"""
Validate the mixed-init model at multiple initial rod configurations.

Now that the model takes init in its prompt, we can ask the *same model*:
"Given init=(180, 100), achieve target X" vs "Given init=(100, 180), achieve target X"
and observe whether actuation pattern preferences differ — the test
Reviewer 1 actually asked for.

Configs evaluated:
  (180, 100)  -- paper default; should reproduce paper's single_b2 dominance
  (100, 180)  -- mirror;        should show single_b1 dominance if policy is adaptive

Each config: 2,000 cases (same seed/cases as paper's 100K validation, for
direct comparability).

Run from WSL with the project venv:
    python validate_mixed.py
or for a quick check:
    python validate_mixed.py --num_cases 50
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import multiprocessing
import re
import shutil
import subprocess
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Paths
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent          # validation/
EXP_DIR = SCRIPT_DIR.parent                            # exp1_initial_rod_variation/
TRAINING_DIR = EXP_DIR / "training"
MY_DIR = EXP_DIR.parent.parent                         # My/
KOMODO_DIR = MY_DIR.parent                             # KOMODO/

PHASE1_MODEL_PATH = TRAINING_DIR / "models" / "phase1_grammar_mixed" / "final_model"
PHASE2_MODEL_PATH = TRAINING_DIR / "models" / "phase2_task_mixed" / "final_model"

KOMODO_EXECUTABLE = KOMODO_DIR / "komodo"
TEMPLATE_FILE = SCRIPT_DIR / "template_init_var"
RUNS_ROOT = SCRIPT_DIR / "runs"

INFERENCE_TEMPERATURE = 0.05
INFERENCE_TOP_P = 0.9
INFERENCE_MAX_TOKENS = 60   # 10 numbers, slightly more than paper's 50

CONFIGS = [
    (180, 100, "default"),
    (100, 180, "mirror"),
]

# ============================================================================
# Model
# ============================================================================

def load_model():
    print(f"[load] Phase 1: {PHASE1_MODEL_PATH}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        str(PHASE1_MODEL_PATH),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    print(f"[load] Phase 2 LoRA: {PHASE2_MODEL_PATH}", flush=True)
    model = PeftModel.from_pretrained(base, str(PHASE2_MODEL_PATH))
    tok = AutoTokenizer.from_pretrained(str(PHASE2_MODEL_PATH))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, tok


def parse_prediction(text: str):
    nums = re.findall(r"[\d.]+", text)
    if len(nums) < 10:
        return None
    try:
        v = [float(x) for x in nums[-6:]]
        return {
            "b1_pos": v[0], "b1_time": v[1], "b1_speed": v[2],
            "b2_pos": v[3], "b2_time": v[4], "b2_speed": v[5],
        }
    except Exception:
        return None


def predict(model, tok, init_b1, init_b2, p_init, p_target):
    prompt = f"[{init_b1}, {init_b2}, {p_init}, {p_target},"
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=INFERENCE_MAX_TOKENS,
            temperature=INFERENCE_TEMPERATURE,
            top_p=INFERENCE_TOP_P,
            do_sample=True,
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    return parse_prediction(text), text

# ============================================================================
# Simulator
# ============================================================================

def render_input(template_path: Path, scenario: dict, out_path: Path):
    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content.format(**scenario))


def run_komodo(input_path: Path):
    try:
        subprocess.run(
            [str(KOMODO_EXECUTABLE), input_path.name],
            cwd=str(input_path.parent),
            check=True, capture_output=True, text=True, encoding="utf-8",
            timeout=60,
        )
        return True, None
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except subprocess.CalledProcessError as e:
        return False, f"komodo failed: {e.stderr[:200]}"
    except FileNotFoundError:
        return False, f"komodo not found: {KOMODO_EXECUTABLE}"


def parse_results(out_file: Path):
    powers = []
    try:
        with open(out_file, "r", encoding="utf-8", errors="ignore") as f:
            in_results = False
            for line in f:
                if "TRANSIENT RESULTS" in line:
                    in_results = True
                    next(f, None); next(f, None)
                    continue
                if in_results and line.strip().startswith("CPU time"):
                    break
                if in_results and line.strip():
                    parts = line.split()
                    if len(parts) >= 4:
                        try: powers.append(float(parts[3]))
                        except (ValueError, IndexError): continue
        if not powers:
            return None
        return {"initial_power": powers[0], "final_power": powers[-1], "peak_power": max(powers)}
    except FileNotFoundError:
        return None


def generate_test_cases(num_cases=2000):
    base = [
        (1.0, 1.05, "+5%"), (1.0, 1.10, "+10%"), (1.0, 1.20, "+20%"),
        (1.0, 1.30, "+30%"), (1.0, 1.40, "+40%"), (1.0, 1.50, "+50%"),
        (1.0, 0.95, "-5%"), (1.0, 0.90, "-10%"), (1.0, 0.80, "-20%"),
        (1.0, 0.70, "-30%"), (1.0, 0.60, "-40%"), (1.0, 0.50, "-50%"),
    ]
    np.random.seed(42)
    cases = list(base)
    for _ in range(num_cases - len(base)):
        f = np.random.uniform(0.5, 1.5)
        cases.append((1.0, round(f, 5), "random"))
    return cases

# ============================================================================
# Worker (KOMODO-only; no GPU). Top-level so multiprocessing.Pool can pickle it.
# ============================================================================

def _execute_komodo_case(task: dict):
    """
    Worker that takes a pre-predicted task and runs KOMODO + parses output.
    Pure CPU + disk; safe to run in a multiprocessing.Pool.
    """
    case_id  = task["case_id"]
    init_b1  = task["init_b1"]
    init_b2  = task["init_b2"]
    p_init   = task["p_init"]
    p_target = task["p_target"]
    desc     = task["desc"]
    pred     = task["pred"]
    raw      = task["raw"]
    run_dir  = Path(task["run_dir"])

    if pred is None:
        return {
            "case_id": case_id, "init_b1": init_b1, "init_b2": init_b2,
            "target_initial": p_init, "target_final": p_target, "description": desc,
            "parsing_success": False, "full_prediction": raw,
        }

    case_dir = run_dir / f"case_{case_id:04d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    scenario = {
        "init_b1": f"{init_b1}.", "init_b2": f"{init_b2}.",
        "b1_pos": pred["b1_pos"], "b1_time": pred["b1_time"], "b1_speed": pred["b1_speed"],
        "b2_pos": pred["b2_pos"], "b2_time": pred["b2_time"], "b2_speed": pred["b2_speed"],
    }
    inp = case_dir / "input.inp"
    render_input(TEMPLATE_FILE, scenario, inp)
    ok, err = run_komodo(inp)
    if not ok:
        shutil.rmtree(case_dir, ignore_errors=True)
        return {
            "case_id": case_id, "init_b1": init_b1, "init_b2": init_b2,
            "target_initial": p_init, "target_final": p_target, "description": desc,
            "parsing_success": True, "predicted_params": pred, "full_prediction": raw,
            "simulation_success": False, "error": err,
        }
    res = parse_results(case_dir / "input.inp.out")
    shutil.rmtree(case_dir, ignore_errors=True)
    if res is None:
        return {
            "case_id": case_id, "init_b1": init_b1, "init_b2": init_b2,
            "target_initial": p_init, "target_final": p_target, "description": desc,
            "parsing_success": True, "predicted_params": pred, "full_prediction": raw,
            "simulation_success": True, "result_parsing_success": False,
        }
    final_err = abs(res["final_power"] - p_target)
    return {
        "case_id": case_id, "init_b1": init_b1, "init_b2": init_b2,
        "target_initial": p_init, "target_final": p_target, "description": desc,
        "parsing_success": True, "predicted_params": pred, "full_prediction": raw,
        "simulation_success": True, "result_parsing_success": True,
        "actual_initial": res["initial_power"], "actual_final": res["final_power"],
        "actual_peak": res["peak_power"], "final_error": final_err,
        "validation_success_1": final_err <= abs(p_target) * 0.01,
        "validation_success_2": final_err <= abs(p_target) * 0.02,
        "validation_success_3": final_err <= abs(p_target) * 0.03,
        "validation_success_5": final_err <= abs(p_target) * 0.05,
        "validation_success_10": final_err <= abs(p_target) * 0.10,
    }

# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_cases", type=int, default=2000)
    ap.add_argument("--only", choices=["default", "mirror"], default=None,
                    help="Run only one config")
    ap.add_argument("--workers", type=int, default=12,
                    help="Parallel KOMODO workers (GPU inference stays sequential)")
    ap.add_argument("--restart", action="store_true",
                    help="Discard partial JSONL and start config from case 1")
    args = ap.parse_args()

    model, tok = load_model()
    cases = generate_test_cases(args.num_cases)

    configs = [c for c in CONFIGS if (args.only is None or c[2] == args.only)]
    for init_b1, init_b2, tag in configs:
        run_dir = RUNS_ROOT / f"{tag}_b1_{init_b1}_b2_{init_b2}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*70}\n[config] {tag}: B1={init_b1}, B2={init_b2}, n={len(cases)}, workers={args.workers}\n{'='*70}", flush=True)

        # Resume detection: skip case_ids already in JSONL
        partial_path = run_dir / "results_partial.jsonl"
        results = []
        done_ids = set()
        if partial_path.exists() and not args.restart:
            with open(partial_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        rec = json.loads(line)
                        results.append(rec)
                        done_ids.add(rec["case_id"])
                    except Exception:
                        continue
            if done_ids:
                print(f"[resume] {len(done_ids)} cases already done in {partial_path.name}", flush=True)
        elif args.restart and partial_path.exists():
            partial_path.unlink()

        # Phase 1: GPU inference (sequential, fast: ~50ms/case)
        pending_tasks = []
        todo = [(i, p0, pf, desc) for i, (p0, pf, desc) in enumerate(cases, 1) if i not in done_ids]
        if todo:
            for i, p0, pf, desc in tqdm(todo, desc=f"predict {tag}", unit="case", ncols=100):
                pred, raw = predict(model, tok, init_b1, init_b2, p0, pf)
                pending_tasks.append({
                    "case_id": i, "init_b1": init_b1, "init_b2": init_b2,
                    "p_init": p0, "p_target": pf, "desc": desc,
                    "pred": pred, "raw": raw, "run_dir": str(run_dir),
                })

        # Phase 2: KOMODO execution (parallel, slow: ~4s/case but K-way parallel)
        succ5 = sum(1 for r in results if r.get("validation_success_5"))
        n_target = len(cases)
        try:
            with open(partial_path, "a", encoding="utf-8") as fjsonl:
                if pending_tasks:
                    with multiprocessing.Pool(processes=args.workers) as pool:
                        pbar = tqdm(
                            pool.imap_unordered(_execute_komodo_case, pending_tasks),
                            total=len(pending_tasks), desc=f"komodo {tag}",
                            unit="case", ncols=110,
                        )
                        for rec in pbar:
                            results.append(rec)
                            if rec.get("validation_success_5"):
                                succ5 += 1
                            fjsonl.write(json.dumps(rec) + "\n")
                            fjsonl.flush()
                            done_count = len(results)
                            pbar.set_postfix(acc5=f"{succ5/done_count*100:.1f}%")
        except KeyboardInterrupt:
            print(f"\n[interrupted] partial preserved: {len(results)}/{n_target} -> {partial_path}", flush=True)
            raise

        # Final consolidated JSON (downstream-compat with paper's plot scripts)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sort results by case_id for deterministic output
        results_sorted = sorted(results, key=lambda r: r["case_id"])
        out = run_dir / f"results_{tag}_{args.num_cases}cases_{ts}.json"
        with open(out, "w") as f:
            json.dump(results_sorted, f, indent=2)
        shutil.copy(out, run_dir / "results_latest.json")

        n = len(results_sorted)
        s1 = sum(1 for r in results_sorted if r.get("validation_success_1"))
        s5 = sum(1 for r in results_sorted if r.get("validation_success_5"))
        s10 = sum(1 for r in results_sorted if r.get("validation_success_10"))
        print(f"\n[done] {tag}: ±1%: {s1/n*100:.1f}%  ±5%: {s5/n*100:.1f}%  ±10%: {s10/n*100:.1f}%", flush=True)
        print(f"  saved: {out}\n")


if __name__ == "__main__":
    main()
