"""
End-to-end orchestrator for Exp 1 (mixed-init experiment).

Runs the 6-step pipeline with progress reporting and resume support:
  1. sample_existing.py            (~10s)   stratified 50K from paper's 100K CSV
  2. generate_mirrored.py          (~3h)    new 50K KOMODO sims at init=(100,180)
  3. combine.py                    (~10s)   merge into mixed-init CSV
  4. phase1_grammar_mixed.py       (~6h)    CPT, 8-token format with init
  5. phase2_task_mixed.py          (~10h)   LoRA, 10-token format with init
  6. validate_mixed.py             (~6h)    2,000 cases x 2 init configs

Each step's stdout/stderr is streamed to the terminal AND copied to logs/<step>.log
so internal tqdm bars (KOMODO sims, transformers Trainer, validation cases) stay
visible. An outer tqdm tracks pipeline progress across steps.

Run from WSL with the project venv:
    PY=/mnt/c/projects/Foundation_Model/KOMODO/My/training/venv/bin/python
    $PY run_all.py                      # full pipeline, skip steps with existing outputs
    $PY run_all.py --no-skip-existing   # rerun everything
    $PY run_all.py --from-step 4        # start at Phase 1 training (data already built)
    $PY run_all.py --only 6             # just re-run validation
    $PY run_all.py --num-cases 50       # smoke test on validation
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================================
# Pipeline definition
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent       # exp1_initial_rod_variation/
DATA_DIR = SCRIPT_DIR / "data_generation"
TRAIN_DIR = SCRIPT_DIR / "training"
VAL_DIR = SCRIPT_DIR / "validation"
LOG_DIR = SCRIPT_DIR / "logs"

# Default python: project venv (where torch/transformers/peft are installed)
DEFAULT_PY = "/mnt/c/projects/Foundation_Model/KOMODO/My/training/venv/bin/python"


def _csv_min_rows(path: Path, expected_data_rows: int):
    """Returns True iff CSV exists with at least header + expected_data_rows lines."""
    if not path.exists():
        return False
    with open(path, "r", encoding="utf-8") as f:
        n = sum(1 for _ in f)
    return n >= expected_data_rows + 1  # +1 for header


def _hf_model_dir_complete(path: Path):
    """Returns True iff HF/PEFT model dir exists with key files."""
    if not path.exists() or not path.is_dir():
        return False
    must_have_one = ["model.safetensors", "pytorch_model.bin",
                     "adapter_model.safetensors", "adapter_model.bin"]
    return any((path / f).exists() for f in must_have_one)


def _val_json_complete(path: Path, expected_cases: int):
    """Returns True iff JSON exists and has at least expected_cases entries."""
    if not path.exists():
        return False
    try:
        import json
        with open(path, "r") as f:
            data = json.load(f)
        return isinstance(data, list) and len(data) >= expected_cases
    except Exception:
        return False


def make_steps(args):
    """Return list of (id, name, cwd, cmd, complete_check)."""
    steps = []

    # 1. Stratified sample of existing 100K CSV  -> 50K rows
    steps.append({
        "id": 1, "name": "sample_existing",
        "cwd": DATA_DIR,
        "cmd": [args.python, "sample_existing.py"],
        "outputs": [DATA_DIR / "subset_default_init_50k.csv"],
        "complete": lambda: _csv_min_rows(DATA_DIR / "subset_default_init_50k.csv", 50000),
        "eta_hours": 0.01,
    })

    # 2. Generate mirrored 50K (KOMODO sims)  -> 50K rows
    steps.append({
        "id": 2, "name": "generate_mirrored",
        "cwd": DATA_DIR,
        "cmd": [args.python, "generate_mirrored.py", "--workers", str(args.workers)],
        "outputs": [DATA_DIR / "subset_mirror_init_50k.csv"],
        "complete": lambda: _csv_min_rows(DATA_DIR / "subset_mirror_init_50k.csv", 50000),
        "eta_hours": 3.0,
    })

    # 3. Combine -> 100K rows
    steps.append({
        "id": 3, "name": "combine",
        "cwd": DATA_DIR,
        "cmd": [args.python, "combine.py"],
        "outputs": [DATA_DIR / "master_dataset_mixed_init_100k.csv"],
        "complete": lambda: _csv_min_rows(DATA_DIR / "master_dataset_mixed_init_100k.csv", 100000),
        "eta_hours": 0.01,
    })

    # 4. Phase 1 grammar (CPT)
    p1_dir = TRAIN_DIR / "models" / "phase1_grammar_mixed" / "final_model"
    steps.append({
        "id": 4, "name": "phase1_grammar",
        "cwd": TRAIN_DIR,
        "cmd": [args.python, "phase1_grammar_mixed.py"],
        "outputs": [p1_dir],
        "complete": lambda: _hf_model_dir_complete(p1_dir),
        "eta_hours": 6.0,
    })

    # 5. Phase 2 task (LoRA)
    p2_dir = TRAIN_DIR / "models" / "phase2_task_mixed" / "final_model"
    steps.append({
        "id": 5, "name": "phase2_task",
        "cwd": TRAIN_DIR,
        "cmd": [args.python, "phase2_task_mixed.py"],
        "outputs": [p2_dir],
        "complete": lambda: _hf_model_dir_complete(p2_dir),
        "eta_hours": 10.0,
    })

    # 6. Validate at both inits (KOMODO calls now parallelized with --workers)
    v_def = VAL_DIR / "runs" / "default_b1_180_b2_100" / "results_latest.json"
    v_mir = VAL_DIR / "runs" / "mirror_b1_100_b2_180"  / "results_latest.json"
    val_cmd = [args.python, "validate_mixed.py",
               "--num_cases", str(args.num_cases),
               "--workers", str(args.workers)]
    steps.append({
        "id": 6, "name": "validate_mixed",
        "cwd": VAL_DIR,
        "cmd": val_cmd,
        "outputs": [v_def, v_mir],
        "complete": lambda: _val_json_complete(v_def, args.num_cases) and _val_json_complete(v_mir, args.num_cases),
        "eta_hours": 0.5,
    })

    return steps


# ============================================================================
# Step runner
# ============================================================================

def output_complete(step):
    """Check whether a step's outputs are not just present but actually complete.
    Falls back to existence check if no `complete` callable is defined."""
    fn = step.get("complete")
    if fn is None:
        return all(p.exists() for p in step.get("outputs", []))
    try:
        return bool(fn())
    except Exception as e:
        print(f"[warn] complete() check raised: {e}", flush=True)
        return False


def run_step(step, log_path):
    """
    Run subprocess with stdout/stderr inherited from parent (so tqdm sees a TTY
    and updates live), AND mirror to a log file via shell `tee`.

    Why shell + tee:
      - `for line in proc.stdout` stalls on tqdm \r updates (no newline => buffered)
      - `bufsize=0` would still mishandle CR rewrites
      - `tee` keeps the parent TTY connected so tqdm refreshes normally
    """
    import shlex

    print(f"\n{'='*80}\n[step {step['id']}/6] {step['name']}\n  cwd: {step['cwd']}\n  cmd: {' '.join(map(str, step['cmd']))}\n  log: {log_path}\n{'='*80}", flush=True)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()

    # Pre-write log header (tee will append after)
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"# step {step['id']}: {step['name']}\n")
        logf.write(f"# started: {datetime.now().isoformat()}\n")
        logf.write(f"# cwd: {step['cwd']}\n")
        logf.write(f"# cmd: {' '.join(map(str, step['cmd']))}\n\n")

    cmd_quoted = " ".join(shlex.quote(str(c)) for c in step["cmd"])
    full = f"({cmd_quoted}) 2>&1 | tee -a {shlex.quote(str(log_path))}"
    ret = subprocess.call(full, cwd=str(step["cwd"]), shell=True, executable="/bin/bash")

    elapsed = time.time() - start
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"\n# finished: {datetime.now().isoformat()}\n")
        logf.write(f"# elapsed: {elapsed:.1f}s\n")
        logf.write(f"# exit: {ret}\n")

    return ret, elapsed


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("--python", default=DEFAULT_PY,
                    help=f"Python interpreter (default: {DEFAULT_PY})")
    ap.add_argument("--from-step", type=int, default=1, choices=range(1, 7),
                    help="Start at this step (default: 1)")
    ap.add_argument("--only", type=int, default=None, choices=range(1, 7),
                    help="Run only this single step")
    ap.add_argument("--no-skip-existing", action="store_true",
                    help="Force re-run of steps even if outputs already exist")
    ap.add_argument("--workers", type=int, default=12,
                    help="Workers for generate_mirrored.py (default: 12)")
    ap.add_argument("--num-cases", type=int, default=2000,
                    help="Cases per init config in validate_mixed.py (default: 2000)")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="Don't abort the pipeline if a step fails")
    args = ap.parse_args()

    # Sanity: Python interpreter exists?
    if not Path(args.python).exists():
        print(f"[error] python interpreter not found: {args.python}", file=sys.stderr)
        print("        Pass --python /path/to/venv/bin/python", file=sys.stderr)
        sys.exit(2)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    steps = make_steps(args)

    # Filter steps based on --from-step / --only
    if args.only is not None:
        active = [s for s in steps if s["id"] == args.only]
    else:
        active = [s for s in steps if s["id"] >= args.from_step]

    skip_existing = not args.no_skip_existing

    # Show plan
    print(f"\n{'#'*80}\n# Exp 1: Initial Rod Position Variation -- pipeline\n{'#'*80}")
    print(f"# python      : {args.python}")
    print(f"# log dir     : {LOG_DIR}")
    print(f"# skip-existing: {skip_existing}")
    print(f"# steps to run: {[s['id'] for s in active]}")
    eta_total = sum(s["eta_hours"] for s in active)
    print(f"# total ETA   : ~{eta_total:.1f}h ({timedelta(hours=eta_total)})")

    # No outer tqdm here — would clash with subprocess tqdm bars writing to the
    # same TTY. We use plain text headers per step instead; each subprocess gets
    # the full TTY for its own tqdm.

    overall_start = time.time()
    summary = []
    n_total = len(active)

    for idx, step in enumerate(active, 1):
        print(f"\n\n>>> pipeline {idx}/{n_total}: step {step['id']} ({step['name']}) — eta ~{step['eta_hours']:.1f}h\n", flush=True)

        # Skip if outputs are COMPLETE (not just present).
        if skip_existing and output_complete(step):
            print(f"[skip] step {step['id']} ({step['name']}): outputs complete", flush=True)
            for o in step["outputs"]:
                print(f"        - {o}", flush=True)
            summary.append((step["id"], step["name"], 0.0, "skipped"))
            continue

        log_path = LOG_DIR / f"step{step['id']}_{step['name']}.log"
        ret, elapsed = run_step(step, log_path)
        summary.append((step["id"], step["name"], elapsed, "ok" if ret == 0 else f"exit={ret}"))

        if ret != 0:
            print(f"\n[fail] step {step['id']} ({step['name']}) exited with {ret}", flush=True)
            print(f"       see {log_path}", flush=True)
            if not args.continue_on_error:
                print(f"\n[abort] use --continue-on-error to ignore failures", flush=True)
                _print_summary(summary, overall_start)
                sys.exit(ret)

    _print_summary(summary, overall_start)


def _print_summary(summary, start):
    total = time.time() - start
    print(f"\n{'='*80}\n# Pipeline summary -- total wall {timedelta(seconds=int(total))}\n{'='*80}")
    print(f"  {'#':>2}  {'name':<22} {'status':<10} {'elapsed':>12}")
    for i, name, elapsed, status in summary:
        e = timedelta(seconds=int(elapsed)) if elapsed else "-"
        print(f"  {i:>2}  {name:<22} {status:<10} {str(e):>12}")


if __name__ == "__main__":
    main()
