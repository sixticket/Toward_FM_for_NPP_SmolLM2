"""
Gain-scheduled PID baseline.

Differs from the paper's single-gain PID in two specific ways and only those:
  (1) Six regime-dependent proportional gains instead of one global Kp.
      Regimes are split by power-change magnitude (small / medium / large)
      and direction (increase / decrease) — six total.
  (2) Calibration sweep is wider (~33 KOMODO sims spanning rod displacements
      [-80, +80] in steps of 5) so that each regime has its own linear fit
      rather than relying on a global 6-shot symmetric fit.

Everything else matches the paper PID baseline:
  - B2 is the primary actuator; B1 is engaged only on B2 saturation
  - Rod motion at fixed DEFAULT_SPEED = 2.0 steps/s
  - b_time = dist/speed (matches `My/PID/pid_validation.py` convention)
  - Same 2,000-case test set (seeded), same KOMODO closed-loop protocol

Pipeline
--------
1. Calibration (--calibrate, ~2 min, ~33 KOMODO sims):
     Sweep delta_step in [-80, +80] step 5 on B2 only.
     Bin completed sims by observed power_delta into the six regimes.
     Per regime, fit y_step = slope * x_power_delta and store Kp_regime = |slope|.
     Cache to calibration_config.json.
2. Validation (--validate, ~12 min with 12 workers):
     For each test case, classify into regime by target power change,
     look up Kp_regime, compute control vector with same saturation logic
     as paper PID, run KOMODO once, score against tolerance bands.

Run from WSL:
    python pid_scheduled.py --calibrate --validate --workers 12
"""

import argparse
import csv
import json
import multiprocessing
import shutil
import subprocess
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
from scipy import stats
from tqdm import tqdm

# ============================================================================
# Paths and constants
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent      # exp2_scheduled_pid/
EXP_DIR = SCRIPT_DIR.parent                        # revision_experiments/
MY_DIR = EXP_DIR.parent                            # My/
KOMODO_DIR = MY_DIR.parent                         # KOMODO/

KOMODO_EXECUTABLE = KOMODO_DIR / "komodo"
TEMPLATE_FILE = MY_DIR / "template"               # original template (init=180,100)
WORK_DIR = SCRIPT_DIR / "runs"
CONFIG_FILE = SCRIPT_DIR / "calibration_config.json"

# Paper-matched constants
INIT_B1_POS = 180.0
INIT_B2_POS = 100.0
DEFAULT_SPEED = 2.0

# Regimes: (label, delta_power_lower, delta_power_upper)
# delta_power = target_final - target_initial; classifier uses inclusive lo, exclusive hi.
REGIMES = [
    ("large_dec",  -1.00, -0.30),
    ("medium_dec", -0.30, -0.10),
    ("small_dec",  -0.10,  0.00),
    ("small_inc",   0.00,  0.10),
    ("medium_inc",  0.10,  0.30),
    ("large_inc",   0.30,  1.00),
]

FALLBACK_KP = 35.5  # global single-gain Kp from paper baseline (used if regime has too few samples)


def classify_regime(target_init: float, target_final: float) -> str:
    delta = target_final - target_init
    for name, lo, hi in REGIMES:
        if lo <= delta < hi:
            return name
    # boundary fallback
    return "large_inc" if delta >= 0 else "large_dec"


# ============================================================================
# Simulator I/O (mirrors My/PID/pid_validation.py)
# ============================================================================

def render(scenario: dict, out_path: Path):
    with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
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


def cleanup(d: Path):
    try:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    except Exception:
        pass


# ============================================================================
# Calibration: sweep B2 displacements, bin by regime, fit per-regime Kp
# ============================================================================

def _run_one_calibration_pickleable(args: tuple):
    """
    Top-level worker for parallel calibration sweep.
    args = (delta_step, work_dir_str)
    Returns (delta_step, power_delta) or (delta_step, None) on failure.
    """
    delta_step, work_dir_str = args
    work_dir = Path(work_dir_str)
    target_b2 = INIT_B2_POS + delta_step
    if target_b2 < 0 or target_b2 > 180:
        return (delta_step, None)
    dist = abs(delta_step)
    time_val = max(0.1, dist / DEFAULT_SPEED) if dist > 0 else 0.0

    case_dir = work_dir / f"calib_{delta_step:+d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    scenario = {
        "b1_pos": INIT_B1_POS, "b1_time": 0.0, "b1_speed": 0.0,
        "b2_pos": round(float(target_b2), 3),
        "b2_time": round(time_val, 3),
        "b2_speed": DEFAULT_SPEED if dist > 0 else 0.0,
    }
    inp = case_dir / "input.inp"
    render(scenario, inp)
    ok, _ = run_komodo(inp)
    if not ok:
        cleanup(case_dir); return (delta_step, None)
    res = parse_results(case_dir / "input.inp.out")
    cleanup(case_dir)
    if res is None:
        return (delta_step, None)
    return (delta_step, res["final_power"] - res["initial_power"])


def _fit_regime_kp(samples_in_regime, regime_name, lo, hi):
    """
    Fit Kp for a regime. Use linear regression slope when R² is reasonable;
    fall back to median |delta_step / power_delta| when fit is poor (which
    captures within-regime nonlinearity better than a poor linear slope).

    Returns (Kp, r2, method).
    """
    if len(samples_in_regime) < 2:
        return FALLBACK_KP, 0.0, "fallback (n<2)"

    x = np.array([pd for ds, pd in samples_in_regime])  # power_delta
    y = np.array([ds for ds, pd in samples_in_regime])  # delta_step
    # Filter out near-zero power_delta to avoid div-by-zero in median fallback
    mask = np.abs(x) > 1e-4
    x_safe = x[mask]; y_safe = y[mask]
    if len(x_safe) < 2:
        return FALLBACK_KP, 0.0, "fallback (samples too close to zero)"

    # Linear regression: y_step = slope * x_power_delta
    slope, intercept, r_val, _, _ = stats.linregress(x_safe, y_safe)
    r2 = float(r_val ** 2)
    kp_lin = float(abs(slope))

    # Median ratio: per-sample Kp_i = |delta_step / power_delta|, take median (robust to outliers)
    ratios = np.abs(y_safe / x_safe)
    kp_med = float(np.median(ratios))

    if r2 >= 0.5 and kp_lin > 0:
        return kp_lin, r2, "linear_regression"
    return kp_med, r2, "median_ratio (low R^2)"


def calibrate_all(sweep_step: int = 1, sweep_dec_max: int = 25, sweep_inc_max: int = 15, workers: int = 12):
    """
    Parallel calibration sweep over B2 delta_steps in [-sweep_dec_max, +sweep_inc_max].

    Asymmetric range reflects reactor's asymmetric reactivity response: small B2
    withdrawal (positive delta_step) produces large positive power changes, while
    insertion (negative delta_step) produces smaller-magnitude changes. The
    sweep step is fine (default 1) to populate the small-power-change regimes
    that linear gain scheduling needs to cover.

    Bin each sim by observed power_delta into a regime; fit one Kp per regime.
    Use linear regression when R² >= 0.5; otherwise fall back to median ratio
    (more robust to within-regime nonlinearity).
    """
    sweep = [d for d in range(-sweep_dec_max, sweep_inc_max + 1, sweep_step) if d != 0]
    print(f"[calib] B2 sweep: dec_max={sweep_dec_max}, inc_max={sweep_inc_max}, step={sweep_step}  ({len(sweep)} sims, {workers} workers)", flush=True)

    work = WORK_DIR / "calib"
    work.mkdir(parents=True, exist_ok=True)

    tasks = [(d, str(work)) for d in sweep]
    samples = []
    with multiprocessing.Pool(processes=workers) as pool:
        pbar = tqdm(
            pool.imap_unordered(_run_one_calibration_pickleable, tasks),
            total=len(tasks), desc="calibration sweep", unit="sim", ncols=110,
        )
        for ds, pd in pbar:
            if pd is not None:
                samples.append((ds, pd))
            pbar.set_postfix(last=f"ds={ds:+d}, dP={(pd if pd is not None else float('nan')):+.3f}", n_ok=len(samples))

    if not samples:
        raise RuntimeError("calibration sweep produced zero valid samples")

    samples.sort()  # by delta_step
    cleanup(work)
    print(f"\n[calib] collected {len(samples)} valid (delta_step, power_delta) samples", flush=True)

    # Diagnostic: show power_delta distribution
    pds = sorted([pd for _, pd in samples])
    print(f"[calib] power_delta range: [{pds[0]:+.3f}, {pds[-1]:+.3f}]  median={pds[len(pds)//2]:+.3f}", flush=True)

    config = {
        "calibrated_at": datetime.now().isoformat(),
        "sweep_step": sweep_step,
        "sweep_dec_max": sweep_dec_max,
        "sweep_inc_max": sweep_inc_max,
        "n_total_samples": len(samples),
        "all_samples": [{"delta_step": ds, "power_delta": pd} for ds, pd in samples],
        "regimes": {},
    }

    print(f"\n[calib] fitting per-regime Kp:", flush=True)
    for name, lo, hi in REGIMES:
        in_regime = [(ds, pd) for ds, pd in samples if lo <= pd < hi]
        if len(in_regime) < 2:
            kp = FALLBACK_KP
            print(f"  {name:12s}: only {len(in_regime)} samples in [{lo:+.2f}, {hi:+.2f}) — fallback Kp={kp}", flush=True)
            config["regimes"][name] = {
                "lo": lo, "hi": hi, "Kp": kp, "n_samples": len(in_regime),
                "r2": 0.0, "status": "fallback (insufficient samples)",
                "method": "fallback",
            }
            continue

        kp, r2, method = _fit_regime_kp(in_regime, name, lo, hi)
        ds_range = (min(ds for ds, _ in in_regime), max(ds for ds, _ in in_regime))
        pd_range = (min(pd for _, pd in in_regime), max(pd for _, pd in in_regime))
        print(f"  {name:12s}: Kp={kp:7.3f}  n={len(in_regime):2d}  R²={r2:.3f}  ds=[{ds_range[0]:+d},{ds_range[1]:+d}]  dP=[{pd_range[0]:+.3f},{pd_range[1]:+.3f}]  ({method})", flush=True)
        config["regimes"][name] = {
            "lo": lo, "hi": hi, "Kp": kp, "n_samples": len(in_regime),
            "r2": r2, "status": "ok", "method": method,
            "ds_range": list(ds_range), "pd_range": list(pd_range),
        }

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n[calib] saved: {CONFIG_FILE}", flush=True)
    return config


# ============================================================================
# Runtime control (mirrors paper PID's saturation logic; only Kp is regime-dependent)
# ============================================================================

def calculate_control(target_initial: float, target_final: float, config: dict):
    regime = classify_regime(target_initial, target_final)
    Kp = float(config["regimes"][regime]["Kp"])

    power_error = target_final - target_initial
    delta_steps = power_error * Kp

    b2_target = INIT_B2_POS + delta_steps
    b1_target = INIT_B1_POS

    # Saturation: spill into B1 if B2 hits 0 or 180 (matches paper PID)
    if b2_target > 180.0:
        rem = b2_target - 180.0
        b2_target = 180.0
        b1_target = min(180.0, INIT_B1_POS + rem)
    elif b2_target < 0.0:
        rem = b2_target  # negative
        b2_target = 0.0
        b1_target = max(0.0, INIT_B1_POS + rem)

    dist_b1 = abs(b1_target - INIT_B1_POS)
    dist_b2 = abs(b2_target - INIT_B2_POS)

    return {
        "b1_pos": round(float(b1_target), 3),
        "b1_time": round(max(0.1, dist_b1 / DEFAULT_SPEED), 3) if dist_b1 > 0 else 0.0,
        "b1_speed": DEFAULT_SPEED if dist_b1 > 0 else 0.0,
        "b2_pos": round(float(b2_target), 3),
        "b2_time": round(max(0.1, dist_b2 / DEFAULT_SPEED), 3) if dist_b2 > 0 else 0.0,
        "b2_speed": DEFAULT_SPEED if dist_b2 > 0 else 0.0,
        "_regime": regime,
        "_Kp_used": Kp,
    }


# ============================================================================
# Worker (top-level for multiprocessing pickle)
# ============================================================================

def _execute_case(task: dict):
    p0 = task["target_initial"]
    pf = task["target_final"]
    desc = task["desc"]
    case_id = task["case_id"]
    work = Path(task["work_dir"])
    config = task["config"]

    control = calculate_control(p0, pf, config)
    regime = control.pop("_regime")
    kp_used = control.pop("_Kp_used")

    case_dir = work / f"case_{case_id:04d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    inp = case_dir / "input.inp"
    render(control, inp)
    ok, err = run_komodo(inp)

    base = {
        "case_id": case_id, "target_initial": p0, "target_final": pf, "description": desc,
        "predicted_params": control, "regime": regime, "Kp_used": kp_used,
    }
    if not ok:
        cleanup(case_dir)
        return {**base, "simulation_success": False, "error": err}
    res = parse_results(case_dir / "input.inp.out")
    cleanup(case_dir)
    if res is None:
        return {**base, "simulation_success": True, "result_parsing_success": False}

    final_err = abs(res["final_power"] - pf)
    return {
        **base,
        "simulation_success": True, "result_parsing_success": True,
        "actual_initial": res["initial_power"], "actual_final": res["final_power"],
        "actual_peak": res["peak_power"], "final_error": final_err,
        "validation_success_1": final_err <= abs(pf) * 0.01,
        "validation_success_2": final_err <= abs(pf) * 0.02,
        "validation_success_3": final_err <= abs(pf) * 0.03,
        "validation_success_5": final_err <= abs(pf) * 0.05,
        "validation_success_10": final_err <= abs(pf) * 0.10,
    }


# ============================================================================
# Test cases (same seed/cases as paper)
# ============================================================================

def generate_test_cases(num_cases: int = 2000):
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
# Validation suite (parallel KOMODO with incremental JSONL)
# ============================================================================

def validate(num_cases: int, restart: bool = False, workers: int = 12):
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"calibration not found: {CONFIG_FILE}\nRun --calibrate first.")
    config = json.load(open(CONFIG_FILE))
    print(f"[val] using calibration ({len(config['regimes'])} regimes), workers={workers}", flush=True)
    for name, _, _ in REGIMES:
        info = config["regimes"][name]
        print(f"        {name:12s}  Kp={info['Kp']:7.3f}  n={info['n_samples']:2d}  R²={info['r2']:.3f}  ({info['status']})", flush=True)

    cases = generate_test_cases(num_cases)
    work = WORK_DIR / "val"
    work.mkdir(parents=True, exist_ok=True)

    partial_path = WORK_DIR / "pid_scheduled_results_partial.jsonl"
    results = []
    done_ids = set()

    if partial_path.exists() and not restart:
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
    elif restart and partial_path.exists():
        partial_path.unlink()

    succ5 = sum(1 for r in results if r.get("validation_success_5"))

    pending = [
        {"case_id": i, "target_initial": p0, "target_final": pf, "desc": desc,
         "config": config, "work_dir": str(work)}
        for i, (p0, pf, desc) in enumerate(cases, 1) if i not in done_ids
    ]

    if pending:
        try:
            with open(partial_path, "a", encoding="utf-8") as fjsonl:
                with multiprocessing.Pool(processes=workers) as pool:
                    pbar = tqdm(
                        pool.imap_unordered(_execute_case, pending),
                        total=len(pending), desc="pid validate",
                        unit="case", ncols=110,
                    )
                    for rec in pbar:
                        results.append(rec)
                        if rec.get("validation_success_5"):
                            succ5 += 1
                        fjsonl.write(json.dumps(rec) + "\n")
                        fjsonl.flush()
                        pbar.set_postfix(acc5=f"{succ5/len(results)*100:.1f}%", saved=len(results))
        except KeyboardInterrupt:
            print(f"\n[interrupted] partial preserved: {len(results)}/{num_cases} -> {partial_path}", flush=True)
            raise
    else:
        print(f"[val] all {num_cases} cases already in JSONL", flush=True)

    # Final consolidation
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_sorted = sorted(results, key=lambda r: r["case_id"])
    out = WORK_DIR / f"pid_scheduled_results_{num_cases}cases_{ts}.json"
    with open(out, "w") as f:
        json.dump(results_sorted, f, indent=2)
    fixed = WORK_DIR / "pid_scheduled_results_latest.json"
    shutil.copy(out, fixed)

    # ---- Summary stats ----
    import statistics as _st
    n = len(results_sorted)
    sim_ok = sum(1 for r in results_sorted if r.get("simulation_success"))
    s_tol = {tol: sum(1 for r in results_sorted if r.get(f"validation_success_{tol}")) for tol in (1, 2, 3, 5, 10)}
    errors = [r["final_error"] * 100 for r in results_sorted if r.get("final_error") is not None]

    print(f"\n{'='*70}\n  Gain-scheduled PID validation summary  (n={n})\n{'='*70}")
    print(f"  sim_ok: {sim_ok}/{n} ({sim_ok/n*100:.1f}%)")
    for tol in (1, 2, 3, 5, 10):
        print(f"  ±{tol:2d}%: {s_tol[tol]/n*100:5.1f}%  ({s_tol[tol]:4d}/{n})")
    if errors:
        sorted_e = sorted(errors)
        p95 = sorted_e[int(0.95 * len(sorted_e))]
        p99 = sorted_e[int(0.99 * len(sorted_e))]
        print(f"\n  err  mean: {_st.mean(errors):.2f}%  median: {_st.median(errors):.2f}%  max: {max(errors):.2f}%")
        print(f"       p95: {p95:.2f}%   p99: {p99:.2f}%")

    print(f"\n  ±5% by power-change magnitude:")
    for label, lo_dp, hi_dp in [("small (≤10%)", 0, 0.1001), ("medium (10-30%)", 0.1001, 0.3001), ("large (>30%)", 0.3001, 100)]:
        sub = [r for r in results_sorted if lo_dp <= abs(r["target_final"] - 1.0) < hi_dp]
        if sub:
            ss = sum(1 for r in sub if r.get("validation_success_5"))
            print(f"    {label:18s}: {ss:4d}/{len(sub):4d} = {ss/len(sub)*100:5.1f}%")

    print(f"\n  ±5% by direction:")
    for label, sub in [
        ("increase", [r for r in results_sorted if r["target_final"] > 1.0]),
        ("decrease", [r for r in results_sorted if r["target_final"] < 1.0]),
    ]:
        if sub:
            ss = sum(1 for r in sub if r.get("validation_success_5"))
            print(f"    {label:18s}: {ss:4d}/{len(sub):4d} = {ss/len(sub)*100:5.1f}%")

    print(f"\n  ±5% by regime (assigned by classifier at runtime):")
    by_regime = {}
    for r in results_sorted:
        by_regime.setdefault(r.get("regime", "unknown"), []).append(r)
    for name, _, _ in REGIMES:
        if name not in by_regime: continue
        sub = by_regime[name]
        ss = sum(1 for r in sub if r.get("validation_success_5"))
        kp = config["regimes"][name]["Kp"]
        print(f"    {name:12s}: {ss:4d}/{len(sub):4d} = {ss/len(sub)*100:5.1f}%   (Kp={kp:.2f})")

    print(f"\n  saved: {out}")
    print(f"         {fixed}")


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("--calibrate", action="store_true", help="run calibration sweep + fit per-regime Kp")
    ap.add_argument("--validate",  action="store_true", help="run 2,000-case validation with cached calibration")
    ap.add_argument("--num_cases", type=int, default=2000)
    ap.add_argument("--workers",   type=int, default=12, help="parallel KOMODO workers for validation")
    ap.add_argument("--restart",   action="store_true", help="discard partial JSONL and start validation from case 1")
    ap.add_argument("--sweep_step",    type=int, default=1,
                    help="calibration: rod-displacement step size (default 1, fine sweep)")
    ap.add_argument("--sweep_dec_max", type=int, default=25,
                    help="calibration: max negative displacement (insertion); default 25")
    ap.add_argument("--sweep_inc_max", type=int, default=15,
                    help="calibration: max positive displacement (withdrawal); default 15 (asymmetric due to reactor sensitivity)")
    args = ap.parse_args()

    if not args.calibrate and not args.validate:
        ap.error("specify --calibrate and/or --validate")

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    if args.calibrate:
        calibrate_all(
            sweep_step=args.sweep_step,
            sweep_dec_max=args.sweep_dec_max,
            sweep_inc_max=args.sweep_inc_max,
            workers=args.workers,
        )
    if args.validate:
        validate(num_cases=args.num_cases, restart=args.restart, workers=args.workers)


if __name__ == "__main__":
    main()
