"""
Compare actuation pattern preferences and success rates between
the two init configs validated by the mixed-init model.

Key question: does the model's pattern preference INVERT between
init=(180,100) and init=(100,180)?

If yes -> preference is adaptive (learned policy reflects physics).
If no  -> preference is hardwired bias (single_b2 win regardless).
"""

import json
import math
import statistics
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_ROOT = SCRIPT_DIR / "validation" / "runs"

PAPER_BASELINE = (
    SCRIPT_DIR.parent.parent
    / "validation/validation_runs_v7_simple/validation_results_v7_simple_2000cases_100K.json"
)


def classify(p: dict, init_b1: float, init_b2: float) -> str:
    """init-aware classification (rod 'moves' iff predicted final pos differs from init)"""
    b1_moves = p["b1_speed"] > 0 and abs(p["b1_pos"] - init_b1) > 0.5
    b2_moves = p["b2_speed"] > 0 and abs(p["b2_pos"] - init_b2) > 0.5
    if not b1_moves and not b2_moves:
        return "no_motion"
    if b1_moves and not b2_moves:
        return "single_b1"
    if b2_moves and not b1_moves:
        return "single_b2"
    if math.isclose(p["b1_time"], p["b2_time"], abs_tol=0.05):
        return "simultaneous"
    return "sequential"


def summarize(results, label, init_b1=180.0, init_b2=100.0):
    n = len(results)
    if n == 0:
        print(f"[{label}] empty"); return
    parsed = sum(1 for r in results if r.get("parsing_success"))
    sim_ok = sum(1 for r in results if r.get("simulation_success"))
    s1 = sum(1 for r in results if r.get("validation_success_1"))
    s5 = sum(1 for r in results if r.get("validation_success_5"))
    s10 = sum(1 for r in results if r.get("validation_success_10"))

    patterns = {"single_b1": 0, "single_b2": 0, "simultaneous": 0, "sequential": 0, "no_motion": 0}
    for r in results:
        p = r.get("predicted_params")
        if p:
            ib1 = r.get("init_b1", init_b1)
            ib2 = r.get("init_b2", init_b2)
            patterns[classify(p, ib1, ib2)] += 1

    errors_pct = [r["final_error"] * 100 for r in results if r.get("final_error") is not None]

    print(f"\n=== {label} (n={n}) ===")
    print(f"  parsing : {parsed/n*100:5.1f}%   sim_ok: {sim_ok/n*100:5.1f}%")
    print(f"  ±1%: {s1/n*100:5.1f}%   ±5%: {s5/n*100:5.1f}%   ±10%: {s10/n*100:5.1f}%")
    if errors_pct:
        print(f"  err  mean: {statistics.mean(errors_pct):6.3f}%  median: {statistics.median(errors_pct):6.3f}%  max: {max(errors_pct):6.2f}%")
    print(f"  pattern usage:")
    for k in ["single_b1", "single_b2", "simultaneous", "sequential", "no_motion"]:
        c = patterns[k]
        print(f"    {k:12s}: {c:4d}  ({c/n*100:5.1f}%)")


def load(path: Path):
    if not path.exists(): return None
    with open(path) as f:
        return json.load(f)


def main():
    # paper baseline (init=180,100, 100K model from paper)
    base = load(PAPER_BASELINE)
    if base:
        summarize(base, "PAPER 100K (B1=180, B2=100)", init_b1=180.0, init_b2=100.0)
    else:
        print(f"[warn] paper baseline not found at {PAPER_BASELINE}")

    # mixed-init model evaluated at each config
    if not RUNS_ROOT.exists():
        print(f"\n[info] no runs/ yet at {RUNS_ROOT}"); return
    for cfg in sorted(RUNS_ROOT.iterdir()):
        latest = cfg / "results_latest.json"
        if latest.exists():
            data = load(latest)
            # init values are saved per-record by validate_mixed.py
            ib1 = data[0].get("init_b1", 180.0) if data else 180.0
            ib2 = data[0].get("init_b2", 100.0) if data else 100.0
            summarize(data, f"MIXED model @ {cfg.name}", init_b1=ib1, init_b2=ib2)

    print("\n[interpretation]")
    print("  If MIXED@default favors single_b2 AND MIXED@mirror favors single_b1")
    print("    -> preference is ADAPTIVE -> defends 'agentic' claim.")
    print("  If MIXED@mirror still favors single_b2")
    print("    -> preference is hardwired bias; reframe paper accordingly.")


if __name__ == "__main__":
    main()
