"""
Recompute policy entropy H(P_runtime) and KL divergence D_KL(P_runtime || P_train)
directly from the raw runtime histograms in validation_results_*_2000cases.json.

Driven by Reviewer 1 Concern 5 (R1#5a/b): the originally reported entropy
(1.38, 1.21, 0.89 nats) and KL (0.18, 0.22, 0.31 nats) values are inconsistent
with the raw histograms in Figure 6b. This script recomputes both quantities
using the same actuation-pattern classifier (b1_time>0 / b2_time>0, simultaneous
vs sequential split by |b1_time - b2_time| < 0.01) that produced Figure 6b.

Outputs:
- printed table for direct copy into Supplementary Table S1 and Response letter
- JSON file `entropy_kl_recomputed.json` for reproducibility
"""

import json
import math
from pathlib import Path
from tqdm import tqdm

CODE_DIR = Path(__file__).resolve().parent
OUT_PATH = CODE_DIR / "entropy_kl_recomputed.json"

VALIDATION_FILES = {
    "1K":   "validation_results_1k_2000cases.json",
    "10K":  "validation_results_10k_2000cases.json",
    "100K": "validation_results_100k_2000cases.json",
}

CLASSES = ["single_b1", "single_b2", "simultaneous", "sequential"]

# Training mixture per manuscript Section 4.4 (line 1416):
# 60% single-bank (30/30 split), 30% simultaneous, 10% sequential
P_TRAIN = {
    "single_b1":    0.30,
    "single_b2":    0.30,
    "simultaneous": 0.30,
    "sequential":   0.10,
}

# Small smoothing for any class missing entirely from a runtime histogram
# (avoids -inf in KL). 1e-3 matches the convention noted in the manuscript revision.
EPS = 1e-3


def classify(params):
    """Same logic as classify_scenario_type in plot.py."""
    if not isinstance(params, dict) or not params:
        return "parsing_failure"
    b1_active = params.get("b1_time", 0) > 0
    b2_active = params.get("b2_time", 0) > 0
    if b1_active and not b2_active:
        return "single_b1"
    if b2_active and not b1_active:
        return "single_b2"
    if b1_active and b2_active:
        if abs(params.get("b1_time", 0) - params.get("b2_time", 0)) < 0.01:
            return "simultaneous"
        return "sequential"
    return "none_active"


def histogram(records):
    """Count actuation classes over a list of validation records."""
    counts = {c: 0 for c in CLASSES}
    other = 0
    for rec in records:
        cls = classify(rec.get("predicted_params"))
        if cls in counts:
            counts[cls] += 1
        else:
            other += 1
    total = sum(counts.values())  # restrict to in-taxonomy attempts
    return counts, other, total


def entropy(probs):
    """H(P) = -sum p log p, with 0 log 0 := 0, in NATS."""
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log(p)
    return h


def kl_divergence(p_runtime, p_train, eps=EPS):
    """D_KL(P || Q) = sum_i P_i log(P_i / Q_i), in NATS.

    Empty runtime bins are smoothed by eps and the distribution is renormalized
    before the sum; this matches the convention announced in the revised text.
    """
    smoothed = [max(p, eps) for p in p_runtime]
    s = sum(smoothed)
    smoothed = [p / s for p in smoothed]
    kl = 0.0
    for p, q in zip(smoothed, p_train):
        if p > 0:
            kl += p * math.log(p / q)
    return kl


def main():
    p_train_vec = [P_TRAIN[c] for c in CLASSES]
    h_max = math.log(len(CLASSES))  # ln 4

    results = {}
    for scale, fname in tqdm(VALIDATION_FILES.items(), desc="Scales"):
        path = CODE_DIR / fname
        with open(path) as f:
            records = json.load(f)

        counts, other, total = histogram(records)
        probs = [counts[c] / total for c in CLASSES] if total > 0 else [0.0] * 4

        h = entropy(probs)
        kl = kl_divergence(probs, p_train_vec)

        results[scale] = {
            "n_records": len(records),
            "n_in_taxonomy": total,
            "n_other": other,  # parsing_failure / none_active
            "counts": counts,
            "probs": dict(zip(CLASSES, probs)),
            "H_nats": h,
            "H_max_nats": h_max,
            "KL_nats": kl,
        }

    # Print a compact table for copy-paste
    print()
    print(f"{'Scale':>6}  {'H(P)':>8}  {'H_max':>8}  {'KL(P||Q)':>10}  | counts")
    print("-" * 72)
    for scale, r in results.items():
        counts_str = ", ".join(f"{c}={r['counts'][c]}" for c in CLASSES)
        print(f"{scale:>6}  {r['H_nats']:>8.4f}  {r['H_max_nats']:>8.4f}  {r['KL_nats']:>10.4f}  | {counts_str}")
    print()
    print("Probabilities (single_b1, single_b2, simultaneous, sequential):")
    for scale, r in results.items():
        probs_str = ", ".join(f"{r['probs'][c]:.4f}" for c in CLASSES)
        print(f"  {scale:>5}: [{probs_str}]  (n_in_taxonomy={r['n_in_taxonomy']}, n_other={r['n_other']})")
    print()
    print(f"Training distribution P_train (single_b1, single_b2, simultaneous, sequential):")
    print(f"        [{', '.join(f'{p:.2f}' for p in p_train_vec)}]")
    print()

    # KL monotonicity check (for Branch A vs Branch B in Response letter Concern 5(b))
    kls = [results[s]["KL_nats"] for s in ["1K", "10K", "100K"]]
    monotonic = kls[0] <= kls[1] <= kls[2]
    print(f"KL across scales: {kls[0]:.4f} -> {kls[1]:.4f} -> {kls[2]:.4f}")
    print(f"Monotonic increase: {monotonic}  -> "
          f"{'Branch A (KL increases monotonically)' if monotonic else 'Branch B (KL not monotonic)'}")
    print()

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
