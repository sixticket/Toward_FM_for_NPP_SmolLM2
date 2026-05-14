"""
Sample 50K stratified rows from the existing master_dataset_100K.csv
(originally generated at init=(B1=180, B2=100) by auto_run_minimal.py).

Stratification matches the 60/30/10 distribution that auto_run_minimal.py
generates: single 60% (split ~50/50 b1/b2), simultaneous 30%, sequential 10%.
We classify each existing row by inspecting which banks moved.

Output adds init_b1=180, init_b2=100 columns.
"""

import argparse
import csv
import math
import random
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent              # data_generation/
EXP_DIR = SCRIPT_DIR.parent                                # exp1_initial_rod_variation/
MY_DIR = EXP_DIR.parent.parent                             # My/
SOURCE_CSV = MY_DIR / "dataset" / "master_dataset_100K.csv"

# Original init that produced the source CSV
SRC_INIT_B1 = 180
SRC_INIT_B2 = 100

# Match auto_run_minimal.py weights:
#   single 60% (split 30/30 between single_b1 and single_b2)
#   simultaneous 30%, sequential 10%
TARGET_FRACTIONS = {
    "single_b1":   0.30,
    "single_b2":   0.30,
    "simultaneous": 0.30,
    "sequential":  0.10,
}


def classify(row) -> str:
    b1_moves = row["b1_speed"] > 0 and abs(row["b1_pos"] - SRC_INIT_B1) > 0
    b2_moves = row["b2_speed"] > 0 and abs(row["b2_pos"] - SRC_INIT_B2) > 0
    if not b1_moves and not b2_moves:
        return "no_motion"
    if b1_moves and not b2_moves:
        return "single_b1"
    if b2_moves and not b1_moves:
        return "single_b2"
    if math.isclose(row["b1_time"], row["b2_time"], abs_tol=0.05):
        return "simultaneous"
    return "sequential"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000, help="target sample size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str,
                    default=str(SCRIPT_DIR / "subset_default_init_50k.csv"))
    args = ap.parse_args()

    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"source CSV missing: {SOURCE_CSV}")

    print(f"[load] {SOURCE_CSV}", flush=True)
    df = pd.read_csv(SOURCE_CSV)
    print(f"  rows: {len(df)}")

    df["pattern"] = df.apply(classify, axis=1)
    print("\n[source distribution]")
    for k, v in df["pattern"].value_counts().items():
        print(f"  {k:12s}: {v:6d}  ({v/len(df)*100:5.1f}%)")

    # Sample stratified
    random.seed(args.seed)
    out_rows = []
    for pattern, frac in TARGET_FRACTIONS.items():
        target = int(round(args.n * frac))
        pool = df[df["pattern"] == pattern]
        if len(pool) < target:
            print(f"  [warn] {pattern}: only {len(pool)} available, requested {target}")
            target = len(pool)
        sampled = pool.sample(n=target, random_state=args.seed)
        out_rows.append(sampled)
        print(f"  sampled {pattern:12s}: {len(sampled):6d}")

    sub = pd.concat(out_rows, ignore_index=True).sample(frac=1, random_state=args.seed)
    sub = sub.drop(columns=["pattern"])
    sub.insert(0, "init_b2", SRC_INIT_B2)
    sub.insert(0, "init_b1", SRC_INIT_B1)

    out_path = Path(args.out)
    sub.to_csv(out_path, index=False)
    print(f"\n[done] wrote {len(sub)} rows -> {out_path}")


if __name__ == "__main__":
    main()
