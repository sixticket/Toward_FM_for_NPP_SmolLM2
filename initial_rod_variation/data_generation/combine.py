"""
Combine the two stratified subsets into a single mixed-init training CSV:
  subset_default_init_50k.csv  (init_b1=180, init_b2=100)
  subset_mirror_init_50k.csv   (init_b1=100, init_b2=180)
->
  master_dataset_mixed_init_100k.csv

Shuffles rows together so training batches mix inits.
"""

import argparse
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULTS = {
    "default": SCRIPT_DIR / "subset_default_init_50k.csv",
    "mirror":  SCRIPT_DIR / "subset_mirror_init_50k.csv",
    "out":     SCRIPT_DIR / "master_dataset_mixed_init_100k.csv",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--default_csv", type=str, default=str(DEFAULTS["default"]))
    ap.add_argument("--mirror_csv",  type=str, default=str(DEFAULTS["mirror"]))
    ap.add_argument("--out", type=str, default=str(DEFAULTS["out"]))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    p_def = Path(args.default_csv)
    p_mir = Path(args.mirror_csv)
    if not p_def.exists():
        raise FileNotFoundError(f"missing: {p_def} (run sample_existing.py first)")
    if not p_mir.exists():
        raise FileNotFoundError(f"missing: {p_mir} (run generate_mirrored.py first)")

    df_def = pd.read_csv(p_def)
    df_mir = pd.read_csv(p_mir)
    print(f"[load] default: {len(df_def)} rows | mirror: {len(df_mir)} rows")

    # ensure consistent columns
    expected = ["init_b1", "init_b2", "b1_pos", "b1_time", "b1_speed",
                "b2_pos", "b2_time", "b2_speed",
                "initial_power", "final_power", "peak_power"]
    for col in expected:
        if col not in df_def.columns:
            raise ValueError(f"default CSV missing column: {col}")
        if col not in df_mir.columns:
            raise ValueError(f"mirror CSV missing column: {col}")

    # rebuild run_id to be unique across the union
    if "run_id" not in df_def.columns:
        df_def.insert(0, "run_id", [f"def_{i:05d}" for i in range(len(df_def))])
    else:
        df_def["run_id"] = [f"def_{i:05d}" for i in range(len(df_def))]
    if "run_id" not in df_mir.columns:
        df_mir.insert(0, "run_id", [f"mir_{i:05d}" for i in range(len(df_mir))])
    else:
        df_mir["run_id"] = [f"mir_{i:05d}" for i in range(len(df_mir))]

    cols = ["run_id"] + expected
    combined = pd.concat([df_def[cols], df_mir[cols]], ignore_index=True)
    combined = combined.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    out_path = Path(args.out)
    combined.to_csv(out_path, index=False)
    print(f"[done] combined {len(combined)} rows -> {out_path}")
    print(f"  init=(180,100) rows: {(combined.init_b1==180).sum()}")
    print(f"  init=(100,180) rows: {(combined.init_b1==100).sum()}")


if __name__ == "__main__":
    main()
