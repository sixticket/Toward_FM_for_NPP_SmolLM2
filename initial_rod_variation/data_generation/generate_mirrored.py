"""
Generate 50K scenarios at init=(B1=100, B2=180) — the MIRROR of the paper's
default (B1=180, B2=100).

Mirror logic: in the original auto_run_minimal.py, B1 is the "coarse bank"
(starts fully withdrawn at 180, can only move down) and B2 is the "fine bank"
(starts at 100, has the steep reactivity gradient). For the mirror init we
swap the ROLES of B1/B2 so the dataset has the same structural difficulty:
B2 becomes coarse (at max 180), B1 becomes fine (at middle 100).

This means the sampling ranges and speed ranges for single_b1 here mirror
the single_b2 ranges of the original (and vice versa).

Output adds init_b1=100, init_b2=180 columns.

Usage (from WSL):
    python generate_mirrored.py --n 50000
"""

import argparse
import csv
import multiprocessing
import os
import random
import shutil
import subprocess
import time
from functools import partial
from pathlib import Path

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent          # data_generation/
EXP_DIR = SCRIPT_DIR.parent                            # exp1_initial_rod_variation/
MY_DIR = EXP_DIR.parent.parent                         # My/
KOMODO_DIR = MY_DIR.parent                             # KOMODO/

KOMODO_EXECUTABLE = KOMODO_DIR / "komodo"
TEMPLATE_FILE = EXP_DIR / "validation" / "template_init_var"  # parameterized template

# Mirror init
INIT_B1_POS = 100
INIT_B2_POS = 180

WORK_DIR = SCRIPT_DIR / "scratch_mirror"

# ============================================================================
# Mirrored scenario generator
# ============================================================================
# Original auto_run_minimal.py sampling:
#   single_b1: b1 in [60,180], speed in [0.5, 3.0]   (coarse bank)
#   single_b2: b2 in [70,160], speed in [0.2, 2.0]   (fine bank)
#   simultaneous: b1 in [80,180], b1_speed [0.5,2.5]; b2 in [70,140], b2_speed [0.2,1.5]
#   sequential:   b2 in [80,150] first, then b1 in [70,180] later
#
# MIRROR (B1 is now fine, B2 is now coarse), so swap b1<->b2 ranges/speeds:

def generate_random_scenario():
    scenario_type = random.choices(
        ["single", "simultaneous", "sequential"],
        weights=[60, 30, 10],
    )[0]

    b1_pos, b1_time, b1_speed = INIT_B1_POS, 0.0, 0.0
    b2_pos, b2_time, b2_speed = INIT_B2_POS, 0.0, 0.0

    if scenario_type == "single":
        if random.random() < 0.5:
            # single_b1 — now mirrors original single_b2 (fine bank)
            b1_pos = random.randint(70, 160)
            b1_time = round(random.uniform(2.0, 15.0), 1)
            b1_speed = round(random.uniform(0.2, 2.0), 1)
        else:
            # single_b2 — now mirrors original single_b1 (coarse bank)
            b2_pos = random.randint(60, 180)
            b2_time = round(random.uniform(2.0, 15.0), 1)
            b2_speed = round(random.uniform(0.5, 3.0), 1)

    elif scenario_type == "simultaneous":
        start_time = round(random.uniform(2.0, 10.0), 1)
        # mirror: b1 takes original b2 ranges, b2 takes original b1 ranges
        b1_pos = random.randint(70, 140)
        b1_time = start_time
        b1_speed = round(random.uniform(0.2, 1.5), 1)
        b2_pos = random.randint(80, 180)
        b2_time = start_time
        b2_speed = round(random.uniform(0.5, 2.5), 1)

    else:  # sequential
        # mirror: now B1 (fine bank) moves first, B2 (coarse) moves later
        b1_pos = random.randint(80, 150)
        b1_time = round(random.uniform(2.0, 8.0), 1)
        b1_speed = round(random.uniform(0.3, 2.0), 1)
        b2_pos = random.randint(70, 180)
        b2_time = b1_time + round(random.uniform(3.0, 8.0), 1)
        b2_speed = round(random.uniform(0.5, 3.0), 1)

    return {
        "b1_pos": b1_pos, "b1_time": b1_time, "b1_speed": b1_speed,
        "b2_pos": b2_pos, "b2_time": b2_time, "b2_speed": b2_speed,
    }


# ============================================================================
# Simulator interface (mirrors auto_run_minimal.py)
# ============================================================================

def render_input(template_path: Path, scenario: dict, output_path: Path):
    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()
    rendered = content.format(
        init_b1=f"{INIT_B1_POS}.", init_b2=f"{INIT_B2_POS}.",
        **scenario,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered)


def run_komodo(input_path: Path):
    try:
        subprocess.run(
            [str(KOMODO_EXECUTABLE), input_path.name],
            cwd=str(input_path.parent),
            check=True,
            capture_output=True, text=True, encoding="utf-8",
            timeout=60,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def parse_summary(out_file: Path):
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
            return {"initial": None, "final": None, "peak": None}
        return {"initial": powers[0], "final": powers[-1], "peak": max(powers)}
    except FileNotFoundError:
        return {"initial": None, "final": None, "peak": None}


def run_single(i: int, dataset_dir: str, template_file: str):
    run_name = f"run_{i:06d}"
    scenario = generate_random_scenario()
    inp = Path(dataset_dir) / f"{run_name}.inp"
    out = Path(dataset_dir) / f"{run_name}.inp.out"
    vtk = Path(dataset_dir) / f"{run_name}.inp.vtk"

    try:
        render_input(Path(template_file), scenario, inp)
        if not run_komodo(inp):
            for p in (inp, out, vtk):
                if p.exists():
                    try: p.unlink()
                    except OSError: pass
            return None
        summary = parse_summary(out)
        for p in (inp, out, vtk):
            if p.exists():
                try: p.unlink()
                except OSError: pass
        time.sleep(0.05)
        return {"run_name": run_name, "scenario": scenario, "summary": summary}
    except Exception:
        for p in (inp, out, vtk):
            if p.exists():
                try: p.unlink()
                except OSError: pass
        return None


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=str,
                    default=str(SCRIPT_DIR / "subset_mirror_init_50k.csv"))
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--restart", action="store_true",
                    help="Discard any existing partial CSV and start fresh from sim 1")
    args = ap.parse_args()

    random.seed(args.seed)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out)
    header = ["run_id", "init_b1", "init_b2",
              "b1_pos", "b1_time", "b1_speed", "b2_pos", "b2_time", "b2_speed",
              "initial_power", "final_power", "peak_power"]

    # ---- resume detection ----
    n_done = 0
    if out_path.exists() and not args.restart:
        with open(out_path, "r", encoding="utf-8") as f:
            n_done = max(0, sum(1 for _ in f) - 1)  # subtract header
        if n_done > 0:
            print(f"[resume] found {out_path.name} with {n_done} existing rows; resuming from sim {n_done + 1}", flush=True)
    if n_done == 0:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    n_remaining = args.n - n_done
    if n_remaining <= 0:
        print(f"[skip] {out_path.name} already has {n_done} rows >= target {args.n}", flush=True)
        shutil.rmtree(WORK_DIR, ignore_errors=True)
        return

    print(f"\n[gen-mirror] target N={args.n}  done={n_done}  remaining={n_remaining}  init=(B1={INIT_B1_POS}, B2={INIT_B2_POS})  workers={args.workers}", flush=True)

    func = partial(run_single, dataset_dir=str(WORK_DIR), template_file=str(TEMPLATE_FILE))
    successful = n_done

    # ---- incremental write: every completed sim is flushed to disk ----
    try:
        with open(out_path, "a", newline="", encoding="utf-8") as fcsv:
            w = csv.writer(fcsv)
            with multiprocessing.Pool(processes=args.workers) as pool:
                pbar = tqdm(
                    pool.imap_unordered(func, range(n_done + 1, args.n + 1)),
                    total=n_remaining, desc="mirror sims", ncols=100,
                )
                for r in pbar:
                    if r is None:
                        continue
                    s = r["scenario"]
                    w.writerow([
                        r["run_name"], INIT_B1_POS, INIT_B2_POS,
                        s["b1_pos"], s["b1_time"], s["b1_speed"],
                        s["b2_pos"], s["b2_time"], s["b2_speed"],
                        r["summary"].get("initial"), r["summary"].get("final"),
                        r["summary"].get("peak"),
                    ])
                    fcsv.flush()
                    successful += 1
                    pbar.set_postfix(saved=successful)
    except KeyboardInterrupt:
        print(f"\n[interrupted] partial CSV preserved: {successful}/{args.n} rows -> {out_path}", flush=True)
        shutil.rmtree(WORK_DIR, ignore_errors=True)
        raise

    # Cleanup scratch
    shutil.rmtree(WORK_DIR, ignore_errors=True)

    print(f"\n[done] {successful}/{args.n} succeeded -> {out_path}")


if __name__ == "__main__":
    main()
