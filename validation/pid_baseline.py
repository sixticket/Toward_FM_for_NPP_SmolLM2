"""
PID Baseline (Minimal Pre-Tuning) Validation Script - Final
- Calibration: 6 samples (Symmetric: +/- Small, Medium, Large)
- Validation: 2,000 cases with Fixed Kp
"""

import os
import numpy as np
import subprocess
import shutil
from pathlib import Path
import json
from datetime import datetime
from scipy import stats
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

# Paths (resolve relative to this script)
SCRIPT_DIR = Path(__file__).resolve().parent  # validation/
REPO_ROOT = SCRIPT_DIR.parent                  # repo root

KOMODO_EXECUTABLE = Path(os.environ.get("KOMODO_EXECUTABLE", str(REPO_ROOT / "komodo")))
TEMPLATE_FILE = REPO_ROOT / "data_generation" / "template"
VALIDATION_DIR = SCRIPT_DIR / "validation_runs_pid_final"
TUNING_FILE = SCRIPT_DIR / "pid_config.json"  # renamed in release

# Control Constants
DEFAULT_SPEED = 2.0
INIT_B1_POS = 180.0
INIT_B2_POS = 100.0

# ============================================================================
# Simulator Utilities
# ============================================================================

def create_input_file(template_path, scenario, output_path):
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found at: {template_path}")
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()
    new_content = template_content.format(**scenario)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

def run_komodo(input_path):
    work_dir = input_path.parent
    executable_path = str(KOMODO_EXECUTABLE)
    if not Path(executable_path).exists():
        return False, f"KOMODO executable not found at: {executable_path}"
    try:
        subprocess.run(
            [executable_path, input_path.name],
            cwd=str(work_dir),
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=60
        )
        return True, None
    except Exception as e:
        return False, str(e)

def parse_simulation_results(output_file):
    powers = []
    try:
        with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
            in_results = False
            for line in f:
                if "TRANSIENT RESULTS" in line:
                    in_results = True
                    next(f); next(f)
                    continue
                if in_results and line.strip().startswith('CPU time'): break
                if in_results and line.strip():
                    parts = line.split()
                    if len(parts) >= 4:
                        try: powers.append(float(parts[3]))
                        except: continue
        if not powers: return None
        return {'initial_power': powers[0], 'final_power': powers[-1]}
    except: return None

def cleanup_case_files(run_dir):
    try:
        if run_dir.exists(): shutil.rmtree(run_dir)
    except: pass

# ============================================================================
# 6-Shot Calibration (Symmetric)
# ============================================================================

def run_calibration_case(delta_step, case_id):
    """Run single simulation for calibration"""
    run_dir = VALIDATION_DIR / f"tune_{case_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    target_b2 = INIT_B2_POS + delta_step
    dist = abs(delta_step)
    time_val = max(0.1, dist / DEFAULT_SPEED)

    params = {
        'b1_pos': INIT_B1_POS, 'b1_time': 0.0, 'b1_speed': 0.0,
        'b2_pos': round(target_b2, 3), 'b2_time': round(time_val, 3),
        'b2_speed': DEFAULT_SPEED
    }

    input_path = run_dir / "input.inp"
    create_input_file(TEMPLATE_FILE, params, input_path)
    success, _ = run_komodo(input_path)

    power_delta = 0.0
    if success:
        res = parse_simulation_results(run_dir / "input.inp.out")
        if res:
            power_delta = res['final_power'] - res['initial_power']

    cleanup_case_files(run_dir)
    return power_delta

def get_or_create_tuned_kp(force_retune=False):
    """
    Run 6-Point Calibration to find average Kp.
    Test points: [10, -10, 30, -30, 50, -50] (Symmetric)
    """
    if not force_retune and TUNING_FILE.exists():
        with open(TUNING_FILE, 'r') as f:
            data = json.load(f)
            kp = data['kp_value']
            print("\n" + "=" * 60)
            print(f"♻️  Loaded Pre-Tuned Kp (6-Shot) from file: {kp:.4f}")
            print("=" * 60)
            return kp

    print("\n🔧 Starting Fast Calibration (6 Samples: Symmetric)...")

    # 6 Representative Points: Small, Medium, Large (Both +/-)
    test_deltas = [10, -10, 30, -30, 50, -50]

    x_power_deltas = []
    y_step_deltas = []

    pbar = tqdm(enumerate(test_deltas), total=len(test_deltas), desc="Calibrating", unit="step")

    for i, delta in pbar:
        power_delta = run_calibration_case(delta, i)

        if abs(power_delta) > 0.00001:
            x_power_deltas.append(power_delta)
            y_step_deltas.append(delta)
            pbar.set_postfix(step=f"{delta:+}", pwr=f"{power_delta:+.4f}")

    if len(x_power_deltas) < 3:
        print("⚠️ Too few valid points! Defaulting to Kp=200.0")
        best_kp = 200.0
    else:
        # Linear Regression to find Kp
        slope, _, r_value, _, _ = stats.linregress(x_power_deltas, y_step_deltas)
        best_kp = abs(slope)
        print(f"\n✅ 6-Shot Calibration Done. R^2: {r_value**2:.4f}, Kp: {best_kp:.4f}")

    with open(TUNING_FILE, 'w') as f:
        json.dump({
            'kp_value': best_kp,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'n_samples': len(x_power_deltas)
        }, f, indent=4)

    print(f"💾 Tuning result saved to: {TUNING_FILE}")
    return best_kp

# ============================================================================
# PID Calculation & Validation
# ============================================================================

def calculate_pid_control(target_initial, target_final, kp_value):
    power_error = target_final - target_initial
    delta_steps = power_error * kp_value

    b2_target = INIT_B2_POS + delta_steps
    b1_target = INIT_B1_POS

    # Saturation Logic
    if b2_target > 180.0:
        rem = b2_target - 180.0
        b2_target = 180.0
        b1_target = min(180.0, INIT_B1_POS + rem)
    elif b2_target < 0.0:
        rem = b2_target
        b2_target = 0.0
        b1_target = max(0.0, INIT_B1_POS + rem)

    dist_b1 = abs(b1_target - INIT_B1_POS)
    dist_b2 = abs(b2_target - INIT_B2_POS)

    b1_time = max(0.1, dist_b1 / DEFAULT_SPEED) if dist_b1 > 0 else 0.0
    b2_time = max(0.1, dist_b2 / DEFAULT_SPEED) if dist_b2 > 0 else 0.0

    return {
        'b1_pos': round(b1_target, 3), 'b1_time': round(b1_time, 3),
        'b1_speed': DEFAULT_SPEED if dist_b1 > 0 else 0.0,
        'b2_pos': round(b2_target, 3), 'b2_time': round(b2_time, 3),
        'b2_speed': DEFAULT_SPEED if dist_b2 > 0 else 0.0
    }

def generate_test_cases(num_cases=2000):
    test_cases = []
    base_cases = [
        (1.0, 1.05, "Small Increase 5%"), (1.0, 1.10, "Increase 10%"),
        (1.0, 1.20, "Medium Increase 20%"), (1.0, 1.30, "Large Increase 30%"),
        (1.0, 1.40, "Major Increase 40%"), (1.0, 1.50, "Massive Increase 50%"),
        (1.0, 0.95, "Small Decrease 5%"), (1.0, 0.90, "Decrease 10%"),
        (1.0, 0.80, "Medium Decrease 20%"), (1.0, 0.70, "Large Decrease 30%"),
        (1.0, 0.60, "Major Decrease 40%"), (1.0, 0.50, "Massive Decrease 50%"),
    ]
    test_cases.extend(base_cases)
    np.random.seed(42)
    remaining = num_cases - len(base_cases)
    for _ in range(remaining):
        final = np.random.uniform(0.5, 1.5)
        test_cases.append((1.0, round(final, 5), "Random"))
    return test_cases

def main():
    print("\n" + "=" * 80)
    print("PID Baseline Validation (6-Shot Symmetric Tuning -> 2000 Cases)")
    print("=" * 80)

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 6-SHOT TUNING
    tuned_kp = get_or_create_tuned_kp(force_retune=False)

    # 2. VALIDATION
    num_cases = 2000
    print(f"\n🚀 Validating 2,000 cases with FIXED Kp = {tuned_kp:.4f}...")

    test_cases = generate_test_cases(num_cases)
    results = []
    success_count_5 = 0

    pbar = tqdm(enumerate(test_cases, 1), total=num_cases, desc="Validating", unit="case")

    for i, (initial, final, desc) in pbar:
        control_params = calculate_pid_control(initial, final, tuned_kp)

        case_dir = VALIDATION_DIR / f"case_{i:04d}"
        case_dir.mkdir(parents=True, exist_ok=True)

        input_path = case_dir / "input.inp"
        create_input_file(TEMPLATE_FILE, control_params, input_path)

        success, _ = run_komodo(input_path)
        res = parse_simulation_results(case_dir / "input.inp.out") if success else None

        if res:
            actual_p = res['final_power']
            err = abs(actual_p - final)
            is_success_5 = err <= abs(final * 0.05)

            if is_success_5: success_count_5 += 1

            results.append({
                'case_id': i,
                'validation_success_5': is_success_5,
                'validation_success_10': err <= abs(final * 0.10),
                'final_error': err,
                'target_final': final,
                'actual_final': actual_p
            })

            current_acc = (success_count_5 / i) * 100
            pbar.set_postfix(Acc_5pct=f"{current_acc:.1f}%", Last_Err=f"{err:.3f}")

        cleanup_case_files(case_dir)

    success_5 = sum(1 for r in results if r.get('validation_success_5'))
    print(f"\n📊 Final Accuracy (±5%): {success_5/num_cases*100:.1f}%")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = VALIDATION_DIR / f"pid_final_results_{timestamp}.json"

    # Save copy for easy plotting
    fixed_save_path = VALIDATION_DIR / "pid_final_results.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    shutil.copy(save_path, fixed_save_path)

    print(f"💾 Results saved: {save_path}")
    print(f"💾 (Plotting Copy): {fixed_save_path}")

if __name__ == "__main__":
    main()