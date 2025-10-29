"""
Model Predictions → Simulator Validation Script (V7.1 Simple)
- Predict control-rod parameters using the Phase 2 V7.1 Simple model
- Run the KOMODO simulator with the predicted parameters
- Compare achieved final power vs. target final power
- Automatically delete *.inp/*.out files after validation (keep JSON only)
- Five-tier accuracy analysis (±1%, ±2%, ±3%, ±5%, ±10%)
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import torch
import re
import subprocess
import shutil
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
from datetime import datetime

# ============================================================================
# Settings
# ============================================================================

SCRIPT_DIR = Path(__file__).parent  # validation/
BASE_DIR = SCRIPT_DIR.parent         # My/
KOMODO_DIR = BASE_DIR.parent         # KOMODO/

# Model paths (V7.1 Simple!)
TRAINING_DIR = BASE_DIR / "training"
PHASE1_MODEL_PATH = TRAINING_DIR / "models/smollm2_unsupervised_numeric/final_model"
PHASE2_MODEL_PATH = TRAINING_DIR / "models/smollm2_supervised_lora_v7_numeric_simple/final_model"  # V7.1!

# Simulator settings
KOMODO_EXECUTABLE = KOMODO_DIR / "komodo"
TEMPLATE_FILE = BASE_DIR / "template"
VALIDATION_DIR = SCRIPT_DIR / "validation_runs_v7_simple"  # V7.1-only directory

# Inference settings
INFERENCE_TEMPERATURE = 0.05
INFERENCE_TOP_P = 0.9
INFERENCE_MAX_TOKENS = 50


# ============================================================================
# Load model
# ============================================================================

def load_model():
    """Load Phase 2 V7.1 Simple model (Phase 1 base + LoRA)."""
    print("=" * 80)
    print("Loading model... (V7.1 Simple)")
    print("=" * 80)

    # Load Phase 1 base model
    print(f"\n✓ Phase 1 model: {PHASE1_MODEL_PATH}")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(PHASE1_MODEL_PATH),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # Load LoRA adapter
    print(f"✓ Phase 2 LoRA (V7.1 Simple): {PHASE2_MODEL_PATH}")
    model = PeftModel.from_pretrained(base_model, str(PHASE2_MODEL_PATH))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(PHASE2_MODEL_PATH))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    print("\n✓ V7.1 Simple model loaded!")
    print("  - No arrows, commas only")
    print("  - Format: [1.0, 1.5, 180, 0.0, ...]")
    print("=" * 80)

    return model, tokenizer


# ============================================================================
# Model prediction - V7.1 Simple
# ============================================================================

def parse_prediction_simple(prediction_text):
    """
    Parse the V7.1 Simple format: extract 8 numbers.
    Example: "[1.0, 1.5, 180, 0.0, 0.0, 100, 0.0, 0.0]"
    First two are inputs; last six are outputs.
    """
    numbers = re.findall(r'[\d.]+', prediction_text)

    if len(numbers) < 8:
        return None

    try:
        # Take the last six as control-rod parameters
        values = [float(x) for x in numbers[-6:]]
        return {
            'b1_pos': values[0],
            'b1_time': values[1],
            'b1_speed': values[2],
            'b2_pos': values[3],
            'b2_time': values[4],
            'b2_speed': values[5]
        }
    except Exception:
        return None


def predict_rod_parameters(model, tokenizer, initial_power, final_power):
    """Predict control-rod parameters with the model (V7.1 Simple)."""
    # V7.1 format: ends with a comma
    prompt = f"[{initial_power}, {final_power},"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=INFERENCE_MAX_TOKENS,
            temperature=INFERENCE_TEMPERATURE,
            top_p=INFERENCE_TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Parse
    parsed = parse_prediction_simple(generated)

    return parsed, generated


# ============================================================================
# Run simulator
# ============================================================================

def create_input_file(template_path, scenario, output_path):
    """Create a KOMODO input file from a template."""
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()

    new_content = template_content.format(**scenario)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(new_content)


def run_komodo(input_path):
    """Execute the KOMODO simulator."""
    work_dir = input_path.parent
    input_filename = input_path.name

    executable_path = str(KOMODO_EXECUTABLE)

    try:
        subprocess.run(
            [executable_path, input_filename],
            cwd=str(work_dir),
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=60
        )
        return True, None
    except subprocess.TimeoutExpired:
        return False, "Simulation timeout (exceeded 60s)"
    except subprocess.CalledProcessError as e:
        return False, f"Simulation failed:\n{e.stderr}"
    except FileNotFoundError:
        return False, f"KOMODO executable not found: {executable_path}"


def parse_simulation_results(output_file):
    """Extract power/time series from the simulator output file."""
    times = []
    powers = []

    try:
        with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
            in_results_section = False
            for line in f:
                if "TRANSIENT RESULTS" in line:
                    in_results_section = True
                    next(f, None)
                    next(f, None)
                    continue

                if in_results_section and line.strip().startswith('CPU time breakdown'):
                    break

                if in_results_section and line.strip():
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            time_val = float(parts[1])
                            power_val = float(parts[3])
                            times.append(time_val)
                            powers.append(power_val)
                        except (ValueError, IndexError):
                            continue

        if not powers:
            return None

        return {
            'initial_power': powers[0],
            'final_power': powers[-1],
            'peak_power': max(powers),
            'times': times,
            'powers': powers
        }

    except FileNotFoundError:
        return None


# ============================================================================
# File cleanup
# ============================================================================

def cleanup_case_files(run_dir):
    """Delete *.inp/*.out files in a case directory (remove the whole folder)."""
    try:
        if run_dir.exists():
            shutil.rmtree(run_dir)
    except Exception as e:
        print(f"  ⚠️ Failed to delete files: {e}")


# ============================================================================
# Validation pipeline
# ============================================================================

def validate_single_case(model, tokenizer, initial_power, final_power, case_id, total_cases, desc=""):
    """Validate a single case with the simulator."""
    print(f"\n{'='*70}", flush=True)
    print(f"Progress: [{case_id:4d}/{total_cases}] ({case_id/total_cases*100:.1f}%) - {desc}", flush=True)
    print(f"Target: {initial_power} → {final_power}", flush=True)
    print(f"{'='*70}", flush=True)

    # 1) Model prediction
    predicted_params, full_prediction = predict_rod_parameters(
        model, tokenizer, initial_power, final_power
    )

    if predicted_params is None:
        print(f"  ❌ Parsing failed!", flush=True)
        return {
            'case_id': case_id,
            'target_initial': initial_power,
            'target_final': final_power,
            'description': desc,
            'parsing_success': False,
            'full_prediction': full_prediction,
            'error': 'Parsing failed'
        }

    print(f"  ✅ Prediction success: {full_prediction}", flush=True)

    # 2) Create input file
    run_dir = VALIDATION_DIR / f"case_{case_id:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    input_file = run_dir / "input.inp"
    create_input_file(TEMPLATE_FILE, predicted_params, input_file)

    # 3) Run simulator
    success, error = run_komodo(input_file)

    if not success:
        print(f"  ❌ Simulation failed: {error}", flush=True)
        cleanup_case_files(run_dir)  # delete even if it fails
        return {
            'case_id': case_id,
            'target_initial': initial_power,
            'target_final': final_power,
            'description': desc,
            'parsing_success': True,
            'predicted_params': predicted_params,
            'full_prediction': full_prediction,
            'simulation_success': False,
            'error': error
        }

    # 4) Parse simulator results
    output_file = run_dir / "input.inp.out"
    results = parse_simulation_results(output_file)

    if results is None:
        print(f"  ❌ Result parsing failed", flush=True)
        cleanup_case_files(run_dir)  # delete even if it fails
        return {
            'case_id': case_id,
            'target_initial': initial_power,
            'target_final': final_power,
            'description': desc,
            'parsing_success': True,
            'predicted_params': predicted_params,
            'full_prediction': full_prediction,
            'simulation_success': True,
            'result_parsing_success': False,
            'error': 'Result parsing failed'
        }

    # 5) Validation (five tiers)
    final_error = abs(results['final_power'] - final_power)
    tolerance_1 = abs(final_power) * 0.01
    tolerance_2 = abs(final_power) * 0.02
    tolerance_3 = abs(final_power) * 0.03
    tolerance_5 = abs(final_power) * 0.05
    tolerance_10 = abs(final_power) * 0.10

    success_1 = final_error <= tolerance_1
    success_2 = final_error <= tolerance_2
    success_3 = final_error <= tolerance_3
    success_5 = final_error <= tolerance_5
    success_10 = final_error <= tolerance_10

    # Emoji by accuracy level
    if success_1:
        status = "🎯"  # very accurate
    elif success_2:
        status = "✅"  # accurate
    elif success_3:
        status = "👍"  # good
    elif success_5:
        status = "📊"  # acceptable
    elif success_10:
        status = "⚠️"  # lenient
    else:
        status = "❌"  # fail

    print(f"  {status} Final power: {results['final_power']:.3f} (error: {final_error:.4f})", flush=True)

    # 6) Clean up (also for successful cases)
    cleanup_case_files(run_dir)
    print(f"  🗑️  Temporary files removed", flush=True)

    return {
        'case_id': case_id,
        'target_initial': initial_power,
        'target_final': final_power,
        'description': desc,
        'parsing_success': True,
        'predicted_params': predicted_params,
        'full_prediction': full_prediction,
        'simulation_success': True,
        'result_parsing_success': True,
        'actual_initial': results['initial_power'],
        'actual_final': results['final_power'],
        'actual_peak': results['peak_power'],
        'final_error': final_error,
        'validation_success_1': success_1,
        'validation_success_2': success_2,
        'validation_success_3': success_3,
        'validation_success_5': success_5,
        'validation_success_10': success_10
    }


def generate_test_cases(num_cases=100):
    """Generate test cases."""
    test_cases = []

    # Core cases (always include)
    base_cases = [
        (1.0, 1.05, "Small increase 5%"),
        (1.0, 1.10, "Increase 10%"),
        (1.0, 1.20, "Medium increase 20%"),
        (1.0, 1.30, "Large increase 30%"),
        (1.0, 1.40, "Big increase 40%"),
        (1.0, 1.50, "Very large increase 50%"),

        (1.0, 0.95, "Small decrease 5%"),
        (1.0, 0.90, "Decrease 10%"),
        (1.0, 0.80, "Medium decrease 20%"),
        (1.0, 0.70, "Large decrease 30%"),
        (1.0, 0.60, "Big decrease 40%"),
        (1.0, 0.50, "Very large decrease 50%"),
    ]

    test_cases.extend(base_cases)

    # Fill the rest with random samples
    np.random.seed(42)
    remaining = num_cases - len(base_cases)

    for _ in range(remaining):
        # Sample final power from 0.5 ~ 1.5
        final = np.random.uniform(0.5, 1.5)
        change = (final - 1.0) / 1.0 * 100
        desc = f"{'Increase' if final > 1.0 else 'Decrease'} {abs(change):.1f}%"
        test_cases.append((1.0, round(final, 5), desc))

    return test_cases


def run_validation_suite(model, tokenizer, num_cases=100):
    """Validate multiple test cases."""
    print("\n" + "🎯 " * 20)
    print(f"V7.1 Simple model → Simulator validation ({num_cases} cases)")
    print("🎯 " * 20)
    print("\n💡 After validation, *.inp/*.out files are auto-deleted (only JSON is kept)")

    test_cases = generate_test_cases(num_cases)

    results = []
    total = len(test_cases)
    for i, (initial, final, desc) in enumerate(test_cases, 1):
        result = validate_single_case(model, tokenizer, initial, final, i, total, desc)
        results.append(result)

        # Interim stats every 50 cases
        if i % 50 == 0:
            temp_parsing = sum(1 for r in results if r.get('parsing_success', False))
            temp_sim = sum(1 for r in results if r.get('simulation_success', False))
            temp_val_1 = sum(1 for r in results if r.get('validation_success_1', False))
            temp_val_2 = sum(1 for r in results if r.get('validation_success_2', False))
            temp_val_5 = sum(1 for r in results if r.get('validation_success_5', False))
            print(f"\n  === Interim stats ({i} cases) ===", flush=True)
            print(f"  Parsing: {temp_parsing}/{i} ({temp_parsing/i*100:.1f}%)", flush=True)
            print(f"  Simulation: {temp_sim}/{i} ({temp_sim/i*100:.1f}%)", flush=True)
            print(f"  Validation (±1%): {temp_val_1}/{i} ({temp_val_1/i*100:.1f}%)", flush=True)
            print(f"  Validation (±2%): {temp_val_2}/{i} ({temp_val_2/i*100:.1f}%)", flush=True)
            print(f"  Validation (±5%): {temp_val_5}/{i} ({temp_val_5/i*100:.1f}%)\n", flush=True)

    # Overall stats
    print("\n\n" + "=" * 80)
    print("📊 Overall Validation Results (V7.1 Simple)")
    print("=" * 80)

    total = len(results)
    parsing_success = sum(1 for r in results if r.get('parsing_success', False))
    simulation_success = sum(1 for r in results if r.get('simulation_success', False))
    validation_success_1 = sum(1 for r in results if r.get('validation_success_1', False))
    validation_success_2 = sum(1 for r in results if r.get('validation_success_2', False))
    validation_success_3 = sum(1 for r in results if r.get('validation_success_3', False))
    validation_success_5 = sum(1 for r in results if r.get('validation_success_5', False))
    validation_success_10 = sum(1 for r in results if r.get('validation_success_10', False))

    print(f"\nTotal cases:            {total}")
    print(f"Parsing success:        {parsing_success} ({parsing_success / total * 100:.1f}%)")

    if parsing_success > 0:
        print(f"Simulation success:      {simulation_success} ({simulation_success / parsing_success * 100:.1f}% of parsed)")

    if simulation_success > 0:
        print(f"\n🎯 Success rates (five tiers):")
        print(f"  ±1%  (very accurate): {validation_success_1} ({validation_success_1 / simulation_success * 100:.1f}%)")
        print(f"  ±2%  (accurate):      {validation_success_2} ({validation_success_2 / simulation_success * 100:.1f}%)")
        print(f"  ±3%  (good):          {validation_success_3} ({validation_success_3 / simulation_success * 100:.1f}%)")
        print(f"  ±5%  (acceptable):    {validation_success_5} ({validation_success_5 / simulation_success * 100:.1f}%)")
        print(f"  ±10% (lenient):       {validation_success_10} ({validation_success_10 / simulation_success * 100:.1f}%)")

    # V7 vs V7.1 comparison
    print(f"\n📊 V7 vs V7.1 comparison:")
    print(f"  V7 parsing:         90.0%")
    print(f"  V7.1 parsing:       {parsing_success / total * 100:.1f}% ⭐")
    if simulation_success > 0:
        print(f"  V7 validation (±5%): 22.2%")
        print(f"  V7.1 validation (±5%): {validation_success_5 / simulation_success * 100:.1f}%")
        print(f"  V7 validation (±10%): 66.7%")
        print(f"  V7.1 validation (±10%): {validation_success_10 / simulation_success * 100:.1f}%")

    # Error stats
    valid_results = [r for r in results if r.get('final_error') is not None]
    if valid_results:
        final_errors = [r['final_error'] for r in valid_results]
        print(f"\nFinal power error stats:")
        print(f"  Mean:       {np.mean(final_errors):.4f}")
        print(f"  Median:     {np.median(final_errors):.4f}")
        print(f"  Max:        {np.max(final_errors):.4f}")
        print(f"  Min:        {np.min(final_errors):.4f}")
        print(f"  Std. dev.:  {np.std(final_errors):.4f}")

    # Success rates by range
    print(f"\nSuccess rates by target range (±5% criterion):")
    ranges = [
        ("Small (±10%)", lambda r: abs(r['target_final'] - 1.0) <= 0.1),
        ("Medium (±30%)", lambda r: 0.1 < abs(r['target_final'] - 1.0) <= 0.3),
        ("Large (±50%)", lambda r: abs(r['target_final'] - 1.0) > 0.3),
    ]

    for range_name, condition in ranges:
        range_results = [r for r in results if r.get('target_final') and condition(r)]
        if range_results:
            range_success = sum(1 for r in range_results if r.get('validation_success_5', False))
            print(f"  {range_name:15s}: {range_success}/{len(range_results)} ({range_success/len(range_results)*100:.1f}%)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = VALIDATION_DIR / f"validation_results_v7_simple_{num_cases}cases_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved: {results_file}")
    print(f"💾 All temporary files have been deleted (JSON only is kept)")

    # Summarize failed/borderline cases
    failed_cases = [r for r in results if not r.get('validation_success_5', False)]
    if failed_cases and len(failed_cases) <= 20:
        print(f"\n⚠️ Failed/borderline cases ({len(failed_cases)}):")
        for r in failed_cases[:20]:
            if r.get('final_error'):
                print(
                    f"  Case {r['case_id']:3d}: {r['target_initial']:.2f}→{r['target_final']:.2f}, "
                    f"actual {r['actual_final']:.3f}, error {r['final_error']:.4f}"
                )

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("V7.1 Simple Model → Simulator Validation Tool")
    print("=" * 80)

    # Prepare directories
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model()

    # Choose number of validation cases
    print("\n📊 Choose number of validation cases:")
    print("  [1] 1,000 cases (±1.9% CI)")
    print("  [2] 2,000 cases (±1.3% CI) ⭐ Recommended")
    print("  [3] 3,000 cases (±1.1% CI)")

    choice = input("\nSelect (1-3, default 2): ").strip() or "2"

    num_cases_map = {
        "1": 1000,
        "2": 2000,
        "3": 3000
    }

    num_cases = num_cases_map.get(choice, 2000)

    # Run validation
    print(f"\n🚀 Starting validation for {num_cases} cases...")
    print("  💡 *.inp/*.out files will be deleted after validation")
    input("\nPress Enter to start...")

    results = run_validation_suite(model, tokenizer, num_cases=num_cases)

    print("\n" + "✅ " * 20)
    print("V7.1 Simple validation complete!")
    print("✅ " * 20)


if __name__ == "__main__":
    main()
