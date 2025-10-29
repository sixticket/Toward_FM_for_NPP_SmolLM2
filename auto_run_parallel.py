# auto_run_og.py
import os
import random
import subprocess
import csv
import sys
import time
import multiprocessing
from functools import partial
from tqdm import tqdm

# --- Configuration variables ---
NUM_SIMULATIONS = 3000
KOMODO_EXECUTABLE = '../komodo'
TEMPLATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'template')
DATASET_DIR = 'dataset'
SAMPLING_INTERVAL = 1.0  # Data sampling interval (seconds)


def generate_random_scenario():
    """
    Scenario generator for foundation model training
    - Type 1: Single control bank operation (60%)
    - Type 2: Two banks operated simultaneously (30%)
    - Type 3: Two banks operated sequentially (10%)
    """
    INITIAL_B1_POS = 180
    INITIAL_B2_POS = 100

    # Three clear scenario modes
    scenario_type = random.choices(
        ['single', 'simultaneous', 'sequential'],
        weights=[60, 30, 10]
    )[0]

    b1_pos, b1_time, b1_speed = INITIAL_B1_POS, 0.0, 0.0
    b2_pos, b2_time, b2_speed = INITIAL_B2_POS, 0.0, 0.0

    if scenario_type == 'single':
        # Move only one control bank (randomly pick b1 or b2)
        if random.random() < 0.5:
            print("[Type: Bank1 only]")
            b1_pos = random.randint(60, 180)
            b1_time = round(random.uniform(2.0, 15.0), 1)
            b1_speed = round(random.uniform(0.5, 3.0), 1)
        else:
            print("[Type: Bank2 only]")
            b2_pos = random.randint(70, 160)
            b2_time = round(random.uniform(2.0, 15.0), 1)
            b2_speed = round(random.uniform(0.2, 2.0), 1)

    elif scenario_type == 'simultaneous':
        # Operate both control banks at the same time
        print("[Type: Simultaneous operation]")
        start_time = round(random.uniform(2.0, 10.0), 1)

        b1_pos = random.randint(80, 180)
        b1_time = start_time
        b1_speed = round(random.uniform(0.5, 2.5), 1)

        b2_pos = random.randint(70, 140)
        b2_time = start_time
        b2_speed = round(random.uniform(0.2, 1.5), 1)

    else:  # sequential
        # Sequential operation (b2 first, then b1)
        print("[Type: Sequential operation]")
        b2_pos = random.randint(80, 150)
        b2_time = round(random.uniform(2.0, 8.0), 1)
        b2_speed = round(random.uniform(0.3, 2.0), 1)

        b1_pos = random.randint(70, 180)
        b1_time = b2_time + round(random.uniform(3.0, 8.0), 1)
        b1_speed = round(random.uniform(0.5, 3.0), 1)

    return {
        "b1_pos": b1_pos, "b1_time": b1_time, "b1_speed": b1_speed,
        "b2_pos": b2_pos, "b2_time": b2_time, "b2_speed": b2_speed,
    }


def create_input_file(template_path, scenario, output_path):
    """Create a new KOMODO input file using the template file and scenario."""
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()

        new_content = template_content.format(**scenario)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        raise
    except KeyError as e:
        print(f"Error: Required key '{e}' for the template is missing in the scenario.")
        raise


def run_komodo(input_path):
    """Run a KOMODO simulation."""
    work_dir = os.path.dirname(input_path)
    input_filename = os.path.basename(input_path)
    executable_path_from_cwd = '../../komodo'

    try:
        result = subprocess.run(
            [executable_path_from_cwd, input_filename],
            cwd=work_dir,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        print(f"Success: Simulation finished for {input_path}.")
        return True
    except FileNotFoundError:
        print(f"Error: Could not find KOMODO executable '{executable_path_from_cwd}'.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error: Simulation failed for {input_path}.")
        print("--- KOMODO ERROR OUTPUT ---")
        print(e.stderr)
        print("--------------------------")
        return False


def parse_and_save_results(output_file, save_path, interval=1.0):
    """
    Parse data from the result file and save it as CSV.
    - Sample data at the specified interval.
    - Return summary info of initial/final/peak power.
    """
    times, powers = [], []
    summary = {"initial": None, "final": None, "peak": None}

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
            print(f"Warning: No power output data found in '{output_file}'.")
            return summary

        # Compute summary
        summary["initial"] = powers[0]
        summary["final"] = powers[-1]
        summary["peak"] = max(powers)

        # Sampling
        sampled_times, sampled_powers = [], []
        last_saved_time = -1.0

        for t, p in zip(times, powers):
            if t >= last_saved_time + interval:
                sampled_times.append(t)
                sampled_powers.append(p)
                last_saved_time = t

        # Save to CSV
        with open(save_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Time(s)', 'Relative_Power'])
            writer.writerows(zip(sampled_times, sampled_powers))

        print(f"Saved results: {save_path} (sampling interval: {interval}s)")
        print(f"  >> Summary: initial={summary['initial']:.4f}, final={summary['final']:.4f}, peak={summary['peak']:.4f}")

        return summary

    except FileNotFoundError:
        print(f"Error: Could not find result file '{output_file}'.")
        return summary


def run_single_simulation(i, dataset_dir, template_file, sampling_interval):
    """Run a single simulation (for parallel processing)."""
    run_name = f"run_{i:04d}"
    print(f"[{i}] Simulation start: {run_name}")

    scenario = generate_random_scenario()
    print(f"[{i}] Scenario: {scenario}")

    input_path = os.path.join(dataset_dir, f"{run_name}.inp")

    # Create input file
    try:
        create_input_file(template_file, scenario, input_path)
    except Exception as e:
        print(f"[{i}] Failed to create input file: {e}")
        return None

    # Run KOMODO
    if not run_komodo(input_path):
        return None

    # Parse results
    output_path = os.path.join(dataset_dir, f"{run_name}.inp.out")
    power_csv_path = os.path.join(dataset_dir, f"{run_name}_power.csv")
    summary = parse_and_save_results(output_path, power_csv_path, interval=sampling_interval)

    # Save scenario
    rod_csv_path = os.path.join(dataset_dir, f"{run_name}_rod_scenario.csv")
    with open(rod_csv_path, 'w', newline='', encoding='utf-8') as f:
        rod_writer = csv.writer(f)
        rod_writer.writerow(scenario.keys())
        rod_writer.writerow(scenario.values())

    time.sleep(0.2)

    return {
        'run_name': run_name,
        'power_csv_path': power_csv_path,
        'rod_csv_path': rod_csv_path,
        'scenario': scenario,
        'summary': summary
    }

def main():
    os.makedirs(DATASET_DIR, exist_ok=True)
    master_csv_path = os.path.join(DATASET_DIR, 'master_dataset_3000.csv')

    # Write header
    sample_scenario = generate_random_scenario()
    header = ['run_id', 'power_data_path', 'rod_scenario_path'] + \
             list(sample_scenario.keys()) + \
             ['initial_power', 'final_power', 'peak_power']

    with open(master_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # Conservative parallel settings
    num_workers = 4  # Fixed at 4 (favor stability)
    print(f"\nStarting parallel processing: using {num_workers} workers (stability mode)\n")

    with multiprocessing.Pool(processes=num_workers) as pool:
        func = partial(
            run_single_simulation,
            dataset_dir=DATASET_DIR,
            template_file=TEMPLATE_FILE,
            sampling_interval=SAMPLING_INTERVAL
        )

        # Progress bar with tqdm
        results = list(tqdm(
            pool.imap(func, range(1, NUM_SIMULATIONS + 1)),
            total=NUM_SIMULATIONS,
            desc="Simulations",
            ncols=100,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed}, remaining: {remaining}]'
        ))

    # Append results to the master CSV
    with open(master_csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for result in results:
            if result is not None:
                row_data = [
                               result['run_name'],
                               result['power_csv_path'],
                               result['rod_csv_path']
                           ] + list(result['scenario'].values()) + [
                               result['summary'].get('initial'),
                               result['summary'].get('final'),
                               result['summary'].get('peak')
                           ]
                writer.writerow(row_data)

    print(f"\nAll simulations complete! Master dataset: '{master_csv_path}'")


if __name__ == "__main__":
    main()
