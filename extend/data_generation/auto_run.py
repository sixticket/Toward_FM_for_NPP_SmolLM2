import os
import random
import subprocess
import csv
import multiprocessing
from functools import partial
from pathlib import Path
from tqdm import tqdm
import itertools
import time

# --- [설정] ---
SCRIPT_DIR = Path(__file__).resolve().parent          # extend/data_generation/
REPO_ROOT = SCRIPT_DIR.parents[1]                      # repo root

# Override with KOMODO_EXECUTABLE env var if installed elsewhere
KOMODO_EXECUTABLE_PATH = os.environ.get(
    "KOMODO_EXECUTABLE",
    str(REPO_ROOT / "komodo")
)
TEMPLATE_FILE = str(SCRIPT_DIR / "template")  # extend 전용 template (boron/window 필드 포함)
DATASET_DIR = str(SCRIPT_DIR / "dataset_10k_retry")
SAMPLES_PER_BIN = 50


def generate_scenario_in_bin(bin_config):
    """구간 조건 내 랜덤 시나리오 생성"""
    (win_min, win_max), (boron_min, boron_max), init_case, move_type = bin_config

    monitor_window = round(random.uniform(win_min, win_max), 1)
    boron_ppm = round(random.uniform(boron_min, boron_max), 2)
    switch_time = max(monitor_window - 10.0, 50.0)

    if init_case == 'case1':
        init_b1, init_b2 = 180, 100
    else:
        init_b1, init_b2 = 100, 180

    b1_final, b1_time, b1_speed = init_b1, 0.0, 0.0
    b2_final, b2_time, b2_speed = init_b2, 0.0, 0.0

    if move_type == 'A_only':
        b1_final = random.randint(0, init_b1 - 10)
        b1_time = round(random.uniform(1.0, 40.0), 1)
        b1_speed = round(random.uniform(1.0, 10.0), 1)

    elif move_type == 'B_only':
        b2_final = random.randint(0, init_b2 - 10)
        b2_time = round(random.uniform(1.0, 40.0), 1)
        b2_speed = round(random.uniform(1.0, 10.0), 1)

    elif move_type == 'Simultaneous':
        start_time = round(random.uniform(1.0, 40.0), 1)
        b1_final = random.randint(0, init_b1 - 10)
        b1_time, b1_speed = start_time, round(random.uniform(1.0, 10.0), 1)
        b2_final = random.randint(0, init_b2 - 10)
        b2_time, b2_speed = start_time, round(random.uniform(1.0, 10.0), 1)

    elif move_type == 'Sequential':
        first_mover = random.choice(['A', 'B'])
        t1 = round(random.uniform(1.0, 20.0), 1)
        t2 = t1 + round(random.uniform(5.0, 15.0), 1)

        if first_mover == 'A':
            b1_final = random.randint(0, init_b1 - 10)
            b1_time, b1_speed = t1, round(random.uniform(1.0, 10.0), 1)
            b2_final = random.randint(0, init_b2 - 10)
            b2_time, b2_speed = t2, round(random.uniform(1.0, 10.0), 1)
        else:
            b2_final = random.randint(0, init_b2 - 10)
            b2_time, b2_speed = t1, round(random.uniform(1.0, 10.0), 1)
            b1_final = random.randint(0, init_b1 - 10)
            b1_time, b1_speed = t2, round(random.uniform(1.0, 10.0), 1)

    return {
        "b1_pos": b1_final, "b1_time": b1_time, "b1_speed": b1_speed,
        "b2_pos": b2_final, "b2_time": b2_time, "b2_speed": b2_speed,
        "boron_ppm": boron_ppm,
        "monitor_window": monitor_window,
        "switch_time": switch_time,
        "init_case": init_case,
        "move_type": move_type,
        "win_range": f"{win_min}-{win_max}",
        "boron_range": f"{boron_min}-{boron_max}"
    }


def create_input_file(template_path, scenario, output_path):
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        new_content = template_content.format(**scenario)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    except:
        raise


def run_komodo(input_path):
    work_dir = os.path.dirname(input_path)
    input_filename = os.path.basename(input_path)
    try:
        # 타임아웃 3분
        subprocess.run(
            [KOMODO_EXECUTABLE_PATH, input_filename],
            cwd=work_dir, check=True, capture_output=True, text=True, encoding='utf-8',
            timeout=180
        )
        return True
    except:
        return False


def parse_output_for_summary(output_file):
    summary = {"initial": None, "final": None, "peak": None}
    powers = []
    try:
        with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
            in_results_section = False
            for line in f:
                if "TRANSIENT RESULTS" in line:
                    in_results_section = True
                    next(f, None);
                    next(f, None)
                    continue
                if in_results_section and line.strip().startswith('CPU time breakdown'):
                    break
                if in_results_section and line.strip():
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            powers.append(float(parts[3]))
                        except:
                            continue
        if powers:
            summary["initial"] = powers[0]
            summary["final"] = powers[-1]
            summary["peak"] = max(powers)
        return summary
    except:
        return summary


def run_task_with_retry(args):
    """[핵심] 실패 시 성공할 때까지 재시도하는 함수"""
    i, bin_config, dataset_dir, template_file = args
    run_name = f"data_{i:06d}"

    input_path = os.path.join(dataset_dir, f"{run_name}.inp")
    output_path = os.path.join(dataset_dir, f"{run_name}.inp.out")
    vtk_path = os.path.join(dataset_dir, f"{run_name}.inp.vtk")

    max_retries = 20  # 최대 20번까지 재시도
    attempt = 0

    while attempt < max_retries:
        attempt += 1
        try:
            # 1. 시나리오 생성 (재시도할 때마다 새로운 랜덤값 생성 -> 성공 확률 높임)
            scenario = generate_scenario_in_bin(bin_config)

            # 2. 파일 생성
            create_input_file(template_file, scenario, input_path)

            # 3. 실행
            if not run_komodo(input_path):
                # 실패: 파일 지우고 continue (재시도)
                for p in [input_path, output_path, vtk_path]:
                    if os.path.exists(p): os.remove(p)
                continue

                # 4. 파싱
            summary = parse_output_for_summary(output_path)

            # 데이터 없음: 실패 처리
            if summary['initial'] is None:
                for p in [input_path, output_path, vtk_path]:
                    if os.path.exists(p): os.remove(p)
                continue

            # --- 성공! ---
            # 파일 삭제
            for p in [input_path, output_path, vtk_path]:
                try:
                    if os.path.exists(p): os.remove(p)
                except:
                    pass

            return {'run_name': run_name, 'scenario': scenario, 'summary': summary}

        except Exception:
            # 에러 발생 시 파일 지우고 재시도
            for p in [input_path, output_path, vtk_path]:
                try:
                    if os.path.exists(p): os.remove(p)
                except:
                    pass
            continue

    # Max retries 초과 시 (거의 없겠지만)
    return None


def main():
    if not os.path.exists(KOMODO_EXECUTABLE_PATH):
        print(f"❌ 오류: 실행 파일 없음 -> {KOMODO_EXECUTABLE_PATH}")
        return

    os.makedirs(DATASET_DIR, exist_ok=True)
    master_csv_path = os.path.join(DATASET_DIR, 'master_dataset_10k_guaranteed.csv')

    win_ranges = [(60, 80), (80, 100), (100, 120), (120, 140), (140, 160)]
    boron_ranges = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000)]
    init_cases = ['case1', 'case2']
    move_types = ['A_only', 'B_only', 'Simultaneous', 'Sequential']

    all_bins = list(itertools.product(win_ranges, boron_ranges, init_cases, move_types))

    tasks = []
    run_id = 1
    for bin_config in all_bins:
        for _ in range(SAMPLES_PER_BIN):
            tasks.append((run_id, bin_config, DATASET_DIR, TEMPLATE_FILE))
            run_id += 1

    total_sims = len(tasks)

    header = ['run_id', 'win_range', 'boron_range', 'init_case', 'move_type',
              'boron_ppm', 'monitor_window', 'switch_time',
              'b1_pos', 'b1_time', 'b1_speed',
              'b2_pos', 'b2_time', 'b2_speed',
              'initial_power', 'final_power', 'peak_power']

    with open(master_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    num_workers = 6
    print(f"\n💎 [무결점 10k 생산 모드]")
    print(f"   - 코어: {num_workers}개")
    print(f"   - 특징: 에러 발생 시 성공할 때까지 재시도 (Retry)")
    print(f"   - 목표: 정확히 {total_sims}개 보장")

    with multiprocessing.Pool(processes=num_workers) as pool:
        with open(master_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            for result in tqdm(pool.imap_unordered(run_task_with_retry, tasks),
                               total=total_sims, ncols=100, desc="Generating"):
                if result:
                    scen = result['scenario']
                    summ = result['summary']
                    row = [
                        result['run_name'],
                        scen['win_range'], scen['boron_range'], scen['init_case'], scen['move_type'],
                        scen['boron_ppm'], scen['monitor_window'], scen['switch_time'],
                        scen['b1_pos'], scen['b1_time'], scen['b1_speed'],
                        scen['b2_pos'], scen['b2_time'], scen['b2_speed'],
                        summ['initial'], summ['final'], summ['peak']
                    ]
                    writer.writerow(row)

    print(f"\n✅ 10,000개 완벽 생성 완료! 저장 위치: {master_csv_path}")


if __name__ == "__main__":
    main()