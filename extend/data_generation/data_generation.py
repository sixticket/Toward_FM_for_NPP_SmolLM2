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
import shutil

# ==============================================================================
# [설정] 100K 대량 생산 모드 (Variable Window)
# ==============================================================================

# 경로 (스크립트 위치 기준)
SCRIPT_DIR = Path(__file__).resolve().parent          # extend/data_generation/
REPO_ROOT = SCRIPT_DIR.parents[1]                      # repo root

# 실행 파일 경로 (override with KOMODO_EXECUTABLE env var)
KOMODO_EXECUTABLE_PATH = os.environ.get(
    "KOMODO_EXECUTABLE",
    str(REPO_ROOT / "komodo")
)
TEMPLATE_FILE = str(SCRIPT_DIR / "template")  # extend 전용 template (boron/window 필드 포함)

# 저장 경로 (스크립트 옆에 생성)
DATASET_DIR = str(SCRIPT_DIR / "dataset_100k_final")

# [핵심] 10만 개 생성 설정
# 200개 구간(Bins) * 500개 = 100,000개
SAMPLES_PER_BIN = 500

# 작업자 프로세스 수 (CPU 코어 수에 맞게 조절, 보통 코어수 - 1 or 2)
NUM_WORKERS = 10

# 배치 쓰기 크기 (디스크 부하 감소용)
BATCH_SIZE = 100


# ==============================================================================
# 로직 함수들
# ==============================================================================

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
    except Exception as e:
        raise e


def run_komodo(input_path):
    work_dir = os.path.dirname(input_path)
    input_filename = os.path.basename(input_path)
    try:
        # 타임아웃 3분 (100k 돌릴 땐 타임아웃 필수)
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
    """실패 시 성공할 때까지 재시도하는 함수 (Robustness)"""
    i, bin_config, dataset_dir, template_file = args
    run_name = f"data_{i:06d}"

    # 임시 파일 경로
    input_path = os.path.join(dataset_dir, f"{run_name}.inp")
    output_path = os.path.join(dataset_dir, f"{run_name}.inp.out")
    vtk_path = os.path.join(dataset_dir, f"{run_name}.inp.vtk")

    max_retries = 50  # 100k 모드에서는 retry 횟수를 늘려서라도 성공시킴
    attempt = 0

    while attempt < max_retries:
        attempt += 1
        try:
            # 1. 시나리오 생성 (매 시도마다 랜덤값 재생성)
            scenario = generate_scenario_in_bin(bin_config)

            # 2. 파일 생성
            create_input_file(template_file, scenario, input_path)

            # 3. 실행
            if not run_komodo(input_path):
                # 실행 실패 시 파일 정리 후 재시도
                for p in [input_path, output_path, vtk_path]:
                    if os.path.exists(p): os.remove(p)
                continue

            # 4. 파싱
            summary = parse_output_for_summary(output_path)

            # 파싱 실패(데이터 없음) 시 재시도
            if summary['initial'] is None:
                for p in [input_path, output_path, vtk_path]:
                    if os.path.exists(p): os.remove(p)
                continue

            # --- 성공 ---
            # 불필요한 파일 즉시 삭제 (10만 개 파일 쌓이면 SSD 느려짐)
            for p in [input_path, output_path, vtk_path]:
                try:
                    if os.path.exists(p): os.remove(p)
                except:
                    pass

            return {'run_name': run_name, 'scenario': scenario, 'summary': summary}

        except Exception:
            # 예외 발생 시 정리 후 재시도
            for p in [input_path, output_path, vtk_path]:
                try:
                    if os.path.exists(p): os.remove(p)
                except:
                    pass
            continue

    return None  # Max retries 초과


# ==============================================================================
# 메인 실행
# ==============================================================================

def main():
    if not os.path.exists(KOMODO_EXECUTABLE_PATH):
        print(f"❌ 오류: 실행 파일 없음 -> {KOMODO_EXECUTABLE_PATH}")
        return

    os.makedirs(DATASET_DIR, exist_ok=True)
    master_csv_path = os.path.join(DATASET_DIR, 'master_dataset_100k.csv')

    # 구간 설정
    win_ranges = [(60, 80), (80, 100), (100, 120), (120, 140), (140, 160)]
    boron_ranges = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000)]
    init_cases = ['case1', 'case2']
    move_types = ['A_only', 'B_only', 'Simultaneous', 'Sequential']

    all_bins = list(itertools.product(win_ranges, boron_ranges, init_cases, move_types))

    # 전체 작업 리스트 생성
    all_tasks = []
    run_id = 1
    for bin_config in all_bins:
        for _ in range(SAMPLES_PER_BIN):
            all_tasks.append((run_id, bin_config, DATASET_DIR, TEMPLATE_FILE))
            run_id += 1

    total_planned = len(all_tasks)

    # --- [이어하기 기능: Resume Logic] ---
    existing_ids = set()
    write_header = True

    if os.path.exists(master_csv_path):
        print(f"🔄 기존 데이터셋 감지됨. 이어하기를 준비합니다...")
        try:
            with open(master_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header_row = next(reader, None)
                if header_row:
                    write_header = False  # 헤더가 이미 있음
                    for row in reader:
                        if row: existing_ids.add(row[0])  # run_name (data_000001)
            print(f"   -> 이미 완료된 작업: {len(existing_ids):,}개")
        except:
            print("   -> 읽기 에러. 새로 시작합니다.")
            write_header = True

    # 이미 한 작업은 제외하고 할 작업만 필터링
    tasks_to_run = []
    for t in all_tasks:
        name = f"data_{t[0]:06d}"
        if name not in existing_ids:
            tasks_to_run.append(t)

    print(f"🚀 남은 작업: {len(tasks_to_run):,} / {total_planned:,}")
    if len(tasks_to_run) == 0:
        print("✅ 모든 작업이 이미 완료되었습니다!")
        return

    # CSV 헤더 작성 (새로 시작하는 경우에만)
    header = ['run_name', 'win_range', 'boron_range', 'init_case', 'move_type',
              'boron_ppm', 'monitor_window', 'switch_time',
              'b1_pos', 'b1_time', 'b1_speed',
              'b2_pos', 'b2_time', 'b2_speed',
              'initial_power', 'final_power', 'peak_power']

    if write_header:
        with open(master_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # --- [멀티프로세싱 실행] ---
    print(f"\n💎 [100K 대량 생산 모드 가동]")
    print(f"   - 코어 수: {NUM_WORKERS}")
    print(f"   - 배치 쓰기: {BATCH_SIZE}개 단위")

    buffer = []  # 메모리에 모아두는 버퍼

    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        # 파일은 append 모드('a')로 엶
        with open(master_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            for result in tqdm(pool.imap_unordered(run_task_with_retry, tasks_to_run),
                               total=len(tasks_to_run), ncols=100, desc="Generating"):
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

                    buffer.append(row)

                    # 버퍼가 차면 디스크에 씀 (I/O 병목 해소)
                    if len(buffer) >= BATCH_SIZE:
                        writer.writerows(buffer)
                        f.flush()  # 강제 기록 (비정상 종료 대비)
                        buffer = []

            # 남은 버퍼 처리
            if buffer:
                writer.writerows(buffer)
                f.flush()

    print(f"\n✅ 100K 생성 완료! 저장 위치: {master_csv_path}")


if __name__ == "__main__":
    main()