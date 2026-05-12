import os
import random
import subprocess
import csv
import time
import multiprocessing
from functools import partial
from pathlib import Path
from tqdm import tqdm

# --- 설정 변수 ---
NUM_SIMULATIONS = 100000  # 100K 데이터 생성

SCRIPT_DIR = Path(__file__).resolve().parent  # data_generation/
REPO_ROOT  = SCRIPT_DIR.parent                # repo root

# KOMODO 실행파일 (환경변수 KOMODO_EXECUTABLE 으로 override 가능)
KOMODO_EXECUTABLE = os.environ.get("KOMODO_EXECUTABLE", str(REPO_ROOT / "komodo"))
TEMPLATE_FILE = str(SCRIPT_DIR / "template")
DATASET_DIR = str(REPO_ROOT / "dataset")  # 학습 스크립트가 여기서 찾음


def generate_random_scenario():
    """Foundation model 학습용 시나리오 생성"""
    INITIAL_B1_POS = 180
    INITIAL_B2_POS = 100

    scenario_type = random.choices(
        ['single', 'simultaneous', 'sequential'],
        weights=[60, 30, 10]
    )[0]

    b1_pos, b1_time, b1_speed = INITIAL_B1_POS, 0.0, 0.0
    b2_pos, b2_time, b2_speed = INITIAL_B2_POS, 0.0, 0.0

    if scenario_type == 'single':
        if random.random() < 0.5:
            b1_pos = random.randint(60, 180)
            b1_time = round(random.uniform(2.0, 15.0), 1)
            b1_speed = round(random.uniform(0.5, 3.0), 1)
        else:
            b2_pos = random.randint(70, 160)
            b2_time = round(random.uniform(2.0, 15.0), 1)
            b2_speed = round(random.uniform(0.2, 2.0), 1)

    elif scenario_type == 'simultaneous':
        start_time = round(random.uniform(2.0, 10.0), 1)
        b1_pos = random.randint(80, 180)
        b1_time = start_time
        b1_speed = round(random.uniform(0.5, 2.5), 1)
        b2_pos = random.randint(70, 140)
        b2_time = start_time
        b2_speed = round(random.uniform(0.2, 1.5), 1)

    else:  # sequential
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
    """템플릿 파일과 시나리오를 사용하여 새로운 KOMODO 입력 파일을 생성합니다."""
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        new_content = template_content.format(**scenario)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다: {e}")
        raise
    except KeyError as e:
        print(f"오류: 템플릿 파일에 필요한 키 '{e}'가 시나리오에 없습니다.")
        raise


def run_komodo(input_path):
    """KOMODO 시뮬레이션을 실행합니다."""
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
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def parse_output_for_summary(output_file):
    """
    ✨ 핵심 변경: 파일에서 요약 정보만 추출 (CSV 저장 안함)
    """
    summary = {"initial": None, "final": None, "peak": None}
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
                            power_val = float(parts[3])
                            powers.append(power_val)
                        except (ValueError, IndexError):
                            continue

        if powers:
            summary["initial"] = powers[0]
            summary["final"] = powers[-1]
            summary["peak"] = max(powers)

        return summary

    except FileNotFoundError:
        return summary


def run_single_simulation(i, dataset_dir, template_file):
    """단일 시뮬레이션 실행 - 파일 즉시 삭제 버전"""
    run_name = f"run_{i:04d}"
    scenario = generate_random_scenario()
    input_path = os.path.join(dataset_dir, f"{run_name}.inp")
    output_path = os.path.join(dataset_dir, f"{run_name}.inp.out")
    vtk_path = os.path.join(dataset_dir, f"{run_name}.inp.vtk")  # ✨ 추가

    try:
        # 1. 입력 파일 생성
        create_input_file(template_file, scenario, input_path)

        # 2. KOMODO 실행
        if not run_komodo(input_path):
            # 실패 시 파일 삭제
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_path):
                os.remove(output_path)
            if os.path.exists(vtk_path):  # ✨ 추가
                os.remove(vtk_path)
            return None

        # 3. 결과 파싱 (요약만)
        summary = parse_output_for_summary(output_path)

        # 4. ✨ 즉시 파일 삭제 (VTK 포함!)
        try:
            os.remove(input_path)
            os.remove(output_path)
            if os.path.exists(vtk_path):  # ✨ 추가
                os.remove(vtk_path)
        except OSError:
            pass

        time.sleep(0.1)

        return {
            'run_name': run_name,
            'scenario': scenario,
            'summary': summary
        }
    
    
    

    except Exception as e:
        # 오류 발생 시 파일 정리
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        if os.path.exists(vtk_path):  # ✨ 추가
            os.remove(vtk_path)
        return None


def main():
    os.makedirs(DATASET_DIR, exist_ok=True)
    master_csv_path = os.path.join(DATASET_DIR, 'master_dataset_100K.csv')

    # 헤더 작성
    sample_scenario = generate_random_scenario()
    header = ['run_id'] + list(sample_scenario.keys()) + \
             ['initial_power', 'final_power', 'peak_power']

    with open(master_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # 병렬 처리 설정
    num_workers = 12
    print(f"\n✨ Master CSV 전용 모드 시작 (중간 파일 자동 삭제)")
    print(f"병렬 처리: {num_workers}개 워커")
    print(f"목표: {NUM_SIMULATIONS}개 시뮬레이션\n")

    with multiprocessing.Pool(processes=num_workers) as pool:
        func = partial(
            run_single_simulation,
            dataset_dir=DATASET_DIR,
            template_file=TEMPLATE_FILE
        )

        results = list(tqdm(
            pool.imap(func, range(1, NUM_SIMULATIONS + 1)),
            total=NUM_SIMULATIONS,
            desc="시뮬레이션",
            ncols=100,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [경과: {elapsed}, 남음: {remaining}]'
        ))

    # 결과를 마스터 CSV에 저장
    successful = 0
    with open(master_csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for result in results:
            if result is not None:
                row_data = [result['run_name']] + \
                           list(result['scenario'].values()) + [
                               result['summary'].get('initial'),
                               result['summary'].get('final'),
                               result['summary'].get('peak')
                           ]
                writer.writerow(row_data)
                successful += 1

    print(f"\n✅ 완료! 성공: {successful}/{NUM_SIMULATIONS}")
    print(f"📁 저장 위치: '{master_csv_path}'")
    print(f"💾 디스크 사용: Master CSV 1개만 (중간 파일 모두 삭제됨)")


if __name__ == "__main__":
    main()