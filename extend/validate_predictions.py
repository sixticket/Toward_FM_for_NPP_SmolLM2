import os
import pandas as pd
import numpy as np
import subprocess
import shutil
import json
import re
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# ============================================================================
# [설정] 검증할 대상 CSV 파일들
# ============================================================================
TARGET_CSVS = [
    "prediction_extended_2k.csv",
    "prediction_adapter_only_2k.csv"
]

# 경로 설정 (override KOMODO_EXECUTABLE env var if installed elsewhere)
SCRIPT_DIR = Path(__file__).resolve().parent  # extend/
REPO_ROOT  = SCRIPT_DIR.parent                # repo root
KOMODO_EXECUTABLE = Path(os.environ.get("KOMODO_EXECUTABLE", str(REPO_ROOT / "komodo")))
TEMPLATE_FILE = SCRIPT_DIR / "data_generation" / "template"  # extend 전용 (window/boron 필드 포함)
SIM_TEMP_DIR = SCRIPT_DIR / "temp_sim_monitor"  # 임시 실행 폴더

FIXED_BORON = 500.0


# ============================================================================
# [유틸] JSON 인코더 & 밴드 분류기
# ============================================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def get_window_band(w):
    try:
        w = float(w)
        if 60 <= w < 70:
            return "60-70"
        elif 70 <= w < 80:
            return "70-80"
        elif 80 <= w < 90:
            return "80-90"
        elif 90 <= w <= 100:
            return "90-100"
        else:
            return "Outlier"
    except:
        return "Error"


def get_delta_band(p_init, p_target):
    try:
        p_init = float(p_init)
        p_target = float(p_target)
        if p_init == 0: return "Error"
        delta = abs(p_target - p_init) / p_init

        if delta < 0.01:
            return "Very Low (<1%)"
        elif 0.01 <= delta < 0.10:
            return "Low (1~10%)"
        elif 0.10 <= delta < 0.30:
            return "Mid (10~30%)"
        elif 0.30 <= delta <= 0.50:
            return "High (30~50%)"
        else:
            return "Very High (>50%)"
    except:
        return "Error"


# ============================================================================
# [시뮬레이터] 실행 함수
# ============================================================================
def create_input_file(template_path, scenario, output_path):
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template.format(**scenario))


def run_komodo(input_path):
    try:
        subprocess.run([str(KOMODO_EXECUTABLE), input_path.name],
                       cwd=str(input_path.parent),
                       check=True, capture_output=True, text=True, timeout=60)
        return True, None
    except Exception as e:
        return False, str(e)


def parse_simulation_output(output_file):
    powers = []
    try:
        with open(output_file, 'r', errors='ignore') as f:
            read = False
            for line in f:
                if "TRANSIENT RESULTS" in line:
                    read = True
                    next(f, None);
                    next(f, None)
                    continue
                if read and "CPU time" in line: break
                if read and line.strip():
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            powers.append(float(parts[3]))
                        except:
                            pass
        if not powers: return None
        return {'final_power': powers[-1]}
    except:
        return None


# ============================================================================
# [검증 로직] 단일 케이스 처리
# ============================================================================
def verify_row(row, idx):
    # 1. 입력 데이터 파싱
    try:
        p_init = float(row['initial_power'])
        p_target = float(row['final_power'])
        window = float(row['monitor_window'])
    except:
        return {'case_id': idx, 'sim_success': False, 'error': 'Invalid Input', 'parsed': False}

    w_band = get_window_band(window)
    d_band = get_delta_band(p_init, p_target)

    # 2. 모델 출력 파싱 성공 여부 확인
    is_parsed = True
    if 'parsed' in row:
        is_parsed = str(row['parsed']).lower() == 'true'
    elif pd.isna(row.get('b1_pos')):  # parsed 컬럼 없으면 b1_pos 값 유무로 판단
        is_parsed = False

    if not is_parsed:
        return {
            'case_id': idx, 'parsed': False, 'sim_success': False,
            'error': 'Parsing Failed',
            'window_band': w_band, 'delta_band': d_band
        }

    # 3. 시뮬레이션 준비 및 실행
    case_dir = SIM_TEMP_DIR / f"case_{idx:04d}"
    inp_path = case_dir / "input.inp"
    out_path = case_dir / "input.inp.out"

    try:
        scenario = {
            "monitor_window": window,
            "boron_ppm": FIXED_BORON,
            "initial_power": p_init,
            "b1_pos": int(np.clip(round(float(row['b1_pos'])), 0, 180)),
            "b1_time": float(row['b1_time']),
            "b1_speed": float(row['b1_speed']),
            "b2_pos": int(np.clip(round(float(row['b2_pos'])), 0, 180)),
            "b2_time": float(row['b2_time']),
            "b2_speed": float(row['b2_speed']),
        }
        create_input_file(TEMPLATE_FILE, scenario, inp_path)
        ok, err = run_komodo(inp_path)
        sim_res = parse_simulation_output(out_path) if ok else None
    except Exception as e:
        ok, sim_res = False, None

    # 임시 파일 삭제
    if case_dir.exists(): shutil.rmtree(case_dir)

    # 4. 결과 판정
    if not ok or sim_res is None:
        return {
            'case_id': idx, 'parsed': True, 'sim_success': False,
            'error': 'Simulation Failed',
            'window_band': w_band, 'delta_band': d_band
        }

    actual = sim_res['final_power']
    error = abs(actual - p_target)
    err_rate = error / abs(p_target) if abs(p_target) > 0 else float('inf')

    return {
        'case_id': idx, 'parsed': True, 'sim_success': True,
        'actual_final': actual, 'p_target': p_target,
        'final_error': error, 'error_rate': err_rate,

        # [통계용 성공 여부 플래그]
        'success_1': err_rate <= 0.01,
        'success_2': err_rate <= 0.02,
        'success_3': err_rate <= 0.03,
        'success_5': err_rate <= 0.05,
        'success_10': err_rate <= 0.10,

        'window_band': w_band,
        'delta_band': d_band
    }


# ============================================================================
# [리포트 생성] 메인 프로세스
# ============================================================================
def process_csv_file(csv_file):
    print(f"\n{'=' * 80}")
    print(f"📂 Processing: {csv_file}")
    print(f"{'=' * 80}")

    file_path = SCRIPT_DIR / csv_file
    if not file_path.exists():
        print("❌ File not found")
        return

    df = pd.read_csv(file_path)
    total = len(df)
    results = []

    # TQDM 진행바와 함께 검증 수행
    for idx, row in tqdm(df.iterrows(), total=total, desc="Verifying"):
        results.append(verify_row(row, idx))

    # 데이터프레임 변환
    res_df = pd.DataFrame(results)
    sim_ok = res_df[res_df['sim_success'] == True]
    parsed_ok = res_df[res_df['parsed'] == True]

    # --- [텍스트 리포트 출력] ---
    print(f"\n📊 [Validation Report] - {csv_file}")
    print(f"  Total Cases:        {total}")
    print(f"  Parsing Success:    {len(parsed_ok)} ({len(parsed_ok) / total * 100:.1f}%)")
    print(f"  Simulation Success: {len(sim_ok)} ({len(sim_ok) / total * 100:.1f}%)")

    if len(sim_ok) > 0:
        # 1. 전체 오차별 성공률 (1, 2, 3, 5, 10%)
        print(f"\n  [Global Accuracy (Based on Sim Success)]")
        for limit in [1, 2, 3, 5, 10]:
            col = f'success_{limit}'
            succ = sim_ok[col].sum()
            print(f"    Target ±{limit}%: {succ:4d}/{len(sim_ok)} ({succ / len(sim_ok) * 100:.1f}%)")

        print(f"    Mean Error:   {sim_ok['error_rate'].mean() * 100:.4f}%")

        # 2. Window 구간별 통계 (±5% 기준)
        print(f"\n  [By Window Band (Success ±5%)]")
        if 'window_band' in sim_ok.columns:
            for band in sorted(sim_ok['window_band'].unique()):
                sub = sim_ok[sim_ok['window_band'] == band]
                if len(sub) == 0: continue
                succ = sub['success_5'].sum()
                print(f"    {band:<7}s: {succ:3d}/{len(sub):3d} ({succ / len(sub) * 100:.1f}%)")

        # 3. Power Delta(난이도)별 상세 통계 (1, 3, 5% 각각 표시)
        print(f"\n  [By Power Delta (Accuracy Breakdown)]")
        order = ["Low (1~10%)", "Mid (10~30%)", "High (30~50%)"]
        if 'delta_band' in sim_ok.columns:
            for band in order:
                sub = sim_ok[sim_ok['delta_band'] == band]
                if len(sub) == 0: continue

                s1 = sub['success_1'].sum()
                s3 = sub['success_3'].sum()
                s5 = sub['success_5'].sum()

                print(
                    f"    {band:<15}: (±1%) {s1 / len(sub) * 100:.0f}% | (±3%) {s3 / len(sub) * 100:.0f}% | (±5%) {s5 / len(sub) * 100:.0f}%")

    # --- [JSON 결과 저장] ---
    out_name = f"verified_{csv_file.replace('.csv', '')}.json"
    with open(SCRIPT_DIR / out_name, 'w') as f:
        json.dump(res_df.to_dict(orient='records'), f, indent=2, cls=NumpyEncoder)
    print(f"\n💾 Saved detailed results: {out_name}")


def main():
    if SIM_TEMP_DIR.exists(): shutil.rmtree(SIM_TEMP_DIR)
    SIM_TEMP_DIR.mkdir(parents=True, exist_ok=True)

    for csv in TARGET_CSVS:
        process_csv_file(csv)

    if SIM_TEMP_DIR.exists(): shutil.rmtree(SIM_TEMP_DIR)
    print("\n🎉 All Verification Tasks Completed!")


if __name__ == "__main__":
    main()