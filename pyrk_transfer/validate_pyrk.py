"""
PyRK Inverse Model Validation (Benchmark Style)
- KOMODO 검증 스크립트와 동일한 구조
- 5단계 정확도 분석 (±1%, ±2%, ±3%, ±5%, ±10%)
- 구간별 성공률 (소폭/중폭/대폭)
- 중간 통계 출력
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import warnings
warnings.filterwarnings("ignore")

# matplotlib 임포트 전에 백엔드 설정
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import torch
import logging
import types
import contextlib
import multiprocessing as mp
from pathlib import Path
import time as time_module
import re
import json
from datetime import datetime

# PyRK
from pyrk.utilities.ur import units
from pyrk.timer import Timer
from pyrk.th_component import THComponent
from pyrk.materials.material import Material
from pyrk.materials.liquid_material import LiquidMaterial
from pyrk.density_model import DensityModel
from pyrk.reactivity_insertion import ReactivityInsertion
from pyrk.db import database
from pyrk.inp import sim_info
from pyrk.driver import solve

# Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============================================================================
# 설정
# ============================================================================
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.CRITICAL)

SCRIPT_DIR = Path(__file__).resolve().parent          # pyrk_transfer/

# 모델 경로 - train_pyrk.py 가 SCRIPT_DIR/models/ 에 출력
PHASE1_MODEL_PATH = SCRIPT_DIR / "models/pyrk_inverse_phase1_v2/final_model"
PHASE2_MODEL_PATH = SCRIPT_DIR / "models/pyrk_inverse_phase2_v2/final_model"

# 검증 결과 저장
VALIDATION_DIR = SCRIPT_DIR / "validation_runs"

# 추론 설정
INFERENCE_TEMPERATURE = 0.05
INFERENCE_TOP_P = 0.9
INFERENCE_MAX_TOKENS = 50

# PyRK 기준값
NOMINAL_POWER = 250.0  # MW (기준 출력)


# ============================================================================
# 모델 로드
# ============================================================================
def load_model():
    """Phase 2 V2 모델 로드 (Phase 1 + LoRA)"""
    print("=" * 80)
    print("모델 로딩 중... (V2 Simple Format)")
    print("=" * 80)

    # Phase 1 기본 모델 로드
    print(f"\n✓ Phase 1 모델: {PHASE1_MODEL_PATH}")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(PHASE1_MODEL_PATH),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # LoRA adapter 로드
    print(f"✓ Phase 2 LoRA: {PHASE2_MODEL_PATH}")
    model = PeftModel.from_pretrained(base_model, str(PHASE2_MODEL_PATH))

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(str(PHASE2_MODEL_PATH))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    print("\n✓ 모델 로드 완료!")
    print("=" * 80)

    return model, tokenizer


# ============================================================================
# 모델 예측
# ============================================================================
def parse_prediction(prediction_text):
    """
    V2 Simple 형식 파싱: 숫자 4개 추출
    예: "[245.44, 180.13, -0.002089, 22.388]"
    앞 2개는 입력, 뒤 2개는 출력 (reactivity, duration)
    """
    try:
        # 모든 숫자 추출 (음수 포함)
        numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', prediction_text)

        # 4개 이상 있어야 함
        if len(numbers) >= 4:
            # 3번째, 4번째가 출력값
            rho = float(numbers[2])
            dur = float(numbers[3])

            # 유효성 검사: rho는 음수, dur는 양수여야 함
            if rho < 0 and dur > 0:
                return {'reactivity': rho, 'duration': dur}

        # 숫자가 2개만 있으면 (생성된 부분만)
        if len(numbers) >= 2:
            # 마지막 2개 시도
            rho = float(numbers[-2])
            dur = float(numbers[-1])
            if rho < 0 and dur > 0:
                return {'reactivity': rho, 'duration': dur}

    except:
        pass

    return None


def predict_control_parameters(model, tokenizer, p_init, p_target):
    """V2 Simple: 앞 2개 주고 뒤 2개 생성"""
    # V2 형식: "[P_init, P_target," 로 시작
    prompt = f"[{p_init:.2f}, {p_target:.2f},"

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
    parsed = parse_prediction(generated)

    return parsed, generated


# ============================================================================
# PyRK 시뮬레이션
# ============================================================================
def run_pyrk_simulation(task_id, rho_target, duration_target):
    """주어진 reactivity, duration으로 PyRK 시뮬레이션 실행"""
    try:
        ti = Timer(t0=0 * units.s, tf=300.0 * units.s, dt=1.0 * units.s)
        power_tot = 250e6 * units.watt

        H_FLOW_EQUILIBRIUM = 29492.3829
        h_fuel = 8500.0 * units.watt / (units.meter ** 2 * units.kelvin)
        A = 752.0 * units.meter ** 2
        T_cool_0 = 580.0 * units.kelvin
        T_fuel_0 = T_cool_0 + power_tot / (h_fuel * A)

        alpha_f = -3.0e-5 * units.delta_k / units.kelvin
        alpha_c = -20.0e-5 * units.delta_k / units.kelvin

        class UO2(Material):
            def __init__(self):
                super().__init__(name="uo2", k=3.0 * units.watt / units.meter / units.kelvin,
                                 cp=310.0 * units.joule / units.kg / units.kelvin,
                                 dm=DensityModel(a=10970.0 * units.kg / units.meter ** 3, model="constant"))

        class Water(LiquidMaterial):
            def __init__(self):
                super().__init__(name="water", k=0.56 * units.watt / units.meter / units.kelvin,
                                 cp=5600.0 * units.joule / units.kg / units.kelvin,
                                 dm=DensityModel(a=720.0 * units.kg / units.meter ** 3, model="constant"),
                                 mu=9.0e-5 * units.pascal * units.second)

        fuel = THComponent(name="fuel", mat=UO2(), vol=3.0 * units.meter ** 3, T0=T_fuel_0,
                           alpha_temp=alpha_f, timer=ti, heatgen=True, power_tot=power_tot)
        cool = THComponent(name="cool", mat=Water(), vol=15.0 * units.meter ** 3, T0=T_cool_0,
                           alpha_temp=alpha_c, timer=ti)
        inlet = THComponent(name="inlet", mat=Water(), vol=15.0 * units.meter ** 3, T0=553.0 * units.kelvin,
                            alpha_temp=0.0 * units.delta_k / units.kelvin, timer=ti)

        fuel.add_convection("cool", h=h_fuel, area=A)
        cool.add_convection("fuel", h=h_fuel, area=A)
        cool.add_convection("inlet", h=H_FLOW_EQUILIBRIUM * units.watt / units.meter ** 2 / units.kelvin, area=A)

        class CustomRampReactivity(ReactivityInsertion):
            def __init__(self, timer, start_time, dur, total_rho):
                super().__init__(timer)
                self.start_time = start_time
                self.end_time = start_time + dur
                self.total_rho = total_rho
                self.slope = total_rho / dur if dur > 0 else 0

            def reactivity(self, t_idx=None):
                if t_idx is None:
                    t_idx = 0
                current_time = (self.timer.t0.magnitude + t_idx * self.timer.dt.magnitude)

                if current_time < self.start_time:
                    return 0.0 * units.delta_k
                elif self.start_time <= current_time < self.end_time:
                    dt = current_time - self.start_time
                    return (self.slope * dt) * units.delta_k
                else:
                    return self.total_rho * units.delta_k

        rho_ext = CustomRampReactivity(timer=ti, start_time=100.0,
                                       dur=duration_target, total_rho=rho_target)

        db_name = f"tmp_val_{task_id}_{os.getpid()}.h5"
        db = database.Database(db_name)
        si = sim_info.SimInfo(timer=ti, components=[fuel, cool, inlet],
                              iso="u235", e="thermal", n_precursors=6, n_decay=11,
                              rho_ext=rho_ext, feedback=True, db=db)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            infile = types.SimpleNamespace(kappa=0.0, nsteps=10000)
            sol = solve(si=si, y=si.y, infile=infile)

        db.close_db()
        if os.path.exists(db_name):
            try:
                os.remove(db_name)
            except:
                pass

        sol = np.array(sol)
        power = sol[:, 0] * 250  # MW
        times = np.linspace(0, 300.0, len(power))
        idx_100 = np.argmin(np.abs(times - 100.0))

        power_initial = power[idx_100]
        power_final = power[-1]
        power_peak = np.max(power[idx_100:])
        power_min = np.min(power[idx_100:])

        if np.isnan(power_final) or np.isinf(power_final):
            return None

        return {
            'initial_power': power_initial,
            'final_power': power_final,
            'peak_power': power_peak,
            'min_power': power_min,
            'success': True
        }

    except Exception as e:
        return None


# ============================================================================
# 검증 파이프라인
# ============================================================================
def validate_single_case(model, tokenizer, initial_power, final_power, case_id, total_cases, desc=""):
    """단일 케이스 검증"""
    print(f"\n{'='*70}", flush=True)
    print(f"진행률: [{case_id:4d}/{total_cases}] ({case_id/total_cases*100:.1f}%) - {desc}", flush=True)
    print(f"목표: {initial_power:.2f} MW → {final_power:.2f} MW", flush=True)
    print(f"{'='*70}", flush=True)

    # 1. 모델 예측
    predicted_params, full_prediction = predict_control_parameters(
        model, tokenizer, initial_power, final_power
    )

    if predicted_params is None:
        print(f"  ❌ 파싱 실패!", flush=True)
        print(f"  📝 모델 출력: {full_prediction[:200]}...", flush=True)  # 디버그
        return {
            'case_id': case_id,
            'target_initial': initial_power,
            'target_final': final_power,
            'description': desc,
            'parsing_success': False,
            'full_prediction': full_prediction,
            'error': 'Parsing failed'
        }

    print(f"  ✅ 예측 성공: rho={predicted_params['reactivity']:.6f}, dur={predicted_params['duration']:.2f}", flush=True)
    print(f"  📝 모델 출력: {full_prediction[:150]}...", flush=True)  # 디버그

    # 2. 시뮬레이터 실행
    results = run_pyrk_simulation(case_id, predicted_params['reactivity'], predicted_params['duration'])

    if results is None:
        print(f"  ❌ 시뮬레이션 실패", flush=True)
        return {
            'case_id': case_id,
            'target_initial': initial_power,
            'target_final': final_power,
            'description': desc,
            'parsing_success': True,
            'predicted_params': predicted_params,
            'full_prediction': full_prediction,
            'simulation_success': False,
            'error': 'Simulation failed'
        }

    # 3. 검증 (5단계)
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

    # 정확도별 이모지
    if success_1:
        status = "🎯"  # 매우 정확
    elif success_2:
        status = "✅"  # 정확
    elif success_3:
        status = "👍"  # 좋음
    elif success_5:
        status = "📊"  # 허용
    elif success_10:
        status = "⚠️"  # 관대
    else:
        status = "❌"  # 실패

    error_pct = final_error / final_power * 100 if final_power > 0 else 0
    print(f"  {status} 최종 출력: {results['final_power']:.2f} MW (오차: {error_pct:.2f}%)", flush=True)

    return {
        'case_id': case_id,
        'target_initial': initial_power,
        'target_final': final_power,
        'description': desc,
        'parsing_success': True,
        'predicted_params': predicted_params,
        'full_prediction': full_prediction,
        'simulation_success': True,
        'actual_initial': results['initial_power'],
        'actual_final': results['final_power'],
        'actual_peak': results['peak_power'],
        'actual_min': results['min_power'],
        'final_error': final_error,
        'final_error_pct': error_pct,
        'validation_success_1': success_1,
        'validation_success_2': success_2,
        'validation_success_3': success_3,
        'validation_success_5': success_5,
        'validation_success_10': success_10
    }


def generate_test_cases(num_cases=2000):
    """
    테스트 케이스 생성 (SMR 출력 감소 시나리오)
    학습 데이터 기준:
    - P_init = 245.44 MW (고정)
    - P_target = 최대 50% 감소 (122.72 ~ 240 MW)

    구간:
    - 소폭: 0~10% 감소 (220.9 ~ 240 MW)
    - 중폭: 10~30% 감소 (171.8 ~ 220.9 MW)
    - 대폭: 30~50% 감소 (122.7 ~ 171.8 MW)
    """
    test_cases = []

    # 학습 데이터의 실제 값
    P_INIT = 245.44  # 학습 데이터의 power_initial (고정값)
    P_TARGET_MIN = P_INIT * 0.5   # 50% 감소 = 122.72 MW
    P_TARGET_MAX = 240.0          # 학습 데이터 max 근처

    # 기본 케이스들 (각 구간별 대표값)
    base_cases = [
        # 소폭 감소 (0~10%)
        (P_INIT, 238.0, "소폭 감소 3%"),
        (P_INIT, 232.0, "소폭 감소 5%"),
        (P_INIT, 225.0, "소폭 감소 8%"),
        (P_INIT, 221.0, "소폭 감소 10%"),
        # 중폭 감소 (10~30%)
        (P_INIT, 215.0, "중폭 감소 12%"),
        (P_INIT, 200.0, "중폭 감소 19%"),
        (P_INIT, 185.0, "중폭 감소 25%"),
        (P_INIT, 172.0, "중폭 감소 30%"),
        # 대폭 감소 (30~50%)
        (P_INIT, 165.0, "대폭 감소 33%"),
        (P_INIT, 150.0, "대폭 감소 39%"),
        (P_INIT, 135.0, "대폭 감소 45%"),
        (P_INIT, 123.0, "대폭 감소 50%"),
    ]

    test_cases.extend(base_cases)

    # 나머지는 랜덤 생성
    np.random.seed(42)
    remaining = num_cases - len(base_cases)

    for i in range(remaining):
        # 50% 감소까지만 랜덤 (122.72 ~ 240 MW)
        target = np.random.uniform(P_TARGET_MIN, P_TARGET_MAX)
        change = (P_INIT - target) / P_INIT * 100
        desc = f"감소 {change:.1f}%"
        test_cases.append((P_INIT, round(target, 2), desc))

    return test_cases


def run_validation_suite(model, tokenizer, num_cases=2000):
    """여러 테스트 케이스 검증"""
    print("\n" + "🎯 " * 20)
    print(f"PyRK SMR Inverse Model → 시뮬레이터 검증 ({num_cases} 케이스)")
    print(f"출력 감소 시나리오: 245.44 MW → 122~240 MW (최대 50% 감소)")
    print("🎯 " * 20)

    test_cases = generate_test_cases(num_cases)

    results = []
    total = len(test_cases)

    start_time = time_module.time()

    for i, (initial, final, desc) in enumerate(test_cases, 1):
        result = validate_single_case(model, tokenizer, initial, final, i, total, desc)
        results.append(result)

        # 50개마다 중간 통계
        if i % 50 == 0:
            elapsed = time_module.time() - start_time
            rate = i / elapsed
            eta = (total - i) / rate / 60

            temp_parsing = sum(1 for r in results if r.get('parsing_success', False))
            temp_sim = sum(1 for r in results if r.get('simulation_success', False))
            temp_val_1 = sum(1 for r in results if r.get('validation_success_1', False))
            temp_val_2 = sum(1 for r in results if r.get('validation_success_2', False))
            temp_val_5 = sum(1 for r in results if r.get('validation_success_5', False))

            print(f"\n  === 중간 통계 ({i}개) | {rate:.1f} 케이스/초 | ETA: {eta:.1f}분 ===", flush=True)
            print(f"  파싱: {temp_parsing}/{i} ({temp_parsing/i*100:.1f}%)", flush=True)
            print(f"  시뮬: {temp_sim}/{i} ({temp_sim/i*100:.1f}%)", flush=True)
            print(f"  검증(±1%): {temp_val_1}/{i} ({temp_val_1/i*100:.1f}%)", flush=True)
            print(f"  검증(±2%): {temp_val_2}/{i} ({temp_val_2/i*100:.1f}%)", flush=True)
            print(f"  검증(±5%): {temp_val_5}/{i} ({temp_val_5/i*100:.1f}%)\n", flush=True)

    total_time = time_module.time() - start_time

    # 전체 통계
    print("\n\n" + "=" * 80)
    print("📊 전체 검증 결과 (SMR 출력 감소 시나리오)")
    print("=" * 80)

    total = len(results)
    parsing_success = sum(1 for r in results if r.get('parsing_success', False))
    simulation_success = sum(1 for r in results if r.get('simulation_success', False))
    validation_success_1 = sum(1 for r in results if r.get('validation_success_1', False))
    validation_success_2 = sum(1 for r in results if r.get('validation_success_2', False))
    validation_success_3 = sum(1 for r in results if r.get('validation_success_3', False))
    validation_success_5 = sum(1 for r in results if r.get('validation_success_5', False))
    validation_success_10 = sum(1 for r in results if r.get('validation_success_10', False))

    print(f"\n총 케이스:              {total}개")
    print(f"소요 시간:              {total_time/60:.1f}분 ({total_time:.0f}초)")
    print(f"처리 속도:              {total/total_time:.2f} 케이스/초")
    print(f"파싱 성공:              {parsing_success}개 ({parsing_success / total * 100:.1f}%)")

    if parsing_success > 0:
        print(f"시뮬레이션 성공:        {simulation_success}개 ({simulation_success / parsing_success * 100:.1f}% of parsed)")

    if simulation_success > 0:
        print(f"\n🎯 검증 성공률 (5단계):")
        print(f"  ±1%  (매우 정확):     {validation_success_1}개 ({validation_success_1 / simulation_success * 100:.1f}%)")
        print(f"  ±2%  (정확):          {validation_success_2}개 ({validation_success_2 / simulation_success * 100:.1f}%)")
        print(f"  ±3%  (좋음):          {validation_success_3}개 ({validation_success_3 / simulation_success * 100:.1f}%)")
        print(f"  ±5%  (허용):          {validation_success_5}개 ({validation_success_5 / simulation_success * 100:.1f}%)")
        print(f"  ±10% (관대):          {validation_success_10}개 ({validation_success_10 / simulation_success * 100:.1f}%)")

    # 오차 통계
    valid_results = [r for r in results if r.get('final_error_pct') is not None]
    if valid_results:
        errors = [r['final_error_pct'] for r in valid_results]
        print(f"\n최종 출력 오차 통계 (%):")
        print(f"  평균:     {np.mean(errors):.2f}%")
        print(f"  중앙값:   {np.median(errors):.2f}%")
        print(f"  최대:     {np.max(errors):.2f}%")
        print(f"  최소:     {np.min(errors):.2f}%")
        print(f"  표준편차: {np.std(errors):.2f}%")

    # 구간별 성공률 (P_init=245.44 기준)
    # 소폭: 0~10% (≥220.9), 중폭: 10~30% (171.8~220.9), 대폭: 30~50% (<171.8)
    print(f"\n구간별 성공률 (±5% 기준):")
    ranges = [
        ("소폭 감소 (0~10%)", lambda r: r['target_final'] >= 220.9),
        ("중폭 감소 (10~30%)", lambda r: 171.8 <= r['target_final'] < 220.9),
        ("대폭 감소 (30~50%)", lambda r: r['target_final'] < 171.8),
    ]

    for range_name, condition in ranges:
        range_results = [r for r in results if r.get('target_final') and condition(r)]
        if range_results:
            range_success = sum(1 for r in range_results if r.get('validation_success_5', False))
            print(f"  {range_name:15s}: {range_success}/{len(range_results)} ({range_success/len(range_results)*100:.1f}%)")

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = VALIDATION_DIR / f"validation_results_smr_{num_cases}cases_{timestamp}.json"

    # JSON 직렬화 가능하도록 변환
    json_results = []
    for r in results:
        jr = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                jr[k] = float(v)
            elif isinstance(v, np.ndarray):
                jr[k] = v.tolist()
            else:
                jr[k] = v
        json_results.append(jr)

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 결과 저장: {results_file}")

    # 실패 케이스 요약
    failed_cases = [r for r in results if not r.get('validation_success_5', False)]
    if failed_cases and len(failed_cases) <= 30:
        print(f"\n⚠️ 실패/경계 케이스 ({len(failed_cases)}개):")
        for r in failed_cases[:30]:
            if r.get('final_error_pct'):
                print(f"  케이스 {r['case_id']:4d}: {r['target_initial']:.0f}→{r['target_final']:.0f} MW, "
                      f"실제 {r['actual_final']:.1f} MW, 오차 {r['final_error_pct']:.2f}%")

    return results


# ============================================================================
# 메인
# ============================================================================
def main():
    print("\n" + "=" * 80)
    print("PyRK SMR Inverse Model V2 → 시뮬레이터 검증 도구")
    print("V2 Simple Format: [P_init, P_target, rho, dur]")
    print("출력 감소 제어 시나리오 (245.44 MW → 122~240 MW, 최대 50% 감소)")
    print("=" * 80)
    print(f"\n경로 설정:")
    print(f"  스크립트: {SCRIPT_DIR}")
    print(f"  Phase 1:  {PHASE1_MODEL_PATH}")
    print(f"  Phase 2:  {PHASE2_MODEL_PATH}")

    # 디렉토리 생성
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    # 모델 경로 확인
    if not PHASE1_MODEL_PATH.exists():
        print(f"\n❌ Phase 1 모델을 찾을 수 없습니다: {PHASE1_MODEL_PATH}")
        return
    if not PHASE2_MODEL_PATH.exists():
        print(f"\n❌ Phase 2 모델을 찾을 수 없습니다: {PHASE2_MODEL_PATH}")
        return

    # 모델 로드
    model, tokenizer = load_model()

    # 검증 케이스 수 선택
    print("\n📊 검증 케이스 수 선택:")
    print("  [1] 500개   (빠른 테스트, ~10분)")
    print("  [2] 1,000개 (±1.9% 신뢰구간, ~20분)")
    print("  [3] 2,000개 (±1.3% 신뢰구간, ~40분) ⭐ 추천")
    print("  [4] 3,000개 (±1.1% 신뢰구간, ~60분)")

    choice = input("\n선택 (1-4, 기본 3): ").strip() or "3"

    num_cases_map = {
        "1": 500,
        "2": 1000,
        "3": 2000,
        "4": 3000
    }

    num_cases = num_cases_map.get(choice, 2000)

    # 검증 실행
    print(f"\n🚀 {num_cases}개 케이스 검증을 시작합니다...")
    print(f"  예상 소요 시간: ~{num_cases//50}분")
    input("\nEnter를 눌러 시작...")

    results = run_validation_suite(model, tokenizer, num_cases=num_cases)

    print("\n" + "✅ " * 20)
    print("PyRK SMR V2 검증 완료!")
    print("✅ " * 20)


if __name__ == "__main__":
    main()