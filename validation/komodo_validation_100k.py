"""
모델 예측 → 시뮬레이터 검증 스크립트 (V7.1 Simple)
- Phase 2 V7.1 Simple 모델로 제어봉 파라미터 예측
- 예측된 파라미터로 KOMODO 시뮬레이터 실행
- 실제 도달 출력 vs 목표 출력 비교
- 검증 완료 후 inp/out 파일 자동 삭제 (JSON만 보존)
- 5단계 정확도 분석 (±1%, ±2%, ±3%, ±5%, ±10%)
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
# 설정
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent  # validation/
REPO_ROOT = SCRIPT_DIR.parent                  # repo root

# 모델 경로 (V7.1 Simple!)
TRAINING_DIR = REPO_ROOT / "training"
PHASE1_MODEL_PATH = TRAINING_DIR / "models/smollm2_unsupervised_numeric_100k/final_model"
PHASE2_MODEL_PATH = TRAINING_DIR / "models/smollm2_supervised_lora_v7_numeric_simple_100k/final_model"  # V7.1!

# 시뮬레이터 설정 (override with KOMODO_EXECUTABLE env var if installed elsewhere)
KOMODO_EXECUTABLE = Path(os.environ.get("KOMODO_EXECUTABLE", str(REPO_ROOT / "komodo")))
TEMPLATE_FILE = REPO_ROOT / "data_generation" / "template"
VALIDATION_DIR = SCRIPT_DIR / "validation_runs_v7_simple"  # V7.1 전용 디렉토리

# 추론 설정
INFERENCE_TEMPERATURE = 0.05
INFERENCE_TOP_P = 0.9
INFERENCE_MAX_TOKENS = 50


# ============================================================================
# 모델 로드
# ============================================================================

def load_model():
    """Phase 2 V7.1 Simple 모델 로드 (Phase 1 + LoRA)"""
    print("=" * 80)
    print("모델 로딩 중... (V7.1 Simple)")
    print("=" * 80)

    # Phase 1 기본 모델 로드
    print(f"\n✓ Phase 1 모델: {PHASE1_MODEL_PATH}")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(PHASE1_MODEL_PATH),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # LoRA adapter 로드
    print(f"✓ Phase 2 LoRA (V7.1 Simple): {PHASE2_MODEL_PATH}")
    model = PeftModel.from_pretrained(base_model, str(PHASE2_MODEL_PATH))

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(str(PHASE2_MODEL_PATH))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    print("\n✓ V7.1 Simple 모델 로드 완료!")
    print("  - 화살표 없음, 쉼표만 사용")
    print("  - 형식: [1.0, 1.5, 180, 0.0, ...]")
    print("=" * 80)

    return model, tokenizer


# ============================================================================
# 모델 예측 - V7.1 Simple!
# ============================================================================

def parse_prediction_simple(prediction_text):
    """
    V7.1 Simple 형식 파싱: 숫자 8개 추출
    예: "[1.0, 1.5, 180, 0.0, 0.0, 100, 0.0, 0.0]"
    앞 2개는 입력, 뒤 6개는 출력
    """
    # 숫자 추출
    numbers = re.findall(r'[\d.]+', prediction_text)

    # 8개 이상 있어야 함
    if len(numbers) < 8:
        return None

    try:
        # 마지막 6개가 출력 (제어봉 파라미터)
        values = [float(x) for x in numbers[-6:]]

        return {
            'b1_pos': values[0],
            'b1_time': values[1],
            'b1_speed': values[2],
            'b2_pos': values[3],
            'b2_time': values[4],
            'b2_speed': values[5]
        }
    except:
        return None


def predict_rod_parameters(model, tokenizer, initial_power, final_power):
    """모델로 제어봉 파라미터 예측 (V7.1 Simple)"""
    # V7.1 형식! 쉼표로 끝!
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

    # 파싱
    parsed = parse_prediction_simple(generated)

    return parsed, generated


# ============================================================================
# 시뮬레이터 실행
# ============================================================================

def create_input_file(template_path, scenario, output_path):
    """템플릿으로 KOMODO 입력 파일 생성"""
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()

    new_content = template_content.format(**scenario)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(new_content)


def run_komodo(input_path):
    """KOMODO 시뮬레이터 실행"""
    work_dir = input_path.parent
    input_filename = input_path.name

    executable_path = str(KOMODO_EXECUTABLE)

    try:
        result = subprocess.run(
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
        return False, "시뮬레이션 타임아웃 (60초 초과)"
    except subprocess.CalledProcessError as e:
        return False, f"시뮬레이션 실패:\n{e.stderr}"
    except FileNotFoundError:
        return False, f"KOMODO 실행 파일을 찾을 수 없음: {executable_path}"


def parse_simulation_results(output_file):
    """시뮬레이션 결과에서 출력 데이터 추출"""
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
# 파일 정리
# ============================================================================

def cleanup_case_files(run_dir):
    """케이스 디렉토리의 inp/out 파일 삭제"""
    try:
        # 디렉토리 전체 삭제
        if run_dir.exists():
            shutil.rmtree(run_dir)
    except Exception as e:
        print(f"  ⚠️ 파일 삭제 실패: {e}")


# ============================================================================
# 검증 파이프라인
# ============================================================================

def validate_single_case(model, tokenizer, initial_power, final_power, case_id, total_cases, desc=""):
    """단일 케이스 검증"""
    print(f"\n{'='*70}", flush=True)
    print(f"진행률: [{case_id:4d}/{total_cases}] ({case_id/total_cases*100:.1f}%) - {desc}", flush=True)
    print(f"목표: {initial_power} → {final_power}", flush=True)
    print(f"{'='*70}", flush=True)

    # 1. 모델 예측
    predicted_params, full_prediction = predict_rod_parameters(
        model, tokenizer, initial_power, final_power
    )

    if predicted_params is None:
        print(f"  ❌ 파싱 실패!", flush=True)
        return {
            'case_id': case_id,
            'target_initial': initial_power,
            'target_final': final_power,
            'description': desc,
            'parsing_success': False,
            'full_prediction': full_prediction,
            'error': 'Parsing failed'
        }

    print(f"  ✅ 예측 성공: {full_prediction}", flush=True)

    # 2. 입력 파일 생성
    run_dir = VALIDATION_DIR / f"case_{case_id:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    input_file = run_dir / "input.inp"
    create_input_file(TEMPLATE_FILE, predicted_params, input_file)

    # 3. 시뮬레이터 실행
    success, error = run_komodo(input_file)

    if not success:
        print(f"  ❌ 시뮬레이션 실패: {error}", flush=True)
        cleanup_case_files(run_dir)  # 실패해도 삭제!
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

    # 4. 결과 파싱
    output_file = run_dir / "input.inp.out"
    results = parse_simulation_results(output_file)

    if results is None:
        print(f"  ❌ 결과 파싱 실패", flush=True)
        cleanup_case_files(run_dir)  # 실패해도 삭제!
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

    # 5. 검증 (5단계)
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

    print(f"  {status} 최종 출력: {results['final_power']:.3f} (오차: {final_error:.4f})", flush=True)

    # 6. 파일 정리 (성공한 케이스도 삭제!)
    cleanup_case_files(run_dir)
    print(f"  🗑️  임시 파일 삭제 완료", flush=True)

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
    """테스트 케이스 생성"""
    test_cases = []

    # 기본 케이스들 (확실히 포함)
    base_cases = [
        (1.0, 1.05, "소폭 증가 5%"),
        (1.0, 1.10, "증가 10%"),
        (1.0, 1.20, "중폭 증가 20%"),
        (1.0, 1.30, "대폭 증가 30%"),
        (1.0, 1.40, "큰 폭 증가 40%"),
        (1.0, 1.50, "매우 큰 증가 50%"),

        (1.0, 0.95, "소폭 감소 5%"),
        (1.0, 0.90, "감소 10%"),
        (1.0, 0.80, "중폭 감소 20%"),
        (1.0, 0.70, "대폭 감소 30%"),
        (1.0, 0.60, "큰 폭 감소 40%"),
        (1.0, 0.50, "매우 큰 감소 50%"),
    ]

    test_cases.extend(base_cases)

    # 나머지는 랜덤 생성
    np.random.seed(42)
    remaining = num_cases - len(base_cases)

    for i in range(remaining):
        # 0.5 ~ 1.5 범위에서 랜덤
        final = np.random.uniform(0.5, 1.5)
        change = (final - 1.0) / 1.0 * 100
        desc = f"{'증가' if final > 1.0 else '감소'} {abs(change):.1f}%"
        test_cases.append((1.0, round(final, 5), desc))

    return test_cases


def run_validation_suite(model, tokenizer, num_cases=100):
    """여러 테스트 케이스 검증"""
    print("\n" + "🎯 " * 20)
    print(f"V7.1 Simple 모델 → 시뮬레이터 검증 ({num_cases} 케이스)")
    print("🎯 " * 20)
    print("\n💡 검증 완료 후 inp/out 파일은 자동 삭제됩니다 (JSON만 보존)")

    test_cases = generate_test_cases(num_cases)

    results = []
    total = len(test_cases)
    for i, (initial, final, desc) in enumerate(test_cases, 1):
        result = validate_single_case(model, tokenizer, initial, final, i, total, desc)
        results.append(result)

        # 50개마다 중간 통계 (2000개 기준)
        if i % 50 == 0:
            temp_parsing = sum(1 for r in results if r.get('parsing_success', False))
            temp_sim = sum(1 for r in results if r.get('simulation_success', False))
            temp_val_1 = sum(1 for r in results if r.get('validation_success_1', False))
            temp_val_2 = sum(1 for r in results if r.get('validation_success_2', False))
            temp_val_5 = sum(1 for r in results if r.get('validation_success_5', False))
            print(f"\n  === 중간 통계 ({i}개) ===", flush=True)
            print(f"  파싱: {temp_parsing}/{i} ({temp_parsing/i*100:.1f}%)", flush=True)
            print(f"  시뮬: {temp_sim}/{i} ({temp_sim/i*100:.1f}%)", flush=True)
            print(f"  검증(±1%): {temp_val_1}/{i} ({temp_val_1/i*100:.1f}%)", flush=True)
            print(f"  검증(±2%): {temp_val_2}/{i} ({temp_val_2/i*100:.1f}%)", flush=True)
            print(f"  검증(±5%): {temp_val_5}/{i} ({temp_val_5/i*100:.1f}%)\n", flush=True)

    # 전체 통계
    print("\n\n" + "=" * 80)
    print("📊 전체 검증 결과 (V7.1 Simple)")
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

    # V7과 비교
    print(f"\n📊 V7 vs V7.1 비교:")
    print(f"  V7 파싱:        90.0%")
    print(f"  V7.1 파싱:      {parsing_success / total * 100:.1f}% ⭐")
    if simulation_success > 0:
        print(f"  V7 검증(±5%):   22.2%")
        print(f"  V7.1 검증(±5%): {validation_success_5 / simulation_success * 100:.1f}%")
        print(f"  V7 검증(±10%):  66.7%")
        print(f"  V7.1 검증(±10%): {validation_success_10 / simulation_success * 100:.1f}%")

    # 오차 통계
    valid_results = [r for r in results if r.get('final_error') is not None]
    if valid_results:
        final_errors = [r['final_error'] for r in valid_results]
        print(f"\n최종 출력 오차 통계:")
        print(f"  평균:   {np.mean(final_errors):.4f}")
        print(f"  중앙값: {np.median(final_errors):.4f}")
        print(f"  최대:   {np.max(final_errors):.4f}")
        print(f"  최소:   {np.min(final_errors):.4f}")
        print(f"  표준편차: {np.std(final_errors):.4f}")

    # 구간별 성공률
    print(f"\n구간별 성공률 (±5% 기준):")
    ranges = [
        ("소폭 (±10%)", lambda r: abs(r['target_final'] - 1.0) <= 0.1),
        ("중폭 (±30%)", lambda r: 0.1 < abs(r['target_final'] - 1.0) <= 0.3),
        ("대폭 (±50%)", lambda r: abs(r['target_final'] - 1.0) > 0.3),
    ]

    for range_name, condition in ranges:
        range_results = [r for r in results if r.get('target_final') and condition(r)]
        if range_results:
            range_success = sum(1 for r in range_results if r.get('validation_success_5', False))
            print(f"  {range_name:15s}: {range_success}/{len(range_results)} ({range_success/len(range_results)*100:.1f}%)")

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = VALIDATION_DIR / f"validation_results_v7_simple_{num_cases}cases_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ 결과 저장: {results_file}")
    print(f"💾 임시 파일은 모두 삭제되었습니다 (JSON만 보존)")

    # 실패 케이스 요약
    failed_cases = [r for r in results if not r.get('validation_success_5', False)]
    if failed_cases and len(failed_cases) <= 20:
        print(f"\n⚠️ 실패/경계 케이스 ({len(failed_cases)}개):")
        for r in failed_cases[:20]:
            if r.get('final_error'):
                print(f"  케이스 {r['case_id']:3d}: {r['target_initial']:.2f}→{r['target_final']:.2f}, "
                      f"실제 {r['actual_final']:.3f}, 오차 {r['final_error']:.4f}")

    return results


# ============================================================================
# 메인
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("V7.1 Simple 모델 → 시뮬레이터 검증 도구")
    print("=" * 80)

    # 디렉토리 생성
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    # 모델 로드
    model, tokenizer = load_model()

    # 검증 케이스 수 선택
    print("\n📊 검증 케이스 수 선택:")
    print("  [1] 1,000개 (±1.9% 신뢰구간, 2.5-4시간)")
    print("  [2] 2,000개 (±1.3% 신뢰구간, 5-8시간) ⭐ 추천")
    print("  [3] 3,000개 (±1.1% 신뢰구간, 7.5-12시간)")

    choice = input("\n선택 (1-3, 기본 2): ").strip() or "2"

    num_cases_map = {
        "1": 1000,
        "2": 2000,
        "3": 3000
    }

    num_cases = num_cases_map.get(choice, 2000)

    # 검증 실행
    print(f"\n🚀 {num_cases}개 케이스 검증을 시작합니다...")
    print(f"  예상 소요 시간: {num_cases*0.15//60:.0f}-{num_cases*0.25//60:.0f}시간")
    print("  💡 inp/out 파일은 검증 후 자동 삭제됩니다")
    input("\nEnter를 눌러 시작...")

    results = run_validation_suite(model, tokenizer, num_cases=num_cases)

    print("\n" + "✅ " * 20)
    print("V7.1 Simple 검증 완료!")
    print("✅ " * 20)


if __name__ == "__main__":
    main()