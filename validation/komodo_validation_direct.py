"""
모델 예측 → 시뮬레이터 검증 스크립트 (Direct Model Validation)
- Phase 1: Hugging Face 원본 모델 사용
- Phase 2: 로컬 LoRA 어댑터 사용
- 기능 추가: TQDM 진행률 표시 + 초반 5개 저장 검증
"""

import os
import sys

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
from tqdm import tqdm  # [추가] 진행률 표시

# ============================================================================
# 설정
# ============================================================================

# 현재 스크립트 위치
SCRIPT_DIR = Path(__file__).resolve().parent  # validation/
REPO_ROOT = SCRIPT_DIR.parent                  # repo root

# [경로 설정] training 폴더에서 Direct LoRA 모델 로드
TRAINING_DIR = REPO_ROOT / "training"

# 모델 경로 설정
PHASE1_MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
PHASE2_MODEL_PATH = TRAINING_DIR / "models" / "smollm2_supervised_lora_v7_numeric_simple_100k_hf_base" / "final_model"

# 시뮬레이터 설정 (override with KOMODO_EXECUTABLE env var if installed elsewhere)
KOMODO_EXECUTABLE = Path(os.environ.get("KOMODO_EXECUTABLE", str(REPO_ROOT / "komodo")))
TEMPLATE_FILE = REPO_ROOT / "data_generation" / "template"

# 검증 결과 저장 디렉토리
VALIDATION_DIR = SCRIPT_DIR / "validation_runs_direct_model"

# 추론 설정
INFERENCE_TEMPERATURE = 0.05
INFERENCE_TOP_P = 0.9
INFERENCE_MAX_TOKENS = 50


# ============================================================================
# 유틸리티 (데이터 정제 및 저장)
# ============================================================================

def make_json_serializable(obj):
    """모든 데이터 타입을 JSON 저장 가능한 형태로 재귀적 변환"""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return str(obj)
    return obj

def save_results(results, filename):
    """결과를 JSON으로 안전하게 저장"""
    clean_results = make_json_serializable(results)
    file_path = VALIDATION_DIR / filename
    with open(file_path, 'w') as f:
        json.dump(clean_results, f, indent=2)
    return file_path

# ============================================================================
# 모델 로드
# ============================================================================

def check_path(path, name):
    tqdm.write(f"🔍 {name} 경로 확인 중: {path}")
    if not path.exists():
        tqdm.write(f"\n❌ [오류] {name} 폴더가 없습니다!")
        return False
    return True

def load_model():
    print("=" * 80)
    print("모델 로딩 (HF Base + Local LoRA)")
    print("=" * 80)

    if not check_path(PHASE2_MODEL_PATH, "Phase 2 (LoRA)"):
        sys.exit(1)

    print(f"\n🚀 Phase 1 모델 로드 중: {PHASE1_MODEL_ID}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            PHASE1_MODEL_ID,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        print("   ✅ Phase 1 (Base) 로드 성공")
    except Exception as e:
        print(f"\n❌ Phase 1 로드 실패: {e}")
        sys.exit(1)

    print(f"🚀 Phase 2 LoRA 로드 중: {PHASE2_MODEL_PATH.name}")
    try:
        model = PeftModel.from_pretrained(
            base_model,
            str(PHASE2_MODEL_PATH),
            local_files_only=True
        )
        print("   ✅ Phase 2 (LoRA) 로드 성공")
    except Exception as e:
        print(f"\n❌ Phase 2 LoRA 로드 실패: {e}")
        sys.exit(1)

    print(f"🚀 토크나이저 로드 시도...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(PHASE2_MODEL_PATH), local_files_only=True)
        print("   ✅ Phase 2 경로에서 토크나이저 로드됨")
    except:
        print("   ⚠️ Phase 2에 토크나이저 없음, Phase 1 (HF Hub)에서 로드...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(PHASE1_MODEL_ID)
            print("   ✅ Phase 1 (HF Hub)에서 토크나이저 로드됨")
        except Exception as e:
            print(f"\n❌ 토크나이저 로드 실패: {e}")
            sys.exit(1)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    return model, tokenizer


# ============================================================================
# 모델 예측 및 시뮬레이션
# ============================================================================

def parse_prediction_simple(prediction_text):
    numbers = re.findall(r'-?\d+\.?\d*', prediction_text)
    if len(numbers) < 8: return None
    try:
        values = [float(x) for x in numbers[-6:]]
        return {
            'b1_pos': values[0], 'b1_time': values[1], 'b1_speed': values[2],
            'b2_pos': values[3], 'b2_time': values[4], 'b2_speed': values[5]
        }
    except:
        return None

def predict_rod_parameters(model, tokenizer, initial_power, final_power):
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
    parsed = parse_prediction_simple(generated)
    return parsed, generated

def create_input_file(template_path, scenario, output_path):
    if not template_path.exists():
         raise FileNotFoundError(f"Template not found: {template_path}")
    with open(template_path, 'r', encoding='utf-8') as f:
        content = f.read()
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content.format(**scenario))

def run_komodo(input_path):
    executable_path = str(KOMODO_EXECUTABLE)
    try:
        subprocess.run([executable_path, input_path.name], cwd=str(input_path.parent),
                       check=True, capture_output=True, text=True, timeout=60)
        return True, None
    except Exception as e:
        return False, str(e)

def parse_simulation_results(output_file):
    try:
        with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        powers = []
        in_results = False
        for line in lines:
            if "TRANSIENT RESULTS" in line:
                in_results = True
                continue
            if in_results and line.strip() and not line.strip().startswith('Time') and not line.strip().startswith('CPU'):
                parts = line.split()
                if len(parts) >= 4:
                    try: powers.append(float(parts[3]))
                    except: pass

        if not powers: return None
        return {'initial_power': powers[0], 'final_power': powers[-1], 'peak_power': max(powers)}
    except:
        return None

def cleanup_case_files(run_dir):
    try: shutil.rmtree(run_dir)
    except: pass

def validate_single_case(model, tokenizer, initial_power, final_power, case_id):
    # TQDM과 충돌하지 않게 tqdm.write 사용 가능하지만, 여기선 로그를 최소화함
    parsed, raw_output = predict_rod_parameters(model, tokenizer, initial_power, final_power)

    if not parsed:
        return {'case_id': case_id, 'parsing_success': False, 'full_prediction': raw_output}

    run_dir = VALIDATION_DIR / f"case_{case_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        create_input_file(TEMPLATE_FILE, parsed, run_dir / "input.inp")
        success, err = run_komodo(run_dir / "input.inp")

        if not success:
            cleanup_case_files(run_dir)
            return {'case_id': case_id, 'parsing_success': True, 'simulation_success': False, 'error': err}

        res = parse_simulation_results(run_dir / "input.inp.out")
        cleanup_case_files(run_dir)

        if not res:
            return {'case_id': case_id, 'parsing_success': True, 'simulation_success': True, 'result_parsing_success': False}

        err_val = abs(res['final_power'] - final_power)
        base_val = abs(final_power) if abs(final_power) > 1e-6 else 1.0
        success_5 = err_val <= (base_val * 0.05)

        return {
            'case_id': case_id,
            'parsing_success': True,
            'simulation_success': True,
            'result_parsing_success': True,
            'target_final': float(final_power),
            'actual_final': float(res['final_power']),
            'final_error': float(err_val),
            'validation_success_5': bool(success_5)
        }
    except Exception as e:
        cleanup_case_files(run_dir)
        return {'case_id': case_id, 'parsing_success': True, 'error': str(e)}

# ============================================================================
# 메인 실행 로직 (TQDM + Early Save)
# ============================================================================

def run_suite(num_cases):
    model, tokenizer = load_model()
    cases = [(1.0, np.round(np.random.uniform(0.5, 1.5), 4), "Random") for _ in range(num_cases)]

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n🚀 검증 시작 ({num_cases} 케이스)")
    print(f"📂 저장 폴더: {VALIDATION_DIR}")
    print("-" * 60)

    # TQDM progress bar
    pbar = tqdm(enumerate(cases), total=num_cases, desc="진행 중", unit="case")

    for i, (init, final, desc) in pbar:
        # 실행
        res = validate_single_case(model, tokenizer, init, final, i+1)
        results.append(res)

        # 상태 표시 업데이트
        if res.get('validation_success_5'):
            status = "Success"
        elif not res.get('parsing_success'):
            status = "ParseErr"
        else:
            status = f"Err:{res.get('final_error', 9.9):.3f}"

        pbar.set_postfix(last_status=status)

        # [추가] 초기 검증 저장 (5개 실행 후)
        if i == 4:
            tqdm.write("\n💾 [중간 점검] 초반 5개 케이스 저장 중...")
            try:
                preview_path = save_results(results, f"preview_first_5_{timestamp}.json")
                tqdm.write(f"   ✅ 저장 성공: {preview_path.name}\n")
            except Exception as e:
                tqdm.write(f"   ❌ 저장 실패: {e}\n")

    # 최종 저장
    results_file = f"results_{num_cases}cases_{timestamp}.json"
    save_path = save_results(results, results_file)

    # 최종 통계
    success_cnt = sum(1 for r in results if r.get('validation_success_5'))
    print("\n" + "=" * 60)
    print(f"📊 최종 결과: {success_cnt}/{num_cases} 성공 (±5%)")
    print(f"📂 전체 결과 파일: {save_path}")
    print("=" * 60)

def main():
    print("\n" + "=" * 80)
    print("Direct Model 시뮬레이터 검증 도구 (TQDM + Safe Save)")
    print("=" * 80)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    print("\n📊 검증 케이스 수 선택:")
    print("  [1] 100개")
    print("  [2] 1,000개")
    print("  [3] 2,000개")
    choice = input("\n선택 (1-3, 기본 2): ").strip() or "2"
    num_cases = { "1": 100, "2": 1000, "3": 2000 }.get(choice, 1000)

    try:
        run_suite(num_cases)
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자 중단! 프로그램을 종료합니다.")

if __name__ == "__main__":
    main()