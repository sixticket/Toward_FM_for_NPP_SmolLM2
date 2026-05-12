"""
SmolLM2-360M Supervised Training for KOMODO (Phase 2) - V7.1 Simple 100K
+ V7.1: 화살표 제거! 완전히 쉼표로만 통합
+ 데이터 형식: "[1.0, 1.5, 180, 0.0, 0.0, 100, 0.0, 0.0]"
+ Temperature 0.05 (더 결정적)
+ Phase 1 V7 100K 모델 기반
+ 100,000개 데이터로 학습!
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from datetime import timedelta
import re
import json


# ============================================================================
# Progress Callback
# ============================================================================

class ProgressCallback(TrainerCallback):
    """학습 진행 상황 표시"""

    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print("\n" + "🚀 " * 20)
        print("SmolLM2 Supervised Training 시작! (Phase 2 - V7.1 Simple 100K!)")
        print("🚀 " * 20 + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            current_step = state.global_step
            progress = (current_step / self.total_steps) * 100

            elapsed = time.time() - self.start_time
            elapsed_str = str(timedelta(seconds=int(elapsed)))

            if current_step > 0:
                avg_time_per_step = elapsed / current_step
                remaining_steps = self.total_steps - current_step
                remaining = avg_time_per_step * remaining_steps
                remaining_str = str(timedelta(seconds=int(remaining)))
            else:
                remaining_str = "계산 중..."

            loss = logs.get('loss', 'N/A')
            learning_rate = logs.get('learning_rate', 'N/A')

            bar_length = 30
            filled = int(bar_length * current_step / self.total_steps)
            bar = '█' * filled + '░' * (bar_length - filled)

            print(f"\n{'=' * 80}")
            print(f"📊 Step {current_step}/{self.total_steps} [{bar}] {progress:.1f}%")
            print(f"{'=' * 80}")
            print(f"📈 Loss: {loss:.4f}" if isinstance(loss, float) else f"📈 Loss: {loss}")
            print(f"🎯 Learning Rate: {learning_rate:.2e}" if isinstance(learning_rate,
                                                                        float) else f"🎯 Learning Rate: {learning_rate}")
            print(f"⏱️  경과 시간: {elapsed_str}")
            print(f"⏳ 남은 시간 (예상): {remaining_str}")

            if 'epoch' in logs:
                print(f"📚 Epoch: {logs['epoch']:.2f}")

            print(f"{'=' * 80}\n")

    def on_epoch_end(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch)
        print("\n" + "🎉 " * 20)
        print(f"Epoch {current_epoch} 완료!")
        print("🎉 " * 20 + "\n")

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        print("\n" + "✅ " * 20)
        print(f"훈련 완료! 총 소요 시간: {total_time_str}")
        print("✅ " * 20 + "\n")


# ============================================================================
# 설정 - V7.1 Simple 100K!
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATASET_PATH = BASE_DIR / "dataset/master_dataset_100K.csv"  # ✅ 100K!
PHASE1_MODEL_PATH = SCRIPT_DIR / "models/smollm2_unsupervised_numeric_100k/final_model"  # ✅ V7 100K!
OUTPUT_DIR = SCRIPT_DIR / "models/smollm2_supervised_lora_v7_numeric_simple_100k"  # ✅ V7.1 100K!
DATA_OUTPUT_DIR = SCRIPT_DIR / "processed_data"
RESULTS_DIR = SCRIPT_DIR / "evaluation_results"

# 모델 설정
MAX_SEQ_LENGTH = 256

# 데이터 분할 (80/10/10 - Stratified)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# LoRA 설정 (V6와 동일)
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# 훈련 설정 - 100K 최적화!
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
EPOCHS = 15  # ✅ 30→15 (100K면 충분)
LEARNING_RATE = 5e-5
WARMUP_STEPS = 200  # ✅ 100→200 (더 긴 워밍업)
WEIGHT_DECAY = 0.01
EVAL_STEPS = 500  # ✅ 250→500 (데이터 10배)
SAVE_STEPS = 2000  # ✅ 1000→2000

# 추론 설정 - V7.1 최적화!
INFERENCE_TEMPERATURE = 0.05  # 더 결정적!
INFERENCE_TOP_P = 0.9
INFERENCE_MAX_TOKENS = 50  # 숫자만이라 짧음

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ============================================================================
# 시나리오 분류
# ============================================================================

def classify_scenario(row):
    """시나리오 자동 분류"""
    b1_active = row['b1_time'] > 0
    b2_active = row['b2_time'] > 0

    if b1_active and not b2_active:
        return 'single_b1'
    elif b2_active and not b1_active:
        return 'single_b2'
    elif b1_active and b2_active:
        if abs(row['b1_time'] - row['b2_time']) < 0.01:
            return 'simultaneous'
        else:
            return 'sequential'
    else:
        return 'none'


# ============================================================================
# 데이터 준비 - V7.1 Simple! (화살표 제거)
# ============================================================================

def create_supervised_format_simple(row):
    """
    V7.1 Simple: 화살표 제거! 완전 통합!
    "[1.0, 1.5, 180, 0.0, 0.0, 100, 0.0, 0.0]"
    순서: [initial_power, final_power, b1_pos, b1_time, b1_speed, b2_pos, b2_time, b2_speed]
    """
    text = (
        f"[{row['initial_power']}, {row['final_power']}, "
        f"{row['b1_pos']}, {row['b1_time']}, {row['b1_speed']}, "
        f"{row['b2_pos']}, {row['b2_time']}, {row['b2_speed']}]"
    )
    return text


def load_and_prepare_data():
    """데이터 로드 및 V7.1 Simple Format 변환"""
    print("=" * 80)
    print("데이터 로딩 (V7.1 Simple Format - Phase 2 100K)")
    print("=" * 80)

    df = pd.read_csv(DATASET_PATH)
    print(f"\n전체 데이터: {len(df):,}개")

    df['scenario'] = df.apply(classify_scenario, axis=1)

    print("\n[시나리오 분포]")
    print("=" * 80)
    scenario_counts = df['scenario'].value_counts()
    print(scenario_counts)

    single_count = (df['scenario'].str.contains('single')).sum()
    simul_count = (df['scenario'] == 'simultaneous').sum()
    seq_count = (df['scenario'] == 'sequential').sum()

    print(f"\n✅ 단일 조작: {single_count:,}개 ({single_count / len(df) * 100:.1f}%)")
    print(f"   - Bank1만: {(df['scenario'] == 'single_b1').sum():,}개")
    print(f"   - Bank2만: {(df['scenario'] == 'single_b2').sum():,}개")
    print(f"✅ 동시 조작: {simul_count:,}개 ({simul_count / len(df) * 100:.1f}%)")
    print(f"✅ 순차 조작: {seq_count:,}개 ({seq_count / len(df) * 100:.1f}%)")

    # V7.1 Simple 형식으로 변환!
    df['text'] = df.apply(create_supervised_format_simple, axis=1)

    print("\n[V7.1 Simple 샘플 데이터 예시]")
    print("=" * 80)
    print(df.iloc[0]['text'])
    print("=" * 80)
    print("🎯 V7.1 혁신: 화살표 제거! 완전히 쉼표로만!")
    print("✅ Phase 1과 형식 동일 (숫자만 늘어남)")
    print("✅ 특수 기호 없음 (-> 제거)")
    print("✅ 파싱 초간단 (숫자 8개)")
    print("✅ 100,000개 데이터로 90-95% 목표!")

    return df


def stratified_split(df):
    """시나리오 비율 유지하며 분할 (80/10/10)"""
    print("\n" + "=" * 80)
    print("Stratified Split (80/10/10 - 시나리오 비율 유지)")
    print("=" * 80)

    train, temp = train_test_split(
        df,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df['scenario'],
        random_state=RANDOM_SEED
    )

    val, test = train_test_split(
        temp,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=temp['scenario'],
        random_state=RANDOM_SEED
    )

    print(f"\n[Train Set: {len(train):,}개]")
    print(train['scenario'].value_counts(normalize=True) * 100)

    print(f"\n[Val Set: {len(val):,}개]")
    print(val['scenario'].value_counts(normalize=True) * 100)

    print(f"\n[Test Set: {len(test):,}개]")
    print(test['scenario'].value_counts(normalize=True) * 100)

    train_clean = train.reset_index(drop=True)
    val_clean = val.reset_index(drop=True)
    test_clean = test.reset_index(drop=True)

    print("\n✅ 모든 세트에서 시나리오 비율 유지됨!")

    return train_clean, val_clean, test_clean


# ============================================================================
# 모델 로드 + LoRA 적용
# ============================================================================

def load_model_with_lora(tokenizer):
    """Phase 1 V7 100K 모델 로드 + LoRA adapter 추가"""
    print("\n" + "=" * 80)
    print("Phase 1 V7 100K 모델 로딩 + LoRA Adapter 추가")
    print("=" * 80)

    if not PHASE1_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Phase 1 V7 100K 모델을 찾을 수 없습니다: {PHASE1_MODEL_PATH}\n"
            "먼저 Phase 1 V7 100K 훈련을 완료하세요!"
        )

    # Phase 1 V7 100K 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        str(PHASE1_MODEL_PATH),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    print(f"\n✓ Phase 1 V7 100K 모델 로드: {PHASE1_MODEL_PATH}")

    # LoRA 설정
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )

    # LoRA adapter 추가
    model = get_peft_model(model, lora_config)

    # 학습 가능한 파라미터 확인
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / total_params

    print(f"\n✓ LoRA Adapter 추가 완료!")
    print(f"  - Rank (r): {LORA_R}")
    print(f"  - Alpha: {LORA_ALPHA}")
    print(f"  - Dropout: {LORA_DROPOUT}")
    print(f"  - Target Modules: {LORA_TARGET_MODULES}")
    print(f"\n📊 파라미터 통계:")
    print(f"  - 학습 가능: {trainable_params:,} ({trainable_percent:.2f}%)")
    print(f"  - 전체: {total_params:,}")
    print(f"  - Phase 1 V7 100K 파라미터: 100% 동결 ✅")

    if torch.cuda.is_available():
        model = model.cuda()

    model.print_trainable_parameters()

    return model


def load_model_and_tokenizer():
    """토크나이저 + LoRA 모델 로드"""
    tokenizer = AutoTokenizer.from_pretrained(str(PHASE1_MODEL_PATH))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model_with_lora(tokenizer)

    return model, tokenizer


# ============================================================================
# 데이터셋 토큰화 - V7.1 Simple (화살표 없음!)
# ============================================================================

def tokenize_supervised_simple(examples, tokenizer):
    """
    V7.1 Simple: 앞 2개 숫자 (input) 마스킹, 뒤 6개 숫자 (output)만 학습
    "[1.0, 1.5, 180, 0.0, 0.0, 100, 0.0, 0.0]"
    """
    texts = examples['text']

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding='max_length',
        return_tensors=None,
    )

    labels = []

    for i in range(len(texts)):
        input_ids = tokenized['input_ids'][i]

        # 텍스트 재구성하여 쉼표 위치 찾기
        text = texts[i]

        # "[1.0, 1.5," 부분을 토큰화해서 길이 확인
        # 앞 2개 숫자 + 2개 쉼표 + 대괄호 = 마스킹할 부분
        prompt_part = text.split(',')[0:2]  # "[1.0", " 1.5"
        prompt_text = ','.join(prompt_part) + ','  # "[1.0, 1.5,"

        prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)['input_ids']
        mask_length = len(prompt_tokens)

        label = input_ids.copy()

        # 앞부분 마스킹
        for j in range(min(mask_length, len(label))):
            label[j] = -100

        labels.append(label)

    tokenized['labels'] = labels
    return tokenized


def prepare_dataset(df, tokenizer):
    """HuggingFace Dataset으로 변환"""
    dataset = Dataset.from_pandas(df[['text']])

    tokenized_dataset = dataset.map(
        lambda x: tokenize_supervised_simple(x, tokenizer),
        batched=True,
        batch_size=100,
        remove_columns=['text'],
        desc="Tokenizing (V7.1 Simple 100K - LoRA)"
    )

    return tokenized_dataset


# ============================================================================
# 훈련
# ============================================================================

def train_model(model, tokenizer, train_df, val_df):
    """모델 훈련 (LoRA V7.1 Simple 100K)"""
    print("\n" + "=" * 80)
    print("Supervised Training 시작 (Phase 2 - V7.1 Simple 100K)")
    print("=" * 80)

    train_dataset = prepare_dataset(train_df, tokenizer)
    val_dataset = prepare_dataset(val_df, tokenizer)

    print(f"\n훈련 데이터: {len(train_dataset):,}개")
    print(f"검증 데이터: {len(val_dataset):,}개")

    total_steps = (len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) * EPOCHS
    print(f"총 훈련 스텝: {total_steps:,}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,  # ✅ 최고 모델 자동 로드
        bf16=torch.cuda.is_available(),
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        seed=RANDOM_SEED,
        report_to="none",
        disable_tqdm=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    progress_callback = ProgressCallback(total_steps)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[progress_callback],
    )

    print(f"\n[훈련 설정 - Phase 2 V7.1 Simple 100K]")
    print(f"  - 데이터: 100,000개 (10배!) ✅")
    print(f"  - 방식: LoRA Adapter만 학습")
    print(f"  - 기반: Phase 1 V7 100K 모델 (숫자 전용!)")
    print(f"  - 형식: 화살표 제거! 완전 통합!")
    print(f"  - LoRA Rank: {LORA_R}")
    print(f"  - LoRA Alpha: {LORA_ALPHA}")
    print(f"  - Learning rate: {LEARNING_RATE:.2e}")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Gradient accum: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  - Effective batch: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  - Warmup: {WARMUP_STEPS} steps")
    print(f"  - Temperature: {INFERENCE_TEMPERATURE} 🎯")
    print(f"  - 🔒 Phase 1 V7 100K 파라미터: 완전 동결")
    print(f"  - 🎯 목표: 90-95% 검증 성공률!")

    print("\n" + "-" * 80)
    print("V7.1 Simple 100K 훈련 시작...")
    print("-" * 80 + "\n")

    trainer.train()

    print("\n" + "=" * 80)
    print("✓ Phase 2 V7.1 Simple 100K 훈련 완료!")
    print("=" * 80)

    return trainer


# ============================================================================
# 저장
# ============================================================================

def save_model(model, tokenizer):
    """LoRA 모델 저장"""
    print("\n" + "=" * 80)
    print("LoRA 모델 저장 (Phase 2 V7.1 Simple 100K)")
    print("=" * 80)

    final_model_dir = OUTPUT_DIR / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)

    # LoRA adapter만 저장
    model.save_pretrained(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    print(f"\n✓ Phase 2 V7.1 100K LoRA 모델 저장 완료: {final_model_dir}")
    print("  - LoRA adapter만 저장됨 (용량 매우 작음!)")
    print("  - Phase 1 V7 100K 모델 + 이 adapter를 같이 로드해야 함")


# ============================================================================
# 추론 및 파싱 - V7.1 Simple!
# ============================================================================

def parse_prediction_simple(prediction_text):
    """
    V7.1 Simple: 숫자 8개 추출 (초간단!)
    예: "[1.0, 1.5, 180, 0.0, 0.0, 100, 0.0, 0.0]"
    앞 2개는 입력, 뒤 6개는 출력
    """
    # 숫자 추출
    numbers = re.findall(r'[\d.]+', prediction_text)

    # 8개 이상 있어야 함
    if len(numbers) < 8:
        return None

    try:
        # 앞 2개는 입력 (건너뜀), 뒤 6개가 출력
        # 하지만 추론 시 입력 2개는 이미 프롬프트로 줬으므로
        # 생성된 부분에서 6개만 추출

        # 전체에서 마지막 6개 추출 (가장 안전)
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


def generate_prediction(model, tokenizer, initial_power, final_power):
    """Power 값으로 제어봉 파라미터 예측 (V7.1 Simple)"""
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

    return generated


# ============================================================================
# Test Set 전체 평가
# ============================================================================

def evaluate_on_test_set(model, tokenizer, test_df):
    """Test Set 전체 평가 + 통계 분석 (V7.1 100K)"""
    print("\n" + "=" * 80)
    print(f"🎯 Test Set 전체 평가 시작 ({len(test_df):,}개) - V7.1 Simple 100K")
    print("=" * 80)

    model.eval()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    parse_success = 0
    parse_fail = 0

    print("\n예측 진행 중...")
    for idx, row in test_df.iterrows():
        if (idx + 1) % 1000 == 0:
            print(f"  처리 중: {idx + 1:,}/{len(test_df):,}...")

        initial_power = row['initial_power']
        final_power = row['final_power']

        # 예측
        prediction_text = generate_prediction(model, tokenizer, initial_power, final_power)
        parsed = parse_prediction_simple(prediction_text)

        # 정답
        ground_truth = {
            'b1_pos': row['b1_pos'],
            'b1_time': row['b1_time'],
            'b1_speed': row['b1_speed'],
            'b2_pos': row['b2_pos'],
            'b2_time': row['b2_time'],
            'b2_speed': row['b2_speed']
        }

        result = {
            'idx': idx,
            'scenario': row['scenario'],
            'initial_power': initial_power,
            'final_power': final_power,
            'prediction_text': prediction_text,
            'parsed_success': parsed is not None,
            'ground_truth': ground_truth,
            'prediction': parsed if parsed else {}
        }

        if parsed:
            parse_success += 1
            # 오차 계산
            errors = {}
            for key in ground_truth:
                errors[f'{key}_error'] = abs(parsed[key] - ground_truth[key])
            result['errors'] = errors
        else:
            parse_fail += 1
            result['errors'] = {}

        results.append(result)

    print(f"\n✓ 예측 완료: {len(results):,}개")
    print(f"  - 파싱 성공: {parse_success:,}개 ({parse_success / len(results) * 100:.1f}%)")
    print(f"  - 파싱 실패: {parse_fail:,}개 ({parse_fail / len(results) * 100:.1f}%)")

    # 버전별 비교
    print(f"\n📊 버전별 파싱 성공률 비교:")
    print(f"  - V6 (필드명, 10K):    87.1%")
    print(f"  - V7 (화살표, 10K):    90.0%")
    print(f"  - V7.1 (쉼표, 10K):    100.0%")
    print(f"  - V7.1 (쉼표, 100K):   {parse_success / len(results) * 100:.1f}% ⭐")

    # 통계 분석
    print("\n" + "=" * 80)
    print("📊 정량적 평가 결과")
    print("=" * 80)

    if parse_success > 0:
        # 전체 통계
        all_errors = {key: [] for key in ['b1_pos', 'b1_time', 'b1_speed', 'b2_pos', 'b2_time', 'b2_speed']}

        for r in results:
            if r['parsed_success']:
                for key in all_errors:
                    all_errors[key].append(r['errors'][f'{key}_error'])

        print("\n[전체 오차 통계]")
        print("-" * 80)
        for key in ['b1_pos', 'b1_time', 'b1_speed', 'b2_pos', 'b2_time', 'b2_speed']:
            errors = all_errors[key]
            mae = np.mean(errors)
            mse = np.mean([e ** 2 for e in errors])
            rmse = np.sqrt(mse)
            median = np.median(errors)

            print(f"{key:12s}: MAE={mae:6.3f}, RMSE={rmse:6.3f}, Median={median:6.3f}")

        # 시나리오별 통계
        print("\n[시나리오별 오차 분석]")
        print("-" * 80)
        scenarios = test_df['scenario'].unique()

        for scenario in scenarios:
            scenario_results = [r for r in results if r['scenario'] == scenario and r['parsed_success']]

            if len(scenario_results) > 0:
                print(f"\n{scenario} ({len(scenario_results):,}개):")

                scenario_errors = {key: [] for key in
                                   ['b1_pos', 'b1_time', 'b1_speed', 'b2_pos', 'b2_time', 'b2_speed']}

                for r in scenario_results:
                    for key in scenario_errors:
                        scenario_errors[key].append(r['errors'][f'{key}_error'])

                for key in ['b1_pos', 'b1_time', 'b1_speed', 'b2_pos', 'b2_time', 'b2_speed']:
                    errors = scenario_errors[key]
                    mae = np.mean(errors)
                    print(f"  {key:12s}: MAE={mae:6.3f}")

    # 결과 저장
    results_file = RESULTS_DIR / "test_set_results_v7_simple_100k.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ 상세 결과 저장: {results_file}")

    # 샘플 출력
    print("\n" + "=" * 80)
    print("🔍 샘플 예측 (처음 5개)")
    print("=" * 80)

    for i in range(min(5, len(results))):
        r = results[i]
        print(f"\n[샘플 {i + 1}] {r['scenario']}")
        print(f"입력: [{r['initial_power']}, {r['final_power']}]")
        print(f"예측: {r['prediction_text']}")

        if r['parsed_success']:
            print("파싱 결과:")
            for key in ['b1_pos', 'b1_time', 'b1_speed', 'b2_pos', 'b2_time', 'b2_speed']:
                pred = r['prediction'][key]
                truth = r['ground_truth'][key]
                error = r['errors'][f'{key}_error']
                print(f"  {key:12s}: {pred:6.1f} (정답: {truth:6.1f}, 오차: {error:5.2f})")
        else:
            print("⚠️ 파싱 실패")
        print("-" * 80)

    return results


# ============================================================================
# 간단한 테스트 (Quick Test)
# ============================================================================

def quick_test(model, tokenizer, test_df):
    """빠른 테스트 (샘플 몇 개) - V7.1 100K"""
    print("\n" + "=" * 80)
    print("⚡ Quick Test (샘플 예측) - V7.1 Simple 100K")
    print("=" * 80)

    model.eval()

    # 테스트 1: 임의 Power 값
    print("\n[테스트 1: 임의 Power 값으로 예측]")
    print("=" * 80)

    test_cases = [
        (1.0, 1.5),
        (1.0, 0.8),
        (1.0, 2.0),
    ]

    for initial, final in test_cases:
        prediction = generate_prediction(model, tokenizer, initial, final)

        print(f"\n입력: [{initial}, {final}]")
        print(f"예측: {prediction}")

        parsed = parse_prediction_simple(prediction)
        if parsed:
            print("✅ 파싱 성공:")
            for key, val in parsed.items():
                print(f"  {key}: {val}")
        else:
            print("⚠️ 파싱 실패")

    # 테스트 2: 실제 데이터
    print("\n" + "=" * 80)
    print("[테스트 2: 실제 데이터 3개 샘플]")
    print("=" * 80)

    for i in range(3):
        sample = test_df.iloc[i]
        prediction = generate_prediction(model, tokenizer, sample['initial_power'], sample['final_power'])

        answer = f"[{sample['initial_power']}, {sample['final_power']}, {sample['b1_pos']}, {sample['b1_time']}, {sample['b1_speed']}, {sample['b2_pos']}, {sample['b2_time']}, {sample['b2_speed']}]"

        print(f"\n[샘플 {i + 1}] {sample['scenario']}")
        print(f"입력: [{sample['initial_power']}, {sample['final_power']}]")
        print(f"예측: {prediction}")
        print(f"정답: {answer}")

        parsed = parse_prediction_simple(prediction)
        if parsed:
            print("✅ 파싱 성공")
        else:
            print("⚠️ 파싱 실패")
        print("-" * 80)

    print("\n📌 V7.1 Simple 100K 특징:")
    print("  ✓ 100,000개 데이터 학습!")
    print("  ✓ 화살표 완전 제거 (-> 없음)")
    print("  ✓ 쉼표만 사용 (단순!)")
    print("  ✓ Phase 1과 형식 동일 (숫자만 늘어남)")
    print("  ✓ Confusion 원인 완전 제거")
    print("  ✓ 파싱 성공률 100% 유지 목표")
    print("  🎯 검증 성공률 90-95% 목표!")


# ============================================================================
# 메인
# ============================================================================

def main():
    """메인 훈련 파이프라인"""
    print("\n")
    print("=" * 80)
    print("KOMODO Foundation Model - Phase 2 V7.1 Simple 100K")
    print("SmolLM2-360M + LoRA (화살표 제거, 완전 통합)")
    print("100,000개 데이터로 90-95% 목표!")
    print("=" * 80)
    print(f"\nGPU: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - {torch.cuda.get_device_name(0)}")
        print(f"  - VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    df = load_and_prepare_data()
    train_df, val_df, test_df = stratified_split(df)

    model, tokenizer = load_model_and_tokenizer()

    trainer = train_model(model, tokenizer, train_df, val_df)

    save_model(model, tokenizer)

    # Quick Test
    quick_test(model, tokenizer, test_df)

    # Full Test Set Evaluation
    print("\n" + "=" * 80)
    print("🎯 Test Set 전체 평가를 시작하시겠습니까?")
    print(f"   ({len(test_df):,}개 예측, 약 30-60분 소요)")
    print("=" * 80)

    user_input = input("전체 평가 실행? (y/n): ").strip().lower()

    if user_input == 'y':
        evaluate_on_test_set(model, tokenizer, test_df)
    else:
        print("\n⏩ 전체 평가 건너뜀")

    print("\n" + "=" * 80)
    print("✓ Phase 2 V7.1 Simple 100K 완료!")
    print("=" * 80)
    print(f"\n모델 저장: {OUTPUT_DIR / 'final_model'}")
    print("\n🎯 V7.1 Simple 100K 혁신:")
    print("  ✅ 100,000개 데이터로 학습! (10배!)")
    print("  ✅ 화살표 완전 제거 (-> 없음!)")
    print("  ✅ 쉼표만 사용 (초간단!)")
    print("  ✅ Phase 1과 형식 동일 (숫자만 늘어남)")
    print("  ✅ 특수 기호 없음")
    print("  ✅ 파싱 초간단 (숫자 8개)")
    print("  ✅ Confusion 완전 제거")
    print("  🎯 목표 달성:")
    print("     - 파싱 성공률: 100% 유지")
    print("     - 검증 성공률: 90-95% 달성 예상!")
    print("\n📊 버전별 검증 성공률 비교 (예상):")
    print("  - V7 (화살표, 10K):    22.2%")
    print("  - V7.1 (쉼표, 10K):    79.0%")
    print("  - V7.1 (쉼표, 100K):   90-95% 목표! 🎯")
    print("\n다음: validation_with_simulator_v7_simple.py로 100-200 케이스 검증")


if __name__ == "__main__":
    main()