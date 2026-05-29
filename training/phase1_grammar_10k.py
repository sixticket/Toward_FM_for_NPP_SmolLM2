"""
SmolLM2-360M Unsupervised Training for KOMODO (Phase 1 - V7 Simple)
+ 숫자만 학습 (필드명 제거)
+ Stratified Split (시나리오 비율 유지)
+ 속도 최적화 (Batch 8)
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 멀티프로세싱 경고 방지

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
from sklearn.model_selection import train_test_split
import time
from datetime import timedelta


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
        print("SmolLM2 Unsupervised Training 시작! (Phase 1 - V7 Simple)")
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
            print(f"🎯 Learning Rate: {learning_rate:.2e}" if isinstance(learning_rate, float) else f"🎯 Learning Rate: {learning_rate}")
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
# 설정
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATASET_PATH = BASE_DIR / "dataset/master_dataset_10K.csv"
OUTPUT_DIR = SCRIPT_DIR / "models/smollm2_unsupervised_numeric_10k"  # V7 전용 디렉토리
DATA_OUTPUT_DIR = SCRIPT_DIR / "processed_data"

# 모델 설정
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"
MAX_SEQ_LENGTH = 256

# 훈련 설정 (90/10/0 추천)
TRAIN_RATIO = 0.9
VAL_RATIO = 0.1
TEST_RATIO = 0.0  # Phase 1에서는 불필요

# V7 최적화: 속도 개선
BATCH_SIZE = 8                   # 2 → 8 (4배 증가)
GRADIENT_ACCUMULATION_STEPS = 2  # 8 → 2 (4배 감소)
EPOCHS = 20
LEARNING_RATE = 5e-5
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01

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
# 데이터 준비 - V7 Simple!
# ============================================================================

def create_simple_format_phase1(row):
    """
    V7 Simple: 숫자만!
    순서: [b1_pos, b1_time, b1_speed, b2_pos, b2_time, b2_speed]
    """
    return (
        f"[{row['b1_pos']}, {row['b1_time']}, {row['b1_speed']}, "
        f"{row['b2_pos']}, {row['b2_time']}, {row['b2_speed']}]"
    )


def load_and_prepare_data():
    """데이터 로드 및 V7 Simple Format 변환"""
    print("=" * 80)
    print("데이터 로딩 (V7 Simple Format - Phase 1)")
    print("=" * 80)

    df = pd.read_csv(DATASET_PATH)
    print(f"\n전체 데이터: {len(df)}개")

    # 시나리오 분류
    df['scenario'] = df.apply(classify_scenario, axis=1)

    # 분포 확인
    print("\n[시나리오 분포]")
    print("=" * 80)
    scenario_counts = df['scenario'].value_counts()
    print(scenario_counts)

    single_count = (df['scenario'].str.contains('single')).sum()
    simul_count = (df['scenario'] == 'simultaneous').sum()
    seq_count = (df['scenario'] == 'sequential').sum()

    print(f"\n✅ 단일 조작: {single_count}개 ({single_count / len(df) * 100:.1f}%)")
    print(f"   - Bank1만: {(df['scenario'] == 'single_b1').sum()}개")
    print(f"   - Bank2만: {(df['scenario'] == 'single_b2').sum()}개")
    print(f"✅ 동시 조작: {simul_count}개 ({simul_count / len(df) * 100:.1f}%)")
    print(f"✅ 순차 조작: {seq_count}개 ({seq_count / len(df) * 100:.1f}%)")

    # V7 Simple format으로 변환
    df['text'] = df.apply(create_simple_format_phase1, axis=1)

    # 샘플 출력
    print("\n[샘플 데이터 예시 - V7 Simple]")
    print("=" * 80)
    print(df.iloc[0]['text'])
    print("=" * 80)
    print("\n✅ 숫자만! (필드명 제거)")
    print("✅ 순서: [b1_pos, b1_time, b1_speed, b2_pos, b2_time, b2_speed]")
    print("✅ Unsupervised CLM: 숫자 패턴 학습")

    return df


def stratified_split(df):
    """시나리오 비율 유지하며 분할"""
    print("\n" + "=" * 80)
    print("Stratified Split (시나리오 비율 유지)")
    print("=" * 80)

    if TEST_RATIO > 0:
        # Train+Val / Test 먼저 분할
        train_val, test = train_test_split(
            df,
            test_size=TEST_RATIO,
            stratify=df['scenario'],
            random_state=RANDOM_SEED
        )

        # Train / Val 분할
        val_size = VAL_RATIO / (1 - TEST_RATIO)
        train, val = train_test_split(
            train_val,
            test_size=val_size,
            stratify=train_val['scenario'],
            random_state=RANDOM_SEED
        )
    else:
        # Test 없이 Train / Val만 분할
        train, val = train_test_split(
            df,
            test_size=VAL_RATIO,
            stratify=df['scenario'],
            random_state=RANDOM_SEED
        )
        test = pd.DataFrame()  # 빈 DataFrame

    # 각 세트의 시나리오 분포 확인
    print(f"\n[Train Set: {len(train)}개]")
    print(train['scenario'].value_counts(normalize=True) * 100)

    print(f"\n[Val Set: {len(val)}개]")
    print(val['scenario'].value_counts(normalize=True) * 100)

    if len(test) > 0:
        print(f"\n[Test Set: {len(test)}개]")
        print(test['scenario'].value_counts(normalize=True) * 100)

    # scenario 컬럼 제거
    train = train.drop('scenario', axis=1).reset_index(drop=True)
    val = val.drop('scenario', axis=1).reset_index(drop=True)
    if len(test) > 0:
        test = test.drop('scenario', axis=1).reset_index(drop=True)

    print("\n✅ 모든 세트에서 시나리오 비율 유지됨!")

    return train, val, test


# ============================================================================
# 모델 로드
# ============================================================================

def load_model_and_tokenizer():
    """SmolLM2-360M 로드"""
    print("\n" + "=" * 80)
    print("SmolLM2-360M 로딩 (V7 Simple)")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # SmolLM2는 기본적으로 pad_token이 있지만, 없으면 eos_token 사용
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    if torch.cuda.is_available():
        model = model.cuda()

    print(f"\n✓ 모델 로드: {MODEL_NAME}")
    print(f"  - Parameters: 360M")
    print(f"  - Training: 4T tokens (FineWeb-Edu, DCLM, The Stack)")
    print(f"  - Release: 2025년 2월 (최신!)")
    print(f"  - V7 Simple: 숫자만 학습!")
    print(f"  - BF16 사용")

    return model, tokenizer


# ============================================================================
# 데이터셋 토큰화
# ============================================================================

def tokenize_unsupervised(examples, tokenizer):
    """Unsupervised: 전체 텍스트를 그대로 학습"""
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding='max_length',
        return_tensors=None,
    )

    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized


def prepare_dataset(df, tokenizer):
    """HuggingFace Dataset으로 변환"""
    dataset = Dataset.from_pandas(df[['text']])

    tokenized_dataset = dataset.map(
        lambda x: tokenize_unsupervised(x, tokenizer),
        batched=True,
        batch_size=100,
        remove_columns=['text'],
        desc="Tokenizing (V7 Simple)"
    )

    return tokenized_dataset


# ============================================================================
# 훈련
# ============================================================================

def train_model(model, tokenizer, train_df, val_df):
    """모델 훈련"""
    print("\n" + "=" * 80)
    print("Unsupervised Training 시작 (Phase 1 - V7 Simple)")
    print("=" * 80)

    train_dataset = prepare_dataset(train_df, tokenizer)
    val_dataset = prepare_dataset(val_df, tokenizer)

    print(f"\n훈련 데이터: {len(train_dataset)}개")
    print(f"검증 데이터: {len(val_dataset)}개")

    total_steps = (len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) * EPOCHS
    print(f"총 훈련 스텝: {total_steps}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,  # 평가는 더 큰 배치
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=250,  # V7: 평가 빈도 감소
        save_strategy="steps",
        save_steps=1000,  # V7: 저장 빈도 감소
        save_total_limit=3,
        bf16=torch.cuda.is_available(),
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        seed=RANDOM_SEED,
        report_to="none",
        disable_tqdm=False,
        dataloader_num_workers=4,      # V7: 멀티프로세싱
        dataloader_pin_memory=True,    # V7: GPU 전송 최적화
    )

    progress_callback = ProgressCallback(total_steps)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[progress_callback],
    )

    print(f"\n[훈련 설정 - Phase 1 V7 Simple]")
    print(f"  - 방식: Unsupervised CLM")
    print(f"  - 모델: SmolLM2-360M (2025년 최신)")
    print(f"  - 형식: 숫자만! [180, 0.0, ...]")
    print(f"  - 시나리오 비율: 유지됨 (Stratified Split)")
    print(f"  - Batch size: {BATCH_SIZE} 🚀 (2→8, 4배 증가)")
    print(f"  - Gradient Accum: {GRADIENT_ACCUMULATION_STEPS} 🚀 (8→2)")
    print(f"  - Effective batch: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - DataLoader workers: 4 (멀티프로세싱)")
    print(f"  - 예상 속도: V6 Phase 1 대비 2-3배 빠름! ⚡")

    print("\n" + "-" * 80)
    print("V7 Simple 훈련 시작...")
    print("-" * 80 + "\n")

    trainer.train()

    print("\n" + "=" * 80)
    print("✓ Phase 1 V7 훈련 완료!")
    print("=" * 80)

    return trainer


# ============================================================================
# 저장 및 테스트
# ============================================================================

def save_model(model, tokenizer):
    """모델 저장"""
    print("\n" + "=" * 80)
    print("모델 저장 (Phase 1 V7)")
    print("=" * 80)

    final_model_dir = OUTPUT_DIR / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    print(f"\n✓ Phase 1 V7 모델 저장 완료: {final_model_dir}")


def test_model(model, tokenizer, test_df):
    """모델 테스트 - V7 Simple"""
    print("\n" + "=" * 80)
    print("Phase 1 V7 모델 테스트 (숫자 패턴 검증)")
    print("=" * 80)

    model.eval()

    # 테스트 1: 빈 시작
    print("\n[테스트 1: 빈 시작에서 생성]")
    print("=" * 80)

    prompt = "["
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # V7: 숫자만이라 짧음
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\n입력: '{prompt}'")
    print(f"\n생성 결과:")
    print("-" * 80)
    print(generated)
    print("-" * 80)

    is_valid = validate_simple_format(generated)
    print(f"\n{'✅' if is_valid else '❌'} 형식 검증: {'통과' if is_valid else '실패'}")

    # 테스트 2: 부분 입력
    print("\n" + "=" * 80)
    print("[테스트 2: 부분 입력에서 이어서 생성]")
    print("=" * 80)

    prompt = "[180, 0.0, 0.0,"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.5,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\n입력: '{prompt}'")
    print(f"\n생성 결과:")
    print("-" * 80)
    print(generated)
    print("-" * 80)

    print("\n📌 Phase 1 V7 목표:")
    print("  ✓ 대괄호 [ ] 구조 이해")
    print("  ✓ 숫자 6개 생성 (쉼표 구분)")
    print("  ✓ 필드명 생성 안함!")
    print("  ✓ V6 대비 토큰 2-3배 절약")


def validate_simple_format(output):
    """V7 Simple 형식 검증"""
    import re

    # 숫자 6개 추출 시도
    numbers = re.findall(r'[\d.]+', output)

    if len(numbers) < 6:
        print(f"  ❌ 숫자 부족: {len(numbers)}개 (6개 필요)")
        return False

    # 대괄호 체크
    if '[' not in output:
        print(f"  ❌ 시작 대괄호 없음")
        return False

    # 필드명이 생성되었는지 체크 (생성되면 안됨!)
    field_keywords = ['pos', 'time', 'speed', 'b1', 'b2', 'initial', 'final', 'power', 'bank']
    for keyword in field_keywords:
        if keyword in output.lower():
            print(f"  ❌ 필드명 발견: '{keyword}' (숫자만 있어야 함!)")
            return False

    print(f"  ✅ 숫자 6개 이상 생성: {len(numbers)}개")
    print(f"  ✅ 필드명 없음 (깔끔!)")

    return True


# ============================================================================
# 메인
# ============================================================================

def main():
    """메인 훈련 파이프라인"""
    print("\n")
    print("=" * 80)
    print("KOMODO Foundation Model - Phase 1 V7 Simple")
    print("SmolLM2-360M + 숫자만 학습 + 속도 최적화")
    print("=" * 80)
    print(f"\nGPU: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - {torch.cuda.get_device_name(0)}")
        print(f"  - VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # 1. 데이터 준비
    df = load_and_prepare_data()
    train_df, val_df, test_df = stratified_split(df)

    # 2. 모델 로드
    model, tokenizer = load_model_and_tokenizer()

    # 3. 훈련
    trainer = train_model(model, tokenizer, train_df, val_df)

    # 4. 저장
    save_model(model, tokenizer)

    # 5. 테스트
    test_model(model, tokenizer, val_df)

    print("\n" + "=" * 80)
    print("✓ Phase 1 V7 완료!")
    print("=" * 80)
    print(f"\n모델 저장: {OUTPUT_DIR / 'final_model'}")
    print("\n📌 V7 개선 사항:")
    print("  ✅ 숫자만 학습 (필드명 제거)")
    print("  ✅ 토큰 2-3배 절약 (15~20 토큰)")
    print("  ✅ Confusion 원인 제거")
    print("  🚀 속도 2-3배 향상 (Batch 8)")
    print("  🎯 Phase 2 V7의 튼튼한 기초 완성!")


if __name__ == "__main__":
    main()