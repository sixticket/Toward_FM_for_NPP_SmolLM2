"""
SmolLM2-360M Supervised Training for KOMODO (Phase 2) - V7.1 Simple 100K
- [Updated] Base model loaded directly from Hugging Face Hub (SmolLM2-360M)
- V7.1: 화살표 제거! 완전히 쉼표로만 통합
- 데이터 형식: "[1.0, 1.5, 180, 0.0, 0.0, 100, 0.0, 0.0]"
- 100,000개 데이터로 학습!
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
# 설정 - V7.1 Simple 100K! [PATH MODIFIED]
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent  # training/
REPO_ROOT = SCRIPT_DIR.parent                  # repo root

# DATASET_PATH expected at <repo>/dataset/ (run data_generation/komodo_generate.py first)
DATASET_PATH = REPO_ROOT / "dataset/master_dataset_100K.csv"

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"

# Direct LoRA baseline outputs models alongside other training models
OUTPUT_DIR = SCRIPT_DIR / "models/smollm2_supervised_lora_v7_numeric_simple_100k_hf_base"

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

# 훈련 설정 - 100K 최적화! (unchanged hyperparams)
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
EPOCHS = 15
LEARNING_RATE = 5e-5
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.01
EVAL_STEPS = 500
SAVE_STEPS = 2000

# 추론 설정
INFERENCE_TEMPERATURE = 0.05
INFERENCE_TOP_P = 0.9
INFERENCE_MAX_TOKENS = 50

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ============================================================================
# 시나리오 분류 (unchanged)
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
# 데이터 준비 - V7.1 Simple! (unchanged)
# ============================================================================

def create_supervised_format_simple(row):
    """
    V7.1 Simple: 화살표 제거! 완전 통합!
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

    if not DATASET_PATH.exists():
         raise FileNotFoundError(f"Dataset not found at expected path: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    print(f"\n전체 데이터: {len(df):,}개")

    df['scenario'] = df.apply(classify_scenario, axis=1)

    print("\n[시나리오 분포]")
    print("=" * 80)
    scenario_counts = df['scenario'].value_counts()
    print(scenario_counts)

    # V7.1 Simple 형식으로 변환!
    df['text'] = df.apply(create_supervised_format_simple, axis=1)

    print("\n[V7.1 Simple 샘플 데이터 예시]")
    print("=" * 80)
    print(df.iloc[0]['text'])
    print("=" * 80)
    print("✅ Phase 2 목표: SmolLM2-360M에 핵제어 로직 주입")
    print("✅ 특수 기호 없음 (-> 제거)")

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
    print(f"[Val Set: {len(val):,}개]")
    print(f"[Test Set: {len(test):,}개]")

    train_clean = train.reset_index(drop=True)
    val_clean = val.reset_index(drop=True)
    test_clean = test.reset_index(drop=True)

    print("✅ 모든 세트에서 시나리오 비율 유지됨!")

    return train_clean, val_clean, test_clean


# ============================================================================
# 모델 로드 + LoRA 적용 [Modified]
# ============================================================================

def load_model_with_lora(tokenizer):
    """Hugging Face SmolLM2-360M 모델 로드 + LoRA adapter 추가"""
    print("\n" + "=" * 80)
    print(f"Phase 2 모델 로딩: {MODEL_NAME} + LoRA Adapter 추가")
    print("=" * 80)

    # Load base model directly from Hugging Face Hub
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    print(f"\n✓ Base Model 로드 완료: {MODEL_NAME}")
    print("  (이 일반 LLM 지식을 Numeric Control Grammar의 초기 지식으로 사용)")

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

    print(f"\n📊 파라미터 통계:")
    print(f"  - 학습 가능: {trainable_params:,} ({trainable_percent:.2f}%)")
    print(f"  - 전체: {total_params:,}")
    print(f"  - 일반 LLM 파라미터: 동결 ✅")

    if torch.cuda.is_available():
        model = model.cuda()

    model.print_trainable_parameters()

    return model


def load_model_and_tokenizer():
    """토크나이저 + LoRA 모델 로드"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model_with_lora(tokenizer)

    return model, tokenizer


# ============================================================================
# 데이터셋 토큰화 - V7.1 Simple (unchanged)
# ============================================================================

def tokenize_supervised_simple(examples, tokenizer):
    """
    V7.1 Simple: 앞 2개 숫자 (input) 마스킹, 뒤 6개 숫자 (output)만 학습
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

        # "[1.0, 1.5," 부분을 토큰화해서 길이 확인
        prompt_part = texts[i].split(',')[0:2]
        prompt_text = ','.join(prompt_part) + ','

        prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)['input_ids']
        mask_length = len(prompt_tokens)

        label = input_ids.copy()

        # 앞부분 마스킹 (-100으로 설정하여 loss 계산에서 제외)
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
# 훈련 (unchanged Trainer logic)
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
        load_best_model_at_end=True,
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
    print(f"  - 기반: {MODEL_NAME} (Hugging Face!)")
    print(f"  - Learning rate: {LEARNING_RATE:.2e}")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Effective batch: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  - Temperature: {INFERENCE_TEMPERATURE} 🎯")
    print(f"  - 🔒 일반 LLM 파라미터: 완전 동결")
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
# 저장 (unchanged)
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

    # Save tokenizer based on the base model name
    tokenizer.save_pretrained(str(final_model_dir))

    print(f"\n✓ Phase 2 V7.1 100K LoRA 모델 저장 완료: {final_model_dir}")
    print("  - LoRA adapter만 저장됨 (용량 매우 작음!)")


# ============================================================================
# 추론 및 파싱 (unchanged)
# ============================================================================

def parse_prediction_simple(prediction_text):
    """
    V7.1 Simple: 숫자 8개 추출 (초간단!)
    """
    numbers = re.findall(r'[\d.]+', prediction_text)

    if len(numbers) < 8:
        return None

    try:
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
# Test Set 전체 평가 (Evaluation logic requires dedicated environment and is omitted here)
# ============================================================================

def evaluate_on_test_set(*args):
    print("\n[Evaluation logic requires dedicated environment and is omitted here for training script clarity.]")
    pass

def quick_test(*args):
    print("\n[Quick test logic omitted for training script clarity.]")
    pass


# ============================================================================
# 메인
# ============================================================================

def main():
    """메인 훈련 파이프라인"""
    print("\n")
    print("=" * 80)
    print("KOMODO Foundation Model - Phase 2 V7.1 Simple 100K")
    print(f"SmolLM2-360M ({MODEL_NAME}) + LoRA (HF Base)")
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

    print("\n" + "=" * 80)
    print("✓ Phase 2 V7.1 Simple 100K 훈련 완료!")
    print("=" * 80)
    print(f"\n모델 저장: {OUTPUT_DIR / 'final_model'}")
    print("\n🎯 목표 달성:")
    print("  ✅ 100,000개 데이터 학습 완료!")
    print("  ✅ LoRA를 통해 일반 LLM에 핵제어 지식 주입 성공!")

if __name__ == "__main__":
    main()