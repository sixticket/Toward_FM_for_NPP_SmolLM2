"""
PyRK Inverse Model Training - V2 Simple Numeric Format
- KOMODO 학습 방식과 동일한 구조
- 데이터 형식: "[P_init, P_target, reactivity, duration]"
- Phase 1: 숫자 형식 학습 (Unsupervised)
- Phase 2: 앞 2개 마스킹, 뒤 2개만 학습 (Supervised LoRA)
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
    TrainerCallback,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
import time
from datetime import timedelta
import gc
import re

# ============================================================================
# 설정
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent  # pyrk_transfer/

# 데이터셋 경로 (data_generation.py 실행 시 같은 폴더에 생성됨)
DATASET_PATH = SCRIPT_DIR / "nuscale_training_dataset_10k.csv"

# 저장 경로
OUTPUT_DIR_PHASE1 = SCRIPT_DIR / "models/pyrk_inverse_phase1_v2"
OUTPUT_DIR_PHASE2 = SCRIPT_DIR / "models/pyrk_inverse_phase2_v2"

# 모델 설정
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"
MAX_SEQ_LENGTH = 128

# 데이터 분할 (80/10/10)
TEST_SPLIT_RATIO = 0.2
VAL_TEST_RATIO = 0.5
RANDOM_SEED = 42

# Phase 1 설정
P1_BATCH = 8
P1_ACCUM = 2
P1_EPOCHS = 3
P1_LR = 5e-5

# Phase 2 설정
P2_BATCH = 8
P2_ACCUM = 2
P2_EPOCHS = 15
P2_LR = 5e-5
P2_WARMUP = 100

# LoRA 설정
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# 추론 설정
INFERENCE_TEMPERATURE = 0.05
INFERENCE_MAX_TOKENS = 50

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================================
# Progress Callback
# ============================================================================
class ProgressCallback(TrainerCallback):
    def __init__(self, total_steps, phase_name="Training"):
        self.total_steps = total_steps
        self.phase_name = phase_name
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print("\n" + "🚀 " * 20)
        print(f"PyRK {self.phase_name} 시작!")
        print("🚀 " * 20 + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            current_step = state.global_step
            if self.total_steps > 0:
                progress = (current_step / self.total_steps) * 100
                filled = int(30 * current_step / self.total_steps)
            else:
                progress = 0
                filled = 0

            elapsed = time.time() - self.start_time
            elapsed_str = str(timedelta(seconds=int(elapsed)))

            if current_step > 0:
                avg_time = elapsed / current_step
                remaining = avg_time * (self.total_steps - current_step)
                remaining_str = str(timedelta(seconds=int(remaining)))
            else:
                remaining_str = "계산 중..."

            loss = logs.get('loss', 'N/A')
            bar = '█' * filled + '░' * (30 - filled)
            print(f"📊 [{self.phase_name}] Step {current_step}/{self.total_steps} [{bar}] {progress:.1f}% | Loss: {loss} | 경과: {elapsed_str} | 남은: {remaining_str}")

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\n🎉 Epoch {int(state.epoch)} 완료!\n")


# ============================================================================
# 데이터 로드 및 분할
# ============================================================================
def load_and_split_data():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"데이터셋 없음: {DATASET_PATH}")

    print(f"Loading Dataset: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)

    print(f"\n[데이터 통계]")
    print(f"  전체: {len(df):,}개")
    print(f"  power_initial: {df['power_initial'].mean():.2f} (고정)")
    print(f"  power_final: {df['power_final'].min():.2f} ~ {df['power_final'].max():.2f}")
    print(f"  reactivity: {df['input_reactivity'].min():.6f} ~ {df['input_reactivity'].max():.6f}")
    print(f"  duration: {df['input_duration'].min():.2f} ~ {df['input_duration'].max():.2f}")

    # 1차 분할: Train(80%) vs Temp(20%)
    train_df, temp_df = train_test_split(df, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_SEED)

    # 2차 분할: Temp(20%) -> Val(10%) vs Test(10%)
    val_df, test_df = train_test_split(temp_df, test_size=VAL_TEST_RATIO, random_state=RANDOM_SEED)

    print(f"\n[Data Split 80/10/10]")
    print(f"  - Train : {len(train_df):,} (80%)")
    print(f"  - Val   : {len(val_df):,} (10%)")
    print(f"  - Test  : {len(test_df):,} (10%)")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ============================================================================
# 데이터 포맷 - V2 Simple (KOMODO 스타일)
# ============================================================================
def create_phase1_format(row):
    """
    Phase 1: Output 형식만 학습
    "[reactivity, duration]"
    """
    return f"[{row['input_reactivity']:.6f}, {row['input_duration']:.4f}]"


def create_phase2_format(row):
    """
    Phase 2: 전체 형식 (앞 2개 마스킹, 뒤 2개만 학습)
    "[P_init, P_target, reactivity, duration]"
    """
    return f"[{row['power_initial']:.2f}, {row['power_final']:.2f}, {row['input_reactivity']:.6f}, {row['input_duration']:.4f}]"


# ============================================================================
# Phase 2 토큰화 (앞 2개 마스킹)
# ============================================================================
def tokenize_with_masking(examples, tokenizer):
    """
    앞 2개 숫자 (P_init, P_target) 마스킹, 뒤 2개 (reactivity, duration)만 학습
    "[245.44, 180.13, -0.002089, 22.388]"
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
        text = texts[i]

        # "[P_init, P_target," 부분까지 마스킹
        # 예: "[245.44, 180.13,"
        parts = text.split(',')
        if len(parts) >= 2:
            prompt_text = ','.join(parts[:2]) + ','  # "[245.44, 180.13,"
            prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)['input_ids']
            mask_length = len(prompt_tokens)
        else:
            mask_length = 0

        label = input_ids.copy()

        # 앞부분 마스킹
        for j in range(min(mask_length, len(label))):
            label[j] = -100

        labels.append(label)

    tokenized['labels'] = labels
    return tokenized


# ============================================================================
# PHASE 1: Grammar Learning (Unsupervised)
# ============================================================================
def run_phase1(train_df, val_df):
    print("\n\n" + "#" * 60)
    print("🚀 PHASE 1: Grammar Learning (Unsupervised)")
    print("   형식: [reactivity, duration] - Output만 학습")
    print("#" * 60)

    # 데이터 포맷팅 - Phase 1 형식 (output만)
    train_ds_df = train_df.copy()
    val_ds_df = val_df.copy()
    train_ds_df['text'] = train_ds_df.apply(create_phase1_format, axis=1)
    val_ds_df['text'] = val_ds_df.apply(create_phase1_format, axis=1)

    print(f"\n[샘플 데이터]")
    print(train_ds_df['text'].iloc[0])

    # 토크나이저 & 모델
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    if torch.cuda.is_available():
        model = model.cuda()

    # 토크나이징
    def tokenize(examples):
        return tokenizer(examples['text'], truncation=True, max_length=MAX_SEQ_LENGTH, padding='max_length')

    train_ds = Dataset.from_pandas(train_ds_df[['text']]).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(val_ds_df[['text']]).map(tokenize, batched=True)

    # Trainer
    total_steps = (len(train_ds) // (P1_BATCH * P1_ACCUM)) * P1_EPOCHS

    OUTPUT_DIR_PHASE1.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(OUTPUT_DIR_PHASE1 / "checkpoints"),
            num_train_epochs=P1_EPOCHS,
            per_device_train_batch_size=P1_BATCH,
            gradient_accumulation_steps=P1_ACCUM,
            learning_rate=P1_LR,
            weight_decay=0.01,
            logging_steps=50,
            save_strategy="epoch",
            bf16=torch.cuda.is_available(),
            report_to="none",
            disable_tqdm=True
        ),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[ProgressCallback(total_steps, "Phase 1")]
    )

    trainer.train()

    # 저장
    print(f"\nSaving Phase 1 Model to {OUTPUT_DIR_PHASE1}...")
    final_dir = OUTPUT_DIR_PHASE1 / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print("✅ Phase 1 완료!")

    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()


# ============================================================================
# PHASE 2: Mapping Learning (Supervised LoRA with Masking)
# ============================================================================
def run_phase2(train_df, val_df, test_df):
    print("\n\n" + "#" * 60)
    print("🚀 PHASE 2: Mapping Learning (Supervised LoRA)")
    print("   형식: [P_init, P_target, rho, dur]")
    print("   앞 2개 마스킹, 뒤 2개만 학습")
    print("#" * 60)

    # 데이터 포맷팅 - Phase 2 형식 (전체, 마스킹 적용)
    train_ds_df = train_df.copy()
    val_ds_df = val_df.copy()
    train_ds_df['text'] = train_ds_df.apply(create_phase2_format, axis=1)
    val_ds_df['text'] = val_ds_df.apply(create_phase2_format, axis=1)

    print(f"\n[샘플 데이터]")
    print(train_ds_df['text'].iloc[0])
    print("   ↑ 앞 2개 (P_init, P_target) 마스킹")
    print("   ↑ 뒤 2개 (reactivity, duration)만 학습")

    # Phase 1 모델 로드
    phase1_path = OUTPUT_DIR_PHASE1 / "final_model"
    if not phase1_path.exists():
        raise FileNotFoundError(f"Phase 1 모델이 없습니다: {phase1_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(phase1_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(phase1_path),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )

    # LoRA 적용
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    if torch.cuda.is_available():
        model = model.cuda()

    model.print_trainable_parameters()

    # 데이터셋 (마스킹 적용)
    train_ds = Dataset.from_pandas(train_ds_df[['text']])
    val_ds = Dataset.from_pandas(val_ds_df[['text']])

    train_ds = train_ds.map(
        lambda x: tokenize_with_masking(x, tokenizer),
        batched=True,
        batch_size=100,
        remove_columns=['text']
    )
    val_ds = val_ds.map(
        lambda x: tokenize_with_masking(x, tokenizer),
        batched=True,
        batch_size=100,
        remove_columns=['text']
    )

    # Trainer
    total_steps = (len(train_ds) // (P2_BATCH * P2_ACCUM)) * P2_EPOCHS

    OUTPUT_DIR_PHASE2.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(OUTPUT_DIR_PHASE2 / "checkpoints"),
            num_train_epochs=P2_EPOCHS,
            per_device_train_batch_size=P2_BATCH,
            gradient_accumulation_steps=P2_ACCUM,
            learning_rate=P2_LR,
            warmup_steps=P2_WARMUP,
            weight_decay=0.01,
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=400,  # eval_steps의 배수
            save_total_limit=3,
            load_best_model_at_end=True,
            bf16=torch.cuda.is_available(),
            lr_scheduler_type="cosine",
            report_to="none",
            disable_tqdm=True
        ),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[ProgressCallback(total_steps, "Phase 2")]
    )

    print(f"\n[훈련 설정]")
    print(f"  - LoRA Rank: {LORA_R}")
    print(f"  - LoRA Alpha: {LORA_ALPHA}")
    print(f"  - Learning rate: {P2_LR:.2e}")
    print(f"  - Epochs: {P2_EPOCHS}")
    print(f"  - Effective batch: {P2_BATCH * P2_ACCUM}")

    trainer.train()

    # 저장
    print(f"\nSaving Phase 2 Model to {OUTPUT_DIR_PHASE2}...")
    final_dir = OUTPUT_DIR_PHASE2 / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # ------------------------------------------------------------------
    # 테스트
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("🧪 FINAL EVALUATION on Test Set")
    print("=" * 60)

    model.eval()
    test_samples = test_df.sample(10, random_state=RANDOM_SEED)

    success = 0
    for _, row in test_samples.iterrows():
        # 추론: 앞 2개만 주고 뒤 2개 생성
        prompt = f"[{row['power_initial']:.2f}, {row['power_final']:.2f},"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=INFERENCE_MAX_TOKENS,
                temperature=INFERENCE_TEMPERATURE,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 파싱
        numbers = re.findall(r'[-\d.]+', gen_text)

        print(f"\n[Test Case]")
        print(f"입력: [{row['power_initial']:.2f}, {row['power_final']:.2f}]")
        print(f"예측: {gen_text}")
        print(f"정답: [{row['input_reactivity']:.6f}, {row['input_duration']:.4f}]")

        if len(numbers) >= 4:
            pred_rho = float(numbers[2])
            pred_dur = float(numbers[3])
            print(f"파싱: rho={pred_rho:.6f}, dur={pred_dur:.4f}")

            # 오차 계산
            rho_error = abs(pred_rho - row['input_reactivity'])
            dur_error = abs(pred_dur - row['input_duration'])
            print(f"오차: rho={rho_error:.6f}, dur={dur_error:.4f}")

            if rho_error < 0.005 and dur_error < 5.0:
                print("✅ 합리적 예측")
                success += 1
            else:
                print("⚠️ 오차 큼")
        else:
            print("❌ 파싱 실패")

    print(f"\n[요약] 합리적 예측: {success}/10")

    print("\n✅ Phase 2 완료!")

    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("🌙 PyRK Inverse Training - V2 Simple Numeric")
    print("   Phase 1: [rho, dur] - Output 형식 학습")
    print("   Phase 2: [P_init, P_target, rho, dur] - 매핑 학습")
    print("=" * 60)

    print(f"\nGPU: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - {torch.cuda.get_device_name(0)}")

    # 데이터 로드
    train_df, val_df, test_df = load_and_split_data()

    # Phase 1
    run_phase1(train_df, val_df)

    print("\n⏳ 5초 대기...\n")
    time.sleep(5)

    # Phase 2
    run_phase2(train_df, val_df, test_df)

    print("\n" + "=" * 60)
    print("🎉 모든 학습 완료!")
    print("=" * 60)
    print(f"\n모델 저장 위치:")
    print(f"  Phase 1: {OUTPUT_DIR_PHASE1 / 'final_model'}")
    print(f"  Phase 2: {OUTPUT_DIR_PHASE2 / 'final_model'}")
    print(f"\n검증 실행:")
    print(f"  python validation_pyrk.py")


if __name__ == "__main__":
    main()