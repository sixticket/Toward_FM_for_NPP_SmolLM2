# smollm2_phase1_unsupervised_numeric_last.py
"""
SmolLM2-360M Unsupervised Training for KOMODO (Phase 1 - V7 Simple)
+ Optimized for 100K dataset
+ Numbers only (remove field names)
+ Stratified split (preserve scenario ratios)
+ Memory optimizations + stronger checkpointing
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # suppress multiprocessing warnings

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
    """Display training progress"""

    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print("\n" + "🚀 " * 20)
        print("SmolLM2 Unsupervised Training START! (Phase 1 - V7 Simple - 100K)")
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
                remaining_str = "calculating..."

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
            print(f"⏱️  Elapsed: {elapsed_str}")
            print(f"⏳ Remaining (est.): {remaining_str}")

            if 'epoch' in logs:
                print(f"📚 Epoch: {logs['epoch']:.2f}")

            print(f"{'=' * 80}\n")

    def on_epoch_end(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch)
        print("\n" + "🎉 " * 20)
        print(f"Epoch {current_epoch} complete!")
        print("🎉 " * 20 + "\n")

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        print("\n" + "✅ " * 20)
        print(f"Training finished! Total time: {total_time_str}")
        print("✅ " * 20 + "\n")


# ============================================================================
# Configuration - optimized for 100K
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATASET_PATH = BASE_DIR / "dataset/master_dataset.csv"  # ✅ 100K dataset!
OUTPUT_DIR = SCRIPT_DIR / "models/smollm2_unsupervised_numeric_100k"  # ✅ for 100K
DATA_OUTPUT_DIR = SCRIPT_DIR / "processed_data"

# Model config
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"
MAX_SEQ_LENGTH = 256

# Train/val/test ratios (90/10/0 recommended)
TRAIN_RATIO = 0.9
VAL_RATIO = 0.1
TEST_RATIO = 0.0  # Not needed in Phase 1

# 100K optimization
BATCH_SIZE = 8                    # keep
GRADIENT_ACCUMULATION_STEPS = 2   # keep
EPOCHS = 5                        # ✅ 20 → 5 (sufficient for 100K)
LEARNING_RATE = 5e-5
WARMUP_STEPS = 500                # ✅ 100 → 500 (smoother start)
WEIGHT_DECAY = 0.01

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ============================================================================
# Scenario classification
# ============================================================================

def classify_scenario(row):
    """Auto-classify scenario"""
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
# Data prep - V7 Simple
# ============================================================================

def create_simple_format_phase1(row):
    """
    V7 Simple: numbers only!
    Order: [b1_pos, b1_time, b1_speed, b2_pos, b2_time, b2_speed]
    """
    return (
        f"[{row['b1_pos']}, {row['b1_time']}, {row['b1_speed']}, "
        f"{row['b2_pos']}, {row['b2_time']}, {row['b2_speed']}]"
    )


def load_and_prepare_data():
    """Load data and convert to V7 Simple format"""
    print("=" * 80)
    print("Loading data (V7 Simple Format - Phase 1 - 100K)")
    print("=" * 80)

    df = pd.read_csv(DATASET_PATH)
    print(f"\nTotal rows: {len(df):,} 🎉")  # ✅ thousands separator

    # Scenario classification
    df['scenario'] = df.apply(classify_scenario, axis=1)

    # Distribution
    print("\n[Scenario distribution]")
    print("=" * 80)
    scenario_counts = df['scenario'].value_counts()
    print(scenario_counts)

    single_count = (df['scenario'].str.contains('single')).sum()
    simul_count = (df['scenario'] == 'simultaneous').sum()
    seq_count = (df['scenario'] == 'sequential').sum()

    print(f"\n✅ Single ops: {single_count:,} ({single_count / len(df) * 100:.1f}%)")
    print(f"   - Bank1 only: {(df['scenario'] == 'single_b1').sum():,}")
    print(f"   - Bank2 only: {(df['scenario'] == 'single_b2').sum():,}")
    print(f"✅ Simultaneous: {simul_count:,} ({simul_count / len(df) * 100:.1f}%)")
    print(f"✅ Sequential: {seq_count:,} ({seq_count / len(df) * 100:.1f}%)")

    # Convert to V7 Simple
    df['text'] = df.apply(create_simple_format_phase1, axis=1)

    # Sample print
    print("\n[Sample - V7 Simple]")
    print("=" * 80)
    print(df.iloc[0]['text'])
    print("=" * 80)
    print("\n✅ Numbers only! (no field names)")
    print("✅ Order: [b1_pos, b1_time, b1_speed, b2_pos, b2_time, b2_speed]")
    print("✅ Unsupervised CLM: learn numeric patterns")

    return df


def stratified_split(df):
    """Split while preserving scenario ratios"""
    print("\n" + "=" * 80)
    print("Stratified Split (preserve scenario ratios)")
    print("=" * 80)

    if TEST_RATIO > 0:
        # Split Train+Val / Test first
        train_val, test = train_test_split(
            df,
            test_size=TEST_RATIO,
            stratify=df['scenario'],
            random_state=RANDOM_SEED
        )

        # Then split Train / Val
        val_size = VAL_RATIO / (1 - TEST_RATIO)
        train, val = train_test_split(
            train_val,
            test_size=val_size,
            stratify=train_val['scenario'],
            random_state=RANDOM_SEED
        )
    else:
        # No test set: only Train / Val
        train, val = train_test_split(
            df,
            test_size=VAL_RATIO,
            stratify=df['scenario'],
            random_state=RANDOM_SEED
        )
        test = pd.DataFrame()  # empty DataFrame

    # Show distributions
    print(f"\n[Train set: {len(train):,}]")
    print(train['scenario'].value_counts(normalize=True) * 100)

    print(f"\n[Val set: {len(val):,}]")
    print(val['scenario'].value_counts(normalize=True) * 100)

    if len(test) > 0:
        print(f"\n[Test set: {len(test):,}]")
        print(test['scenario'].value_counts(normalize=True) * 100)

    # Drop scenario column
    train = train.drop('scenario', axis=1).reset_index(drop=True)
    val = val.drop('scenario', axis=1).reset_index(drop=True)
    if len(test) > 0:
        test = test.drop('scenario', axis=1).reset_index(drop=True)

    print("\n✅ Scenario ratios preserved across all splits!")

    return train, val, test


# ============================================================================
# Load model
# ============================================================================

def load_model_and_tokenizer():
    """Load SmolLM2-360M"""
    print("\n" + "=" * 80)
    print("Loading SmolLM2-360M (V7 Simple - 100K)")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # SmolLM2 usually has a pad_token; if not, use eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    if torch.cuda.is_available():
        model = model.cuda()

    print(f"\n✓ Model loaded: {MODEL_NAME}")
    print(f"  - Parameters: 360M")
    print(f"  - Training: 4T tokens (FineWeb-Edu, DCLM, The Stack)")
    print(f"  - Release: Feb 2025 (latest)")
    print(f"  - V7 Simple: numbers only!")
    print(f"  - BF16 enabled")
    print(f"  - Data: 100K (10× the 10K) 🚀")

    return model, tokenizer


# ============================================================================
# Dataset tokenization - optimized for 100K
# ============================================================================

def tokenize_unsupervised(examples, tokenizer):
    """Unsupervised: learn from full text as-is"""
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
    """Convert to HuggingFace Dataset (memory-efficient)"""
    dataset = Dataset.from_pandas(df[['text']])

    # ✅ Batch tuned for 100K (faster with acceptable memory)
    tokenized_dataset = dataset.map(
        lambda x: tokenize_unsupervised(x, tokenizer),
        batched=True,
        batch_size=500,  # ✅ 100 → 500 (faster)
        remove_columns=['text'],
        desc="Tokenizing (V7 Simple - 100K)"
    )

    return tokenized_dataset


# ============================================================================
# Training - optimized for 100K
# ============================================================================

def train_model(model, tokenizer, train_df, val_df):
    """Train the model"""
    print("\n" + "=" * 80)
    print("Starting Unsupervised Training (Phase 1 - V7 Simple - 100K)")
    print("=" * 80)

    train_dataset = prepare_dataset(train_df, tokenizer)
    val_dataset = prepare_dataset(val_df, tokenizer)

    print(f"\nTrain samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")

    total_steps = (len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) * EPOCHS
    print(f"Total training steps: {total_steps:,}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,  # larger batch for eval
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_steps=100,            # ✅ 50 → 100
        eval_strategy="steps",
        eval_steps=1000,              # ✅ 250 → 1000
        save_strategy="steps",
        save_steps=2000,              # ✅ 1000 → 2000
        save_total_limit=5,           # ✅ 3 → 5
        bf16=torch.cuda.is_available(),
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        seed=RANDOM_SEED,
        report_to="none",
        disable_tqdm=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        load_best_model_at_end=True,  # ✅ auto-load the best model
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    progress_callback = ProgressCallback(total_steps)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[progress_callback],
    )

    print(f"\n[Training config - Phase 1 V7 Simple - 100K]")
    print(f"  - Mode: Unsupervised CLM")
    print(f"  - Model: SmolLM2-360M (2025)")
    print(f"  - Format: numbers only! [180, 0.0, ...]")
    print(f"  - Data: 100,000 samples (10× the 10K) 🚀")
    print(f"  - Scenario ratios: preserved (Stratified Split)")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Gradient accum: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  - Effective batch: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  - Epochs: {EPOCHS} ⚡ (3 epochs are usually enough for 100K!)")
    print(f"  - Warmup steps: {WARMUP_STEPS} (smoother ramp-up)")
    print(f"  - DataLoader workers: 4 (multiprocessing)")
    print(f"  - Estimated duration: ~8–12 hours ⏱️")
    print(f"  - Checkpoints: every 2000 steps (safer)")

    print("\n" + "-" * 80)
    print("Starting V7 Simple 100K training...")
    print("-" * 80 + "\n")

    trainer.train()

    print("\n" + "=" * 80)
    print("✓ Phase 1 V7 100K training complete!")
    print("=" * 80)

    return trainer


# ============================================================================
# Save & Test
# ============================================================================

def save_model(model, tokenizer):
    """Save the model"""
    print("\n" + "=" * 80)
    print("Saving model (Phase 1 V7 - 100K)")
    print("=" * 80)

    final_model_dir = OUTPUT_DIR / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    print(f"\n✓ Phase 1 V7 100K model saved to: {final_model_dir}")


def test_model(model, tokenizer, test_df):
    """Model test - V7 Simple"""
    print("\n" + "=" * 80)
    print("Phase 1 V7 Model Test (numeric pattern check - 100K)")
    print("=" * 80)

    model.eval()

    # Test 1: generate from empty start
    print("\n[Test 1: generate from empty prefix]")
    print("=" * 80)

    prompt = "["
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # short for V7 numbers-only
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\nInput: '{prompt}'")
    print(f"\nGenerated:")
    print("-" * 80)
    print(generated)
    print("-" * 80)

    is_valid = validate_simple_format(generated)
    print(f"\n{'✅' if is_valid else '❌'} Format check: {'PASS' if is_valid else 'FAIL'}")

    # Test 2: continue from partial input
    print("\n" + "=" * 80)
    print("[Test 2: continue from partial input]")
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

    print(f"\nInput: '{prompt}'")
    print(f"\nGenerated:")
    print("-" * 80)
    print(generated)
    print("-" * 80)

    print("\n📌 Phase 1 V7 100K goals:")
    print("  ✓ Understand bracket [ ] structure")
    print("  ✓ Generate 6 numbers (comma-separated)")
    print("  ✓ Do NOT generate field names")
    print("  ✓ Learn perfect numeric grammar with 100K data")
    print("  ✓ Basis to fix the 21% failure cases!")


def validate_simple_format(output):
    """Validate V7 Simple format"""
    import re

    # Try extracting 6 numbers
    numbers = re.findall(r'[\d.]+', output)

    if len(numbers) < 6:
        print(f"  ❌ Not enough numbers: {len(numbers)} (need 6)")
        return False

    # Must contain opening bracket
    if '[' not in output:
        print(f"  ❌ Missing opening bracket")
        return False

    # Ensure no field names are generated (numbers only!)
    field_keywords = ['pos', 'time', 'speed', 'b1', 'b2', 'initial', 'final', 'power', 'bank']
    for keyword in field_keywords:
        if keyword in output.lower():
            print(f"  ❌ Found field name: '{keyword}' (must be numbers only!)")
            return False

    print(f"  ✅ Generated ≥6 numbers: {len(numbers)}")
    print(f"  ✅ No field names (clean)")

    return True


# ============================================================================
# Main
# ============================================================================

def main():
    """Main training pipeline"""
    print("\n")
    print("=" * 80)
    print("KOMODO Foundation Model - Phase 1 V7 Simple - 100K")
    print("SmolLM2-360M + numbers-only training + 100,000 samples")
    print("=" * 80)
    print(f"\nGPU: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - {torch.cuda.get_device_name(0)}")
        print(f"  - VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # 1) Data
    df = load_and_prepare_data()
    train_df, val_df, test_df = stratified_split(df)

    # 2) Model
    model, tokenizer = load_model_and_tokenizer()

    # 3) Train
    trainer = train_model(model, tokenizer, train_df, val_df)

    # 4) Save
    save_model(model, tokenizer)

    # 5) Quick test
    test_model(model, tokenizer, val_df)

    print("\n" + "=" * 80)
    print("✓ Phase 1 V7 100K DONE!")
    print("=" * 80)
    print(f"\nModel saved to: {OUTPUT_DIR / 'final_model'}")
    print("\n📌 100K improvements:")
    print("  ✅ 100,000 samples (10× the 10K)")
    print("  ✅ 5 epochs (enough for 100K)")
    print("  ✅ Wider pattern coverage")
    print("  ✅ Foundation to fix the 21% failure cases")
    print("  🎯 Phase 2 V7 100K target: 90–95%")
    print("\nNext: python smollm2_phase2_supervised_lora_numeric_simple_last.py")


if __name__ == "__main__":
    main()

'''
## 🔧 Key changes (100K optimization)

1. **Dataset path**: `master_dataset.csv` (100K)
2. **Epochs**: 20 → 5 (5 epochs are enough for 100K)
3. **Warmup**: 100 → 500 (smoother ramp-up)
4. **Tokenization batch**: 100 → 500 (memory / speed balance)
5. **Logging**: every 100 steps
6. **Eval**: every 1000 steps
7. **Checkpointing**: every 2000 steps
8. **Save total**: 3 → 5 (keep more)
9. **Load best model**: enabled

## ⏱️ Estimated duration

90K samples × 3 epochs ≈ 8–12 hours
'''
