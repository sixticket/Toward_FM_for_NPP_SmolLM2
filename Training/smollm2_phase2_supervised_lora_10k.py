# smollm2_phase2_supervised_lora_10k.py
"""
SmolLM2-360M Supervised Training for KOMODO (Phase 2) - V7.1 Simple
+ V7.1: Remove arrows; commas-only format
+ Data format: "[1.0, 1.5, 180, 0.0, 0.0, 100, 0.0, 0.0]"
+ Temperature 0.05 (more deterministic)
+ Based on Phase 1 V7 model
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
    """Display training progress"""

    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print("\n" + "🚀 " * 20)
        print("SmolLM2 Supervised Training START! (Phase 2 - V7.1 Simple)")
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
# Configuration - V7.1 Simple
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATASET_PATH = BASE_DIR / "dataset/master_dataset.csv"
PHASE1_MODEL_PATH = SCRIPT_DIR / "models/smollm2_unsupervised_numeric/final_model"  # V7 base
OUTPUT_DIR = SCRIPT_DIR / "models/smollm2_supervised_lora_v7_numeric_simple"        # V7.1 output
DATA_OUTPUT_DIR = SCRIPT_DIR / "processed_data"
RESULTS_DIR = SCRIPT_DIR / "evaluation_results"

# Model settings
MAX_SEQ_LENGTH = 256

# Data split (80/10/10 - Stratified)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# LoRA settings (same as V6)
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training settings (same as original 10K config)
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
EPOCHS = 30
LEARNING_RATE = 5e-5
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
EVAL_STEPS = 250

# Inference settings - V7.1 optimization
INFERENCE_TEMPERATURE = 0.05  # more deterministic
INFERENCE_TOP_P = 0.9
INFERENCE_MAX_TOKENS = 50     # short because numbers-only

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
# Data prep - V7.1 Simple (no arrows)
# ============================================================================

def create_supervised_format_simple(row):
    """
    V7.1 Simple: arrows removed; commas-only
    "[1.0, 1.5, 180, 0.0, 0.0, 100, 0.0, 0.0]"
    Order: [initial_power, final_power, b1_pos, b1_time, b1_speed, b2_pos, b2_time, b2_speed]
    """
    text = (
        f"[{row['initial_power']}, {row['final_power']}, "
        f"{row['b1_pos']}, {row['b1_time']}, {row['b1_speed']}, "
        f"{row['b2_pos']}, {row['b2_time']}, {row['b2_speed']}]"
    )
    return text


def load_and_prepare_data():
    """Load data and convert to V7.1 Simple format"""
    print("=" * 80)
    print("Loading data (V7.1 Simple Format - Phase 2)")
    print("=" * 80)

    df = pd.read_csv(DATASET_PATH)
    print(f"\nTotal rows: {len(df)}")

    df['scenario'] = df.apply(classify_scenario, axis=1)

    print("\n[Scenario distribution]")
    print("=" * 80)
    scenario_counts = df['scenario'].value_counts()
    print(scenario_counts)

    single_count = (df['scenario'].str.contains('single')).sum()
    simul_count = (df['scenario'] == 'simultaneous').sum()
    seq_count = (df['scenario'] == 'sequential').sum()

    print(f"\n✅ Single ops: {single_count} ({single_count / len(df) * 100:.1f}%)")
    print(f"   - Bank1 only: {(df['scenario'] == 'single_b1').sum()}")
    print(f"   - Bank2 only: {(df['scenario'] == 'single_b2').sum()}")
    print(f"✅ Simultaneous: {simul_count} ({simul_count / len(df) * 100:.1f}%)")
    print(f"✅ Sequential: {seq_count} ({seq_count / len(df) * 100:.1f}%)")

    # Convert to V7.1 Simple
    df['text'] = df.apply(create_supervised_format_simple, axis=1)

    print("\n[V7.1 Simple sample]")
    print("=" * 80)
    print(df.iloc[0]['text'])
    print("=" * 80)
    print("🎯 V7.1: arrows removed; commas-only")
    print("✅ Same structure as Phase 1 (just two more numbers up front)")
    print("✅ No special symbols (no '->')")
    print("✅ Easy parsing (8 numbers)")

    return df


def stratified_split(df):
    """Split preserving scenario ratios (80/10/10)"""
    print("\n" + "=" * 80)
    print("Stratified Split (80/10/10 - preserve scenario ratios)")
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

    print(f"\n[Train set: {len(train)}]")
    print(train['scenario'].value_counts(normalize=True) * 100)

    print(f"\n[Val set: {len(val)}]")
    print(val['scenario'].value_counts(normalize=True) * 100)

    print(f"\n[Test set: {len(test)}]")
    print(test['scenario'].value_counts(normalize=True) * 100)

    train_clean = train.reset_index(drop=True)
    val_clean = val.reset_index(drop=True)
    test_clean = test.reset_index(drop=True)

    print("\n✅ Scenario ratios preserved across all splits!")

    return train_clean, val_clean, test_clean


# ============================================================================
# Load model + apply LoRA
# ============================================================================

def load_model_with_lora(tokenizer):
    """Load Phase 1 V7 model and attach a LoRA adapter"""
    print("\n" + "=" * 80)
    print("Loading Phase 1 V7 model + adding LoRA adapter")
    print("=" * 80)

    if not PHASE1_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find Phase 1 V7 model: {PHASE1_MODEL_PATH}\n"
            "Please complete Phase 1 V7 training first!"
        )

    model = AutoModelForCausalLM.from_pretrained(
        str(PHASE1_MODEL_PATH),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    print(f"\n✓ Phase 1 V7 model loaded: {PHASE1_MODEL_PATH}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / total_params

    print(f"\n✓ LoRA adapter attached!")
    print(f"  - Rank (r): {LORA_R}")
    print(f"  - Alpha: {LORA_ALPHA}")
    print(f"  - Dropout: {LORA_DROPOUT}")
    print(f"  - Target modules: {LORA_TARGET_MODULES}")
    print(f"\n📊 Parameter stats:")
    print(f"  - Trainable: {trainable_params:,} ({trainable_percent:.2f}%)")
    print(f"  - Total: {total_params:,}")
    print(f"  - Phase 1 V7 weights: 100% frozen ✅")

    if torch.cuda.is_available():
        model = model.cuda()

    model.print_trainable_parameters()

    return model


def load_model_and_tokenizer():
    """Load tokenizer + LoRA-augmented model"""
    tokenizer = AutoTokenizer.from_pretrained(str(PHASE1_MODEL_PATH))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model_with_lora(tokenizer)

    return model, tokenizer


# ============================================================================
# Dataset tokenization - V7.1 Simple (no arrows)
# ============================================================================

def tokenize_supervised_simple(examples, tokenizer):
    """
    V7.1 Simple: mask the first two numbers (inputs) and train only on the last six (outputs)
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

        text = texts[i]
        # Mask the prefix "[1.0, 1.5," (two numbers + commas + opening bracket)
        prompt_part = text.split(',')[0:2]  # "[1.0", " 1.5"
        prompt_text = ','.join(prompt_part) + ','  # "[1.0, 1.5,"

        prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)['input_ids']
        mask_length = len(prompt_tokens)

        label = input_ids.copy()
        for j in range(min(mask_length, len(label))):
            label[j] = -100

        labels.append(label)

    tokenized['labels'] = labels
    return tokenized


def prepare_dataset(df, tokenizer):
    """Convert to HuggingFace Dataset"""
    dataset = Dataset.from_pandas(df[['text']])

    tokenized_dataset = dataset.map(
        lambda x: tokenize_supervised_simple(x, tokenizer),
        batched=True,
        batch_size=100,
        remove_columns=['text'],
        desc="Tokenizing (V7.1 Simple - LoRA)"
    )

    return tokenized_dataset


# ============================================================================
# Training
# ============================================================================

def train_model(model, tokenizer, train_df, val_df):
    """Train the model (LoRA V7.1 Simple)"""
    print("\n" + "=" * 80)
    print("Supervised Training START (Phase 2 - V7.1 Simple)")
    print("=" * 80)

    train_dataset = prepare_dataset(train_df, tokenizer)
    val_dataset = prepare_dataset(val_df, tokenizer)

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    total_steps = (len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) * EPOCHS
    print(f"Total training steps: {total_steps}")

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
        save_steps=1000,
        save_total_limit=3,
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

    print(f"\n[Training config - Phase 2 V7.1 Simple]")
    print(f"  - Method: train LoRA adapter only")
    print(f"  - Base: Phase 1 V7 model (numbers-only)")
    print(f"  - Format: arrows removed; unified commas-only")
    print(f"  - LoRA Rank: {LORA_R}")
    print(f"  - LoRA Alpha: {LORA_ALPHA}")
    print(f"  - Learning rate: {LEARNING_RATE:.2e}")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Gradient accum: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  - Effective batch: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  - Temperature: {INFERENCE_TEMPERATURE} 🎯")
    print(f"  - 🔒 Phase 1 V7 parameters: frozen")

    print("\n" + "-" * 80)
    print("Starting V7.1 Simple training...")
    print("-" * 80 + "\n")

    trainer.train()

    print("\n" + "=" * 80)
    print("✓ Phase 2 V7.1 Simple training complete!")
    print("=" * 80)

    return trainer


# ============================================================================
# Save
# ============================================================================

def save_model(model, tokenizer):
    """Save the LoRA adapter"""
    print("\n" + "=" * 80)
    print("Saving LoRA model (Phase 2 V7.1 Simple)")
    print("=" * 80)

    final_model_dir = OUTPUT_DIR / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)

    # Save only the LoRA adapter
    model.save_pretrained(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    print(f"\n✓ Phase 2 V7.1 LoRA model saved: {final_model_dir}")
    print("  - Only the LoRA adapter is saved (very small size)")
    print("  - Must be loaded together with the Phase 1 V7 base model")


# ============================================================================
# Inference & parsing - V7.1 Simple
# ============================================================================

def parse_prediction_simple(prediction_text):
    """
    V7.1 Simple: extract 8 numbers (super simple)
    Example: "[1.0, 1.5, 180, 0.0, 0.0, 100, 0.0, 0.0]"
    First two are inputs; last six are outputs.
    """
    numbers = re.findall(r'[\d.]+', prediction_text)

    if len(numbers) < 8:
        return None

    try:
        values = [float(x) for x in numbers[-6:]]  # last six as outputs (safest)
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
    """Predict control-rod parameters from power values (V7.1 Simple)"""
    # V7.1 format: ends with a comma
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
# Evaluate on the full Test set
# ============================================================================

def evaluate_on_test_set(model, tokenizer, test_df):
    """Evaluate on the full Test set (1,000 samples) + stats (V7.1)"""
    print("\n" + "=" * 80)
    print("🎯 Starting full Test set evaluation (1,000) - V7.1 Simple")
    print("=" * 80)

    model.eval()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    parse_success = 0
    parse_fail = 0

    print("\nRunning predictions...")
    for idx, row in test_df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"  Progress: {idx + 1}/{len(test_df)}...")

        initial_power = row['initial_power']
        final_power = row['final_power']

        prediction_text = generate_prediction(model, tokenizer, initial_power, final_power)
        parsed = parse_prediction_simple(prediction_text)

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
            errors = {}
            for key in ground_truth:
                errors[f'{key}_error'] = abs(parsed[key] - ground_truth[key])
            result['errors'] = errors
        else:
            parse_fail += 1
            result['errors'] = {}

        results.append(result)

    print(f"\n✓ Predictions done: {len(results)}")
    print(f"  - Parse success: {parse_success} ({parse_success / len(results) * 100:.1f}%)")
    print(f"  - Parse fail:    {parse_fail} ({parse_fail / len(results) * 100:.1f}%)")

    print(f"\n📊 Parsing success rate by version:")
    print(f"  - V6 (fields):    87.1%")
    print(f"  - V7 (arrows):    90.0%")
    print(f"  - V7.1 (commas):  {parse_success / len(results) * 100:.1f}% ⭐")

    print("\n" + "=" * 80)
    print("📊 Quantitative evaluation")
    print("=" * 80)

    if parse_success > 0:
        all_errors = {key: [] for key in ['b1_pos', 'b1_time', 'b1_speed', 'b2_pos', 'b2_time', 'b2_speed']}

        for r in results:
            if r['parsed_success']:
                for key in all_errors:
                    all_errors[key].append(r['errors'][f'{key}_error'])

        print("\n[Overall error stats]")
        print("-" * 80)
        for key in ['b1_pos', 'b1_time', 'b1_speed', 'b2_pos', 'b2_time', 'b2_speed']:
            errors = all_errors[key]
            mae = np.mean(errors)
            mse = np.mean([e ** 2 for e in errors])
            rmse = np.sqrt(mse)
            median = np.median(errors)

            print(f"{key:12s}: MAE={mae:6.3f}, RMSE={rmse:6.3f}, Median={median:6.3f}")

        print("\n[Error by scenario]")
        print("-" * 80)
        scenarios = test_df['scenario'].unique()

        for scenario in scenarios:
            scenario_results = [r for r in results if r['scenario'] == scenario and r['parsed_success']]

            if len(scenario_results) > 0:
                print(f"\n{scenario} ({len(scenario_results)}):")

                scenario_errors = {key: [] for key in
                                   ['b1_pos', 'b1_time', 'b1_speed', 'b2_pos', 'b2_time', 'b2_speed']}

                for r in scenario_results:
                    for key in scenario_errors:
                        scenario_errors[key].append(r['errors'][f'{key}_error'])

                for key in ['b1_pos', 'b1_time', 'b1_speed', 'b2_pos', 'b2_time', 'b2_speed']:
                    errors = scenario_errors[key]
                    mae = np.mean(errors)
                    print(f"  {key:12s}: MAE={mae:6.3f}")

    results_file = RESULTS_DIR / "test_set_results_v7_simple.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Detailed results saved: {results_file}")

    print("\n" + "=" * 80)
    print("🔍 Sample predictions (first 5)")
    print("=" * 80)

    for i in range(min(5, len(results))):
        r = results[i]
        print(f"\n[Sample {i + 1}] {r['scenario']}")
        print(f"Input: [{r['initial_power']}, {r['final_power']}]")
        print(f"Prediction: {r['prediction_text']}")

        if r['parsed_success']:
            print("Parsed values:")
            for key in ['b1_pos', 'b1_time', 'b1_speed', 'b2_pos', 'b2_time', 'b2_speed']:
                pred = r['prediction'][key]
                truth = r['ground_truth'][key]
                error = r['errors'][f'{key}_error']
                print(f"  {key:12s}: {pred:6.1f} (truth: {truth:6.1f}, error: {error:5.2f})")
        else:
            print("⚠️ Parse failed")
        print("-" * 80)

    return results


# ============================================================================
# Quick test
# ============================================================================

def quick_test(model, tokenizer, test_df):
    """Quick test (a few samples) - V7.1"""
    print("\n" + "=" * 80)
    print("⚡ Quick Test (sample predictions) - V7.1 Simple")
    print("=" * 80)

    model.eval()

    # Test 1: arbitrary power values
    print("\n[TEST 1: predict from arbitrary power values]")
    print("=" * 80)

    test_cases = [
        (1.0, 1.5),
        (1.0, 0.8),
        (1.0, 2.0),
    ]

    for initial, final in test_cases:
        prediction = generate_prediction(model, tokenizer, initial, final)

        print(f"\nInput: [{initial}, {final}]")
        print(f"Prediction: {prediction}")

        parsed = parse_prediction_simple(prediction)
        if parsed:
            print("✅ Parse success:")
            for key, val in parsed.items():
                print(f"  {key}: {val}")
        else:
            print("⚠️ Parse failed")

    # Test 2: real samples
    print("\n" + "=" * 80)
    print("[TEST 2: 3 real samples from dataset]")
    print("=" * 80)

    for i in range(3):
        sample = test_df.iloc[i]
        prediction = generate_prediction(model, tokenizer, sample['initial_power'], sample['final_power'])

        answer = f"[{sample['initial_power']}, {sample['final_power']}, {sample['b1_pos']}, {sample['b1_time']}, {sample['b1_speed']}, {sample['b2_pos']}, {sample['b2_time']}, {sample['b2_speed']}]"

        print(f"\n[Sample {i + 1}] {sample['scenario']}")
        print(f"Input: [{sample['initial_power']}, {sample['final_power']}]")
        print(f"Prediction: {prediction}")
        print(f"Answer: {answer}")

        parsed = parse_prediction_simple(prediction)
        if parsed:
            print("✅ Parse success")
        else:
            print("⚠️ Parse failed")
        print("-" * 80)

    print("\n📌 V7.1 Simple highlights:")
    print("  ✓ Arrows fully removed (no '->')")
    print("  ✓ Commas only (simple)")
    print("  ✓ Same structure as Phase 1 (just more numbers)")
    print("  ✓ Eliminates sources of confusion")
    print("  ✓ Target parsing success ≥95%")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main training pipeline"""
    print("\n")
    print("=" * 80)
    print("KOMODO Foundation Model - Phase 2 V7.1 Simple")
    print("SmolLM2-360M + LoRA (arrows removed, fully unified)")
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
    print("🎯 Run full Test set evaluation?")
    print("   (1,000 predictions; ~10–20 minutes)")
    print("=" * 80)

    user_input = input("Run full evaluation? (y/n): ").strip().lower()

    if user_input == 'y':
        evaluate_on_test_set(model, tokenizer, test_df)
    else:
        print("\n⏩ Skipping full evaluation")

    print("\n" + "=" * 80)
    print("✓ Phase 2 V7.1 Simple DONE!")
    print("=" * 80)
    print(f"\nModel saved: {OUTPUT_DIR / 'final_model'}")
    print("\n🎯 V7.1 Simple recap:")
    print("  ✅ Arrows completely removed (no '->')")
    print("  ✅ Commas only (super simple)")
    print("  ✅ Same structure as Phase 1 (more numbers)")
    print("  ✅ No special symbols")
    print("  ✅ Parsing is trivial (8 numbers)")
    print("  ✅ Confusion eliminated")
    print("  🎯 Expected parsing success: ≥95%")
    print("\n📊 Version comparison:")
    print("  - V6 (fields):    87.1%")
    print("  - V7 (arrows):    90.0%")
    print("  - V7.1 (commas):  95%+ target ⭐")


if __name__ == "__main__":
    main()
