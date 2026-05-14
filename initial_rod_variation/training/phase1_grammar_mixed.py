"""
Phase 1 (CPT, grammar learning) on the mixed-init 100K dataset.

Format change vs paper:
    Paper Phase 1: "[b1_pos, b1_time, b1_speed, b2_pos, b2_time, b2_speed]"  (6 numbers)
    Mixed Phase 1: "[init_b1, init_b2, b1_pos, b1_time, b1_speed, b2_pos, b2_time, b2_speed]"  (8 numbers)

This teaches the grammar of valid commands *given init*, which is the
prerequisite for Phase 2 to learn init-conditional policies.

Run from WSL with the project venv.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainerCallback, TrainingArguments,
)

# ============================================================================
# Paths and config
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent           # training/
EXP_DIR = SCRIPT_DIR.parent                             # exp1_initial_rod_variation/
DATA_DIR = EXP_DIR / "data_generation"
DATASET_PATH = DATA_DIR / "master_dataset_mixed_init_100k.csv"

OUTPUT_DIR = SCRIPT_DIR / "models" / "phase1_grammar_mixed"
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"

MAX_SEQ_LENGTH = 256
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
EPOCHS = 5                # 100K -> 5 epochs (matches paper's 100K Phase 1)
LEARNING_RATE = 5e-5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
TRAIN_RATIO = 0.9
VAL_RATIO = 0.1
SEED = 42

np.random.seed(SEED)


# ============================================================================
# Helpers
# ============================================================================

def classify_scenario(row):
    b1 = row["b1_time"] > 0
    b2 = row["b2_time"] > 0
    if b1 and not b2: return "single_b1"
    if b2 and not b1: return "single_b2"
    if b1 and b2:
        return "simultaneous" if abs(row["b1_time"] - row["b2_time"]) < 0.01 else "sequential"
    return "none"


def make_text(row):
    """[init_b1, init_b2, b1_pos, b1_time, b1_speed, b2_pos, b2_time, b2_speed]"""
    return (
        f"[{int(row['init_b1'])}, {int(row['init_b2'])}, "
        f"{row['b1_pos']}, {row['b1_time']}, {row['b1_speed']}, "
        f"{row['b2_pos']}, {row['b2_time']}, {row['b2_speed']}]"
    )


def tokenize_unsupervised(examples, tokenizer):
    tok = tokenizer(
        examples["text"], truncation=True, max_length=MAX_SEQ_LENGTH,
        padding="max_length", return_tensors=None,
    )
    tok["labels"] = tok["input_ids"].copy()
    return tok


class ProgressCallback(TrainerCallback):
    def __init__(self, total_steps):
        self.total_steps = total_steps; self.start = None
    def on_train_begin(self, args, state, control, **kw):
        self.start = time.time()
    def on_log(self, args, state, control, logs=None, **kw):
        if not logs: return
        i = state.global_step
        if i == 0: return
        elapsed = time.time() - self.start
        eta = elapsed / i * (self.total_steps - i)
        loss = logs.get("loss", "N/A")
        loss_s = f"{loss:.4f}" if isinstance(loss, float) else str(loss)
        print(f"  [step {i}/{self.total_steps}] loss={loss_s}  elapsed={timedelta(seconds=int(elapsed))}  eta={timedelta(seconds=int(eta))}", flush=True)


# ============================================================================
# Main
# ============================================================================

def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"mixed dataset missing: {DATASET_PATH}\n"
                                "Run data_generation/{sample_existing.py, generate_mirrored.py, combine.py} first.")
    print(f"[load] {DATASET_PATH}", flush=True)
    df = pd.read_csv(DATASET_PATH)
    print(f"  rows: {len(df)}")
    print(f"  init=(180,100): {(df.init_b1==180).sum()}")
    print(f"  init=(100,180): {(df.init_b1==100).sum()}")

    df["scenario"] = df.apply(classify_scenario, axis=1)
    df["text"] = df.apply(make_text, axis=1)
    print(f"\n[sample] {df.iloc[0]['text']}", flush=True)

    # stratify by (init, scenario) so both inits keep their pattern balance
    df["strat"] = df["init_b1"].astype(str) + "_" + df["scenario"]
    train_df, val_df = train_test_split(
        df, test_size=VAL_RATIO, stratify=df["strat"], random_state=SEED,
    )
    print(f"\n[split] train={len(train_df)}  val={len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n[model] {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        model = model.cuda()

    train_ds = Dataset.from_pandas(train_df[["text"]]).map(
        lambda x: tokenize_unsupervised(x, tokenizer), batched=True, batch_size=500,
        remove_columns=["text"], desc="tokenize-train",
    )
    val_ds = Dataset.from_pandas(val_df[["text"]]).map(
        lambda x: tokenize_unsupervised(x, tokenizer), batched=True, batch_size=500,
        remove_columns=["text"], desc="tokenize-val",
    )

    total_steps = (len(train_ds) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) * EPOCHS
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_steps=100,
        eval_strategy="steps", eval_steps=1000,
        save_strategy="steps", save_steps=2000, save_total_limit=5,
        bf16=torch.cuda.is_available(),
        optim="adamw_torch", lr_scheduler_type="cosine",
        seed=SEED, report_to="none",
        dataloader_num_workers=4, dataloader_pin_memory=True,
        load_best_model_at_end=True, metric_for_best_model="loss", greater_is_better=False,
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        callbacks=[ProgressCallback(total_steps)],
    )

    # Auto-resume from latest checkpoint if one exists in OUTPUT_DIR.
    def _latest_ckpt(out_dir: Path):
        if not out_dir.exists():
            return None
        cps = [d for d in out_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
        if not cps:
            return None
        return max(cps, key=lambda d: int(d.name.split("-")[1]))

    resume_ckpt = _latest_ckpt(OUTPUT_DIR)
    if resume_ckpt is not None:
        print(f"\n[resume] from {resume_ckpt}", flush=True)
        print(f"[train] total_steps={total_steps}, epochs={EPOCHS}, batch={BATCH_SIZE}*{GRADIENT_ACCUMULATION_STEPS}\n", flush=True)
        trainer.train(resume_from_checkpoint=str(resume_ckpt))
    else:
        print(f"\n[train] total_steps={total_steps}, epochs={EPOCHS}, batch={BATCH_SIZE}*{GRADIENT_ACCUMULATION_STEPS} (fresh start)\n", flush=True)
        trainer.train()

    final = OUTPUT_DIR / "final_model"
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final))
    tokenizer.save_pretrained(str(final))
    print(f"\n[done] saved -> {final}")


if __name__ == "__main__":
    main()
