import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ============================================================================
# [PATHS] 스크립트 위치 기준 (어디서 실행해도 동일)
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent          # extend/
REPO_ROOT  = SCRIPT_DIR.parent                        # repo root

# Phase 1/2 100K 모델 (training/ 에서 학습된 결과물)
BASE_MODEL_PATH = REPO_ROOT / "training/models/smollm2_unsupervised_numeric_100k/final_model"
LORA_PATH      = REPO_ROOT / "training/models/smollm2_supervised_lora_v7_numeric_simple_100k/final_model"

# Variable-window 10K 데이터셋 (data_generation 으로 생성)
DATASET_PATH   = SCRIPT_DIR / "data_generation/dataset_10k_window_unfiltered/master_dataset_10000.csv"

OUTPUT_ROOT    = SCRIPT_DIR / "models"

# ============================================================================
# [HYPERPARAMS]
# ============================================================================
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-4   # LoRA LR
ENCODER_LR = 5e-4      # Encoder LR
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 64


def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_window_bin(df: pd.DataFrame) -> pd.DataFrame:
    """
    monitor_window (예: 60~100)을 4개 구간으로 binning해서 stratify에 사용
    """
    df = df.copy()
    bins = [59.9, 70.0, 80.0, 90.0, 100.1]
    labels = ["60_70", "70_80", "80_90", "90_100"]
    df["window_bin"] = pd.cut(df["monitor_window"].astype(float), bins=bins, labels=labels, include_lowest=True)
    df["window_bin"] = df["window_bin"].astype(str)
    return df


# ============================================================================
# 1) Input Encoder (Input: [monitor_window, P_init, P_target] -> 2 tokens)
# ============================================================================
class InputEncoder(nn.Module):
    def __init__(self, input_dim=3, num_output_tokens=2, hidden_size=576,
                 encoder_hidden=128, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, encoder_hidden),
                nn.LayerNorm(encoder_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = encoder_hidden

        self.encoder = nn.Sequential(*layers)
        self.projection = nn.Linear(encoder_hidden, num_output_tokens * hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)

        self.num_output_tokens = num_output_tokens
        self.hidden_size = hidden_size

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.encoder(x)
        x = self.projection(x)
        x = x.view(x.size(0), self.num_output_tokens, self.hidden_size)
        return self.output_norm(x)


# ============================================================================
# 2) Model Wrapper
# ============================================================================
class WindowKOMODOModel(nn.Module):
    def __init__(self, base_path, lora_path, encoder_hidden=128, encoder_layers=2):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(str(base_path))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("  [Model] Loading Base LLM & LoRA...")
        self.llm = AutoModelForCausalLM.from_pretrained(str(base_path), torch_dtype=torch.float32)
        self.llm = PeftModel.from_pretrained(self.llm, str(lora_path))

        self.input_encoder = InputEncoder(
            input_dim=3,
            hidden_size=self.llm.config.hidden_size,
            encoder_hidden=encoder_hidden,
            num_layers=encoder_layers,
            dropout=0.1
        )
        self.embed_tokens = self.llm.get_input_embeddings()

    def configure_training(self, train_lora: bool):
        # Encoder always train
        for p in self.input_encoder.parameters():
            p.requires_grad = True

        if train_lora:
            print("  [Config] Mode: Adapter + LoRA (Train LoRA)")
            for name, p in self.llm.named_parameters():
                if "lora" in name.lower():
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        else:
            print("  [Config] Mode: Adapter Only (Freeze LLM+LoRA)")
            for p in self.llm.parameters():
                p.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  [Config] Trainable Params: {trainable:,}")

    def forward(self, input_values, output_token_ids=None, labels=None):
        input_embeds = self.input_encoder(input_values)  # (B, 2, H)

        if output_token_ids is not None:
            output_embeds = self.embed_tokens(output_token_ids)  # (B, L, H)
            combined_embeds = torch.cat([input_embeds, output_embeds], dim=1)  # (B, 2+L, H)
        else:
            combined_embeds = input_embeds

        if labels is not None:
            dummy = torch.full((input_values.size(0), 2), -100, dtype=labels.dtype, device=labels.device)
            padded_labels = torch.cat([dummy, labels], dim=1)
            outputs = self.llm(inputs_embeds=combined_embeds, labels=padded_labels, return_dict=True)
            return outputs.loss

        return self.llm(inputs_embeds=combined_embeds, return_dict=True).logits


# ============================================================================
# 3) Dataset
# ============================================================================
class WindowKOMODODataset(Dataset):
    def __init__(self, df, tokenizer, max_length=64):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Inputs
        monitor_window = float(row["monitor_window"]) / 100.0  # 60~100 -> 0.60~1.00
        p_init = float(row["initial_power"])
        p_target = float(row["final_power"])
        input_tensor = torch.tensor([monitor_window, p_init, p_target], dtype=torch.float32)

        # Output text (6 control vars)
        output_text = (
            f"[{row['b1_pos']}, {row['b1_time']}, {row['b1_speed']}, "
            f"{row['b2_pos']}, {row['b2_time']}, {row['b2_speed']}]"
        )

        enc = self.tokenizer(
            output_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        ids = enc["input_ids"].squeeze(0)
        labels = ids.clone()
        labels[enc["attention_mask"].squeeze(0) == 0] = -100

        return {"input_values": input_tensor, "output_ids": ids, "labels": labels}


# ============================================================================
# 4) Eval helpers
# ============================================================================
@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        loss = model(
            batch["input_values"].to(device),
            batch["output_ids"].to(device),
            batch["labels"].to(device)
        )
        total += loss.item()
        n += 1
    return total / max(1, n)


def save_best(model, out_path: Path, train_lora: bool):
    save_dict = {"encoder": model.input_encoder.state_dict()}
    if train_lora:
        save_dict["lora"] = {k: v for k, v in model.llm.state_dict().items() if "lora" in k.lower()}
    torch.save(save_dict, out_path)


def load_best(model, ckpt_path: Path, train_lora: bool, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.input_encoder.load_state_dict(ckpt["encoder"], strict=True)
    if train_lora and "lora" in ckpt:
        model.llm.load_state_dict(ckpt["lora"], strict=False)


# ============================================================================
# 5) Experiment Runner (Stratified 80/10/10)
# ============================================================================
def run_experiment(exp_name, dataset_path, train_lora, output_dir_name):
    print("\n" + "=" * 80)
    print(f"🧪 STARTING: {exp_name}")
    print(f"   - Train LoRA: {train_lora}")
    print(f"   - Output Dir: {output_dir_name}")
    print("=" * 80)

    dataset_path = Path(dataset_path)
    print(f"   ▶ Checking Dataset Path: {dataset_path}")
    if not dataset_path.exists():
        print(f"❌ Error: Dataset not found at {dataset_path}")
        return

    df = pd.read_csv(dataset_path)

    # Strat key = window_bin + pattern
    df = add_window_bin(df)

    # 80/10/10 stratified split
    try:
        df["strat"] = df["window_bin"] + "_" + df["pattern"].astype(str)

        train_df, temp_df = train_test_split(
            df,
            test_size=0.2,
            stratify=df["strat"],
            random_state=RANDOM_SEED
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,                 # 0.2의 절반 => 0.1/0.1
            stratify=temp_df["strat"],
            random_state=RANDOM_SEED
        )

    except Exception as e:
        print(f"  [Warn] Stratified split failed -> fallback random split. Reason: {e}")
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_SEED)

    print(f"   Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Model
    model = WindowKOMODOModel(BASE_MODEL_PATH, LORA_PATH).to(DEVICE)
    model.configure_training(train_lora=train_lora)

    train_ds = WindowKOMODODataset(train_df, model.tokenizer, max_length=MAX_LENGTH)
    val_ds = WindowKOMODODataset(val_df, model.tokenizer, max_length=MAX_LENGTH)
    test_ds = WindowKOMODODataset(test_df, model.tokenizer, max_length=MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Optimizer / Scheduler
    params = [{"params": model.input_encoder.parameters(), "lr": ENCODER_LR}]
    if train_lora:
        lora_params = [p for n, p in model.llm.named_parameters() if "lora" in n.lower() and p.requires_grad]
        params.append({"params": lora_params, "lr": LEARNING_RATE})

    optimizer = torch.optim.AdamW(params, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[p["lr"] for p in params],
        total_steps=len(train_loader) * EPOCHS,
        pct_start=WARMUP_RATIO
    )

    # Output dir
    output_dir = OUTPUT_ROOT / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "best_model.pt"

    best_val = float("inf")
    history = []

    # Train loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        if train_lora:
            model.llm.train()
        else:
            model.llm.eval()

        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch}", leave=False)
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)

            loss = model(
                batch["input_values"].to(DEVICE),
                batch["output_ids"].to(DEVICE),
                batch["labels"].to(DEVICE)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = total_loss / max(1, len(train_loader))
        val_loss = evaluate_loss(model, val_loader, DEVICE)

        print(f"   [{exp_name}] Ep {epoch}: Train {train_loss:.4f}, Val {val_loss:.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        pd.DataFrame(history).to_csv(output_dir / "loss_history.csv", index=False)

        if val_loss < best_val:
            best_val = val_loss
            save_best(model, ckpt_path, train_lora=train_lora)
            print(f"     -> Best Model Saved (val={best_val:.4f}) to {ckpt_path.name}")

    # Final test eval (load best)
    load_best(model, ckpt_path, train_lora=train_lora, device=DEVICE)
    test_loss = evaluate_loss(model, test_loader, DEVICE)
    print(f"✅ Finished: {exp_name}")
    print(f"   Best Val Loss: {best_val:.4f}")
    print(f"   Test Loss (best-val checkpoint): {test_loss:.4f}\n")

    # Cleanup
    del model, optimizer, scheduler, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()


# ============================================================================
# Main
# ============================================================================
def main():
    seed_everything(RANDOM_SEED)

    print("[PATH CHECK]")
    print(" SCRIPT_DIR   =", SCRIPT_DIR)
    print(" EXTEND_DIR   =", EXTEND_DIR)
    print(" MY_DIR       =", MY_DIR)
    print(" BASE_MODEL   =", BASE_MODEL_PATH, "exists:", BASE_MODEL_PATH.exists())
    print(" LORA_PATH    =", LORA_PATH,      "exists:", LORA_PATH.exists())
    print(" DATASET_PATH =", DATASET_PATH,   "exists:", DATASET_PATH.exists())
    print(" OUTPUT_ROOT  =", OUTPUT_ROOT)

    print("=" * 80)
    print("🌙 WINDOW DUAL EXPERIMENT RUNNER (No Boron, + Monitor Window Input, Stratified 80/10/10)")
    print("=" * 80)

    # 1) Adapter Only
    run_experiment(
        exp_name="EXP 1: Window Adapter Only",
        dataset_path=DATASET_PATH,
        train_lora=False,
        output_dir_name="smollm2_window_encoder_only_10k"
    )

    # 2) Adapter + LoRA
    run_experiment(
        exp_name="EXP 2: Window Full Training",
        dataset_path=DATASET_PATH,
        train_lora=True,
        output_dir_name="smollm2_window_encoder_extended_10k"
    )

    print("\n🎉🎉🎉 ALL EXPERIMENTS COMPLETED! 🎉🎉🎉")


if __name__ == "__main__":
    main()
