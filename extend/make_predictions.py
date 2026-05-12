import os
import gc
import random
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============================================================================
# [경로 설정]
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent  # extend/
REPO_ROOT  = SCRIPT_DIR.parent                # repo root

# 1. Phase 1/2 100K 모델 (training/ 에서 학습된 결과물)
BASE_MODEL_PATH = REPO_ROOT / "training/models/smollm2_unsupervised_numeric_100k/final_model"
LORA_PATH = REPO_ROOT / "training/models/smollm2_supervised_lora_v7_numeric_simple_100k/final_model"

# 2. 검증할 타겟 모델들 (train_window.py 가 SCRIPT_DIR/models/ 에 출력)
TARGET_MODELS = [
    {
        "name": "Extended (Adapter+LoRA)",
        "path": SCRIPT_DIR / "models/smollm2_window_encoder_extended_10k",
        "output_csv": "prediction_extended_2k.csv"
    },
    {
        "name": "Adapter Only (Encoder Only)",
        "path": SCRIPT_DIR / "models/smollm2_window_encoder_only_10k",
        "output_csv": "prediction_adapter_only_2k.csv"
    }
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# [모델 클래스] (기존과 동일)
# ============================================================================
class InputEncoder(torch.nn.Module):
    def __init__(self, input_dim=3, num_output_tokens=2, hidden_size=576, encoder_hidden=128, num_layers=2,
                 dropout=0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                torch.nn.Linear(in_dim, encoder_hidden),
                torch.nn.LayerNorm(encoder_hidden),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout)
            ])
            in_dim = encoder_hidden
        self.encoder = torch.nn.Sequential(*layers)
        self.projection = torch.nn.Linear(encoder_hidden, num_output_tokens * hidden_size)
        self.output_norm = torch.nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.projection(x)
        x = x.view(x.size(0), 2, -1)
        return self.output_norm(x)


class WindowKOMODOModel(torch.nn.Module):
    def __init__(self, base_path, lora_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(str(base_path))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(str(base_path), torch_dtype=torch.float32)
        self.llm = PeftModel.from_pretrained(self.llm, str(lora_path))
        self.input_encoder = InputEncoder(hidden_size=self.llm.config.hidden_size)

    def forward_generate(self, input_values, max_new_tokens=50):
        input_embeds = self.input_encoder(input_values)
        outputs = self.llm.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            temperature=0.001
        )
        return outputs


# ============================================================================
# [유틸 함수]
# ============================================================================
def parse_prediction(text):
    try:
        content = text.split('[')[-1].split(']')[0]
        vals = [float(x.strip()) for x in content.split(',')]
        if len(vals) == 6:
            return vals
    except:
        pass
    return None


def generate_test_cases(num_cases=2000):
    cases = []
    monitor_windows = [60.0, 70.0, 80.0, 90.0, 100.0]

    for i in range(num_cases):
        mw = random.choice(monitor_windows) + random.uniform(-1.0, 1.0)
        mw = max(60.0, min(100.0, round(mw, 1)))

        p_init = 1.0
        p_target = p_init * random.uniform(0.5, 1.5)

        cases.append({
            "run_id": f"val_{i:04d}",
            "monitor_window": mw,
            "initial_power": round(p_init, 2),
            "final_power": round(p_target, 4)
        })
    return cases


# ============================================================================
# [메인 실행]
# ============================================================================
def main():
    print("🚀 [Validation Generator] Starting Dual Model Inference...")
    print(f"⚡ Device: {DEVICE}")
    print(f"📂 Base Model Dir: {BASE_MODEL_PATH}")

    # 경로 존재 확인
    if not BASE_MODEL_PATH.exists():
        print(f"❌ Error: Base model not found at {BASE_MODEL_PATH}")
        return

    print("\n🎲 Generating 2,000 Test Cases (Range: 50% ~ 150%)...")
    test_cases = generate_test_cases(2000)

    for target in TARGET_MODELS:
        model_name = target["name"]
        model_dir = target["path"]
        output_file = SCRIPT_DIR / target["output_csv"]

        print(f"\n" + "=" * 60)
        print(f"🔮 Processing: {model_name}")
        print(f"   Model Path: {model_dir}")
        print("=" * 60)

        if not model_dir.exists():
            print(f"❌ Error: Target model directory not found: {model_dir}")
            continue

        try:
            model = WindowKOMODOModel(BASE_MODEL_PATH, LORA_PATH).to(DEVICE)

            ckpt_path = model_dir / "best_model.pt"
            if not ckpt_path.exists():
                print(f"❌ Error: best_model.pt not found at {ckpt_path}")
                continue

            checkpoint = torch.load(ckpt_path, map_location=DEVICE)
            model.input_encoder.load_state_dict(checkpoint['encoder'])

            if 'lora' in checkpoint:
                print("   -> Loading Finetuned LoRA weights...")
                model.llm.load_state_dict(checkpoint['lora'], strict=False)
            else:
                print("   -> Using Base LoRA weights (Adapter Only mode)...")

            model.eval()

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            continue

        results = []
        parse_count = 0

        with torch.no_grad():
            for case in tqdm(test_cases, desc=f"Inference"):
                inp_tensor = torch.tensor([[
                    case['monitor_window'] / 100.0,
                    case['initial_power'],
                    case['final_power']
                ]], dtype=torch.float32).to(DEVICE)

                gen_ids = model.forward_generate(inp_tensor)
                output_text = model.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

                res = case.copy()
                res['full_text'] = output_text

                vals = parse_prediction(output_text)
                if vals:
                    res['b1_pos'], res['b1_time'], res['b1_speed'] = vals[0], vals[1], vals[2]
                    res['b2_pos'], res['b2_time'], res['b2_speed'] = vals[3], vals[4], vals[5]
                    res['parsed'] = True
                    parse_count += 1
                else:
                    res['b1_pos'] = res['b1_time'] = res['b1_speed'] = None
                    res['b2_pos'] = res['b2_time'] = res['b2_speed'] = None
                    res['parsed'] = False

                results.append(res)

        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"✅ Saved: {output_file.name} (Parsed: {parse_count}/2000)")

        del model
        torch.cuda.empty_cache()
        gc.collect()

    print("\n🎉 All Done! GPU 작업 완료.")


if __name__ == "__main__":
    main()