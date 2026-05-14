# Toward FM for NPP — SmolLM2

Official code for **"Agentic Physical AI toward a Domain-Specific Foundation Model for Nuclear Reactor Control"** (Lee et al., 2026).

A compact 360M-parameter language model (SmolLM2) that generates physically valid PWR control-rod commands, validated through closed-loop execution in the KOMODO reactor simulator.

## Overview

- **Backbone**: SmolLM2-360M (HuggingFace)
- **Task**: (P_init, P_target) → 6-parameter rod command `(b1_pos, b1_time, b1_speed, b2_pos, b2_time, b2_speed)`
- **Curriculum**: Phase 1 (grammar, CPT) + Phase 2 (task, LoRA r=32 α=64)
- **Validation**: 2,000 independent closed-loop KOMODO runs per scale
- **Hardware**: trained on a single RTX 3070 (8GB)

## Results

Closed-loop success rate at ±5% tolerance (2,000 runs each):

| Scale | ±1% | ±5% | ±10% | Severe failures (>10%) |
|---|---|---|---|---|
| 1K | 6.0% | 36.9% | 61.5% | 771 |
| 10K | 26.2% | 83.9% | 96.0% | 89 |
| **100K** | **92.0%** | **97.4%** | **100.0%** | **0** |

Scaling from 10K→100K shows super-linear gain (α=1.24 at ±1%) and ~500× variance collapse. Despite balanced 30/30/30/10 training over four actuation families, the 100K model concentrates 76.1% of runtime on `single_b2`.

**Baselines** (±5% on identical 2,000 runs):

| Method | ±5% | Notes |
|---|---|---|
| PID (6-shot tuned) | 43.8% | saturates in large regime |
| Direct LoRA (no Phase 1) | ~0% | mean err >100%, validates curriculum |
| **Proposed (100K)** | **97.4%** | — |

**PyRK transfer** (point kinetics, 10K, same curriculum, zero architectural change): >94% at ±1%.

## Structure

```
.
├── data_generation/        # KOMODO PWR data (1K, 100K)
│   ├── komodo_generate.py
│   ├── komodo_generate_1k.py
│   └── template
├── training/               # Phase 1/2 + Direct LoRA baseline
│   ├── phase1_grammar_{1k,100k}.py
│   ├── phase2_task_{1k,100k}.py
│   └── direct_lora_baseline.py
├── validation/             # Closed-loop + PID
│   ├── komodo_validation_{1k,10k,100k,direct}.py
│   ├── pid_baseline.py
│   └── pid_config.json
├── extend/                 # Variable monitoring window (60-100s)
│   ├── data_generation/{data_generation.py, auto_run.py, template}
│   ├── train_window.py
│   ├── make_predictions.py        # step 1
│   └── validate_predictions.py    # step 2
├── pyrk_transfer/          # PyRK point-kinetics generalization
│   ├── data_generation.py
│   ├── train_pyrk.py
│   └── validate_pyrk.py
├── plotting/
│   ├── plot_main.py
│   └── plot_comparison.py
├── analysis/             # Recomputed entropy / KL divergence (Reviewer 1, Concern 5)
│   ├── recompute_entropy_kl.py
│   ├── entropy_kl_recomputed.json
│   └── validation_results_{1k,10k,100k}_2000cases.json
└── initial_rod_variation/    # Mixed-initialization training and 2-config validation (Reviewer 1, Concern 2)
    ├── run_all.py
    ├── analyze.py
    ├── plot_init_variation.py
    ├── data_generation/{sample_existing,generate_mirrored,combine}.py
    ├── training/{phase1_grammar,phase2_task}_mixed.py
    └── validation/{validate_mixed.py, template_init_var}
```

## Setup

1. Install [KOMODO](https://github.com/imronuke/KOMODO) and either place the `komodo` binary at the repo root or export:
   ```bash
   export KOMODO_EXECUTABLE=/path/to/komodo
   ```
2. Python deps:
   ```bash
   pip install -r requirements.txt
   ```
   PyRK transfer requires `pip install pyrk` separately.

## Usage

```bash
# 1. Data → writes to ./dataset/
python data_generation/komodo_generate.py        # 100K, ~33h @ 12 cores
python data_generation/komodo_generate_1k.py     # 1K

# 2. Train (RTX 3070)
python training/phase1_grammar_100k.py           # ~5-6h
python training/phase2_task_100k.py              # ~10-12h

# 3. Validate (2,000 runs each)
python validation/komodo_validation_100k.py
python validation/pid_baseline.py
python validation/komodo_validation_direct.py    # Direct LoRA ablation

# 4. Plot
python plotting/plot_main.py
python plotting/plot_comparison.py
```

Variable window (Sec. 2.7):
```bash
python extend/data_generation/data_generation.py
python extend/train_window.py
python extend/make_predictions.py
python extend/validate_predictions.py
```

PyRK transfer (Sec. 2.7):
```bash
python pyrk_transfer/data_generation.py
python pyrk_transfer/train_pyrk.py
python pyrk_transfer/validate_pyrk.py
```

Reproduce Supplementary Table S1 (policy entropy and KL divergence from raw runtime histograms — Reviewer 1, Concern 5):
```bash
python analysis/recompute_entropy_kl.py
```

Initial-rod-position variation experiment (Section 2.8 — Reviewer 1, Concern 2):
```bash
cd initial_rod_variation
python run_all.py          # end-to-end pipeline (data + train + validate)
python plot_init_variation.py
```
See `initial_rod_variation/README.md` for step-by-step details and `results.md` for the full analysis.

## Data & Models

Not included. Regenerate via scripts above or contact authors for weights/datasets.

## Citation

```bibtex
@article{lee2025agentic,
  title={Agentic Physical AI toward a Domain-Specific Foundation Model for Nuclear Reactor Control},
  author={Lee, Yoonpyo and Kobayashi, Kazuma and Puppala, Sai and Talukder, Sajedul and Koric, Seid and Chakraborty, Souvik and Alam, Syed Bahauddin},
  journal={arXiv preprint arXiv:2512.23292},
  year={2025}
}
```

## Contact

- Yoonpyo Lee — Hanyang Univ. <yoonpyo2@illinois.edu>
- Syed Bahauddin Alam — UIUC <alams@illinois.edu>

## Acknowledgments

[KOMODO](https://github.com/imronuke/KOMODO) (Imron, 2019), [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) (HuggingFace), Hanyang NPRE, UIUC NPRE.

## License

MIT.
