# Toward Foundation Models for Nuclear Engineering Control Systems

SmolLM2-360M based nuclear reactor control rod parameter prediction model.

## 📖 Paper

**Title:** Toward Foundation Models for Nuclear Engineering Control Systems

**Authors:** Yoonpyo Lee (Hanyang University), Syed Bahauddin Alam (UIUC)

**Status:** Under Review

## 🎯 Overview

- **Model:** SmolLM2-360M (360M parameters)
- **Task:** Predict 6-parameter control vector from power change `(initial_power, target_power)`
- **Training:** Two-phase curriculum learning
- **Performance:** 
  - 10K model: 83.9% accuracy (±5% tolerance)
  - 100K model: 97.4% accuracy (±5% tolerance)

## 📁 Repository Structure
```
.
├── auto_run_parallel.py              # Parallel data generation (100K scenarios)
├── training/
│   ├── smollm2_phase1_unsupervised_100k.py
│   └── smollm2_phase2_supervised_lora_100k.py
└── validation/
    └── validation_with_simulator_100k.py
```

## 🚀 Quick Start

### Prerequisites

1. **KOMODO Simulator** (required)
```bash
   git clone https://github.com/imronuke/KOMODO
   cd KOMODO
   # Follow KOMODO installation instructions
```
   
   **Reference:** Imron, M. (2019). Development and verification of open reactor simulator ADPRES. *Annals of Nuclear Energy*, 133, 580-588.

2. **Python Environment**
```bash
   pip install torch transformers peft accelerate
   pip install numpy pandas scikit-learn matplotlib tqdm
```

### Usage

#### 1. Generate Training Data
```bash
# Generate 100K synthetic scenarios (takes ~33 hours on 12 cores)
python auto_run_parallel.py
```

**Output:** Dataset with diverse control scenarios
- Single bank operations (60%)
- Simultaneous dual bank (30%)
- Sequential dual bank (10%)

#### 2. Train Phase 1: Unsupervised Learning
```bash
# For 10K dataset (~1 hour on RTX 3070)
python training/smollm2_phase1_unsupervised_10k.py

# For 100K dataset (~5-6 hours on RTX 3070)
python training/smollm2_phase1_unsupervised_100k.py
```

**Goal:** Learn numeric grammar of control parameters

#### 3. Train Phase 2: Supervised LoRA Fine-tuning
```bash
# For 10K dataset (~2-3 hours)
python training/smollm2_phase2_supervised_lora_10k.py

# For 100K dataset (~10-12 hours)
python training/smollm2_phase2_supervised_lora_100k.py
```

**Goal:** Map (initial_power, target_power) → control vector

#### 4. Validate with Simulator
```bash
# Validate 10K model (2,000 independent runs)
python validation/validation_with_simulator_10k.py

# Validate 100K model (2,000 independent runs)
python validation/validation_with_simulator_100k.py
```

## 📊 Results

### Validation Performance (2,000 simulator runs each)

| Model | Training Data | Parsing | ±5% Success | ±10% Success | MAE |
|-------|---------------|---------|-------------|--------------|-----|
| 10K   | 10,000       | 100%    | 83.9%       | 96.0%        | 0.0329 |
| 100K  | 100,000      | 100%    | **97.4%**   | **100.0%**   | **0.0061** |

### Key Findings

- **Scaling Effect:** 10× data → +13.5%p accuracy improvement
- **Error Reduction:** Maximum error decreased from 0.42 to 0.10
- **Actuation Patterns:** Models prefer single-bank solutions (especially Bank 2)
- **Stability:** 100K model shows zero failures at ±10% tolerance

## 🔬 Method

### Input Format
```python
[initial_power, target_power, b1_pos, b1_time, b1_speed, b2_pos, b2_time, b2_speed]
# Example: [1.0, 1.5, 180, 0.0, 0.0, 113, 14.2, 0.6]
```

### Two-Phase Training Strategy

**Phase 1: Unsupervised CLM**
- Learns grammar of control parameters (6 numbers)
- No power values in training
- Full fine-tuning on SmolLM2-360M

**Phase 2: Supervised LoRA**
- Maps power change to control vector
- LoRA adapters (r=32, α=64)
- Freezes Phase 1 weights

## 💾 Data & Models

### Training Data
- **Full 100K dataset:** Available upon request
- **Generation:** Use `auto_run_parallel.py` (requires KOMODO)
- **Time:** ~33 hours on 12-core CPU

### Pre-trained Models
- Models not included due to size (~1.4GB each)
- Train using provided scripts or contact authors for weights

## 📄 Citation
```bibtex
@article{lee2025toward,
  title={Toward Foundation Models for Nuclear Engineering Control Systems},
  author={Lee, Yoonpyo and Alam, Syed Bahauddin},
  journal={Under Review},
  year={2025}
}
```

## 🙏 Acknowledgments

- **KOMODO Simulator:** Imron, M. (2019) - https://github.com/imronuke/KOMODO
- **SmolLM2 Model:** HuggingFace Team
- Hanyang University Nuclear Engineering Department
- University of Illinois Urbana-Champaign (NPRE)

## 📧 Contact

- **Yoonpyo Lee:** Hanyang University lukeyounpyo@hanyang.ac.kr)
- **Syed Bahauddin Alam:** UIUC (alams@illinois.edu)

## 📝 License

MIT License - See LICENSE file for details

---

**Note:** This repository contains training and validation code only. KOMODO simulator must be installed separately. Trained models and large datasets are not included but can be generated using the provided scripts.
