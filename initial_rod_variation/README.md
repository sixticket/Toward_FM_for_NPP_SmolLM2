# Exp 1: Initial Rod Position Variation (revised design)

Addresses **Reviewer 1, Concern #2**:
> A straightforward test would be to vary the initial rod positions and report whether the same preference persists, reverses, or breaks down.

## Why a re-train (not just re-validate)

Paper's 100K model has no `init` in its prompt. Just changing simulator init
without re-training tests OOD execution, not whether the single_b2 preference
is policy or init-bias. The model can't even see init, so its output
distribution would be identical regardless of new init.

To answer the reviewer's question, we **re-train on a mixed-init dataset
with init in the prompt**, then evaluate at multiple inits and observe whether
pattern preferences invert.

## Design

### Dataset (50K + 50K = 100K, paper-equivalent scale)

| Subset | Init (B1, B2) | Source |
|---|---|---|
| default | (180, 100) | Stratified sample of 50K rows from `My/dataset/master_dataset_100K.csv` (paper data, untouched). 60/30/10 actuation balance preserved. |
| mirror  | (100, 180) | New 50K KOMODO sims at the mirrored init. Sampling logic mirrors `auto_run_minimal.py` with B1/B2 ranges and speeds **swapped** so that the "fine bank in steep gradient" structural role is preserved. |

Both subsets get `init_b1`, `init_b2` columns; combined and shuffled into
`master_dataset_mixed_init_100k.csv`.

### Prompt format change

| | Old (paper) | New (mixed) |
|---|---|---|
| Phase 1 | `[b1_pos, b1_time, b1_speed, b2_pos, b2_time, b2_speed]` | `[init_b1, init_b2, b1_pos, b1_time, b1_speed, b2_pos, b2_time, b2_speed]` |
| Phase 2 | `[init_p, target_p, b1_pos, ..., b2_speed]` (mask 2) | `[init_b1, init_b2, init_p, target_p, b1_pos, ..., b2_speed]` (mask 4) |

Architecture (SmolLM2-360M), LoRA (r=32, alpha=64, q/k/v/o_proj),
optimizer/scheduler all identical to paper's 100K Phase 2.

### Validation
Same 2,000-case test set as paper's 100K validation, run twice — once at
`(180, 100)` and once at `(100, 180)` — through the **same** mixed-trained
model.

## Files

```
exp1_initial_rod_variation/
├── README.md                           (this file)
├── data_generation/
│   ├── sample_existing.py              50K stratified sample from existing 100K CSV
│   ├── generate_mirrored.py            50K new sims at init=(100,180), mirror logic
│   └── combine.py                      merge into master_dataset_mixed_init_100k.csv
├── training/
│   ├── phase1_grammar_mixed.py         CPT, 8-number format with init at front
│   └── phase2_task_mixed.py            LoRA, 10-number format with init at front
├── validation/
│   ├── template_init_var               KOMODO template with {init_b1}/{init_b2}
│   └── validate_mixed.py               2,000 cases × 2 configs through mixed model
└── analyze.py                          actuation pattern compare across configs
```

## Run order

### Easy mode: orchestrator
Set `PY` to your local Python (the venv with `torch`, `transformers`, `peft`)
and `EXP` to your local checkout of this folder, then:
```bash
PY=/path/to/your/venv/bin/python
EXP=/path/to/initial_rod_variation

$PY $EXP/run_all.py                       # full pipeline, auto-skips existing outputs
$PY $EXP/run_all.py --from-step 4         # resume at Phase 1 training
$PY $EXP/run_all.py --only 6              # just re-run validation
$PY $EXP/run_all.py --num-cases 50        # smoke (50 cases instead of 2000)
$PY $EXP/run_all.py --no-skip-existing    # force rerun everything
```

`run_all.py` streams each subprocess's output live (so internal tqdm bars
from KOMODO sim, transformers Trainer, validation cases all stay visible)
AND copies it to `logs/step{N}_<name>.log`. Outer tqdm tracks pipeline progress.

### Manual mode (run individual stages)
```bash
# 1. Build mixed-init dataset (~3h: mostly the 50K mirror sims)
cd $EXP/data_generation
$PY sample_existing.py
$PY generate_mirrored.py --workers 12
$PY combine.py

# 2. Train (~16h on RTX 3070)
cd $EXP/training
$PY phase1_grammar_mixed.py
$PY phase2_task_mixed.py

# 3. Validate at both inits (~6h: 4,000 model+sim cases)
cd $EXP/validation
$PY validate_mixed.py

# 4. Analyze
cd $EXP
$PY analyze.py
```

Smoke test (no training, just generate one mirror sample):
```bash
$PY $EXP/data_generation/generate_mirrored.py --n 100 --workers 4
```

## Total wallclock

~24h end-to-end (mostly Phase 1 + Phase 2 training and the 50K mirror sims).
Fits within the 4-week revision window with margin.

## Expected outcome

| MIXED@(180,100) | MIXED@(100,180) | Interpretation |
|---|---|---|
| single_b2 wins | single_b1 wins | **adaptive policy** — strongest defense of the agentic claim |
| single_b2 wins | single_b2 wins | **hardwired bias** — paper must reframe ("policy is init-coupled") |
| mixed/uniform | mixed/uniform | model defaulted to spread; fall back to data-distribution explanation |

Whichever outcome we get, it's an honest answer to Reviewer 1's exact question.
