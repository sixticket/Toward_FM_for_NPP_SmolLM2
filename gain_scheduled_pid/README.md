# Exp 2 — Gain-Scheduled PID Baseline

Stronger classical baseline requested by **Reviewer 1 #4** ("PID is straw man") and **Reviewer 2 #6** ("compare with stronger control architectures"). The supervisor's response letter draft commits to this specific baseline.

## What's different from the paper PID

The paper baseline is a single-gain proportional controller (Kp=35.5 from 6-shot symmetric calibration). This implementation differs in **exactly two ways**, and in only those:

1. **Six regime-dependent Kps** instead of one global Kp. Regimes split by power-change magnitude (small/medium/large) × direction (increase/decrease).
2. **Wider calibration sweep** (~33 KOMODO sims spanning rod displacement [-80, +80] step 5) so each regime gets a meaningful linear fit. Original PID uses 6 symmetric points.

Everything else matches paper PID:
- B2 is the primary actuator; B1 is engaged only on B2 saturation
- Rod motion at fixed `DEFAULT_SPEED = 2.0` steps/s
- `b_time = dist/speed` convention (mirrors `My/PID/pid_validation.py`)
- Same 2,000-case test set (seed=42), same KOMODO closed-loop protocol
- Same tolerance bands (±1, 2, 3, 5, 10%)

## Why this is the right "stronger baseline"

- **Industry standard**: gain scheduling is the textbook first step beyond fixed-gain PID for nonlinear plants.
- **Same architecture as paper PID**: makes the comparison clean — gap (single-gain → scheduled PID) → LLM is interpretable, not confounded by additional banks/horizons/feedback.
- **Falsifiable hypothesis**: if gain scheduling closes most of the gap to LLM, it indicates the paper's headline gap (LLM 97.4% vs PID 43.8%) was largely a tuning artifact. If it closes only modestly, the gap reflects a real model-class limit of feedback control.

## Run

From this folder, with the project Python environment active:
```bash
python pid_scheduled.py --calibrate --validate --workers 12
```

Combined timing on the paper's hardware (12-core CPU + RTX 3070):
- Calibration: ~33 sims × ~5s = **~3 min**
- Validation: 2,000 sims / 12 workers ≈ **~12-15 min**
- **Total ~15-20 min**

Smoke test (faster):
```bash
python pid_scheduled.py --calibrate --validate --num_cases 100
```

Resume after interruption:
```bash
# JSONL preserves progress; just rerun. To force fresh start:
python pid_scheduled.py --validate --restart
```

## Output

- `calibration_config.json` — per-regime Kp + R² + sample counts (cached)
- `runs/pid_scheduled_results_partial.jsonl` — incremental save (one row per case)
- `runs/pid_scheduled_results_2000cases_<ts>.json` — final consolidated JSON
- `runs/pid_scheduled_results_latest.json` — fixed-name copy for downstream plotting (same schema as paper validation)

The validate command prints a full summary at the end (per-tolerance success, error stats, magnitude/direction strata, per-regime success with Kp shown).

## Expected outcome

| Method | ±5% success | Mean err |
|---|---|---|
| Paper single-gain PID | 43.8% | 6.34% |
| **Gain-scheduled PID (this work)** | **?** | **?** |
| LLM 100K (paper) | 97.4% | 0.61% |

Reasonable expectation: ±5% in the **55-75%** range. Higher would suggest gain scheduling alone closes most of the LLM gap (negative for paper narrative). Lower or comparable to single-gain would suggest the regime-classification doesn't help much — feedback control hits a ceiling regardless of tuning sophistication (positive for paper narrative).

Numbers go into the response letter's Concern 4 / Comment 6 placeholder paragraph.

## Files

```
exp2_scheduled_pid/
├── README.md                       (this file)
├── pid_scheduled.py                main script: --calibrate / --validate
├── calibration_config.json         (created by --calibrate)
└── runs/                           (created by --validate)
    ├── pid_scheduled_results_partial.jsonl
    ├── pid_scheduled_results_<n>cases_<ts>.json
    └── pid_scheduled_results_latest.json
```
