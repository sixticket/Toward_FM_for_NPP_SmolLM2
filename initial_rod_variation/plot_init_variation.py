"""
Plot for Initial Rod Position Variation experiment (Reviewer 1, Concern 2).

Two-panel figure following the paper's Figure 4 style:
  (a) Runtime actuation pattern shift: Default vs Mirror initialization
      (4 actuation classes, grouped bars).
  (b) Closed-loop success rate by tolerance: Default vs Mirror
      (3 tolerance bands, grouped bars).

Default config (B1=180, B2=100)  -> paper-blue (NATURE_COLORS['100K'])
Mirror  config (B1=100, B2=180)  -> ORANGE_COLORS['training'] for contrast

Inputs:
  validation/runs/default_b1_180_b2_100/results_latest.json
  validation/runs/mirror_b1_100_b2_180/results_latest.json

Outputs (in ./figures/):
  Figure_init_variation.{png, pdf, svg}
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Style (mirrors statistic_results/code/plot.py)
# ----------------------------------------------------------------------------
plt.rcParams['font.family']      = 'Arial'
plt.rcParams['font.size']        = 7
plt.rcParams['axes.labelsize']   = 7
plt.rcParams['axes.titlesize']   = 8
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize']  = 7
plt.rcParams['ytick.labelsize']  = 7
plt.rcParams['legend.fontsize']  = 7
plt.rcParams['figure.dpi']       = 300
plt.rcParams['savefig.dpi']      = 600
plt.rcParams['pdf.fonttype']     = 42
plt.rcParams['ps.fonttype']      = 42

NATURE_COLORS = {
    '1K':       '#DEEBF7',
    '10K':      '#9ECAE1',
    '100K':     '#4292C6',
    'emphasis': '#08519C',
}
EDGE_COLORS = {
    '1K':       '#9ECAE1',
    '10K':      '#4292C6',
    '100K':     '#2171B5',
    'emphasis': '#08519C',
}
ORANGE_COLORS = {
    'design':   '#FDB863',
    'training': '#E66101',
}
ORANGE_EDGES = {
    'design':   '#E66101',
    'training': '#B35806',
}

DEFAULT_FACE = NATURE_COLORS['100K']
DEFAULT_EDGE = EDGE_COLORS['100K']
MIRROR_FACE  = ORANGE_COLORS['training']
MIRROR_EDGE  = ORANGE_EDGES['training']

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
EXP_DIR = Path(__file__).resolve().parent
DEFAULT_JSON = EXP_DIR / "validation" / "runs" / "default_b1_180_b2_100" / "results_latest.json"
MIRROR_JSON  = EXP_DIR / "validation" / "runs" / "mirror_b1_100_b2_180"  / "results_latest.json"
OUT_DIR = EXP_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)

CLASSES      = ['single_b1', 'single_b2', 'simultaneous', 'sequential']
CLASS_LABELS = ['Single B1', 'Single B2', 'Simultaneous', 'Sequential']
TOLS         = [1, 5, 10]
TOL_LABELS   = [r'$\pm1\%$', r'$\pm5\%$', r'$\pm10\%$']


def classify(params):
    """Match classify_scenario_type in plot.py."""
    if not isinstance(params, dict) or not params:
        return None
    b1_t = params.get('b1_time', 0)
    b2_t = params.get('b2_time', 0)
    b1_active = b1_t > 0
    b2_active = b2_t > 0
    if b1_active and not b2_active:
        return 'single_b1'
    if b2_active and not b1_active:
        return 'single_b2'
    if b1_active and b2_active:
        if abs(b1_t - b2_t) < 0.01:
            return 'simultaneous'
        return 'sequential'
    return None


def summarize(json_path, label):
    with open(json_path) as f:
        records = json.load(f)
    n = len(records)
    counts = {c: 0 for c in CLASSES}
    success = {t: 0 for t in TOLS}
    for rec in tqdm(records, desc=f"  {label:<8s}"):
        cls = classify(rec.get('predicted_params'))
        if cls in counts:
            counts[cls] += 1
        for t in TOLS:
            if rec.get(f'validation_success_{t}', False):
                success[t] += 1
    return {
        'n':       n,
        'counts':  counts,
        'pct':     [counts[c] / n * 100 for c in CLASSES],
        'success': [success[t] / n * 100 for t in TOLS],
    }


def main():
    print("Loading validation results...")
    default = summarize(DEFAULT_JSON, "Default")
    mirror  = summarize(MIRROR_JSON,  "Mirror")

    print()
    print(f"{'Class':<14s}  {'Default':>10s}  {'Mirror':>10s}")
    for c, lbl in zip(CLASSES, CLASS_LABELS):
        print(f"  {lbl:<12s}  {default['pct'][CLASSES.index(c)]:>9.2f}%  {mirror['pct'][CLASSES.index(c)]:>9.2f}%")
    print()
    for t, p_d, p_m in zip(TOLS, default['success'], mirror['success']):
        print(f"  S(±{t}%):     {p_d:>9.2f}%  {p_m:>9.2f}%")
    print()

    # ------------------------------------------------------------------------
    # Figure: 1x2 panel (matches paper Fig 4 layout)
    # ------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.8))
    width = 0.35

    # Panel a: actuation pattern shift
    ax = axes[0]
    x = np.arange(len(CLASSES))
    bars_d = ax.bar(x - width/2, default['pct'], width,
                    color=DEFAULT_FACE, edgecolor=DEFAULT_EDGE,
                    linewidth=1.0, label='Default (B1=180, B2=100)', alpha=0.85)
    bars_m = ax.bar(x + width/2, mirror['pct'], width,
                    color=MIRROR_FACE, edgecolor=MIRROR_EDGE,
                    linewidth=1.0, label='Mirror (B1=100, B2=180)', alpha=0.85)
    for bars, vals in zip([bars_d, bars_m], [default['pct'], mirror['pct']]):
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 1,
                        f'{v:.1f}%', ha='center', va='bottom', fontsize=6)
    ax.set_ylabel('Runtime actuation frequency (%)', fontweight='bold', fontsize=7)
    ax.set_xlabel('Actuation pattern', fontweight='bold', fontsize=7)
    ax.set_title('a', loc='left', fontweight='bold', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LABELS, fontsize=7)
    ax.set_ylim(0, 95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.0)
    for s in ax.spines.values():
        s.set_linewidth(1.0)

    # Panel b: success rate by tolerance
    ax = axes[1]
    x = np.arange(len(TOLS))
    bars_d2 = ax.bar(x - width/2, default['success'], width,
                     color=DEFAULT_FACE, edgecolor=DEFAULT_EDGE,
                     linewidth=1.0, alpha=0.85)
    bars_m2 = ax.bar(x + width/2, mirror['success'], width,
                     color=MIRROR_FACE, edgecolor=MIRROR_EDGE,
                     linewidth=1.0, alpha=0.85)
    for bars, vals in zip([bars_d2, bars_m2], [default['success'], mirror['success']]):
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 1,
                        f'{v:.1f}%', ha='center', va='bottom', fontsize=6)
    ax.set_ylabel('Closed-loop success rate (%)', fontweight='bold', fontsize=7)
    ax.set_xlabel('Tolerance band', fontweight='bold', fontsize=7)
    ax.set_title('b', loc='left', fontweight='bold', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(TOL_LABELS, fontsize=7)
    ax.set_ylim(0, 115)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.0)
    for s in ax.spines.values():
        s.set_linewidth(1.0)

    # Figure-level legend (top center, outside subplots)
    handles = [bars_d, bars_m]
    labels  = ['Default (B1=180, B2=100)', 'Mirror (B1=100, B2=180)']
    fig.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(0.5, 1.04), ncol=2,
               fontsize=7, frameon=False)

    plt.tight_layout()

    out = OUT_DIR / "varied_init_pattern.pdf"
    plt.savefig(out, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"  saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
