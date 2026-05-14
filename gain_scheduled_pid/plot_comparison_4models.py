"""
Updated Figure 5 — 4-panel comparison with gain-scheduled PID added.

Same style as the paper's ablation_plot_together.py (Nature 4-panel layout).
Models compared:
  - PID (single gain)         — paper baseline
  - Gain-sched. PID           — strengthened baseline (this revision)
  - Direct (LoRA)             — curriculum ablation
  - Proposed (100K)           — main model
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# Nature style (matches ablation_plot_together.py)
# ============================================================================
plt.rcParams['font.family']      = 'sans-serif'
plt.rcParams['font.sans-serif']  = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size']        = 7
plt.rcParams['axes.labelsize']   = 7
plt.rcParams['axes.titlesize']   = 8
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize']  = 7
plt.rcParams['ytick.labelsize']  = 7
plt.rcParams['legend.fontsize']  = 6
plt.rcParams['figure.dpi']       = 300
plt.rcParams['savefig.dpi']      = 600
plt.rcParams['lines.linewidth']  = 1.0
plt.rcParams['axes.linewidth']   = 0.8
plt.rcParams['pdf.fonttype']     = 42
plt.rcParams['ps.fonttype']      = 42

# 4-model colour palette — PID family in grey, LLM family in blue
MODEL_COLORS = {
    'PID':              '#BDBDBD',  # Light grey
    'Gain-sched. PID':  '#636363',  # Medium grey (strengthened baseline)
    'Direct (LoRA)':    '#6BAED6',  # Light blue
    'Proposed (100K)':  '#08519C',  # Dark blue
}
EDGE_COLORS = {
    'PID':              '#737373',
    'Gain-sched. PID':  '#252525',
    'Direct (LoRA)':    '#3182BD',
    'Proposed (100K)':  '#08306B',
}
STAT_COLORS = {
    'threshold_5':  '#D55E00',
    'threshold_10': '#CC79A7',
}

# ============================================================================
# Paths
# ============================================================================
HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# Set these paths to your local data layout before running.
# Each entry points to the closed-loop validation results (JSON) for one model.
# Each JSON is a list of per-case dicts with at least the keys:
#   target_initial, target_final, actual_initial, actual_final,
#   final_error, validation_success_5
#
# The Gain-scheduled PID path defaults to the runs/ folder produced by
# `pid_scheduled.py --validate`; the other three (single-gain PID, Direct LoRA,
# Proposed 100K) come from the main paper's validation outputs and are not
# bundled with this repo --- replace the placeholders below with your paths.
# ----------------------------------------------------------------------------
FILES = {
    'PID':              Path('PATH/TO/pid_final_results.json'),
    'Gain-sched. PID':  HERE / 'runs' / 'pid_scheduled_results_latest.json',
    'Direct (LoRA)':    Path('PATH/TO/results_2000_only_LoRA.json'),
    'Proposed (100K)':  Path('PATH/TO/validation_results_100k_2000cases.json'),
}

OUTPUT_DIR = HERE / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# Data loading
# ============================================================================
def classify_power_bin(target_final):
    if pd.isna(target_final):
        return 'Unknown'
    delta_p = abs(target_final - 1.0)
    if delta_p <= 0.10001:
        return 'Small'
    if delta_p <= 0.30001:
        return 'Medium'
    return 'Large'


def load_data(files_dict):
    dfs = []
    for label, path in files_dict.items():
        path = Path(path)
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = data.get('results', [data])
            df = pd.DataFrame(data)
            df['model'] = label

            # Use relative error (matches paper's definition in eq. 1519):
            #   epsilon = |P_achieved - P_target| / |P_target| * 100 (%)
            if 'actual_final' in df.columns and 'target_final' in df.columns:
                df['pct_error'] = (
                    (df['actual_final'] - df['target_final']).abs()
                    / df['target_final'].abs()
                ) * 100
            elif 'final_error' in df.columns:
                # Fallback: final_error is the absolute diff; convert to relative if target known
                df['pct_error'] = df['final_error'] * 100
            else:
                raise RuntimeError(f"{label}: no error columns found")

            # Use the pre-computed validation_success_5 flag when available
            # (it already uses the correct relative-error definition).
            # NaN / None / False all map to 0; True maps to 1.
            if 'validation_success_5' in df.columns:
                df['is_success_5'] = df['validation_success_5'].apply(
                    lambda x: 1 if x is True else 0
                )
            else:
                df['is_success_5'] = (df['pct_error'] <= 5.0).astype(int)
            if 'target_final' in df.columns:
                df['power_bin'] = df['target_final'].apply(classify_power_bin)

            dfs.append(df)
            print(f"  loaded {label}: {len(df)} cases (success ±5% = {df['is_success_5'].mean()*100:.1f}%)")
        except Exception as e:
            print(f"  FAILED to load {label} ({path}): {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else None


# ============================================================================
# Plotting
# ============================================================================
def setup_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(width=0.8)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.3)


def create_figure(df, output_dir):
    print("Creating 4-model comparison figure...")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    plt.subplots_adjust(wspace=0.3, hspace=0.45, top=0.90)

    models = ['PID', 'Gain-sched. PID', 'Direct (LoRA)', 'Proposed (100K)']
    models = [m for m in models if m in df['model'].unique()]
    short_labels = {
        'PID':              'PID',
        'Gain-sched. PID':  'GS-PID',
        'Direct (LoRA)':    'Direct',
        'Proposed (100K)':  'Proposed',
    }
    short_list = [short_labels[m] for m in models]

    # ------------------------------------------------------------------------
    # Panel a — Overall ±5% success rate
    # ------------------------------------------------------------------------
    ax = axes[0, 0]
    setup_axis(ax)
    rates = df.groupby('model')['is_success_5'].mean() * 100
    rates = rates.reindex(models)
    bars = ax.bar(
        models, rates,
        color=[MODEL_COLORS[m] for m in models],
        edgecolor=[EDGE_COLORS[m] for m in models],
        linewidth=1.0, width=0.6, alpha=0.9,
    )
    ax.set_ylabel('Success rate (±5%) [%]', fontweight='bold')
    ax.set_title('a', loc='left', fontweight='bold', fontsize=9)
    ax.set_ylim(0, 115)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 2,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=6, fontweight='bold')
    ax.set_xticklabels(short_list)

    # ------------------------------------------------------------------------
    # Panel b — Regime-stratified ±5% success rate
    # ------------------------------------------------------------------------
    ax = axes[0, 1]
    setup_axis(ax)
    bin_order = ['Small', 'Medium', 'Large']
    width = 0.2
    x = np.arange(len(bin_order))
    n = len(models)
    centred_offsets = (np.arange(n) - (n - 1) / 2.) * width

    for i, model in enumerate(models):
        subset = df[df['model'] == model]
        rates = []
        for b in bin_order:
            b_data = subset[subset['power_bin'] == b]
            rates.append(b_data['is_success_5'].mean() * 100 if len(b_data) else 0)
        ax.bar(
            x + centred_offsets[i], rates, width, label=model,
            color=MODEL_COLORS[model], edgecolor=EDGE_COLORS[model],
            linewidth=1.0, alpha=0.9,
        )
    ax.set_ylabel('Success rate [%]', fontweight='bold')
    ax.set_xlabel('Power change regime', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{b}\n(±{val}%)' for b, val in zip(bin_order, [10, 30, 50])])
    ax.set_title('b', loc='left', fontweight='bold', fontsize=9)
    ax.set_ylim(0, 115)
    # Legend is placed at figure level (see below), no per-panel legend here.

    # ------------------------------------------------------------------------
    # Panel c — Error distribution (violin + box, log y)
    # ------------------------------------------------------------------------
    ax = axes[1, 0]
    setup_axis(ax)
    plot_data = []
    for model in models:
        subset = df[df['model'] == model].copy()
        subset['pct_error'] = subset['pct_error'].replace(0, 1e-4)
        plot_data.append(subset)
    combined = pd.concat(plot_data)

    sns.violinplot(
        data=combined, x='model', y='pct_error', order=models,
        ax=ax, linewidth=0.8, inner=None, cut=0,
    )
    for i, coll in enumerate(ax.collections):
        if i < len(models):
            coll.set_facecolor(MODEL_COLORS[models[i]])
            coll.set_edgecolor(EDGE_COLORS[models[i]])
            coll.set_alpha(0.3)
    sns.boxplot(
        data=combined, x='model', y='pct_error', order=models,
        ax=ax, width=0.15, showfliers=False,
        boxprops={'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 0.8},
        whiskerprops={'linewidth': 0.8}, capprops={'linewidth': 0.8},
        medianprops={'color': 'black', 'linewidth': 1.0},
    )
    ax.set_yscale('log')
    ax.set_ylabel('Terminal error [%] (log scale)', fontweight='bold')
    ax.set_xlabel('')
    ax.set_xticklabels(short_list)
    ax.set_title('c', loc='left', fontweight='bold', fontsize=9)
    ax.axhline(5.0, color=STAT_COLORS['threshold_5'], linestyle='--', linewidth=0.8, label='±5%')
    ax.legend(loc='upper right', fontsize=6, frameon=False)

    # ------------------------------------------------------------------------
    # Panel d — CDF of error
    # ------------------------------------------------------------------------
    ax = axes[1, 1]
    setup_axis(ax)
    for model in models:
        err = df[df['model'] == model]['pct_error'].dropna()
        s = np.sort(err)
        y = np.arange(len(s)) / float(len(s) - 1) * 100
        ax.plot(s, y, label=model, color=MODEL_COLORS[model], linewidth=1.5)
    ax.set_xscale('log')
    ax.set_xlim(0.01, 100)
    ax.set_ylim(0, 105)
    ax.axvline(5.0, color=STAT_COLORS['threshold_5'], linestyle='--', linewidth=0.8, label='±5% limit')
    ax.set_xlabel('Terminal error [%]', fontweight='bold')
    ax.set_ylabel('Cumulative probability [%]', fontweight='bold')
    ax.set_title('d', loc='left', fontweight='bold', fontsize=9)
    # Only the threshold line legend remains on panel d
    th_handles = [h for h in ax.get_lines() if h.get_label() == '±5% limit']
    if th_handles:
        ax.legend(handles=th_handles, loc='lower right', fontsize=6, frameon=False)

    # ------------------------------------------------------------------------
    # Figure-level legend for the four models (top center, outside subplots)
    # ------------------------------------------------------------------------
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(facecolor=MODEL_COLORS[m], edgecolor=EDGE_COLORS[m],
                       linewidth=1.0, alpha=0.9, label=m)
        for m in models
    ]
    fig.legend(handles=legend_handles, loc='upper center',
               bbox_to_anchor=(0.5, 0.97), ncol=len(models),
               fontsize=7, frameon=False, columnspacing=2.0)

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------
    out_pdf = output_dir / 'Figure_Comparison_Combined.pdf'
    out_png = output_dir / 'Figure_Comparison_Combined.png'
    plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    plt.savefig(out_png, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"  saved: {out_pdf}")
    print(f"  saved: {out_png}")
    plt.close()


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print(" 4-model comparison figure (with gain-scheduled PID)")
    print("=" * 60)
    df = load_data(FILES)
    if df is None:
        print("Failed to load any data")
        return
    create_figure(df, OUTPUT_DIR)

    print("\nSummary statistics:")
    stats = df.groupby('model')['pct_error'].describe(percentiles=[.5, .95, .99])
    print(stats[['mean', '50%', '95%', 'max']])


if __name__ == "__main__":
    main()
