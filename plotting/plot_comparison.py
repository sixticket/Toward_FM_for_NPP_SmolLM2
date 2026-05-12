import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# ============================================================================
# 1. NATURE STYLE CONFIGURATION
# ============================================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 7
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 6
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['axes.linewidth'] = 0.8

# Color Palette
MODEL_COLORS = {
    'PID Control': '#999999',      # Grey
    'Direct (LoRA)': '#E69F00',    # Orange
    'Proposed (100K)': '#009E73'   # Bluish Green
}

EDGE_COLORS = {
    'PID Control': '#4D4D4D',
    'Direct (LoRA)': '#A65628',
    'Proposed (100K)': '#005A32'
}

# File Paths (validation 스크립트 결과물 위치 기준)
SCRIPT_DIR = Path(__file__).resolve().parent  # plotting/
REPO_ROOT  = SCRIPT_DIR.parent                # repo root
VALIDATION_DIR = REPO_ROOT / "validation"

FILES = {
    'PID Control':     str(VALIDATION_DIR / "validation_runs_pid_final" / "pid_final_results.json"),
    'Direct (LoRA)':   str(VALIDATION_DIR / "validation_runs_direct_model" / "results_2000cases_only_LoRA.json"),
    'Proposed (100K)': str(VALIDATION_DIR / "validation_runs_v7_simple" / "validation_results_v7_simple_2000cases_100K.json"),
}

# ============================================================================
# 2. DATA LOADING & PROCESSING
# ============================================================================
def classify_power_bin(target_final):
    if pd.isna(target_final): return 'Unknown'
    # Target Change from 1.0
    delta_p = abs(target_final - 1.0)
    if delta_p <= 0.10001: return 'Small'
    elif delta_p <= 0.30001: return 'Medium'
    else: return 'Large'

def load_data(files_dict):
    dfs = []
    for label, path in files_dict.items():
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            # 리스트/딕셔너리 구조 처리
            if isinstance(data, dict):
                data = data.get('results', [data])

            df = pd.DataFrame(data)
            df['model'] = label

            # [수정됨] 백분율 에러 계산 (Absolute Difference * 100)
            # 이유: 초기 출력이 1.0이므로 차이 자체가 % 단위임 (0.01차이 = 1%)
            if 'actual_final' in df.columns and 'target_final' in df.columns:
                df['pct_error'] = abs(df['actual_final'] - df['target_final']) * 100
            elif 'final_error' in df.columns:
                # final_error가 이미 |Actual - Target|이라고 가정
                df['pct_error'] = df['final_error'] * 100
            else:
                print(f"⚠️ Warning: Cannot calculate error for {label}")
                continue

            # 성공 여부 (5% 이내)
            df['is_success_5'] = (df['pct_error'] <= 5.0).astype(int)

            # Power Bin 분류
            if 'target_final' in df.columns:
                df['power_bin'] = df['target_final'].apply(classify_power_bin)

            dfs.append(df)
            print(f"  ✓ Loaded {label}: {len(df)} cases")

        except Exception as e:
            print(f"  ✗ Error loading {label}: {e}")

    return pd.concat(dfs, ignore_index=True) if dfs else None

# ============================================================================
# 3. PLOTTING FUNCTIONS
# ============================================================================

def setup_axis(ax):
    """Apply Nature style spines and ticks"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(width=0.8)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.3)

def create_figure_comparison(df):
    """
    Create a 4-panel figure comparing the models.
    """
    print("Creating Comparison Figure...")

    # Figure Size (Width ~180mm for double column)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    # Model Order
    models = ['PID Control', 'Direct (LoRA)', 'Proposed (100K)']
    models = [m for m in models if m in df['model'].unique()]

    # ------------------------------------------------------------------------
    # Panel A: Overall Success Rate
    # ------------------------------------------------------------------------
    ax = axes[0, 0]
    setup_axis(ax)

    success_rates = df.groupby('model')['is_success_5'].mean() * 100
    success_rates = success_rates.reindex(models)

    bars = ax.bar(models, success_rates,
                  color=[MODEL_COLORS[m] for m in models],
                  edgecolor=[EDGE_COLORS[m] for m in models],
                  linewidth=1.0, width=0.6, alpha=0.9)

    ax.set_ylabel('Success rate (±5%) [%]', fontweight='bold')
    ax.set_title('a', loc='left', fontweight='bold', fontsize=9)
    ax.set_ylim(0, 115)

    # Text labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=6, fontweight='bold')

    # Shorten x-labels
    short_labels = [m.replace(' Control', '').replace(' (LoRA)', '').replace(' (100K)', '') for m in models]
    ax.set_xticklabels(short_labels)

    # ------------------------------------------------------------------------
    # Panel B: Success Rate by Regime
    # ------------------------------------------------------------------------
    ax = axes[0, 1]
    setup_axis(ax)

    bin_order = ['Small', 'Medium', 'Large']
    width = 0.25
    x = np.arange(len(bin_order))

    for i, model in enumerate(models):
        subset = df[df['model'] == model]
        rates = []
        for b in bin_order:
            b_data = subset[subset['power_bin'] == b]
            if len(b_data) > 0:
                rates.append(b_data['is_success_5'].mean() * 100)
            else:
                rates.append(0)

        offset = (i - 1) * width
        ax.bar(x + offset, rates, width, label=model,
               color=MODEL_COLORS[model], edgecolor=EDGE_COLORS[model],
               linewidth=1.0, alpha=0.9)

    ax.set_ylabel('Success rate [%]', fontweight='bold')
    ax.set_xlabel('Power change regime', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{b}\n(±{val}%)' for b, val in zip(bin_order, [10, 30, 50])])
    ax.set_title('b', loc='left', fontweight='bold', fontsize=9)
    ax.legend(loc='best', fontsize=5, frameon=False)
    ax.set_ylim(0, 115)

    # ------------------------------------------------------------------------
    # Panel C: Error Distribution (Violin)
    # ------------------------------------------------------------------------
    ax = axes[1, 0]
    setup_axis(ax)

    # Log scale data preparation (replace 0 with small value for log scale)
    plot_data = []
    for model in models:
        subset = df[df['model'] == model].copy()
        # Log scale visualization requires non-zero. Replace 0 error with 0.001%
        subset['log_error'] = subset['pct_error'].replace(0, 1e-3)
        plot_data.append(subset)

    combined_plot_df = pd.concat(plot_data)

    # Violin Plot
    parts = sns.violinplot(data=combined_plot_df, x='model', y='pct_error', order=models,
                           ax=ax, linewidth=0.8, inner=None, cut=0)

    # Set colors manually
    for i, collection in enumerate(ax.collections):
        if i < len(models):
            collection.set_facecolor(MODEL_COLORS[models[i]])
            collection.set_edgecolor(EDGE_COLORS[models[i]])
            collection.set_alpha(0.3)

    # Box Plot Overlay
    sns.boxplot(data=combined_plot_df, x='model', y='pct_error', order=models,
                ax=ax, width=0.15, showfliers=False,
                boxprops={'facecolor': 'none', 'edgecolor': 'black', 'linewidth': 0.8},
                whiskerprops={'linewidth': 0.8}, capprops={'linewidth': 0.8},
                medianprops={'color': 'black', 'linewidth': 1.0})

    ax.set_yscale('log')
    ax.set_ylabel('Terminal error [%] (Log scale)', fontweight='bold')
    ax.set_xlabel('')
    ax.set_xticklabels(short_labels)
    ax.set_title('c', loc='left', fontweight='bold', fontsize=9)

    # Threshold lines
    ax.axhline(5.0, color='#D55E00', linestyle='--', linewidth=0.8, label='±5%')
    ax.legend(loc='upper right', fontsize=6, frameon=False)

    # ------------------------------------------------------------------------
    # Panel D: Tail Risk (CDF)
    # ------------------------------------------------------------------------
    ax = axes[1, 1]
    setup_axis(ax)

    for model in models:
        subset = df[df['model'] == model]['pct_error'].dropna()
        sorted_data = np.sort(subset)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1) * 100

        ax.plot(sorted_data, yvals, label=model,
                color=MODEL_COLORS[model], linewidth=1.5)

    ax.set_xscale('log')
    ax.set_xlim(0.01, 100)
    ax.set_ylim(0, 105)

    ax.axvline(5.0, color='#D55E00', linestyle='--', linewidth=0.8, label='±5% limit')

    ax.set_xlabel('Terminal error [%]', fontweight='bold')
    ax.set_ylabel('Cumulative probability [%]', fontweight='bold')
    ax.set_title('d', loc='left', fontweight='bold', fontsize=9)

    # Legend
    ax.legend(loc='lower right', fontsize=6, frameon=False)

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------
    plt.tight_layout()
    plt.savefig('Figure_Comparison_3Models.png', dpi=600, bbox_inches='tight', facecolor='white')
    print("✓ Saved Figure to 'Figure_Comparison_3Models.png'")
    # plt.show()

# ============================================================================
# 4. EXECUTION
# ============================================================================
if __name__ == "__main__":
    df = load_data(FILES)
    if df is not None:
        create_figure_comparison(df)

        # 간단한 통계 출력
        print("\n=== Summary Statistics (Percent Error) ===")
        stats = df.groupby('model')['pct_error'].describe(percentiles=[.5, .95])
        print(stats[['mean', '50%', '95%', 'max']])