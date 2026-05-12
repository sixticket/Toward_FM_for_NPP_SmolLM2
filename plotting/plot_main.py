import pandas as pd
import numpy as np
import json
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
import sys
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_validation_results(results, output_dir, model_name='1K'):
    """Visualize validation results"""
    print("\n" + "=" * 80)
    print(f"📊 Creating visualization... ({model_name})")
    print("=" * 80)

    # Filter successful cases only
    valid_results = [r for r in results if r.get('simulation_success', False)]

    if not valid_results:
        print("⚠️ No data to visualize.")
        return

    # Prepare data
    target_changes = [(r['target_final'] - r['target_initial']) / r['target_initial'] * 100
                      for r in valid_results]
    final_errors = [r['final_error'] for r in valid_results]
    success_5 = [r.get('validation_success_5', False) for r in valid_results]
    success_10 = [r.get('validation_success_10', False) for r in valid_results]

    # Create figure
    fig = plt.figure(figsize=(20, 12))

    # 1. Error Distribution Histogram
    ax1 = plt.subplot(2, 3, 1)
    n, bins, patches = plt.hist(final_errors, bins=50, edgecolor='black', alpha=0.7, label='Error Distribution')
    mean_line = plt.axvline(np.mean(final_errors), color='r', linestyle='--',
                            label=f'Mean: {np.mean(final_errors):.4f}', linewidth=2)
    median_line = plt.axvline(np.median(final_errors), color='g', linestyle='--',
                              label=f'Median: {np.median(final_errors):.4f}', linewidth=2)
    plt.xlabel('Terminal Power Error', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'V7.1 Simple {model_name}: Error Distribution', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # 2. Target Change vs Error Scatter Plot (✅ LEGEND ADDED)
    ax2 = plt.subplot(2, 3, 2)

    # Separate data by success/failure
    success_changes = [target_changes[i] for i in range(len(target_changes)) if success_5[i]]
    success_errors = [final_errors[i] for i in range(len(final_errors)) if success_5[i]]
    fail_changes = [target_changes[i] for i in range(len(target_changes)) if not success_5[i]]
    fail_errors = [final_errors[i] for i in range(len(final_errors)) if not success_5[i]]

    # Plot with explicit labels
    plt.scatter(success_changes, success_errors, c='green', alpha=0.6, s=50, label='Success (±5%)')
    plt.scatter(fail_changes, fail_errors, c='red', alpha=0.6, s=50, label='Failure (±5%)')

    plt.axhline(0.05, color='orange', linestyle='--', label='±5% Threshold', linewidth=2)
    plt.axhline(0.10, color='yellow', linestyle='--', label='±10% Threshold', linewidth=2)
    plt.xlabel('Target Change (%)', fontsize=12)
    plt.ylabel('Terminal Power Error', fontsize=12)
    plt.title('Target Change vs Error', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # 3. 5-Tier Accuracy Bar Chart
    ax3 = plt.subplot(2, 3, 3)
    success_1 = sum(1 for r in valid_results if r.get('validation_success_1', False))
    success_2 = sum(1 for r in valid_results if r.get('validation_success_2', False))
    success_3 = sum(1 for r in valid_results if r.get('validation_success_3', False))
    success_5_count = sum(1 for r in valid_results if r.get('validation_success_5', False))
    success_10_count = sum(1 for r in valid_results if r.get('validation_success_10', False))

    total = len(valid_results)
    percentages = [
        success_1 / total * 100,
        success_2 / total * 100,
        success_3 / total * 100,
        success_5_count / total * 100,
        success_10_count / total * 100
    ]

    labels = ['±1%', '±2%', '±3%', '±5%', '±10%']
    colors_list = ['#2ecc71', '#27ae60', '#3498db', '#f39c12', '#e74c3c']
    bars = plt.bar(labels, percentages, color=colors_list)

    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.title('5-Tier Accuracy Analysis', fontsize=14, fontweight='bold')
    plt.ylim(0, 105)

    # Display percentages on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Add legend for color meaning
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_list[i], label=labels[i]) for i in range(len(labels))]
    plt.legend(handles=legend_elements, loc='upper right', title='Tolerance Band')
    plt.grid(True, alpha=0.3, axis='y')

    # 4. Success Rate by Power Change Range
    ax4 = plt.subplot(2, 3, 4)
    ranges = {
        'Small\n(±10%)': [r for r in valid_results if abs(r['target_final'] - 1.0) <= 0.1],
        'Medium\n(±30%)': [r for r in valid_results if 0.1 < abs(r['target_final'] - 1.0) <= 0.3],
        'Large\n(±50%)': [r for r in valid_results if abs(r['target_final'] - 1.0) > 0.3],
    }

    range_names = []
    range_success_rates = []
    range_counts = []

    for name, range_results in ranges.items():
        if range_results:
            range_names.append(name)
            success = sum(1 for r in range_results if r.get('validation_success_5', False))
            range_success_rates.append(success / len(range_results) * 100)
            range_counts.append(f"n={len(range_results)}")

    range_colors = ['#3498db', '#9b59b6', '#e74c3c']
    bars = plt.bar(range_names, range_success_rates, color=range_colors)
    plt.ylabel('Success Rate (%) at ±5%', fontsize=12)
    plt.title('Success Rate by Range', fontsize=14, fontweight='bold')
    plt.ylim(0, 105)

    # Display percentages and counts on bars
    for bar, pct, count in zip(bars, range_success_rates, range_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{pct:.1f}%\n{count}', ha='center', va='bottom', fontweight='bold')

    # Add legend
    legend_elements = [Patch(facecolor=range_colors[i], label=range_names[i].replace('\n', ' '))
                       for i in range(len(range_names))]
    plt.legend(handles=legend_elements, loc='upper right', title='Power Change Range')
    plt.grid(True, alpha=0.3, axis='y')

    # 5. Error Box Plot by Range
    ax5 = plt.subplot(2, 3, 5)
    box_data = []
    box_labels = []
    for name, range_results in ranges.items():
        if range_results:
            errors = [r['final_error'] for r in range_results]
            box_data.append(errors)
            box_labels.append(name.replace('\n', ' '))

    if box_data:
        bp = plt.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], range_colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Add boxplot legend
        legend_elements = [
            plt.Line2D([0], [0], color='orange', linewidth=2, label='Median'),
            plt.Line2D([0], [0], color='green', linewidth=2, label='Mean (green line)'),
            Patch(facecolor='lightblue', alpha=0.6, label='IQR (25%-75%)')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

    plt.ylabel('Terminal Power Error', fontsize=12)
    plt.title('Error Distribution by Range', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    # 6. Cumulative Distribution Function (CDF)
    ax6 = plt.subplot(2, 3, 6)
    sorted_errors = np.sort(final_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    plt.plot(sorted_errors, cumulative, linewidth=2, color='#2c3e50', label='CDF')
    plt.axvline(0.05, color='orange', linestyle='--', label='±5% Threshold', linewidth=2)
    plt.axvline(0.10, color='yellow', linestyle='--', label='±10% Threshold', linewidth=2)
    plt.xlabel('Terminal Power Error', fontsize=12)
    plt.ylabel('Cumulative Probability (%)', fontsize=12)
    plt.title('Error Cumulative Distribution Function (CDF)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(f'V7.1 Simple {model_name} Validation Results (n={len(valid_results)})',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = output_dir / f"validation_plot_v7_simple_{model_name.lower()}_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Graph saved: {plot_file}")

    plt.close()


def plot_comparison_all_models(df, output_dir):
    """Compare all models: 1K, 10K, 100K"""
    print("\n📊 Creating model comparison graph...")

    versions = []
    parsing_rates = []
    success_5_rates = []
    success_10_rates = []

    for model_name in sorted(df['model'].unique()):
        model_df = df[df['model'] == model_name]
        valid_results = model_df[model_df['simulation_success'] == True]

        if len(valid_results) == 0:
            continue

        versions.append(f'V7.1\n({model_name})')

        # Parsing
        parsing = (model_df['parsing_success'].sum() / len(model_df) * 100)
        parsing_rates.append(parsing)

        # Validation success rate
        success_5 = (valid_results['validation_success_5'].sum() / len(valid_results) * 100)
        success_5_rates.append(success_5)

        success_10 = (valid_results['validation_success_10'].sum() / len(valid_results) * 100)
        success_10_rates.append(success_10)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['#e74c3c', '#f39c12', '#2ecc71']  # 1K, 10K, 100K
    model_labels = ['1K', '10K', '100K']

    # Parsing Success Rate
    ax1 = axes[0]
    bars1 = ax1.bar(versions, parsing_rates, color=colors[:len(versions)])
    ax1.set_ylabel('Parsing Success Rate (%)', fontsize=12)
    ax1.set_title('Parsing Success Rate Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 105)
    for bar, rate in zip(bars1, parsing_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=f'{model_labels[i]} Model')
                       for i in range(len(versions))]
    ax1.legend(handles=legend_elements, loc='lower right')
    ax1.grid(True, alpha=0.3, axis='y')

    # ±5% Validation Success Rate
    ax2 = axes[1]
    bars2 = ax2.bar(versions, success_5_rates, color=colors[:len(versions)])
    ax2.set_ylabel('Validation Success Rate (%)', fontsize=12)
    ax2.set_title('Validation Success Rate (±5%)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 105)
    for bar, rate in zip(bars2, success_5_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

    legend_elements = [Patch(facecolor=colors[i], label=f'{model_labels[i]} Model')
                       for i in range(len(versions))]
    ax2.legend(handles=legend_elements, loc='lower right')
    ax2.grid(True, alpha=0.3, axis='y')

    # ±10% Validation Success Rate
    ax3 = axes[2]
    bars3 = ax3.bar(versions, success_10_rates, color=colors[:len(versions)])
    ax3.set_ylabel('Validation Success Rate (%)', fontsize=12)
    ax3.set_title('Validation Success Rate (±10%)', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 105)
    for bar, rate in zip(bars3, success_10_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

    legend_elements = [Patch(facecolor=colors[i], label=f'{model_labels[i]} Model')
                       for i in range(len(versions))]
    ax3.legend(handles=legend_elements, loc='lower right')
    ax3.grid(True, alpha=0.3, axis='y')

    fig.suptitle('V7.1 Simple: 1K vs 10K vs 100K Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = output_dir / f"comparison_plot_all_models_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison graph saved: {plot_file}")
    plt.close()


def plot_comparison_all_models(df, output_dir):
    """Compare all models: 1K, 10K, 100K"""
    print("\n📊 Creating model comparison graph...")

    versions = []
    parsing_rates = []
    success_5_rates = []
    success_10_rates = []

    for model_name in sorted(df['model'].unique()):
        model_df = df[df['model'] == model_name]
        valid_results = model_df[model_df['simulation_success'] == True]

        if len(valid_results) == 0:
            continue

        versions.append(f'V7.1\n({model_name})')

        # Parsing
        parsing = (model_df['parsing_success'].sum() / len(model_df) * 100)
        parsing_rates.append(parsing)

        # Validation success rate
        success_5 = (valid_results['validation_success_5'].sum() / len(valid_results) * 100)
        success_5_rates.append(success_5)

        success_10 = (valid_results['validation_success_10'].sum() / len(valid_results) * 100)
        success_10_rates.append(success_10)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['#e74c3c', '#f39c12', '#2ecc71']  # 1K, 10K, 100K

    # Parsing Success Rate
    ax1 = axes[0]
    bars1 = ax1.bar(versions, parsing_rates, color=colors[:len(versions)])
    ax1.set_ylabel('Parsing Success Rate (%)', fontsize=12)
    ax1.set_title('Parsing Success Rate Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 105)
    for bar, rate in zip(bars1, parsing_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # ±5% Validation Success Rate
    ax2 = axes[1]
    bars2 = ax2.bar(versions, success_5_rates, color=colors[:len(versions)])
    ax2.set_ylabel('Validation Success Rate (%)', fontsize=12)
    ax2.set_title('Validation Success Rate (±5%)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 105)
    for bar, rate in zip(bars2, success_5_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # ±10% Validation Success Rate
    ax3 = axes[2]
    bars3 = ax3.bar(versions, success_10_rates, color=colors[:len(versions)])
    ax3.set_ylabel('Validation Success Rate (%)', fontsize=12)
    ax3.set_title('Validation Success Rate (±10%)', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 105)
    for bar, rate in zip(bars3, success_10_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    fig.suptitle('V7.1 Simple: 1K vs 10K vs 100K Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = output_dir / f"comparison_plot_all_models_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison graph saved: {plot_file}")
    plt.close()


# --- Helper Functions for Data Processing ---

def classify_power_bin(target_final):
    """Classify into Small, Medium, Large bins"""
    if pd.isna(target_final):
        return 'Unknown'
    delta_p = abs(target_final - 1.0)

    if delta_p <= 0.1:
        return 'Small'
    elif delta_p <= 0.3:
        return 'Medium'
    else:
        return 'Large'


def classify_scenario_type(params):
    """Classify actuation pattern from predicted parameters"""
    if not isinstance(params, dict) or not params:
        return 'parsing_failure'

    b1_active = params.get('b1_time', 0) > 0
    b2_active = params.get('b2_time', 0) > 0

    if b1_active and not b2_active:
        return 'single_b1'
    elif not b1_active and b2_active:
        return 'single_b2'
    elif b1_active and b2_active:
        if abs(params.get('b1_time', 0) - params.get('b2_time', 0)) < 0.01:
            return 'simultaneous'
        else:
            return 'sequential'
    else:
        return 'none_active'


def load_and_process_data(files_to_load):
    """Load JSON files and create unified DataFrame"""
    dfs_to_concat = []

    for model_name, file_path in files_to_load.items():
        print(f"Attempting to load {model_name} data from {file_path}...")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            df['model'] = model_name
            dfs_to_concat.append(df)
            print(f"  ...Success: Loaded {len(df)} runs for {model_name} model.")
        except FileNotFoundError:
            print(f"  ...Warning: File not found: {file_path}. Skipping this file.")
        except Exception as e:
            print(f"  ...Error: Failed to load {file_path}. Error: {e}")

    if not dfs_to_concat:
        print("Error: No data files were successfully loaded. Exiting.")
        return None

    df = pd.concat(dfs_to_concat, ignore_index=True)
    print(f"\nSuccessfully combined data. Total runs: {len(df)}")
    print("Model distribution:")
    print(df['model'].value_counts())

    # Data cleaning
    df['predicted_params'] = df['predicted_params'].apply(
        lambda x: {} if x is None or (isinstance(x, float) and np.isnan(x)) else x
    )
    df['target_final'] = df['target_final'].fillna(1.0)

    # Success/failure columns
    success_cols = [col for col in df.columns if 'validation_success' in col]
    for col in success_cols:
        df[col] = df[col].fillna(False)
        new_col_name = f"is_{col.replace('validation_success_', 'success_')}"
        df[new_col_name] = df[col].astype(int)

    df['is_failure_5pct'] = (1 - df['is_success_5']).astype(int)
    df['delta_p'] = (df['target_final'] - 1.0).abs()

    # Classification
    df['power_bin'] = df['target_final'].apply(classify_power_bin)
    df['scenario_type'] = df['predicted_params'].apply(classify_scenario_type)

    print("Data processing complete.\n")
    return df


# --- Task 1.1: Bootstrap Confidence Intervals ---

def run_task_1_1_bootstrap_ci(df, n_bootstrap=10000, output_dir=Path('.')):
    """Bootstrap Confidence Intervals for Success Rates"""
    print("\n" + "=" * 80)
    print("Task 1.1: Bootstrap Confidence Intervals (95% CI)")
    print("=" * 80)

    bands = [
        {'label': '±1%', 'col': 'is_success_1'},
        {'label': '±2%', 'col': 'is_success_2'},
        {'label': '±3%', 'col': 'is_success_3'},
        {'label': '±5%', 'col': 'is_success_5'},
        {'label': '±10%', 'col': 'is_success_10'}
    ]

    results = []

    for model_name in sorted(df['model'].unique()):
        model_df = df[df['model'] == model_name]
        n_total = len(model_df)

        print(f"\n--- {model_name} Model (n={n_total}) ---")

        for band in bands:
            col = band['col']
            successes = model_df[col].sum()
            success_rate = successes / n_total

            bootstrap_rates = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(
                    model_df[col].values,
                    size=n_total,
                    replace=True
                )
                bootstrap_rates.append(sample.mean())

            ci_lower = np.percentile(bootstrap_rates, 2.5)
            ci_upper = np.percentile(bootstrap_rates, 97.5)

            results.append({
                'Model': model_name,
                'Tolerance': band['label'],
                'Success_Rate': success_rate,
                'CI_Lower_2.5%': ci_lower,
                'CI_Upper_97.5%': ci_upper,
                'CI_Width': ci_upper - ci_lower,
                'Successes': successes,
                'Total': n_total
            })

            print(f"  {band['label']}: {success_rate:.1%} "
                  f"[{ci_lower:.1%}, {ci_upper:.1%}] "
                  f"(width: {(ci_upper - ci_lower):.1%})")

    results_df = pd.DataFrame(results)

    print("\n--- Summary Table ---")
    print(results_df.to_string(index=False, float_format="%.4f"))

    results_df.to_csv(output_dir / 'bootstrap_ci_results.csv', index=False)
    print(f"\n✓ Results saved to: {output_dir / 'bootstrap_ci_results.csv'}")
    return results_df


# --- Task 1.2: Stratification ---

def run_task_1_2_stratification(df, output_dir=Path('.')):
    """Scenario Type × Power Bin Stratification"""
    print("\n" + "=" * 80)
    print("Task 1.2: Stratification (Scenario Type × Power Bin)")
    print("=" * 80)

    main_scenarios = ['single_b1', 'single_b2', 'simultaneous', 'sequential']
    stratified_df = df[df['scenario_type'].isin(main_scenarios)].copy()

    all_results = []

    for model_name in sorted(df['model'].unique()):
        model_df = stratified_df[stratified_df['model'] == model_name]

        print(f"\n--- {model_name} Model ---")

        for scenario in main_scenarios:
            for power_bin in ['Small', 'Medium', 'Large']:
                subset = model_df[
                    (model_df['scenario_type'] == scenario) &
                    (model_df['power_bin'] == power_bin)
                    ]

                if len(subset) > 0:
                    success_rate = subset['is_success_5'].mean()
                    n_cases = len(subset)

                    all_results.append({
                        'Model': model_name,
                        'Scenario': scenario,
                        'Power_Bin': power_bin,
                        'Success_Rate': success_rate,
                        'N_Cases': n_cases
                    })

        pivot = model_df.groupby(['scenario_type', 'power_bin'])['is_success_5'].agg([
            ('Success_Rate_%', lambda x: x.mean() * 100),
            ('N', 'count')
        ]).round(1)

        print(pivot.to_string())

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / 'stratification_results.csv', index=False)
    print(f"\n✓ Results saved to: {output_dir / 'stratification_results.csv'}")
    return results_df


# --- Task 1.3: FDR Multiple Comparison ---

def run_task_1_3_fdr_comparison(df, output_dir=Path('.')):
    """Pairwise FDR Multiple Comparison Correction"""
    print("\n" + "=" * 80)
    print("Task 1.3: Pairwise FDR Multiple Comparison Correction")
    print("=" * 80)

    bands_to_test = [
        {'label': '±1%', 'col': 'is_success_1'},
        {'label': '±2%', 'col': 'is_success_2'},
        {'label': '±3%', 'col': 'is_success_3'},
        {'label': '±5%', 'col': 'is_success_5'},
        {'label': '±10%', 'col': 'is_success_10'}
    ]

    models = sorted(df['model'].unique())
    model_pairs = list(itertools.combinations(models, 2))

    if len(models) < 2:
        print("Only one model found, skipping comparison.")
        return

    all_p_values = []
    test_results_data = []

    for model_a, model_b in model_pairs:
        for band in bands_to_test:
            col = band['col']

            total_a = len(df[df['model'] == model_a])
            success_a = df[df['model'] == model_a][col].sum()
            fail_a = total_a - success_a

            total_b = len(df[df['model'] == model_b])
            success_b = df[df['model'] == model_b][col].sum()
            fail_b = total_b - success_b

            contingency = np.array([[success_a, fail_a], [success_b, fail_b]])

            chi2, p, dof, expected = chi2_contingency(contingency, correction=False)
            all_p_values.append(p)

            test_results_data.append({
                'Comparison': f"{model_a} vs {model_b}",
                'Band': band['label'],
                f'{model_a}_Success': f"{success_a}/{total_a} ({success_a / total_a:.1%})",
                f'{model_b}_Success': f"{success_b}/{total_b} ({success_b / total_b:.1%})",
                'Chi2': chi2,
                'Original_P_Value': p
            })

    reject, corrected_p_values, _, _ = multipletests(all_p_values, alpha=0.05, method='fdr_bh')

    for i, result in enumerate(test_results_data):
        result['Corrected_P_Value (fdr_bh)'] = corrected_p_values[i]
        result['Significant (alpha=0.05)'] = reject[i]

    results_df = pd.DataFrame(test_results_data)
    print(results_df.to_string(index=False))

    results_df.to_csv(output_dir / 'fdr_comparison_results.csv', index=False)
    print(f"\n✓ Results saved to: {output_dir / 'fdr_comparison_results.csv'}")


# --- Task 2: Failure Analysis ---

def run_task_2_failure_distribution(df, output_dir=Path('.')):
    """Failure Distribution Data Extraction"""
    print("\n" + "=" * 80)
    print("Task 2: Failure Distribution Analysis")
    print("=" * 80)

    failed_df = df[df['is_failure_5pct'] == 1].copy()

    failed_df['P_initial'] = 1.0
    failed_df['P_target'] = failed_df['target_final']

    print(f"\nTotal failures across all models: {len(failed_df)}")

    for model_name in sorted(df['model'].unique()):
        model_failures = failed_df[failed_df['model'] == model_name]

        if len(model_failures) == 0:
            print(f"\n{model_name}: No failures (100% success)")
            continue

        print(f"\n{model_name} Model ({len(model_failures)} failures):")
        print(f"  Mean |ΔP|: {model_failures['delta_p'].mean():.4f}")
        print(f"  Median |ΔP|: {model_failures['delta_p'].median():.4f}")
        print(f"  Std Dev |ΔP|: {model_failures['delta_p'].std():.4f}")

        bin_counts = model_failures['power_bin'].value_counts()
        for bin_name in ['Small', 'Medium', 'Large']:
            count = bin_counts.get(bin_name, 0)
            pct = (count / len(model_failures) * 100) if len(model_failures) > 0 else 0
            print(f"    {bin_name}: {count} ({pct:.1f}%)")

    output_cols = ['model', 'case_id', 'P_initial', 'P_target', 'actual_final',
                   'delta_p', 'power_bin', 'scenario_type', 'final_error']
    available_cols = [col for col in output_cols if col in failed_df.columns]

    failed_df[available_cols].to_csv(output_dir / 'failure_distribution_data.csv', index=False)
    print(f"\n✓ Failure data saved to: {output_dir / 'failure_distribution_data.csv'}")


# --- Task 3: Actuation Pattern ---

def run_task_3_actuation_pattern(df, output_dir=Path('.')):
    """Actuation Pattern Quantification"""
    print("\n" + "=" * 80)
    print("Task 3: Actuation Pattern Quantification")
    print("=" * 80)

    success_rate_by_pattern = df.groupby(['model', 'scenario_type']).agg({
        'is_success_5': ['mean', 'count']
    }).reset_index()

    success_rate_by_pattern.columns = ['Model', 'Scenario_Type', 'Success_Rate', 'Total_Cases']
    success_rate_by_pattern['Success_Rate'] = (success_rate_by_pattern['Success_Rate'] * 100).round(1)

    print("\n--- Success Rate (±5%) by Model and Actuation Pattern ---")
    print(success_rate_by_pattern.to_string(index=False))

    print("\n--- Actuation Pattern Distribution (%) by Model ---")
    for model_name in sorted(df['model'].unique()):
        model_df = df[df['model'] == model_name]
        pattern_dist = model_df['scenario_type'].value_counts(normalize=True) * 100

        print(f"\n{model_name} Model:")
        for pattern, pct in pattern_dist.items():
            count = model_df['scenario_type'].value_counts()[pattern]
            print(f"  {pattern}: {count} ({pct:.1f}%)")

    if 'final_error' in df.columns:
        valid_errors_df = df[df['final_error'].notna()]
        error_stats_by_pattern = valid_errors_df.groupby(['model', 'scenario_type'])['final_error'].describe(
            percentiles=[.25, .5, .75]
        )

        error_stats_by_pattern = error_stats_by_pattern.rename(columns={
            'mean': 'Mean_Error', 'std': 'Std_Dev', '50%': 'Median',
            '25%': '25th', '75%': '75th'
        })
        error_stats_by_pattern = error_stats_by_pattern[['Mean_Error', 'Median', 'Std_Dev',
                                                         '25th', '75th', 'min', 'max']]

        print("\n\n--- Error Distribution by Model and Actuation Pattern ---")
        print(error_stats_by_pattern.to_string(float_format="%.4f"))

    success_rate_by_pattern.to_csv(output_dir / 'actuation_pattern_success_rates.csv', index=False)
    print(f"\n✓ Results saved to: {output_dir / 'actuation_pattern_success_rates.csv'}")


# --- Task 4: Percentile Analysis ---

def run_task_4_percentile_analysis(df, output_dir=Path('.')):
    """Percentile Analysis"""
    print("\n" + "=" * 80)
    print("Task 4: Percentile Analysis")
    print("=" * 80)

    if 'final_error' not in df.columns:
        print("Warning: 'final_error' column not found.")
        return

    valid_errors = df[df['final_error'].notna()]
    percentiles_to_calc = [.25, .5, .75, .95]

    stats_df = valid_errors.groupby('model')['final_error'].describe(percentiles=percentiles_to_calc)

    stats_df = stats_df.rename(columns={
        'mean': 'Mean', 'std': 'Std_Dev', 'min': 'Min', 'max': 'Max',
        '25%': '25th_Percentile', '50%': 'Median',
        '75%': '75th_Percentile', '95%': '95th_Percentile'
    })

    final_columns = [
        'Mean', 'Median', '25th_Percentile',
        '75th_Percentile', '95th_Percentile', 'Std_Dev', 'Min', 'Max'
    ]
    final_stats = stats_df[final_columns]

    print("\n--- Terminal Power Error Statistics (All Models) ---")
    print(final_stats.to_string(float_format="%.4f"))

    if len(final_stats) >= 2:
        models = sorted(valid_errors['model'].unique())
        print("\n--- Improvement Analysis ---")
        for i in range(len(models) - 1):
            model_a = models[i]
            model_b = models[i + 1]

            mean_a = final_stats.loc[model_a, 'Mean']
            mean_b = final_stats.loc[model_b, 'Mean']
            improvement = (mean_a - mean_b) / mean_a * 100

            median_a = final_stats.loc[model_a, 'Median']
            median_b = final_stats.loc[model_b, 'Median']
            median_improvement = (median_a - median_b) / median_a * 100

            print(f"\n{model_a} → {model_b}:")
            print(f"  Mean error: {mean_a:.4f} → {mean_b:.4f} ({improvement:+.1f}%)")
            print(f"  Median error: {median_a:.4f} → {median_b:.4f} ({median_improvement:+.1f}%)")

    final_stats.to_csv(output_dir / 'percentile_analysis_results.csv')
    print(f"\n✓ Results saved to: {output_dir / 'percentile_analysis_results.csv'}")


# --- Main ---

def main():
    """Main function"""

    # 경로 설정 (스크립트 위치 기준)
    SCRIPT_DIR = Path(__file__).resolve().parent  # plotting/
    REPO_ROOT = SCRIPT_DIR.parent                  # repo root

    # 결과 JSON 위치: <repo>/validation/validation_runs_*/  (검증 스크립트가 생성)
    VALIDATION_DIR = REPO_ROOT / "validation"

    # Output directory (figures 저장)
    output_dir = SCRIPT_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Output directory: {output_dir}")
    print("=" * 80)

    # Validation JSON 파일 경로 - 환경에 따라 파일명/타임스탬프가 다를 수 있으므로
    # 필요시 직접 지정하거나 환경변수로 override 하세요.
    files_to_load = {
        '1K':   str(VALIDATION_DIR / "validation_runs_v7_simple_1k" / "validation_results_v7_simple_1k_2000cases.json"),
        '10K':  str(VALIDATION_DIR / "validation_runs_v7_simple"    / "validation_results_v7_simple_2000cases_10K.json"),
        '100K': str(VALIDATION_DIR / "validation_runs_v7_simple"    / "validation_results_v7_simple_2000cases_100K.json"),
    }

    # Load data
    master_df = load_and_process_data(files_to_load)

    if master_df is None:
        print("Data loading failed. Exiting.")
        return

    # Run all analyses
    print("\n" + "=" * 80)
    print("STARTING ALL ANALYSES (1K, 10K, 100K)")
    print("=" * 80)

    # Task 1: Statistical Rigor
    run_task_1_1_bootstrap_ci(master_df, n_bootstrap=10000, output_dir=output_dir)
    run_task_1_2_stratification(master_df, output_dir=output_dir)
    run_task_1_3_fdr_comparison(master_df, output_dir=output_dir)

    # Task 2: Failure Analysis
    run_task_2_failure_distribution(master_df, output_dir=output_dir)

    # Task 3: Actuation Patterns
    run_task_3_actuation_pattern(master_df, output_dir=output_dir)

    # Task 4: Percentiles
    run_task_4_percentile_analysis(master_df, output_dir=output_dir)

    # Visualization - Overall comparison
    plot_comparison_all_models(master_df, output_dir)

    # Individual model graphs
    print("\n" + "=" * 80)
    print("Creating individual model graphs...")
    print("=" * 80)

    for model_name in sorted(master_df['model'].unique()):
        print(f"\nCreating {model_name} model graph...")
        model_df = master_df[master_df['model'] == model_name]

        # Convert DataFrame to results format
        results = []
        for idx, row in model_df.iterrows():
            result = {
                'simulation_success': row.get('simulation_success', False),
                'validation_success_1': row.get('validation_success_1', False),
                'validation_success_2': row.get('validation_success_2', False),
                'validation_success_3': row.get('validation_success_3', False),
                'validation_success_5': row.get('validation_success_5', False),
                'validation_success_10': row.get('validation_success_10', False),
                'target_initial': row.get('target_initial', 1.0),
                'target_final': row.get('target_final', 1.0),
                'final_error': row.get('final_error', None)
            }
            results.append(result)

        # Create individual graph
        plot_validation_results(results, output_dir, model_name=model_name)

    print("\n" + "=" * 80)
    print("ALL ANALYSES COMPLETE")
    print("=" * 80)

    print(f"\n📁 All files saved to: {output_dir}")
    print("\nGenerated files:")
    output_files = [
        "bootstrap_ci_results.csv",
        "stratification_results.csv",
        "fdr_comparison_results.csv",
        "failure_distribution_data.csv",
        "actuation_pattern_success_rates.csv",
        "percentile_analysis_results.csv",
        "comparison_plot_all_models_[timestamp].png",
        "validation_plot_v7_simple_1k_[timestamp].png",
        "validation_plot_v7_simple_10k_[timestamp].png",
        "validation_plot_v7_simple_100k_[timestamp].png"
    ]

    for f in output_files:
        print(f"  ✓ {f}")

    print(f"\n📂 Check files: ls {output_dir}")


if __name__ == "__main__":
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', None)

    main()