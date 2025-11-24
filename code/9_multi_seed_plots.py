import json
import matplotlib.pyplot as plt
import pandas as pd
from utils.paths import PROJECT_ROOT

# Configuration
RESULTS_DIR = PROJECT_ROOT / "results"

# Selected scales for cleaner visualization
SELECTED_SCALES = [0.0, 0.1, 0.4, 0.8, 1.2, 1.6, 2.0]

# Model names
MODELS = {
    "bert": "BERT-base",
    "modern_bert": "Modern BERT-base"
}


def load_results(model_name, bias_type):
    """
    Load aggregated results for a specific model and bias type.
    
    Args:
        model_name: 'bert' or 'modern_bert'
        bias_type: 'gender' or 'age'
    
    Returns:
        DataFrame with results
    """
    results_path = RESULTS_DIR / f"neuron_scaling_bias_mitigation_{bias_type}" / model_name / "intervention_res.json"
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Separate baseline
    baseline = df[df['mode'] == 'baseline'].iloc[0].to_dict()
    df = df[df['mode'] != 'baseline'].copy()
    
    # Modify zero mode to have scale=0.0 for consistent plotting
    df.loc[df['mode'] == 'zero', 'scale'] = 0.0
    
    # Convert accuracies to percentages
    df['task_accuracy'] = df['task_accuracy'] * 100
    if bias_type == 'gender':
        df['gender_balanced_accuracy'] = df['gender_balanced_accuracy'] * 100
        baseline['gender_balanced_accuracy'] = baseline['gender_balanced_accuracy'] * 100
    else:
        df['age_balanced_accuracy'] = df['age_balanced_accuracy'] * 100
        baseline['age_balanced_accuracy'] = baseline['age_balanced_accuracy'] * 100
    
    baseline['task_accuracy'] = baseline['task_accuracy'] * 100
    
    return df, baseline


def plot_results(df, baseline, model_name, bias_type, plots_dir):
    """
    Create plots for task accuracy and bias balanced accuracy.
    
    Args:
        df: DataFrame with results
        baseline: Dictionary with baseline metrics
        model_name: 'bert' or 'modern_bert'
        bias_type: 'gender' or 'age'
        plots_dir: Path to save plots
    """
    bias_metric = f"{bias_type}_balanced_accuracy"
    bias_label = f"{bias_type.capitalize()} Balanced Accuracy"
    
    # Plot 1: Task Accuracy
    plt.figure(figsize=(12, 6))
    for scale in SELECTED_SCALES:
        if scale == 0.0:
            zero_df = df[df['mode'] == 'zero']
            plt.plot(zero_df['coverage'].mul(100), zero_df['task_accuracy'],
                     marker='o', label='Zeroing-out', linewidth=2)
        else:
            mode = 'scale_up' if scale > 1.0 else 'scale_down'
            subset = df[(df['mode'] == mode) & (df['scale'] == scale)]
            if not subset.empty:
                plt.plot(subset['coverage'].mul(100), subset['task_accuracy'],
                         marker='o', label=f'Scale {scale}')
    
    plt.axhline(baseline['task_accuracy'], color='red', linestyle='--', label='Baseline', linewidth=2)
    plt.xlabel('Coverage (%)', fontsize=12)
    plt.ylabel('Task Accuracy (%)', fontsize=12)
    plt.title(f'Task Accuracy vs Coverage - {MODELS[model_name]} ({bias_type.capitalize()})', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, axis='y', alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    plt.tight_layout()
    
    # Save plot
    filename = f"task_accuracy_{bias_type}_{model_name}.png"
    plt.savefig(plots_dir / filename, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {filename}")
    plt.close()
    
    # Plot 2: Bias Balanced Accuracy
    plt.figure(figsize=(12, 6))
    for scale in SELECTED_SCALES:
        if scale == 0.0:
            zero_df = df[df['mode'] == 'zero']
            plt.plot(zero_df['coverage'].mul(100), zero_df[bias_metric],
                     marker='o', label='Zeroing-out', linewidth=2)
        else:
            mode = 'scale_up' if scale > 1.0 else 'scale_down'
            subset = df[(df['mode'] == mode) & (df['scale'] == scale)]
            if not subset.empty:
                plt.plot(subset['coverage'].mul(100), subset[bias_metric],
                         marker='o', label=f'Scale {scale}')
    
    plt.axhline(baseline[bias_metric], color='red', linestyle='--', label='Baseline', linewidth=2)
    plt.xlabel('Coverage (%)', fontsize=12)
    plt.ylabel(f'{bias_label} (%)', fontsize=12)
    plt.title(f'{bias_label} vs Coverage - {MODELS[model_name]} ({bias_type.capitalize()})', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, axis='y', alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    plt.tight_layout()
    
    # Save plot
    filename = f"{bias_type}_balanced_accuracy_{model_name}.png"
    plt.savefig(plots_dir / filename, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {filename}")
    plt.close()


def plot_combined_metrics(df, baseline, model_name, bias_type, plots_dir):
    """
    Create combined plot showing both task accuracy and bias balanced accuracy.
    
    Args:
        df: DataFrame with results
        baseline: Dictionary with baseline metrics
        model_name: 'bert' or 'modern_bert'
        bias_type: 'gender' or 'age'
        plots_dir: Path to save plots
    """
    bias_metric = f"{bias_type}_balanced_accuracy"
    bias_label = f"{bias_type.capitalize()} Balanced Accuracy"
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Plot 1: Task Accuracy
    for scale in SELECTED_SCALES:
        if scale == 0.0:
            zero_df = df[df['mode'] == 'zero']
            ax1.plot(zero_df['coverage'].mul(100), zero_df['task_accuracy'],
                     marker='o', label='Zeroing-out', linewidth=2)
        else:
            mode = 'scale_up' if scale > 1.0 else 'scale_down'
            subset = df[(df['mode'] == mode) & (df['scale'] == scale)]
            if not subset.empty:
                ax1.plot(subset['coverage'].mul(100), subset['task_accuracy'],
                         marker='o', label=f'Scale {scale}')
    
    ax1.axhline(baseline['task_accuracy'], color='red', linestyle='--', label='Baseline', linewidth=2)
    ax1.set_xlabel('Coverage (%)', fontsize=12)
    ax1.set_ylabel('Task Accuracy (%)', fontsize=12)
    ax1.set_title('Task Accuracy vs Coverage', fontsize=13)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    # Plot 2: Bias Balanced Accuracy
    for scale in SELECTED_SCALES:
        if scale == 0.0:
            zero_df = df[df['mode'] == 'zero']
            ax2.plot(zero_df['coverage'].mul(100), zero_df[bias_metric],
                     marker='o', label='Zeroing-out', linewidth=2)
        else:
            mode = 'scale_up' if scale > 1.0 else 'scale_down'
            subset = df[(df['mode'] == mode) & (df['scale'] == scale)]
            if not subset.empty:
                ax2.plot(subset['coverage'].mul(100), subset[bias_metric],
                         marker='o', label=f'Scale {scale}')
    
    ax2.axhline(baseline[bias_metric], color='red', linestyle='--', label='Baseline', linewidth=2)
    ax2.set_xlabel('Coverage (%)', fontsize=12)
    ax2.set_ylabel(f'{bias_label} (%)', fontsize=12)
    ax2.set_title(f'{bias_label} vs Coverage', fontsize=13)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    fig.suptitle(f'{MODELS[model_name]} - {bias_type.capitalize()} Bias Mitigation', fontsize=15, y=1.02)
    plt.tight_layout()
    
    # Save plot
    filename = f"combined_metrics_{bias_type}_{model_name}.png"
    plt.savefig(plots_dir / filename, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {filename}")
    plt.close()


def create_comparison_plot(results_dict, bias_type, metric, plots_dir):
    """
    Create comparison plot across models for a specific metric.
    
    Args:
        results_dict: Dictionary with model results {model_name: (df, baseline)}
        bias_type: 'gender' or 'age'
        metric: 'task_accuracy' or bias balanced accuracy metric
        plots_dir: Path to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, (model_name, (df, baseline)) in enumerate(results_dict.items()):
        ax = axes[idx]
        
        for scale in SELECTED_SCALES:
            if scale == 0.0:
                zero_df = df[df['mode'] == 'zero']
                ax.plot(zero_df['coverage'].mul(100), zero_df[metric],
                        marker='o', label='Zeroing-out', linewidth=2)
            else:
                mode = 'scale_up' if scale > 1.0 else 'scale_down'
                subset = df[(df['mode'] == mode) & (df['scale'] == scale)]
                if not subset.empty:
                    ax.plot(subset['coverage'].mul(100), subset[metric],
                            marker='o', label=f'Scale {scale}')
        
        ax.axhline(baseline[metric], color='red', linestyle='--', label='Baseline', linewidth=2)
        ax.set_xlabel('Coverage (%)', fontsize=12)
        
        if metric == 'task_accuracy':
            ax.set_ylabel('Task Accuracy (%)', fontsize=12)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        else:
            label = bias_type.capitalize() + ' Balanced Accuracy'
            ax.set_ylabel(f'{label} (%)', fontsize=12)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        
        ax.set_title(f'{MODELS[model_name]}', fontsize=13)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    metric_name = 'Task Accuracy' if metric == 'task_accuracy' else f'{bias_type.capitalize()} Balanced Accuracy'
    fig.suptitle(f'{metric_name} Comparison - {bias_type.capitalize()} Bias Mitigation', fontsize=15, y=1.02)
    plt.tight_layout()
    
    # Save plot
    filename = f"comparison_{metric}_{bias_type}.png"
    plt.savefig(plots_dir / filename, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {filename}")
    plt.close()


def main():
    """Main execution function."""
    print(f"\n{'#'*70}")
    print("# Multi-seed Bias Mitigation - Plotting Results")
    print(f"{'#'*70}\n")
    
    # Process each bias type
    for bias_type in ['gender', 'age']:
        print(f"\n{'='*70}")
        print(f"Processing {bias_type.upper()} bias mitigation results")
        print(f"{'='*70}\n")
        
        results_dict = {}
        
        # Load results for each model
        for model_name in MODELS.keys():
            print(f"[INFO] Loading {model_name} results...")
            df, baseline = load_results(model_name, bias_type)
            results_dict[model_name] = (df, baseline)
            
            # Create plots directory for this model
            plots_dir = RESULTS_DIR / "plots" / model_name
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"[INFO] Creating plots for {model_name}...")
            plot_results(df, baseline, model_name, bias_type, plots_dir)
            plot_combined_metrics(df, baseline, model_name, bias_type, plots_dir)
        
        # Create comparison plots in shared plots directory
        print(f"\n[INFO] Creating comparison plots for {bias_type}...")
        comparison_plots_dir = RESULTS_DIR / "plots"
        comparison_plots_dir.mkdir(parents=True, exist_ok=True)
        
        bias_metric = f"{bias_type}_balanced_accuracy"
        create_comparison_plot(results_dict, bias_type, 'task_accuracy', comparison_plots_dir)
        create_comparison_plot(results_dict, bias_type, bias_metric, comparison_plots_dir)
    
    print(f"\n{'#'*70}")
    print(f"# All plots saved to: {RESULTS_DIR / 'plots'}")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
