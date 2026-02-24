import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.paths import PROJECT_ROOT
# from scipy.interpolate import make_interp_spline

# Configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

MODELS = ['bert', 'modern_bert']
MODEL_DISPLAY_NAMES = {'bert': 'BERT', 'modern_bert': 'ModernBERT'}
LAYER_STRATEGIES = ['top_3', 'first_half', 'second_half', 'all']
COLORS = {
    'steering': '#2E86AB',  # Blue
    'neuron_scaling': '#A23B72',  # Purple
}
MARKERS = {
    'top_3': 'o',
    'first_half': 's',
    'second_half': '^',
    'all': 'D'
}
LAYER_STRATEGY_DISPLAY_NAMES = {
    'top_3': 'Top 3',
    'first_half': 'First Half',
    'second_half': 'Second Half',
    'all': 'All'
}


def load_baseline_results():
    """Load baseline (no intervention) results from three-head model."""
    baseline_path = PROJECT_ROOT / "results" / "three_head_training_combined" / "performance_eval.json"

    with open(baseline_path, 'r') as f:
        data = json.load(f)

    baselines = {}
    for model_name in MODELS:
        model_data = data[model_name]
        baselines[model_name] = {
            'task_accuracy': model_data['average_task_accuracy'],
            'gender_accuracy': model_data['average_gender_accuracy'],
            'gender_balanced_accuracy': model_data['average_gender_balanced_accuracy'],
            'age_accuracy': model_data['average_age_accuracy'],
            'age_balanced_accuracy': model_data['average_age_balanced_accuracy']
        }

    return baselines


def load_steering_results(model_name):
    """Load steering vector results."""
    results_path = PROJECT_ROOT / "results" / "steering_vectors_combined" / model_name / "steering_results.json"

    with open(results_path, 'r') as f:
        data = json.load(f)

    return data


def load_neuron_scaling_results(model_name):
    """Load neuron scaling results."""
    results_path = (
        PROJECT_ROOT / "results" / "neuron_scaling_bias_mitigation_combined"
        / model_name / "intervention_res.json"
    )

    with open(results_path, 'r') as f:
        data = json.load(f)

    return data


def prepare_steering_data(steering_results, baseline):
    """Prepare steering vector data for plotting."""
    data = {
        'top_3': [], 'first_half': [], 'second_half': [], 'all': []
    }

    for entry in steering_results:
        layers = entry['layers']
        if layers in data:
            data[layers].append({
                'coefficient': entry['coefficient'],
                'task_accuracy': entry['task_accuracy'],
                'gender_balanced_accuracy': entry['gender_balanced_accuracy'],
                'age_balanced_accuracy': entry['age_balanced_accuracy'],
                # Calculate reductions from baseline
                'task_reduction': baseline['task_accuracy'] - entry['task_accuracy'],
                'gender_reduction': baseline['gender_balanced_accuracy'] - entry['gender_balanced_accuracy'],
                'age_reduction': baseline['age_balanced_accuracy'] - entry['age_balanced_accuracy']
            })

    return data


def prepare_neuron_scaling_data(neuron_results, baseline):
    """Prepare neuron scaling data for plotting."""
    data = {
        'top_3': [], 'first_half': [], 'second_half': [], 'all': []
    }

    # Map neuron scaling layer names to our standard names
    layer_mapping = {
        'top_3_layers': 'top_3',
        'first_half_layers': 'first_half',
        'second_half_layers': 'second_half',
        'all_layers': 'all'
    }

    for entry in neuron_results:
        layers_key = entry.get('layers', entry.get('layer_strategy', ''))
        layers = layer_mapping.get(layers_key, layers_key)

        if layers in data:
            # Combine scaling_factor and method for identification
            scaling = entry.get('scaling_factor', 0)
            method = entry.get('method', 'unknown')

            data[layers].append({
                'scaling_factor': scaling,
                'method': method,
                'task_accuracy': entry['task_accuracy'],
                'gender_balanced_accuracy': entry['gender_balanced_accuracy'],
                'age_balanced_accuracy': entry['age_balanced_accuracy'],
                # Calculate reductions from baseline
                'task_reduction': baseline['task_accuracy'] - entry['task_accuracy'],
                'gender_reduction': baseline['gender_balanced_accuracy'] - entry['gender_balanced_accuracy'],
                'age_reduction': baseline['age_balanced_accuracy'] - entry['age_balanced_accuracy']
            })

    return data


def plot_pareto_front(model_name, bias_type, steering_data, neuron_data, baseline, output_dir):
    """
    Plot Pareto front: Task Accuracy vs Bias Reduction
    Shows which method achieves better trade-offs.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    bias_key = f'{bias_type}_balanced_accuracy'

    # Plot each layer strategy
    for layer_strategy in LAYER_STRATEGIES:
        # Steering vectors
        steering_layer = steering_data[layer_strategy]
        if steering_layer:
            task_accs = [d['task_accuracy'] for d in steering_layer]
            bias_accs = [d[bias_key] for d in steering_layer]

            ax.scatter(bias_accs, task_accs,
                       c=COLORS['steering'], marker=MARKERS[layer_strategy],
                       s=100, alpha=0.7, edgecolors='black', linewidth=1,
                       label=f'Steering - {LAYER_STRATEGY_DISPLAY_NAMES[layer_strategy]}')

        # Neuron scaling
        neuron_layer = neuron_data[layer_strategy]
        if neuron_layer:
            task_accs = [d['task_accuracy'] for d in neuron_layer]
            bias_accs = [d[bias_key] for d in neuron_layer]

            ax.scatter(bias_accs, task_accs,
                       c=COLORS['neuron_scaling'], marker=MARKERS[layer_strategy],
                       s=100, alpha=0.7, edgecolors='black', linewidth=1,
                       label=f'Neuron Scaling - {LAYER_STRATEGY_DISPLAY_NAMES[layer_strategy]}')

    # Plot baseline
    ax.scatter([baseline[bias_key]], [baseline['task_accuracy']],
               c='red', marker='*', s=400, edgecolors='black', linewidth=2,
               label='Baseline (No Intervention)', zorder=10)

    ax.set_xlabel(f'{bias_type.capitalize()} Balanced Accuracy', fontsize=16, weight='bold')
    ax.set_ylabel('Task Accuracy', fontsize=16, weight='bold')
    ax.set_title(
        f'{MODEL_DISPLAY_NAMES[model_name]}: Task Accuracy vs {bias_type.capitalize()} Bias\n'
        f'(Lower bias accuracy = More bias removed)',
        fontsize=18, weight='bold', pad=20)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_pareto_{bias_type}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved Pareto front for {MODEL_DISPLAY_NAMES[model_name]} - {bias_type}")


def plot_pareto_front_regression(model_name, bias_type, steering_data, neuron_data, baseline, output_dir):
    """
    Plot Pareto front with regression curves instead of scattered dots.
    Shows smooth trade-off curves between task accuracy and bias reduction.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    bias_key = f'{bias_type}_balanced_accuracy'

    # Plot each layer strategy
    for layer_strategy in LAYER_STRATEGIES:
        # Steering vectors
        steering_layer = steering_data[layer_strategy]
        if steering_layer and len(steering_layer) > 2:  # Need at least 3 points for fitting
            bias_vals = [x[bias_key] for x in steering_layer]
            task_vals = [x['task_accuracy'] for x in steering_layer]

            # Sort by bias values for smooth curve
            sorted_pairs = sorted(zip(bias_vals, task_vals))
            bias_sorted = [x[0] for x in sorted_pairs]
            task_sorted = [x[1] for x in sorted_pairs]

            # Fit polynomial (degree 2 for smooth curve)
            if len(bias_sorted) >= 3:
                z = np.polyfit(bias_sorted, task_sorted, min(2, len(bias_sorted) - 1))
                p = np.poly1d(z)

                # Create smooth curve
                bias_smooth = np.linspace(min(bias_sorted), max(bias_sorted), 100)
                task_smooth = p(bias_smooth)

                ax.plot(bias_smooth, task_smooth,
                        color=COLORS['steering'], linestyle='-', linewidth=2.5,
                        marker=MARKERS[layer_strategy], markevery=20, markersize=8,
                        label=f'Steering - {LAYER_STRATEGY_DISPLAY_NAMES[layer_strategy]}', alpha=0.8)

        # Neuron scaling
        neuron_layer = neuron_data[layer_strategy]
        if neuron_layer and len(neuron_layer) > 2:
            bias_vals = [x[bias_key] for x in neuron_layer]
            task_vals = [x['task_accuracy'] for x in neuron_layer]

            # Sort by bias values for smooth curve
            sorted_pairs = sorted(zip(bias_vals, task_vals))
            bias_sorted = [x[0] for x in sorted_pairs]
            task_sorted = [x[1] for x in sorted_pairs]

            # Fit polynomial (degree 2 for smooth curve)
            if len(bias_sorted) >= 3:
                z = np.polyfit(bias_sorted, task_sorted, min(2, len(bias_sorted) - 1))
                p = np.poly1d(z)

                # Create smooth curve
                bias_smooth = np.linspace(min(bias_sorted), max(bias_sorted), 100)
                task_smooth = p(bias_smooth)

                ax.plot(bias_smooth, task_smooth,
                        color=COLORS['neuron_scaling'], linestyle='-', linewidth=2.5,
                        marker=MARKERS[layer_strategy], markevery=20, markersize=8,
                        label=f'Neuron Scaling - {LAYER_STRATEGY_DISPLAY_NAMES[layer_strategy]}', alpha=0.8)

    # Plot baseline
    ax.scatter([baseline[bias_key]], [baseline['task_accuracy']],
               c='red', marker='*', s=400, edgecolors='black', linewidth=2,
               label='Baseline (No Intervention)', zorder=10)

    ax.set_xlabel(f'{bias_type.capitalize()} Balanced Accuracy', fontsize=16, weight='bold')
    ax.set_ylabel('Task Accuracy', fontsize=16, weight='bold')
    ax.set_title(
        f'{MODEL_DISPLAY_NAMES[model_name]}: Task Accuracy vs {bias_type.capitalize()} Bias (Regression Curves)\n'
        f'(Lower bias accuracy = More bias removed)',
        fontsize=18, weight='bold', pad=20)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_pareto_regression_{bias_type}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved Pareto front (regression) for {MODEL_DISPLAY_NAMES[model_name]} - {bias_type}")


def plot_layer_strategy_comparison(model_name, bias_type, steering_data, neuron_data, baseline, output_dir):
    """
    Compare layer strategies directly for each method.
    Shows which layers are most effective.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{MODEL_DISPLAY_NAMES[model_name]}: Layer Strategy Comparison - {bias_type.capitalize()} Bias',
                 fontsize=18, weight='bold')

    bias_key = f'{bias_type}_balanced_accuracy'

    # Steering vectors
    for layer_strategy in LAYER_STRATEGIES:
        steering_layer = sorted(steering_data[layer_strategy], key=lambda x: x['coefficient'])
        if steering_layer:
            x = [d['coefficient'] for d in steering_layer]
            y = [d['task_accuracy'] for d in steering_layer]
            bias = [d[bias_key] for d in steering_layer]

            # Task accuracy
            ax1.plot(x, y, marker=MARKERS[layer_strategy], label=LAYER_STRATEGY_DISPLAY_NAMES[layer_strategy],
                     linewidth=2, markersize=8, alpha=0.7)

            # Bias accuracy
            ax2.plot(x, bias, marker=MARKERS[layer_strategy], label=LAYER_STRATEGY_DISPLAY_NAMES[layer_strategy],
                     linewidth=2, markersize=8, alpha=0.7)

    # Baselines
    ax1.axhline(y=baseline['task_accuracy'], color='red', linestyle=':', linewidth=2, alpha=0.8)
    ax2.axhline(y=baseline[bias_key], color='red', linestyle=':', linewidth=2, alpha=0.8)

    ax1.set_xlabel('Steering Coefficient', fontsize=14, weight='bold')
    ax1.set_ylabel('Task Accuracy', fontsize=14, weight='bold')
    ax1.set_title('Task Accuracy by Layer Strategy', fontsize=16, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Steering Coefficient', fontsize=14, weight='bold')
    ax2.set_ylabel(f'{bias_type.capitalize()} Balanced Accuracy', fontsize=14, weight='bold')
    ax2.set_title(f'{bias_type.capitalize()} Bias by Layer Strategy', fontsize=16, weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_layers_steering_{bias_type}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved layer strategy comparison for {model_name} - {bias_type} (steering)")


def plot_neuron_scaling_by_layer(model_name, bias_type, neuron_results, baseline, output_dir):
    """
    Plot neuron scaling results like the reference image:
    - 4 rows (one per layer strategy)
    - 2 columns (Task Accuracy, Bias Accuracy)
    - X-axis: Coverage (%)
    - Lines: Different scales (Zeroing-out, Scale 0.1, 0.4, 0.8, 1.2, 1.6, 2.0)
    """
    bias_key = f'{bias_type}_balanced_accuracy'

    # Create 4x2 subplot grid
    fig, axes = plt.subplots(4, 2, figsize=(20, 20))
    fig.suptitle(f'{MODEL_DISPLAY_NAMES[model_name]} - {bias_type.capitalize()} Bias Mitigation',
                 fontsize=18, weight='bold', y=0.995)

    # Colors for different scales
    scale_colors = {
        'zero': '#1f77b4',      # Blue for zeroing
        0.1: '#ff7f0e',         # Orange
        0.4: '#2ca02c',         # Green
        0.8: '#d62728',         # Red
        1.2: '#9467bd',         # Purple
        1.6: '#8c564b',         # Brown
        2.0: '#e377c2',         # Pink
    }



    for row_idx, layer_strategy in enumerate(LAYER_STRATEGIES):
        ax_task = axes[row_idx, 0]
        ax_bias = axes[row_idx, 1]

        # Organize data by scale/mode
        data_by_scale = {}

        for entry in neuron_results:
            if entry.get('mode') == 'baseline':
                continue

            # Map layer names
            entry_layers = entry.get('layers', '')
            if entry_layers in ['top_3_layers', 'top_3'] and layer_strategy != 'top_3':
                continue
            if entry_layers in ['first_half_layers', 'first_half'] and layer_strategy != 'first_half':
                continue
            if entry_layers in ['second_half_layers', 'second_half'] and layer_strategy != 'second_half':
                continue
            if entry_layers in ['all_layers', 'all'] and layer_strategy != 'all':
                continue
            if entry_layers not in ['top_3_layers', 'top_3', 'first_half_layers', 'first_half',
                                    'second_half_layers', 'second_half', 'all_layers', 'all']:
                continue

            mode = entry.get('mode')
            scale = entry.get('scale')
            coverage = entry.get('coverage', 0)

            if mode == 'zero':
                key = 'zero'
            elif scale is not None:
                key = scale
            else:
                continue

            if key not in data_by_scale:
                data_by_scale[key] = []

            data_by_scale[key].append({
                'coverage': coverage * 100,  # Convert to %
                'task_accuracy': entry['task_accuracy'] * 100,
                'bias_accuracy': entry[bias_key] * 100
            })

        # Plot each scale
        for scale_key in ['zero', 0.1, 0.4, 0.8, 1.2, 1.6, 2.0]:
            if scale_key not in data_by_scale:
                continue

            data_points = sorted(data_by_scale[scale_key], key=lambda x: x['coverage'])
            if not data_points:
                continue

            x = [d['coverage'] for d in data_points]
            y_task = [d['task_accuracy'] for d in data_points]
            y_bias = [d['bias_accuracy'] for d in data_points]

            label = 'Zeroing-out' if scale_key == 'zero' else f'Scale {scale_key}'
            color = scale_colors.get(scale_key, '#000000')

            ax_task.plot(x, y_task, marker='o', label=label, color=color,
                         linewidth=2, markersize=6)
            ax_bias.plot(x, y_bias, marker='o', label=label, color=color,
                         linewidth=2, markersize=6)

        # Add baseline
        ax_task.axhline(y=baseline['task_accuracy'] * 100, color='red',
                        linestyle='--', linewidth=2, label='Baseline')
        ax_bias.axhline(y=baseline[bias_key] * 100, color='red',
                        linestyle='--', linewidth=2, label='Baseline')

        # Formatting
        ax_task.set_xlabel('Coverage (%)', fontsize=14)
        ax_task.set_ylabel('Task Accuracy (%)', fontsize=14)
        ax_task.set_title(f'{LAYER_STRATEGY_DISPLAY_NAMES[layer_strategy]} Layers - Task Accuracy vs Coverage', fontsize=14, weight='bold')
        ax_task.legend(fontsize=10, loc='best')
        ax_task.grid(True, alpha=0.3)
        ax_task.set_xticks([5, 10, 15, 20])
        ax_task.set_xlim(3, 22)

        ax_bias.set_xlabel('Coverage (%)', fontsize=14)
        ax_bias.set_ylabel(f'{bias_type.capitalize()} Balanced Accuracy (%)', fontsize=14)
        ax_bias.set_title(
            f'{LAYER_STRATEGY_DISPLAY_NAMES[layer_strategy]} Layers - {bias_type.capitalize()} '
            f'Balanced Accuracy vs Coverage',
            fontsize=14, weight='bold')
        ax_bias.legend(fontsize=10, loc='best')
        ax_bias.grid(True, alpha=0.3)
        ax_bias.set_xticks([5, 10, 15, 20])
        ax_bias.set_xlim(3, 22)

    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_layers_neuron_{bias_type}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved neuron layer comparison for {bias_type} bias")


# Threshold for acceptable task accuracy (95% of baseline)
TASK_ACCURACY_THRESHOLD = 0.95


def select_best_configuration(data, baseline, threshold=TASK_ACCURACY_THRESHOLD):
    """
    Select best configuration using threshold-based Pareto approach.

    Criterion: Among configurations where task accuracy >= threshold * baseline,
    select the one with lowest average bias (best debiasing).

    If no configuration meets the threshold, fall back to the one with
    minimum task accuracy loss.

    Args:
        data: List of configuration results
        baseline: Baseline results dict
        threshold: Minimum fraction of baseline task accuracy to retain (default 95%)

    Returns:
        Best configuration dict, or None if data is empty
    """
    if not data:
        return None

    baseline_task = baseline['task_accuracy']
    min_task = baseline_task * threshold

    # Filter configurations meeting task threshold
    valid = [x for x in data if x['task_accuracy'] >= min_task]

    if valid:
        # Among valid, pick best debiasing (lowest average bias accuracy)
        best = min(valid, key=lambda x: (x['gender_balanced_accuracy'] + x['age_balanced_accuracy']) / 2)
    else:
        # Fallback: pick configuration with minimum task loss
        best = max(data, key=lambda x: x['task_accuracy'])

    return best


def plot_combined_pareto_front(model_name, steering_data, neuron_data, baseline, output_dir):
    """
    Plot combined Pareto front showing both gender and age biases.
    X-axis: Average bias (gender + age) / 2
    Y-axis: Task accuracy
    Shows overall debiasing effectiveness.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each layer strategy
    for layer_strategy in LAYER_STRATEGIES:
        # Steering vectors
        steering_layer = steering_data[layer_strategy]
        if steering_layer:
            task_accs = [d['task_accuracy'] for d in steering_layer]
            avg_bias = [(d['gender_balanced_accuracy'] + d['age_balanced_accuracy']) / 2
                        for d in steering_layer]

            ax.scatter(avg_bias, task_accs,
                       c=COLORS['steering'], marker=MARKERS[layer_strategy],
                       s=100, alpha=0.7, edgecolors='black', linewidth=1,
                       label=f'Steering - {LAYER_STRATEGY_DISPLAY_NAMES[layer_strategy]}')

        # Neuron scaling
        neuron_layer = neuron_data[layer_strategy]
        if neuron_layer:
            task_accs = [d['task_accuracy'] for d in neuron_layer]
            avg_bias = [(d['gender_balanced_accuracy'] + d['age_balanced_accuracy']) / 2
                        for d in neuron_layer]

            ax.scatter(avg_bias, task_accs,
                       c=COLORS['neuron_scaling'], marker=MARKERS[layer_strategy],
                       s=100, alpha=0.7, edgecolors='black', linewidth=1,
                       label=f'Neuron Scaling - {LAYER_STRATEGY_DISPLAY_NAMES[layer_strategy]}')

    # Plot baseline
    baseline_avg_bias = (baseline['gender_balanced_accuracy'] + baseline['age_balanced_accuracy']) / 2
    ax.scatter([baseline_avg_bias], [baseline['task_accuracy']],
               c='red', marker='*', s=400, edgecolors='black', linewidth=2,
               label='Baseline (No Intervention)', zorder=10)

    ax.set_xlabel('Average Bias Accuracy (Gender + Age) / 2', fontsize=16, weight='bold')
    ax.set_ylabel('Task Accuracy', fontsize=16, weight='bold')
    ax.set_title(
        f'{MODEL_DISPLAY_NAMES[model_name]}: Task Accuracy vs Combined Bias\n'
        f'(Lower bias accuracy = More bias removed)',
        fontsize=18, weight='bold', pad=20)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_pareto_combined.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved combined Pareto front for {MODEL_DISPLAY_NAMES[model_name]}")


def plot_combined_pareto_front_regression(model_name, steering_data, neuron_data, baseline, output_dir):
    """
    Plot combined Pareto front with regression curves showing both gender and age biases.
    X-axis: Average bias (gender + age) / 2
    Y-axis: Task accuracy
    Shows overall debiasing effectiveness with smooth curves.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each layer strategy
    for layer_strategy in LAYER_STRATEGIES:
        # Steering vectors
        steering_layer = steering_data[layer_strategy]
        if steering_layer and len(steering_layer) > 2:
            avg_bias = [(x['gender_balanced_accuracy'] + x['age_balanced_accuracy']) / 2
                        for x in steering_layer]
            task_vals = [x['task_accuracy'] for x in steering_layer]

            # Sort by bias values for smooth curve
            sorted_pairs = sorted(zip(avg_bias, task_vals))
            bias_sorted = [x[0] for x in sorted_pairs]
            task_sorted = [x[1] for x in sorted_pairs]

            # Fit polynomial (degree 2 for smooth curve)
            if len(bias_sorted) >= 3:
                z = np.polyfit(bias_sorted, task_sorted, min(2, len(bias_sorted) - 1))
                p = np.poly1d(z)

                # Create smooth curve
                bias_smooth = np.linspace(min(bias_sorted), max(bias_sorted), 100)
                task_smooth = p(bias_smooth)

                ax.plot(bias_smooth, task_smooth,
                        color=COLORS['steering'], linestyle='-', linewidth=2.5,
                        marker=MARKERS[layer_strategy], markevery=20, markersize=8,
                        label=f'Steering - {LAYER_STRATEGY_DISPLAY_NAMES[layer_strategy]}', alpha=0.8)

        # Neuron scaling
        neuron_layer = neuron_data[layer_strategy]
        if neuron_layer and len(neuron_layer) > 2:
            avg_bias = [(x['gender_balanced_accuracy'] + x['age_balanced_accuracy']) / 2
                        for x in neuron_layer]
            task_vals = [x['task_accuracy'] for x in neuron_layer]

            # Sort by bias values for smooth curve
            sorted_pairs = sorted(zip(avg_bias, task_vals))
            bias_sorted = [x[0] for x in sorted_pairs]
            task_sorted = [x[1] for x in sorted_pairs]

            # Fit polynomial (degree 2 for smooth curve)
            if len(bias_sorted) >= 3:
                z = np.polyfit(bias_sorted, task_sorted, min(2, len(bias_sorted) - 1))
                p = np.poly1d(z)

                # Create smooth curve
                bias_smooth = np.linspace(min(bias_sorted), max(bias_sorted), 100)
                task_smooth = p(bias_smooth)

                ax.plot(bias_smooth, task_smooth,
                        color=COLORS['neuron_scaling'], linestyle='-', linewidth=2.5,
                        marker=MARKERS[layer_strategy], markevery=20, markersize=8,
                        label=f'Neuron Scaling - {LAYER_STRATEGY_DISPLAY_NAMES[layer_strategy]}', alpha=0.8)

    # Plot baseline
    baseline_avg_bias = (baseline['gender_balanced_accuracy'] + baseline['age_balanced_accuracy']) / 2
    ax.scatter([baseline_avg_bias], [baseline['task_accuracy']],
               c='red', marker='*', s=400, edgecolors='black', linewidth=2,
               label='Baseline (No Intervention)', zorder=10)

    ax.set_xlabel('Average Bias Accuracy (Gender + Age) / 2', fontsize=16, weight='bold')
    ax.set_ylabel('Task Accuracy', fontsize=16, weight='bold')
    ax.set_title(
        f'{MODEL_DISPLAY_NAMES[model_name]}: Task Accuracy vs Combined Bias (Regression Curves)\n'
        f'(Lower bias accuracy = More bias removed)',
        fontsize=18, weight='bold', pad=20)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_pareto_regression_combined.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved combined Pareto front (regression) for {MODEL_DISPLAY_NAMES[model_name]}")


def plot_combined_layer_strategy_comparison(model_name, steering_data, neuron_data, baseline, output_dir):
    """
    Compare layer strategies showing both gender and age biases together.
    Shows which layers are most effective for combined debiasing.
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(f'{MODEL_DISPLAY_NAMES[model_name]}: Layer Strategy Comparison - Combined Bias',
                 fontsize=18, weight='bold')

    # Steering vectors
    for layer_strategy in LAYER_STRATEGIES:
        steering_layer = sorted(steering_data[layer_strategy], key=lambda x: x['coefficient'])
        if steering_layer:
            x = [d['coefficient'] for d in steering_layer]
            y_task = [d['task_accuracy'] for d in steering_layer]
            y_gender = [d['gender_balanced_accuracy'] for d in steering_layer]
            y_age = [d['age_balanced_accuracy'] for d in steering_layer]

            # Task accuracy
            axes[0].plot(x, y_task, marker=MARKERS[layer_strategy], label=LAYER_STRATEGY_DISPLAY_NAMES[layer_strategy],
                         linewidth=2, markersize=8, alpha=0.7)

            # Gender bias
            axes[1].plot(x, y_gender, marker=MARKERS[layer_strategy], label=LAYER_STRATEGY_DISPLAY_NAMES[layer_strategy],
                         linewidth=2, markersize=8, alpha=0.7)

            # Age bias
            axes[2].plot(x, y_age, marker=MARKERS[layer_strategy], label=LAYER_STRATEGY_DISPLAY_NAMES[layer_strategy],
                         linewidth=2, markersize=8, alpha=0.7)

    # Baselines
    axes[0].axhline(y=baseline['task_accuracy'], color='red', linestyle=':', linewidth=2, alpha=0.8)
    axes[1].axhline(y=baseline['gender_balanced_accuracy'], color='red', linestyle=':', linewidth=2, alpha=0.8)
    axes[2].axhline(y=baseline['age_balanced_accuracy'], color='red', linestyle=':', linewidth=2, alpha=0.8)

    axes[0].set_xlabel('Steering Coefficient', fontsize=14, weight='bold')
    axes[0].set_ylabel('Task Accuracy', fontsize=14, weight='bold')
    axes[0].set_title('Task Accuracy by Layer Strategy', fontsize=16, weight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Steering Coefficient', fontsize=14, weight='bold')
    axes[1].set_ylabel('Gender Balanced Accuracy', fontsize=14, weight='bold')
    axes[1].set_title('Gender Bias by Layer Strategy', fontsize=16, weight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('Steering Coefficient', fontsize=14, weight='bold')
    axes[2].set_ylabel('Age Balanced Accuracy', fontsize=14, weight='bold')
    axes[2].set_title('Age Bias by Layer Strategy', fontsize=16, weight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_layers_steering_combined.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved combined layer strategy comparison for {model_name} (steering)")


def plot_best_results(model_name, steering_data, neuron_data, baseline, output_dir):
    """
    Bar chart comparing best results from each method.
    Shows optimal trade-off points for all 4 layer strategies.

    Selection criterion (threshold-based Pareto):
    1. Filter configs where task_accuracy >= baseline * 95%
    2. Among valid, pick lowest average bias (best debiasing)
    3. Fallback: if none valid, pick minimum task loss
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    threshold_pct = (1 - TASK_ACCURACY_THRESHOLD) * 100
    fig.suptitle(
        f'{MODEL_DISPLAY_NAMES[model_name]}: Best Results Comparison\n'
        f'(Task accuracy threshold: {threshold_pct:.0f}% max drop)',
        fontsize=18, weight='bold')

    # For each layer strategy, find best steering and neuron configurations
    metrics = ['task_accuracy', 'gender_balanced_accuracy', 'age_balanced_accuracy']
    metric_labels = ['Task Accuracy', 'Gender Balanced\nAccuracy', 'Age Balanced\nAccuracy']

    for idx, layer_strategy in enumerate(LAYER_STRATEGIES):
        ax = axes[idx // 2, idx % 2]

        steering_layer = steering_data[layer_strategy]
        neuron_layer = neuron_data[layer_strategy]

        # Select best using threshold-based approach
        best_steering = select_best_configuration(steering_layer, baseline)
        best_neuron = select_best_configuration(neuron_layer, baseline)

        # Prepare data for bar chart
        x = np.arange(len(metrics))
        width = 0.25

        baseline_vals = [baseline[m] for m in metrics]
        steering_vals = [best_steering[m] for m in metrics] if best_steering else [0, 0, 0]
        neuron_vals = [best_neuron[m] for m in metrics] if best_neuron else [0, 0, 0]

        ax.bar(x - width, baseline_vals, width, label='Baseline', color='red', alpha=0.7, edgecolor='black')
        if best_steering:
            ax.bar(x, steering_vals, width, label='Steering (Best)',
                   color=COLORS['steering'], alpha=0.7, edgecolor='black')
        if best_neuron:
            ax.bar(x + width, neuron_vals, width, label='Neuron (Best)',
                   color=COLORS['neuron_scaling'], alpha=0.7, edgecolor='black')

        # Add value labels on bars
        for i, b in enumerate(baseline_vals):
            ax.text(i - width, b + 0.01, f'{b:.3f}', ha='center', va='bottom', fontsize=14)
        if best_steering:
            for i, s in enumerate(steering_vals):
                ax.text(i, s + 0.01, f'{s:.3f}', ha='center', va='bottom', fontsize=14)
        if best_neuron:
            for i, n in enumerate(neuron_vals):
                ax.text(i + width, n + 0.01, f'{n:.3f}', ha='center', va='bottom', fontsize=14)

        ax.set_ylabel('Score', fontsize=14, weight='bold')
        ax.set_title(f'Layer Strategy: {LAYER_STRATEGY_DISPLAY_NAMES[layer_strategy]}', fontsize=16, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_best_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved best results comparison for {model_name}")


def main():
    """Generate all comparison plots."""
    print("\n" + "="*70)
    print("Bias Mitigation Methods Comparison - Plotting")
    print("="*70 + "\n")

    # Load baseline
    print("[1/4] Loading baseline results...")
    baselines = load_baseline_results()

    for model_name in MODELS:
        print(f"\n{'='*70}")
        print(f"Processing {MODEL_DISPLAY_NAMES[model_name]}")
        print(f"{'='*70}\n")

        # Create output directory
        output_dir = PROJECT_ROOT / "results" / "plots" / "method_comparison" / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load results
        print(f"[2/4] Loading results for {model_name}...")
        steering_results = load_steering_results(model_name)
        neuron_results = load_neuron_scaling_results(model_name)
        baseline = baselines[model_name]

        # Prepare data
        print("[3/4] Preparing data...")
        steering_data = prepare_steering_data(steering_results, baseline)
        neuron_data = prepare_neuron_scaling_data(neuron_results, baseline)

        # Generate plots
        print("[4/4] Generating plots...")

        # Pareto fronts for gender, age, and combined (scatter plots)
        plot_pareto_front(model_name, 'gender', steering_data, neuron_data, baseline, output_dir)
        plot_pareto_front(model_name, 'age', steering_data, neuron_data, baseline, output_dir)
        plot_combined_pareto_front(model_name, steering_data, neuron_data, baseline, output_dir)

        # Pareto fronts with regression curves
        plot_pareto_front_regression(model_name, 'gender', steering_data, neuron_data, baseline, output_dir)
        plot_pareto_front_regression(model_name, 'age', steering_data, neuron_data, baseline, output_dir)
        plot_combined_pareto_front_regression(model_name, steering_data, neuron_data, baseline, output_dir)

        # Layer strategy comparisons (steering) for gender, age, and combined
        plot_layer_strategy_comparison(model_name, 'gender', steering_data, neuron_data, baseline, output_dir)
        plot_layer_strategy_comparison(model_name, 'age', steering_data, neuron_data, baseline, output_dir)
        plot_combined_layer_strategy_comparison(model_name, steering_data, neuron_data, baseline, output_dir)

        # Neuron scaling plots (new format with layer strategies)
        plot_neuron_scaling_by_layer(model_name, 'gender', neuron_results, baseline, output_dir)
        plot_neuron_scaling_by_layer(model_name, 'age', neuron_results, baseline, output_dir)

        # Best results
        plot_best_results(model_name, steering_data, neuron_data, baseline, output_dir)

        print(f"\n[OK] All plots saved to: {output_dir}")

    print("\n" + "="*70)
    print("All plots generated successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
