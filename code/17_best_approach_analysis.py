import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from utils.paths import PROJECT_ROOT

# Configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

MODELS = ['bert', 'modern_bert']
BIAS_TYPES = ['gender', 'age', 'combined']

# Thresholds for valid configurations
TASK_THRESHOLD = 0.95  # Max 5% drop in task accuracy
SECONDARY_EPSILON = 0.02  # Max 2% increase in secondary bias

COLORS = {
    'neuron_scaling': '#A23B72',  # Purple
    'steering': '#2E86AB',  # Blue
    'baseline': '#E63946',  # Red
}


def load_baseline(model_name, bias_type):
    """Load baseline results for the given model and bias type."""
    if bias_type == 'combined':
        path = PROJECT_ROOT / "results" / "three_head_training_combined" / "performance_eval.json"
    elif bias_type == 'gender':
        path = PROJECT_ROOT / "results" / "two_head_training_gender" / "performance_eval.json"
    else:  # age
        path = PROJECT_ROOT / "results" / "two_head_training_age" / "performance_eval.json"

    with open(path, 'r') as f:
        data = json.load(f)

    return data[model_name]


def load_neuron_scaling_results(model_name, bias_type):
    """Load neuron scaling results."""
    path = (PROJECT_ROOT / "results" /
            f"neuron_scaling_bias_mitigation_{bias_type}" /
            model_name / "intervention_res.json")

    if not path.exists():
        return []

    with open(path, 'r') as f:
        data = json.load(f)

    # Filter out baseline entry
    return [d for d in data if d.get('mode') != 'baseline']


def load_steering_results(model_name, bias_type):
    """Load steering vector results."""
    path = PROJECT_ROOT / "results" / f"steering_vectors_{bias_type}" / model_name / "steering_results.json"

    if not path.exists():
        return []

    with open(path, 'r') as f:
        data = json.load(f)

    # Filter out baseline (coefficient = 0)
    return [d for d in data if d.get('coefficient', 0) > 0]


def get_metric_keys(bias_type):
    """Get the metric keys for the given bias type."""
    if bias_type == 'gender':
        return {
            'primary': 'gender_balanced_accuracy',
            'secondary': 'age_balanced_accuracy' if bias_type != 'gender' else None,
            'task': 'task_accuracy'
        }
    elif bias_type == 'age':
        return {
            'primary': 'age_balanced_accuracy',
            'secondary': 'gender_balanced_accuracy' if bias_type != 'age' else None,
            'task': 'task_accuracy'
        }
    else:  # combined
        return {
            'primary_gender': 'gender_balanced_accuracy',
            'primary_age': 'age_balanced_accuracy',
            'task': 'task_accuracy'
        }


def select_best_config(configs, baseline, bias_type):
    """
    Select best configuration using constrained optimization.

    Constraints:
    1. task_acc >= TASK_THRESHOLD * baseline_task
    2. secondary_bias <= baseline_secondary + SECONDARY_EPSILON (if applicable)

    Objective: Maximize primary bias reduction

    Returns:
        Best config dict or None if no valid configs
    """
    if not configs:
        return None

    baseline_task = baseline['average_task_accuracy']
    min_task = baseline_task * TASK_THRESHOLD

    # Set up metric extractors based on bias type
    if bias_type == 'gender':
        baseline_primary = baseline['average_gender_balanced_accuracy']
        # For gender-only results, we don't have age info (two-head model)
        # So we skip secondary constraint for single-attribute mitigation
        has_secondary = 'age_balanced_accuracy' in configs[0] if configs else False
        if has_secondary:
            baseline_secondary = baseline.get('average_age_balanced_accuracy', 1.0)
        else:
            baseline_secondary = None

        def get_primary(config):
            return config['gender_balanced_accuracy']

        def get_secondary(config):
            return config.get('age_balanced_accuracy', 0)

    elif bias_type == 'age':
        baseline_primary = baseline['average_age_balanced_accuracy']
        has_secondary = 'gender_balanced_accuracy' in configs[0] if configs else False
        if has_secondary:
            baseline_secondary = baseline.get('average_gender_balanced_accuracy', 1.0)
        else:
            baseline_secondary = None

        def get_primary(config):
            return config['age_balanced_accuracy']

        def get_secondary(config):
            return config.get('gender_balanced_accuracy', 0)

    else:  # combined
        baseline_gender = baseline['average_gender_balanced_accuracy']
        baseline_age = baseline['average_age_balanced_accuracy']
        baseline_primary = (baseline_gender + baseline_age) / 2
        baseline_secondary = None  # No secondary for combined

        def get_primary(config):
            return (config['gender_balanced_accuracy'] + config['age_balanced_accuracy']) / 2

        def get_secondary(config):
            return 0

    # Filter by task constraint: task_acc >= TASK_THRESHOLD * baseline_task
    valid_configs = [config for config in configs if config['task_accuracy'] >= min_task]

    # Filter by secondary constraint (if applicable): secondary_bias <= baseline_secondary + EPSILON
    if baseline_secondary is not None:
        valid_configs = [config for config in valid_configs 
                        if get_secondary(config) <= baseline_secondary + SECONDARY_EPSILON]

    if not valid_configs:
        # Fallback: return config with minimum task drop
        return max(configs, key=lambda config: config['task_accuracy'])

    # Among valid configs, select the one with best primary bias reduction
    # Lower primary bias = better debiasing; tie-breaker: higher task accuracy
    best = min(valid_configs, key=lambda config: (get_primary(config), -config['task_accuracy']))

    return best


def analyze_single_bias(model_name, bias_type):
    """Analyze a single bias type for a model."""
    baseline = load_baseline(model_name, bias_type)
    neuron_results = load_neuron_scaling_results(model_name, bias_type)
    steering_results = load_steering_results(model_name, bias_type)

    best_neuron = select_best_config(neuron_results, baseline, bias_type)
    best_steering = select_best_config(steering_results, baseline, bias_type)

    # Extract baseline metrics based on bias type
    baseline_task = baseline['average_task_accuracy']

    if bias_type == 'combined':
        baseline_gender = baseline['average_gender_balanced_accuracy']
        baseline_age = baseline['average_age_balanced_accuracy']
        baseline_primary = (baseline_gender + baseline_age) / 2

        def get_primary(config):
            """Calculate primary bias metric: average of gender and age."""
            if config is None:
                return baseline_primary
            return (config['gender_balanced_accuracy'] + config['age_balanced_accuracy']) / 2

        def get_gender(config):
            """Extract gender bias metric."""
            if config is None:
                return baseline_gender
            return config['gender_balanced_accuracy']

        def get_age(config):
            """Extract age bias metric."""
            if config is None:
                return baseline_age
            return config['age_balanced_accuracy']

    elif bias_type == 'gender':
        baseline_primary = baseline['average_gender_balanced_accuracy']
        baseline_gender = baseline_primary
        baseline_age = baseline.get('average_age_balanced_accuracy', None)

        def get_primary(config):
            """Primary bias metric: gender balanced accuracy."""
            if config is None:
                return baseline_primary
            return config['gender_balanced_accuracy']

        def get_gender(config):
            """Same as primary for gender-only mitigation."""
            return get_primary(config)

        def get_age(config):
            """Age metric (may not exist for two-head model)."""
            if config is None:
                return baseline_age
            return config.get('age_balanced_accuracy', baseline_age)

    else:  # age
        baseline_primary = baseline['average_age_balanced_accuracy']
        baseline_age = baseline_primary
        baseline_gender = baseline.get('average_gender_balanced_accuracy', None)

        def get_primary(config):
            """Primary bias metric: age balanced accuracy."""
            if config is None:
                return baseline_primary
            return config['age_balanced_accuracy']

        def get_age(config):
            """Same as primary for age-only mitigation."""
            return get_primary(config)

        def get_gender(config):
            """Gender metric (may not exist for two-head model)."""
            if config is None:
                return baseline_gender
            return config.get('gender_balanced_accuracy', baseline_gender)

    # Build result dict
    result = {
        'model': model_name,
        'bias_type': bias_type,
        'baseline': {
            'task_accuracy': baseline_task,
            'gender_balanced_accuracy': baseline_gender,
            'age_balanced_accuracy': baseline_age,
            'primary_bias': baseline_primary
        },
        'neuron_scaling': None,
        'steering': None,
        'winner': None
    }

    # Process neuron scaling
    if best_neuron:
        neuron_primary = get_primary(best_neuron)
        neuron_reduction = baseline_primary - neuron_primary
        result['neuron_scaling'] = {
            'config': {
                'mode': best_neuron.get('mode'),
                'scale': best_neuron.get('scale'),
                'coverage': best_neuron.get('coverage'),
                'layers': best_neuron.get('layers')
            },
            'task_accuracy': best_neuron['task_accuracy'],
            'task_drop': baseline_task - best_neuron['task_accuracy'],
            'task_drop_pct': (baseline_task - best_neuron['task_accuracy']) / baseline_task * 100,
            'gender_balanced_accuracy': get_gender(best_neuron),
            'age_balanced_accuracy': get_age(best_neuron),
            'primary_bias': neuron_primary,
            'primary_reduction': neuron_reduction,
            'primary_reduction_pct': neuron_reduction / baseline_primary * 100 if baseline_primary > 0 else 0
        }

    # Process steering
    if best_steering:
        steering_primary = get_primary(best_steering)
        steering_reduction = baseline_primary - steering_primary
        result['steering'] = {
            'config': {
                'coefficient': best_steering.get('coefficient'),
                'layers': best_steering.get('layers')
            },
            'task_accuracy': best_steering['task_accuracy'],
            'task_drop': baseline_task - best_steering['task_accuracy'],
            'task_drop_pct': (baseline_task - best_steering['task_accuracy']) / baseline_task * 100,
            'gender_balanced_accuracy': get_gender(best_steering),
            'age_balanced_accuracy': get_age(best_steering),
            'primary_bias': steering_primary,
            'primary_reduction': steering_reduction,
            'primary_reduction_pct': steering_reduction / baseline_primary * 100 if baseline_primary > 0 else 0
        }

    # Determine winner
    if result['neuron_scaling'] and result['steering']:
        n_red = result['neuron_scaling']['primary_reduction']
        s_red = result['steering']['primary_reduction']
        if n_red > s_red:
            result['winner'] = 'neuron_scaling'
        elif s_red > n_red:
            result['winner'] = 'steering'
        else:
            # Tie-breaker: higher task accuracy
            if result['neuron_scaling']['task_accuracy'] >= result['steering']['task_accuracy']:
                result['winner'] = 'neuron_scaling'
            else:
                result['winner'] = 'steering'
    elif result['neuron_scaling']:
        result['winner'] = 'neuron_scaling'
    elif result['steering']:
        result['winner'] = 'steering'

    return result


def plot_comparison(all_results, output_dir):
    """Create comparison plots for all models and bias types."""

    # Plot 1: Primary bias reduction comparison (bar chart)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Best Bias Reduction Method Comparison: Neuron Scaling vs Steering Vectors\n'
                 f'(Task threshold: {TASK_THRESHOLD*100:.0f}%)',
                 fontsize=16, weight='bold')

    for i, model in enumerate(MODELS):
        for j, bias_type in enumerate(BIAS_TYPES):
            ax = axes[i, j]
            result = all_results[model][bias_type]

            methods = ['Baseline', 'Neuron Scaling', 'Steering Vectors']
            task_accs = [result['baseline']['task_accuracy']]
            primary_biases = [result['baseline']['primary_bias']]
            colors_list = [COLORS['baseline']]

            if result['neuron_scaling']:
                task_accs.append(result['neuron_scaling']['task_accuracy'])
                primary_biases.append(result['neuron_scaling']['primary_bias'])
                colors_list.append(COLORS['neuron_scaling'])
            else:
                task_accs.append(0)
                primary_biases.append(0)
                colors_list.append('#cccccc')

            if result['steering']:
                task_accs.append(result['steering']['task_accuracy'])
                primary_biases.append(result['steering']['primary_bias'])
                colors_list.append(COLORS['steering'])
            else:
                task_accs.append(0)
                primary_biases.append(0)
                colors_list.append('#cccccc')

            x = np.arange(len(methods))
            width = 0.35

            bars1 = ax.bar(x - width/2, task_accs, width, label='Task Accuracy',
                           color=[c if c != '#cccccc' else '#cccccc' for c in colors_list],
                           alpha=0.7, edgecolor='black')
            bars2 = ax.bar(x + width/2, primary_biases, width, label=f'{bias_type.capitalize()} Bias',
                           color=[c if c != '#cccccc' else '#cccccc' for c in colors_list],
                           alpha=0.4, edgecolor='black', hatch='//')

            # Add value labels
            for bar, val in zip(bars1, task_accs):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9)
            for bar, val in zip(bars2, primary_biases):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9)

            # Mark winner
            if result['winner']:
                winner_idx = 1 if result['winner'] == 'neuron_scaling' else 2
                ax.annotate('★ BEST', xy=(winner_idx, 0.05), fontsize=12, color='green',
                            weight='bold', ha='center')

            ax.set_ylabel('Score', fontsize=11)
            ax.set_title(f'{model.upper()} - {bias_type.capitalize()} Bias', fontsize=12, weight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(methods, fontsize=10)
            ax.legend(loc='upper right', fontsize=9)
            ax.set_ylim([0, 1.0])
            ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'best_approach_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Bias Reduction Percentage (horizontal bar chart)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Bias Reduction (%)\nHigher = Better Debiasing', fontsize=16, weight='bold')

    for i, model in enumerate(MODELS):
        ax = axes[i]

        labels = []
        neuron_reductions = []
        steering_reductions = []

        for bias_type in BIAS_TYPES:
            result = all_results[model][bias_type]
            labels.append(bias_type.capitalize())

            if result['neuron_scaling']:
                neuron_reductions.append(result['neuron_scaling']['primary_reduction_pct'])
            else:
                neuron_reductions.append(0)

            if result['steering']:
                steering_reductions.append(result['steering']['primary_reduction_pct'])
            else:
                steering_reductions.append(0)

        y = np.arange(len(labels))
        height = 0.35

        bars1 = ax.barh(y - height/2, neuron_reductions, height, label='Neuron Scaling',
                        color=COLORS['neuron_scaling'], alpha=0.8, edgecolor='black')
        bars2 = ax.barh(y + height/2, steering_reductions, height, label='Steering Vectors',
                        color=COLORS['steering'], alpha=0.8, edgecolor='black')

        # Add value labels
        for bar, val in zip(bars1, neuron_reductions):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', ha='left', va='center', fontsize=10, weight='bold')
        for bar, val in zip(bars2, steering_reductions):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', ha='left', va='center', fontsize=10, weight='bold')

        ax.set_xlabel('Bias Reduction (%)', fontsize=12, weight='bold')
        ax.set_title(f'{model.upper()}', fontsize=14, weight='bold')
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=11)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim([0, max(max(neuron_reductions), max(steering_reductions)) * 1.2 + 5])

    plt.tight_layout()
    plt.savefig(output_dir / 'bias_reduction_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved comparison plots to {output_dir}")


def print_summary(all_results):
    """Print detailed console summary."""
    print("\n" + "="*80)
    print("BEST APPROACH ANALYSIS - SUMMARY")
    print("="*80)
    print(f"Task Threshold: {TASK_THRESHOLD*100:.0f}% of baseline")
    print(f"Secondary Epsilon: {SECONDARY_EPSILON*100:.0f}% max increase")
    print("="*80)

    overall_wins = {'neuron_scaling': 0, 'steering': 0}

    for model in MODELS:
        print(f"\n{'#'*80}")
        print(f"# MODEL: {model.upper()}")
        print(f"{'#'*80}")

        for bias_type in BIAS_TYPES:
            result = all_results[model][bias_type]
            print(f"\n  [{bias_type.upper()} BIAS]")
            print(f"  {'-'*60}")

            # Baseline
            baseline = result['baseline']
            print("  Baseline:")
            print(f"    Task Accuracy: {baseline['task_accuracy']:.4f}")
            if baseline['gender_balanced_accuracy'] is not None:
                print(f"    Gender Bal Acc: {baseline['gender_balanced_accuracy']:.4f}")
            if baseline['age_balanced_accuracy'] is not None:
                print(f"    Age Bal Acc: {baseline['age_balanced_accuracy']:.4f}")
            print(f"    Primary Bias: {baseline['primary_bias']:.4f}")

            # Neuron Scaling
            print("\n  Neuron Scaling (Best Config):")
            if result['neuron_scaling']:
                ns = result['neuron_scaling']
                cfg = ns['config']
                print(f"    Config: mode={cfg['mode']}, scale={cfg['scale']}, "
                      f"coverage={cfg['coverage']}, layers={cfg['layers']}")
                print(f"    Task Accuracy: {ns['task_accuracy']:.4f} "
                      f"(drop: {ns['task_drop_pct']:.2f}%)")
                print(f"    Primary Bias: {ns['primary_bias']:.4f} "
                      f"(reduction: {ns['primary_reduction_pct']:.2f}%)")
            else:
                print("    No valid configuration found")

            # Steering Vectors
            print("\n  Steering Vectors (Best Config):")
            if result['steering']:
                sv = result['steering']
                cfg = sv['config']
                print(f"    Config: coefficient={cfg['coefficient']}, layers={cfg['layers']}")
                print(f"    Task Accuracy: {sv['task_accuracy']:.4f} "
                      f"(drop: {sv['task_drop_pct']:.2f}%)")
                print(f"    Primary Bias: {sv['primary_bias']:.4f} "
                      f"(reduction: {sv['primary_reduction_pct']:.2f}%)")
            else:
                print("    No valid configuration found")

            # Winner
            print("\n  >>> WINNER: ", end="")
            if result['winner']:
                winner_name = result['winner'].replace('_', ' ').title()
                print(f"{winner_name} <<<")
                overall_wins[result['winner']] += 1
            else:
                print("No clear winner <<<")

    # Overall summary
    print("\n" + "="*80)
    print("OVERALL WINNER TALLY")
    print("="*80)
    print(f"  Neuron Scaling: {overall_wins['neuron_scaling']} wins")
    print(f"  Steering Vectors: {overall_wins['steering']} wins")
    print()

    if overall_wins['neuron_scaling'] > overall_wins['steering']:
        print("  >>> OVERALL BEST METHOD: NEURON SCALING <<<")
    elif overall_wins['steering'] > overall_wins['neuron_scaling']:
        print("  >>> OVERALL BEST METHOD: STEERING VECTORS <<<")
    else:
        print("  >>> OVERALL: TIE <<<")

    print("="*80 + "\n")


def main():
    """Run the best approach analysis."""
    start_time = datetime.now()
    print("\n" + "#"*80)
    print("# Best Approach Analysis: Neuron Scaling vs Steering Vectors")
    print(f"# Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("#"*80 + "\n")

    # Output directory
    output_dir = PROJECT_ROOT / "results" / "best_approach_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze all combinations
    all_results = {}

    for model in MODELS:
        print(f"[{model.upper()}] Analyzing...")
        all_results[model] = {}

        for bias_type in BIAS_TYPES:
            print(f"  Processing {bias_type} bias...")
            result = analyze_single_bias(model, bias_type)
            all_results[model][bias_type] = result

    # Save results to JSON
    results_file = output_dir / "best_approach_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Results saved to: {results_file}")

    # Generate plots
    print("\n[PLOTS] Generating comparison plots...")
    plot_comparison(all_results, output_dir)

    # Print console summary
    print_summary(all_results)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Analysis completed in {duration:.1f} seconds")
    print(f"Results saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
