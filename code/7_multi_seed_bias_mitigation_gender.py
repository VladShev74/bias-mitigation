import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer
from torch.cuda.amp import autocast
from utils.paths import PROJECT_ROOT, PAN16_PICKLE_DIR
from utils.models_config import MODEL_IDS
from utils.model_architectures import BertWithTwoHeads

# Configuration
SEEDS = [42, 123, 1337]
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Intervention parameters
COVERAGE_LIST = [0.05, 0.1, 0.15, 0.2]  # Percentage of neurons to intervene on
SCALE_DOWN = [round(w, 1) for w in torch.arange(0.1, 1.0, 0.1).tolist()]
SCALE_UP = [round(w, 1) for w in torch.arange(1.1, 2.1, 0.1).tolist()]
LAYERS_STRATEGIES = ['all', 'first_half', 'second_half', 'top_3']


def load_test_data():
    """Load test data for evaluation."""
    test_path = PAN16_PICKLE_DIR / "test.pkl"
    test_list = pd.read_pickle(test_path)
    test_df = pd.DataFrame(test_list)

    # Convert gender labels to binary
    test_df['gender'] = test_df['gender'].apply(lambda x: 1 if x == 'female' else 0)

    return test_df


def load_intervention_map(model_name):
    """Load neuron intervention map for the model."""
    map_path = PROJECT_ROOT / "results" / "activation_differences" / model_name / "neuron_map_gender.json"

    with open(map_path, 'r') as f:
        intervention_map = json.load(f)

    # Convert string keys to integers and dict values to lists
    processed_map = {}
    for layer_str, neuron_dict in intervention_map.items():
        layer_num = int(layer_str)
        # Get neuron IDs sorted by importance (already sorted in the dict)
        neuron_list = [int(n) for n in neuron_dict.keys()]
        processed_map[layer_num] = neuron_list

    return processed_map


def load_baseline_results(model_name):
    """Load baseline performance from training results."""
    results_path = PROJECT_ROOT / "results" / "two_head_training_gender" / "performance_eval.json"

    with open(results_path, 'r') as f:
        all_results = json.load(f)

    model_results = all_results[model_name]

    return {
        "mode": "baseline",
        "scale": None,
        "coverage": None,
        "task_accuracy": model_results['average_task_accuracy'],
        "gender_accuracy": model_results['average_gender_accuracy'],
        "gender_balanced_accuracy": model_results['average_gender_balanced_accuracy']
    }


def get_hook_fn(neuron_indices, mode, scale):
    """
    Create hook function for neuron intervention.

    Args:
        neuron_indices: List of neuron indices to intervene on
        mode: 'zero' to zero out neurons, 'scale' to scale them
        scale: Scaling factor (ignored if mode is 'zero')
    """
    def hook(module, input, output):
        # For BERT: output is tuple (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Clone to avoid in-place modification issues
        hidden_states = hidden_states.clone()
        cls_output = hidden_states[:, 0, :]  # CLS token

        for idx in neuron_indices:
            if mode == "zero":
                cls_output[:, idx] = 0.0
            elif mode == "scale":
                cls_output[:, idx] *= scale

        hidden_states[:, 0, :] = cls_output

        # Return in the same format as received
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        else:
            return hidden_states

    return hook


def register_hooks(model, intervention_map, mode, scale, model_name):
    """
    Register forward hooks on model layers.

    Args:
        model: The model to register hooks on
        intervention_map: Dict mapping layer numbers to neuron lists
        mode: 'zero' or 'scale'
        scale: Scaling factor
        model_name: 'bert' or 'modern_bert'

    Returns:
        List of hook handles
    """
    hooks = []

    for layer_num, neurons in intervention_map.items():
        hook_fn = get_hook_fn(neurons, mode, scale)

        # Layer numbers in JSON are CSV indices (1-indexed after embedding)
        # Convert to encoder layer index: layer_num - 1
        encoder_layer_idx = layer_num - 1

        if model_name == "bert":
            handle = model.bert.encoder.layer[encoder_layer_idx].output.register_forward_hook(hook_fn)
        else:  # modern_bert
            handle = model.bert.layers[encoder_layer_idx].register_forward_hook(hook_fn)

        hooks.append(handle)

    return hooks


def evaluate_model(model, texts, tokenizer, task_labels, gender_labels, model_name):
    """
    Evaluate model on test set.

    Returns:
        Tuple of (task_accuracy, gender_accuracy, gender_balanced_accuracy)
    """
    model.eval()
    num_samples = len(texts)

    # Create batches
    all_task_preds = []
    all_gender_preds = []

    with torch.no_grad():
        for i in tqdm(range(0, num_samples, BATCH_SIZE), desc="Evaluating", leave=False):
            end_idx = min(i + BATCH_SIZE, num_samples)
            batch_texts = texts[i:end_idx]

            # Tokenize batch
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(DEVICE)
            attention_mask = encoded['attention_mask'].to(DEVICE)

            # Forward pass with FP16 mixed precision
            with autocast(dtype=torch.float16):
                task_logits, gender_logits = model(
                    input_ids,
                    attention_mask
                )

            task_preds = torch.argmax(task_logits, dim=1).cpu().numpy()
            gender_preds = torch.argmax(gender_logits, dim=1).cpu().numpy()

            all_task_preds.extend(task_preds)
            all_gender_preds.extend(gender_preds)

    # Convert to numpy arrays
    all_task_preds = np.array(all_task_preds)
    all_gender_preds = np.array(all_gender_preds)
    task_labels = np.array(task_labels)
    gender_labels = np.array(gender_labels)

    # Task accuracy
    task_accuracy = (all_task_preds == task_labels).mean()

    # Gender accuracy (unbalanced)
    gender_accuracy = (all_gender_preds == gender_labels).mean()

    # Gender balanced accuracy (average of per-class accuracies)
    gender_acc_per_class = []
    for gender in [0, 1]:
        mask = gender_labels == gender
        if mask.sum() > 0:
            acc = (all_gender_preds[mask] == gender_labels[mask]).mean()
            gender_acc_per_class.append(acc)

    gender_balanced_accuracy = np.mean(gender_acc_per_class)

    return float(task_accuracy), float(gender_accuracy), float(gender_balanced_accuracy)


def run_experiment(model,
                   texts,
                   tokenizer,
                   task_labels,
                   gender_labels,
                   mode,
                   scale,
                   coverage,
                   layers_strategy,
                   intervention_map,
                   model_name):
    """
    Run single intervention experiment.

    Args:
        model: The model to evaluate
        texts: Test texts
        tokenizer: Tokenizer
        task_labels: Task labels
        gender_labels: Gender labels
        mode: 'zero' or 'scale'
        scale: Scaling factor
        coverage: Fraction of neurons to intervene on
        layers_strategy: Which layers to intervene on ('all', 'first_half', 'second_half', 'top_3')
        intervention_map: Full neuron map
        model_name: 'bert' or 'modern_bert'

    Returns:
        Dict with task_accuracy, gender_accuracy, and gender_balanced_accuracy
    """
    # Determine number of neurons based on coverage
    total_neurons = 768  # Both BERT and Modern BERT have 768 hidden size
    num_neurons = int(total_neurons * coverage)

    # Filter layers based on strategy
    all_layers = sorted(intervention_map.keys())
    if layers_strategy == 'all':
        selected_layers = all_layers
    elif layers_strategy == 'first_half':
        mid_point = len(all_layers) // 2
        selected_layers = all_layers[:mid_point]
    elif layers_strategy == 'second_half':
        mid_point = len(all_layers) // 2
        selected_layers = all_layers[mid_point:]
    elif layers_strategy == 'top_3':
        selected_layers = all_layers[-3:]
    else:
        raise ValueError(f"Unknown layers strategy: {layers_strategy}")

    # Create coverage map (top-k neurons per selected layer)
    coverage_map = {
        layer: neurons[:num_neurons]
        for layer, neurons in intervention_map.items()
        if layer in selected_layers
    }

    # Register hooks
    hooks = register_hooks(model, coverage_map, mode, scale, model_name)

    # Evaluate
    task_acc, gender_acc, gender_bal_acc = evaluate_model(model,
                                                          texts,
                                                          tokenizer,
                                                          task_labels,
                                                          gender_labels,
                                                          model_name)

    # Remove hooks
    for h in hooks:
        h.remove()

    return {
        "task_accuracy": task_acc,
        "gender_accuracy": gender_acc,
        "gender_balanced_accuracy": gender_bal_acc
    }


def run_seed_experiments(model_name, seed, test_df, tokenizer, intervention_map):
    """Run all experiments for a single seed."""
    print(f"\n{'='*70}")
    print(f"Processing seed {seed} for {model_name}")
    print(f"{'='*70}")

    # Setup results directory and file
    seed_dir = PROJECT_ROOT / "results" / "neuron_scaling_bias_mitigation_gender" / model_name / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    results_file = seed_dir / "intervention_res.json"

    # Load existing results if available
    if results_file.exists():
        print(f"[INFO] Loading existing results from {results_file}")
        with open(results_file, 'r') as f:
            seed_results = json.load(f)
        # Create set of completed configs for quick lookup
        completed = {(r['mode'], r['scale'], r['coverage'], r.get('layers', 'top_3')) for r in seed_results}
    else:
        seed_results = []
        completed = set()

    # Load model
    model_path = PROJECT_ROOT / "models" / "two_head_gender" / model_name / f"seed_{seed}"
    model_id = MODEL_IDS[model_name]

    # Initialize model architecture (BertWithTwoHeads works for both BERT and Modern BERT)
    model = BertWithTwoHeads(model_id=model_id, num_task_labels=2)

    # Load weights
    weights_path = model_path / "model_weights.pth"
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=False))
    model.to(DEVICE)
    model.eval()

    # Extract test data
    texts = test_df['text'].tolist()
    task_labels = test_df['task_label'].values
    gender_labels = test_df['gender'].values

    # Define interventions
    interventions = [
        {"mode": "zero", "scales": [1.0], "coverages": COVERAGE_LIST},
        {"mode": "scale_down", "scales": SCALE_DOWN, "coverages": COVERAGE_LIST},
        {"mode": "scale_up", "scales": SCALE_UP, "coverages": COVERAGE_LIST}
    ]

    for intervention in interventions:
        mode = intervention["mode"]
        scales = intervention["scales"]
        coverages = intervention["coverages"]

        # Reverse neuron order for upscaling (least important first)
        if mode == "scale_up":
            current_map = {layer: list(reversed(neurons)) for layer, neurons in intervention_map.items()}
        else:
            current_map = intervention_map

        print(f"\n  Running intervention: {mode}")

        for layers_strategy in LAYERS_STRATEGIES:
            for scale in scales:
                for coverage in tqdm(coverages, desc=f"  {mode} {layers_strategy} scale={scale}", leave=False):
                    # Check if this config was already computed
                    config_key = (mode, None if mode == "zero" else scale, coverage, layers_strategy)
                    if config_key in completed:
                        continue

                    hook_mode = "scale" if mode in ["scale_down", "scale_up"] else "zero"

                    result = run_experiment(
                        model, texts, tokenizer, task_labels, gender_labels,
                        hook_mode, scale, coverage, layers_strategy, current_map, model_name
                    )

                    entry = {
                        "mode": mode,
                        "scale": None if mode == "zero" else scale,
                        "coverage": coverage,
                        "layers": layers_strategy,
                        "task_accuracy": result["task_accuracy"],
                        "gender_accuracy": result["gender_accuracy"],
                        "gender_balanced_accuracy": result["gender_balanced_accuracy"]
                    }

                    seed_results.append(entry)
                    completed.add(config_key)

                    # Save intermediate results after each experiment
                    with open(results_file, 'w') as f:
                        json.dump(seed_results, f, indent=2)

    return seed_results


def aggregate_results(all_seed_results, baseline):
    """
    Aggregate results across seeds.

    Returns:
        List of aggregated results with mean values
    """
    # Group by (mode, scale, coverage, layers)
    results_by_config = {}

    for seed_results in all_seed_results:
        for entry in seed_results:
            key = (entry["mode"], entry["scale"], entry["coverage"], entry.get("layers", "top_3"))

            if key not in results_by_config:
                results_by_config[key] = []

            results_by_config[key].append({
                "task_accuracy": entry["task_accuracy"],
                "gender_accuracy": entry["gender_accuracy"],
                "gender_balanced_accuracy": entry["gender_balanced_accuracy"]
            })

    # Start with baseline
    aggregated = [baseline]

    # Sort configs by intervention type order: zero, scale_down, scale_up, then by layers, scale, coverage
    intervention_order = {"zero": 0, "scale_down": 1, "scale_up": 2}
    layers_order = {"top_3": 0, "first_half": 1, "second_half": 2, "all": 3}
    sorted_configs = sorted(
        results_by_config.keys(),
        key=lambda x: (intervention_order.get(x[0], 3), layers_order.get(x[3], 4),
                       x[1] or 0, x[2])
    )

    # Compute means for each config
    for key in sorted_configs:
        mode, scale, coverage, layers = key
        results = results_by_config[key]

        mean_task_acc = np.mean([r["task_accuracy"] for r in results])
        mean_gender_acc = np.mean([r["gender_accuracy"] for r in results])
        mean_gender_bal_acc = np.mean([r["gender_balanced_accuracy"] for r in results])

        aggregated.append({
            "mode": mode,
            "scale": scale,
            "coverage": coverage,
            "layers": layers,
            "task_accuracy": float(mean_task_acc),
            "gender_accuracy": float(mean_gender_acc),
            "gender_balanced_accuracy": float(mean_gender_bal_acc)
        })

    return aggregated


def main():
    """Main execution function."""
    start_time = datetime.now()
    print(f"\n{'#'*70}")
    print("# Multi-seed Gender Bias Mitigation")
    print(f"# Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")

    # Process each model
    for model_name, model_id in MODEL_IDS.items():
        print(f"\n{'#'*70}")
        print(f"# Processing Model: {model_name}")
        print(f"{'#'*70}")

        # Load test data and tokenizer
        print("[OK] Loading test data...")
        test_df = load_test_data()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"[OK] Test samples: {len(test_df)}")

        # Load intervention map
        print("[OK] Loading intervention map...")
        intervention_map = load_intervention_map(model_name)
        print(f"[OK] Intervention layers: {sorted(intervention_map.keys())}")

        # Load baseline
        baseline = load_baseline_results(model_name)
        print(f"[OK] Baseline - Task Acc: {baseline['task_accuracy']:.4f}, "
              f"Gender Bal Acc: {baseline['gender_balanced_accuracy']:.4f}")

        # Results storage
        all_seed_results = []

        # Run experiments for each seed
        for seed in SEEDS:
            seed_results = run_seed_experiments(model_name, seed, test_df, tokenizer, intervention_map)
            all_seed_results.append(seed_results)

            print(f"[OK] Completed results for seed {seed}")

        # Aggregate across seeds
        print("\n[OK] Aggregating results across seeds...")
        aggregated_results = aggregate_results(all_seed_results, baseline)

        # Save aggregated results
        model_dir = PROJECT_ROOT / "results" / "neuron_scaling_bias_mitigation_gender" / model_name
        with open(model_dir / "intervention_res.json", 'w') as f:
            json.dump(aggregated_results, f, indent=2)

        print(f"[OK] Saved aggregated results for {model_name}")
        print(f"[OK] Total configurations: {len(aggregated_results)} (including baseline)")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60

    print(f"\n{'#'*70}")
    print(f"# Pipeline completed in {duration:.1f} minutes")
    print(f"# Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
