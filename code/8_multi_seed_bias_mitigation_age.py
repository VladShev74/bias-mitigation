import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer
from utils.paths import PROJECT_ROOT, PAN16_PICKLE_DIR
from utils.models_config import MODEL_IDS
from utils.model_architectures import BertWithTwoHeadsAge


# Configuration
SEEDS = [42, 123, 1337]
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Intervention parameters
COVERAGE_LIST = [0.05, 0.1, 0.15, 0.2]  # Percentage of neurons to intervene on
SCALE_DOWN = [round(w, 1) for w in torch.arange(0.1, 1.0, 0.1).tolist()]
SCALE_UP = [round(w, 1) for w in torch.arange(1.1, 2.1, 0.1).tolist()]


def load_test_data():
    """Load test data for evaluation."""
    test_path = PAN16_PICKLE_DIR / "test.pkl"
    test_list = pd.read_pickle(test_path)
    test_df = pd.DataFrame(test_list)

    # Map age strings to integers
    age_mapping = {'18-24': 0, '25-34': 1, '35-49': 2, '50-64': 3, '65-xx': 4}
    test_df['age'] = test_df['age'].map(age_mapping)
    
    # Gender labels to binary
    test_df['gender'] = test_df['gender'].apply(lambda x: 1 if x == 'female' else 0)

    return test_df


def load_intervention_map(model_name):
    """Load neuron intervention map for the model."""
    map_path = PROJECT_ROOT / "results" / "activation_differences" / model_name / "neuron_map_age.json"

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
    results_path = PROJECT_ROOT / "results" / "two_head_training_age" / "performance_eval.json"

    with open(results_path, 'r') as f:
        all_results = json.load(f)

    model_results = all_results[model_name]

    return {
        "mode": "baseline",
        "scale": None,
        "coverage": None,
        "task_accuracy": model_results['average_task_accuracy'],
        "age_balanced_accuracy": model_results['average_age_balanced_accuracy']
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

        # Layer numbers in JSON are CSV layer indices (1-indexed for transformer layers)
        # Need to subtract 1 to get 0-indexed encoder layer indices
        encoder_layer_idx = layer_num - 1

        if model_name == "bert":
            handle = model.bert.encoder.layer[encoder_layer_idx].output.register_forward_hook(hook_fn)
        else:  # modern_bert
            handle = model.bert.layers[encoder_layer_idx].register_forward_hook(hook_fn)

        hooks.append(handle)

    return hooks


def evaluate_model(model, texts, tokenizer, task_labels, age_labels, model_name):
    """
    Evaluate model on test set.

    Returns:
        Tuple of (task_accuracy, age_balanced_accuracy)
    """
    model.eval()
    num_samples = len(texts)

    # Create batches
    all_task_preds = []
    all_age_preds = []

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

            # Forward pass - get both heads
            task_logits, age_logits = model(
                input_ids,
                attention_mask
            )

            task_preds = torch.argmax(task_logits, dim=1).cpu().numpy()
            age_preds = torch.argmax(age_logits, dim=1).cpu().numpy()

            all_task_preds.extend(task_preds)
            all_age_preds.extend(age_preds)

    # Convert to numpy arrays
    all_task_preds = np.array(all_task_preds)
    all_age_preds = np.array(all_age_preds)
    task_labels = np.array(task_labels)
    age_labels = np.array(age_labels)

    # Task accuracy (topic classification)
    task_accuracy = (all_task_preds == task_labels).mean()

    # Age balanced accuracy (average of per-class accuracies)
    age_acc_per_class = []
    for age_group in range(5):  # 5 age groups
        mask = age_labels == age_group
        if mask.sum() > 0:
            acc = (all_age_preds[mask] == age_labels[mask]).mean()
            age_acc_per_class.append(acc)

    age_balanced_accuracy = np.mean(age_acc_per_class)

    return float(task_accuracy), float(age_balanced_accuracy)


def run_experiment(model, texts, tokenizer, task_labels, age_labels, mode, scale, coverage,
                   intervention_map, model_name):
    """
    Run single intervention experiment.

    Args:
        model: The model to evaluate
        texts: Test texts
        tokenizer: Tokenizer
        task_labels: Task labels
        age_labels: Age labels
        mode: 'zero' or 'scale'
        scale: Scaling factor
        coverage: Fraction of neurons to intervene on
        intervention_map: Full neuron map
        model_name: 'bert' or 'modern_bert'

    Returns:
        Dict with task_accuracy and age_balanced_accuracy
    """
    # Determine number of neurons based on coverage
    total_neurons = 768  # Both BERT and Modern BERT have 768 hidden size
    num_neurons = int(total_neurons * coverage)

    # Create coverage map (top-k neurons per layer)
    coverage_map = {
        layer: neurons[:num_neurons]
        for layer, neurons in intervention_map.items()
    }

    # Register hooks
    hooks = register_hooks(model, coverage_map, mode, scale, model_name)

    # Evaluate
    task_acc, age_bal_acc = evaluate_model(model, texts, tokenizer, task_labels, age_labels, model_name)

    # Remove hooks
    for h in hooks:
        h.remove()

    return {
        "task_accuracy": task_acc,
        "age_balanced_accuracy": age_bal_acc
    }


def run_seed_experiments(model_name, seed, test_df, tokenizer, intervention_map):
    """Run all experiments for a single seed."""
    print(f"\n{'='*70}")
    print(f"Processing seed {seed} for {model_name}")
    print(f"{'='*70}")

    # Load model
    model_path = PROJECT_ROOT / "models" / "two_head_age" / model_name / f"seed_{seed}"
    model_id = MODEL_IDS[model_name]

    # Initialize model architecture (BertWithTwoHeadsAge works for both BERT and Modern BERT)
    model = BertWithTwoHeadsAge(model_id=model_id, num_task_labels=2, num_age_groups=5)

    # Load weights
    weights_path = model_path / "model_weights.pth"
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=False))
    model.to(DEVICE)
    model.eval()

    # Extract test data
    texts = test_df['text'].tolist()
    task_labels = test_df['task_label'].values
    age_labels = test_df['age'].values

    # Results for this seed
    seed_results = []

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

        for scale in scales:
            for coverage in tqdm(coverages, desc=f"  {mode} scale={scale}", leave=False):
                hook_mode = "scale" if mode in ["scale_down", "scale_up"] else "zero"

                result = run_experiment(
                    model, texts, tokenizer, task_labels, age_labels,
                    hook_mode, scale, coverage, current_map, model_name
                )

                entry = {
                    "mode": mode,
                    "scale": None if mode == "zero" else scale,
                    "coverage": coverage,
                    "task_accuracy": result["task_accuracy"],
                    "age_balanced_accuracy": result["age_balanced_accuracy"]
                }

                seed_results.append(entry)

    return seed_results


def aggregate_results(all_seed_results, baseline):
    """
    Aggregate results across seeds.

    Returns:
        List of aggregated results with mean values
    """
    aggregated = [baseline]  # Start with baseline

    # Group by (mode, scale, coverage)
    results_by_config = {}

    for seed_results in all_seed_results:
        for entry in seed_results:
            key = (entry["mode"], entry["scale"], entry["coverage"])

            if key not in results_by_config:
                results_by_config[key] = []

            results_by_config[key].append({
                "task_accuracy": entry["task_accuracy"],
                "age_balanced_accuracy": entry["age_balanced_accuracy"]
            })

    # Compute means
    for key, results in results_by_config.items():
        mode, scale, coverage = key

        mean_task_acc = np.mean([r["task_accuracy"] for r in results])
        mean_age_bal_acc = np.mean([r["age_balanced_accuracy"] for r in results])

        aggregated.append({
            "mode": mode,
            "scale": scale,
            "coverage": coverage,
            "task_accuracy": float(mean_task_acc),
            "age_balanced_accuracy": float(mean_age_bal_acc)
        })

    return aggregated


def main():
    """Main execution function."""
    start_time = datetime.now()
    print(f"\n{'#'*70}")
    print("# Multi-seed Age Bias Mitigation")
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
              f"Age Bal Acc: {baseline['age_balanced_accuracy']:.4f}")

        # Results storage
        all_seed_results = []

        # Run experiments for each seed
        for seed in SEEDS:
            seed_results = run_seed_experiments(model_name, seed, test_df, tokenizer, intervention_map)
            all_seed_results.append(seed_results)

            # Save individual seed results
            seed_dir = PROJECT_ROOT / "results" / "neuron_scaling_bias_mitigation_age" / model_name / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            with open(seed_dir / "intervention_res.json", 'w') as f:
                json.dump(seed_results, f, indent=2)

            print(f"[OK] Saved results for seed {seed}")

        # Aggregate across seeds
        print("\n[OK] Aggregating results across seeds...")
        aggregated_results = aggregate_results(all_seed_results, baseline)

        # Save aggregated results
        model_dir = PROJECT_ROOT / "results" / "neuron_scaling_bias_mitigation_age" / model_name
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
