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
from utils.model_architectures import BertWithTwoHeadsAge


# Configuration
SEEDS = [42, 123, 1337]
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Steering coefficients (0.0 = baseline, no steering)
COEFFICIENTS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]

# Layer strategies
LAYERS_STRATEGIES = ['all', 'first_half', 'second_half', 'top_3']


def load_test_data():
    """Load test data for evaluation."""
    test_path = PAN16_PICKLE_DIR / "test.pkl"
    test_list = pd.read_pickle(test_path)
    test_df = pd.DataFrame(test_list)

    # Map labels
    age_mapping = {'18-24': 0, '25-34': 1, '35-49': 2, '50-64': 3, '65-xx': 4}
    test_df['age'] = test_df['age'].map(age_mapping)
    test_df['gender'] = test_df['gender'].apply(lambda x: 1 if x == 'female' else 0)

    return test_df


def get_age_vector_pan16(model, tokenizer, test_df, n_samples=2000):
    """Compute age steering vector using difference-in-means on PAN16 test set."""
    print("  [OK] Computing age vector (difference-in-means)...")

    # Filter age groups: 0 = 18-24, 4 = 65-xx
    df_young = test_df[test_df['age'] == 0]
    df_old = test_df[test_df['age'] == 4]

    n_young = min(n_samples, len(df_young))
    n_old = min(n_samples, len(df_old))

    texts_young = df_young.sample(n=n_young, random_state=42)['text'].tolist()
    texts_old = df_old.sample(n=n_old, random_state=42)['text'].tolist()

    def get_centroid(texts):
        embeddings = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i+32]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                out = model.bert(**enc).last_hidden_state[:, 0, :]
                embeddings.append(out.cpu())
        return torch.cat(embeddings).mean(dim=0)

    mean_young = get_centroid(texts_young)
    mean_old = get_centroid(texts_old)

    # Compute unit-normalized steering vector (standard approach)
    v_age = mean_old - mean_young
    v_age = v_age / torch.norm(v_age)  # Unit normalization

    print("  [OK] Age vector computed (unit normalized)")
    return v_age.to(DEVICE)


class AgeSteeringHook:
    """Forward hook that applies age-only steering via simple subtraction.

    For BERT: steers only CLS token (index 0)
    For ModernBERT: steers all tokens for better coverage
    """
    def __init__(self, v_age, coefficient, steer_all_tokens=False):
        self.v_age = v_age
        self.coefficient = coefficient
        self.steer_all_tokens = steer_all_tokens

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        h_modified = hidden_states.clone()

        if self.steer_all_tokens:
            # ModernBERT: steer all tokens
            h_modified = h_modified - self.coefficient * self.v_age
        else:
            # BERT: steer only CLS token
            cls = h_modified[:, 0, :]  # [batch_size, 768]
            cls = cls - self.coefficient * self.v_age
            h_modified[:, 0, :] = cls

        if isinstance(output, tuple):
            return (h_modified,) + output[1:]
        return h_modified


def register_steering_hooks(model, v_age, coefficient, layers_strategy, model_name):
    """Register steering hooks on selected layers based on strategy."""
    hooks = []

    if model_name == "bert":
        all_layers = list(range(len(model.bert.encoder.layer)))
        steer_all_tokens = False  # BERT: CLS only
    else:  # modern_bert
        all_layers = list(range(len(model.bert.layers)))
        steer_all_tokens = True  # ModernBERT: all tokens

    # Select layers based on strategy
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

    # Register hooks on selected layers
    hook_fn = AgeSteeringHook(v_age, coefficient, steer_all_tokens)

    for layer_idx in selected_layers:
        if model_name == "bert":
            handle = model.bert.encoder.layer[layer_idx].output.register_forward_hook(hook_fn)
        else:  # modern_bert
            handle = model.bert.layers[layer_idx].register_forward_hook(hook_fn)
        hooks.append(handle)

    return hooks


def evaluate_model(model, texts, tokenizer, labels_dict):
    """Evaluate model on test set and return accuracy metrics."""
    model.eval()
    all_preds = {'task': [], 'age': []}

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Evaluating", leave=False):
            batch_texts = texts[i:i+BATCH_SIZE]

            enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
            input_ids = enc['input_ids'].to(DEVICE)
            attn_mask = enc['attention_mask'].to(DEVICE)

            with autocast(dtype=torch.float16):
                t_logits, a_logits = model(input_ids, attn_mask)

            all_preds['task'].extend(torch.argmax(t_logits, dim=1).cpu().numpy())
            all_preds['age'].extend(torch.argmax(a_logits, dim=1).cpu().numpy())

    preds_task = np.array(all_preds['task'])
    preds_age = np.array(all_preds['age'])

    labels_task = labels_dict['task']
    labels_age = labels_dict['age']

    metrics = {}

    # Task accuracy
    metrics['task_accuracy'] = float((preds_task == labels_task).mean())

    # Age accuracy (unbalanced)
    metrics['age_accuracy'] = float((preds_age == labels_age).mean())

    # Age balanced accuracy
    a_accs = [np.mean(preds_age[labels_age == a] == labels_age[labels_age == a]) for a in range(5)]
    metrics['age_balanced_accuracy'] = float(np.mean(a_accs))

    return metrics


def aggregate_and_save(all_seed_results, model_name):
    """Aggregate results across seeds and save to JSON."""
    # Group by (coefficient, layers)
    grouped = {}
    for seed_res in all_seed_results:
        for entry in seed_res:
            c = entry['coefficient']
            layers = entry.get('layers', 'top_3')
            key = (c, layers)

            if key not in grouped:
                grouped[key] = {k: [] for k in entry if k not in ['coefficient', 'layers']}

            for metric_key in grouped[key]:
                grouped[key][metric_key].append(entry[metric_key])

    # Compute statistics
    final_output = []

    # Sort by layers strategy, then coefficient
    layers_order = {"top_3": 0, "first_half": 1, "second_half": 2, "all": 3}
    sorted_keys = sorted(grouped.keys(), key=lambda x: (layers_order.get(x[1], 4), x[0]))

    for coeff, layers in sorted_keys:
        data = grouped[(coeff, layers)]

        # Extract metric values
        task_vals = data.get('task_accuracy', [])
        age_vals = data.get('age_accuracy', [])
        age_bal_vals = data.get('age_balanced_accuracy', [])

        obj = {
            'coefficient': coeff,
            'layers': layers,
            'task_accuracy': float(np.mean(task_vals)),
            'age_accuracy': float(np.mean(age_vals)),
            'age_balanced_accuracy': float(np.mean(age_bal_vals))
        }

        final_output.append(obj)

    # Save results
    out_dir = PROJECT_ROOT / "results" / "steering_vectors_age" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "steering_results.json"
    with open(out_path, 'w') as f:
        json.dump(final_output, f, indent=2)

    print(f"[OK] Saved results for {model_name}")


def main():
    """Run age-only steering vector bias mitigation experiments."""
    start_time = datetime.now()
    print(f"\n{'#'*70}")
    print("# Steering Vectors - Age-Only Bias Mitigation")
    print(f"# Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")

    # Load test data
    test_df = load_test_data()
    texts = test_df['text'].tolist()
    labels_dict = {
        'task': test_df['task_label'].values,
        'age': test_df['age'].values
    }

    for model_name, model_id in MODEL_IDS.items():
        print(f"\n{'='*70}")
        print(f"Processing {model_name}")
        print(f"{'='*70}\n")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model_results_dir = PROJECT_ROOT / "results" / "steering_vectors_age" / model_name
        all_seed_results = []

        for seed in SEEDS:
            print(f"\n[Seed {seed}] Starting evaluation...")

            # Setup results file for this seed
            seed_results_file = model_results_dir / f"seed_{seed}" / "steering_results.json"
            seed_results_file.parent.mkdir(parents=True, exist_ok=True)

            # Load existing results if available
            if seed_results_file.exists():
                print(f"[INFO] Loading existing results from {seed_results_file}")
                with open(seed_results_file, 'r') as f:
                    seed_results = json.load(f)
                completed = {(r['coefficient'], r.get('layers', 'top_3')) for r in seed_results}
            else:
                seed_results = []
                completed = set()

            # Load model (use two-head age model, same as neuron scaling)
            model_path = PROJECT_ROOT / "models" / "two_head_age" / model_name / f"seed_{seed}"
            model = BertWithTwoHeadsAge(model_id=model_id, num_task_labels=2, num_age_groups=5)

            weights = torch.load(model_path / "model_weights.pth", map_location=DEVICE, weights_only=False)
            model.load_state_dict(weights)
            model.to(DEVICE)
            model.eval()

            # Compute age steering vector only
            v_age = get_age_vector_pan16(model, tokenizer, test_df)

            # Sweep coefficients and layer strategies
            for layers_strategy in LAYERS_STRATEGIES:
                for coeff in COEFFICIENTS:
                    # Check if already computed
                    if (coeff, layers_strategy) in completed:
                        continue

                    # Register hooks on selected layers (age only)
                    hooks = register_steering_hooks(model, v_age, coeff, layers_strategy, model_name)

                    # Evaluate
                    metrics = evaluate_model(model, texts, tokenizer, labels_dict)

                    # Reorder fields: coefficient, layers, then metrics
                    ordered_metrics = {
                        'coefficient': coeff,
                        'layers': layers_strategy,
                        'task_accuracy': metrics['task_accuracy'],
                        'age_accuracy': metrics['age_accuracy'],
                        'age_balanced_accuracy': metrics['age_balanced_accuracy']
                    }

                    print(f"  [Coeff {coeff}, {layers_strategy}] Task: {ordered_metrics['task_accuracy']:.4f} | "
                          f"Age Bal: {ordered_metrics['age_balanced_accuracy']:.4f}")

                    seed_results.append(ordered_metrics)
                    completed.add((coeff, layers_strategy))

                    # Save intermediate results after each experiment
                    with open(seed_results_file, 'w') as f:
                        json.dump(seed_results, f, indent=2)

                    # Remove hooks
                    for handle in hooks:
                        handle.remove()

            all_seed_results.append(seed_results)
            print(f"[OK] Completed results for seed {seed}")

        # Aggregate across seeds
        aggregate_and_save(all_seed_results, model_name)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60

    print(f"\n{'#'*70}")
    print(f"# Pipeline completed in {duration:.1f} minutes")
    print(f"# Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
