import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import json
import matplotlib.pyplot as plt
from utils.paths import WINOGENDER_DATA, PROJECT_ROOT
from utils.models_config import MODEL_IDS


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128


def get_cls_embeddings(text, tokenizer, model):
    """
    Tokenize input text, pass through model, and extract CLS embeddings from all layers.
    Returns a list of tensors (one per layer).
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LENGTH)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states  # Tuple: (embedding_output, layer1_output, ..., layer_n_output)

    # Extract CLS token ([0] index) from each hidden state
    cls_embeddings = [state[:, 0, :].squeeze(0).cpu() for state in hidden_states]
    return cls_embeddings  # List of tensors (one per layer), each of shape (hidden_dim,)


def l2_distance(vec1, vec2):
    """Compute L2 (Euclidean) distance between two vectors."""
    return torch.norm(vec1 - vec2, p=2).item()


def analyze_gender_bias(model_name, model_id, df):
    """
    Analyze gender bias using counterfactual Winogender pairs.

    Args:
        model_name (str): Name of the model for display
        model_id (str): Hugging Face model ID
        df (pd.DataFrame): Winogender data with 'original_text' and 'counterfactual_text' columns

    Returns:
        dict: Dictionary containing layer indices and average L2 distances
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, output_hidden_states=True)
    model.to(DEVICE)
    model.eval()

    # Get the number of layers dynamically
    num_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer

    # Initialize a list to collect distances per layer
    layer_distances = [[] for _ in range(num_layers)]
    # Also collect per-layer neuron-wise absolute differences
    neuron_diffs_all = [[] for _ in range(num_layers)]

    # Loop through each counterfactual pair
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {model_name}"):
        original_text = row['original_text']
        counterfactual_text = row['counterfactual_text']

        # Get CLS embeddings for both texts
        cls_original = get_cls_embeddings(text=original_text, tokenizer=tokenizer, model=model)
        cls_counterfactual = get_cls_embeddings(text=counterfactual_text, tokenizer=tokenizer, model=model)
        # Calculate L2 distance and neuron-wise abs diff for each layer
        for layer_idx in range(num_layers):
            diff_vec = (cls_original[layer_idx] - cls_counterfactual[layer_idx]).abs()
            dist = torch.norm(diff_vec, p=2).item()
            layer_distances[layer_idx].append(dist)
            neuron_diffs_all[layer_idx].append(diff_vec.numpy())

    # Compute average distance per layer
    average_l2_distances = [np.mean(layer) for layer in layer_distances]

    return {
        'num_layers': num_layers,
        'distances': average_l2_distances,
        'neuron_diffs': neuron_diffs_all
    }


def save_results(model_name, results):
    """
    Save gender bias analysis results to CSV and PNG plot.

    Args:
        model_name (str): Name of the model
        results (dict): Dictionary with 'num_layers' and 'distances' keys
    """
    num_layers = results['num_layers']
    average_l2_distances = results['distances']

    # Create output directory for results
    results_dir = PROJECT_ROOT / "results" / "activation_differences" / model_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Check if results already exist
    csv_file = results_dir / "gender_signal.csv"
    plot_file = results_dir / "activation_diff_gender.png"
    if csv_file.exists() and plot_file.exists():
        print(f"[SKIPPING] Results already exist for {model_name}")
        return

    # Save to CSV
    csv_data = {
        'layer': list(range(num_layers)),
        'gender_l2': average_l2_distances
    }
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(csv_file, index=False)

    # Print results
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")
    print(f"Average L2 Distance per Layer (0 = Embedding, 1-{num_layers-1} = Transformer Layers):\n")
    for i, avg_dist in enumerate(average_l2_distances):
        print(f"  Layer {i:2d}: {avg_dist:.6f}")

    # Plot and save results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_layers), average_l2_distances[1:], marker='o', linewidth=2, markersize=6)
    plt.title(f"Gender Bias: L2 Distance between Original and Counterfactual CLS Embeddings\nModel: {model_name}",
              fontsize=12, fontweight='bold')
    plt.xlabel("Layer", fontsize=11)
    plt.ylabel("Average L2 Distance", fontsize=11)
    plt.xticks(range(1, num_layers))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plot_file = results_dir / "activation_diff_gender.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Results saved to: {csv_file}")
    print(f"[OK] Plot saved to: {plot_file}")
    plt.close()


def generate_neuron_intervention_map(model_name, results):
    """
    Generate and save neuron intervention maps for bias mitigation.

    Args:
        model_name (str): Name of the model
        results (dict): Dictionary with 'num_layers', 'distances', and 'neuron_diffs' keys
    """
    average_l2_distances = results['distances']
    neuron_diffs_all = results['neuron_diffs']

    # Create output directory for results
    results_dir = PROJECT_ROOT / "results" / "activation_differences" / model_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Check if neuron map already exists
    json_path = results_dir / "neuron_map_gender.json"
    if json_path.exists():
        print(f"[SKIPPING] Neuron intervention map already exists for {model_name}")
        return

    # Determine top-3 layers by average L2 distance (highest first)
    # Skip embedding layer (index 0) and use transformer layers only (1-12 for BERT, 1-22 for Modern BERT)
    transformer_distances = average_l2_distances[1:]  # Exclude embedding layer
    top_3_indices = np.argsort(transformer_distances)[-3:]  # Get top 3 indices in transformer_distances
    top_3_layers = sorted((top_3_indices + 1).tolist())

    print("\nGenerating neuron intervention maps...")
    print(f"Top-3 layers by L2 distance (CSV indices): {top_3_layers}")

    # Compute and save per-neuron average absolute difference for top-3 layers
    neuron_map_dict = {}
    for layer in top_3_layers:
        if len(neuron_diffs_all[layer]) == 0:
            continue
        diffs_matrix = np.stack(neuron_diffs_all[layer], axis=0)  # (num_samples, hidden_dim)
        avg_abs_diff = diffs_matrix.mean(axis=0)  # (hidden_dim,)

        # Get sorted neuron indices (descending by importance)
        sorted_indices = np.argsort(avg_abs_diff)[::-1]
        # Store as dict with neuron_id: importance_score for transparency
        neuron_map_dict[layer] = {
            int(neuron_id): float(avg_abs_diff[neuron_id])
            for neuron_id in sorted_indices
        }

    # Save JSON intervention map (for direct reuse) with attribute name
    with open(json_path, 'w') as f:
        json.dump(neuron_map_dict, f, indent=2)
    print(f"  [OK] Saved neuron intervention map: {json_path}")
    print("       Ready for use as INTERVENTION_MAP in bias mitigation experiments")


# Main execution
if __name__ == "__main__":
    df = pd.read_csv(WINOGENDER_DATA)
    print(f"[OK] Loaded Winogender data: {len(df)} counterfactual pairs\n")

    for model_name, model_id in MODEL_IDS.items():
        results = analyze_gender_bias(model_name, model_id, df)
        save_results(model_name, results)
        generate_neuron_intervention_map(model_name, results)
