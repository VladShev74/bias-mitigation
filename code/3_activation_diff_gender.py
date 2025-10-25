import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
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

    # Loop through each counterfactual pair
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {model_name}"):
        original_text = row['original_text']
        counterfactual_text = row['counterfactual_text']

        # Get CLS embeddings for both texts
        cls_original = get_cls_embeddings(text=original_text, tokenizer=tokenizer, model=model)
        cls_counterfactual = get_cls_embeddings(text=counterfactual_text, tokenizer=tokenizer, model=model)
        # Calculate L2 distance for each layer
        for layer_idx in range(num_layers):
            dist = l2_distance(cls_original[layer_idx], cls_counterfactual[layer_idx])
            layer_distances[layer_idx].append(dist)

    # Compute average distance per layer
    average_l2_distances = [np.mean(layer) for layer in layer_distances]

    return {
        'num_layers': num_layers,
        'distances': average_l2_distances
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

    # Save to CSV
    csv_data = {
        'layer': list(range(num_layers)),
        'gender_l2': average_l2_distances
    }
    csv_df = pd.DataFrame(csv_data)
    csv_file = results_dir / "gender_signal.csv"
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


# Main execution
if __name__ == "__main__":
    df = pd.read_csv(WINOGENDER_DATA)
    print(f"[OK] Loaded Winogender data: {len(df)} counterfactual pairs\n")

    for model_name, model_id in MODEL_IDS.items():
        results = analyze_gender_bias(model_name, model_id, df)
        save_results(model_name, results)
