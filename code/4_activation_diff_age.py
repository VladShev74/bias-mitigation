import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

from utils.paths import PROJECT_ROOT
from utils.models_config import MODEL_IDS

# Age class mapping
AGE_GROUPS = ["18-24", "25-34", "35-49", "50-64", "65-xx"]
AGE_GROUP_MAP = {age: idx for idx, age in enumerate(AGE_GROUPS)}


def _layer_index_from_name(name: str) -> int:
    """Extract layer index from filename like 'X_train_layer5.npy'."""
    m = re.search(r"layer(\d+)\.npy$", name)
    if not m:
        raise ValueError(f"Cannot parse layer index: {name}")
    return int(m.group(1))


def load_embeddings(split: str, model_name: str) -> tuple[list[np.ndarray], np.ndarray]:
    """Load embeddings and age labels from disk."""
    data_dir = PROJECT_ROOT / "data" / "pan16_embeddings" / "age" / model_name / split

    # Load embeddings per layer
    layer_files = sorted(
        [p for p in data_dir.glob(f"X_{split}_layer*.npy")],
        key=lambda p: _layer_index_from_name(p.name),
    )
    X_layers = [np.load(p) for p in layer_files]

    # Load labels
    y = pd.read_pickle(data_dir / f"y_{split}.pkl")

    # Convert to numpy array
    if isinstance(y, pd.Series):
        y = y.values

    # Convert string labels to integers if needed
    if isinstance(y[0], str):
        y = np.array([AGE_GROUP_MAP[label] for label in y], dtype=int)
    else:
        y = np.array(y, dtype=int)

    return X_layers, y


def compute_group_centroids(X: np.ndarray, y: np.ndarray, n_groups: int = 5) -> dict[int, np.ndarray]:
    """Compute mean embedding for each age group."""
    centroids = {}
    for group_id in range(n_groups):
        mask = (y == group_id)
        if mask.sum() == 0:
            raise ValueError(f"No samples in group {group_id}")
        centroids[group_id] = X[mask].mean(axis=0)
    return centroids


def l2_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute L2 distance between two vectors."""
    return float(np.linalg.norm(vec1 - vec2))


def avg_pairwise_distance(centroids: dict[int, np.ndarray]) -> float:
    """Average L2 distance between all pairs of centroids."""
    distances = []
    group_ids = sorted(centroids.keys())
    for i in range(len(group_ids)):
        for j in range(i + 1, len(group_ids)):
            d = l2_distance(centroids[group_ids[i]], centroids[group_ids[j]])
            distances.append(d)
    return float(np.mean(distances))


def plot_l2_distances(distances: list[float], model_name: str, save_dir: Path):
    """Plot L2 distances per layer."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(distances)), distances, marker='o', linewidth=2, markersize=6)
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Avg Pairwise L2 Distance (Age Centroids)", fontsize=12)
    plt.title(f"Age Activation Differences Across Layers\nModel: {model_name}", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = save_dir / "activation_diff_age.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


def analyze_model(model_name: str):
    """Analyze age signal in a single model."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    # Load embeddings
    print("Loading embeddings...")
    X_train_layers, y_train = load_embeddings("train", model_name)
    X_val_layers, y_val = load_embeddings("val", model_name)
    n_layers = len(X_train_layers)
    print(f"Loaded {n_layers} layers from train ({len(y_train):,}) and val ({len(y_val):,}) sets")

    # Compute L2 distances between age group centroids
    print("\nComputing activation differences...")
    train_distances = []
    val_distances = []

    for li in range(n_layers):
        # Train set distances
        centroids_train = compute_group_centroids(X_train_layers[li], y_train, n_groups=5)
        train_dist = avg_pairwise_distance(centroids_train)
        train_distances.append(train_dist)

        # Val set distances
        centroids_val = compute_group_centroids(X_val_layers[li], y_val, n_groups=5)
        val_dist = avg_pairwise_distance(centroids_val)
        val_distances.append(val_dist)

        print(f"  Layer {li:2d}/{n_layers-1}: Train L2={train_dist:.4f} | Val L2={val_dist:.4f}")

    # Save results
    results_dir = PROJECT_ROOT / "results" / "activation_differences" / model_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    df = pd.DataFrame({
        "layer": list(range(n_layers)),
        "train_l2": train_distances,
        "val_l2": val_distances,
    })
    csv_path = results_dir / "age_signal.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Save plot
    plot_path = plot_l2_distances(train_distances, model_name, results_dir)
    print(f"Plot saved to: {plot_path}")

    # Summary
    top_layer_train = int(np.argmax(train_distances))
    top_layer_val = int(np.argmax(val_distances))
    print(f"\nTop layer (Train): {top_layer_train} (L2={train_distances[top_layer_train]:.4f})")
    print(f"Top layer (Val):   {top_layer_val} (L2={val_distances[top_layer_val]:.4f})")
    print(f"{'='*60}")


def main():
    """Analyze age activation differences for all models."""
    print(f"Project root: {PROJECT_ROOT}")
    for model_name in MODEL_IDS.keys():
        try:
            analyze_model(model_name)
        except Exception as e:
            print(f"ERROR processing {model_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
