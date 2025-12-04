import json
import numpy as np
from utils.paths import PROJECT_ROOT
from utils.models_config import MODEL_IDS


def combine_neuron_maps_zscore(model_name):
    """Combine gender and age neuron maps using z-score normalization."""
    input_dir = PROJECT_ROOT / "results" / "activation_differences" / model_name
    output_dir = input_dir

    # Load neuron maps
    with open(input_dir / "neuron_map_gender.json", 'r') as f:
        gender_map = json.load(f)
    with open(input_dir / "neuron_map_age.json", 'r') as f:
        age_map = json.load(f)

    combined_map = {}
    layers = sorted(gender_map.keys(), key=int)

    print(f"  [OK] Combining maps for {model_name} using z-score normalization...")

    for layer in layers:
        g_neurons = gender_map[layer]
        a_neurons = age_map[layer]

        # Extract values for statistics
        g_vals = np.array(list(g_neurons.values()))
        a_vals = np.array(list(a_neurons.values()))

        # Calculate mean and standard deviation per layer
        g_mean, g_std = np.mean(g_vals), np.std(g_vals)
        a_mean, a_std = np.mean(a_vals), np.std(a_vals)

        # Avoid division by zero
        g_std = g_std if g_std > 1e-9 else 1.0
        a_std = a_std if a_std > 1e-9 else 1.0

        layer_combined = {}
        all_neuron_indices = set(g_neurons.keys()) | set(a_neurons.keys())

        for idx in all_neuron_indices:
            raw_g = g_neurons.get(idx, 0.0)
            raw_a = a_neurons.get(idx, 0.0)

            # Compute z-scores
            z_g = (raw_g - g_mean) / g_std
            z_a = (raw_a - a_mean) / a_std

            # Combine z-scores
            layer_combined[idx] = float(z_g + z_a)

        # Sort by combined importance (descending)
        sorted_neurons = dict(sorted(layer_combined.items(), key=lambda item: item[1], reverse=True))
        combined_map[layer] = sorted_neurons

    # Save combined map
    output_path = output_dir / "neuron_map_combined.json"
    with open(output_path, 'w') as f:
        json.dump(combined_map, f, indent=2)

    print(f"  [OK] Saved combined map to: {output_path}")


def main():
    """Main execution function."""
    print(f"\n{'#'*70}")
    print("# Combining Gender and Age Neuron Maps")
    print(f"{'#'*70}\n")

    for model_name in MODEL_IDS.keys():
        print(f"[OK] Processing model: {model_name}")
        try:
            combine_neuron_maps_zscore(model_name)
        except FileNotFoundError as e:
            print(f"[ERROR] Skipping {model_name}: {e}")

    print(f"\n{'#'*70}")
    print("# Combination complete")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
