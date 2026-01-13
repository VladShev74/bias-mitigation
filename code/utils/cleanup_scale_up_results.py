import json
from pathlib import Path

# Get PROJECT_ROOT directly (utils folder -> code folder -> project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def cleanup_scale_up_results():
    """Remove scale_up results from all neuron scaling experiments."""

    # Directories to process
    bias_types = ['age', 'gender', 'combined']
    models = ['bert', 'modern_bert']
    seeds = [42, 123, 1337]

    print("\n" + "="*70)
    print("Cleaning up scale_up results from neuron scaling experiments")
    print("="*70 + "\n")

    total_removed = 0

    for bias_type in bias_types:
        print(f"\n[{bias_type.upper()}] Processing...")

        results_dir = PROJECT_ROOT / "results" / f"neuron_scaling_bias_mitigation_{bias_type}"

        if not results_dir.exists():
            print(f"  [SKIP] Directory not found: {results_dir}")
            continue

        for model_name in models:
            model_dir = results_dir / model_name

            if not model_dir.exists():
                print(f"  [SKIP] Model directory not found: {model_name}")
                continue

            print(f"\n  [{model_name}]")

            # Delete aggregated results file
            aggregated_file = model_dir / "intervention_res.json"
            if aggregated_file.exists():
                aggregated_file.unlink()
                print(f"    [DELETED] Aggregated results: {aggregated_file.name}")
            else:
                print("    [SKIP] No aggregated results file found")

            # Process each seed
            for seed in seeds:
                seed_dir = model_dir / f"seed_{seed}"
                results_file = seed_dir / "intervention_res.json"

                if not results_file.exists():
                    print(f"    [SKIP] seed_{seed}: No results file")
                    continue

                # Load results
                with open(results_file, 'r') as f:
                    results = json.load(f)

                # Count and filter
                original_count = len(results)
                filtered_results = [r for r in results if r.get('mode') != 'scale_up']
                removed_count = original_count - len(filtered_results)

                if removed_count > 0:
                    # Save filtered results
                    with open(results_file, 'w') as f:
                        json.dump(filtered_results, f, indent=2)

                    print(f"    [OK] seed_{seed}: Removed {removed_count} scale_up entries "
                          f"({len(filtered_results)} remaining)")
                    total_removed += removed_count
                else:
                    print(f"    [OK] seed_{seed}: No scale_up entries found")

    print("\n" + "="*70)
    print(f"Cleanup complete! Total entries removed: {total_removed}")
    print("="*70 + "\n")


if __name__ == "__main__":
    cleanup_scale_up_results()
