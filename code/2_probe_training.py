import pandas as pd
from transformers import AutoModel
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
from utils.models_config import MODEL_IDS
from utils.paths import PROJECT_ROOT

# Suppress convergence warnings
warnings.filterwarnings('ignore', category=UserWarning)


def load_embeddings(split: str, model_name: str, label_column: str):
    X_layers = []
    storage_dir = PROJECT_ROOT / "data" / "pan16_embeddings" / label_column / model_name

    for i in range(13):
        path = storage_dir / split / f"X_{split}_layer{i}.npy"
        X = np.load(path)
        X_layers.append(X)

    y_path = storage_dir / split / f"y_{split}.pkl"
    y = pd.read_pickle(y_path)
    y = np.array(y)

    return X_layers, y


for model_name, model_id in MODEL_IDS.items():
    for label_column in ["gender", "age"]:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}, Task: {label_column}")
        print(f"{'='*60}\n")

        # Create output directory for results
        results_dir = PROJECT_ROOT / "results" / "log_reg_probe" / model_name / label_column
        results_dir.mkdir(parents=True, exist_ok=True)

        print("Loading training data...")
        X_train_layers, y_train = load_embeddings("train", model_name, label_column)
        print("Training data loaded:", [x.shape for x in X_train_layers])

        print("Loading validation data...")
        X_val_layers, y_val = load_embeddings("val", model_name, label_column)
        print("Validation data loaded:", [x.shape for x in X_val_layers])

        print("\nTraining and evaluating probing classifiers:\n")

        model = AutoModel.from_pretrained(model_id)

        # Store results
        results = []

        for i in range(model.config.num_hidden_layers + 1):
            print(f"Layer {i} of Model {model_name}:", end=" ")

            clf = LogisticRegression(max_iter=5000, solver='lbfgs')
            clf.fit(X_train_layers[i], y_train)

            val_preds = clf.predict(X_val_layers[i])
            acc = accuracy_score(y_val, val_preds)

            print(f"Validation Accuracy: {acc:.4f}")

            results.append({
                "layer": i,
                "accuracy": acc,
                "model": model_name,
                "task": label_column
            })

        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_file = results_dir / "results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")
