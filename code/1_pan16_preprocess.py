import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm
from utils.models_config import MODEL_IDS
from utils.paths import PROJECT_ROOT, PAN16_PICKLE_DIR

# Config for saving embeddings
BATCH_SIZE = 64
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load from pickle
train_list = pd.read_pickle(PAN16_PICKLE_DIR.joinpath("train.pkl"))
val_list = pd.read_pickle(PAN16_PICKLE_DIR.joinpath("validation.pkl"))

# Convert list of dicts into DataFrames
train_df = pd.DataFrame(train_list)
val_df = pd.DataFrame(val_list)


def get_cls_embeddings_batch(texts, tokenizer, model, max_length=128):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states  # Tuple of layers (including embedding layer)
    cls_by_layer = [layer[:, 0, :].cpu().numpy() for layer in hidden_states]  # Each: (batch_size, hidden_dim)

    return cls_by_layer  # List of all layers


def embeddings_exist(split_name, model_name, label_column, num_layers):
    """Check if all embeddings and labels already exist for the given split, model, and label column."""
    storage_dir = PROJECT_ROOT / "data" / "pan16_embeddings" / label_column / model_name
    output_dir = storage_dir / split_name

    # Check if all layer files and labels pickle exist
    required_files = [f"X_{split_name}_layer{i}.npy" for i in range(num_layers)] + [f"y_{split_name}.pkl"]

    for file_name in required_files:
        file_path = output_dir / file_name
        if not file_path.exists():
            return False
    return True


def save_cls_embeddings(data_df, split_name, model_name, label_column, model):
    """
    Save CLS embeddings for a given split and model.

    Args:
        data_df: DataFrame with text and labels
        split_name: "train" or "val"
        model_name: Name of the model (e.g., "bert", "modern_bert")
        label_column: Column name in data_df to use as labels ("gender" or "age")
        model: The loaded transformer model
    """
    # Determine number of layers dynamically
    num_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer

    # Check if embeddings already exist
    if embeddings_exist(split_name, model_name, label_column, num_layers):
        print(f"Embeddings for {split_name} set, model {model_name}, and "
              f"label {label_column} already exist. Skipping...")
        return

    storage_dir = PROJECT_ROOT / "data" / "pan16_embeddings" / label_column / model_name
    output_dir = storage_dir / split_name
    print(f"Creating embeddings for {split_name} set, model {model_name}, "
          f"label {label_column} ({num_layers} layers)...")
    print(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_layers = [[] for _ in range(num_layers)]
    y_labels = []

    for start in tqdm(range(0, len(data_df), BATCH_SIZE), desc=f"Processing {split_name} set ({label_column})"):
        end = min(start + BATCH_SIZE, len(data_df))
        batch_texts = data_df.iloc[start:end]['text'].tolist()
        batch_labels = data_df.iloc[start:end][label_column].tolist()

        # Apply label transformation based on label column
        if label_column == "gender":
            batch_labels = [1 if label == 'female' else 0 for label in batch_labels]

        cls_layers = get_cls_embeddings_batch(batch_texts, tokenizer, model)

        for i in range(num_layers):
            X_layers[i].append(cls_layers[i])
        y_labels.extend(batch_labels)

    # Save each layer
    for i in range(num_layers):
        layer_array = np.vstack(X_layers[i])
        np.save(os.path.join(output_dir, f"X_{split_name}_layer{i}.npy"), layer_array)

    # Save labels
    pd.to_pickle(y_labels, os.path.join(output_dir, f"y_{split_name}.pkl"))
    print(f"Saved {split_name} embeddings to: {output_dir}")


# Initialize BERT tokenizer and model (outputs hidden states from all layers)
for model_name, model_id in MODEL_IDS.items():
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"{'='*60}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, output_hidden_states=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()  # set model to evaluation mode (no dropout, etc.)

    # Process gender task
    save_cls_embeddings(train_df, "train", model_name, "gender", model)
    save_cls_embeddings(val_df, "val", model_name, "gender", model)

    # Process age task
    save_cls_embeddings(train_df, "train", model_name, "age", model)
    save_cls_embeddings(val_df, "val", model_name, "age", model)
