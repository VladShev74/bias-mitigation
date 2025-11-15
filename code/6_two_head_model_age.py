import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler
from torch.utils.data import DataLoader
from utils.paths import PROJECT_ROOT
from utils.models_config import MODEL_IDS
from utils.model_architectures import BertWithTwoHeadsAge


# Configuration
SEEDS = [42, 123, 1337]
NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Age group mapping
AGE_GROUPS = ["18-24", "25-34", "35-49", "50-64", "65-xx"]
AGE_GROUP_MAP = {age: idx for idx, age in enumerate(AGE_GROUPS)}


class CustomDatasetAge(torch.utils.data.Dataset):
    """Dataset for two-head training with task and age labels."""

    def __init__(self, dataframe, tokenizer, max_length=MAX_LENGTH):
        self.texts = dataframe['text'].tolist()
        self.task_labels = dataframe['task_label'].tolist()

        # Convert age strings to integers
        age_labels = dataframe['age'].tolist()
        if isinstance(age_labels[0], str):
            self.age_labels = [AGE_GROUP_MAP[age] for age in age_labels]
        else:
            self.age_labels = age_labels

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'task_label': torch.tensor(self.task_labels[idx]),
            'age_label': torch.tensor(self.age_labels[idx])
        }


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data():
    """Load training and validation data from pickle files."""
    data_dir = PROJECT_ROOT / "raw_data" / "pan16_raw" / "pickle_format"

    train_list = pd.read_pickle(data_dir / "train.pkl")
    train_df = pd.DataFrame(train_list)

    val_list = pd.read_pickle(data_dir / "validation.pkl")
    val_df = pd.DataFrame(val_list)

    return train_df, val_df


def train_model(model_name: str, model_id: str, train_dataset, seed: int, save_path: Path):
    """
    Train a two-head model with specified seed.

    Args:
        model_name: Name of the model (e.g., 'bert', 'modern_bert')
        model_id: Hugging Face model ID
        train_dataset: Training dataset
        seed: Random seed for reproducibility
        save_path: Path to save model weights and metadata

    Returns:
        Trained model
    """
    # Check if model already trained
    model_weights_path = save_path / "model_weights.pth"
    if model_weights_path.exists():
        print(f"\n{'='*70}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] SKIPPING Training {model_name} with seed {seed}")
        print(f"[OK] Model already exists at: {save_path}")
        print(f"{'='*70}")

        # Load and return existing model
        model = BertWithTwoHeadsAge(model_id=model_id, num_task_labels=2, num_age_groups=5)
        model.to(DEVICE)
        model.load_state_dict(torch.load(model_weights_path, map_location=DEVICE))
        return model

    start_time = datetime.now()
    print(f"\n{'='*70}")
    print(f"[{start_time.strftime('%H:%M:%S')}] Training {model_name} with seed {seed}")
    print(f"{'='*70}")

    set_seed(seed)

    # Initialize model
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing model...")
    model = BertWithTwoHeadsAge(model_id=model_id, num_task_labels=2, num_age_groups=5)
    model.to(DEVICE)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Model loaded on {DEVICE}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    num_training_steps = len(train_dataset) * NUM_EPOCHS // BATCH_SIZE
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Training loop
    model.train()
    training_history = []

    for epoch in range(NUM_EPOCHS):
        total_task_loss, total_age_loss = 0, 0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            task_labels = batch['task_label'].to(DEVICE)
            age_labels = batch['age_label'].to(DEVICE)

            optimizer.zero_grad()
            task_logits, age_logits = model(input_ids, attention_mask)

            task_loss = loss_fn(task_logits, task_labels)
            age_loss = loss_fn(age_logits, age_labels)
            total_loss = task_loss + age_loss

            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_task_loss += task_loss.item()
            total_age_loss += age_loss.item()
            num_batches += 1

            progress_bar.set_postfix({
                'task_loss': f"{task_loss.item():.4f}",
                'age_loss': f"{age_loss.item():.4f}"
            })

        avg_task_loss = total_task_loss / num_batches
        avg_age_loss = total_age_loss / num_batches

        print(f"  [Epoch {epoch+1}] Task Loss: {avg_task_loss:.4f} | Age Loss: {avg_age_loss:.4f}")

        training_history.append({
            'epoch': epoch + 1,
            'task_loss': avg_task_loss,
            'age_loss': avg_age_loss
        })

    # Save model
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saving model...")
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / "model_weights.pth")

    # Save training metadata
    metadata = {
        'model_name': model_name,
        'model_id': model_id,
        'seed': seed,
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_age_groups': 5,
        'age_groups': AGE_GROUPS,
        'training_history': training_history,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(save_path / "training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"[{end_time.strftime('%H:%M:%S')}] Training completed in {duration:.1f}s")
    print(f"[OK] Model saved to: {save_path}")
    return model


def evaluate_model(model, val_df, tokenizer):
    """
    Evaluate model on validation set.

    Args:
        model: Trained model
        val_df: Validation dataframe
        tokenizer: Tokenizer for the model

    Returns:
        Tuple of (task_accuracy, age_accuracy, age_balanced_accuracy)
    """
    model.eval()
    val_dataset = CustomDatasetAge(val_df, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    task_correct = 0
    age_correct = 0
    total = 0

    # For balanced accuracy - track per age group
    age_tp = {i: 0 for i in range(5)}  # True positives per age group
    age_total = {i: 0 for i in range(5)}  # Total per age group

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            task_labels = batch['task_label'].to(DEVICE)
            age_labels = batch['age_label'].to(DEVICE)

            task_logits, age_logits = model(input_ids, attention_mask)

            task_preds = torch.argmax(task_logits, dim=1)
            age_preds = torch.argmax(age_logits, dim=1)

            task_correct += (task_preds == task_labels).sum().item()
            age_correct += (age_preds == age_labels).sum().item()
            total += task_labels.size(0)

            # Per-age-group accuracy for balanced accuracy
            for age_group in range(5):
                mask = (age_labels == age_group)
                age_total[age_group] += mask.sum().item()
                age_tp[age_group] += ((age_preds == age_labels) & mask).sum().item()

    task_accuracy = task_correct / total
    age_accuracy = age_correct / total

    # Balanced accuracy: average of per-class accuracies
    age_acc_per_class = [age_tp[g] / age_total[g] if age_total[g] > 0 else 0 for g in range(5)]
    age_balanced_accuracy = np.mean(age_acc_per_class)

    return task_accuracy, age_accuracy, age_balanced_accuracy


def train_and_evaluate_all_models():
    """Train and evaluate all models across all seeds."""
    overall_start = datetime.now()
    print(f"\n[{overall_start.strftime('%H:%M:%S')}] Starting two-head training pipeline (AGE)...")
    print("[OK] Loading data...")
    train_df, val_df = load_data()
    print(f"[OK] Train samples: {len(train_df)} | Val samples: {len(val_df)}")
    print(f"[OK] Age groups: {AGE_GROUPS}\n")

    results = {}

    for model_name, model_id in MODEL_IDS.items():
        print(f"\n{'#'*70}")
        print(f"# Processing Model: {model_name}")
        print(f"{'#'*70}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        train_dataset = CustomDatasetAge(train_df, tokenizer)

        model_results = []

        for seed in SEEDS:
            # Define save path: models/two_head_age/{model_name}/seed_{seed}/
            save_path = PROJECT_ROOT / "models" / "two_head_age" / model_name / f"seed_{seed}"

            # Train model
            model = train_model(model_name, model_id, train_dataset, seed, save_path)

            # Evaluate
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Evaluating {model_name} (seed {seed})...")

            # Check if evaluation already exists
            eval_file_models = save_path / "evaluation_results.json"
            results_seed_dir = PROJECT_ROOT / "results" / "two_head_training_age" / model_name / f"seed_{seed}"
            eval_file_results = results_seed_dir / "evaluation_results.json"

            if eval_file_models.exists() and eval_file_results.exists():
                print("[OK] Evaluation results already exist, loading from disk...")
                with open(eval_file_models, 'r') as f:
                    eval_data = json.load(f)
                task_acc = eval_data['task_accuracy']
                age_acc = eval_data['age_accuracy']
                age_bal_acc = eval_data['age_balanced_accuracy']
            else:
                metrics = evaluate_model(model, val_df, tokenizer)
                task_acc, age_acc, age_bal_acc = metrics

            print(f"  Task Accuracy: {task_acc:.4f}")
            print(f"  Age Accuracy: {age_acc:.4f}")
            print(f"  Age Balanced Accuracy: {age_bal_acc:.4f}")

            model_results.append({
                'seed': seed,
                'task_accuracy': float(task_acc),
                'age_accuracy': float(age_acc),
                'age_balanced_accuracy': float(age_bal_acc)
            })

            # Save individual results in both locations
            eval_data = {
                'task_accuracy': float(task_acc),
                'age_accuracy': float(age_acc),
                'age_balanced_accuracy': float(age_bal_acc)
            }

            # Save with model weights
            with open(save_path / "evaluation_results.json", 'w') as f:
                json.dump(eval_data, f, indent=4)

            # Save in results folder
            results_seed_dir = PROJECT_ROOT / "results" / "two_head_training_age" / model_name / f"seed_{seed}"
            results_seed_dir.mkdir(parents=True, exist_ok=True)
            with open(results_seed_dir / "evaluation_results.json", 'w') as f:
                json.dump(eval_data, f, indent=4)

        # Compute and save average metrics for this model
        avg_task_acc = np.mean([r['task_accuracy'] for r in model_results])
        avg_age_acc = np.mean([r['age_accuracy'] for r in model_results])
        avg_age_bal_acc = np.mean([r['age_balanced_accuracy'] for r in model_results])

        print(f"\n{'='*70}")
        print(f"Average Results for {model_name}:")
        print(f"{'='*70}")
        print(f"  Task Accuracy: {avg_task_acc:.4f}")
        print(f"  Age Accuracy: {avg_age_acc:.4f}")
        print(f"  Age Balanced Accuracy: {avg_age_bal_acc:.4f}")

        results[model_name] = {
            'per_seed_results': model_results,
            'average_task_accuracy': float(avg_task_acc),
            'average_age_accuracy': float(avg_age_acc),
            'average_age_balanced_accuracy': float(avg_age_bal_acc)
        }

        # Save per-model average results in results folder
        model_results_dir = PROJECT_ROOT / "results" / "two_head_training_age" / model_name
        with open(model_results_dir / "average_results.json", 'w') as f:
            json.dump({
                'model_name': model_name,
                'average_task_accuracy': float(avg_task_acc),
                'average_age_accuracy': float(avg_age_acc),
                'average_age_balanced_accuracy': float(avg_age_bal_acc),
                'per_seed_results': model_results,
                'num_seeds': len(SEEDS),
                'seeds': SEEDS,
                'age_groups': AGE_GROUPS
            }, f, indent=4)

    # Save overall results
    results_dir = PROJECT_ROOT / "results" / "two_head_training_age"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "performance_eval.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    overall_end = datetime.now()
    total_duration = (overall_end - overall_start).total_seconds()
    print(f"\n{'#'*70}")
    print(f"[{overall_end.strftime('%H:%M:%S')}] Pipeline completed in {total_duration/60:.1f} minutes")
    print(f"[OK] All results saved to: {results_file}")
    print(f"{'#'*70}\n")

    # Print final summary of all results
    print(f"\n{'='*70}")
    print("FINAL EVALUATION SUMMARY (AGE)")
    print(f"{'='*70}\n")

    for model_name, model_results in results.items():
        print(f"Model: {model_name}")
        print(f"{'-'*70}")
        print(f"  Average Task Accuracy:        {model_results['average_task_accuracy']:.4f}")
        print(f"  Average Age Accuracy:         {model_results['average_age_accuracy']:.4f}")
        print(f"  Average Age Balanced Accuracy: {model_results['average_age_balanced_accuracy']:.4f}")
        print("\n  Per-seed results:")
        for seed_result in model_results['per_seed_results']:
            print(f"    Seed {seed_result['seed']:4d}: Task={seed_result['task_accuracy']:.4f} | "
                  f"Age={seed_result['age_accuracy']:.4f} | "
                  f"Balanced={seed_result['age_balanced_accuracy']:.4f}")
        print()

    print(f"{'='*70}\n")

    return results


if __name__ == "__main__":
    results = train_and_evaluate_all_models()
