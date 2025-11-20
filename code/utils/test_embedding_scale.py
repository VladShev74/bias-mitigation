from transformers import AutoModel, AutoTokenizer
import torch
from models_config import MODEL_IDS

# Test all configured models
for name, model_id in MODEL_IDS.items():
    print(f"\n{'='*50}")
    print(f"Testing {name}")
    print(f"{'='*50}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, output_hidden_states=True)
    model.eval()

    # Test sentence
    text = "The doctor examined the patient carefully."
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states

    print(f"Number of layers: {len(hidden_states)}")
    print(f"Hidden size: {hidden_states[0].shape[-1]}")

    # Check statistics of middle and last layers
    for layer_idx in [0, len(hidden_states)//2, -1]:
        layer_output = hidden_states[layer_idx]
        cls_embedding = layer_output[:, 0, :].cpu().numpy()
        print(f"\nLayer {layer_idx}:")
        print(f"  Mean: {cls_embedding.mean():.4f}")
        print(f"  Std: {cls_embedding.std():.4f}")
        print(f"  Min: {cls_embedding.min():.4f}")
        print(f"  Max: {cls_embedding.max():.4f}")
