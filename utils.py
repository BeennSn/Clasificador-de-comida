import os
import io
from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Load class names (21-class mapping used in training)
CLASSES_PATH = os.path.join("food-101", "meta", "classes.txt")

def _load_classes(path: str = CLASSES_PATH) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        classes = f.read().splitlines()
    classes_21 = classes[:20] + ["other"]
    return classes_21


def _build_model(device: torch.device, checkpoint_path: str):
    backbone = models.densenet201(weights=None)

    # Classifier: 1920 -> 1024 -> 101
    classifier = nn.Sequential(
        nn.Linear(1920, 1024),
        nn.LeakyReLU(),
        nn.Linear(1024, 101),
    )
    backbone.classifier = classifier

    # Head: 101 -> 21
    head = nn.Linear(101, len(_load_classes()))

    # Complete model
    model = nn.Sequential(backbone, head)

    # Load checkpoint
    state = torch.load(checkpoint_path, map_location=device)

    if isinstance(state, dict):
        if 'model_state_dict' in state:
            sd = state['model_state_dict']
        elif 'state_dict' in state:
            sd = state['state_dict']
        else:
            sd = state
    else:
        sd = state

    sd_clean = {k.replace('module.', ''): v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd_clean, strict=False)
    if missing:
        print(f"Warning: Missing keys in checkpoint: {missing[:5]}...")
    if unexpected:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected[:5]}...")

    model.to(device)
    model.eval()
    return model


_test_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def preprocess_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return _test_transforms(img).unsqueeze(0)


def load_model_and_classes(checkpoint_path: str = "./ckpt_finetuned.pt") -> Tuple[torch.nn.Module, List[str], torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = _build_model(device, checkpoint_path)
    classes = _load_classes()
    return model, classes, device


def predict_from_bytes(image_bytes: bytes, model: torch.nn.Module, classes: List[str], device: torch.device, topk: int = 3):
    """Return top-k predictions as list of (label, probability)."""
    tensor = preprocess_image(image_bytes).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().squeeze(0)

    topk_probs, topk_idx = torch.topk(probs, k=topk)
    results = []
    for p, idx in zip(topk_probs.tolist(), topk_idx.tolist()):
        label = classes[idx] if idx < len(classes) else "unknown"
        results.append({"label": label, "confidence": float(p)})

    return results
