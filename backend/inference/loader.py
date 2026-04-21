import sys
from pathlib import Path
from typing import Dict

import torch

BASE_DIR = Path(__file__).resolve().parent.parent          # backend/
INFERENCE_DIR = Path(__file__).resolve().parent            # backend/inference/
MODELS_DIR = BASE_DIR / "models"                           # backend/models/
WEIGHTS_DIR = INFERENCE_DIR / "weights"

# Make backend/models importable as top-level modules like:
# import admm_model, import ensemble, import unet
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

# If utils.py is in backend/, make that importable too
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

_MODEL_CACHE: Dict[str, torch.nn.Module] = {}

MODEL_FILES = {
    "admm_converged": "model_admm_converged.pt",
    "admm_bounded": "model_admm_bounded.pt",
    "le_admm": "model_le_admm.pt",
    "le_admm_s": "model_le_admm_s.pt",
    "le_admm_u": "model_le_admm_u.pt",
    "unet": "model_unet.pt",
}

DISPLAY_NAMES = {
    "admm_converged": "ADMM converged",
    "admm_bounded": "ADMM bounded",
    "le_admm": "Le-ADMM",
    "le_admm_s": "Le-ADMM*",
    "le_admm_u": "Le-ADMM-U",
    "unet": "U-Net",
}


def _patch_loaded_model(model, device_str: str):
    if hasattr(model, "cuda_device"):
        model.cuda_device = device_str

    if hasattr(model, "admm_model") and hasattr(model.admm_model, "cuda_device"):
        model.admm_model.cuda_device = device_str

    model.eval()
    return model


def load_model(model_name: str, device: torch.device):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model: {model_name}")

    weights_path = WEIGHTS_DIR / MODEL_FILES[model_name]
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file: {weights_path}")

    device_str = str(device)

    model = torch.load(
        weights_path,
        map_location=device_str,
        weights_only=False,
    )

    model = _patch_loaded_model(model, device_str)
    _MODEL_CACHE[model_name] = model
    return model


def list_models():
    return [
        {
            "name": name,
            "display_name": DISPLAY_NAMES[name],
            "weights_found": (WEIGHTS_DIR / filename).exists(),
        }
        for name, filename in MODEL_FILES.items()
    ]