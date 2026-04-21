import sys
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

_MODEL_CACHE = None

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


def load_all_models(device: torch.device):
    global _MODEL_CACHE

    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    my_device = str(device)

    le_admm_u = torch.load(
        WEIGHTS_DIR / "model_le_admm_u.pt",
        map_location=my_device,
        weights_only=False,
    )
    le_admm_u.admm_model.cuda_device = my_device
    le_admm_u.eval()

    unet = torch.load(
        WEIGHTS_DIR / "model_unet.pt",
        map_location=my_device,
        weights_only=False,
    )
    unet.eval()

    le_admm_s = torch.load(
        WEIGHTS_DIR / "model_le_admm_s.pt",
        map_location=my_device,
        weights_only=False,
    )
    le_admm_s.cuda_device = my_device
    le_admm_s.eval()

    le_admm = torch.load(
        WEIGHTS_DIR / "model_le_admm.pt",
        map_location=my_device,
        weights_only=False,
    )
    le_admm.cuda_device = my_device
    le_admm.eval()

    admm_auto = torch.load(
        WEIGHTS_DIR / "model_admm_converged.pt",
        map_location=my_device,
        weights_only=False,
    )
    admm_auto.cuda_device = my_device
    admm_auto.eval()

    admm_auto5 = torch.load(
        WEIGHTS_DIR / "model_admm_bounded.pt",
        map_location=my_device,
        weights_only=False,
    )
    admm_auto5.cuda_device = my_device
    admm_auto5.eval()

    _MODEL_CACHE = {
        "admm_converged": admm_auto,
        "admm_bounded": admm_auto5,
        "le_admm": le_admm,
        "le_admm_s": le_admm_s,
        "le_admm_u": le_admm_u,
        "unet": unet,
    }

    return _MODEL_CACHE


def load_model(model_name: str, device: torch.device):
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model: {model_name}")

    my_device = str(device)

    if model_name == "le_admm_u":
        model = torch.load(WEIGHTS_DIR / "model_le_admm_u.pt", map_location=my_device, weights_only=False)
        model.admm_model.cuda_device = my_device
    elif model_name == "unet":
        model = torch.load(WEIGHTS_DIR / "model_unet.pt", map_location=my_device, weights_only=False)
    elif model_name == "le_admm_s":
        model = torch.load(WEIGHTS_DIR / "model_le_admm_s.pt", map_location=my_device, weights_only=False)
        model.cuda_device = my_device
    elif model_name == "le_admm":
        model = torch.load(WEIGHTS_DIR / "model_le_admm.pt", map_location=my_device, weights_only=False)
        model.cuda_device = my_device
    elif model_name == "admm_converged":
        model = torch.load(WEIGHTS_DIR / "model_admm_converged.pt", map_location=my_device, weights_only=False)
        model.cuda_device = my_device
    elif model_name == "admm_bounded":
        model = torch.load(WEIGHTS_DIR / "model_admm_bounded.pt", map_location=my_device, weights_only=False)
        model.cuda_device = my_device
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.eval()
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