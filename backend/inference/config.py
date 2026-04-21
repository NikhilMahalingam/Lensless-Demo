from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INFERENCE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = INFERENCE_DIR / "weights"
OUTPUTS_DIR = INFERENCE_DIR / "outputs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Matches the repo's preprocessing behavior:
# image / 4095.0 - 0.008273973, then downsample by 4
RAW_DIVISOR = 4095.0
RAW_OFFSET = 0.008273973
DOWNSAMPLE_FACTOR = 4

# You will replace these filenames with your real downloaded weights.
MODEL_REGISTRY = {
    "unet_small": {
        "type": "unet_small",
        "weights": WEIGHTS_DIR / "unet_small.pth",
        "display_name": "U-Net Small",
    },
    "unet_270x480": {
        "type": "unet_270x480",
        "weights": WEIGHTS_DIR / "unet_270x480.pth",
        "display_name": "U-Net 270x480",
    },
    # Add ADMM-based models later after one feedforward model works.
}