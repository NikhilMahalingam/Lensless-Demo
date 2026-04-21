from pathlib import Path
import uuid

import numpy as np
from PIL import Image

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def preplot_repo_style(chw: np.ndarray) -> np.ndarray:
    """
    Match the LenslessLearning repo's preplot(image) behavior:
    - CHW -> HWC
    - swap channels [2,1,0]
    - clip to [0,1]
    - flip vertically
    - crop borders [60:, 62:-38, :]
    """
    if chw.ndim != 3:
        raise ValueError(f"Expected CHW array, got shape {chw.shape}")

    image = np.transpose(chw, (1, 2, 0))

    image_color = np.zeros_like(image)
    image_color[:, :, 0] = image[:, :, 2]
    image_color[:, :, 1] = image[:, :, 1]
    image_color[:, :, 2] = image[:, :, 0]

    out_image = np.flipud(np.clip(image_color, 0, 1))

    # Apply same crop as notebook preplot()
    if out_image.shape[0] > 60 and out_image.shape[1] > 100:
        out_image = out_image[60:, 62:-38, :]

    return out_image


def to_uint8(hwc: np.ndarray) -> np.ndarray:
    hwc = (hwc * 255.0).clip(0, 255).astype(np.uint8)
    return hwc


def save_hwc_png(hwc: np.ndarray) -> str:
    filename = f"{uuid.uuid4().hex}.png"
    out_path = OUTPUTS_DIR / filename
    Image.fromarray(hwc).save(out_path)
    return filename


def save_preview_from_chw(array_chw: np.ndarray) -> str:
    hwc = preplot_repo_style(array_chw)
    return save_hwc_png(to_uint8(hwc))


def save_chw_output(array_chw: np.ndarray) -> str:
    hwc = preplot_repo_style(array_chw)
    return save_hwc_png(to_uint8(hwc))