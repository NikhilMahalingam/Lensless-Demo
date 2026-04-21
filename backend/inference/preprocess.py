from pathlib import Path

import cv2
import numpy as np
import torch


def _load_npy_chw(file_path: str) -> np.ndarray:
    arr = np.load(file_path)

    if arr.ndim != 3:
        raise ValueError(f".npy input must have 3 dims, got shape {arr.shape}")

    # assume HWC and convert to CHW
    return arr.transpose((2, 0, 1))


def _load_regular_image_chw(file_path: str) -> np.ndarray:
    img = cv2.imread(file_path, -1)
    if img is None:
        raise ValueError("Failed to read uploaded image with OpenCV.")

    img = img.astype(np.float32)

    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    return img.transpose((2, 0, 1))


def load_input_array(file_path: str) -> np.ndarray:
    suffix = Path(file_path).suffix.lower()

    if suffix == ".npy":
        return _load_npy_chw(file_path)

    return _load_regular_image_chw(file_path)


def to_batched_tensor(image_chw: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(image_chw, dtype=torch.float32, device=device).unsqueeze(0)