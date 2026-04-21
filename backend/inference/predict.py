import time
from typing import Dict, Optional

import torch

from .loader import DISPLAY_NAMES, MODEL_FILES, load_model
from .postprocess import save_chw_output, save_preview_from_chw
from .preprocess import load_input_array, to_batched_tensor


def build_preview(file_path: str) -> str:
    array_chw = load_input_array(file_path)
    return save_preview_from_chw(array_chw)


def run_one_model(model_name: str, x: torch.Tensor, device: torch.device) -> Dict:
    model = load_model(model_name, device)

    start = time.perf_counter()
    with torch.no_grad():
        y = model(x)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    output_chw = y[0].detach().cpu().numpy()
    image_file = save_chw_output(output_chw)

    return {
        "name": model_name,
        "display_name": DISPLAY_NAMES[model_name],
        "inference_time_ms": elapsed_ms,
        "output_file": image_file,
    }


def run_prediction(
    lensless_file_path: str,
    model_name: str = "unet",
    lensed_file_path: Optional[str] = None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lensless_chw = load_input_array(lensless_file_path)
    x = to_batched_tensor(lensless_chw, device)

    lensless_preview = build_preview(lensless_file_path)
    lensed_preview = build_preview(lensed_file_path) if lensed_file_path else None

    if model_name == "all":
        results = []
        for name in MODEL_FILES.keys():
            results.append(run_one_model(name, x, device))
        return {
            "mode": "all",
            "input_shape": list(x.shape),
            "lensless_preview_file": lensless_preview,
            "lensed_preview_file": lensed_preview,
            "results": results,
        }

    result = run_one_model(model_name, x, device)
    return {
        "mode": "single",
        "input_shape": list(x.shape),
        "lensless_preview_file": lensless_preview,
        "lensed_preview_file": lensed_preview,
        "result": result,
    }