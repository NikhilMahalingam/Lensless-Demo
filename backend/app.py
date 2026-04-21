from pathlib import Path
import shutil
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from inference.loader import list_models
from inference.postprocess import OUTPUTS_DIR
from inference.predict import run_prediction

app = FastAPI(title="Lensless Reconstruction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("tmp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/models")
def get_models():
    models = list_models()
    models.append({
        "name": "all",
        "display_name": "Run all models",
        "weights_found": True,
    })
    return {"models": models}


@app.post("/reconstruct")
async def reconstruct(
    lensless_file: UploadFile = File(...),
    model_name: str = Form(...),
    lensed_file: Optional[UploadFile] = File(None),
):
    lensless_suffix = Path(lensless_file.filename).suffix or ".npy"
    lensless_temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{lensless_suffix}"

    lensed_temp_path = None

    try:
        with lensless_temp_path.open("wb") as buffer:
            shutil.copyfileobj(lensless_file.file, buffer)

        if lensed_file is not None and lensed_file.filename:
            lensed_suffix = Path(lensed_file.filename).suffix or ".npy"
            lensed_temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{lensed_suffix}"
            with lensed_temp_path.open("wb") as buffer:
                shutil.copyfileobj(lensed_file.file, buffer)

        response = run_prediction(
            lensless_file_path=str(lensless_temp_path),
            model_name=model_name,
            lensed_file_path=str(lensed_temp_path) if lensed_temp_path else None,
        )

        response["lensless_preview_url"] = f"/outputs/{response['lensless_preview_file']}"
        if response["lensed_preview_file"]:
            response["lensed_preview_url"] = f"/outputs/{response['lensed_preview_file']}"
        else:
            response["lensed_preview_url"] = None

        if response["mode"] == "single":
            response["result"]["image_url"] = f"/outputs/{response['result']['output_file']}"
        else:
            for result in response["results"]:
                result["image_url"] = f"/outputs/{result['output_file']}"

        return response

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    finally:
        lensless_temp_path.unlink(missing_ok=True)
        if lensed_temp_path is not None:
            lensed_temp_path.unlink(missing_ok=True)