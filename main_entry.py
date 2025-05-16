from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import tempfile
import os
import shutil
from model.main import ModelWrapper


app = FastAPI(
    title="Generic Model Serving API",
    description="Endpoints to load and inference models with audio/image/text inputs",
    version="1.0.0"
)

# Global singleton for simplicity; in production use a pool or registry
model_wrapper: Optional[ModelWrapper] = None

@app.post("/load_model")
async def load_model(
    provider: str = Form(..., description="Model provider, e.g., 'huggingface'"),
    model_name: str = Form(..., description="Full model identifier, e.g., 'openai/whisper-large'"),
    task: str = Form(..., description="Task type")
):
    """
    Load a model into memory (and optionally Azure cache).
    """
    global model_wrapper
    try:
        model_wrapper = ModelWrapper(provider, model_name, task)
        model_wrapper.load_model()
        return {"message": f"Model '{model_name}' loaded successfully from {provider}."}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load error: {e}")

@app.post("/infer")
async def infer(
    text: Optional[str] = Form(None, description="Text input for text-based models"),
    image: Optional[UploadFile] = File(None, description="Image file for vision models"),
    audio: Optional[UploadFile] = File(None, description="Audio file for audio models")
):
    """
    Run inference on the previously loaded model. Returns `model_output`.
    Accepts one or more of: text, image, audio.
    """
    global model_wrapper
    if not model_wrapper or not model_wrapper.model:
        raise HTTPException(status_code=400, detail="No model loaded. Call /load_model first.")

    # Prepare input payload
    input_data: List[dict] = []
    # Handle text
    if text:
        input_data.append({"text": text})
    # Handle image upload
    if image:
        suffix = os.path.splitext(image.filename)[1] or ".jpg"
        tmp_img = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        shutil.copyfileobj(image.file, tmp_img)
        tmp_img.close()
        input_data.append({"image": tmp_img.name})
    # Handle audio upload
    if audio:
        suffix = os.path.splitext(audio.filename)[1] or ".wav"
        tmp_aud = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        shutil.copyfileobj(audio.file, tmp_aud)
        tmp_aud.close()
        input_data.append({"audio": tmp_aud.name})

    if not input_data:
        raise HTTPException(status_code=422, detail="At least one of text, image, or audio must be provided.")

    try:
        # Run inference and capture output
        output = model_wrapper.run_inference(input_data=input_data, task=model_wrapper.task)
        return {"model_output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

