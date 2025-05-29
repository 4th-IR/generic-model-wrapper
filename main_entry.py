from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from typing import Optional, List, Dict, Any
import tempfile, os, shutil
from model.main import ModelWrapper
import time
from utils.process_inputs import TextInput, process_audio, process_images

app = FastAPI(
    title="Generic Model Wrapper API",
    description="Endpoints to load and inference models with audio/image/text inputs",
    version="1.0.01"
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/save_model")
async def load_model(
    provider: str = Form(..., description="Model provider, e.g., 'huggingface'"),
    model_name: str = Form(..., description="Full model identifier, e.g., 'openai/whisper-large'"),
    task: str = Form(..., description="Task type")
    ):
    """
    Load a model into memory (and optionally Azure cache)
    """
    try:
        model_wrapper = ModelWrapper(provider, model_name, task)
        model_wrapper.load_model()
        app.state.model_wrapper = model_wrapper

        # Return message first (pretend to notify user)
        load_msg = f"Model '{model_name}' loaded successfully from {provider}. Starting inference..."

        return {"message": load_msg}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load or inference error: {e}")

@app.post("/load_inference")
async def inference_model(
    request: Request,
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
):
    model_wrapper = request.app.state.model_wrapper
    content: Dict[str, Any] = {}

    print(content)
    # Text
    if text:
        try:
            validated = TextInput(text=text)
            content["text"] = validated.text
            print(validated.text)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        

    # Image
    if image:
        try:
            tmp_img_path = process_images(image)
            content["image"] = tmp_img_path
            print(tmp_img_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image error: {str(e)}")

    # Audio
    if audio:
        try:
            tmp_aud_path = process_audio(audio)
            content["audio"] = tmp_aud_path
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Audio error: {str(e)}")

    if not content:
        raise HTTPException(status_code=422, detail="At least one of text, image, or audio must be provided.")

    # Properly formatted messages list
    messages = [{"role": "user", "content": content}]

    try:
        model_output = model_wrapper.run_inference(messages=messages, task=model_wrapper.task)
    finally:
        # Clean up temp files if they were created
        for key in ("image", "audio"):
            path = content.get(key)
            if path and os.path.exists(path):
                os.unlink(path)

    print(model_output)
    return JSONResponse(content=model_output)


