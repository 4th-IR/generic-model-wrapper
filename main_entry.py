from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from typing import Optional, List
import tempfile, os, shutil
from model.main import ModelWrapper


# modules
from utils.process_inputs import process_images, process_audio, TextInput

app = FastAPI(
    title="Generic Model Wrapper API",
    description="Endpoints to load and inference models with audio/image/text inputs",
    version="1.0.01"
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/load_model")
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


@app.post("/infer")
async def inference_model(
    request: Request,
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
):
    # Load wrapper object
    model_wrapper = request.app.state.model_wrapper

    input_data: List[dict] = []
    if text:
        try:
            validated = TextInput(text=text)  
            input_data.append({"text": validated.text})

            # print("Validated text input:", validated.text)
            # print("Input data going into inference:", input_data)

        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
    # Image
    if image:
        try:
            tmp_img_path = process_images(image)
        
            input_data.append({"image": tmp_img_path})
            print("Validated image path:", tmp_img_path)
            print("Input data going into inference:", input_data)
        
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
    # Audio
    if audio:
        try:
            tmp_aud_path = process_audio(audio)
            
            input_data.append({"audio": tmp_aud_path})
            print("Validated audio path:", tmp_aud_path)
            print("Input data going into inference:", input_data)

        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected audio processing error: {str(e)}")

            
    if not input_data:
        raise HTTPException(status_code=422, detail="At least one of text, image, or audio must be provided.")


    try:
        output = model_wrapper.run_inference(input_data=input_data, task=model_wrapper.task)
    finally:
        # 4) Clean up temp files
        for item in input_data:
            for path in item.values():
                if os.path.exists(path):
                    os.unlink(path)

    return {"results": output}
