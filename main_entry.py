from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from starlette.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, validator
from typing import List, Optional, Literal, Dict
import time
import os
import psutil
import uuid
from utils.prepare_binary_input import process_file
import json

# Internal dependencies
from pipeline.model_loading_pipeline import load_model
from pipeline.model_inference import model_inference
from utils.logger import get_logger

LOG = get_logger('test')
app = FastAPI()

VALID_TASKS = {
    "image": ["image_classification", "image-captioning"],
    "audio": ["audio_transcription", "speech-to-text"],
    "text": ["text_processing", "document-summary"]
}

# ----- Request Schemas -----
class SampleInputItem(BaseModel):
    image: Optional[str] = None
    text: Optional[str] = None

    @validator('image')
    def validate_local_image_path(cls, v):
        if v and v.strip().lower().startswith(('http://', 'https://')):
            raise ValueError("Only local file paths are allowed for 'image'; URLs are not permitted.")
        return v


class ModelInput(BaseModel):
    model_provider: str
    model_category: str
    model_name: str
    task: str
    sample_input: List[SampleInputItem]


# ----- Response Schema -----
class ModelTestResponse(BaseModel):
    model_name: str
    total_time_taken_mins: float
    memory_used: float
    load_status: int
    load_error: Optional[str]
    inference_status: int
    inference_error: Optional[str]
    model_output: Optional[str]


# -------- OpenAI-Compatible Schema --------
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class OpenAIChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False


@app.post("/test-model", response_model=ModelTestResponse)
def test_model_endpoint(model_input: ModelInput):
    start_time = time.time()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 2)

    load_status = 0
    inference_status = 0
    load_error = None
    inference_error = None
    model_output = None

    model_config = {
        "model_provider": model_input.model_provider,
        "model_category": model_input.model_category,
        "model_name": model_input.model_name,
        "task": model_input.task,
        "sample_input": [item.dict(exclude_none=True) for item in model_input.sample_input]
    }

    try:
        LOG.info(f"Loading model: {model_input.model_name}")
        load_model(model_config)
        load_status = 1
        LOG.info("Model loaded successfully.")
    except Exception as e:
        load_error = str(e)
        LOG.exception("Model loading failed.")
        # Skip inference if loading fails
        return ModelTestResponse(
            model_name=model_input.model_name,
            total_time_taken_mins=0.0,
            memory_used=0.0,
            load_status=load_status,
            load_error=load_error,
            inference_status=inference_status,
            inference_error=inference_error,
            model_output=model_output,
        )

    try:
        LOG.info("Running inference...")
        model_output = model_inference(model_config)
        inference_status = 1
        LOG.info("Inference completed.")
    except Exception as e:
        inference_error = str(e)
        LOG.exception("Inference failed.")

    end_time = time.time()
    mem_after = process.memory_info().rss / (1024 ** 2)
    mem_used = max(mem_after - mem_before, 0)
    total_time = round((end_time - start_time) / 60, 2)

    return ModelTestResponse(
        model_name=model_input.model_name,
        total_time_taken_mins=total_time,
        memory_used=mem_used,
        load_status=load_status,
        load_error=load_error,
        inference_status=inference_status,
        inference_error=inference_error,
        model_output=model_output,
    )


@app.post("/v1/inference/completions")
async def inference(
    model: str = Form(...),
    task: str = Form(...),
    messages: str = Form(...),  # JSON-encoded list of ChatMessage
    stream: Optional[bool] = Form(False),
    file: Optional[UploadFile] = File(None)  # <- make file optional
):
    try:
        # Validate and parse messages
        parsed_messages = [ChatMessage(**m) for m in json.loads(messages)]

        # Map task to internal handler type
        task_type = None
        for k, aliases in VALID_TASKS.items():
            if task.lower() in aliases:
                task_type = k
                break
        if not task_type:
            raise HTTPException(status_code=400, detail=f"Unsupported task: {task}")

        # Only process file if provided
        file_data = None
        if file:
            file_data = process_file(file, task_type=task_type)

        # Prepare sample input
        sample_input = [{
            "text": msg.content,
            "role": msg.role,
            "file_data": file_data
        } for msg in parsed_messages]

        # Build config
        model_config = {
            "model_name": model,
            "task": task,
            "sample_input": sample_input
        }

        # Run inference
        LOG.info(f"Running multimodal inference for model: {model}, task: {task}")
        output = model_inference(model_config)

        if stream:
            def generate():
                for word in output.split():
                    chunk = {
                        "choices": [{
                            "delta": {"content": word + " "},
                            "index": 0,
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    time.sleep(0.05)
                yield "data: [DONE]\n\n"
            return StreamingResponse(generate(), media_type="text/event-stream")

        return JSONResponse({
            "id": f"mmmdl-{uuid.uuid4().hex}",
            "object": "inference.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output
                },
                "finish_reason": "stop"
            }]
        })

    except Exception as e:
        LOG.exception("Multimodal inference failed.")
        raise HTTPException(status_code=500, detail=str(e))