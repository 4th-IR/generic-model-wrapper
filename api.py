import uvicorn
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Any, Dict, Optional
import os

# Assuming model and utils are accessible from the root directory
# Adjust imports based on actual project structure if needed
from model.main import ModelWrapper
from utils.logger import get_logger

# Load environment variables (ensure HUGGINGFACE_TOKEN and Azure creds are set)
from dotenv import load_dotenv
load_dotenv()

LOG = get_logger('api')

app = FastAPI(
    title="AI Model Inference Service",
    description="A FastAPI service to run inference on various AI models.",
    version="1.0.0"
)

class InferenceRequest(BaseModel):
    provider: str
    model_name: str
    pipeline_type: str
    input_data: Any
    task: Optional[str] = None
    kwargs: Optional[Dict[str, Any]] = {}

class InferenceResponse(BaseModel):
    result: Any

# Global model cache (simple implementation)
# For production, consider a more robust caching/loading strategy
model_cache: Dict[str, ModelWrapper] = {}

def get_model_instance(provider: str, model_name: str, pipeline_type: str) -> ModelWrapper:
    """Gets or creates a ModelWrapper instance."""
    cache_key = f"{provider}_{model_name}_{pipeline_type}"
    if cache_key not in model_cache:
        LOG.info(f"Creating and loading new model instance: {cache_key}")
        try:
            model_wrapper = ModelWrapper(provider=provider, model_name=model_name, pipeline_type=pipeline_type)
            model_wrapper.load_model() # Load the model upon creation
            model_cache[cache_key] = model_wrapper
        except Exception as e:
            LOG.error(f"Failed to load model {cache_key}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to load model {model_name}: {str(e)}")
    else:
        LOG.info(f"Using cached model instance: {cache_key}")
    return model_cache[cache_key]

@app.post("/infer", response_model=InferenceResponse)
async def run_inference_endpoint(request: InferenceRequest = Body(...)):
    """Runs inference using the specified model and input data."""
    LOG.info(f"Received inference request for model: {request.model_name} ({request.provider}) with task: {request.task or request.pipeline_type}")
    try:
        # Get or create the model instance
        model_wrapper = get_model_instance(
            provider=request.provider,
            model_name=request.model_name,
            pipeline_type=request.pipeline_type
        )

        # Run inference
        LOG.info("Running inference...")
        start_time = time.time()
        result = model_wrapper.run_inference(
            input_data=request.input_data,
            task=request.task or request.pipeline_type, # Use pipeline_type as default task if task not provided
            **request.kwargs
        )
        end_time = time.time()
        LOG.info(f"Inference completed in {end_time - start_time:.2f} seconds.")

        return InferenceResponse(result=result)

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions (e.g., from model loading)
        raise http_exc
    except Exception as e:
        LOG.error(f"Inference failed for model {request.model_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "AI Model Inference Service is running. Use the /infer endpoint to run predictions."}

if __name__ == "__main__":
    # Ensure necessary environment variables are set before starting
    required_env_vars = ["HUGGINGFACE_TOKEN", "AZURE_CONNECTION_STRING", "AZURE_CONTAINER_NAME", "AZURE_STORAGE_ACCOUNT"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        LOG.warning(f"Missing environment variables: {', '.join(missing_vars)}. Model loading/saving might fail.")

    port = int(os.getenv("PORT", 8000))
    LOG.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)