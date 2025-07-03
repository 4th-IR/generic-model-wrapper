import tempfile
from typing import Literal, Optional, Dict

from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from core.wrapper import wrapper
from core.config import settings


app = FastAPI(
    title="Generic Model Wrapper Saving API",
    description="Endpoints to save models to blob storage",
    version="1.0.01",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/docs")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/save_model")
async def save_model(
    provider: str = Body(
        "huggingface", description="Model provider, e.g., 'huggingface', 'spacy'"
    ),
    task: Optional[
        Literal[
            "audio-classification",
            "automatic-speech-recognition",
            "depth-estimation",
            "document-question-answering",
            "feature-extraction",
            "fill-mask",
            "image-classification",
            "image-feature-extraction",
            "image-segmentation",
            "image-text-to-text",
            "image-to-image",
            "image-to-text",
            "mask-generation",
            "ner",
            "object-detection",
            "question-answering",
            "sentiment-analysis",
            "summarization",
            "table-question-answering",
            "text-classification",
            "text-generation",
            "text-to-audio",
            "text-to-speech",
            "text2text-generation",
            "token-classification",
            "translation",
            "video-classification",
            "visual-question-answering",
            "vqa",
            "zero-shot-audio-classification",
            "zero-shot-classification",
            "zero-shot-image-classification",
            "zero-shot-object-detection",
        ]
    ] = Body(None, description="Task type"),
    model_identifier: str = Body(
        None,
        description="Full model identifier, e.g., 'openai/whisper-large', 'en_core_web_sm'",
    ),
    kwargs: Optional[Dict[str, str]] = Body(
        None,
        description="Any extra config or parameters to be passed to the model save function",
    ),
):
    """
    Save model to storage
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:

            result = wrapper.load_from_provider(
                provider,
                model_identifier,
                task,
                kwargs,
                tmpdir,
            )

            message = f"Model '{model_identifier}' not downloaded from {provider} successfully."

            if result:
                wrapper.save_to_storage(model_identifier, tmpdir)

                # Return message first (pretend to notify user)
                message = f"Model '{model_identifier}' {'' if result else 'not'} saved to storage successfully from {provider}."

                wrapper.cleanup(model_identifier.replace("/", "_"), provider)

            return {"message": message}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load error: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
