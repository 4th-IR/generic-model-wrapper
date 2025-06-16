import io
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

import torch
import numpy as np
from PIL import Image
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModel, AutoProcessor, AutoFeatureExtractor

from core.wrapper import wrapper
from core.process_inputs import load_audio, load_image


app = FastAPI(
    title="Generic Model Wrapper API",
    description="Endpoints to load and inference models with audio/image/text inputs",
    version="1.0.01",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/load_model")
def load_model():
    result = wrapper.load_from_storage()
    return f"Model was {'' if result else 'not'} loaded successfully."


@torch.inference_mode()
@app.post("/run_inference")
def run_inference(
    task: Literal[
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
    ] = Form(...),
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    sample_rate: Optional[int] = Form(16000),
):
    task_pipeline_details = PIPELINE_REGISTRY.supported_tasks.get(
        task, None
    ) or PIPELINE_REGISTRY.supported_tasks.get(
        PIPELINE_REGISTRY.task_aliases.get(task), None
    )
    if task_pipeline_details:

        processor = None

        try:
            processor = AutoProcessor.from_pretrained(wrapper.model_save_path)
        except:
            try:
                processor = AutoFeatureExtractor.from_pretrained(
                    wrapper.model_save_path
                )
            except:
                # raise RuntimeError("Couldn't find a good processor for the model")
                raise HTTPException(
                    500, "Couldn't find a good processor from the available files"
                )
        automodel_class: AutoModel = task_pipeline_details.get("pt")[0]
        model = automodel_class.from_pretrained(wrapper.model_save_path)
        model.eval()

        loaded_audio = load_audio(audio) if audio else None

        input_dict = {
            "text": text,
            "images": load_image(image) if image else None,
            "audio": loaded_audio,
            "raw_speech": loaded_audio,
            "sampling_rate": sample_rate if audio else None,
        }

        filtered_input_dict = {k: v for k, v in input_dict.items() if v is not None}

        inputs = processor(
            **filtered_input_dict,
            return_tensors="pt",
        )
        outputs = model(**inputs)

        if task in [
            "automatic-speech-recognition",
            "fill-mask",
        ]:

            predicted_ids = outputs.logits.argmax(dim=-1)
            return processor.decode(predicted_ids[0], skip_special_tokens=True)

        elif task in [
            "audio-classification",
        ]:
            predicted_ids = outputs.logits.argmax(dim=-1)

            return model.config.id2label[predicted_ids.item()]

        elif task in [
            "depth-estimation",
        ]:

            depth = outputs.predicted_depth[0]
            depth_np = depth.detach().numpy()

            norm = (depth_np - depth_np.min()) / (np.ptp(depth_np))
            depth_img = Image.fromarray((norm * 255).astype(np.uint8))
            img_io = io.BytesIO()
            depth_img.save(img_io, format="PNG")  # or "JPEG", depending on use case
            img_io.seek(0)

            return StreamingResponse(img_io, media_type="image/png")

        elif task in [
            "feature-extraction",
        ]:
            return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    return "Unsupported Task"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
