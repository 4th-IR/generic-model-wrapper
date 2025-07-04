import io
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoFeatureExtractor

from core.wrapper import wrapper
from core.process_inputs import load_audio, load_image
from core.config import settings


app = FastAPI(
    title=f"Generic Model Wrapper Inference API - {settings.MODEL_IDENTIFIER} for {settings.TASK}",
    description="Endpoints to load and inference models with audio/image/text inputs",
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


@app.post("/load_model")
def load_model(force_redownload: bool = False):
    result = wrapper.load_from_storage(force_redownload)
    return f"Model was {'' if result else 'not'} loaded successfully."


@torch.inference_mode()
@app.post("/run_inference")
def run_inference(
    #     task: Optional[
    #         Literal[
    #             "audio-classification",
    #             "automatic-speech-recognition",
    #             "depth-estimation",
    #             "document-question-answering",
    #             "feature-extraction",
    #             "fill-mask",
    #             "image-classification",
    #             "image-feature-extraction",
    #             "image-segmentation",
    #             "image-text-to-text",
    #             "image-to-image",
    #             "image-to-text",
    #             "mask-generation",
    #             "ner",
    #             "object-detection",
    #             "question-answering",
    #             "sentiment-analysis",
    #             "summarization",
    #             "table-question-answering",
    #             "text-classification",
    #             "text-generation",
    #             "text-to-audio",
    #             "text-to-speech",
    #             "text2text-generation",
    #             "token-classification",
    #             "translation",
    #             "video-classification",
    #             "visual-question-answering",
    #             "vqa",
    #             "zero-shot-audio-classification",
    #             "zero-shot-classification",
    #             "zero-shot-image-classification",
    #             "zero-shot-object-detection",
    #         ]
    #     ] = Form(...),
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    sample_rate: Optional[int] = Form(16000),
):
    if settings.PROVIDER == "huggingface":

        processor = None

        try:
            processor = AutoProcessor.from_pretrained(wrapper.model_save_path)
        except:
            try:
                processor = AutoFeatureExtractor.from_pretrained(
                    wrapper.model_save_path
                )
            except:
                raise HTTPException(
                    500, "Couldn't find a good processor from the available files"
                )

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
        outputs = wrapper.model(**inputs)

        if settings.TASK in [
            "automatic-speech-recognition",
            "fill-mask",
        ]:

            predicted_ids = outputs.logits.argmax(dim=-1)
            return processor.decode(predicted_ids[0], skip_special_tokens=True)

        elif settings.TASK in [
            "audio-classification",
        ]:
            predicted_ids = outputs.logits.argmax(dim=-1)

            return wrapper.model.config.id2label[predicted_ids.item()]

        elif settings.TASK in [
            "ner",
            "token-classification",
        ]:
            predicted_ids = outputs.logits.argmax(dim=-1)
            print(inputs)

            input_ids = inputs["input_ids"][0]
            word_ids = inputs.word_ids()
            word_list = text.split()
            tokens = processor.convert_ids_to_tokens(input_ids)

            labels = [wrapper.model.config.id2label[t.item()] for t in predicted_ids[0]]

            entities = []
            for token, label, id in zip(tokens, labels, word_ids):
                if id is not None:
                    entities.append(
                        {
                            "token": token,
                            "text": word_list[id],
                            "label": label,
                        }
                    )

            return entities

        elif settings.TASK in [
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

        elif settings.TASK in [
            "feature-extraction",
        ]:
            return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        return "Unsupported Task"

    elif settings.PROVIDER == "spacy":
        doc = wrapper.model(text)

        result = {
            "entities": [
                {
                    attr: (
                        getattr(token, attr).tolist()
                        if isinstance(getattr(token, attr), np.ndarray)
                        else getattr(token, attr)
                    )
                    for attr in dir(token)
                    if not attr.startswith("_")
                    and not callable(getattr(token, attr))
                    and isinstance(
                        getattr(token, attr),
                        (
                            str,
                            int,
                            float,
                            bool,
                            np.ndarray,
                        ),
                    )
                }
                for token in doc
            ]
        }

        return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
