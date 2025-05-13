import tempfile
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoProcessor,
    AutoImageProcessor,
    AutoModel
)
import os

def from_huggingface(model_name: str):
    save_path = tempfile.gettempdir()
    safe_model_name = model_name.replace("/", "_") 
    save_path = os.path.join("models", safe_model_name)
    os.makedirs(save_path, exist_ok=True)

    # Step 1: Load config
    try:
        config = AutoConfig.from_pretrained(model_name)
    except Exception as e:
        raise RuntimeError(f"Could not load config for {model_name}: {e}")

    # Step 2: Infer correct AutoModel class from config
    try:
        model = AutoModel.from_config(config)
        model_class = model.__class__
    except Exception as e:
        raise RuntimeError(f"Could not infer model class: {e}")

    # Instantiate model
    try:
        model = model_class.from_pretrained(model_name)
        model.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    except Exception as e:
        raise RuntimeError(f"Could not load model weights: {e}")

    # Try saving tokenizer (if available)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_path)
        print("Tokenizer saved.")
    except Exception:
        print("No tokenizer found.")

    # Try saving processor (for audio/vision/multimodal models)
    for processor_cls in [AutoProcessor, AutoImageProcessor]:
        try:
            processor = processor_cls.from_pretrained(model_name)
            processor.save_pretrained(save_path)
            print(f"{processor_cls.__name__} saved.")
        except Exception:
            continue

    # figure out how to make the special cases work

# from_huggingface("Salesforce/blip-image-captioning-base")

