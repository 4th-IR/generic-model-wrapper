import torch
import torchvision.models as models
import torchaudio.models as audio_models
import torchaudio.pipelines
from torchvision.models import get_model_weights
import os


def from_torch(model_name: str, save_path: str, hub_repo: str = None):
    model = None

    # Load model from Torch Hub
    if hub_repo:
        try:
            model = torch.hub.load(hub_repo, model_name, pretrained=True)  # Fix
            print(f"Model '{model_name}' loaded from Torch Hub repository '{hub_repo}'.")
        except Exception as e:
            print(f"Could not load model '{model_name}' from Torch Hub repository '{hub_repo}': {e}")

    # Load model from torchvision.models
    if model is None and hasattr(models, model_name):
        try:
            model = getattr(models, model_name)(weights=get_model_weights(model_name).DEFAULT)  # Fix
            print(f"Model '{model_name}' loaded from torchvision.models.")
        except Exception as e:
            print(f"Could not load model '{model_name}' from torchvision.models: {e}")

    # Load model from torchaudio.models
    if model is None and hasattr(audio_models, model_name):
        try:
            model = getattr(audio_models, model_name)()
            print(f"Model '{model_name}' loaded from torchaudio.models.")
        except Exception as e:
            print(f"Could not load model '{model_name}' from torchaudio.models: {e}")

    # Load model from torchaudio.pipelines
    if model is None:
        pipeline_models = {attr.upper(): getattr(torchaudio.pipelines, attr) for attr in dir(torchaudio.pipelines) if attr.isupper()}  # Fix
        if model_name.upper() in pipeline_models:  
            try:
                bundle = pipeline_models[model_name.upper()]
                model = bundle.get_model()
                print(f"Model '{model_name}' loaded from torchaudio.pipelines.")
            except Exception as e:
                print(f"Could not load model '{model_name}' from torchaudio.pipelines: {e}")

    if model is None:
        raise ValueError(f"Model '{model_name}' not found in Torch Hub, torchvision.models, torchaudio.models, or torchaudio.pipelines.")

    os.makedirs(save_path, exist_ok=True)
    model_save_path = os.path.join(save_path, f"{model_name}.pt")

    # accordng to my research this is the best way to save models for production: torch.jit.script 
    model = torch.jit.script(model)
    torch.jit.save(model, model_save_path)

    print(f"Model '{model_name}' saved to '{model_save_path}'")

    return model_save_path
