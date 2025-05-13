from transformers.models import auto

auto_model_classes = [name for name in dir(auto) if name.startswith("AutoModel")]
print(auto_model_classes)

