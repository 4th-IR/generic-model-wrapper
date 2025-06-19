import tempfile

from typing import Dict
from transformers import pipeline

from logging import getLogger

logs = getLogger("huggingface_route")


def download_from_huggingface(
    model_name: str, task: str, model_path: str, kwargs: Dict[str, str]
):

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dict = {"task": task, "model": model_name, "kwargs": kwargs}

            filtered_input_dict = {k: v for k, v in input_dict.items() if v is not None}

            # Load pipeline
            pipe = pipeline(**filtered_input_dict, model_kwargs={"cache_dir": tmpdir})

            # Save model
            pipe.save_pretrained(model_path)

            # Try saving tokenizer or processor
            if hasattr(pipe, "tokenizer") and pipe.tokenizer is not None:
                pipe.tokenizer.save_pretrained(model_path)
                print("Tokenizer saved.")
            elif hasattr(pipe, "processor") and pipe.processor is not None:
                pipe.processor.save_pretrained(model_path)
                print("Processor saved.")
            else:
                print("No tokenizer or processor found to save.")

        return True
    except Exception as e:
        logs.error(e, exc_info=True)
        return False
