from transformers import pipeline


def download_from_huggingface(model_name: str, task: str, model_path):

    try:
        # Load pipeline
        pipe = pipeline(task=task, model=model_name)

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
    except:
        return False
