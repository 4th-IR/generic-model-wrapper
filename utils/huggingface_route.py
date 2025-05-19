"""
Function to handle all downloads & inferencing from the huggingface repo
"""

def download_from_huggingface(model_name: str, model_path):
    try:
        if model_name in ['deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B']:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(model_path)  
            model.save_pretrained(model_path)   

        if model_name in ['microsoft/git-base']:
            from transformers import AutoModelForCausalLM, AutoProcessor
            model = AutoModelForCausalLM.from_pretrained(model_name)
            processor = AutoProcessor.from_pretrained(model_name)
            processor.save_pretrained(model_path)  
            model.save_pretrained(model_path) 

        if model_name in ["openai/whisper-large"]:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            processor = WhisperProcessor.from_pretrained(model_name)
            model = WhisperForConditionalGeneration.from_pretrained(model_name)
            processor.save_pretrained(model_path)  
            model.save_pretrained(model_path)

        if model_name in ['Salesforce/blip2-opt-2.7b']:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            processor = Blip2Processor.from_pretrained(model_name, cache_dir=model_path)
            model = Blip2ForConditionalGeneration.from_pretrained(model_name, cache_dir=model_path)
            processor.save_pretrained(model_path)  
            model.save_pretrained(model_path)

        if model_name in ['Salesforce/blip-image-captioning-base']:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            processor = BlipProcessor.from_pretrained(model_name)
            model = BlipForConditionalGeneration.from_pretrained(model_name)
            processor.save_pretrained(model_path)  
            model.save_pretrained(model_path)

        if model_name in ["gpt2"]:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(model_name)
            tokenizer.save_pretrained(model_path)  
            model.save_pretrained(model_path) 

    except Exception as e:
        print(f"Error downloading {model_name}: {e}")

