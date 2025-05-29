""" A Generic Model Wrapper Class to Standardize Framework Agnostic Model Loading and Inference """


#external 
import os
import torch
import time
import tempfile
from tqdm import tqdm 
from typing import Any
from PIL import Image
import numpy as np
import tensorflow as tf
from fastapi import HTTPException
from huggingface_hub import login
from azure.storage.blob import BlobServiceClient
import torchaudio
from torchvision.transforms import transforms
import shutil



#internal 
from utils.logger import get_logger
from utils.env_manager import AZURE_BLOB_URI, AZURE_CONNECTION_STRING, AZURE_CONTAINER_NAME, AZURE_STORAGE_ACCOUNT
from utils.resource_manager import timer
from utils.huggingface_route import download_from_huggingface


LOG = get_logger('model')

class ModelWrapper:
    def __init__(self, provider, model_name, task):
        self.model_provider = provider.lower() if provider else None
        self.model_name = model_name
        self.task = task.lower() if task else None
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_save_path = None 
        self.processor = None


        # Azure storage setup        
        if not AZURE_CONNECTION_STRING:
            raise ValueError("Azure connection string is missing! Check environment variables.")
        if not AZURE_CONTAINER_NAME:
            raise ValueError("Azure container name is missing! Check environment variables.")
        
        self.blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        self.container_client = self.blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
        self.azure_config = True  # Enable Azure model loading
        
        # model name to ensure proper dir creation
        self.safe_model_name = self.model_name.replace("/", "__")
        # temp files to store the model loaded for inferencing
        self.temp_model_inference_path = tempfile.mkdtemp()
        

    def upload_directory_to_azure(self, local_dir, blob_prefix=""):
        """
        Uploads a local directory to Azure Blob Storage.
        Ensures that blob_prefix is safe by replacing '/' with '_'.
        """
        # Sanitize the blob prefix if needed
        safe_blob_prefix = blob_prefix.replace("/", "__") if blob_prefix else ""

        for root, _, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                blob_name = os.path.relpath(local_path, local_dir)
                
                if safe_blob_prefix:
                    blob_name = f"{safe_blob_prefix}/{blob_name}"
                
                blob_client = self.container_client.get_blob_client(blob_name)
                with open(local_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)


    def save_to_azure(self): 
        """Saves the model directory/files to Azure Blob Storage under a folder named after the model."""
        
        LOG.info(f"Attempting to save model '{self.model_name}' to Azure container '{AZURE_CONTAINER_NAME}'.")

        if not self.model_save_path or not os.path.exists(self.model_save_path):
            error_msg = f"Model save path '{self.model_save_path}' is not valid or model hasn't been saved locally first."
            LOG.error(error_msg)
            raise ValueError(error_msg)

        try:
            LOG.info(f'Local model path to upload: {self.model_save_path}')
            
            # Pass the safe model name as the blob prefix
            self.upload_directory_to_azure(
                local_dir=self.model_save_path,
                blob_prefix=self.safe_model_name
            )

            LOG.info(f"Successfully saved '{self.model_name}' to Azure Blob Storage in container '{AZURE_CONTAINER_NAME}' under folder '{self.safe_model_name}'.")
        except Exception as e:
            LOG.error(f'Failed to save model {self.model_name} to Azure: {e}', exc_info=True)
            raise RuntimeError(f'Model saving to Azure failed: {e}')

        LOG.info(f"{self.model_name} saved to Azure Blob Storage under folder '{self.safe_model_name}'")


    def load_from_azure(self):

        # Check for blobs in the self.safe_model_name folder
        blob_list = list(self.container_client.list_blobs(name_starts_with=f"{self.safe_model_name}/"))
        if not blob_list:
            LOG.warning(f"No blobs found for model '{self.model_name}' under folder '{self.safe_model_name}/'")
            print("Model blob folder does not exist")
            return False

        # Download each file under the model folder
        for blob in blob_list:
            blob_client = self.container_client.get_blob_client(blob)

            # Get relative path within the model folder
            relative_path = os.path.relpath(blob.name, start=f"{self.safe_model_name}/")
            local_path = os.path.join(self.temp_model_inference_path, relative_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            with open(local_path, "wb") as file:
                download_stream = blob_client.download_blob()
                file.write(download_stream.readall())
                LOG.info(f"Downloaded: {blob.name} -> {local_path}")

        LOG.info(f"{self.model_name} downloaded from Azure to: {self.temp_model_inference_path}")
        LOG.info("Model successfully loaded from Azure and initialized for inference.")
        return True

    
    def load_model(self):

        """Loads models from Azure Storage if available; otherwise, downloads from a repository."""

        try:      
            if self.azure_config:
                LOG.info("Checking Azure Storage for model...")
                print("Checking Azure Storage for model...")

                if self.load_from_azure():
                    LOG.info(f"Loaded model from Azure Storage: {self.model_name}")
                    print(f"Loaded model from Azure Storage: {self.model_name}")
                
                else:
                    LOG.info(f"Downloading model from {self.model_provider}...")

                    if self.model_provider == "huggingface":
                        LOG.info(f"Loading Hugging Face model: {self.model_name}")
                        temp_download_dir = tempfile.gettempdir()
                        save_path = os.path.join(temp_download_dir, self.safe_model_name)
                        self.model_save_path = save_path

                        download_from_huggingface(model_name=self.model_name, model_path=save_path)
                        
                        if self.azure_config:
                            LOG.info("Saving model from Provider to Azure...")
                            self.save_to_azure()

                        if os.path.exists(save_path):
                            shutil.rmtree(save_path)
                            LOG.info(f"Deleted temporary directory: {save_path}")

                    else:
                        LOG.error(f"Unknown model provider: {self.model_provider}")
                        raise ValueError(f"Unsupported model provider: {self.model_provider}")

                    LOG.info("Model loaded successfully.")

        except Exception as e:
            LOG.error(f"Error loading model: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
        
    
    def run_inference(self, messages, task: str = None, **kwargs: Any):
        """Runs inference with enhanced multimodal support"""

        try:
            
            if self.model_provider == "huggingface":
                LOG.info(f"Direct HF {task} inference")
                      
                def parse_messages(messages):
                    img = txt = aud = None

                    for message in messages:
                        if message['role'] != 'user':
                            continue
                        content = message['content']

                        if isinstance(content, dict):
                            img = content.get('image', img)
                            txt = content.get('text', txt)
                            aud = content.get('audio', aud)
                        # elif isinstance(content, str):
                        #     txt = content
                        print('image', img)
                        print('text', txt)
                        print('audio', aud)
                    return img, txt, aud
                
                img, txt, aud = parse_messages(messages)
                # Inferencing script for whisper
                if self.model_name == "openai/whisper-large":
                    from transformers import WhisperProcessor, WhisperForConditionalGeneration
                    self.processor = WhisperProcessor.from_pretrained(self.temp_model_inference_path)
                    self.model = WhisperForConditionalGeneration.from_pretrained(self.temp_model_inference_path) 
                    waveform, sample_rate = torchaudio.load(aud)

                    if sample_rate != 16000:
                        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                        waveform = resampler(waveform)
                        sample_rate = 16000

                    waveform = waveform.mean(dim=0)  

                    inputs = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt")

                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            inputs["input_features"], 
                            attention_mask=inputs.get("attention_mask"),
                            max_new_tokens=200
                        ) 

                    model_output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
      
                if self.model_name == 'Salesforce/blip2-opt-2.7b':
                    from transformers import Blip2Processor, Blip2ForConditionalGeneration
                    self.processor = Blip2Processor.from_pretrained(self.temp_model_inference_path)
                    self.model = Blip2ForConditionalGeneration.from_pretrained(self.temp_model_inference_path)

                    image = Image.open(img).convert("RGB")
                    inputs = self.processor(images=image, text=txt, return_tensors="pt")

                    with torch.no_grad():
                        generated_ids = self.model.generate(**inputs)
                        model_output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                if self.model_name == "Salesforce/blip-image-captioning-base":
                    from transformers import BlipProcessor, BlipForConditionalGeneration
                    self.processor = BlipProcessor.from_pretrained(self.temp_model_inference_path)
                    self.model = BlipForConditionalGeneration.from_pretrained(self.temp_model_inference_path)
                    image = Image.open(img).convert('RGB')

                    # conditional image captioning
                    text = txt
                    inputs = self.processor(image, text, return_tensors="pt")

                    out = self.model.generate(**inputs)
                    model_output = self.processor.decode(out[0], skip_special_tokens=True)

                if self.model_name == 'microsoft/git-base':
                    from transformers import GitProcessor, GitForCausalLM
                    self.model = GitForCausalLM.from_pretrained(self.temp_model_inference_path)
                    self.processor = GitProcessor.from_pretrained(self.temp_model_inference_path)

                    image = Image.open(img).convert('RGB')

                    pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
                    generated_ids = self.model.generate(pixel_values=pixel_values, max_new_tokens=50)
                    model_output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                if self.model_name == 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B':  
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    self.model = AutoModelForCausalLM.from_pretrained(self.temp_model_inference_path)
                    self.tokenizer = AutoTokenizer.from_pretrained(self.temp_model_inference_path)  
                    prompt = txt
                    inputs = self.tokenizer(prompt, return_tensors="pt")

                    # Generate output
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=100,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=self.tokenizer.eos_token_id
                        )

                    # Decode and return result
                    model_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                if self.model_name == 'gpt2':
                    from transformers import GPT2LMHeadModel, GPT2Tokenizer
                    self.tokenizer = GPT2Tokenizer.from_pretrained(self.temp_model_inference_path)
                    self.model = GPT2LMHeadModel.from_pretrained(self.temp_model_inference_path)

                    prompt = txt

                    # Ensure the tokenizer has a pad token
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                        self.model.config.pad_token_id = self.tokenizer.pad_token_id

                    # Tokenize input and get attention mask
                    inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
                    input_ids = inputs["input_ids"]
                    attention_mask = inputs["attention_mask"]

                    # Generate text from the prompt
                    with torch.no_grad():
                        output = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,        
                            max_length=100,                       
                            num_return_sequences=1,
                            no_repeat_ngram_size=2,
                            do_sample=True,
                            temperature=0.9,
                            top_k=50,
                            top_p=0.95,
                            pad_token_id=self.tokenizer.pad_token_id  
                        )

                    # Decode and print the generated text
                    model_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    
            messages.append({
                            "role": "assistant",
                            "content": model_output
                        })

            return messages

        except Exception as e:
            LOG.error(f"Inference failed: {str(e)}")
            raise RuntimeError(f"Inference error: {str(e)}") from e

if __name__ == "__main__":
    model_name = 'microsoft/git-base'
    provider = "huggingface"
    task = "vqa"
    # Instantiate the wrapper
    model_wrapper = ModelWrapper(provider=provider, model_name=model_name, task=task)

    # Load the model
    model_wrapper.load_model()

    # Run inference with input as a list of dictionaries
    output = model_wrapper.run_inference(messages = [
        {
            "role": "user",
            "content": {
                "text": "What is this image showing?",
                "image": "/home/model-wrapper/tests/assets/images/animal_pictures/dog1.jpg",
                # "audio": "/home/model-wrapper/tests/assets/audios/audio3.mp3"
            }
        }
    ]
    )

    print(output)
    print(f'{ "*" * 5 }')
    print(output[-1])
