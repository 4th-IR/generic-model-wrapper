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


LOG = get_logger('model')

class ModelWrapper:
    def __init__(self, provider, model_name, task):
        self.model_provider = provider
        self.model_name = model_name
        self.task = task
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

        self.safe_model_name = self.model_name.replace("/", "_")


    def upload_directory_to_azure(self, local_dir, blob_prefix=""):
        """
        Uploads a local directory to Azure Blob Storage.
        Ensures that blob_prefix is safe by replacing '/' with '_'.
        """
        # Sanitize the blob prefix if needed
        safe_blob_prefix = blob_prefix.replace("/", "_") if blob_prefix else ""

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
        print("<---SAVING TO AZURE--->")
        
        
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


    @timer
    def load_from_azure(self):
        print("<---LOADING MODEL--->")
        
        if not self.azure_config:
            LOG.warning("Azure configuration is disabled.")
            return False


        # Check for blobs in the self.safe_model_name folder
        blob_list = list(self.container_client.list_blobs(name_starts_with=f"{self.safe_model_name}/"))
        if not blob_list:
            LOG.warning(f"No blobs found for model '{self.model_name}' under folder '{self.safe_model_name}/'")
            print("Model blob folder does not exist")
            return False

        # Create a temporary directory to save the model files
        temp_model_path = tempfile.mkdtemp()
        LOG.info(f"Created temporary directory for model download: {temp_model_path}")

        # Download each file under the model folder
        for blob in blob_list:
            blob_client = self.container_client.get_blob_client(blob)

            # Get relative path within the model folder
            relative_path = os.path.relpath(blob.name, start=f"{self.safe_model_name}/")
            local_path = os.path.join(temp_model_path, relative_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            with open(local_path, "wb") as file:
                download_stream = blob_client.download_blob()
                file.write(download_stream.readall())
                LOG.info(f"Downloaded: {blob.name} -> {local_path}")

        LOG.info(f"{self.model_name} downloaded from Azure to: {temp_model_path}")
    
        # Load the model based on the specified provider.
        provider = self.model_provider.lower()
        if provider == "huggingface":
            if self.model_name in ['deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B']:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.model = AutoModelForCausalLM.from_pretrained(temp_model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(temp_model_path) 

            if self.model_name in ["openai/whisper-large"]:
                from transformers import WhisperProcessor, WhisperForConditionalGeneration
                self.processor = WhisperProcessor.from_pretrained(temp_model_path)
                self.model = WhisperForConditionalGeneration.from_pretrained(temp_model_path) 

            if self.model_name in ['Salesforce/blip2-opt-2.7b']:
                from transformers import Blip2Processor, Blip2ForConditionalGeneration
                self.processor = Blip2Processor.from_pretrained(temp_model_path)
                self.model = Blip2ForConditionalGeneration.from_pretrained(temp_model_path)

        else:
            LOG.warning(f"Unsupported model provider: {self.model_provider}")
            return False

        LOG.info("Model successfully loaded from Azure and initialized for inference.")
        return True

    @timer
    def load_model(self):

        """Loads models from Azure Storage if available; otherwise, downloads from a repository."""

        try:
            
            if self.azure_config:
                LOG.info("Checking Azure Storage for model...")
                print("Checking Azure Storage for model...")

                if self.load_from_azure():
                    LOG.info(f"Loaded model from Azure Storage: {self.model_name}")
                    print(f"Loaded model from Azure Storage: {self.model_name}")
                    return
                
                else:
                    LOG.info(f"Downloading model from {self.model_provider}...")

                    if self.model_provider == "huggingface":
                        LOG.info(f"Loading Hugging Face model: {self.model_name}")
                        temp_dir = tempfile.gettempdir()
                        save_path = os.path.join(temp_dir, self.safe_model_name)
                        self.model_save_path = save_path

                        if self.model_name in ['deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B']:
                            from transformers import AutoModelForCausalLM, AutoTokenizer
                            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                            self.tokenizer.save_pretrained(save_path)  
                            self.model.save_pretrained(save_path)     

                        if self.model_name in ["openai/whisper-large"]:
                            from transformers import WhisperProcessor, WhisperForConditionalGeneration
                            self.processor = WhisperProcessor.from_pretrained(self.model_name)
                            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
                            self.model.config.forced_decoder_ids = None
                            self.processor.save_pretrained(save_path)  
                            self.model.save_pretrained(save_path)

                        if self.model_name in ['Salesforce/blip2-opt-2.7b']:
                            from transformers import Blip2Processor, Blip2ForConditionalGeneration
                            self.processor = Blip2Processor.from_pretrained(self.model_name)
                            self.model = Blip2ForConditionalGeneration.from_pretrained(self.model_name)
                            self.processor.save_pretrained(save_path)  
                            self.model.save_pretrained(save_path)

                    else:
                        LOG.error(f"Unknown model provider: {self.model_provider}")
                        raise ValueError(f"Unsupported model provider: {self.model_provider}")

                    LOG.info("Model loaded successfully.")

                    if self.azure_config:
                        LOG.info("Saving model from Provider to Azure...")
                        self.save_to_azure()

                    if os.path.exists(save_path):
                        shutil.rmtree(save_path)
                        LOG.info(f"Deleted temporary directory: {save_path}")

        except Exception as e:
            LOG.error(f"Error loading model: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
        

    @timer
    def run_inference(self, input_data: Any, task: str = None, **kwargs: Any):
        """Runs inference with enhanced multimodal support"""
        try:
       
            if self.model_provider == "huggingface":
                LOG.info(f"Direct HF {task} inference")
                img = None
                txt = None
                aud = None
                for item in input_data:
                    if 'image' in item:
                        img = item['image']
                    elif 'text' in item:
                        txt = item['text']
                    elif 'audio' in item:
                        aud = item['audio']         

                # Inferencing script for whisper
                if self.model_name in ["openai/whisper-large"]:
                    waveform, sample_rate = torchaudio.load(aud)

                    # Resample if needed
                    if sample_rate != 16000:
                        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                        waveform = resampler(waveform)
                        sample_rate = 16000

                    # Whisper expects mono-channel
                    waveform = waveform.mean(dim=0)

                    inputs = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt")

                    with torch.no_grad():
                        generated_ids = self.model.generate(inputs["input_features"])
                        model_output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    
                if self.model_name in ['deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B']:
                    
                    inputs = self.tokenizer(txt, return_tensors="pt")

                    with torch.no_grad():
                        output = self.model.generate(**inputs, max_length=100)

                    model_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    

                if self.model_name in ['Salesforce/blip2-opt-2.7b']:
                    image = Image.open(img).convert("RGB")
                    inputs = self.processor(images=image, text=txt, return_tensors="pt")

                    with torch.no_grad():
                        generated_ids = self.model.generate(**inputs)
                        model_output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                return model_output

        except Exception as e:
            LOG.error(f"Inference failed: {str(e)}")
            raise RuntimeError(f"Inference error: {str(e)}") from e

    
