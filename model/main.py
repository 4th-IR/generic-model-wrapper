import os
import keras 
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
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModelForImageClassification
from utils.torch_route import from_torch
from utils.tensorflow_route import from_tensorflow


#internal 
from utils.logger import get_logger
from utils.env_manager import AZURE_BLOB_URI, AZURE_CONNECTION_STRING, AZURE_CONTAINER_NAME, AZURE_STORAGE_ACCOUNT
from utils.resource_manager import timer

HGF_TOKEN = os.getenv("HUGGINGFACE ")
login(HGF_TOKEN)



LOG = get_logger('model')


class ModelWrapper:
    def __init__(self, provider, model_name, pipeline_type):
        self.model_provider = provider
        self.model_name = model_name
        self.task = pipeline_type
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.saved_path = "models_saved"
        self.model_save_path = None 

        os.makedirs(self.saved_path, exist_ok=True)

        # Azure storage setup        
        if not AZURE_CONNECTION_STRING:
            raise ValueError("Azure connection string is missing! Check environment variables.")
        if not AZURE_CONTAINER_NAME:
            raise ValueError("Azure container name is missing! Check environment variables.")
        
        self.blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        self.container_client = self.blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
        self.azure_config = True  # Enable Azure model loading

    @timer
    def save_to_azure(self):
        print("<---SAVING TO AZURE--->")
    
        if not self.model and not self.model_save_path:
            LOG.error("Attempted to save a model that is not loaded.")
            raise ValueError("Model is not loaded and saved")
        
        file_size = os.stat(self.model_save_path).st_size
        blob_client = self.container_client.get_blob_client(blob=self.model_name)
        
        with tqdm(total=file_size, unit="B", unit_scale=True, desc="Upload Progress") as progress_bar:
            def progress_callback(bytes_uploaded):
                # Update progress_bar: calculate the difference between the current uploaded bytes
                # and the last reported amount (progress_bar.n).
                progress_bar.update(bytes_uploaded - progress_bar.n)

            with open(self.model_save_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True, progress_hook=progress_callback, timeout=600)

        LOG.info(f"{self.model_name} saved to Azure Blob Storage")

    @timer
    def load_from_azure(self):
        print("<---LOADING MODEL--->")
        if not self.azure_config:
            LOG.warning("Azure configuration is disabled.")
            return False

        # Get the blob client for the model and check if it exists.
        blob_client = self.container_client.get_blob_client(blob=self.model_name)
        if not blob_client.exists():
            LOG.warning("Model blob does not exist")
            print("Model blob does not exist")
            return False

        # Create a temporary file to hold the downloaded model.
        # Using delete=False because we need to pass the file path to the loader.
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            temp_model_path = tmp.name
            download_stream = blob_client.download_blob()
            tmp.write(download_stream.readall())

        LOG.info(f"{self.model_name} downloaded from Azure to temporary file {temp_model_path}")

        # Load the model based on the specified provider.
        provider = self.model_provider.lower()
        if provider == "huggingface":
            try:
                # For Hugging Face, typically from_pretrained expects a directory.
                # If your temporary file is a compressed archive, consider extracting it.
                self.model = AutoModelForCausalLM.from_pretrained(temp_model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(temp_model_path)
                self.pipeline = pipeline(self.task, model=self.model, tokenizer=self.tokenizer)
            except Exception as e:
                LOG.error(f"Failed to load Hugging Face model: {e}")
                return False

        elif provider == "pytorch":
            try:
                # Try loading as a TorchScript model.
                self.model = torch.jit.load(temp_model_path, map_location="cpu")
                self.model.eval()
            except RuntimeError as e:
                print(f"torch.jit.load() failed: {e}. Trying torch.load() instead...")
                try:
                    # Fallback: load as a full PyTorch model.
                    self.model = torch.load(temp_model_path, map_location="cpu")
                    self.model.eval()
                except Exception as e2:
                    LOG.error(f"Failed to load PyTorch model: {e2}")
                    return False

        elif provider == "tensorflow":
            try:
                # TensorFlow's load_model expects the SavedModel format. 
                # Make sure your temporary file is a valid SavedModel.
                self.model = tf.keras.models.load_model(temp_model_path)
                print("TensorFlow model successfully loaded.")
            except Exception as e:
                LOG.error(f"Failed to load TensorFlow model: {e}")
                return False

        else:
            LOG.warning(f"Unsupported model provider: {self.model_provider}")
            return False

        LOG.info("Model successfully loaded from Azure and initialized for inference.")
        return True

    @timer
    def load_model(self):

        """Loads models from Azure Storage if available; otherwise, downloads from a repository."""
        # since we are using a local directory, we will check it before 

        try:
            # for root, _, files in os.walk(self.saved_path):
            #     if self.model_name in files:
            #         print("Model found Locally")

            
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
                        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                        self.pipeline = pipeline(self.task, model=self.model, tokenizer=self.tokenizer)
                        temp_dir = tempfile.gettempdir()
                        self.pipeline.save_pretrained(temp_dir)
                        self.model_save_path = temp_dir

                    elif self.model_provider == "pytorch":
                        LOG.info(f"Loading PyTorch model from {self.model_name}")
                        print(f"Loading PyTorch model from {self.model_name}")
                        self.model_save_path = from_torch(self.model_name) 

                    elif self.model_provider == "tensorflow":
                        LOG.info(f"Loading TensorFlow model from {self.model_name}")
                        print(f"Loading TensorFlow model from {self.model_name}")
                        self.model_save_path = from_tensorflow(self.model_name)

                    else:
                        LOG.error(f"Unknown model provider: {self.model_provider}")
                        raise ValueError(f"Unsupported model provider: {self.model_provider}")

                    LOG.info("Model loaded successfully.")

                    if self.azure_config:
                        LOG.info("Saving model from Provider to Azure...")
                        self.save_to_azure()

        except Exception as e:
            LOG.error(f"Error loading model: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
        

    @timer
    def run_inference(self, input_data: Any, **kwargs: Any):
        """Runs inference using the loaded model 
        Args:
            input 
        """
        if self.pipeline:
            LOG.info("Using Hugging Face pipeline for inference.")
            try:
                return self.pipeline(input_data, **kwargs)
            except Exception as e:
                LOG.error(f"Hugging Face pipeline inference failed: {e}")

        elif self.provider == "huggingface" and self.model is not None:
            LOG.info("Attempting direct Hugging Face model inference.")
            try:
                if isinstance(input_data, str) and self.tokenizer:
                    inputs = self.tokenizer(input_data, return_tensors="pt", truncation=True, padding=True)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    return outputs
                elif isinstance(input_data, (Image.Image, str)) and self.feature_extractor and isinstance(self.model, AutoModelForImageClassification):
                    try:
                        if isinstance(input_data, str):
                            image = Image.open(input_data).convert("RGB")
                        else:
                            image = input_data.convert("RGB")
                        inputs = self.feature_extractor(images=[image], return_tensors="pt")
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                        return torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
                    except Exception as e:
                        LOG.warning(f"Image classification inference failed: {e}")
                else:
                    LOG.warning("Could not automatically determine Hugging Face inference method based on input.")
                    return None
            except Exception as e:
                LOG.error(f"Direct Hugging Face inference failed: {e}")
                return None

        elif self.provider == "pytorch" and isinstance(self.model, torch.nn.Module):
            LOG.info("Attempting direct PyTorch model inference.")
            try:
                self.model.eval()
                with torch.no_grad():
                    if isinstance(input_data, np.ndarray):
                        input_tensor = torch.from_numpy(input_data).float()
                        if len(input_tensor.shape) == 3:
                            input_tensor = input_tensor.unsqueeze(0).permute(0, 3, 1, 2)
                        elif len(input_tensor.shape) == 2:
                            input_tensor = input_tensor.unsqueeze(0)
                    elif isinstance(input_data, torch.Tensor):
                        input_tensor = input_data
                        if len(input_tensor.shape) == 3:
                            input_tensor = input_tensor.unsqueeze(0).permute(0, 3, 1, 2)
                        elif len(input_tensor.shape) == 2:
                            input_tensor = input_tensor.unsqueeze(0)
                    else:
                        LOG.warning(f"Unsupported input type for PyTorch inference: {type(input_data)}")
                        return None
                    return self.model(input_tensor)
            except Exception as e:
                LOG.error(f"PyTorch inference failed: {e}")
                return None

        elif self.provider == "tensorflow" and isinstance(self.model, keras.Model):
            LOG.info("Attempting direct TensorFlow model inference.")
            try:
                return self.model.predict(input_data, **kwargs)
            except Exception as e:
                LOG.error(f"TensorFlow inference failed: {e}")
                return None

        LOG.error("Could not automatically determine inference method.")
        raise RuntimeError("Could not automatically determine inference method.")



# if __name__ == "__main__":
   
#     """
#     I wil be testing these models tomorrow and updating the sheet. 
#     """



#     from tests.inferencing import inference_model
#     import pandas as pd
#     import openpyxl
#     import psutil
    
#     models_dict = {
#     "model_1": {"model_provider": "pytorch", "model_category": "torch_audio", "model_name": "WAV2VEC2_BASE"},
#     "model_2": {"model_provider": "pytorch", "model_category": "torch_audio", "model_name": "EMFORMER_RNNT_BASE_LIBRISPEECH"},
#     "model_3": {"model_provider": "pytorch", "model_category": "torch_vision", "model_name": "ResNet50"},
#     "model_4": {"model_provider": "pytorch", "model_category": "torch_audio", "model_name": "Wav2Vec2Bundle"},
#     "model_5": {"model_provider": "pytorch", "model_category": "torch_vision", "model_name": "inception_v3"},
#     "model_6": {"model_provider": "pytorch", "model_category": "torch_audio", "model_name": "EMFORMER_RNNT_BASE_LIBRISPEECH"},
#     "model_7": {"model_provider": "pytorch", "model_category": "torch_vision", "model_name": "shufflenet_v2_x1_5"},
#     "model_8": {"model_provider": "pytorch", "model_category": "torch_audio", "model_name": "TACOTRON2_WAVERNN_CHAR_LJSPEECH"},
#         }

#     excel_file = 'torch_model_metrics.xlsx'

#     model_inference_metrics = []

#     for model in models_dict.values():

#         model_name = model["model_name"]
#         model_provider = model["model_provider"]
#         model_category = model["model_category"]


#         start_inference_time = time.time()
#         process = psutil.Process(os.getpid())

#         mem_before = process.memory_info().rss / (1024 ** 2)

#         model_wrapper = ModelWrapper(model_provider, model_name, model_category)
#         model_wrapper.load_model()

#         print("/nDownload completed")

#         inference_model(model_provider, model_name, model_category)

#         end_inference_time = time.time()

#         mem_after = process.memory_info().rss / (1024 ** 2)

#         mem_used = mem_after - mem_before


#         TOTAL_TIME_TAKEN = round((end_inference_time - start_inference_time)/60, 2)
#         model_data = {'model_name': model_name,
#                         'model_provider': model_provider,
#                         "model_category": model_category,
#                         "total_time_taken(mins)": TOTAL_TIME_TAKEN,
#                         "memory_used": mem_used}
                                
#         model_inference_metrics.append(model_data)
#         print("/n Metrics obtained", model_inference_metrics)

#     df = pd.DataFrame(model_inference_metrics)

#     if os.path.exists(excel_file):
#         existing_df = pd.read_excel(excel_file)
#         updated_df = pd.concat([existing_df, df], ignore_index=True)
#     else:
#         updated_df = df

#     updated_df.to_excel(excel_file, index=False)

#     print(f"Saved {len(df)} model entries to {excel_file}")


