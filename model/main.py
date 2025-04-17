""" A Generic Model Wrapper Class to Standardize Framework Agnostic Model Loading and Inference """


#external 
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
import torchaudio



#internal 
from utils.logger import get_logger
from utils.env_manager import AZURE_BLOB_URI, AZURE_CONNECTION_STRING, AZURE_CONTAINER_NAME, AZURE_STORAGE_ACCOUNT
from utils.resource_manager import timer
from utils.torch_route import from_torch
from utils.tensorflow_route import from_tensorflow

HGF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
login(HGF_TOKEN)



LOG = get_logger('model')


class ModelWrapper:
    def __init__(self, provider, model_name, pipeline_type):
        self.model_provider = provider
        self.model_name = model_name
        self.task = pipeline_type
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.pipeline = None
        self.saved_path = "models_saved"
        self.model_save_path = None 

        #os.makedirs(self.saved_path, exist_ok=True)

        # Azure storage setup        
        if not AZURE_CONNECTION_STRING:
            raise ValueError("Azure connection string is missing! Check environment variables.")
        if not AZURE_CONTAINER_NAME:
            raise ValueError("Azure container name is missing! Check environment variables.")
        
        self.blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        self.container_client = self.blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
        self.azure_config = True  # Enable Azure model loading

    #@timer
    def save_to_azure(self):
        print("<---SAVING TO AZURE--->")
    
        if not self.model and not self.model_save_path:
            LOG.error("Attempted to save a model that is not loaded.")
            raise ValueError("Model is not loaded and saved")
        
        #file_size = os.stat(self.model_save_path).st_size
        #blob_client = self.container_client.get_blob_client(blob=self.model_name)
        
        # with tqdm(total=file_size, unit="B", unit_scale=True, desc="Upload Progress") as progress_bar:
        #     def progress_callback(bytes_uploaded):
        #         # Update progress_bar: calculate the difference between the current uploaded bytes
        #         # and the last reported amount (progress_bar.n).
        #         progress_bar.update(bytes_uploaded - progress_bar.n)

        # with open(self.model_save_path, "rb") as data:
        #     blob_client.upload_blob(data, overwrite=True)

        try:
            self.upload_directory_to_azure(
                self.model_save_path
            )
        except Exception as e:
            LOG.warn(f'Failed to save with error {e}')

        LOG.info(f"{self.model_name} saved to Azure Blob Storage")

    #@timer
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
                        save_path = os.path.join(temp_dir, self.model_name)
                        self.model.save_pretrained(save_path)
                        self.tokenizer.save_pretrained(save_path)
                        self.model_save_path = save_path

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
    def run_inference(self, input_data: Any, task: str = None, **kwargs: Any):
        """Runs inference with enhanced multi-modal support"""
        try:
            # Common preprocessing for all frameworks
            def _convert_to_tensor(data):
                if isinstance(data, np.ndarray):
                    return torch.from_numpy(data) if self.provider == "pytorch" else tf.convert_to_tensor(data)
                return data

            # Hugging Face Pipeline First (now with audio support)
            if self.pipeline:
                LOG.info(f"Using HF pipeline for {task or 'auto'} task")
                # Handle audio inputs specifically
                if task in ["automatic-speech-recognition", "audio-classification"]:
                    if isinstance(input_data, (str, bytes, np.ndarray)):
                        return self.pipeline(input_data, task=task, **kwargs)
                return self.pipeline(input_data, **kwargs)
            
            # Hugging Face Direct Execution
            if self.provider == "huggingface" and self.model:
                LOG.info(f"Direct HF {task} inference")
                
                # Audio Processing
                if task in ["audio-classification", "automatic-speech-recognition"]:
                    audio = self._load_audio(input_data)
                    inputs = self.feature_extractor(
                        audio, 
                        sampling_rate=16000, 
                        return_tensors="pt" if self.device == "cpu" else "tf"
                    )
                    outputs = self.model(**inputs)
                    return self._postprocess_audio(outputs, task)

                # Text Generation & Seq2Seq
                if task in ["text-generation", "text2text-generation"]:
                    inputs = self.tokenizer(input_data, return_tensors="pt" if self.device == "cpu" else "tf")
                    outputs = self.model.generate(
                        **inputs,
                        max_length=kwargs.get("max_length", 512),
                        num_beams=kwargs.get("num_beams", 5)
                    )
                    return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Object Detection & Segmentation
                if task in ["object-detection", "image-segmentation"]:
                    image = self._load_image(input_data)
                    inputs = self.image_processor(images=image, return_tensors="pt" if self.device == "cpu" else "tf")
                    outputs = self.model(**inputs)
                    return self.image_processor.post_process_object_detection(
                        outputs, 
                        threshold=kwargs.get("threshold", 0.9)
                    ) if task == "object-detection" else \
                    self.image_processor.post_process_semantic_segmentation(outputs)

                # Sequence Classification Fallback
                if isinstance(input_data, str):
                    inputs = self.tokenizer(input_data, return_tensors="pt" if self.device == "cpu" else "tf")
                    return self.model(**inputs).logits

            # PyTorch Specific Processing
            if self.provider == "pytorch" and isinstance(self.model, torch.nn.Module):
                # Audio Processing
                if task == "audio-classification":
                    waveform = _convert_to_tensor(input_data).float()
                    mel_spec = torchaudio.transforms.MelSpectrogram()(waveform)
                    return self.model(mel_spec.unsqueeze(0))
                
                # Object Detection
                if task == "object-detection":
                    from torchvision import transforms
                    preprocess = transforms.Compose([
                        transforms.Resize(800),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    img_tensor = preprocess(input_data).unsqueeze(0)
                    with torch.no_grad():
                        outputs = self.model(img_tensor)
                    return [{k: v.cpu() for k, v in t.items()} for t in outputs]

            # TensorFlow Specific Processing  
            if self.provider == "tensorflow" and isinstance(self.model, tf.keras.Model):
                # Audio Processing
                if task == "audio-classification":
                    spec = tf.signal.stft(_convert_to_tensor(input_data), frame_length=255, frame_step=128)
                    spec = tf.abs(spec)[..., tf.newaxis]
                    return self.model.predict(spec[tf.newaxis, ...])
                
                # Image Segmentation
                if task == "image-segmentation":
                    img = tf.image.resize(input_data, (256, 256))
                    img = tf.cast(img, tf.float32) / 255.0
                    return self.model.predict(img[tf.newaxis, ...])[0]

            # Common Postprocessing
            def _postprocess_audio(outputs, task):
                if task == "audio-classification":
                    return torch.nn.functional.softmax(outputs.logits, dim=-1)
                return outputs.logits  # For speech recognition

        except Exception as e:
            LOG.error(f"Inference failed: {str(e)}")
            raise RuntimeError(f"Inference error: {str(e)}") from e


    def _load_image(self, input_data):
        if isinstance(input_data, str):
            return Image.open(input_data).convert("RGB")
        return input_data

    def _load_audio(self, input_data):
        if isinstance(input_data, str):
            waveform, sample_rate = torchaudio.load(input_data)
            return waveform.numpy()
        return input_data
    
    def upload_directory_to_azure(self, local_dir, blob_prefix=""):
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                blob_name = os.path.relpath(local_path, local_dir)
                if blob_prefix:
                    blob_name = f"{blob_prefix}/{blob_name}"
                
                blob_client = self.container_client.get_blob_client(blob_name)
                with open(local_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)

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


