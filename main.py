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
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoProcessor,
    AutoImageProcessor,
    AutoModel,
    pipeline,
)
import torchaudio
from torchvision.transforms import transforms
from configs.hf_task_mapping import task_automodel_mapping
from transformers import BlipProcessor, BlipForConditionalGeneration



#internal 
from utils.logger import get_logger
from utils.env_manager import AZURE_BLOB_URI, AZURE_CONNECTION_STRING, AZURE_CONTAINER_NAME, AZURE_STORAGE_ACCOUNT
from utils.resource_manager import timer
from utils.torch_route import from_torch
from utils.tensorflow_route import from_tensorflow

# HGF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
# login(HGF_TOKEN)



LOG = get_logger('model')


class ModelWrapper:
    def __init__(self, provider, model_name, pipeline_type, model_category):
        self.model_provider = provider
        self.model_name = model_name
        self.task = pipeline_type
        self.model_category = model_category
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.saved_path = "models_saved"
        self.model_save_path = None 
        self.processor = None

        #os.makedirs(self.saved_path, exist_ok=True)

        # Azure storage setup        
        if not AZURE_CONNECTION_STRING:
            raise ValueError("Azure connection string is missing! Check environment variables.")
        if not AZURE_CONTAINER_NAME:
            raise ValueError("Azure container name is missing! Check environment variables.")
        
        self.blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        self.container_client = self.blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
        self.azure_config = True  # Enable Azure model loading

   
    def save_to_azure(self): 
        """Saves the model directory/files to Azure Blob Storage under a folder named after the model."""
        print("<---SAVING TO AZURE--->")
        safe_model_name = self.model_name.replace("/", "_")
        
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
                blob_prefix=safe_model_name
            )

            LOG.info(f"Successfully saved '{self.model_name}' to Azure Blob Storage in container '{AZURE_CONTAINER_NAME}' under folder '{safe_model_name}'.")
        except Exception as e:
            LOG.error(f'Failed to save model {self.model_name} to Azure: {e}', exc_info=True)
            raise RuntimeError(f'Model saving to Azure failed: {e}')

        LOG.info(f"{self.model_name} saved to Azure Blob Storage under folder '{safe_model_name}'")


    #@timer
    def load_from_azure(self):
        print("<---LOADING MODEL--->")
        
        if not self.azure_config:
            LOG.warning("Azure configuration is disabled.")
            return False

        # Use the safe model name (matching how it was saved)
        safe_model_name = self.model_name.replace("/", "_")

        # Check for blobs in the safe_model_name folder
        blob_list = list(self.container_client.list_blobs(name_starts_with=f"{safe_model_name}/"))
        if not blob_list:
            LOG.warning(f"No blobs found for model '{self.model_name}' under folder '{safe_model_name}/'")
            print("Model blob folder does not exist")
            return False

        # Create a temporary directory to save the model files
        temp_model_path = tempfile.mkdtemp()
        LOG.info(f"Created temporary directory for model download: {temp_model_path}")

        # Download each file under the model folder
        for blob in blob_list:
            blob_client = self.container_client.get_blob_client(blob)

            # Get relative path within the model folder
            relative_path = os.path.relpath(blob.name, start=f"{safe_model_name}/")
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
            # For Hugging Face, typically from_pretrained expects a directory.
            # try:
            #     config = AutoConfig.from_pretrained(self.model_name)
            # except Exception as e:
            #     raise RuntimeError(f"Could not load config for {self.model_name}: {e}")

            # # Step 2: Infer correct AutoModel class from config
            # try:
            #     self.model = AutoModel.from_config(config)
            #     model_class = self.model.__class__
            # except Exception as e:
            #     raise RuntimeError(f"Could not infer model class: {e}")

            # # Instantiate model
            # try:
            #     self.model = model_class.from_pretrained(self.model_name)
            #     self.model.save_pretrained(temp_model_path)
            #     print(f"Model saved to {temp_model_path}")
            # except Exception as e:
            #     raise RuntimeError(f"Could not load model weights: {e}")

            # # Try saving tokenizer (if available)
            # try:
            #     self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            #     self.tokenizer.save_pretrained(temp_model_path)
            #     print("Tokenizer saved.")
            # except Exception:
            #     print("No tokenizer found.")
            #     self.tokenizer = None

            # # Try saving processor (for audio/vision/multimodal models)
            #     for processor_cls in [AutoProcessor, AutoImageProcessor]:
            #         try:
            #             processor = processor_cls.from_pretrained(self.model_name)
            #             processor.save_pretrained(temp_model_path)
            #             print(f"{processor_cls.__name__} saved.")
            #         except Exception:
            #             processor = None
            #             continue
                
            #     if self.tokenizer:
            #         self.pipeline = pipeline(self.task, model=self.model, tokenizer=self.tokenizer)
            #     if self.processor:
            #         self.pipeline = pipeline(self.task, model=self.model, processor=self.processor)

            self.processor = BlipProcessor.from_pretrained(temp_model_path)
            self.model = BlipForConditionalGeneration.from_pretrained(temp_model_path)   
            self.pipeline = pipeline(self.task, model=self.model, processor=self.processor)

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
                        save_path = os.path.join(temp_dir, self.model_name)
                        self.model_save_path = save_path

                        # Step 1: Load config
                        try:
                            config = AutoConfig.from_pretrained(self.model_name)
                        except Exception as e:
                            raise RuntimeError(f"Could not load config for {self.model_name}: {e}")

                        # Step 2: Infer correct AutoModel class from config
                        try:
                            model = AutoModel.from_config(config)
                            model_class = model.__class__
                        except Exception as e:
                            raise RuntimeError(f"Could not infer model class: {e}")

                        # Instantiate model
                        try:
                            model = model_class.from_pretrained(self.model_name)
                            model.save_pretrained(save_path)
                            print(f"Model saved to {save_path}")
                        except Exception as e:
                            raise RuntimeError(f"Could not load model weights: {e}")

                        # Try saving tokenizer (if available)
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                            tokenizer.save_pretrained(save_path)
                            print("Tokenizer saved.")
                        except Exception:
                            print("No tokenizer found.")

                        # Try saving processor (for audio/vision/multimodal models)
                        for processor_cls in [AutoProcessor, AutoImageProcessor]:
                            try:
                                processor = processor_cls.from_pretrained(self.model_name)
                                processor.save_pretrained(save_path)
                                print(f"{processor_cls.__name__} saved.")
                            except Exception:
                                continue
  
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
                # return self.pipeline(input_data, **kwargs)
            
            # Hugging Face Direct Execution
            if self.model_provider == "huggingface" and self.model:
                LOG.info(f"Direct HF {task} inference")
                
                from PIL import image
                import requests
                if self.model_category == "multimodal":
                    try:
                        img_path = None
                        text = None

                        for item in input_data:
                            if 'image' in item:
                                img_path = item['image']
                            elif 'text' in item:
                                text = item['text']

                        if not img_path or not text:
                            raise ValueError("Both image and text must be present in the sample_input")
                        
                        raw_image = Image.open(img_path).convert('RGB')
                        
                        inputs = self.processor(raw_image, text, return_tensors="pt")

                        out = self.model.generate(**inputs)
                        decoded_output = self.processor.decode(out[0], skip_special_tokens=True)
                        print(f"[INFO] Decoded Output: {decoded_output}")
                        return decoded_output 

                    except Exception as e:
                        print("Couldn't inference", e)
                    # >>> a photography of a woman and her dog

                    
                    # inference for QWEN
                    # query = self.tokenizer.from_list_format(input_data)
                    # inputs = self.tokenizer(query, return_tensors='pt')
                    # inputs = inputs.to(self.model.device)
                    # pred = self.model.generate(**inputs)
                    # response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
                    # print(response)
                    # image = self.tokenizer.draw_bbox_on_latest_picture(response)
                    # if image:
                    #     image.save('2.jpg')
                    # else:
                    #     print("no box")

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
                if self.task in ["text-generation", "text2text-generation"]:
                   
                    inputs = self.tokenizer(input_data, return_tensors="pt").to(model.device)

                    # Generate output
                    output =self.__module__odel.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=True,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.95
                    )

                    # Decode and print result
                    generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    return generated_text

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
            if self.model_provider == "pytorch" and isinstance(self.model, torch.nn.Module):
                # Audio Processing
                # if task == "audio-classification":
                #     waveform = _convert_to_tensor(input_data).float()
                #     mel_spec = torchaudio.transforms.MelSpectrogram()(waveform)
                #     return self.model(mel_spec.unsqueeze(0))
                if self.model_category=="audio":
                    try:
                        torchaudio.set_audio_backend("soundfile")
                        print("== audio model inferencing beginning ==")
                        # Load the pre-trained ConvTasNet model
                        # bundle = f"{torchaudio.pipelines}.{model_name}"
                        model = torch.load(f"{self.model_save_path}/{self.model_name}.pt", weights_only=False)
                        model.eval()

                        # Load the audio file
                        
                        waveform, sample_rate = torchaudio.load(input_data)

                        # Resample if the audio sample rate doesn't match the model's expected sample rate
                        expected_sampling_rate = 16000
                        if sample_rate != expected_sampling_rate:
                            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=expected_sampling_rate)
                            waveform = resampler(waveform)
                            sample_rate = expected_sampling_rate

                        # Ensure the waveform is mono
                        if waveform.shape[0] > 1:
                            waveform = torch.mean(waveform, dim=0, keepdim=True)

                        waveform = waveform.unsqueeze(0) 

                        # Perform source separation
                        with torch.no_grad():
                            separated_sources = model(waveform)
                        print("Separated sources shape:", separated_sources.shape)
                    except Exception as e:
                        print("Error inferencing torch audio model", e)
                
                # Object Detection
                # if task == "object-detection":
                #     from torchvision import transforms
                #     preprocess = transforms.Compose([
                #         transforms.Resize(800),
                #         transforms.ToTensor(),
                #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                #     ])
                #     img_tensor = preprocess(input_data).unsqueeze(0)
                #     with torch.no_grad():
                #         outputs = self.model(img_tensor)
                #     return [{k: v.cpu() for k, v in t.items()} for t in outputs]
                if self.model_category=="vision":
                    try:
                        model = torch.load(f"{self.model_save_path}/{self.model_name}.pt", weights_only=False)
                        model.eval()

                        preprocess = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])

                        # Load the image and apply the preprocessing pipeline.
                         # Replace with the actual image path
                        input_image = Image.open(input_data).convert("RGB")
                        input_tensor = preprocess(input_image)
                        input_batch = input_tensor.unsqueeze(0)

                        # Move to GPU if available
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        model.to(device)
                        input_batch = input_batch.to(device)

                        # Perform inference.
                        with torch.no_grad():
                            output = model(input_batch)

                        # Apply softmax to convert logits to probabilities.
                        probabilities = torch.nn.functional.softmax(output[0], dim=0)
                        predicted_prob, predicted_class = torch.max(probabilities, dim=0)

                        print(f"Predicted class: {predicted_class.item()}, Probability: {predicted_prob.item():.4f}")

                        print("== Vision model inferencing successful ==")
                        # print(F"{TOTAL_TIME_TAKEN} minutes in total")

                    except Exception as e:
                        print("Error during torch vision model inferencing", e)

            # TensorFlow Specific Processing  
            if self.model_provider == "tensorflow" and isinstance(self.model, tf.keras.Model):
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


