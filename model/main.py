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
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoFeatureExtractor, AutoImageProcessor
import torchaudio



#internal 
from utils.logger import get_logger
from utils.env_manager import AZURE_BLOB_URI, AZURE_CONNECTION_STRING, AZURE_CONTAINER_NAME, AZURE_STORAGE_ACCOUNT
from utils.resource_manager import timer
from utils.torch_route import from_torch
from utils.tensorflow_route import from_tensorflow
from utils.preprocessing import prepare_multimodal_input, convert_to_tensor

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
        self.processor = None # For multimodal processors
        self.image_processor = None # Specific image processor
        self.feature_extractor = None # Specific audio feature extractor
        self.pipeline = None
        self.saved_path = "temp_models"
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
        """Saves the model directory/files to Azure Blob Storage under a folder named after the model."""
        print("<---SAVING TO AZURE--->")
        LOG.info(f"Attempting to save model '{self.model_name}' to Azure container '{AZURE_CONTAINER_NAME}'.")

        if not self.model_save_path or not os.path.exists(self.model_save_path):
            error_msg = f"Model save path '{self.model_save_path}' is not valid or model hasn't been saved locally first."
            LOG.error(error_msg)
            raise ValueError(error_msg)

        try:
            LOG.info(f'Local model path to upload: {self.model_save_path}')
            # Use the model_name as the root folder (prefix) in the blob container
            self.upload_directory_to_azure(
                local_path=self.model_save_path,
                blob_prefix=self.model_name
            )
            LOG.info(f"Successfully saved '{self.model_name}' to Azure Blob Storage in container '{AZURE_CONTAINER_NAME}' under folder '{self.model_name}'.")
        except Exception as e:
            LOG.error(f'Failed to save model {self.model_name} to Azure: {e}', exc_info=True)
            raise RuntimeError(f'Model saving to Azure failed: {e}')

    #@timer
    def load_from_azure(self):
        """Loads the model from Azure Blob Storage, expecting a directory structure named after the model."""
        print("<---LOADING MODEL FROM AZURE--->")
        LOG.info(f"Attempting to load model '{self.model_name}' from Azure container '{AZURE_CONTAINER_NAME}'.")
        if not self.azure_config:
            LOG.warning("Azure configuration is disabled. Cannot load from Azure.")
            return False

        # Standard approach: Look for a directory named after the model.
        # The prefix should be the model name followed by a slash to list contents *within* that directory.
        blob_prefix = self.model_name + "/"
        blob_list = list(self.container_client.list_blobs(name_starts_with=blob_prefix))

        if blob_list:
            # Found blobs under the model_name prefix, indicating a directory structure.
            LOG.info(f"Found directory structure for '{self.model_name}' in Azure container '{AZURE_CONTAINER_NAME}'. Loading directory.")
            return self._load_directory_from_azure(self.model_name) # Pass model_name as the prefix
        else:
            # If no directory structure, log a warning and indicate failure.
            # We no longer support loading single files directly named after the model as the standard.
            LOG.warning(f"No directory structure found with prefix '{blob_prefix}' in Azure container '{AZURE_CONTAINER_NAME}'. Model not found or not saved in the standard format.")
            # Optionally, you could check for the single file as a legacy fallback here, but it's discouraged.
            # blob_client = self.container_client.get_blob_client(self.model_name)
            # if self._check_blob_exists(blob_client):
            #     LOG.warning(f"Found single blob '{self.model_name}'. Attempting legacy load. Standard is directory structure.")
            #     return self._load_single_file_from_azure(self.model_name)
            return False
            
    def _check_blob_exists(self, blob_client):
        try:
            blob_client.get_blob_properties()
            return True
        except Exception as e:
            return False

    def _load_single_file_from_azure(self, blob_name):
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_file_path = os.path.join(tmp_dir, os.path.basename(blob_name))
            blob_client = self.container_client.get_blob_client(blob_name)
            try:
                with open(local_file_path, "wb") as file:
                    download_stream = blob_client.download_blob()
                    file.write(download_stream.readall())
                LOG.info(f"Single file model downloaded from Azure to: {local_file_path}")
                return self._load_model_from_local_path(local_file_path)
            except Exception as e:
                LOG.error(f"Error downloading single file model from Azure: {e}")
                return False

    def _load_directory_from_azure(self, blob_prefix):
        with tempfile.TemporaryDirectory() as tmp_dir:
            blobs = self.container_client.list_blobs(name_starts_with=blob_prefix + "/")
            for blob in blobs:
                relative_path = os.path.relpath(blob.name, blob_prefix)
                local_path = os.path.join(tmp_dir, relative_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                blob_client = self.container_client.get_blob_client(blob.name)
                try:
                    with open(local_path, "wb") as file:
                        download_stream = blob_client.download_blob()
                        file.write(download_stream.readall())
                except Exception as e:
                    LOG.error(f"Error downloading blob {blob.name}: {e}")
                    return False
            LOG.info(f"Model directory downloaded from Azure to: {tmp_dir}")
            return self._load_model_from_local_path(tmp_dir)

    def _load_model_from_local_path(self, local_path):
        """Loads the model from a local directory path based on the provider."""
        LOG.info(f"Loading model for provider '{self.model_provider}' from local path: {local_path}")
        try:
            provider = self.model_provider.lower()
            if provider == "huggingface":
                # Hugging Face models: Load model, tokenizer, and potentially processor/feature_extractor/image_processor
                self.model = AutoModelForCausalLM.from_pretrained(local_path)
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(local_path)
                    LOG.info("HF AutoTokenizer loaded.")
                except Exception:
                    LOG.warning(f"Could not load AutoTokenizer from {local_path}. Tokenizer might be part of the processor.")
                    self.tokenizer = None
                try:
                    # Attempt to load a general processor first (common for multimodal)
                    self.processor = AutoProcessor.from_pretrained(local_path)
                    LOG.info("HF AutoProcessor loaded.")
                    # If processor exists, it might contain tokenizer/feature_extractor/image_processor
                    if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer:
                        self.tokenizer = self.processor.tokenizer
                    if hasattr(self.processor, 'feature_extractor') and self.processor.feature_extractor:
                        self.feature_extractor = self.processor.feature_extractor
                    if hasattr(self.processor, 'image_processor') and self.processor.image_processor:
                        self.image_processor = self.processor.image_processor
                except Exception:
                    LOG.info(f"Could not load AutoProcessor from {local_path}. Attempting individual components.")
                    self.processor = None
                    # Load individual components if processor loading failed or they are separate
                    if not self.image_processor:
                        try:
                            self.image_processor = AutoImageProcessor.from_pretrained(local_path)
                            LOG.info("HF AutoImageProcessor loaded.")
                        except Exception:
                            LOG.info(f"Could not load AutoImageProcessor from {local_path}.")
                    if not self.feature_extractor:
                         try:
                            self.feature_extractor = AutoFeatureExtractor.from_pretrained(local_path)
                            LOG.info("HF AutoFeatureExtractor loaded.")
                         except Exception:
                            LOG.info(f"Could not load AutoFeatureExtractor from {local_path}.")

                # Optionally create pipeline if task is suitable and components are available
                try:
                    # Pipeline creation might need specific components depending on the task
                    pipeline_args = {'model': self.model}
                    if self.tokenizer: pipeline_args['tokenizer'] = self.tokenizer
                    if self.image_processor: pipeline_args['image_processor'] = self.image_processor
                    if self.feature_extractor: pipeline_args['feature_extractor'] = self.feature_extractor
                    # Avoid creating pipeline for base causal LM unless task explicitly needs it
                    if self.task and self.task not in ["text-generation", "feature-extraction"]: 
                        self.pipeline = pipeline(self.task, **pipeline_args)
                        LOG.info(f"Hugging Face pipeline for task '{self.task}' created.")
                    else:
                        LOG.info(f"HF components loaded for task '{self.task}'. Pipeline not created by default or task unsuitable.")
                        self.pipeline = None
                except Exception as pipe_error:
                    LOG.warning(f"Could not automatically create HF pipeline for task '{self.task}': {pipe_error}. Components loaded directly.")
                    self.pipeline = None
            elif provider == "pytorch":
                # PyTorch models: Search for the model file within the downloaded directory.
                found_model_file = None
                possible_extensions = (".pt", ".pth", ".bin") # Common extensions
                for filename in os.listdir(local_path):
                    if filename.endswith(possible_extensions):
                        # Prioritize standard names if multiple files exist
                        if filename in ["pytorch_model.bin", "model.pt", "model.pth"]:
                             found_model_file = os.path.join(local_path, filename)
                             break # Found a standard name
                        elif not found_model_file: # Keep the first one found if no standard name
                             found_model_file = os.path.join(local_path, filename)

                if found_model_file:
                    LOG.info(f"Found PyTorch model file: {found_model_file}")
                    # Use torch.load for flexibility (can load state_dict or full model)
                    # Map location ensures model loads correctly regardless of saving device
                    self.model = torch.load(found_model_file, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                    # If it's a state dict, you might need to instantiate the model class first
                    # This example assumes torch.load loads the whole model object or a JIT model
                    if isinstance(self.model, dict) and 'state_dict' in self.model:
                         LOG.warning("Loaded state_dict, model class instantiation might be required.")
                         # Add logic here to load state_dict into your model class if needed
                    elif isinstance(self.model, torch.nn.Module):
                        self.model.eval() # Set to evaluation mode
                    LOG.info("PyTorch model loaded successfully.")
                else:
                    # If no single file, maybe it's a torch.jit.load compatible directory?
                    try:
                        self.model = torch.jit.load(local_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                        if isinstance(self.model, torch.nn.Module):
                             self.model.eval()
                        LOG.info("PyTorch model loaded successfully using torch.jit.load on the directory.")
                    except Exception as jit_e:
                        LOG.error(f"Could not find a suitable PyTorch model file ({possible_extensions}) in {local_path}, and torch.jit.load failed: {jit_e}")
                        return False
            elif provider == "tensorflow":
                # TensorFlow/Keras models are typically saved as a directory (SavedModel format)
                self.model = tf.keras.models.load_model(local_path)
                LOG.info("TensorFlow/Keras model loaded successfully.")
            else:
                LOG.error(f"Unsupported model provider '{self.model_provider}' for loading from local path.")
                return False

            LOG.info(f"Model '{self.model_name}' ({self.model_provider}) loaded successfully from local path: {local_path}")
            self.model_save_path = local_path # Update save path to the loaded path
            return True
        except Exception as e:
            LOG.error(f"Failed to load model from local path {local_path}: {e}", exc_info=True)
            return False

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
                    LOG.info(f"Model not found in Azure. Downloading from {self.model_provider}...")
                    if self.model_provider == "huggingface":
                        LOG.info(f"Loading Hugging Face model/components: {self.model_name}")
                        # Load all potential components
                        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                        try:
                            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                        except Exception: self.tokenizer = None
                        try:
                            self.processor = AutoProcessor.from_pretrained(self.model_name)
                            # If processor loaded, extract sub-components
                            if hasattr(self.processor, 'tokenizer'): self.tokenizer = self.processor.tokenizer
                            if hasattr(self.processor, 'feature_extractor'): self.feature_extractor = self.processor.feature_extractor
                            if hasattr(self.processor, 'image_processor'): self.image_processor = self.processor.image_processor
                        except Exception: self.processor = None
                        # Load individual components if processor failed or they are separate
                        if not self.image_processor:
                            try: self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
                            except Exception: pass
                        if not self.feature_extractor:
                            try: self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
                            except Exception: pass
                        
                        # Attempt pipeline creation (similar logic to _load_model_from_local_path)
                        try:
                            pipeline_args = {'model': self.model}
                            if self.tokenizer: pipeline_args['tokenizer'] = self.tokenizer
                            if self.image_processor: pipeline_args['image_processor'] = self.image_processor
                            if self.feature_extractor: pipeline_args['feature_extractor'] = self.feature_extractor
                            if self.task and self.task not in ["text-generation", "feature-extraction"]:
                                self.pipeline = pipeline(self.task, **pipeline_args)
                                LOG.info(f"HF pipeline for task '{self.task}' created after download.")
                            else: self.pipeline = None
                        except Exception as pipe_error:
                            LOG.warning(f"Could not create HF pipeline after download: {pipe_error}")
                            self.pipeline = None

                        # Save all loaded components
                        temp_dir = tempfile.gettempdir()
                        save_path = os.path.join(temp_dir, self.model_name.replace('/', '_')) # Sanitize name for path
                        os.makedirs(save_path, exist_ok=True)
                        self.model.save_pretrained(save_path)
                        if self.tokenizer: self.tokenizer.save_pretrained(save_path)
                        if self.processor: self.processor.save_pretrained(save_path)
                        # Save individual components only if they weren't part of a processor
                        if not self.processor and self.image_processor: self.image_processor.save_pretrained(save_path)
                        if not self.processor and self.feature_extractor: self.feature_extractor.save_pretrained(save_path)
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

                    LOG.info("Model loaded successfully from provider.")
                    if self.azure_config:
                        LOG.info("Saving model from Provider to Azure...")
                        self.save_to_azure()

        except Exception as e:
            LOG.error(f"Error loading model: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

        

    @timer
    def run_inference(self, input_data: Any, task: str = None, **kwargs: Any):
        """Runs inference using the loaded model, handling multimodal inputs via preprocessing."""
        effective_task = task or self.task
        LOG.info(f"Running inference for task: {effective_task} with provider: {self.model_provider}")
        
        if not self.model:
             error_msg = "Model is not loaded. Cannot run inference."
             LOG.error(error_msg)
             raise RuntimeError(error_msg)

        try:
            # --- Input Preparation --- 
            # Use the new preprocessing utility
            # Pass necessary components (processor, tokenizer, etc.) from self
            prepared_input = prepare_multimodal_input(
                input_data=input_data,
                task=effective_task,
                provider=self.model_provider.lower(),
                processor=getattr(self, 'processor', None),
                tokenizer=getattr(self, 'tokenizer', None),
                feature_extractor=getattr(self, 'feature_extractor', None),
                image_processor=getattr(self, 'image_processor', None),
                # Add target_sr if needed, e.g., from model config
                # target_sr=self.config.get('target_sample_rate', 16000) 
            )
            LOG.debug(f"Prepared input type: {type(prepared_input)}")

            # --- Inference Execution --- 
            
            # Hugging Face Pipeline (if available and suitable)
            # Pipelines often handle preprocessing internally, but we prepare input first for consistency
            # Check if pipeline exists and task matches
            if self.pipeline and self.pipeline.task == effective_task:
                LOG.info(f"Using existing HF pipeline for task '{effective_task}'.")
                # Pipelines might expect raw data or preprocessed data depending on implementation
                # Passing prepared_input might work for some, raw input_data for others.
                # Let's try passing the originally provided input_data first, as pipelines often handle loading.
                try:
                    # Special handling for audio pipelines that might expect paths/bytes
                    if effective_task in ["automatic-speech-recognition", "audio-classification"] and isinstance(input_data, (str, bytes)):
                         return self.pipeline(input_data, **kwargs)
                    # For other pipelines, try with the original input data
                    return self.pipeline(input_data, **kwargs)
                except Exception as pipe_exec_err:
                     LOG.warning(f"HF pipeline execution failed with raw input: {pipe_exec_err}. Trying with prepared input.")
                     # Fallback: Try passing the dictionary from prepare_multimodal_input if it returned one
                     if isinstance(prepared_input, dict):
                          # Need to ensure keys match what the pipeline expects internally
                          return self.pipeline(**prepared_input, **kwargs)
                     else:
                          # If prepared_input isn't a dict, pass it directly
                          return self.pipeline(prepared_input, **kwargs)

            # Direct Model Execution (if no suitable pipeline)
            LOG.info(f"No suitable pipeline found or used. Running direct model inference for task '{effective_task}'.")
            
            # Ensure inputs are tensors for PyTorch/TensorFlow if needed
            # Note: prepared_input might already be tensors if HF processor was used
            if self.model_provider.lower() in ["pytorch", "tensorflow"]:
                 # Convert numpy arrays within the prepared input (if it's a dict or single array)
                 if isinstance(prepared_input, dict):
                      tensor_input = {k: convert_to_tensor(v, self.model_provider.lower()) for k, v in prepared_input.items()}
                 else:
                      tensor_input = convert_to_tensor(prepared_input, self.model_provider.lower())
            else:
                 # For HF, prepared_input should already be in the correct format (usually dict of tensors)
                 tensor_input = prepared_input

            # --- Framework-Specific Inference Logic --- 
            if self.model_provider.lower() == "huggingface":
                # Input should be a dictionary ready for model(**input)
                if not isinstance(tensor_input, dict):
                     error_msg = f"Hugging Face direct inference expects a dictionary of inputs, but got {type(tensor_input)}."
                     LOG.error(error_msg)
                     raise TypeError(error_msg)
                
                # Move tensors to the correct device (assuming self.device is set)
                device = getattr(self, 'device', 'cpu') # Default to CPU if not set
                tensor_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tensor_input.items()}
                
                LOG.debug(f"Running HF model.forward with input keys: {list(tensor_input.keys())}")
                with torch.no_grad() if device == 'cpu' else tf.device(device): # Context manager might vary
                     outputs = self.model(**tensor_input)
                LOG.debug("HF model.forward completed.")
                
                # --- Post-processing (Example for HF, needs task-specific logic) --- 
                if effective_task in ["text-generation", "text2text-generation"] and self.tokenizer:
                     # Model might return logits, need generate for sequence
                     # Re-run with generate if needed, or process logits
                     LOG.warning("Direct HF text-generation might require model.generate(). Processing raw logits.")
                     # This part needs refinement based on the specific model's output structure
                     if hasattr(outputs, 'logits'):
                          # Simple argmax for classification-like tasks from logits
                          predicted_ids = torch.argmax(outputs.logits, dim=-1)
                          return self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
                     else:
                          return outputs # Return raw output
                elif effective_task == "image-classification":
                     return torch.nn.functional.softmax(outputs.logits, dim=-1)
                elif effective_task == "audio-classification":
                     return torch.nn.functional.softmax(outputs.logits, dim=-1)
                elif effective_task == "automatic-speech-recognition":
                     # Process logits to text (example using argmax, real decoding is more complex)
                     predicted_ids = torch.argmax(outputs.logits, dim=-1)
                     return self.tokenizer.batch_decode(predicted_ids)[0]
                elif effective_task in ["object-detection", "image-segmentation"] and self.image_processor:
                     # Use processor's post-processing if available
                     if effective_task == "object-detection" and hasattr(self.image_processor, 'post_process_object_detection'):
                          return self.image_processor.post_process_object_detection(outputs, threshold=kwargs.get("threshold", 0.9))
                     elif effective_task == "image-segmentation" and hasattr(self.image_processor, 'post_process_semantic_segmentation'):
                          return self.image_processor.post_process_semantic_segmentation(outputs)
                     else:
                          LOG.warning(f"No specific HF post-processing found for {effective_task}. Returning raw outputs.")
                          return outputs
                else:
                     # Default: return raw outputs (often logits or a model-specific object)
                     LOG.debug(f"Returning raw model outputs for task {effective_task}.")
                     return outputs

            elif self.model_provider.lower() == "pytorch":
                # PyTorch model expects specific tensor inputs
                # This part requires knowledge of the specific PyTorch model's forward method
                LOG.warning("Direct PyTorch inference requires model-specific input handling.")
                # Example: Assume model takes a single tensor input
                if isinstance(tensor_input, dict):
                     # Try to find the primary input tensor if dict was prepared
                     input_key = next((k for k in ['pixel_values', 'input_values', 'input_ids'] if k in tensor_input), None)
                     if input_key:
                          model_input = tensor_input[input_key]
                     else:
                          raise ValueError("Could not determine primary input tensor for PyTorch model from dict.")
                elif isinstance(tensor_input, torch.Tensor):
                     model_input = tensor_input
                else:
                     raise TypeError(f"Unsupported input type for direct PyTorch inference: {type(tensor_input)}")
                
                # Add batch dimension if missing
                if model_input.ndim == 3 and effective_task in ["image-classification", "object-detection"]:
                     model_input = model_input.unsqueeze(0)
                elif model_input.ndim == 1 and effective_task in ["audio-classification"]:
                     model_input = model_input.unsqueeze(0)
                
                device = getattr(self, 'device', 'cpu')
                model_input = model_input.to(device)
                self.model.to(device)
                self.model.eval()
                
                LOG.debug(f"Running PyTorch model.forward with input shape: {model_input.shape}")
                with torch.no_grad():
                    outputs = self.model(model_input)
                LOG.debug("PyTorch model.forward completed.")
                # Add task-specific post-processing for PyTorch models here
                return outputs 

            elif self.model_provider.lower() == "tensorflow":
                # TensorFlow model expects specific tensor inputs
                LOG.warning("Direct TensorFlow inference requires model-specific input handling.")
                # Example: Assume model takes a single tensor input
                if isinstance(tensor_input, dict):
                     input_key = next((k for k in ['pixel_values', 'input_values', 'input_ids'] if k in tensor_input), None)
                     if input_key:
                          model_input = tensor_input[input_key]
                     else:
                          raise ValueError("Could not determine primary input tensor for TF model from dict.")
                elif tf.is_tensor(tensor_input):
                     model_input = tensor_input
                else:
                     raise TypeError(f"Unsupported input type for direct TF inference: {type(tensor_input)}")
                
                # Add batch dimension if missing (TF often expects batch)
                if len(model_input.shape) == 3 and effective_task in ["image-classification", "object-detection"]:
                     model_input = tf.expand_dims(model_input, axis=0)
                elif len(model_input.shape) == 1 and effective_task in ["audio-classification"]:
                     model_input = tf.expand_dims(model_input, axis=0)

                LOG.debug(f"Running TF model.predict/call with input shape: {model_input.shape}")
                outputs = self.model(model_input) # Or self.model.predict(model_input)
                LOG.debug("TF model.predict/call completed.")
                # Add task-specific post-processing for TF models here
                return outputs

            else:
                LOG.error(f"Inference logic not implemented for provider: {self.model_provider}")
                raise NotImplementedError(f"Inference not implemented for provider {self.model_provider}")

        except Exception as e:
            LOG.error(f"Inference failed for task '{effective_task}': {e}", exc_info=True)
            # Consider raising a more specific exception or returning an error status
            raise RuntimeError(f"Inference error during task '{effective_task}': {str(e)}") from e
    
    def upload_directory_to_azure(self, local_path, blob_prefix=""):
        LOG.info(f'Attempting to upload path: {local_path} to Azure with prefix: {blob_prefix}')
        try:
            blob_client = self.container_client.get_blob_client(os.path.join(blob_prefix, os.path.basename(local_path)))

            if os.path.isfile(local_path):
                LOG.info(f'Uploading single file: {local_path} as {blob_client.blob_name}')
                with open(local_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                LOG.info(f'Successfully uploaded file: {local_path} to {blob_client.blob_name}')

            elif os.path.isdir(local_path):
                LOG.info(f'Uploading directory: {local_path} as prefix: {blob_prefix}')
                for root, _, files in os.walk(local_path):
                    for file in files:
                        local_file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(local_file_path, local_path)
                        blob_name = os.path.join(blob_prefix, relative_path)
                        blob_client = self.container_client.get_blob_client(blob_name)
                        LOG.info(f'Uploading file: {local_file_path} as {blob_client.blob_name}')
                        with open(local_file_path, "rb") as data:
                            blob_client.upload_blob(data, overwrite=True)
                        LOG.info(f'Successfully uploaded file: {local_file_path} to {blob_client.blob_name}')
                LOG.info(f'Successfully uploaded directory: {local_path} to prefix: {blob_prefix}')
            else:
                LOG.warning(f'Provided local path: {local_path} is neither a file nor a directory. Skipping upload.')

        except Exception as e:
            LOG.error(f'Saving to Azure failed with error: {e}')
            raise RuntimeError(f'Model saving to Azure failed: {e}')

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


