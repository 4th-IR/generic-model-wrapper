import os
import numpy as np
import torch
import tensorflow as tf
from PIL import Image
import torchaudio
from io import BytesIO
import base64
import requests

from utils.logger import get_logger

LOG = get_logger('preprocessing')

# --- Image Loading & Preprocessing --- 

def load_image(image_input: Any) -> Image.Image:
    """Loads an image from various input types (path, URL, base64, bytes, PIL Image)."""
    if isinstance(image_input, Image.Image):
        LOG.debug("Input is already a PIL Image.")
        return image_input.convert("RGB")
    elif isinstance(image_input, str):
        if image_input.startswith(('http://', 'https://')):
            try:
                LOG.debug(f"Loading image from URL: {image_input}")
                response = requests.get(image_input, stream=True)
                response.raise_for_status()
                image = Image.open(response.raw).convert("RGB")
                LOG.debug("Image loaded successfully from URL.")
                return image
            except requests.exceptions.RequestException as e:
                LOG.error(f"Failed to load image from URL {image_input}: {e}")
                raise ValueError(f"Could not fetch image from URL: {e}") from e
        elif os.path.isfile(image_input):
            try:
                LOG.debug(f"Loading image from path: {image_input}")
                image = Image.open(image_input).convert("RGB")
                LOG.debug("Image loaded successfully from path.")
                return image
            except FileNotFoundError:
                LOG.error(f"Image file not found at path: {image_input}")
                raise ValueError(f"Image file not found: {image_input}")
            except Exception as e:
                LOG.error(f"Failed to load image from path {image_input}: {e}")
                raise ValueError(f"Could not load image from path: {e}") from e
        else:
            # Try decoding as base64 string
            try:
                LOG.debug("Attempting to load image from base64 string.")
                # Remove potential data URI prefix (e.g., "data:image/jpeg;base64,")
                if ',' in image_input:
                    image_input = image_input.split(',', 1)[1]
                img_bytes = base64.b64decode(image_input)
                image = Image.open(BytesIO(img_bytes)).convert("RGB")
                LOG.debug("Image loaded successfully from base64 string.")
                return image
            except (base64.binascii.Error, ValueError, Exception) as e:
                LOG.error(f"Input string is not a valid path, URL, or base64 image: {e}")
                raise ValueError("Invalid image input string. Must be a file path, URL, or base64 encoded string.")
    elif isinstance(image_input, bytes):
        try:
            LOG.debug("Loading image from bytes.")
            image = Image.open(BytesIO(image_input)).convert("RGB")
            LOG.debug("Image loaded successfully from bytes.")
            return image
        except Exception as e:
            LOG.error(f"Failed to load image from bytes: {e}")
            raise ValueError(f"Could not load image from bytes: {e}") from e
    else:
        LOG.error(f"Unsupported image input type: {type(image_input)}")
        raise TypeError(f"Unsupported image input type: {type(image_input)}. Expected path, URL, base64 string, bytes, or PIL Image.")

# --- Audio Loading & Preprocessing --- 

def load_audio(audio_input: Any, target_sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    """Loads audio from various input types (path, URL, base64, bytes, numpy array, torch tensor) 
       and resamples to the target sample rate.
       Returns a tuple of (waveform_numpy, sample_rate).
    """
    waveform = None
    original_sr = None

    if isinstance(audio_input, (np.ndarray, torch.Tensor)):
        LOG.debug("Input is already a numpy array or torch tensor.")
        # Assume sample rate is provided or default (needs improvement if SR varies)
        # This part might need adjustment based on how tensors/arrays are passed
        waveform = audio_input.numpy() if isinstance(audio_input, torch.Tensor) else audio_input
        original_sr = target_sample_rate # Assuming input tensor/array is already at target SR
        LOG.warning("Assuming input tensor/array has target sample rate. Provide sample rate if different.")

    elif isinstance(audio_input, str):
        if audio_input.startswith(('http://', 'https://')):
            try:
                LOG.debug(f"Loading audio from URL: {audio_input}")
                response = requests.get(audio_input)
                response.raise_for_status()
                # Use BytesIO to load from memory
                waveform, original_sr = torchaudio.load(BytesIO(response.content))
                LOG.debug(f"Audio loaded successfully from URL. Original SR: {original_sr}")
            except requests.exceptions.RequestException as e:
                LOG.error(f"Failed to load audio from URL {audio_input}: {e}")
                raise ValueError(f"Could not fetch audio from URL: {e}") from e
            except Exception as e:
                 LOG.error(f"Failed to decode audio from URL {audio_input}: {e}")
                 raise ValueError(f"Could not decode audio from URL: {e}") from e
        elif os.path.isfile(audio_input):
            try:
                LOG.debug(f"Loading audio from path: {audio_input}")
                waveform, original_sr = torchaudio.load(audio_input)
                LOG.debug(f"Audio loaded successfully from path. Original SR: {original_sr}")
            except FileNotFoundError:
                LOG.error(f"Audio file not found at path: {audio_input}")
                raise ValueError(f"Audio file not found: {audio_input}")
            except Exception as e:
                LOG.error(f"Failed to load audio from path {audio_input}: {e}")
                raise ValueError(f"Could not load audio from path: {e}") from e
        else:
            # Try decoding as base64 string
            try:
                LOG.debug("Attempting to load audio from base64 string.")
                if ',' in audio_input:
                    audio_input = audio_input.split(',', 1)[1]
                audio_bytes = base64.b64decode(audio_input)
                waveform, original_sr = torchaudio.load(BytesIO(audio_bytes))
                LOG.debug(f"Audio loaded successfully from base64 string. Original SR: {original_sr}")
            except (base64.binascii.Error, ValueError, Exception) as e:
                LOG.error(f"Input string is not a valid path, URL, or base64 audio: {e}")
                raise ValueError("Invalid audio input string. Must be a file path, URL, or base64 encoded string.")

    elif isinstance(audio_input, bytes):
        try:
            LOG.debug("Loading audio from bytes.")
            waveform, original_sr = torchaudio.load(BytesIO(audio_input))
            LOG.debug(f"Audio loaded successfully from bytes. Original SR: {original_sr}")
        except Exception as e:
            LOG.error(f"Failed to load audio from bytes: {e}")
            raise ValueError(f"Could not load audio from bytes: {e}") from e
    else:
        LOG.error(f"Unsupported audio input type: {type(audio_input)}")
        raise TypeError(f"Unsupported audio input type: {type(audio_input)}. Expected path, URL, base64, bytes, numpy array, or torch tensor.")

    # Ensure waveform is a numpy array (mono)
    if waveform.ndim > 1 and waveform.shape[0] > 1: # Check if stereo
        waveform = torch.mean(waveform, dim=0, keepdim=True) # Convert to mono
        LOG.debug("Converted stereo audio to mono.")
    waveform_np = waveform.squeeze().numpy()

    # Resample if necessary
    if original_sr != target_sample_rate:
        LOG.debug(f"Resampling audio from {original_sr} Hz to {target_sample_rate} Hz.")
        resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sample_rate)
        waveform_resampled = resampler(waveform) # Resample original tensor
        waveform_np = waveform_resampled.squeeze().numpy()
        LOG.debug("Resampling complete.")
        
    return waveform_np, target_sample_rate

# --- Tensor Conversion --- 

def convert_to_tensor(data: Any, provider: str) -> Any:
    """Converts numpy arrays to the appropriate tensor type (PyTorch or TensorFlow)."""
    if isinstance(data, np.ndarray):
        if provider == "pytorch":
            return torch.from_numpy(data)
        elif provider == "tensorflow":
            return tf.convert_to_tensor(data)
        else:
            LOG.warning(f"Unknown provider '{provider}' for tensor conversion. Returning numpy array.")
            return data
    # Add handling for lists of numpy arrays if needed
    elif isinstance(data, list) and all(isinstance(item, np.ndarray) for item in data):
         if provider == "pytorch":
            return [torch.from_numpy(item) for item in data]
         elif provider == "tensorflow":
            return [tf.convert_to_tensor(item) for item in data]
         else:
            LOG.warning(f"Unknown provider '{provider}' for list tensor conversion. Returning list of numpy arrays.")
            return data
    return data # Return data unchanged if not a numpy array or list of arrays

# --- Multimodal Input Handling --- 

def prepare_multimodal_input(input_data: Any, task: str, provider: str, processor: Any = None, tokenizer: Any = None, feature_extractor: Any = None, image_processor: Any = None, target_sr: int = 16000) -> Any:
    """Prepares potentially multimodal input data based on the task and provider.
    
    Args:
        input_data: The raw input data. Can be text, image data, audio data, or a dict 
                    mapping modality names (e.g., 'text', 'image', 'audio') to their data.
        task: The inference task (e.g., 'visual-question-answering', 'image-to-text').
        provider: The model provider ('huggingface', 'pytorch', 'tensorflow').
        processor: A multimodal processor (e.g., from Hugging Face).
        tokenizer: A tokenizer (primarily for text).
        feature_extractor: An audio feature extractor.
        image_processor: An image processor.
        target_sr: Target sample rate for audio.

    Returns:
        The processed input suitable for the model.
    """
    LOG.info(f"Preparing input for task '{task}' with provider '{provider}'.")

    # --- Hugging Face Specific Handling --- 
    if provider == "huggingface":
        # Use multimodal processor if available (preferred)
        if processor:
            LOG.debug("Using Hugging Face multimodal processor.")
            try:
                # Handle dict input for multimodal processors
                if isinstance(input_data, dict):
                    processed_inputs = {}
                    if 'text' in input_data:
                        processed_inputs['text'] = input_data['text'] # Assume processor handles text tokenization
                    if 'image' in input_data:
                        image = load_image(input_data['image'])
                        processed_inputs['images'] = image # Processor expects 'images' key
                    if 'audio' in input_data:
                         # Processors might expect raw waveform + sampling rate or pre-extracted features
                         waveform, sr = load_audio(input_data['audio'], target_sample_rate=target_sr)
                         # Check processor documentation for exact input format
                         processed_inputs['audio'] = waveform 
                         processed_inputs['sampling_rate'] = sr
                    
                    # Let the processor handle the combination
                    # Adjust return_tensors based on expected framework ('pt' or 'tf')
                    # This needs refinement based on actual processor usage
                    inputs = processor(**processed_inputs, return_tensors="pt") # Default to PT for now
                    LOG.debug("Processor successfully processed multimodal dict input.")
                    return inputs
                else:
                    # If not a dict, assume processor can handle the single input type
                    # This might need adjustment based on specific processor
                    LOG.warning("Input data is not a dict. Assuming processor can handle single modality input.")
                    if task in ["automatic-speech-recognition", "audio-classification"]:
                        waveform, sr = load_audio(input_data, target_sample_rate=target_sr)
                        inputs = processor(waveform, sampling_rate=sr, return_tensors="pt")
                    elif task in ["image-classification", "object-detection", "image-segmentation", "image-to-text"]:
                        image = load_image(input_data)
                        inputs = processor(images=image, return_tensors="pt")
                    elif task in ["text-generation", "text-classification", "token-classification", "question-answering"]:
                         inputs = processor(text=input_data, return_tensors="pt")
                    else: # Generic fallback
                         inputs = processor(input_data, return_tensors="pt") 
                    LOG.debug("Processor successfully processed single modality input.")
                    return inputs

            except Exception as e:
                LOG.error(f"Error using Hugging Face processor: {e}", exc_info=True)
                raise ValueError(f"Failed to process input with HF processor: {e}") from e
        
        # Fallback to individual components if no multimodal processor
        LOG.debug("No multimodal processor provided. Using individual components (tokenizer/image_processor/feature_extractor).")
        processed_inputs = {}
        return_tensors = "pt" # Default, adjust if TF needed

        if isinstance(input_data, dict):
            if 'text' in input_data and tokenizer:
                LOG.debug("Tokenizing text input.")
                # Tokenizer might return multiple items, handle potential collision
                tokenized = tokenizer(input_data['text'], return_tensors=return_tensors, padding=True, truncation=True)
                processed_inputs.update(tokenized)
            if 'image' in input_data and image_processor:
                LOG.debug("Processing image input.")
                image = load_image(input_data['image'])
                img_processed = image_processor(images=image, return_tensors=return_tensors)
                # Careful about key names (e.g., pixel_values)
                processed_inputs.update(img_processed)
            if 'audio' in input_data and feature_extractor:
                LOG.debug("Extracting features from audio input.")
                waveform, sr = load_audio(input_data['audio'], target_sample_rate=target_sr)
                audio_features = feature_extractor(waveform, sampling_rate=sr, return_tensors=return_tensors)
                # Careful about key names (e.g., input_features)
                processed_inputs.update(audio_features)
            
            if not processed_inputs:
                 LOG.error("Multimodal input dict provided, but no matching processors/tokenizers found or input keys are incorrect.")
                 raise ValueError("Could not process multimodal input dict with available components.")
            LOG.debug("Successfully processed multimodal dict with individual components.")
            return processed_inputs
        
        # Handle single modality input with individual components
        elif task in ["automatic-speech-recognition", "audio-classification"] and feature_extractor:
             LOG.debug("Extracting features from single audio input.")
             waveform, sr = load_audio(input_data, target_sample_rate=target_sr)
             return feature_extractor(waveform, sampling_rate=sr, return_tensors=return_tensors)
        elif task in ["image-classification", "object-detection", "image-segmentation", "image-to-text"] and image_processor:
             LOG.debug("Processing single image input.")
             image = load_image(input_data)
             return image_processor(images=image, return_tensors=return_tensors)
        elif task in ["text-generation", "text-classification", "token-classification", "question-answering"] and tokenizer:
             LOG.debug("Tokenizing single text input.")
             return tokenizer(input_data, return_tensors=return_tensors, padding=True, truncation=True)
        else:
             LOG.error(f"Cannot process input for task '{task}'. No suitable processor/tokenizer/extractor found for input type {type(input_data)}.")
             raise ValueError(f"Unsupported input type or missing processor/tokenizer for task '{task}'.")

    # --- Generic PyTorch/TensorFlow Handling (Less specific, needs model-specific logic) --- 
    elif provider in ["pytorch", "tensorflow"]:
        LOG.warning(f"Generic preprocessing for {provider}. Model-specific preprocessing might be required.")
        if isinstance(input_data, dict):
             # Basic loading for common modalities
             processed_dict = {}
             if 'text' in input_data:
                 processed_dict['text'] = input_data['text'] # Keep as string, model needs to handle tokenization
             if 'image' in input_data:
                 processed_dict['image'] = load_image(input_data['image']) # Load as PIL
                 # Convert to tensor later if needed by the specific model
             if 'audio' in input_data:
                 waveform_np, sr = load_audio(input_data['audio'], target_sample_rate=target_sr)
                 processed_dict['audio'] = waveform_np # Keep as numpy
                 processed_dict['sampling_rate'] = sr
             LOG.debug(f"Loaded multimodal dict for {provider}: { {k: type(v) for k, v in processed_dict.items()} }")
             return processed_dict # Return dict of loaded data
        else:
            # Handle single modality
            if task in ["automatic-speech-recognition", "audio-classification"]:
                waveform_np, sr = load_audio(input_data, target_sample_rate=target_sr)
                LOG.debug(f"Loaded single audio input for {provider}.")
                return waveform_np # Return numpy array
            elif task in ["image-classification", "object-detection", "image-segmentation", "image-to-text"]:
                image = load_image(input_data)
                LOG.debug(f"Loaded single image input for {provider}.")
                return image # Return PIL image
            elif isinstance(input_data, str): # Assume text otherwise
                 LOG.debug(f"Passing through single text input for {provider}.")
                 return input_data # Pass string directly
            else:
                 LOG.warning(f"Passing through unknown single input type {type(input_data)} for {provider}.")
                 return input_data # Pass through unknown types

    else:
        LOG.error(f"Unsupported provider '{provider}' for preprocessing.")
        raise ValueError(f"Unsupported provider: {provider}")