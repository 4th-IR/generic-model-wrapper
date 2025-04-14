# Universal Model Wrapper

This project provides a unified solution to load, convert, and save machine learning models from various sources and frameworks. It supports:

- **PyTorch Models:** Download models from Torch Hub, torchvision, torchaudio models/pipelines, and save them using TorchScript.
- **TensorFlow Models:** Load models from tf.keras.applications, TF-Image-Models (tfimm), Keras Hub, or even fall back to KaggleHub if necessary.
- **Hugging Face Models:** Integration to load and run pipelines for natural language processing tasks.
- **Azure Blob Storage:** Save and load models from Azure Blob Storage so that model artifacts can be centrally managed and shared.

---

## File Structure

- **`torch_route.py`**  
  Contains the `from_torch()` function for:
  - Loading models from Torch Hub, torchvision, or torchaudio.
  - Converting the loaded model to TorchScript.
  - Saving the model as a `.pt` file.

- **`tensorflow_route.py`**  
  Contains the `from_tensorflow()` function for:
  - Downloading TensorFlow/Keras models from tf.keras.applications.
  - Falling back to TF-Image-Models (tfimm) or Keras Hub (with various model types) when the model is not available in tf.keras.applications.
  - Option to download models via KaggleHub when required.

- **Main Application / Model Wrapper** (e.g., in `model_wrapper.py`)  
  Implements the `ModelWrapper` class, which:
  - Manages model loading based on the provider (PyTorch, TensorFlow, or Hugging Face).
  - Checks local and Azure Blob Storage for existing models.
  - Downloads and saves models if not found locally.
  - Integrates with Azure Blob Storage for cloud-based model management using environment variables for credentials.
  - Provides detailed logging and error handling.

---

**Unified Inference Function: inference_model(model_provider, model_name, model_category)**

-TorchVision 

-Torchaudio 

-Keras Hub Models:

  <!-- Image Classification

  Text Classification

  Seq2Seq Language Models

  Causal Language Models

  Text-to-Image Generation

  Image-to-Image Generation -->
  

**How to Use**
Call the function with appropriate parameters:

inference_model(
    model_provider="tensorflow", 
    model_name="efficientnetv2_b0_imagenet1k", 
    model_category="image_classifier"
)

---

***File Requirements***
Ensure the following directories and files exist:

*./animal_pctures/* with test images like dog1.jpg

*./audios/* with audio test files like audio2.wav

*./models_saved/* for model storage

---

## Prerequisites

- **Python 3.10 (depending on your TensorFlow variant requirements)**
- **PyTorch, Torchaudio, & Torchvision:** For handling PyTorch models.
- **TensorFlow & Keras & Kaggle :** For handling TensorFlow models.
- **Transformers:** For Hugging Face models and pipelines.
- **Azure Storage Blob SDK:** For saving and retrieving models to/from Azure.
- **Additional Libraries:** tfimm, kagglehub, keras_hub, python-dotenv, torchaudio, soundfile, pandas, psutil, matplotlib etc.

Make sure to install the required packages by using a package manager like pip from the `requirements.txt`.