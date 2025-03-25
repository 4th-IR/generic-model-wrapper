import os
import io
import torch
import logging
import tempfile
import requests
from fastapi import HTTPException
from azure.storage.blob import BlobServiceClient
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModel

# Azure configuration from environment variables
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = "model-files"
AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
AZURE_MODEL_FOLDER = "generic-model-wrapper-fastapi"
AZURE_BLOB_URI = f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net/{AZURE_CONTAINER_NAME}"

# Logging setup
LOG_FILE = "model.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

class ModelWrapper:
    def __init__(self, provider, model_name, pipeline_type):
        """
        Initialize the ModelWrapper.
        provider: 'huggingface', 'pytorch', 'tensorflow', or 'github'
        model_name: model identifier or local path (or GitHub URL for 'github')
        pipeline_type: task type (e.g., "text-generation")
        """
        self.provider = provider
        self.model_name = model_name
        self.task = pipeline_type
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        # Azure blob name: use a filename that replaces slashes
        self.azure_blob_name = f"{AZURE_MODEL_FOLDER}/{model_name.replace('/', '_')}.pt"
        
        # Azure storage setup if connection string exists
        if AZURE_CONNECTION_STRING:
            self.blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
            self.container_client = self.blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
            self.azure_config = True
        else:
            self.azure_config = False

    def save_to_azure(self):
        """Save the entire PyTorch model to Azure Blob Storage."""
        if not self.model:
            logging.error("Attempted to save a model that is not loaded.")
            raise ValueError("Model is not loaded.")
        
        # Save the model to a temporary file
        temp_dir = tempfile.gettempdir()
        local_model_path = os.path.join(temp_dir, self.azure_blob_name)
        os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
        torch.save(self.model, local_model_path)  # Save the full model

        blob_client = self.container_client.get_blob_client(self.azure_blob_name)
        with open(local_model_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        logging.info(f"Model saved to Azure Blob Storage: {self.azure_blob_name}")

    def load_from_azure(self):
        """Stream the entire model directly from Azure Blob Storage into memory."""
        if not self.azure_config:
            logging.warning("Azure configuration is disabled.")
            return False

        blob_client = self.container_client.get_blob_client(self.azure_blob_name)
        if not blob_client.exists():
            logging.warning(f"Model blob does not exist: {self.azure_blob_name}")
            return False

        # Use an in-memory buffer to stream the model
        stream = io.BytesIO()
        blob_client.download_blob().readinto(stream)
        stream.seek(0)
        self.model = torch.load(stream, map_location="cpu")
        logging.info(f"Model streamed from Azure: {self.azure_blob_name}")

        # If the provider is huggingface, also load the tokenizer and create a pipeline
        if self.provider == "huggingface":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.pipeline = pipeline(self.task, model=self.model, tokenizer=self.tokenizer)
        return True

    def download_from_github(self, github_url):
        """
        Download the model file from GitHub and return the local path.
        github_url: direct URL to the model file on GitHub.
        """
        temp_dir = tempfile.gettempdir()
        local_model_path = os.path.join(temp_dir, self.azure_blob_name)
        os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
        
        response = requests.get(github_url, stream=True)
        if response.status_code == 200:
            with open(local_model_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            logging.info(f"Model downloaded from GitHub and saved at: {local_model_path}")
            return local_model_path
        else:
            raise Exception(f"Failed to download model from GitHub. Status Code: {response.status_code}")

    def load_model(self, github_url=None):
        """
        Loads the model.
        1. Try to load from Azure.
        2. If not available and provider is 'github', download from GitHub and save to Azure.
        3. Otherwise, load from the provider's repository.
        """
        try:
            if self.azure_config:
                logging.info("Checking Azure Storage for model...")
                if self.load_from_azure():
                    logging.info(f"Loaded model from Azure Storage: {self.model_name}")
                    return

            # If not loaded from Azure, load from original source
            logging.info(f"Downloading model from {self.provider}...")
            if self.provider == "huggingface":
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.pipeline = pipeline(self.task, model=self.model, tokenizer=self.tokenizer)
            elif self.provider == "pytorch":
                # Assume model_name is a local path
                self.model = torch.load(self.model_name, map_location="cpu")
            elif self.provider == "tensorflow":
                from tensorflow import keras
                self.model = keras.models.load_model(self.model_name)
            elif self.provider == "github":
                if not github_url:
                    raise ValueError("GitHub URL must be provided for provider 'github'")
                local_model_path = self.download_from_github(github_url)
                self.model = torch.load(local_model_path, map_location="cpu")
            else:
                logging.error(f"Unknown model provider: {self.provider}")
                raise ValueError(f"Unsupported model provider: {self.provider}")

            logging.info("Model loaded successfully from source.")
            # Save the full model to Azure for future use
            if self.azure_config:
                logging.info("Saving model to Azure...")
                self.save_to_azure()
        except Exception as e:
            logging.error(f"Error loading model: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

    def run_inference(self, text):
        """
        Runs inference on the given text input.
        For Hugging Face pipelines, this calls the pipeline.
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline is not initialized. Load the model first.")
        logging.info(f"Running inference on input: {text}")
        return self.pipeline(text)

# Example usage:
if __name__ == "__main__":
    import time
    # Example: using a Hugging Face model with GitHub fallback (if desired)
    # For GitHub provider, pass the direct URL to the model file
    # You could switch provider to "huggingface" to directly download from Hugging Face.
    #github_model_url = "https://raw.githubusercontent.com/user/repo/main/model.pt"
    
    # For demonstration, we use the "huggingface" provider
    model_wrapper = ModelWrapper("huggingface", 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', "text-generation")
    load_time = time.time()
    model_wrapper.load_model()  # For GitHub provider, you would pass github_url=github_model_url
    end_load_time = time.time()
    print(f"Model loading took: {end_load_time - load_time} seconds")

    start = time.time()
    result = model_wrapper.run_inference("The movie was awesome")
    end = time.time()
    print(result)
    print(f"Inference took: {end - start} seconds")