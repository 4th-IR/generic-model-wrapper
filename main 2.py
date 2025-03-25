import os
import torch
import logging
import tempfile
from fastapi import HTTPException
from azure.storage.blob import BlobServiceClient
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Azure Configuration
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = "model-files"
AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
AZURE_MODEL_FOLDER = "generic-model-wrapper-fastapi"
AZURE_BLOB_URI = f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net/{AZURE_CONTAINER_NAME}"

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
        self.model_provider = provider
        self.model_name = model_name
        self.task = pipeline_type
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.azure_blob_name = f"{AZURE_MODEL_FOLDER}/{model_name.replace('/', '_')}.pt"

        # Azure storage setup
        if AZURE_CONNECTION_STRING:
            self.blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
            self.container_client = self.blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
            self.azure_config = True  # Enable Azure model loading
        else:
            self.azure_config = False

    def save_to_azure(self):
        """Saves the model to Azure Blob Storage."""
        if not self.model:
            logging.error("Attempted to save a model that is not loaded.")
            raise ValueError("Model is not loaded.")

        temp_dir = tempfile.gettempdir()
        local_model_path = os.path.join(temp_dir, self.azure_blob_name)
        torch.save(self.model.state_dict(), local_model_path)

        blob_client = self.container_client.get_blob_client(self.azure_blob_name)
        with open(local_model_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        logging.info(f"Model saved to Azure Blob Storage: {self.azure_blob_name}")

    def load_from_azure(self):
        """Loads the model from Azure Blob Storage."""
        if not self.azure_config:
            logging.warning("Azure configuration is disabled.")
            return False

        temp_dir = tempfile.gettempdir()
        local_model_path = os.path.join(temp_dir, self.azure_blob_name)
        blob_client = self.container_client.get_blob_client(self.azure_blob_name)

        if not blob_client.exists():
            logging.warning(f"Model blob does not exist: {self.azure_blob_name}")
            return False

        with open(local_model_path, "wb") as file:
            download_stream = blob_client.download_blob()
            file.write(download_stream.readall())

        logging.info(f"Model downloaded from Azure: {self.azure_blob_name}")

        # Load model state dict
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)                   ## fetching the model by name 
        self.model.load_state_dict(torch.load(local_model_path, map_location="cpu"))
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.pipeline = pipeline(self.task, model=self.model, tokenizer=self.tokenizer)

        logging.info("Model successfully loaded from Azure and initialized for inference.")
        return True

    def load_model(self):
        """Loads models from Azure Storage if available; otherwise, downloads from a repository."""
        try:
            if self.azure_config:
                logging.info("Checking Azure Storage for model...")
                if self.load_from_azure():
                    logging.info(f"Loaded model from Azure Storage: {self.model_name}")
                    return

            logging.info(f"Downloading model from {self.model_provider}...")

            if self.model_provider == "huggingface":
                logging.info(f"Loading Hugging Face model: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                self.pipeline = pipeline(self.task, model=self.model, tokenizer=self.tokenizer)
            elif self.model_provider == "pytorch":
                logging.info(f"Loading PyTorch model from {self.model_name}")
                self.model = torch.load(self.model_name, map_location="cpu")
            elif self.model_provider == "tensorflow":
                logging.info(f"Loading TensorFlow model from {self.model_name}")
                from tensorflow import keras
                self.model = keras.models.load_model(self.model_name)
            else:
                logging.error(f"Unknown model provider: {self.model_provider}")
                raise ValueError(f"Unsupported model provider: {self.model_provider}")

            logging.info("Model loaded successfully.")

            if self.azure_config:
                logging.info("Saving model to Azure...")
                self.save_to_azure()
        except Exception as e:
            logging.error(f"Error loading model: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

    def run_inference(self, text):
        """Runs inference on the model."""
        if not self.pipeline:
            raise RuntimeError("Pipeline is not initialized. Load the model first.")

        logging.info(f"Running inference on input: {text}")
        return self.pipeline(text)

# Test
if __name__ == "__main__":
    import time
    model_wrapper = ModelWrapper("huggingface", 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', "text-generation")

    start = time.time()
    model_wrapper.load_model()
    end = time.time()
    print(f"Model loading took {end - start:.2f} seconds")

    # Run inference
    start = time.time()
    result = model_wrapper.run_inference(["This is a sample text."])
    end = time.time()
    print(f"Inference took {end - start:.2f} seconds")
    print(result)
    
