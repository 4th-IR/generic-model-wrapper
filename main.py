import os
import torch
import logging
from fastapi import HTTPException
from azure.storage.blob import BlobServiceClient
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from torch_route import from_torch
from tensorflow_route import from_tensorflow
from dotenv import load_dotenv
import tensorflow as tf
import time
from huggingface_hub import login

HGF_TOKEN = os.getenv("HUGGINGFACE ")
login(HGF_TOKEN)


load_dotenv()


AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")
AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
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
        self.saved_path = "models_saved"

        os.makedirs(self.saved_path, exist_ok=True)

        # Azure storage setup        
        if not AZURE_CONNECTION_STRING:
            raise ValueError("Azure connection string is missing! Check environment variables.")
        if not AZURE_CONTAINER_NAME:
            raise ValueError("Azure container name is missing! Check environment variables.")
        
        self.blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        self.container_client = self.blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
        self.azure_config = True  # Enable Azure model loading

    def save_to_azure(self):
        print("<---SAVING TO AZURE--->")
    
        if not self.model:
            logging.error("Attempted to save a model that is not loaded.")
            raise ValueError("Model is not loaded.")
        
        for root, _, files in os.walk(self.saved_path):
            if self.model_name + ".pt" in files:
                print("Model found Locally")
                local_model_path = os.path.join(root, f"{self.model_name}.pt")
                blob_client = self.container_client.get_blob_client(blob=self.model_name)
                with open(local_model_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)

        logging.info(f"{self.model_name} saved to Azure Blob Storage")


    def load_from_azure(self):
        print("<---LOADING MODEL--->")
        if not self.azure_config:
            logging.warning("Azure configuration is disabled.")
           
            return False

        path = "./" + self.saved_path
        local_model_path = os.path.join(path, f"{self.model_name}.pt")
        blob_client = self.container_client.get_blob_client(blob=self.model_name)

        if not blob_client.exists():
            logging.warning("Model blob does not exist")
            print("Model blob does not exist")
            return False

        with open(local_model_path, "wb") as file:
            download_stream = blob_client.download_blob()
            file.write(download_stream.readall())

        logging.info(f"{self.model_name} downloaded from Azure")

        # Load model 
        if self.model_provider == "huggingface":
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.pipeline = pipeline(self.task, model=self.model, tokenizer=self.tokenizer)

        elif self.model_provider == "pytorch":
            try:
                # Try loading as a TorchScript model
                self.model = torch.jit.load(local_model_path, map_location="cpu")
                self.model.eval()
            except RuntimeError as e:
                print(f"torch.jit.load() failed: {e}. Trying torch.load() instead...")

                # Load as a full PyTorch model
                self.model = torch.load(local_model_path, weights_only=False)
                self.model.eval()

        elif self.model_provider == "tensorflow":
            try:
                # Try loading as a TorchScript model
                self.model = tf.keras.models.load_model(local_model_path)
                print("Model successfully loaded")
                return self.model
            except Exception as e:
                print("Failed to load tensorflow model", e)
                
        logging.info("Model successfully loaded from Azure and initialized for inference.")
        return True

    def load_model(self):

        """Loads models from Azure Storage if available; otherwise, downloads from a repository."""
        # since we are using a local directory, we will check it before 

        try:
            for root, _, files in os.walk(self.saved_path):
                if self.model_name in files:
                    print("Model found Locally")

            else: 
                if self.azure_config:
                    logging.info("Checking Azure Storage for model...")
                    print("Checking Azure Storage for model...")
                    if self.load_from_azure():
                        logging.info(f"Loaded model from Azure Storage: {self.model_name}")
                        print(f"Loaded model from Azure Storage: {self.model_name}")
                        return

                logging.info(f"Downloading model from {self.model_provider}...")

                if self.model_provider == "huggingface":
                    logging.info(f"Loading Hugging Face model: {self.model_name}")
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                    self.pipeline = pipeline(self.task, model=self.model, tokenizer=self.tokenizer)

                elif self.model_provider == "pytorch":
                    logging.info(f"Loading PyTorch model from {self.model_name}")
                    print(f"Loading PyTorch model from {self.model_name}")
                    self.model = from_torch(self.model_name, self.saved_path) 

                elif self.model_provider == "tensorflow":
                    logging.info(f"Loading TensorFlow model from {self.model_name}")
                    print(f"Loading TensorFlow model from {self.model_name}")
                    self.model = from_tensorflow(self.model_name, self.saved_path)

                else:
                    logging.error(f"Unknown model provider: {self.model_provider}")
                    raise ValueError(f"Unsupported model provider: {self.model_provider}")

            logging.info("Model loaded successfully.")

            # if self.azure_config:
            #     logging.info("Saving model to Azure...")
            #     self.save_to_azure()
        except Exception as e:
            logging.error(f"Error loading model: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")


if __name__ == "__main__":
   
    """
    I wil be testing these models tomorrow and updating the sheet. 
    """



    from inferencing import inference_model
    import pandas as pd
    import openpyxl
    import psutil
    
    models_dict = {
    "model_1": {"model_provider": "pytorch", "model_category": "torch_audio", "model_name": "WAV2VEC2_BASE"},
    "model_2": {"model_provider": "pytorch", "model_category": "torch_audio", "model_name": "EMFORMER_RNNT_BASE_LIBRISPEECH"},
    "model_3": {"model_provider": "pytorch", "model_category": "torch_vision", "model_name": "ResNet50"},
    "model_4": {"model_provider": "pytorch", "model_category": "torch_audio", "model_name": "Wav2Vec2Bundle"},
    "model_5": {"model_provider": "pytorch", "model_category": "torch_vision", "model_name": "inception_v3"},
    "model_6": {"model_provider": "pytorch", "model_category": "torch_audio", "model_name": "EMFORMER_RNNT_BASE_LIBRISPEECH"},
    "model_7": {"model_provider": "pytorch", "model_category": "torch_vision", "model_name": "shufflenet_v2_x1_5"},
    "model_8": {"model_provider": "pytorch", "model_category": "torch_audio", "model_name": "TACOTRON2_WAVERNN_CHAR_LJSPEECH"},
        }

    excel_file = 'torch_model_metrics.xlsx'

    model_inference_metrics = []

    for model in models_dict.values():

        model_name = model["model_name"]
        model_provider = model["model_provider"]
        model_category = model["model_category"]


        start_inference_time = time.time()
        process = psutil.Process(os.getpid())

        mem_before = process.memory_info().rss / (1024 ** 2)

        model_wrapper = ModelWrapper(model_provider, model_name, model_category)
        model_wrapper.load_model()

        print("/nDownload completed")

        inference_model(model_provider, model_name, model_category)

        end_inference_time = time.time()

        mem_after = process.memory_info().rss / (1024 ** 2)

        mem_used = mem_after - mem_before


        TOTAL_TIME_TAKEN = round((end_inference_time - start_inference_time)/60, 2)
        model_data = {'model_name': model_name,
                        'model_provider': model_provider,
                        "model_category": model_category,
                        "total_time_taken(mins)": TOTAL_TIME_TAKEN,
                        "memory_used": mem_used}
                                
        model_inference_metrics.append(model_data)
        print("/n Metrics obtained", model_inference_metrics)

    df = pd.DataFrame(model_inference_metrics)

    if os.path.exists(excel_file):
        existing_df = pd.read_excel(excel_file)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        updated_df = df

    updated_df.to_excel(excel_file, index=False)

    print(f"Saved {len(df)} model entries to {excel_file}")


