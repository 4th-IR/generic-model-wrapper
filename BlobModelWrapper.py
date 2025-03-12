import mlflow
from mlflow.models import Model
from transformers import pipeline, AutoModel, AutoTokenizer
import transformers
from typing import Optional, Dict, Any, Union
import os
import torch
import tensorflow
import json
import tempfile
import logging
from pathlib import Path
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from time import time
from typing import Dict, Optional, List, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_wrapper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BlobModelWrapper(mlflow.pyfunc.PythonModel):
    """
    A model wrapper that loads models from providers (HuggingFace, TensorFlow, etc.)
    and saves/loads them from Azure Blob Storage using only BlobServiceClient. Supports:
    - HuggingFace models and pipelines
    - TensorFlow models
    - PyTorch models
    - Custom preprocessors
    - Automatic component detection
    """
    
    def __init__(self, 
                 model_provider: str,
                 model_name: str,
                 task: Optional[str] = None,
                 azure_config: Dict[str, str] = None,
                 **kwargs):
        """
        Initialize the model wrapper
        Args:
            model_provider: Source of the model ('huggingface', 'tensorflow', 'pytorch', etc.)
            model_name: Name/path of the model from the provider
            task: Task type for pipeline-based models
            azure_config: Azure storage configuration with connection_string
            **kwargs: Additional arguments for model loading
        """
        self.model_provider = model_provider
        self.model_name = model_name
        self.task = task
        self.azure_config = azure_config or {}
        self.kwargs = kwargs
        self.model = None
        self.preprocessor = None
        self._blob_service_client = None
        
        # Get storage account name from environment if not provided
        if 'account_name' not in self.azure_config:
            account_name = os.getenv('AZURE_STORAGE_ACCOUNT')
            if account_name:
                self.azure_config['account_name'] = account_name
            else:
                logger.warning("Azure storage account name not found in environment variables")

    @property
    def blob_service_client(self) -> BlobServiceClient:
        """Initialize Azure BlobServiceClient using connection string"""
        if self._blob_service_client is None and self.azure_config:
            if 'connection_string' not in self.azure_config:
                raise ValueError("Azure connection string is required but not provided")
                
            logger.info(f"Initializing Azure BlobServiceClient with account: {self.azure_config.get('account_name')}")
            try:
                self._blob_service_client = BlobServiceClient.from_connection_string(
                    self.azure_config['connection_string']
                )
                logger.info("Azure BlobServiceClient initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Azure BlobServiceClient: {str(e)}")
                raise
        return self._blob_service_client

    def _load_from_provider(self):
        """Load model from provider"""
        logger.info(f"Loading model '{self.model_name}' from provider '{self.model_provider}'")
        if self.model_provider == "huggingface":
            if self.task:
                self.model = pipeline(
                    task=self.task,
                    model=self.model_name,
                    **self.kwargs
                )
                # Extract preprocessor if available
                self.preprocessor = getattr(self.model, "tokenizer", None) or \
                                  getattr(self.model, "feature_extractor", None) or \
                                  getattr(self.model, "image_processor", None)
            else:
                self.model = AutoModel.from_pretrained(self.model_name, **self.kwargs)
                self.preprocessor = AutoTokenizer.from_pretrained(self.model_name, **self.kwargs)
        
        elif self.model_provider == "tensorflow":
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_name)
        
        elif self.model_provider == "pytorch":
            import torch.hub
            self.model = torch.hub.load(*self.model_name.split(':'), **self.kwargs)
        
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

    def _download_blob_to_dir(self, container_client: ContainerClient, prefix: str, local_dir: str):
        """Download all blobs with given prefix to local directory"""
        logger.info(f"Downloading blobs with prefix '{prefix}' to {local_dir}")
        
        # List all blobs with the prefix
        blob_list = container_client.list_blobs(name_starts_with=prefix)
        
        # Download each blob
        for blob in blob_list:
            # Get relative path from prefix
            rel_path = blob.name[len(prefix):].lstrip('/')
            local_path = os.path.join(local_dir, rel_path)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download blob
            blob_client = container_client.get_blob_client(blob.name)
            with open(local_path, "wb") as file:
                data = blob_client.download_blob()
                file.write(data.readall())
                
        logger.info(f"Downloaded {sum(1 for _ in container_client.list_blobs(name_starts_with=prefix))} files")

    def _upload_directory(self, container_client: ContainerClient, local_dir: str, prefix: str = ""):
        """Upload directory contents to blob storage with retry logic"""
        MAX_CHUNK_SIZE = 4 * 1024 * 1024  # 4MB chunks
        MAX_RETRIES = 3
        
        logger.info(f"Uploading directory {local_dir} to container with prefix '{prefix}'")
        
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, local_dir)
                blob_path = f"{prefix}/{rel_path}" if prefix else rel_path
                
                # Get file size
                file_size = os.path.getsize(local_path)
                
                # Use chunked upload for large files
                if file_size > MAX_CHUNK_SIZE:
                    logger.info(f"Uploading large file {rel_path} ({file_size} bytes) to {blob_path} in chunks")
                    self._upload_large_file(container_client, local_path, blob_path, MAX_CHUNK_SIZE, MAX_RETRIES)
                else:
                    # Small file upload with retries
                    for retry in range(MAX_RETRIES):
                        try:
                            logger.info(f"Uploading {rel_path} ({file_size} bytes) to {blob_path}")
                            blob_client = container_client.get_blob_client(blob_path)
                            with open(local_path, "rb") as data:
                                blob_client.upload_blob(data, overwrite=True)
                            break
                        except Exception as e:
                            if retry < MAX_RETRIES - 1:
                                wait_time = (retry + 1) * 2  # Exponential backoff
                                logger.warning(f"Retry {retry+1}/{MAX_RETRIES} for {blob_path}: {str(e)}. Waiting {wait_time}s")
                                time.sleep(wait_time)
                            else:
                                logger.error(f"Failed to upload {blob_path} after {MAX_RETRIES} retries: {str(e)}")
                                raise

    def _upload_large_file(self, container_client: ContainerClient, local_path: str, blob_path: str, chunk_size: int, max_retries: int):
        """Upload a large file to Azure Blob Storage in chunks"""
        from azure.storage.blob import BlobBlock
        import uuid
        import time
        
        blob_client = container_client.get_blob_client(blob_path)
        file_size = os.path.getsize(local_path)
        
        # Create block list
        block_list = []
        
        with open(local_path, 'rb') as file_handle:
            # Read file in chunks
            block_number = 0
            while True:
                read_data = file_handle.read(chunk_size)
                if not read_data:
                    break  # End of file
                
                # Create unique block id
                block_id = str(uuid.uuid4())
                encoded_block_id = block_id.encode('utf-8')
                block_list.append(encoded_block_id)
                
                # Upload block with retry logic
                for retry in range(max_retries):
                    try:
                        blob_client.stage_block(block_id=encoded_block_id, data=read_data)
                        # Log progress for large uploads
                        block_number += 1
                        progress = min(100, (block_number * chunk_size / file_size) * 100)
                        logger.info(f"Uploaded block {block_number}, {progress:.1f}% complete")
                        break
                    except Exception as e:
                        if retry < max_retries - 1:
                            wait_time = (retry + 1) * 2  # Exponential backoff
                            logger.warning(f"Block upload retry {retry+1}/{max_retries}: {str(e)}. Waiting {wait_time}s")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Failed to upload block after {max_retries} retries: {str(e)}")
                            raise
        
        # Commit the blocks
        for retry in range(max_retries):
            try:
                blob_client.commit_block_list(block_list)
                logger.info(f"Committed all blocks for {blob_path}")
                break
            except Exception as e:
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 2  # Exponential backoff
                    logger.warning(f"Block commit retry {retry+1}/{max_retries}: {str(e)}. Waiting {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to commit blocks after {max_retries} retries: {str(e)}")
                    raise

    def load_context(self, context):
        """Load model from local artifacts or Azure Blob Storage."""
        logger.info("Loading model context")
        model_files = str(context.artifacts["model_files"])
        
        # Load config
        config_path = os.path.join(model_files, "blob_wrapper_config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Set parameters from config
        self.model_provider = config["model_provider"]
        self.model_name = config["model_name"]
        self.task = config["task"]
        self.kwargs = config["kwargs"]
        
        # Load model based on provider
        if self.model_provider == "huggingface":
            if self.task:
                self.model = pipeline(
                    task=self.task,
                    model=model_files,  # Use local path
                    **self.kwargs
                )
            else:
                self.model = AutoModel.from_pretrained(model_files)
                self.preprocessor = AutoTokenizer.from_pretrained(model_files)
        elif self.model_provider in ["tensorflow", "pytorch"]:
            self.model_name = os.path.join(model_files, f"model.{self.model_provider}")
            if self.model_provider == "tensorflow":
                import tensorflow as tf
                self.model = tf.keras.models.load_model(self.model_name)
            else:
                self.model = torch.load(self.model_name)

        logger.info("Model loaded successfully")

    def predict(self, context, model_input: List[Union[str, Dict, Any]]):
        """Unified prediction interface for all model types."""

        self.load_context(context)

        print('loaded mode: ', self.model)
        logger.info(f"Predicting with {self.model_provider} model")
        results = []
        for input_item in model_input:
            try:
                # Framework-specific logic happens here
                if hasattr(self.model, "predict"):
                    # TensorFlow and some HuggingFace models
                    result = self.model.predict(input_item)
                elif hasattr(self.model, "generate"):
                    # HuggingFace generation models
                    if self.preprocessor:
                        inputs = self.preprocessor(input_item, return_tensors="pt")
                        outputs = self.model.generate(inputs)
                        result = self.preprocessor.decode(outputs[0])
                    else:
                        result = self.model.generate(input_item)
                elif hasattr(self.model, "call"):
                    # PyTorch models
                    if self.preprocessor:
                        input_item = self.preprocessor(input_item)
                    result = self.model(input_item)
                else:
                    raise NotImplementedError(f"Model prediction interface not recognized for {self.model_provider} model")
                results.append(result)
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                # Add model info to help with debugging
                logger.error(f"Model provider: {self.model_provider}, Model type: {type(self.model)}")
                raise
        return results

    @classmethod
    def save(
        cls,
        model_name: str,
        model_provider: str = "huggingface",
        task: Optional[str] = None,
        azure_config: Dict[str, str] = None,
        storage_path: Optional[str] = None,
        **kwargs
    ):
        """Save model to local path or Azure Blob Storage."""
        # Create wrapper and load model
        wrapper = cls(
            model_provider=model_provider,
            model_name=model_name,
            task=task,
            azure_config=azure_config,
            **kwargs
        )
        wrapper._load_from_provider()

        # Generate paths
        model_path = Path(model_name).name
        container_name = "model-files"

        logger.info(f"Saving model {model_name} to temporary directory...")
        # Save everything to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directories
            mlflow_dir = os.path.join(temp_dir, "mlflow_artifacts")
            model_dir = os.path.join(temp_dir, "model_files")
            os.makedirs(mlflow_dir, exist_ok=True)
            os.makedirs(model_dir, exist_ok=True)

            # Save config
            config = {
                "model_provider": model_provider,
                "model_name": model_name,
                "task": task,
                "kwargs": kwargs
            }
            config_path = os.path.join(model_dir, "blob_wrapper_config.json")
            with open(config_path, 'w') as f:
                json.dump({
                    "model_provider": model_provider,
                    "model_name": model_name,
                    "task": task,
                    "kwargs": kwargs
                }, f)

            # Save model based on provider
            if model_provider == "huggingface":
                if hasattr(wrapper.model, "save_pretrained"):
                    wrapper.model.save_pretrained(model_dir)
                if wrapper.preprocessor and hasattr(wrapper.preprocessor, "save_pretrained"):
                    wrapper.preprocessor.save_pretrained(model_dir)
            elif model_provider in ["tensorflow", "pytorch"]:
                model_path = os.path.join(model_dir, f"model.{model_provider}")
                if model_provider == "tensorflow":
                    wrapper.model.save(model_path)
                else:
                    torch.save(wrapper.model.state_dict(), model_path)

            logger.info("Saving MLflow model with local artifacts...")

            print('model_dir: ', model_dir)
            print(f"Saving MLflow model with local artifacts to {mlflow_dir}")
            print('This is the wrapper to save: ', wrapper)
            # Save MLflow model with local artifacts
            mlflow.pyfunc.save_model(
                path=mlflow_dir,
                python_model=wrapper,
                artifacts={
                    "model_files": str(model_dir),
                    "azure_config": config_path
                },
                pip_requirements=[
                    f"torch=={torch.__version__}",
                    f"transformers=={transformers.__version__}",
                    f"tensorflow=={tensorflow.__version__}",
                    f"mlflow=={mlflow.__version__}",
                    "azure-storage-blob>=12.0.0",
                    "accelerate>=0.20.0",
                    "python-dotenv>=1.0.0"
                ]
            )

             # Save Azure config separately to avoid MLflow artifact handling issues
            azure_config_path = os.path.join(model_dir, "azure_config.json")
            with open(azure_config_path, 'w') as f:
                json.dump({
                    "container": container_name,
                    "account_name": azure_config['account_name'],
                    "connection_string": azure_config['connection_string']
                }, f)

            # If Azure config is provided, upload to Azure
            if azure_config:
                logger.info("Uploading files to Azure...")
                # Get container client
                container_client = wrapper.blob_service_client.get_container_client(container_name)
                # Ensure container exists
                try:
                    container_client = wrapper.blob_service_client.create_container(container_name)
                    logger.info(f"Created container: {container_name}")
                except ResourceExistsError:
                    container_client = wrapper.blob_service_client.get_container_client(container_name)
                    logger.info(f"Using existing container: {container_name}")

                # Upload both model and MLflow files
                wrapper._upload_directory(container_client, model_dir, f"{storage_path}/model-files")
                wrapper._upload_directory(container_client, mlflow_dir, f"{storage_path}")
                logger.info("Model saved to Azure successfully")
            else:
                logger.info("Model saved locally successfully")

        return wrapper