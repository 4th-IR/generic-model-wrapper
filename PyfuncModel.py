import mlflow
from mlflow.models import Model
from transformers import pipeline, AutoModel, AutoTokenizer
from adlfs import AzureBlobFileSystem
from typing import Optional, Dict, Any, Union
import os
import torch
import json
import tempfile
import logging
from pathlib import Path

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

class UniversalModelWrapper(mlflow.pyfunc.PythonModel):
    """
    A model wrapper that loads models from providers (HuggingFace, TensorFlow, etc.)
    and saves/loads them from Azure Blob Storage. Supports:
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
        self._fs = None
        
        # Get storage account name from environment if not provided
        if 'account_name' not in self.azure_config:
            account_name = os.getenv('AZURE_STORAGE_ACCOUNT')
            if account_name:
                self.azure_config['account_name'] = account_name
            else:
                logger.warning("Azure storage account name not found in environment variables")

    @property
    def fs(self):
        """Initialize Azure filesystem using connection string"""
        if self._fs is None and self.azure_config:
            if 'connection_string' not in self.azure_config:
                raise ValueError("Azure connection string is required but not provided")
                
            logger.info(f"Initializing Azure filesystem with account: {self.azure_config.get('account_name')}")
            try:
                self._fs = AzureBlobFileSystem(
                    connection_string=self.azure_config['connection_string']
                )
                logger.info("Azure filesystem initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Azure filesystem: {str(e)}")
                raise
        return self._fs

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
        
        elif self.model_provider == "gluoncv":
            import gluoncv
            self.model = gluoncv.model_zoo.get_model(self.model_name, **self.kwargs)
        
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

    def _save_to_azure(self, fs_path: str, local_dir: str):
        """Save model to Azure Blob Storage using BlobServiceClient with chunking"""
        from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
        from azure.core.exceptions import ResourceExistsError
        import time
        
        MAX_CHUNK_SIZE = 4 * 1024 * 1024  # 4MB chunks
        MAX_RETRIES = 3
        
        logger.info(f"Saving model to Azure: {fs_path}")
        
        # Save config
        config = {
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "task": self.task,
            "kwargs": self.kwargs
        }
        
        config_path = os.path.join(local_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f)

        # Save model based on provider
        if self.model_provider == "huggingface":
            if hasattr(self.model, "save_pretrained"):
                self.model.save_pretrained(local_dir)
            if self.preprocessor and hasattr(self.preprocessor, "save_pretrained"):
                self.preprocessor.save_pretrained(local_dir)
        
        elif self.model_provider in ["tensorflow", "pytorch", "gluoncv"]:
            model_path = os.path.join(local_dir, f"model.{self.model_provider}")
            if self.model_provider == "tensorflow":
                self.model.save(model_path)
            else:
                torch.save(self.model.state_dict(), model_path)

        # Upload to Azure using BlobServiceClient with chunking
        logger.info(f"Uploading files from {local_dir}")
        try:
            # Create BlobServiceClient
            blob_service_client = BlobServiceClient.from_connection_string(self.azure_config['connection_string'])
            
            # Get container name and model path from fs_path
            container_name, blob_prefix = fs_path.split('/', 1) if '/' in fs_path else (fs_path, "")
            
            # Ensure container exists
            try:
                container_client = blob_service_client.create_container(container_name)
                logger.info(f"Created container: {container_name}")
            except ResourceExistsError:
                container_client = blob_service_client.get_container_client(container_name)
                logger.info(f"Using existing container: {container_name}")
            
            # Upload files with retry logic
            for root, _, files in os.walk(local_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    rel_path = os.path.relpath(local_path, local_dir)
                    blob_path = f"{blob_prefix}/{rel_path}" if blob_prefix else rel_path
                    
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
                                with open(local_path, 'rb') as f:
                                    container_client.upload_blob(name=blob_path, data=f, overwrite=True)
                                break
                            except Exception as e:
                                if retry < MAX_RETRIES - 1:
                                    wait_time = (retry + 1) * 2  # Exponential backoff
                                    logger.warning(f"Retry {retry+1}/{MAX_RETRIES} for {blob_path}: {str(e)}. Waiting {wait_time}s")
                                    time.sleep(wait_time)
                                else:
                                    logger.error(f"Failed to upload {blob_path} after {MAX_RETRIES} retries: {str(e)}")
                                    raise
            
            logger.info("All files uploaded successfully")
        except Exception as e:
            logger.error(f"Failed to upload files to Azure: {str(e)}")
            raise
    
    def _upload_large_file(self, container_client, local_path, blob_path, chunk_size, max_retries):
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
        """Load model from Azure"""
        logger.info("Loading model context from Azure")
        
        # Get Azure paths and storage options
        azure_uri = context.artifacts.get("azure_uri")
        storage_options = context.artifacts.get("storage_options")
        
        if azure_uri and storage_options:
            # Use MLflow's built-in remote storage support
            logger.info(f"Loading model directly from Azure URI: {azure_uri}")
            try:
                # Try loading directly from Azure
                self.model = mlflow.pyfunc.load_model(azure_uri, storage_options=storage_options)
                logger.info("Model loaded directly from Azure")
                return
            except Exception as e:
                logger.warning(f"Failed to load model directly from Azure: {str(e)}")
                logger.info("Falling back to manual download")
            
        # Fallback to manual download
        fs_path = context.artifacts["azure_path"]
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Downloading files from {fs_path}")
            self.fs.get(fs_path, temp_dir, recursive=True)
            
            # Load config
            config_path = os.path.join(temp_dir, "config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Set model parameters
            self.model_provider = config["model_provider"]
            self.task = config["task"]
            self.kwargs = config["kwargs"]
            
            # Set model path for provider loading
            if self.model_provider == "huggingface":
                self.model_name = temp_dir
            elif self.model_provider in ["tensorflow", "pytorch", "gluoncv"]:
                self.model_name = os.path.join(temp_dir, f"model.{self.model_provider}")
            
            # Load model
            self._load_from_provider()
            logger.info("Model loaded from local files")

    def predict(self, context, model_input: list[Union[str, Dict, Any]]):
        """Unified prediction interface"""
        # Handle batch input
        results = []
        for input_item in model_input:
            if hasattr(self.model, "predict"):
                result = self.model.predict(input_item)
            elif hasattr(self.model, "generate"):
                # Handle text generation models
                if self.preprocessor:
                    inputs = self.preprocessor(input_item, return_tensors="pt")
                    outputs = self.model.generate(**inputs)
                    result = self.preprocessor.decode(outputs[0])
                else:
                    result = self.model.generate(**input_item)
            elif hasattr(self.model, "__call__"):
                # Handle PyTorch/GluonCV models
                if self.preprocessor:
                    input_item = self.preprocessor(input_item)
                result = self.model(input_item)
            else:
                raise NotImplementedError("Model prediction interface not recognized")
            results.append(result)
        return results

    @classmethod
    def save(cls, 
             model_name: str,
             model_provider: str = "huggingface",
             task: Optional[str] = None,
             azure_config: Dict[str, str] = None,
             **kwargs):
        """Save model to Azure Blob Storage"""
        # Get Azure config from environment if not provided
        if azure_config is None:
            connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            account_name = os.getenv('AZURE_STORAGE_ACCOUNT')
            if not connection_string or not account_name:
                raise ValueError("Azure connection string and account name are required")
            azure_config = {
                'connection_string': connection_string,
                'account_name': account_name
            }
            
        # Create wrapper and load model
        wrapper = cls(
            model_provider=model_provider,
            model_name=model_name,
            task=task,
            azure_config=azure_config,
            **kwargs
        )
        wrapper._load_from_provider()
        
        # Generate Azure paths
        model_path = Path(model_name).name
        container_name = "model-files"
        fs_path = f"{container_name}/{model_path}"
        azure_uri = f"wasbs://{container_name}@{azure_config['account_name']}.blob.core.windows.net/{model_path}"
        
        logger.info(f"Using Azure URI: {azure_uri}")
        
        # Save everything to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directories
            model_dir = os.path.join(temp_dir, "model")
            mlflow_dir = os.path.join(temp_dir, "mlflow")
            os.makedirs(model_dir)
            os.makedirs(mlflow_dir)
            
            # Save model files
            wrapper._save_to_azure(fs_path, model_dir)
            
            # Save MLflow model
            mlflow.pyfunc.save_model(
                path=mlflow_dir,
                python_model=wrapper,
                artifacts={
                    "azure_path": fs_path,
                    "azure_uri": azure_uri,
                    "storage_options": {
                        "connection_string": connection_string,
                        "account_name": azure_config['account_name'],
                        "container": container_name
                    }
                },
                pip_requirements=[
                    "torch",
                    "transformers",
                    "tensorflow",
                    "gluoncv",
                    "mlflow",
                    "adlfs",
                    "azure-storage-blob",
                    "accelerate"
                ]
            )
            
            # Upload MLflow model
            logger.info("Uploading MLflow model")
            for root, _, files in os.walk(mlflow_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    rel_path = os.path.relpath(local_path, mlflow_dir)
                    azure_path = f"{fs_path}/{rel_path}"
                    
                    logger.info(f"Uploading MLflow file: {rel_path}")
                    with open(local_path, 'rb') as f:
                        wrapper.fs.pipe(azure_path, f.read())
            
        logger.info("Model saved successfully")
        return wrapper