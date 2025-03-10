import os
import shutil
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import mlflow
import logging
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from PyfuncModel import UniversalModelWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def main():
    # Get Azure configuration from environment
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
    
    if not connection_string or not account_name:
        logger.error("Required Azure configuration not found in environment variables")
        return
        
    azure_config = {
        "connection_string": connection_string,
        "account_name": account_name
    }

    # Model configuration - use tiny model for faster testing
    model_name = "prajjwal1/bert-tiny"  # Tiny model (4MB vs 400MB) for much faster testing
    container_name = "model-files"
    model_path = Path(model_name).name
    full_model_path = f"{container_name}/{model_path}"
    
    # Create BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    
    # Ensure container exists
    try:
        container_client = blob_service_client.get_container_client(container_name)
        container_client.get_container_properties()
        logger.info(f"Container {container_name} exists")
    except ResourceNotFoundError:
        logger.info(f"Creating container: {container_name}")
        container_client = blob_service_client.create_container(container_name)
    
    logger.info("1. Saving model to Azure...")
    
    try:
        # Save the model to Azure
        wrapper = UniversalModelWrapper.save(
            model_name=model_name,
            model_provider="huggingface",
            task="text-classification",  # Text classification task (sentiment analysis)
            azure_config=azure_config
        )
        logger.info(f"Model saved to Azure path: {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model to Azure: {str(e)}")
        return
    
    logger.info("2. Loading model from Azure...")
    
    try:
        logger.info("Attempting to load model directly from Azure...")
        
        # First approach: Try to load directly from Azure blob storage
        try:
            # Set Azure storage options for MLflow
            storage_options = {
                "connection_string": connection_string,
                "account_name": account_name,
                "container": container_name
            }
            
            # Create a MLflow-compatible model URI
            model_uri = f"models:/{full_model_path}"
            logger.info(f"Loading model from: {model_uri}")
            
            loaded_model = mlflow.pyfunc.load_model(
                model_uri=model_uri,
                storage_options=storage_options
            )
            logger.info("Model loaded successfully from Azure")
        except Exception as azure_err:
            logger.warning(f"Failed to load directly from Azure: {str(azure_err)}")
            
            # Second approach: Download model files to temp directory and load from there
            logger.info("Falling back to downloading model files first...")
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # Download MLflow model directory from Azure
                    logger.info(f"Downloading model files to: {temp_dir}")
                    download_count = 0
                    mlflow_model_found = False
                    
                    # First, check if model files exist
                    blobs = list(container_client.list_blobs(name_starts_with=model_path))
                    if not blobs:
                        logger.error(f"No files found in Azure Blob Storage at path: {model_path}")
                        raise FileNotFoundError(f"No model files in {model_path}")
                        
                    logger.info(f"Found {len(blobs)} files to download")
                    
                    # Download each blob
                    for blob in blobs:
                        # Skip directories (if any)
                        if blob.name.endswith('/'):
                            continue
                            
                        # Create relative path and local directories
                        rel_path = blob.name[len(model_path):].lstrip('/')
                        local_path = os.path.join(temp_dir, rel_path)
                        os.makedirs(os.path.dirname(local_path), exist_ok=True)
                        
                        # Check for MLflow model file
                        if "MLmodel" in blob.name:
                            mlflow_model_found = True
                            mlflow_dir = os.path.dirname(local_path)
                            logger.info(f"Found MLflow model file at {blob.name}, will load from {mlflow_dir}")
                        
                        # Download blob content
                        logger.info(f"Downloading {blob.name} to {local_path}")
                        blob_client = container_client.get_blob_client(blob.name)
                        with open(local_path, "wb") as download_file:
                            download_file.write(blob_client.download_blob().readall())
                        download_count += 1
                        
                    logger.info(f"Downloaded {download_count} files to {temp_dir}")
                    
                    # Now load the model from the local directory
                    if mlflow_model_found:
                        logger.info(f"Loading MLflow model from: {mlflow_dir}")
                        loaded_model = mlflow.pyfunc.load_model(mlflow_dir)
                    else:
                        # If no MLflow model file found, try using UniversalModelWrapper directly
                        logger.info(f"No MLflow model file found, trying to load model files directly")
                        wrapper = UniversalModelWrapper(
                            model_provider="huggingface",
                            model_name=os.path.join(temp_dir),
                            task="text-classification",
                            azure_config=azure_config
                        )
                        loaded_model = wrapper
                except Exception as download_err:
                    logger.error(f"Failed to download and load model: {str(download_err)}")
                    raise
        
        logger.info("Model loaded successfully")
        
        # Test the model
        logger.info("3. Testing model prediction...")
        try:
            # Test the model with some example texts
            test_inputs = ["This movie is great!", "I didn't like this film at all"]
            logger.info(f"Input texts: {test_inputs}")
            
            predictions = loaded_model.predict(test_inputs)
            logger.info(f"Predictions: {predictions}")
        except Exception as e:
            logger.error(f"Failed to make prediction: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to load model from Azure: {str(e)}")
        return

if __name__ == "__main__":
    main()