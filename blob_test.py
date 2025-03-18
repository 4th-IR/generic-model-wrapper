import os
import logging
import mlflow
from dotenv import load_dotenv
from BlobModelWrapper import BlobModelWrapper
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_wrapper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_azure_container(connection_string: str, container_name: str = "model-files"):
    """Setup Azure container if it doesn't exist"""
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    
    try:
        container_client = blob_service_client.get_container_client(container_name)
        container_client.get_container_properties()
        logger.info(f"Container {container_name} exists")
    except ResourceNotFoundError:
        logger.info(f"Creating container: {container_name}")
        blob_service_client.create_container(container_name)
    
    return blob_service_client

def test_save_and_load():
    """Test saving and loading a model using BlobModelWrapper"""
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get Azure credentials
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
    
    if not connection_string or not account_name:
        raise ValueError("Azure credentials not found in environment variables")
    
    azure_config = {
        "connection_string": connection_string,
        "account_name": account_name
    }
    
    # Setup Azure container
    setup_azure_container(connection_string)
    
    # Test parameters
    model_name = "prajjwal1/bert-tiny"  # Small model for testing
    container_name = "model-files"
    test_texts = ["This movie was great!", "This product is terrible."]

    try:
        # 1. Save model to Azure using proper staging
        logger.info(f"Saving model {model_name} to Azure...")
        
        # Create unique path using timestamp
        import time
        unique_path = f"{Path(model_name).name}-{int(time.time())}"
        
        # Save through wrapper
        saved_path = BlobModelWrapper.save(
            model_name=model_name,
            model_provider="huggingface",
            task="text-classification",
            azure_config=azure_config,
            storage_path=unique_path  # Add unique path parameter
        )
        logger.info(f"Model saved to Azure path: {saved_path}")

        # 2. Load model from Azure using MLflow
        model_uri = f"wasbs://{container_name}@{account_name}.blob.core.windows.net/{unique_path}"
        logger.info(f"Loading model from: {model_uri}")

        ## save model URI and Model Config to MIS database 
        ## in inference service, fetch model URI and pass to the mlflow.pyfunc.load_model(model_uri)
        
        print(model_uri)
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model loaded successfully from Azure")

        # 3. Make predictions
        logger.info("Making predictions...")
        predictions = loaded_model.predict(test_texts)
        
        for text, pred in zip(test_texts, predictions):
            logger.info(f"Input: {text}")
            logger.info(f"Prediction: {pred[0]}\n")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_save_and_load()