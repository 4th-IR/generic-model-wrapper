import os
import logging
import mlflow
from dotenv import load_dotenv
from BlobModelWrapper import BlobModelWrapper
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError

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
    model_name = "prajjwal1/bert-tiny"  # Small 4MB model for testing
    test_texts = [
        "This movie is fantastic! I really enjoyed it.",
        "This was a terrible waste of time.",
        "The product works exactly as described.",
        "I'm not sure how I feel about this.",
    ]
    
    try:
        # Configure MLflow to use Azure storage
        
        #1. Save model to Azure
        # logger.info(f"Saving model {model_name} to Azure...")
        # wrapper = BlobModelWrapper.save(
        #     model_name=model_name,
        #     model_provider="huggingface",
        #     task="text-classification",
        #     azure_config=azure_config
        # )
        # logger.info("Model saved successfully")
        
        # # 2. Test immediate predictions
        # logger.info("Testing predictions with saved model...")
        # predictions = wrapper.predict(None, test_texts)
        # logger.info("Predictions from saved model:")
        # for text, pred in zip(test_texts, predictions):
        #     logger.info(f"Text: {text}")
        #     logger.info(f"Prediction: {pred}\n")
        
        
        
        # # 3. Create new instance and load model
        # logger.info("Testing model loading...")
        new_wrapper = BlobModelWrapper(
            model_provider="huggingface",
            model_name=model_name,
            task="text-classification",
            azure_config=azure_config
        )

        #print('wrapper: ', new_wrapper.load_context())
        print('wrapper.model: ', new_wrapper.model)
        
        # Load the model (this will happen automatically in predict)
        # predictions = new_wrapper.predict(None, test_texts)
        # logger.info("Predictions from loaded model:")
        # for text, pred in zip(test_texts, predictions):
        #     logger.info(f"Text: {text}")
        #     logger.info(f"Prediction: {pred}\n")
        
        # logger.info("All tests completed successfully!")

        # loaded_model = mlflow.pyfunc.load_model(
        #     "wasbs://model-files@questaistorageaccount.blob.core.windows.net/model-files/bert-tiny"
        # )

        # # Make predictions
        # predictions = loaded_model.predict(["This is a test input"])
        # print(predictions)
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_save_and_load() 