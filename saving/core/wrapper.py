"""A Generic Model Wrapper Class to Standardize Framework Agnostic Model Loading and Inference"""

import os
import shutil

from azure.storage.blob import BlobServiceClient

from core.huggingface_route import download_from_huggingface
from core.config import settings

from logging import getLogger

logs = getLogger("model")


class ModelWrapper:

    def __init__(self):

        # Azure storage setup
        if not settings.AZURE_STORAGE_CONNECTION_STRING:
            raise ValueError(
                "Azure connection string is missing! Check environment variables."
            )
        if not settings.AZURE_CONTAINER_NAME:
            raise ValueError(
                "Azure container name is missing! Check environment variables."
            )

        self.blob_service_client = BlobServiceClient.from_connection_string(
            settings.AZURE_STORAGE_CONNECTION_STRING
        )
        self.container_client = self.blob_service_client.get_container_client(
            settings.AZURE_CONTAINER_NAME
        )
        self.azure_config = True  # Enable Azure model loading
        self.models_base_dir = "./models/"

    def load_from_provider(
        self,
        model_provider=None,
        model_identifier=None,
        task=None,
        kwargs=None,
        model_save_path=None,
    ):
        """
        Loads model from provider
        """

        if model_provider == "huggingface":
            model_downloaded = download_from_huggingface(
                model_identifier,
                task,
                model_save_path,
                kwargs,
            )

            return model_downloaded

        return False

    def save_to_storage(self, model_identifier, model_save_path):
        """Saves the model directory/files to Azure Blob Storage under a folder named after the model."""

        logs.info(
            f"Attempting to save model '{model_identifier}' to Azure container '{settings.AZURE_CONTAINER_NAME}'."
        )

        if not model_save_path or not os.path.exists(model_save_path):
            error_msg = f"Model save path '{model_save_path}' is not valid or model hasn't been saved locally first."
            logs.error(error_msg)
            raise ValueError(error_msg)

        try:
            logs.info(f"Local model path to upload: {model_save_path}")

            # Sanitize the blob prefix if needed
            safe_model_name = model_identifier.replace("/", "_")
            safe_blob_prefix = safe_model_name

            for root, _, files in os.walk(model_save_path):
                for file in files:
                    local_path = os.path.join(root, file)
                    blob_name = os.path.relpath(local_path, model_save_path)

                    if safe_blob_prefix:
                        blob_name = f"{safe_blob_prefix}/{blob_name}"

                    blob_client = self.container_client.get_blob_client(blob_name)
                    with open(local_path, "rb") as data:
                        blob_client.upload_blob(data, overwrite=True)

            logs.info(
                f"Successfully saved '{model_identifier}' to Azure Blob Storage in container '{settings.AZURE_CONTAINER_NAME}' under folder '{safe_model_name}'."
            )
        except Exception as e:
            logs.error(
                f"Failed to save model {model_identifier} to Azure: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Model saving to Azure failed: {e}")

        logs.info(
            f"{model_identifier} saved to Azure Blob Storage under folder '{safe_model_name}'"
        )

    # def clear_model_folder(self, model_safe_name):
    #     if os.path.exists(self.models_base_dir):

    #         model_path = os.path.join(self.models_base_dir, model_safe_name)

    #         if os.path.isdir(model_path):
    #             shutil.rmtree(model_path)
    #             logs.info(f"Deleted model directory: {model_path}")
    #         else:
    #             logs.info(f"No such model directory: {model_path}")


wrapper = ModelWrapper()
