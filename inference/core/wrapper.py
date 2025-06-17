"""A Generic Model Wrapper Class to Standardize Framework Agnostic Model Loading and Inference"""

import json

import os

from azure.storage.blob import BlobServiceClient
from fastapi import HTTPException

from core.config import settings

from logging import getLogger

LOG = getLogger("model")


class ModelWrapper:

    def __init__(self):

        # if os.path.exists("./wrapper_config.json"):
        #     with open("./wrapper_config.json") as f:
        #         content = f.read()
        #         wrapper_config = json.loads(content)

        #         self.model_provider = wrapper_config.get("provider", None)
        #         self.model_identifier = wrapper_config.get("model_identifier", None)
        #         self.task = wrapper_config.get("task", None)
        #         self.model_save_path = "models/" + self.model_identifier
        # else:
        self.model_provider = settings.PROVIDER
        self.model_identifier = settings.MODEL_IDENTIFIER
        self.task = settings.TASK
        self.model_save_path = "./models/" + self.model_identifier

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

        # model name to ensure proper dir creation
        self.safe_model_name = self.model_identifier.replace("/", "_")
        # temp files to store the model loaded for inferencing

        self.load_from_storage()

    def load_from_storage(self, force_redownload=False):
        """Loads models from Azure Storage if available"""

        try:
            if self.azure_config:
                LOG.info("Checking Azure Storage for model...")
                print("Checking Azure Storage for model...")
                model_loaded = None

                blob_list = list(
                    self.container_client.list_blobs(
                        name_starts_with=f"{self.safe_model_name}/"
                    )
                )
                if not blob_list:
                    LOG.warning(
                        f"No blobs found for model '{self.model_identifier}' under folder '{self.safe_model_name}/'"
                    )
                    print("Model blob folder does not exist")
                    model_loaded = False

                else:

                    # Download each file under the model folder
                    for blob in blob_list:
                        blob_client = self.container_client.get_blob_client(blob)

                        # Get relative path within the model folder
                        relative_path = os.path.relpath(
                            blob.name, start=f"{self.safe_model_name}/"
                        )
                        local_path = os.path.join(self.model_save_path, relative_path)

                        # Skip download if file already exists unless using force
                        if os.path.exists(local_path) and not force_redownload:
                            LOG.info(f"Skipping (already exists): {local_path}")
                            continue

                        os.makedirs(os.path.dirname(local_path), exist_ok=True)

                        with open(local_path, "wb") as file:
                            download_stream = blob_client.download_blob()
                            file.write(download_stream.readall())
                            LOG.info(f"Downloaded: {blob.name} -> {local_path}")

                    LOG.info(
                        f"{self.model_identifier} downloaded from Azure to: {self.model_save_path}"
                    )
                    LOG.info(
                        "Model successfully loaded from Azure and initialized for inference."
                    )
                    model_loaded = True

                if model_loaded:
                    LOG.info(
                        f"Loaded model from Azure Storage: {self.model_identifier}"
                    )
                    print(f"Loaded model from Azure Storage: {self.model_identifier}")
                return model_loaded

        except Exception as e:
            LOG.error(f"Error loading model: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Model loading failed: {str(e)}"
            )


wrapper = ModelWrapper()
