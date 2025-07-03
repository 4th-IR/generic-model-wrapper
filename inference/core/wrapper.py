"""A Generic Model Wrapper Class to Standardize Framework Agnostic Model Loading and Inference"""

import json

import os

from azure.storage.blob import BlobServiceClient

from fastapi import HTTPException

import spacy
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModel, AutoProcessor, AutoFeatureExtractor

from core.config import settings

from logging import getLogger

logs = getLogger("model")


class ModelWrapper:

    def __init__(self):
        self.model_provider = settings.PROVIDER
        self.model_identifier = settings.MODEL_IDENTIFIER
        self.task = settings.TASK
        self.model_save_path = "./models/" + self.model_identifier
        self.model = None

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
                logs.info("Checking Azure Storage for model...")
                print("Checking Azure Storage for model...")
                model_loaded = None

                blob_list = list(
                    self.container_client.list_blobs(
                        name_starts_with=f"{self.safe_model_name}/"
                    )
                )
                if not blob_list:
                    logs.warning(
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
                            logs.info(f"Skipping (already exists): {local_path}")
                            continue

                        os.makedirs(os.path.dirname(local_path), exist_ok=True)

                        with open(local_path, "wb") as file:
                            download_stream = blob_client.download_blob()
                            file.write(download_stream.readall())
                            logs.info(f"Downloaded: {blob.name} -> {local_path}")

                    logs.info(
                        f"{self.model_identifier} downloaded from Azure to: {self.model_save_path}"
                    )
                    logs.info(
                        "Model successfully loaded from Azure and initialized for inference."
                    )

                if settings.PROVIDER == "huggingface":
                    task_pipeline_details = PIPELINE_REGISTRY.supported_tasks.get(
                        settings.TASK, None
                    ) or PIPELINE_REGISTRY.supported_tasks.get(
                        PIPELINE_REGISTRY.task_aliases.get(settings.TASK), None
                    )
                    if task_pipeline_details:
                        automodel_class: AutoModel = task_pipeline_details.get("pt")[0]
                        self.model = automodel_class.from_pretrained(
                            self.model_save_path
                        )
                        self.model.eval()
                        model_loaded = True
                elif settings.PROVIDER == "spacy":
                    self.model = spacy.load(self.model_save_path)
                    model_loaded = True

                model_loaded = True

                if model_loaded:
                    logs.info(
                        f"Loaded model from Azure Storage: {self.model_identifier}"
                    )
                    print(f"Loaded model from Azure Storage: {self.model_identifier}")
                return model_loaded

        except Exception as e:
            logs.error(f"Error loading model: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Model loading failed: {str(e)}"
            )


wrapper = ModelWrapper()
