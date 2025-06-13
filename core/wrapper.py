"""A Generic Model Wrapper Class to Standardize Framework Agnostic Model Loading and Inference"""

import json

# external
import os
import shutil
import tempfile
import time

from azure.storage.blob import BlobServiceClient
from fastapi import HTTPException

from core.huggingface_route import download_from_huggingface
from core.config import settings

# internal
from logging import getLogger

LOG = getLogger("model")


class ModelWrapper:

    def __init__(self):

        if os.path.exists("./wrapper_config.json"):
            with open("./wrapper_config.json") as f:
                content = f.read()
                wrapper_config = json.loads(content)

                self.model_provider = wrapper_config.get("provider", None)
                self.model_identifier = wrapper_config.get("model_identifier", None)
                self.task = wrapper_config.get("task", None)
                self.model_save_path = "models/" + self.model_identifier
        else:
            self.model_provider = settings.PROVIDER
            self.model_identifier = settings.MODEL_IDENTIFIER
            self.task = settings.TASK

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

    def upload_directory_to_storage(self, local_dir, blob_prefix=""):
        """
        Uploads a local directory to Azure Blob Storage.
        Ensures that blob_prefix is safe by replacing '/' with '_'.
        """
        # Sanitize the blob prefix if needed
        safe_blob_prefix = blob_prefix.replace("/", "_") if blob_prefix else ""

        for root, _, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                blob_name = os.path.relpath(local_path, local_dir)

                if safe_blob_prefix:
                    blob_name = f"{safe_blob_prefix}/{blob_name}"

                blob_client = self.container_client.get_blob_client(blob_name)
                with open(local_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)

    def save_to_storage(self):
        """Saves the model directory/files to Azure Blob Storage under a folder named after the model."""

        LOG.info(
            f"Attempting to save model '{self.model_identifier}' to Azure container '{settings.AZURE_CONTAINER_NAME}'."
        )

        if not self.model_save_path or not os.path.exists(self.model_save_path):
            error_msg = f"Model save path '{self.model_save_path}' is not valid or model hasn't been saved locally first."
            LOG.error(error_msg)
            raise ValueError(error_msg)

        try:
            LOG.info(f"Local model path to upload: {self.model_save_path}")

            # Pass the safe model name as the blob prefix
            self.upload_directory_to_storage(
                local_dir=self.model_save_path, blob_prefix=self.safe_model_name
            )

            LOG.info(
                f"Successfully saved '{self.model_identifier}' to Azure Blob Storage in container '{settings.AZURE_CONTAINER_NAME}' under folder '{self.safe_model_name}'."
            )
        except Exception as e:
            LOG.error(
                f"Failed to save model {self.model_identifier} to Azure: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Model saving to Azure failed: {e}")

        LOG.info(
            f"{self.model_identifier} saved to Azure Blob Storage under folder '{self.safe_model_name}'"
        )

    def load_from_storage(self):
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

                # else:
                #     LOG.info(f"Downloading model from {self.model_provider}...")
                #     temp_download_dir = tempfile.gettempdir()
                #     save_path = os.path.join(temp_download_dir, self.safe_model_name)
                #     self.model_save_path = save_path

                #     if self.model_provider == "huggingface":
                #         LOG.info(f"Loading Hugging Face model: {self.model_identifier}")
                #         download_from_huggingface(
                #             model_name=self.model_identifier,
                #             task=self.task,
                #             model_path=self.model_save_path,
                #         )

                #     # make some changes
                #     elif self.model_provider == "pytorch":
                #         LOG.info(f"Loading PyTorch model from {self.model_identifier}")
                #         print(f"Loading PyTorch model from {self.model_identifier}")
                #         # self.model_save_path = download_from_torch(
                #         #     self.model_identifier
                #         # )

                #     elif self.model_provider == "tensorflow":
                #         LOG.info(
                #             f"Loading TensorFlow model from {self.model_identifier}"
                #         )
                #         print(f"Loading TensorFlow model from {self.model_identifier}")
                #         # self.model_save_path = download_from_tensorflow(
                #         #     self.model_identifier
                #         # )

                #     if self.azure_config:
                #         LOG.info("Saving model from Provider to Azure...")
                #         self.save_to_azure()

                #     if os.path.exists(save_path):
                #         shutil.rmtree(save_path)
                #         LOG.info(f"Deleted temporary directory: {save_path}")

                #     else:
                #         LOG.error(f"Unknown model provider: {self.model_provider}")
                #         raise ValueError(
                #             f"Unsupported model provider: {self.model_provider}"
                #         )

                #     LOG.info("Model loaded successfully.")

        except Exception as e:
            LOG.error(f"Error loading model: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Model loading failed: {str(e)}"
            )

    def load_from_provider(
        self,
        model_identifier=None,
        task=None,
        model_save_path=None,
    ):
        """
        Loads model from provider
        """

        if self.model_provider == "huggingface":
            model_downloaded = download_from_huggingface(
                model_identifier if model_identifier else self.model_identifier,
                task if task else self.task,
                model_save_path if model_save_path else self.model_save_path,
            )

            return model_downloaded

        return False


wrapper = ModelWrapper()


# if __name__ == "__main__":

#     import openpyxl
#     import pandas as pd
#     import psutil

#     models_dict = {
#         # "model_3": {"model_provider": "huggingface","task": "text-generation", "model_name": "mistralai/Mistral-7B-Instruct-v0.3"},
#         "model_4": {
#             "model_provider": "huggingface",
#             "task": "text-generation",
#             "model_name": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
#         },
#     }
#     excel_file = "model_saving_metrics.xlsx"

#     model_inference_metrics = []

#     for model in models_dict.values():

#         model_name = model["model_name"]
#         model_provider = model["model_provider"]
#         model_task = model["task"]

#         start_inference_time = time.time()
#         process = psutil.Process(os.getpid())

#         mem_before = process.memory_info().rss / (1024**2)

#         model_wrapper = ModelWrapper(model_provider, model_name, model_task)
#         model_wrapper.save_model()

#         end_download_time = time.time()
#         print("/nDownload completed")

#         mem_after = process.memory_info().rss / (1024**2)

#         mem_used = mem_after - mem_before

#         TOTAL_TIME_TAKEN = round((end_download_time - start_inference_time) / 60, 2)
#         model_data = {
#             "model_name": model_name,
#             "model_provider": model_provider,
#             "model_task": model_task,
#             "total_time_taken(mins)": TOTAL_TIME_TAKEN,
#             "memory_used": mem_used,
#         }

#         model_inference_metrics.append(model_data)
#         print("/n Metrics obtained", model_inference_metrics)

#     df = pd.DataFrame(model_inference_metrics)

#     if os.path.exists(excel_file):
#         existing_df = pd.read_excel(excel_file)
#         updated_df = pd.concat([existing_df, df], ignore_index=True)
#     else:
#         updated_df = df

#     updated_df.to_excel(excel_file, index=False)

#     print(f"Saved {len(df)} model entries to {excel_file}")
