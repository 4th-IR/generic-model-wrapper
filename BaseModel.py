from transformers import pipeline
from adlfs import AzureBlobFileSystem
from typing import Optional, Dict, Any


class UniversalPipelineModel:
    """
    Unified model interface using Hugging Face pipelines with Azure Blob Storage integration
    Handles:
    - Text models (classification, generation)
    - Image models (classification, segmentation)
    - Multi-modal pipelines
    - Custom processors
    """
    
    def __init__(
        self,
        azure_path: str,
        task: Optional[str] = None,
        framework: Optional[str] = None,
        **pipeline_kwargs
    ):
        """
        :param azure_path: Azure Blob path (e.g., "az://my-container/models/bert-base")
        :param task: Pipeline task type (auto-detected if None)
        """
        self.fs = AzureBlobFileSystem(
            account_name="<account>",
            credential="<key_or_token>",
        )
        
        # Mount Azure path to local filesystem
        self.local_path = f"/tmp/{azure_path.split('/')[-1]}"
        self.fs.get(azure_path, self.local_path, recursive=True)
        
        # Initialize pipeline with automatic component detection
        self.pipeline = pipeline(
            task=task,
            model=self.local_path,
            framework=framework,
            **pipeline_kwargs
        )
        
        # Store references to components
        self.model = self.pipeline.model
        self.tokenizer = getattr(self.pipeline, "tokenizer", None)
        self.feature_extractor = getattr(self.pipeline, "feature_extractor", None)
        self.image_processor = getattr(self.pipeline, "image_processor", None)

    def predict(self, inputs, **kwargs):
        """Unified prediction interface"""
        return self.pipeline(inputs, **kwargs)
    
    def save_to_azure(self, azure_path: str):
        """Save updated model/processor to Azure"""
        with self.fs.open(azure_path, "wb") as f:
            self.pipeline.save_pretrained(f.name)
        
        # Sync local files
        self.fs.put(self.local_path, azure_path, recursive=True)

    @classmethod
    def from_pretrained(cls, azure_path: str, **kwargs):
        """Alternative constructor matching HF pattern"""
        return cls(azure_path=azure_path, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fs.rm(self.local_path, recursive=True)