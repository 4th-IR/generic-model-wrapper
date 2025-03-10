# UniversalModelWrapper: MLflow Model Management with Azure Integration

## Overview

The `UniversalModelWrapper` is a versatile MLflow-based solution for managing machine learning models with seamless Azure Blob Storage integration. It provides a standardized interface for loading, saving, and deploying models from various ML frameworks (HuggingFace, TensorFlow, PyTorch) while abstracting away storage complexity.

## Features

- **Multi-framework support**: Works with HuggingFace, TensorFlow, PyTorch, and other ML frameworks
- **Seamless Azure Storage integration**: Direct model loading from Azure Blob Storage
- **Chunked upload/download**: Efficient handling of large model files with retry logic
- **MLflow compatibility**: Full integration with MLflow's model management capabilities
- **Unified prediction interface**: Standard interface regardless of underlying model type
- **Automatic component detection**: Auto-discovers model components like tokenizers and preprocessors

## Architecture

The wrapper is built on top of MLflow's `PythonModel` class and extends it with enhanced functionality:

```text
UniversalModelWrapper (extends mlflow.pyfunc.PythonModel)
├── Model Loading Engine
│   ├── HuggingFace Loader
│   ├── TensorFlow Loader
│   ├── PyTorch Loader
│   └── Custom Loaders
├── Model Storage Interface
│   ├── Local Storage
│   └── Azure Blob Storage
└── Prediction Service
    ├── Unified Prediction API
    └── Type Adaptation
```

## Getting Started

### Prerequisites

- Python 3.8+
- MLflow
- Azure Storage libraries
- ML framework of your choice (HuggingFace, TensorFlow, PyTorch)

### Installation

```bash
pip install mlflow azure-storage-blob adlfs transformers torch tensorflow
```

### Environment Configuration

Create a `.env` file with your Azure Storage credentials:

```env
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
AZURE_STORAGE_ACCOUNT=your_account_name
```

## Usage Examples

### Saving a Model to Azure Blob Storage

```python
from PyfuncModel import UniversalModelWrapper

# Save a HuggingFace model to Azure Blob Storage
wrapper = UniversalModelWrapper.save(
    model_name="bert-base-uncased",
    model_provider="huggingface",
    task="text-classification",
    azure_config={
        "connection_string": connection_string,
        "account_name": account_name
    }
)
```

### Loading a Model from Azure Blob Storage

```python
import mlflow

# Load model directly from Azure using MLflow
model_path = "model-files/bert-base-uncased"  # container/model_name
storage_options = {
    "connection_string": connection_string,
    "account_name": account_name,
    "container": "model-files"
}

# Direct loading from Azure
loaded_model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_path}",
    storage_options=storage_options
)

# Make predictions
predictions = loaded_model.predict(["This movie is great!", "I didn't like this film"])
```

## Detailed Use Case: End-to-End Model Deployment

This example demonstrates the complete workflow of fetching a model from HuggingFace, saving it to Azure Blob Storage, and loading it for predictions.

```python
import os
from pathlib import Path
from dotenv import load_dotenv
import mlflow
import logging
from azure.storage.blob import BlobServiceClient
from PyfuncModel import UniversalModelWrapper

# Load environment variables
load_dotenv()

# Get Azure configuration from environment
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
azure_config = {
    "connection_string": connection_string,
    "account_name": account_name
}

# 1. Define model configuration
model_name = "prajjwal1/bert-tiny"  # A small model for testing
container_name = "model-files"
model_path = Path(model_name).name
full_model_path = f"{container_name}/{model_path}"

# 2. Ensure Azure container exists
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)
if not container_client.exists():
    container_client = blob_service_client.create_container(container_name)

# 3. Save the model to Azure
wrapper = UniversalModelWrapper.save(
    model_name=model_name,
    model_provider="huggingface",
    task="text-classification",
    azure_config=azure_config
)
print(f"Model saved to Azure path: {full_model_path}")

# 4. Load model directly from Azure
storage_options = {
    "connection_string": connection_string,
    "account_name": account_name,
    "container": container_name
}

loaded_model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{full_model_path}",
    storage_options=storage_options
)

# 5. Make predictions
test_inputs = ["This movie is great!", "I didn't like this film at all"]
predictions = loaded_model.predict(test_inputs)
print(f"Predictions: {predictions}")
```

## Implementation Details

### Model Saving Process

The saving process follows these steps:

1. Load the model from the source provider (e.g., HuggingFace)
2. Save the model files to a temporary local directory
3. Save metadata and configuration in a structured format
4. Upload all files to Azure Blob Storage using efficient chunking
5. Create and save an MLflow model wrapper that points to these files

### Model Loading Process

The loading process has two paths:

1. **Direct Azure Loading**: Attempts to load the model directly from Azure using MLflow's remote model support
2. **Fallback Loading**: If direct loading fails, downloads files to a temporary directory and loads from there

### Chunked Upload for Large Files

For large model files, the wrapper implements chunked upload with these features:

- Configurable chunk size (default: 4MB)
- Retry logic with exponential backoff
- Progress reporting
- Block-based upload for large files

## Common Issues and Solutions

### Connection Timeouts

If you experience connection timeouts during model upload:

- Reduce chunk size (modify `MAX_CHUNK_SIZE` value)
- Increase retry count and timeout values
- Consider using a more stable network connection
- For very large models, consider using Azure Storage Explorer for initial upload

### Model Loading Errors

If model loading fails:

- Verify Azure credentials and permissions
- Check container and path existence
- Examine logs for specific error messages
- Try the fallback local loading approach

## Limitations and Future Improvements

### Current Limitations

- Serialization can be challenging for very large models (>10GB)
- Network latency when retrieving models from cloud storage
- Limited support for specialized model types

### Future Improvements

- Implement model caching with Redis or similar technology
- Add support for model versioning and A/B testing
- Implement concurrent/parallel uploads for faster model saving
- Add compression options for more efficient storage
- Support for additional cloud providers (AWS, GCP)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.