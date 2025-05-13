from azure.storage.blob import BlobServiceClient
from utils.env_manager import AZURE_BLOB_URI, AZURE_CONNECTION_STRING, AZURE_CONTAINER_NAME, AZURE_STORAGE_ACCOUNT


def list_blob_folders_and_files():
    service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container_client = service_client.get_container_client(AZURE_CONTAINER_NAME)

    print(f"\nBlobs in container '{AZURE_CONTAINER_NAME}' grouped by folders:\n")

    blobs = container_client.list_blobs()
    folder_map = {}

    for blob in blobs:
        parts = blob.name.split("/")
        if len(parts) > 1:
            folder = parts[0]
            folder_map.setdefault(folder, []).append(blob.name)
        else:g
            folder_map.setdefault("<root>", []).append(blob.name)

    for folder, files in folder_map.items():
        print(f"[Folder] {folder}/")
        for file in files:
            print(f"  └── {file.split('/', 1)[-1]}")
        print()

if __name__ == "__main__":
    list_blob_folders_and_files()