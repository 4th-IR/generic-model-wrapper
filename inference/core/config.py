from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    AZURE_STORAGE_CONNECTION_STRING: str
    AZURE_CONTAINER_NAME: str
    AZURE_STORAGE_ACCOUNT: str
    MODEL_IDENTIFIER: Optional[str] = None
    TASK: Optional[str] = None
    PROVIDER: str = "huggingface"


settings = Settings()
