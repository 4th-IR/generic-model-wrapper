from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    ALLOWED_ORIGINS: str = "*"
    AZURE_STORAGE_CONNECTION_STRING: str
    AZURE_CONTAINER_NAME: str
    AZURE_STORAGE_ACCOUNT: str


settings = Settings()
