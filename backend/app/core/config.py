from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql://mlops:mlops@localhost:5432/mlops"
    mlflow_tracking_uri: str = "http://localhost:5000"
    upload_dir: str = "./data/uploads"
    api_key: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
