from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    mongodb_uri: str | None = None
    mongodb_database: str = "churnstream"
    mongodb_customers_collection: str = "customers"
    mongodb_predictions_collection: str = "churn_predictions"

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_model_name: str = "churn_model"
    mlflow_model_alias: str = "champion"

    churn_threshold: float = 0.5
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

