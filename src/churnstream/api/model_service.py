from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

from churnstream.core.config import Settings
from churnstream.features.customer_features import CUSTOMER_FEATURES

@dataclass(frozen=True)
class PredictionResult:
    prediction: int
    probability: float

class ModelService:
    def __init__(
            self, 
            model: Any, 
            model_name: str, 
            model_version: str, 
            model_alias: str, 
            threshold: float,
    ) -> None:
        
        self.model = model
        self.model_name = model_name
        self.model_version = model_version
        self.model_alias = model_alias
        self.threshold = threshold

    @classmethod
    def load(cls, settings: Settings) -> "ModelService":
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

        model_uri = f"models:/{settings.mlflow_model_name}@{settings.mlflow_model_alias}"

        client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

        model_version = client.get_model_version_by_alias(
            name=settings.mlflow_model_name,
            alias=settings.mlflow_model_alias,
        )

        model = mlflow.sklearn.load_model(model_uri)

        return cls(
            model=model,
            model_name=settings.mlflow_model_name,
            model_version=str(model_version.version),
            model_alias = settings.mlflow_model_alias,
            threshold = settings.churn_threshold,
        )
    
    def predict(
            self,
            input_data: dict[str, object],
    ) -> PredictionResult:
        dataframe = pd.DataFrame(
            [input_data],
            columns=CUSTOMER_FEATURES.input_features,
        )

        probability = float(
            self.model.predict_proba(dataframe)[0, 1]
        )

        prediction = int(
            probability >= self.threshold
        )

        return PredictionResult(
            prediction=prediction,
            probability=probability,
        )
    


