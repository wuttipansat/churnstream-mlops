from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow.sklearn
import pandas as pd
from pathlib import Path

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
            threshold: float,
    ) -> None:
        
        self.model = model
        self.threshold = threshold

    @classmethod
    def load(cls, settings: Settings) -> "ModelService":
        model_path = Path(settings.local_model_path).resolve()
        
        if not (model_path / "MLmodel").exists():
            raise FileNotFoundError(f"MLflow model not found: {model_path}")
        
        model = mlflow.sklearn.load_model(model_path.as_uri())

        return cls(
            model=model,
            threshold=settings.churn_threshold,
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
    


