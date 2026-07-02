from __future__ import annotations

import os
import shutil
from pathlib import Path

import mlflow
import mlflow.sklearn

from churnstream.core.config import get_settings

settings = get_settings()

OUTPUT_PATH = Path("models/churn_model")

def main() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model_uri = f"models:/{settings.mlflow_model_name}@{settings.mlflow_model_alias}"

    print(f"Loading model: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    if OUTPUT_PATH.exists():
        shutil.rmtree(OUTPUT_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    mlflow.sklearn.save_model(sk_model=model, path=str(OUTPUT_PATH))

    print(f"Model exported to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()