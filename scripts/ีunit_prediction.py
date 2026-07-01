import sys
from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from pydantic import ValidationError

from churnstream.core.config import get_settings
from churnstream.prediction.predict import predict

SAMPLE_CUSTOMER = {
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 5,
    "Balance": 125000.50,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 75000.00,
}

def get_model_uri(
        client: MlflowClient,
        model_name: str,
        model_alias: str | None,
) -> str:
    
    if model_alias:
        try:
            model_version = (
                client.get_model_version_by_alias(
                    name=model_name,
                    alias=model_alias
                )
            )

            print(f"Using model: {model_name}@{model_alias} version {model_version.version}")

            return f"models:/{model_name}/{model_version.version}"
        
        except MlflowException:
            print(f"Alias '{model_alias}' not found. Using latest version.")

    model_versions = list(
        client.search_model_versions(
            filter_string=f"name='{model_name}'"
        )
    )

    if not model_versions:
        raise ValueError(f"Model '{model_name}' not found.")
    
    latest_version = max(
        model_version,
        key=lambda item: int(item.version)
    )

    print(f"Using model: {model_name} version {latest_version.version}")

    return f"model:/{model_name}/{latest_version.version}"

def main() -> int:
    settings = get_settings()
    print("=== Unit Prediction ===")
    print(f"MLflow tracking URI: {settings.mlflow_tracking_uri}")
    print(f"Registered model: {settings.mlflow_model_name}")
    print(f"Prediction threshold: {settings.churn_threshold}")

    try:
        mlflow.set_tracking_uri(
            settings.mlflow_tracking_uri
        )

        print("Finding model...")

        client = MlflowClient(
            tracking_uri=settings.mlflow_tracking_uri
        )

        model_uri = get_model_uri(
            client=client,
            model_name=settings.mlflow_model_name,
            model_alias=settings.mlflow_model_alias,
        )

        print(f"Model URI: {model_uri}")
        print("Loading Model...")

        model = mlflow.sklearn.load_model(
            model_uri
        )

        print("Model loading: PASSED")
        print("Running prediction...")

        result = predict(
            model=model,
            customer_data=SAMPLE_CUSTOMER,
            threshold=settings.churn_threshold,
        )

        prediction = result["prediction"]
        probability = result["churn_probability"]

        prediction_label = (
            "CHURN"
            if prediction == 1
            else "NOT CHURN"
        )

        print("\n=== Prediction Result ===")
        print(f"Prediction: {prediction}")
        print(f"Result: {prediction_label}")
        print(
            f"Churn probability: "
            f"{probability:.4f}"
        )
        print(
            f"Churn percentage: "
            f"{probability * 100:.2f}%"
        )

        print("\nUnit prediction: PASSED")

        return 0

    except ValidationError as error:
        print("\nInput validation: FAILED")
        print(error)

        return 1

    except KeyboardInterrupt:
        print("\nPrediction cancelled")

        return 1

    except Exception as error:
        print("\nUnit prediction: FAILED")
        print(
            f"{type(error).__name__}: {error}"
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
