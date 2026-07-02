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

def main():
    settings = get_settings()

    model_path = Path(__file__).resolve().parents[1] / "models" / "churn_model"

    print(f"Loading model: {model_path}")

    model = mlflow.sklearn.load_model(str(model_path))

    result = predict(model, customer_data=SAMPLE_CUSTOMER, threshold=settings.churn_threshold)

    prediction = result["prediction"]
    probability = result["churn_probability"]

    print(f"Prediction: {prediction}")
    print(f"Result: {'CHURN' if prediction else 'NOT CHURN'}")
    print(f"Probability: {probability:.4f}")
    print("UNIT prediction: PASSED")

if __name__ == "__main__":
    raise SystemExit(main())
