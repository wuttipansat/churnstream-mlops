from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import(
    OneHotEncoder,
    StandardScaler,

)

from churnstream.core.config import get_settings
from churnstream.features.customer_features import (
    CUSTOMER_FEATURES,
    build_features,
    build_target
)
from churnstream.features.feature_engineering import CustomerFeatureEngineer
from churnstream.validation.customer_schema import validate_customer_dataset

RANDOM_STATE = 42
TEST_SIZE = 0.2
EXPERIMENT_NAME = "churn-prediction"

def find_latest_snapshot(
        directory: Path = Path("data/raw")
) -> Path:
    snapshots = list(
        directory.glob("customer*.parquet")
    )

    if not snapshots:
        raise FileNotFoundError(
            "No customer Parquet snapshot found."
        )
    
    return max(
        snapshots,
        key=lambda path: path.stat().st_mtime,
    )

def build_model_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers = [
            (
                "numeric",
                numeric_pipeline,
                list(CUSTOMER_FEATURES.model_numeric),
            ),
            (
                "categorical",
                categorical_pipeline,
                list(CUSTOMER_FEATURES.model_categorical),
            ),
        ],
        remainder='drop'
    )

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            (
                "feature_engineering",
                CustomerFeatureEngineer(),
            ),
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                classifier
            ),
        ]
    )

def calculate_metrics(
        y_true: pd.Series,
        predictions,
        probabilities,
) -> dict[str, float]:
    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
    }

def train_model() -> None:
    settings = get_settings()

    mlflow.set_tracking_uri(
        settings.mlflow_tracking_uri
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    snapshot_path = find_latest_snapshot()

    print(f"Using dataset: {snapshot_path}")

    dataframe = pd.read_parquet(
        snapshot_path
    )

    validated_dataframe = validate_customer_dataset(dataframe)

    features = build_features(
        validated_dataframe
    )

    target = build_target(
        validated_dataframe
    )

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    model = build_model_pipeline()

    with mlflow.start_run(
        run_name="logistic-regression-baseline"
    ):
        model.fit(
            X_train,
            y_train
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        predictions = (probabilities >= settings.churn_threshold).astype(int)

        metrics = calculate_metrics(
            y_true = y_test,
            predictions=predictions,
            probabilities=probabilities,
        )

        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "max_iter": 1_000,
                "class_weight": "balanced",
                "threshold": (
                    settings.churn_threshold
                ),
                "training_rows": len(X_train),
                "testing_rows": len(X_test),
                "feature_engineering_version": "v1",
                "raw_feature_count": len(
                    CUSTOMER_FEATURES.input_features
                ),
                "engineered_feature_count": (
                    len(
                        CUSTOMER_FEATURES.engineered_categorical
                    )
                    + len(
                        CUSTOMER_FEATURES.engineered_numeric
                    )
                ),
                "total_model_feature_count": len(
                    CUSTOMER_FEATURES.model_features
                )
            }
        )

        mlflow.log_metrics(metrics)

        mlflow.log_dict(
            {
                "target": CUSTOMER_FEATURES.target,
                "identifiers": list(
                    CUSTOMER_FEATURES.identifiers
                ),
                "raw_numeric": list(
                    CUSTOMER_FEATURES.numeric
                ),
                "raw_categorical": list(
                    CUSTOMER_FEATURES.categorical
                ),
                "engineered_numeric": list(
                    CUSTOMER_FEATURES.engineered_numeric
                ),
                "engineered_categorical": list(
                    CUSTOMER_FEATURES.engineered_categorical
                ),
                "input_features": list(
                    CUSTOMER_FEATURES.input_features
                ),
                "model_features": list(
                    CUSTOMER_FEATURES.model_features
                ),
            },
            "metadata/customer_features.json"
        )

        input_example = X_train.head(5).copy()

        signature = infer_signature(
            input_example,
            model.predict(input_example)
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=input_example,
            signature=signature,
            registered_model_name=(
                settings.mlflow_model_name
            ),
            serialization_format="cloudpickle"
        )

        print("\n=== Metrics ===")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        print(
            f"\nRegistered model: "
            f"{settings.mlflow_model_name}"
        )

        print(
            f"Model URI: {model_info.model_uri}"
        )