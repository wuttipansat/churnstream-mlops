from typing import Any, Mapping

import pandas as pd

from churnstream.features.customer_features import build_features
from churnstream.validation import validate_prediction_input

def predict(
        model,
        customer_data: Mapping[str, Any],
        threshold: float = 0.5,
) -> dict[str, float | int]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1")
    
    validated_customer = validate_prediction_input(customer_data)

    customer_dataframe = pd.DataFrame(
        [validated_customer.model_dump()]
    )

    features = build_features(customer_dataframe)

    probability = float(model.predict_proba(features)[0, 1])

    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "churn_probability": probability
    }

