from dataclasses import dataclass

import pandas as pd

@dataclass(frozen=True)
class CustomerFeatureConfig:
    target: str
    identifiers: tuple[str, ...]
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]

    @property
    def model_features(self) -> tuple[str, ...]:
        return self.numeric + self.categorical
    
CUSTOMER_FEATURES = CustomerFeatureConfig(
    target="Exited",
    identifiers=(
        "RowNumber",
        "CustomerId",
        "Surname"
    ),
    numeric=(
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ),
    categorical=(
        "Geography",
        "Gender",
    ),
)

def build_features(
        dataframe: pd.DataFrame,
) -> pd.DataFrame:
    
    required_features = set(
            CUSTOMER_FEATURES.model_features
        )

    missing_features = required_features.difference(
        dataframe.columns
    )
    
    if missing_features:
        missing_text = ", ".join(
            sorted(missing_features)
        )

        raise ValueError(
            f"Missing model features: {missing_text}"
        )
    
    return dataframe.loc[
        :,
        list(CUSTOMER_FEATURES.model_features),
    ].copy()

def build_target(
        dataframe: pd.DataFrame
) -> pd.Series:
    
    target_column = CUSTOMER_FEATURES.target

    if target_column not in dataframe.columns:
        raise ValueError(
            f"Missing target column: {target_column}"
        )
    
    return dataframe[target_column].copy()


