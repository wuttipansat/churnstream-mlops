from dataclasses import dataclass

import pandas as pd

@dataclass(frozen=True)
class CustomerFeatureConfig:
    target: str
    identifiers: tuple[str, ...]
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    engineered_numeric: tuple[str, ...]
    engineered_categorical: tuple[str, ...]

    @property
    def input_features(self) -> tuple[str, ...]:
        return self.numeric + self.categorical
    
    @property
    def model_numeric(self) -> tuple[str, ...]:
        return self.numeric + self.engineered_numeric
    
    @property
    def model_categorical(self) -> tuple[str, ...]:
        return self.categorical + self.engineered_categorical
    
    @property
    def model_features(self) -> tuple[str, ...]:
        return self.model_numeric + self.model_categorical
    
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
    engineered_numeric=(
        "BalanceSalaryRatio",
        "BalancePerProduct",
        "TenureAgeRatio",
        "IsZeroBalance",
        "HasMultipleProducts"
    ),
    engineered_categorical=(
        "AgeGroup",
    ),
)

def build_features(
        dataframe: pd.DataFrame,
) -> pd.DataFrame:
    
    required_features = set(
            CUSTOMER_FEATURES.input_features
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
    
    features = dataframe.loc[
        :,
        list(CUSTOMER_FEATURES.input_features),
    ].copy()

    numeric_columns = list(CUSTOMER_FEATURES.numeric)

    features[numeric_columns] = features[numeric_columns].astype("float64")

    return features

def build_target(
        dataframe: pd.DataFrame
) -> pd.Series:
    
    target_column = CUSTOMER_FEATURES.target

    if target_column not in dataframe.columns:
        raise ValueError(
            f"Missing target column: {target_column}"
        )
    
    return dataframe[target_column].astype("int64").copy()


