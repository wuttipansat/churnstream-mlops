from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from churnstream.features.customer_features import(
    CUSTOMER_FEATURES,
)

class CustomerFeatureEngineer(
    BaseEstimator,
    TransformerMixin
):
    
    def fit(self, X: pd.DataFrame, y: Any=None) -> "CustomerFeatureEngineer":
        self._validate_input(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(X)

        result = X.copy()

        result["BalanceSalaryRatio"] = (
            self._safe_divide(
                numerator=result["Balance"],
                denominator=result["EstimatedSalary"]
            )
        )

        result["BalancePerProduct"] = (
            self._safe_divide(
                numerator=result["Balance"],
                denominator=result["NumOfProducts"]
            )
        )

        result["TenureAgeRatio"] = (
            self._safe_divide(
                numerator=result["Tenure"],
                denominator=result["Age"]
            )
        )

        result["IsZeroBalance"] = (
            result["Balance"].eq(0).astype("float64")
        )

        result["HasMultipleProducts"] = (
            result["NumOfProducts"]
            .gt(1)
            .astype("float64")
        )

        result["AgeGroup"] = pd.cut(
            result["Age"],
            bins=[0, 30, 45, 60, np.inf],
            labels=["18-30", "31-45", "46-60", "61+"],
            include_lowest=True,
        ).astype("object")

        return result
    
    @staticmethod
    def _safe_divide(
        numerator: pd.Series,
        denominator: pd.Series,
    ) -> pd.Series:
        numerator_float = pd.to_numeric(
            numerator, 
            errors="coerce"
        ).astype("float64")

        denominator_float = pd.to_numeric(
            denominator,
            errors="coerce"
        )

        safe_denominator = denominator_float.where(
            denominator_float.ne(0)
        )

        result = numerator_float.div(safe_denominator)

        return result.replace(
            [np.inf, -np.inf],
            np.nan,
        )

    @staticmethod
    def _validate_input(
        dataframe: pd.DataFrame
    ) -> None:
        if not isinstance(
            dataframe,
            pd.DataFrame
        ):
            raise TypeError(
                "CustomerFeatureEngineer expects a pandas DataFrame"
            )
        
        required_columns = set(
            CUSTOMER_FEATURES.input_features
        )

        missing_columns = (
            required_columns.difference(
                dataframe.columns
            )
        )

        if missing_columns:
            missing_text = ", ".join(
                sorted(missing_columns)
            )

            raise ValueError(
                f"Missing columns for feature engineering: {missing_text}"
            )
        
    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(CUSTOMER_FEATURES.model_features, dtype=object)