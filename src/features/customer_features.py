from dataclasses import dataclass

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