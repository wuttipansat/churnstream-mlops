from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

class CustomerPredictionInput(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        strict=True,
    )

    CreditScore: int = Field(ge=0)
    Geography: str = Field(min_length=1)
    Gender: Literal["Male", "Female"]

    Age: int = Field(ge=18)
    Tenure: int = Field(ge=0)
    Balance: float = Field(ge=0)
    NumOfProducts: int = Field(ge=0)

    HasCrCard: Literal[0, 1]
    IsActiveMember: Literal[0, 1]

    EstimatedSalary: float = Field(ge=0)

    @field_validator("Geography")
    @classmethod
    def validate_geography(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Geography must not be empty.")
        
        return value
    
def validate_prediction_input(
        customer_data: Mapping[str, Any],
) -> CustomerPredictionInput:
    
    return CustomerPredictionInput.model_validate(customer_data)
