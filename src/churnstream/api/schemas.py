from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Literal

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: int = Field(gt=0)

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

class PredictionResponse(BaseModel):
    request_id: str
    customer_id: int
    churn_prediction: int
    churn_probability: float
    model_name: str
    model_version: str
    model_alias: str
    threshold: float
    replayed: bool = False

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    model_alias: str