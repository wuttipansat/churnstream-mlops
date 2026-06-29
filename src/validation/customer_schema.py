import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

class CustomerDatasetSchema(pa.DataFrameModel):
    RowNumber: Series[int] = pa.Field(
        ge=1,
        nullable=False,
    )

    CustomerId: Series[int] = pa.Field(
        ge=1,
        nullable=False,
        unique=True
    )

    Surname: Series[str] = pa.Field(
        nullable=False,
    )

    CreditScore: Series[int] = pa.Field(
        ge=0,
        nullable=False,
    )

    Geography: Series[str] = pa.Field(
        nullable=False,
    )

    Gender: Series[str] = pa.Field(
        isin=["Male", "Female"],
        nullable=False,
    )

    Age: Series[int] = pa.Field(
        ge=18,
        nullable=False,
    )

    Balance: Series[float] = pa.Field(
        ge=0,
        nullable=False
    )

    HasCrCard: Series[int] = pa.Field(
        isin=[0, 1],
        nullable=False,
    )

    IsActiveMember: Series[int] = pa.Field(
        isin=[0, 1],
        nullable=False,
    )

    EstimatedSalary: Series[float] = pa.Field(
        ge=0,
        nullable=False,
    )

    Exited: Series[int] = pa.Field(
        isin=[0, 1],
        nullable=False,
    )

    class Config:
        coerce = True
        strict = True


def validate_customer_dataset(
        dataframe: pd.DataFrame,
) -> pd.DataFrame:
    return CustomerDatasetSchema.validate(
        dataframe,
        lazy=True
    )

