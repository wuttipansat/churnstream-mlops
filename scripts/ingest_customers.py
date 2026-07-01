from datetime import timezone, datetime
from pathlib import Path

import pandas as pd
from pandera.errors import SchemaError

from churnstream.core.config import get_settings
from churnstream.ingestion.mongodb_repository import MongoDBRepository
from churnstream.validation.customer_schema import validate_customer_dataset


def main() -> None:
    settings = get_settings()

    with MongoDBRepository(settings) as repository:
        documents = repository.read_customers()
    
    if not documents:
        raise RuntimeError(
            "No customer documents found in MongoDB"
        )
    dataframe = pd.DataFrame(documents)

    try:
        validated_dataframe = validate_customer_dataset(dataframe)

    except SchemaError as error:
        print(f"Data validation failed: {error.failure_cases.to_string(index=False)}")

        raise SystemExit(1) from error
    
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d-T%H%M%SZ"
    )


    output_path = Path(f"data/raw/customers_{timestamp}.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    validated_dataframe.to_parquet(output_path, index=False)
    
    print(f"Saved {len(dataframe)} customers to {output_path}")


if __name__ == "__main__":
    main()