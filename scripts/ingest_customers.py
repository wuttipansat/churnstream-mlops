from pathlib import Path

import pandas as pd

from churnstream.core.config import get_settings
from churnstream.ingestion.mongodb_repository import MongoDBRepository

def main() -> None:
    settings = get_settings()

    with MongoDBRepository(settings) as repository:
        documents = repository.read_customers()

    dataframe = pd.DataFrame(documents)

    output_path = Path("data/raw/customers.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    dataframe.to_parquet(output_path, index=False)
    
    print(f"Saved {len(dataframe)} customers to {output_path}")


if __name__ == "__main__":
    main()