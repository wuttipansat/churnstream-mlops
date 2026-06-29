import pandas as pd

from churnstream.core.config import get_settings
from churnstream.ingestion.mongodb_repository import MongoDBRepository

def main() -> None:
    settings = get_settings()

    with MongoDBRepository(settings) as repository:
        documents = repository.read_customers()

    if not documents:
        print("No customer documents found.")
        return
    
    dataframe = pd.DataFrame(documents)

    print("\n=== Shape ===")
    print(dataframe.shape)

    print("\n=== Column ===")
    print(dataframe.columns.tolist())

    print("\n=== Data Type ===")
    print(dataframe.dtypes)

    print("\n=== Missing Values ===")
    print(dataframe.isna().sum())

    print("\n Example rows ===")
    print(dataframe.head())

    print("\n=== Geography ===")
    print(dataframe["Geography"].value_counts(dropna=False))

    print("\n=== Gender ===")
    print(dataframe["Gender"].value_counts(dropna=False))

    print("\n=== Exited ===")
    print(dataframe["Exited"].value_counts(dropna=False))

if __name__ == "__main__":
    main()