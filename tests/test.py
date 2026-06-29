from churnstream.core.config  import get_settings
from churnstream.ingestion.mongodb_repository import (
    MongoDBRepository,
)

import pandas as pd

settings = get_settings()

repository = MongoDBRepository(settings)

repository.ping()

customers = repository.read_customers()

df = pd.DataFrame(customers)

print(f"Number of customers: {len(df)}")
print(df.head(10))

repository.close()