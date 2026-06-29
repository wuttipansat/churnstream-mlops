import os

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

load_dotenv()

mongodb_uri = os.getenv("MONGODB_URI")
database_name = os.getenv("MONGODB_DATABASE")

client: MongoClient | None = None

try:
    client = MongoClient(
        mongodb_uri,
        serverSelectionTimeoutMS=500
    )

    client.admin.command("ping")

    database = client[database_name]
    collection = database["customers"]

    documents = list(
        collection.find(
            {},
            {"_id": 0},
        ).limit(100)
    )

    dataframe = pd.DataFrame(documents)
    print("Successfully connect to MongoDB")
    print(f"Data count: {len(dataframe)}")
    print(dataframe.head(10))

except PyMongoError as error:
    print(f"Failed to connect to MongoDB: {error}")