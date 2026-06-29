from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from pymongo.database import Database
from pymongo.collection import Collection

from churnstream.core.config import Settings

logger = logging.getLogger(__name__)

class MongoDBRepository:
    def __init__(self, settings: Settings) -> None:
        self._client: MongoClient[dict[str, Any]] = MongoClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=5_000,
        )

        self._database: Database[dict[str, Any]] = self._client[
            settings.mongodb_database
        ]

        self.customers: Collection[dict[str, Any]] = self._database[
            settings.mongodb_customers_collection
        ]

        self.predictions: Collection[dict[str, Any]] = self._database[
            settings.mongodb_predictions_collection
        ]

    def ping(self) -> None:
        self._client.admin.command("ping")
        logger.info("MongoDB connection successfully")

    def read_customers(
            self,
            query: dict[str, Any] | None = None,
            limit: int | None = None,
    ) -> list[dict[str, Any]]:
        cursor = self.customers.find(
            query or {},
            {"_id": 0}
        )

        if limit is not None:
            cursor = cursor.limit(limit)

        return list(cursor)
    
    def __enter__(self) -> "MongoDBRepository":
        return self
    
    def __exit__(self, *_: object) -> None:
        self.close()
    
    def close(self) -> None:
        self._client.close()