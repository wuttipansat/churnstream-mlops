from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request

from churnstream.api.model_service import ModelService
from churnstream.api.schemas import HealthResponse, PredictionRequest, PredictionResponse
from churnstream.core.config import get_settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    model_service = ModelService.load(settings)

    app.state.settings = settings
    app.state.model_service = model_service

    yield

app = FastAPI(
    title="ChurnStream Prediction API",
    version="0.1.0",
    lifespan=lifespan,
)

@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    service: ModelService = request.app.state.model_service

    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=service.model_name,
        model_version = service.model_version,
        model_alias=service.model_alias,
    )

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    service: ModelService = request.app.state.model_service

    model_input = payload.model_dump(
        exclude={"customer_id"},
    )

    result=service.predict(model_input)

    return PredictionResponse(
        request_id = str(uuid4()),
        customer_id=payload.customer_id,
        churn_prediction=result.prediction,
        churn_probability=result.probability,
        model_name=service.model_name,
        model_version=service.model_version,
        model_alias=service.model_alias,
        threshold=service.threshold,
    )