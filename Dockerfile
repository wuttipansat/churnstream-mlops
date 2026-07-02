FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

COPY requirements.txt pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps .

RUN useradd --create-home appuser
USER appuser

CMD ["sh", "-c", "uvicorn churnstream.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
