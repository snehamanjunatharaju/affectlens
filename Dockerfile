FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SAVE_IMAGES=false \
    AFFECTLENS_MODEL_PATH=/models/affectnet-enet-b2-7.onnx

WORKDIR /app
COPY pyproject.toml README.md LICENSE-APACHE-2.0 THIRD_PARTY_NOTICES.md ./
COPY affectlens ./affectlens
RUN mkdir -p /models
COPY artifacts/affectnet-enet-b2-7.onnx artifacts/affectnet-enet-b2-7.preprocess.json artifacts/affectnet-enet-b2-7.uncertainty.json /models/
RUN pip install --no-cache-dir .
RUN useradd --system --uid 10001 --no-create-home app

EXPOSE 8000
USER 10001:10001
CMD ["sh", "-c", "uvicorn affectlens.api:app --host 0.0.0.0 --port ${PORT:-10000}"]
