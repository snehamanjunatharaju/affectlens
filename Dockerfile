FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SAVE_IMAGES=false \
    AFFECTLENS_MODEL_PATH=/models/fer-ferplus.onnx

WORKDIR /app
COPY pyproject.toml README.md LICENSE-APACHE-2.0 THIRD_PARTY_NOTICES.md ./
COPY affectlens ./affectlens
RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "affectlens.api:app", "--host", "0.0.0.0", "--port", "8000"]
