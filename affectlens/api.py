import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .inference import ExpressionModel, InferenceUnavailable, InvalidImage
from .monitoring import PredictionMonitor
from .rate_limit import SlidingWindowLimiter
from .settings import Settings

STATIC_DIR = Path(__file__).parent / "static"


class HealthResponse(BaseModel):
    status: Literal["ready", "needs_model"]
    model_loaded: bool
    uncertainty_mode: Literal["unavailable", "split_conformal", "score_threshold"]
    fusion_loaded: bool


class AnalysisResponse(BaseModel):
    label: str | None
    confidence: float
    probabilities: dict[str, float]
    abstained: bool
    reason: str | None = None
    prediction_set: list[str]
    base_label: str | None = None
    modality_disagreement: bool = False
    fusion_applied: bool = False
    uncertainty_mode: Literal["split_conformal", "score_threshold"]
    disclaimer: str
    retention: str
    uncertainty_scope: str


class MonitoringResponse(BaseModel):
    total_count: int
    recent_count: int
    mean_latency_ms: float | None
    p95_latency_ms: float | None
    confidence_histogram: list[int]
    predicted_class_distribution: dict[str, float]
    prediction_drift_psi: float | None
    drift_threshold: float | None = None
    drift_alert: bool
    note: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    app.state.settings = settings
    app.state.model = ExpressionModel(settings.model_path, settings.confidence_threshold, settings.face_cascade_path)
    app.state.rate_limiter = SlidingWindowLimiter(settings.rate_limit_per_minute)
    baseline = {}
    if settings.model_path:
        baseline_path = settings.model_path.with_suffix(".monitoring.json")
        if baseline_path.is_file():
            baseline = json.loads(baseline_path.read_text()).get("predicted_class_distribution", {})
    app.state.monitor = PredictionMonitor(baseline)
    yield


app = FastAPI(
    title="AffectLens",
    description="Local-first facial expression research demo. Predictions are not measurements of inner emotional state.",
    version=__version__,
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health", response_model=HealthResponse)
def health(request: Request):
    model = request.app.state.model
    uncertainty_mode = "unavailable" if not model.ready else (
        "split_conformal" if model.conformal_quantile is not None else "score_threshold"
    )
    return {
        "status": "ready" if model.ready else "needs_model",
        "model_loaded": model.ready,
        "uncertainty_mode": uncertainty_mode,
        "fusion_loaded": model.fusion_session is not None,
    }


@app.get("/api/metrics", response_model=MonitoringResponse)
def metrics(request: Request):
    """Return process-local aggregates; no image or per-request record is retained."""
    return request.app.state.monitor.snapshot()


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze(request: Request):
    started_at = time.perf_counter()
    settings = request.app.state.settings
    if settings.save_images:
        raise HTTPException(status_code=503, detail="Image persistence is disabled by this build.")
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type not in {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}:
        raise HTTPException(status_code=415, detail="Use a JPEG, PNG, WebP, HEIC, or HEIF image.")
    client_key = request.client.host if request.client else "unknown"
    allowed, retry_after = request.app.state.rate_limiter.check(client_key)
    if not allowed:
        raise HTTPException(status_code=429, detail="Too many analysis requests; try again shortly.", headers={"Retry-After": str(retry_after)})
    face_crop = request.headers.get("x-affectlens-face-crop") == "mediapipe-landmarks"
    blendshape_header = request.headers.get("x-affectlens-blendshapes")
    blendshapes = None
    if blendshape_header:
        try:
            parsed = json.loads(blendshape_header)
            if not isinstance(parsed, dict) or len(parsed) > 64 or any(
                not isinstance(name, str) or not isinstance(value, (int, float)) or not 0 <= value <= 1
                for name, value in parsed.items()
            ):
                raise ValueError("invalid blendshape values")
            blendshapes = parsed
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Blendshape input must be a JSON object with scores from 0 to 1.") from exc
    payload = bytearray()
    async for chunk in request.stream():
        payload.extend(chunk)
        if len(payload) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="Image exceeds the configured upload limit.")
    try:
        prediction_args = {"face_crop": face_crop}
        if blendshapes is not None:
            prediction_args["blendshapes"] = blendshapes
        prediction = request.app.state.model.predict(bytes(payload), **prediction_args)
    except InferenceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InvalidImage as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    request.app.state.monitor.record(prediction.probabilities, (time.perf_counter() - started_at) * 1000)
    return {
        "label": prediction.label,
        "confidence": prediction.confidence,
        "probabilities": prediction.probabilities,
        "abstained": prediction.abstained,
        "reason": prediction.reason,
        "prediction_set": prediction.prediction_set,
        "base_label": prediction.base_label,
        "modality_disagreement": prediction.modality_disagreement,
        "fusion_applied": prediction.fusion_applied,
        "uncertainty_mode": "split_conformal" if request.app.state.model.conformal_quantile is not None else "score_threshold",
        "disclaimer": "A visible expression is not proof of a person's inner emotional state.",
        "retention": "Image processed in memory and not saved by this service.",
        "uncertainty_scope": "Conformal coverage applies only when inference examples match the calibration data and preprocessing distribution; it is not guaranteed for out-of-domain webcam images.",
    }
