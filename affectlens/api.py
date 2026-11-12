from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .inference import ExpressionModel, InferenceUnavailable, InvalidImage
from .settings import Settings

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    app.state.settings = settings
    app.state.model = ExpressionModel(settings.model_path, settings.confidence_threshold, settings.face_cascade_path)
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


@app.get("/api/health")
def health(request: Request):
    model = request.app.state.model
    uncertainty_mode = "unavailable" if not model.ready else (
        "split_conformal" if model.conformal_quantile is not None else "score_threshold"
    )
    return {
        "status": "ready" if model.ready else "needs_model",
        "model_loaded": model.ready,
        "uncertainty_mode": uncertainty_mode,
    }


@app.post("/api/analyze")
async def analyze(request: Request):
    settings = request.app.state.settings
    if settings.save_images:
        raise HTTPException(status_code=503, detail="Image persistence is disabled by this build.")
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Use a JPEG, PNG, or WebP image.")
    face_crop = request.headers.get("x-affectlens-face-crop") == "mediapipe-landmarks"
    payload = bytearray()
    async for chunk in request.stream():
        payload.extend(chunk)
        if len(payload) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="Image exceeds the configured upload limit.")
    try:
        prediction = request.app.state.model.predict(bytes(payload), face_crop=face_crop)
    except InferenceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InvalidImage as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "label": prediction.label,
        "confidence": prediction.confidence,
        "probabilities": prediction.probabilities,
        "abstained": prediction.abstained,
        "reason": prediction.reason,
        "prediction_set": prediction.prediction_set,
        "uncertainty_mode": "split_conformal" if request.app.state.model.conformal_quantile is not None else "score_threshold",
        "disclaimer": "A visible expression is not proof of a person's inner emotional state.",
        "retention": "Image processed in memory and not saved by this service.",
        "uncertainty_scope": "Conformal coverage applies only when inference examples match the calibration data and preprocessing distribution; it is not guaranteed for out-of-domain webcam images.",
    }
