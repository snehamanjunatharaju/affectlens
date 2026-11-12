import builtins
import io
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from affectlens import api
from affectlens.inference import ExpressionModel, Prediction, decode_upload
from affectlens.monitoring import PredictionMonitor
from affectlens.rate_limit import SlidingWindowLimiter
from affectlens.settings import Settings


class FakeModel:
    conformal_quantile = 0.5
    ready = True

    def predict(self, _raw, *, face_crop=False):
        return Prediction("happy", 0.9, {"happy": 0.9, "sad": 0.1}, False, None, ("happy",))


def image_bytes():
    output = io.BytesIO()
    Image.new("RGB", (12, 8), (128, 128, 128)).save(output, format="JPEG")
    return output.getvalue()


def test_decode_applies_orientation_and_discards_exif():
    image = Image.new("RGB", (8, 12), "red")
    exif = Image.Exif()
    exif[274] = 6
    output = io.BytesIO()
    image.save(output, format="JPEG", exif=exif)

    decoded = decode_upload(output.getvalue())

    assert decoded.shape == (8, 12, 3)
    assert isinstance(decoded, np.ndarray)


def test_active_model_sidecar_matches_benchmark_resize():
    model = ExpressionModel(Path("artifacts/affectnet-enet-b2-7.onnx"), threshold=0.55)
    assert model.resize_interpolation == cv2.INTER_CUBIC


def test_analyze_request_does_not_write_any_file(monkeypatch):
    settings = Settings(None, None, 0.55, 8 * 1024 * 1024, False)
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda _cls: settings))
    monkeypatch.setattr(api, "ExpressionModel", lambda *_args, **_kwargs: FakeModel())
    payload = image_bytes()

    with TestClient(api.app) as client:
        def reject_write(*_args, **_kwargs):
            raise AssertionError("/api/analyze attempted a filesystem write")

        monkeypatch.setattr(builtins, "open", reject_write)
        monkeypatch.setattr(io, "open", reject_write)
        monkeypatch.setattr(os, "open", reject_write)
        monkeypatch.setattr(Path, "write_bytes", reject_write)
        monkeypatch.setattr(Path, "write_text", reject_write)
        monkeypatch.setattr(cv2, "imwrite", reject_write)

        response = client.post("/api/analyze", content=payload, headers={"content-type": "image/jpeg"})

    assert response.status_code == 200
    assert response.json()["retention"] == "Image processed in memory and not saved by this service."


def test_save_images_must_remain_disabled(monkeypatch):
    monkeypatch.setenv("SAVE_IMAGES", "true")
    with pytest.raises(ValueError, match="never written to disk"):
        Settings.from_env()


def test_rate_limiter_returns_retry_after():
    limiter = SlidingWindowLimiter(limit=1, window_seconds=60)
    assert limiter.check("127.0.0.1") == (True, 0)
    allowed, retry_after = limiter.check("127.0.0.1")
    assert not allowed
    assert 1 <= retry_after <= 60


def test_monitor_flags_distribution_shift_without_storing_images():
    monitor = PredictionMonitor({"happy": 1.0})
    for _ in range(50):
        monitor.record({"sad": 0.8, "happy": 0.2}, 12.0)
    snapshot = monitor.snapshot()
    assert snapshot["recent_count"] == 50
    assert snapshot["drift_alert"]
    assert snapshot["mean_latency_ms"] == 12.0
