from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from .labels import LABELS


class InferenceUnavailable(RuntimeError):
    pass


class InvalidImage(ValueError):
    pass


@dataclass(frozen=True)
class Prediction:
    label: str | None
    confidence: float
    probabilities: dict[str, float]
    abstained: bool
    reason: str | None = None
    prediction_set: tuple[str, ...] = ()


class ExpressionModel:
    """Runs a seven-class FER-compatible ONNX model on one detected face."""

    def __init__(self, model_path: Path | None, threshold: float, cascade_path: Path | None = None):
        self.threshold = threshold
        bundled_cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        selected_cascade = cascade_path or bundled_cascade
        detector_paths = [selected_cascade]
        if cascade_path is None:
            detector_paths.extend([
                Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt2.xml",
                Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt.xml",
            ])
        self.face_detectors = [cv2.CascadeClassifier(str(path)) for path in dict.fromkeys(detector_paths)]
        self.face_detectors = [detector for detector in self.face_detectors if not detector.empty()]
        if not self.face_detectors:
            raise RuntimeError(f"Could not load face detector: {selected_cascade}")
        self.session = None
        self.conformal_quantile = None
        self.input_layout = None
        if model_path:
            if not model_path.is_file():
                raise RuntimeError(f"Model file does not exist: {model_path}")
            self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            input_shape = self.session.get_inputs()[0].shape
            if len(input_shape) != 4:
                raise RuntimeError("Model input must be rank 4 with a single grayscale channel")
            if input_shape[1] == 1:
                self.input_layout = "nchw"
            elif input_shape[-1] == 1:
                self.input_layout = "nhwc"
            else:
                raise RuntimeError("Model input must use a single grayscale channel in NCHW or NHWC layout")
            sidecar = model_path.with_suffix(".uncertainty.json")
            if sidecar.is_file():
                import json

                calibration = json.loads(sidecar.read_text())
                if calibration.get("labels") != list(LABELS):
                    raise RuntimeError(f"Calibration label order does not match {sidecar}")
                quantile = calibration.get("quantile")
                if not isinstance(quantile, (int, float)) or not 0 <= quantile <= 1:
                    raise RuntimeError(f"Invalid conformal quantile in {sidecar}")
                self.conformal_quantile = float(quantile)

    @property
    def ready(self) -> bool:
        return self.session is not None

    def predict(self, raw: bytes, *, face_crop: bool = False) -> Prediction:
        if self.session is None:
            raise InferenceUnavailable("No model configured. Set AFFECTLENS_MODEL_PATH to a FER-compatible ONNX model.")
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidImage("Upload must be a supported image file.")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if face_crop:
            crop = gray
        else:
            faces = self._detect_faces(gray)
            if len(faces) == 0:
                return Prediction(None, 0.0, {}, True, "no_face")
            if len(faces) > 1:
                return Prediction(None, 0.0, {}, True, "multiple_faces")
            x, y, width, height = map(int, faces[0])
            crop = gray[y : y + height, x : x + width]
        input_meta = self.session.get_inputs()[0]
        shape = input_meta.shape
        channels_first = self.input_layout == "nchw"
        height_in, width_in = (48, 48)
        height_axis, width_axis = (2, 3) if channels_first else (1, 2)
        if len(shape) == 4 and isinstance(shape[height_axis], int) and isinstance(shape[width_axis], int):
            height_in, width_in = shape[height_axis], shape[width_axis]
        crop = cv2.resize(crop, (width_in, height_in)).astype(np.float32) / 255.0
        crop = (crop - 0.5) / 0.5
        if channels_first:
            tensor = crop[None, None, :, :]
        else:
            tensor = crop[None, :, :, None]
        logits = np.asarray(self.session.run(None, {input_meta.name: tensor})[0]).reshape(-1)
        if logits.size != len(LABELS) or not np.isfinite(logits).all():
            raise InferenceUnavailable("Model output must contain seven finite class scores.")
        # Accept logits or probabilities and normalize either form safely.
        if np.all(logits >= 0) and np.isclose(logits.sum(), 1.0, atol=1e-3):
            probabilities = logits / logits.sum()
        else:
            shifted = logits - logits.max()
            exp = np.exp(shifted)
            probabilities = exp / exp.sum()
        best = int(probabilities.argmax())
        confidence = float(probabilities[best])
        if self.conformal_quantile is not None:
            selected = tuple(LABELS[i] for i, score in enumerate(probabilities) if 1.0 - score <= self.conformal_quantile)
            abstained = len(selected) != 1
            reason = "empty_prediction_set" if not selected else "ambiguous_prediction_set" if len(selected) > 1 else None
        else:
            abstained = confidence < self.threshold
            selected = () if abstained else (LABELS[best],)
            reason = "low_confidence" if abstained else None
        return Prediction(
            None if abstained else LABELS[best],
            confidence,
            {label: float(probabilities[i]) for i, label in enumerate(LABELS)},
            abstained,
            reason,
            selected,
        )

    def _detect_faces(self, gray: np.ndarray) -> np.ndarray:
        """Retry with alternate frontal cascades and contrast-normalized input for close crops."""
        equalized = cv2.equalizeHist(gray)
        variants = [(gray, 1.0), (equalized, 1.0)]
        if max(gray.shape) < 1200:
            enlarged = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            variants.append((cv2.equalizeHist(enlarged), 1.5))
        for detector in self.face_detectors:
            for image, scale in variants:
                found = detector.detectMultiScale(
                    image,
                    scaleFactor=1.05,
                    minNeighbors=4,
                    minSize=(32, 32),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )
                if len(found):
                    mapped = np.asarray(found, dtype=np.float32) / scale
                    mapped = np.rint(mapped).astype(np.int32)
                    mapped[:, 0] = np.clip(mapped[:, 0], 0, gray.shape[1] - 1)
                    mapped[:, 1] = np.clip(mapped[:, 1], 0, gray.shape[0] - 1)
                    mapped[:, 2] = np.minimum(mapped[:, 2], gray.shape[1] - mapped[:, 0])
                    mapped[:, 3] = np.minimum(mapped[:, 3], gray.shape[0] - mapped[:, 1])
                    return mapped
        return np.empty((0, 4), dtype=np.int32)
