from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError

from .labels import LABELS


class InferenceUnavailable(RuntimeError):
    pass


class InvalidImage(ValueError):
    pass


def decode_upload(raw: bytes) -> np.ndarray:
    """Decode orientation-corrected RGB pixels while discarding all source metadata."""
    try:
        with Image.open(BytesIO(raw)) as upload:
            if upload.width * upload.height > 40_000_000:
                raise InvalidImage("Image dimensions exceed the configured pixel limit.")
            return np.asarray(ImageOps.exif_transpose(upload).convert("RGB"))
    except InvalidImage:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImage("Upload must be a supported image file.") from exc


@dataclass(frozen=True)
class Prediction:
    label: str | None
    confidence: float
    probabilities: dict[str, float]
    abstained: bool
    reason: str | None = None
    prediction_set: tuple[str, ...] = ()
    base_label: str | None = None
    modality_disagreement: bool = False
    fusion_applied: bool = False


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
        self.input_channels = 1
        self.input_size = 48
        self.mean = [0.5]
        self.std = [0.5]
        self.color_order = "rgb"
        self.resize_interpolation = cv2.INTER_LINEAR
        self.temperature = 1.0
        self.fusion_session = None
        self.fusion_metadata = None
        if model_path:
            if not model_path.is_file():
                raise RuntimeError(f"Model file does not exist: {model_path}")
            self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            input_shape = self.session.get_inputs()[0].shape
            if len(input_shape) != 4:
                raise RuntimeError("Model input must be rank 4 with a single grayscale channel")
            if input_shape[1] in (1, 3):
                self.input_layout = "nchw"
                self.input_channels = input_shape[1]
            elif input_shape[-1] in (1, 3):
                self.input_layout = "nhwc"
                self.input_channels = input_shape[-1]
            else:
                raise RuntimeError("Model input must use one or three channels in NCHW or NHWC layout")
            sidecar = model_path.with_suffix(".preprocess.json")
            if sidecar.is_file():
                import json
                preprocess = json.loads(sidecar.read_text())
                self.input_size = int(preprocess.get("size", self.input_size))
                self.mean = preprocess.get("mean", self.mean)
                self.std = preprocess.get("std", self.std)
                self.color_order = preprocess.get("color_order", "rgb")
                interpolation = preprocess.get("interpolation", "linear")
                if len(self.mean) != self.input_channels or len(self.std) != self.input_channels or any(float(value) <= 0 for value in self.std):
                    raise RuntimeError(f"Invalid preprocessing parameters in {sidecar}")
                if self.color_order not in {"rgb", "bgr"}:
                    raise RuntimeError(f"Invalid color_order in {sidecar}")
                if interpolation not in {"linear", "cubic"}:
                    raise RuntimeError(f"Invalid interpolation in {sidecar}")
                self.resize_interpolation = cv2.INTER_CUBIC if interpolation == "cubic" else cv2.INTER_LINEAR
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
                self.temperature = float(calibration.get("temperature", 1.0))
                if not 0.05 <= self.temperature <= 20.0:
                    raise RuntimeError(f"Invalid temperature in {sidecar}")
            fusion_model = model_path.with_suffix(".fusion.onnx")
            fusion_sidecar = model_path.with_suffix(".fusion.json")
            if fusion_model.is_file() != fusion_sidecar.is_file():
                raise RuntimeError("Fusion model and metadata sidecar must both be present")
            if fusion_model.is_file():
                import json

                self.fusion_metadata = json.loads(fusion_sidecar.read_text())
                if self.fusion_metadata.get("labels") != list(LABELS):
                    raise RuntimeError(f"Fusion label order does not match {fusion_sidecar}")
                names = self.fusion_metadata.get("blendshape_names")
                means = self.fusion_metadata.get("feature_mean")
                stds = self.fusion_metadata.get("feature_std")
                if not isinstance(names, list) or len(names) != 52 or len(set(names)) != 52:
                    raise RuntimeError(f"Invalid blendshape names in {fusion_sidecar}")
                if not isinstance(means, list) or not isinstance(stds, list) or len(means) != 52 or len(stds) != 52 or any(float(value) <= 0 for value in stds):
                    raise RuntimeError(f"Invalid blendshape normalization in {fusion_sidecar}")
                temperature = float(self.fusion_metadata.get("temperature", 1.0))
                quantile = float(self.fusion_metadata.get("quantile", -1.0))
                if not 0.05 <= temperature <= 20.0 or not 0 <= quantile <= 1:
                    raise RuntimeError(f"Invalid fusion calibration in {fusion_sidecar}")
                self.fusion_session = ort.InferenceSession(str(fusion_model), providers=["CPUExecutionProvider"])

    @property
    def ready(self) -> bool:
        return self.session is not None

    def predict(self, raw: bytes, *, face_crop: bool = False, blendshapes: dict[str, float] | None = None) -> Prediction:
        if self.session is None:
            raise InferenceUnavailable("No model configured. Set AFFECTLENS_MODEL_PATH to a FER-compatible ONNX model.")
        image = decode_upload(raw)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if face_crop:
            crop = gray
            color_crop = image
        else:
            faces = self._detect_faces(gray)
            if len(faces) == 0:
                return Prediction(None, 0.0, {}, True, "no_face")
            if len(faces) > 1:
                return Prediction(None, 0.0, {}, True, "multiple_faces")
            x, y, width, height = map(int, faces[0])
            crop = gray[y : y + height, x : x + width]
            color_crop = image[y : y + height, x : x + width]
        input_meta = self.session.get_inputs()[0]
        shape = input_meta.shape
        channels_first = self.input_layout == "nchw"
        height_in, width_in = (self.input_size, self.input_size)
        height_axis, width_axis = (2, 3) if channels_first else (1, 2)
        if len(shape) == 4 and isinstance(shape[height_axis], int) and isinstance(shape[width_axis], int):
            height_in, width_in = shape[height_axis], shape[width_axis]
        if self.input_channels == 3:
            color_crop = cv2.cvtColor(color_crop, cv2.COLOR_RGB2BGR) if self.color_order == "bgr" else color_crop
            resized = cv2.resize(color_crop, (width_in, height_in), interpolation=self.resize_interpolation).astype(np.float32) / 255.0
            if channels_first:
                tensor = resized.transpose(2, 0, 1)[None, :, :, :]
            else:
                tensor = resized[None, :, :, :]
        else:
            resized = cv2.resize(crop, (width_in, height_in), interpolation=self.resize_interpolation).astype(np.float32) / 255.0
            tensor = resized[None, None, :, :] if channels_first else resized[None, :, :, None]
        mean = np.asarray(self.mean, dtype=np.float32).reshape((1, self.input_channels, 1, 1) if channels_first else (1, 1, 1, self.input_channels))
        std = np.asarray(self.std, dtype=np.float32).reshape((1, self.input_channels, 1, 1) if channels_first else (1, 1, 1, self.input_channels))
        tensor = (tensor - mean) / std
        logits = np.asarray(self.session.run(None, {input_meta.name: tensor})[0]).reshape(-1)
        if logits.size != len(LABELS) or not np.isfinite(logits).all():
            raise InferenceUnavailable("Model output must contain seven finite class scores.")
        # Accept logits or probabilities and normalize either form safely.
        if np.all(logits >= 0) and np.isclose(logits.sum(), 1.0, atol=1e-3):
            probabilities = logits / logits.sum()
        else:
            shifted = logits / self.temperature - (logits / self.temperature).max()
            exp = np.exp(shifted)
            probabilities = exp / exp.sum()
        base_best = int(probabilities.argmax())
        fusion_applied = False
        quantile = self.conformal_quantile
        if self.fusion_session is not None and blendshapes is not None:
            names = self.fusion_metadata["blendshape_names"]
            if all(name in blendshapes for name in names):
                raw_features = np.asarray([blendshapes[name] for name in names], dtype=np.float32)
                means = np.asarray(self.fusion_metadata["feature_mean"], dtype=np.float32)
                stds = np.asarray(self.fusion_metadata["feature_std"], dtype=np.float32)
                feature_vector = np.concatenate([((raw_features - means) / stds), np.ones(1, dtype=np.float32)])[None, :]
                fusion_inputs = self.fusion_session.get_inputs()
                fusion_logits = np.asarray(self.fusion_session.run(None, {
                    fusion_inputs[0].name: probabilities.astype(np.float32)[None, :],
                    fusion_inputs[1].name: feature_vector,
                })[0]).reshape(-1)
                if fusion_logits.size != len(LABELS) or not np.isfinite(fusion_logits).all():
                    raise InferenceUnavailable("Fusion model output must contain seven finite class scores.")
                probabilities = np.exp(fusion_logits / float(self.fusion_metadata["temperature"]) - np.max(fusion_logits / float(self.fusion_metadata["temperature"])))
                probabilities /= probabilities.sum()
                fusion_applied = True
                quantile = float(self.fusion_metadata["quantile"])
        best = int(probabilities.argmax())
        confidence = float(probabilities[best])
        if quantile is not None:
            selected = tuple(LABELS[i] for i, score in enumerate(probabilities) if 1.0 - score <= quantile)
            disagreement = fusion_applied and best != base_best
            if disagreement and LABELS[base_best] not in selected:
                selected = (*selected, LABELS[base_best])
            abstained = len(selected) != 1
            reason = "modality_disagreement" if disagreement else "empty_prediction_set" if not selected else "ambiguous_prediction_set" if len(selected) > 1 else None
        else:
            abstained = confidence < self.threshold
            selected = () if abstained else (LABELS[best],)
            disagreement = fusion_applied and best != base_best
            reason = "modality_disagreement" if disagreement else "low_confidence" if abstained else None
            if disagreement:
                abstained = True
                selected = tuple(dict.fromkeys((*selected, LABELS[base_best])))
        return Prediction(
            None if abstained else LABELS[best],
            confidence,
            {label: float(probabilities[i]) for i, label in enumerate(LABELS)},
            abstained,
            reason,
            selected,
            LABELS[base_best] if fusion_applied else None,
            disagreement,
            fusion_applied,
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
