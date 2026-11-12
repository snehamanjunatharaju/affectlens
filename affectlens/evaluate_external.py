"""Evaluate an external seven-class ONNX model on browser-aligned FER-style images."""

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .inference import ExpressionModel
from .labels import LABELS
from .train_fusion import image_rows, media_pipe_features


def preprocess(crops: list[bytes]) -> np.ndarray:
    images = []
    for raw in crops:
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode aligned face crop")
        image = cv2.resize(image, (260, 260), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        image -= np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
        image /= np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
        images.append(image.transpose(2, 0, 1))
    return np.asarray(images, dtype=np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def expected_calibration_error(expected: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    ece = 0.0
    for lower, upper in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        selected = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if selected.any():
            ece += float(selected.mean()) * abs(float((predicted[selected] == expected[selected]).mean()) - float(confidence[selected].mean()))
    return ece


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path)
    parser.add_argument("--task-model", type=Path, default=Path("affectlens/static/vendor/mediapipe/face_landmarker.task"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-calibration-sidecar", action="store_true", help="fit temperature and conformal threshold from validation split only")
    parser.add_argument("--limit-per-class", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.write_calibration_sidecar and args.split != "validation":
        parser.error("Calibration sidecars may only be fit on the validation split")
    rows = image_rows(args.data / args.split)
    if args.limit_per_class is not None:
        counts = {label: 0 for label in range(len(LABELS))}
        selected = []
        for row in rows:
            if counts[row[1]] < args.limit_per_class:
                selected.append(row)
                counts[row[1]] += 1
        rows = selected
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.task_model), delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        output_face_blendshapes=True,
        num_faces=1,
    )
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    baseline = ExpressionModel(args.baseline_model, threshold=0.0) if args.baseline_model else None
    input_name = session.get_inputs()[0].name
    probabilities, baseline_predictions, expected, mesh_found = [], [], [], 0
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            crops = []
            for path, label in batch:
                _features, _names, crop = media_pipe_features(path, landmarker, None)
                if crop is not None:
                    mesh_found += 1
                raw = crop or path.read_bytes()
                crops.append(raw)
                if baseline is not None:
                    baseline_predictions.append(baseline.predict(raw, face_crop=True).probabilities)
                expected.append(label)
            logits = np.asarray(session.run(None, {input_name: preprocess(crops)})[0])
            probabilities.extend(softmax(logits).tolist())
            print(f"{args.split}: evaluated {min(start + len(batch), len(rows))}/{len(rows)}")
    expected = np.asarray(expected)
    probabilities = np.asarray(probabilities)
    predicted = probabilities.argmax(axis=1)
    report = {
        "model": str(args.model),
        "split": args.split,
        "sample_count": len(expected),
        "mesh_aligned_fraction": float(mesh_found / len(expected)),
        "accuracy": float((predicted == expected).mean()),
        "macro_f1": float(f1_score(expected, predicted, labels=list(range(len(LABELS))), average="macro", zero_division=0)),
        "classes": classification_report(expected, predicted, labels=list(range(len(LABELS))), target_names=list(LABELS), output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=list(range(len(LABELS)))).tolist(),
        "label_order": list(LABELS),
        "preprocessing": "MediaPipe eye-aligned crop where detected; otherwise original image. BGR, 260x260, ImageNet mean/std per EmotiEffNet B2 reference implementation.",
    }
    uncertainty_sidecar = args.model.with_suffix(".uncertainty.json")
    if uncertainty_sidecar.is_file():
        calibration = json.loads(uncertainty_sidecar.read_text())
        temperature = float(calibration["temperature"])
        quantile = float(calibration["quantile"])
        calibrated = softmax(np.log(np.clip(probabilities, 1e-8, 1.0)) / temperature)
        prediction_sets = calibrated >= (1.0 - quantile)
        singleton = prediction_sets.sum(axis=1) == 1
        report["calibrated_uncertainty"] = {
            "temperature": temperature,
            "conformal_quantile": quantile,
            "ece": expected_calibration_error(expected, calibrated),
            "coverage": float(prediction_sets[np.arange(len(expected)), expected].mean()),
            "mean_set_size": float(prediction_sets.sum(axis=1).mean()),
            "singleton_coverage": float(singleton.mean()),
            "singleton_accuracy": float((prediction_sets.argmax(axis=1)[singleton] == expected[singleton]).mean()) if singleton.any() else None,
        }
    if baseline is not None:
        baseline_predicted = np.asarray([
            max(probability, key=probability.get, default=LABELS[0]) for probability in baseline_predictions
        ])
        baseline_expected = np.asarray([LABELS[label] for label in expected])
        report["baseline"] = {
            "model": str(args.baseline_model),
            "preprocessing": "Same MediaPipe-aligned JPEG crop as the external model; current model's own input preprocessing.",
            "accuracy": float((baseline_predicted == baseline_expected).mean()),
            "macro_f1": float(f1_score(baseline_expected, baseline_predicted, labels=list(LABELS), average="macro", zero_division=0)),
            "classes": classification_report(baseline_expected, baseline_predicted, labels=list(LABELS), output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(baseline_expected, baseline_predicted, labels=list(LABELS)).tolist(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.write_calibration_sidecar:
        from scipy.optimize import minimize_scalar

        log_probabilities = np.log(np.clip(probabilities, 1e-8, 1.0))
        def nll(temperature):
            scaled = log_probabilities / temperature
            scaled -= scaled.max(axis=1, keepdims=True)
            calibrated = np.exp(scaled)
            calibrated /= calibrated.sum(axis=1, keepdims=True)
            return float(-np.log(np.clip(calibrated[np.arange(len(expected)), expected], 1e-12, 1.0)).mean())
        fit = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
        temperature = float(fit.x)
        scaled = log_probabilities / temperature
        calibrated = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        calibrated /= calibrated.sum(axis=1, keepdims=True)
        nonconformity = np.sort(1.0 - calibrated[np.arange(len(expected)), expected])
        rank = min(len(nonconformity) - 1, int(np.ceil((len(nonconformity) + 1) * 0.90)) - 1)
        sidecar = args.model.with_suffix(".uncertainty.json")
        sidecar.write_text(json.dumps({
            "method": "split_conformal_classification",
            "alpha": 0.1,
            "quantile": float(nonconformity[rank]),
            "labels": list(LABELS),
            "training_labels": "FER+ validation labels for calibration only",
            "temperature": temperature,
            "calibration_count": int(len(expected)),
            "calibration_nll": nll(temperature),
            "calibration_coverage": float((calibrated[np.arange(len(expected)), expected] >= 1.0 - nonconformity[rank]).mean()),
            "calibration_mean_set_size": float((calibrated >= 1.0 - nonconformity[rank]).sum(axis=1).mean()),
            "note": "Calibration fitted on FER+ validation with browser-aligned preprocessing; coverage does not transfer automatically to other domains.",
        }, indent=2) + "\n")
        args.model.with_suffix(".preprocess.json").write_text(json.dumps({
            "size": 260,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "color_order": "bgr",
        }, indent=2) + "\n")
    summary = {key: report[key] for key in ("sample_count", "mesh_aligned_fraction", "accuracy", "macro_f1")}
    if "baseline" in report:
        summary["baseline"] = {key: report["baseline"][key] for key in ("accuracy", "macro_f1")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
