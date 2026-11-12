"""Measure classifier quality and selective coverage under controlled image shifts."""

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .labels import ALIASES, IMAGE_EXTENSIONS, LABELS


def corruptions(image: np.ndarray) -> dict[str, np.ndarray]:
    height, width = image.shape
    float_image = image.astype(np.float32)
    dim = np.clip(float_image * 0.8 - 35, 0, 255).astype(np.uint8)
    low_contrast = np.clip((float_image - 127.5) * 0.55 + 127.5, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(image, (5, 5), sigmaX=1.2)
    occluded = image.copy()
    occluded[: max(1, height // 4), : max(1, width // 4)] = 0
    return {
        "clean": image,
        "dim_lighting": dim,
        "low_contrast": low_contrast,
        "gaussian_blur": blur,
        "corner_occlusion": occluded,
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    if np.all(logits >= 0) and np.allclose(logits.sum(axis=1), 1.0, atol=1e-3):
        return logits / logits.sum(axis=1, keepdims=True)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def summarize(labels: np.ndarray, probabilities: np.ndarray, quantile: float) -> dict:
    predictions = probabilities.argmax(axis=1)
    classes = []
    f1s = []
    for index, label in enumerate(LABELS):
        mask = labels == index
        tp = int(np.sum(mask & (predictions == index)))
        fp = int(np.sum(~mask & (predictions == index)))
        fn = int(np.sum(mask & (predictions != index)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        classes.append({"label": label, "precision": precision, "recall": recall, "f1": f1, "support": int(mask.sum())})
        f1s.append(f1)
    sets = probabilities >= (1.0 - quantile)
    sizes = sets.sum(axis=1)
    singleton = sizes == 1
    return {
        "samples": int(len(labels)),
        "accuracy": float(np.mean(predictions == labels)),
        "macro_f1": float(np.mean(f1s)),
        "classes": classes,
        "conformal_marginal_coverage": float(sets[np.arange(len(labels)), labels].mean()),
        "singleton_coverage": float(singleton.mean()),
        "singleton_accuracy": float(np.mean(predictions[singleton] == labels[singleton])) if singleton.any() else None,
        "mean_prediction_set_size": float(sizes.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Dataset root with test/<class>/ folders")
    parser.add_argument("--model", type=Path, required=True, help="ONNX model exported by AffectLens")
    parser.add_argument("--output", type=Path, default=Path("reports/robustness.json"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    sidecar = args.model.with_suffix(".uncertainty.json")
    if not sidecar.is_file():
        parser.error(f"Missing conformal calibration sidecar: {sidecar}")
    calibration = json.loads(sidecar.read_text())
    if calibration.get("labels") != list(LABELS) or not 0 <= calibration.get("quantile", -1) <= 1:
        parser.error("Calibration sidecar label order or quantile is invalid")
    test_root = args.data / "test"
    if not test_root.is_dir():
        parser.error(f"Missing test directory: {test_root}")

    labels = []
    variants: dict[str, list[np.ndarray]] = {}
    counts = Counter()
    for folder in sorted(test_root.iterdir()):
        canonical = ALIASES.get(folder.name.lower()) if folder.is_dir() else None
        if canonical is None:
            continue
        label_index = LABELS.index(canonical)
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            image = cv2.resize(image, (48, 48))
            labels.append(label_index)
            for name, variant in corruptions(image).items():
                variants.setdefault(name, []).append(variant)
            counts[canonical] += 1
    missing = [label for label in LABELS if counts[label] == 0]
    if missing:
        parser.error(f"Test split is missing classes: {missing}")
    if not labels:
        parser.error("No readable test images found")

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    results = {}
    labels_array = np.asarray(labels, dtype=np.int64)
    for name, images in variants.items():
        tensors = (np.asarray(images, dtype=np.float32)[:, None, :, :] / 127.5) - 1.0
        chunks = []
        for start in range(0, len(tensors), args.batch_size):
            scores = session.run(None, {input_meta.name: tensors[start : start + args.batch_size]})[0]
            chunks.append(softmax(np.asarray(scores).reshape(-1, len(LABELS))))
        probabilities = np.concatenate(chunks, axis=0)
        results[name] = summarize(labels_array, probabilities, float(calibration["quantile"]))

    report = {
        "model": args.model.name,
        "calibration_method": calibration.get("method"),
        "calibration_alpha": calibration.get("alpha"),
        "conformal_quantile": calibration["quantile"],
        "dataset_split": "test",
        "results": results,
        "limitations": [
            "This evaluates the pre-cropped face classifier, not the face detector or full upload pipeline.",
            "Synthetic corruptions are controlled probes, not substitutes for real-world domain evaluation.",
            "Conformal coverage relies on calibration/test exchangeability; the corrupted sets are intentionally shifted.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Robustness report written to {args.output}")


if __name__ == "__main__":
    main()
