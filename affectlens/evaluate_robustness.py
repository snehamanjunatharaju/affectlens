"""Measure classifier quality and selective coverage under controlled image shifts."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from .evaluate import expected_calibration_error
from .inference import ExpressionModel
from .labels import ALIASES, IMAGE_EXTENSIONS, LABELS


def corruptions(image: np.ndarray) -> dict[str, np.ndarray]:
    height, width = image.shape
    float_image = image.astype(np.float32)
    dim = np.clip(float_image * 0.5, 0, 255).astype(np.uint8)
    bright = np.clip(float_image * 1.5 + 25, 0, 255).astype(np.uint8)
    low_contrast = np.clip((float_image - 127.5) * 0.55 + 127.5, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(image, (5, 5), sigmaX=1.2)
    noise = np.random.default_rng(17).normal(0, 22, image.shape)
    noisy = np.clip(float_image + noise, 0, 255).astype(np.uint8)
    occluded = image.copy()
    # A dark band across the eye region acts as a deterministic sunglasses proxy.
    cv2.rectangle(occluded, (0, height // 3), (width - 1, height // 2), 0, thickness=-1)
    result = {
        "clean": image,
        "dim_lighting": dim,
        "high_brightness": bright,
        "low_contrast": low_contrast,
        "gaussian_blur": blur,
        "gaussian_noise": noisy,
        "eye_band_occlusion": occluded,
    }
    for degrees in (15, 30, 45):
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
        result[f"rotation_{degrees}deg"] = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=127)
    return result


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
    ece, reliability = expected_calibration_error(labels, probabilities)
    return {
        "samples": int(len(labels)),
        "accuracy": float(np.mean(predictions == labels)),
        "macro_f1": float(np.mean(f1s)),
        "ece": ece,
        "reliability_bins": reliability,
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
    parser.add_argument("--limit", type=int, help="Optional deterministic cap for quick local runs")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    model = ExpressionModel(args.model, 0.55)
    quantile = model.conformal_quantile
    if quantile is None:
        parser.error(f"Missing conformal calibration sidecar for {args.model}")
    test_root = args.data / "test"
    if not test_root.is_dir():
        parser.error(f"Missing test directory: {test_root}")

    labels = []
    variants: dict[str, list[np.ndarray]] = {}
    counts = {label: 0 for label in LABELS}
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
            if args.limit is not None and len(labels) >= args.limit:
                break
            labels.append(label_index)
            for name, variant in corruptions(image).items():
                variants.setdefault(name, []).append(variant)
            counts[canonical] += 1
        if args.limit is not None and len(labels) >= args.limit:
            break
    missing = [label for label in LABELS if counts[label] == 0]
    if missing:
        parser.error(f"Test split is missing classes: {missing}")
    if not labels:
        parser.error("No readable test images found")

    results = {}
    labels_array = np.asarray(labels, dtype=np.int64)
    for name, images in variants.items():
        probabilities = []
        for image in images:
            success, encoded = cv2.imencode(".png", image)
            if not success:
                raise RuntimeError("Could not encode synthetic robustness image")
            prediction = model.predict(encoded.tobytes(), face_crop=True)
            probabilities.append([prediction.probabilities[label] for label in LABELS])
        results[name] = summarize(labels_array, np.asarray(probabilities), float(quantile))

    report = {
        "model": args.model.name,
        "calibration_method": "temperature scaling + split conformal" if model.conformal_quantile is not None else None,
        "calibration_alpha": json.loads(args.model.with_suffix(".uncertainty.json").read_text()).get("alpha"),
        "conformal_quantile": quantile,
        "dataset_split": "test",
        "results": results,
        "limitations": [
            "This evaluates the pre-cropped face classifier, not the face detector or full upload pipeline.",
            "Noise, occlusion, and pose changes are synthetic controlled probes, not substitutes for real-world evaluation.",
            "Conformal coverage relies on calibration/test exchangeability; the corrupted sets are intentionally shifted.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    chart_path = Path("docs/robustness.png")
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    names = list(results)
    accuracy = [results[name]["accuracy"] for name in names]
    clean = results.get("clean", {}).get("accuracy", 0.0)
    figure, axis = plt.subplots(figsize=(11, 5))
    bars = axis.bar(names, accuracy, color=["#56708a" if name == "clean" else "#a8cf51" for name in names])
    axis.set_ylabel("Accuracy")
    axis.set_title(f"Accuracy by controlled image condition (clean={clean:.1%})")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=30)
    axis.bar_label(bars, fmt="%.1f%%", padding=3)
    figure.tight_layout()
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)
    print(f"Robustness report written to {args.output}")


if __name__ == "__main__":
    main()
