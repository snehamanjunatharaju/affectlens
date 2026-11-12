"""Evaluate an exported ONNX model on a labeled FER-style folder split."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .inference import ExpressionModel
from .labels import ALIASES, IMAGE_EXTENSIONS, LABELS


def expected_calibration_error(labels, probabilities, bins=15):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    rows, ece = [], 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        mask = (confidence >= lower) & (confidence < upper if index < bins - 1 else confidence <= upper)
        count = int(mask.sum())
        accuracy = float((predicted[mask] == labels[mask]).mean()) if count else None
        average_confidence = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(labels) * abs(accuracy - average_confidence)
        rows.append({"lower": lower, "upper": upper, "count": count, "accuracy": accuracy, "confidence": average_confidence})
    return float(ece), rows


def evaluate(model_path: Path, split: Path):
    model = ExpressionModel(model_path, 0.55)
    samples = []
    for folder in sorted(split.iterdir()):
        if not folder.is_dir() or folder.name.casefold() not in ALIASES:
            continue
        label = LABELS.index(ALIASES[folder.name.casefold()])
        samples.extend((path, label) for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not samples:
        raise ValueError(f"No labeled images found in {split}")
    expected, probabilities, sets = [], [], []
    for path, label in samples:
        raw = path.read_bytes()
        prediction = model.predict(raw, face_crop=True)
        expected.append(label)
        probabilities.append([prediction.probabilities.get(name, 0.0) for name in LABELS])
        sets.append(prediction.prediction_set)
    expected = np.asarray(expected)
    probabilities = np.asarray(probabilities)
    predicted = probabilities.argmax(axis=1)
    matrix = confusion_matrix(expected, predicted, labels=list(range(len(LABELS))))
    ece, reliability = expected_calibration_error(expected, probabilities)
    conformal = None
    if model.conformal_quantile is not None:
        covered = np.asarray([LABELS[label] in selected for label, selected in zip(expected, sets)])
        conformal = {
            "target_coverage": 1.0 - json.loads(model_path.with_suffix(".uncertainty.json").read_text()).get("alpha", 0.1),
            "marginal_coverage": float(covered.mean()),
            "mean_set_size": float(np.mean([len(selected) for selected in sets])),
            "singleton_rate": float(np.mean([len(selected) == 1 for selected in sets])),
        }
    order = np.argsort(-probabilities.max(axis=1))
    risk_coverage = []
    for coverage in np.linspace(0.1, 1.0, 10):
        count = max(1, int(round(len(order) * coverage)))
        selected = order[:count]
        risk_coverage.append({"coverage": float(count / len(order)), "risk": float((predicted[selected] != expected[selected]).mean())})
    return {
        "model": str(model_path),
        "split": str(split),
        "sample_count": len(samples),
        "accuracy": float((predicted == expected).mean()),
        "macro_f1": float(f1_score(expected, predicted, labels=list(range(len(LABELS))), average="macro", zero_division=0)),
        "classes": classification_report(expected, predicted, labels=list(range(len(LABELS))), target_names=list(LABELS), output_dict=True, zero_division=0),
        "confusion_matrix": matrix.tolist(),
        "ece": ece,
        "reliability_bins": reliability,
        "conformal": conformal,
        "risk_coverage": risk_coverage,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="Folder containing labeled class subfolders")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.model, args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ("sample_count", "accuracy", "macro_f1", "ece", "conformal")}, indent=2))


if __name__ == "__main__":
    main()
