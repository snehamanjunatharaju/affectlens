"""Fit temperature scaling and conformal quantiles on a disjoint calibration folder."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.model_selection import train_test_split

from .inference import ExpressionModel
from .labels import ALIASES, IMAGE_EXTENSIONS, LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True, help="Training folder; 15%% is held out for calibration")
    parser.add_argument("--output", type=Path, help="Optional separate calibration JSON output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    if not 0 < args.alpha < 1:
        parser.error("--alpha must be between zero and one")
    samples = []
    for folder in sorted(args.train_split.iterdir()):
        if folder.is_dir() and folder.name.casefold() in ALIASES:
            label = LABELS.index(ALIASES[folder.name.casefold()])
            samples.extend((path, label, hashlib.sha256(path.read_bytes()).hexdigest()) for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not samples:
        parser.error("No labeled images found in --train-split")
    group_labels = {}
    groups = {}
    for index, (path, label, group) in enumerate(samples):
        groups.setdefault(group, []).append(index)
        group_labels[group] = label
    group_ids = np.asarray(sorted(groups))
    labels = np.asarray([group_labels[group] for group in group_ids])
    _, calibration_groups = train_test_split(group_ids, test_size=0.15, random_state=args.seed, stratify=labels)
    indices = [index for group in calibration_groups for index in groups[group]]

    model = ExpressionModel(args.model, threshold=0.55)
    old_path = args.model.with_suffix(".uncertainty.json")
    old = json.loads(old_path.read_text()) if old_path.is_file() else {}
    old_temperature = float(old.get("temperature", 1.0))
    probs, expected = [], []
    for index in indices:
        path, label, _ = samples[index]
        pred = model.predict(path.read_bytes(), face_crop=True)
        # Undo the old temperature transform in logit space before refitting.
        probabilities = np.asarray([pred.probabilities.get(name, 0.0) for name in LABELS])
        logits = np.log(np.clip(probabilities, 1e-9, 1.0)) * old_temperature
        probs.append(logits)
        expected.append(label)
    logits = np.asarray(probs)
    expected = np.asarray(expected)

    def nll(log_temperature):
        temperature = np.exp(log_temperature)
        scaled = logits / temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        logsum = np.log(np.exp(scaled).sum(axis=1))
        return float(np.mean(logsum - scaled[np.arange(len(expected)), expected]))

    result = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    temperature = float(np.exp(result.x))
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    calibrated = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
    scores = np.sort(1 - calibrated[np.arange(len(expected)), expected])
    rank = min(len(scores) - 1, int(np.ceil((len(scores) + 1) * (1 - args.alpha))) - 1)
    quantile = float(scores[rank])
    sets = calibrated >= 1 - quantile
    result_row = {
        "method": "split_conformal_classification",
        "alpha": args.alpha,
        "quantile": quantile,
        "labels": list(LABELS),
        "training_labels": old.get("training_labels", "majority labels"),
        "temperature": temperature,
        "calibration_count": len(expected),
        "calibration_nll": float(result.fun),
        "calibration_coverage": float(sets[np.arange(len(expected)), expected].mean()),
        "calibration_mean_set_size": float(sets.sum(axis=1).mean()),
        "note": "Temperature and conformal quantile fit on a stratified held-out 15% of train only.",
    }
    output = args.output or old_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result_row, indent=2))
    print(json.dumps(result_row, indent=2))


if __name__ == "__main__":
    main()
