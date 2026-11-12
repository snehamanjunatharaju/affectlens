"""Use out-of-fold HOG/logistic probabilities with Cleanlab to rank likely label issues."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from cleanlab.filter import find_label_issues
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .labels import ALIASES, IMAGE_EXTENSIONS, LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="FER+ dataset root; train/ and validation/ are audited")
    parser.add_argument("--output", type=Path, default=Path("data/flagged_labels.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/cleanlab-label-audit.json"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    samples = []
    for split in ("train", "validation"):
        root = args.data / split
        if not root.is_dir():
            continue
        for folder in sorted(root.iterdir()):
            if folder.is_dir() and folder.name.casefold() in ALIASES:
                label = LABELS.index(ALIASES[folder.name.casefold()])
                for path in sorted(folder.rglob("*")):
                    if path.suffix.lower() in IMAGE_EXTENSIONS:
                        samples.append((path, split, label))
    if not samples:
        parser.error("No labeled train/validation images found")
    labels = np.asarray([row[2] for row in samples], dtype=np.int64)
    if min(np.bincount(labels, minlength=len(LABELS))) < args.folds:
        parser.error("Each class must have at least --folds samples")

    hog = cv2.HOGDescriptor((48, 48), (16, 16), (8, 8), (8, 8), 9)
    features = []
    for path, _split, _label in samples:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (48, 48):
            parser.error(f"Unreadable or non-48x48 image: {path}")
        features.append(hog.compute(image).reshape(-1))
    features = np.asarray(features, dtype=np.float32)
    folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=500))
    out_of_fold = cross_val_predict(classifier, features, labels, cv=folds, method="predict_proba", n_jobs=1)
    issue_indices = find_label_issues(labels=labels, pred_probs=out_of_fold, return_indices_ranked_by="self_confidence")
    predicted = out_of_fold.argmax(axis=1)
    rows = []
    for rank, index in enumerate(issue_indices, 1):
        path, split, label = samples[int(index)]
        rows.append({
            "rank": rank,
            "source_index": path.stem.removeprefix("ferplus_"),
            "split": split,
            "relative_path": path.relative_to(args.data).as_posix(),
            "observed_label": LABELS[label],
            "oof_top_label": LABELS[int(predicted[index])],
            "oof_probability_for_observed_label": float(out_of_fold[index, label]),
            "oof_prediction_confidence": float(out_of_fold[index, predicted[index]]),
            "method": "Cleanlab confident-learning issue ranking using 5-fold out-of-fold HOG/logistic probabilities",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]) if rows else ["rank", "relative_path"])
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "samples": len(samples),
        "folds": args.folds,
        "flagged_count": len(rows),
        "flagged_fraction": len(rows) / len(samples),
        "feature_model": "OpenCV HOG + class-balanced logistic regression",
        "probability_method": "Stratified out-of-fold predict_proba; no image is predicted by a model fit on its fold.",
        "oof_accuracy": float(accuracy_score(labels, predicted)),
        "oof_macro_f1": float(f1_score(labels, predicted, labels=list(range(len(LABELS))), average="macro", zero_division=0)),
        "limitations": [
            "Flags are review candidates, not ground-truth corrections.",
            "The audit covers only FER+ train and validation images and uses the majority class as a discrete target.",
            "HOG/logistic probabilities provide a lightweight independent quality signal, not the final AffectLens classifier.",
            "FER+ annotator disagreement is expected and may cause apparent label issues without a true labeling error.",
        ],
        "csv": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
