"""Train and evaluate a late fusion of FER scores and MediaPipe blendshapes."""

import argparse
import json
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .inference import ExpressionModel
from .labels import ALIASES, IMAGE_EXTENSIONS, LABELS


class FusionHead(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(32, len(LABELS)),
        )

    def forward(self, image_scores, blendshapes):
        return self.layers(torch.cat((image_scores, blendshapes), dim=1))


def image_rows(split: Path):
    rows = []
    for folder in sorted(split.iterdir()):
        if not folder.is_dir() or folder.name.lower() not in ALIASES:
            continue
        label = LABELS.index(ALIASES[folder.name.lower()])
        rows.extend((path, label) for path in sorted(folder.rglob("*")) if path.suffix.lower() in IMAGE_EXTENSIONS)
    return rows


def media_pipe_features(image_path: Path, landmarker, names: list[str] | None):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    image = cv2.cvtColor(cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image))
    if not result.face_blendshapes:
        return (None if names is None else np.zeros(len(names) + 1, dtype=np.float32)), names, None
    categories = result.face_blendshapes[0]
    current_names = [item.category_name for item in categories]
    if names is None:
        names = current_names
    if current_names != names:
        values = {item.category_name: item.score for item in categories}
        vector = [float(values.get(name, 0.0)) for name in names]
    else:
        vector = [float(item.score) for item in categories]
    landmarks = result.face_landmarks[0]
    points = [(point.x * 256, point.y * 256) for point in landmarks]
    xs, ys = zip(*points)
    face_width, face_height = max(xs) - min(xs), max(ys) - min(ys)
    side = max(face_width, face_height) * 1.18
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2 - face_height * 0.02
    eye_left, eye_right = points[33], points[263]
    angle = math.atan2(eye_right[1] - eye_left[1], eye_right[0] - eye_left[0])
    matrix = cv2.getRotationMatrix2D((center_x, center_y), math.degrees(angle), 256 / side)
    matrix[0, 2] += 128 - center_x
    matrix[1, 2] += 128 - center_y
    aligned = cv2.warpAffine(image, matrix, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(119, 119, 119))
    success, encoded = cv2.imencode(".jpg", cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not success:
        raise RuntimeError(f"Could not encode aligned face: {image_path}")
    return np.asarray([*vector, 1.0], dtype=np.float32), names, encoded.tobytes()


def collect_split(split: Path, base_model: ExpressionModel, landmarker, names):
    rows = image_rows(split)
    probabilities, features, labels = [], [], []
    for index, (path, label) in enumerate(rows, 1):
        vector, names, aligned_crop = media_pipe_features(path, landmarker, names)
        crop = aligned_crop if aligned_crop is not None else path.read_bytes()
        prediction = base_model.predict(crop, face_crop=True)
        probabilities.append([prediction.probabilities.get(name, 0.0) for name in LABELS])
        features.append(vector)
        labels.append(label)
        if index % 500 == 0:
            print(f"{split.name}: extracted {index}/{len(rows)}")
    if names is None:
        raise RuntimeError(f"MediaPipe did not find a face in any image under {split}")
    features = [vector if vector is not None else np.zeros(len(names) + 1, dtype=np.float32) for vector in features]
    return np.asarray(probabilities, dtype=np.float32), np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64), names


def class_metrics(expected, predicted):
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(f1_score(expected, predicted, labels=list(range(len(LABELS))), average="macro", zero_division=0)),
        "classes": classification_report(expected, predicted, labels=list(range(len(LABELS))), target_names=list(LABELS), output_dict=True, zero_division=0),
    }


def softmax(logits, temperature=1.0):
    values = np.asarray(logits, dtype=np.float64) / temperature
    values -= values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(logits, labels):
    tensor_logits = torch.tensor(logits, dtype=torch.float64)
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    temperature = torch.nn.Parameter(torch.ones((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([temperature], lr=0.05, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(tensor_logits / temperature.clamp(0.05, 20.0), tensor_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp(0.05, 20.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Existing single-image ONNX model")
    parser.add_argument("--task-model", type=Path, default=Path("affectlens/static/vendor/mediapipe/face_landmarker.task"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/fer-ferplus"), help="Output stem; active fusion sidecars use this model's stem")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    if not (args.data / "validation").is_dir() or not (args.data / "test").is_dir():
        parser.error("Fusion training needs separate validation/ and test/ splits")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    base_model = ExpressionModel(args.model, threshold=0.5)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.task_model), delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        output_face_blendshapes=True,
        num_faces=1,
    )
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        val_probs, val_features, val_labels, names = collect_split(args.data / "validation", base_model, landmarker, None)
        test_probs, test_features, test_labels, _ = collect_split(args.data / "test", base_model, landmarker, names)

    valid_fraction = float(np.mean(val_features[:, -1] > 0))
    test_valid_fraction = float(np.mean(test_features[:, -1] > 0))
    if valid_fraction < 0.80 or test_valid_fraction < 0.80:
        parser.error(f"MediaPipe mesh was available on only {valid_fraction:.1%} validation and {test_valid_fraction:.1%} test images; refusing to train an unreliable fusion head")

    indices = np.arange(len(val_labels))
    fit_indices, holdout_indices = train_test_split(indices, test_size=0.30, random_state=args.seed, stratify=val_labels)
    selection_indices, calibration_indices = train_test_split(
        holdout_indices, test_size=0.50, random_state=args.seed, stratify=val_labels[holdout_indices]
    )
    feature_mean = val_features[fit_indices, :-1].mean(axis=0)
    feature_std = np.maximum(val_features[fit_indices, :-1].std(axis=0), 0.05)
    normalized_features = np.concatenate([
        (val_features[:, :-1] - feature_mean) / feature_std,
        val_features[:, -1:],
    ], axis=1)
    normalized_test_features = np.concatenate([
        (test_features[:, :-1] - feature_mean) / feature_std,
        test_features[:, -1:],
    ], axis=1)

    def tensors(probs, features, labels, selected):
        return TensorDataset(torch.from_numpy(probs[selected]), torch.from_numpy(features[selected]), torch.from_numpy(labels[selected]))

    fit_loader = DataLoader(tensors(val_probs, normalized_features, val_labels, fit_indices), batch_size=128, shuffle=True)
    selection_x = torch.from_numpy(np.concatenate([val_probs[selection_indices], normalized_features[selection_indices]], axis=1))
    calibration_x = torch.from_numpy(np.concatenate([val_probs[calibration_indices], normalized_features[calibration_indices]], axis=1))
    test_x = torch.from_numpy(np.concatenate([test_probs, normalized_test_features], axis=1))
    model = FusionHead(val_probs.shape[1] + normalized_features.shape[1])
    counts = np.bincount(val_labels[fit_indices], minlength=len(LABELS))
    class_weights = counts.sum() / (len(LABELS) * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.002)
    best_f1, best_state, stale = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        for image_scores, blendshapes, labels in fit_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(image_scores, blendshapes), labels)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            selection_logits = model(selection_x[:, :7], selection_x[:, 7:]).numpy()
        selection = class_metrics(val_labels[selection_indices], selection_logits.argmax(axis=1))
        print(f"epoch={epoch + 1:03d} selection_accuracy={selection['accuracy']:.4f} selection_macro_f1={selection['macro_f1']:.4f}")
        if selection["macro_f1"] > best_f1:
            best_f1, best_state, stale = selection["macro_f1"], {key: value.clone() for key, value in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= 12:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        selection_logits = model(selection_x[:, :7], selection_x[:, 7:]).numpy()
        calibration_logits = model(calibration_x[:, :7], calibration_x[:, 7:]).numpy()
        test_logits = model(test_x[:, :7], test_x[:, 7:]).numpy()
    temperature = fit_temperature(calibration_logits, val_labels[calibration_indices])
    calibration_probs = softmax(calibration_logits, temperature)
    nonconformity = np.sort(1 - calibration_probs[np.arange(len(calibration_indices)), val_labels[calibration_indices]])
    rank = min(len(nonconformity) - 1, int(np.ceil((len(nonconformity) + 1) * 0.9)) - 1)
    quantile = float(nonconformity[rank])
    test_calibrated = softmax(test_logits, temperature)
    test_sets = test_calibrated >= 1 - quantile
    test_predictions = test_calibrated.argmax(axis=1)
    baseline_metrics = class_metrics(test_labels, test_probs.argmax(axis=1))
    fusion_metrics = class_metrics(test_labels, test_predictions)
    selection_baseline = class_metrics(val_labels[selection_indices], val_probs[selection_indices].argmax(axis=1))
    selection_fusion = class_metrics(val_labels[selection_indices], selection_logits.argmax(axis=1))

    activate = selection_fusion["macro_f1"] >= selection_baseline["macro_f1"] + 0.01 and selection_fusion["accuracy"] >= selection_baseline["accuracy"] - 0.005
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model_suffix = ".fusion.onnx" if activate else ".fusion-candidate.onnx"
    metadata_suffix = ".fusion.json" if activate else ".fusion-candidate.json"
    model_path = args.output.with_suffix(model_suffix)
    torch.onnx.export(
        model,
        (torch.zeros(1, 7), torch.zeros(1, len(names) + 1)),
        str(model_path),
        input_names=["image_scores", "blendshapes"],
        output_names=["logits"],
        dynamic_shapes=({0: torch.export.Dim("batch")}, {0: torch.export.Dim("batch")}),
        dynamo=True,
        opset_version=18,
    )
    metadata = {
        "labels": list(LABELS),
        "blendshape_names": names,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "temperature": temperature,
        "quantile": quantile,
        "alpha": 0.10,
        "mesh_valid_fraction": valid_fraction,
        "test_mesh_valid_fraction": test_valid_fraction,
        "training_split": "FER+ validation split only; split into fit, selection, and calibration subsets",
        "note": "Late fusion of single-image FER probabilities and MediaPipe blendshape coefficients. Not a measurement of inner emotional state.",
    }
    args.output.with_suffix(metadata_suffix).write_text(json.dumps(metadata, indent=2) + "\n")
    base_top = test_probs.argmax(axis=1)
    disagreement = test_predictions != base_top
    test_sets[np.arange(len(test_labels))[disagreement], base_top[disagreement]] = True
    test_coverage = float(test_sets[np.arange(len(test_labels)), test_labels].mean())
    report = {
        "selection": {"image_only": selection_baseline, "fusion": selection_fusion},
        "test": {"image_only": baseline_metrics, "fusion": fusion_metrics,
                 "conformal_coverage": test_coverage, "mean_set_size": float(test_sets.sum(axis=1).mean()),
                 "singleton_coverage": float((test_sets.sum(axis=1) == 1).mean())},
        "class_order": list(LABELS),
        "feature_count": len(names),
        "mesh_valid_fraction": valid_fraction,
        "test_mesh_valid_fraction": test_valid_fraction,
        "activated_by_selection": activate,
        "model": str(model_path),
    }
    args.output.with_suffix(".fusion.metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"activated_by_selection": activate, "selection": report["selection"], "test": {"image_only_accuracy": baseline_metrics["accuracy"], "fusion_accuracy": fusion_metrics["accuracy"], "image_only_macro_f1": baseline_metrics["macro_f1"], "fusion_macro_f1": fusion_metrics["macro_f1"], "conformal_coverage": test_coverage, "mean_set_size": report["test"]["mean_set_size"]}}, indent=2))


if __name__ == "__main__":
    main()
