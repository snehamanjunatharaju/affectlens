"""Train a compact FER baseline, report held-out metrics, and export ONNX."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from .audit_data import audit_dataset
from .labels import ALIASES, IMAGE_EXTENSIONS, LABELS


class FolderFaces(Dataset):
    def __init__(self, root: Path, transform):
        self.root = root
        self.transform = transform
        soft_targets = {}
        sidecar = root / "soft_labels.jsonl"
        if sidecar.is_file():
            with sidecar.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    row = json.loads(line)
                    target = np.asarray(row["target"], dtype=np.float32)
                    if target.shape != (len(LABELS),) or not np.isfinite(target).all() or np.any(target < 0) or not np.isclose(target.sum(), 1.0, atol=1e-4):
                        raise ValueError(f"Invalid soft target on {sidecar}:{line_number}")
                    soft_targets[row["file"]] = target
        self.uses_soft_labels = bool(soft_targets)
        self.samples: list[tuple[Path, int, np.ndarray]] = []
        for folder in sorted(root.iterdir()):
            if not folder.is_dir() or folder.name.lower() not in ALIASES:
                continue
            label = LABELS.index(ALIASES[folder.name.lower()])
            for path in folder.rglob("*"):
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                target = soft_targets.get(path.relative_to(root).as_posix())
                if target is None:
                    target = np.eye(len(LABELS), dtype=np.float32)[label]
                self.samples.append((path, label, target))
        if not self.samples:
            raise ValueError(f"No labeled image files found under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("L")
            return self.transform(image), label, torch.from_numpy(target.copy())


class CompactFER(nn.Module):
    """Small depthwise CNN baseline for 48x48 grayscale FER crops."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.Hardswish(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=32, bias=False),
            nn.Conv2d(32, 64, 1, bias=False), nn.BatchNorm2d(64), nn.Hardswish(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, groups=64, bias=False),
            nn.Conv2d(64, 128, 1, bias=False), nn.BatchNorm2d(128), nn.Hardswish(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1, groups=128, bias=False),
            nn.Conv2d(128, 192, 1, bias=False), nn.BatchNorm2d(192), nn.Hardswish(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(192, len(LABELS)))

    def forward(self, image):
        return self.classifier(self.features(image))


def metrics(model, loader, device):
    model.eval()
    expected, probabilities = [], []
    with torch.no_grad():
        for images, labels, _targets in loader:
            scores = model(images.to(device)).softmax(dim=1)
            expected.extend(labels.numpy().tolist())
            probabilities.extend(scores.cpu().numpy().tolist())
    probabilities = np.asarray(probabilities)
    predicted = probabilities.argmax(axis=1)
    matrix = confusion_matrix(expected, predicted, labels=list(range(len(LABELS))))
    return {
        "accuracy": float(np.trace(matrix) / max(1, matrix.sum())),
        "macro_f1": float(f1_score(expected, predicted, labels=list(range(len(LABELS))), average="macro", zero_division=0)),
        "classes": classification_report(expected, predicted, labels=list(range(len(LABELS))), target_names=list(LABELS), output_dict=True, zero_division=0),
        "confusion_matrix": matrix.tolist(),
        "expected": expected,
        "probabilities": probabilities,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Folder with train/<class>/ and test/<class>/")
    parser.add_argument("--output", type=Path, default=Path("artifacts/fer.onnx"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.10, help="Target conformal miscoverage rate")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must be strictly between 0 and 1")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    augment = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    plain = transforms.Compose([transforms.Resize((48, 48)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    if not (args.data / "train").is_dir() or not (args.data / "test").is_dir():
        parser.error("--data must contain train/ and test/ split directories")
    train_aug = FolderFaces(args.data / "train", augment)
    train_eval = FolderFaces(args.data / "train", plain)
    test_data = FolderFaces(args.data / "test", plain)
    has_validation = (args.data / "validation").is_dir()
    selection_data = FolderFaces(args.data / "validation", plain) if has_validation else train_eval
    data_audit = audit_dataset(args.data)
    audit_issues = []
    checked_splits = ["train", "test"] + (["validation"] if has_validation else [])
    for split in checked_splits:
        split_report = data_audit["splits"].get(split, {})
        if split_report.get("missing_classes"):
            audit_issues.append(f"{split} is missing {split_report['missing_classes']}")
        if split_report.get("corrupt_images"):
            audit_issues.append(f"{split} contains {split_report['corrupt_images']} unreadable image(s)")
    if data_audit["cross_split_duplicate_groups"]:
        audit_issues.append(f"{data_audit['cross_split_duplicate_groups']} exact duplicate group(s) cross dataset splits")
    if data_audit["conflicting_label_groups"]:
        audit_issues.append(f"{data_audit['conflicting_label_groups']} exact duplicate group(s) have conflicting labels")
    for split in ("validation", "test"):
        repeated = data_audit["splits"].get(split, {}).get("duplicate_content_groups", 0)
        if repeated:
            audit_issues.append(f"{split} contains {repeated} repeated image group(s), which can bias evaluation")
    if audit_issues:
        parser.error("Dataset audit failed: " + "; ".join(audit_issues) + ". See the audit report and fix the source data before training.")
    train_classes = {label for _, label, _target in train_aug.samples}
    test_classes = {label for _, label, _target in test_data.samples}
    missing_train = [LABELS[i] for i in range(len(LABELS)) if i not in train_classes]
    missing_test = [LABELS[i] for i in range(len(LABELS)) if i not in test_classes]
    if missing_train or missing_test:
        parser.error(f"All seven classes are required; missing from train={missing_train}, test={missing_test}")
    duplicate_groups: dict[str, list[int]] = defaultdict(list)
    group_labels: dict[str, int] = {}
    for index, (path, label, _target) in enumerate(train_aug.samples):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        duplicate_groups[digest].append(index)
        group_labels[digest] = label
    group_ids = np.asarray(sorted(duplicate_groups))
    labels_by_group = np.asarray([group_labels[digest] for digest in group_ids])
    def expand_groups(group_set):
        return [index for digest in group_set for index in duplicate_groups[digest]]

    class_counts = np.bincount(labels_by_group, minlength=len(LABELS))
    if np.any(class_counts < 7):
        parser.error("Need at least seven unique image groups per class for stratified fit, selection, and calibration splits")
    labels_lookup = dict(zip(group_ids, labels_by_group))
    if has_validation:
        fit_groups, calibration_groups = train_test_split(
            group_ids, test_size=0.15, random_state=args.seed, stratify=labels_by_group
        )
        selection_loader = DataLoader(selection_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    else:
        fit_groups, holdout_groups = train_test_split(
            group_ids, test_size=0.30, random_state=args.seed, stratify=labels_by_group
        )
        selection_groups, calibration_groups = train_test_split(
            holdout_groups,
            test_size=0.50,
            random_state=args.seed,
            stratify=np.asarray([labels_lookup[digest] for digest in holdout_groups]),
        )
        selection_indices = [index for digest in selection_groups for index in duplicate_groups[digest]]
        selection_loader = DataLoader(Subset(train_eval, selection_indices), batch_size=args.batch_size, shuffle=False, num_workers=0)
    fit_indices = expand_groups(fit_groups)
    calibration_indices = expand_groups(calibration_groups)
    train_loader = DataLoader(Subset(train_aug, fit_indices), batch_size=args.batch_size, shuffle=True, num_workers=0)
    calibration_loader = DataLoader(Subset(train_eval, calibration_indices), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = CompactFER().to(device)
    class_mass = np.stack([train_aug.samples[i][2] for i in fit_indices]).sum(axis=0)
    weights = class_mass.sum() / (len(LABELS) * np.maximum(class_mass, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_f1, best_state, stale = -1.0, None, 0

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for images, labels, targets in train_loader:
            images, labels, targets = images.to(device), labels.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
        scheduler.step()
        validation = metrics(model, selection_loader, device)
        print(f"epoch={epoch + 1:02d} loss={loss_sum / len(fit_indices):.4f} selection_macro_f1={validation['macro_f1']:.4f}")
        if validation["macro_f1"] > best_f1:
            best_f1 = validation["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    model.to(device)
    calibration_metrics = metrics(model, calibration_loader, device)
    calibration_y = np.asarray(calibration_metrics.pop("expected"))
    calibration_probs = calibration_metrics.pop("probabilities")
    nonconformity = np.sort(1.0 - calibration_probs[np.arange(len(calibration_y)), calibration_y])
    alpha = args.alpha
    rank = min(len(nonconformity) - 1, int(np.ceil((len(nonconformity) + 1) * (1 - alpha))) - 1)
    conformal_quantile = float(nonconformity[rank])
    calibration_sets = calibration_probs >= (1.0 - conformal_quantile)
    calibration_singleton = calibration_sets.sum(axis=1) == 1
    calibration_metrics["conformal"] = {
        "alpha": alpha,
        "score": "1 - probability assigned to the true class",
        "quantile": conformal_quantile,
        "sample_count": int(len(calibration_y)),
        "singleton_coverage": float(calibration_singleton.mean()),
        "singleton_accuracy": float((calibration_sets.argmax(axis=1)[calibration_singleton] == calibration_y[calibration_singleton]).mean()) if calibration_singleton.any() else None,
    }
    test_metrics = metrics(model, test_loader, device)
    test_y = np.asarray(test_metrics.pop("expected"))
    test_probs = test_metrics.pop("probabilities")
    test_sets = test_probs >= (1.0 - conformal_quantile)
    test_singleton = test_sets.sum(axis=1) == 1
    test_metrics["conformal"] = {
        "alpha": alpha,
        "marginal_coverage": float(test_sets[np.arange(len(test_y)), test_y].mean()),
        "singleton_coverage": float(test_singleton.mean()),
        "singleton_accuracy": float((test_sets.argmax(axis=1)[test_singleton] == test_y[test_singleton]).mean()) if test_singleton.any() else None,
        "note": "Finite-sample coverage assumes calibration and test examples are exchangeable.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    export_model = model.to("cpu")
    torch.onnx.export(
        export_model, torch.zeros(1, 1, 48, 48), str(args.output),
        input_names=["face"], output_names=["logits"],
        dynamic_shapes=({0: torch.export.Dim("batch")},),
        dynamo=True,
        opset_version=18,
    )
    calibration_path = args.output.with_suffix(".uncertainty.json")
    calibration_path.write_text(json.dumps({
        "method": "split_conformal_classification",
        "alpha": alpha,
        "quantile": conformal_quantile,
        "labels": list(LABELS),
        "training_labels": "FER+ annotator distributions" if train_aug.uses_soft_labels else "single hard labels",
    }, indent=2))
    report_path = args.output.with_suffix(".metrics.json")
    report_path.write_text(json.dumps({
        "seed": args.seed,
        "best_selection_macro_f1": best_f1,
        "selection_source": "validation split" if has_validation else "stratified subset of train split",
        "calibration": calibration_metrics,
        "test": test_metrics,
        "labels": list(LABELS),
        "note": "Single benchmark split; no demographic or cross-dataset generalization claim.",
    }, indent=2))
    print(f"ONNX model: {args.output}\nTest accuracy: {test_metrics['accuracy']:.4f}\nTest macro-F1: {test_metrics['macro_f1']:.4f}\nTest conformal coverage: {test_metrics['conformal']['marginal_coverage']:.4f}\nMetrics: {report_path}")


if __name__ == "__main__":
    main()
