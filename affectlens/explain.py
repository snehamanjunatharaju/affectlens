"""Build Grad-CAM failure examples from an exported CompactFER ONNX checkpoint."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
import torch
from onnx import numpy_helper
from torch import nn

from .inference import ExpressionModel
from .labels import ALIASES, IMAGE_EXTENSIONS, LABELS


class CompactExport(nn.Module):
    """CompactFER with Conv-BN fused weights recovered from its ONNX graph."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.Hardswish(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=32, bias=False),
            nn.Conv2d(32, 64, 1), nn.Hardswish(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, groups=64, bias=False),
            nn.Conv2d(64, 128, 1), nn.Hardswish(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1, groups=128, bias=False),
            nn.Conv2d(128, 192, 1), nn.Hardswish(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(192, len(LABELS)))

    def forward(self, image):
        return self.classifier(self.features(image))


def recover_compact_model(model_path: Path):
    graph = onnx.load(str(model_path), load_external_data=True)
    initializers = {item.name: np.array(numpy_helper.to_array(item), copy=True) for item in graph.graph.initializer}
    model = CompactExport().eval()
    state = model.state_dict()
    for module_index, onnx_index, has_bias in ((0, 0, True), (2, 3, False), (3, 4, True), (5, 7, False), (6, 8, True), (8, 11, False), (9, 12, True)):
        state[f"features.{module_index}.weight"] = torch.from_numpy(initializers[f"features.{onnx_index}.weight"])
        if has_bias:
            fused_bias_name = f"features.{onnx_index}.weight_bias"
            state[f"features.{module_index}.bias"] = torch.from_numpy(initializers[fused_bias_name])
    state["classifier.1.weight"] = torch.from_numpy(initializers["classifier.2.weight"])
    state["classifier.1.bias"] = torch.from_numpy(initializers["classifier.2.bias"])
    model.load_state_dict(state)

    session = __import__("onnxruntime").InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    probe = np.linspace(-1, 1, 48 * 48, dtype=np.float32).reshape(1, 1, 48, 48)
    onnx_output = session.run(None, {session.get_inputs()[0].name: probe})[0].reshape(-1)
    with torch.no_grad():
        torch_output = model(torch.from_numpy(probe)).numpy().reshape(-1)
    if not np.allclose(onnx_output, torch_output, atol=1e-4, rtol=1e-4):
        raise ValueError("Could not faithfully recover CompactFER weights from ONNX; Grad-CAM is unsafe for this architecture.")
    return model


def gradcam(model, image: np.ndarray, class_index: int) -> np.ndarray:
    activations, gradients = [], []
    layer = model.features[9]
    def capture(_module, _inputs, output):
        activations.append(output)
        output.register_hook(lambda gradient: gradients.append(gradient))
    layer.register_forward_hook(capture)
    tensor = torch.from_numpy((image.astype(np.float32)[None, None] / 127.5) - 1.0)
    model.zero_grad(set_to_none=True)
    score = model(tensor)[0, class_index]
    score.backward()
    activation, gradient = activations[0], gradients[0]
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    heat = torch.relu((weights * activation).sum(dim=1))[0].detach().numpy()
    heat -= heat.min()
    if heat.max() > 0:
        heat /= heat.max()
    return cv2.resize(heat, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="Held-out test split with class folders")
    parser.add_argument("--output", type=Path, default=Path("docs/failures"))
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--filename-prefix", help="Optional source filename filter, e.g. PublicTest_")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

    inference = ExpressionModel(args.model, 0.55)
    candidates = []
    for folder in sorted(args.data.iterdir()):
        if not folder.is_dir() or folder.name.casefold() not in ALIASES:
            continue
        true_label = LABELS.index(ALIASES[folder.name.casefold()])
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTENSIONS or (args.filename_prefix and not path.name.startswith(args.filename_prefix)):
                continue
            prediction = inference.predict(path.read_bytes(), face_crop=True)
            scores = [prediction.probabilities.get(label, 0.0) for label in LABELS]
            predicted = int(np.argmax(scores))
            if predicted != true_label:
                candidates.append((max(scores), path, true_label, predicted))
    candidates.sort(key=lambda row: row[0], reverse=True)
    if not candidates:
        parser.error("No wrong predictions found")

    model = recover_compact_model(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    failures = []
    for rank, (confidence, path, true_label, predicted) in enumerate(candidates[:args.limit], 1):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        heat = gradcam(model, image, predicted)
        resized = cv2.resize(image, (288, 288), interpolation=cv2.INTER_CUBIC)
        heat = cv2.resize(heat, (288, 288), interpolation=cv2.INTER_CUBIC)
        colored = cv2.applyColorMap(np.uint8(np.clip(heat, 0, 1) * 255), cv2.COLORMAP_TURBO)
        overlay = cv2.addWeighted(cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR), 0.58, colored, 0.42, 0)
        stem = f"{rank:02d}-{LABELS[true_label]}-as-{LABELS[predicted]}"
        cv2.imwrite(str(args.output / f"{stem}-face.png"), resized)
        cv2.imwrite(str(args.output / f"{stem}-gradcam.png"), overlay)
        failures.append({
            "rank": rank,
            "source_file": path.name,
            "true_label": LABELS[true_label],
            "predicted_label": LABELS[predicted],
            "confidence": confidence,
            "face_image": f"{stem}-face.png",
            "gradcam_image": f"{stem}-gradcam.png",
            "possible_factors": "Label disagreement, expression ambiguity, pose, lighting, or occlusion are hypotheses only; no individual cause was verified.",
        })
    (args.output / "index.json").write_text(json.dumps(failures, indent=2))
    print(f"Saved {len(failures)} held-out benchmark errors with Grad-CAM overlays under {args.output}")


if __name__ == "__main__":
    main()
