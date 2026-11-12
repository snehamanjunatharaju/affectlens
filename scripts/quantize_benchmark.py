"""Build an INT8 ONNX copy and compare PyTorch, ONNX, and quantized CPU inference."""

import argparse
import json
import random
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from PIL import Image, ImageOps

from affectlens.evaluate import evaluate
from affectlens.explain import recover_compact_model
from affectlens.labels import IMAGE_EXTENSIONS


def face_tensor(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("L").resize((size, size))
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return array[None, None, :, :]


class FERCalibrationReader(CalibrationDataReader):
    def __init__(self, paths: list[Path], size: int):
        self.rows = iter([{"face": face_tensor(path, size)} for path in paths])

    def get_next(self):
        return next(self.rows, None)


def latency(call, warmup: int = 30, iterations: int = 250) -> dict:
    for _ in range(warmup):
        call()
    values = []
    for _ in range(iterations):
        start = time.perf_counter()
        call()
        values.append((time.perf_counter() - start) * 1000)
    return {
        "iterations": iterations,
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(statistics.mean(values)),
    }


def artifact_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.parent.glob(path.name + "*"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True, help="FER training split; used only for quantization calibration")
    parser.add_argument("--test-data", type=Path, required=True, help="Held-out labeled split for quality evaluation")
    parser.add_argument("--output", type=Path, required=True, help="Quantized ONNX output path")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--calibration-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    preprocessing_path = args.model.with_suffix(".preprocess.json")
    preprocessing = json.loads(preprocessing_path.read_text()) if preprocessing_path.exists() else {}
    if preprocessing.get("channels", 1) != 1:
        parser.error("This benchmark currently supports grayscale compact models only.")
    size = int(preprocessing.get("size", 48))
    candidates = [path for path in args.train_data.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    if len(candidates) < 16:
        parser.error("The training split needs at least 16 readable image files.")
    selected = random.Random(args.seed).sample(candidates, min(args.calibration_samples, len(candidates)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(args.model),
        str(args.output),
        FERCalibrationReader(selected, size),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    for suffix in (".preprocess.json", ".uncertainty.json"):
        sidecar = args.model.with_suffix(suffix)
        if sidecar.exists():
            shutil.copy2(sidecar, args.output.with_suffix(suffix))

    sample = face_tensor(selected[0], size)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    fp_session = ort.InferenceSession(str(args.model), sess_options=options, providers=["CPUExecutionProvider"])
    int8_session = ort.InferenceSession(str(args.output), sess_options=options, providers=["CPUExecutionProvider"])
    fp_call = lambda: fp_session.run(None, {fp_session.get_inputs()[0].name: sample})
    int8_call = lambda: int8_session.run(None, {int8_session.get_inputs()[0].name: sample})

    torch_model = recover_compact_model(args.model).eval()
    torch.set_num_threads(1)
    torch_input = torch.from_numpy(sample)
    with torch.inference_mode():
        torch_call = lambda: torch_model(torch_input)
        torch_latency = latency(torch_call)
    fp_latency = latency(fp_call)
    int8_latency = latency(int8_call)

    fp_quality = evaluate(args.model, args.test_data)
    int8_quality = evaluate(args.output, args.test_data)
    report = {
        "source_model": str(args.model),
        "quantized_model": str(args.output),
        "calibration_source": str(args.train_data),
        "calibration_samples": len(selected),
        "quantization": "ONNX Runtime QDQ static INT8; train images only for activation ranges",
        "benchmark": {
            "runtime": "CPU, single intra-op thread, batch size 1",
            "torch": {**torch_latency, "artifact_bytes": int(args.model.with_suffix(".pt").stat().st_size) if args.model.with_suffix(".pt").exists() else None},
            "onnx_fp32": {**fp_latency, "artifact_bytes": artifact_size(args.model)},
            "onnx_int8": {**int8_latency, "artifact_bytes": artifact_size(args.output)},
        },
        "heldout_quality": {
            "split": str(args.test_data),
            "fp32": {key: fp_quality[key] for key in ("sample_count", "accuracy", "macro_f1", "ece", "conformal")},
            "int8": {key: int8_quality[key] for key in ("sample_count", "accuracy", "macro_f1", "ece", "conformal")},
        },
        "note": "Latency varies by device, power state, runtime, and build. The FP32 calibration sidecar is reused to expose quantization-induced calibration/coverage changes.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["benchmark"], indent=2))
    print(json.dumps(report["heldout_quality"], indent=2))


if __name__ == "__main__":
    main()
