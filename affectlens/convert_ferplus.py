"""Build a FER+ soft-label dataset from FER training pixels and ten-rater labels."""

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .labels import LABELS

PIXEL_COUNT = 48 * 48
VOTE_COLUMNS = ("anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise")
FERPLUS_TO_LABEL = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happiness": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprise",
}
NON_EXPRESSION_COLUMNS = ("contempt", "unknown", "nf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Labeled FER2013 training CSV")
    parser.add_argument("--ferplus", type=Path, required=True, help="Microsoft FER+ ten-rater annotation CSV")
    parser.add_argument("--output", type=Path, required=True, help="New output folder; existing folders are never overwritten")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.csv.is_file() or not args.ferplus.is_file():
        parser.error("Both --csv and --ferplus input files must exist")
    if args.output.exists():
        parser.error(f"Output already exists; choose a new folder: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        with args.csv.open("r", newline="", encoding="utf-8-sig") as file:
            raw_reader = csv.DictReader(file)
            raw_columns = {name.strip().casefold(): name for name in (raw_reader.fieldnames or [])}
            if not {"emotion", "pixels"} <= raw_columns.keys():
                parser.error("FER CSV must have emotion and pixels columns")
            raw_rows = list(raw_reader)
        with args.ferplus.open("r", newline="", encoding="utf-8-sig") as file:
            label_reader = csv.DictReader(file)
            label_columns = {name.strip().casefold(): name for name in (label_reader.fieldnames or [])}
            required = {"usage", "image name", *VOTE_COLUMNS, *NON_EXPRESSION_COLUMNS}
            if not required <= label_columns.keys():
                parser.error(f"FER+ CSV is missing columns: {sorted(required - label_columns.keys())}")
            label_rows = list(label_reader)
    except UnicodeDecodeError as exc:
        parser.error(f"Input CSVs must be UTF-8 encoded: {exc}")

    training_labels = [row for row in label_rows if row[label_columns["usage"]].strip().casefold() == "training"]
    if len(raw_rows) != len(training_labels):
        parser.error(
            f"FER training rows ({len(raw_rows)}) do not match FER+ Training annotations ({len(training_labels)}); "
            "refusing a potentially misaligned join"
        )

    groups = {}
    skipped_low_agreement = 0
    skipped_no_emotion_votes = 0
    for index, (raw, annotation) in enumerate(zip(raw_rows, training_labels)):
        try:
            emotion_id = int(raw[raw_columns["emotion"]])
            pixels = np.fromstring(raw[raw_columns["pixels"]], dtype=np.int16, sep=" ")
            emotion_votes = np.asarray([int(annotation[label_columns[column]]) for column in VOTE_COLUMNS], dtype=np.float32)
            other_votes = sum(int(annotation[label_columns[column]]) for column in NON_EXPRESSION_COLUMNS)
        except (TypeError, ValueError) as exc:
            parser.error(f"Malformed FER or FER+ row {index + 2}: {exc}")
        if emotion_id not in range(7) or pixels.size != PIXEL_COUNT or np.any(pixels < 0) or np.any(pixels > 255):
            parser.error(f"Invalid emotion or 48x48 pixel data on FER row {index + 2}")
        total_emotion_votes = float(emotion_votes.sum())
        if total_emotion_votes == 0:
            skipped_no_emotion_votes += 1
            continue
        if total_emotion_votes < 4 or other_votes > total_emotion_votes:
            skipped_low_agreement += 1
            continue

        target = np.zeros(len(LABELS), dtype=np.float64)
        for column, votes in zip(VOTE_COLUMNS, emotion_votes):
            target[LABELS.index(FERPLUS_TO_LABEL[column])] += votes
        target /= target.sum()
        pixel_bytes = pixels.astype(np.uint8).tobytes()
        digest = hashlib.sha256(pixel_bytes).digest()
        if digest not in groups:
            groups[digest] = {"pixels": pixel_bytes, "target_sum": target, "copies": 1, "row": index}
        else:
            groups[digest]["target_sum"] += target
            groups[digest]["copies"] += 1

    unique_images = []
    for digest, group in groups.items():
        target = group["target_sum"] / group["copies"]
        unique_images.append({
            "digest": digest.hex(),
            "pixels": group["pixels"],
            "target": target,
            "label": int(target.argmax()),
            "row": group["row"],
        })

    rng = np.random.default_rng(args.seed)
    assignments = {}
    for label in range(len(LABELS)):
        indices = [i for i, item in enumerate(unique_images) if item["label"] == label]
        if len(indices) < 3:
            parser.error(f"FER+ has fewer than 3 usable unique images for {LABELS[label]}")
        shuffled = rng.permutation(indices)
        n_validation = max(1, round(len(indices) * 0.15))
        n_test = max(1, round(len(indices) * 0.15))
        for rank, index in enumerate(shuffled):
            assignments[int(index)] = "validation" if rank < n_validation else "test" if rank < n_validation + n_test else "train"

    counts = {split: {label: 0 for label in LABELS} for split in ("train", "validation", "test")}
    with tempfile.TemporaryDirectory(prefix=f".{args.output.name}-", dir=args.output.parent) as temporary:
        staged = Path(temporary)
        for index, item in enumerate(unique_images):
            split = assignments[index]
            label = LABELS[item["label"]]
            relative = Path(label) / f"ferplus_{item['row']:06d}.png"
            target_dir = staged / split / label
            target_dir.mkdir(parents=True, exist_ok=True)
            image = np.frombuffer(item["pixels"], dtype=np.uint8).reshape(48, 48)
            success, encoded = cv2.imencode(".png", image)
            if not success:
                parser.error(f"Could not encode FER+ image from source row {item['row'] + 2}")
            (target_dir / relative.name).write_bytes(encoded.tobytes())
            sidecar = staged / split / "soft_labels.jsonl"
            with sidecar.open("a", encoding="utf-8") as output:
                output.write(json.dumps({"file": relative.as_posix(), "target": item["target"].tolist()}) + "\n")
            counts[split][label] += 1
        os.replace(staged, args.output)

    print(json.dumps({
        "output": str(args.output),
        "source_training_rows": len(raw_rows),
        "ferplus_training_rows": len(training_labels),
        "unique_images_after_deduplication": len(unique_images),
        "skipped_no_emotion_votes": skipped_no_emotion_votes,
        "skipped_low_agreement_or_nonface": skipped_low_agreement,
        "counts": counts,
        "label_type": "seven-class soft distributions from FER+ crowd votes",
    }, indent=2))


if __name__ == "__main__":
    main()
