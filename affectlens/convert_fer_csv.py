"""Convert FER-2013 CSV rows into labeled image folders."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np

from .labels import LABELS

EMOTION_IDS = {
    0: "angry", 1: "disgust", 2: "fear", 3: "happy",
    4: "sad", 5: "surprise", 6: "neutral",
}
USAGE_SPLITS = {"training": "train", "publictest": "validation", "privatetest": "test"}
PIXEL_COUNT = 48 * 48


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="FER-2013 CSV obtained under its dataset terms")
    parser.add_argument("--output", type=Path, required=True, help="New output folder; existing folders are never overwritten")
    parser.add_argument(
        "--train-only-split", action="store_true",
        help="Create stratified train/validation/test subsets from a labeled emotion,pixels CSV",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train-only stratification")
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error(f"CSV file does not exist: {args.csv}")
    if args.output.exists():
        parser.error(f"Output already exists; choose a new folder: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        with args.csv.open("r", newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames:
                parser.error("CSV has no header")
            columns = {name.strip().casefold(): name for name in reader.fieldnames}
            if not {"emotion", "pixels"} <= columns.keys():
                parser.error("CSV must have emotion and pixels columns")
            has_usage = "usage" in columns
            if has_usage == args.train_only_split:
                parser.error("Use --train-only-split only with a CSV that has no Usage column")
            records = []
            for row_number, row in enumerate(reader, start=2):
                try:
                    emotion_id = int(row[columns["emotion"]])
                    pixel_text = row[columns["pixels"]]
                    pixels = np.fromstring(pixel_text, dtype=np.int16, sep=" ")
                except (TypeError, ValueError) as exc:
                    parser.error(f"Invalid emotion or pixels on CSV row {row_number}: {exc}")
                if emotion_id not in EMOTION_IDS:
                    parser.error(f"Unknown emotion value on CSV row {row_number}")
                if pixels.size != PIXEL_COUNT or np.any(pixels < 0) or np.any(pixels > 255):
                    parser.error(f"Expected exactly {PIXEL_COUNT} grayscale byte values on row {row_number}")
                usage = row[columns["usage"]].strip().casefold() if has_usage else None
                if has_usage and usage not in USAGE_SPLITS:
                    parser.error(f"Unknown usage value on CSV row {row_number}")
                records.append((row_number, emotion_id, pixels.astype(np.uint8), usage))
    except UnicodeDecodeError as exc:
        parser.error(f"CSV must be UTF-8 encoded: {exc}")

    if args.train_only_split:
        # Identical FER pixel rows must stay together; conflicting labels are omitted
        # rather than leaking between splits or teaching contradictory targets.
        pixel_groups = {}
        for record in records:
            digest = hashlib.sha256(record[2].tobytes()).digest()
            pixel_groups.setdefault(digest, []).append(record)
        records = []
        for group in pixel_groups.values():
            if len({record[1] for record in group}) == 1:
                records.append(group[0])
        by_class = {emotion_id: [] for emotion_id in EMOTION_IDS}
        for record in records:
            by_class[record[1]].append(record)
        rng = np.random.default_rng(args.seed)
        split_records = []
        for emotion_id, class_records in by_class.items():
            if len(class_records) < 3:
                parser.error(f"Training CSV needs at least 3 examples of emotion {emotion_id}")
            order = rng.permutation(len(class_records))
            n_validation = max(1, round(len(order) * 0.15))
            n_test = max(1, round(len(order) * 0.15))
            for rank, index in enumerate(order):
                split = "validation" if rank < n_validation else "test" if rank < n_validation + n_test else "train"
                row_number, label_id, pixels, _ = class_records[index]
                split_records.append((row_number, label_id, pixels, split))
        records = split_records

    counts = {split: {label: 0 for label in LABELS} for split in ("train", "validation", "test")}
    with tempfile.TemporaryDirectory(prefix=f".{args.output.name}-", dir=args.output.parent) as temporary:
        staged = Path(temporary)
        for row_number, emotion_id, pixels, usage in records:
            split = usage if args.train_only_split else USAGE_SPLITS[usage]
            label = EMOTION_IDS[emotion_id]
            target_dir = staged / split / label
            target_dir.mkdir(parents=True, exist_ok=True)
            success, encoded = cv2.imencode(".png", pixels.reshape(48, 48))
            if not success:
                parser.error(f"Could not encode image from CSV row {row_number}")
            (target_dir / f"fer2013_{row_number:06d}.png").write_bytes(encoded.tobytes())
            counts[split][label] += 1

        incomplete = {split: [label for label, count in values.items() if count == 0] for split, values in counts.items()}
        incomplete = {split: labels for split, labels in incomplete.items() if labels}
        if incomplete:
            parser.error(f"CSV is missing classes in required splits: {incomplete}")
        os.replace(staged, args.output)

    print(json.dumps({"output": str(args.output), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
