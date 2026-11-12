"""Audit a FER-style image folder without exporting or storing image contents."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .labels import ALIASES, IMAGE_EXTENSIONS, LABELS


def audit_dataset(root: Path) -> dict:
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    digest_records: dict[str, list[tuple[str, str]]] = defaultdict(list)
    output = {
        "dataset": root.name,
        "splits": {},
        "duplicate_content_groups": 0,
        "cross_split_duplicate_groups": 0,
        "conflicting_label_groups": 0,
        "warnings": [],
    }
    split_names = ["train"]
    if (root / "validation").is_dir():
        split_names.append("validation")
    split_names.append("test")
    for split in split_names:
        split_root = root / split
        if not split_root.is_dir():
            output["warnings"].append(f"Missing split directory: {split}")
            continue
        counts = Counter()
        dimensions = Counter()
        corrupt = 0
        unrecognized = []
        for folder in sorted(split_root.iterdir()):
            if not folder.is_dir():
                continue
            canonical = ALIASES.get(folder.name.lower())
            if canonical is None:
                unrecognized.append(folder.name)
                continue
            for path in folder.rglob("*"):
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                counts[canonical] += 1
                payload = path.read_bytes()
                digest_records[hashlib.sha256(payload).hexdigest()].append((split, canonical))
                image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    corrupt += 1
                else:
                    height, width = image.shape[:2]
                    dimensions[f"{width}x{height}"] += 1
        missing = [label for label in LABELS if counts[label] == 0]
        output["splits"][split] = {
            "total_images": int(sum(counts.values())),
            "class_counts": {label: int(counts[label]) for label in LABELS},
            "missing_classes": missing,
            "corrupt_images": corrupt,
            "dimensions": dict(sorted(dimensions.items())),
            "unrecognized_class_folders": unrecognized,
        }
        if missing:
            output["warnings"].append(f"{split} is missing classes: {', '.join(missing)}")
        if corrupt:
            output["warnings"].append(f"{split} contains {corrupt} unreadable image(s)")
    duplicate_hashes = [records for records in digest_records.values() if len(records) > 1]
    output["duplicate_content_groups"] = len(duplicate_hashes)
    output["cross_split_duplicate_groups"] = sum(len({split for split, _ in records}) > 1 for records in duplicate_hashes)
    output["conflicting_label_groups"] = sum(
        len({label for _, label in records}) > 1 for records in duplicate_hashes
    )
    for split in output["splits"]:
        output["splits"][split]["duplicate_content_groups"] = sum(
            sum(record_split == split for record_split, _ in records) > 1 for records in duplicate_hashes
        )
    if output["cross_split_duplicate_groups"]:
        output["warnings"].append("Exact duplicate image content occurs across dataset splits; investigate split leakage")
    if output["conflicting_label_groups"]:
        output["warnings"].append("Some exact duplicate images have conflicting labels; investigate annotation quality")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Folder containing train/ and test/")
    parser.add_argument("--output", type=Path, help="Optional JSON report path; otherwise print the report")
    args = parser.parse_args()
    report = audit_dataset(args.data)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
        print(f"Audit report written to {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
