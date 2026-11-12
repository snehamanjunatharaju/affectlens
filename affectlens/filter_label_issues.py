"""Create a reproducible copy with only top-ranked Cleanlab candidates removed."""

import argparse
import csv
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--issues", type=Path, default=Path("data/flagged_labels.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("reports/cleanlab-filtered-data.json"))
    parser.add_argument("--fraction", type=float, default=0.10, help="Maximum fraction of train+validation items to remove, ranked first")
    args = parser.parse_args()
    if not 0 < args.fraction < 1:
        parser.error("--fraction must be strictly between zero and one")
    if args.output.exists():
        parser.error(f"Output exists; choose a new directory: {args.output}")
    with args.issues.open(newline="", encoding="utf-8") as source:
        issues = list(csv.DictReader(source))
    total_train = sum(1 for split in ("train", "validation") for path in (args.data / split).rglob("*") if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    maximum = round(total_train * args.fraction)
    selected = issues[:maximum]
    removed = {(row["split"], row["relative_path"]) for row in selected}
    args.output.mkdir(parents=True)
    counts = {split: 0 for split in ("train", "validation", "test")}
    for split in counts:
        source_root = args.data / split
        if not source_root.is_dir():
            continue
        target_root = args.output / split
        target_root.mkdir(parents=True, exist_ok=True)
        for source in source_root.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(source_root).as_posix()
            if source.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and (split, f"{split}/{relative}") in removed:
                continue
            target = target_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.name == "soft_labels.jsonl":
                rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
                kept = [row for row in rows if (split, f"{split}/{row['file']}") not in removed]
                target.write_text("".join(json.dumps(row) + "\n" for row in kept), encoding="utf-8")
            else:
                shutil.copy2(source, target)
            if source.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                counts[split] += 1
    report = {"source": str(args.data), "issues": str(args.issues), "requested_fraction": args.fraction, "ranked_candidates_removed": len(selected), "removed_by_split": {split: sum(1 for item in selected if item["split"] == split) for split in ("train", "validation")}, "remaining_image_counts": counts, "output": str(args.output), "note": "Cleanlab flags are uncertain review candidates; only the top-ranked capped fraction is removed, and no labels are automatically rewritten."}
    (args.output / "filter-report.json").write_text(json.dumps(report, indent=2))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
