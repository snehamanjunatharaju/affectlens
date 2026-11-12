"""Render calibration and robustness reports as publication-friendly PNGs."""

import argparse
import json
from pathlib import Path


def reliability(report: dict, output: Path):
    from PIL import Image, ImageDraw, ImageFont

    bins = report.get("reliability_bins") or report.get("test", {}).get("reliability_bins")
    if not bins:
        raise ValueError("Report has no reliability_bins")
    plotted = [row for row in bins if row.get("count", 0) and row.get("accuracy") is not None]
    if not plotted:
        raise ValueError("No populated reliability bins")
    confidence = [row["confidence"] for row in plotted]
    accuracy = [row["accuracy"] for row in plotted]
    counts = [row["count"] for row in plotted]
    ece = report.get("ece", report.get("test", {}).get("ece"))
    width, height = 800, 680
    left, top, right, bottom = 105, 90, 740, 590
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((left, 30), f"Reliability diagram · ECE={ece:.3f}" if ece is not None else "Reliability diagram", fill="#1f2937", font=font)
    for tick in range(6):
        value = tick / 5
        x = left + value * (right - left)
        y = bottom - value * (bottom - top)
        draw.line((x, top, x, bottom), fill="#e4e8ee", width=1)
        draw.line((left, y, right, y), fill="#e4e8ee", width=1)
        draw.text((x - 8, bottom + 12), f"{value:.1f}", fill="#354254", font=font)
        draw.text((left - 40, y - 5), f"{value:.1f}", fill="#354254", font=font)
    draw.line((left, bottom, right, top), fill="#8090a0", width=2)
    points = [(left + conf * (right - left), bottom - acc * (bottom - top)) for conf, acc in zip(confidence, accuracy)]
    if len(points) > 1:
        draw.line(points, fill="#78a52d", width=4)
    for (x, y), count in zip(points, counts):
        radius = max(4, min(11, count // 30))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#a8cf51", outline="#42651b", width=1)
    draw.text((320, 635), "Mean calibrated confidence", fill="#354254", font=font)
    draw.text((18, 80), "Accuracy", fill="#354254", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reliability(json.loads(args.report.read_text()), args.output)
    print(f"Reliability diagram written to {args.output}")


if __name__ == "__main__":
    main()
