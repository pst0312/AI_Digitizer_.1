#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


MIN_COLUMN_RATIO = 0.4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect vertical shift lines in graph crops and update JSON metadata safely."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="output_pdf",
        help="Root folder containing output_pdf/ structure (default: output_pdf)",
    )
    return parser.parse_args()


def safe_load_json(json_path: Path):
    try:
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[ERROR] Failed to read JSON {json_path}: {exc}")
        return None


def safe_write_json(json_path: Path, data: dict):
    try:
        with json_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2))
            f.write("\n")
    except Exception as exc:
        print(f"[ERROR] Failed to write JSON {json_path}: {exc}")


def validate_plot_bounds(data: dict):
    plot_bounds = data.get("plot_bounds")
    if not isinstance(plot_bounds, dict):
        return None
    for key in ("x1", "y1", "x2", "y2"):
        if key not in plot_bounds:
            return None
    return plot_bounds


def detect_vertical_shift_line(gray_crop: np.ndarray):
    if gray_crop.size == 0:
        return None

    blur = cv2.GaussianBlur(gray_crop, (5, 5), 0)
    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    profile = np.sum(binary.astype(np.uint32), axis=0)
    if profile.size == 0:
        return None

    peak_x = int(np.argmax(profile))
    peak_value = int(profile[peak_x])
    height = int(gray_crop.shape[0])
    max_possible = 255 * height
    if peak_value < int(round(max_possible * MIN_COLUMN_RATIO)):
        return None

    return peak_x


def process_pair(png_path: Path, json_path: Path):
    data = safe_load_json(json_path)
    if data is None:
        return False

    plot_bounds = validate_plot_bounds(data)
    if plot_bounds is None:
        print(f"[SKIP] Missing or invalid plot_bounds in {json_path}")
        return False

    image_width = data.get("image_width_px")
    image_height = data.get("image_height_px")
    if not isinstance(image_width, int) or not isinstance(image_height, int):
        print(f"[SKIP] Missing image_width_px/image_height_px in {json_path}")
        return False

    x1 = int(round(plot_bounds["x1"]))
    y1 = int(round(plot_bounds["y1"]))
    x2 = int(round(plot_bounds["x2"]))
    y2 = int(round(plot_bounds["y2"]))

    if x2 <= x1 or y2 <= y1:
        print(f"[SKIP] Invalid plot_bounds rectangle in {json_path}")
        return False

    try:
        image = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Unable to open PNG")
    except Exception as exc:
        print(f"[SKIP] Failed to load image {png_path}: {exc}")
        return False

    if image.shape[1] != image_width or image.shape[0] != image_height:
        print(f"[WARN] Image dimensions differ from JSON for {png_path}")

    if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
        print(f"[SKIP] plot_bounds out of bounds for {png_path}")
        return False

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        print(f"[SKIP] Empty crop for {png_path}")
        return False

    relative_x = detect_vertical_shift_line(crop)
    if relative_x is None:
        print(f"[SKIP] No valid vertical shift line found in {png_path}")
        return False

    absolute_x = x1 + relative_x
    crop_width = x2 - x1
    if crop_width <= 0:
        print(f"[SKIP] Invalid crop width for {png_path}")
        return False

    data["vertical_markers_px"] = [absolute_x]
    data["vertical_markers_relative_x"] = [round(relative_x / float(crop_width), 4)]

    safe_write_json(json_path, data)
    print(f"[UPDATE] Detected vertical shift line in {png_path.name}, updated {json_path.name}")
    return True


def main():
    args = parse_args()
    root_dir = Path(args.root).resolve()
    if not root_dir.is_dir():
        raise SystemExit(f"Root directory not found: {root_dir}")

    total = 0
    updated = 0
    skipped = 0

    for current_root, _, files in os.walk(str(root_dir)):
        for filename in sorted(files):
            if not filename.lower().endswith(".png"):
                continue
            png_path = Path(current_root) / filename
            json_path = png_path.with_suffix(".json")
            if not json_path.exists():
                continue

            total += 1
            if process_pair(png_path, json_path):
                updated += 1
            else:
                skipped += 1

    print(f"\nScanned {total} PNG/JSON pairs: updated={updated}, skipped={skipped}")


if __name__ == "__main__":
    main()
