#!/usr/bin/env python3
"""
Detect vertical shift lines in graph crops and update JSON metadata safely.

Key improvements over original:
  - Requires the candidate column to be a LOCAL PEAK in the column profile,
    not just the global maximum. A sharp absorption edge is flanked by other
    dark columns; a true vertical line stands alone.
  - Checks NEIGHBOR CONTRAST: the candidate column must be significantly
    darker than the mean of its immediate neighbors. Absorption edges fail
    this because adjacent columns are nearly as dark.
  - Enforces PIXEL CONTINUITY along the candidate column: dark pixels must
    form an unbroken (or near-unbroken) run, not a scattered profile that
    happens to sum high.
  - Raises MIN_COLUMN_RATIO to reduce easy false positives.
  - Adds a WIDTH GATE: after finding the peak column, it measures how many
    adjacent columns also exceed a high threshold. Real shift lines are 1-3px
    wide; peak edges spread across many columns.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


# ── tuneable constants ────────────────────────────────────────────────────────

# Fraction of column height that must be dark for the peak column to qualify.
# Raised from 0.40 to reduce easy false positives from shallow peaks.
MIN_COLUMN_RATIO = 0.55

# A true vertical marker must be at least this much darker than its neighbours.
# Expressed as a fraction of the peak column's own value.
# e.g. 0.35 means neighbours must average ≤ 65 % of the peak column's sum.
NEIGHBOR_CONTRAST_THRESHOLD = 0.35

# Half-width of the neighbourhood used for the contrast check (columns).
NEIGHBOR_HALF_WIDTH = 6

# Maximum number of columns (centred on peak) that may exceed
# MIN_COLUMN_RATIO * height before the candidate is rejected as a "broad feature"
# (i.e. a peak edge rather than a single vertical line).
MAX_LINE_WIDTH_PX = 4

# Minimum fraction of the crop height that must be covered by a CONTINUOUS run
# of dark pixels in the candidate column.
# A true vertical line is continuous; an absorption peak edge is not (the line
# only exists where the curve is near-vertical).
MIN_CONTINUITY_FRACTION = 0.55

# Minimum gap (in pixels) between a bright run break before it counts as
# discontinuous.  Small gaps from scan noise are tolerated.
CONTINUITY_GAP_TOLERANCE = 6


# ── helpers ───────────────────────────────────────────────────────────────────

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


# ── core detector ─────────────────────────────────────────────────────────────

def _largest_continuous_run(col_binary: np.ndarray, gap_tolerance: int) -> int:
    """
    Return the length of the largest continuous run of non-zero pixels in a
    1-D binary column, allowing internal gaps of up to `gap_tolerance` pixels.
    """
    pixels = col_binary.astype(bool)
    if not pixels.any():
        return 0

    best = 0
    run_start = None
    gap_count = 0

    for i, v in enumerate(pixels):
        if v:
            if run_start is None:
                run_start = i
            gap_count = 0
            best = max(best, i - run_start + 1)
        else:
            if run_start is not None:
                gap_count += 1
                if gap_count > gap_tolerance:
                    run_start = None
                    gap_count = 0

    return best


def detect_vertical_shift_line(gray_crop: np.ndarray):
    """
    Return the x-coordinate (relative to crop) of a vertical shift line,
    or None if no convincing line is found.

    Strategy
    --------
    1. Binarise with Otsu (ink = 255).
    2. Build a column-sum profile.
    3. Find the global peak column; reject if below MIN_COLUMN_RATIO.
    4. Reject if the peak is NOT a local maximum (i.e. a neighbouring column
       is as dark or darker — typical of a broad absorption edge).
    5. Reject if the peak column's sum is not sufficiently larger than the
       mean of its ±NEIGHBOR_HALF_WIDTH neighbours (contrast check).
    6. Measure the width of the "dark band" around the peak; reject if wider
       than MAX_LINE_WIDTH_PX (broad feature = peak edge, not a line).
    7. Check pixel continuity along the candidate column; reject if the
       longest unbroken dark run is shorter than MIN_CONTINUITY_FRACTION of
       the crop height.
    """
    if gray_crop.size == 0:
        return None

    height, width = gray_crop.shape

    # ── 1. binarise ──────────────────────────────────────────────────────────
    blur = cv2.GaussianBlur(gray_crop, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # ── 2. column profile ────────────────────────────────────────────────────
    profile = np.sum(binary.astype(np.float32), axis=0)  # shape (width,)
    if profile.size == 0:
        return None

    # ── 3. global peak + minimum height gate ────────────────────────────────
    peak_x = int(np.argmax(profile))
    peak_value = float(profile[peak_x])
    max_possible = 255.0 * height

    if peak_value < max_possible * MIN_COLUMN_RATIO:
        return None

    # ── 4. local-maximum check ───────────────────────────────────────────────
    # The peak column must be strictly greater than BOTH immediate neighbours.
    # An absorption edge has many adjacent columns with nearly identical sums.
    left_val  = float(profile[peak_x - 1]) if peak_x > 0          else 0.0
    right_val = float(profile[peak_x + 1]) if peak_x < width - 1  else 0.0

    if peak_value <= left_val or peak_value <= right_val:
        return None

    # ── 5. neighbour contrast check ──────────────────────────────────────────
    lo = max(0,         peak_x - NEIGHBOR_HALF_WIDTH)
    hi = min(width - 1, peak_x + NEIGHBOR_HALF_WIDTH)
    neighbour_indices = [i for i in range(lo, hi + 1) if i != peak_x]
    if neighbour_indices:
        neighbour_mean = float(np.mean(profile[neighbour_indices]))
        # neighbours must average less than (1 - threshold) * peak
        if neighbour_mean > peak_value * (1.0 - NEIGHBOR_CONTRAST_THRESHOLD):
            return None

    # ── 6. width gate ────────────────────────────────────────────────────────
    # Count how many consecutive columns around the peak also exceed the
    # minimum height ratio (i.e. are "nearly as dark").
    threshold_sum = max_possible * MIN_COLUMN_RATIO
    band_width = 1  # the peak column itself
    # expand left
    for dx in range(1, width):
        x = peak_x - dx
        if x < 0 or profile[x] < threshold_sum:
            break
        band_width += 1
    # expand right
    for dx in range(1, width):
        x = peak_x + dx
        if x >= width or profile[x] < threshold_sum:
            break
        band_width += 1

    if band_width > MAX_LINE_WIDTH_PX:
        return None

    # ── 7. continuity check ──────────────────────────────────────────────────
    col_pixels = binary[:, peak_x]
    longest_run = _largest_continuous_run(col_pixels, CONTINUITY_GAP_TOLERANCE)
    if longest_run < height * MIN_CONTINUITY_FRACTION:
        return None

    return peak_x


# ── file processing ───────────────────────────────────────────────────────────

def process_pair(png_path: Path, json_path: Path):
    data = safe_load_json(json_path)
    if data is None:
        return False

    plot_bounds = validate_plot_bounds(data)
    if plot_bounds is None:
        print(f"[SKIP] Missing or invalid plot_bounds in {json_path}")
        return False

    image_width  = data.get("image_width_px")
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
    crop_width  = x2 - x1

    data["vertical_markers_px"]         = [absolute_x]
    data["vertical_markers_relative_x"] = [round(relative_x / float(crop_width), 4)]

    safe_write_json(json_path, data)
    print(
        f"[UPDATE] Detected vertical shift line in {png_path.name}, "
        f"updated {json_path.name}"
    )
    return True


def main():
    args = parse_args()
    root_dir = Path(args.root).resolve()
    if not root_dir.is_dir():
        raise SystemExit(f"Root directory not found: {root_dir}")

    total = updated = skipped = 0

    for current_root, _, files in os.walk(str(root_dir)):
        for filename in sorted(files):
            if not filename.lower().endswith(".png"):
                continue
            png_path  = Path(current_root) / filename
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