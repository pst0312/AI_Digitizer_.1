#!/usr/bin/env python3
"""
Detect vertical shift lines (solid or dashed) in graph crops and update JSON metadata safely.
"""

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np


# ── tuneable constants ────────────────────────────────────────────────────────

# Lowered from 0.55 to 0.25 to accommodate the empty gaps in dashed lines.
MIN_COLUMN_RATIO = 0.25

# A true vertical marker must be at least this much darker than its neighbours.
NEIGHBOR_CONTRAST_THRESHOLD = 0.35
NEIGHBOR_HALF_WIDTH = 6

# Maximum number of columns (centred on peak) that may exceed a relative width threshold.
MAX_LINE_WIDTH_PX = 4

# Minimum vertical coverage across the entire column height (for dashes + solids)
MIN_VERTICAL_SPAN_FRACTION = 0.85

# The plot-box frame is itself a full-height column that would pass every gate.
# Ignore peak candidates within this margin of either crop edge.
EDGE_EXCLUSION_FRACTION = 0.02
EDGE_EXCLUSION_MIN_PX = 5


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

def _validate_vertical_distribution(col_binary: np.ndarray) -> bool:
    """
    Checks if the column acts like a global vertical line (solid or dashed).
    Ensures pixels span from top to bottom and occupy multiple vertical sections.
    """
    height = len(col_binary)
    nz_indices = np.where(col_binary > 0)[0]
    if len(nz_indices) == 0:
        return False

    # 1. Vertical Span Check: First to last pixel must cover most of the height
    span = (nz_indices[-1] - nz_indices[0]) / height
    if span < MIN_VERTICAL_SPAN_FRACTION:
        return False

    # 2. Zone Distribution Check: Divide column into 4 vertical zones.
    # A true shift line intersects almost all zones, whereas localized curves do not.
    zones = np.array_split(col_binary, 4)
    populated_zones = sum(1 for zone in zones if np.any(zone > 0))
    if populated_zones < 3:
        return False

    return True


def _candidate_passes_gates(
    profile: np.ndarray, binary: np.ndarray, peak_x: int, width: int
) -> bool:
    """Run the per-column tests that distinguish a true shift line from noise."""
    peak_value = float(profile[peak_x])

    # ── local-maximum check ──────────────────────────────────────────────────
    left_val  = float(profile[peak_x - 1]) if peak_x > 0          else 0.0
    right_val = float(profile[peak_x + 1]) if peak_x < width - 1  else 0.0
    if peak_value <= left_val or peak_value <= right_val:
        return False

    # ── neighbour contrast check ─────────────────────────────────────────────
    lo = max(0,         peak_x - NEIGHBOR_HALF_WIDTH)
    hi = min(width - 1, peak_x + NEIGHBOR_HALF_WIDTH)
    neighbour_indices = [i for i in range(lo, hi + 1) if i != peak_x]
    if neighbour_indices:
        neighbour_mean = float(np.mean(profile[neighbour_indices]))
        if neighbour_mean > peak_value * (1.0 - NEIGHBOR_CONTRAST_THRESHOLD):
            return False

    # ── relative width gate ──────────────────────────────────────────────────
    # We measure width relative to the peak value itself instead of a global floor.
    # This prevents noise from expanding the width bounds of low-density dashed lines.
    threshold_sum = peak_value * 0.50
    band_width = 1
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
        return False

    # ── distribution and span check ──────────────────────────────────────────
    if not _validate_vertical_distribution(binary[:, peak_x]):
        return False

    return True


def detect_vertical_shift_line(gray_crop: np.ndarray):
    """
    Return the x-coordinate (relative to crop) of a vertical shift line,
    or None if no convincing line is found.
    """
    if gray_crop.size == 0:
        return None

    height, width = gray_crop.shape

    # ── 1. binarise (using a smaller blur to keep thin dashed lines distinct) ──
    blur = cv2.GaussianBlur(gray_crop, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # ── 2. column profile ────────────────────────────────────────────────────
    profile = np.sum(binary.astype(np.float32), axis=0)  # shape (width,)
    if profile.size == 0:
        return None

    max_possible = 255.0 * height
    min_peak = max_possible * MIN_COLUMN_RATIO

    # ── 3. exclude the plot-frame borders ────────────────────────────────────
    # The plot box's own left/right edges are full-height columns that pass every
    # gate, so suppress candidates within a margin of either crop edge.
    edge_margin = max(EDGE_EXCLUSION_MIN_PX, int(round(width * EDGE_EXCLUSION_FRACTION)))
    working = profile.copy()
    working[:edge_margin] = 0.0
    if edge_margin > 0:
        working[width - edge_margin:] = 0.0

    # ── 4. evaluate candidate peaks strongest-first ──────────────────────────
    # A failed gate only rules out that one column (e.g. a sharp curve peak), so
    # suppress it and fall through to the next-best candidate. The minimum-height
    # gate is monotonic, so once the strongest remaining column is too short no
    # later candidate can qualify and we stop.
    for _ in range(width):
        peak_x = int(np.argmax(working))
        peak_value = float(working[peak_x])
        if peak_value < min_peak:
            return None
        if _candidate_passes_gates(profile, binary, peak_x, width):
            return peak_x
        lo = max(0, peak_x - 1)
        hi = min(width, peak_x + 2)
        working[lo:hi] = 0.0

    return None


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
        # Clear any stale markers so downstream stages don't mask a phantom line.
        if data.get("vertical_markers_px") or data.get("vertical_markers_relative_x"):
            data["vertical_markers_px"] = []
            data["vertical_markers_relative_x"] = []
            safe_write_json(json_path, data)
            print(f"[UPDATE] Cleared stale vertical markers in {json_path.name}")
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