#!/usr/bin/env python3
"""Stage 3.0.5 - OpenCV plot-bounds detection + axis-call crop.

Runs after size measurement (stage 3) and BEFORE the axis/metadata call (3.1).

The plot-region pixel rectangle (``plot_bounds``) is pure geometry: a vision
model that sees a downscaled, tiled copy of the image cannot estimate it
reliably, which is the root cause of the pixel-cropping error in the autonomous
runs. This stage derives ``plot_bounds`` directly from the pixels with OpenCV
and records it in each graph JSON as the authoritative crop. By design it never
asks a language model and is never cross-checked against one.

It also writes an *edited* copy of the PNG - the plot cropped to the detected
bounds (plus a small margin so the axis tick values and titles stay visible) -
into a sibling ``axis_crops/`` folder. Stage 3.1 feeds that focused crop to the
vision model so the model reads axis labels/limits and the series count off the
plot itself rather than the surrounding page. The original ``graphs/`` PNG is
never modified, so digitization still runs on the untouched image using the
pixel coordinates detected here.

Bounds detection works for framed and open (axis-only) plots alike: a
horizontal axis line's horizontal extent fixes x1/x2 and a vertical axis line's
vertical extent fixes y1/y2, so one of each fully defines the box even when no
full frame is printed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "output_pdf"

# Frame/axis-line detection tuning.
BRIDGE_FRACTION = 0.02        # close gaps up to this fraction of the smaller side (tick breaks)
LINE_KERNEL_FRACTION = 0.3    # an axis line must span >= this fraction of width/height
LINE_COVERAGE_FRACTION = 0.3  # a row/col is an axis line if this fraction of it is ink
MIN_BOX_FRACTION = 0.2        # the detected box must be at least this fraction of each dimension

# Crop margin for the axis-call image, as a fraction of the plot size. Keeps the
# tick-value labels and axis titles (which sit just outside the frame) visible.
AXIS_CROP_MARGIN_FRACTION = 0.12


def _binarize_inv(gray: np.ndarray) -> np.ndarray:
    _, line_mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    return line_mask


def detect_plot_bounds(gray: np.ndarray):
    """Return the axis rectangle {x1, y1, x2, y2}, or None if uncertain.

    Long horizontal/vertical ink lines are the axes/frame. Bridging short gaps
    first (so a tick-broken baseline reconnects) then opening with a long line
    kernel keeps only those lines and drops the curve ink. The horizontal-line
    columns give x1/x2; the vertical-line rows give y1/y2. Returns None when no
    plausible box emerges, so the caller can keep the model's bounds rather than
    crop on a bad guess.
    """
    if gray.ndim != 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if height < 20 or width < 20:
        return None

    line_mask = _binarize_inv(gray)

    bridge = max(8, int(round(min(height, width) * BRIDGE_FRACTION)))
    closed_h = cv2.morphologyEx(
        line_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (bridge, 1)),
    )
    closed_v = cv2.morphologyEx(
        line_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, bridge)),
    )

    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(10, int(width * LINE_KERNEL_FRACTION)), 1)
    )
    v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(10, int(height * LINE_KERNEL_FRACTION)))
    )
    horiz = cv2.morphologyEx(closed_h, cv2.MORPH_OPEN, h_kernel)
    vert = cv2.morphologyEx(closed_v, cv2.MORPH_OPEN, v_kernel)

    h_rows = np.flatnonzero((horiz.sum(axis=1) / 255.0) >= LINE_COVERAGE_FRACTION * width)
    v_cols = np.flatnonzero((vert.sum(axis=0) / 255.0) >= LINE_COVERAGE_FRACTION * height)
    if h_rows.size == 0 or v_cols.size == 0:
        return None

    # x-extent of the horizontal axis line(s); y-extent of the vertical axis line(s).
    x_cols = np.flatnonzero(horiz[h_rows].any(axis=0))
    y_rows = np.flatnonzero(vert[:, v_cols].any(axis=1))
    if x_cols.size == 0 or y_rows.size == 0:
        return None

    x1, x2 = int(x_cols.min()), int(x_cols.max())
    y1, y2 = int(y_rows.min()), int(y_rows.max())

    if (x2 - x1) < MIN_BOX_FRACTION * width or (y2 - y1) < MIN_BOX_FRACTION * height:
        return None

    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def write_axis_crop(gray_or_rgb_path: Path, bounds: dict, crop_path: Path,
                    image_width: int, image_height: int) -> bool:
    """Write the bounds-plus-margin crop used only for the axis call."""
    margin_x = int(round((bounds["x2"] - bounds["x1"]) * AXIS_CROP_MARGIN_FRACTION))
    margin_y = int(round((bounds["y2"] - bounds["y1"]) * AXIS_CROP_MARGIN_FRACTION))
    cx1 = max(0, bounds["x1"] - margin_x)
    cy1 = max(0, bounds["y1"] - margin_y)
    cx2 = min(image_width, bounds["x2"] + margin_x)
    cy2 = min(image_height, bounds["y2"] + margin_y)
    if cx2 - cx1 < 5 or cy2 - cy1 < 5:
        return False
    with Image.open(gray_or_rgb_path) as img:
        crop = img.crop((cx1, cy1, cx2, cy2))
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(crop_path)
    return True


# --------------------------------------------------------------------------- #
# Tree walking / JSON update
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect plot_bounds with OpenCV and write the axis-call crop."
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        default=None,
        help="Path to the root output_pdf/ directory (default: output_pdf at the repo root)",
    )
    parser.add_argument(
        "--single",
        metavar="PNG_PATH",
        help="Process one specific graph PNG path instead of walking the full tree",
    )
    return parser.parse_args()


def safe_load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def find_graphs_folders(root_dir: str):
    for current_root, dirs, _ in os.walk(root_dir):
        if os.path.basename(current_root).lower() == "graphs":
            continue
        if "graphs" not in dirs:
            continue
        graphs_path = os.path.join(current_root, "graphs")
        dirs.remove("graphs")
        if os.path.isdir(graphs_path):
            yield graphs_path


def axis_crop_path_for(png_path: Path) -> Path:
    # Sibling of graphs/, never inside it, so later stages that walk graphs/*.png
    # do not mistake the crop for a graph.
    page_dir = png_path.parent.parent
    return page_dir / "axis_crops" / png_path.name


def process_png_file(png_path: Path, summary: dict):
    json_path = png_path.with_suffix(".json")
    metadata = safe_load_json(json_path)
    filename = png_path.name

    if metadata is None or metadata.get("image_measured") is not True:
        print(f"Skipping {filename} - run 3_png_size_prep.py first.")
        summary["skipped"] += 1
        return

    image_width = int(metadata.get("image_width_px", 0))
    image_height = int(metadata.get("image_height_px", 0))
    if image_width <= 0 or image_height <= 0:
        print(f"Skipping {filename} - invalid image dimensions in JSON.")
        summary["skipped"] += 1
        return

    try:
        gray = np.array(Image.open(png_path).convert("L"))
    except Exception:
        print(f"Skipping {filename} - could not read PNG.")
        summary["failed"] += 1
        return

    bounds = detect_plot_bounds(gray)
    if bounds is None:
        metadata["plot_bounds_source"] = "llm"
        metadata["axis_crop_written"] = False
        print(f"  {filename}: no axis box found; axis call will use the full image.")
        try:
            json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            summary["no_bounds"] += 1
        except OSError:
            summary["failed"] += 1
        return

    metadata["plot_bounds"] = bounds
    metadata["plot_bounds_source"] = "opencv"

    crop_path = axis_crop_path_for(png_path)
    try:
        wrote = write_axis_crop(png_path, bounds, crop_path, image_width, image_height)
    except Exception:
        wrote = False
    metadata["axis_crop_written"] = bool(wrote)
    if not wrote:
        print(f"  {filename}: bounds {bounds} found but crop not written.")

    try:
        json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        summary["processed"] += 1
        print(f"  {filename}: plot_bounds {bounds} (opencv)")
    except OSError:
        print(f"  {filename}: failed to write JSON.")
        summary["failed"] += 1


def main():
    args = parse_args()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_dir = os.path.abspath(
        args.root_dir if args.root_dir else os.path.join(repo_root, "output_pdf")
    )
    if not os.path.isdir(root_dir):
        raise SystemExit(f"Root directory not found: {root_dir}")

    summary = {"total_pngs": 0, "processed": 0, "no_bounds": 0, "skipped": 0, "failed": 0}

    if args.single:
        png_path = Path(args.single)
        if not png_path.is_absolute():
            png_path = Path(root_dir) / args.single
        png_path = png_path.resolve()
        if not (png_path.is_file() and png_path.parent.name.lower() == "graphs"):
            raise SystemExit(
                f"Single file must be an existing PNG inside a graphs/ folder: {args.single}"
            )
        summary["total_pngs"] = 1
        process_png_file(png_path, summary)
    else:
        for graphs_path in find_graphs_folders(root_dir):
            for entry in sorted(os.listdir(graphs_path)):
                if not entry.lower().endswith(".png"):
                    continue
                summary["total_pngs"] += 1
                process_png_file(Path(graphs_path) / entry, summary)

    print("\nOpenCV geometry complete.")
    print(f"Total PNGs found in graphs/ folders: {summary['total_pngs']}")
    print(f"Bounds detected (opencv): {summary['processed']}")
    print(f"No box found (axis call uses full image): {summary['no_bounds']}")
    print(f"Skipped (not measured): {summary['skipped']}")
    print(f"Failed: {summary['failed']}")


if __name__ == "__main__":
    main()
