#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


DEFAULT_ROOT = Path(__file__).parent / "output_pdf"

# Run/tracking parameters for geometric series separation
RUN_GAP_PX = 2                     # vertical gap that splits one column run into two
MAX_RUN_HEIGHT_FRACTION = 0.85     # runs taller than this fraction of plot height are axis junk
STRUCTURAL_LINE_FRACTION = 0.7     # lines spanning this fraction of width/height are frame, not data
EDGE_BAND_FRACTION = 0.03          # runs entirely inside this top/bottom edge band are frame remnants
BASE_MATCH_TOL_FRACTION = 0.025    # matching tolerance right after a confirmed point
GAP_TOL_GROWTH_FRACTION = 0.015    # tolerance growth per skipped column (dashed gaps)
MAX_MATCH_TOL_FRACTION = 0.15      # tolerance ceiling


def parse_args():
    parser = argparse.ArgumentParser(
        description="Digitize graphs by intensity clustering and skeleton pixel tracing"
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=None,
        help="Root output_pdf folder (default: output_pdf)",
    )
    parser.add_argument(
        "--single",
        metavar="PNG_PATH",
        help="Process one specific graph PNG path instead of walking the full tree",
    )
    return parser.parse_args()


def is_graphs_png(png_path: Path) -> bool:
    return (
        png_path.is_file()
        and png_path.suffix.lower() == ".png"
        and png_path.parent.name.lower() == "graphs"
    )


def find_graphs_folders(root_dir: Path):
    for current_root, dirs, _ in os.walk(str(root_dir)):
        if os.path.basename(current_root).lower() == "graphs":
            continue
        if "graphs" not in dirs:
            continue
        graphs_path = Path(current_root) / "graphs"
        dirs.remove("graphs")
        if graphs_path.is_dir():
            yield graphs_path


def resolve_single_path(single_arg: str, root_dir: Path) -> Path:
    candidate = Path(single_arg)
    if not candidate.is_absolute():
        candidate = root_dir / single_arg
    return candidate.resolve()


def safe_load_json(json_path: Path):
    if not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def clamp_plot_bounds(bounds: dict, image_width: int, image_height: int):
    x1 = int(round(bounds.get("x1", 0)))
    y1 = int(round(bounds.get("y1", 0)))
    x2 = int(round(bounds.get("x2", 0)))
    y2 = int(round(bounds.get("y2", 0)))

    x1 = max(0, min(x1, image_width - 1))
    x2 = max(0, min(x2, image_width))
    y1 = max(0, min(y1, image_height - 1))
    y2 = max(0, min(y2, image_height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid plot_bounds after clamping: {(x1, y1, x2, y2)} "
            f"for image size {(image_width, image_height)}"
        )

    return x1, y1, x2, y2


def crop_plot(image: np.ndarray, bounds: dict, image_width: int, image_height: int):
    x1, y1, x2, y2 = clamp_plot_bounds(bounds, image_width, image_height)
    return image[y1:y2, x1:x2], x1, y1, x2, y2


def detect_vertical_markers(gray_crop: np.ndarray, manual_markers=None):
    height, width = gray_crop.shape[:2]
    if height < 5 or width < 5:
        return [], np.zeros((height, width), dtype=np.uint8), gray_crop.copy()

    marker_mask = np.zeros((height, width), dtype=np.uint8)
    marker_positions = []

    # Scenario A: Use explicit manual coordinates from your JSON if provided
    if manual_markers and len(manual_markers) > 0:
        marker_positions = sorted(list(set([int(round(x)) for x in manual_markers])))
        
        # Draw vertical mask bands around each known shift marker location
        # A thickness buffer of 3-5 pixels ensures we catch bleeding pixels
        buffer_px = 3 
        for x_pos in marker_positions:
            x1 = max(0, x_pos - buffer_px)
            x2 = min(width, x_pos + buffer_px + 1)
            cv2.rectangle(marker_mask, (x1, 0), (x2, height), 255, thickness=-1)

    # Scenario B: Fall back to automatic morphology if JSON has no markers
    else:
        _, bin_inv = cv2.threshold(
            gray_crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        kernel_height = max(3, int(round(height * 0.6)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))
        vertical_candidates = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            vertical_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h < int(round(height * 0.6)):
                continue
            cv2.drawContours(marker_mask, [contour], -1, 255, thickness=-1)
            marker_positions.append(int(round(x + w / 2)))
        marker_positions = sorted(set(marker_positions))

    # Inpaint the regions to seamlessly wipe them out from the data path tracing
    if np.any(marker_mask):
        inpainted = cv2.inpaint(gray_crop, marker_mask, 5, cv2.INPAINT_TELEA)
    else:
        inpainted = gray_crop.copy()

    return marker_positions, marker_mask, inpainted

def extract_dark_pixels(gray_image: np.ndarray):
    _, line_mask = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    ys, xs = np.where(line_mask > 0)
    return xs, ys, line_mask


def remove_structural_lines(line_mask: np.ndarray) -> np.ndarray:
    """Subtract long horizontal/vertical lines (plot frame, gridlines) from the mask."""
    height, width = line_mask.shape[:2]
    h_len = max(3, int(round(width * STRUCTURAL_LINE_FRACTION)))
    v_len = max(3, int(round(height * STRUCTURAL_LINE_FRACTION)))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    horiz = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, horiz_kernel)
    vert = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, vert_kernel)
    cleaned = cv2.subtract(line_mask, cv2.add(horiz, vert))
    return cleaned if np.any(cleaned) else line_mask


def extract_column_runs(line_mask: np.ndarray):
    """For each column, collapse contiguous dark-pixel runs to (center, top, bottom)."""
    height, width = line_mask.shape[:2]
    max_run_height = max(3, int(round(height * MAX_RUN_HEIGHT_FRACTION)))
    edge_band = max(2, int(round(height * EDGE_BAND_FRACTION)))
    runs_per_column = []
    for x in range(width):
        ys = np.flatnonzero(line_mask[:, x])
        if ys.size == 0:
            runs_per_column.append([])
            continue
        split_points = np.flatnonzero(np.diff(ys) > RUN_GAP_PX)
        segments = np.split(ys, split_points + 1)
        runs = [
            (float(seg.mean()), float(seg[0]), float(seg[-1]))
            for seg in segments
            if seg.size <= max_run_height
            and seg[-1] >= edge_band
            and seg[0] <= height - 1 - edge_band
        ]
        runs_per_column.append(runs)
    return runs_per_column


def find_seed_column(runs_per_column, num_series: int):
    """Pick the middle of the longest stretch of columns with exactly num_series runs."""
    best_start, best_length = -1, 0
    start = -1
    for x, runs in enumerate(runs_per_column + [[]]):
        if len(runs) == num_series:
            if start < 0:
                start = x
        elif start >= 0:
            if x - start > best_length:
                best_start, best_length = start, x - start
            start = -1
    if best_length == 0:
        return None
    return best_start + best_length // 2


def match_runs_to_series(runs, last_intervals, last_x, x, height, min_share_gap):
    """Assign column runs to series by proximity to each series' last matched run.

    Distance is measured between [top, bottom] intervals, so a series that just
    traversed a tall near-vertical stroke (sharp peak/dip) can continue from
    either end of it.
    """
    num_series = len(last_intervals)
    base_tol = height * BASE_MATCH_TOL_FRACTION
    growth = height * GAP_TOL_GROWTH_FRACTION
    tol_cap = height * MAX_MATCH_TOL_FRACTION

    gaps = [abs(x - last_x[s]) for s in range(num_series)]
    tolerances = [
        min(tol_cap, base_tol + growth * max(0, gaps[s] - 1))
        for s in range(num_series)
    ]

    def interval_distance(series_idx, run_idx):
        _, run_top, run_bottom = runs[run_idx]
        last_top, last_bottom = last_intervals[series_idx]
        if run_top > last_bottom:
            return run_top - last_bottom
        if last_top > run_bottom:
            return last_top - run_bottom
        return 0.0

    candidates = sorted(
        (interval_distance(s, r), s, r)
        for s in range(num_series)
        for r in range(len(runs))
        if interval_distance(s, r) <= tolerances[s]
    )

    # First pass: unique greedy matching so re-splitting curves separate again
    assignment = {}
    used_runs = set()
    for dist, s, r in candidates:
        if s in assignment or r in used_runs:
            continue
        assignment[s] = r
        used_runs.add(r)

    # Second pass: let unmatched series share an already claimed run, either at
    # tight distance (curves gradually merging) or after a long gap (series lost
    # its curve and is recovering). Short dashed-gap columns stay unassigned so a
    # dotted series does not zigzag onto a close neighboring curve.
    for dist, s, r in candidates:
        if s in assignment:
            continue
        if dist <= base_tol or gaps[s] >= min_share_gap:
            assignment[s] = r

    return assignment


def cluster_series(gray_image: np.ndarray, num_series: int):
    xs, ys, line_mask = extract_dark_pixels(gray_image)
    if xs.size == 0:
        raise ValueError("No dark pixels found inside plot crop")

    height, width = gray_image.shape[:2]
    line_mask = remove_structural_lines(line_mask)
    runs_per_column = extract_column_runs(line_mask)

    seed_x = find_seed_column(runs_per_column, num_series)
    if seed_x is None:
        raise ValueError(
            f"No column with exactly {num_series} separable runs; cannot seed tracking"
        )

    series_y = np.full((num_series, width), np.nan, dtype=np.float64)

    # Seed assignment: series ordered top-to-bottom at the seed column
    seed_runs = sorted(runs_per_column[seed_x], key=lambda run: run[0])
    for s in range(num_series):
        series_y[s, seed_x] = seed_runs[s][0]

    min_share_gap = max(10, int(round(width * 0.02)))

    # Track outward from the seed in both directions
    for direction in (1, -1):
        last_intervals = [(run[1], run[2]) for run in seed_runs]
        last_x = [seed_x] * num_series
        x = seed_x + direction
        while 0 <= x < width:
            runs = runs_per_column[x]
            if runs:
                assignment = match_runs_to_series(
                    runs, last_intervals, last_x, x, height, min_share_gap
                )
                for s, r in assignment.items():
                    series_y[s, x] = runs[r][0]
                    last_intervals[s] = (runs[r][1], runs[r][2])
                    last_x[s] = x
            x += direction

    clusters = []
    for cluster_id in range(num_series):
        row = series_y[cluster_id]
        assigned = np.flatnonzero(~np.isnan(row))
        point_count = int(assigned.size)
        coverage_ratio = float(point_count) / float(width) if width > 0 else 0.0

        if point_count == 0:
            clusters.append(
                {
                    "id": cluster_id + 1,
                    "style": "solid",
                    "intensity_center": 0.0,
                    "point_count": 0,
                    "coverage_ratio": 0.0,
                    "pixels": [],
                }
            )
            continue

        sample_y = np.clip(np.round(row[assigned]).astype(int), 0, height - 1)
        intensity_center = float(gray_image[sample_y, assigned].mean())

        filled = np.interp(np.arange(width), assigned, row[assigned])
        points = [
            (
                float(x) / float(width),
                1.0 - float(y_val) / float(height),
            )
            for x, y_val in enumerate(filled)
        ]

        style = "solid" if coverage_ratio > 0.85 else "dashed"

        clusters.append(
            {
                "id": cluster_id + 1,
                "style": style,
                "intensity_center": float(round(intensity_center, 2)),
                "point_count": point_count,
                "coverage_ratio": float(round(coverage_ratio, 4)),
                "pixels": points,
            }
        )

    return clusters


def write_series_csv(csv_path: Path, points):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write("x_relative,y_relative\n")
        for x_rel, y_rel in points:
            f.write(f"{x_rel:.6f},{y_rel:.6f}\n")


def process_graph_png(png_path: Path, json_path: Path, root_dir: Path, summary: dict):
    metadata = safe_load_json(json_path)
    if metadata is None:
        summary["skipped"] += 1
        return

    if metadata.get("vision_extracted") is not True or metadata.get("image_measured") is not True:
        summary["skipped"] += 1
        return

    plot_bounds = metadata.get("plot_bounds")
    if not isinstance(plot_bounds, dict):
        summary["skipped"] += 1
        return

    num_series = metadata.get("num_series")
    try:
        num_series = int(num_series)
    except Exception:
        summary["skipped"] += 1
        return

    if num_series <= 0:
        summary["skipped"] += 1
        return

    image_width = int(metadata.get("image_width_px", 0))
    image_height = int(metadata.get("image_height_px", 0))
    if image_width <= 0 or image_height <= 0:
        summary["skipped"] += 1
        return

    try:
        with Image.open(png_path) as img:
            img = img.convert("L")
            image = np.array(img)
    except Exception:
        summary["failed"] += 1
        return

    try:
        cropped, x1, y1, x2, y2 = crop_plot(image, plot_bounds, image_width, image_height)
    except Exception:
        summary["failed"] += 1
        return

    try:
        # Check if manual shift markers exist in your JSON metadata
        # Adjust the key name ("shift_markers_x" or "vertical_markers_px") to match your exact JSON structure
        manual_markers = metadata.get("vertical_markers_px", [])
        
        vertical_markers_px, marker_mask, cleaned = detect_vertical_markers(
            cropped, 
            manual_markers=manual_markers
        )
    except Exception:
        summary["failed"] += 1
        return

    try:
        clusters = cluster_series(cleaned, num_series)
    except Exception:
        summary["failed"] += 1
        return

    page_dir = png_path.parent.parent
    digitized_dir = page_dir / "digitized"
    csv_entries = []
    for cluster in clusters:
        csv_name = f"{png_path.stem}_series_{cluster['id']}_{cluster['style']}.csv"
        csv_path = digitized_dir / csv_name
        write_series_csv(csv_path, cluster["pixels"])
        cluster["csv_path"] = str(Path("digitized") / csv_name).replace("\\", "/")
        del cluster["pixels"]
        csv_entries.append(cluster)

    relative_markers = [round(float(x) / float(cropped.shape[1]), 4) for x in vertical_markers_px]
    needs_manual = any(cluster["coverage_ratio"] < 0.70 for cluster in csv_entries)

    metadata.update(
        {
            "digitized": True,
            "digitization_method": "intensity_cluster_skeleton",
            "axis_scale": "relative",
            "vertical_markers_px": vertical_markers_px,
            "vertical_markers_relative_x": relative_markers,
            "series": csv_entries,
            "needs_manual_review": needs_manual,
        }
    )

    try:
        json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        summary["digitized"] += 1
        if needs_manual:
            summary["flagged"] += 1
    except Exception:
        summary["failed"] += 1


def process_graphs_folder(graphs_path: Path, root_dir: Path, summary: dict):
    for entry in sorted(graphs_path.glob("*.png")):
        summary["total_pngs"] += 1
        process_graph_png(entry, entry.with_suffix(".json"), root_dir, summary)


def main():
    args = parse_args()
    root_dir = Path(args.root).resolve() if args.root else DEFAULT_ROOT.resolve()
    if not root_dir.is_dir():
        raise SystemExit(f"Root directory not found: {root_dir}")

    summary = {
        "total_pngs": 0,
        "skipped": 0,
        "digitized": 0,
        "flagged": 0,
        "failed": 0,
    }

    if args.single:
        png_path = resolve_single_path(args.single, root_dir)
        if not is_graphs_png(png_path):
            raise SystemExit(
                f"Single file must be an existing PNG inside a graphs/ folder: {args.single}"
            )
        summary["total_pngs"] = 1
        process_graph_png(png_path, png_path.with_suffix(".json"), root_dir, summary)
    else:
        for graphs_path in find_graphs_folders(root_dir):
            process_graphs_folder(graphs_path, root_dir, summary)

    print("Processing complete.")
    print(f"Total PNGs found: {summary['total_pngs']}")
    print(f"Skipped: {summary['skipped']}")
    print(f"Digitized: {summary['digitized']}")
    print(f"Flagged for review: {summary['flagged']}")
    print(f"Failed: {summary['failed']}")


if __name__ == "__main__":
    main()
