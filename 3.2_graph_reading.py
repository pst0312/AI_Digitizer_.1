#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans


DEFAULT_ROOT = Path(__file__).parent / "output_pdf"


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


def detect_vertical_markers(gray_crop: np.ndarray):
    height, width = gray_crop.shape[:2]
    if height < 5 or width < 5:
        return [], np.zeros((height, width), dtype=np.uint8), gray_crop.copy()

    _, bin_inv = cv2.threshold(
        gray_crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    kernel_height = max(3, int(round(height * 0.6)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))
    vertical_candidates = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        vertical_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    marker_mask = np.zeros_like(vertical_candidates, dtype=np.uint8)
    marker_positions = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h < int(round(height * 0.6)):
            continue
        cv2.drawContours(marker_mask, [contour], -1, 255, thickness=-1)
        marker_positions.append(int(round(x + w / 2)))

    marker_positions = sorted(set(marker_positions))
    if np.any(marker_mask):
        inpainted = cv2.inpaint(gray_crop, marker_mask, 3, cv2.INPAINT_TELEA)
    else:
        inpainted = gray_crop.copy()

    return marker_positions, marker_mask, inpainted


def extract_dark_pixels(gray_image: np.ndarray):
    _, line_mask = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    ys, xs = np.where(line_mask > 0)
    return xs, ys, line_mask


def cluster_series(gray_image: np.ndarray, num_series: int):
    xs, ys, line_mask = extract_dark_pixels(gray_image)
    if xs.size == 0:
        raise ValueError("No dark pixels found inside plot crop")

    intensities = gray_image[ys, xs].astype(np.float32).reshape(-1, 1)
    if len(intensities) < num_series:
        raise ValueError(
            f"Insufficient dark pixels ({len(intensities)}) for {num_series} series"
        )

    model = KMeans(n_clusters=num_series, random_state=0, n_init=10)
    labels = model.fit_predict(intensities)

    clusters = []
    height, width = gray_image.shape[:2]

    for cluster_id in range(num_series):
        mask = labels == cluster_id
        cluster_x = xs[mask]
        cluster_y = ys[mask]
        point_count = int(cluster_x.size)
        intensity_center = float(model.cluster_centers_[cluster_id][0])

        if cluster_x.size == 0:
            clusters.append(
                {
                    "id": cluster_id + 1,
                    "style": "solid",
                    "intensity_center": intensity_center,
                    "point_count": 0,
                    "coverage_ratio": 0.0,
                    "pixels": [],
                }
            )
            continue

        df = pd.DataFrame({"x": cluster_x, "y": cluster_y})
        grouped = df.groupby("x")["y"].median()

        timeline = pd.DataFrame(
            {"x": np.arange(width)}
        )
        timeline["y"] = np.nan
        timeline.loc[grouped.index, "y"] = grouped.values
        coverage_ratio = float(grouped.index.size) / float(width) if width > 0 else 0.0
        timeline["y"] = timeline["y"].interpolate(method="linear", limit_direction="both")

        if timeline["y"].isna().any():
            raise ValueError(
                f"Cluster {cluster_id + 1} could not be fully interpolated across width={width}"
            )

        points = [
            (
                float(x) / float(width),
                1.0 - float(y_val) / float(height),
            )
            for x, y_val in zip(timeline["x"].to_numpy(), timeline["y"].to_numpy())
        ]

        style = "solid" if coverage_ratio > 0.85 else "dashed"

        clusters.append(
            {
                "id": cluster_id + 1,
                "style": style,
                "intensity_center": intensity_center,
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
        vertical_markers_px, marker_mask, cleaned = detect_vertical_markers(cropped)
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
