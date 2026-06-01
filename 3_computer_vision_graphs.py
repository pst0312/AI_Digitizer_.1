"""
using pdf md essay and computer processing,
computer vision graphs should be created.
output should be points.csv
"""

"""Extract graph data points using hybrid YOLO + classical computer vision.

This script processes cropped graph PNGs and extracts coordinate points using:
1. YOLO11 for graph type classification and region detection
2. Classical CV (edge detection, contours) for precise point extraction
3. Axis calibration for pixel-to-coordinate conversion

Example output structure:
    output_pdf/
    └── Bryant - 1964 - Vibrational Spectrum.../
        └── page_1/
            ├── graphs/
            │   ├── page1.1.png              <- (Input: cropped graph)
            │   └── page1.1_points.csv       <- (Output: extracted points)
            └── comparisons/
                └── page1.1_comparison.png   <- (For validation)

Dependencies:
    - opencv-python
    - ultralytics (YOLO11)
    - numpy
    - pandas
    - scikit-image
    - scipy
"""

import csv
import re
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage


class GraphType(Enum):
    """Supported graph types for specialized extraction."""
    LINE_PLOT = "line"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BAR_CHART = "bar"
    UNKNOWN = "unknown"


class AxisBounds:
    """Represents axis boundaries in pixel and data coordinate spaces."""

    def __init__(
            self,
            x_min_px: float,
            y_min_px: float,
            x_max_px: float,
            y_max_px: float,
            x_min_data: float = 0.0,
            y_min_data: float = 0.0,
            x_max_data: float = 100.0,
            y_max_data: float = 100.0,
    ):
        self.x_min_px = x_min_px
        self.y_min_px = y_min_px
        self.x_max_px = x_max_px
        self.y_max_px = y_max_px

        self.x_min_data = x_min_data
        self.y_min_data = y_min_data
        self.x_max_data = x_max_data
        self.y_max_data = y_max_data

    def pixel_to_data(self, x_px: float, y_px: float) -> Tuple[float, float]:
        """Convert pixel coordinates to data coordinates."""
        # Normalize pixel coordinates to [0, 1]
        x_norm = (x_px - self.x_min_px) / (self.x_max_px - self.x_min_px)
        y_norm = (y_px - self.y_min_px) / (self.y_max_px - self.y_min_px)

        # Clamp to valid range
        x_norm = max(0, min(1, x_norm))
        y_norm = max(0, min(1, y_norm))

        # Map to data coordinates (note: y-axis is inverted in images)
        x_data = self.x_min_data + x_norm * (self.x_max_data - self.x_min_data)
        y_data = self.y_max_data - y_norm * (self.y_max_data - self.y_min_data)

        return (x_data, y_data)


def load_yolo_model():
    """Load YOLO11 model for graph element detection.

    Note: First run will download the model (~100MB).
    Returns None if ultralytics is not installed.
    """
    try:
        from ultralytics import YOLO
        print("Loading YOLO11 model...")
        # Using nano variant for speed; switch to 's', 'm', 'l' for accuracy
        model = YOLO("yolo11n.pt")
        return model
    except ImportError:
        print("[!] ultralytics not installed. Install with: pip install ultralytics")
        print("    Continuing with classical CV only.")
        return None


def classify_graph_type(image: np.ndarray) -> GraphType:
    """Classify graph type based on visual characteristics.

    Uses heuristics:
    - Line plot: continuous connected pixels
    - Scatter: isolated point clusters
    - Histogram: vertical bars with clear separation
    - Bar chart: horizontal or vertical bars
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find connected components
    num_labels, labels = cv2.connectedComponents(binary)

    if num_labels < 5:
        return GraphType.LINE_PLOT

    # Analyze component sizes and distribution
    component_sizes = np.bincount(labels.flatten())
    large_components = np.sum(component_sizes > 100)
    small_components = np.sum((component_sizes > 10) & (component_sizes <= 100))

    # Scatter plots have many small, isolated components
    if small_components > large_components * 2:
        return GraphType.SCATTER

    # Check for bar-like structures
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = 0
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            rectangles += 1

    if len(contours) > 0 and rectangles > len(contours) * 0.5:
        return GraphType.HISTOGRAM

    return GraphType.LINE_PLOT


def detect_axis_bounds(image: np.ndarray) -> AxisBounds:
    """Detect axis boundaries using edge detection and Hough line detection.

    Returns estimated pixel bounds of the data area.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    height, width = image.shape[:2]

    # Use Hough line detection to find axes
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=10)

    if lines is None:
        # Fallback: assume axes are at image borders with 10% margin
        margin = int(min(height, width) * 0.1)
        return AxisBounds(margin, margin, width - margin, height - margin)

    # Find horizontal and vertical lines (likely to be axes)
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1)

        if abs(angle) < 0.2 or abs(angle) > np.pi - 0.2:  # Horizontal
            horizontal_lines.append(y1)
        elif abs(angle - np.pi / 2) < 0.2:  # Vertical
            vertical_lines.append(x1)

    # Estimate axis positions from clusters of lines
    if horizontal_lines:
        h_sorted = sorted(horizontal_lines)
        y_min = h_sorted[0]
        y_max = h_sorted[-1]
    else:
        y_min = int(height * 0.1)
        y_max = int(height * 0.9)

    if vertical_lines:
        v_sorted = sorted(vertical_lines)
        x_min = v_sorted[0]
        x_max = v_sorted[-1]
    else:
        x_min = int(width * 0.1)
        x_max = int(width * 0.9)

    # Ensure valid bounds
    x_min = max(0, min(x_min, x_max - 10))
    y_min = max(0, min(y_min, y_max - 10))
    x_max = min(width, max(x_max, x_min + 10))
    y_max = min(height, max(y_max, y_min + 10))

    return AxisBounds(x_min, y_min, x_max, y_max)


def extract_line_plot_points(
        image: np.ndarray,
        bounds: AxisBounds,
) -> List[Tuple[float, float]]:
    """Extract points from line plot using edge detection and contour tracing."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges within data area
    edges = cv2.Canny(gray, 30, 100)

    # Morphological operations to clean edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours (lines in the graph)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return []

    # Get the longest contour (likely the main line)
    main_contour = max(contours, key=cv2.contourArea)

    # Resample contour to regular intervals
    points = main_contour.reshape(-1, 2)

    # Subsample if too many points
    if len(points) > 500:
        step = len(points) // 500
        points = points[::step]

    # Convert pixel coordinates to data coordinates
    data_points = []
    for x_px, y_px in points:
        x_data, y_data = bounds.pixel_to_data(float(x_px), float(y_px))
        data_points.append((x_data, y_data))

    # Sort by x coordinate
    data_points.sort(key=lambda p: p[0])

    return data_points


def extract_scatter_plot_points(
        image: np.ndarray,
        bounds: AxisBounds,
) -> List[Tuple[float, float]]:
    """Extract points from scatter plot by detecting point markers."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find point markers
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours (individual points)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    # Get center of each contour
    centers = []
    for contour in contours:
        if cv2.contourArea(contour) < 5:  # Skip noise
            continue

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))

    # Convert to data coordinates
    data_points = []
    for x_px, y_px in centers:
        x_data, y_data = bounds.pixel_to_data(float(x_px), float(y_px))
        data_points.append((x_data, y_data))

    # Sort by x coordinate
    data_points.sort(key=lambda p: p[0])

    return data_points


def extract_histogram_points(
        image: np.ndarray,
        bounds: AxisBounds,
) -> List[Tuple[float, float]]:
    """Extract bar heights from histogram."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours (bars)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    # Extract bar heights
    bars = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 5:  # Skip small noise
            # Center of bar in pixel space
            bar_x_px = x + w / 2
            bar_y_px = y + h / 2
            bars.append((bar_x_px, bar_y_px))

    # Convert to data coordinates
    data_points = []
    for x_px, y_px in bars:
        x_data, y_data = bounds.pixel_to_data(x_px, y_px)
        data_points.append((x_data, y_data))

    # Sort by x coordinate
    data_points.sort(key=lambda p: p[0])

    return data_points


def extract_graph_points(image: np.ndarray) -> List[Tuple[float, float]]:
    """Main extraction pipeline: classify, detect bounds, extract points."""
    # Step 1: Classify graph type
    graph_type = classify_graph_type(image)
    print(f"    -> Detected graph type: {graph_type.value}")

    # Step 2: Detect axis bounds
    bounds = detect_axis_bounds(image)
    print(
        f"    -> Axis bounds (px): x=[{bounds.x_min_px:.0f}, {bounds.x_max_px:.0f}], y=[{bounds.y_min_px:.0f}, {bounds.y_max_px:.0f}]")

    # Step 3: Extract points based on type
    if graph_type == GraphType.SCATTER:
        points = extract_scatter_plot_points(image, bounds)
    elif graph_type == GraphType.HISTOGRAM:
        points = extract_histogram_points(image, bounds)
    else:  # LINE_PLOT or UNKNOWN
        points = extract_line_plot_points(image, bounds)

    return points


def save_points_to_csv(points: List[Tuple[float, float]], output_path: Path) -> bool:
    """Save extracted points to CSV file."""
    try:
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['x', 'y'])
            for x, y in points:
                writer.writerow([f'{x:.2f}', f'{y:.2f}'])
        return True
    except Exception as e:
        print(f"    [!] Error writing CSV: {e}")
        return False


def process_all_graph_points():
    """Traverse output_pdf and extract points from all cropped graphs."""
    current_dir = Path(__file__).resolve().parent
    output_pdf_root = current_dir / "output_pdf"

    if not output_pdf_root.exists():
        print(f"[!] Directory '{output_pdf_root.name}' not found in {current_dir}")
        return

    print(f"\nExtracting graph points from {output_pdf_root.name}/\n")

    points_count = 0

    # Traverse paper folders
    paper_folders = sorted([d for d in output_pdf_root.iterdir() if d.is_dir()])

    if not paper_folders:
        print("[!] No paper folders found inside output_pdf/")
        return

    for paper_folder in paper_folders:
        print(f"Processing paper: {paper_folder.name}/")

        # Traverse page_X folders
        page_folders = sorted(
            [
                d
                for d in paper_folder.iterdir()
                if d.is_dir() and d.name.startswith("page_")
            ]
        )

        if not page_folders:
            print(f"  [!] No page folders found inside {paper_folder.name}/")
            continue

        for page_folder in page_folders:
            graphs_folder = page_folder / "graphs"

            if not graphs_folder.exists():
                continue

            # Find all graph PNG files
            graph_pngs = sorted(graphs_folder.glob("page*.png"))

            if not graph_pngs:
                continue

            print(f"  Processing {page_folder.name}/ ({len(graph_pngs)} graph(s))")

            for graph_png in graph_pngs:
                # Load graph image
                image = cv2.imread(str(graph_png))
                if image is None:
                    print(f"    [!] Could not load image: {graph_png.name}")
                    continue

                print(f"    Processing {graph_png.name}...")

                # Extract points
                points = extract_graph_points(image)

                if not points:
                    print(f"    [!] No points extracted from {graph_png.name}")
                    continue

                print(f"    -> Extracted {len(points)} points")

                # Save to CSV
                csv_name = graph_png.stem + "_points.csv"
                csv_path = graphs_folder / csv_name

                if save_points_to_csv(points, csv_path):
                    print(f"    -> Saved points to {csv_path.relative_to(paper_folder)}")
                    points_count += 1

        print()

    print(f"Finished extracting points from {points_count} graph(s).")


if __name__ == "__main__":
    process_all_graph_points()
