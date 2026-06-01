"""
Pseudo-code structure:
1. Load the original graph PNG
2. Parse points from points.csv
3. Create a blank canvas matching original graph dimensions
4. Draw detected points on the canvas (circles, markers, etc.)
5. Optionally: connect points with lines if they form a curve/trend
6. Create side-by-side composite image:
   - Left: Original PNG
   - Right: Points overlay (or blank background + points)
7. Save as output (e.g., comparison_page1.1.png)

Create side-by-side comparison of original graphs vs detected points.

This script reads cropped graph PNGs and their corresponding points.csv files,
then creates a visual comparison showing:
- Left: Original graph image
- Right: Detected points connected with lines overlaid on the original

Example output structure:
    output_pdf/
    └── Bryant - 1964 - Vibrational Spectrum.../
        └── page_1/
            ├── graphs/
            │   ├── page1.1.png          <- (Original graph)
            │   └── page1.1_points.csv   <- (Detected points)
            └── comparisons/
                └── page1.1_comparison.png  <- (Side-by-side output)

Dependencies:
    - opencv-python (cv2)
    - pandas (for CSV reading)
    - numpy
"""

import cv2  # Fast image operations
from PIL import Image, ImageDraw  # Alternative with better font support
import pandas as pd  # Parse points.csv
import numpy as np  # Coordinate transformations
import csv
from pathlib import Path
from typing import List, Tuple

def load_points_from_csv(csv_path: Path) -> List[Tuple[float, float]]:
   """Load (x, y) coordinates from CSV file.

   Expected CSV format:
       x,y
       123.45,456.78
       234.56,567.89
       ...

   Args:
       csv_path: Path to the CSV file

   Returns:
       List of (x, y) tuples in pixel coordinates
   """
   try:
      df = pd.read_csv(csv_path)
      points = list(zip(df['x'], df['y']))
      return points
   except Exception as e:
      print(f"[!] Error reading CSV {csv_path.name}: {e}")
      return []


def draw_points_and_lines(
        image: np.ndarray,
        points: List[Tuple[float, float]],
        point_color: Tuple[int, int, int] = (0, 255, 0),
        line_color: Tuple[int, int, int] = (255, 0, 0),
        point_radius: int = 5,
        line_thickness: int = 2,
) -> np.ndarray:
   """Draw detected points and connecting lines on image.

   Args:
       image: Original graph image (numpy array)
       points: List of (x, y) coordinate tuples
       point_color: BGR color for points (default: bright green)
       line_color: BGR color for connecting lines (default: red)
       point_radius: Radius of point circles in pixels
       line_thickness: Thickness of connecting lines

   Returns:
       Image with drawn points and lines
   """
   # Create a copy to avoid modifying original
   overlay = image.copy()

   if not points or len(points) < 2:
      print(f"    [!] Insufficient points ({len(points)}) for line drawing")
      return overlay

   # Convert points to integer pixel coordinates
   pixel_points = [(int(x), int(y)) for x, y in points]

   # Draw connecting lines between consecutive points
   for i in range(len(pixel_points) - 1):
      pt1 = tuple(pixel_points[i])
      pt2 = tuple(pixel_points[i + 1])
      cv2.line(overlay, pt1, pt2, line_color, line_thickness)

   # Draw points (circles) on top of lines
   for pt in pixel_points:
      cv2.circle(overlay, pt, point_radius, point_color, -1)  # -1 fills the circle

   return overlay


def create_side_by_side_comparison(
        original_image_path: Path,
        points_csv_path: Path,
        output_path: Path,
        gap_width: int = 10,
) -> bool:
   """Create and save a side-by-side comparison image.

   Args:
       original_image_path: Path to original graph PNG
       points_csv_path: Path to points CSV file
       output_path: Where to save the comparison image
       gap_width: Pixel gap between left and right images

   Returns:
       True if successful, False otherwise
   """
   # Load original image
   original = cv2.imread(str(original_image_path))
   if original is None:
      print(f"    [!] Could not load image: {original_image_path.name}")
      return False

   # Load points from CSV
   points = load_points_from_csv(points_csv_path)
   if not points:
      print(f"    [!] No valid points found in {points_csv_path.name}")
      return False

   print(f"    -> Loaded {len(points)} points from CSV")

   # Create overlay with points and connecting lines
   overlay_image = draw_points_and_lines(original, points)

   # Get dimensions
   height, width = original.shape[:2]

   # Create side-by-side composite
   # Total width = original + gap + overlay
   composite_width = width * 2 + gap_width
   composite_height = height

   # Create blank canvas (white background)
   composite = np.ones(
      (composite_height, composite_width, 3),
      dtype=np.uint8
   ) * 255

   # Place original on left
   composite[0:height, 0:width] = original

   # Place overlay on right (after gap)
   composite[0:height, width + gap_width:width * 2 + gap_width] = overlay_image

   # Save composite image
   success = cv2.imwrite(str(output_path), composite)
   if success:
      print(f"    -> Saved comparison: {output_path.relative_to(output_path.parent.parent.parent)}")
      return True
   else:
      print(f"    [!] Failed to save: {output_path.name}")
      return False


def process_all_graph_comparisons():
   """Traverse output_pdf directory and create comparisons for all graphs."""
   current_dir = Path(__file__).resolve().parent
   output_pdf_root = current_dir / "output_pdf"

   if not output_pdf_root.exists():
      print(f"[!] Directory '{output_pdf_root.name}' not found in {current_dir}")
      return

   print(f"\nCreating graph comparisons in {output_pdf_root.name}/\n")

   comparison_count = 0

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

         # Create comparisons subfolder
         comparisons_folder = page_folder / "comparisons"
         comparisons_folder.mkdir(exist_ok=True)

         # Find all graph PNG files
         graph_pngs = sorted(graphs_folder.glob("page*.png"))

         if not graph_pngs:
            continue

         print(f"  Processing {page_folder.name}/ ({len(graph_pngs)} graph(s))")

         for graph_png in graph_pngs:
            # Expected CSV name: page1.1.png -> page1.1_points.csv
            csv_name = graph_png.stem + "_points.csv"
            csv_path = graphs_folder / csv_name

            if not csv_path.exists():
               print(f"    [!] No CSV found for {graph_png.name}")
               continue

            # Create output comparison filename
            comparison_name = graph_png.stem + "_comparison.png"
            comparison_path = comparisons_folder / comparison_name

            # Create comparison
            if create_side_by_side_comparison(graph_png, csv_path, comparison_path):
               comparison_count += 1

      print()

   print(f"Finished creating {comparison_count} comparison(s).")


if __name__ == "__main__":
   process_all_graph_comparisons()
