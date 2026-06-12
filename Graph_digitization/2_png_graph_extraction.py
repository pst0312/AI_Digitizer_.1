"""Extract graphs from nested page folders in output_pdf structure.

This script traverses the output_pdf directory, finds page PNG files inside
page_X folders, runs DocLayout-YOLO on each, and saves cropped graphs into
a new `graphs` subfolder with naming convention: page{n}.{crop_count}.png

Example output structure:
    output_pdf/
    └── Bryant - 1964 - Vibrational Spectrum.../
        └── page_1/
            ├── page_1_Bryant...pdf
            ├── page_1_Bryant...png  <- (Model input)
            └── graphs/
                ├── page1.1.png      <- (Cropped graph 1)
                └── page1.2.png      <- (Cropped graph 2)
"""

import os
import re
from pathlib import Path

import cv2
from doclayout_yolo import YOLOv10


def load_model():
    """Load the local DocLayout-YOLO model."""
    repo_root = Path(__file__).resolve().parent.parent
    model_filename = "doclayout_yolo_docstructbench_imgsz1024.pt"
    model_path = repo_root / model_filename

    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find '{model_filename}' in {repo_root}. "
            f"Please download it from Hugging Face and place it here."
        )

    print("Loading local DocLayout-YOLO model...")
    return YOLOv10(str(model_path))


def extract_page_number(folder_name: str) -> int | None:
    """Extract the page number from a folder name like 'page_1'."""
    match = re.search(r"page[_]?(\d+)", folder_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def find_page_png(page_folder: Path) -> Path | None:
    """Find the page PNG file inside a page folder (e.g., page_1_Bryant...png)."""
    valid_extensions = {".png", ".jpg", ".jpeg", ".tiff"}
    for file_path in page_folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            return file_path
    return None


def process_output_pdf_nested():
    """Traverse output_pdf directory and extract graphs from each page."""
    repo_root = Path(__file__).resolve().parent.parent
    output_pdf_root = repo_root / "output_pdf"

    if not output_pdf_root.exists():
        print(f"[!] Directory '{output_pdf_root.name}' not found in {repo_root}")
        return

    model = load_model()

    print(f"\nProcessing nested structure in {output_pdf_root.name}/\n")

    # Traverse paper folders inside output_pdf
    paper_folders = sorted([d for d in output_pdf_root.iterdir() if d.is_dir()])

    if not paper_folders:
        print("[!] No paper folders found inside output_pdf/")
        return

    for paper_folder in paper_folders:
        print(f"Processing paper: {paper_folder.name}/")

        # Traverse page_X folders inside each paper folder
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
            # Extract page number (e.g., "page_1" -> 1)
            page_num = extract_page_number(page_folder.name)
            if page_num is None:
                print(f"  [!] Could not parse page number from {page_folder.name}/")
                continue

            # Find the page PNG file inside this folder
            page_png = find_page_png(page_folder)
            if not page_png:
                print(f"  [!] No PNG image found in {page_folder.name}/")
                continue

            print(f"  Analyzing {page_png.name}...")

            # Create graphs subfolder inside the page folder
            graphs_folder = page_folder / "graphs"
            graphs_folder.mkdir(exist_ok=True)

            # Run YOLO prediction
            results = model.predict(str(page_png), imgsz=1024, conf=0.25, device="cpu")

            # Load image via OpenCV for cropping
            image = cv2.imread(str(page_png))
            height, width, _ = image.shape

            crop_count = 0

            # Loop through detected boundaries
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                # Filter specifically for document regions likely containing graphs
                if label in ["figure", "picture", "table"]:
                    crop_count += 1

                    # Extract raw pixel coordinates
                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())

                    # Apply a 5% padding to avoid clipping labels or axis numbers
                    pad_x = int((xmax - xmin) * 0.05)
                    pad_y = int((ymax - ymin) * 0.05)

                    xmin = max(0, xmin - pad_x)
                    ymin = max(0, ymin - pad_y)
                    xmax = min(width, xmax + pad_x)
                    ymax = min(height, ymax + pad_y)

                    # Perform the pixel slice
                    cropped_segment = image[ymin:ymax, xmin:xmax]

                    # Save with naming convention: page{n}.{crop_count}.png
                    output_file_name = f"page{page_num}.{crop_count}.png"
                    output_file_path = graphs_folder / output_file_name

                    cv2.imwrite(str(output_file_path), cropped_segment)
                    print(
                        f"    -> Extracted {label} to {output_file_path.relative_to(paper_folder)}"
                    )

            if crop_count == 0:
                print(f"    -> No graphs or figures detected on {page_png.name}.")
            else:
                print(f"    -> Found {crop_count} graph(s) on {page_png.name}.")

        print()


if __name__ == "__main__":
    process_output_pdf_nested()
