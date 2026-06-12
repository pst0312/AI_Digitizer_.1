"""Interactive CLI to review cropped graph images and sort them.

Usage:
    python3 2.1_cli_png_mngmt.py [root]

- root: optional path to `output_pdf` (default: ./output_pdf)

Behavior:
- Traverses output_pdf/<paper>/<page_X>/graphs/*.png
- For each graph crop, displays it side-by-side with the parent page image
  in a non-blocking Matplotlib window.
- Prompts in terminal: [K]eep, [M]ove to tables, [D]elete (k/m/d)
- Closes the viewer immediately when a choice is entered.

Requirements:
- Python with `opencv-python` and `matplotlib` installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import re

import cv2
import matplotlib.pyplot as plt


VALID_EXTS = {".png", ".jpg", ".jpeg", ".tiff"}


def extract_page_number(folder_name: str) -> int | None:
    m = re.search(r"page[_]?(\d+)", folder_name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def find_parent_page_image(page_folder: Path) -> Path | None:
    """Find a page image in the parent page folder that starts with the folder name.

    Example: page_folder.name == 'page_1' -> look for files starting with 'page_1_'
    """
    prefix = f"{page_folder.name}_"
    for p in page_folder.iterdir():
        if p.is_file() and p.suffix.lower() in VALID_EXTS and p.name.startswith(prefix):
            return p
    # fallback: any image file in parent
    for p in page_folder.iterdir():
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            return p
    return None


def display_side_by_side(crop_path: Path, parent_path: Path | None) -> plt.Figure:
    """Open a non-blocking Matplotlib window showing crop and parent side-by-side.

    Returns the Matplotlib figure so it can be closed programmatically.
    """
    # Read images with OpenCV (BGR) and convert to RGB
    crop = cv2.imread(str(crop_path))
    if crop is None:
        raise RuntimeError(f"Failed to read image: {crop_path}")
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    parent = None
    if parent_path:
        parent_img = cv2.imread(str(parent_path))
        if parent_img is not None:
            parent = cv2.cvtColor(parent_img, cv2.COLOR_BGR2RGB)

    # Create figure
    cols = 2 if parent is not None else 1
    fig, axes = plt.subplots(1, cols, figsize=(12, 7))
    if cols == 1:
        ax = axes
        ax.imshow(crop)
        ax.set_title(f"Crop: {crop_path.name}")
        ax.axis("off")
    else:
        ax_crop, ax_parent = axes
        ax_crop.imshow(crop)
        ax_crop.set_title(f"Crop: {crop_path.name}")
        ax_crop.axis("off")

        ax_parent.imshow(parent)
        ax_parent.set_title(f"Parent: {parent_path.name}")
        ax_parent.axis("off")

    plt.tight_layout()
    # Non-blocking show
    plt.show(block=False)
    # Give GUI event loop a moment to draw
    plt.pause(0.1)
    return fig


def prompt_action() -> str:
    """Prompt user for action until a valid choice is provided."""
    valid = {"k": "keep", "m": "move", "d": "delete", "q": "quit"}
    while True:
        try:
            resp = input("[K]eep, [M]ove to tables, or [D]elete? (k/m/d, q to quit): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return "q"
        if resp in valid:
            return resp
        print("Invalid choice. Please enter 'k', 'm', 'd', or 'q'.")


def process_graph_file(graph_path: Path, paper_folder: Path) -> bool:
    """Display graph and parent, prompt for action, perform it.

    Returns False if user chose to quit, True to continue.
    """
    page_folder = graph_path.parent.parent
    page_num = extract_page_number(page_folder.name)

    parent_page = find_parent_page_image(page_folder)

    try:
        fig = display_side_by_side(graph_path, parent_page)
    except Exception as e:
        print(f"Error displaying images: {e}")
        fig = None

    action = prompt_action()

    # Close the viewer immediately
    if fig is not None:
        try:
            plt.close(fig)
        except Exception:
            pass

    if action == "q":
        return False

    if action == "k":
        # keep in place
        print(f"Keeping {graph_path.relative_to(paper_folder)}")
        return True

    if action == "m":
        tables_folder = page_folder / "tables"
        tables_folder.mkdir(exist_ok=True)
        dest = tables_folder / graph_path.name
        graph_path.replace(dest)
        print(f"Moved to {dest.relative_to(paper_folder)}")
        return True

    if action == "d":
        try:
            graph_path.unlink()
            print(f"Deleted {graph_path.relative_to(paper_folder)}")
        except Exception as e:
            print(f"Failed to delete {graph_path}: {e}")
        return True

    return True


def traverse_and_review(root: Path) -> None:
    if not root.exists() or not root.is_dir():
        print(f"Root folder not found: {root}")
        return

    paper_folders = sorted([p for p in root.iterdir() if p.is_dir()])
    if not paper_folders:
        print(f"No paper folders found in {root}")
        return

    for paper in paper_folders:
        print(f"\nPaper: {paper.name}")
        page_folders = sorted([d for d in paper.iterdir() if d.is_dir() and d.name.startswith("page_")])
        for page in page_folders:
            graphs_folder = page / "graphs"
            if not graphs_folder.exists():
                continue
            graph_files = sorted([f for f in graphs_folder.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTS])
            if not graph_files:
                continue
            print(f" Reviewing {page.name} ({len(graph_files)} graph(s))")
            for graph in graph_files:
                cont = process_graph_file(graph, paper)
                if not cont:
                    print("User requested quit. Exiting traversal.")
                    return


def main():
    parser = argparse.ArgumentParser(description="Interactive graph review for output_pdf structure")
    parser.add_argument(
        "root",
        nargs="?",
        default=None,
        help="Root output_pdf folder (default: output_pdf at the repo root)",
    )
    args = parser.parse_args()

    root = (
        Path(args.root).resolve()
        if args.root
        else Path(__file__).resolve().parent.parent / "output_pdf"
    )
    traverse_and_review(root)


if __name__ == "__main__":
    main()
