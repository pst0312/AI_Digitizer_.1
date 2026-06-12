#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure PNGs in graphs/ folders and write JSON metadata"
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        default=None,
        help="Path to the root output_pdf/ directory (default: output_pdf at the repo root)",
    )
    return parser.parse_args()


def process_graphs_folder(graphs_path, root_dir, counters):
    counters["graphs_folders"] += 1

    for entry in os.listdir(graphs_path):
        entry_path = os.path.join(graphs_path, entry)
        if not os.path.isfile(entry_path):
            continue

        if not entry.lower().endswith(".png"):
            continue

        counters["png_measured"] += 1
        with Image.open(entry_path) as img:
            width, height = img.size

        rel_png_path = os.path.relpath(entry_path, root_dir)
        json_path = os.path.splitext(entry_path)[0] + ".json"

        if os.path.exists(json_path):
            counters["json_updated"] += 1
            with open(json_path, "r", encoding="utf-8") as f:
                try:
                    metadata = json.load(f)
                except ValueError:
                    metadata = {}
        else:
            counters["json_created"] += 1
            metadata = {"source_png": rel_png_path}

        metadata["image_width_px"] = int(width)
        metadata["image_height_px"] = int(height)
        metadata["image_measured"] = True

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            f.write("\n")


def main():
    args = parse_args()
    root_dir = (
        os.path.abspath(args.root_dir)
        if args.root_dir
        else str(Path(__file__).resolve().parent.parent / "output_pdf")
    )

    counters = {
        "graphs_folders": 0,
        "png_measured": 0,
        "json_created": 0,
        "json_updated": 0,
    }

    for current_root, dirs, files in os.walk(root_dir):
        # If this folder is itself a graphs/ folder, skip it here.
        # We handle graphs/ folders from their parent directories so we do not recurse into them.
        if os.path.basename(current_root).lower() == "graphs":
            continue

        if "graphs" not in dirs:
            # No graphs/ folder in this branch, skip processing entirely.
            continue

        graphs_path = os.path.join(current_root, "graphs")

        # Prevent os.walk from descending into graphs/ subfolders.
        # This ensures we only process PNG files inside graphs/ and never walk deeper inside that folder.
        dirs.remove("graphs")

        if not os.path.isdir(graphs_path):
            continue

        process_graphs_folder(graphs_path, root_dir, counters)

    print("Processing complete.")
    print(f"Total graphs/ folders found: {counters['graphs_folders']}")
    print(f"Total PNG files measured: {counters['png_measured']}")
    print(f"Total JSON files created: {counters['json_created']}")
    print(f"Total JSON files updated: {counters['json_updated']}")


if __name__ == "__main__":
    main()
