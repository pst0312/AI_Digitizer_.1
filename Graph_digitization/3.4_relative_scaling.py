#!/usr/bin/env python3

import argparse
import csv
import json
import os
import re
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "output_pdf"

SERIES_CSV_PATTERN = re.compile(r"^(.+)_series_\d+_(?:solid|dashed)\.csv$")
VERTICAL_SHIFT_PATTERN = re.compile(r"^(.+)_vertical_shift\.csv$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert digitized graph CSVs from relative (0-1) coordinates "
                     "to true axis values using axis_x_origin/final or legacy axis_x_min/max "
                     "and axis_y_origin/final or legacy axis_y_min/max from the graph's JSON metadata. "
                     "Writes copies; originals are untouched."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=None,
        help="Root output_pdf folder (default: output_pdf)",
    )
    parser.add_argument(
        "--single",
        metavar="CSV_PATH",
        help="Process one specific digitized series CSV instead of walking the full tree",
    )
    return parser.parse_args()


def safe_load_json(json_path: Path):
    if not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_digitized_folders(root_dir: Path):
    for current_root, dirs, _ in os.walk(str(root_dir)):
        if os.path.basename(current_root).lower() == "digitized":
            continue
        if "digitized" in dirs:
            yield Path(current_root) / "digitized"


def get_axis_bounds(metadata: dict):
    """Return (x_origin, x_final, y_origin, y_final) as floats, or None if invalid.

    This prefers explicit origin/final fields (axis_x_origin/axis_x_final,
    axis_y_origin/axis_y_final) but falls back to legacy axis_x_min/max and
    axis_y_min/max for compatibility.
    """
    origin_keys = (
        "axis_x_origin",
        "axis_x_final",
        "axis_y_origin",
        "axis_y_final",
    )
    if all(k in metadata and metadata[k] not in (None, "") for k in origin_keys):
        try:
            return (
                float(metadata["axis_x_origin"]),
                float(metadata["axis_x_final"]),
                float(metadata["axis_y_origin"]),
                float(metadata["axis_y_final"]),
            )
        except (TypeError, ValueError):
            return None

    required = ("axis_x_min", "axis_x_max", "axis_y_min", "axis_y_max")
    if not all(k in metadata and metadata[k] not in (None, "") for k in required):
        return None
    try:
        return tuple(float(metadata[k]) for k in required)
    except (TypeError, ValueError):
        return None


def rescale_csv(csv_path: Path, json_path: Path, output_path: Path):
    """Read a relative-scale series CSV and write a true-scale copy. Returns True on success."""
    metadata = safe_load_json(json_path)
    if metadata is None:
        print(f"  Skipping {csv_path.name}: missing or unreadable JSON ({json_path.name})")
        return False

    bounds = get_axis_bounds(metadata)
    if bounds is None:
        print(
            f"  Skipping {csv_path.name}: {json_path.name} is missing "
            "axis_x_origin/axis_x_final/axis_y_origin/axis_y_final or legacy "
            "axis_x_min/axis_x_max/axis_y_min/axis_y_max"
        )
        return False

    x_min, x_max, y_min, y_max = bounds

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("r", encoding="utf-8", newline="") as src, \
            output_path.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)

        next(reader, None)  # skip original header (x_relative,y_relative)
        writer.writerow(["x", "y"])

        for row in reader:
            if len(row) < 2:
                continue
            x_rel, y_rel = float(row[0]), float(row[1])
            x_val = x_min + x_rel * (x_max - x_min)
            y_val = y_min + y_rel * (y_max - y_min)
            writer.writerow([f"{x_val:.6f}", f"{y_val:.6f}"])

    return True


def rescale_vertical_shift_csv(csv_path: Path, json_path: Path, output_path: Path):
    """Read a vert_grapher shift CSV (x_relative,y_relative + metadata columns) and write a
    copy with true-scale 'x'/'y' columns appended. Returns True on success."""
    metadata = safe_load_json(json_path)
    if metadata is None:
        print(f"  Skipping {csv_path.name}: missing or unreadable JSON ({json_path.name})")
        return False

    bounds = get_axis_bounds(metadata)
    if bounds is None:
        print(
            f"  Skipping {csv_path.name}: {json_path.name} is missing "
            "axis_x_origin/axis_x_final/axis_y_origin/axis_y_final or legacy "
            "axis_x_min/axis_x_max/axis_y_min/axis_y_max"
        )
        return False

    x_min, x_max, y_min, y_max = bounds

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("r", encoding="utf-8", newline="") as src, \
            output_path.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        if "x_relative" not in (reader.fieldnames or []) or "y_relative" not in (reader.fieldnames or []):
            print(f"  Skipping {csv_path.name}: missing x_relative/y_relative columns")
            return False

        fieldnames = list(reader.fieldnames)
        for col in ("x", "y"):
            if col not in fieldnames:
                fieldnames.append(col)

        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            x_rel, y_rel = float(row["x_relative"]), float(row["y_relative"])
            row["x"] = f"{x_min + x_rel * (x_max - x_min):.6f}"
            row["y"] = f"{y_min + y_rel * (y_max - y_min):.6f}"
            writer.writerow(row)

    return True


def identify_csv(csv_path: Path):
    """Return (graph_stem, rescaler_func) for a recognized digitized CSV, or (None, None)."""
    match = SERIES_CSV_PATTERN.match(csv_path.name)
    if match:
        return match.group(1), rescale_csv

    match = VERTICAL_SHIFT_PATTERN.match(csv_path.name)
    if match:
        return match.group(1), rescale_vertical_shift_csv

    return None, None


def process_digitized_folder(digitized_path: Path, summary: dict):
    graphs_path = digitized_path.parent / "graphs"
    output_path = digitized_path.parent / "digitized_scaled"

    for csv_path in sorted(digitized_path.glob("*.csv")):
        summary["total_csvs"] += 1

        graph_stem, rescaler = identify_csv(csv_path)
        if graph_stem is None:
            print(f"  Skipping {csv_path.name}: filename does not match a recognized pattern")
            summary["skipped"] += 1
            continue

        json_path = graphs_path / f"{graph_stem}.json"
        out_csv = output_path / csv_path.name

        if rescaler(csv_path, json_path, out_csv):
            summary["scaled"] += 1
        else:
            summary["skipped"] += 1


def resolve_single_path(single_arg: str, root_dir: Path) -> Path:
    candidate = Path(single_arg)
    if not candidate.is_absolute():
        candidate = root_dir / single_arg
    return candidate.resolve()


def main():
    args = parse_args()
    root_dir = Path(args.root).resolve() if args.root else DEFAULT_ROOT.resolve()
    if not root_dir.is_dir():
        raise SystemExit(f"Root directory not found: {root_dir}")

    summary = {"total_csvs": 0, "scaled": 0, "skipped": 0}

    if args.single:
        csv_path = resolve_single_path(args.single, root_dir)
        if not csv_path.is_file() or csv_path.suffix.lower() != ".csv":
            raise SystemExit(f"Single file must be an existing CSV: {args.single}")

        graph_stem, rescaler = identify_csv(csv_path)
        if graph_stem is None:
            raise SystemExit(
                f"Filename does not match '<stem>_series_<id>_<style>.csv' or "
                f"'<stem>_vertical_shift.csv': {csv_path.name}"
            )

        digitized_path = csv_path.parent
        graphs_path = digitized_path.parent / "graphs"
        json_path = graphs_path / f"{graph_stem}.json"
        out_csv = digitized_path.parent / "digitized_scaled" / csv_path.name

        summary["total_csvs"] = 1
        if rescaler(csv_path, json_path, out_csv):
            summary["scaled"] += 1
        else:
            summary["skipped"] += 1
    else:
        for digitized_path in find_digitized_folders(root_dir):
            print(f"Processing {digitized_path}")
            process_digitized_folder(digitized_path, summary)

    print("\nProcessing complete.")
    print(f"Total CSVs found: {summary['total_csvs']}")
    print(f"Scaled (copies written to digitized_scaled/): {summary['scaled']}")
    print(f"Skipped: {summary['skipped']}")


if __name__ == "__main__":
    main()
