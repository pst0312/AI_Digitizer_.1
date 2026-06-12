#!/usr/bin/env python3

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "output_pdf"


def parse_args():
    parser = argparse.ArgumentParser(
        description="For each digitized graph with a detected vertical shift line, "
                     "write a CSV recording each series' relative value at the shift location."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=None,
        help="Root output_pdf folder (default: output_pdf)",
    )
    parser.add_argument(
        "--single",
        metavar="JSON_PATH",
        help="Process one specific graph JSON (in a graphs/ folder) instead of walking the full tree",
    )
    return parser.parse_args()


def safe_load_json(json_path: Path):
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_graph_jsons(root_dir: Path):
    for current_root, _, files in os.walk(str(root_dir)):
        if os.path.basename(current_root).lower() != "graphs":
            continue
        for filename in sorted(files):
            if filename.lower().endswith(".json") and not filename.lower().endswith("_pre.json"):
                yield Path(current_root) / filename


def load_series_xy(csv_path: Path):
    xs, ys = [], []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            xs.append(float(row[0]))
            ys.append(float(row[1]))
    return np.array(xs), np.array(ys)


def process_graph_json(json_path: Path, summary: dict):
    metadata = safe_load_json(json_path)
    if metadata is None:
        print(f"  Skipping {json_path.name}: missing or unreadable JSON")
        summary["skipped"] += 1
        return

    markers_relative = metadata.get("vertical_markers_relative_x") or []
    markers_px = metadata.get("vertical_markers_px") or []
    series_list = metadata.get("series") or []

    page_dir = json_path.parent.parent
    shift_csv = page_dir / "digitized" / f"{json_path.stem}_vertical_shift.csv"

    if not markers_relative:
        # Remove any shift CSV left from a previous run so stale phantom markers
        # don't resurface in overlays now that this graph has no detected line.
        for stale in (
            shift_csv,
            page_dir / "digitized_scaled" / shift_csv.name,
        ):
            if stale.exists():
                stale.unlink()
                print(f"  Removed stale {stale}")
        summary["no_marker"] += 1
        return

    if not series_list:
        print(f"  Skipping {json_path.name}: no digitized series found")
        summary["skipped"] += 1
        return

    rows = []
    for idx, x_rel in enumerate(markers_relative):
        marker_px = markers_px[idx] if idx < len(markers_px) else ""
        for series in series_list:
            csv_rel = series.get("csv_path")
            if not csv_rel:
                continue
            series_csv = page_dir / csv_rel
            if not series_csv.exists():
                print(f"  Skipping series {series.get('id')} for {json_path.name}: {series_csv} not found")
                continue

            xs, ys = load_series_xy(series_csv)
            if xs.size == 0:
                continue

            y_at_marker = float(np.interp(x_rel, xs, ys))
            rows.append(
                {
                    "marker_index": idx,
                    "x_px": marker_px,
                    "x_relative": x_rel,
                    "series_id": series.get("id"),
                    "style": series.get("style"),
                    "y_relative": round(y_at_marker, 6),
                }
            )

    if not rows:
        summary["skipped"] += 1
        return

    out_path = shift_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["marker_index", "x_px", "x_relative", "series_id", "style", "y_relative"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Wrote {out_path}")
    summary["written"] += 1


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

    summary = {"total": 0, "written": 0, "no_marker": 0, "skipped": 0}

    if args.single:
        json_path = resolve_single_path(args.single, root_dir)
        if not json_path.is_file() or json_path.suffix.lower() != ".json":
            raise SystemExit(f"Single file must be an existing JSON: {args.single}")
        summary["total"] = 1
        process_graph_json(json_path, summary)
    else:
        for json_path in find_graph_jsons(root_dir):
            summary["total"] += 1
            process_graph_json(json_path, summary)

    print("\nProcessing complete.")
    print(f"Total graph JSONs found: {summary['total']}")
    print(f"Vertical-shift CSVs written: {summary['written']}")
    print(f"No vertical shift marker: {summary['no_marker']}")
    print(f"Skipped: {summary['skipped']}")


if __name__ == "__main__":
    main()
