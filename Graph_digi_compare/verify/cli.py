"""CLI orchestrator.

Wires discovery -> calibration -> pixel-space overlay -> hi-res PNG and writes a
run manifest. Each run produces a timestamped batch of overlay PNGs (one per
graph) for review or for feeding back into model-improvement workflows.

Run as a module from the repo root so ``import verify.*`` resolves::

    python -m Graph_digi_compare.verify.cli
    python -m Graph_digi_compare.verify.cli --filter page6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `import verify.*` work whether launched as a module or a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verify import config  # noqa: E402
from verify.discovery import discover_graphs, load_calibration  # noqa: E402
from verify.storage import (  # noqa: E402
    create_run_dir,
    manifest_entry,
    save_figure_png,
    write_manifest,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay each digitized graph's CSV onto its original PNG cutout and "
            "save a pixel-space overlay PNG per graph (a batch per run)."
        )
    )
    parser.add_argument(
        "root_dir", nargs="?", default=str(config.DEFAULT_ROOT),
        help="Root output_pdf directory to scan (default: output_pdf).",
    )
    parser.add_argument(
        "--scaled", action="store_true",
        help="Read series from digitized_scaled/*.csv instead of digitized/*.csv.",
    )
    parser.add_argument(
        "--alpha", type=float, default=config.DEFAULT_IMG_ALPHA,
        help=f"Background PNG opacity (default: {config.DEFAULT_IMG_ALPHA}).",
    )
    parser.add_argument(
        "--dpi", type=int, default=config.DEFAULT_DPI,
        help=f"Output PNG resolution (default: {config.DEFAULT_DPI}).",
    )
    parser.add_argument(
        "--filter", metavar="SUBSTR", default=None,
        help="Only process graphs whose plot_key contains this substring.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    csv_folder = config.SCALED_FOLDER if args.scaled else config.RELATIVE_FOLDER
    records = discover_graphs(root, csv_folder=csv_folder)
    if args.filter:
        records = [r for r in records if args.filter in r.plot_key]

    if not records:
        raise RuntimeError(f"No digitized graphs found under {root} ({csv_folder}/).")

    run_dir = create_run_dir()
    print(f"Run directory: {run_dir}")

    # Import the matplotlib renderer lazily to keep startup light.
    from verify.overlay import build_overlay_figure
    import matplotlib.pyplot as plt

    entries: list[dict] = []
    failures: list[dict] = []
    for record in records:
        try:
            cal = load_calibration(record)
            fig = build_overlay_figure(record, cal, img_alpha=args.alpha)
            out_png = save_figure_png(fig, run_dir, record.plot_key, dpi=args.dpi)
            plt.close(fig)
            entries.append(
                manifest_entry(record, out_png, spaces=["pixel"])
            )
            print(f"  rendered {record.plot_key} -> {out_png.name}")
        except Exception as exc:  # keep going on bad graphs (incl. CalibrationError)
            failures.append({"plot_key": record.plot_key, "error": str(exc)})
            print(f"  FAILED {record.plot_key}: {exc}")

    manifest_path = write_manifest(
        run_dir, entries, args=vars(args), failures=failures
    )
    print(
        f"Done: {len(entries)} rendered, {len(failures)} failed. "
        f"Manifest: {manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
