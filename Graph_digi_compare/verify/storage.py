"""Agent C — run isolation and artifact storage.

Each execution gets its own ``runs/run_YYYYMMDD_HHMMSS/`` folder holding the
high-resolution comparison PNGs and a ``manifest.json`` describing what was
rendered. Keeping every run separate makes it easy to diff digitization quality
across pipeline changes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from verify.config import DEFAULT_DPI, DEFAULT_RUNS_DIR

if TYPE_CHECKING:  # avoid importing matplotlib at module load
    from matplotlib.figure import Figure

    from verify.discovery import GraphRecord


def create_run_dir(base: Path = DEFAULT_RUNS_DIR) -> Path:
    """Create and return a fresh timestamped run directory with an ``images/`` subdir."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"run_{timestamp}"
    (run_dir / "images").mkdir(parents=True, exist_ok=True)
    return run_dir


def _safe_name(plot_key: str) -> str:
    return plot_key.replace("/", "_").replace("\\", "_")


def save_figure_png(
    fig: "Figure", run_dir: Path, plot_key: str, *, dpi: int = DEFAULT_DPI
) -> Path:
    """Save ``fig`` as a high-resolution PNG under ``run_dir/images/``."""
    target = run_dir / "images" / f"{_safe_name(plot_key)}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    return target


def write_manifest(
    run_dir: Path,
    entries: list[dict[str, Any]],
    *,
    args: dict[str, Any] | None = None,
    failures: list[dict[str, Any]] | None = None,
) -> Path:
    """Write ``manifest.json`` summarizing the run.

    ``entries`` is a list of per-graph dicts (paths, output PNG, rendered
    spaces). ``failures`` records graphs that errored so the run is auditable.
    """
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "args": args or {},
        "graph_count": len(entries),
        "failure_count": len(failures or []),
        "graphs": entries,
        "failures": failures or [],
    }
    target = run_dir / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2))
    return target


def manifest_entry(
    record: "GraphRecord", output_png: Path, spaces: list[str]
) -> dict[str, Any]:
    """Build one manifest record for a successfully rendered graph."""
    return {
        "plot_key": record.plot_key,
        "doc": record.doc,
        "png_source": str(record.png_path),
        "json_source": str(record.json_path),
        "series": [str(s.csv_path) for s in record.series],
        "vertical_shift_csv": (
            str(record.vertical_shift_csv) if record.vertical_shift_csv else None
        ),
        "output_png": str(output_png),
        "rendered_spaces": spaces,
    }
