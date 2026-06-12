"""Shared configuration constants for the verification tool.

Centralizes the magic numbers (paths, DPI, opacity, colors) used by the
overlay renderer.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
# Default root of the digitizer output and where run folders are written.
DEFAULT_ROOT = Path("output_pdf")
DEFAULT_RUNS_DIR = Path("Graph_digi_compare/runs")

# CSV source folders produced by the digitization pipeline.
RELATIVE_FOLDER = "digitized"          # x_relative, y_relative in [0, 1]
SCALED_FOLDER = "digitized_scaled"     # x, y in true axis units

# --- Rendering -------------------------------------------------------------
DEFAULT_DPI = 300
DEFAULT_IMG_ALPHA = 0.5

# High-contrast palette cycled across series (color-blind friendly-ish).
SERIES_COLORS = (
    "#e6194B",  # red
    "#3cb44b",  # green
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#f032e6",  # magenta
    "#bfef45",  # lime
)

PLOT_BOUNDS_COLOR = "#222222"
VERTICAL_SHIFT_COLOR = "#000000"


def series_color(index: int) -> str:
    """Return a stable color for the ``index``-th series (wraps around)."""
    return SERIES_COLORS[index % len(SERIES_COLORS)]
