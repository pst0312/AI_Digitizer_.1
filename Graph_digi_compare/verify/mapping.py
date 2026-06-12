"""Agent B — coordinate mapping (the core contract of the whole tool).

Pure, side-effect-free transforms between three coordinate systems:

* **relative**   ``(x_relative, y_relative)`` in ``[0, 1]`` (the ``digitized`` CSVs)
* **scaled**     ``(x, y)`` in true axis units (the ``digitized_scaled`` CSVs)
* **pixel**      image pixel coordinates, origin top-left, y growing downward

Pixel mapping uses the calibration's ``plot_bounds`` (the pixel corners of the
axis box, with ``y1`` the top edge and ``y2`` the bottom edge)::

    pixel_x = x1 + x_relative * (x2 - x1)
    pixel_y = y2 - y_relative * (y2 - y1)

The y term is *subtracted* from the bottom edge because relative ``y = 1`` is the
top of the plot (``y1``), while image pixel rows increase downward. Getting this
flip wrong is the classic digitizer-overlay bug, so it is unit-tested.

This module imports no matplotlib and performs no I/O so it can be reused
unchanged by both the static (matplotlib) and interactive (plotly) renderers.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Calibration:
    """Pixel + axis calibration for one graph (loaded from its JSON)."""

    x1: float  # plot-box pixel corners
    y1: float  # top edge (smaller pixel-y)
    x2: float
    y2: float  # bottom edge (larger pixel-y)
    img_w: int
    img_h: int
    x_min: float  # axis data range; x_max may be < x_min (descending axis)
    x_max: float
    y_min: float
    y_max: float


# --- scalar transforms -----------------------------------------------------
def relative_to_pixel(
    x_rel: float, y_rel: float, cal: Calibration
) -> tuple[float, float]:
    """Map a relative ``[0, 1]`` coordinate to image pixel space."""
    px = cal.x1 + x_rel * (cal.x2 - cal.x1)
    py = cal.y2 - y_rel * (cal.y2 - cal.y1)
    return px, py


def scaled_to_relative(x: float, y: float, cal: Calibration) -> tuple[float, float]:
    """Map a true-axis-unit coordinate to relative ``[0, 1]`` space.

    Works for descending axes (``x_max < x_min``): the division by a negative
    span flips the sign consistently so endpoints still land at 0 and 1.
    """
    x_rel = (x - cal.x_min) / (cal.x_max - cal.x_min)
    y_rel = (y - cal.y_min) / (cal.y_max - cal.y_min)
    return x_rel, y_rel


# --- vectorized DataFrame transforms ---------------------------------------
def _detect_kind(df: pd.DataFrame) -> str:
    """Return ``"relative"`` or ``"scaled"`` based on the DataFrame columns."""
    if {"x_relative", "y_relative"} <= set(df.columns):
        return "relative"
    if {"x", "y"} <= set(df.columns):
        return "scaled"
    raise ValueError(
        f"DataFrame has neither (x_relative, y_relative) nor (x, y); got {list(df.columns)}"
    )


def map_relative_df(df: pd.DataFrame, cal: Calibration) -> pd.DataFrame:
    """Add ``pixel_x``/``pixel_y`` columns for an ``x_relative,y_relative`` frame."""
    out = df.copy()
    out["pixel_x"] = cal.x1 + out["x_relative"] * (cal.x2 - cal.x1)
    out["pixel_y"] = cal.y2 - out["y_relative"] * (cal.y2 - cal.y1)
    return out


def map_scaled_df(df: pd.DataFrame, cal: Calibration) -> pd.DataFrame:
    """Add ``pixel_x``/``pixel_y`` columns for an ``x,y`` (true-units) frame."""
    out = df.copy()
    x_rel = (out["x"] - cal.x_min) / (cal.x_max - cal.x_min)
    y_rel = (out["y"] - cal.y_min) / (cal.y_max - cal.y_min)
    out["pixel_x"] = cal.x1 + x_rel * (cal.x2 - cal.x1)
    out["pixel_y"] = cal.y2 - y_rel * (cal.y2 - cal.y1)
    return out


def map_to_pixels(df: pd.DataFrame, cal: Calibration) -> pd.DataFrame:
    """Auto-detect the column convention and add ``pixel_x``/``pixel_y``.

    Accepts either ``digitized`` (``x_relative,y_relative``) or
    ``digitized_scaled`` (``x,y``) frames, so the overlay maps both to the same
    pixel coordinates.
    """
    if _detect_kind(df) == "relative":
        return map_relative_df(df, cal)
    return map_scaled_df(df, cal)
