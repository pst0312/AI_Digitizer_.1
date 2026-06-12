"""Agent D — pixel-space overlay (matplotlib).

Renders one comparison figure per graph: the original cutout PNG as a
semi-transparent background with the digitized CSV curves drawn on top at the
exact pixel coordinates implied by the calibration ``plot_bounds``. Because the
overlay lives in raw image pixels, alignment reflects digitization (tracing)
accuracy directly.

The renderer only builds and returns the figure; saving the batch of PNGs is the
storage layer's job (see ``verify/storage.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")  # headless: safe for batch runs without a display
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from verify import config  # noqa: E402
from verify.mapping import Calibration, map_to_pixels  # noqa: E402

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from PIL.Image import Image

    from verify.discovery import GraphRecord


def _load_image(png_path):
    from PIL import Image as PILImage

    return PILImage.open(png_path).convert("RGBA")


def _draw_pixel_panel(
    ax: "Axes", record: "GraphRecord", cal: Calibration, img: "Image", alpha: float
) -> None:
    """Image in raw pixels with the CSV curves mapped via plot_bounds."""
    ax.imshow(
        img,
        extent=(0, cal.img_w, cal.img_h, 0),  # origin top-left, y downward
        alpha=alpha,
        zorder=0,
        aspect="equal",
    )

    for i, series in enumerate(record.series):
        mapped = map_to_pixels(pd.read_csv(series.csv_path), cal)
        ax.plot(
            mapped["pixel_x"],
            mapped["pixel_y"],
            color=config.series_color(i),
            linewidth=0.9,
            marker="o",
            markersize=1.6,
            label=series.label,
            zorder=2,
        )

    # plot-bounds sanity rectangle
    ax.add_patch(
        Rectangle(
            (cal.x1, cal.y1),
            cal.x2 - cal.x1,
            cal.y2 - cal.y1,
            fill=False,
            edgecolor=config.PLOT_BOUNDS_COLOR,
            linestyle="--",
            linewidth=0.8,
            zorder=1,
        )
    )

    if record.vertical_shift_csv:
        shift = map_to_pixels(pd.read_csv(record.vertical_shift_csv), cal)
        ax.scatter(
            shift["pixel_x"], shift["pixel_y"],
            marker="x", s=40, color=config.VERTICAL_SHIFT_COLOR,
            label="vertical shift", zorder=3,
        )

    ax.set_xlim(0, cal.img_w)
    ax.set_ylim(cal.img_h, 0)
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")


def build_overlay_figure(
    record: "GraphRecord",
    cal: Calibration,
    *,
    img_alpha: float = config.DEFAULT_IMG_ALPHA,
) -> "Figure":
    """Build the pixel-space overlay figure for one graph. Caller saves it."""
    img = _load_image(record.png_path)

    fig, ax = plt.subplots(figsize=(10, 7))
    _draw_pixel_panel(ax, record, cal, img, img_alpha)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles, labels,
            loc="upper center", bbox_to_anchor=(0.5, -0.08),
            ncol=min(len(labels), 6), fontsize="small", frameon=False,
        )

    fig.suptitle(record.name, fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig
