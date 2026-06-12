"""Unit tests for the coordinate-mapping contract (verify.mapping)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Allow `import verify.*` when running pytest from the repo root or elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verify.mapping import (  # noqa: E402
    Calibration,
    map_relative_df,
    map_scaled_df,
    relative_to_pixel,
    scaled_to_relative,
)

# A calibration with a DESCENDING x-axis (3800 -> 3100), like the real data.
CAL = Calibration(
    x1=100.0, y1=20.0, x2=900.0, y2=500.0,
    img_w=1000, img_h=600,
    x_min=3800.0, x_max=3100.0, y_min=0.0, y_max=1.0,
)


def test_relative_corners_map_to_plot_bounds():
    # (0, 0) -> bottom-left pixel: x1, y2
    assert relative_to_pixel(0.0, 0.0, CAL) == pytest.approx((100.0, 500.0))
    # (1, 1) -> top-right pixel: x2, y1
    assert relative_to_pixel(1.0, 1.0, CAL) == pytest.approx((900.0, 20.0))
    # (1, 0) -> bottom-right; (0, 1) -> top-left
    assert relative_to_pixel(1.0, 0.0, CAL) == pytest.approx((900.0, 500.0))
    assert relative_to_pixel(0.0, 1.0, CAL) == pytest.approx((100.0, 20.0))


def test_midpoint_maps_to_box_center():
    px, py = relative_to_pixel(0.5, 0.5, CAL)
    assert px == pytest.approx((100.0 + 900.0) / 2)
    assert py == pytest.approx((20.0 + 500.0) / 2)


def test_descending_axis_round_trips():
    # x_min (3800) -> relative 0 ; x_max (3100) -> relative 1
    assert scaled_to_relative(3800.0, 0.0, CAL) == pytest.approx((0.0, 0.0))
    assert scaled_to_relative(3100.0, 1.0, CAL) == pytest.approx((1.0, 1.0))
    # midpoint value 3450 -> 0.5
    x_rel, _ = scaled_to_relative(3450.0, 0.0, CAL)
    assert x_rel == pytest.approx(0.5)


def test_relative_and_scaled_dfs_agree_on_pixels():
    rel = pd.DataFrame({"x_relative": [0.0, 0.5, 1.0], "y_relative": [0.0, 0.5, 1.0]})
    # Same three points expressed in true axis units.
    scaled = pd.DataFrame({"x": [3800.0, 3450.0, 3100.0], "y": [0.0, 0.5, 1.0]})

    rel_px = map_relative_df(rel, CAL)[["pixel_x", "pixel_y"]]
    scaled_px = map_scaled_df(scaled, CAL)[["pixel_x", "pixel_y"]]

    pd.testing.assert_frame_equal(rel_px, scaled_px, check_exact=False)
