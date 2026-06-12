"""Agent A — discovery.

Walks the digitizer output tree and pairs each graph-cutout PNG with its
calibration JSON and the digitized series CSVs. Produces immutable
``GraphRecord`` objects consumed by the mapping and overlay layers.

The pipeline lays each graph out as::

    output_pdf/<doc>/page_N/
        graphs/<plot_key>.png          # the cutout image
        graphs/<plot_key>.json         # calibration (NOT the *_pre.json)
        digitized/<plot_key>_series_<k>_<style>.csv      # x_relative, y_relative
        digitized/<plot_key>_vertical_shift.csv          # optional
        digitized_scaled/<plot_key>_series_<k>_<style>.csv  # x, y (true units)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from verify.mapping import Calibration

# Mirrors the digitization pipeline's <plot_key>_series_<k>_<style> CSV naming.
SERIES_PATTERN = re.compile(r"(?P<plot_key>.+?)_series_(?P<series>\d+)_(?P<style>.+)$")


class CalibrationError(ValueError):
    """Raised when a graph's JSON lacks usable plot-bounds / axis calibration."""


@dataclass(frozen=True)
class SeriesData:
    """A single digitized curve within a graph."""

    csv_path: Path
    label: str  # e.g. "series_1_solid"
    style: str  # "solid" | "dashed" | ...
    series_index: int


@dataclass(frozen=True)
class GraphRecord:
    """Everything needed to overlay one graph: image, calibration and series."""

    plot_key: str  # e.g. "page6.1"
    doc: str  # the source paper folder name
    page_dir: Path  # .../page_N
    png_path: Path  # graphs/<plot_key>.png
    json_path: Path  # graphs/<plot_key>.json
    series: list[SeriesData]
    vertical_shift_csv: Path | None

    @property
    def name(self) -> str:
        """Human-friendly identifier used in figure titles and the manifest."""
        return f"{self.doc} / {self.plot_key}"


def _series_for_plot(
    folder: Path, plot_key: str
) -> tuple[list[SeriesData], Path | None]:
    """Collect series CSVs and the optional vertical-shift CSV in ``folder``."""
    series: list[SeriesData] = []
    vertical_shift: Path | None = None
    if not folder.is_dir():
        return series, vertical_shift

    for csv_path in sorted(folder.glob(f"{plot_key}_*.csv")):
        if csv_path.stem == f"{plot_key}_vertical_shift":
            vertical_shift = csv_path
            continue
        match = SERIES_PATTERN.match(csv_path.stem)
        if not match or match.group("plot_key") != plot_key:
            continue
        series.append(
            SeriesData(
                csv_path=csv_path,
                label=f"series_{match.group('series')}_{match.group('style')}",
                style=match.group("style"),
                series_index=int(match.group("series")),
            )
        )

    series.sort(key=lambda s: s.series_index)
    return series, vertical_shift


def discover_graphs(
    root: Path, *, csv_folder: str = "digitized"
) -> list[GraphRecord]:
    """Find every digitized graph under ``root``.

    A graph qualifies when a ``graphs/<plot_key>.png`` has a sibling calibration
    ``<plot_key>.json`` (the ``*_pre.json`` preprocessing sidecar is ignored).
    Series CSVs are read from ``csv_folder`` (``digitized`` or
    ``digitized_scaled``). Results are sorted by ``(doc, plot_key)``.
    """
    records: list[GraphRecord] = []
    for png_path in sorted(root.rglob("graphs/*.png")):
        plot_key = png_path.stem
        json_path = png_path.with_suffix(".json")
        if not json_path.is_file():
            print(f"Skipping {png_path}: no calibration JSON ({json_path.name})")
            continue

        page_dir = png_path.parent.parent  # .../page_N (graphs/ is one level down)
        # doc = the folder directly under the output root.
        doc = png_path.relative_to(root).parts[0]

        series, vertical_shift = _series_for_plot(page_dir / csv_folder, plot_key)
        if not series:
            print(f"Skipping {plot_key}: no series CSVs in {csv_folder}/")
            continue

        records.append(
            GraphRecord(
                plot_key=plot_key,
                doc=doc,
                page_dir=page_dir,
                png_path=png_path,
                json_path=json_path,
                series=series,
                vertical_shift_csv=vertical_shift,
            )
        )

    records.sort(key=lambda r: (r.doc, r.plot_key))
    return records


def load_calibration(record: GraphRecord) -> Calibration:
    """Read ``record``'s JSON into a :class:`Calibration`.

    Raises :class:`CalibrationError` if plot bounds are missing or degenerate
    (zero-width / zero-height box), which would make the pixel mapping blow up.
    """
    data = json.loads(record.json_path.read_text())

    bounds = data.get("plot_bounds")
    if not bounds:
        raise CalibrationError(f"{record.json_path}: missing 'plot_bounds'")

    try:
        x1, y1 = float(bounds["x1"]), float(bounds["y1"])
        x2, y2 = float(bounds["x2"]), float(bounds["y2"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError(f"{record.json_path}: bad plot_bounds: {exc}") from exc

    if x1 == x2 or y1 == y2:
        raise CalibrationError(
            f"{record.json_path}: degenerate plot_bounds (zero width or height)"
        )

    return Calibration(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        img_w=int(data.get("image_width_px", 0)),
        img_h=int(data.get("image_height_px", 0)),
        x_min=float(data.get("axis_x_min", 0.0)),
        x_max=float(data.get("axis_x_max", 1.0)),
        y_min=float(data.get("axis_y_min", 0.0)),
        y_max=float(data.get("axis_y_max", 1.0)),
    )
