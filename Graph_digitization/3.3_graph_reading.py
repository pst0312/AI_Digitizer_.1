#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "output_pdf"

# Run/tracking parameters for geometric series separation
RUN_GAP_PX = 2                     # vertical gap that splits one column run into two
MAX_RUN_HEIGHT_FRACTION = 0.95     # runs taller than this fraction of plot height are axis junk
STRUCTURAL_LINE_FRACTION = 0.7     # lines spanning this fraction of width/height are frame, not data
FULL_HEIGHT_LINE_FRACTION = 0.92   # vertical strokes shorter than this are data (spikes), not frame
EDGE_BAND_FRACTION = 0.03          # runs entirely inside this top/bottom edge band are frame remnants
BASE_MATCH_TOL_FRACTION = 0.025    # matching tolerance right after a confirmed point
GAP_TOL_GROWTH_FRACTION = 0.015    # tolerance growth per skipped column (dashed gaps)
MAX_MATCH_TOL_FRACTION = 0.15      # tolerance ceiling
CONVERGE_GAP_COLS = 3              # consecutive misses before a series may share a nearby run
CONVERGE_SHARE_TOL_FACTOR = 2.0    # share distance ceiling (xbase_tol) for convergence
ENDPOINT_SNAP_FRACTION = 0.05      # snap a series' end to the shared endpoint within this
CHAIN_GAP_PX = 12                  # vertical gap bridged when stacking dot/dash runs in a column
SPIKE_RUN_FACTOR = 3.0             # interval taller than this x line thickness = steep stroke
SPIKE_BLOCK_MAX_SKIP = 3           # tall columns this close together belong to one steep stroke
STEEP_TOL_FACTOR = 1.0             # extra tolerance per pixel of last-interval height on steep strokes
STEEP_TOL_MAX_FRACTION = 0.30      # ceiling for that steep-stroke tolerance bonus
JUMP_HISTORY_COLS = 5              # recent matches remembered for the jump-size gate bonus
STYLE_PENALTY_FRACTION = 0.03      # ranking penalty for runs whose style mismatches the series
SOLID_COMPONENT_MIN_FRACTION = 0.05  # component x-extent (fraction of width) marking a solid stroke


def parse_args():
    parser = argparse.ArgumentParser(
        description="Digitize graphs by intensity clustering and skeleton pixel tracing"
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=None,
        help="Root output_pdf folder (default: output_pdf)",
    )
    parser.add_argument(
        "--single",
        metavar="PNG_PATH",
        help="Process one specific graph PNG path instead of walking the full tree",
    )
    return parser.parse_args()


def is_graphs_png(png_path: Path) -> bool:
    return (
        png_path.is_file()
        and png_path.suffix.lower() == ".png"
        and png_path.parent.name.lower() == "graphs"
    )


def find_graphs_folders(root_dir: Path):
    for current_root, dirs, _ in os.walk(str(root_dir)):
        if os.path.basename(current_root).lower() == "graphs":
            continue
        if "graphs" not in dirs:
            continue
        graphs_path = Path(current_root) / "graphs"
        dirs.remove("graphs")
        if graphs_path.is_dir():
            yield graphs_path


def resolve_single_path(single_arg: str, root_dir: Path) -> Path:
    candidate = Path(single_arg)
    if not candidate.is_absolute():
        candidate = root_dir / single_arg
    return candidate.resolve()


def safe_load_json(json_path: Path):
    if not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def clamp_plot_bounds(bounds: dict, image_width: int, image_height: int):
    x1 = int(round(bounds.get("x1", 0)))
    y1 = int(round(bounds.get("y1", 0)))
    x2 = int(round(bounds.get("x2", 0)))
    y2 = int(round(bounds.get("y2", 0)))

    x1 = max(0, min(x1, image_width - 1))
    x2 = max(0, min(x2, image_width))
    y1 = max(0, min(y1, image_height - 1))
    y2 = max(0, min(y2, image_height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid plot_bounds after clamping: {(x1, y1, x2, y2)} "
            f"for image size {(image_width, image_height)}"
        )

    return x1, y1, x2, y2


def crop_plot(image: np.ndarray, bounds: dict, image_width: int, image_height: int):
    x1, y1, x2, y2 = clamp_plot_bounds(bounds, image_width, image_height)
    return image[y1:y2, x1:x2], x1, y1, x2, y2


def detect_vertical_markers(gray_crop: np.ndarray, manual_markers=None):
    height, width = gray_crop.shape[:2]
    if height < 5 or width < 5:
        return [], np.zeros((height, width), dtype=np.uint8), gray_crop.copy()

    marker_mask = np.zeros((height, width), dtype=np.uint8)
    marker_positions = []

    # Scenario A: Use explicit crop-relative marker columns if provided.
    # (Callers convert absolute JSON markers to crop columns before passing.)
    if manual_markers and len(manual_markers) > 0:
        marker_positions = sorted(list(set([int(round(x)) for x in manual_markers])))
        
        # Draw vertical mask bands around each known shift marker location
        # A thickness buffer of 3-5 pixels ensures we catch bleeding pixels
        buffer_px = 3 
        for x_pos in marker_positions:
            x1 = max(0, x_pos - buffer_px)
            x2 = min(width, x_pos + buffer_px + 1)
            cv2.rectangle(marker_mask, (x1, 0), (x2, height), 255, thickness=-1)

    # Scenario B: Fall back to automatic morphology if JSON has no markers.
    # The span bar matches 3.2.1's shift-line criterion (0.85): anything shorter
    # is a near-vertical data spike, and inpainting it would erase a real peak.
    else:
        _, bin_inv = cv2.threshold(
            gray_crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        kernel_height = max(3, int(round(height * 0.85)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))
        vertical_candidates = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            vertical_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h < int(round(height * 0.85)):
                continue
            cv2.drawContours(marker_mask, [contour], -1, 255, thickness=-1)
            marker_positions.append(int(round(x + w / 2)))
        marker_positions = sorted(set(marker_positions))

    # Inpaint the regions to seamlessly wipe them out from the data path tracing
    if np.any(marker_mask):
        inpainted = cv2.inpaint(gray_crop, marker_mask, 5, cv2.INPAINT_TELEA)
    else:
        inpainted = gray_crop.copy()

    return marker_positions, marker_mask, inpainted

def extract_dark_pixels(gray_image: np.ndarray):
    _, line_mask = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    ys, xs = np.where(line_mask > 0)
    return xs, ys, line_mask


def remove_structural_lines(line_mask: np.ndarray) -> np.ndarray:
    """Subtract long horizontal/vertical lines (plot frame, gridlines) from the mask."""
    height, width = line_mask.shape[:2]
    h_len = max(3, int(round(width * STRUCTURAL_LINE_FRACTION)))
    v_len = max(3, int(round(height * STRUCTURAL_LINE_FRACTION)))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    horiz = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, horiz_kernel)
    vert = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, vert_kernel)

    # The vertical opening also catches near-vertical spike strokes (sharp peaks
    # and absorption dips). Only frame borders and full-height gridlines are
    # structural: keep a vertical stroke for subtraction only if it spans nearly
    # the whole crop height or sits at the left/right crop border.
    full_height = int(round(height * FULL_HEIGHT_LINE_FRACTION))
    edge_margin = max(5, int(round(width * 0.02)))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(vert, connectivity=8)
    vert_structural = np.zeros_like(vert)
    for label in range(1, n_labels):
        x, _, w, h, _ = stats[label]
        if h >= full_height or x <= edge_margin or (x + w) >= width - edge_margin:
            vert_structural[labels == label] = 255

    cleaned = cv2.subtract(line_mask, cv2.add(horiz, vert_structural))
    return cleaned if np.any(cleaned) else line_mask


def extract_column_runs(line_mask: np.ndarray):
    """For each column, collapse contiguous dark-pixel runs to (center, top, bottom)."""
    height, width = line_mask.shape[:2]
    max_run_height = max(3, int(round(height * MAX_RUN_HEIGHT_FRACTION)))
    edge_band = max(2, int(round(height * EDGE_BAND_FRACTION)))
    runs_per_column = []
    for x in range(width):
        ys = np.flatnonzero(line_mask[:, x])
        if ys.size == 0:
            runs_per_column.append([])
            continue
        split_points = np.flatnonzero(np.diff(ys) > RUN_GAP_PX)
        segments = np.split(ys, split_points + 1)
        # Drop runs whose CENTER sits inside the top/bottom edge band: those are
        # frame-line remnants, and letting a series anchor to them creates a flat
        # ghost stroke along the plot border.
        runs = [
            (float(seg.mean()), float(seg[0]), float(seg[-1]))
            for seg in segments
            if seg.size <= max_run_height
            and edge_band <= seg.mean() <= height - 1 - edge_band
        ]
        runs_per_column.append(runs)
    return runs_per_column


def classify_runs(line_mask: np.ndarray, runs_per_column):
    """Label each run 'solid' or 'dashed' from its connected component's x-extent.

    Dash/dot fragments are small in BOTH dimensions; a solid curve forms
    components that are long in at least one direction (wide normally, tall on
    a near-vertical flank even if a scan artifact severs it from the rest of
    the curve). At a solid/dashed intersection the merged component is large,
    so touching dashes read as 'solid' there - harmless, since distance still
    dominates matching.
    """
    height, width = line_mask.shape[:2]
    solid_min = max(20, int(round(width * SOLID_COMPONENT_MIN_FRACTION)))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(line_mask, connectivity=8)

    run_classes_per_column = []
    for x, runs in enumerate(runs_per_column):
        classes = []
        for _, top, bottom in runs:
            seg = labels[int(top) : int(bottom) + 1, x]
            seg = seg[seg > 0]
            if seg.size == 0:
                classes.append("dashed")
                continue
            label = int(np.bincount(seg).argmax())
            comp_extent = max(
                int(stats[label, cv2.CC_STAT_WIDTH]),
                int(stats[label, cv2.CC_STAT_HEIGHT]),
            )
            classes.append("solid" if comp_extent >= solid_min else "dashed")
        run_classes_per_column.append(classes)
    return run_classes_per_column


def find_seed_column(runs_per_column, num_series: int):
    """Pick the middle of the longest stretch of columns with exactly num_series runs."""
    best_start, best_length = -1, 0
    start = -1
    for x, runs in enumerate(runs_per_column + [[]]):
        if len(runs) == num_series:
            if start < 0:
                start = x
        elif start >= 0:
            if x - start > best_length:
                best_start, best_length = start, x - start
            start = -1
    if best_length == 0:
        return None
    return best_start + best_length // 2


def segment_regions(width: int, marker_cols: np.ndarray):
    """Split [0, width) into maximal column ranges separated by marker bands.

    With no markers this yields a single whole-crop region, so tracking behaves
    exactly as the old single-seed path.
    """
    regions = []
    start = None
    for x in range(width):
        if marker_cols[x]:
            if start is not None:
                regions.append((start, x - 1))
                start = None
        elif start is None:
            start = x
    if start is not None:
        regions.append((start, width - 1))
    return regions


def chain_stacked_runs(runs, assignment):
    """Extend each matched run with unclaimed runs stacked within CHAIN_GAP_PX.

    Dotted/dashed strokes on a steep flank appear as several short runs stacked
    vertically in one column; only one can be matched directly, so absorb the
    rest into the matched interval. Runs claimed by any series never chain, so a
    neighbouring curve's own stroke cannot be swallowed.
    """
    claimed = set(assignment.values())
    intervals = {}
    for s, r in assignment.items():
        top, bottom = runs[r][1], runs[r][2]
        changed = True
        while changed:
            changed = False
            for idx, (_, run_top, run_bottom) in enumerate(runs):
                if idx in claimed:
                    continue
                if run_top > bottom:
                    gap = run_top - bottom
                elif top > run_bottom:
                    gap = top - run_bottom
                else:
                    gap = 0.0
                if gap <= CHAIN_GAP_PX:
                    top = min(top, run_top)
                    bottom = max(bottom, run_bottom)
                    claimed.add(idx)
                    changed = True
        intervals[s] = (top, bottom)
    return intervals


def seed_and_track_region(
    runs_per_column,
    run_classes,
    series_styles,
    num_series,
    lo,
    hi,
    height,
    series_y,
    series_iv,
    style_counts,
    min_share_gap,
):
    """Seed within [lo, hi] (top-to-bottom) and track outward, staying in-region.

    Returns True if the region was seeded. Series identity is the top-to-bottom
    order at the region's seed column, so the same series index lines up across
    regions split by a shift line.
    """
    region_runs = runs_per_column[lo : hi + 1]
    local_seed = find_seed_column(region_runs, num_series)
    if local_seed is None:
        return False
    seed_x = lo + local_seed

    seed_order = sorted(range(len(runs_per_column[seed_x])), key=lambda r: runs_per_column[seed_x][r][0])
    seed_runs = [runs_per_column[seed_x][r] for r in seed_order]
    for s in range(num_series):
        series_y[s, seed_x] = seed_runs[s][0]
        series_iv[s, seed_x] = (seed_runs[s][1], seed_runs[s][2])
        style_counts[s][run_classes[seed_x][seed_order[s]]] += 1

    for direction in (1, -1):
        last_intervals = [(run[1], run[2]) for run in seed_runs]
        last_x = [seed_x] * num_series
        last_center = [run[0] for run in seed_runs]
        jump_history = [[] for _ in range(num_series)]
        x = seed_x + direction
        while lo <= x <= hi:
            runs = runs_per_column[x]
            if runs:
                # A dotted curve advances in vertical steps of roughly its dot
                # pitch; the series' own recent jumps tell the gate how far the
                # next dot may legitimately sit.
                jump_bonus = [
                    max(hist) if hist else 0.0 for hist in jump_history
                ]
                assignment = match_runs_to_series(
                    runs,
                    run_classes[x],
                    series_styles,
                    last_intervals,
                    last_x,
                    x,
                    height,
                    min_share_gap,
                    jump_bonus,
                )
                chained = chain_stacked_runs(runs, assignment)
                for s, r in assignment.items():
                    top, bottom = chained[s]
                    if (top, bottom) == (runs[r][1], runs[r][2]):
                        series_y[s, x] = runs[r][0]
                    else:
                        series_y[s, x] = 0.5 * (top + bottom)
                    series_iv[s, x] = (top, bottom)
                    last_intervals[s] = (top, bottom)
                    last_x[s] = x
                    jump_history[s].append(abs(series_y[s, x] - last_center[s]))
                    if len(jump_history[s]) > JUMP_HISTORY_COLS:
                        jump_history[s].pop(0)
                    last_center[s] = series_y[s, x]
                    style_counts[s][run_classes[x][r]] += 1
            x += direction
    return True


def match_runs_to_series(
    runs,
    run_classes,
    series_styles,
    last_intervals,
    last_x,
    x,
    height,
    min_share_gap,
    jump_bonus=None,
):
    """Assign column runs to series by proximity to each series' last matched run.

    Distance is measured between [top, bottom] intervals, so a series that just
    traversed a tall near-vertical stroke (sharp peak/dip) can continue from
    either end of it. When series styles are known (second tracking pass), a run
    whose solid/dashed class mismatches the series is ranked behind same-style
    runs at comparable distance, which keeps identities straight at crossings.
    """
    num_series = len(last_intervals)
    base_tol = height * BASE_MATCH_TOL_FRACTION
    growth = height * GAP_TOL_GROWTH_FRACTION
    tol_cap = height * MAX_MATCH_TOL_FRACTION
    style_penalty = height * STYLE_PENALTY_FRACTION

    gaps = [abs(x - last_x[s]) for s in range(num_series)]
    # A tall last interval means the series is on a near-vertical stroke (or a
    # chained stack of dots on one): its continuation in the next column can be
    # displaced by roughly that stroke's height, so widen the gate accordingly.
    steep_cap = height * STEEP_TOL_MAX_FRACTION
    steep_bonus = [
        min(steep_cap, STEEP_TOL_FACTOR * (last_intervals[s][1] - last_intervals[s][0]))
        for s in range(num_series)
    ]
    if jump_bonus is None:
        jump_bonus = [0.0] * num_series
    tolerances = [
        min(tol_cap, base_tol + growth * max(0, gaps[s] - 1))
        + steep_bonus[s]
        + min(steep_cap, jump_bonus[s])
        for s in range(num_series)
    ]

    def interval_distance(series_idx, run_idx):
        _, run_top, run_bottom = runs[run_idx]
        last_top, last_bottom = last_intervals[series_idx]
        if run_top > last_bottom:
            return run_top - last_bottom
        if last_top > run_bottom:
            return last_top - run_bottom
        return 0.0

    def style_matches(series_idx, run_idx):
        if series_styles is None or series_styles[series_idx] is None:
            return True
        return series_styles[series_idx] == run_classes[run_idx]

    # Candidates gate on raw distance (so a lone mismatched run is still usable)
    # but rank by distance plus a penalty for style mismatch. A tall candidate
    # run is itself a steep stroke; the series attaches at its near end, so its
    # height also relaxes the gate (mirror of the last-interval steep bonus).
    candidates = []
    for s in range(num_series):
        for r in range(len(runs)):
            dist = interval_distance(s, r)
            run_bonus = min(steep_cap, 0.5 * (runs[r][2] - runs[r][1]))
            if dist > tolerances[s] + run_bonus:
                continue
            rank = dist if style_matches(s, r) else dist + style_penalty
            candidates.append((rank, dist, s, r))
    candidates.sort()

    # First pass: unique greedy matching so re-splitting curves separate again
    assignment = {}
    used_runs = set()
    for rank, dist, s, r in candidates:
        if s in assignment or r in used_runs:
            continue
        assignment[s] = r
        used_runs.add(r)

    # Second pass: let unmatched series share an already claimed run when curves
    # converge or a series is recovering a lost curve. Three escalating cases:
    #   - tight distance: curves gradually merging onto the same stroke;
    #   - a few consecutive misses with a nearby same-style run: a converging
    #     series whose own (e.g. dashed) curve dropped out should follow the
    #     neighbour to the shared endpoint rather than flat-line;
    #   - a long gap: the series lost its curve entirely and is recovering.
    # Plain short dashed-gap columns still stay unassigned so a dotted series does
    # not zigzag onto a close neighbour mid-trace, and a dashed series in a gap
    # cannot hijack a crossing solid stroke (style must match to converge-share).
    converge_tol = base_tol * CONVERGE_SHARE_TOL_FACTOR
    for rank, dist, s, r in candidates:
        if s in assignment:
            continue
        # Tight-distance sharing is for genuinely coincident curves, which meet
        # on a thin stroke. A tall steep stroke of the other style is a
        # neighbour's peak flank: a dashed series must coast its dash gap there,
        # not ride the solid line away from its own curve.
        run_height = runs[r][2] - runs[r][1] + 1.0
        coincident = style_matches(s, r) or run_height <= base_tol
        if (
            (dist <= base_tol and coincident)
            or (
                gaps[s] >= CONVERGE_GAP_COLS
                and dist <= converge_tol
                and style_matches(s, r)
            )
            or gaps[s] >= min_share_gap
        ):
            assignment[s] = r

    return assignment


def refine_steep_strokes(series_y: np.ndarray, series_iv: np.ndarray):
    """Rewrite y values inside steep strokes so spike tips are not clipped.

    Tracking stores one matched interval per column; emitting its center halves
    the amplitude of any narrow peak/dip whose stroke is much taller than the
    line thickness. For each block of tall-interval columns decide, from the
    flanking trace, whether it is a spike (emit the interval end pointing away
    from the baseline, capturing the tip) or a monotonic steep flank (emit a
    linear path clipped into each column's interval).
    """
    num_series, width = series_y.shape
    for s in range(num_series):
        cols = np.flatnonzero(~np.isnan(series_y[s]))
        if cols.size < 3:
            continue
        tops = series_iv[s, cols, 0]
        bottoms = series_iv[s, cols, 1]
        heights = bottoms - tops + 1.0
        typical = float(np.median(heights))
        tall_thresh = max(10.0, typical * SPIKE_RUN_FACTOR)
        tall = heights > tall_thresh

        i = 0
        n = cols.size
        while i < n:
            if not tall[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and tall[j + 1] and cols[j + 1] - cols[j] <= SPIKE_BLOCK_MAX_SKIP:
                j += 1

            block = cols[i : j + 1]
            block_tops = series_iv[s, block, 0]
            block_bottoms = series_iv[s, block, 1]
            block_top = float(block_tops.min())
            block_bottom = float(block_bottoms.max())

            y_left = float(series_y[s, cols[i - 1]]) if i > 0 else None
            y_right = float(series_y[s, cols[j + 1]]) if j + 1 < n else None
            if y_left is None and y_right is None:
                i = j + 1
                continue
            if y_left is None:
                y_left = y_right
            if y_right is None:
                y_right = y_left

            if abs(y_left - y_right) >= 0.5 * (block_bottom - block_top):
                # Steep monotonic flank: walk from one side to the other,
                # constrained to each column's actual stroke extent.
                targets = np.linspace(y_left, y_right, block.size + 2)[1:-1]
                series_y[s, block] = np.clip(targets, block_tops, block_bottoms)
            else:
                # Spike: emit the stroke edge pointing away from the baseline so
                # the extremum column carries the full tip value.
                baseline = 0.5 * (y_left + y_right)
                if (block_bottom - baseline) >= (baseline - block_top):
                    series_y[s, block] = block_bottoms
                else:
                    series_y[s, block] = block_tops
            i = j + 1


def cluster_series(gray_image: np.ndarray, num_series: int, marker_cols=None):
    xs, ys, line_mask = extract_dark_pixels(gray_image)
    if xs.size == 0:
        raise ValueError("No dark pixels found inside plot crop")

    height, width = gray_image.shape[:2]
    line_mask = remove_structural_lines(line_mask)
    runs_per_column = extract_column_runs(line_mask)
    run_classes = classify_runs(line_mask, runs_per_column)

    # Columns covered by a vertical shift-line band carry no real curve data:
    # blank their runs so no series can be matched across the discontinuity.
    if marker_cols is None:
        marker_cols = np.zeros(width, dtype=bool)
    else:
        marker_cols = np.asarray(marker_cols, dtype=bool)
        for x in range(min(width, marker_cols.size)):
            if marker_cols[x]:
                runs_per_column[x] = []
                run_classes[x] = []

    min_share_gap = max(10, int(round(width * 0.02)))
    regions = segment_regions(width, marker_cols)

    def track_all(series_styles):
        """One full tracking pass; returns (series_y, series_iv, style_counts)."""
        series_y = np.full((num_series, width), np.nan, dtype=np.float64)
        series_iv = np.full((num_series, width, 2), np.nan, dtype=np.float64)
        style_counts = [{"solid": 0, "dashed": 0} for _ in range(num_series)]

        # Seed and track each shift-line-delimited region independently. A single
        # seed in one region cannot follow a curve across a shift-line break, so
        # the opposite side would otherwise be left to interpolate a diagonal.
        seeded_any = False
        for lo, hi in regions:
            if seed_and_track_region(
                runs_per_column, run_classes, series_styles, num_series,
                lo, hi, height, series_y, series_iv, style_counts, min_share_gap,
            ):
                seeded_any = True

        # Fallback: if no region had a clean seed column, seed once over the
        # whole crop so a graph still digitizes rather than fails.
        if not seeded_any:
            if find_seed_column(runs_per_column, num_series) is None:
                raise ValueError(
                    f"No column with exactly {num_series} separable runs; cannot seed tracking"
                )
            seed_and_track_region(
                runs_per_column, run_classes, series_styles, num_series,
                0, width - 1, height, series_y, series_iv, style_counts, min_share_gap,
            )
        return series_y, series_iv, style_counts

    # Pass 1 (style-blind) establishes which series is solid vs dashed from the
    # connected-component class of the runs it matched. Pass 2 re-tracks with
    # those styles so each series prefers strokes of its own kind at crossings.
    _, _, style_counts = track_all(None)
    series_styles = [
        ("solid" if counts["solid"] >= counts["dashed"] else "dashed")
        if (counts["solid"] + counts["dashed"]) > 0
        else None
        for counts in style_counts
    ]
    series_y, series_iv, _ = track_all(series_styles)

    refine_steep_strokes(series_y, series_iv)

    # Shared endpoints: a series that tracked to within ENDPOINT_SNAP_FRACTION of
    # the outermost tracked column is snapped out to it, so converging curves end
    # together. Series that fall short by more than that are left to terminate at
    # their own last real column instead of being flat-extrapolated to the edge.
    assigned_per_series = [
        np.flatnonzero(~np.isnan(series_y[c])) for c in range(num_series)
    ]
    non_empty = [a for a in assigned_per_series if a.size]
    common_first = int(min((a[0] for a in non_empty), default=0))
    common_last = int(max((a[-1] for a in non_empty), default=width - 1))
    snap_tol = max(1, int(round(width * ENDPOINT_SNAP_FRACTION)))

    clusters = []
    for cluster_id in range(num_series):
        row = series_y[cluster_id]
        assigned = assigned_per_series[cluster_id]
        point_count = int(assigned.size)
        coverage_ratio = float(point_count) / float(width) if width > 0 else 0.0

        if point_count == 0:
            clusters.append(
                {
                    "id": cluster_id + 1,
                    "style": "solid",
                    "intensity_center": 0.0,
                    "point_count": 0,
                    "coverage_ratio": 0.0,
                    "pixels": [],
                }
            )
            continue

        sample_y = np.clip(np.round(row[assigned]).astype(int), 0, height - 1)
        intensity_center = float(gray_image[sample_y, assigned].mean())

        first_col, last_col = int(assigned[0]), int(assigned[-1])
        lo_col = common_first if (first_col - common_first) <= snap_tol else first_col
        hi_col = common_last if (common_last - last_col) <= snap_tol else last_col

        filled = np.interp(np.arange(width), assigned, row[assigned])
        # Emit points only within this series' (snapped) tracked span, skipping
        # shift-line columns. Outside the span we do not fabricate a y; inside,
        # np.interp bridges short dashed gaps.
        points = [
            (
                float(x) / float(width),
                1.0 - float(filled[x]) / float(height),
            )
            for x in range(lo_col, hi_col + 1)
            if not marker_cols[x]
        ]

        # Prefer the connected-component style from tracking; fall back to the
        # old coverage heuristic only if the pass-1 style could not be decided.
        style = series_styles[cluster_id] or (
            "solid" if coverage_ratio > 0.85 else "dashed"
        )

        clusters.append(
            {
                "id": cluster_id + 1,
                "style": style,
                "intensity_center": float(round(intensity_center, 2)),
                "point_count": point_count,
                "coverage_ratio": float(round(coverage_ratio, 4)),
                "pixels": points,
            }
        )

    return clusters


def write_series_csv(csv_path: Path, points):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write("x_relative,y_relative\n")
        for x_rel, y_rel in points:
            f.write(f"{x_rel:.6f},{y_rel:.6f}\n")


def process_graph_png(png_path: Path, json_path: Path, root_dir: Path, summary: dict):
    metadata = safe_load_json(json_path)
    if metadata is None:
        summary["skipped"] += 1
        return

    if metadata.get("vision_extracted") is not True or metadata.get("image_measured") is not True:
        summary["skipped"] += 1
        return

    plot_bounds = metadata.get("plot_bounds")
    if not isinstance(plot_bounds, dict):
        summary["skipped"] += 1
        return

    num_series = metadata.get("num_series")
    try:
        num_series = int(num_series)
    except Exception:
        summary["skipped"] += 1
        return

    if num_series <= 0:
        summary["skipped"] += 1
        return

    image_width = int(metadata.get("image_width_px", 0))
    image_height = int(metadata.get("image_height_px", 0))
    if image_width <= 0 or image_height <= 0:
        summary["skipped"] += 1
        return

    try:
        with Image.open(png_path) as img:
            img = img.convert("L")
            image = np.array(img)
    except Exception:
        summary["failed"] += 1
        return

    try:
        cropped, x1, y1, x2, y2 = crop_plot(image, plot_bounds, image_width, image_height)
    except Exception:
        summary["failed"] += 1
        return

    try:
        # vertical_markers_px in the JSON are ABSOLUTE image x. The crop used for
        # tracing starts at x1, so convert each marker to a crop-relative column
        # and discard any that fall outside this crop before masking.
        crop_width = cropped.shape[1]
        manual_markers_abs = metadata.get("vertical_markers_px", []) or []
        manual_markers_crop = [
            int(round(x)) - x1
            for x in manual_markers_abs
            if 0 <= int(round(x)) - x1 < crop_width
        ]

        marker_positions_crop, marker_mask, cleaned = detect_vertical_markers(
            cropped,
            manual_markers=manual_markers_crop,
        )
    except Exception:
        summary["failed"] += 1
        return

    try:
        # Columns touched by the shift-line mask are no-data; pass them through so
        # the tracer neither matches nor emits fabricated points there.
        marker_cols = np.any(marker_mask > 0, axis=0) if marker_mask.size else None
        clusters = cluster_series(cleaned, num_series, marker_cols=marker_cols)
    except Exception:
        summary["failed"] += 1
        return

    page_dir = png_path.parent.parent
    digitized_dir = page_dir / "digitized"

    # A series' style label is part of its CSV filename, so a style flip leaves
    # the old file behind; purge this graph's previous series CSVs (and their
    # scaled copies) before writing the fresh set.
    for old_dir in (digitized_dir, page_dir / "digitized_scaled"):
        for stale in old_dir.glob(f"{png_path.stem}_series_*.csv"):
            stale.unlink()

    csv_entries = []
    for cluster in clusters:
        csv_name = f"{png_path.stem}_series_{cluster['id']}_{cluster['style']}.csv"
        csv_path = digitized_dir / csv_name
        write_series_csv(csv_path, cluster["pixels"])
        cluster["csv_path"] = str(Path("digitized") / csv_name).replace("\\", "/")
        del cluster["pixels"]
        csv_entries.append(cluster)

    # Convert crop-relative marker columns back to the storage contract:
    # vertical_markers_px = absolute image x; relative = crop_column / crop_width.
    vertical_markers_px = [x1 + cx for cx in marker_positions_crop]
    relative_markers = [round(cx / float(crop_width), 4) for cx in marker_positions_crop]
    needs_manual = any(cluster["coverage_ratio"] < 0.70 for cluster in csv_entries)

    metadata.update(
        {
            "digitized": True,
            "digitization_method": "intensity_cluster_skeleton",
            "axis_scale": "relative",
            "vertical_markers_px": vertical_markers_px,
            "vertical_markers_relative_x": relative_markers,
            "series": csv_entries,
            "needs_manual_review": needs_manual,
        }
    )

    try:
        json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        summary["digitized"] += 1
        if needs_manual:
            summary["flagged"] += 1
    except Exception:
        summary["failed"] += 1


def process_graphs_folder(graphs_path: Path, root_dir: Path, summary: dict):
    for entry in sorted(graphs_path.glob("*.png")):
        summary["total_pngs"] += 1
        process_graph_png(entry, entry.with_suffix(".json"), root_dir, summary)


def main():
    args = parse_args()
    root_dir = Path(args.root).resolve() if args.root else DEFAULT_ROOT.resolve()
    if not root_dir.is_dir():
        raise SystemExit(f"Root directory not found: {root_dir}")

    summary = {
        "total_pngs": 0,
        "skipped": 0,
        "digitized": 0,
        "flagged": 0,
        "failed": 0,
    }

    if args.single:
        png_path = resolve_single_path(args.single, root_dir)
        if not is_graphs_png(png_path):
            raise SystemExit(
                f"Single file must be an existing PNG inside a graphs/ folder: {args.single}"
            )
        summary["total_pngs"] = 1
        process_graph_png(png_path, png_path.with_suffix(".json"), root_dir, summary)
    else:
        for graphs_path in find_graphs_folders(root_dir):
            process_graphs_folder(graphs_path, root_dir, summary)

    print("Processing complete.")
    print(f"Total PNGs found: {summary['total_pngs']}")
    print(f"Skipped: {summary['skipped']}")
    print(f"Digitized: {summary['digitized']}")
    print(f"Flagged for review: {summary['flagged']}")
    print(f"Failed: {summary['failed']}")


if __name__ == "__main__":
    main()
