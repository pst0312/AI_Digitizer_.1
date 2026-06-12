from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt


def prompt_mode():
    while True:
        resp = input("Select data collection mode:\n  1 - Minimal (plot bounds + series count only)\n  2 - Full (plot bounds + series count + axis labels, units, and limits)\nEnter 1 or 2: ")
        if resp in ("1", "2"):
            return int(resp)
        print("Please enter exactly 1 or 2.")


def prompt_positive_int(prompt_text):
    while True:
        val = input(prompt_text)
        try:
            ival = int(val)
            if ival > 0:
                return ival
        except Exception:
            pass
        print("Invalid input — please enter a positive integer.")


def prompt_float(prompt_text, allow_blank=False):
    while True:
        val = input(prompt_text)
        if allow_blank and val == "":
            return ""
        try:
            return float(val)
        except Exception:
            print("Invalid input — please enter a number.")


def process_png(png_path: Path, json_path: Path, mode: int):
    """Process a single PNG: show image, capture two clicks, then prompt CLI inputs.

    Coordinate derivation comment:
    matplotlib returns image-data coordinates with y=0 at the top. We receive two
    clicks: click1 (axis origin bottom-left), click2 (top-right corner). To derive
    plot bounds (x1,y1,x2,y2) in image coordinates:
      - x1 = click1_x  (left edge, from origin click)
      - y1 = click2_y  (top edge, from top-right click)
      - x2 = click2_x  (right edge, from top-right click)
      - y2 = click1_y  (bottom edge, from origin click)
    These are stored as floats rounded to 1 decimal place.
    """

    existing = {}
    if json_path.exists():
        try:
            existing = json.loads(json_path.read_text())
        except Exception:
            existing = {}

    # If vision_extracted is present and True, caller will decide whether to skip.

    # Load and show image
    img = plt.imread(str(png_path))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.set_title("Click 1: axis origin (bottom-left) | Click 2: top-right corner")

    # Display non-blocking so the window remains visible during CLI prompts
    plt.show(block=False)
    plt.pause(0.1)

    try:
        clicks = plt.ginput(2, timeout=0)
        if len(clicks) != 2:
            raise RuntimeError("Did not receive two clicks")

        (c1x, c1y), (c2x, c2y) = clicks

        x1 = round(float(c1x), 1)
        y1 = round(float(c2y), 1)
        x2 = round(float(c2x), 1)
        y2 = round(float(c1y), 1)

        # Collect CLI inputs while the image window remains open
        num_series = prompt_positive_int("Enter number of series (integer): ")

        data = {
            "plot_bounds": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "num_series": num_series,
            "vision_extracted": True,
            "manually_entered": True,
            "needs_manual_review": False,
            "collection_mode": int(mode),
        }

        if mode == 2:
            axis_x_label = input("axis_x_label (string): ")
            axis_x_units = input("axis_x_units (string, press enter to leave blank): ")
            axis_x_origin = prompt_float("axis_x_origin (float, press enter for default 0-1 scale): ", allow_blank=True)
            if axis_x_origin == "":
                axis_x_origin, axis_x_final = 0.0, 1.0
            else:
                axis_x_final = prompt_float("axis_x_final (float): ")
            axis_y_label = input("axis_y_label (string): ")
            axis_y_units = input("axis_y_units (string, press enter to leave blank): ")
            axis_y_origin = prompt_float("axis_y_origin (float, press enter for default 0-1 scale): ", allow_blank=True)
            if axis_y_origin == "":
                axis_y_origin, axis_y_final = 0.0, 1.0
            else:
                axis_y_final = prompt_float("axis_y_final (float): ")

            data.update({
                "axis_x_label": axis_x_label,
                "axis_x_units": axis_x_units if axis_x_units != "" else "",
                "axis_x_origin": axis_x_origin,
                "axis_x_final": axis_x_final,
                "axis_x_min": axis_x_origin,
                "axis_x_max": axis_x_final,
                "axis_y_label": axis_y_label,
                "axis_y_units": axis_y_units if axis_y_units != "" else "",
                "axis_y_origin": axis_y_origin,
                "axis_y_final": axis_y_final,
                "axis_y_min": axis_y_origin,
                "axis_y_max": axis_y_final,
            })

        # Close window only after CLI prompts are completed
        plt.close(fig)

        # JSON merge logic: preserve protected fields from existing JSON and then
        # merge new data, ensuring we do not overwrite image measurement fields.
        protected = {k: existing[k] for k in ("image_width_px", "image_height_px", "image_measured", "source_png") if k in existing}
        merged = {}
        merged.update(existing)  # start with everything we had
        merged.update(data)      # then write collected fields (this won't touch protected ones if present)
        merged.update(protected) # re-apply protected to ensure they are not overwritten

        json_path.write_text(json.dumps(merged, indent=2))
        print(f"Saved: {json_path.relative_to(json_path.parents[2]).as_posix() if json_path.exists() else json_path}")
        return True, None

    except Exception as e:
        # If user closed the window manually or an error occurred, mark as aborted
        plt.close('all')
        aborted = existing.copy()
        aborted["vision_extracted"] = False
        aborted["manually_aborted"] = True
        try:
            json_path.write_text(json.dumps(aborted, indent=2))
        except Exception:
            pass
        print(f"Warning: user aborted or error for {png_path.name}: {e}")
        return False, e


def main():
    parser = argparse.ArgumentParser(description="Manual metadata collection for graph images")
    parser.add_argument("root", nargs="?", help="root folder path", default=None)
    parser.add_argument("--review", action="store_true", help="Process all graphs even if already vision_extracted")
    args = parser.parse_args()

    base = Path(args.root) if args.root else (Path(__file__).resolve().parent.parent / "output_pdf")
    base = base.resolve()

    mode = prompt_mode()

    total_graphs_folders = 0
    total_skipped = 0
    total_collected_mode1 = 0
    total_collected_mode2 = 0
    total_failed = 0

    # Walk for graphs/ subfolders only
    for graphs_dir in base.rglob('graphs'):
        if not graphs_dir.is_dir():
            continue
        total_graphs_folders += 1

        # Only process PNGs directly in this graphs/ folder
        for png in sorted(graphs_dir.glob('*.png')):
            json_path = png.with_suffix('.json')
            existing = {}
            if json_path.exists():
                try:
                    existing = json.loads(json_path.read_text())
                except Exception:
                    existing = {}

            if (not args.review) and existing.get('vision_extracted') is True:
                print(f"Skipping {png.name} — already processed. Use --review to override.")
                total_skipped += 1
                continue

            ok, err = process_png(png, json_path, mode)
            if ok:
                if mode == 1:
                    total_collected_mode1 += 1
                else:
                    total_collected_mode2 += 1
            else:
                total_failed += 1

    # Summary
    print("\nSummary:")
    print(f"- Total graphs/ folders found: {total_graphs_folders}")
    print(f"- Total PNGs skipped (already processed): {total_skipped}")
    print(f"- Total PNGs collected (mode 1): {total_collected_mode1}")
    print(f"- Total PNGs collected (mode 2): {total_collected_mode2}")
    print(f"- Total PNGs failed or aborted: {total_failed}")


if __name__ == '__main__':
    main()
