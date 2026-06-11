#!/usr/bin/env python3

import argparse
import json
import math
import os
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

BORDER_ERODE_PX = 3                  
GAUSSIAN_KERNEL = (3, 3)               
GAUSSIAN_SIGMA = .8                   
ADAPTIVE_BLOCK_SIZE = 25                
ADAPTIVE_C = 4                      
MIN_COMPONENT_AREA = 12                 
MAX_SKEW_ANGLE_DEG = 15                 
SKEW_METHOD_DISAGREEMENT_THRESHOLD_DEG = 5 

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "output_pdf"

STAGE_NAMES = [
    "grayscale",
    "deskew",
    "border_erode",
    "gaussian_blur",
    "adaptive_threshold",
    "morph_open",
    "component_filter",
]

def isolate_graph_structural_lines(gray_img: np.ndarray) -> np.ndarray:
    """Removes text and diagonal data lines, leaving only grid lines and axes."""
    # Invert to binary (white lines on black background)
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Create horizontal and vertical structural elements
    scale = 20  # Higher means longer lines required
    k_width = max(2, gray_img.shape[1] // scale)
    k_height = max(2, gray_img.shape[0] // scale)
    
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_width, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_height))
    
    # Extract horizontal and vertical structures
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)
    
    # Combine them and invert back to black-on-white
    combined = cv2.add(horiz, vert)
    return cv2.bitwise_not(combined)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess graph PNGs in output_pdf/ before JSON measurement and digitization"
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Path to output_pdf/ directory (default: output_pdf next to script)",
    )
    parser.add_argument(
        "--single",
        metavar="PNG_PATH",
        help="Process one PNG only (for debugging)",
    )
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Do not save originals/ backup (use with caution)",
    )
    parser.add_argument(
        "--rotation-only",
        action="store_true",
        help="Only run rotation correction, skip all other steps",
    )
    parser.add_argument(
        "--no-rotation",
        action="store_true",
        help="Skip rotation correction, run all other steps",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate images to graphs/debug/ at each stage",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done, do not write any files",
    )
    return parser.parse_args()


def extract_page_number(folder_name: str) -> str:
    match = re.search(r"page[_]?(\d+)", folder_name, re.IGNORECASE)
    return match.group(1) if match else folder_name


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


def load_grayscale(png_path: Path) -> np.ndarray:
    image = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image: {png_path}")
    if image.ndim == 3:
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 2:
        return image
    raise RuntimeError(f"Unsupported image shape: {image.shape}")


def save_image(image: np.ndarray, output_path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def save_debug_image(image: np.ndarray, debug_dir: Path, stage_name: str, dry_run: bool) -> str:
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / f"{stage_name}.png"
    if not dry_run:
        cv2.imwrite(str(debug_path), image)
    return str(debug_path)


def detect_skew_angle(gray_img: np.ndarray) -> tuple[float, bool, list[str]]:
    warnings: list[str] = []
    angles: list[float] = []

    def clamp_angle(angle: float) -> float:
        return max(-MAX_SKEW_ANGLE_DEG, min(MAX_SKEW_ANGLE_DEG, angle))

    # --- 1. Isolate the axes to eliminate noisy data lines ---
    clean_axes_img = isolate_graph_structural_lines(gray_img)
    # Use a blurred version for smoother profiling
    gray_small = cv2.GaussianBlur(clean_axes_img, (3, 3), 0)

    def hough_line_angle(image: np.ndarray) -> float | None:
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=max(30, min(image.shape) // 10),
            maxLineGap=20,
        )
        if lines is None:
            return None
        angle_values = []
        for x1, y1, x2, y2 in lines[:, 0]:
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            if abs(angle) <= 45:
                angle_values.append(angle)
        if not angle_values:
            return None
        weighted_angles: list[float] = []
        weighted_lengths: list[float] = []
        for (x1, y1, x2, y2), angle in zip(lines[:, 0], angle_values):
            weighted_angles.append(angle)
            weighted_lengths.append(math.hypot(x2 - x1, y2 - y1))
        weighted_sum = sum(a * w for a, w in zip(weighted_angles, weighted_lengths))
        total_weight = sum(weighted_lengths)
        return clamp_angle(weighted_sum / total_weight) if total_weight else None

    def projection_profile_angle(image: np.ndarray) -> float | None:
        best_correction = 0.0
        best_variance = -1.0
        for correction in np.linspace(-MAX_SKEW_ANGLE_DEG, MAX_SKEW_ANGLE_DEG, 31):
            rotated = rotate_image(image, correction, border_value=255)
            profile = np.sum(255 - rotated.astype(np.float32), axis=1)
            variance = float(np.var(profile))
            if variance > best_variance:
                best_variance = variance
                best_correction = correction
        return clamp_angle(-best_correction)

    def bounding_rect_angle(image: np.ndarray) -> float | None:
        _, bin_inv = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(bin_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 10:
            return None
        rect = cv2.minAreaRect(largest)
        angle = rect[-1]
        if angle < -45:
            angle += 90
        return clamp_angle(-angle)

    try:
        hough_res = hough_line_angle(gray_small)
        if hough_res is not None: angles.append(clamp_angle(hough_res))
    except Exception: hough_res = None

    try:
        proj_res = projection_profile_angle(gray_small)
        if proj_res is not None: angles.append(clamp_angle(proj_res))
    except Exception: proj_res = None

    try:
        bound_res = bounding_rect_angle(gray_small)
        if bound_res is not None: angles.append(clamp_angle(bound_res))
    except Exception: bound_res = None

    if not angles:
        warnings.append("skew estimation unavailable")
        return 0.0, True, warnings

    median_angle = float(np.median(np.array(angles, dtype=np.float32)))
    disagreement = max(abs(angle - median_angle) for angle in angles)
    
    # --- 2. Smart Automatic Resolution ---
    if disagreement > SKEW_METHOD_DISAGREEMENT_THRESHOLD_DEG:
        # If projection profile found a strong match, trust it over the others 
        # because Hough gets distracted by text clusters.
        if proj_res is not None:
            warnings.append(f"Disagreement resolved automatically: trusting Projection Profile ({proj_res:.2f}°)")
            return clamp_angle(proj_res), False, warnings
        else:
            warnings.append("skew estimation disagreement exceeds threshold; fallback failed")
            return 0.0, True, warnings

    return clamp_angle(median_angle), False, warnings


def rotate_image(gray_img: np.ndarray, angle_deg: float, border_value: int = 255) -> np.ndarray:
    height, width = gray_img.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        gray_img,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def correct_rotation(gray_img: np.ndarray, angle_deg: float) -> np.ndarray:
    return rotate_image(gray_img, -angle_deg, border_value=255)


def erode_border(img: np.ndarray, px: int = BORDER_ERODE_PX) -> np.ndarray:
    if img.shape[0] <= px * 2 or img.shape[1] <= px * 2:
        return img
    return img[px:-px, px:-px]


def adaptive_threshold(img: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C,
    )


def morph_open(img: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def filter_components(binary: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    clean = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA:
            clean[labels == i] = 255
    return clean


def build_backup_path(png_path: Path, graph_index: int) -> Path:
    originals_dir = png_path.parent / "originals"
    page_folder = png_path.parent.parent
    page_number = extract_page_number(page_folder.name)
    backup_name = f"page{page_number}.{graph_index}_original.png"
    return originals_dir / backup_name


def build_pre_json_path(png_path: Path) -> Path:
    return png_path.with_name(f"{png_path.stem}_pre.json")


def write_preprocess_json(
    json_path: Path,
    png_path: Path,
    root_dir: Path,
    skew_angle: float,
    skew_corrected: bool,
    needs_rotation_review: bool,
    stages_applied: list[str],
    warnings: list[str],
    debug_images: list[str],
    dry_run: bool,
) -> None:
    metadata = {
        "source_png": str(png_path.relative_to(root_dir)).replace("\\", "/"),
        "preprocessed": True,
        "skew_angle_detected_deg": float(round(skew_angle, 4)),
        "skew_corrected": skew_corrected,
        "needs_rotation_review": needs_rotation_review,
        "border_eroded_px": BORDER_ERODE_PX,
        "adaptive_block_size": ADAPTIVE_BLOCK_SIZE,
        "adaptive_C": ADAPTIVE_C,
        "min_component_area": MIN_COMPONENT_AREA,
        "stages_applied": stages_applied,
        "warnings": warnings,
    }
    if debug_images:
        metadata["debug_images"] = debug_images
    if dry_run:
        return
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def process_graph_png(
    png_path: Path,
    root_dir: Path,
    args,
    counters: dict[str, int],
    graph_index: int,
) -> None:
    pre_json_path = build_pre_json_path(png_path)
    if pre_json_path.exists():
        try:
            existing = json.loads(pre_json_path.read_text(encoding="utf-8"))
            if existing.get("preprocessed") is True:
                counters["already_preprocessed"] += 1
                return
        except Exception:
            pass

    if not args.skip_backup:
        backup_path = build_backup_path(png_path, graph_index)
        if backup_path.exists():
            counters["backup_skipped"] += 1
        else:
            if args.dry_run:
                print(f"Would back up {png_path} to {backup_path}")
            else:
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(png_path), str(backup_path))
                counters["backups_created"] += 1

    try:
        gray = load_grayscale(png_path)
    except Exception as exc:
        print(f"Failed to read {png_path}: {exc}")
        counters["failed"] += 1
        return

    debug_images: list[str] = []
    if args.debug:
        debug_dir = png_path.parent / "debug"
        debug_images.append(str(Path("debug") / f"stage1_grayscale.png"))
        save_debug_image(gray, debug_dir, "stage1_grayscale", args.dry_run)

    stages_applied = ["grayscale"]
    skew_angle = 0.0
    skew_corrected = False
    needs_rotation_review = False
    warnings: list[str] = []

    if args.no_rotation:
        warnings.append("rotation correction skipped by --no-rotation")
        rotated = gray.copy()
    else:
        skew_angle, review_flag, skew_warnings = detect_skew_angle(gray)
        warnings.extend(skew_warnings)
        if review_flag:
            needs_rotation_review = True
            skew_corrected = False
            rotated = gray.copy()
        else:
            rotated = correct_rotation(gray, skew_angle)
            skew_corrected = bool(abs(skew_angle) > 1e-6)
            if rotated.shape[0] > int(round(gray.shape[0] * 1.1)) or rotated.shape[1] > int(round(gray.shape[1] * 1.1)):
                warnings.append("rotation correction produced unexpected image dimensions")
                rotated = gray.copy()
                needs_rotation_review = True
                skew_corrected = False
        stages_applied.append("deskew")
        if args.debug:
            debug_dir = png_path.parent / "debug"
            debug_images.append(str(Path("debug") / f"stage2_deskewed.png"))
            save_debug_image(rotated, debug_dir, "stage2_deskewed", args.dry_run)

    final_image = rotated

    if not args.rotation_only:
        eroded = erode_border(final_image, BORDER_ERODE_PX)
        if eroded.shape != final_image.shape:
            stages_applied.append("border_erode")
        else:
            warnings.append("border erosion skipped because the image is too small")
        if args.debug:
            debug_dir = png_path.parent / "debug"
            debug_images.append(str(Path("debug") / f"stage3_border_eroded.png"))
            save_debug_image(eroded, debug_dir, "3_border_eroded", args.dry_run)

        blurred = cv2.GaussianBlur(eroded, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)
        stages_applied.append("gaussian_blur")
        if args.debug:
            debug_dir = png_path.parent / "debug"
            debug_images.append(str(Path("debug") / f"stage4_blurred.png"))
            save_debug_image(blurred, debug_dir, "4_blurred", args.dry_run)

        thresholded = adaptive_threshold(blurred)
        stages_applied.append("adaptive_threshold")
        if args.debug:
            debug_dir = png_path.parent / "debug"
            debug_images.append(str(Path("debug") / f"stage5_adaptive_threshold.png"))
            save_debug_image(thresholded, debug_dir, "5_adaptive_threshold", args.dry_run)

        opened = morph_open(thresholded)
        stages_applied.append("morph_open")
        if args.debug:
            debug_dir = png_path.parent / "debug"
            debug_images.append(str(Path("debug") / f"stage6_morph_open.png"))
            save_debug_image(opened, debug_dir, "6_morph_open", args.dry_run)

        cleaned = filter_components(opened)
        stages_applied.append("component_filter")
        if args.debug:
            debug_dir = png_path.parent / "debug"
            debug_images.append(str(Path("debug") / f"stage7_component_filtered.png"))
            save_debug_image(cleaned, debug_dir, "7_component_filtered", args.dry_run)

        final_image = cv2.bitwise_not(cleaned)
    else:
        final_image = cv2.bitwise_not(final_image)

    if args.debug:
        debug_dir = png_path.parent / "debug"
        debug_images.append(str(Path("debug") / f"stage8_final_output.png"))
        save_debug_image(final_image, debug_dir, "8_final_output", args.dry_run)

    if args.dry_run:
        print(f"Would write processed image to {png_path}")
        print(f"Would write sidecar JSON to {pre_json_path}")
        counters["processed"] += 1
        return

    save_image(final_image, png_path, args.dry_run)
    write_preprocess_json(
        pre_json_path,
        png_path,
        root_dir,
        skew_angle,
        skew_corrected,
        needs_rotation_review,
        stages_applied,
        warnings,
        debug_images,
        args.dry_run,
    )
    counters["processed"] += 1
    if needs_rotation_review:
        counters["rotation_review"] += 1


def process_graphs_folder(graphs_path: Path, root_dir: Path, args, counters: dict[str, int]) -> None:
    png_paths = sorted(graphs_path.glob("*.png"))
    if not png_paths:
        counters["skipped_no_graphs"] += 1
        return

    for graph_index, png_path in enumerate(png_paths, start=1):
        counters["total_pngs"] += 1
        process_graph_png(png_path, root_dir, args, counters, graph_index)


def main() -> None:
    args = parse_args()
    if args.rotation_only and args.no_rotation:
        raise SystemExit("--rotation-only and --no-rotation cannot be used together")
    root_dir = Path(args.root_dir).resolve()
    if not root_dir.is_dir():
        raise SystemExit(f"Root directory not found: {root_dir}")

    counters = {
        "total_pngs": 0,
        "skipped_no_graphs": 0,
        "already_preprocessed": 0,
        "processed": 0,
        "rotation_review": 0,
        "failed": 0,
        "backups_created": 0,
        "backup_skipped": 0,
    }

    if args.single:
        png_path = resolve_single_path(args.single, root_dir)
        if not png_path.is_file() or png_path.suffix.lower() != ".png" or png_path.parent.name.lower() != "graphs":
            raise SystemExit(
                f"Single file must be an existing PNG inside a graphs/ folder: {args.single}"
            )
        counters["total_pngs"] = 1
        process_graph_png(png_path, root_dir, args, counters, graph_index=1)
    else:
        for graphs_path in find_graphs_folders(root_dir):
            process_graphs_folder(graphs_path, root_dir, args, counters)

    print("Processing complete.")
    print(f"Total PNGs found:        {counters['total_pngs']}")
    print(f"Skipped (no graphs/):    {counters['skipped_no_graphs']}")
    print(f"Already preprocessed:    {counters['already_preprocessed']}")
    print(f"Processed successfully:  {counters['processed']}")
    print(f"Flagged for rotation review: {counters['rotation_review']}")
    print(f"Failed:                  {counters['failed']}")


if __name__ == "__main__":
    main()
