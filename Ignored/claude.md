# AI Digitizer Function Reference Guide

**Quick navigation guide for all functions across the project.**

---

## File: 1_pdf_to_png.py
**Purpose**: Splits multi-page PDFs into individual page PNGs

### Functions:

| Function | Purpose |
|----------|---------|
| `resolve_input_pdf(input_pdf_filename: str \| None)` | Resolves input PDF path from CLI argument or default file location. Checks absolute, relative, and script directories. |
| `prepare_output_folders(source_pdf: Path)` | Creates output directory structure for processed PDFs. Returns output_root and paper_folder paths. |
| `copy_original_pdf(source_pdf: Path, paper_folder: Path)` | Copies the original source PDF into the output paper folder for archival. |
| `render_pdf_pages_to_png(pdf_path: Path, output_folder: Path, dpi: int)` | Converts PDF pages to PNG images at specified DPI resolution using pdf2image. |

---

## File: 2_png_graph_extraction.py
**Purpose**: Extracts graphs from page PNGs using DocLayout-YOLO model

### Functions:

| Function | Purpose |
|----------|---------|
| `load_model()` | Loads the local DocLayout-YOLO pre-trained model from disk. Raises FileNotFoundError if model file missing. |
| `extract_page_number(folder_name: str)` | Extracts numeric page number from folder names like 'page_1' or 'page_2'. Returns int or None. |
| `find_page_png(page_folder: Path)` | Searches for image files (.png, .jpg, .jpeg, .tiff) in a page folder. Returns Path or None. |
| `process_output_pdf_nested()` | Main traversal function. Walks output_pdf structure, detects graphs per page, crops and saves to graphs/ subfolders. |

---

## File: 2.1_cli_png_mngmt.py
**Purpose**: Interactive CLI to review, sort, and categorize extracted graph crops

### Functions:

| Function | Purpose |
|----------|---------|
| `extract_page_number(folder_name: str)` | Parses page number from folder names using regex. Returns int or None. |
| `find_parent_page_image(page_folder: Path)` | Locates the original page PNG in the parent folder for side-by-side display. |
| `display_side_by_side(crop_path: Path, parent_path: Path \| None)` | Opens non-blocking Matplotlib figure showing graph crop and parent page side-by-side. Returns figure object for programmatic closing. |
| `prompt_action()` | Prompts user for action choice (Keep/Move/Delete) until valid input provided. Returns single character. |

---

## File: 2.2_clean_empty_png.py
**Purpose**: Cleanup utility to remove empty graphs/ and tables/ directories

### Functions:

| Function | Purpose |
|----------|---------|
| `cleanup_empty_folders(root_dir_name: str)` | Walks output_pdf tree bottom-up and removes any empty 'graphs' or 'tables' folders. Prints summary statistics. |

---

## File: 2.3_png_pre.py
**Purpose**: Preprocesses graph PNGs (deskew, denoise, isolation) before digitization

### Functions:

| Function | Purpose |
|----------|---------|
| `isolate_graph_structural_lines(gray_img: np.ndarray)` | Removes text and diagonal data lines using morphological operations. Returns binary image with only grid/axis lines. |
| `parse_args()` | Parses command-line arguments for preprocessing pipeline (root_dir, single file, skip-backup, rotation-only, etc.). |
| `extract_page_number(folder_name: str)` | Extracts numeric page ID from folder name string. |

---

## File: 3_png_size_prep.py
**Purpose**: Measures PNG dimensions and creates JSON metadata files

### Functions:

| Function | Purpose |
|----------|---------|
| `parse_args()` | Parses CLI arguments to specify root output_pdf directory (default: ./output_pdf). |
| `process_graphs_folder(graphs_path, root_dir, counters)` | Iterates through PNG files in a graphs/ folder, reads dimensions, creates/updates JSON metadata with image_width_px and image_height_px. |
| `main()` | Entry point. Walks output_pdf tree, finds graphs/ folders, processes each PNG, prints summary counters. |

---

## File: 3.1_ai_bound_axis_call.py
**Purpose**: Calls OpenAI Vision API to extract axis metadata from graph images

### Functions:

| Function | Purpose |
|----------|---------|
| `parse_args()` | Parses arguments: root_dir (default: output_pdf) or --single PNG_PATH for single-file processing. |
| `load_api_client()` | Loads OpenAI API client using AI_KEY from info.env environment variables. Raises RuntimeError if key missing. |
| `read_json(path)` | Safely reads JSON file. Returns dict or None on error. |
| `write_json(path, data)` | Writes data dict to JSON file with 2-space indentation and trailing newline. |
| `is_graphs_png(path)` | Checks if path is a valid PNG file inside a 'graphs' directory. Returns boolean. |
| `build_user_prompt(retry_message: str \| None)` | Constructs user prompt for OpenAI. Optionally appends retry message for resubmission. |
| `extract_json_blob(text: str)` | Extracts JSON object from model response text by finding outer braces. Raises ValueError if not found. |
| `send_vision_request(client, png_path, prompt_text)` | Encodes PNG to base64, sends to GPT-4o with system prompt, parses JSON response. Returns dict. |
| `validate_plot_bounds(bounds, image_width, image_height)` | Verifies plot_bounds are valid pixel coordinates within image. Raises ValueError on invalid bounds. |
| `validate_vision_payload(payload, image_width, image_height)` | Validates all required fields returned by OpenAI (axis labels, units, bounds, series count). |

---

## File: 3.1ALT_human_metadata_input.py
**Purpose**: Alternative manual input for axis metadata when AI extraction fails

---

## File: 3.2.1_vert_line_detection.py
**Purpose**: Detects vertical shift lines (solid or dashed) in graph crops

### Functions:

| Function | Purpose |
|----------|---------|
| `parse_args()` | Parses root folder argument (default: output_pdf) for line detection. |
| `safe_load_json(json_path: Path)` | Safely reads JSON metadata file. Returns dict or None on error with error logging. |
| `safe_write_json(json_path: Path, data: dict)` | Safely writes JSON metadata with 2-space indentation. Logs errors. |
| `validate_plot_bounds(data: dict)` | Checks if plot_bounds dict exists and contains x1, y1, x2, y2. Returns bounds dict or None. |
| `_validate_vertical_distribution(col_binary: np.ndarray)` | Checks if column binary data represents a true vertical line (solid or dashed). Tests span, zone coverage. Returns boolean. |
| `detect_vertical_shift_line(gray_crop: np.ndarray)` | Analyzes column profiles to find vertical shift line x-coordinate. Uses peak detection, contrast, width validation. Returns int or None. |

---

## File: 3.2.2_vert_grapher.py
**Purpose**: (Empty file - placeholder for vertical graph visualization or processing)

---

## File: 3.3_graph_reading.py
**Purpose**: Digitizes graphs using intensity clustering and skeleton pixel tracing

### Functions:

| Function | Purpose |
|----------|---------|
| `parse_args()` | Parses CLI: root folder or --single PNG_PATH for targeted processing. |
| `is_graphs_png(png_path: Path)` | Validates that path is PNG in a graphs/ folder. Returns boolean. |
| `find_graphs_folders(root_dir: Path)` | Generator that yields all graphs/ folder paths inside output_pdf structure. |
| `resolve_single_path(single_arg: str, root_dir: Path)` | Resolves single file argument to absolute path. Handles relative and absolute inputs. |
| `safe_load_json(json_path: Path)` | Safely reads JSON metadata. Returns dict or None on error. |
| `clamp_plot_bounds(bounds: dict, image_width: int, image_height: int)` | Constrains plot boundary coordinates to valid image dimensions. Returns clamped tuple (x1, y1, x2, y2). |
| `crop_plot(image: np.ndarray, bounds: dict, image_width: int, image_height: int)` | Extracts plot region from image using bounds. Returns cropped array and boundary coordinates. |
| `detect_vertical_markers(gray_crop: np.ndarray, manual_markers: list \| None)` | Identifies vertical shift line positions. Uses manual JSON coordinates if provided, falls back to morphology. Returns positions, mask, inpainted image. |
| `extract_dark_pixels(gray_image: np.ndarray)` | Finds all dark pixels (data traces) in binary-inverted image. Returns xs, ys, line_mask. |

---

## Pipeline Overview

```
1_pdf_to_png.py
    ↓ (Creates page PNGs)
2_png_graph_extraction.py
    ↓ (Crops individual graphs)
2.1_cli_png_mngmt.py
    ↓ (Manual review & sort)
2.2_clean_empty_png.py
    ↓ (Cleanup)
2.3_png_pre.py
    ↓ (Preprocess graphs)
3_png_size_prep.py
    ↓ (Measure & create metadata)
3.1_ai_bound_axis_call.py
    ↓ (Extract axis info via AI)
3.1ALT_human_metadata_input.py
    ↓ (Manual fallback)
3.2.1_vert_line_detection.py
    ↓ (Detect shift markers)
3.3_graph_reading.py
    ↓ (Digitize data)
Output: JSON with digitized graph data
```

---

## Common Patterns

### Error Handling
- Most functions use `try/except` with silent fallback (returning None)
- JSON operations protected by `safe_load_json()` / `safe_write_json()`
- Path validation checks existence before reading

### Path Resolution
- Most scripts accept optional `root_dir` argument (default: `output_pdf`)
- Single-file processing available via `--single` flag for debugging
- Relative paths resolved from script directory first, then CWD

### JSON Metadata Structure
```json
{
  "source_png": "path/to/page_X/graphs/pageX.Y.png",
  "image_width_px": 800,
  "image_height_px": 600,
  "image_measured": true,
  "axis_x_label": "Wavelength (nm)",
  "axis_y_label": "Intensity",
  "axis_x_units": "nm",
  "axis_y_units": "",
  "axis_x_min": 400,
  "axis_x_max": 800,
  "axis_y_min": 0,
  "axis_y_max": 100,
  "plot_bounds": {"x1": 50, "y1": 40, "x2": 750, "y2": 550},
  "num_series": 1,
  "vertical_markers": [150, 350, 550]
}
```

