#!/usr/bin/env python3

import argparse
import base64
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image


SYSTEM_PROMPT = (
    "You are a graph axis extraction assistant. "
    "You will receive exactly one graph image and must respond with a single JSON object only. "
    "Do not include any markdown, commentary, or extra fields. "
    "Return exactly these fields: axis_x_label, axis_y_label, axis_x_units, axis_y_units, "
    "axis_x_min, axis_x_max, axis_y_min, axis_y_max, plot_bounds, num_series. "
    "Use empty strings for units when none are present. "
    "The plot_bounds object must contain x1, y1, x2, y2 with top-left origin pixel coordinates."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run OpenAI vision extraction for graph axes inside graphs/ folders"
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        default=None,
        help="Path to the root output_pdf/ directory (default: output_pdf at the repo root)",
    )
    parser.add_argument(
        "--single",
        metavar="PNG_PATH",
        help="Process one specific graph PNG path instead of walking the full tree",
    )
    return parser.parse_args()


def load_api_client():
    load_dotenv("info.env")
    api_key = os.getenv("AI_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenAI API key not found. Set AI_KEY in the environment or info.env."
        )
    return OpenAI(api_key=api_key)


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def is_graphs_png(path):
    return (
        os.path.isfile(path)
        and path.lower().endswith(".png")
        and os.path.basename(os.path.dirname(path)).lower() == "graphs"
    )


def build_user_prompt(retry_message=None):
    prompt = (
        "Analyze the attached graph image and return exactly one JSON object with the required fields. "
        "Do not add explanations or formatting."
    )
    if retry_message:
        prompt += " " + retry_message
    return prompt


def extract_json_blob(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model response")
    return text[start : end + 1]


def send_vision_request(client, png_path, prompt_text):
    # Base64 encode the local file so ChatCompletions can read it
    with open(png_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                ],
            },
        ],
    )
    
    response_text = response.choices[0].message.content
    json_blob = extract_json_blob(response_text)
    return json.loads(json_blob)


def validate_plot_bounds(bounds, image_width, image_height):
    if not isinstance(bounds, dict):
        raise ValueError("plot_bounds must be a JSON object")

    x1 = float(bounds.get("x1"))
    y1 = float(bounds.get("y1"))
    x2 = float(bounds.get("x2"))
    y2 = float(bounds.get("y2"))

    # Ensure bounds are inside the image and that the coordinates define a valid rectangle.
    if not (0 <= x1 < x2 <= image_width):
        raise ValueError(
            f"plot_bounds x values invalid: x1={x1}, x2={x2}, width={image_width}"
        )
    if not (0 <= y1 < y2 <= image_height):
        raise ValueError(
            f"plot_bounds y values invalid: y1={y1}, y2={y2}, height={image_height}"
        )
    return True


def validate_vision_payload(payload, image_width, image_height):
    required_fields = [
        "axis_x_label",
        "axis_y_label",
        "axis_x_units",
        "axis_y_units",
        "axis_x_min",
        "axis_x_max",
        "axis_y_min",
        "axis_y_max",
        "plot_bounds",
        "num_series",
    ]
    for field in required_fields:
        if field not in payload:
            raise ValueError(f"Missing required field: {field}")

    plot_bounds = payload["plot_bounds"]
    validate_plot_bounds(plot_bounds, image_width, image_height)

    # Convert numeric axis values and series count for consistency.
    payload["axis_x_min"] = float(payload["axis_x_min"])
    payload["axis_x_max"] = float(payload["axis_x_max"])
    payload["axis_y_min"] = float(payload["axis_y_min"])
    payload["axis_y_max"] = float(payload["axis_y_max"])
    payload["num_series"] = int(payload["num_series"])

    return payload


def get_retry_message(payload, width, height):
    if payload is None:
        return (
            "The previous request failed to return valid data. "
            f"Please analyze the image and return valid JSON with plot_bounds inside [0, {width}] x [0, {height}]."
        )
    bounds = payload.get("plot_bounds")
    if not bounds or not isinstance(bounds, dict):
        return (
            "The previous response did not provide valid plot_bounds. "
            f"Please return plot_bounds inside [0, {width}] x [0, {height}] and ensure x1 < x2 and y1 < y2."
        )
    return (
        "The previous response contained invalid plot_bounds: "
        f"{bounds}. Please correct them to fit inside [0, {width}] x [0, {height}] with x1 < x2 and y1 < y2."
    )


def process_png_file(png_path, client, summary):
    json_path = os.path.splitext(png_path)[0] + ".json"
    metadata = read_json(json_path)
    filename = os.path.basename(png_path)

    if metadata is None or metadata.get("image_measured") is not True:
        print(f"Skipping {filename} — run step0_measure_images.py first.")
        summary["skipped_not_measured"] += 1
        return

    if metadata.get("vision_extracted") is True:
        print(f"Already processed: {filename}")
        summary["skipped_already_processed"] += 1
        return

    image_width = int(metadata.get("image_width_px", 0))
    image_height = int(metadata.get("image_height_px", 0))
    if image_width <= 0 or image_height <= 0:
        print(f"Skipping {filename} — invalid image dimensions in JSON.")
        summary["skipped_not_measured"] += 1
        return

    try:
        with Image.open(png_path) as img:
            img.verify()
    except Exception:
        print(f"Skipping {filename} — invalid PNG image file.")
        summary["failed"] += 1
        return

    # Attempt the API call and validate plot_bounds. If invalid, retry up to 3 times.
    max_attempts = 3
    payload = None
    for attempt in range(1, max_attempts + 1):
        try:
            prompt_text = build_user_prompt(
                None if attempt == 1 else get_retry_message(payload, image_width, image_height)
            )
            payload = send_vision_request(client, png_path, prompt_text)
            validate_vision_payload(payload, image_width, image_height)

            # Preserve existing measurement fields and add vision extraction results.
            metadata.update(payload)
            metadata["vision_extracted"] = True
            metadata["needs_manual_review"] = False
            write_json(json_path, metadata)
            summary["processed"] += 1
            return
        except Exception as exc:
            if attempt == max_attempts:
                print(
                    f"Failed to process {filename} after {max_attempts} attempts: {exc}"
                )
                metadata["vision_extracted"] = False
                metadata["needs_manual_review"] = True
                write_json(json_path, metadata)
                summary["failed"] += 1
                return
            continue


def process_graphs_folder(graphs_path, client, summary):
    for entry in sorted(os.listdir(graphs_path)):
        entry_path = os.path.join(graphs_path, entry)
        if not os.path.isfile(entry_path):
            continue
        if not entry.lower().endswith(".png"):
            continue

        summary["total_pngs"] += 1
        process_png_file(entry_path, client, summary)


def find_graphs_folders(root_dir):
    for current_root, dirs, files in os.walk(root_dir):
        if os.path.basename(current_root).lower() == "graphs":
            # Skip processing within a graphs/ folder directly; we handle graphs/ from its parent.
            continue

        if "graphs" not in dirs:
            # No graphs/ subfolder in this branch, so skip the branch entirely.
            continue

        graphs_path = os.path.join(current_root, "graphs")
        dirs.remove("graphs")
        if os.path.isdir(graphs_path):
            yield graphs_path


def resolve_single_path(single_arg, root_dir):
    candidate = os.path.abspath(single_arg)
    if not os.path.exists(candidate):
        candidate = os.path.abspath(os.path.join(root_dir, single_arg))
    return candidate


def main():
    args = parse_args()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_dir = os.path.abspath(
        args.root_dir if args.root_dir else os.path.join(repo_root, "output_pdf")
    )
    if not os.path.isdir(root_dir):
        raise SystemExit(f"Root directory not found: {root_dir}")

    client = load_api_client()

    summary = {
        "total_pngs": 0,
        "skipped_not_measured": 0,
        "skipped_already_processed": 0,
        "processed": 0,
        "failed": 0,
    }

    if args.single:
        single_path = resolve_single_path(args.single, root_dir)
        if not is_graphs_png(single_path):
            raise SystemExit(
                f"Single file must be an existing PNG inside a graphs/ folder: {args.single}"
            )
        summary["total_pngs"] = 1
        process_png_file(single_path, client, summary)
    else:
        for graphs_path in find_graphs_folders(root_dir):
            process_graphs_folder(graphs_path, client, summary)

    print("Processing complete.")
    print(f"Total PNGs found in graphs/ folders: {summary['total_pngs']}")
    print(f"Skipped (not measured): {summary['skipped_not_measured']}")
    print(f"Skipped (already done): {summary['skipped_already_processed']}")
    print(f"Processed: {summary['processed']}")
    print(f"Failed: {summary['failed']}")


if __name__ == "__main__":
    main()