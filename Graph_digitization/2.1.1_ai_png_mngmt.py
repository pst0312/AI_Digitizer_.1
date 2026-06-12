"""AI-driven review of cropped graph images, mirroring 2.1_cli_png_mngmt.py.

Usage:
    python3 2.1_ai_png_mngmt.py [root] [--dry-run] [--limit N]

- root: optional path to `output_pdf` (default: ./output_pdf)
- --dry-run: classify but do not move/delete any files
- --limit N: stop after classifying N images (useful for testing)

Behavior:
- Traverses output_pdf/<paper>/<page_X>/graphs/*.png
- For each crop, asks the AI (via AI_KEY in info.env) whether it is a
  'graph', 'table', or should be 'delete'd.
- graph -> kept in place
- table -> moved to the sibling `tables/` folder
- delete -> file is removed

Requirements:
- Python with `openai`, `python-dotenv`, and `pillow` installed.
- AI_KEY set in info.env at the repo root.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from dotenv import load_dotenv
import os
from openai import OpenAI


VALID_EXTS = {".png", ".jpg", ".jpeg", ".tiff"}

SYSTEM_PROMPT = (
    "You are an image triage assistant reviewing cropped images automatically extracted "
    "from pages of a scientific paper. Each image is a candidate figure crop. "
    "Classify the image into exactly one of three categories: "
    "'graph' (a plotted chart/graph with axes and data series, suitable for digitization), "
    "'table' (a data table or other tabular layout), or "
    "'delete' (not useful - e.g. body text, an equation, a logo, a blank or noisy region, "
    "or any crop that is not a graph or table). "
    "Respond with a single JSON object only, no markdown or extra commentary, "
    "with exactly two fields: 'classification' (one of 'graph', 'table', 'delete') "
    "and 'reason' (a short explanation)."
)


def load_api_client() -> OpenAI:
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / "info.env")
    api_key = os.getenv("AI_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenAI API key not found. Set AI_KEY in the environment or info.env."
        )
    return OpenAI(api_key=api_key)


def extract_json_blob(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model response")
    return text[start : end + 1]


def classify_image(client: OpenAI, png_path: Path, max_attempts: int = 2) -> dict:
    """Ask the AI to classify a single image. Returns {'classification': ..., 'reason': ...}."""
    with open(png_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Classify the attached cropped image.",
                            },
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
            payload = json.loads(extract_json_blob(response_text))
            classification = payload.get("classification", "").strip().lower()
            if classification not in {"graph", "table", "delete"}:
                raise ValueError(f"Invalid classification: {classification!r}")
            return {
                "classification": classification,
                "reason": payload.get("reason", ""),
            }
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Classification failed after {max_attempts} attempts: {last_error}"
                )
    raise RuntimeError(f"Classification failed: {last_error}")


def perform_action(graph_path: Path, page_folder: Path, paper_folder: Path, classification: str) -> str:
    """Apply the classification result to the file. Returns a status string."""
    if classification == "graph":
        return f"Kept {graph_path.relative_to(paper_folder)} (graph)"

    if classification == "table":
        tables_folder = page_folder / "tables"
        tables_folder.mkdir(exist_ok=True)
        dest = tables_folder / graph_path.name
        graph_path.replace(dest)
        return f"Moved {graph_path.relative_to(paper_folder)} -> {dest.relative_to(paper_folder)} (table)"

    if classification == "delete":
        graph_path.unlink()
        return f"Deleted {graph_path.relative_to(paper_folder)} (not a graph or table)"

    return f"Unknown classification '{classification}' for {graph_path.relative_to(paper_folder)}"


def traverse_and_classify(root: Path, client: OpenAI, dry_run: bool, limit: int | None) -> None:
    if not root.exists() or not root.is_dir():
        print(f"Root folder not found: {root}")
        return

    paper_folders = sorted([p for p in root.iterdir() if p.is_dir()])
    if not paper_folders:
        print(f"No paper folders found in {root}")
        return

    summary = {"total": 0, "graph": 0, "table": 0, "delete": 0, "failed": 0}

    for paper in paper_folders:
        print(f"\nPaper: {paper.name}")
        page_folders = sorted([d for d in paper.iterdir() if d.is_dir() and d.name.startswith("page_")])
        for page in page_folders:
            graphs_folder = page / "graphs"
            if not graphs_folder.exists():
                continue
            graph_files = sorted([f for f in graphs_folder.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTS])
            if not graph_files:
                continue
            print(f" Reviewing {page.name} ({len(graph_files)} graph(s))")
            for graph in graph_files:
                if limit is not None and summary["total"] >= limit:
                    print("Reached limit. Stopping.")
                    print_summary(summary)
                    return

                summary["total"] += 1
                try:
                    result = classify_image(client, graph)
                except Exception as exc:
                    print(f"  Failed to classify {graph.name}: {exc}")
                    summary["failed"] += 1
                    continue

                classification = result["classification"]
                reason = result["reason"]
                summary[classification] += 1

                if dry_run:
                    print(f"  [dry-run] {graph.relative_to(paper)} -> {classification} ({reason})")
                    continue

                try:
                    status = perform_action(graph, page, paper, classification)
                    print(f"  {status} - {reason}")
                except Exception as exc:
                    print(f"  Failed to apply action to {graph.name}: {exc}")
                    summary["failed"] += 1

    print_summary(summary)


def print_summary(summary: dict) -> None:
    print("\nSummary:")
    print(f"  Total reviewed: {summary['total']}")
    print(f"  Graphs kept:    {summary['graph']}")
    print(f"  Tables moved:   {summary['table']}")
    print(f"  Deleted:        {summary['delete']}")
    print(f"  Failed:         {summary['failed']}")


def main():
    parser = argparse.ArgumentParser(description="AI-driven graph review for output_pdf structure")
    parser.add_argument(
        "root",
        nargs="?",
        default=None,
        help="Root output_pdf folder (default: output_pdf at the repo root)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify images without moving or deleting files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after classifying this many images",
    )
    args = parser.parse_args()

    root = (
        Path(args.root).resolve()
        if args.root
        else Path(__file__).resolve().parent.parent / "output_pdf"
    )

    client = load_api_client()
    traverse_and_classify(root, client, args.dry_run, args.limit)


if __name__ == "__main__":
    main()
