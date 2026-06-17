#!/usr/bin/env python3
"""Batch-run the full graph digitization pipeline end-to-end.

Stages (in order): PDF->PNG, graph extraction, graph curation, cleanup,
preprocessing, size measurement, axis/metadata extraction, vertical marker
detection, digitization, vertical marker series extraction, true-axis
scaling, and an optional overlay verification pass.

Default mode is fully autonomous (AI-driven curation + axis extraction via
2.1.1_ai_png_mngmt.py and 3.1_ai_bound_axis_call.py). Pass --human to use the
interactive stages instead (2.1_cli_png_mngmt.py and
3.1ALT_human_metadata_input.py).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
GRAPH_DIG = REPO_ROOT / "Graph_digitization"
INPUT_PDF_DIR = REPO_ROOT / "Input_pdf"
MODEL_FILE = REPO_ROOT / "doclayout_yolo_docstructbench_imgsz1024.pt"
ENV_FILE = REPO_ROOT / "info.env"
RUN_LOG_DIR = REPO_ROOT / "pipeline_runs"

REQUIRED_PACKAGES = [
    "pypdf", "pdf2image", "cv2", "doclayout_yolo", "numpy", "PIL",
    "dotenv", "matplotlib",
]
AI_PACKAGES = ["openai"]


class StageError(RuntimeError):
    pass


def check_preflight(human_mode: bool) -> None:
    problems = []

    missing_pkgs = []
    for pkg in REQUIRED_PACKAGES + ([] if human_mode else AI_PACKAGES):
        try:
            __import__(pkg)
        except ImportError:
            missing_pkgs.append(pkg)
    if missing_pkgs:
        problems.append(f"Missing Python packages: {', '.join(missing_pkgs)}")

    if not MODEL_FILE.exists():
        problems.append(
            f"Missing DocLayout-YOLO model file: {MODEL_FILE.name} "
            "(required by stage 2, graph extraction)"
        )

    if shutil.which("pdftoppm") is None:
        problems.append(
            "Poppler ('pdftoppm') not found on PATH "
            "(required by pdf2image for stage 1, PDF rendering)"
        )

    if not human_mode:
        if not ENV_FILE.exists():
            problems.append(
                f"Missing {ENV_FILE.name} at repo root "
                "(must define AI_KEY for autonomous mode)"
            )
        elif "AI_KEY" not in ENV_FILE.read_text():
            problems.append(f"AI_KEY not found in {ENV_FILE.name} (required for autonomous mode)")

    if problems:
        print("Pre-flight check failed:")
        for p in problems:
            print(f"  - {p}")
        raise StageError("Fix the issues above before running the pipeline.")

    print("Pre-flight checks passed.")


def discover_pdfs(requested: list[str]) -> list[Path]:
    if requested:
        resolved = []
        for item in requested:
            p = Path(item)
            if not p.is_absolute():
                candidate = INPUT_PDF_DIR / item
                p = candidate if candidate.exists() else (REPO_ROOT / item)
            if not p.exists():
                raise StageError(f"PDF not found: {item}")
            resolved.append(p)
        return resolved

    if not INPUT_PDF_DIR.exists():
        raise StageError(f"Input_pdf/ directory not found at {INPUT_PDF_DIR}")
    pdfs = sorted(INPUT_PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise StageError(f"No PDFs found in {INPUT_PDF_DIR}")
    return pdfs


def run_stage(name: str, cmd: list[str], log_path: Path, continue_on_error: bool) -> bool:
    print(f"\n=== {name} ===")
    print(f"$ {' '.join(cmd)}")
    start = time.time()
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        process.wait()
    elapsed = time.time() - start
    ok = process.returncode == 0
    status = "OK" if ok else f"FAILED (exit {process.returncode})"
    print(f"--- {name}: {status} in {elapsed:.1f}s (log: {log_path.relative_to(REPO_ROOT)}) ---")
    if not ok and not continue_on_error:
        raise StageError(f"Stage '{name}' failed (exit {process.returncode}). See {log_path}")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Batch-run the full graph digitization pipeline (PDF -> digitized, scaled CSVs)."
    )
    parser.add_argument(
        "pdfs", nargs="*",
        help="PDF filename(s) under Input_pdf/, or full paths. Default: every PDF in Input_pdf/.",
    )
    parser.add_argument(
        "--human", action="store_true",
        help="Use interactive human-input stages (manual graph curation + manual axis/metadata "
             "entry) instead of the default fully autonomous AI stages.",
    )
    parser.add_argument(
        "--root", default=None,
        help="Override the output_pdf root directory (default: output_pdf at repo root).",
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip the final overlay verification stage (Graph_digi_compare/run_compare.py).",
    )
    parser.add_argument(
        "--skip-pdf-split", action="store_true",
        help="Skip stage 1 (PDF -> PNG); use when output_pdf is already populated.",
    )
    parser.add_argument(
        "--ai-dry-run", action="store_true",
        help="(Autonomous mode only) Classify graph crops without moving/deleting files.",
    )
    parser.add_argument(
        "--ai-limit", type=int, default=None,
        help="(Autonomous mode only) Stop AI graph curation after classifying N images.",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Keep running remaining stages even if one stage fails (default: stop at first failure).",
    )
    args = parser.parse_args()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    log_dir = RUN_LOG_DIR / f"run_{run_id}"
    log_dir.mkdir(parents=True, exist_ok=True)

    mode_label = "HUMAN" if args.human else "AUTONOMOUS (AI)"
    print(f"Graph digitization batch pipeline — mode: {mode_label}")
    print(f"Logs: {log_dir.relative_to(REPO_ROOT)}")

    try:
        check_preflight(args.human)

        root_arg = [args.root] if args.root else []

        if not args.skip_pdf_split:
            pdfs = discover_pdfs(args.pdfs)
            print(f"\nFound {len(pdfs)} PDF(s) to process: {', '.join(p.name for p in pdfs)}")
            for i, pdf in enumerate(pdfs, 1):
                run_stage(
                    f"1. PDF -> PNG ({pdf.name})",
                    [sys.executable, str(GRAPH_DIG / "1_pdf_to_png.py"), str(pdf)],
                    log_dir / f"01_pdf_to_png_{i}.log",
                    args.continue_on_error,
                )
        else:
            print("\nSkipping stage 1 (--skip-pdf-split).")

        run_stage(
            "2. Graph extraction (DocLayout-YOLO)",
            [sys.executable, str(GRAPH_DIG / "2_png_graph_extraction.py")],
            log_dir / "02_graph_extraction.log",
            args.continue_on_error,
        )

        if args.human:
            print(
                "\nStage 2.1 is interactive: review each crop and choose "
                "keep/move/delete (k/m/d) in the terminal."
            )
            curation_cmd = [sys.executable, str(GRAPH_DIG / "2.1_cli_png_mngmt.py"), *root_arg]
        else:
            curation_cmd = [sys.executable, str(GRAPH_DIG / "2.1.1_ai_png_mngmt.py"), *root_arg]
            if args.ai_dry_run:
                curation_cmd.append("--dry-run")
            if args.ai_limit is not None:
                curation_cmd += ["--limit", str(args.ai_limit)]
        run_stage("2.1 Graph curation", curation_cmd, log_dir / "02_1_curation.log", args.continue_on_error)

        run_stage(
            "2.2 Clean empty folders",
            [sys.executable, str(GRAPH_DIG / "2.2_clean_empty_png.py")],
            log_dir / "02_2_clean_empty.log",
            args.continue_on_error,
        )

        run_stage(
            "2.3 PNG preprocessing",
            [sys.executable, str(GRAPH_DIG / "2.3_png_pre.py"), *root_arg],
            log_dir / "02_3_png_pre.log",
            args.continue_on_error,
        )

        run_stage(
            "3. PNG size measurement",
            [sys.executable, str(GRAPH_DIG / "3_png_size_prep.py"), *root_arg],
            log_dir / "03_png_size_prep.log",
            args.continue_on_error,
        )

        if args.human:
            print(
                "\nStage 3.1 is interactive: a mode prompt will appear in the terminal. "
                "Select mode 2 (Full) so axis_x/y_origin/final are captured — mode 1 "
                "(Minimal) will cause stage 3.4 (true-axis scaling) to skip every series."
            )
            metadata_cmd = [sys.executable, str(GRAPH_DIG / "3.1ALT_human_metadata_input.py"), *root_arg]
        else:
            metadata_cmd = [sys.executable, str(GRAPH_DIG / "3.1_ai_bound_axis_call.py"), *root_arg]
        run_stage("3.1 Axis/metadata extraction", metadata_cmd, log_dir / "03_1_metadata.log", args.continue_on_error)

        run_stage(
            "3.2.1 Vertical marker detection",
            [sys.executable, str(GRAPH_DIG / "3.2.1_vert_line_detection.py"), *root_arg],
            log_dir / "03_2_1_vert_detection.log",
            args.continue_on_error,
        )

        run_stage(
            "3.3 Graph digitization",
            [sys.executable, str(GRAPH_DIG / "3.3_graph_reading.py"), *root_arg],
            log_dir / "03_3_graph_reading.log",
            args.continue_on_error,
        )

        run_stage(
            "3.2.2 Vertical marker series extraction",
            [sys.executable, str(GRAPH_DIG / "3.2.2_vert_grapher.py"), *root_arg],
            log_dir / "03_2_2_vert_grapher.log",
            args.continue_on_error,
        )

        run_stage(
            "3.4 True-axis scaling",
            [sys.executable, str(GRAPH_DIG / "3.4_relative_scaling.py"), *root_arg],
            log_dir / "03_4_relative_scaling.log",
            args.continue_on_error,
        )

        if not args.skip_verify:
            run_stage(
                "Verification overlay",
                [sys.executable, str(REPO_ROOT / "Graph_digi_compare" / "run_compare.py"), *root_arg],
                log_dir / "04_verify.log",
                args.continue_on_error,
            )
        else:
            print("\nSkipping verification overlay (--skip-verify).")

    except StageError as e:
        print(f"\nPipeline aborted: {e}")
        sys.exit(1)

    print(f"\nPipeline finished. Logs saved under {log_dir.relative_to(REPO_ROOT)}.")


if __name__ == "__main__":
    main()
