"""Split a multi-page PDF into individual page PDFs and PNGs.

Default input file:
    Input_pdf/Bryant - 1964 - Vibrational Spectrum of Sodium Azide Single Crystals 2.pdf

Output structure:
    output_pdf/
    └── Bryant - 1964 - Vibrational Spectrum of Sodium Azide Single Crystals 2/
        ├── Bryant - 1964 - Vibrational Spectrum of Sodium Azide Single Crystals 2.pdf
        ├── page_1/
        │   └── page_1_Bryant - 1964 - Vibrational Spectrum of Sodium Azide Single Crystals 2.png
        └── page_2/
            └── page_2_Bryant - 1964 - Vibrational Spectrum of Sodium Azide Single Crystals 2.png

Dependencies:
- `pypdf` for PDF reading and writing
- `pdf2image` for PDF page rasterization
- `poppler` system package for `pdf2image`

Install dependencies with pip:
    python -m pip install pypdf pdf2image

Install Poppler:
- macOS: brew install poppler
- Ubuntu/Debian: sudo apt-get install poppler-utils
- Windows: install Poppler and add its `bin` folder to PATH, or set POPPLER_PATH.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from pypdf import PdfReader
from pdf2image import convert_from_path

DEFAULT_INPUT_RELATIVE = Path(
    "Input_pdf/Bryant - 1964 - Vibrational Spectrum of Sodium Azide Single Crystals 2.pdf"
)
DEFAULT_OUTPUT_ROOT = Path("output_pdf")


def resolve_input_pdf(input_pdf_filename: str | None = None) -> Path:
    """Resolve the input PDF path from CLI or default file location."""
    script_dir = Path(__file__).resolve().parent

    if input_pdf_filename:
        input_path = Path(input_pdf_filename)
        if input_path.is_absolute():
            candidate = input_path
        else:
            candidate = (Path.cwd() / input_path).resolve()

        if candidate.exists():
            return candidate

        fallback = (script_dir / input_path).resolve()
        if fallback.exists():
            return fallback

        raise FileNotFoundError(
            f"Source PDF not found: {input_pdf_filename}. Checked: {candidate} and {fallback}."
        )

    default_path = (script_dir / DEFAULT_INPUT_RELATIVE).resolve()
    if default_path.exists():
        return default_path

    cwd_path = (Path.cwd() / DEFAULT_INPUT_RELATIVE).resolve()
    if cwd_path.exists():
        return cwd_path

    raise FileNotFoundError(
        f"Source PDF not found: {DEFAULT_INPUT_RELATIVE}. Checked script directory: {default_path} and current working directory: {cwd_path}."
    )


def prepare_output_folders(source_pdf: Path) -> tuple[Path, Path]:
    """Create output root and paper folder for all generated files."""
    script_dir = Path(__file__).resolve().parent
    output_root = (script_dir / DEFAULT_OUTPUT_ROOT).resolve()
    paper_folder = output_root / source_pdf.stem
    paper_folder.mkdir(parents=True, exist_ok=True)
    return output_root, paper_folder


def copy_original_pdf(source_pdf: Path, paper_folder: Path) -> Path:
    """Copy the original source PDF into the paper output folder."""
    target_path = paper_folder / source_pdf.name
    if not target_path.exists():
        shutil.copy2(source_pdf, target_path)
        print(f"Copied original PDF to: {target_path.relative_to(paper_folder.parent)}")
    else:
        print(f"Original PDF already exists: {target_path.relative_to(paper_folder.parent)}")
    return target_path


def render_pdf_pages_to_png(pdf_path: Path, output_folder: Path, dpi: int = 300) -> None:
    """Render each PDF page to a PNG inside individual sub-folders."""
    pdf_path = pdf_path.resolve()
    base_name = pdf_path.stem

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    poppler_path = os.environ.get("POPPLER_PATH")
    images = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        poppler_path=poppler_path,
    )

    for page_number, image in enumerate(images, start=1):
        page_folder = output_folder / f"page_{page_number}"
        page_folder.mkdir(parents=True, exist_ok=True)

        output_png_name = f"page_{page_number}_{base_name}.png"
        output_png_path = page_folder / output_png_name
        image.save(output_png_path, format="PNG")

        print(f"Created PNG: {output_png_path.relative_to(output_folder)} ({page_number}/{total_pages})")


def process_pdf(input_pdf_filename: str | None = None) -> None:
    """Main entrypoint: resolve input, copy original, and generate page outputs."""
    source_pdf = resolve_input_pdf(input_pdf_filename)
    output_root, paper_folder = prepare_output_folders(source_pdf)

    copy_original_pdf(source_pdf, paper_folder)
    render_pdf_pages_to_png(source_pdf, paper_folder)

    print(f"\nFinished processing {source_pdf.name}.")
    print(f"Output root: {output_root}")
    print(f"Paper folder: {paper_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a PDF into single-page PDFs and PNGs organized by individual folders."
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        help=(
            "Path to the source PDF file. Defaults to the Bryant PDF inside Input_pdf/ "
            "or the same file in the current working directory."
        ),
    )
    args = parser.parse_args()
    process_pdf(args.pdf_path)