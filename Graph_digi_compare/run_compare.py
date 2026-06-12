#!/usr/bin/env python3
"""One-shot launcher for the Graph Digitizer comparison pipeline.

This is the convenience entry point for the whole ``Graph_digi_compare`` folder:
it wires up the import path and the working directory so you can run the full
overlay batch from anywhere, without remembering the ``python -m`` invocation.

It delegates to ``verify.cli`` (discovery -> mapping -> overlay -> storage), so
every option that CLI accepts works here too and is simply passed through::

    python run_compare.py                      # overlay every graph under output_pdf
    python run_compare.py --filter page6       # only graphs whose key contains "page6"
    python run_compare.py --scaled             # read digitized_scaled/*.csv instead
    python run_compare.py --alpha 0.4 --dpi 200
    python run_compare.py /path/to/other/output_pdf

Each run writes a timestamped batch of overlay PNGs plus a manifest to
``Graph_digi_compare/runs/run_YYYYMMDD_HHMMSS/``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# This file lives in <repo>/Graph_digi_compare/. The pipeline's default paths
# ("output_pdf", "Graph_digi_compare/runs") are relative to the repo root, so
# anchor the process there regardless of where the script was launched from.
THIS_DIR = Path(__file__).resolve().parent          # .../Graph_digi_compare
REPO_ROOT = THIS_DIR.parent                          # .../AI_Digitizer_.1

# Make ``import verify.*`` resolve when run as a plain script.
sys.path.insert(0, str(THIS_DIR))


def main() -> int:
    # Anchor at the repo root so the pipeline's relative default paths
    # ("output_pdf", "Graph_digi_compare/runs") resolve no matter the cwd.
    os.chdir(REPO_ROOT)

    # Delegate to the CLI, which validates the root and reports a clear error
    # ("Root directory not found") if output_pdf (or a passed-in root) is absent.
    from verify.cli import main as cli_main

    return cli_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
