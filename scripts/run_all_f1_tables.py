#!/usr/bin/env python3
"""
Run all paper tables related to F1 and kappa.

This script regenerates:
- Summary CSVs + val-opt F1 tables: output/f1/val_opt/
- Soft-label + val-opt kappa tables: output/f1/kappa/{soft_labels,val_opt}/
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all F1 and kappa table generators.")
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip scripts/comprehensive_f1_analysis.py (useful if dependencies like sklearn are unavailable).",
    )
    args = parser.parse_args()

    # Ensure paths work no matter where this is invoked from.
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    # Produces output/summary/*.csv used by downstream table generators.
    if not args.skip_analysis:
        if importlib.util.find_spec("sklearn") is None:
            print(
                "\nSkipping: scripts/comprehensive_f1_analysis.py (missing dependency: sklearn). "
                "Re-run with an environment that has scikit-learn installed, or pass --skip-analysis."
            )
        else:
            run([sys.executable, "scripts/comprehensive_f1_analysis.py"])

    # Val-opt F1 LaTeX tables.
    run([sys.executable, "scripts/generate_valopt_f1_tables.py"])

    # Soft-label + val-opt kappa LaTeX tables (includes soft_labels/main.tex).
    run([sys.executable, "scripts/generate_kappa_tables.py"])

    # Soft-label F1: post-process category tables + build main.tex even if analysis is skipped.
    run(
        [
            sys.executable,
            "-c",
            "import scripts.comprehensive_f1_analysis as m; "
            "m.postprocess_soft_category_tables('output/f1/soft'); "
            "m.write_main_tex_soft('output/f1/soft')",
        ]
    )


if __name__ == "__main__":
    main()
