#!/usr/bin/env python3
"""
Run both soft-label and val-opt table generation.
"""

import subprocess
import sys


def run(cmd: list[str]) -> None:
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main() -> None:
    run([sys.executable, "scripts/comprehensive_f1_analysis.py"])
    run([sys.executable, "scripts/generate_valopt_f1_tables.py"])


if __name__ == "__main__":
    main()
