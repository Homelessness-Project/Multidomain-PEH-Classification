#!/usr/bin/env python3
"""Bootstrap 95% CIs for pooled prevalence gaps on the gold-standard set."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["reddit", "news", "meeting_minutes", "x"]
MODELS = ["llama", "phi4", "qwen", "gemini", "grok", "gpt4"]
COL_MAP = {
    "not in my backyard": "Perception_not in my backyard",
    "provide a fact or claim": "Comment_provide a fact or claim",
    "harmful generalization": "Perception_harmful generalization",
    "express their opinion": "Comment_express their opinion",
}


def pooled_gap(category: str) -> tuple[float, float, float, int, int]:
    col = COL_MAP[category]
    gold_parts, pred_parts = [], []
    for src in SOURCES:
        soft = pd.read_csv(REPO_ROOT / "output/annotation/soft_labels" / f"{src}_soft_labels.csv")
        gold = (soft[category].fillna(0) >= 0.5).astype(int).values
        preds = []
        for model in MODELS:
            for few in ["none", src]:
                path = (
                    REPO_ROOT
                    / "output"
                    / src
                    / model
                    / f"classified_comments_{src}_gold_subset_{model}_{few}_flags.csv"
                )
                if path.exists():
                    df = pd.read_csv(path, low_memory=False)
                    preds.append(pd.to_numeric(df[col], errors="coerce").fillna(0).values)
        n = min(len(gold), min(len(p) for p in preds))
        gold_parts.append(gold[:n])
        pred_parts.append(np.mean(np.stack([p[:n] for p in preds], axis=0), axis=0))
    g = np.concatenate(gold_parts)
    p = np.concatenate(pred_parts)
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(5000):
        idx = rng.integers(0, len(g), len(g))
        boot.append((p[idx].mean() - g[idx].mean()) * 100)
    boot = np.array(boot)
    return (p.mean() - g.mean()) * 100, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), len(g), int(g.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "output/f1/gold_error_analysis/prevalence_gap_bootstrap.csv")
    args = parser.parse_args()
    rows = []
    for cat in COL_MAP:
        gap, lo, hi, n, n_pos = pooled_gap(cat)
        rows.append({"category": cat, "gap_pp": gap, "ci_lo": lo, "ci_hi": hi, "n": n, "gold_positives": n_pos})
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
