#!/usr/bin/env python3
"""
GPT vs human gold bias audit (systematic prevalence + per-label P/R/F1).

Joins human soft labels with GPT-4.1 gold-subset flag files on normalized text,
reports per-label prevalence (human vs GPT), precision/recall/F1 treating human
(soft >= threshold) as reference, and flags unstable rare positives.

Usage (from repo root):
    python scripts/gpt_human_bias_audit.py
    python scripts/gpt_human_bias_audit.py --source reddit x
    python scripts/gpt_human_bias_audit.py --soft-threshold 0.5 --rare-positive-max 5

Outputs:
    output/annotation/gpt4_bias_audit/summary_by_source.csv
    output/annotation/gpt4_bias_audit/per_label_metrics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCES = ["reddit", "x", "news", "meeting_minutes"]

RAW_FILES = {
    "reddit": "annotation/reddit_raw_scores.csv",
    "x": "annotation/x_raw_scores.csv",
    "news": "annotation/news_raw_scores.csv",
    "meeting_minutes": "annotation/meeting_minutes_raw_scores.csv",
}

SOFT_FILES = {
    s: f"output/annotation/soft_labels/{s}_soft_labels.csv" for s in SOURCES
}

GPT_FLAG_FILES = {
    "reddit": "output/reddit/gpt4/classified_comments_reddit_gold_subset_gpt4_reddit_flags.csv",
    "x": "output/x/gpt4/classified_comments_x_gold_subset_gpt4_x_flags.csv",
    "news": "output/news/gpt4/classified_comments_news_gold_subset_gpt4_news_flags.csv",
    "meeting_minutes": (
        "output/meeting_minutes/gpt4/"
        "classified_comments_meeting_minutes_gold_subset_gpt4_meeting_minutes_flags.csv"
    ),
}

# Human-facing label order (matches soft label CSV columns)
LABEL_COLUMNS = [
    "ask a genuine question",
    "ask a rhetorical question",
    "provide a fact or claim",
    "provide an observation",
    "express their opinion",
    "express others opinions",
    "money aid allocation",
    "government critique",
    "societal critique",
    "solutions/interventions",
    "personal interaction",
    "media portrayal",
    "not in my backyard",
    "harmful generalization",
    "deserving/undeserving",
    "racist",
]

# GPT flag CSV columns -> soft label names
GPT_TO_SOFT = {
    "Comment_ask a genuine question": "ask a genuine question",
    "Comment_ask a rhetorical question": "ask a rhetorical question",
    "Comment_provide a fact or claim": "provide a fact or claim",
    "Comment_provide an observation": "provide an observation",
    "Comment_express their opinion": "express their opinion",
    "Comment_express others opinions": "express others opinions",
    "Critique_money aid allocation": "money aid allocation",
    "Critique_government critique": "government critique",
    "Critique_societal critique": "societal critique",
    "Response_solutions/interventions": "solutions/interventions",
    "Perception_personal interaction": "personal interaction",
    "Perception_media portrayal": "media portrayal",
    "Perception_not in my backyard": "not in my backyard",
    "Perception_harmful generalization": "harmful generalization",
    "Perception_deserving/undeserving": "deserving/undeserving",
    "Racist_Flag": "racist",
}


def _text_column(df: pd.DataFrame) -> str:
    for c in (
        "Deidentified_Comment",
        "Deidentified text",
        "Deidentified_paragraph",
        "Comment",
    ):
        if c in df.columns:
            return c
    raise ValueError(f"No text column found. Columns: {df.columns.tolist()}")


def _load_human_frame(source: str) -> tuple[pd.DataFrame, int]:
    """Return dataframe with Comment + soft label columns; n_raw before dedupe."""
    raw_path = REPO_ROOT / RAW_FILES[source]
    soft_path = REPO_ROOT / SOFT_FILES[source]
    raw = pd.read_csv(raw_path, low_memory=False)
    soft = pd.read_csv(soft_path, low_memory=False)
    if len(raw) != len(soft):
        raise ValueError(
            f"{source}: raw_scores ({len(raw)}) and soft_labels ({len(soft)}) length mismatch"
        )
    tc = _text_column(raw)
    out = pd.DataFrame(
        {"Comment": raw[tc].astype(str).str.strip()}
    )
    for col in LABEL_COLUMNS:
        if col not in soft.columns:
            raise ValueError(f"{source}: soft labels missing column {col!r}")
        out[col] = pd.to_numeric(soft[col], errors="coerce").fillna(0.0)
    n_raw = len(out)
    out = out.drop_duplicates(subset=["Comment"], keep="first")
    return out, n_raw


def _load_gpt_flags(source: str) -> pd.DataFrame:
    path = REPO_ROOT / GPT_FLAG_FILES[source]
    if not path.exists():
        raise FileNotFoundError(path)
    gpt = pd.read_csv(path, low_memory=False)
    if "Comment" not in gpt.columns:
        raise ValueError(f"{path} missing Comment column")
    gpt = gpt.copy()
    gpt["Comment"] = gpt["Comment"].astype(str).str.strip()
    gpt = gpt.drop_duplicates(subset=["Comment"], keep="first")
    for gcol, scol in GPT_TO_SOFT.items():
        if gcol not in gpt.columns:
            raise ValueError(f"{path} missing GPT column {gcol!r}")
        gpt[scol] = pd.to_numeric(gpt[gcol], errors="coerce").fillna(0).astype(int).clip(0, 1)
    return gpt[["Comment"] + LABEL_COLUMNS]


def _binary_prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Precision, recall, F1 for positive class (no sklearn)."""
    yt = y_true.astype(int)
    yp = y_pred.astype(int)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _micro_prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Micro-averaged P/R/F1 over flattened binary decisions."""
    return _binary_prf(y_true, y_pred)


def run_audit(
    sources: list[str],
    soft_threshold: float,
    rare_positive_max: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    per_label_rows = []

    for source in sources:
        human, n_raw = _load_human_frame(source)
        gpt = _load_gpt_flags(source)
        gpt_ren = gpt.rename(columns={c: f"{c}_gpt" for c in LABEL_COLUMNS})
        merged = human.merge(gpt_ren, on="Comment", how="inner")

        n_dup_raw = n_raw - len(human)
        row_summary = {
            "source": source,
            "n_raw_human_rows": n_raw,
            "n_duplicate_text_dropped": n_dup_raw,
            "n_human_unique": len(human),
            "n_gpt_unique": len(gpt),
            "n_merged": len(merged),
        }

        if len(merged) == 0:
            summary_rows.append(row_summary)
            continue

        all_true = []
        all_pred = []
        f1_for_macro = []

        for lab in LABEL_COLUMNS:
            y_true = (pd.to_numeric(merged[lab], errors="coerce").fillna(0) >= soft_threshold).astype(int).values
            y_pred = (
                pd.to_numeric(merged[f"{lab}_gpt"], errors="coerce").fillna(0).astype(int).clip(0, 1).values
            )
            p, r, f1 = _binary_prf(y_true, y_pred)
            f1_for_macro.append(f1)
            prev_h = float(y_true.mean())
            prev_g = float(y_pred.mean())
            pos_h = int(y_true.sum())
            pos_g = int(y_pred.sum())
            unstable = pos_h <= rare_positive_max

            per_label_rows.append(
                {
                    "source": source,
                    "label": lab,
                    "n_merged": len(merged),
                    "human_positive_count": pos_h,
                    "gpt_positive_count": pos_g,
                    "prevalence_human": round(prev_h, 6),
                    "prevalence_gpt": round(prev_g, 6),
                    "delta_prevalence_gpt_minus_human": round(prev_g - prev_h, 6),
                    "precision_gpt_vs_human": round(p, 6),
                    "recall_gpt_vs_human": round(r, 6),
                    "f1_gpt_vs_human": round(f1, 6),
                    "low_human_positive_support": unstable,
                }
            )
            all_true.extend(y_true.tolist())
            all_pred.extend(y_pred.tolist())

        micro_p, micro_r, micro_f1 = _micro_prf(
            np.array(all_true, dtype=int),
            np.array(all_pred, dtype=int),
        )
        dprev = [abs(r["delta_prevalence_gpt_minus_human"]) for r in per_label_rows if r["source"] == source]
        row_summary.update(
            {
                "micro_precision": round(float(micro_p), 6),
                "micro_recall": round(float(micro_r), 6),
                "micro_f1": round(float(micro_f1), 6),
                "macro_f1_mean_per_label": round(float(np.mean(f1_for_macro)), 6),
                "mean_abs_delta_prevalence": round(float(np.mean(dprev)), 6),
            }
        )
        summary_rows.append(row_summary)

    summary_df = pd.DataFrame(summary_rows)
    per_label_df = pd.DataFrame(per_label_rows)
    if not per_label_df.empty:
        per_label_df["human_positive_rare_cap"] = rare_positive_max
    summary_path = out_dir / "summary_by_source.csv"
    per_label_path = out_dir / "per_label_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    per_label_df.to_csv(per_label_path, index=False)
    print(f"Wrote {summary_path}")
    print(f"Wrote {per_label_path}")
    print(summary_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        nargs="*",
        default=SOURCES,
        choices=SOURCES,
        help="Subset of sources (default: all)",
    )
    parser.add_argument(
        "--soft-threshold",
        type=float,
        default=0.5,
        help="Binarize human soft labels at this threshold (default 0.5)",
    )
    parser.add_argument(
        "--rare-positive-max",
        type=int,
        default=5,
        help="Flag labels with human positive count <= this (default 5)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "output/annotation/gpt4_bias_audit",
        help="Output directory for CSVs",
    )
    args = parser.parse_args()

    run_audit(
        sources=list(args.source),
        soft_threshold=args.soft_threshold,
        rare_positive_max=args.rare_positive_max,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
