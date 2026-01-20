#!/usr/bin/env python3
"""
Annotator agreement breakdown for raw annotation scores (0..3).

For each source, counts (across all categories, all items):
  - full_positive: 3/3 positive  -> score == 3
  - two_of_three_positive: 2/3   -> score == 2
  - two_of_three_negative: 1/3   -> score == 1
  - full_negative: 0/3           -> score == 0

Outputs a LaTeX table (.tex) suitable for pasting into the paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


SOURCES: List[Tuple[str, str, str]] = [
    ("reddit", "annotation/reddit_raw_scores.csv", "Deidentified_Comment"),
    ("x", "annotation/x_raw_scores.csv", "Deidentified text"),
    ("news", "annotation/news_raw_scores.csv", "Deidentified_paragraph"),
    ("meeting_minutes", "annotation/meeting_minutes_raw_scores.csv", "Deidentified_paragraph"),
]

# Gold-standard sampled files (used to filter raw-score rows down to the gold set)
GOLD_STANDARD: Dict[str, Tuple[str, str]] = {
    "reddit": ("gold_standard/sampled_reddit_comments.csv", "Comment"),
    "x": ("gold_standard/sampled_twitter_posts.csv", "Deidentified_text"),
    "news": ("gold_standard/sampled_lexisnexis_news.csv", "Deidentified_paragraph_text"),
    "meeting_minutes": ("gold_standard/sampled_meeting_minutes.csv", "Deidentified_paragraph"),
}

CATEGORIES: List[str] = [
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

# Column name variants present in some files
COL_ALIASES: Dict[str, List[str]] = {
    "ask a rhetorical question": ["ask a rhetorical question", "ask a rheorical question"],
    "racist": ["racist", "Racist"],
}


def _find_col(df: pd.DataFrame, category: str) -> str | None:
    if category in df.columns:
        return category
    for alt in COL_ALIASES.get(category, []):
        if alt in df.columns:
            return alt
    return None


def _normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip().lower()


def load_gold_text_set(source: str) -> set[str]:
    """
    Load the gold-standard sampled texts for `source`, normalized for matching.
    """
    path_col = GOLD_STANDARD.get(source)
    if not path_col:
        return set()
    path_str, col = path_col
    path = Path(path_str)
    if not path.exists():
        return set()
    df = pd.read_csv(path, low_memory=False)
    if col not in df.columns:
        return set()
    return set(df[col].apply(_normalize_text).tolist())


def summarize_source_overall(source: str, csv_path: Path, raw_text_col: str) -> tuple[int, Dict[str, int]]:
    df = pd.read_csv(csv_path, low_memory=False)
    gold_texts = load_gold_text_set(source)
    if gold_texts and raw_text_col in df.columns:
        df = df[df[raw_text_col].apply(_normalize_text).isin(gold_texts)].copy()

    cols = []
    for cat in CATEGORIES:
        col = _find_col(df, cat)
        if col is not None:
            cols.append(col)

    if not cols:
        return int(len(df)), {
            "full_positive": 0,
            "two_of_three_positive": 0,
            "two_of_three_negative": 0,
            "full_negative": 0,
            "total": 0,
        }

    mat = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    vals = mat.ravel()
    vals = vals[~np.isnan(vals)]
    vals_i = vals.astype(int)

    full_positive = int(np.sum(vals_i == 3))
    two_pos = int(np.sum(vals_i == 2))
    two_neg = int(np.sum(vals_i == 1))
    full_negative = int(np.sum(vals_i == 0))
    total = int(len(vals_i))

    return int(len(df)), {
        "full_positive": full_positive,
        "two_of_three_positive": two_pos,
        "two_of_three_negative": two_neg,
        "full_negative": full_negative,
        "total": total,
    }


def summarize_source_per_category(
    source: str,
    csv_path: Path,
    raw_text_col: str,
) -> tuple[int, Dict[str, Dict[str, int]]]:
    """
    Returns:
      n_items, {category: {full_positive, two_of_three_positive, two_of_three_negative, full_negative, total}}
    over the GOLD STANDARD sampled subset only.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    gold_texts = load_gold_text_set(source)
    if gold_texts and raw_text_col in df.columns:
        df = df[df[raw_text_col].apply(_normalize_text).isin(gold_texts)].copy()

    out: Dict[str, Dict[str, int]] = {}
    for cat in CATEGORIES:
        col = _find_col(df, cat)
        if col is None:
            out[cat] = {"full_positive": 0, "two_of_three_positive": 0, "two_of_three_negative": 0, "full_negative": 0, "total": 0}
            continue
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)].astype(int)
        out[cat] = {
            "full_positive": int(np.sum(vals == 3)),
            "two_of_three_positive": int(np.sum(vals == 2)),
            "two_of_three_negative": int(np.sum(vals == 1)),
            "full_negative": int(np.sum(vals == 0)),
            "total": int(len(vals)),
        }

    return int(len(df)), out


def pct(n: int, d: int) -> str:
    if d <= 0:
        return "0.0%"
    return f"{(100.0 * n / d):.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize annotator agreement (0..3) into a LaTeX table.")
    parser.add_argument(
        "--out",
        default="output/annotation/annotator_agreement_breakdown.tex",
        help="Output .tex path (default: output/annotation/annotator_agreement_breakdown.tex)",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("% Auto-generated by scripts/annotator_agreement_breakdown.py")
    lines.append("% Gold-standard only (matched by normalized text).")
    lines.append("")

    for source, rel_csv, raw_text_col in SOURCES:
        csv_path = Path(rel_csv)
        if not csv_path.exists():
            continue

        n_items, per_cat = summarize_source_per_category(source, csv_path, raw_text_col)

        # totals across categories for this source
        totals = {"full_positive": 0, "two_of_three_positive": 0, "two_of_three_negative": 0, "full_negative": 0, "total": 0}
        for cat in CATEGORIES:
            for k in totals:
                totals[k] += int(per_cat[cat].get(k, 0))

        lines.append(r"\begin{table*}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\begin{tabular}{lrrrrr}")
        lines.append(r"\toprule")
        lines.append(rf"Category & 3/3 Pos & 2/3 Pos & 2/3 Neg & 3/3 Neg & Total \\")
        lines.append(r"\midrule")

        for cat in CATEGORIES:
            c = per_cat[cat]
            total = int(c["total"])
            lines.append(
                f"{cat} "
                f"& {c['full_positive']} ({pct(int(c['full_positive']), total)}) "
                f"& {c['two_of_three_positive']} ({pct(int(c['two_of_three_positive']), total)}) "
                f"& {c['two_of_three_negative']} ({pct(int(c['two_of_three_negative']), total)}) "
                f"& {c['full_negative']} ({pct(int(c['full_negative']), total)}) "
                f"& {total} \\\\"
            )

        lines.append(r"\midrule")
        lines.append(
            f"TOTAL "
            f"& {totals['full_positive']} ({pct(int(totals['full_positive']), int(totals['total']))}) "
            f"& {totals['two_of_three_positive']} ({pct(int(totals['two_of_three_positive']), int(totals['total']))}) "
            f"& {totals['two_of_three_negative']} ({pct(int(totals['two_of_three_negative']), int(totals['total']))}) "
            f"& {totals['full_negative']} ({pct(int(totals['full_negative']), int(totals['total']))}) "
            f"& {totals['total']} \\\\"
        )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(
            rf"\caption{{Annotator agreement breakdown by category for {source} (gold-standard set; $n={n_items}$ items). Raw scores: 0--3.}}"
        )
        lines.append(rf"\label{{tab:annotator_agreement_breakdown_{source}}}")
        lines.append(r"\end{table*}")
        lines.append("")

    out_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()

