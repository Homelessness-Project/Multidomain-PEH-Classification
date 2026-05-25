#!/usr/bin/env python3
"""
Qualitative error analysis: why do LLMs over-tag NIMBY on the gold-standard set?

Compares gold-negative / model-positive (false positive) posts to:
  - true negatives (gold-negative, no model tags NIMBY)
  - true positives (gold-positive NIMBY)

Reports lexical trigger rates, simple structural cues, and exports example snippets.

Usage (repo root):
    .venv/bin/python scripts/nimby_qualitative_error_analysis.py
    .venv/bin/python scripts/nimby_qualitative_error_analysis.py --min-model-fp 3

Outputs (default output/f1/gold_error_analysis/nimby_qualitative/):
    trigger_rates_by_group.csv
    trigger_odds_ratios.csv
    nimby_fp_examples.csv
    fact_claim_fn_examples.csv
    summary.txt
    snippets_for_paper.tex
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCES = ["reddit", "news", "meeting_minutes", "x"]
PROMPT_MODELS = ["llama", "phi4", "qwen", "gemini", "grok", "gpt4"]
NIMBY = "not in my backyard"
FACT = "provide a fact or claim"
COL_NIMBY = "Perception_not in my backyard"
COL_FACT = "Comment_provide a fact or claim"

# Lexical / structural triggers hypothesized to drive NIMBY over-tagging.
TRIGGER_PATTERNS: Dict[str, str] = {
    "affordable_housing": r"\baffordable housing\b",
    "housing_general": r"\b(housing|shelter|homeless|unhoused|encampment)\b",
    "proximity_local": (
        r"\b(neighborhood|neighbourhood|nearby|next door|down the street|"
        r"in my (area|backyard|neighborhood)|our (area|block|street)|local)\b"
    ),
    "explicit_nimby": r"\b(nimby|not in my backyard|backyard)\b",
    "opposition_verbs": (
        r"\b(oppose|opposed|against|don't want|do not want|dont want|"
        r"fight|block|prevent|stop|reject|refuse)\b"
    ),
    "rhetorical_cue": r"\b(did you read|how can you|what about|why would)\b",
    "question_mark": r"\?",
    "policy_numbers": r"(\d+\s*%|\b\d+\s*(million|units|percent)\b)",
    "development": r"\b(development|zoning|build|building|units)\b",
    "shelter_siting": r"\b(shelter|siting|site the|located)\b",
}


def _few_shot_suffix(source: str, shot_type: str = "zero_shot") -> str:
    return "none" if shot_type == "zero_shot" else source


def _prediction_path(source: str, model: str, shot_type: str = "zero_shot") -> Path:
    few = _few_shot_suffix(source, shot_type)
    return (
        REPO_ROOT
        / "output"
        / source
        / model
        / f"classified_comments_{source}_gold_subset_{model}_{few}_flags.csv"
    )


def _load_soft(source: str) -> pd.DataFrame:
    path = REPO_ROOT / "output" / "annotation" / "soft_labels" / f"{source}_soft_labels.csv"
    return pd.read_csv(path)


def _load_preds(source: str, model: str) -> Optional[pd.DataFrame]:
    path = _prediction_path(source, model)
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    if COL_NIMBY not in df.columns:
        return None
    out = pd.DataFrame(
        {
            "text": df["Comment"].astype(str),
            COL_NIMBY: pd.to_numeric(df[COL_NIMBY], errors="coerce").fillna(0).astype(int),
            COL_FACT: (
                pd.to_numeric(df.get(COL_FACT, 0), errors="coerce").fillna(0).astype(int)
                if COL_FACT in df.columns
                else 0
            ),
        }
    )
    out["source"] = source
    out["model"] = model
    return out


def _gold_bin(soft: pd.Series, threshold: float) -> int:
    return int(float(soft) >= threshold)


def _trigger_matrix(texts: pd.Series) -> pd.DataFrame:
    rows = []
    for text in texts:
        t = str(text).lower()
        rows.append(
            {
                k: bool(re.search(p, t, flags=re.IGNORECASE))
                for k, p in TRIGGER_PATTERNS.items()
            }
        )
    return pd.DataFrame(rows)


def _build_unified_table(threshold: float = 0.5) -> pd.DataFrame:
    """One row per gold-subset item (first model's text), with gold + vote counts."""
    chunks: List[pd.DataFrame] = []
    for source in SOURCES:
        soft = _load_soft(source)
        preds = [p for m in PROMPT_MODELS if (p := _load_preds(source, m)) is not None]
        if not preds:
            continue
        n = min(len(soft), *(len(p) for p in preds))
        base = preds[0].iloc[:n].copy()
        base["gold_nimby"] = (soft[NIMBY].fillna(0).iloc[:n].values >= threshold).astype(int)
        base["gold_fact"] = (soft[FACT].fillna(0).iloc[:n].values >= threshold).astype(int)
        nimby_votes = np.zeros(n, dtype=int)
        fact_misses = np.zeros(n, dtype=int)
        for pred in preds:
            nimby_votes += pred[COL_NIMBY].iloc[:n].values
            fact_misses += (1 - pred[COL_FACT].iloc[:n].values) * base["gold_fact"].values
        if base is None:
            continue
        base["nimby_model_votes"] = nimby_votes
        base["fact_fn_votes"] = fact_misses.astype(int)
        base["any_nimby_fp"] = ((base["gold_nimby"] == 0) & (base["nimby_model_votes"] > 0)).astype(int)
        base["consensus_nimby_fp"] = (
            (base["gold_nimby"] == 0) & (base["nimby_model_votes"] >= 3)
        ).astype(int)
        chunks.append(base)
    return pd.concat(chunks, ignore_index=True)


def _rate_table(df: pd.DataFrame, mask: pd.Series, label: str) -> pd.DataFrame:
    trig = _trigger_matrix(df.loc[mask, "text"])
    rates = trig.mean().rename(label)
    out = pd.DataFrame({label: rates})
    out["n"] = int(mask.sum())
    return out


def _odds_ratio(fp_rate: float, tn_rate: float, eps: float = 1e-4) -> float:
    fp_rate = min(max(fp_rate, eps), 1 - eps)
    tn_rate = min(max(tn_rate, eps), 1 - eps)
    return (fp_rate / (1 - fp_rate)) / (tn_rate / (1 - tn_rate))


def _pick_examples(
    df: pd.DataFrame,
    mask: pd.Series,
    sort_col: str,
    ascending: bool,
    n: int = 8,
) -> pd.DataFrame:
    sub = df.loc[mask].copy()
    sub = sub.sort_values(sort_col, ascending=ascending)
    return sub.head(n)[
        ["source", "text", "gold_nimby", "nimby_model_votes", "gold_fact", "fact_fn_votes"]
    ]


def _latex_escape(s: str, max_len: int = 220) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def run_analysis(out_dir: Path, min_model_fp: int, threshold: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _build_unified_table(threshold=threshold)

    gold_pos = df["gold_nimby"] == 1
    gold_neg = df["gold_nimby"] == 0
    consensus_fp = gold_neg & (df["nimby_model_votes"] >= min_model_fp)
    any_fp = gold_neg & (df["nimby_model_votes"] > 0)
    true_neg = gold_neg & (df["nimby_model_votes"] == 0)
    gold_tp = gold_pos

    groups = {
        "consensus_fp": consensus_fp,
        "any_fp": any_fp,
        "true_negative": true_neg,
        "gold_nimby_positive": gold_tp,
    }

    rate_frames = []
    for name, mask in groups.items():
        if mask.sum() == 0:
            continue
        rate_frames.append(_rate_table(df, mask, name))
    rates = pd.concat(rate_frames, axis=1)
    rates.to_csv(out_dir / "trigger_rates_by_group.csv")

    fp_rates = rates["consensus_fp"].dropna()
    tn_rates = rates["true_negative"].dropna()
    odds = []
    for trig in fp_rates.index:
        odds.append(
            {
                "trigger": trig,
                "fp_rate": fp_rates[trig],
                "tn_rate": tn_rates.get(trig, np.nan),
                "odds_ratio_vs_tn": _odds_ratio(fp_rates[trig], tn_rates.get(trig, 0.01)),
                "fp_rate_minus_tn": fp_rates[trig] - tn_rates.get(trig, 0.0),
            }
        )
    odds_df = pd.DataFrame(odds).sort_values("odds_ratio_vs_tn", ascending=False)
    odds_df.to_csv(out_dir / "trigger_odds_ratios.csv", index=False)

    nimby_examples = _pick_examples(
        df, consensus_fp, "nimby_model_votes", ascending=False, n=15
    )
    nimby_examples.to_csv(out_dir / "nimby_fp_examples.csv", index=False)

    fact_fn = (df["gold_fact"] == 1) & (df["fact_fn_votes"] >= 5)
    fact_examples = _pick_examples(df, fact_fn, "fact_fn_votes", ascending=False, n=15)
    fact_examples.to_csv(out_dir / "fact_claim_fn_examples.csv", index=False)

    n_all = len(df)
    n_fp = int(consensus_fp.sum())
    n_any = int(any_fp.sum())
    n_gold_nimby = int(gold_pos.sum())
    mean_votes_fp = df.loc[consensus_fp, "nimby_model_votes"].mean() if n_fp else 0.0

    top_or = odds_df.head(6)
    low_opp_fp = fp_rates.get("opposition_verbs", 0) - tn_rates.get("opposition_verbs", 0)

    lines = [
        "NIMBY qualitative error analysis (gold-standard set)",
        f"Items: {n_all} | Gold NIMBY+: {n_gold_nimby}",
        f"Consensus FP (gold-, >={min_model_fp} models tag NIMBY): {n_fp} ({100*n_fp/n_all:.1f}%)",
        f"Any-model FP: {n_any} ({100*n_any/n_all:.1f}%)",
        f"Mean model votes on consensus FP: {mean_votes_fp:.2f}",
        "",
        "Top trigger enrichment (FP vs true negative rate delta):",
    ]
    for _, row in odds_df.sort_values("fp_rate_minus_tn", ascending=False).head(8).iterrows():
        lines.append(
            f"  {row['trigger']}: FP {row['fp_rate']:.2%} vs TN {row['tn_rate']:.2%} "
            f"(Δ {row['fp_rate_minus_tn']:+.2%}, OR {row['odds_ratio_vs_tn']:.2f})"
        )
    lines.append(f"\nOpposition-verb rate delta (FP - TN): {low_opp_fp:+.2%}")

    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

  # Paper snippets
    fp_row = df.loc[consensus_fp & df["text"].str.contains("25%", case=False, na=False)]
    if fp_row.empty:
        fp_row = df.loc[consensus_fp].head(1)
    fp_text = fp_row.iloc[0]["text"] if len(fp_row) else ""

    fact_row = df.loc[
        (df["gold_fact"] == 1)
        & (df["fact_fn_votes"] >= 5)
        & df["text"].str.contains("demolished", case=False, na=False)
    ]
    if fact_row.empty:
        fact_row = df.loc[(df["gold_fact"] == 1) & (df["fact_fn_votes"] >= 5)].head(1)
    fact_text = fact_row.iloc[0]["text"] if len(fact_row) else ""

    housing_fp = fp_rates.get("housing_general", 0)
    housing_tn = tn_rates.get("housing_general", 0)
    aff_fp = fp_rates.get("affordable_housing", 0)
    prox_fp = fp_rates.get("proximity_local", 0)
    prox_tn = tn_rates.get("proximity_local", 0)
    opp_fp = fp_rates.get("opposition_verbs", 0)
    opp_tp = rates["gold_nimby_positive"].get("opposition_verbs", np.nan) if "gold_nimby_positive" in rates.columns else np.nan
    quest_fp = fp_rates.get("question_mark", 0)
    quest_tp = rates["gold_nimby_positive"].get("question_mark", np.nan) if "gold_nimby_positive" in rates.columns else np.nan

    tex = [
        "% Auto-generated by scripts/nimby_qualitative_error_analysis.py",
        f"% Consensus NIMBY FP: n={n_fp}, housing_general FP rate {housing_fp:.1%} vs TN {housing_tn:.1%}",
        "",
        "% Example FP (pro-housing; gold not NIMBY):",
        f"% {_latex_escape(fp_text)}",
        "",
        "% Example fact/claim FN:",
        f"% {_latex_escape(fact_text)}",
    ]
    (out_dir / "snippets_for_paper.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWrote outputs to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NIMBY over-tagging qualitative analysis")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "output" / "f1" / "gold_error_analysis" / "nimby_qualitative",
    )
    parser.add_argument(
        "--min-model-fp",
        type=int,
        default=3,
        help="Minimum models tagging NIMBY for consensus false positive",
    )
    parser.add_argument("--soft-threshold", type=float, default=0.5)
    args = parser.parse_args()
    run_analysis(args.out_dir, args.min_model_fp, args.soft_threshold)


if __name__ == "__main__":
    main()
