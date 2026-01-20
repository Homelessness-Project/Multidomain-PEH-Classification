#!/usr/bin/env python3
"""
Generate per-category Cohen's kappa LaTeX tables (16 categories).

This script produces BOTH:

Soft-label κ:
  - Computed from gold-subset prediction CSVs vs *soft* labels derived from raw annotation scores
    (scaled to [0,1] and thresholded at >= 0.5)
  - Reported per source (Reddit/News/Meeting Minutes/X)
  - Output: output/f1/kappa/soft_labels/<model>_kappa_soft.tex
    Columns are grouped by source: (Zero-shot, Few-shot) × source

Val-opt κ:
  - Fine-tuned κ read from: nlp_outputs/**/gpt_pseudolabel_*valopt_results.json (per_category_kappa)
  - Prompt κ (Zero/Few) computed against thresholded labels (same as soft-label binarization)
  - Reported per source (Reddit/News/Meeting Minutes/X)
  - Output: output/f1/kappa/val_opt/<model>_kappa_valopt.tex
    Columns are grouped by source: (Zero-shot, Few-shot, Gold-only, GPT-pseudolabel, LoRA (GPT)) × source
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_SOFT_THRESHOLD = 0.5
DEFAULT_EVAL_THRESHOLD = "val_opt"
DEFAULT_OUTDIR = Path("output/f1/kappa")

# Order + inclusion for generated main.tex files
ORDERED_MODELS: List[str] = ["llama", "phi4", "qwen", "gemini", "grok", "gpt4"]

SOURCE_ORDER: List[str] = ["reddit", "news", "meeting_minutes", "x"]

SOURCE_DISPLAY: Dict[str, str] = {
    "reddit": "Reddit",
    "news": "News",
    "meeting_minutes": "Meeting Minutes",
    "x": "X (Twitter)",
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


CATEGORY_DISPLAY: Dict[str, str] = {
    "ask a genuine question": "Ask Genuine Question",
    "ask a rhetorical question": "Ask Rhetorical Question",
    "provide a fact or claim": "Provide Fact/Claim",
    "provide an observation": "Provide Observation",
    "express their opinion": "Express Opinion",
    "express others opinions": "Express Others Opinions",
    "money aid allocation": "Money Aid Allocation",
    "government critique": "Government Critique",
    "societal critique": "Societal Critique",
    "solutions/interventions": "Solutions/Interventions",
    "personal interaction": "Personal Interaction",
    "media portrayal": "Media Portrayal",
    "not in my backyard": "Not in My Backyard",
    "harmful generalization": "Harmful Generalization",
    "deserving/undeserving": "Deserving/Undeserving",
    "racist": "Racist",
}


FLAG_COLS: Dict[str, str] = {
    "ask a genuine question": "Comment_ask a genuine question",
    "ask a rhetorical question": "Comment_ask a rhetorical question",
    "provide a fact or claim": "Comment_provide a fact or claim",
    "provide an observation": "Comment_provide an observation",
    "express their opinion": "Comment_express their opinion",
    "express others opinions": "Comment_express others opinions",
    "money aid allocation": "Critique_money aid allocation",
    "government critique": "Critique_government critique",
    "societal critique": "Critique_societal critique",
    "solutions/interventions": "Response_solutions/interventions",
    "personal interaction": "Perception_personal interaction",
    "media portrayal": "Perception_media portrayal",
    "not in my backyard": "Perception_not in my backyard",
    "harmful generalization": "Perception_harmful generalization",
    "deserving/undeserving": "Perception_deserving/undeserving",
    "racist": "Racist_Flag",
}


SOURCE_CONFIG = {
    # Raw annotation score files (counts 0..3) with deidentified text for alignment.
    "reddit": {"raw_scores_file": "annotation/reddit_raw_scores.csv", "text_col": "Deidentified_Comment"},
    "x": {"raw_scores_file": "annotation/x_raw_scores.csv", "text_col": "Deidentified text"},
    "news": {"raw_scores_file": "annotation/news_raw_scores.csv", "text_col": "Deidentified_paragraph"},
    "meeting_minutes": {"raw_scores_file": "annotation/meeting_minutes_raw_scores.csv", "text_col": "Deidentified_paragraph"},
}


RAW_COL_CANON: Dict[str, str] = {
    "ask a rheorical question": "ask a rhetorical question",
    "Racist": "racist",
}


def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip().lower()


def cohen_kappa_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = int(len(y_true))
    if n == 0:
        return float("nan")
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    p0 = (tp + tn) / n
    p_yes_true = (tp + fn) / n
    p_yes_pred = (tp + fp) / n
    p_no_true = (tn + fp) / n
    p_no_pred = (tn + fn) / n
    pe = (p_yes_true * p_yes_pred) + (p_no_true * p_no_pred)
    denom = 1.0 - pe
    if denom == 0.0:
        return float("nan")
    return (p0 - pe) / denom


def cohen_kappa_from_expected_counts(tp: float, tn: float, fp: float, fn: float) -> float:
    """
    Cohen's kappa from (possibly fractional) expected contingency counts.
    """
    n = tp + tn + fp + fn
    if n <= 0:
        return float("nan")
    p0 = (tp + tn) / n
    p_yes_true = (tp + fn) / n
    p_yes_pred = (tp + fp) / n
    p_no_true = (tn + fp) / n
    p_no_pred = (tn + fn) / n
    pe = (p_yes_true * p_yes_pred) + (p_no_true * p_no_pred)
    denom = 1.0 - pe
    if denom == 0.0:
        return float("nan")
    return (p0 - pe) / denom


def fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NaN"
    return f"{x:.2f}"


def load_soft_label_map(source: str) -> Dict[str, np.ndarray]:
    """
    Build mapping from normalized text -> soft label probabilities in [0,1].
    Raw annotation files store integer counts (typically 0..3); we scale by max observed (usually 3).
    """
    cfg = SOURCE_CONFIG[source]
    df = pd.read_csv(cfg["raw_scores_file"], low_memory=False)
    df["_norm"] = df[cfg["text_col"]].apply(normalize_text)

    raw_cols_by_cat: Dict[str, str] = {}
    for col in df.columns:
        canon = RAW_COL_CANON.get(col, col)
        if canon in CATEGORIES:
            raw_cols_by_cat[canon] = col

    cols: List[np.ndarray] = []
    for cat in CATEGORIES:
        col = raw_cols_by_cat.get(cat)
        if col is None:
            cols.append(np.zeros(len(df), dtype=float))
        else:
            cols.append(pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float).values)

    raw_mat = np.vstack(cols).T if cols else np.zeros((len(df), len(CATEGORIES)), dtype=float)
    max_val = float(np.nanmax(raw_mat)) if raw_mat.size else 0.0
    denom = max(1.0, max_val)  # expected 3.0
    soft_probs = raw_mat / denom
    return dict(zip(df["_norm"], soft_probs))


def load_raw_score_counts(source: str) -> pd.DataFrame:
    """
    Load raw annotation *count* scores (0..3) for a source.
    These are aggregated across annotators per example.
    """
    cfg = SOURCE_CONFIG[source]
    return pd.read_csv(cfg["raw_scores_file"], low_memory=False)


def gold_positive_counts_from_raw(source: str) -> Dict[str, int]:
    """
    Count gold-standard positives per category for a source.
    Gold is the majority vote among 3 annotators, i.e., positive if aggregated count >= 2.
    """
    df = load_raw_score_counts(source)
    raw_cols_by_cat: Dict[str, str] = {}
    for col in df.columns:
        canon = RAW_COL_CANON.get(col, col)
        if canon in CATEGORIES:
            raw_cols_by_cat[canon] = col

    counts: Dict[str, int] = {}
    for cat in CATEGORIES:
        col = raw_cols_by_cat.get(cat)
        if col is None:
            counts[cat] = 0
            continue
        k = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float).to_numpy()
        k = np.clip(k, 0.0, 3.0)
        counts[cat] = int(np.sum(k >= 2.0))
    return counts


def compute_human_vs_gold_kappas(source: str) -> Dict[str, float]:
    """
    Compute Cohen's kappa between an *individual annotator* and the gold standard,
    using only aggregated count data (0..3) per example.

    Since we do not have per-annotator labels, we compute the *expected* kappa for
    a randomly selected annotator, where gold is the majority vote (>=2 of 3).
    """
    df = load_raw_score_counts(source)

    # Map/normalize raw columns to canonical category names.
    raw_cols_by_cat: Dict[str, str] = {}
    for col in df.columns:
        canon = RAW_COL_CANON.get(col, col)
        if canon in CATEGORIES:
            raw_cols_by_cat[canon] = col

    kappas: Dict[str, float] = {}
    for cat in CATEGORIES:
        col = raw_cols_by_cat.get(cat)
        if col is None:
            kappas[cat] = float("nan")
            continue

        k = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float).to_numpy()
        # Clamp defensively
        k = np.clip(k, 0.0, 3.0)

        # Gold label is majority vote among 3 annotators: positive if k>=2
        gold = (k >= 2.0).astype(float)

        # For a random annotator, P(annotator positive) = k/3.
        p_pos = k / 3.0
        p_neg = 1.0 - p_pos

        # Expected contingency contributions per item:
        # If gold==1: TP += p_pos, FN += p_neg
        # If gold==0: FP += p_pos, TN += p_neg
        tp = float(np.sum(gold * p_pos))
        fn = float(np.sum(gold * p_neg))
        fp = float(np.sum((1.0 - gold) * p_pos))
        tn = float(np.sum((1.0 - gold) * p_neg))

        kappas[cat] = cohen_kappa_from_expected_counts(tp=tp, tn=tn, fp=fp, fn=fn)

    return kappas


def load_pred_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = []
    for cat in CATEGORIES:
        col = FLAG_COLS[cat]
        if col in df.columns:
            cols.append((df[col].fillna(0).astype(float).values > 0.5).astype(int))
        else:
            cols.append(np.zeros(len(df), dtype=int))
    return np.vstack(cols).T


def compute_kappas_from_preds(path: Path, source: str, label_threshold: float) -> np.ndarray:
    label_by_text = load_soft_label_map(source)
    df = pd.read_csv(path, low_memory=False)
    df["_norm"] = df["Comment"].apply(normalize_text)
    df = df[df["_norm"].isin(label_by_text.keys())].copy()
    if df.empty:
        return np.full(len(CATEGORIES), np.nan, dtype=float)
    y_true_soft = np.vstack([label_by_text[t] for t in df["_norm"]])
    y_true = (y_true_soft >= label_threshold).astype(int)
    y_pred = load_pred_matrix(df)
    kappas = np.zeros(len(CATEGORIES), dtype=float)
    for i in range(len(CATEGORIES)):
        kappas[i] = cohen_kappa_binary(y_true[:, i], y_pred[:, i])
    return kappas


def aggregate_prompt(root: Path, label_threshold: float) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    buckets: Dict[str, Dict[str, Dict[str, List[np.ndarray]]]] = {}
    for path in root.glob("output/*/*/classified_comments_*_gold_subset_*_flags.csv"):
        stem = path.name.replace("classified_comments_", "").replace("_flags.csv", "")
        if "_gold_subset_" not in stem:
            continue
        source = stem.split("_gold_subset_", 1)[0]
        rest = stem.split("_gold_subset_", 1)[1]
        model = rest.split("_", 1)[0]
        if source not in SOURCE_CONFIG:
            continue
        method = "zero_shot" if path.name.endswith("_none_flags.csv") else "few_shot"
        kappas = compute_kappas_from_preds(path, source, label_threshold)
        buckets.setdefault(model, {}).setdefault(method, {}).setdefault(source, []).append(kappas)

    out: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for model, by_method in buckets.items():
        out[model] = {}
        for method, by_source in by_method.items():
            out[model][method] = {}
            for source, mats in by_source.items():
                out[model][method][source] = (
                    np.nanmean(np.vstack(mats), axis=0) if mats else np.full(len(CATEGORIES), np.nan)
                )
    return out


def aggregate_finetuned_valopt(root: Path, eval_threshold: str) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    buckets: Dict[str, Dict[str, Dict[str, List[np.ndarray]]]] = {}
    for path in root.glob("nlp_outputs/**/gpt_pseudolabel_*valopt_results.json"):
        with open(path, "r") as f:
            data = json.load(f)
        training = data.get("training_config", {})
        if training.get("eval_threshold") != eval_threshold:
            continue
        model = training.get("model")
        source = training.get("source")
        if not model or not source:
            continue

        if training.get("train_on_gold") == "gold_only":
            key = "Gold-only"
        elif training.get("use_lora"):
            key = "LoRA (GPT)"
        else:
            key = "GPT-pseudolabel"

        per_kappa = data.get("per_category_kappa") or {}
        if not per_kappa and isinstance(data.get("val_metrics"), dict):
            per_kappa = data["val_metrics"].get("per_label_kappa") or {}

        kappas = np.array([float(per_kappa.get(cat, np.nan)) for cat in CATEGORIES], dtype=float)
        buckets.setdefault(model, {}).setdefault(key, {}).setdefault(source, []).append(kappas)

    out: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for model, by_key in buckets.items():
        out[model] = {}
        for key, by_source in by_key.items():
            out[model][key] = {}
            for source, mats in by_source.items():
                out[model][key][source] = (
                    np.nanmean(np.vstack(mats), axis=0) if mats else np.full(len(CATEGORIES), np.nan)
                )
    return out


def write_model_wide_table(
    path: Path,
    caption: str,
    label: str,
    methods: List[str],
    data: Dict[str, Dict[str, np.ndarray]],
    gold_pos_counts: Dict[str, Dict[str, int]] | None = None,
) -> None:
    """
    Write one wide table per model with columns grouped by source.

    data: method -> source -> kappas (len(CATEGORIES))
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    n_cols = len(SOURCE_ORDER) * len(methods)
    colspec = "l" + "c" * n_cols

    header1_parts = ["Category"]
    for source in SOURCE_ORDER:
        header1_parts.append(rf"\multicolumn{{{len(methods)}}}{{c}}{{{SOURCE_DISPLAY.get(source, source)}}}")

    header2_parts = [""]
    for _ in SOURCE_ORDER:
        header2_parts.extend(methods)

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(header1_parts) + r" \\",
        " & ".join(header2_parts) + r" \\",
        r"\midrule",
    ]

    for i, cat in enumerate(CATEGORIES):
        row = [CATEGORY_DISPLAY.get(cat, cat)]
        for source in SOURCE_ORDER:
            pos_count = None
            if gold_pos_counts is not None:
                pos_count = gold_pos_counts.get(source, {}).get(cat)
            for method in methods:
                vals = data.get(method, {}).get(source)
                if vals is None:
                    row.append("NaN")
                else:
                    s = fmt(float(vals[i]))
                    # Mark κ values for sparse categories (<5 positives in gold)
                    if pos_count is not None and pos_count < 5 and s != "NaN":
                        s = f"{s}*"
                    row.append(s)
        lines.append(" & ".join(row) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table*}",
    ]
    path.write_text("\n".join(lines))


def write_main_tex(dir_path: Path) -> None:
    """
    Write a main.tex that concatenates all .tex tables in this directory (excluding main.tex).
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    lines = ["% Auto-generated. Concatenated tables (no \\input).", ""]

    for model in ORDERED_MODELS:
        if dir_path.name == "soft_labels":
            p = dir_path / f"{model}_kappa_soft.tex"
        else:
            p = dir_path / f"{model}_kappa_valopt.tex"
        if not p.exists():
            continue
        lines.append(f"% ===== {p.name} =====")
        lines.append(p.read_text())
        lines.append("")
    (dir_path / "main.tex").write_text("\n".join(lines))


def cleanup_legacy_outputs(outdir: Path) -> None:
    """
    Remove legacy/unused kappa tables so the output directories stay clean.
    We keep only per-source tables for ORDERED_MODELS plus main.tex.
    """
    for sub in ["soft_labels", "val_opt", "humans_vs_gold"]:
        d = outdir / sub
        if not d.exists():
            continue
        for p in d.glob("*.tex"):
            if p.name == "main.tex":
                continue
            keep = False
            for model in ORDERED_MODELS:
                if sub == "soft_labels" and p.name == f"{model}_kappa_soft.tex":
                    keep = True
                if sub == "val_opt" and p.name == f"{model}_kappa_valopt.tex":
                    keep = True
            if sub == "humans_vs_gold" and p.name == "humans_vs_gold_kappa.tex":
                keep = True
            if not keep:
                p.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Cohen's kappa LaTeX tables.")
    # Script runs with no args; these are optional overrides.
    parser.add_argument("--soft_threshold", type=float, default=DEFAULT_SOFT_THRESHOLD)
    parser.add_argument("--eval_threshold", default=DEFAULT_EVAL_THRESHOLD)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    root = Path(".")
    outdir = Path(args.outdir)
    soft_outdir = outdir / "soft_labels"
    val_outdir = outdir / "val_opt"
    humans_outdir = outdir / "humans_vs_gold"
    outdir.mkdir(parents=True, exist_ok=True)
    cleanup_legacy_outputs(outdir)

    # Soft-label tables: prompt models only (Zero vs Few), grouped by source.
    prompt = aggregate_prompt(root, args.soft_threshold)
    gold_pos_counts = {source: gold_positive_counts_from_raw(source) for source in SOURCE_ORDER}
    for model in ORDERED_MODELS:
        if model not in prompt:
            continue
        by_method = prompt[model]
        methods = ["Zero-shot", "Few-shot"]
        data: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in methods}
        for source in SOURCE_ORDER:
            data["Zero-shot"][source] = by_method.get("zero_shot", {}).get(source, np.full(len(CATEGORIES), np.nan))
            data["Few-shot"][source] = by_method.get("few_shot", {}).get(source, np.full(len(CATEGORIES), np.nan))

        write_model_wide_table(
            soft_outdir / f"{model}_kappa_soft.tex",
            f"Soft-label Cohen's $\\kappa$ by category ({model}). (* indicates $<$5 positive examples in gold labels; interpret with caution.)",
            f"tab:{model}_kappa_soft",
            methods,
            data,
            gold_pos_counts=gold_pos_counts,
        )
    write_main_tex(soft_outdir)

    # Val-opt tables: prompt (Zero/Few) + finetuned (Gold-only / GPT / LoRA), grouped by source.
    finetuned = aggregate_finetuned_valopt(root, args.eval_threshold)
    for model in ORDERED_MODELS:
        if model not in prompt and model not in finetuned:
            continue
        methods = ["Zero-shot", "Few-shot", "Gold-only", "GPT-pseudolabel", "LoRA (GPT)"]
        data: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in methods}
        for source in SOURCE_ORDER:
            # Prompt
            data["Zero-shot"][source] = (
                prompt.get(model, {}).get("zero_shot", {}).get(source, np.full(len(CATEGORIES), np.nan))
            )
            data["Few-shot"][source] = (
                prompt.get(model, {}).get("few_shot", {}).get(source, np.full(len(CATEGORIES), np.nan))
            )
            # Finetuned (val-opt)
            data["Gold-only"][source] = (
                finetuned.get(model, {}).get("Gold-only", {}).get(source, np.full(len(CATEGORIES), np.nan))
            )
            data["GPT-pseudolabel"][source] = (
                finetuned.get(model, {}).get("GPT-pseudolabel", {}).get(source, np.full(len(CATEGORIES), np.nan))
            )
            data["LoRA (GPT)"][source] = (
                finetuned.get(model, {}).get("LoRA (GPT)", {}).get(source, np.full(len(CATEGORIES), np.nan))
            )

        write_model_wide_table(
            val_outdir / f"{model}_kappa_valopt.tex",
            f"Validation-optimized Cohen's $\\kappa$ by category ({model}). (* indicates $<$5 positive examples in gold labels; interpret with caution.)",
            f"tab:{model}_kappa_valopt",
            methods,
            data,
            gold_pos_counts=gold_pos_counts,
        )
    write_main_tex(val_outdir)

    # Human-vs-gold kappa (expected kappa for a randomly selected annotator vs majority-vote gold).
    humans_outdir.mkdir(parents=True, exist_ok=True)
    by_source: Dict[str, Dict[str, float]] = {}
    for source in SOURCE_ORDER:
        by_source[source] = compute_human_vs_gold_kappas(source)

    # Write a single wide table: columns are sources (one value each) and rows are categories.
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Category & Reddit & News & Meeting Minutes & X (Twitter) \\",
        r"\midrule",
    ]
    for cat in CATEGORIES:
        row = [CATEGORY_DISPLAY.get(cat, cat)]
        for source in SOURCE_ORDER:
            s = fmt(float(by_source.get(source, {}).get(cat, np.nan)))
            if gold_pos_counts.get(source, {}).get(cat, 0) < 5 and s != "NaN":
                s = f"{s}*"
            row.append(s)
        lines.append(" & ".join(row) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Cohen's $\kappa$ by category between annotators and the gold standard (expected $\kappa$ for a randomly selected annotator vs majority-vote gold). (* indicates $<$5 positive examples in gold labels; interpret with caution.)}",
        r"\label{tab:humans_vs_gold_kappa}",
        r"\end{table*}",
    ]
    (humans_outdir / "humans_vs_gold_kappa.tex").write_text("\n".join(lines))

    # Concatenated main.tex for easy copy/paste
    (humans_outdir / "main.tex").write_text(
        "\n".join(
            [
                "% Auto-generated. Concatenated tables (no \\input).",
                "",
                "% ===== humans_vs_gold_kappa.tex =====",
                (humans_outdir / "humans_vs_gold_kappa.tex").read_text(),
                "",
            ]
        )
    )


if __name__ == "__main__":
    main()

