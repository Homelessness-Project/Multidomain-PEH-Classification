#!/usr/bin/env python3
"""
Compare zero-shot, few-shot, and LoRA GPT-pseudolabel results.
Outputs a CSV summary with macro/micro F1 and macro kappa.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = [
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

FLAG_COLS = {
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
    "reddit": {
        "gold_file": "gold_standard/sampled_reddit_comments.csv",
        "gold_text_col": "Comment",
        "soft_labels_file": "output/annotation/soft_labels/reddit_soft_labels.csv",
        "pred_text_col": "Comment",
    },
    "x": {
        "gold_file": "gold_standard/sampled_twitter_posts.csv",
        "gold_text_col": "Deidentified_text",
        "soft_labels_file": "output/annotation/soft_labels/x_soft_labels.csv",
        "pred_text_col": "Comment",
    },
    "news": {
        "gold_file": "gold_standard/sampled_lexisnexis_news.csv",
        "gold_text_col": "Deidentified_paragraph_text",
        "soft_labels_file": "output/annotation/soft_labels/news_soft_labels.csv",
        "pred_text_col": "Comment",
    },
    "meeting_minutes": {
        "gold_file": "gold_standard/sampled_meeting_minutes.csv",
        "gold_text_col": "Deidentified_paragraph",
        "soft_labels_file": "output/annotation/soft_labels/meeting_minutes_soft_labels.csv",
        "pred_text_col": "Comment",
    },
}


def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip().lower()


def f1_macro_micro(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, list[float]]:
    label_f1s = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        denom = (2 * tp + fp + fn)
        label_f1s.append(0.0 if denom == 0 else (2 * tp / denom))
    macro = float(np.mean(label_f1s)) if label_f1s else 0.0
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    denom = (2 * tp + fp + fn)
    micro = 0.0 if denom == 0 else (2 * tp / denom)
    return macro, micro, label_f1s


def kappa_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(y_true)
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
    denom = 1 - pe
    if denom == 0:
        return float("nan")
    return (p0 - pe) / denom


def kappa_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    kappas = []
    for i in range(y_true.shape[1]):
        kappas.append(kappa_binary(y_true[:, i], y_pred[:, i]))
    return float(np.nanmean(kappas)) if kappas else float("nan")


def load_gold_labels(source: str) -> tuple[dict, int]:
    cfg = SOURCE_CONFIG[source]
    gold_df = pd.read_csv(cfg["gold_file"])
    soft_df = pd.read_csv(cfg["soft_labels_file"])
    gold_df = gold_df.reset_index(drop=True)
    soft_df = soft_df.reset_index(drop=True)
    gold_df["_norm"] = gold_df[cfg["gold_text_col"]].apply(normalize_text)
    label_by_text = dict(zip(gold_df["_norm"], soft_df[CATEGORIES].values))
    return label_by_text, len(gold_df)


def parse_source_model(path: Path) -> tuple[str | None, str | None]:
    match = re.search(r"classified_comments_(.+?)_gold_subset_(.+?)_", path.name)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def evaluate_zero_few(path: Path, label_by_text: dict, pred_text_col: str) -> tuple[float, float, float, int, list[float]]:
    df = pd.read_csv(path)
    df["_norm"] = df[pred_text_col].apply(normalize_text)
    df = df[df["_norm"].isin(label_by_text.keys())].copy()
    if df.empty:
        return float("nan"), float("nan"), float("nan"), 0, [float("nan")] * len(CATEGORIES)
    y_true = np.vstack([label_by_text[t] for t in df["_norm"]])
    y_true = (y_true >= 0.5).astype(int)
    preds = []
    for cat in CATEGORIES:
        col = FLAG_COLS[cat]
        if col in df.columns:
            preds.append((df[col].fillna(0).astype(float).values > 0.5).astype(int))
        else:
            preds.append(np.zeros(len(df), dtype=int))
    y_pred = np.vstack(preds).T
    macro, micro, label_f1s = f1_macro_micro(y_true, y_pred)
    macro_kappa = kappa_macro(y_true, y_pred)
    return macro, micro, macro_kappa, len(df), label_f1s


def main() -> None:
    root = Path(".")
    rows = []
    per_category_rows = []

    # Zero-shot and few-shot files
    zero_files = list(root.glob("output/*/*/classified_comments_*_gold_subset_*_none_flags.csv"))
    few_files = [
        p for p in root.glob("output/*/*/classified_comments_*_gold_subset_*_flags.csv")
        if "_none_flags.csv" not in p.name
    ]

    for path in sorted(zero_files + few_files):
        source, model = parse_source_model(path)
        if source not in SOURCE_CONFIG:
            continue
        label_by_text, _ = load_gold_labels(source)
        macro, micro, kappa, n, label_f1s = evaluate_zero_few(path, label_by_text, SOURCE_CONFIG[source]["pred_text_col"])
        method = "zero_shot" if "_none_flags.csv" in path.name else "few_shot"
        rows.append({
            "source": source,
            "model": model,
            "method": method,
            "macro_f1": macro,
            "micro_f1": micro,
            "macro_kappa": kappa,
            "n_eval": n,
            "file": str(path),
        })
        for cat, f1 in zip(CATEGORIES, label_f1s):
            per_category_rows.append({
                "source": source,
                "model": model,
                "method": method,
                "category": cat,
                "f1": f1,
                "n_eval": n,
                "file": str(path),
            })

    # LoRA GPT-pseudolabel results
    for path in sorted(root.glob("nlp_outputs/**/gpt_pseudolabel*_results.json")):
        with open(path, "r") as f:
            results = json.load(f)
        training = results.get("training_config", {})
        rows.append({
            "source": training.get("source"),
            "model": training.get("model"),
            "method": "lora_gpt" if training.get("use_lora") else "ft_gpt",
            "macro_f1": results.get("macro_f1"),
            "micro_f1": results.get("micro_f1"),
            "macro_kappa": results.get("macro_kappa") or results.get("macro_kappa", float("nan")),
            "n_eval": None,
            "file": str(path),
        })
        per_f1 = results.get("per_category_f1", {})
        for cat in CATEGORIES:
            per_category_rows.append({
                "source": training.get("source"),
                "model": training.get("model"),
                "method": "lora_gpt" if training.get("use_lora") else "ft_gpt",
                "category": cat,
                "f1": per_f1.get(cat, float("nan")),
                "n_eval": None,
                "file": str(path),
            })

    out_df = pd.DataFrame(rows)
    out_dir = Path("output/summary")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "zero_few_lora_comparison.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")
    
    per_path = out_dir / "zero_few_lora_per_category.csv"
    pd.DataFrame(per_category_rows).to_csv(per_path, index=False)
    print(f"Wrote {len(per_category_rows)} rows to {per_path}")


if __name__ == "__main__":
    main()
