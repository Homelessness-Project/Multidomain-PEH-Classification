#!/usr/bin/env python3
"""
Gold-subset error analysis: LLM false positives/negatives vs soft labels (16 categories).

Compares each prompt model's gold-subset flag predictions to human soft labels
(binarized at --soft-threshold, default 0.5). Reports per-category TP/FP/FN/TN,
FPR/FNR, and prevalence delta (predicted − gold) to flag over- vs under-detection.

Usage (repo root):
    python scripts/gold_subset_llm_error_analysis.py
    python scripts/gold_subset_llm_error_analysis.py --top 8 --shot few_shot
    python scripts/gold_subset_llm_error_analysis.py --models gpt4 gemini --sources news

Outputs (default output/f1/gold_error_analysis/):
    per_model_category.csv   # one row per source × model × shot × category
    model_summary.csv        # totals / macro rates per source × model × shot
    category_summary.csv     # mean delta & error rates aggregated across models
    over_detection_rank.csv  # categories ranked by mean prevalence delta (positive)
    under_detection_rank.csv # categories ranked by mean prevalence delta (negative)
    latex/                   # EMNLP-ready tables (under-detection across 6 LLMs)
        underdetection_delta_pooled.tex
        finetuned_pseudolabel_delta_pooled.tex   # BERT/RoBERTa/ModernBERT + LoRA on GPT labels
        finetuned_pseudolabel_negative_only_pooled.tex
        main.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCES = ["reddit", "news", "meeting_minutes", "x"]
PROMPT_MODELS = ["llama", "phi4", "qwen", "gemini", "grok", "gpt4"]
SHOT_TYPES = ["zero_shot", "few_shot"]

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

MODEL_COL_TO_SOFT = {
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

CATEGORY_ORDER = list(LABEL_COLUMNS)
# Racist excluded from publication tables (sparse gold support).
TABLE_CATEGORY_ORDER = [c for c in CATEGORY_ORDER if c != "racist"]

CATEGORY_DISPLAY = {
    "ask a genuine question": "Ask Genuine Question",
    "ask a rhetorical question": "Ask Rhetorical Question",
    "provide a fact or claim": "Provide Fact/Claim",
    "provide an observation": "Provide Observation",
    "express their opinion": "Express Opinion",
    "express others opinions": "Express Others' Opinions",
    "money aid allocation": "Money/Aid Allocation",
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

SOURCE_DISPLAY = {
    "reddit": "Reddit",
    "news": "News",
    "meeting_minutes": "Meeting Minutes",
    "x": "X (Twitter)",
}

MODEL_DISPLAY = {
    "llama": "LLaMA",
    "phi4": "Phi-4",
    "qwen": "Qwen",
    "gemini": "Gemini",
    "grok": "Grok",
    "gpt4": "GPT-4",
}

# Fine-tuned on GPT-4 pseudolabels; evaluated on gold test split (val-opt thresholds).
FINETUNED_MODELS = [
    "bert-base-uncased",
    "roberta-base",
    "modernbert-base",
    "llama",
    "phi4",
    "qwen",
]
FINETUNED_USE_LORA = {
    "bert-base-uncased": False,
    "roberta-base": False,
    "modernbert-base": False,
    "llama": True,
    "phi4": True,
    "qwen": True,
}
FINETUNED_DISPLAY = {
    "bert-base-uncased": "BERT",
    "roberta-base": "RoBERTa",
    "modernbert-base": "ModernBERT",
    "llama": "LLaMA (LoRA)",
    "phi4": "Phi-4 (LoRA)",
    "qwen": "Qwen (LoRA)",
}

FINETUNED_LATEX_NOTE_POOLED = (
    r"Fine-tuned on GPT-4 pseudolabels; gold test split, val-opt thresholds. "
    r"Gap $\hat{\pi}_{\mathrm{model}}-\pi_{\mathrm{human}}$ (pp), pooled across sources. "
    r"Negative = under-detection. Overall: row mean; footer: column means."
)

LATEX_NOTE = (
    r"Gap $\hat{\pi}_{\mathrm{model}}-\pi_{\mathrm{human}}$ (pp), zero/few-shot pooled. "
    r"$\tau{=}0.5$; negative = under-detection. "
    r"{$^*$}${<}5$ human positives in that source."
)

LATEX_NOTE_POOLED = (
    r"Gap $\hat{\pi}_{\mathrm{model}}-\pi_{\mathrm{human}}$ (pp), pooled over shots and four sources. "
    r"$\tau{=}0.5$; negative = under-detection. Overall: row mean; footer: column means."
)

FNR_LATEX_NOTE = (
    r"Mean FNR (\%), zero/few-shot pooled; higher = more missed human positives. "
    r"{$^*$}${<}5$ human positives in that source."
)

FNR_LATEX_NOTE_POOLED = (
    r"Mean FNR (\%), pooled over shots and four sources. Overall: row mean; footer: column means."
)


def _binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    yt = y_true.astype(int)
    yp = y_pred.astype(int)
    return {
        "tp": int(np.sum((yt == 1) & (yp == 1))),
        "fp": int(np.sum((yt == 0) & (yp == 1))),
        "fn": int(np.sum((yt == 1) & (yp == 0))),
        "tn": int(np.sum((yt == 0) & (yp == 0))),
    }


def _rates_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    pos_h = tp + fn
    neg_h = fp + tn
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fp / neg_h if neg_h > 0 else 0.0,
        "fnr": fn / pos_h if pos_h > 0 else 0.0,
    }


def _few_shot_suffix(source: str, shot_type: str) -> str:
    return "none" if shot_type == "zero_shot" else source


def _prediction_path(source: str, model: str, shot_type: str) -> Path:
    few = _few_shot_suffix(source, shot_type)
    return (
        REPO_ROOT
        / "output"
        / source
        / model
        / f"classified_comments_{source}_gold_subset_{model}_{few}_flags.csv"
    )


def _load_soft_labels(source: str) -> pd.DataFrame | None:
    path = REPO_ROOT / "output" / "annotation" / "soft_labels" / f"{source}_soft_labels.csv"
    if not path.exists():
        print(f"  [skip] soft labels not found: {path}")
        return None
    df = pd.read_csv(path)
    missing = [c for c in LABEL_COLUMNS if c not in df.columns]
    if missing:
        print(f"  [skip] {source} soft labels missing columns: {missing}")
        return None
    return df


def _load_predictions(source: str, model: str, shot_type: str) -> pd.DataFrame | None:
    path = _prediction_path(source, model, shot_type)
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame(index=df.index)
    for model_col, soft_col in MODEL_COL_TO_SOFT.items():
        if model_col in df.columns:
            out[soft_col] = (
                pd.to_numeric(df[model_col], errors="coerce").fillna(0).astype(int).clip(0, 1)
            )
    if out.shape[1] != len(LABEL_COLUMNS):
        return None
    return out


def _align_gold_pred(
    soft: pd.DataFrame, pred: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align by row index; trim to common length if needed."""
    n = min(len(soft), len(pred))
    if len(soft) != len(pred):
        print(
            f"    warning: length mismatch soft={len(soft)} pred={len(pred)}; using first {n} rows"
        )
    gold = soft[LABEL_COLUMNS].iloc[:n].copy()
    return gold, pred.iloc[:n]


def _per_category_rows(
    gold_bin: pd.DataFrame,
    pred_bin: pd.DataFrame,
    *,
    source: str,
    model: str,
    shot_type: str,
    soft_threshold: float,
) -> list[dict]:
    rows: list[dict] = []
    n = len(gold_bin)
    for lab in LABEL_COLUMNS:
        y_true = (pd.to_numeric(gold_bin[lab], errors="coerce").fillna(0) >= soft_threshold).astype(
            int
        ).values
        y_pred = pred_bin[lab].astype(int).values
        c = _binary_counts(y_true, y_pred)
        r = _rates_from_counts(c["tp"], c["fp"], c["fn"], c["tn"])
        prev_gold = float(y_true.mean())
        prev_pred = float(y_pred.mean())
        delta = prev_pred - prev_gold
        rows.append(
            {
                "source": source,
                "model": model,
                "shot_type": shot_type,
                "category": lab,
                "n": n,
                "gold_positive_count": int(y_true.sum()),
                "pred_positive_count": int(y_pred.sum()),
                "prevalence_gold": round(prev_gold, 6),
                "prevalence_pred": round(prev_pred, 6),
                "delta_prevalence_pred_minus_gold": round(delta, 6),
                "detection_bias": (
                    "over" if delta > 0.01 else ("under" if delta < -0.01 else "balanced")
                ),
                "tp": c["tp"],
                "fp": c["fp"],
                "fn": c["fn"],
                "tn": c["tn"],
                "fpr": round(r["fpr"], 6),
                "fnr": round(r["fnr"], 6),
                "precision": round(r["precision"], 6),
                "recall": round(r["recall"], 6),
                "f1": round(r["f1"], 6),
                "low_gold_support": int(y_true.sum()) < 5,
            }
        )
    return rows


def _model_summary_from_detail(detail: pd.DataFrame) -> pd.DataFrame:
    agg = (
        detail.groupby(["source", "model", "shot_type"], as_index=False)
        .agg(
            n=("n", "first"),
            total_fp=("fp", "sum"),
            total_fn=("fn", "sum"),
            total_tp=("tp", "sum"),
            macro_fpr=("fpr", "mean"),
            macro_fnr=("fnr", "mean"),
            macro_f1=("f1", "mean"),
            mean_abs_delta_prevalence=("delta_prevalence_pred_minus_gold", lambda s: s.abs().mean()),
            mean_delta_prevalence=("delta_prevalence_pred_minus_gold", "mean"),
            categories_over=("detection_bias", lambda s: int((s == "over").sum())),
            categories_under=("detection_bias", lambda s: int((s == "under").sum())),
        )
        .sort_values(["source", "total_fp", "total_fn"], ascending=[True, False, False])
    )
    return agg


def _category_summary_from_detail(detail: pd.DataFrame) -> pd.DataFrame:
    return (
        detail.groupby(["source", "category"], as_index=False)
        .agg(
            mean_delta_prevalence=("delta_prevalence_pred_minus_gold", "mean"),
            mean_abs_delta_prevalence=("delta_prevalence_pred_minus_gold", lambda s: s.abs().mean()),
            mean_fpr=("fpr", "mean"),
            mean_fnr=("fnr", "mean"),
            mean_f1=("f1", "mean"),
            total_fp=("fp", "sum"),
            total_fn=("fn", "sum"),
            n_models=("model", "nunique"),
        )
        .sort_values(["source", "mean_abs_delta_prevalence"], ascending=[True, False])
    )


def _gold_support_flags(detail: pd.DataFrame) -> dict[tuple[str, str], bool]:
    """(source, category) -> low support if gold positives < 5 in that source."""
    support = (
        detail.groupby(["source", "category"], as_index=False)["gold_positive_count"]
        .first()
        .set_index(["source", "category"])["gold_positive_count"]
    )
    return {key: int(support.loc[key]) < 5 for key in support.index}


def _gold_support_pooled(detail: pd.DataFrame) -> dict[str, bool]:
    """category -> low support if sum of gold positives across sources < 5."""
    per_source = (
        detail.groupby(["source", "category"], as_index=False)["gold_positive_count"]
        .first()
        .groupby("category")["gold_positive_count"]
        .sum()
    )
    return {cat: int(per_source.get(cat, 0)) < 5 for cat in TABLE_CATEGORY_ORDER}


def _pivot_metric(
    detail: pd.DataFrame,
    *,
    metric_col: str,
    models: list[str],
    by_source: bool,
) -> tuple[pd.DataFrame, dict[tuple[str, str], bool]]:
    """Mean metric per category × model, optionally per source. Averages zero/few shot."""
    low_support = _gold_support_flags(detail)
    sub = detail[detail["model"].isin(models)].copy()
    if by_source:
        pivot = (
            sub.groupby(["source", "category", "model"], as_index=False)[metric_col]
            .mean()
            .pivot_table(index=["source", "category"], columns="model", values=metric_col)
        )
        pivot = pivot.reindex(columns=models)
    else:
        pivot = (
            sub.groupby(["category", "model"], as_index=False)[metric_col]
            .mean()
            .pivot(index="category", columns="model", values=metric_col)
        )
        pivot = pivot.reindex(index=TABLE_CATEGORY_ORDER, columns=models)
    return pivot, low_support


def _format_cell(
    value: float,
    *,
    as_percent: bool,
    signed_delta: bool,
    balanced_eps: float = 0.01,
    negative_only: bool = False,
) -> str:
    if pd.isna(value):
        return "---"
    if as_percent:
        v = value * 100.0
    else:
        v = value
    eps = balanced_eps * (100.0 if as_percent else 1.0)
    if abs(v) < eps:
        return "0"
    if negative_only and signed_delta and v > eps:
        return "---"
    if signed_delta and v > 0:
        return f"+{v:.1f}" if as_percent else f"+{v:.2f}"
    return f"{v:.1f}" if as_percent else f"{v:.2f}"


def _latex_model_header(models: list[str], display_map: dict[str, str] | None = None) -> list[str]:
    d = display_map or MODEL_DISPLAY
    return [d.get(m, m.replace("-", " ").title()) for m in models]


def _row_mean(numeric: list[float]) -> float:
    arr = np.asarray(numeric, dtype=float)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _overall_cell(
    value: float,
    *,
    as_percent: bool,
    signed_delta: bool,
    negative_only: bool,
) -> str:
    s = _format_cell(
        value,
        as_percent=as_percent,
        signed_delta=signed_delta,
        negative_only=negative_only,
    )
    return s


def _cells_for_row(
    *,
    category_label: str,
    numeric: list[float],
    models: list[str],
    signed_delta: bool,
    as_percent: bool,
    negative_only: bool,
    low_support_star: bool,
    source: str | None,
    category: str,
    low_support: dict[tuple[str, str], bool],
    low_support_pooled: dict[str, bool] | None = None,
) -> list[str]:
    """Build table row: category + model cells + Overall (row mean across models)."""
    cells = [category_label]
    row_mean = _row_mean(numeric)

    for model, v in zip(models, numeric):
        s = _format_cell(
            v,
            as_percent=as_percent,
            signed_delta=signed_delta,
            negative_only=negative_only,
        )
        star = False
        if source is not None:
            star = low_support.get((source, category), False) and not pd.isna(v)
        elif low_support_pooled is not None:
            star = low_support_pooled.get(category, False) and not pd.isna(v)
        elif low_support_star and not pd.isna(v):
            star = True
        if star and s not in ("---", "0"):
            s = f"{s}$^*$"
        cells.append(s)

    cells.append(
        _overall_cell(
            row_mean,
            as_percent=as_percent,
            signed_delta=signed_delta,
            negative_only=False,
        )
    )
    return cells


def _summary_row_cells(
    pivot: pd.DataFrame,
    *,
    models: list[str],
    index_slice,
    signed_delta: bool,
    as_percent: bool,
    negative_only: bool,
    row_label: str = r"\textit{Overall Avg}",
) -> list[str]:
    """Footer row: macro mean per model column + grand Overall."""
    if index_slice is None:
        sub = pivot
    else:
        sub = pivot.loc[index_slice]

    cells = [row_label]
    col_means: list[float] = []
    for model in models:
        if model in sub.columns:
            col_means.append(float(sub[model].mean()))
        else:
            col_means.append(float("nan"))
        cells.append(
            _overall_cell(
                col_means[-1],
                as_percent=as_percent,
                signed_delta=signed_delta,
                negative_only=False,
            )
        )
    grand = _row_mean(col_means)
    cells.append(
        _overall_cell(
            grand,
            as_percent=as_percent,
            signed_delta=signed_delta,
            negative_only=False,
        )
    )
    return cells


def _write_underdetection_table(
    pivot: pd.DataFrame,
    *,
    models: list[str],
    low_support: dict[tuple[str, str], bool],
    out_path: Path,
    caption: str,
    label: str,
    metric_col_kind: str,
    by_source: bool,
    table_star: bool,
    negative_only: bool = False,
    model_display: dict[str, str] | None = None,
    low_support_pooled: dict[str, bool] | None = None,
) -> None:
    """
    metric_col_kind: 'delta' (signed pp gap) or 'fnr' (0-1 rate, shown as %)
    negative_only: for delta tables, show only under-detection (omit over-detection cells)
    """
    signed_delta = metric_col_kind == "delta"
    as_percent = True
    lines: list[str] = []
    env = "table*" if table_star else "table"
    lines.append(rf"\begin{{{env}}}[htbp]")
    lines.append(r"\centering")
    if table_star:
        lines.append(r"\small")
        lines.append(r"\setlength{\tabcolsep}{3pt}")
    n_model_cols = len(models)
    n_data_cols = n_model_cols + 1  # + Overall (row mean across models)
    col_spec = "l" + "c" * n_data_cols
    if table_star:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header = ["Category"] + _latex_model_header(models, model_display) + ["Overall"]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    if by_source:
        for source in SOURCES:
            if source not in pivot.index.get_level_values(0):
                continue
            lines.append(
                rf"\multicolumn{{{n_data_cols + 1}}}{{l}}{{\textit{{{SOURCE_DISPLAY.get(source, source)}}}}} \\"
            )
            for category in TABLE_CATEGORY_ORDER:
                if (source, category) not in pivot.index:
                    continue
                row_vals = pivot.loc[(source, category)]
                numeric = [row_vals.get(model, np.nan) for model in models]
                cells = _cells_for_row(
                    category_label=CATEGORY_DISPLAY.get(category, category.title()),
                    numeric=numeric,
                    models=models,
                    signed_delta=signed_delta,
                    as_percent=as_percent,
                    negative_only=negative_only,
                    low_support_star=False,
                    source=source,
                    category=category,
                    low_support=low_support,
                    low_support_pooled=None,
                )
                lines.append(" & ".join(cells) + r" \\")
            # Per-source footer: macro average within this source
            src_index = [(source, c) for c in TABLE_CATEGORY_ORDER if (source, c) in pivot.index]
            if src_index:
                lines.append(r"\midrule")
                src_label = rf"\textit{{{SOURCE_DISPLAY.get(source, source)} Avg}}"
                lines.append(
                    " & ".join(
                        _summary_row_cells(
                            pivot,
                            models=models,
                            index_slice=src_index,
                            signed_delta=signed_delta,
                            as_percent=as_percent,
                            negative_only=negative_only,
                            row_label=src_label,
                        )
                    )
                    + r" \\"
                )
            lines.append(r"\addlinespace[0.4em]")
        if lines and lines[-1].startswith(r"\addlinespace"):
            lines.pop()
        # Grand footer across all sources
        lines.append(r"\midrule")
        lines.append(
            " & ".join(
                _summary_row_cells(
                    pivot,
                    models=models,
                    index_slice=None,
                    signed_delta=signed_delta,
                    as_percent=as_percent,
                    negative_only=negative_only,
                    row_label=r"\textit{Overall Avg (pooled)}",
                )
            )
            + r" \\"
        )
    else:
        for category in TABLE_CATEGORY_ORDER:
            if category not in pivot.index:
                continue
            row_vals = pivot.loc[category]
            numeric = [row_vals.get(m, np.nan) for m in models]
            cells = _cells_for_row(
                category_label=CATEGORY_DISPLAY.get(category, category.title()),
                numeric=numeric,
                models=models,
                signed_delta=signed_delta,
                as_percent=as_percent,
                negative_only=negative_only,
                low_support_star=False,
                source=None,
                category=category,
                low_support=low_support,
                low_support_pooled=low_support_pooled,
            )
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")
        lines.append(
            " & ".join(
                _summary_row_cells(
                    pivot,
                    models=models,
                    index_slice=None,
                    signed_delta=signed_delta,
                    as_percent=as_percent,
                    negative_only=negative_only,
                )
            )
            + r" \\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if table_star:
        lines.append("}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(rf"\end{{{env}}}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_test_gold_binary(source: str, soft_threshold: float) -> tuple[np.ndarray, int]:
    """Gold test split (50/50) matching gpt_only finetune evaluation."""
    path = REPO_ROOT / "output" / "annotation" / "soft_labels" / f"{source}_soft_labels.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    soft = pd.read_csv(path)
    label_mat = np.zeros((len(soft), len(LABEL_COLUMNS)), dtype=float)
    for i, cat in enumerate(LABEL_COLUMNS):
        if cat not in soft.columns:
            continue
        col = "Racist" if cat == "racist" and "Racist" in soft.columns else cat
        label_mat[:, i] = (
            pd.to_numeric(soft[col], errors="coerce").fillna(0) >= soft_threshold
        ).astype(float)
    _, test_labels = train_test_split(label_mat, test_size=0.5, random_state=42)
    return test_labels, len(test_labels)


def _delta_pp_from_precision_recall(
    precision: float, recall: float, gold_pos: float, n: int
) -> float:
    """Prevalence gap (percentage points) from aggregate P/R on the test split."""
    if n <= 0:
        return float("nan")
    p = 0.0 if precision is None or (isinstance(precision, float) and np.isnan(precision)) else float(precision)
    r = 0.0 if recall is None or (isinstance(recall, float) and np.isnan(recall)) else float(recall)
    g = float(gold_pos)
    if g <= 0:
        return 0.0 if (p <= 0 and r <= 0) else float("nan")
    gold_prev = g / n
    if p <= 0:
        return -gold_prev * 100.0 if r <= 0 else float("nan")
    return gold_prev * (r / p - 1.0) * 100.0


def _find_finetuned_results_json(source: str, model: str) -> Path | None:
    """Best val-opt JSON for model fine-tuned on GPT pseudolabels (not gold-only training)."""
    candidates: list[Path] = []
    for path in (REPO_ROOT / "nlp_outputs").rglob("gpt_pseudolabel*.json"):
        name = path.name.lower()
        if "train-gold_only" in name or "train-gold-only" in name:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        tc = data.get("training_config") or {}
        if tc.get("source") != source or tc.get("model") != model:
            continue
        if tc.get("train_on_gold") == "gold_only":
            continue
        if bool(tc.get("use_lora")) != FINETUNED_USE_LORA.get(model, False):
            continue
        if tc.get("eval_threshold") not in (None, "val_opt") and "valopt" not in name:
            continue
        candidates.append(path)
    if not candidates:
        return None
    # Prefer validation-optimized results files.
    candidates.sort(
        key=lambda p: (
            0 if "valopt" in p.name else 1,
            len(p.name),
        )
    )
    return candidates[0]


def _finetuned_low_support(soft_threshold: float) -> tuple[dict[tuple[str, str], bool], dict[str, bool]]:
    per_source: dict[tuple[str, str], bool] = {}
    pooled_counts: dict[str, float] = {cat: 0.0 for cat in TABLE_CATEGORY_ORDER}
    for source in SOURCES:
        try:
            test_labels, _ = _load_test_gold_binary(source, soft_threshold)
        except FileNotFoundError:
            continue
        for i, cat in enumerate(LABEL_COLUMNS):
            if cat == "racist":
                continue
            n_pos = float(test_labels[:, i].sum())
            per_source[(source, cat)] = n_pos < 5
            if cat in pooled_counts:
                pooled_counts[cat] += n_pos
    pooled = {cat: pooled_counts[cat] < 5 for cat in TABLE_CATEGORY_ORDER}
    return per_source, pooled


def _build_finetuned_delta_pivot(
    models: list[str], soft_threshold: float
) -> pd.DataFrame:
    """Pooled mean prevalence gap per category × fine-tuned model."""
    records: list[dict] = []
    for source in SOURCES:
        try:
            test_labels, n_test = _load_test_gold_binary(source, soft_threshold)
        except FileNotFoundError:
            print(f"  [skip] no soft labels for finetuned eval: {source}")
            continue
        for model in models:
            path = _find_finetuned_results_json(source, model)
            if path is None:
                print(f"  [missing] {model} @ {source}")
                continue
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            prec = data.get("per_category_precision") or {}
            rec = data.get("per_category_recall") or {}
            for cat in TABLE_CATEGORY_ORDER:
                idx = LABEL_COLUMNS.index(cat)
                gold_pos = float(test_labels[:, idx].sum())
                delta = _delta_pp_from_precision_recall(
                    prec.get(cat), rec.get(cat), gold_pos, n_test
                )
                records.append(
                    {
                        "source": source,
                        "model": model,
                        "category": cat,
                        "delta_prevalence_pred_minus_gold": delta / 100.0,
                    }
                )
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    pivot = (
        df.groupby(["category", "model"], as_index=False)["delta_prevalence_pred_minus_gold"]
        .mean()
        .pivot(index="category", columns="model", values="delta_prevalence_pred_minus_gold")
    )
    return pivot.reindex(index=TABLE_CATEGORY_ORDER, columns=models)


def generate_finetuned_pseudolabel_latex(
    out_dir: Path,
    *,
    models: list[str] | None = None,
    soft_threshold: float = 0.5,
) -> None:
    """Pooled LaTeX tables for models fine-tuned on GPT pseudolabels vs gold test labels."""
    models = models or FINETUNED_MODELS
    latex_dir = out_dir / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)
    print("\nFine-tuned (GPT pseudolabel) prevalence-gap tables...")
    pivot = _build_finetuned_delta_pivot(models, soft_threshold)
    if pivot.empty:
        print("  No finetuned val-opt results found under nlp_outputs/.")
        return
    _, finetuned_pooled_low = _finetuned_low_support(soft_threshold)
    _write_underdetection_table(
        pivot,
        models=models,
        low_support={},
        out_path=latex_dir / "finetuned_pseudolabel_delta_pooled.tex",
        caption=(
            r"Prevalence gaps (GPT-pseudolabel fine-tuned models, pooled). "
            + FINETUNED_LATEX_NOTE_POOLED
        ),
        label="tab:finetuned_pseudolabel_delta_pooled",
        metric_col_kind="delta",
        by_source=False,
        table_star=True,
        model_display=FINETUNED_DISPLAY,
        low_support_pooled=finetuned_pooled_low,
    )
    _write_underdetection_table(
        pivot,
        models=models,
        low_support={},
        out_path=latex_dir / "finetuned_pseudolabel_negative_only_pooled.tex",
        caption=(
            r"Under-detection only (GPT-pseudolabel fine-tuned models, pooled). "
            r"``0'' = match; ``---'' = over-detection. " + FINETUNED_LATEX_NOTE_POOLED
        ),
        label="tab:finetuned_pseudolabel_negative_only_pooled",
        metric_col_kind="delta",
        by_source=False,
        table_star=False,
        negative_only=True,
        model_display=FINETUNED_DISPLAY,
        low_support_pooled=finetuned_pooled_low,
    )
    pivot_pp = pivot * 100.0
    pivot_pp.to_csv(latex_dir / "finetuned_pseudolabel_delta_pooled.csv")
    print(f"  Wrote {latex_dir}/finetuned_pseudolabel_delta_pooled.tex")


def generate_emnlp_latex_tables(
    detail: pd.DataFrame,
    out_dir: Path,
    models: list[str],
    *,
    soft_threshold: float = 0.5,
) -> None:
    """Write pooled and by-source LaTeX tables for under-detection (all six LLMs)."""
    latex_dir = out_dir / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)
    low_support = _gold_support_flags(detail)
    low_support_pooled = _gold_support_pooled(detail)

    delta_pooled, _ = _pivot_metric(
        detail, metric_col="delta_prevalence_pred_minus_gold", models=models, by_source=False
    )
    delta_by_source, _ = _pivot_metric(
        detail, metric_col="delta_prevalence_pred_minus_gold", models=models, by_source=True
    )
    fnr_pooled, _ = _pivot_metric(detail, metric_col="fnr", models=models, by_source=False)
    fnr_by_source, _ = _pivot_metric(detail, metric_col="fnr", models=models, by_source=True)

    _write_underdetection_table(
        delta_pooled,
        models=models,
        low_support=low_support,
        out_path=latex_dir / "underdetection_delta_pooled.tex",
        caption=(
            r"Prevalence gaps vs.\ soft labels (six prompt LLMs, gold subset, pooled). "
            + LATEX_NOTE_POOLED
        ),
        label="tab:underdetection_delta_pooled",
        metric_col_kind="delta",
        by_source=False,
        table_star=True,
        low_support_pooled=low_support_pooled,
    )
    _write_underdetection_table(
        delta_pooled,
        models=models,
        low_support=low_support,
        out_path=latex_dir / "underdetection_negative_only_pooled.tex",
        caption=(
            r"Under-detection only (six prompt LLMs, pooled). "
            r"``0'' = match; ``---'' = over-detection. " + LATEX_NOTE_POOLED
        ),
        label="tab:underdetection_negative_only_pooled",
        metric_col_kind="delta",
        by_source=False,
        table_star=False,
        negative_only=True,
        low_support_pooled=low_support_pooled,
    )
    _write_underdetection_table(
        delta_by_source,
        models=models,
        low_support=low_support,
        out_path=latex_dir / "underdetection_delta_by_source.tex",
        caption=(r"Prevalence gaps by source (six prompt LLMs). " + LATEX_NOTE),
        label="tab:underdetection_delta_by_source",
        metric_col_kind="delta",
        by_source=True,
        table_star=True,
    )
    _write_underdetection_table(
        fnr_pooled,
        models=models,
        low_support=low_support,
        out_path=latex_dir / "underdetection_fnr_pooled.tex",
        caption=(
            r"False-negative rates (six prompt LLMs, gold subset, pooled). " + FNR_LATEX_NOTE_POOLED
        ),
        label="tab:underdetection_fnr_pooled",
        metric_col_kind="fnr",
        by_source=False,
        table_star=False,
        low_support_pooled=low_support_pooled,
    )
    _write_underdetection_table(
        fnr_by_source,
        models=models,
        low_support=low_support,
        out_path=latex_dir / "underdetection_fnr_by_source.tex",
        caption=(r"False-negative rates by source (six prompt LLMs). " + FNR_LATEX_NOTE),
        label="tab:underdetection_fnr_by_source",
        metric_col_kind="fnr",
        by_source=True,
        table_star=True,
    )

    # Compact CSV for supplementary / easy sorting
    pooled_long = (
        detail[detail["category"].isin(TABLE_CATEGORY_ORDER)]
        .groupby(["category", "model"], as_index=False)
        .agg(
            mean_delta_pp=("delta_prevalence_pred_minus_gold", lambda s: s.mean() * 100),
            mean_fnr_pct=("fnr", lambda s: s.mean() * 100),
            mean_recall_pct=("recall", lambda s: s.mean() * 100),
        )
        .sort_values(["mean_delta_pp", "category"])
    )
    pooled_long.to_csv(latex_dir / "underdetection_summary_pooled.csv", index=False)

    generate_finetuned_pseudolabel_latex(out_dir, soft_threshold=soft_threshold)
    finetuned_delta = latex_dir / "finetuned_pseudolabel_delta_pooled.tex"

    main_lines = [
        "% Auto-generated EMNLP under-detection tables (gold subset vs soft labels).",
        "% Requires: \\usepackage{booktabs}",
        "",
        "% ===== Main text (recommended): under-detection only, six LLMs =====",
    ]
    with open(latex_dir / "underdetection_negative_only_pooled.tex", encoding="utf-8") as f:
        main_lines.append(f.read())
    main_lines += [
        "",
        "% ===== Signed prevalence gaps (under- and over-detection) =====",
    ]
    with open(latex_dir / "underdetection_delta_pooled.tex", encoding="utf-8") as f:
        main_lines.append(f.read())
    finetuned_delta = latex_dir / "finetuned_pseudolabel_delta_pooled.tex"
    if finetuned_delta.exists():
        main_lines += [
            "",
            "% ===== Fine-tuned on GPT pseudolabels (pooled) =====",
        ]
        with open(finetuned_delta, encoding="utf-8") as f:
            main_lines.append(f.read())
    main_lines += [
        "",
        "% ===== Appendix: by-source prevalence gap =====",
    ]
    with open(latex_dir / "underdetection_delta_by_source.tex", encoding="utf-8") as f:
        main_lines.append(f.read())
    main_lines += [
        "",
        "% ===== Appendix: pooled FNR =====",
    ]
    with open(latex_dir / "underdetection_fnr_pooled.tex", encoding="utf-8") as f:
        main_lines.append(f.read())
    (latex_dir / "main.tex").write_text("\n".join(main_lines) + "\n", encoding="utf-8")
    print(f"Wrote LaTeX tables to {latex_dir}/")


def _print_rankings(detail: pd.DataFrame, model_summary: pd.DataFrame, top: int) -> None:
    print("\n" + "=" * 72)
    print("HIGHEST FALSE POSITIVES (total FP, pooled over 16 categories)")
    print("=" * 72)
    cols = ["source", "model", "shot_type", "total_fp", "total_fn", "macro_fpr", "macro_fnr"]
    print(model_summary.sort_values("total_fp", ascending=False).head(top)[cols].to_string(index=False))

    print("\n" + "=" * 72)
    print("HIGHEST FALSE NEGATIVES (total FN)")
    print("=" * 72)
    print(model_summary.sort_values("total_fn", ascending=False).head(top)[cols].to_string(index=False))

    print("\n" + "=" * 72)
    print("MOST OVER-DETECTED CATEGORIES (mean delta prevalence across models)")
    print("=" * 72)
    over = (
        detail.groupby(["source", "category"], as_index=False)["delta_prevalence_pred_minus_gold"]
        .mean()
        .rename(columns={"delta_prevalence_pred_minus_gold": "mean_delta"})
        .sort_values("mean_delta", ascending=False)
    )
    print(over.head(top).to_string(index=False))

    print("\n" + "=" * 72)
    print("MOST UNDER-DETECTED CATEGORIES (mean delta prevalence across models)")
    print("=" * 72)
    under = over.sort_values("mean_delta", ascending=True)
    print(under.head(top).to_string(index=False))

    print("\n" + "=" * 72)
    print("WORST PER-CATEGORY FPR (model × category, top rows)")
    print("=" * 72)
    show = detail.sort_values("fpr", ascending=False).head(top)[
        ["source", "model", "shot_type", "category", "fpr", "fp", "gold_positive_count"]
    ]
    print(show.to_string(index=False))


def run_analysis(
    *,
    sources: list[str],
    models: list[str],
    shot_types: list[str],
    soft_threshold: float,
    out_dir: Path,
    top: int,
    write_latex: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for source in sources:
        soft = _load_soft_labels(source)
        if soft is None:
            continue
        print(f"\n--- {source} ({len(soft)} gold rows) ---")

        for model in models:
            for shot_type in shot_types:
                pred = _load_predictions(source, model, shot_type)
                if pred is None:
                    print(f"  [missing] {model} {shot_type}")
                    continue
                gold_bin, pred_bin = _align_gold_pred(soft, pred)
                rows = _per_category_rows(
                    gold_bin,
                    pred_bin,
                    source=source,
                    model=model,
                    shot_type=shot_type,
                    soft_threshold=soft_threshold,
                )
                all_rows.extend(rows)
                fp_sum = sum(r["fp"] for r in rows)
                fn_sum = sum(r["fn"] for r in rows)
                print(f"  {model:6} {shot_type:9}  FP={fp_sum:4d}  FN={fn_sum:4d}")

    if not all_rows:
        print("No results — check that gold-subset flag CSVs exist under output/<source>/<model>/.")
        return

    detail = pd.DataFrame(all_rows)
    model_summary = _model_summary_from_detail(detail)
    category_summary = _category_summary_from_detail(detail)

    over_rank = (
        detail.groupby(["source", "category"], as_index=False)["delta_prevalence_pred_minus_gold"]
        .mean()
        .rename(columns={"delta_prevalence_pred_minus_gold": "mean_delta_prevalence"})
        .query("mean_delta_prevalence > 0")
        .sort_values(["source", "mean_delta_prevalence"], ascending=[True, False])
    )
    under_rank = (
        detail.groupby(["source", "category"], as_index=False)["delta_prevalence_pred_minus_gold"]
        .mean()
        .rename(columns={"delta_prevalence_pred_minus_gold": "mean_delta_prevalence"})
        .query("mean_delta_prevalence < 0")
        .sort_values(["source", "mean_delta_prevalence"], ascending=[True, True])
    )

    detail.to_csv(out_dir / "per_model_category.csv", index=False)
    model_summary.to_csv(out_dir / "model_summary.csv", index=False)
    category_summary.to_csv(out_dir / "category_summary.csv", index=False)
    over_rank.to_csv(out_dir / "over_detection_rank.csv", index=False)
    under_rank.to_csv(out_dir / "under_detection_rank.csv", index=False)

    print(f"\nWrote CSVs to {out_dir}/")
    if write_latex:
        generate_emnlp_latex_tables(detail, out_dir, models=models)
    _print_rankings(detail, model_summary, top=top)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gold-subset FP/FN and over/under-detection analysis for prompt LLMs."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=SOURCES,
        choices=SOURCES,
        help="Data sources to include",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=PROMPT_MODELS,
        help="Prompt models (default: all six)",
    )
    parser.add_argument(
        "--shot",
        choices=["all", "zero_shot", "few_shot"],
        default="all",
        help="Limit to zero-shot, few-shot, or both",
    )
    parser.add_argument(
        "--soft-threshold",
        type=float,
        default=0.5,
        help="Binarize soft labels at this threshold (default 0.5)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "output" / "f1" / "gold_error_analysis",
        help="Output directory for CSVs",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=12,
        help="How many top rows to print per ranking table",
    )
    parser.add_argument(
        "--no-latex",
        action="store_true",
        help="Skip EMNLP LaTeX table generation",
    )
    parser.add_argument(
        "--latex-only",
        action="store_true",
        help="Regenerate LaTeX from existing per_model_category.csv (no recompute)",
    )
    parser.add_argument(
        "--finetuned-latex-only",
        action="store_true",
        help="Only generate fine-tuned-on-pseudolabels pooled LaTeX tables",
    )
    args = parser.parse_args()

    shot_types = SHOT_TYPES if args.shot == "all" else [args.shot]

    if args.finetuned_latex_only:
        generate_finetuned_pseudolabel_latex(
            args.out_dir, soft_threshold=args.soft_threshold
        )
        return

    if args.latex_only:
        csv_path = args.out_dir / "per_model_category.csv"
        if not csv_path.exists():
            raise SystemExit(f"--latex-only requires {csv_path}")
        detail = pd.read_csv(csv_path)
        generate_emnlp_latex_tables(
            detail, args.out_dir, models=args.models, soft_threshold=args.soft_threshold
        )
        return

    run_analysis(
        sources=args.sources,
        models=args.models,
        shot_types=shot_types,
        soft_threshold=args.soft_threshold,
        out_dir=args.out_dir,
        top=args.top,
        write_latex=not args.no_latex,
    )


if __name__ == "__main__":
    main()
