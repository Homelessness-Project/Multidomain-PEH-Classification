#!/usr/bin/env python3
"""
Generate val_opt F1 tables from summary CSVs.
Outputs LaTeX tables under output/f1/val_opt by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_ORDER = ["reddit", "news", "meeting_minutes", "x"]
SOURCE_DISPLAY = {
    "reddit": "Reddit",
    "news": "News",
    "meeting_minutes": "Meeting Minutes",
    "x": "X (Twitter)",
}

# Order matters: it controls the ordering in output/f1/val_opt/main.tex
# (and which models are included at all).
LOCAL_MODELS = ["llama", "phi4", "qwen"]  # no Gemma/Gemma3
API_MODELS = ["gemini", "grok", "gpt4"]
TRANSFORMER_METRICS = {
    "bert-base-uncased": "bert_base_uncased_original_metrics.json",
    "roberta-base": "roberta_base_original_metrics.json",
    "modernbert-base": "modernbert_base_original_metrics.json",
}

CATEGORY_ORDER = [
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

CATEGORY_DISPLAY = {
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


def fmt_percent(value: float | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    return f"{value * 100:.2f}"


def load_positive_counts() -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {s: {} for s in SOURCE_ORDER}
    for source in SOURCE_ORDER:
        soft_path = Path("output/annotation/soft_labels") / f"{source}_soft_labels.csv"
        raw_path = Path("annotation") / f"{source}_raw_scores.csv"
        if soft_path.exists():
            df = pd.read_csv(soft_path)
            for cat in CATEGORY_ORDER:
                if cat in df.columns:
                    counts[source][cat] = int((df[cat] >= 0.5).sum())
                elif cat == "racist" and "Racist" in df.columns:
                    counts[source][cat] = int((df["Racist"] >= 0.5).sum())
        elif raw_path.exists():
            df = pd.read_csv(raw_path)
            for cat in CATEGORY_ORDER:
                if cat in df.columns:
                    counts[source][cat] = int((df[cat] >= 2).sum())
                elif cat == "racist" and "Racist" in df.columns:
                    counts[source][cat] = int((df["Racist"] >= 2).sum())
        else:
            for cat in CATEGORY_ORDER:
                counts[source][cat] = 0
    return counts


def filter_summary(
    summary: pd.DataFrame,
    lora_eval_threshold: str,
) -> pd.DataFrame:
    summary = summary.copy()
    summary["eval_threshold"] = summary["eval_threshold"].fillna("")
    keep_zero_few = summary["method"].isin(["zero_shot", "few_shot"])
    keep_lora = (summary["method"] == "lora_gpt") & (summary["eval_threshold"] == lora_eval_threshold)
    keep_ft = summary["method"].isin(["ft_gpt", "ft_gold"])
    return summary[keep_zero_few | keep_lora | keep_ft].copy()


def build_lookup(df: pd.DataFrame) -> dict[tuple[str, str, str], dict]:
    lookup: dict[tuple[str, str, str], dict] = {}
    for _, row in df.iterrows():
        lookup[(row["source"], row["model"], row["method"])] = row.to_dict()
    return lookup


def write_summary_csvs(
    summary: pd.DataFrame,
    per_category: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "zero_few_lora_comparison.csv", index=False)
    per_category.to_csv(output_dir / "zero_few_lora_per_category.csv", index=False)


def write_main_tex(dir_path: Path) -> None:
    """
    Write a main.tex that concatenates all .tex tables in this directory (excluding main.tex).
    This makes it easy to copy/paste many tables at once without any \\input links.
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Main table first, then per-model sections in paper order.
    preferred_names: list[str] = []
    preferred_names.append("detailed_macro_f1_table.tex")
    preferred_names.append("llm_zero_few_summary_table.tex")

    # Individual models: details only (category tables).
    for model in LOCAL_MODELS + API_MODELS:
        preferred_names.append(f"{model}_category_table.tex")

    # Transformer baselines: details only (no macro/micro summary tables).
    for model in TRANSFORMER_METRICS.keys():
        preferred_names.append(f"{model}_category_table.tex")

    lines = ["% Auto-generated. Concatenated tables (no \\input).", ""]
    # Only include the preferred set (in order). This avoids including
    # unwanted legacy files (e.g., gemma) and summary tables the paper
    # does not need (e.g., transformer macro/micro tables).
    for name in preferred_names:
        p = dir_path / name
        if not p.exists():
            continue
        lines.append(f"% ===== {p.name} =====")
        lines.append(p.read_text())
        lines.append("")
    (dir_path / "main.tex").write_text("\n".join(lines))


def load_transformer_metrics() -> tuple[list[dict], list[dict]]:
    rows = []
    per_category_rows = []
    for source in SOURCE_ORDER:
        for model, filename in TRANSFORMER_METRICS.items():
            path = Path("nlp_outputs") / source / filename
            if not path.exists():
                continue
            data = pd.read_json(path, typ="series").to_dict()
            rows.append(
                {
                    "source": source,
                    "model": model,
                    "method": "ft_gold",
                    "macro_f1": float(data.get("macro_f1", np.nan)),
                    "micro_f1": float(data.get("micro_f1", np.nan)),
                    "macro_kappa": np.nan,
                    "n_eval": float(data.get("test_size", np.nan)),
                    "label_source": "gold_labels",
                    "label_threshold": None,
                    "file": str(path),
                    "eval_threshold": "val_opt",
                }
            )
            label_f1 = data.get("label_f1_scores", {})
            for category, f1 in label_f1.items():
                per_category_rows.append(
                    {
                        "source": source,
                        "model": model,
                        "method": "ft_gold",
                        "category": category,
                        "f1": float(f1) if f1 is not None else np.nan,
                        "n_eval": float(data.get("test_size", np.nan)),
                        "label_source": "gold_labels",
                        "label_threshold": None,
                        "file": str(path),
                        "eval_threshold": "val_opt",
                    }
                )
    return rows, per_category_rows


def load_gpt_pseudolabel_results(lora_eval_threshold: str) -> tuple[list[dict], list[dict]]:
    rows = []
    per_category_rows = []
    for path in Path("nlp_outputs").rglob("gpt_pseudolabel*_results.json"):
        with open(path, "r") as f:
            data = json.load(f)
        training = data.get("training_config", {})
        source = training.get("source")
        model = training.get("model")
        if source not in SOURCE_ORDER or not model:
            continue
        eval_threshold = training.get("eval_threshold", "")
        use_lora = bool(training.get("use_lora"))
        method = "lora_gpt" if use_lora else "ft_gpt"
        if method == "lora_gpt" and eval_threshold != lora_eval_threshold:
            continue
        rows.append(
            {
                "source": source,
                "model": model,
                "method": method,
                "macro_f1": float(data.get("macro_f1", np.nan)),
                "micro_f1": float(data.get("micro_f1", np.nan)),
                "macro_kappa": float(data.get("macro_kappa", np.nan)) if data.get("macro_kappa") is not None else np.nan,
                "n_eval": None,
                "label_source": "gold_labels",
                "label_threshold": None,
                "file": str(path),
                "eval_threshold": eval_threshold,
            }
        )
        per_f1 = data.get("per_category_f1", {})
        for category in CATEGORY_ORDER:
            per_category_rows.append(
                {
                    "source": source,
                    "model": model,
                    "method": method,
                    "category": category,
                    "f1": float(per_f1.get(category, np.nan)),
                    "n_eval": None,
                    "label_source": "gold_labels",
                    "label_threshold": None,
                    "file": str(path),
                    "eval_threshold": eval_threshold,
                }
            )
    return rows, per_category_rows


def cleanup_legacy_tables(output_dir: Path) -> None:
    """
    Remove tables we intentionally no longer generate/include:
    - Gemma/Gemma3 tables
    - Transformer macro/micro summary tables (we keep transformer *detail* tables only)
    """
    unwanted_names = {
        "gemma3_macro_micro_table.tex",
        "gemma3_category_table.tex",
        # prompt-model macro/micro tables (category tables only requested)
        *(f"{m}_macro_micro_table.tex" for m in (LOCAL_MODELS + API_MODELS)),
        # transformer summaries (details only requested)
        *(f"{m}_macro_micro_table.tex" for m in TRANSFORMER_METRICS.keys()),
    }
    for name in unwanted_names:
        p = output_dir / name
        if p.exists():
            p.unlink()


def create_detailed_table(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    lookup = build_lookup(summary)

    columns = [
        ("llama", "zero_shot"),
        ("llama", "lora_gpt"),
        ("phi4", "zero_shot"),
        ("phi4", "lora_gpt"),
        ("qwen", "zero_shot"),
        ("qwen", "lora_gpt"),
        ("gemini", "zero_shot"),
        ("grok", "zero_shot"),
        ("gpt4", "zero_shot"),
        ("bert-base-uncased", "ft_gold"),
        ("bert-base-uncased", "ft_gpt"),
        ("roberta-base", "ft_gold"),
        ("roberta-base", "ft_gpt"),
        ("modernbert-base", "ft_gold"),
        ("modernbert-base", "ft_gpt"),
    ]

    def fetch(source: str, model: str, method: str, metric: str) -> tuple[str, float | None]:
        row = lookup.get((source, model, method))
        if not row:
            return "", None
        raw = row.get(metric)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return "", None
        return fmt_percent(raw), float(raw) * 100.0

    def bold_best(cells: list[tuple[str, float | None]]) -> list[str]:
        values = [v for _, v in cells if v is not None]
        if not values:
            return [s for s, _ in cells]
        best = max(values)
        out: list[str] = []
        for s, v in cells:
            if v is not None and abs(v - best) < 1e-9 and s:
                out.append(f"\\textbf{{{s}}}")
            else:
                out.append(s)
        return out

    rows_macro = {}
    rows_micro = {}
    for source in SOURCE_ORDER:
        rows_macro[source] = [fetch(source, m, method, "macro_f1") for m, method in columns]
        rows_micro[source] = [fetch(source, m, method, "micro_f1") for m, method in columns]

    avg_macro = []
    avg_micro = []
    avg_macro_vals: list[float | None] = []
    avg_micro_vals: list[float | None] = []
    for i in range(len(columns)):
        values_macro = []
        values_micro = []
        for source in SOURCE_ORDER:
            s_macro, v_macro = rows_macro[source][i]
            s_micro, v_micro = rows_micro[source][i]
            if v_macro is not None:
                values_macro.append(v_macro)
            if v_micro is not None:
                values_micro.append(v_micro)
        avg_macro_vals.append(float(np.mean(values_macro)) if values_macro else None)
        avg_micro_vals.append(float(np.mean(values_micro)) if values_micro else None)
        avg_macro.append(f"{avg_macro_vals[-1]:.2f}" if avg_macro_vals[-1] is not None else "")
        avg_micro.append(f"{avg_micro_vals[-1]:.2f}" if avg_micro_vals[-1] is not None else "")

    lines = [
        r"\begin{table*}[h!]",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lccccccccccccccc}",
        r"\toprule",
        r"Data Source & \multicolumn{2}{c}{LLaMA} & \multicolumn{2}{c}{Phi-4} & \multicolumn{2}{c}{Qwen} & Gemini & Grok & GPT-4 & \multicolumn{2}{c}{BERT}& \multicolumn{2}{c}{RoBERTa}& \multicolumn{2}{c}{ModernBERT}\\",
        r" & Zero & Finetuned& Zero & Finetuned& Zero & Finetuned& Zero & Zero & Zero & Gold & GPT & Gold & GPT & Gold & GPT\\",
        r"& & (GPT)& & (GPT)& & (GPT)& & & & (Gold)&(GPT)& (Gold)&(GPT)& (Gold)&(GPT)\\",
        r"\midrule",
    ]

    for source in SOURCE_ORDER:
        lines.append(
            f"{SOURCE_DISPLAY[source]} (Macro) & " + " & ".join(bold_best(rows_macro[source])) + r" \\"
        )
        lines.append(
            f"{SOURCE_DISPLAY[source]} (Micro) & " + " & ".join(bold_best(rows_micro[source])) + r" \\"
        )

    lines.append("Avg (Macro) & " + " & ".join(bold_best(list(zip(avg_macro, avg_macro_vals)))) + r" \\")
    lines.append("Avg (Micro) & " + " & ".join(bold_best(list(zip(avg_micro, avg_micro_vals)))) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\centering",
            r"\caption{Validation-optimized Macro and Micro F1 Scores for All Models by Data Source}",
            r"\label{tab:detailed_f1_scores_reordered}",
            r"\end{table*}",
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "detailed_macro_f1_table.tex").write_text("\n".join(lines))


def create_llm_zero_few_table(summary: pd.DataFrame, output_dir: Path) -> None:
    """
    Val-opt macro/micro F1 summary for the 6 LLMs (zero-shot vs few-shot).
    """
    lookup = build_lookup(summary)

    llms = ["llama", "phi4", "qwen", "gemini", "grok", "gpt4"]
    columns: list[tuple[str, str]] = []
    for m in llms:
        columns.append((m, "zero_shot"))
        columns.append((m, "few_shot"))

    def fetch(source: str, model: str, method: str, metric: str) -> tuple[str, float | None]:
        row = lookup.get((source, model, method))
        if not row:
            return "--", None
        raw = row.get(metric)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return "--", None
        return fmt_percent(raw), float(raw) * 100.0

    def bold_best(cells: list[tuple[str, float | None]]) -> list[str]:
        values = [v for _, v in cells if v is not None]
        if not values:
            return [s for s, _ in cells]
        best = max(values)
        out: list[str] = []
        for s, v in cells:
            if v is not None and abs(v - best) < 1e-9 and s not in ("", "--"):
                out.append(f"\\textbf{{{s}}}")
            else:
                out.append(s)
        return out

    macro_cells_by_source: dict[str, list[tuple[str, float | None]]] = {}
    micro_cells_by_source: dict[str, list[tuple[str, float | None]]] = {}
    for source in SOURCE_ORDER:
        macro_cells_by_source[source] = [fetch(source, m, meth, "macro_f1") for m, meth in columns]
        micro_cells_by_source[source] = [fetch(source, m, meth, "micro_f1") for m, meth in columns]

    avg_macro: list[tuple[str, float | None]] = []
    avg_micro: list[tuple[str, float | None]] = []
    for i in range(len(columns)):
        vals_macro = [
            macro_cells_by_source[s][i][1] for s in SOURCE_ORDER if macro_cells_by_source[s][i][1] is not None
        ]
        vals_micro = [
            micro_cells_by_source[s][i][1] for s in SOURCE_ORDER if micro_cells_by_source[s][i][1] is not None
        ]
        avg_m = float(np.mean(vals_macro)) if vals_macro else None
        avg_u = float(np.mean(vals_micro)) if vals_micro else None
        avg_macro.append((f"{avg_m:.2f}" if avg_m is not None else "--", avg_m))
        avg_micro.append((f"{avg_u:.2f}" if avg_u is not None else "--", avg_u))

    # Build headers: group by model, each with (Zero, Few)
    header1 = r"Data Source"
    header1 += " & " + " & ".join([rf"\multicolumn{{2}}{{c}}{{{m.upper()}}}" for m in llms]) + r" \\"
    header2 = r""
    header2 += " & " + " & ".join(["Zero & Few"] * len(llms)) + r" \\"

    lines = [
        r"\begin{table*}[h!]",
        r"\centering",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l" + ("c" * len(columns)) + r"}",
        r"\toprule",
        header1,
        # underline each model group (2 cols per model)
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}",
        header2,
        r"\midrule",
    ]

    for source in SOURCE_ORDER:
        lines.append(rf"\rowcolor{{gray!10}} {SOURCE_DISPLAY[source]} (Macro) & " + " & ".join(bold_best(macro_cells_by_source[source])) + r" \\")
        lines.append(f"{SOURCE_DISPLAY[source]} (Micro) & " + " & ".join(bold_best(micro_cells_by_source[source])) + r" \\")
        lines.append(r"\addlinespace[0.35em]")

    lines.append(r"\midrule")
    lines.append(rf"\rowcolor{{gray!10}} Avg (Macro) & " + " & ".join(bold_best(avg_macro)) + r" \\")
    lines.append("Avg (Micro) & " + " & ".join(bold_best(avg_micro)) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\caption{Validation-optimized Macro and Micro F1 Scores for LLMs (Zero-shot vs Few-shot)}",
        r"\label{tab:llm_zero_few_valopt}",
        r"\end{table*}",
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "llm_zero_few_summary_table.tex").write_text("\n".join(lines))


def create_macro_micro_tables(
    summary: pd.DataFrame,
    output_dir: Path,
    model: str,
    method_pair: tuple[str, str],
) -> None:
    lookup = build_lookup(summary)
    first_method, second_method = method_pair

    macro_row = ["Macro F1"]
    micro_row = ["Micro F1"]
    for source in SOURCE_ORDER:
        row_first = lookup.get((source, model, first_method))
        row_second = lookup.get((source, model, second_method))

        macro_first = row_first["macro_f1"] if row_first is not None else None
        macro_second = row_second["macro_f1"] if row_second is not None else None
        micro_first = row_first["micro_f1"] if row_first is not None else None
        micro_second = row_second["micro_f1"] if row_second is not None else None

        macro_first_str = fmt_percent(macro_first) if macro_first is not None else "--"
        macro_second_str = fmt_percent(macro_second) if macro_second is not None else "--"
        micro_first_str = fmt_percent(micro_first) if micro_first is not None else "--"
        micro_second_str = fmt_percent(micro_second) if micro_second is not None else "--"

        if macro_first is not None and macro_second is not None:
            if macro_first > macro_second:
                macro_first_str = f"\\textbf{{{macro_first_str}}}"
            elif macro_second > macro_first:
                macro_second_str = f"\\textbf{{{macro_second_str}}}"
        if micro_first is not None and micro_second is not None:
            if micro_first > micro_second:
                micro_first_str = f"\\textbf{{{micro_first_str}}}"
            elif micro_second > micro_first:
                micro_second_str = f"\\textbf{{{micro_second_str}}}"

        macro_row.extend([macro_first_str, macro_second_str])
        micro_row.extend([micro_first_str, micro_second_str])

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{l *{8}{c}}",
        r"\toprule",
        r"Data Source & \multicolumn{2}{c}{Reddit} & \multicolumn{2}{c}{News} & \multicolumn{2}{c}{Meeting Minutes} & \multicolumn{2}{c}{X (Twitter)} \\",
        rf"& {method_pair[0].replace('_', ' ').title()} & {method_pair[1].replace('_', ' ').title()} & "
        rf"{method_pair[0].replace('_', ' ').title()} & {method_pair[1].replace('_', ' ').title()} & "
        rf"{method_pair[0].replace('_', ' ').title()} & {method_pair[1].replace('_', ' ').title()} & "
        rf"{method_pair[0].replace('_', ' ').title()} & {method_pair[1].replace('_', ' ').title()} \\",
        r"\midrule",
        " & ".join(macro_row) + r" \\",
        " & ".join(micro_row) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        rf"\centering\caption{{Validation-optimized Macro and Micro F1 Scores for {model.upper()} Model}}",
        rf"\label{{tab:{model}_macro_micro}}",
        r"\end{table}",
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{model}_macro_micro_table.tex").write_text("\n".join(lines))


def create_category_table(
    per_category: pd.DataFrame,
    output_dir: Path,
    model: str,
    method_pair: tuple[str, str],
    positive_counts: dict[str, dict[str, int]],
) -> None:
    df = per_category[(per_category["model"] == model) & (per_category["method"].isin(method_pair))].copy()
    if df.empty:
        return

    lookup = {}
    for _, row in df.iterrows():
        key = (row["category"], row["source"], row["method"])
        lookup[key] = row["f1"]

    header = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\begin{tabular}{l *{8}{c}}",
        r"\toprule",
        r"Category & \multicolumn{2}{c}{Reddit} & \multicolumn{2}{c}{News} & \multicolumn{2}{c}{Meeting Minutes} & \multicolumn{2}{c}{X (Twitter)} \\",
        rf"& {method_pair[0].replace('_', ' ').title()} & {method_pair[1].replace('_', ' ').title()} & "
        rf"{method_pair[0].replace('_', ' ').title()} & {method_pair[1].replace('_', ' ').title()} & "
        rf"{method_pair[0].replace('_', ' ').title()} & {method_pair[1].replace('_', ' ').title()} & "
        rf"{method_pair[0].replace('_', ' ').title()} & {method_pair[1].replace('_', ' ').title()} \\",
        r"\midrule",
    ]

    lines = []
    star_used = False
    for category in CATEGORY_ORDER:
        row = [CATEGORY_DISPLAY.get(category, category.title())]
        has_value = False
        for source in SOURCE_ORDER:
            pos_count = positive_counts.get(source, {}).get(category, 0)
            first_value = lookup.get((category, source, method_pair[0]))
            second_value = lookup.get((category, source, method_pair[1]))

            first_str = fmt_percent(first_value) if first_value is not None else "--"
            second_str = fmt_percent(second_value) if second_value is not None else "--"

            if pos_count < 5:
                if first_value is not None:
                    first_str = f"{first_str}*"
                    star_used = True
                if second_value is not None:
                    second_str = f"{second_str}*"
                    star_used = True

            if first_value is not None and second_value is not None:
                if first_value > second_value:
                    first_str = f"\\textbf{{{first_str}}}"
                elif second_value > first_value:
                    second_str = f"\\textbf{{{second_str}}}"

            row.extend([first_str, second_str])
            if first_value is not None or second_value is not None:
                has_value = True
        if has_value:
            lines.append(" & ".join(row) + r" \\")

    caption = rf"\centering\caption{{Validation-optimized Category-wise F1 Scores for {model.upper()} Model}}"
    if star_used:
        caption = (
            rf"\centering\caption{{Validation-optimized Category-wise F1 Scores for {model.upper()} Model "
            r"(* indicates $<$5 positive examples in gold labels; interpret with caution.)}"
        )

    footer = [
        r"\bottomrule",
        r"\end{tabular}",
        caption,
        rf"\label{{tab:{model}_category_breakdown}}",
        r"\end{table*}",
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{model}_category_table.tex").write_text("\n".join(header + lines + footer))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate val_opt F1 tables from summary CSVs.")
    parser.add_argument(
        "--summary",
        default="output/summary/zero_few_lora_comparison.csv",
        help="Path to zero/few/lora comparison CSV.",
    )
    parser.add_argument(
        "--per-category",
        default="output/summary/zero_few_lora_per_category.csv",
        help="Path to per-category comparison CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/f1/val_opt",
        help="Output directory for val_opt tables.",
    )
    parser.add_argument(
        "--lora-eval-threshold",
        default="val_opt",
        help="Which eval_threshold to keep for LoRA rows (default: val_opt).",
    )
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    per_category = pd.read_csv(args.per_category)

    transformer_rows, transformer_per_category = load_transformer_metrics()
    if transformer_rows:
        transformer_df = pd.DataFrame(transformer_rows).dropna(axis=1, how="all")
        summary = pd.concat([summary, transformer_df], ignore_index=True)
    if transformer_per_category:
        transformer_cat_df = pd.DataFrame(transformer_per_category).dropna(axis=1, how="all")
        per_category = pd.concat([per_category, transformer_cat_df], ignore_index=True)

    gpt_rows, gpt_per_category = load_gpt_pseudolabel_results(args.lora_eval_threshold)
    if gpt_rows:
        gpt_df = pd.DataFrame(gpt_rows).dropna(axis=1, how="all")
        summary = pd.concat([summary, gpt_df], ignore_index=True)
    if gpt_per_category:
        gpt_cat_df = pd.DataFrame(gpt_per_category).dropna(axis=1, how="all")
        per_category = pd.concat([per_category, gpt_cat_df], ignore_index=True)

    # Drop duplicates keeping latest file path
    summary = summary.drop_duplicates(subset=["source", "model", "method", "eval_threshold"], keep="last")
    per_category = per_category.drop_duplicates(
        subset=["source", "model", "method", "category", "eval_threshold"],
        keep="last",
    )

    summary = filter_summary(summary, args.lora_eval_threshold)
    per_category = per_category[
        per_category["method"].isin(["zero_shot", "few_shot", "lora_gpt", "ft_gpt", "ft_gold"])
    ].copy()
    per_category["eval_threshold"] = per_category["eval_threshold"].fillna("")
    per_category = per_category[
        (per_category["method"].isin(["zero_shot", "few_shot", "ft_gpt", "ft_gold"]))
        | ((per_category["method"] == "lora_gpt") & (per_category["eval_threshold"] == args.lora_eval_threshold))
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_legacy_tables(output_dir)
    write_summary_csvs(summary, per_category, output_dir)
    create_detailed_table(summary, output_dir)
    create_llm_zero_few_table(summary, output_dir)
    positive_counts = load_positive_counts()

    # Per-model tables (ordered for paper).
    for model in LOCAL_MODELS:
        create_category_table(per_category, output_dir, model, ("zero_shot", "lora_gpt"), positive_counts)

    for model in API_MODELS:
        create_category_table(per_category, output_dir, model, ("zero_shot", "few_shot"), positive_counts)

    # Transformer baselines: include details only (no macro/micro summary tables).
    for model in TRANSFORMER_METRICS.keys():
        create_category_table(per_category, output_dir, model, ("ft_gold", "ft_gpt"), positive_counts)

    write_main_tex(output_dir)


if __name__ == "__main__":
    main()
