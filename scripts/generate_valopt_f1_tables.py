#!/usr/bin/env python3
"""
Generate val_opt F1 tables from summary CSVs.
Outputs LaTeX tables under output/f1/val_opt by default.
"""

from __future__ import annotations

import argparse
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

LOCAL_MODELS = ["llama", "qwen", "phi4", "gemma3"]
API_MODELS = ["gpt4", "gemini", "grok"]
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


def filter_summary(
    summary: pd.DataFrame,
    lora_eval_threshold: str,
) -> pd.DataFrame:
    summary = summary.copy()
    summary["eval_threshold"] = summary["eval_threshold"].fillna("")
    keep_zero_few = summary["method"].isin(["zero_shot", "few_shot"])
    keep_lora = (summary["method"] == "lora_gpt") & (summary["eval_threshold"] == lora_eval_threshold)
    keep_ft = summary["method"] == "ft_gpt"
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
        ("bert-base-uncased", "lora_gpt"),
        ("roberta-base", "ft_gold"),
        ("roberta-base", "lora_gpt"),
        ("modernbert-base", "ft_gold"),
        ("modernbert-base", "lora_gpt"),
    ]

    def fetch(source: str, model: str, method: str, metric: str) -> str:
        row = lookup.get((source, model, method))
        if not row:
            return ""
        return fmt_percent(row.get(metric))

    rows_macro = {}
    rows_micro = {}
    for source in SOURCE_ORDER:
        rows_macro[source] = [fetch(source, m, method, "macro_f1") for m, method in columns]
        rows_micro[source] = [fetch(source, m, method, "micro_f1") for m, method in columns]

    avg_macro = []
    avg_micro = []
    for i in range(len(columns)):
        values_macro = []
        values_micro = []
        for source in SOURCE_ORDER:
            v_macro = rows_macro[source][i]
            v_micro = rows_micro[source][i]
            if v_macro:
                values_macro.append(float(v_macro))
            if v_micro:
                values_micro.append(float(v_micro))
        avg_macro.append(f"{np.mean(values_macro):.2f}" if values_macro else "")
        avg_micro.append(f"{np.mean(values_micro):.2f}" if values_micro else "")

    lines = [
        r"\begin{table*}[h!]",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lccccccccccccclclcl}",
        r"\toprule",
        r"Data Source & \multicolumn{2}{c}{LLaMA} & \multicolumn{2}{c}{Phi-4} & \multicolumn{2}{c}{Qwen} & Gemini & & Grok & & GPT-4 & & \multicolumn{2}{c}{BERT}& \multicolumn{2}{c}{RoBERTa}& \multicolumn{2}{c}{ModernBERT}\\",
        r" & Zero & Finetuned& Zero & Finetuned& Zero & Finetuned& Zero & & Zero & & Zero & & Finetuned& Finetuned& Finetuned& Finetuned& Finetuned&Finetuned\\",
        r"& & (GPT)& & (GPT)& & (GPT)& & & & & & & (Gold)&(GPT)& (Gold)&(GPT)& (Gold)&(GPT)\\",
        r"\midrule",
    ]

    for source in SOURCE_ORDER:
        lines.append(f"{SOURCE_DISPLAY[source]} (Macro) & " + " & ".join(rows_macro[source]) + r" \\")
        lines.append(f"{SOURCE_DISPLAY[source]} (Micro) & " + " & ".join(rows_micro[source]) + r" \\")

    lines.append("Avg (Macro) & " + " & ".join(avg_macro) + r" \\")
    lines.append("Avg (Micro) & " + " & ".join(avg_micro) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\centering",
            r"\caption{Macro and Micro F1 Scores for All Models by Data Source}",
            r"\label{tab:detailed_f1_scores_reordered}",
            r"\end{table*}",
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "detailed_macro_f1_table.tex").write_text("\n".join(lines))


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
        for method in [first_method, second_method]:
            row = lookup.get((source, model, method))
            macro_row.append(fmt_percent(row["macro_f1"]) if row is not None else "--")
            micro_row.append(fmt_percent(row["micro_f1"]) if row is not None else "--")

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
        rf"\centering\caption{{Macro and Micro F1 Scores for {model.upper()} Model}}",
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
    for category in CATEGORY_ORDER:
        row = [CATEGORY_DISPLAY.get(category, category.title())]
        for source in SOURCE_ORDER:
            for method in method_pair:
                value = lookup.get((category, source, method))
                row.append(fmt_percent(value) if value is not None else "--")
        lines.append(" & ".join(row) + r" \\")

    footer = [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\centering\caption{{Category-wise F1 Scores for {model.upper()} Model}}",
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
        summary = pd.concat([summary, pd.DataFrame(transformer_rows)], ignore_index=True)
    if transformer_per_category:
        per_category = pd.concat([per_category, pd.DataFrame(transformer_per_category)], ignore_index=True)

    summary = filter_summary(summary, args.lora_eval_threshold)
    per_category = per_category[
        per_category["method"].isin(["zero_shot", "few_shot", "lora_gpt", "ft_gpt"])
    ].copy()
    per_category["eval_threshold"] = per_category["eval_threshold"].fillna("")
    per_category = per_category[
        (per_category["method"].isin(["zero_shot", "few_shot", "ft_gpt"]))
        | ((per_category["method"] == "lora_gpt") & (per_category["eval_threshold"] == args.lora_eval_threshold))
    ]

    output_dir = Path(args.output_dir)
    write_summary_csvs(summary, per_category, output_dir)
    create_detailed_table(summary, output_dir)

    # Per-model tables
    for model in API_MODELS:
        create_macro_micro_tables(summary, output_dir, model, ("zero_shot", "few_shot"))
        create_category_table(per_category, output_dir, model, ("zero_shot", "few_shot"))

    for model in LOCAL_MODELS:
        create_macro_micro_tables(summary, output_dir, model, ("zero_shot", "lora_gpt"))
        create_category_table(per_category, output_dir, model, ("zero_shot", "lora_gpt"))


if __name__ == "__main__":
    main()
