#!/usr/bin/env python3
"""
Comprehensive pseudo-label bias audit: GPT vs human (gold subset) with slices.

Computes per-label prevalence, full confusion counts, TPR/FPR/FNR, and micro-averaged
P/R/F1 stratified by source and city cluster (large vs small vs other), using the same
city lists as scripts/bias_temporal_analysis.py.

Human reference: soft labels binarized at --soft-threshold.
Predicted: GPT-4 flags on the gold subset (0/1).

Usage (repo root):
    python scripts/comprehensive_pseudo_label_bias_audit.py

Outputs (default output/annotation/comprehensive_bias_audit/):
    summary_by_source.csv
    per_label_metrics.csv          # includes tp, fp, fn, tn, tpr, fpr, fnr, specificity
    per_label_by_city_cluster.csv  # same metrics by source × city_cluster × label
    slice_micro_metrics.csv        # micro P/R/F1 by source × city_cluster
    prevalence_by_slice.csv        # human vs GPT positive rate by slice × label
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

LARGE_CITIES = {"san francisco", "portland", "buffalo", "baltimore", "el paso"}
SMALL_CITIES = {"kalamazoo", "south bend", "rockford", "scranton", "fayetteville"}


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


def _city_series(raw: pd.DataFrame) -> pd.Series:
    for c in ("City", "city"):
        if c in raw.columns:
            return raw[c].astype(str).str.lower().str.strip()
    return pd.Series([""] * len(raw), index=raw.index)


def _city_cluster(city_lower: str) -> str:
    if not city_lower or city_lower in ("nan", "none"):
        return "unknown"
    if city_lower in LARGE_CITIES:
        return "large_city_cluster"
    if city_lower in SMALL_CITIES:
        return "small_city_cluster"
    return "other_city"


def _load_human_frame(source: str) -> tuple[pd.DataFrame, int]:
    raw_path = REPO_ROOT / RAW_FILES[source]
    soft_path = REPO_ROOT / SOFT_FILES[source]
    raw = pd.read_csv(raw_path, low_memory=False)
    soft = pd.read_csv(soft_path, low_memory=False)
    if len(raw) != len(soft):
        raise ValueError(
            f"{source}: raw_scores ({len(raw)}) and soft_labels ({len(soft)}) length mismatch"
        )
    tc = _text_column(raw)
    city_s = _city_series(raw)
    out = pd.DataFrame(
        {
            "Comment": raw[tc].astype(str).str.strip(),
            "city_normalized": city_s,
        }
    )
    out["city_cluster"] = out["city_normalized"].map(_city_cluster)
    for col in LABEL_COLUMNS:
        if col not in soft.columns:
            raise ValueError(f"{source}: soft labels missing column {col!r}")
        out[col] = pd.to_numeric(soft[col], errors="coerce").fillna(0.0)
    n_raw = len(out)
    out = out.drop_duplicates(subset=["Comment", "city_normalized"], keep="first")
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
    if "City" in gpt.columns:
        gpt["city_normalized"] = gpt["City"].astype(str).str.lower().str.strip()
    else:
        gpt["city_normalized"] = ""
    gpt["city_cluster"] = gpt["city_normalized"].map(_city_cluster)
    gpt = gpt.drop_duplicates(subset=["Comment", "city_normalized"], keep="first")
    for gcol, scol in GPT_TO_SOFT.items():
        if gcol not in gpt.columns:
            raise ValueError(f"{path} missing GPT column {gcol!r}")
        gpt[scol] = pd.to_numeric(gpt[gcol], errors="coerce").fillna(0).astype(int).clip(0, 1)
    return gpt[["Comment", "city_normalized", "city_cluster"] + LABEL_COLUMNS]


def _binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    yt = y_true.astype(int)
    yp = y_pred.astype(int)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _rates_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    pos_h = tp + fn
    neg_h = fp + tn
    tpr = tp / pos_h if pos_h > 0 else 0.0
    fpr = fp / neg_h if neg_h > 0 else 0.0
    fnr = fn / pos_h if pos_h > 0 else 0.0
    spec = tn / neg_h if neg_h > 0 else 0.0
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tpr": tpr,
        "fpr": fpr,
        "fnr": fnr,
        "specificity": spec,
    }


def _micro_prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    c = _binary_counts(y_true, y_pred)
    r = _rates_from_counts(c["tp"], c["fp"], c["fn"], c["tn"])
    return r["precision"], r["recall"], r["f1"]


def _merge_human_gpt(source: str) -> tuple[pd.DataFrame, dict]:
    human, n_raw = _load_human_frame(source)
    gpt = _load_gpt_flags(source)
    # Avoid duplicate city_cluster columns on merge (keep human-side cluster).
    gpt = gpt.drop(columns=["city_cluster"], errors="ignore")
    gpt_ren = gpt.rename(columns={c: f"{c}_gpt" for c in LABEL_COLUMNS})
    merged = human.merge(
        gpt_ren,
        on=["Comment", "city_normalized"],
        how="inner",
    )
    meta = {
        "source": source,
        "n_raw_human_rows": n_raw,
        "n_human_unique": len(human),
        "n_gpt_unique": len(gpt),
        "n_merged": len(merged),
        "n_dup_raw": n_raw - len(human),
    }
    return merged, meta


def _per_label_block(
    merged: pd.DataFrame,
    soft_threshold: float,
    rare_positive_max: int,
    slice_key: tuple[str, ...],
    slice_vals: tuple,
) -> tuple[list[dict], list[int], list[int]]:
    """Rows for per_label table; also return flattened true/pred for micro over slice."""
    rows = []
    all_true: list[int] = []
    all_pred: list[int] = []
    f1_list: list[float] = []

    mask = np.ones(len(merged), dtype=bool)
    for k, v in zip(slice_key, slice_vals):
        mask &= merged[k].values == v
    sub = merged.loc[mask]
    n_sub = len(sub)
    if n_sub == 0:
        return rows, all_true, all_pred

    for lab in LABEL_COLUMNS:
        y_true = (
            pd.to_numeric(sub[lab], errors="coerce").fillna(0) >= soft_threshold
        ).astype(int).values
        y_pred = (
            pd.to_numeric(sub[f"{lab}_gpt"], errors="coerce")
            .fillna(0)
            .astype(int)
            .clip(0, 1)
            .values
        )
        c = _binary_counts(y_true, y_pred)
        r = _rates_from_counts(c["tp"], c["fp"], c["fn"], c["tn"])
        prev_h = float(y_true.mean())
        prev_g = float(y_pred.mean())
        pos_h = int(y_true.sum())
        unstable = pos_h <= rare_positive_max

        row = {
            "n_slice": n_sub,
            "label": lab,
            "human_positive_count": pos_h,
            "gpt_positive_count": int(y_pred.sum()),
            "prevalence_human": round(prev_h, 6),
            "prevalence_gpt": round(prev_g, 6),
            "delta_prevalence_gpt_minus_human": round(prev_g - prev_h, 6),
            "tp": c["tp"],
            "fp": c["fp"],
            "fn": c["fn"],
            "tn": c["tn"],
            "precision_gpt_vs_human": round(r["precision"], 6),
            "recall_gpt_vs_human": round(r["recall"], 6),
            "f1_gpt_vs_human": round(r["f1"], 6),
            "tpr": round(r["tpr"], 6),
            "fpr": round(r["fpr"], 6),
            "fnr": round(r["fnr"], 6),
            "specificity": round(r["specificity"], 6),
            "low_human_positive_support": unstable,
        }
        for k, v in zip(slice_key, slice_vals):
            row[k] = v
        rows.append(row)
        f1_list.append(r["f1"])
        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())

    return rows, all_true, all_pred


def run_comprehensive_audit(
    sources: list[str],
    soft_threshold: float,
    rare_positive_max: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    per_label_rows = []
    by_cluster_rows = []
    slice_micro_rows = []
    prevalence_rows = []

    for source in sources:
        merged, meta = _merge_human_gpt(source)
        summary_row = dict(meta)
        if len(merged) == 0:
            summary_rows.append(summary_row)
            continue

        # Full source: per label
        block, all_true, all_pred = _per_label_block(
            merged, soft_threshold, rare_positive_max, (), ()
        )
        for r in block:
            r["source"] = source
            per_label_rows.append(r)

        micro_p, micro_r, micro_f1 = _micro_prf(
            np.array(all_true, dtype=int),
            np.array(all_pred, dtype=int),
        )
        f1s = [x["f1_gpt_vs_human"] for x in block]
        dprev = [abs(x["delta_prevalence_gpt_minus_human"]) for x in block]
        summary_row.update(
            {
                "micro_precision": round(float(micro_p), 6),
                "micro_recall": round(float(micro_r), 6),
                "micro_f1": round(float(micro_f1), 6),
                "macro_f1_mean_per_label": round(float(np.mean(f1s)), 6),
                "mean_abs_delta_prevalence": round(float(np.mean(dprev)), 6),
            }
        )
        summary_rows.append(summary_row)

        # By city cluster within source
        for cluster in sorted(merged["city_cluster"].unique()):
            b, at, ap = _per_label_block(
                merged,
                soft_threshold,
                rare_positive_max,
                ("city_cluster",),
                (cluster,),
            )
            for r in b:
                r["source"] = source
                by_cluster_rows.append(r)
            if len(at) > 0:
                mp, mr, mf = _micro_prf(np.array(at, dtype=int), np.array(ap, dtype=int))
                slice_micro_rows.append(
                    {
                        "source": source,
                        "city_cluster": cluster,
                        "micro_precision": round(mp, 6),
                        "micro_recall": round(mr, 6),
                        "micro_f1": round(mf, 6),
                        "n_rows": int(merged["city_cluster"].eq(cluster).sum()),
                    }
                )
            for r in b:
                prevalence_rows.append(
                    {
                        "source": source,
                        "city_cluster": cluster,
                        "label": r["label"],
                        "n_slice": r["n_slice"],
                        "prevalence_human": r["prevalence_human"],
                        "prevalence_gpt": r["prevalence_gpt"],
                        "delta_prevalence": r["delta_prevalence_gpt_minus_human"],
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    per_label_df = pd.DataFrame(per_label_rows)
    by_cluster_df = pd.DataFrame(by_cluster_rows)
    slice_micro_df = pd.DataFrame(slice_micro_rows)
    prev_df = pd.DataFrame(prevalence_rows)

    if not per_label_df.empty:
        per_label_df["human_positive_rare_cap"] = rare_positive_max
        per_label_df["soft_threshold"] = soft_threshold
    if not by_cluster_df.empty:
        by_cluster_df["human_positive_rare_cap"] = rare_positive_max
        by_cluster_df["soft_threshold"] = soft_threshold

    summary_df.to_csv(out_dir / "summary_by_source.csv", index=False)
    per_label_df.to_csv(out_dir / "per_label_metrics.csv", index=False)
    by_cluster_df.to_csv(out_dir / "per_label_by_city_cluster.csv", index=False)
    slice_micro_df.to_csv(out_dir / "slice_micro_metrics.csv", index=False)
    prev_df.to_csv(out_dir / "prevalence_by_slice.csv", index=False)

    print(f"Wrote {out_dir / 'summary_by_source.csv'}")
    print(f"Wrote {out_dir / 'per_label_metrics.csv'}")
    print(f"Wrote {out_dir / 'per_label_by_city_cluster.csv'}")
    print(f"Wrote {out_dir / 'slice_micro_metrics.csv'}")
    print(f"Wrote {out_dir / 'prevalence_by_slice.csv'}")
    print()
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
        help="Flag labels with human positive count <= this within each slice (default 5)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "output/annotation/comprehensive_bias_audit",
        help="Output directory for CSVs",
    )
    args = parser.parse_args()

    run_comprehensive_audit(
        sources=list(args.source),
        soft_threshold=args.soft_threshold,
        rare_positive_max=args.rare_positive_max,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
