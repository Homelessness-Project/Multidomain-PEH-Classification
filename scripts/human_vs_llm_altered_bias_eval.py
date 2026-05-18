#!/usr/bin/env python3
"""
Compare "human text" vs "LLM-altered human text" for NIMBY framing.

Idea:
- Use your existing gold-standard texts (human-written / human-collected).
- Create an LLM-altered version of each text (e.g., paraphrase or "neutral rewrite").
- Hide provenance (original vs altered) in a blinded annotation CSV so raters can't tell which is which.
- Optionally attach existing human gold labels (from soft labels threshold) for NIMBY on the original text.

Outputs:
- blinded_items.csv: one row per item shown to raters (mixed original + altered; randomized order)
- key.csv: mapping item_id -> (pair_id, variant=original|altered)
- pairs.csv: one row per pair with original/altered text and any attached gold NIMBY label (if available)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from utils import call_api_llm, extract_field


def _load_gold_texts(source: str) -> Tuple[pd.DataFrame, str]:
    path_map = {
        "reddit": ("gold_standard/sampled_reddit_comments.csv", "Comment"),
        "x": ("gold_standard/sampled_twitter_posts.csv", "Deidentified_text"),
        "news": ("gold_standard/sampled_lexisnexis_news.csv", "Deidentified_paragraph_text"),
        "meeting_minutes": ("gold_standard/sampled_meeting_minutes.csv", "Deidentified_paragraph"),
    }
    path, col = path_map[source]
    df = pd.read_csv(path)
    if col not in df.columns:
        raise ValueError(f"Missing {col} in {path}")
    df[col] = df[col].fillna("").astype(str)
    df = df[df[col].str.len() > 0].reset_index(drop=True)
    return df, col


def _load_soft_labels(source: str) -> Optional[pd.DataFrame]:
    path = f"output/annotation/soft_labels/{source}_soft_labels.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df


def _stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _trim(text: str, max_chars: int) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if len(t) <= max_chars:
        return t
    t = t[:max_chars]
    if " " in t:
        t = t.rsplit(" ", 1)[0]
    return t + "…"


def _alter_prompt(text: str, source: str, mode: str) -> str:
    if mode == "paraphrase":
        return f"""Paraphrase the following {source} text about homelessness.

Constraints:
- Preserve meaning and stance as much as possible.
- Keep length roughly similar.
- Do not add new facts, locations, or numbers.
- Keep any bracketed deidentification tokens (e.g., [PERSON], [ORGANIZATION]) intact.

Text:
\"\"\"{text}\"\"\"

Output ONLY the paraphrase (no preamble).
"""
    if mode == "neutral_rewrite":
        return f"""Rewrite the following {source} text about homelessness into a more neutral, less emotionally loaded style.

Constraints:
- Preserve the core meaning and the main point.
- Do not add new facts, locations, or numbers.
- Do not explicitly mention "bias" or "mitigation".
- Keep any bracketed deidentification tokens (e.g., [PERSON], [ORGANIZATION]) intact.

Text:
\"\"\"{text}\"\"\"

Output ONLY the rewrite (no preamble).
"""
    raise ValueError(f"Unknown alter mode: {mode}")


def alter_texts(
    texts: List[str],
    *,
    llm_model: str,
    source: str,
    mode: str,
    max_new_tokens: int,
) -> List[str]:
    outs = []
    for t in texts:
        prompt = _alter_prompt(t, source=source, mode=mode)
        out = call_api_llm(prompt, llm_model, max_tokens=max_new_tokens)
        outs.append(out.strip())
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["reddit", "x", "news", "meeting_minutes"], required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alter_model", choices=["gpt4", "gemini", "grok"], default="gpt4")
    ap.add_argument("--alter_mode", choices=["paraphrase", "neutral_rewrite"], default="neutral_rewrite")
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--max_chars", type=int, default=800)
    ap.add_argument("--nimby_threshold", type=float, default=0.5, help="Soft-label threshold for gold NIMBY on originals (if available).")
    ap.add_argument("--out_dir", type=str, default="output/nimby_bias_eval/human_vs_llm_altered")
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or f"{args.source}_{args.n}_{args.alter_model}_{args.alter_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    df, text_col = _load_gold_texts(args.source)
    soft = _load_soft_labels(args.source)
    # Align soft labels to gold by row index if present
    if soft is not None and len(soft) != len(df):
        m = min(len(soft), len(df))
        soft = soft.iloc[:m].reset_index(drop=True)
        df = df.iloc[:m].reset_index(drop=True)

    idxs = [rng.randrange(0, len(df)) for _ in range(args.n)]
    originals_raw = [df.loc[i, text_col] for i in idxs]
    originals = [_trim(t, args.max_chars) for t in originals_raw]

    # Attach gold NIMBY label on originals if available
    gold_nimby = None
    if soft is not None and "not in my backyard" in soft.columns:
        gold_nimby = [(float(soft.loc[i, "not in my backyard"]) >= args.nimby_threshold) for i in idxs]

    altered_raw = alter_texts(
        originals,
        llm_model=args.alter_model,
        source=args.source,
        mode=args.alter_mode,
        max_new_tokens=args.max_new_tokens,
    )
    altered = [_trim(t, args.max_chars) for t in altered_raw]

    # Build pair table
    pair_rows = []
    for k, (orig, alt) in enumerate(zip(originals, altered)):
        pair_id = f"{tag}_pair_{k:05d}"
        pair_rows.append(
            {
                "pair_id": pair_id,
                "source": args.source,
                "text_original": orig,
                "text_altered": alt,
                "gold_nimby_original": (gold_nimby[k] if gold_nimby is not None else ""),
            }
        )
    pairs_df = pd.DataFrame(pair_rows)
    pairs_path = os.path.join(args.out_dir, f"{tag}.pairs.csv")
    pairs_df.to_csv(pairs_path, index=False)

    # Build blinded item list (each pair contributes 2 items)
    items = []
    key = []
    for row in pair_rows:
        pair_id = row["pair_id"]
        for variant, text in [("original", row["text_original"]), ("altered", row["text_altered"])]:
            item_id = f"{pair_id}_{variant}_{_stable_id(text)}"
            items.append(
                {
                    "item_id": item_id,
                    "pair_id": pair_id,
                    "text": text,
                    "nimby_label_yesno": "",  # for human annotation
                    "confidence_1to5": "",     # optional
                }
            )
            key.append({"item_id": item_id, "pair_id": pair_id, "variant": variant})

    rng.shuffle(items)
    items_path = os.path.join(args.out_dir, f"{tag}.blinded_items.csv")
    key_path = os.path.join(args.out_dir, f"{tag}.key.csv")
    pd.DataFrame(items).to_csv(items_path, index=False)
    pd.DataFrame(key).to_csv(key_path, index=False)

    meta = {
        "tag": tag,
        "source": args.source,
        "n_pairs": args.n,
        "alter_model": args.alter_model,
        "alter_mode": args.alter_mode,
        "paths": {"pairs": pairs_path, "blinded_items": items_path, "key": key_path},
    }
    meta_path = os.path.join(args.out_dir, f"{tag}.meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

