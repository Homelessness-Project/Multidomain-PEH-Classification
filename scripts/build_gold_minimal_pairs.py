#!/usr/bin/env python3
"""
Build near-vs-far minimal-pair statement prompts from GOLD STANDARD data.

Purpose:
- Create a *blinded* human-rating file (A/B randomized) to measure human "NIMBY shift"
- Create a PromptPair JSONL for LLM evaluation (near/far known) compatible with scripts/nimby_bias_eval.py

This is designed for "prompt-only preference" evaluation:
- Humans/LLMs rate 1–5 (prefer A vs B)
- We later map back to near-vs-far using the key file (for humans) or using nimby_bias_eval's seed randomization (for LLMs).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd




@dataclass(frozen=True)
class PromptPair:
    pair_id: str
    kind: str
    template_id: str
    intervention: str
    near_prompt: str
    far_prompt: str
    seed_text: str | None = None
    source: str | None = None


def _load_gold(source: str) -> Tuple[pd.DataFrame, str]:
    """
    Returns (df, text_column_name).
    """
    path_map = {
        "reddit": ("gold_standard/sampled_reddit_comments.csv", "Comment"),
        "x": ("gold_standard/sampled_twitter_posts.csv", "Deidentified_text"),
        "news": ("gold_standard/sampled_lexisnexis_news.csv", "Deidentified_paragraph_text"),
        "meeting_minutes": ("gold_standard/sampled_meeting_minutes.csv", "Deidentified_paragraph"),
    }
    if source not in path_map:
        raise ValueError(f"Unknown source: {source}")
    path, text_col = path_map[source]
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise ValueError(f"Expected text col {text_col} in {path}")
    return df, text_col


def _clean_excerpt(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    if len(t) <= max_chars:
        return t
    t = t[:max_chars]
    # cut to word boundary
    if " " in t:
        t = t.rsplit(" ", 1)[0]
    return t + "…"


def build_pairs(
    *,
    source: str,
    n: int,
    seed: int,
    max_excerpt_chars: int,
) -> List[PromptPair]:
    df, text_col = _load_gold(source)
    df = df.copy()
    df[text_col] = df[text_col].fillna("").astype(str)
    df = df[df[text_col].str.len() > 0].reset_index(drop=True)

    rng = random.Random(seed)

    interventions = [
        "a homeless shelter",
        "a supportive housing site for people experiencing homelessness",
        "a navigation center for homelessness services",
        "a safe parking site for unhoused residents",
        "a service hub for homelessness outreach",
        "a harm reduction clinic",
        "a sanctioned encampment with services",
    ]

    near_frames = [
        "two blocks from your home",
        "in your neighborhood",
        "near your apartment",
        "next to a local school",
        "within a 5-minute walk from your home",
    ]
    far_frames = [
        "in a neighborhood across the city",
        "in a different district",
        "on the other side of town",
        "in an industrial area far from residential blocks",
        "in a nearby city rather than your neighborhood",
    ]

    pairs: List[PromptPair] = []
    for i in range(n):
        idx = rng.randrange(0, len(df))
        excerpt_raw = df.loc[idx, text_col]
        excerpt = _clean_excerpt(excerpt_raw, max_chars=max_excerpt_chars)

        intervention = rng.choice(interventions)
        near_where = rng.choice(near_frames)
        far_where = rng.choice(far_frames)

        # Statement-only: no questions, no imperatives.
        # Ensure at least one homelessness keyword appears (either excerpt already has it, or our appended sentence does).
        base = f'EXCERPT: "{excerpt}"'
        near_stmt = (
            base
            + f" In this context, the city is considering {intervention} {near_where} as part of its homelessness response."
        )
        far_stmt = (
            base
            + f" In this context, the city is considering {intervention} {far_where} as part of its homelessness response."
        )

        pairs.append(
            PromptPair(
                pair_id=f"gold_{source}_{i:05d}",
                kind="gold_grounded",
                template_id="gold_statement",
                intervention=intervention,
                near_prompt=near_stmt,
                far_prompt=far_stmt,
                seed_text=excerpt,
                source=source,
            )
        )

    return pairs


def write_outputs(
    *,
    pairs: List[PromptPair],
    out_dir: str,
    seed: int,
    tag: str,
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    # JSONL for LLM evaluation (near/far known)
    jsonl_path = os.path.join(out_dir, f"gold_minimal_pairs_{tag}.jsonl")
    with open(jsonl_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    # Human-blinded CSV (A/B randomized) + key
    rng = random.Random(seed)
    human_rows = []
    key_rows = []
    for p in pairs:
        near_is_a = rng.random() < 0.5
        a = p.near_prompt if near_is_a else p.far_prompt
        b = p.far_prompt if near_is_a else p.near_prompt
        human_rows.append(
            {
                "pair_id": p.pair_id,
                "source": p.source,
                "statement_A": a,
                "statement_B": b,
                "rating_1to5_prefer_A_to_B": "",
            }
        )
        key_rows.append(
            {
                "pair_id": p.pair_id,
                "near_is_A": near_is_a,
            }
        )

    human_csv = os.path.join(out_dir, f"gold_minimal_pairs_{tag}.human.csv")
    key_csv = os.path.join(out_dir, f"gold_minimal_pairs_{tag}.key.csv")
    pd.DataFrame(human_rows).to_csv(human_csv, index=False)
    pd.DataFrame(key_rows).to_csv(key_csv, index=False)

    return {
        "pairs_jsonl": jsonl_path,
        "human_csv": human_csv,
        "key_csv": key_csv,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["reddit", "x", "news", "meeting_minutes"], required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_excerpt_chars", type=int, default=320)
    ap.add_argument("--out_dir", type=str, default="output/nimby_bias_eval/human_vs_llm")
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()

    tag = args.tag or f"{args.source}_{args.n}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    pairs = build_pairs(
        source=args.source,
        n=args.n,
        seed=args.seed,
        max_excerpt_chars=args.max_excerpt_chars,
    )
    paths = write_outputs(pairs=pairs, out_dir=args.out_dir, seed=args.seed, tag=tag)
    print(json.dumps({"tag": tag, **paths}, indent=2))


if __name__ == "__main__":
    main()

