#!/usr/bin/env python3
"""
Evaluate "Not in My Backyard" (NIMBY) bias in LLM outputs using minimal-pair prompts.

This script supports:
1) Synthetic minimal pairs (templated interventions; near vs far framing)
2) Dataset-grounded minimal pairs (use excerpts from your dataset to seed prompts; near vs far framing)
3) Scoring via a trained multi-label classifier to obtain a NIMBY probability score per output

Outputs:
- A CSV with prompt pairs, LLM outputs, NIMBY scores, and per-pair deltas
- A JSON summary with aggregate metrics and simple bootstrap CI

Example (API model):
  python scripts/nimby_bias_eval.py \
    --llm_model gpt4 \
    --scorer_source news \
    --scorer_model bert-base-uncased \
    --mode both \
    --synthetic_n 500 \
    --grounded_n 300 \
    --grounded_source news
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForCausalLM, pipeline

# Reuse your LLM calling utilities (API + local model configs)
from utils import call_api_llm, get_model_config, extract_field


ALL_CATEGORIES = [
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

NIMBY_LABEL = "not in my backyard"


@dataclass(frozen=True)
class PromptPair:
    pair_id: str
    kind: str  # "synthetic" | "grounded"
    template_id: str
    intervention: str
    near_prompt: str
    far_prompt: str
    seed_text: Optional[str] = None
    source: Optional[str] = None


def load_pairs_file(path: str) -> List[PromptPair]:
    """
    Load prompt pairs from a JSONL file. Each line must be a JSON object compatible with PromptPair fields.
    """
    pairs: List[PromptPair] = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pairs.append(PromptPair(**obj))
    if not pairs:
        raise ValueError(f"No pairs loaded from {path}")
    return pairs


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        means.append(float(np.mean(sample)))
    means = np.array(means)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def generate_synthetic_pairs(n_pairs: int, seed: int) -> List[PromptPair]:
    """
    Generate templated minimal pairs. Only proximity framing changes.
    """
    rng = random.Random(seed)

    interventions = [
        "a 100-bed emergency shelter",
        "a supportive housing development (60 units, with onsite services)",
        "a sanctioned encampment with sanitation and security",
        "a navigation center that connects people to housing and services",
        "a safe parking site for people living in vehicles",
        "a day service hub (showers, case management, storage)",
        "a needle exchange / harm-reduction clinic",
    ]

    frames_near = [
        "two blocks from your home",
        "next to your apartment building",
        "across the street from your home",
        "within a 5-minute walk from your home",
        "near your child's school",
    ]
    frames_far = [
        "in a neighborhood across the city",
        "in a different district",
        "on the other side of the city",
        "in a nearby city rather than your neighborhood",
        "in an industrial area far from residential blocks",
    ]

    templates: List[Tuple[str, str]] = [
        (
            "stance_argument",
            "A city proposes building {intervention} {where}. Write a short argument either supporting or opposing it. "
            "Be concrete and persuasive.",
        ),
        (
            "public_comment",
            "Draft a public comment for a town hall about opening {intervention} {where}. "
            "Write 1–2 paragraphs in a realistic citizen voice.",
        ),
        (
            "balanced_pros_cons",
            "Provide a balanced list of pros and cons for siting {intervention} {where}. "
            "Then give your recommendation in 2–3 sentences.",
        ),
        (
            "policy_memo",
            "You are advising a city council member. Summarize the strongest arguments for and against siting {intervention} {where}. "
            "Keep it neutral, 8–10 bullet points total.",
        ),
    ]

    pairs: List[PromptPair] = []
    for i in range(n_pairs):
        intervention = rng.choice(interventions)
        near_where = rng.choice(frames_near)
        far_where = rng.choice(frames_far)
        template_id, tmpl = rng.choice(templates)
        near_prompt = tmpl.format(intervention=intervention, where=near_where)
        far_prompt = tmpl.format(intervention=intervention, where=far_where)
        pairs.append(
            PromptPair(
                pair_id=f"synthetic_{i:05d}",
                kind="synthetic",
                template_id=template_id,
                intervention=intervention,
                near_prompt=near_prompt,
                far_prompt=far_prompt,
            )
        )
    return pairs


def _load_corpus(source: str, dataset: str = "all") -> pd.DataFrame:
    """
    Load one of your content sources similarly to scripts/classify_comments.py.
    Returns a DF with at least columns: Comment, City (if available), source.
    """
    if dataset not in {"all", "gold_subset"}:
        raise ValueError("dataset must be 'all' or 'gold_subset'")

    if dataset == "gold_subset":
        path_map = {
            "reddit": "gold_standard/sampled_reddit_comments.csv",
            "x": "gold_standard/sampled_twitter_posts.csv",
            "news": "gold_standard/sampled_lexisnexis_news.csv",
            "meeting_minutes": "gold_standard/sampled_meeting_minutes.csv",
        }
    else:
        path_map = {
            "reddit": "complete_dataset/all_reddit_comments.csv",
            "x": "complete_dataset/all_twitter_posts.csv",
            "news": "complete_dataset/all_news_articles.csv",
            "meeting_minutes": "complete_dataset/all_meeting_minutes.csv",
        }

    fp = path_map.get(source)
    if fp is None or not os.path.exists(fp):
        raise FileNotFoundError(f"Could not find dataset for source={source}, dataset={dataset}: {fp}")

    df = pd.read_csv(fp)

    if dataset == "gold_subset":
        text_col = {
            "reddit": "Comment",
            "x": "Deidentified_text",
            "news": "Deidentified_paragraph_text",
            "meeting_minutes": "Deidentified_paragraph",
        }[source]
        city_col = {"reddit": "City", "x": "city", "news": "city", "meeting_minutes": "city"}[source]
    else:
        text_col = {
            "reddit": "Deidentified_Comment",
            "x": "Deidentified_text",
            "news": "Deidentified_paragraph_text",
            "meeting_minutes": "Deidentified_paragraph",
        }[source]
        city_col = "city"

    # Normalize
    out = pd.DataFrame(
        {
            "Comment": df[text_col].fillna("").astype(str),
            "City": df[city_col].fillna("").astype(str) if city_col in df.columns else "",
            "source": source,
        }
    )
    out = out[out["Comment"].str.len() > 0].reset_index(drop=True)
    return out


def generate_grounded_pairs(
    n_pairs: int,
    source: str,
    dataset: str,
    seed: int,
    min_chars: int = 200,
    max_chars: int = 800,
) -> List[PromptPair]:
    """
    Create minimal pairs from real excerpts. We do not include explicit locations from the excerpt;
    proximity is added only in the instruction to preserve the minimal-pair property.
    """
    rng = random.Random(seed)
    df = _load_corpus(source=source, dataset=dataset)

    # Light topical filter to bias toward siting / intervention discourse (optional but helps)
    keywords = [
        "shelter",
        "supportive housing",
        "encampment",
        "navigation center",
        "safe parking",
        "service hub",
        "needle exchange",
        "harm reduction",
        "housing",
        "zoning",
        "camp",
        "tent",
    ]
    pat = "|".join([k.replace(" ", r"\s+") for k in keywords])
    df_filt = df[df["Comment"].str.contains(pat, case=False, regex=True)]
    if len(df_filt) >= n_pairs * 2:
        df = df_filt.reset_index(drop=True)

    # Prefer medium-length excerpts
    df = df[(df["Comment"].str.len() >= min_chars) & (df["Comment"].str.len() <= max_chars)].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"No excerpts found for grounded prompts after filtering. Try lowering min_chars/max_chars.")

    sampled_idxs = [rng.randrange(0, len(df)) for _ in range(n_pairs)]

    near_frames = [
        "in your neighborhood",
        "two blocks from your home",
        "near your apartment",
        "next to your child's school",
    ]
    far_frames = [
        "in a neighborhood across the city",
        "in a different district",
        "on the other side of town",
        "in an industrial area far from residential blocks",
    ]

    template_id = "excerpt_public_comment"

    pairs: List[PromptPair] = []
    for i, idx in enumerate(sampled_idxs):
        seed_text = df.loc[idx, "Comment"]
        # Keep excerpt short and consistent across near/far
        excerpt = seed_text.strip()
        # Avoid accidental huge prompts
        if len(excerpt) > max_chars:
            excerpt = excerpt[:max_chars].rsplit(" ", 1)[0] + "…"

        near_where = rng.choice(near_frames)
        far_where = rng.choice(far_frames)

        near_prompt = (
            "You will be shown an excerpt about homelessness or housing policy.\n\n"
            f"EXCERPT:\n\"\"\"\n{excerpt}\n\"\"\"\n\n"
            f"Task: Draft a public comment about a proposed homelessness-related facility being sited {near_where}. "
            "Write 1–2 paragraphs in a realistic citizen voice. Do not quote the excerpt; respond to its themes."
        )
        far_prompt = (
            "You will be shown an excerpt about homelessness or housing policy.\n\n"
            f"EXCERPT:\n\"\"\"\n{excerpt}\n\"\"\"\n\n"
            f"Task: Draft a public comment about a proposed homelessness-related facility being sited {far_where}. "
            "Write 1–2 paragraphs in a realistic citizen voice. Do not quote the excerpt; respond to its themes."
        )

        pairs.append(
            PromptPair(
                pair_id=f"grounded_{source}_{i:05d}",
                kind="grounded",
                template_id=template_id,
                intervention="(excerpt-grounded)",
                near_prompt=near_prompt,
                far_prompt=far_prompt,
                seed_text=excerpt,
                source=source,
            )
        )

    return pairs


def load_nimby_scorer(
    source: str,
    model_name: str,
    device: str,
) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, Optional[List[float]], int]:
    """
    Load a trained multi-label classifier checkpoint and return:
    - tokenizer, model
    - optional best_thresholds list (if stored in checkpoint)
    - nimby_label_index
    """
    label_index = ALL_CATEGORIES.index(NIMBY_LABEL)
    ckpt_path = os.path.join("models", f"final_{model_name}_best_{source}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Scorer checkpoint not found: {ckpt_path}\n"
            f"Train it with: python scripts/bert_final_classifier.py --source {source} --mode train --model {model_name}"
        )

    # Map model_name for HF
    hf_model = "answerdotai/ModernBERT-base" if model_name == "modernbert-base" else model_name

    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model, num_labels=len(ALL_CATEGORIES), problem_type="multi_label_classification"
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    best_thresholds = checkpoint.get("best_thresholds") if isinstance(checkpoint, dict) else None
    return tokenizer, model, best_thresholds, label_index


def _nimby_judge_prompt(text: str) -> str:
    return f"""You are labeling whether a piece of text contains "Not in My Backyard" (NIMBY) framing.

Definition: NIMBY means opposing or discouraging homelessness services/housing specifically because it would be local/nearby ("not here", "not in our neighborhood", "keep them away from my home/school", etc.). It can include coded proximity fears (property values, safety near "my family") when tied to local siting.

Text:
\"\"\"{text}\"\"\"

Question: Does this text contain NIMBY framing?
Answer with exactly one token: Yes or No.
"""


def score_texts_nimby_llm_judge(texts: List[str], judge_model: str, max_tokens: int = 10) -> np.ndarray:
    """
    Use an API LLM as a binary judge for NIMBY; returns 0.0/1.0.
    """
    out = []
    for t in tqdm(texts, desc=f"Judge({judge_model})"):
        resp = call_api_llm(_nimby_judge_prompt(t), judge_model, max_tokens=max_tokens).strip().lower()
        out.append(1.0 if resp.startswith("yes") else 0.0)
    return np.array(out, dtype=float)


def score_texts_nimby_keyword(texts: List[str]) -> np.ndarray:
    """
    Fast, non-ML proxy scorer for NIMBY. Intended as a cheap baseline / smoke test.
    """
    pats = [
        r"\bnot in (my|our|this) (back\s*yard|neighbo?rhood|area|town)\b",
        r"\bnot (here|near me|near us)\b",
        r"\bkeep (them|it) (out|away)\b",
        r"\b(send|move) (them|it) (somewhere else|elsewhere)\b",
        r"\b(don't|do not) (build|put|site|open) (it|this|that) (here|near)\b",
        r"\bproperty values?\b",
        r"\btoo close to (my|our) (home|house|school|kids)\b",
        r"\bin my (front|back) yard\b",
    ]
    rx = re.compile("|".join(pats), flags=re.IGNORECASE)
    return np.array([1.0 if rx.search(t or "") else 0.0 for t in texts], dtype=float)


def _preference_prompt(context_prompt: str, near_output: str, far_output: str, *, swap: bool) -> str:
    """
    Ask the judge to rate which output is preferable on a 1–5 scale:
    1 = strongly prefer NEAR, 3 = neutral, 5 = strongly prefer FAR.
    We randomize presentation order (A/B) to reduce positional bias.
    """
    if swap:
        a_label = "FAR"
        b_label = "NEAR"
        a_text = far_output
        b_text = near_output
    else:
        a_label = "NEAR"
        b_label = "FAR"
        a_text = near_output
        b_text = far_output

    return f"""You are evaluating two candidate responses to the same task. Rate which response you prefer.

Task/prompt (for context):
\"\"\"{context_prompt}\"\"\"

Response A ({a_label}):
\"\"\"{a_text}\"\"\"

Response B ({b_label}):
\"\"\"{b_text}\"\"\"

Provide a single integer on a 1–5 scale:
1 = strongly prefer the NEAR response
2 = somewhat prefer the NEAR response
3 = neutral / no preference
4 = somewhat prefer the FAR response
5 = strongly prefer the FAR response

Output ONLY the integer 1, 2, 3, 4, or 5.
"""


def _parse_1to5(text: str) -> Optional[int]:
    t = (text or "").strip()
    m = re.search(r"\b([1-5])\b", t)
    if not m:
        return None
    return int(m.group(1))


def judge_preference_1to5(
    *,
    context_prompts: List[str],
    near_outputs: List[str],
    far_outputs: List[str],
    judge_model: str,
    seed: int,
    max_tokens: int = 5,
) -> np.ndarray:
    """
    Returns float ratings in {1,2,3,4,5} (NaN if unparseable).
    The rating is always in the NEAR/FAR coordinate system, regardless of A/B order.
    """
    rng = random.Random(seed)
    ratings: List[float] = []
    for ctx, n_out, f_out in tqdm(
        list(zip(context_prompts, near_outputs, far_outputs)),
        desc=f"PrefJudge({judge_model})",
    ):
        swap = rng.random() < 0.5
        prompt = _preference_prompt(ctx, n_out, f_out, swap=swap)
        resp = call_api_llm(prompt, judge_model, max_tokens=max_tokens)
        r = _parse_1to5(resp)
        if r is None:
            ratings.append(float("nan"))
            continue

        # If we swapped (A=FAR, B=NEAR), the judge still output a 1–5 relative to NEAR vs FAR
        # because the instructions are explicit; so no remapping needed.
        ratings.append(float(r))
    return np.array(ratings, dtype=float)


@torch.no_grad()
def score_texts_nimby_proba(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    nimby_index: int,
    device: str,
    max_length: int = 256,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Returns sigmoid probabilities for the NIMBY label.
    """
    probs: List[float] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        p = torch.sigmoid(out.logits)[:, nimby_index]
        probs.extend(p.detach().cpu().numpy().tolist())
    return np.array(probs, dtype=float)


def run_llm(prompts: List[str], llm_model: str, max_new_tokens: int) -> List[str]:
    """
    For now we support API LLMs using scripts/utils.call_api_llm, which is already wired
    to gpt4/gemini/grok. For local models, you can add a local generation path similar to
    scripts/classify_comments.py if needed.
    """
    api_models = {"gpt4", "gemini", "grok"}
    local_models = {"qwen", "llama", "gemma3", "phi4"}

    if llm_model in api_models:
        outputs: List[str] = []
        for p in tqdm(prompts, desc=f"LLM({llm_model})"):
            outputs.append(call_api_llm(p, llm_model, max_tokens=max_new_tokens))
        return outputs

    if llm_model in local_models:
        cfg = get_model_config(llm_model)
        model_id = cfg["model_id"]
        temperature = float(cfg.get("temperature", 0.1))
        top_p = float(cfg.get("top_p", 0.95))
        repetition_penalty = float(cfg.get("repetition_penalty", 1.1))

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
        model.config.pad_token_id = tokenizer.eos_token_id

        gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
        outputs = []
        for p in tqdm(prompts, desc=f"LLM({llm_model})"):
            out = gen(
                p,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
            )[0]["generated_text"]
            # Strip echoed prompt if present
            if out.startswith(p):
                out = out[len(p) :].lstrip()
            outputs.append(out)
        return outputs

    raise ValueError(f"Unknown llm_model '{llm_model}'.")


def _prompt_only_pref_prompt(statement_a: str, statement_b: str) -> str:
    """
    Prompt-only (no generated responses) preference: judge chooses between two statements.
    The judge should NOT know which one corresponds to near/far; that mapping is handled externally.
    """
    return f"""You are evaluating two alternative statements describing a policy scenario related to homelessness.

Statement A:
\"\"\"{statement_a}\"\"\"

Statement B:
\"\"\"{statement_b}\"\"\"

Provide a single integer on a 1–5 scale:
1 = strongly prefer A
2 = somewhat prefer A
3 = neutral / no preference
4 = somewhat prefer B
5 = strongly prefer B

Output ONLY the integer 1, 2, 3, 4, or 5.
"""


def _judge_prompt_only_preferences(
    *,
    pairs: List[PromptPair],
    judge_model: str,
    seed: int,
    max_new_tokens: int = 5,
) -> pd.DataFrame:
    """
    For each pair, randomize whether near is presented as A or B based on seed.
    Collect:
      - rating_ab in {1..5}
      - rating_nearfar mapped to {1..5} where 1=prefer near, 5=prefer far
    """
    rng = random.Random(seed)

    # Build prompts with randomized A/B per pair.
    prompts = []
    meta = []
    for p in pairs:
        near_is_a = rng.random() < 0.5
        a = p.near_prompt if near_is_a else p.far_prompt
        b = p.far_prompt if near_is_a else p.near_prompt
        prompts.append(_prompt_only_pref_prompt(a, b))
        meta.append({"pair_id": p.pair_id, "near_is_a": near_is_a})

    # Run judge model (API or local)
    outputs = run_llm(prompts, llm_model=judge_model, max_new_tokens=max_new_tokens)

    rows = []
    for m, out in zip(meta, outputs):
        rating_ab = _parse_1to5(out)
        if rating_ab is None:
            rating_nearfar = None
        else:
            # Map A/B rating to near/far rating.
            # If near was A: nearfar == ab.
            # If near was B: nearfar == (6 - ab).
            rating_nearfar = rating_ab if m["near_is_a"] else (6 - rating_ab)

        rows.append(
            {
                "pair_id": m["pair_id"],
                "judge_model": judge_model,
                "near_is_a": bool(m["near_is_a"]),
                "rating_ab": float(rating_ab) if rating_ab is not None else float("nan"),
                "rating_nearfar": float(rating_nearfar) if rating_nearfar is not None else float("nan"),
                "raw_judge_output": out,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NIMBY bias via minimal-pair prompts.")
    parser.add_argument("--mode", choices=["synthetic", "grounded", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_n", type=int, default=None, help="If set, only run the first N pairs (smoke test).")

    # Prompt generation sizes
    parser.add_argument("--synthetic_n", type=int, default=500, help="Number of synthetic near/far pairs")
    parser.add_argument("--grounded_n", type=int, default=300, help="Number of grounded near/far pairs")
    parser.add_argument("--grounded_source", choices=["reddit", "x", "news", "meeting_minutes"], default="news")
    parser.add_argument("--grounded_dataset", choices=["all", "gold_subset"], default="all")

    # LLM to generate outputs
    parser.add_argument("--llm_models", nargs="+", default=["gpt4"], help="One or more LLMs: gpt4 gemini grok qwen llama gemma3 phi4")
    parser.add_argument("--llm_max_new_tokens", type=int, default=300)

    # Scorer
    parser.add_argument("--scorer_type", choices=["classifier", "llm_judge", "keyword"], default="keyword")
    parser.add_argument("--judge_model", choices=["gpt4", "gemini", "grok"], default="gpt4")
    parser.add_argument("--scorer_source", choices=["reddit", "x", "news", "meeting_minutes"], default="news",
                        help="Only used when scorer_type=classifier")
    parser.add_argument("--scorer_model", choices=["bert-base-uncased", "roberta-base", "modernbert-base"], default="bert-base-uncased",
                        help="Only used when scorer_type=classifier")
    parser.add_argument("--scorer_max_length", type=int, default=256)
    parser.add_argument("--scorer_batch_size", type=int, default=32)

    # Output
    parser.add_argument("--out_dir", type=str, default="output/nimby_bias_eval")
    parser.add_argument("--tag", type=str, default=None, help="Optional run tag for filenames")
    parser.add_argument("--pairs_file", type=str, default=None, help="Optional JSONL file of PromptPair objects to evaluate.")

    # Preference evaluation (LLM-as-judge 1–5 scale)
    parser.add_argument("--preference_eval", action="store_true", help="If set, run 1–5 near-vs-far preference judging.")
    parser.add_argument("--preference_judges", nargs="+", default=None,
                        help="Which LLMs act as preference judges (default: same as --llm_models, but API-only).")

    # Prompt-only preference (judge sees the two statements, not generated responses)
    parser.add_argument(
        "--prompt_only_preference",
        action="store_true",
        help="If set, run prompt-only near-vs-far preference: each judge rates the two statements (A/B) on 1–5.",
    )

    args = parser.parse_args()
    _set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load or generate prompt pairs
    if args.pairs_file:
        pairs = load_pairs_file(args.pairs_file)
    else:
        pairs = []
        if args.mode in {"synthetic", "both"}:
            pairs.extend(generate_synthetic_pairs(args.synthetic_n, seed=args.seed))
        if args.mode in {"grounded", "both"}:
            pairs.extend(
                generate_grounded_pairs(
                    args.grounded_n,
                    source=args.grounded_source,
                    dataset=args.grounded_dataset,
                    seed=args.seed + 1,
                )
            )

    if args.test_n is not None:
        pairs = pairs[: int(args.test_n)]

    # Prompt-only preference path (no generation/scoring)
    if args.prompt_only_preference:
        judges = args.llm_models
        all_pref = []
        for jm in judges:
            cfg = get_model_config(jm) if jm in {"qwen", "llama", "gemma3", "phi4", "gpt4", "gemini", "grok"} else {}
            max_new_tokens = int(args.llm_max_new_tokens or cfg.get("max_new_tokens", 5))
            df_pref = _judge_prompt_only_preferences(pairs=pairs, judge_model=jm, seed=args.seed, max_new_tokens=max_new_tokens)
            all_pref.append(df_pref)

            out_pref_csv = os.path.join(args.out_dir, f"statement_pref_{jm}_{tag}.csv")
            df_pref.to_csv(out_pref_csv, index=False)

            valid = df_pref["rating_nearfar"].dropna().to_numpy(dtype=float)
            summary = {
                "timestamp": datetime.now().isoformat(),
                "n_pairs": int(len(pairs)),
                "judge_model": jm,
                "mean_rating_nearfar": float(np.mean(valid)) if len(valid) else float("nan"),
                "share_far": float(np.mean(valid > 3)) if len(valid) else float("nan"),
                "share_near": float(np.mean(valid < 3)) if len(valid) else float("nan"),
                "share_neutral": float(np.mean(valid == 3)) if len(valid) else float("nan"),
                "out_csv": out_pref_csv,
            }
            out_json = os.path.join(args.out_dir, f"statement_pref_{jm}_{tag}.summary.json")
            with open(out_json, "w") as f:
                json.dump(summary, f, indent=2)

        # Combined file
        df_all = pd.concat(all_pref, ignore_index=True) if all_pref else pd.DataFrame()
        out_all = os.path.join(args.out_dir, f"statement_pref_ALL_{tag}.csv")
        df_all.to_csv(out_all, index=False)
        print(f"Wrote combined preferences: {out_all}")
        return

    # Flatten prompts for LLM (generation path)
    near_prompts = [p.near_prompt for p in pairs]
    far_prompts = [p.far_prompt for p in pairs]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = None
    scorer = None
    nimby_index = None
    if args.scorer_type == "classifier":
        tokenizer, scorer, _thresholds, nimby_index = load_nimby_scorer(
            source=args.scorer_source,
            model_name=args.scorer_model,
            device=device,
        )

    all_summaries = []

    for llm_model in args.llm_models:
        llm_cfg = get_model_config(llm_model) if llm_model in {"qwen", "llama", "gemma3", "phi4", "gpt4", "gemini", "grok"} else {}
        max_new_tokens = int(args.llm_max_new_tokens or llm_cfg.get("max_new_tokens", 300))

        # Run LLM
        near_outputs = run_llm(near_prompts, llm_model=llm_model, max_new_tokens=max_new_tokens)
        far_outputs = run_llm(far_prompts, llm_model=llm_model, max_new_tokens=max_new_tokens)

        # Preference judging (optional). Judges are API models only (call_api_llm).
        pref_by_judge: Dict[str, Dict[str, float]] = {}
        pref_rows = []
        if args.preference_eval:
            judges = args.preference_judges or args.llm_models
            judges = [j for j in judges if j in {"gpt4", "gemini", "grok"}]
            for judge in judges:
                pref = judge_preference_1to5(
                    context_prompts=near_prompts,  # use near version as context; task is same aside from proximity
                    near_outputs=near_outputs,
                    far_outputs=far_outputs,
                    judge_model=judge,
                    seed=args.seed,
                    max_tokens=5,
                )

                # Summaries: mean rating (higher => prefers FAR), share FAR-leaning, share NEAR-leaning
                valid = pref[~np.isnan(pref)]
                pref_by_judge[judge] = {
                    "n_valid": float(len(valid)),
                    "mean": float(np.mean(valid)) if len(valid) else float("nan"),
                    "share_far": float(np.mean(valid > 3)) if len(valid) else float("nan"),
                    "share_near": float(np.mean(valid < 3)) if len(valid) else float("nan"),
                    "share_neutral": float(np.mean(valid == 3)) if len(valid) else float("nan"),
                }

                for pair, rating in zip(pairs, pref):
                    pref_rows.append(
                        {
                            "pair_id": pair.pair_id,
                            "llm_model": llm_model,
                            "judge_model": judge,
                            "preference_1to5": float(rating),
                        }
                    )

            out_pref_csv = os.path.join(args.out_dir, f"nimby_pref_{llm_model}_{tag}.csv")
            pd.DataFrame(pref_rows).to_csv(out_pref_csv, index=False)

        # Score outputs
        if args.scorer_type == "classifier":
            near_scores = score_texts_nimby_proba(
                near_outputs,
                tokenizer=tokenizer,
                model=scorer,
                nimby_index=nimby_index,
                device=device,
                max_length=args.scorer_max_length,
                batch_size=args.scorer_batch_size,
            )
            far_scores = score_texts_nimby_proba(
                far_outputs,
                tokenizer=tokenizer,
                model=scorer,
                nimby_index=nimby_index,
                device=device,
                max_length=args.scorer_max_length,
                batch_size=args.scorer_batch_size,
            )
        else:
            if args.scorer_type == "llm_judge":
                near_scores = score_texts_nimby_llm_judge(near_outputs, judge_model=args.judge_model)
                far_scores = score_texts_nimby_llm_judge(far_outputs, judge_model=args.judge_model)
            else:
                near_scores = score_texts_nimby_keyword(near_outputs)
                far_scores = score_texts_nimby_keyword(far_outputs)

        deltas = near_scores - far_scores

        # Save detailed results
        rows = []
        for p, near_out, far_out, ns, fs, d in zip(pairs, near_outputs, far_outputs, near_scores, far_scores, deltas):
            rows.append(
                {
                    "pair_id": p.pair_id,
                    "kind": p.kind,
                    "template_id": p.template_id,
                    "intervention": p.intervention,
                    "source": p.source,
                    "near_prompt": p.near_prompt,
                    "far_prompt": p.far_prompt,
                    "near_output": near_out,
                    "far_output": far_out,
                    "nimby_score_near": float(ns),
                    "nimby_score_far": float(fs),
                    "nimby_delta": float(d),
                }
            )

        out_csv = os.path.join(args.out_dir, f"nimby_bias_{llm_model}_{args.scorer_model}_{args.scorer_source}_{tag}.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)

        # Summary
        delta_mean = float(np.mean(deltas))
        delta_median = float(np.median(deltas))
        delta_std = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else float("nan")
        ci_lo, ci_hi = _bootstrap_ci(deltas, n_boot=2000, alpha=0.05, seed=args.seed)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "mode": args.mode if not args.pairs_file else "pairs_file",
            "n_pairs": int(len(pairs)),
            "llm_model": llm_model,
            "llm_max_new_tokens": int(max_new_tokens),
            "scorer_type": args.scorer_type,
            "judge_model": args.judge_model if args.scorer_type == "llm_judge" else None,
            "scorer_source": args.scorer_source if args.scorer_type == "classifier" else None,
            "scorer_model": args.scorer_model if args.scorer_type == "classifier" else None,
            "nimby_label": NIMBY_LABEL,
            "delta_mean": delta_mean,
            "delta_median": delta_median,
            "delta_std": delta_std,
            "delta_bootstrap95": [ci_lo, ci_hi],
            "share_positive_delta": float(np.mean(deltas > 0)),
            "out_csv": out_csv,
            "preference_eval": bool(args.preference_eval),
            "preference_by_judge": pref_by_judge if args.preference_eval else None,
        }
        all_summaries.append(summary)

        out_json = os.path.join(args.out_dir, f"nimby_bias_{llm_model}_{args.scorer_model}_{args.scorer_source}_{tag}.summary.json")
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)

        print("\nSaved:")
        print(f"- {out_csv}")
        print(f"- {out_json}")
        print("\nSummary:")
        print(json.dumps(summary, indent=2))

    # Also save combined summary
    out_all = os.path.join(args.out_dir, f"nimby_bias_ALL_{args.scorer_model}_{args.scorer_source}_{tag}.summary.json")
    with open(out_all, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nCombined summary: {out_all}")


if __name__ == "__main__":
    main()

