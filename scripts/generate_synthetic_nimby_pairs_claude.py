#!/usr/bin/env python3
"""
Generate synthetic NIMBY minimal-pair prompts using Claude Code CLI.

This mirrors the setup style in the adjacent `cst_agents` repository: we call the local
`claude` CLI non-interactively (stdin prompt -> stdout response).

Output format: JSONL where each line is a PromptPair-compatible object for
scripts/nimby_bias_eval.py --pairs_file <jsonl>.

Example:
  python3 scripts/generate_synthetic_nimby_pairs_claude.py --n 500 --out output/nimby_bias_eval/synth_pairs_claude.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List

from claude_cli import claude_code_version_text, run_claude


def _load_prompt_template() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, ".claude", "prompts", "generate_synthetic_nimby_pairs.md")
    with open(path, "r") as f:
        return f.read()

def _load_common() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, ".claude", "prompts", "_nimby_pairs_common.md")
    with open(path, "r") as f:
        return f.read()

def _load_prompt_template_path(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def _extract_json(text: str) -> List[Dict[str, Any]]:
    """
    Accept either:
    - pure JSON array
    - fenced ```json ... ```
    """
    t = text.strip()
    fence = re.search(r"```json\s*(.*?)\s*```", t, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        t = fence.group(1).strip()
    # Try array parse
    data = json.loads(t)
    if not isinstance(data, list):
        raise ValueError("Claude output was not a JSON array.")
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--chunk_size", type=int, default=None, help="If set, generate in chunks and append to JSONL.")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="If output exists, resume by appending remaining chunks.")
    ap.add_argument("--claude_model", type=str, default="haiku", help="Claude model alias/name to use (e.g. haiku, sonnet, opus).")
    ap.add_argument("--prompt_file", type=str, default=None, help="Optional .claude prompt template path.")
    args = ap.parse_args()

    out_path = args.out or f"output/nimby_bias_eval/synth_pairs_claude_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"Generating {args.n} pairs via Claude…")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmpl = _load_prompt_template_path(args.prompt_file) if args.prompt_file else _load_prompt_template()
    common = _load_common()
    tmpl = tmpl.replace("{{COMMON}}", common.strip())
    total_n = int(args.n)
    chunk_size = int(args.chunk_size) if args.chunk_size else total_n

    # Resume support: if file exists, count lines and continue from there
    start_idx = 0
    if args.resume and os.path.exists(out_path):
        with open(out_path, "r") as f:
            start_idx = sum(1 for _ in f if _.strip())
        if start_idx >= total_n:
            print(f"Output already has {start_idx} lines (>= {total_n}); nothing to do.")
            return
        print(f"Resuming: found {start_idx} existing pairs in {out_path}")

    ver = claude_code_version_text()
    if ver:
        print(f"Claude Code: {ver}")

    remaining = total_n - start_idx
    n_chunks = (remaining + chunk_size - 1) // chunk_size
    print(f"Plan: {remaining} remaining pairs in {n_chunks} chunk(s) (chunk_size={chunk_size})")

    # Append mode if resuming or chunking; otherwise overwrite
    file_mode = "a" if (args.resume or os.path.exists(out_path) or chunk_size != total_n) else "w"

    written = 0
    for chunk_i in range(n_chunks):
        this_n = min(chunk_size, remaining - written)
        this_seed = args.seed + chunk_i
        prompt = tmpl.replace("{{N}}", str(this_n)).replace("{{SEED}}", str(this_seed))

        print(f"Chunk {chunk_i+1}/{n_chunks}: requesting {this_n} pairs (seed={this_seed})")
        res = run_claude(prompt, timeout_s=1800, model=args.claude_model)
        if res.exit_code != 0:
            raise RuntimeError(
                f"Claude failed in chunk {chunk_i+1}/{n_chunks} (exit {res.exit_code}).\n"
                f"Stderr:\n{res.stderr}\nStdout:\n{res.stdout}"
            )

        items = _extract_json(res.stdout)
        if len(items) != this_n:
            raise ValueError(f"Expected {this_n} items, got {len(items)} (chunk {chunk_i+1}/{n_chunks})")

        # Rewrite pair_ids to be globally unique and sequential across chunks
        # (keep shape compatible with PromptPair JSONL)
        for j, obj in enumerate(items):
            global_idx = start_idx + written + j
            obj["pair_id"] = f"claude_synth_{global_idx:05d}"

        with open(out_path, file_mode) as f:
            for obj in items:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            f.flush()

        file_mode = "a"
        written += this_n
        print(f"Wrote chunk {chunk_i+1}: total now {start_idx + written}/{total_n} lines -> {out_path}")

    print(f"Done. Wrote {written} new pairs to {out_path} (total={start_idx + written})")


if __name__ == "__main__":
    main()

