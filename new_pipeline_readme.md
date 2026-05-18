# NIMBY bias & related evaluation pipeline

This document consolidates the **near-vs-far minimal-pair**, **synthetic pair generation (Claude Code)**, **gold-standard minimal pairs**, **human-text vs LLM-altered text**, and **GPT-only relabeling** workflows added for evaluating “Not in My Backyard” (NIMBY) framing and LLM behavior.

---

## Prerequisites

- Python 3 with project venv: `.venv/bin/python` (recommended).
- **API keys** (optional, for API models): `.env` with `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GROK_API_KEY` as needed (see main `README.md`).
- **Claude Code CLI** (optional): local `claude` on `PATH` for synthetic pair generation; authenticated via `claude auth login`. Override binary with `CLAUDE_CMD` if needed.
- **Dependencies**: `requirements.txt` plus `torch`, `transformers`, etc. Local HF models need network access to download weights unless cached.

---

## Concepts (methodology in brief)

1. **Counterfactual near vs far**  
   Two prompts differ only in **proximity framing** (near speaker vs far). Any systematic shift in model preference or NIMBY-coded output is attributed to that attribute.

2. **Synthetic minimal pairs (Claude)**  
   JSONL of `near_prompt` / `far_prompt` pairs. Two template families:
   - **Request-style**: `generate_synthetic_nimby_pairs.md` (may use imperatives/questions).
   - **Statement-only**: `generate_synthetic_nimby_pairs_statement.md` (no `?`, no “write/draft/provide” instructions).

3. **1–5 preference (models only, prompt-only)**  
   `nimby_bias_eval.py --prompt_only_preference` presents two statements as A/B (order randomized by `--seed`), model outputs 1–5 (prefer A ↔ B), mapped to **near-vs-far**: 1 = prefer near, 3 = neutral, 5 = prefer far. **No new human 1–5 labels are required** unless you run a separate study.

4. **Gold-standard grounded pairs**  
   `build_gold_minimal_pairs.py` samples deidentified gold text, appends a fixed near/far intervention sentence, exports JSONL + optional blinded human CSV/key for *future* annotation.

5. **Human text vs LLM-altered text**  
   `human_vs_llm_altered_bias_eval.py`: original gold excerpt → API LLM rewrite (`neutral_rewrite` or `paraphrase`) → `pairs.csv` + blinded `blinded_items.csv` + `key.csv`.

6. **No new human annotators: GPT relabel**  
   `gpt_relabel_nimby_shift.py` labels NIMBY Yes/No on `text_original` and `text_altered` with the same judge model and reports **Δ = altered − original** with bootstrap CI.

7. **Formal writeup**  
   See `docs/emnlp_nimby_bias_methodology.md` for EMNLP-oriented framing: **Track A** = counterfactual / model preference; **Track B** = agreement with existing human soft labels on originals (not counterfactual).

---

## File map

| Piece | Path |
|--------|------|
| Claude prompt (shared) | `.claude/prompts/_nimby_pairs_common.md` |
| Claude prompt (request templates) | `.claude/prompts/generate_synthetic_nimby_pairs.md` |
| Claude prompt (statement-only) | `.claude/prompts/generate_synthetic_nimby_pairs_statement.md` |
| Claude CLI wrapper | `scripts/claude_cli.py` |
| Generate synthetic JSONL (Claude) | `scripts/generate_synthetic_nimby_pairs_claude.py` |
| Main eval (gen + score + pref) | `scripts/nimby_bias_eval.py` |
| Gold minimal pairs | `scripts/build_gold_minimal_pairs.py` |
| Alter human text + blinded export | `scripts/human_vs_llm_altered_bias_eval.py` |
| GPT judge + shift | `scripts/gpt_relabel_nimby_shift.py` |
| Methodology doc | `docs/emnlp_nimby_bias_methodology.md` |

Typical output directory: `output/nimby_bias_eval/` (and subdirs `human_vs_llm/`, `human_vs_llm_altered/`).

---

## 1. Generate synthetic pairs with Claude Code (chunked + resumable)

Templates inject `{{COMMON}}` via the generator; `{{N}}` and `{{SEED}}` are substituted per chunk.

**Statement-style (500 pairs, 100 per chunk, Haiku, resume):**

```bash
.venv/bin/python -u -B scripts/generate_synthetic_nimby_pairs_claude.py \
  --n 500 \
  --chunk_size 100 \
  --resume \
  --claude_model haiku \
  --prompt_file .claude/prompts/generate_synthetic_nimby_pairs_statement.md \
  --out output/nimby_bias_eval/synth_pairs_claude_statement.jsonl
```

**Request-style template** (default file if you omit `--prompt_file`):

```bash
.venv/bin/python -u -B scripts/generate_synthetic_nimby_pairs_claude.py \
  --n 500 \
  --chunk_size 100 \
  --resume \
  --claude_model haiku \
  --out output/nimby_bias_eval/synth_pairs_claude_request.jsonl
```

Flags:

- `--prompt_file` — path to `.md` template (must include `{{COMMON}}` placeholder for the shared block, or use the two provided files that already do).
- `--claude_model` — e.g. `haiku`, `sonnet` (passed to `claude -p --model …`).
- `--chunk_size` — smaller chunks if one-shot JSON hits limits; progress is appended after each chunk.
- `--resume` — continue from existing line count in `--out`.
- If `--out` is omitted, the script writes under `output/nimby_bias_eval/synth_pairs_claude_<timestamp>.jsonl`.

---

## 2. Prompt-only 1–5 preference across six LLMs (synthetic / any pairs JSONL)

Uses `near_prompt` and `far_prompt` from JSONL; A/B randomized per pair with `--seed`.

```bash
.venv/bin/python -u -B scripts/nimby_bias_eval.py \
  --pairs_file output/nimby_bias_eval/synth_pairs_claude_statement.jsonl \
  --prompt_only_preference \
  --llm_models llama qwen phi4 gpt4 gemini grok \
  --llm_max_new_tokens 5 \
  --seed 42 \
  --out_dir output/nimby_bias_eval \
  --tag stmt_pref_500
```

**Outputs:**

- `statement_pref_<model>_<tag>.csv`
- `statement_pref_<model>_<tag>.summary.json`
- `statement_pref_ALL_<tag>.csv`

**Note:** Local models (`llama`, `qwen`, `phi4`) require Hugging Face downloads / GPU unless cached; API models (`gpt4`, `gemini`, `grok`) need keys and network.

---

## 3. Other `nimby_bias_eval.py` modes (brief)

- `--mode synthetic` / `grounded` / `both` — build pairs in-script (not from `--pairs_file`).
- `--pairs_file` — JSONL of `PromptPair`-shaped objects.
- `--test_n N` — smoke test on first N pairs.
- `--scorer_type keyword` | `llm_judge` | `classifier` — for generation path (not used with `--prompt_only_preference`).
- `--preference_eval` — 1–5 judging of **model-generated** near/far outputs (API judges).
- Full help: `.venv/bin/python scripts/nimby_bias_eval.py --help`

---

## 4. Gold-standard minimal pairs (dataset-grounded statements)

Samples uniformly from gold CSVs (no keyword filter). Produces (for tag `gold_news200`):

- `gold_minimal_pairs_gold_news200.jsonl` — `near_prompt` / `far_prompt` for eval
- `gold_minimal_pairs_gold_news200.human.csv` / `gold_minimal_pairs_gold_news200.key.csv` — blinded layout if you add human raters later

If `--tag` is omitted, the script uses `{source}_{n}_{timestamp}`.

```bash
.venv/bin/python -u -B scripts/build_gold_minimal_pairs.py \
  --source news \
  --n 200 \
  --seed 42 \
  --out_dir output/nimby_bias_eval/human_vs_llm \
  --tag gold_news200
```

Then run `--prompt_only_preference` with `--pairs_file` pointing at the generated `.jsonl`.

---

## 5. Human text → LLM-altered text + blinded items

```bash
.venv/bin/python -u -B scripts/human_vs_llm_altered_bias_eval.py \
  --source news \
  --n 200 \
  --seed 42 \
  --alter_model gpt4 \
  --alter_mode neutral_rewrite \
  --out_dir output/nimby_bias_eval/human_vs_llm_altered \
  --tag news200_neutralrewrite
```

**Outputs:**

- `<tag>.pairs.csv` — `pair_id`, `text_original`, `text_altered`, optional `gold_nimby_original` from soft labels
- `<tag>.blinded_items.csv` — mixed originals + altered for annotation
- `<tag>.key.csv` — maps `item_id` → original vs altered
- `<tag>.meta.json`

---

## 6. GPT (or Gemini/Grok) relabel — no new human annotators

After you have `<tag>.pairs.csv`:

```bash
.venv/bin/python -u -B scripts/gpt_relabel_nimby_shift.py \
  --pairs_csv output/nimby_bias_eval/human_vs_llm_altered/news200_neutralrewrite.pairs.csv \
  --model gpt4 \
  --out_dir output/nimby_bias_eval/human_vs_llm_altered \
  --tag news200_neutralrewrite_judged
```

**Outputs:**

- `nimby_shift_<model>_<tag>.csv` — per-pair `gpt_nimby_original`, `gpt_nimby_altered`, delta
- `nimby_shift_<model>_<tag>.summary.json` — mean Δ, bootstrap CI, shares increase/decrease

If `--tag` is omitted, the suffix is a timestamp (see script).

Labeling order within each pair is randomized to reduce order effects.

---

## 7. Comparing to “human” without new annotations

- **Human baseline on originals:** use existing gold / soft labels (e.g. `output/annotation/soft_labels/<source>_soft_labels.csv`, column `not in my backyard`) — already merged into `pairs.csv` as `gold_nimby_original` when aligned by row index.

- **Model counterfactual (synthetic pairs):** 1–5 near-vs-far preference or NIMBY-scored generations — does not require human 1–5 on the same pairs.

For a fuller EMNLP-style discussion of what is and is not comparable, read `docs/emnlp_nimby_bias_methodology.md`.

---

## 8. `utils.py` note

`spacy` / `pydeidentify` are **optional imports** so scripts that only need `call_api_llm` / `get_model_config` can run without spaCy installed. De-identification utilities still require those packages when used.

---

## 9. Duplication vs main README

The main `README.md` contains a shorter **“NIMBY Bias Evaluation”** section at the end. This file (`new_pipeline_readme.md`) is the **single extended reference** for all NIMBY pipeline scripts, prompts, outputs, and the human-altered + GPT-relabel branch.
