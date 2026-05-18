# Multimodal Homelessness Dataset

This repository contains code and data for analyzing content (Reddit, X/Twitter, news articles, and meeting minutes) across multiple cities using both open-source and API-based large language models (LLMs) for classification and mitigation. Supported models include Llama 3.2, Qwen 2.5, Phi-4, GPT-4.1, Gemini 2.5 Pro, Grok-4, and fine-tuned BERT models.

## Setup

### 1. Create a Python Environment
You can use either `venv` (standard Python) or `conda` (Anaconda/Miniconda):

**Using venv:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Using conda:**
```bash
conda create -n venv python=3.10 -y
conda activate venv
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. (Optional, but recommended) Set Up API Keys for LLM APIs
If you want to use GPT-4.1, Gemini 2.5 Pro, or Grok-4, create a `.env` file in your project root:

```
OPENAI_API_KEY=your-openai-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
GROK_API_KEY=your-grok-api-key-here
```

**Never commit your API keys to version control.**

---

## Data Sources

The project now supports four main data sources:

### 1. Complete Datasets
All collected data is available in the `complete_dataset/` directory:
- `all_reddit_comments.csv` - Reddit comments across cities
- `all_twitter_posts.csv` - X/Twitter posts across cities  
- `all_newspaper_articles.csv` - News articles across cities
- `all_meeting_minutes.csv` - Meeting minutes across cities

### 2. Gold Standard Datasets
Human-annotated gold standard data is available in `gold_standard/`:
- `sampled_reddit_comments.csv` - Sampled Reddit comments for annotation
- `sampled_twitter_posts.csv` - Sampled X/Twitter posts for annotation
- `sampled_lexisnexis_news.csv` - Sampled news articles for annotation
- `sampled_meeting_minutes.csv` - Sampled meeting minutes for annotation

### 3. Annotation Data
Raw annotation scores and processed outputs are available:
- Raw scores: `annotation/{source}_raw_scores.csv` for each source
- Soft labels: `output/annotation/soft_labels/{source}_soft_labels.csv` for each source
- Agreement statistics: `output/annotation/agreement_stats.csv`

### 4. Census Data
County-level census data and clustering results are available in `census_data/`:
- State-level data: `all_states_2023.csv`, `all_states_2024.csv`
- County-level data with various clustering approaches
- Geographic boundary files and analysis scripts

---

## Model Setup

### 1. Download Required Models / API Setup
Download the following models from HuggingFace or use API-based LLMs:
- [Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [Qwen 2.5 7B Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Phi-4 Mini Instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- **GPT-4.1 (API, OpenAI)**
- **Gemini 2.5 Pro (API, Google)**
- **Grok-4 (API, xAI)**
- **BERT (Fine-tuned on annotation data)**

#### API Key Setup
To use GPT-4.1, Gemini 2.5 Pro, or Grok-4, you must provide API keys. The recommended way is to create a `.env` file in your project root with the following contents:

```
OPENAI_API_KEY=your-openai-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
GROK_API_KEY=your-grok-api-key-here
```

- Only set the keys for the APIs you plan to use.
- The code will automatically load this file using [python-dotenv](https://pypi.org/project/python-dotenv/). Install it with:
  ```bash
  pip install python-dotenv
  ```
- If `python-dotenv` is not installed, set the variables in your shell environment.
- **Never commit your API keys to version control.**

#### API Model Costs
The script estimates and prints API costs for each run (based on input/output tokens):

| Model         | Input Cost (per 1M tokens) | Output Cost (per 1M tokens) |
|--------------|----------------------------|-----------------------------|
| GPT-4.1      | $2.00                      | $8.00                       |
| Gemini 2.5   | $2.50                      | $15.00                      |
| Grok-4       | $3.00                      | $15.00                      |

---

## Annotation and Classification

### 1. Gold Standard / Soft Labeling
The human-annotated gold standard and soft label annotation data are available for all sources:
- Raw scores: `annotation/{source}_raw_scores.csv` for each source
- Processed outputs:
  - `output/annotation/agreement_stats.csv`
  - `output/annotation/soft_labels/{source}_soft_labels.csv` for each source

To generate these yourself:
```bash
python scripts/annotator_agreement.py
```

### 2. BERT Fine-tuning
Fine-tune BERT models on the annotation data for each source:

**Train BERT for a single source:**
```bash
python scripts/finetune_bert_simple.py --source reddit --epochs 3 --batch_size 16 --learning_rate 2e-5
python scripts/finetune_bert_simple.py --source x --epochs 3 --batch_size 16 --learning_rate 2e-5
python scripts/finetune_bert_simple.py --source news --epochs 3 --batch_size 16 --learning_rate 2e-5
python scripts/finetune_bert_simple.py --source meeting_minutes --epochs 3 --batch_size 16 --learning_rate 2e-5
```

**Train BERT for all sources:**
```bash
python scripts/run_bert_all_sources_simple.py
```

### 3. LoRA Fine-tuning on GPT Pseudolabels
Train local LLMs with LoRA using GPT pseudolabels. Use `val_opt` to optimize per-label thresholds on validation (recommended for publication results).

```bash
python scripts/finetune_on_gpt_pseudolabels.py --source reddit --model llama --use_lora --eval_threshold val_opt
python scripts/finetune_on_gpt_pseudolabels.py --source reddit --model qwen --use_lora --eval_threshold val_opt
python scripts/finetune_on_gpt_pseudolabels.py --source reddit --model phi4 --use_lora --eval_threshold val_opt
```

Notes:
- `--eval_threshold val_opt` uses per-label thresholds optimized on validation.
- `--eval_threshold fixed0p50` forces a global 0.5 threshold for all labels.

---

## Run All Models (One Command)

Run all transformer models (BERT, RoBERTa, ModernBERT) with and without SMOTE across all sources (reddit, x, news, meeting_minutes), and also run traditional baselines (Logistic Regression, SVM, Random Forest) without SMOTE.

```bash
python scripts/run_final_all_models_all_sources.py --also
```

Notes:
- Skips already-completed runs automatically by checking for metrics files at `nlp_outputs/{source}/{model}_{original|smote}_metrics.json`.
- Appends per-category rows after each transformer run to `nlp_outputs/all_transformer_results.csv` with columns: `category, country, train, val, test, synthetic, macro_f1, micro_f1, subset_accuracy, hamming_loss, precision, recall, f1, accuracy, roc_auc, average_precision`.
- Per-model transformer outputs per source:
  - CSV: `nlp_outputs/{source}/{model}_{original|smote}.csv`
  - JSON: `nlp_outputs/{source}/{model}_{original|smote}_metrics.json`
  - Weights: `models/final_{model}_best_{source}.pt`
- Traditional baselines outputs per source (overall comparisons):
  - `output/{source}/benchmark/benchmark_comparison_{source}.csv`
  - `output/{source}/benchmark/benchmark_summary_{source}.json`

---

## Classification and Mitigation

**Note:** The valid options for `--source` are: `reddit`, `x`, `news`, and `meeting_minutes`. The valid options for `--model` are: `llama`, `qwen`, `phi4`, `gpt4`, `gemini`, `grok`, and `bert`.

### 3. Classification
You can classify comments using any of the supported models, including API-based LLMs and fine-tuned BERT. For API models, you can use the `--test` flag to process only 10 comments and see the estimated cost:

**Zero-shot (default):**
```bash
python scripts/classify_comments.py --model llama --source reddit --dataset gold_subset
python scripts/classify_comments.py --model qwen --source reddit --dataset gold_subset
python scripts/classify_comments.py --model phi4 --source reddit --dataset gold_subset
python scripts/classify_comments.py --model gpt4 --source reddit --dataset gold_subset --test
python scripts/classify_comments.py --model gemini --source reddit --dataset gold_subset --test
python scripts/classify_comments.py --model grok --source reddit --dataset gold_subset --test
python scripts/classify_comments.py --model bert --source reddit --dataset gold_subset
```

**Few-shot (with examples):**
```bash
python scripts/classify_comments.py --model llama --source reddit --dataset gold_subset --few_shot reddit
python scripts/classify_comments.py --model qwen --source reddit --dataset gold_subset --few_shot reddit
python scripts/classify_comments.py --model phi4 --source reddit --dataset gold_subset --few_shot reddit
python scripts/classify_comments.py --model gpt4 --source reddit --dataset gold_subset --few_shot reddit --test
python scripts/classify_comments.py --model gemini --source reddit --dataset gold_subset --few_shot reddit --test
python scripts/classify_comments.py --model grok --source reddit --dataset gold_subset --few_shot reddit --test
python scripts/classify_comments.py --model bert --source reddit --dataset gold_subset --few_shot reddit
```

- The `--model` argument specifies which model to use (`llama`, `qwen`, `phi4`, `gpt4`, `gemini`, `grok`, or `bert`).
- The `--source` argument specifies the data source (`reddit`, `x`, `news`, `meeting_minutes`).
- The `--dataset` argument specifies which dataset to use (`all`, `gold_subset`).
- The `--few_shot` argument appends five few-shot examples to the end of the prompt. Supported values: `reddit`, `x`, `news`, `meeting_minutes`.
- If `--few_shot` is not specified, zero-shot classification is used.
- The output will be saved to `output/{source}/{model}/classified_comments_{source}_{dataset}_{model}_{few_shot}_flags.csv`.
- You can override the input or output file with `--input` and `--output` arguments.
- The script will automatically load your API keys from `.env` if present.
- The `--test` flag is recommended for API models to avoid unexpected costs.

### 4. Mitigation
To mitigate and reclassify comments using any of the supported models (mitigation always includes reclassification):
```bash
python scripts/mitigate_comments.py --model llama --source reddit --dataset gold_subset
python scripts/mitigate_comments.py --model qwen --source reddit --dataset gold_subset
python scripts/mitigate_comments.py --model phi4 --source reddit --dataset gold_subset
python scripts/mitigate_comments.py --model gpt4 --source reddit --dataset gold_subset --test
python scripts/mitigate_comments.py --model gemini --source reddit --dataset gold_subset --test
python scripts/mitigate_comments.py --model grok --source reddit --dataset gold_subset --test
python scripts/mitigate_comments.py --model bert --source reddit --dataset gold_subset
```

---

## Analysis

### 5. Comprehensive F1 Analysis
Generate comprehensive F1 score analysis across all models and sources:

```bash
python scripts/comprehensive_f1_analysis.py
```

This script:
- Loads soft labels and model predictions for all sources (soft-label evaluation)
- Calculates macro F1, precision, and recall for each model
- Generates LaTeX tables for publication
- Creates detailed comparison tables
- Outputs results to `output/f1/soft/` directory

### 6. Zero/Few/LoRA Comparison (Soft Labels + Gold)
Generate consolidated CSV summaries for zero-shot, few-shot, and LoRA results:

```bash
python scripts/compare_zero_few_lora_results.py --soft_label_threshold 0.5
```

Outputs:
- `output/summary/zero_few_lora_comparison.csv`
- `output/summary/zero_few_lora_per_category.csv`

### 7. Val-Opt Tables (Gold + Val-Opt LoRA)
Create LaTeX tables and CSVs under `output/f1/val_opt/` using val-opt LoRA rows plus zero/few-shot results:

```bash
python scripts/generate_valopt_f1_tables.py
```

### 7.1 One-Command F1 Tables (Soft + Val-Opt)
Run both the soft-label and validation-optimized table generation in one step:

```bash
python scripts/run_all_f1_tables.py
```

### 8. Research Chart Generation
Create publication-quality charts for city analysis across all data sources:

```bash
python scripts/create_research_charts.py
```

This script:
- Generates charts for each data source (Reddit, X, News, Meeting Minutes)
- Creates combined analysis across all sources
- Produces city size comparisons and category correlations
- Creates a weighted confusion matrix that accounts for different post volumes across sources
- Generates a comprehensive 16x16 category relationship matrix showing correlations between all classification categories
- Outputs publication-ready PDFs to `output/charts/gpt_research_analysis/`
- Creates annotation agreement charts in `output/charts/`

### 9. Soft Label Evaluation
Evaluate model performance against soft labels:

```bash
python scripts/evaluate_soft_labels.py
```

This script compares model predictions against the soft label gold standard and generates detailed performance metrics.

### 10. Statistics and Visualization
All statistics and charts are available in the `output/charts/` directory.

To generate these yourself:
```bash
python scripts/calculate_intercoder_reliability.py
```

---

## Output Structure

The project generates extensive outputs organized by:

### Model Outputs
- `output/{source}/{model}/` - Classification results for each model and source
- `output/{source}/bert/` - BERT fine-tuning results and metrics

### Analysis Outputs
- `output/f1/soft/` - Soft-label F1 tables and LaTeX files
- `output/f1/val_opt/` - Val-opt F1 tables and LaTeX files
- `output/charts/gpt_research_analysis/` - Publication-quality research charts
- `output/charts/` - Annotation agreement charts
- `output/annotation/` - Agreement statistics and soft labels

### Data Organization
- `complete_dataset/` - All collected data across sources
- `gold_standard/` - Human-annotated gold standard data
- `annotation/` - Raw annotation scores
- `census_data/` - Geographic and demographic data

---

## Data Collection

### Reddit Data Collection
Run the following script to collect Reddit data:
```bash
python scripts/get_reddit_data.py
```

**Note:** You'll need to:
- Replace `CLIENT_ID`, `CLIENT_SECRET`, and `USER_AGENT` with your Reddit API credentials
- Specify your target subreddit name
- The script outputs 3 CSVs in `data/<city>/reddit/`:
  - `all_comments.csv` (not included due to identifiable information)
  - `filtered_comments.csv` (not included due to identifiable information)
  - `statistics.csv` (included)

After data collection, run:
```bash
python scripts/random_reddit_sample.py
```
This generates a random set of 50 Reddit comments per city.

### Gold Subset Deidentified Data (for LLM Evaluation)
The gold subset deidentified dataset (500 comments total, 50 from each of 10 cities) is available at:
[`output/gold_subset_reddit_comments_by_city_deidentified.csv`](output/gold_subset_reddit_comments_by_city_deidentified.csv)

**Note:** This file is used for LLM evaluation and is NOT the human-annotated gold standard. The true human gold standard is stored separately (see annotation section above).

To generate this yourself:
```bash
python scripts/deidentify_comments.py
```

---

## NIMBY Bias Evaluation (Near vs Far)

This repo includes scripts to evaluate whether LLM behavior shifts toward **“Not in My Backyard” (NIMBY)** framing when an intervention is described as **nearby** vs **far away**.

### Generate synthetic near/far prompt pairs with Claude Code (resumable)

Prompt templates live in `.claude/prompts/`:
- `generate_synthetic_nimby_pairs.md`: request/task-style prompts
- `generate_synthetic_nimby_pairs_statement.md`: **statement-only** prompts (no questions, no “write/draft/provide”, no `?`)
- `_nimby_pairs_common.md`: shared constraints (including required homelessness keywords)

Generate 500 **statement-style** pairs (chunked + resumable):

```bash
.venv/bin/python -u -B scripts/generate_synthetic_nimby_pairs_claude.py \
  --n 500 \
  --chunk_size 100 \
  --resume \
  --claude_model haiku \
  --prompt_file .claude/prompts/generate_synthetic_nimby_pairs_statement.md \
  --out output/nimby_bias_eval/synth_pairs_claude_statement.jsonl
```

### Prompt-only preference evaluation (1–5 scale) across 6 LLMs

Given a JSONL of near/far pairs (e.g., the statement file above), each model rates **A vs B** on a 1–5 scale; A/B order is randomized by `--seed`, and results are mapped back to a **near-vs-far** 1–5 scale:
- 1 = prefer **near**
- 3 = neutral
- 5 = prefer **far**

Run on 500 statement pairs:

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

Outputs:
- `output/nimby_bias_eval/statement_pref_<model>_<tag>.csv`
- `output/nimby_bias_eval/statement_pref_<model>_<tag>.summary.json`
- `output/nimby_bias_eval/statement_pref_ALL_<tag>.csv`