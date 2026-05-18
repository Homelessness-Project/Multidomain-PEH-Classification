## EMNLP methodology note: LLM “NIMBY shift” + comparison to human-labeled NIMBY

### Why this is a strong EMNLP-style bias methodology
Many “bias” evaluations in NLP conflate (i) the **base rate** of a label in a dataset with (ii) the model’s **counterfactual sensitivity** to a protected or decision-relevant attribute. For *Not-In-My-Backyard* (NIMBY) framing, the key attribute is **proximity**: opposition that increases when a homelessness intervention is framed as *near me/us* rather than *far away*.

This methodology targets bias as a **within-item counterfactual shift** (for models):
\[
\Delta = s(\text{near}) - s(\text{far}),
\]
where \(s(\cdot)\) is a scalar outcome derived from model judgments. Because the near and far versions are identical except for proximity framing, \(\Delta\) is interpretable as the effect of proximity under a controlled perturbation—an evaluation pattern common in EMNLP bias/robustness work.

### Key design principles
- **Counterfactual control**: near/far statements differ only by proximity phrasing; everything else is held constant.
- **Blinding**: raters see A/B alternatives without being told which is near/far.
- **Paired analysis**: use paired statistics over minimal pairs (more power, lower variance, clearer causal attribution).
- **Ecological validity**: prompts are *dataset-grounded* (seed excerpts from your gold standard) rather than purely synthetic.
- **Two complementary comparisons**:
  - **Counterfactual sensitivity (models)** via near-vs-far minimal pairs (synthetic or dataset-grounded).
  - **Human-labeled baseline (dataset)** by comparing model judgments on *original gold-standard items* to existing human labels for `not in my backyard`.

### Two evaluation tracks
#### Track A: Counterfactual minimal pairs (near vs far) — model-only
We build a set of \(N\) pairs from the gold standard sources (Reddit, X, news, meeting minutes). Each pair contains:
- a short **deidentified excerpt** from the gold standard (context)
- a **near** statement: intervention *near the speaker*
- a **far** statement: same intervention *far away*

In this repository, `scripts/build_gold_minimal_pairs.py` creates these statement-only pairs and exports a PromptPair JSONL for LLM evaluation.

Important constraint: **If you do not collect new human ratings**, Track A is a *model-only* counterfactual analysis (humans are not part of \(\Delta\)).

#### Track B: Human-labeled NIMBY in the gold standard — model vs humans (non-counterfactual)
Separately, you can compare model judgments to **existing human labels** on the original gold standard:
- humans provide a binary/multilabel annotation including `not in my backyard`
- models can be asked to label the same original items (or you can apply a trained classifier)

This answers: “Does the model’s NIMBY labeling behavior match the human-labeled gold standard?” but it is **not a near-vs-far counterfactual test**.

### Outcome measures
#### Track A (counterfactual, model-only): prompt-only preference mapped to near-vs-far
Each *model* outputs an integer \(r \in \{1,2,3,4,5\}\) preferring A vs B. A/B order is randomized, then mapped back to the near-vs-far axis:
- **1** = strong preference for **near**
- **3** = neutral
- **5** = strong preference for **far**

This is a direct measure of “NIMBY shift” as *preference for distance* under a controlled proximity perturbation.

If you later collect human A/B ratings, the exact same mapping enables a direct human-vs-model comparison on the same pairs.

#### Track B (non-counterfactual, human-labeled): agreement with gold labels
Use the existing human label \(y \in \{0,1\}\) for `not in my backyard` on each gold item and compare against a model-produced score \(\hat{y}\) (binary or probabilistic).
Report:
- F1 / precision / recall (and per-source breakdown)
- calibration / prevalence gap (optional)

#### Secondary outcomes (optional, EMNLP-friendly additions)
- **Binary far preference**: \(\mathbb{1}[r_{near/far} > 3]\)
- **Calibration to human labels** (if desired): compare against gold `not in my backyard` labels on the same excerpts (note: this is not counterfactual; treat as a separate analysis).

### Statistical analysis plan
#### Track A (paired, counterfactual; models)
Report for each model:
- mean \( \bar{r}_{near/far} \) and median
- share preferring far / near / neutral
- **bootstrap 95% CI** for the mean (resampling over pairs)

#### Track B (gold-label agreement; models vs humans)
Report per model:
- F1 / precision / recall on the human-labeled `not in my backyard` label
- optionally stratify by source (reddit/news/meeting minutes/X) or by intervention keywords

### Threats to validity and mitigation
- **Position bias**: mitigated by A/B randomization.
- **Demand effects**: mitigated by blinding (no “near/far” labels in the survey).
- **Prompt confounds**: ensure near/far differ only in the proximity phrase; avoid changing capacity, services, or governance details.
- **Excerpt leakage**: avoid long verbatim copyrighted excerpts in public artifacts; keep excerpts short and deidentified (this repo uses deidentified fields).

### How to run (repo commands)
#### Track A: build gold minimal pairs and evaluate model preference
Example:

```bash
.venv/bin/python -u -B scripts/build_gold_minimal_pairs.py \
  --source news \
  --n 200 \
  --seed 42 \
  --out_dir output/nimby_bias_eval/human_vs_llm \
  --tag gold_news200
```

Then run prompt-only preference evaluation across models:
```bash
.venv/bin/python -u -B scripts/nimby_bias_eval.py \
  --pairs_file output/nimby_bias_eval/human_vs_llm/gold_minimal_pairs_gold_news200.jsonl \
  --prompt_only_preference \
  --llm_models llama qwen phi4 gpt4 gemini grok \
  --llm_max_new_tokens 5 \
  --seed 42 \
  --out_dir output/nimby_bias_eval/human_vs_llm \
  --tag llm_on_gold_news200
```

### How this fits into an EMNLP paper section
You can describe this as:
- **A counterfactual minimal-pair evaluation** for proximity-driven exclusionary framing (NIMBY), measured as distance preference on near-vs-far statement pairs (Track A).
- **A human-labeled validity check** where models’ NIMBY labeling on original gold items is compared to existing human annotations (Track B).

In an EMNLP “Ethics / Limitations” section, note that:
- NIMBY concerns can include legitimate safety/resource considerations; the metric quantifies *systematic proximity sensitivity*, not moral correctness.
- Deidentification and excerpt length control reduce privacy/copyright risk.

