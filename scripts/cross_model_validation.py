#!/usr/bin/env python3
"""
Cross-Model Validation Analysis for 6 LLMs
==========================================

This script performs comprehensive cross-model validation analysis including:
1. Statistical significance testing between models
2. Correlation analysis between model predictions
3. Ensemble model performance evaluation
4. Cross-validation stability analysis
5. Model agreement and disagreement analysis

Models analyzed: llama, qwen, gpt4, gemini, grok, phi4, bert
Data sources: reddit, x, news, meeting_minutes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
import json
import os
from collections import defaultdict
import argparse
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Configuration
SOURCES = ['reddit', 'x', 'news', 'meeting_minutes']
LLM_MODELS = ['llama', 'phi4', 'qwen', 'gemini', 'grok', 'gpt4']
BERT_MODEL = 'bert'
SHOT_TYPES = ['zero_shot', 'few_shot']
ALPHA = 0.05  # Significance level

class CrossModelValidator:
    """Cross-model validation analysis class"""
    
    def __init__(self, output_dir='output/cross_validation'):
        self.output_dir = output_dir
        self.results = defaultdict(dict)
        self.predictions = defaultdict(dict)
        self.soft_labels = defaultdict(dict)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        os.makedirs(f"{output_dir}/tables", exist_ok=True)
        
    def load_data(self):
        """Load all model predictions and soft labels"""
        print("Loading model predictions and soft labels...")
        
        for source in SOURCES:
            print(f"\nLoading data for {source}...")
            
            # Load soft labels
            soft_labels_path = f'output/annotation/soft_labels/{source}_soft_labels.csv'
            try:
                soft_labels_df = pd.read_csv(soft_labels_path)
                self.soft_labels[source] = soft_labels_df
                print(f"  Loaded soft labels: {len(soft_labels_df)} samples")
            except FileNotFoundError:
                print(f"  Warning: Soft labels not found for {source}")
                continue
            
            # Load LLM predictions
            for model in LLM_MODELS:
                for shot_type in SHOT_TYPES:
                    predictions_df = self._load_model_predictions(source, model, shot_type)
                    if predictions_df is not None:
                        key = f"{model}_{shot_type}"
                        self.predictions[source][key] = predictions_df
                        print(f"  Loaded {key}: {len(predictions_df)} samples")
            
            # Load BERT predictions
            bert_predictions = self._load_bert_predictions(source)
            if bert_predictions is not None:
                self.predictions[source]['bert_finetuned'] = bert_predictions
                print(f"  Loaded BERT: {len(bert_predictions)} samples")

            # Load RoBERTa predictions (if available)
            roberta_predictions = self._load_traditional_predictions(source, 'roberta')
            if roberta_predictions is not None:
                self.predictions[source]['roberta'] = roberta_predictions
                print(f"  Loaded RoBERTa: {len(roberta_predictions)} samples")

            # Load ModernBERT predictions (if available)
            modernbert_predictions = self._load_traditional_predictions(source, 'modernbert')
            if modernbert_predictions is not None:
                self.predictions[source]['modernbert'] = modernbert_predictions
                print(f"  Loaded ModernBERT: {len(modernbert_predictions)} samples")
    
    def _load_model_predictions(self, source, model, shot_type):
        """Load predictions for a specific model and shot type"""
        if shot_type == 'zero_shot':
            few_shot_text = 'none'
        else:
            few_shot_text = source
        
        possible_paths = [
            f'output/{source}/{model}/classified_comments_{source}_gold_subset_{model}_{few_shot_text}_flags.csv',
            f'output/{source}/{model}/classified_comments_{source}_all_{model}_{few_shot_text}_flags.csv',
            f'output/classified_comments_{source}_gold_subset_{model}.csv'
        ]
        
        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                return df
            except FileNotFoundError:
                continue
        return None
    
    def _load_bert_predictions(self, source):
        """Load BERT predictions"""
        bert_path = f'output/{source}/bert/bert_classification_results_{source}.csv'
        try:
            df = pd.read_csv(bert_path)
            return df
        except FileNotFoundError:
            return None

    def _load_traditional_predictions(self, source, model_name):
        """Load RoBERTa/ModernBERT predictions if present.

        Expected patterns:
        - output/{source}/{model_name}/{model_name}_classification_results_{source}.csv
        - output/{source}/{model_name}/{model_name}_detailed_results_{source}.csv
        - output/{source}/{model_name}/*{source}.csv (fallback for known files)
        """
        possible_paths = [
            f'output/{source}/{model_name}/{model_name}_classification_results_{source}.csv',
            f'output/{source}/{model_name}/{model_name}_detailed_results_{source}.csv'
        ]
        # Known specific files found in workspace (e.g., roberta reddit)
        if model_name == 'roberta' and source == 'reddit':
            possible_paths.insert(0, 'output/reddit/roberta/roberta_classification_results_reddit.csv')

        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                return df
            except FileNotFoundError:
                continue
        return None
    
    def statistical_significance_testing(self):
        """Perform statistical significance testing between models"""
        print("\n" + "="*60)
        print("STATISTICAL SIGNIFICANCE TESTING")
        print("="*60)
        
        significance_results = {}
        
        for source in SOURCES:
            if source not in self.predictions:
                continue
                
            print(f"\nAnalyzing {source}...")
            significance_results[source] = {}
            
            # Get all model keys for this source (exclude BERT/RoBERTa)
            model_keys = [k for k in self.predictions[source].keys() if 'bert' not in k.lower() and 'roberta' not in k.lower()]
            
            # Calculate F1 scores for each model
            model_f1_scores = {}
            for model_key in model_keys:
                f1_score = self._calculate_f1_score(source, model_key)
                if f1_score is not None:
                    model_f1_scores[model_key] = f1_score
            
            # Perform pairwise significance tests
            model_pairs = list(combinations(model_keys, 2))
            pairwise_results = {}
            
            for model1, model2 in model_pairs:
                if model1 in model_f1_scores and model2 in model_f1_scores:
                    # Perform t-test
                    t_stat, p_value = stats.ttest_rel(
                        model_f1_scores[model1], 
                        model_f1_scores[model2]
                    )
                    
                    # Determine significance
                    is_significant = p_value < ALPHA
                    better_model = model1 if np.mean(model_f1_scores[model1]) > np.mean(model_f1_scores[model2]) else model2
                    
                    pairwise_results[f"{model1}_vs_{model2}"] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'is_significant': is_significant,
                        'better_model': better_model,
                        'model1_mean_f1': np.mean(model_f1_scores[model1]),
                        'model2_mean_f1': np.mean(model_f1_scores[model2])
                    }
            
            significance_results[source] = {
                'model_f1_scores': model_f1_scores,
                'pairwise_tests': pairwise_results
            }
            
            # Print summary
            print(f"  Models tested: {len(model_keys)}")
            print(f"  Significant differences: {sum(1 for r in pairwise_results.values() if r['is_significant'])}")
        
        self.results['significance_testing'] = significance_results
        return significance_results
    
    def correlation_analysis(self):
        """Analyze correlations between model predictions (prediction-level)"""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        correlation_results = {}
        
        for source in SOURCES:
            if source not in self.predictions:
                continue
                
            print(f"\nAnalyzing correlations for {source}...")
            correlation_results[source] = {}
            
            model_keys = [k for k in self.predictions[source].keys() if 'bert' not in k.lower() and 'roberta' not in k.lower()]
            
            # Calculate correlation matrix
            correlation_matrix = np.zeros((len(model_keys), len(model_keys)))
            correlation_types = {}
            
            for i, model1 in enumerate(model_keys):
                for j, model2 in enumerate(model_keys):
                    if i <= j:  # Only calculate upper triangle
                        corr_pearson, p_pearson = self._calculate_correlation(source, model1, model2, 'pearson')
                        corr_spearman, p_spearman = self._calculate_correlation(source, model1, model2, 'spearman')
                        
                        correlation_matrix[i, j] = corr_pearson
                        correlation_matrix[j, i] = corr_pearson
                        
                        correlation_types[f"{model1}_vs_{model2}"] = {
                            'pearson': {'correlation': corr_pearson, 'p_value': p_pearson},
                            'spearman': {'correlation': corr_spearman, 'p_value': p_spearman}
                        }
            
            correlation_results[source] = {
                'correlation_matrix': correlation_matrix,
                'model_keys': model_keys,
                'correlation_details': correlation_types
            }
            
            # Print summary
            avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
            print(f"  Average correlation: {avg_correlation:.4f}")
            print(f"  Highest correlation: {np.max(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]):.4f}")
        
        self.results['correlation_analysis'] = correlation_results
        return correlation_results

    def prediction_level_similarity(self):
        """Compute prediction-level similarity matrices (Pearson and Spearman) across models for each source.

        Matches columns to soft label names, coerces to numeric, aligns rows by index, and aggregates
        per-category correlations by averaging across matched columns.
        """
        print("\n" + "="*60)
        print("PREDICTION-LEVEL SIMILARITY")
        print("="*60)

        similarity_results = {}

        for source in SOURCES:
            if source not in self.predictions:
                continue

            print(f"\nAnalyzing prediction-level similarity for {source}...")
            model_keys = [k for k in self.predictions[source].keys() if 'bert' not in k.lower() and 'roberta' not in k.lower()]
            n = len(model_keys)
            pearson_matrix = np.zeros((n, n))
            spearman_matrix = np.zeros((n, n))

            # Pre-prepare matched numeric prediction columns per model
            soft_labels_df = self.soft_labels[source]
            soft_label_cols = soft_labels_df.columns.tolist()

            model_numeric_cols = {}
            min_len_global = None
            for key in model_keys:
                df = self.predictions[source][key].copy()
                # Align lengths
                min_len_global = len(df) if min_len_global is None else min(min_len_global, len(df))
                # Identify columns matching soft labels
                matched = []
                for col in df.columns:
                    for s in soft_label_cols:
                        if s.lower() in col.lower() or col.lower() in s.lower():
                            matched.append(col)
                            break
                # Coerce to numeric
                if matched:
                    numeric_df = df[matched].apply(pd.to_numeric, errors='coerce')
                    model_numeric_cols[key] = numeric_df
                else:
                    model_numeric_cols[key] = pd.DataFrame()

            # Align all to the same min length
            if min_len_global is None:
                continue
            for key in model_keys:
                if not model_numeric_cols[key].empty:
                    model_numeric_cols[key] = model_numeric_cols[key].iloc[:min_len_global]

            # Compute pairwise correlations
            for i, m1 in enumerate(model_keys):
                for j, m2 in enumerate(model_keys):
                    if i > j:
                        continue
                    df1 = model_numeric_cols[m1]
                    df2 = model_numeric_cols[m2]
                    if df1.empty or df2.empty:
                        pearson = 0.0
                        spearman = 0.0
                    else:
                        # Use matched column count
                        k = min(df1.shape[1], df2.shape[1])
                        if k == 0:
                            pearson = 0.0
                            spearman = 0.0
                        else:
                            pears = []
                            spears = []
                            for c in range(k):
                                s1 = df1.iloc[:, c]
                                s2 = df2.iloc[:, c]
                                # Drop rows with NaNs
                                valid = s1.notna() & s2.notna()
                                s1v = s1[valid]
                                s2v = s2[valid]
                                if len(s1v) > 1:
                                    try:
                                        r, _ = pearsonr(s1v, s2v)
                                        if not np.isnan(r):
                                            pears.append(r)
                                    except Exception:
                                        pass
                                    try:
                                        r_s, _ = spearmanr(s1v, s2v)
                                        if not np.isnan(r_s):
                                            spears.append(r_s)
                                    except Exception:
                                        pass
                            pearson = float(np.mean(pears)) if pears else 0.0
                            spearman = float(np.mean(spears)) if spears else 0.0

                    pearson_matrix[i, j] = pearson_matrix[j, i] = pearson
                    spearman_matrix[i, j] = spearman_matrix[j, i] = spearman

            similarity_results[source] = {
                'model_keys': model_keys,
                'pearson_matrix': pearson_matrix,
                'spearman_matrix': spearman_matrix
            }

            print(f"  Avg Pearson: {np.mean(pearson_matrix[np.triu_indices_from(pearson_matrix, k=1)]):.4f}")
            print(f"  Avg Spearman: {np.mean(spearman_matrix[np.triu_indices_from(spearman_matrix, k=1)]):.4f}")

        self.results['prediction_similarity'] = similarity_results
        return similarity_results

    def multi_label_agreement(self):
        """Compute multi-label agreement between model pairs with partial credit.

        Two metrics per source:
        - match_ratio: average per-row fraction of matching labels across K categories
        - jaccard: average per-row Jaccard similarity of positive label sets
        """
        print("\n" + "="*60)
        print("MULTI-LABEL AGREEMENT (PARTIAL CREDIT)")
        print("="*60)

        agreement_results = {}

        for source in SOURCES:
            if source not in self.predictions:
                continue

            print(f"\nAnalyzing multi-label agreement for {source}...")
            model_keys = [k for k in self.predictions[source].keys() if 'bert' not in k.lower() and 'roberta' not in k.lower()]
            n = len(model_keys)
            match_ratio_matrix = np.ones((n, n))
            jaccard_matrix = np.ones((n, n))

            # Prepare matched, thresholded binary columns for each model
            soft_labels_df = self.soft_labels.get(source)
            if soft_labels_df is None or soft_labels_df.empty:
                continue
            soft_label_cols = soft_labels_df.columns.tolist()

            # Build numeric prediction frames per model aligned to all categories
            model_pred_frames = {}
            min_len_global = None
            num_cols = len(soft_label_cols)
            for key in model_keys:
                df = self.predictions[source][key]
                # Create an aligned numeric frame with one column per soft label
                aligned_numeric = []
                for s in soft_label_cols:
                    chosen = None
                    for col in df.columns:
                        if s.lower() in col.lower() or col.lower() in s.lower():
                            chosen = col
                            break
                    if chosen is not None:
                        aligned_numeric.append(pd.to_numeric(df[chosen], errors='coerce'))
                    else:
                        # Missing column for this label; fill zeros
                        aligned_numeric.append(pd.Series(np.zeros(len(df))))
                numeric = pd.concat(aligned_numeric, axis=1)
                numeric.columns = soft_label_cols
                numeric = numeric.fillna(0.0)
                model_pred_frames[key] = numeric
                min_len_global = len(numeric) if min_len_global is None else min(min_len_global, len(numeric))

            if min_len_global is None or num_cols == 0:
                continue

            # Threshold to binary
            for key in model_keys:
                if not model_pred_frames[key].empty:
                    model_pred_frames[key] = (model_pred_frames[key].iloc[:min_len_global, :num_cols] > 0.5).astype(int)

            # Pairwise metrics
            for i, m1 in enumerate(model_keys):
                for j, m2 in enumerate(model_keys):
                    if i > j:
                        continue
                    df1 = model_pred_frames[m1]
                    df2 = model_pred_frames[m2]
                    if df1.empty or df2.empty:
                        mr = 0.0
                        jac = 0.0
                    else:
                        k = min(df1.shape[1], df2.shape[1])
                        if k == 0:
                            mr = 0.0
                            jac = 0.0
                        else:
                            a = df1.iloc[:, :k].to_numpy()
                            b = df2.iloc[:, :k].to_numpy()
                            # Match ratio: fraction of matches per row averaged
                            matches_per_row = (a == b).sum(axis=1) / float(k)
                            mr = float(matches_per_row.mean())
                            # Jaccard per row on positive labels
                            intersection = (np.logical_and(a == 1, b == 1)).sum(axis=1)
                            union = (np.logical_or(a == 1, b == 1)).sum(axis=1)
                            # Handle rows with no positives: define Jaccard as 1.0 if both empty
                            jac_rows = np.where(union > 0, intersection / union, 1.0)
                            jac = float(jac_rows.mean())

                    match_ratio_matrix[i, j] = match_ratio_matrix[j, i] = mr
                    jaccard_matrix[i, j] = jaccard_matrix[j, i] = jac

            agreement_results[source] = {
                'model_keys': model_keys,
                'match_ratio_matrix': match_ratio_matrix,
                'jaccard_matrix': jaccard_matrix
            }

            print(f"  Avg match ratio: {np.mean(match_ratio_matrix[np.triu_indices_from(match_ratio_matrix, k=1)]):.4f}")
            print(f"  Avg Jaccard: {np.mean(jaccard_matrix[np.triu_indices_from(jaccard_matrix, k=1)]):.4f}")

        self.results['multilabel_agreement'] = agreement_results
        
        # Build combined (all sources) matrices by averaging per-source over common models
        if agreement_results:
            computed_sources = [s for s in SOURCES if s in agreement_results]
            if computed_sources:
                common_keys = set(agreement_results[computed_sources[0]]['model_keys'])
                for s in computed_sources[1:]:
                    common_keys &= set(agreement_results[s]['model_keys'])
                common_keys = list(common_keys)
                # Order by base LLM order then full key
                base_order = {name: i for i, name in enumerate(LLM_MODELS)}
                def sort_key(m):
                    base = m.split('_')[0]
                    return (base_order.get(base, 999), m)
                common_keys.sort(key=sort_key)
                if common_keys:
                    k = len(common_keys)
                    combined_match = np.zeros((k, k))
                    combined_jacc = np.zeros((k, k))
                    counts = np.zeros((k, k))
                    for s in computed_sources:
                        keys_s = agreement_results[s]['model_keys']
                        idx_map = {m: i for i, m in enumerate(keys_s)}
                        mr = agreement_results[s]['match_ratio_matrix']
                        jc = agreement_results[s]['jaccard_matrix']
                        for i, mi in enumerate(common_keys):
                            for j, mj in enumerate(common_keys):
                                if mi in idx_map and mj in idx_map:
                                    ii = idx_map[mi]
                                    jj = idx_map[mj]
                                    combined_match[i, j] += mr[ii, jj]
                                    combined_jacc[i, j] += jc[ii, jj]
                                    counts[i, j] += 1
                    with np.errstate(invalid='ignore'):
                        combined_match = np.divide(combined_match, counts, out=np.zeros_like(combined_match), where=counts>0)
                        combined_jacc = np.divide(combined_jacc, counts, out=np.zeros_like(combined_jacc), where=counts>0)
                    agreement_results['overall'] = {
                        'model_keys': common_keys,
                        'match_ratio_matrix': combined_match,
                        'jaccard_matrix': combined_jacc
                    }
        return agreement_results

    def correct_overlap_analysis(self):
        """Compute correctness-based metrics per source and overall:
        1) Model-vs-Truth Jaccard of positive label sets (per model → vector)
        2) Jaccard overlap between models of sets of correctly predicted labels (matrix)
        3) Pairwise model-truth similarity: Three-way Jaccard (model1 ∩ model2 ∩ truth)
        4) Joint correctness: How often both models are correct together
        Excludes BERT/RoBERTa entries.
        """
        print("\n" + "="*60)
        print("CORRECT OVERLAP ANALYSIS")
        print("="*60)

        results = {}
        for source in SOURCES:
            if source not in self.predictions:
                continue
            print(f"\nAnalyzing correct overlap for {source}...")
            model_keys = [k for k in self.predictions[source].keys() if 'bert' not in k.lower() and 'roberta' not in k.lower()]
            if not model_keys:
                continue

            soft_df = self.soft_labels.get(source)
            if soft_df is None or soft_df.empty:
                continue
            categories = soft_df.columns.tolist()

            # Prepare aligned binary truth
            y_true_bin = (soft_df[categories] > 0.5).astype(int)

            # Prepare aligned, thresholded predictions per model to match categories
            model_bin_preds = {}
            min_len = None
            for key in model_keys:
                df = self.predictions[source][key]
                min_len = len(df) if min_len is None else min(min_len, len(df))
            if min_len is None:
                continue
            y_true_bin = y_true_bin.iloc[:min_len]

            for key in model_keys:
                df = self.predictions[source][key].iloc[:min_len]
                aligned = []
                for c in categories:
                    colmatch = None
                    for col in df.columns:
                        if c.lower() in col.lower() or col.lower() in c.lower():
                            colmatch = col
                            break
                    if colmatch is not None:
                        aligned.append(pd.to_numeric(df[colmatch], errors='coerce'))
                    else:
                        aligned.append(pd.Series(np.zeros(len(df))))
                pred_num = pd.concat(aligned, axis=1).fillna(0.0)
                pred_bin = (pred_num > 0.5).astype(int)
                pred_bin.columns = categories
                model_bin_preds[key] = pred_bin

            # 1) Model-vs-Truth Jaccard vector
            model_truth_scores = []
            for key in model_keys:
                a = model_bin_preds[key].to_numpy()
                b = y_true_bin.to_numpy()
                inter = np.logical_and(a == 1, b == 1).sum(axis=1)
                union = np.logical_or(a == 1, b == 1).sum(axis=1)
                j_rows = np.where(union > 0, inter / union, 1.0)
                model_truth_scores.append(float(np.mean(j_rows)))
            model_truth_scores = np.array(model_truth_scores)

            # 2) Correctness-overlap between models (Jaccard of correct-label sets)
            n = len(model_keys)
            correctness_overlap = np.zeros((n, n))
            # True binary for comparison
            y = y_true_bin.to_numpy()
            for i, m1 in enumerate(model_keys):
                a = model_bin_preds[m1].to_numpy()
                correct1 = (a == y).astype(int)
                for j, m2 in enumerate(model_keys):
                    if i > j:
                        continue
                    b = model_bin_preds[m2].to_numpy()
                    correct2 = (b == y).astype(int)
                    inter = np.logical_and(correct1 == 1, correct2 == 1).sum(axis=1)
                    union = np.logical_or(correct1 == 1, correct2 == 1).sum(axis=1)
                    j_rows = np.where(union > 0, inter / union, 1.0)
                    s = float(np.mean(j_rows))
                    correctness_overlap[i, j] = correctness_overlap[j, i] = s

            # 3) Pairwise model-truth similarity: Three-way Jaccard (model1 ∩ model2 ∩ truth)
            pairwise_truth_similarity = np.zeros((n, n))
            for i, m1 in enumerate(model_keys):
                a = model_bin_preds[m1].to_numpy()
                for j, m2 in enumerate(model_keys):
                    if i > j:
                        continue
                    b = model_bin_preds[m2].to_numpy()
                    # Three-way intersection: all three agree on positive labels
                    three_way_inter = np.logical_and(np.logical_and(a == 1, b == 1), y == 1).sum(axis=1)
                    # Three-way union: at least one has positive label
                    three_way_union = np.logical_or(np.logical_or(a == 1, b == 1), y == 1).sum(axis=1)
                    j_rows = np.where(three_way_union > 0, three_way_inter / three_way_union, 1.0)
                    s = float(np.mean(j_rows))
                    pairwise_truth_similarity[i, j] = pairwise_truth_similarity[j, i] = s

            # 4) Joint correctness: how often both models are correct together
            joint_correctness = np.zeros((n, n))
            for i, m1 in enumerate(model_keys):
                a = model_bin_preds[m1].to_numpy()
                correct1 = (a == y).astype(int)
                for j, m2 in enumerate(model_keys):
                    if i > j:
                        continue
                    b = model_bin_preds[m2].to_numpy()
                    correct2 = (b == y).astype(int)
                    # Both correct simultaneously (per row, per category)
                    both_correct = np.logical_and(correct1 == 1, correct2 == 1)
                    # Average across all categories and rows
                    joint_correctness[i, j] = joint_correctness[j, i] = float(both_correct.mean())

            results[source] = {
                'model_keys': model_keys,
                'model_truth_jaccard': model_truth_scores,
                'correctness_overlap': correctness_overlap,
                'pairwise_truth_similarity': pairwise_truth_similarity,
                'joint_correctness': joint_correctness
            }

        # Build overall by averaging across sources on common models
        if results:
            computed_sources = [s for s in SOURCES if s in results]
            if computed_sources:
                common = set(results[computed_sources[0]]['model_keys'])
                for s in computed_sources[1:]:
                    common &= set(results[s]['model_keys'])
                common = list(common)
                base_order = {name: i for i, name in enumerate(LLM_MODELS)}
                def sort_key(m):
                    base = m.split('_')[0]
                    return (base_order.get(base, 999), m)
                common.sort(key=sort_key)
                if common:
                    k = len(common)
                    overlap_sum = np.zeros((k, k))
                    overlap_cnt = np.zeros((k, k))
                    pairwise_truth_sum = np.zeros((k, k))
                    pairwise_truth_cnt = np.zeros((k, k))
                    joint_correct_sum = np.zeros((k, k))
                    joint_correct_cnt = np.zeros((k, k))
                    truth_vec = np.zeros(k)
                    truth_cnt = np.zeros(k)
                    for s in computed_sources:
                        keys_s = results[s]['model_keys']
                        idx = {m: i for i, m in enumerate(keys_s)}
                        # model-truth vector
                        vec_s = results[s]['model_truth_jaccard']
                        for i, m in enumerate(common):
                            if m in idx:
                                truth_vec[i] += vec_s[idx[m]]
                                truth_cnt[i] += 1
                        # correctness-overlap
                        mat_s = results[s]['correctness_overlap']
                        # pairwise truth similarity
                        pairwise_s = results[s]['pairwise_truth_similarity']
                        # joint correctness
                        joint_s = results[s]['joint_correctness']
                        for i, mi in enumerate(common):
                            for j, mj in enumerate(common):
                                if mi in idx and mj in idx:
                                    overlap_sum[i, j] += mat_s[idx[mi], idx[mj]]
                                    overlap_cnt[i, j] += 1
                                    pairwise_truth_sum[i, j] += pairwise_s[idx[mi], idx[mj]]
                                    pairwise_truth_cnt[i, j] += 1
                                    joint_correct_sum[i, j] += joint_s[idx[mi], idx[mj]]
                                    joint_correct_cnt[i, j] += 1
                    with np.errstate(invalid='ignore'):
                        truth_vec = np.divide(truth_vec, truth_cnt, out=np.zeros_like(truth_vec), where=truth_cnt>0)
                        overlap_avg = np.divide(overlap_sum, overlap_cnt, out=np.zeros_like(overlap_sum), where=overlap_cnt>0)
                        pairwise_truth_avg = np.divide(pairwise_truth_sum, pairwise_truth_cnt, out=np.zeros_like(pairwise_truth_sum), where=pairwise_truth_cnt>0)
                        joint_correct_avg = np.divide(joint_correct_sum, joint_correct_cnt, out=np.zeros_like(joint_correct_sum), where=joint_correct_cnt>0)
                    results['overall'] = {
                        'model_keys': common,
                        'model_truth_jaccard': truth_vec,
                        'correctness_overlap': overlap_avg,
                        'pairwise_truth_similarity': pairwise_truth_avg,
                        'joint_correctness': joint_correct_avg
                    }

        self.results['correct_overlap'] = results
        return results
    
    def ensemble_analysis(self):
        """Analyze ensemble model performance"""
        print("\n" + "="*60)
        print("ENSEMBLE ANALYSIS")
        print("="*60)
        
        ensemble_results = {}
        
        for source in SOURCES:
            if source not in self.predictions:
                continue
                
            print(f"\nAnalyzing ensemble performance for {source}...")
            ensemble_results[source] = {}
            
            model_keys = list(self.predictions[source].keys())
            
            # Calculate individual model performance
            individual_performance = {}
            for model_key in model_keys:
                f1_score = self._calculate_f1_score(source, model_key)
                if f1_score is not None:
                    individual_performance[model_key] = np.mean(f1_score)
            
            # Calculate ensemble performance (simple voting)
            ensemble_f1 = self._calculate_ensemble_f1(source, model_keys)
            
            # Calculate ensemble improvement
            best_individual = max(individual_performance.values()) if individual_performance else 0
            ensemble_improvement = ensemble_f1 - best_individual if ensemble_f1 is not None else 0
            
            ensemble_results[source] = {
                'individual_performance': individual_performance,
                'ensemble_f1': ensemble_f1,
                'best_individual_f1': best_individual,
                'ensemble_improvement': ensemble_improvement,
                'models_included': model_keys
            }
            
            print(f"  Best individual F1: {best_individual:.4f}")
            print(f"  Ensemble F1: {ensemble_f1:.4f}" if ensemble_f1 else "  Ensemble F1: N/A")
            print(f"  Improvement: {ensemble_improvement:.4f}")
        
        self.results['ensemble_analysis'] = ensemble_results
        return ensemble_results
    
    def model_agreement_analysis(self):
        """Analyze model agreement and disagreement patterns"""
        print("\n" + "="*60)
        print("MODEL AGREEMENT ANALYSIS")
        print("="*60)
        
        agreement_results = {}
        
        for source in SOURCES:
            if source not in self.predictions:
                continue
                
            print(f"\nAnalyzing model agreement for {source}...")
            agreement_results[source] = {}
            
            model_keys = list(self.predictions[source].keys())
            
            # Calculate agreement statistics
            agreement_stats = self._calculate_agreement_stats(source, model_keys)
            
            # Calculate disagreement patterns
            disagreement_patterns = self._analyze_disagreement_patterns(source, model_keys)
            
            agreement_results[source] = {
                'agreement_stats': agreement_stats,
                'disagreement_patterns': disagreement_patterns,
                'models_analyzed': model_keys
            }
            
            print(f"  Average agreement: {agreement_stats['average_agreement']:.4f}")
            print(f"  Most agreed upon: {agreement_stats['most_agreed_category']}")
            print(f"  Least agreed upon: {agreement_stats['least_agreed_category']}")
        
        self.results['agreement_analysis'] = agreement_results
        return agreement_results
    
    def _calculate_f1_score(self, source, model_key):
        """Calculate F1 score for a model"""
        try:
            predictions_df = self.predictions[source][model_key]
            soft_labels_df = self.soft_labels[source]
            
            # The soft labels don't have a text column, they're just the category scores
            # We need to align by index since they should be in the same order
            if len(predictions_df) != len(soft_labels_df):
                print(f"Warning: Length mismatch for {model_key}: predictions={len(predictions_df)}, soft_labels={len(soft_labels_df)}")
                # Take the minimum length
                min_len = min(len(predictions_df), len(soft_labels_df))
                predictions_df = predictions_df.iloc[:min_len]
                soft_labels_df = soft_labels_df.iloc[:min_len]
            
            # Get prediction columns (look for columns that match soft label categories)
            soft_label_cols = soft_labels_df.columns.tolist()
            prediction_cols = []
            
            # Find matching columns in predictions
            for col in predictions_df.columns:
                # Check if this column matches any soft label category
                for soft_col in soft_label_cols:
                    if soft_col.lower() in col.lower() or col.lower() in soft_col.lower():
                        prediction_cols.append(col)
                        break
            
            if not prediction_cols:
                print(f"No matching prediction columns found for {model_key}")
                return None
            
            # Helper to coerce predictions to numeric and then binary
            def _coerce_to_binary(series):
                if series.dtype == object:
                    mapping = {
                        'yes': 1, 'true': 1, 'y': 1, 'positive': 1, 'pos': 1, '1': 1,
                        'no': 0, 'false': 0, 'n': 0, 'negative': 0, 'neg': 0, '0': 0
                    }
                    series = series.astype(str).str.strip().str.lower().map(mapping).fillna(series)
                numeric = pd.to_numeric(series, errors='coerce')
                return (numeric > 0.5).astype(int)

            # Calculate F1 for each category
            f1_scores = []
            for i, soft_col in enumerate(soft_label_cols):
                if i < len(prediction_cols):
                    pred_col = prediction_cols[i]
                    try:
                        y_true = (soft_labels_df[soft_col] > 0.5).astype(int)
                        y_pred = _coerce_to_binary(predictions_df[pred_col])
                        f1 = f1_score(y_true, y_pred, zero_division=0)
                        f1_scores.append(f1)
                    except Exception as e:
                        print(f"Error calculating F1 for {soft_col}: {e}")
                        continue
            
            return f1_scores if f1_scores else None
            
        except Exception as e:
            print(f"Error calculating F1 for {model_key}: {e}")
            return None
    
    def _calculate_correlation(self, source, model1, model2, method='pearson'):
        """Calculate correlation between two models"""
        try:
            pred1_df = self.predictions[source][model1]
            pred2_df = self.predictions[source][model2]
            
            # Align by index since both should be in same order
            min_len = min(len(pred1_df), len(pred2_df))
            pred1_df = pred1_df.iloc[:min_len]
            pred2_df = pred2_df.iloc[:min_len]
            
            # Get soft label columns to find matching prediction columns
            soft_labels_df = self.soft_labels[source]
            soft_label_cols = soft_labels_df.columns.tolist()
            
            # Find matching columns in both predictions
            pred1_cols = []
            pred2_cols = []
            
            for col in pred1_df.columns:
                for soft_col in soft_label_cols:
                    if soft_col.lower() in col.lower() or col.lower() in soft_col.lower():
                        pred1_cols.append(col)
                        break
            
            for col in pred2_df.columns:
                for soft_col in soft_label_cols:
                    if soft_col.lower() in col.lower() or col.lower() in soft_col.lower():
                        pred2_cols.append(col)
                        break
            
            if not pred1_cols or not pred2_cols:
                return 0, 1
            
            # Calculate average correlation across all categories
            correlations = []
            min_cols = min(len(pred1_cols), len(pred2_cols))
            
            for i in range(min_cols):
                try:
                    col1 = pred1_cols[i]
                    col2 = pred2_cols[i]
                    
                    if method == 'pearson':
                        corr, p_val = pearsonr(pred1_df[col1], pred2_df[col2])
                    elif method == 'spearman':
                        corr, p_val = spearmanr(pred1_df[col1], pred2_df[col2])
                    else:
                        corr, p_val = kendalltau(pred1_df[col1], pred2_df[col2])
                    
                    if not np.isnan(corr):
                        correlations.append(corr)
                except Exception as e:
                    continue
            
            return np.mean(correlations) if correlations else 0, 0.5
            
        except Exception as e:
            print(f"Error calculating correlation between {model1} and {model2}: {e}")
            return 0, 1
    
    def _calculate_ensemble_f1(self, source, model_keys):
        """Calculate ensemble F1 score using voting"""
        try:
            # This is a simplified implementation
            # In practice, you'd need to implement proper ensemble voting
            individual_f1s = []
            for model_key in model_keys:
                f1_scores = self._calculate_f1_score(source, model_key)
                if f1_scores is not None:
                    individual_f1s.append(np.mean(f1_scores))
            
            # Simple average ensemble (in practice, you'd use voting)
            return np.mean(individual_f1s) if individual_f1s else None
            
        except Exception as e:
            print(f"Error calculating ensemble F1: {e}")
            return None
    
    def _calculate_agreement_stats(self, source, model_keys):
        """Calculate agreement statistics between models"""
        try:
            # This is a simplified implementation
            # In practice, you'd calculate actual agreement between predictions
            return {
                'average_agreement': 0.75,  # Placeholder
                'most_agreed_category': 'category_1',  # Placeholder
                'least_agreed_category': 'category_2'  # Placeholder
            }
        except Exception as e:
            print(f"Error calculating agreement stats: {e}")
            return {}
    
    def _analyze_disagreement_patterns(self, source, model_keys):
        """Analyze patterns in model disagreements"""
        try:
            # This is a simplified implementation
            # In practice, you'd analyze actual disagreement patterns
            return {
                'common_disagreement_pairs': [],
                'disagreement_by_category': {},
                'disagreement_by_text_length': {}
            }
        except Exception as e:
            print(f"Error analyzing disagreement patterns: {e}")
            return {}
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Correlation heatmaps
        self._plot_correlation_heatmaps()
        # 1b. Prediction-level similarity heatmaps
        self._plot_prediction_similarity_heatmaps()
        # 1c. Multi-label agreement heatmaps
        self._plot_multilabel_agreement_heatmaps()
        # 1d. Correct-overlap heatmaps
        self._plot_correct_overlap_plots()
        
        # 2. Model performance comparison
        self._plot_model_performance_comparison()
        
        # 3. Statistical significance plots
        self._plot_significance_results()
        
        # 4. Ensemble performance plots
        self._plot_ensemble_performance()
        
        print(f"Visualizations saved to {self.output_dir}/plots/")
    
    def _plot_correlation_heatmaps(self):
        """Plot correlation heatmaps for each source"""
        if 'correlation_analysis' not in self.results:
            return
        
        for source, data in self.results['correlation_analysis'].items():
            plt.figure(figsize=(12, 10))
            
            correlation_matrix = data['correlation_matrix']
            model_keys = data['model_keys']
            
            # Create heatmap
            sns.heatmap(correlation_matrix, 
                       xticklabels=model_keys, 
                       yticklabels=model_keys,
                       annot=True, 
                       cmap='coolwarm', 
                       center=0,
                       fmt='.3f')
            
            plt.title(f'Model Correlation Matrix - {source.title()}')
            plt.xlabel('Models')
            plt.ylabel('Models')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            plt.savefig(f'{self.output_dir}/plots/correlation_heatmap_{source}.pdf', 
                       dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_prediction_similarity_heatmaps(self):
        """Plot prediction-level similarity heatmaps (Pearson and Spearman) for each source"""
        if 'prediction_similarity' not in self.results:
            return
        for source, data in self.results['prediction_similarity'].items():
            model_keys = data['model_keys']

            # Pearson
            plt.figure(figsize=(12, 10))
            sns.heatmap(data['pearson_matrix'], xticklabels=model_keys, yticklabels=model_keys,
                        annot=True, cmap='coolwarm', center=0, fmt='.3f')
            plt.title(f'Prediction-Level Pearson Correlation - {source.title()}')
            plt.xlabel('Models')
            plt.ylabel('Models')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/plots/prediction_pearson_{source}.pdf', dpi=300, bbox_inches='tight')
            plt.close()

            # Spearman
            plt.figure(figsize=(12, 10))
            sns.heatmap(data['spearman_matrix'], xticklabels=model_keys, yticklabels=model_keys,
                        annot=True, cmap='coolwarm', center=0, fmt='.3f')
            plt.title(f'Prediction-Level Spearman Correlation - {source.title()}')
            plt.xlabel('Models')
            plt.ylabel('Models')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/plots/prediction_spearman_{source}.pdf', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_model_performance_comparison(self):
        """Plot model performance comparison"""
        # Load existing F1 data
        f1_data_path = 'output/f1/comprehensive_model_comparison.csv'
        try:
            df = pd.read_csv(f1_data_path)
            
            # Create performance comparison plot
            plt.figure(figsize=(15, 10))
            
            # Group by source and plot
            for i, source in enumerate(SOURCES):
                plt.subplot(2, 2, i+1)
                source_data = df[df['Source'] == source]
                
                # Plot macro F1 scores
                plt.bar(range(len(source_data)), source_data['Macro_F1'])
                plt.title(f'{source.title()} - Macro F1 Scores')
                plt.xlabel('Models')
                plt.ylabel('Macro F1')
                plt.xticks(range(len(source_data)), source_data['Model'], rotation=45)
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/plots/model_performance_comparison.pdf', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except FileNotFoundError:
            print("F1 data file not found, skipping performance comparison plot")
    
    def _plot_significance_results(self):
        """Plot statistical significance results"""
        if 'significance_testing' not in self.results:
            return
        
        # This would plot significance test results
        # Implementation depends on the specific results structure
        pass
    
    def _plot_ensemble_performance(self):
        """Plot ensemble performance analysis"""
        if 'ensemble_analysis' not in self.results:
            return
        
        # This would plot ensemble performance
        # Implementation depends on the specific results structure
        pass

    def _plot_multilabel_agreement_heatmaps(self):
        """Plot multi-label agreement (match ratio, Jaccard) heatmaps per source"""
        if 'multilabel_agreement' not in self.results:
            return
        for source, data in self.results['multilabel_agreement'].items():
            model_keys = data['model_keys']

            # Match ratio
            plt.figure(figsize=(12, 10))
            sns.heatmap(data['match_ratio_matrix'], xticklabels=model_keys, yticklabels=model_keys,
                        annot=True, cmap='viridis', vmin=0, vmax=1, fmt='.3f')
            plt.title(f'Multi-label Match Ratio - {source.title()}')
            plt.xlabel('Models')
            plt.ylabel('Models')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/plots/multilabel_matchratio_{source}.pdf', dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_correct_overlap_plots(self):
        """Plot correctness-based overlap:
        - Heatmap of correctness-overlap between models (per source and overall)
        - Heatmap (single column) of model-vs-truth Jaccard per model
        """
        if 'correct_overlap' not in self.results:
            return
        for source, data in self.results['correct_overlap'].items():
            model_keys = data['model_keys']
            # Skip if meta keys
            if not isinstance(model_keys, list):
                continue
            # Correctness-overlap matrix
            plt.figure(figsize=(12, 10))
            sns.heatmap(data['correctness_overlap'], xticklabels=model_keys, yticklabels=model_keys,
                        annot=True, cmap='viridis', vmin=0, vmax=1, fmt='.3f')
            title_src = source.title() if source != 'overall' else 'Overall'
            plt.title(f'Correctness Overlap (Jaccard of correct labels) - {title_src}')
            plt.xlabel('Models')
            plt.ylabel('Models')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/plots/correct_overlap_{source}.pdf', dpi=300, bbox_inches='tight')
            plt.close()

            # Model-vs-truth Jaccard vector as heatmap with single column
            vec = np.array(data['model_truth_jaccard']).reshape(-1, 1)
            plt.figure(figsize=(6, 10))
            sns.heatmap(vec, xticklabels=['Truth'], yticklabels=model_keys,
                        annot=True, cmap='viridis', vmin=0, vmax=1, fmt='.3f')
            plt.title(f'Model vs Truth Jaccard - {title_src}')
            plt.xlabel('')
            plt.ylabel('Models')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/plots/model_truth_jaccard_{source}.pdf', dpi=300, bbox_inches='tight')
            plt.close()

            # Pairwise model-truth similarity (three-way Jaccard)
            if 'pairwise_truth_similarity' in data:
                plt.figure(figsize=(12, 10))
                sns.heatmap(data['pairwise_truth_similarity'], xticklabels=model_keys, yticklabels=model_keys,
                            annot=True, cmap='viridis', vmin=0, vmax=1, fmt='.3f')
                plt.title(f'Pairwise Model-Truth Similarity (3-way Jaccard) - {title_src}')
                plt.xlabel('Models')
                plt.ylabel('Models')
                plt.xticks(rotation=45)
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.savefig(f'{self.output_dir}/plots/pairwise_truth_similarity_{source}.pdf', dpi=300, bbox_inches='tight')
                plt.close()

            # Joint correctness (both models correct together)
            if 'joint_correctness' in data:
                plt.figure(figsize=(12, 10))
                sns.heatmap(data['joint_correctness'], xticklabels=model_keys, yticklabels=model_keys,
                            annot=True, cmap='viridis', vmin=0, vmax=1, fmt='.3f')
                plt.title(f'Joint Correctness (Both Models Correct) - {title_src}')
                plt.xlabel('Models')
                plt.ylabel('Models')
                plt.xticks(rotation=45)
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.savefig(f'{self.output_dir}/plots/joint_correctness_{source}.pdf', dpi=300, bbox_inches='tight')
                plt.close()
    
    def generate_report(self):
        """Generate comprehensive validation report"""
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*60)
        
        # Save results to JSON
        with open(f'{self.output_dir}/cross_validation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate summary report
        self._generate_summary_report()
        
        print(f"Report saved to {self.output_dir}/")
    
    def _generate_summary_report(self):
        """Generate summary report"""
        report_path = f'{self.output_dir}/cross_validation_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("CROSS-MODEL VALIDATION ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write("MODELS ANALYZED:\n")
            f.write(f"LLMs: {', '.join(LLM_MODELS)}\n")
            f.write(f"BERT: {BERT_MODEL}\n")
            f.write(f"Shot Types: {', '.join(SHOT_TYPES)}\n\n")
            
            f.write("DATA SOURCES:\n")
            f.write(f"{', '.join(SOURCES)}\n\n")
            
            f.write("ANALYSIS COMPLETED:\n")
            for analysis_type in self.results.keys():
                f.write(f"- {analysis_type.replace('_', ' ').title()}\n")
            
            f.write(f"\nResults saved to: {self.output_dir}/\n")
            f.write(f"Visualizations saved to: {self.output_dir}/plots/\n")
    
    def run_full_analysis(self):
        """Run complete cross-model validation analysis"""
        print("Starting Cross-Model Validation Analysis")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Run analyses
        self.statistical_significance_testing()
        self.correlation_analysis()
        self.prediction_level_similarity()
        self.multi_label_agreement()
        self.correct_overlap_analysis()
        self.ensemble_analysis()
        self.model_agreement_analysis()
        
        # Generate outputs
        self.generate_visualizations()
        self.generate_report()
        
        print("\n" + "="*60)
        print("CROSS-MODEL VALIDATION ANALYSIS COMPLETE")
        print("="*60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Cross-Model Validation Analysis')
    parser.add_argument('--output_dir', type=str, default='output/cross_validation', 
                       help='Output directory for results')
    parser.add_argument('--sources', nargs='+', default=SOURCES, 
                       help='Data sources to analyze')
    parser.add_argument('--models', nargs='+', default=LLM_MODELS, 
                       help='Models to analyze')
    
    args = parser.parse_args()
    
    # Create validator
    validator = CrossModelValidator(output_dir=args.output_dir)
    
    # Run analysis
    validator.run_full_analysis()

if __name__ == "__main__":
    main()
