#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import os
import json
from collections import defaultdict
import argparse

# Configuration
SOURCES = ['reddit', 'x', 'news', 'meeting_minutes']
MODELS = ['llama', 'qwen', 'gpt4', 'gemini', 'grok', 'phi4', 'bert']
SHOT_TYPES = ['zero_shot', 'few_shot']

# Soft label threshold
SOFT_LABEL_THRESHOLD = 0.5

def load_soft_labels(source):
    """Load soft labels for a source."""
    soft_labels_path = f'output/annotation/soft_labels/{source}_soft_labels.csv'
    try:
        df = pd.read_csv(soft_labels_path)
        print(f"Loaded soft labels for {source}: {len(df)} samples")
        return df
    except FileNotFoundError:
        print(f"Soft labels not found for {source}: {soft_labels_path}")
        return None

def load_model_predictions(source, model, shot_type):
    """Load model predictions for a source, model, and shot type."""
    if shot_type == 'zero_shot':
        few_shot_text = 'none'
    else:
        few_shot_text = source
    
    # Try different possible file paths
    possible_paths = [
        f'output/{source}/{model}/classified_comments_{source}_gold_subset_{model}_{few_shot_text}_flags.csv',
        f'output/{source}/{model}/classified_comments_{source}_all_{model}_{few_shot_text}_flags.csv',
        f'output/classified_comments_{source}_gold_subset_{model}.csv'
    ]
    
    for path in possible_paths:
        try:
            df = pd.read_csv(path)
            print(f"Loaded {model} {shot_type} for {source}: {len(df)} samples")
            return df
        except FileNotFoundError:
            continue
    
    print(f"No predictions found for {model} {shot_type} on {source}")
    return None

def load_bert_results(source):
    """Load BERT fine-tuned results."""
    bert_metrics_path = f'output/{source}/bert/bert_metrics_{source}.json'
    try:
        with open(bert_metrics_path, 'r') as f:
            bert_results = json.load(f)
        print(f"Loaded BERT results for {source}: macro F1 = {bert_results['macro_f1']:.4f}")
        return bert_results
    except FileNotFoundError:
        print(f"BERT results not found for {source}")
        return None

def map_columns_to_soft_labels(df, source):
    """Map model output columns to soft label columns."""
    # Define mapping from model output columns to soft label columns
    column_mapping = {
        'Comment_ask a genuine question': 'ask a genuine question',
        'Comment_ask a rhetorical question': 'ask a rhetorical question', 
        'Comment_provide a fact or claim': 'provide a fact or claim',
        'Comment_provide an observation': 'provide an observation',
        'Comment_express their opinion': 'express their opinion',
        'Comment_express others opinions': 'express others opinions',
        'Critique_money aid allocation': 'money aid allocation',
        'Critique_government critique': 'government critique',
        'Critique_societal critique': 'societal critique',
        'Response_solutions/interventions': 'solutions/interventions',
        'Perception_personal interaction': 'personal interaction',
        'Perception_media portrayal': 'media portrayal',
        'Perception_not in my backyard': 'not in my backyard',
        'Perception_harmful generalization': 'harmful generalization',
        'Perception_deserving/undeserving': 'deserving/undeserving',
        'Racist_Flag': 'racist'
    }
    
    # Rename columns to match soft labels
    df_mapped = df.copy()
    for model_col, soft_col in column_mapping.items():
        if model_col in df.columns:
            df_mapped[soft_col] = df[model_col]
    
    return df_mapped

def calculate_metrics(predictions, soft_labels, source):
    """Calculate macro F1 and per-category metrics."""
    if predictions is None or soft_labels is None:
        return None
    
    # Get common categories between predictions and soft labels
    pred_cols = [col for col in predictions.columns if col not in ['Comment', 'City', 'City_original']]
    soft_cols = [col for col in soft_labels.columns if col not in ['Comment', 'City']]
    
    common_categories = list(set(pred_cols) & set(soft_cols))
    print(f"Common categories for {source}: {len(common_categories)}")
    
    if not common_categories:
        return None
    
    # Calculate metrics for each category
    category_metrics = {}
    all_true = []
    all_pred = []
    
    for category in common_categories:
        if category in predictions.columns and category in soft_labels.columns:
            # Get predictions and true labels
            pred_values = predictions[category].fillna(0).astype(int)
            soft_values = soft_labels[category].fillna(0)
            
            # Convert soft labels to binary using threshold
            true_values = (soft_values >= SOFT_LABEL_THRESHOLD).astype(int)
            
            # Ensure same length
            min_len = min(len(pred_values), len(true_values))
            pred_values = pred_values[:min_len]
            true_values = true_values[:min_len]
            
            # Calculate metrics
            f1 = f1_score(true_values, pred_values, average='binary', zero_division=0)
            precision = precision_score(true_values, pred_values, average='binary', zero_division=0)
            recall = recall_score(true_values, pred_values, average='binary', zero_division=0)
            
            category_metrics[category] = {
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
            
            all_true.extend(true_values)
            all_pred.extend(pred_values)
    
    # Calculate overall metrics
    if all_true and all_pred:
        macro_f1 = f1_score(all_true, all_pred, average='macro', zero_division=0)
        micro_f1 = f1_score(all_true, all_pred, average='micro', zero_division=0)
    else:
        macro_f1 = 0
        micro_f1 = 0
    
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'category_metrics': category_metrics,
        'num_categories': len(common_categories)
    }

def create_latex_table(summary_df):
    """Create LaTeX table with best models for each source"""
    # Find best model for each source
    best_models = []
    for source in summary_df['Source'].unique():
        source_data = summary_df[summary_df['Source'] == source]
        
        # Find best macro F1 score
        best_idx = source_data['Macro_F1'].idxmax()
        best_row = source_data.loc[best_idx]
        
        best_models.append({
            'Source': source,
            'Model': best_row['Model'],
            'Macro_F1': best_row['Macro_F1'],
            'Micro_F1': best_row['Micro_F1']
        })
    
    # Create LaTeX table
    latex_table = []
    latex_table.append(r"\begin{table}[htbp]")
    latex_table.append(r"\centering")
    latex_table.append(r"\centering\caption{Best Macro F1 Scores by Data Source}")
    latex_table.append(r"\label{tab:best_macro_f1_scores}")
    latex_table.append(r"\begin{tabular}{lccc}")
    latex_table.append(r"\toprule")
    latex_table.append(r"Data Source & Model & Macro F1 & Micro F1 \\")
    latex_table.append(r"\midrule")
    
    for model in best_models:
        source_display = model['Source'].replace('_', ' ').title()
        if source_display == 'X':
            source_display = 'X (Twitter)'
        
        model_display = model['Model'].replace('_', ' ').upper()
        # Round to 2 significant figures only at the end
        macro_f1 = f"{model['Macro_F1'] * 100:.2f}"
        micro_f1 = f"{model['Micro_F1'] * 100:.2f}"
        
        latex_table.append(f"{source_display} & {model_display} & {macro_f1} & {micro_f1} \\\\")
    
    latex_table.append(r"\bottomrule")
    latex_table.append(r"\end{tabular}")
    latex_table.append(r"\end{table}")
    
    return "\n".join(latex_table)

def create_detailed_latex_table(summary_df):
    """Create detailed LaTeX table with all models"""
    latex_table = []
    latex_table.append(r"\begin{table*}[htbp]")
    latex_table.append(r"\centering")
    latex_table.append(r"\centering\caption{Macro and Micro F1 Scores for All Models by Data Source}")
    latex_table.append(r"\label{tab:detailed_f1_scores}")
    latex_table.append(r"\resizebox{\textwidth}{!}{")
    latex_table.append(r"\begin{tabular}{lcccccccccccc}")
    latex_table.append(r"\toprule")
    latex_table.append(r"Data Source & \multicolumn{2}{c}{GPT-4} & \multicolumn{2}{c}{LLaMA} & \multicolumn{2}{c}{Qwen} & \multicolumn{2}{c}{Phi-4} & \multicolumn{2}{c}{Grok} & \multicolumn{2}{c}{Gemini} & BERT \\")
    latex_table.append(r"& Zero & Few & Zero & Few & Zero & Few & Zero & Few & Zero & Few & Zero & Few & Fine-tuned \\")
    latex_table.append(r"\midrule")
    
    # Dynamically read sample sizes from complete dataset files
    sample_sizes = {}
    source_file_mapping = {
        'reddit': 'complete_dataset/all_reddit_comments.csv',
        'x': 'complete_dataset/all_twitter_posts.csv',
        'news': 'complete_dataset/all_newspaper_articles.csv',
        'meeting_minutes': 'complete_dataset/all_meeting_minutes.csv'
    }
    
    for source, file_path in source_file_mapping.items():
        try:
            # Count lines in the CSV file (subtract 1 for header)
            with open(file_path, 'r') as f:
                line_count = sum(1 for line in f) - 1  # Subtract header
            sample_sizes[source] = line_count
            print(f"Read {source}: {line_count} samples from {file_path}")
        except FileNotFoundError:
            print(f"Warning: {file_path} not found, using default size of 1000")
            sample_sizes[source] = 1000
        except Exception as e:
            print(f"Error reading {file_path}: {e}, using default size of 1000")
            sample_sizes[source] = 1000
    
    total_samples = sum(sample_sizes.values())
    print(f"Total samples across all sources: {total_samples}")
    
    # Store all values for weighted average calculation
    all_macro_values = {model: {'zero': [], 'few': []} for model in ['gpt4', 'llama', 'qwen', 'phi4', 'grok', 'gemini']}
    all_micro_values = {model: {'zero': [], 'few': []} for model in ['gpt4', 'llama', 'qwen', 'phi4', 'grok', 'gemini']}
    bert_macro_values = []
    bert_micro_values = []
    
    # Group by source
    for source in SOURCES:
        source_data = summary_df[summary_df['Source'] == source]
        if source_data.empty:
            continue
        
        # Format source name
        source_display = source.replace('_', ' ').title()
        if source_display == 'X':
            source_display = 'X (Twitter)'
        
        # Create macro F1 row
        macro_row = [f"{source_display} (Macro)"]
        macro_values = []
        
        # Add macro F1 data for each model
        for model in ['gpt4', 'llama', 'qwen', 'phi4', 'grok', 'gemini']:
            for shot_type in ['zero_shot', 'few_shot']:
                model_key = f"{model}_{shot_type}"
                model_data = source_data[source_data['Model'] == model_key]
                
                if not model_data.empty:
                    macro_f1 = model_data.iloc[0]['Macro_F1']
                    macro_values.append(macro_f1)
                    # Round to 2 significant figures only at the end
                    macro_row.append(f"{macro_f1 * 100:.2f}")
                    
                    # Store for weighted average
                    if shot_type == 'zero_shot':
                        all_macro_values[model]['zero'].append((macro_f1, sample_sizes[source]))
                    else:
                        all_macro_values[model]['few'].append((macro_f1, sample_sizes[source]))
                else:
                    macro_values.append(0)
                    macro_row.append("--")
        
        # Add BERT macro F1 data
        bert_data = source_data[source_data['Model'] == 'bert_finetuned']
        if not bert_data.empty:
            bert_macro = bert_data.iloc[0]['Macro_F1']
            macro_values.append(bert_macro)
            # Round to 2 significant figures only at the end
            macro_row.append(f"{bert_macro * 100:.2f}")
            bert_macro_values.append((bert_macro, sample_sizes[source]))
        else:
            macro_values.append(0)
            macro_row.append("--")
        
        # Find best macro F1 and bold it
        valid_macro_values = [v for v in macro_values if v > 0]
        if valid_macro_values:
            best_macro = max(valid_macro_values)
            for i, val in enumerate(macro_values):
                if val == best_macro and val > 0:
                    macro_row[i + 1] = f"\\textbf{{{macro_row[i + 1]}}}"
        
        latex_table.append(" & ".join(macro_row) + " \\\\")
        
        # Create micro F1 row
        micro_row = [f"{source_display} (Micro)"]
        micro_values = []
        
        # Add micro F1 data for each model
        for model in ['gpt4', 'llama', 'qwen', 'phi4', 'grok', 'gemini']:
            for shot_type in ['zero_shot', 'few_shot']:
                model_key = f"{model}_{shot_type}"
                model_data = source_data[source_data['Model'] == model_key]
                
                if not model_data.empty:
                    micro_f1 = model_data.iloc[0]['Micro_F1']
                    micro_values.append(micro_f1)
                    # Round to 2 significant figures only at the end
                    micro_row.append(f"{micro_f1 * 100:.2f}")
                    
                    # Store for weighted average
                    if shot_type == 'zero_shot':
                        all_micro_values[model]['zero'].append((micro_f1, sample_sizes[source]))
                    else:
                        all_micro_values[model]['few'].append((micro_f1, sample_sizes[source]))
                else:
                    micro_values.append(0)
                    micro_row.append("--")
        
        # Add BERT micro F1 data
        bert_data = source_data[source_data['Model'] == 'bert_finetuned']
        if not bert_data.empty:
            bert_micro = bert_data.iloc[0]['Micro_F1']
            micro_values.append(bert_micro)
            # Round to 2 significant figures only at the end
            micro_row.append(f"{bert_micro * 100:.2f}")
            bert_micro_values.append((bert_micro, sample_sizes[source]))
        else:
            micro_values.append(0)
            micro_row.append("--")
        
        # Find best micro F1 and bold it
        valid_micro_values = [v for v in micro_values if v > 0]
        if valid_micro_values:
            best_micro = max(valid_micro_values)
            for i, val in enumerate(micro_values):
                if val == best_micro and val > 0:
                    micro_row[i + 1] = f"\\textbf{{{micro_row[i + 1]}}}"
        
        latex_table.append(" & ".join(micro_row) + " \\\\")
        
        # Add a small gap between sources
        if source != SOURCES[-1]:  # Don't add gap after last source
            latex_table.append(r"\addlinespace[0.5em]")
    
    # Calculate weighted averages
    def calculate_weighted_average(values_weights):
        if not values_weights:
            return 0
        total_weight = sum(weight for _, weight in values_weights)
        weighted_sum = sum(value * weight for value, weight in values_weights)
        return weighted_sum / total_weight if total_weight > 0 else 0
    
    # Add weighted average rows
    latex_table.append(r"\addlinespace[0.5em]")
    
    # Weighted average macro F1 row
    weighted_macro_row = ["Weighted Avg (Macro)"]
    weighted_macro_values = []
    
    for model in ['gpt4', 'llama', 'qwen', 'phi4', 'grok', 'gemini']:
        for shot_type in ['zero', 'few']:
            weighted_avg = calculate_weighted_average(all_macro_values[model][shot_type])
            weighted_macro_values.append(weighted_avg)
            # Round to 2 significant figures only at the end
            weighted_macro_row.append(f"{weighted_avg * 100:.2f}")
    
    # Add BERT weighted average
    bert_weighted_macro = calculate_weighted_average(bert_macro_values)
    weighted_macro_values.append(bert_weighted_macro)
    # Round to 2 significant figures only at the end
    weighted_macro_row.append(f"{bert_weighted_macro * 100:.2f}")
    
    # Bold best weighted macro F1
    valid_weighted_macro = [v for v in weighted_macro_values if v > 0]
    if valid_weighted_macro:
        best_weighted_macro = max(valid_weighted_macro)
        for i, val in enumerate(weighted_macro_values):
            if val == best_weighted_macro and val > 0:
                weighted_macro_row[i + 1] = f"\\textbf{{{weighted_macro_row[i + 1]}}}"
    
    latex_table.append(" & ".join(weighted_macro_row) + " \\\\")
    
    # Weighted average micro F1 row
    weighted_micro_row = ["Weighted Avg (Micro)"]
    weighted_micro_values = []
    
    for model in ['gpt4', 'llama', 'qwen', 'phi4', 'grok', 'gemini']:
        for shot_type in ['zero', 'few']:
            weighted_avg = calculate_weighted_average(all_micro_values[model][shot_type])
            weighted_micro_values.append(weighted_avg)
            # Round to 2 significant figures only at the end
            weighted_micro_row.append(f"{weighted_avg * 100:.2f}")
    
    # Add BERT weighted average
    bert_weighted_micro = calculate_weighted_average(bert_micro_values)
    weighted_micro_values.append(bert_weighted_micro)
    # Round to 2 significant figures only at the end
    weighted_micro_row.append(f"{bert_weighted_micro * 100:.2f}")
    
    # Bold best weighted micro F1
    valid_weighted_micro = [v for v in weighted_micro_values if v > 0]
    if valid_weighted_micro:
        best_weighted_micro = max(valid_weighted_micro)
        for i, val in enumerate(weighted_micro_values):
            if val == best_weighted_micro and val > 0:
                weighted_micro_row[i + 1] = f"\\textbf{{{weighted_micro_row[i + 1]}}}"
    
    latex_table.append(" & ".join(weighted_micro_row) + " \\\\")
    
    latex_table.append(r"\bottomrule")
    latex_table.append(r"\end{tabular}")
    latex_table.append(r"}")
    latex_table.append(r"\end{table*}")
    
    return "\n".join(latex_table)

def create_individual_model_tables(all_results):
    """Create individual LaTeX tables for each model showing category-wise F1 scores"""
    
    # Define category display names
    category_display_names = {
        'ask a genuine question': 'Ask Genuine Question',
        'ask a rhetorical question': 'Ask Rhetorical Question',
        'ask a rheorical question': 'Ask Rhetorical Question',  # Handle typo in BERT data
        'provide a fact or claim': 'Provide Fact/Claim',
        'provide an observation': 'Provide Observation',
        'express their opinion': 'Express Opinion',
        'express others opinions': 'Express Others Opinions',
        'money aid allocation': 'Money Aid Allocation',
        'government critique': 'Government Critique',
        'societal critique': 'Societal Critique',
        'solutions/interventions': 'Solutions/Interventions',
        'personal interaction': 'Personal Interaction',
        'media portrayal': 'Media Portrayal',
        'not in my backyard': 'Not in My Backyard',
        'harmful generalization': 'Harmful Generalization',
        'deserving/undeserving': 'Deserving/Undeserving',
        'racist': 'Racist',
        'Racist': 'Racist'  # Handle case variations
    }
    
    # Define source display names
    source_display_names = {
        'reddit': 'Reddit',
        'news': 'News',
        'meeting_minutes': 'Meeting Minutes',
        'x': 'X (Twitter)'
    }
    
    # Get all unique categories across all results
    all_categories = set()
    for source_results in all_results.values():
        for model_results in source_results.values():
            if 'category_metrics' in model_results:
                all_categories.update(model_results['category_metrics'].keys())
    
    # Sort categories for consistent ordering
    sorted_categories = sorted(all_categories)
    
    # Create tables for each model
    for model in MODELS:
        print(f"Creating table for {model.upper()}...")
        
        # Collect data for this model across all sources
        model_data = {}
        macro_micro_data = {}
        
        for source in SOURCES:
            if source in all_results:
                if model == 'bert':
                    # Handle BERT separately
                    model_key = 'bert_finetuned'
                    if model_key in all_results[source]:
                        if 'category_metrics' in all_results[source][model_key]:
                            model_data[f"{source}_bert"] = all_results[source][model_key]['category_metrics']
                        macro_micro_data[f"{source}_bert"] = {
                            'macro_f1': all_results[source][model_key]['macro_f1'],
                            'micro_f1': all_results[source][model_key]['micro_f1']
                        }
                else:
                    # Handle other models with zero/few shot
                    for shot_type in ['zero_shot', 'few_shot']:
                        model_key = f"{model}_{shot_type}"
                        if model_key in all_results[source]:
                            if 'category_metrics' in all_results[source][model_key]:
                                model_data[f"{source}_{shot_type}"] = all_results[source][model_key]['category_metrics']
                            macro_micro_data[f"{source}_{shot_type}"] = {
                                'macro_f1': all_results[source][model_key]['macro_f1'],
                                'micro_f1': all_results[source][model_key]['micro_f1']
                            }
        
        if not model_data:
            continue
        
        # Create LaTeX table
        latex_table = []
        latex_table.append(r"\begin{table*}[htbp]")
        latex_table.append(r"\centering")
        
        if model == 'bert':
            # BERT table format (no zero/few shot distinction)
            latex_table.append(r"\begin{tabular}{l *{4}{c}}")
            latex_table.append(r"\toprule")
            latex_table.append(r"Category & Reddit & News & Meeting Minutes & X (Twitter) \\")
            latex_table.append(r"\midrule")
            
            # Add rows for each category
            processed_categories = set()
            for category in sorted_categories:
                # Skip unnamed categories and handle display names
                if category.startswith('Unnamed:') or category not in category_display_names:
                    continue
                
                # Skip if we've already processed this category (handle duplicates)
                category_display = category_display_names[category]
                if category_display in processed_categories:
                    continue
                processed_categories.add(category_display)
                
                row_data = [category_display]
                
                # Add data for each source
                for source in SOURCES:
                    key = f"{source}_bert"
                    if key in model_data and category in model_data[key]:
                        # BERT has direct F1 scores, not nested structure
                        f1_score = model_data[key][category]
                        if isinstance(f1_score, dict):
                            f1_score = f1_score['f1']
                        f1_score = f1_score * 100  # Convert to percentage
                        # Round to 2 significant figures only at the end
                        row_data.append(f"{f1_score:.2f}")
                    else:
                        row_data.append("--")
                
                latex_table.append(" & ".join(row_data) + " \\\\")
            
            latex_table.append(r"\bottomrule")
            latex_table.append(r"\end{tabular}")
            latex_table.append(fr"\centering\caption{{Category-wise F1 Scores for BERT Fine-tuned Model}}")
            latex_table.append(fr"\label{{tab:bert_category_breakdown}}")
        else:
            # Other models table format (with zero/few shot distinction)
            latex_table.append(r"\begin{tabular}{l *{8}{c}}")
            latex_table.append(r"\toprule")
            latex_table.append(r"Category & \multicolumn{2}{c}{Reddit} & \multicolumn{2}{c}{News} & \multicolumn{2}{c}{Meeting Minutes} & \multicolumn{2}{c}{X (Twitter)} \\")
            latex_table.append(r"& Zero & Few & Zero & Few & Zero & Few & Zero & Few \\")
            latex_table.append(r"\midrule")
            
            # Add rows for each category
            for category in sorted_categories:
                if category in category_display_names:
                    category_display = category_display_names[category]
                    row_data = [category_display]
                    
                    # Add data for each source and shot type
                    for source in SOURCES:
                        zero_shot_key = f"{source}_zero_shot"
                        few_shot_key = f"{source}_few_shot"
                        
                        zero_shot_value = None
                        few_shot_value = None
                        
                        if zero_shot_key in model_data and category in model_data[zero_shot_key]:
                            zero_shot_value = model_data[zero_shot_key][category]['f1'] * 100
                        
                        if few_shot_key in model_data and category in model_data[few_shot_key]:
                            few_shot_value = model_data[few_shot_key][category]['f1'] * 100
                        
                        # Format zero-shot value
                        if zero_shot_value is not None:
                            # Round to 2 significant figures only at the end
                            zero_shot_str = f"{zero_shot_value:.2f}"
                        else:
                            zero_shot_str = "--"
                        
                        # Format few-shot value
                        if few_shot_value is not None:
                            # Round to 2 significant figures only at the end
                            few_shot_str = f"{few_shot_value:.2f}"
                        else:
                            few_shot_str = "--"
                        
                        # Bold the better score
                        if zero_shot_value is not None and few_shot_value is not None:
                            if zero_shot_value > few_shot_value:
                                zero_shot_str = f"\\textbf{{{zero_shot_str}}}"
                            elif few_shot_value > zero_shot_value:
                                few_shot_str = f"\\textbf{{{few_shot_str}}}"
                        elif zero_shot_value is not None:
                            zero_shot_str = f"\\textbf{{{zero_shot_str}}}"
                        elif few_shot_value is not None:
                            few_shot_str = f"\\textbf{{{few_shot_str}}}"
                        
                        row_data.extend([zero_shot_str, few_shot_str])
                    
                    latex_table.append(" & ".join(row_data) + " \\\\")
            
            latex_table.append(r"\bottomrule")
            latex_table.append(r"\end{tabular}")
            latex_table.append(fr"\centering\caption{{Category-wise F1 Scores for {model.upper()} Model}}")
            latex_table.append(fr"\label{{tab:{model}_category_breakdown}}")
        
        latex_table.append(r"\end{table*}")
        
        # Create macro/micro F1 table
        macro_micro_table = []
        macro_micro_table.append(r"\begin{table}[htbp]")
        macro_micro_table.append(r"\centering")
        
        if model == 'bert':
            # BERT macro/micro table
            macro_micro_table.append(r"\begin{tabular}{lcc}")
            macro_micro_table.append(r"\toprule")
            macro_micro_table.append(r"Data Source & Macro F1 & Micro F1 \\")
            macro_micro_table.append(r"\midrule")
            
            for source in SOURCES:
                key = f"{source}_bert"
                if key in macro_micro_data:
                    source_display = source_display_names[source]
                    # Round to 2 significant figures only at the end
                    macro_f1 = f"{macro_micro_data[key]['macro_f1'] * 100:.2f}"
                    micro_f1 = f"{macro_micro_data[key]['micro_f1'] * 100:.2f}"
                    macro_micro_table.append(f"{source_display} & {macro_f1} & {micro_f1} \\\\")
            
            macro_micro_table.append(r"\bottomrule")
            macro_micro_table.append(r"\end{tabular}")
            macro_micro_table.append(fr"\centering\caption{{Macro and Micro F1 Scores for BERT Fine-tuned Model}}")
            macro_micro_table.append(fr"\label{{tab:bert_macro_micro}}")
        else:
            # Other models macro/micro table
            macro_micro_table.append(r"\begin{tabular}{l *{8}{c}}")
            macro_micro_table.append(r"\toprule")
            macro_micro_table.append(r"Data Source & \multicolumn{2}{c}{Reddit} & \multicolumn{2}{c}{News} & \multicolumn{2}{c}{Meeting Minutes} & \multicolumn{2}{c}{X (Twitter)} \\")
            macro_micro_table.append(r"& Zero & Few & Zero & Few & Zero & Few & Zero & Few \\")
            macro_micro_table.append(r"\midrule")
            
            # Add macro F1 row
            macro_row = ["Macro F1"]
            for source in SOURCES:
                for shot_type in ['zero_shot', 'few_shot']:
                    key = f"{source}_{shot_type}"
                    if key in macro_micro_data:
                        # Round to 2 significant figures only at the end
                        macro_f1 = f"{macro_micro_data[key]['macro_f1'] * 100:.2f}"
                        macro_row.append(macro_f1)
                    else:
                        macro_row.append("--")
            macro_micro_table.append(" & ".join(macro_row) + " \\\\")
            
            # Add micro F1 row
            micro_row = ["Micro F1"]
            for source in SOURCES:
                for shot_type in ['zero_shot', 'few_shot']:
                    key = f"{source}_{shot_type}"
                    if key in macro_micro_data:
                        # Round to 2 significant figures only at the end
                        micro_f1 = f"{macro_micro_data[key]['micro_f1'] * 100:.2f}"
                        micro_row.append(micro_f1)
                    else:
                        micro_row.append("--")
            macro_micro_table.append(" & ".join(micro_row) + " \\\\")
            
            macro_micro_table.append(r"\bottomrule")
            macro_micro_table.append(r"\end{tabular}")
            macro_micro_table.append(fr"\centering\caption{{Macro and Micro F1 Scores for {model.upper()} Model}}")
            macro_micro_table.append(fr"\label{{tab:{model}_macro_micro}}")
        
        macro_micro_table.append(r"\end{table}")
        
        # Save the tables
        output_dir = 'output/f1/soft'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save category table
        table_content = "\n".join(latex_table)
        with open(f'{output_dir}/{model}_category_table.tex', 'w') as f:
            f.write(table_content)
        
        # Save macro/micro table
        macro_micro_content = "\n".join(macro_micro_table)
        with open(f'{output_dir}/{model}_macro_micro_table.tex', 'w') as f:
            f.write(macro_micro_content)
        
        print(f"Saved {model.upper()} tables to {output_dir}/{model}_category_table.tex and {output_dir}/{model}_macro_micro_table.tex")

def main():
    """Main function to run comprehensive F1 analysis and generate LaTeX tables"""
    parser = argparse.ArgumentParser(description='Comprehensive F1 analysis and LaTeX table generation')
    parser.add_argument('--output_dir', type=str, default='output/f1/soft', help='Output directory for results')
    args = parser.parse_args()
    
    # Store all results
    all_results = defaultdict(dict)
    
    # Process each source
    for source in SOURCES:
        print(f"\n{'='*60}")
        print(f"Processing {source}")
        print(f"{'='*60}")
        
        # Load soft labels
        soft_labels = load_soft_labels(source)
        if soft_labels is None:
            continue
        
        # Process each model and shot type
        for model in MODELS:
            for shot_type in SHOT_TYPES:
                print(f"\n--- {model} {shot_type} ---")
                
                # Load model predictions
                predictions = load_model_predictions(source, model, shot_type)
                if predictions is None:
                    continue
                
                # Map columns to soft labels
                predictions_mapped = map_columns_to_soft_labels(predictions, source)
                
                # Calculate metrics
                metrics = calculate_metrics(predictions_mapped, soft_labels, source)
                if metrics:
                    all_results[source][f"{model}_{shot_type}"] = metrics
                    print(f"Macro F1: {metrics['macro_f1']:.4f}")
        
        # Load BERT results
        bert_results = load_bert_results(source)
        if bert_results:
            all_results[source]['bert_finetuned'] = {
                'macro_f1': bert_results['macro_f1'],
                'micro_f1': bert_results['micro_f1'],
                'category_metrics': bert_results['label_f1_scores'],
                'num_categories': bert_results['num_labels']
            }
    
    # Create summary table
    print(f"\n{'='*80}")
    print("COMPREHENSIVE MODEL COMPARISON RESULTS")
    print(f"{'='*80}")
    
    summary_data = []
    
    for source in SOURCES:
        if source in all_results:
            for model_key, metrics in all_results[source].items():
                summary_data.append({
                    'Source': source,
                    'Model': model_key,
                    'Macro_F1': metrics['macro_f1'],
                    'Micro_F1': metrics['micro_f1'],
                    'Num_Categories': metrics['num_categories']
                })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Keep full precision for calculations, only round for display
    summary_df_rounded = summary_df.copy()
    # Don't round here - keep full precision for all calculations
    
    # Save results
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full precision data
    summary_df.to_csv(f'{output_dir}/comprehensive_model_comparison.csv', index=False)
    
    # Also save rounded version for easy reading
    summary_df_rounded_display = summary_df.copy()
    summary_df_rounded_display['Macro_F1'] = summary_df_rounded_display['Macro_F1'].round(2)
    summary_df_rounded_display['Micro_F1'] = summary_df_rounded_display['Micro_F1'].round(2)
    summary_df_rounded_display.to_csv(f'{output_dir}/comprehensive_model_comparison_rounded.csv', index=False)
    
    # Print summary table
    print("\nSummary Table:")
    # Round to 2 significant figures only for display
    summary_df_display = summary_df_rounded.copy()
    summary_df_display['Macro_F1'] = summary_df_display['Macro_F1'].round(2)
    summary_df_display['Micro_F1'] = summary_df_display['Micro_F1'].round(2)
    print(summary_df_display.to_string(index=False, float_format='%.2f'))
    
    # Calculate overall averages
    print(f"\n{'='*80}")
    print("OVERALL AVERAGES")
    print(f"{'='*80}")
    
    # Average by model type - Macro F1
    model_averages_macro = summary_df.groupby('Model')['Macro_F1'].agg(['mean', 'std', 'count'])
    print("\nAverage Macro F1 by Model:")
    # Round to 2 significant figures only for display
    print(model_averages_macro.round(2))
    
    # Average by model type - Micro F1
    model_averages_micro = summary_df.groupby('Model')['Micro_F1'].agg(['mean', 'std', 'count'])
    print("\nAverage Micro F1 by Model:")
    # Round to 2 significant figures only for display
    print(model_averages_micro.round(2))
    
    # Average by source - Macro F1
    source_averages_macro = summary_df.groupby('Source')['Macro_F1'].agg(['mean', 'std', 'count'])
    print("\nAverage Macro F1 by Source:")
    # Round to 2 significant figures only for display
    print(source_averages_macro.round(2))
    
    # Average by source - Micro F1
    source_averages_micro = summary_df.groupby('Source')['Micro_F1'].agg(['mean', 'std', 'count'])
    print("\nAverage Micro F1 by Source:")
    # Round to 2 significant figures only for display
    print(source_averages_micro.round(2))
    
    # Overall averages
    overall_avg_macro = summary_df['Macro_F1'].mean()
    overall_avg_micro = summary_df['Micro_F1'].mean()
    print(f"\nOverall Average Macro F1: {overall_avg_macro:.2f}")
    print(f"Overall Average Micro F1: {overall_avg_micro:.2f}")
    
    # Generate LaTeX tables
    print(f"\n{'='*80}")
    print("GENERATING LATEX TABLES")
    print(f"{'='*80}")
    
    # Create main table
    main_table = create_latex_table(summary_df)
    with open(f'{output_dir}/macro_f1_table.tex', 'w') as f:
        f.write(main_table)
    print("Main table saved as: {}/macro_f1_table.tex".format(output_dir))
    
    # Create detailed table
    detailed_table = create_detailed_latex_table(summary_df)
    with open(f'{output_dir}/detailed_macro_f1_table.tex', 'w') as f:
        f.write(detailed_table)
    print("Detailed table saved as: {}/detailed_macro_f1_table.tex".format(output_dir))
    
    # Create individual model category tables
    create_individual_model_tables(all_results)
    
    # Save detailed results
    with open(f'{output_dir}/detailed_model_comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir}/")
    print("\n=== MAIN TABLE ===")
    print(main_table)

if __name__ == "__main__":
    main() 