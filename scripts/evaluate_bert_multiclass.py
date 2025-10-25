#!/usr/bin/env python3
"""
Evaluate BERT Multiclass Classifier

This script evaluates the BERT multiclass classifier and compares it with other models.
It provides comprehensive metrics and visualizations for all 16 categories.

Usage:
    python scripts/evaluate_bert_multiclass.py --source reddit
    python scripts/evaluate_bert_multiclass.py --source all
"""

import argparse
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report
import warnings
warnings.filterwarnings('ignore')

# Define all 16 categories
ALL_CATEGORIES = [
    # Comment Types (6)
    'ask a genuine question',
    'ask a rhetorical question',
    'provide a fact or claim',
    'provide an observation',
    'express their opinion',
    'express others opinions',
    # Critique Categories (3)
    'money aid allocation',
    'government critique',
    'societal critique',
    # Response Categories (1)
    'solutions/interventions',
    # Perception Types (5)
    'personal interaction',
    'media portrayal',
    'not in my backyard',
    'harmful generalization',
    'deserving/undeserving',
    # Racist Classification (1)
    'racist'
]

def load_bert_results(source):
    """Load BERT multiclass results"""
    metrics_path = f'output/{source}/bert/bert_metrics_{source}.json'
    if not os.path.exists(metrics_path):
        print(f"BERT metrics not found for {source}: {metrics_path}")
        return None
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics

def load_soft_labels(source):
    """Load soft labels for comparison"""
    soft_labels_path = f'output/annotation/soft_labels/{source}_soft_labels.csv'
    if not os.path.exists(soft_labels_path):
        print(f"Soft labels not found for {source}: {soft_labels_path}")
        return None
    
    return pd.read_csv(soft_labels_path)

def load_other_model_results(source, model, shot_type):
    """Load results from other models for comparison"""
    # Map model names to file patterns
    model_patterns = {
        'llama': f'output/{source}/llama/classified_comments_{source}_gold_subset_llama_{shot_type}_flags.csv',
        'qwen': f'output/{source}/qwen/classified_comments_{source}_gold_subset_qwen_{shot_type}_flags.csv',
        'gpt4': f'output/{source}/gpt4/classified_comments_{source}_gold_subset_gpt4_{shot_type}_flags.csv',
        'gemini': f'output/{source}/gemini/classified_comments_{source}_gold_subset_gemini_{shot_type}_flags.csv',
        'grok': f'output/{source}/grok/classified_comments_{source}_gold_subset_grok_{shot_type}_flags.csv'
    }
    
    file_path = model_patterns.get(model)
    if not file_path or not os.path.exists(file_path):
        print(f"Model results not found: {file_path}")
        return None
    
    return pd.read_csv(file_path)

def map_columns_to_categories(df, source):
    """Map model output columns to our 16 categories"""
    # Define mapping from model output columns to our categories
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
    
    # Create new DataFrame with mapped columns
    df_mapped = df.copy()
    for model_col, category in column_mapping.items():
        if model_col in df.columns:
            df_mapped[category] = df[model_col]
        else:
            print(f"Warning: Column {model_col} not found in model results")
            df_mapped[category] = 0
    
    return df_mapped

def calculate_metrics(predictions, true_labels, source):
    """Calculate comprehensive metrics"""
    if len(predictions) != len(true_labels):
        print(f"Warning: Prediction and true label lengths don't match")
        return None
    
    # Calculate per-category metrics
    category_metrics = {}
    for category in ALL_CATEGORIES:
        if category in predictions.columns and category in true_labels.columns:
            pred = predictions[category].values
            true = true_labels[category].values
            
            # Convert to binary if needed
            if pred.dtype == 'float64':
                pred = (pred > 0.5).astype(int)
            if true.dtype == 'float64':
                true = (true > 0.5).astype(int)
            
            precision, recall, f1, support = precision_recall_fscore_support(
                true, pred, average='binary', zero_division=0
            )
            
            category_metrics[category] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support
            }
    
    # Calculate macro and micro F1
    all_pred = []
    all_true = []
    
    for category in ALL_CATEGORIES:
        if category in predictions.columns and category in true_labels.columns:
            pred = predictions[category].values
            true = true_labels[category].values
            
            if pred.dtype == 'float64':
                pred = (pred > 0.5).astype(int)
            if true.dtype == 'float64':
                true = (true > 0.5).astype(int)
            
            all_pred.extend(pred)
            all_true.extend(true)
    
    macro_f1 = f1_score(all_true, all_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(all_true, all_pred, average='micro', zero_division=0)
    
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'category_metrics': category_metrics
    }

def create_comparison_plot(bert_results, other_results, source, output_dir):
    """Create comparison plot of F1 scores"""
    
    # Prepare data for plotting
    categories = []
    bert_f1s = []
    other_f1s = []
    
    for category in ALL_CATEGORIES:
        if category in bert_results['label_f1_scores']:
            categories.append(category)
            bert_f1s.append(bert_results['label_f1_scores'][category])
            
            # Get corresponding F1 from other model
            if category in other_results['category_metrics']:
                other_f1s.append(other_results['category_metrics'][category]['f1'])
            else:
                other_f1s.append(0)
    
    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'Category': categories,
        'BERT': bert_f1s,
        'Other Model': other_f1s
    })
    
    # Create plot
    plt.figure(figsize=(15, 8))
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, plot_data['BERT'], width, label='BERT', alpha=0.8)
    plt.bar(x + width/2, plot_data['Other Model'], width, label='Other Model', alpha=0.8)
    
    plt.xlabel('Categories')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Score Comparison: BERT vs Other Model ({source})')
    plt.xticks(x, categories, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = f'{output_dir}/bert_vs_other_comparison_{source}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to: {plot_path}")

def create_category_heatmap(bert_results, source, output_dir):
    """Create heatmap of F1 scores by category"""
    
    # Prepare data
    categories = []
    f1_scores = []
    
    for category in ALL_CATEGORIES:
        if category in bert_results['label_f1_scores']:
            categories.append(category)
            f1_scores.append(bert_results['label_f1_scores'][category])
    
    # Create heatmap data
    heatmap_data = pd.DataFrame({
        'Category': categories,
        'F1 Score': f1_scores
    })
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    
    # Reshape data for heatmap
    heatmap_matrix = heatmap_data['F1 Score'].values.reshape(1, -1)
    
    sns.heatmap(
        heatmap_matrix,
        annot=True,
        fmt='.3f',
        xticklabels=categories,
        yticklabels=['F1 Score'],
        cmap='RdYlBu_r',
        cbar_kws={'label': 'F1 Score'}
    )
    
    plt.title(f'BERT Multiclass F1 Scores by Category ({source})')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = f'{output_dir}/bert_category_heatmap_{source}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Category heatmap saved to: {plot_path}")

def print_detailed_results(bert_results, source):
    """Print detailed results"""
    
    print(f"\n{'='*80}")
    print(f"BERT MULTICLASS RESULTS FOR {source.upper()}")
    print(f"{'='*80}")
    
    print(f"Overall Metrics:")
    print(f"  Macro F1: {bert_results['macro_f1']:.4f}")
    print(f"  Micro F1: {bert_results['micro_f1']:.4f}")
    print(f"  Best Threshold: {bert_results['best_threshold']}")
    print(f"  Test Size: {bert_results['test_size']}")
    print(f"  Number of Categories: {bert_results['num_labels']}")
    
    print(f"\nPer-Category F1 Scores:")
    print(f"{'Category':<30} {'F1 Score':<10}")
    print(f"{'-'*40}")
    
    for category in ALL_CATEGORIES:
        if category in bert_results['label_f1_scores']:
            f1 = bert_results['label_f1_scores'][category]
            print(f"{category:<30} {f1:<10.4f}")
    
    # Find best and worst performing categories
    f1_scores = [(cat, bert_results['label_f1_scores'][cat]) 
                  for cat in ALL_CATEGORIES 
                  if cat in bert_results['label_f1_scores']]
    
    f1_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 5 Categories:")
    for i, (category, f1) in enumerate(f1_scores[:5]):
        print(f"  {i+1}. {category}: {f1:.4f}")
    
    print(f"\nBottom 5 Categories:")
    for i, (category, f1) in enumerate(f1_scores[-5:]):
        print(f"  {len(f1_scores)-4+i}. {category}: {f1:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate BERT Multiclass Classifier')
    parser.add_argument('--source', type=str, required=True,
                       choices=['reddit', 'x', 'news', 'meeting_minutes', 'all'],
                       help='Source to evaluate')
    parser.add_argument('--compare_with', type=str, default=None,
                       choices=['llama', 'qwen', 'gpt4', 'gemini', 'grok'],
                       help='Model to compare with')
    parser.add_argument('--shot_type', type=str, default='none',
                       choices=['none', 'reddit', 'x', 'news', 'meeting_minutes'],
                       help='Shot type for comparison model')
    parser.add_argument('--output_dir', type=str, default='output/evaluation',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    if args.source == 'all':
        sources = ['reddit', 'x', 'news', 'meeting_minutes']
    else:
        sources = [args.source]
    
    for source in sources:
        print(f"\n{'='*80}")
        print(f"EVALUATING {source.upper()}")
        print(f"{'='*80}")
        
        # Load BERT results
        bert_results = load_bert_results(source)
        if bert_results is None:
            print(f"Skipping {source} - no BERT results found")
            continue
        
        # Print detailed results
        print_detailed_results(bert_results, source)
        
        # Create category heatmap
        create_category_heatmap(bert_results, source, args.output_dir)
        
        # Compare with other model if specified
        if args.compare_with:
            print(f"\nComparing with {args.compare_with} ({args.shot_type} shot)...")
            
            other_results = load_other_model_results(source, args.compare_with, args.shot_type)
            if other_results is not None:
                # Load soft labels for comparison
                soft_labels = load_soft_labels(source)
                if soft_labels is not None:
                    # Map other model results to our categories
                    other_mapped = map_columns_to_categories(other_results, source)
                    
                    # Calculate metrics for other model
                    other_metrics = calculate_metrics(other_mapped, soft_labels, source)
                    
                    if other_metrics:
                        print(f"BERT Macro F1: {bert_results['macro_f1']:.4f}")
                        print(f"{args.compare_with} Macro F1: {other_metrics['macro_f1']:.4f}")
                        print(f"BERT Micro F1: {bert_results['micro_f1']:.4f}")
                        print(f"{args.compare_with} Micro F1: {other_metrics['micro_f1']:.4f}")
                        
                        # Create comparison plot
                        create_comparison_plot(bert_results, other_metrics, source, args.output_dir)
        
        print(f"\nEvaluation completed for {source}")

if __name__ == "__main__":
    main() 