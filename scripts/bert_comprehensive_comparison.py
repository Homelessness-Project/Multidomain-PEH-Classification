#!/usr/bin/env python3
"""
Comprehensive BERT Comparison Script

This script compares different BERT approaches:
1. Original BERT (full fine-tuning)
2. Frozen BERT (99% frozen - best from analysis)
3. Improved BERT (focal loss, soft labels, augmentation)
4. Improved RoBERTa (focal loss, soft labels, augmentation)

Usage:
    python scripts/bert_comprehensive_comparison.py --source reddit --epochs 3
"""

import argparse
import pandas as pd
import numpy as np
import torch
import subprocess
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def run_bert_experiment(source, model_type, epochs=3):
    """Run a specific BERT experiment"""
    
    print(f"\n{'='*60}")
    print(f"RUNNING {model_type.upper()} EXPERIMENT")
    print(f"{'='*60}")
    
    if model_type == "original":
        cmd = [
            "python", "scripts/bert_multiclass_classifier.py",
            "--source", source, "--dataset", "gold_subset", "--mode", "train",
            "--epochs", str(epochs)
        ]
    elif model_type == "frozen":
        cmd = [
            "python", "scripts/bert_frozen_layers_classifier.py",
            "--source", source, "--dataset", "gold_subset", "--mode", "train",
            "--epochs", str(epochs)
        ]
    elif model_type == "improved_bert":
        cmd = [
            "python", "scripts/bert_improved_classifier.py",
            "--source", source, "--model", "bert-base-uncased", "--mode", "train",
            "--epochs", str(epochs), "--use_focal_loss", "--use_soft_labels", "--augment"
        ]
    elif model_type == "improved_roberta":
        cmd = [
            "python", "scripts/bert_improved_classifier.py",
            "--source", source, "--model", "roberta-base", "--mode", "train",
            "--epochs", str(epochs), "--use_focal_loss", "--use_soft_labels", "--augment"
        ]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        if result.returncode == 0:
            print(f"✅ {model_type} experiment completed successfully")
            return True
        else:
            print(f"❌ {model_type} experiment failed:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {model_type} experiment timed out")
        return False
    except Exception as e:
        print(f"❌ {model_type} experiment error: {e}")
        return False

def load_results(source, model_type):
    """Load results from a specific experiment"""
    
    if model_type == "original":
        results_file = f"output/{source}/bert/bert_metrics_{source}.json"
    elif model_type == "frozen":
        results_file = f"output/{source}/bert_frozen/bert_frozen_metrics_{source}.json"
    elif model_type in ["improved_bert", "improved_roberta"]:
        results_file = f"output/{source}/bert_improved/bert_improved_metrics_{source}.json"
    else:
        return None
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    else:
        print(f"Results file not found: {results_file}")
        return None

def create_comparison_visualization(results, source):
    """Create visualization comparing all approaches"""
    
    os.makedirs('output/bert_comparison', exist_ok=True)
    
    # Prepare data
    model_names = []
    macro_f1s = []
    micro_f1s = []
    
    for model_type, result in results.items():
        if result:
            model_names.append(model_type.replace('_', ' ').title())
            macro_f1s.append(result['macro_f1'])
            micro_f1s.append(result['micro_f1'])
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Macro F1 comparison
    bars1 = ax1.bar(model_names, macro_f1s, alpha=0.7, color='skyblue')
    ax1.set_title(f'Macro F1 Comparison - {source.title()}')
    ax1.set_ylabel('Macro F1 Score')
    ax1.set_ylim(0, max(macro_f1s) * 1.1)
    
    # Add value labels on bars
    for bar, value in zip(bars1, macro_f1s):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Micro F1 comparison
    bars2 = ax2.bar(model_names, micro_f1s, alpha=0.7, color='lightcoral')
    ax2.set_title(f'Micro F1 Comparison - {source.title()}')
    ax2.set_ylabel('Micro F1 Score')
    ax2.set_ylim(0, max(micro_f1s) * 1.1)
    
    # Add value labels on bars
    for bar, value in zip(bars2, micro_f1s):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'output/bert_comparison/bert_comparison_{source}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to output/bert_comparison/bert_comparison_{source}.png")

def create_detailed_comparison_table(results, source):
    """Create detailed comparison table"""
    
    comparison_data = []
    
    for model_type, result in results.items():
        if result:
            comparison_data.append({
                'Model': model_type.replace('_', ' ').title(),
                'Macro F1': f"{result['macro_f1']:.4f}",
                'Micro F1': f"{result['micro_f1']:.4f}",
                'Best Threshold': f"{result['best_threshold']:.2f}",
                'Test Size': result['test_size'],
                'Num Labels': result['num_labels']
            })
    
    df = pd.DataFrame(comparison_data)
    
    # Save comparison table
    os.makedirs('output/bert_comparison', exist_ok=True)
    df.to_csv(f'output/bert_comparison/bert_comparison_{source}.csv', index=False)
    
    print(f"\n{'='*80}")
    print(f"BERT COMPREHENSIVE COMPARISON - {source.upper()}")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Comprehensive BERT Comparison')
    parser.add_argument('--source', type=str, required=True, 
                       choices=['reddit', 'x', 'news', 'meeting_minutes'],
                       help='Data source to use')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--skip_experiments', action='store_true', 
                       help='Skip running experiments, only analyze existing results')
    
    args = parser.parse_args()
    
    print(f"BERT COMPREHENSIVE COMPARISON")
    print(f"Source: {args.source}")
    print(f"Epochs: {args.epochs}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define experiments to run
    experiments = [
        "original",
        "frozen", 
        "improved_bert",
        "improved_roberta"
    ]
    
    results = {}
    
    if not args.skip_experiments:
        # Run experiments
        for experiment in experiments:
            success = run_bert_experiment(args.source, experiment, args.epochs)
            if success:
                print(f"✅ {experiment} completed")
            else:
                print(f"❌ {experiment} failed")
    
    # Load results
    print(f"\n{'='*60}")
    print("LOADING RESULTS")
    print(f"{'='*60}")
    
    for experiment in experiments:
        result = load_results(args.source, experiment)
        if result:
            results[experiment] = result
            print(f"✅ Loaded {experiment} results")
        else:
            print(f"❌ Could not load {experiment} results")
    
    if results:
        # Create comparison
        comparison_df = create_detailed_comparison_table(results, args.source)
        create_comparison_visualization(results, args.source)
        
        # Find best performing model
        best_macro = max(results.items(), key=lambda x: x[1]['macro_f1'] if x[1] else 0)
        best_micro = max(results.items(), key=lambda x: x[1]['micro_f1'] if x[1] else 0)
        
        print(f"\n{'='*60}")
        print("BEST PERFORMING MODELS")
        print(f"{'='*60}")
        print(f"Best Macro F1: {best_macro[0]} ({best_macro[1]['macro_f1']:.4f})")
        print(f"Best Micro F1: {best_micro[0]} ({best_micro[1]['micro_f1']:.4f})")
        
        print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("No results to analyze")

if __name__ == "__main__":
    main()

