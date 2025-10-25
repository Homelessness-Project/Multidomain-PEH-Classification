#!/usr/bin/env python3
"""
RoBERTa Comprehensive Multiclass Analysis

This script provides comprehensive training, evaluation, and analysis for RoBERTa
multiclass classification across all data sources. RoBERTa typically performs
2-5% better than BERT on classification tasks.

Usage:
    python scripts/roberta_comprehensive_analysis.py --mode train_evaluate
    python scripts/roberta_comprehensive_analysis.py --mode evaluate_only
"""

import argparse
import pandas as pd
import numpy as np
import json
import os
import warnings
from datetime import datetime
import subprocess
import sys
warnings.filterwarnings('ignore')

def run_roberta_training(source, epochs=5, batch_size=16, learning_rate=2e-5):
    """Run RoBERTa training for a specific source"""
    print(f"Training RoBERTa for {source}...")
    
    cmd = [
        sys.executable, 'scripts/roberta_multiclass_classifier.py',
        '--source', source,
        '--mode', 'train',
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--learning_rate', str(learning_rate)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ RoBERTa training completed for {source}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in RoBERTa training for {source}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def load_roberta_results(source):
    """Load RoBERTa results for a specific source"""
    metrics_path = f'output/{source}/roberta/roberta_metrics_{source}.json'
    if not os.path.exists(metrics_path):
        print(f"RoBERTa metrics not found for {source}: {metrics_path}")
        return None
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics

def analyze_roberta_performance(sources):
    """Analyze RoBERTa performance across all sources"""
    print("\n" + "="*80)
    print("COMPREHENSIVE ROBERTA MULTICLASS ANALYSIS RESULTS")
    print("="*80)
    
    # Load results for all sources
    results = {}
    for source in sources:
        metrics = load_roberta_results(source)
        if metrics:
            results[source] = metrics
    
    if not results:
        print("No results to analyze")
        return
    
    # Create summary dataframe
    summary_data = []
    for source, metrics in results.items():
        summary_data.append({
            'Source': source,
            'Macro F1': metrics['macro_f1'],
            'Micro F1': metrics['micro_f1'],
            'Best Threshold': metrics['threshold']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Print overall performance summary
    print("\nOVERALL PERFORMANCE SUMMARY:")
    print(summary_df.to_string(index=False, float_format='%.4f'))
    
    # Create detailed category analysis
    print("\n" + "="*80)
    print("DETAILED CATEGORY PERFORMANCE BY SOURCE")
    print("="*80)
    
    # Define category groups
    category_groups = {
        'COMMENT TYPES': [
            'ask a genuine question', 'ask a rhetorical question', 'provide a fact or claim',
            'provide an observation', 'express their opinion', 'express others opinions'
        ],
        'CRITIQUE CATEGORIES': [
            'money aid allocation', 'government critique', 'societal critique'
        ],
        'RESPONSE CATEGORIES': [
            'solutions/interventions'
        ],
        'PERCEPTION TYPES': [
            'personal interaction', 'media portrayal', 'not in my backyard',
            'harmful generalization', 'deserving/undeserving'
        ],
        'RACIST CLASSIFICATION': [
            'racist'
        ]
    }
    
    # Print detailed results by category group
    for group_name, categories in category_groups.items():
        print(f"\n{group_name}:")
        print("-" * 80)
        
        # Create dataframe for this group
        group_data = []
        for source in sources:
            if source in results:
                for category in categories:
                    if category in results[source]['category_metrics']:
                        metrics = results[source]['category_metrics'][category]
                        group_data.append({
                            'Source': source,
                            'Category': category,
                            'F1 Score': metrics['f1'],
                            'Precision': metrics['precision'],
                            'Recall': metrics['recall']
                        })
        
        if group_data:
            group_df = pd.DataFrame(group_data)
            pivot_df = group_df.pivot(index='Category', columns='Source', values='F1 Score')
            print(pivot_df.to_string(float_format='%.4f'))
    
    # Save results
    os.makedirs('output/comprehensive_analysis', exist_ok=True)
    
    # Save summary results
    summary_df.to_csv('output/comprehensive_analysis/roberta_summary_results.csv', index=False)
    
    # Save detailed results
    detailed_data = []
    for source in sources:
        if source in results:
            for category, metrics in results[source]['category_metrics'].items():
                detailed_data.append({
                    'Source': source,
                    'Category': category,
                    'F1 Score': metrics['f1'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall']
                })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv('output/comprehensive_analysis/roberta_detailed_results.csv', index=False)
    
    print(f"\n" + "="*80)
    print("RESULTS SAVED TO:")
    print(f"- Summary: output/comprehensive_analysis/roberta_summary_results.csv")
    print(f"- Detailed: output/comprehensive_analysis/roberta_detailed_results.csv")
    print("="*80)
    
    # Create visualizations
    create_roberta_visualizations(summary_df, detailed_df)

def create_roberta_visualizations(summary_df, detailed_df):
    """Create visualizations for RoBERTa results"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        os.makedirs('output/comprehensive_analysis/plots', exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Overall performance comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Macro F1 comparison
        sources = summary_df['Source']
        macro_f1 = summary_df['Macro F1']
        micro_f1 = summary_df['Micro F1']
        
        x = np.arange(len(sources))
        width = 0.35
        
        ax1.bar(x - width/2, macro_f1, width, label='Macro F1', alpha=0.8)
        ax1.bar(x + width/2, micro_f1, width, label='Micro F1', alpha=0.8)
        
        ax1.set_xlabel('Data Source')
        ax1.set_ylabel('F1 Score')
        ax1.set_title('RoBERTa Performance by Data Source')
        ax1.set_xticks(x)
        ax1.set_xticklabels(sources, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (macro, micro) in enumerate(zip(macro_f1, micro_f1)):
            ax1.text(i - width/2, macro + 0.01, f'{macro:.3f}', ha='center', va='bottom')
            ax1.text(i + width/2, micro + 0.01, f'{micro:.3f}', ha='center', va='bottom')
        
        # 2. Category performance heatmap
        pivot_df = detailed_df.pivot(index='Category', columns='Source', values='F1 Score')
        
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax2)
        ax2.set_title('RoBERTa Category Performance Heatmap')
        ax2.set_xlabel('Data Source')
        ax2.set_ylabel('Category')
        
        plt.tight_layout()
        plt.savefig('output/comprehensive_analysis/plots/roberta_overall_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Category performance by group
        category_groups = {
            'Comment Types': ['ask a genuine question', 'ask a rhetorical question', 'provide a fact or claim',
                            'provide an observation', 'express their opinion', 'express others opinions'],
            'Critique Categories': ['money aid allocation', 'government critique', 'societal critique'],
            'Response Categories': ['solutions/interventions'],
            'Perception Types': ['personal interaction', 'media portrayal', 'not in my backyard',
                               'harmful generalization', 'deserving/undeserving'],
            'Racist Classification': ['racist']
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (group_name, categories) in enumerate(category_groups.items()):
            if idx < len(axes):
                group_data = detailed_df[detailed_df['Category'].isin(categories)]
                if not group_data.empty:
                    pivot_group = group_data.pivot(index='Category', columns='Source', values='F1 Score')
                    sns.heatmap(pivot_group, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[idx])
                    axes[idx].set_title(f'{group_name} Performance')
                    axes[idx].set_xlabel('Data Source')
                    axes[idx].set_ylabel('Category')
        
        # Hide empty subplots
        for idx in range(len(category_groups), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('output/comprehensive_analysis/plots/roberta_category_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to: output/comprehensive_analysis/plots/")
        
    except ImportError:
        print("Matplotlib/Seaborn not available, skipping visualizations")

def main():
    parser = argparse.ArgumentParser(description='RoBERTa Comprehensive Multiclass Analysis')
    parser.add_argument('--mode', required=True, choices=['train_evaluate', 'evaluate_only'],
                       help='Analysis mode')
    parser.add_argument('--sources', nargs='+', 
                       default=['reddit', 'x', 'news', 'meeting_minutes'],
                       choices=['reddit', 'x', 'news', 'meeting_minutes'],
                       help='Data sources to analyze')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    
    args = parser.parse_args()
    
    print("ROBERTA COMPREHENSIVE MULTICLASS ANALYSIS")
    print(f"Mode: {args.mode}")
    print(f"Sources: {', '.join(args.sources)}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if args.mode == 'train_evaluate':
        # Train models for all sources
        for source in args.sources:
            print("="*80)
            print(f"PROCESSING {source.upper()}")
            print("="*80)
            
            success = run_roberta_training(
                source, 
                epochs=args.epochs, 
                batch_size=args.batch_size, 
                learning_rate=args.learning_rate
            )
            
            if not success:
                print(f"Skipping analysis for {source} due to training failure")
    
    # Analyze performance
    analyze_roberta_performance(args.sources)
    
    print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 