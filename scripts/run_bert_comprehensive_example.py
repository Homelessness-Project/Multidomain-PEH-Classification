#!/usr/bin/env python3
"""
Example Script for BERT Comprehensive Analysis

This script demonstrates how to use the comprehensive BERT multiclass analysis
to train models and get detailed F1 scores for each category for each source.

Usage:
    python scripts/run_bert_comprehensive_example.py
"""

import subprocess
import sys
import os
from datetime import datetime

def run_comprehensive_analysis(mode='evaluate_only', sources=None):
    """Run comprehensive BERT analysis"""
    
    if sources is None:
        sources = ['reddit', 'x', 'news', 'meeting_minutes']
    
    print(f"🚀 BERT COMPREHENSIVE MULTICLASS ANALYSIS")
    print(f"Mode: {mode}")
    print(f"Sources: {', '.join(sources)}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Build command
    cmd = [
        'python', 'scripts/bert_comprehensive_analysis.py',
        '--mode', mode,
        '--sources'
    ] + sources + [
        '--epochs', '3',  # Reduced for example
        '--batch_size', '16',
        '--learning_rate', '2e-5',
        '--max_length', '256',
        '--test_size', '0.2',
        '--val_size', '0.1',
        '--seed', '42'
    ]
    
    try:
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, text=True)
        print(f"\n✅ Analysis completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running analysis: {e}")
        return False

def main():
    """Main function to run comprehensive analysis examples"""
    
    print("BERT COMPREHENSIVE MULTICLASS ANALYSIS EXAMPLES")
    print("="*60)
    
    # Example 1: Evaluate existing models (if available)
    print("\n1. EVALUATING EXISTING MODELS")
    print("-" * 40)
    print("This will evaluate BERT models if they exist, or show errors if not trained yet.")
    
    success = run_comprehensive_analysis(mode='evaluate_only')
    
    if success:
        print("\n📊 Results Summary:")
        print("- Check output/comprehensive_analysis/bert_summary_results.csv for overall performance")
        print("- Check output/comprehensive_analysis/bert_detailed_results.csv for per-category F1 scores")
        print("- Check output/comprehensive_analysis/plots/ for visualizations")
    
    # Example 2: Train and evaluate (commented out to avoid long training)
    print("\n2. TRAINING AND EVALUATING MODELS")
    print("-" * 40)
    print("To train new models and evaluate them, run:")
    print("python scripts/bert_comprehensive_analysis.py --mode train_evaluate")
    print("\nNote: This will take several hours to complete training for all sources.")
    
    # Example 3: Single source example
    print("\n3. SINGLE SOURCE EXAMPLE")
    print("-" * 40)
    print("To train and evaluate just Reddit data:")
    print("python scripts/bert_comprehensive_analysis.py --mode train_evaluate --sources reddit")
    
    # Example 4: Custom parameters
    print("\n4. CUSTOM PARAMETERS EXAMPLE")
    print("-" * 40)
    print("To use custom training parameters:")
    print("python scripts/bert_comprehensive_analysis.py --mode train_evaluate --epochs 10 --batch_size 32 --learning_rate 1e-5")
    
    print("\n" + "="*60)
    print("📁 OUTPUT FILES:")
    print("- output/comprehensive_analysis/bert_summary_results.csv")
    print("- output/comprehensive_analysis/bert_detailed_results.csv")
    print("- output/comprehensive_analysis/plots/overall_performance.png")
    print("- output/comprehensive_analysis/plots/category_performance_heatmap.png")
    print("="*60)

if __name__ == "__main__":
    main() 