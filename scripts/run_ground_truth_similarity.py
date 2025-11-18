#!/usr/bin/env python3
"""
Run Ground Truth Similarity Analysis
====================================

This script runs only the correct overlap analysis to test ground truth similarity
between model overlap. It computes:
1. Model-vs-Truth Jaccard of positive label sets (per model → vector)
2. Jaccard overlap between models of sets of correctly predicted labels (matrix)
3. Pairwise model-truth similarity: Three-way Jaccard (model1 ∩ model2 ∩ truth)
4. Joint correctness: How often both models are correct together
"""

import sys
import os

# Add scripts directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from cross_model_validation import CrossModelValidator
import json

def main():
    """Run ground truth similarity analysis"""
    print("="*60)
    print("GROUND TRUTH SIMILARITY ANALYSIS")
    print("="*60)
    
    # Create validator
    validator = CrossModelValidator(output_dir='output/cross_validation')
    
    # Load data (required before running analysis)
    print("\nLoading data...")
    validator.load_data()
    
    # Run only the correct overlap analysis
    print("\nRunning correct overlap analysis...")
    results = validator.correct_overlap_analysis()
    
    # Generate visualizations for correct overlap
    print("\nGenerating visualizations...")
    validator._plot_correct_overlap_plots()
    
    # Save results to JSON
    output_file = 'output/cross_validation/ground_truth_similarity_results.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Visualizations saved to: output/cross_validation/plots/")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for source, data in results.items():
        if source == 'overall':
            print(f"\n{source.upper()}:")
        else:
            print(f"\n{source.upper()}:")
        
        if 'model_keys' in data and 'model_truth_jaccard' in data:
            print(f"  Models analyzed: {len(data['model_keys'])}")
            print(f"  Model-vs-Truth Jaccard scores:")
            for i, model in enumerate(data['model_keys']):
                score = data['model_truth_jaccard'][i]
                print(f"    {model}: {score:.4f}")
            
            if 'correctness_overlap' in data:
                avg_overlap = data['correctness_overlap'][
                    data['correctness_overlap'] != 1.0
                ].mean() if data['correctness_overlap'].size > 0 else 0
                print(f"  Average correctness overlap: {avg_overlap:.4f}")
            
            if 'pairwise_truth_similarity' in data:
                avg_pairwise = data['pairwise_truth_similarity'][
                    data['pairwise_truth_similarity'] != 1.0
                ].mean() if data['pairwise_truth_similarity'].size > 0 else 0
                print(f"  Average pairwise model-truth similarity: {avg_pairwise:.4f}")
            
            if 'joint_correctness' in data:
                avg_joint = data['joint_correctness'][
                    data['joint_correctness'] != 1.0
                ].mean() if data['joint_correctness'].size > 0 else 0
                print(f"  Average joint correctness: {avg_joint:.4f}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()

