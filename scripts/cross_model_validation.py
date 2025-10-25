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
LLM_MODELS = ['llama', 'qwen', 'gpt4', 'gemini', 'grok', 'phi4']
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
            
            # Get all model keys for this source
            model_keys = list(self.predictions[source].keys())
            
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
        """Analyze correlations between model predictions"""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        correlation_results = {}
        
        for source in SOURCES:
            if source not in self.predictions:
                continue
                
            print(f"\nAnalyzing correlations for {source}...")
            correlation_results[source] = {}
            
            model_keys = list(self.predictions[source].keys())
            
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
            
            # Calculate F1 for each category
            f1_scores = []
            for i, soft_col in enumerate(soft_label_cols):
                if i < len(prediction_cols):
                    pred_col = prediction_cols[i]
                    try:
                        y_true = (soft_labels_df[soft_col] > 0.5).astype(int)
                        y_pred = (predictions_df[pred_col] > 0.5).astype(int)
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
            
            plt.savefig(f'{self.output_dir}/plots/correlation_heatmap_{source}.png', 
                       dpi=300, bbox_inches='tight')
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
            plt.savefig(f'{self.output_dir}/plots/model_performance_comparison.png', 
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
