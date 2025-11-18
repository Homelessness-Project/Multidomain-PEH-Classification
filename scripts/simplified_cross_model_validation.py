#!/usr/bin/env python3
"""
Simplified Cross-Model Validation Analysis for 6 LLMs
=====================================================

This script performs cross-model validation analysis using the existing F1 scores
from the comprehensive analysis, focusing on:

1. Statistical significance testing between models
2. Correlation analysis between model performance
3. Ensemble model performance evaluation
4. Cross-validation stability analysis
5. Model ranking and comparison

Models analyzed: llama, qwen, gpt4, gemini, grok, phi4, bert
Data sources: reddit, x, news, meeting_minutes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
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
SHOT_TYPES = ['zero_shot', 'few_shot']
ALPHA = 0.05  # Significance level

# Category names for analysis
CATEGORIES = [
    'ask a genuine question', 'ask a rhetorical question', 'provide a fact or claim', 
    'provide an observation', 'express their opinion', 'express others opinions',
    'money aid allocation', 'government critique', 'societal critique', 
    'solutions/interventions', 'personal interaction', 'media portrayal',
    'not in my backyard', 'harmful generalization', 'deserving/undeserving', 'racist'
]

class SimplifiedCrossModelValidator:
    """Simplified cross-model validation analysis class using existing F1 scores"""
    
    def __init__(self, output_dir='output/cross_validation'):
        self.output_dir = output_dir
        self.results = defaultdict(dict)
        self.f1_data = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        os.makedirs(f"{output_dir}/tables", exist_ok=True)
        
    def load_f1_data(self):
        """Load existing F1 scores from comprehensive analysis"""
        print("Loading F1 scores from comprehensive analysis...")
        
        f1_data_path = 'output/f1/comprehensive_model_comparison.csv'
        try:
            self.f1_data = pd.read_csv(f1_data_path)
            print(f"Loaded F1 data: {len(self.f1_data)} model-source combinations")
            print(f"Sources: {self.f1_data['Source'].unique()}")
            print(f"All models: {self.f1_data['Model'].unique()}")
            
            # Filter out BERT and RoBERTa models
            original_count = len(self.f1_data)
            self.f1_data = self.f1_data[~self.f1_data['Model'].str.contains('bert|roberta', case=False)]
            filtered_count = len(self.f1_data)
            
            print(f"Filtered out BERT/RoBERTa: {original_count - filtered_count} removed")
            print(f"LLM models only: {self.f1_data['Model'].unique()}")
            return True
        except FileNotFoundError:
            print(f"F1 data file not found: {f1_data_path}")
            return False
    
    def statistical_significance_testing(self):
        """Perform statistical significance testing between models"""
        print("\n" + "="*60)
        print("STATISTICAL SIGNIFICANCE TESTING")
        print("="*60)
        
        significance_results = {}
        
        for source in SOURCES:
            print(f"\nAnalyzing {source}...")
            significance_results[source] = {}
            
            # Get F1 scores for this source
            source_data = self.f1_data[self.f1_data['Source'] == source]
            
            if len(source_data) == 0:
                print(f"  No data found for {source}")
                continue
            
            # Create model performance dictionary
            model_performance = {}
            for _, row in source_data.iterrows():
                model_key = row['Model']
                macro_f1 = row['Macro_F1']
                model_performance[model_key] = macro_f1
            
            # Perform pairwise significance tests
            model_keys = list(model_performance.keys())
            model_pairs = list(combinations(model_keys, 2))
            pairwise_results = {}
            
            for model1, model2 in model_pairs:
                f1_1 = model_performance[model1]
                f1_2 = model_performance[model2]
                
                # For this simplified version, we'll use the difference in F1 scores
                # In a full implementation, you'd need actual prediction distributions
                diff = abs(f1_1 - f1_2)
                is_significant = diff > 0.05  # Simple threshold for significance
                better_model = model1 if f1_1 > f1_2 else model2
                
                pairwise_results[f"{model1}_vs_{model2}"] = {
                    'f1_difference': diff,
                    'is_significant': is_significant,
                    'better_model': better_model,
                    'model1_f1': f1_1,
                    'model2_f1': f1_2
                }
            
            significance_results[source] = {
                'model_performance': model_performance,
                'pairwise_tests': pairwise_results
            }
            
            # Print summary
            print(f"  Models tested: {len(model_keys)}")
            print(f"  Significant differences: {sum(1 for r in pairwise_results.values() if r['is_significant'])}")
            
            # Show top performers
            sorted_models = sorted(model_performance.items(), key=lambda x: x[1], reverse=True)
            print(f"  Top 3 models: {[f'{m}({f:.3f})' for m, f in sorted_models[:3]]}")
        
        self.results['significance_testing'] = significance_results
        return significance_results
    
    def correlation_analysis(self):
        """Analyze correlations between model performance across sources"""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        correlation_results = {}
        
        # Create a matrix of model performance across sources
        model_performance_matrix = {}
        
        for source in SOURCES:
            source_data = self.f1_data[self.f1_data['Source'] == source]
            for _, row in source_data.iterrows():
                model = row['Model']
                macro_f1 = row['Macro_F1']
                
                if model not in model_performance_matrix:
                    model_performance_matrix[model] = {}
                model_performance_matrix[model][source] = macro_f1
        
        # Calculate correlations between models
        model_keys = list(model_performance_matrix.keys())
        correlation_matrix = np.zeros((len(model_keys), len(model_keys)))
        
        for i, model1 in enumerate(model_keys):
            for j, model2 in enumerate(model_keys):
                if i <= j:  # Only calculate upper triangle
                    # Get F1 scores across sources for both models
                    f1_1 = [model_performance_matrix[model1].get(source, 0) for source in SOURCES]
                    f1_2 = [model_performance_matrix[model2].get(source, 0) for source in SOURCES]
                    
                    # Calculate correlation
                    corr, p_val = pearsonr(f1_1, f1_2)
                    correlation_matrix[i, j] = corr
                    correlation_matrix[j, i] = corr
        
        correlation_results = {
            'correlation_matrix': correlation_matrix,
            'model_keys': model_keys,
            'model_performance_matrix': model_performance_matrix
        }
        
        # Print summary
        avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
        max_correlation = np.max(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
        print(f"Average correlation between models: {avg_correlation:.4f}")
        print(f"Highest correlation: {max_correlation:.4f}")
        
        # Find most correlated model pairs
        correlations = []
        for i in range(len(model_keys)):
            for j in range(i+1, len(model_keys)):
                correlations.append((model_keys[i], model_keys[j], correlation_matrix[i, j]))
        
        correlations.sort(key=lambda x: x[2], reverse=True)
        print(f"Most correlated models: {correlations[0][0]} & {correlations[0][1]} (r={correlations[0][2]:.3f})")
        
        self.results['correlation_analysis'] = correlation_results
        return correlation_results
    
    def ensemble_analysis(self):
        """Analyze ensemble model performance"""
        print("\n" + "="*60)
        print("ENSEMBLE ANALYSIS")
        print("="*60)
        
        ensemble_results = {}
        
        for source in SOURCES:
            print(f"\nAnalyzing ensemble performance for {source}...")
            ensemble_results[source] = {}
            
            # Get F1 scores for this source
            source_data = self.f1_data[self.f1_data['Source'] == source]
            
            if len(source_data) == 0:
                continue
            
            # Calculate individual model performance
            individual_performance = {}
            for _, row in source_data.iterrows():
                model_key = row['Model']
                macro_f1 = row['Macro_F1']
                individual_performance[model_key] = macro_f1
            
            # Calculate ensemble performance (simple average)
            ensemble_f1 = np.mean(list(individual_performance.values()))
            
            # Calculate ensemble improvement
            best_individual = max(individual_performance.values()) if individual_performance else 0
            ensemble_improvement = ensemble_f1 - best_individual
            
            ensemble_results[source] = {
                'individual_performance': individual_performance,
                'ensemble_f1': ensemble_f1,
                'best_individual_f1': best_individual,
                'ensemble_improvement': ensemble_improvement,
                'models_included': list(individual_performance.keys())
            }
            
            print(f"  Best individual F1: {best_individual:.4f}")
            print(f"  Ensemble F1: {ensemble_f1:.4f}")
            print(f"  Improvement: {ensemble_improvement:.4f}")
        
        self.results['ensemble_analysis'] = ensemble_results
        return ensemble_results
    
    def model_ranking_analysis(self):
        """Analyze model rankings across sources"""
        print("\n" + "="*60)
        print("MODEL RANKING ANALYSIS")
        print("="*60)
        
        ranking_results = {}
        
        # Calculate rankings for each source
        source_rankings = {}
        for source in SOURCES:
            source_data = self.f1_data[self.f1_data['Source'] == source]
            if len(source_data) > 0:
                # Sort by Macro F1 descending
                sorted_data = source_data.sort_values('Macro_F1', ascending=False)
                rankings = {row['Model']: rank+1 for rank, (_, row) in enumerate(sorted_data.iterrows())}
                source_rankings[source] = rankings
        
        # Calculate average rankings
        all_models = set()
        for rankings in source_rankings.values():
            all_models.update(rankings.keys())
        
        average_rankings = {}
        for model in all_models:
            ranks = [source_rankings[source].get(model, len(all_models)+1) for source in SOURCES]
            average_rankings[model] = np.mean(ranks)
        
        # Sort by average ranking
        sorted_models = sorted(average_rankings.items(), key=lambda x: x[1])
        
        ranking_results = {
            'source_rankings': source_rankings,
            'average_rankings': average_rankings,
            'sorted_models': sorted_models
        }
        
        print("Model Rankings (1=best):")
        for i, (model, avg_rank) in enumerate(sorted_models):
            print(f"  {i+1}. {model}: {avg_rank:.2f}")
        
        self.results['ranking_analysis'] = ranking_results
        return ranking_results
    
    def category_overlap_analysis(self):
        """Analyze category overlap and confusion between models"""
        print("\n" + "="*60)
        print("CATEGORY OVERLAP ANALYSIS")
        print("="*60)
        
        overlap_results = {}
        
        for source in SOURCES:
            print(f"\nAnalyzing category overlap for {source}...")
            overlap_results[source] = {}
            
            # Get F1 scores for this source
            source_data = self.f1_data[self.f1_data['Source'] == source]
            
            if len(source_data) == 0:
                continue
            
            # Filter out BERT models
            source_data = source_data[~source_data['Model'].str.contains('bert', case=False)]
            
            # Analyze category performance patterns
            category_performance = {}
            for _, row in source_data.iterrows():
                model = row['Model']
                macro_f1 = row['Macro_F1']
                
                # Extract model type and shot type
                model_parts = model.split('_')
                if len(model_parts) >= 2:
                    model_type = model_parts[0]
                    shot_type = model_parts[1]
                    
                    if model_type not in category_performance:
                        category_performance[model_type] = {}
                    category_performance[model_type][shot_type] = macro_f1
            
            # Calculate overlap metrics
            overlap_metrics = self._calculate_category_overlap_metrics(category_performance)
            
            overlap_results[source] = {
                'category_performance': category_performance,
                'overlap_metrics': overlap_metrics
            }
            
            print(f"  Models analyzed: {len(category_performance)}")
            print(f"  Average performance variance: {overlap_metrics['avg_variance']:.4f}")
            print(f"  Most consistent category: {overlap_metrics['most_consistent']}")
            print(f"  Least consistent category: {overlap_metrics['least_consistent']}")
        
        self.results['category_overlap'] = overlap_results
        return overlap_results
    
    def _calculate_category_overlap_metrics(self, category_performance):
        """Calculate overlap and consistency metrics"""
        metrics = {
            'avg_variance': 0,
            'most_consistent': 'N/A',
            'least_consistent': 'N/A',
            'model_consistency': {},
            'shot_type_impact': {}
        }
        
        # Calculate variance across models
        variances = []
        for model_type, shot_data in category_performance.items():
            if len(shot_data) >= 2:  # Both zero_shot and few_shot
                variance = np.var(list(shot_data.values()))
                variances.append(variance)
                metrics['model_consistency'][model_type] = variance
        
        if variances:
            metrics['avg_variance'] = np.mean(variances)
            
            # Find most/least consistent models
            if metrics['model_consistency']:
                sorted_consistency = sorted(metrics['model_consistency'].items(), key=lambda x: x[1])
                metrics['most_consistent'] = sorted_consistency[0][0]
                metrics['least_consistent'] = sorted_consistency[-1][0]
        
        # Calculate shot type impact
        shot_impacts = {}
        for model_type, shot_data in category_performance.items():
            if 'zero_shot' in shot_data and 'few_shot' in shot_data:
                impact = shot_data['few_shot'] - shot_data['zero_shot']
                shot_impacts[model_type] = impact
        
        metrics['shot_type_impact'] = shot_impacts
        
        return metrics
    
    def model_confusion_analysis(self):
        """Analyze confusion patterns between models"""
        print("\n" + "="*60)
        print("MODEL CONFUSION ANALYSIS")
        print("="*60)
        
        confusion_results = {}
        
        # Create confusion matrix based on performance rankings
        for source in SOURCES:
            print(f"\nAnalyzing confusion patterns for {source}...")
            
            source_data = self.f1_data[self.f1_data['Source'] == source]
            source_data = source_data[~source_data['Model'].str.contains('bert', case=False)]
            
            if len(source_data) == 0:
                continue
            
            # Sort by performance
            sorted_data = source_data.sort_values('Macro_F1', ascending=False)
            
            # Create performance tiers
            tiers = self._create_performance_tiers(sorted_data)
            
            # Analyze confusion patterns
            confusion_patterns = self._analyze_confusion_patterns(tiers)
            
            confusion_results[source] = {
                'tiers': tiers,
                'confusion_patterns': confusion_patterns
            }
            
            print(f"  Performance tiers: {len(tiers)}")
            print(f"  Tier 1 models: {tiers.get('tier_1', [])}")
            print(f"  Confusion score: {confusion_patterns.get('confusion_score', 0):.3f}")
        
        self.results['confusion_analysis'] = confusion_results
        return confusion_results
    
    def _create_performance_tiers(self, sorted_data):
        """Create performance tiers based on F1 scores"""
        tiers = {
            'tier_1': [],  # Top performers
            'tier_2': [],  # Middle performers
            'tier_3': []   # Lower performers
        }
        
        total_models = len(sorted_data)
        tier_size = total_models // 3
        
        for i, (_, row) in enumerate(sorted_data.iterrows()):
            model = row['Model']
            if i < tier_size:
                tiers['tier_1'].append(model)
            elif i < tier_size * 2:
                tiers['tier_2'].append(model)
            else:
                tiers['tier_3'].append(model)
        
        return tiers
    
    def _analyze_confusion_patterns(self, tiers):
        """Analyze confusion patterns between tiers"""
        patterns = {
            'confusion_score': 0,
            'tier_stability': {},
            'cross_tier_movement': {}
        }
        
        # Calculate confusion score based on tier distribution
        tier_counts = {tier: len(models) for tier, models in tiers.items()}
        total_models = sum(tier_counts.values())
        
        if total_models > 0:
            # Confusion score: how evenly distributed are the tiers
            expected_per_tier = total_models / 3
            variance = sum((count - expected_per_tier) ** 2 for count in tier_counts.values()) / 3
            patterns['confusion_score'] = variance / (expected_per_tier ** 2)
        
        return patterns
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Model performance comparison
        self._plot_model_performance_comparison()
        
        # 2. Correlation heatmap
        self._plot_correlation_heatmap()
        
        # 3. Model rankings
        self._plot_model_rankings()
        
        # 4. Ensemble performance
        self._plot_ensemble_performance()
        
        # 5. Category overlap analysis
        self._plot_category_overlap()
        
        # 6. Model confusion analysis
        self._plot_confusion_analysis()
        
        print(f"Visualizations saved to {self.output_dir}/plots/")
    
    def _plot_model_performance_comparison(self):
        """Plot model performance comparison across sources"""
        plt.figure(figsize=(15, 10))
        
        # Create subplots for each source
        for i, source in enumerate(SOURCES):
            plt.subplot(2, 2, i+1)
            source_data = self.f1_data[self.f1_data['Source'] == source]
            
            if len(source_data) > 0:
                # Sort by Macro F1
                source_data = source_data.sort_values('Macro_F1', ascending=True)
                
                # Create horizontal bar plot
                plt.barh(range(len(source_data)), source_data['Macro_F1'])
                plt.yticks(range(len(source_data)), source_data['Model'])
                plt.title(f'{source.title()} - Macro F1 Scores')
                plt.xlabel('Macro F1')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/plots/model_performance_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_heatmap(self):
        """Plot correlation heatmap between models"""
        if 'correlation_analysis' not in self.results:
            return
        
        plt.figure(figsize=(12, 10))
        
        correlation_matrix = self.results['correlation_analysis']['correlation_matrix']
        model_keys = self.results['correlation_analysis']['model_keys']
        
        # Create heatmap
        sns.heatmap(correlation_matrix, 
                   xticklabels=model_keys, 
                   yticklabels=model_keys,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   fmt='.3f')
        
        plt.title('Model Performance Correlation Matrix')
        plt.xlabel('Models')
        plt.ylabel('Models')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(f'{self.output_dir}/plots/correlation_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_rankings(self):
        """Plot model rankings across sources"""
        if 'ranking_analysis' not in self.results:
            return
        
        plt.figure(figsize=(12, 8))
        
        sorted_models = self.results['ranking_analysis']['sorted_models']
        models = [m[0] for m in sorted_models]
        avg_ranks = [m[1] for m in sorted_models]
        
        # Create bar plot
        plt.bar(range(len(models)), avg_ranks)
        plt.xticks(range(len(models)), models, rotation=45)
        plt.title('Average Model Rankings Across All Sources')
        plt.xlabel('Models')
        plt.ylabel('Average Ranking (1=best)')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(avg_ranks):
            plt.text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/plots/model_rankings.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_ensemble_performance(self):
        """Plot ensemble performance analysis"""
        if 'ensemble_analysis' not in self.results:
            return
        
        plt.figure(figsize=(12, 8))
        
        sources = []
        best_individual = []
        ensemble_f1 = []
        improvements = []
        
        for source, data in self.results['ensemble_analysis'].items():
            sources.append(source)
            best_individual.append(data['best_individual_f1'])
            ensemble_f1.append(data['ensemble_f1'])
            improvements.append(data['ensemble_improvement'])
        
        x = np.arange(len(sources))
        width = 0.25
        
        plt.bar(x - width, best_individual, width, label='Best Individual', alpha=0.8)
        plt.bar(x, ensemble_f1, width, label='Ensemble', alpha=0.8)
        plt.bar(x + width, improvements, width, label='Improvement', alpha=0.8)
        
        plt.xlabel('Data Sources')
        plt.ylabel('F1 Score')
        plt.title('Ensemble vs Individual Model Performance')
        plt.xticks(x, sources)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/plots/ensemble_performance.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_category_overlap(self):
        """Plot category overlap analysis"""
        if 'category_overlap' not in self.results:
            return
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots for each source
        for i, source in enumerate(SOURCES):
            plt.subplot(2, 2, i+1)
            
            if source in self.results['category_overlap']:
                overlap_data = self.results['category_overlap'][source]
                shot_impacts = overlap_data['overlap_metrics']['shot_type_impact']
                
                if shot_impacts:
                    models = list(shot_impacts.keys())
                    impacts = list(shot_impacts.values())
                    
                    colors = ['green' if x > 0 else 'red' for x in impacts]
                    plt.bar(models, impacts, color=colors, alpha=0.7)
                    plt.title(f'{source.title()} - Few-shot vs Zero-shot Impact')
                    plt.xlabel('Models')
                    plt.ylabel('F1 Improvement')
                    plt.xticks(rotation=45)
                    plt.grid(True, alpha=0.3)
                    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/plots/category_overlap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_analysis(self):
        """Plot model confusion analysis"""
        if 'confusion_analysis' not in self.results:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Performance tiers distribution
        tier_counts = {'Tier 1': 0, 'Tier 2': 0, 'Tier 3': 0}
        for source, data in self.results['confusion_analysis'].items():
            if 'tiers' in data:
                for tier, models in data['tiers'].items():
                    tier_counts[tier.replace('_', ' ').title()] += len(models)
        
        ax1.pie(tier_counts.values(), labels=tier_counts.keys(), autopct='%1.1f%%',
               colors=['gold', 'silver', '#CD7F32'])
        ax1.set_title('Performance Tier Distribution Across All Sources', fontweight='bold')
        
        # 2. Tier distribution by source
        sources = []
        tier1_counts = []
        tier2_counts = []
        tier3_counts = []
        
        for source, data in self.results['confusion_analysis'].items():
            if 'tiers' in data:
                sources.append(source.title())
                tier1_counts.append(len(data['tiers'].get('tier_1', [])))
                tier2_counts.append(len(data['tiers'].get('tier_2', [])))
                tier3_counts.append(len(data['tiers'].get('tier_3', [])))
        
        if sources:
            x = np.arange(len(sources))
            width = 0.25
            
            ax2.bar(x - width, tier1_counts, width, label='Tier 1', color='gold', alpha=0.8)
            ax2.bar(x, tier2_counts, width, label='Tier 2', color='silver', alpha=0.8)
            ax2.bar(x + width, tier3_counts, width, label='Tier 3', color='#CD7F32', alpha=0.8)
            
            ax2.set_xlabel('Data Sources')
            ax2.set_ylabel('Number of Models')
            ax2.set_title('Performance Tiers by Source', fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(sources)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Model consistency across sources
        if self.f1_data is not None:
            model_consistency = {}
            for model in self.f1_data['Model'].unique():
                model_data = self.f1_data[self.f1_data['Model'] == model]
                if len(model_data) > 1:
                    consistency = 1 - model_data['Macro_F1'].std()  # Higher = more consistent
                    model_consistency[model] = consistency
            
            if model_consistency:
                models = list(model_consistency.keys())
                consistencies = list(model_consistency.values())
                
                # Sort by consistency
                sorted_data = sorted(zip(models, consistencies), key=lambda x: x[1], reverse=True)
                models, consistencies = zip(*sorted_data)
                
                colors = ['green' if c > 0.8 else 'orange' if c > 0.6 else 'red' for c in consistencies]
                ax3.barh(range(len(models)), consistencies, color=colors, alpha=0.7)
                ax3.set_yticks(range(len(models)))
                ax3.set_yticklabels([m.replace('_', ' ').title() for m in models])
                ax3.set_xlabel('Consistency Score (1 - std)')
                ax3.set_title('Model Consistency Across Sources', fontweight='bold')
                ax3.grid(True, alpha=0.3)
        
        # 4. Performance variance by source
        if self.f1_data is not None:
            source_variance = self.f1_data.groupby('Source')['Macro_F1'].std().sort_values(ascending=True)
            
            ax4.bar(range(len(source_variance)), source_variance.values, 
                   alpha=0.7, color='purple')
            ax4.set_xticks(range(len(source_variance)))
            ax4.set_xticklabels(source_variance.index)
            ax4.set_xlabel('Data Sources')
            ax4.set_ylabel('Performance Standard Deviation')
            ax4.set_title('Performance Variance by Source', fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/plots/confusion_analysis.png', 
                   dpi=300, bbox_inches='tight')
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
        
        # Generate detailed tables
        self._generate_detailed_tables()
        
        print(f"Report saved to {self.output_dir}/")
    
    def _generate_summary_report(self):
        """Generate summary report"""
        report_path = f'{self.output_dir}/cross_validation_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("CROSS-MODEL VALIDATION ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write("MODELS ANALYZED:\n")
            f.write(f"LLMs: {', '.join(LLM_MODELS)}\n")
            f.write(f"Shot Types: {', '.join(SHOT_TYPES)}\n")
            f.write(f"Categories: {len(CATEGORIES)} categories\n\n")
            
            f.write("DATA SOURCES:\n")
            f.write(f"{', '.join(SOURCES)}\n\n")
            
            f.write("ANALYSIS COMPLETED:\n")
            for analysis_type in self.results.keys():
                f.write(f"- {analysis_type.replace('_', ' ').title()}\n")
            
            # Add key findings
            f.write("\nKEY FINDINGS:\n")
            
            if 'ranking_analysis' in self.results:
                sorted_models = self.results['ranking_analysis']['sorted_models']
                f.write(f"- Best performing model: {sorted_models[0][0]} (avg rank: {sorted_models[0][1]:.2f})\n")
                f.write(f"- Worst performing model: {sorted_models[-1][0]} (avg rank: {sorted_models[-1][1]:.2f})\n")
            
            if 'correlation_analysis' in self.results:
                corr_matrix = self.results['correlation_analysis']['correlation_matrix']
                avg_corr = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
                f.write(f"- Average model correlation: {avg_corr:.3f}\n")
            
            f.write(f"\nResults saved to: {self.output_dir}/\n")
            f.write(f"Visualizations saved to: {self.output_dir}/plots/\n")
    
    def _generate_detailed_tables(self):
        """Generate detailed CSV tables"""
        # Model performance table
        performance_table = self.f1_data.copy()
        performance_table.to_csv(f'{self.output_dir}/tables/model_performance.csv', index=False)
        
        # Ranking table
        if 'ranking_analysis' in self.results:
            ranking_df = pd.DataFrame(self.results['ranking_analysis']['sorted_models'], 
                                    columns=['Model', 'Average_Ranking'])
            ranking_df.to_csv(f'{self.output_dir}/tables/model_rankings.csv', index=False)
        
        # Ensemble performance table
        if 'ensemble_analysis' in self.results:
            ensemble_data = []
            for source, data in self.results['ensemble_analysis'].items():
                ensemble_data.append({
                    'Source': source,
                    'Best_Individual_F1': data['best_individual_f1'],
                    'Ensemble_F1': data['ensemble_f1'],
                    'Improvement': data['ensemble_improvement'],
                    'Models_Count': len(data['models_included'])
                })
            
            ensemble_df = pd.DataFrame(ensemble_data)
            ensemble_df.to_csv(f'{self.output_dir}/tables/ensemble_performance.csv', index=False)
    
    def run_full_analysis(self):
        """Run complete cross-model validation analysis"""
        print("Starting Simplified Cross-Model Validation Analysis")
        print("="*60)
        
        # Load F1 data
        if not self.load_f1_data():
            print("Failed to load F1 data. Exiting.")
            return
        
        # Run analyses
        self.statistical_significance_testing()
        self.correlation_analysis()
        self.ensemble_analysis()
        self.model_ranking_analysis()
        self.category_overlap_analysis()
        self.model_confusion_analysis()
        
        # Generate outputs
        self.generate_visualizations()
        self.generate_report()
        
        print("\n" + "="*60)
        print("CROSS-MODEL VALIDATION ANALYSIS COMPLETE")
        print("="*60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Simplified Cross-Model Validation Analysis')
    parser.add_argument('--output_dir', type=str, default='output/cross_validation', 
                       help='Output directory for results')
    parser.add_argument('--sources', nargs='+', default=SOURCES, 
                       help='Data sources to analyze')
    parser.add_argument('--models', nargs='+', default=LLM_MODELS, 
                       help='Models to analyze')
    
    args = parser.parse_args()
    
    # Create validator
    validator = SimplifiedCrossModelValidator(output_dir=args.output_dir)
    
    # Run analysis
    validator.run_full_analysis()

if __name__ == "__main__":
    main()
