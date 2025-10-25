#!/usr/bin/env python3
"""
Enhanced Cross-Model Overlap Analysis
====================================

This script adds sophisticated overlap and similarity metrics to the existing
cross-model validation analysis, providing deeper insights into model behavior.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
import os
from itertools import combinations

class EnhancedOverlapAnalyzer:
    """Enhanced overlap analysis with sophisticated metrics"""
    
    def __init__(self, results_dir='output/cross_validation'):
        self.results_dir = results_dir
        self.f1_data = None
        self.enhanced_results = {}
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load F1 data"""
        f1_path = 'output/f1/comprehensive_model_comparison.csv'
        if os.path.exists(f1_path):
            self.f1_data = pd.read_csv(f1_path)
            # Filter out BERT
            self.f1_data = self.f1_data[~self.f1_data['Model'].str.contains('bert', case=False)]
            print(f"Loaded F1 data: {len(self.f1_data)} model-source combinations")
    
    def run_enhanced_analysis(self):
        """Run enhanced overlap analysis"""
        print("Running Enhanced Overlap Analysis...")
        print("="*60)
        
        # 1. Model Similarity Analysis
        self.model_similarity_analysis()
        
        # 2. Performance Clustering Analysis
        self.performance_clustering_analysis()
        
        # 3. Category-Specific Overlap Analysis
        self.category_specific_overlap()
        
        # 4. Shot Type Impact Analysis
        self.shot_type_impact_analysis()
        
        # 5. Cross-Source Consistency Analysis
        self.cross_source_consistency()
        
        # 6. Model Agreement Analysis
        self.model_agreement_analysis()
        
        # 7. Performance Gap Analysis
        self.performance_gap_analysis()
        
        # Generate enhanced visualizations
        self.generate_enhanced_visualizations()
        
        # Save results
        self.save_enhanced_results()
        
        print("Enhanced overlap analysis complete!")
    
    def model_similarity_analysis(self):
        """Analyze model similarity using multiple metrics"""
        print("\n1. MODEL SIMILARITY ANALYSIS")
        print("-" * 40)
        
        similarity_results = {}
        
        for source in ['reddit', 'x', 'news', 'meeting_minutes']:
            source_data = self.f1_data[self.f1_data['Source'] == source]
            
            if len(source_data) == 0:
                continue
            
            # Create model performance matrix
            model_scores = {}
            for _, row in source_data.iterrows():
                model_scores[row['Model']] = row['Macro_F1']
            
            # Calculate similarity metrics using performance differences
            models = list(model_scores.keys())
            n_models = len(models)
            
            # Create similarity matrix based on performance differences
            similarity_matrix = np.eye(n_models)
            for i in range(n_models):
                for j in range(i+1, n_models):
                    # Calculate similarity based on performance difference
                    score1 = model_scores[models[i]]
                    score2 = model_scores[models[j]]
                    # Similarity = 1 - normalized difference
                    max_score = max(score1, score2)
                    similarity = 1 - (abs(score1 - score2) / (max_score + 1e-10))
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
            
            # Calculate similarity statistics
            upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            
            similarity_results[source] = {
                'similarity_matrix': similarity_matrix.tolist(),
                'model_names': models,
                'avg_similarity': np.mean(upper_triangle),
                'max_similarity': np.max(upper_triangle),
                'min_similarity': np.min(upper_triangle),
                'similarity_std': np.std(upper_triangle)
            }
            
            print(f"  {source.title()}:")
            print(f"    Average similarity: {np.mean(upper_triangle):.4f}")
            print(f"    Max similarity: {np.max(upper_triangle):.4f}")
            print(f"    Similarity std: {np.std(upper_triangle):.4f}")
        
        self.enhanced_results['model_similarity'] = similarity_results
    
    def performance_clustering_analysis(self):
        """Analyze performance clustering patterns"""
        print("\n2. PERFORMANCE CLUSTERING ANALYSIS")
        print("-" * 40)
        
        clustering_results = {}
        
        # Overall clustering across all sources
        all_scores = []
        model_names = []
        
        for _, row in self.f1_data.iterrows():
            all_scores.append(row['Macro_F1'])
            model_names.append(row['Model'])
        
        if len(all_scores) > 3:
            # Perform K-means clustering
            scores_array = np.array(all_scores).reshape(-1, 1)
            
            # Try different numbers of clusters
            best_silhouette = -1
            best_n_clusters = 2
            
            for n_clusters in range(2, min(6, len(set(model_names)))):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(scores_array)
                
                if len(set(cluster_labels)) > 1:
                    silhouette = silhouette_score(scores_array, cluster_labels)
                    if silhouette > best_silhouette:
                        best_silhouette = silhouette
                        best_n_clusters = n_clusters
            
            # Final clustering with best number of clusters
            kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(scores_array)
            
            # Analyze clusters
            cluster_analysis = {}
            for cluster_id in range(best_n_clusters):
                cluster_models = [model_names[i] for i in range(len(model_names)) if cluster_labels[i] == cluster_id]
                cluster_scores = [all_scores[i] for i in range(len(all_scores)) if cluster_labels[i] == cluster_id]
                
                cluster_analysis[f'cluster_{cluster_id}'] = {
                    'models': cluster_models,
                    'avg_score': np.mean(cluster_scores),
                    'score_std': np.std(cluster_scores),
                    'size': len(cluster_models)
                }
            
            clustering_results['overall'] = {
                'n_clusters': best_n_clusters,
                'silhouette_score': best_silhouette,
                'cluster_analysis': cluster_analysis
            }
            
            print(f"  Optimal clusters: {best_n_clusters}")
            print(f"  Silhouette score: {best_silhouette:.4f}")
            
            for cluster_id, analysis in cluster_analysis.items():
                print(f"    {cluster_id}: {analysis['size']} models, avg F1: {analysis['avg_score']:.4f}")
        
        self.enhanced_results['performance_clustering'] = clustering_results
    
    def category_specific_overlap(self):
        """Analyze overlap patterns specific to different category types"""
        print("\n3. CATEGORY-SPECIFIC OVERLAP ANALYSIS")
        print("-" * 40)
        
        # Define category groups based on the 16 categories
        category_groups = {
            'communication': ['ask a genuine question', 'ask a rhetorical question', 
                             'provide a fact or claim', 'provide an observation', 
                             'express their opinion', 'express others opinions'],
            'critique': ['money aid allocation', 'government critique', 'societal critique'],
            'response': ['solutions/interventions'],
            'perception': ['personal interaction', 'media portrayal', 'not in my backyard'],
            'harmful': ['harmful generalization', 'deserving/undeserving', 'racist']
        }
        
        category_overlap = {}
        
        for source in ['reddit', 'x', 'news', 'meeting_minutes']:
            source_data = self.f1_data[self.f1_data['Source'] == source]
            
            if len(source_data) == 0:
                continue
            
            # Analyze overlap within each category group
            group_overlap = {}
            
            for group_name, categories in category_groups.items():
                # For this analysis, we'll use the overall F1 scores as proxies
                # since we don't have category-specific F1 scores in the current data
                
                # Calculate model performance variance for this group
                model_performance = {}
                for _, row in source_data.iterrows():
                    model_performance[row['Model']] = row['Macro_F1']
                
                if len(model_performance) > 1:
                    scores = list(model_performance.values())
                    variance = np.var(scores)
                    cv = np.std(scores) / np.mean(scores)  # Coefficient of variation
                    
                    group_overlap[group_name] = {
                        'variance': variance,
                        'coefficient_of_variation': cv,
                        'performance_range': np.max(scores) - np.min(scores),
                        'models_analyzed': len(model_performance)
                    }
            
            category_overlap[source] = group_overlap
            
            print(f"  {source.title()}:")
            for group, metrics in group_overlap.items():
                print(f"    {group}: CV={metrics['coefficient_of_variation']:.4f}, "
                      f"Range={metrics['performance_range']:.4f}")
        
        self.enhanced_results['category_specific_overlap'] = category_overlap
    
    def shot_type_impact_analysis(self):
        """Analyze the impact of shot type on model overlap"""
        print("\n4. SHOT TYPE IMPACT ANALYSIS")
        print("-" * 40)
        
        shot_impact_results = {}
        
        for source in ['reddit', 'x', 'news', 'meeting_minutes']:
            source_data = self.f1_data[self.f1_data['Source'] == source]
            
            if len(source_data) == 0:
                continue
            
            # Separate zero-shot and few-shot models
            zero_shot_data = source_data[source_data['Model'].str.contains('zero_shot')]
            few_shot_data = source_data[source_data['Model'].str.contains('few_shot')]
            
            if len(zero_shot_data) > 0 and len(few_shot_data) > 0:
                # Calculate performance statistics for each shot type
                zero_shot_scores = zero_shot_data['Macro_F1'].values
                few_shot_scores = few_shot_data['Macro_F1'].values
                
                # Calculate overlap metrics
                zero_shot_variance = np.var(zero_shot_scores)
                few_shot_variance = np.var(few_shot_scores)
                
                # Calculate improvement consistency
                model_improvements = []
                for _, zero_row in zero_shot_data.iterrows():
                    model_base = zero_row['Model'].replace('_zero_shot', '')
                    few_row = few_shot_data[few_shot_data['Model'].str.contains(model_base)]
                    if len(few_row) > 0:
                        improvement = few_row.iloc[0]['Macro_F1'] - zero_row['Macro_F1']
                        model_improvements.append(improvement)
                
                shot_impact_results[source] = {
                    'zero_shot_variance': zero_shot_variance,
                    'few_shot_variance': few_shot_variance,
                    'variance_ratio': few_shot_variance / (zero_shot_variance + 1e-10),
                    'avg_improvement': np.mean(model_improvements) if model_improvements else 0,
                    'improvement_consistency': 1 - np.std(model_improvements) if model_improvements else 0,
                    'models_with_improvement': sum(1 for x in model_improvements if x > 0) if model_improvements else 0
                }
                
                print(f"  {source.title()}:")
                print(f"    Zero-shot variance: {zero_shot_variance:.6f}")
                print(f"    Few-shot variance: {few_shot_variance:.6f}")
                print(f"    Avg improvement: {np.mean(model_improvements):.4f}" if model_improvements else "    Avg improvement: N/A")
        
        self.enhanced_results['shot_type_impact'] = shot_impact_results
    
    def cross_source_consistency(self):
        """Analyze consistency of model performance across sources"""
        print("\n5. CROSS-SOURCE CONSISTENCY ANALYSIS")
        print("-" * 40)
        
        consistency_results = {}
        
        # Group by model
        model_groups = self.f1_data.groupby('Model')
        
        for model_name, model_data in model_groups:
            if len(model_data) >= 2:  # Need at least 2 sources
                scores = model_data['Macro_F1'].values
                sources = model_data['Source'].values
                
                # Calculate consistency metrics
                consistency_score = 1 - (np.std(scores) / (np.mean(scores) + 1e-10))
                rank_consistency = self._calculate_rank_consistency(model_data)
                
                consistency_results[model_name] = {
                    'performance_std': np.std(scores),
                    'performance_cv': np.std(scores) / (np.mean(scores) + 1e-10),
                    'consistency_score': max(0, consistency_score),
                    'rank_consistency': rank_consistency,
                    'sources_analyzed': len(sources),
                    'performance_range': np.max(scores) - np.min(scores)
                }
        
        # Sort by consistency
        sorted_consistency = sorted(consistency_results.items(), 
                                  key=lambda x: x[1]['consistency_score'], reverse=True)
        
        print("  Model Consistency Rankings:")
        for i, (model, metrics) in enumerate(sorted_consistency[:5]):
            print(f"    {i+1}. {model}: {metrics['consistency_score']:.4f}")
        
        self.enhanced_results['cross_source_consistency'] = consistency_results
    
    def _calculate_rank_consistency(self, model_data):
        """Calculate rank consistency across sources"""
        # Create ranking for each source
        source_rankings = {}
        for source in model_data['Source'].unique():
            source_data = self.f1_data[self.f1_data['Source'] == source]
            sorted_data = source_data.sort_values('Macro_F1', ascending=False)
            
            for rank, (_, row) in enumerate(sorted_data.iterrows()):
                if row['Model'] == model_data.iloc[0]['Model']:
                    source_rankings[source] = rank + 1
                    break
        
        if len(source_rankings) >= 2:
            ranks = list(source_rankings.values())
            return 1 - (np.std(ranks) / (np.mean(ranks) + 1e-10))
        return 0
    
    def model_agreement_analysis(self):
        """Analyze agreement patterns between models"""
        print("\n6. MODEL AGREEMENT ANALYSIS")
        print("-" * 40)
        
        agreement_results = {}
        
        for source in ['reddit', 'x', 'news', 'meeting_minutes']:
            source_data = self.f1_data[self.f1_data['Source'] == source]
            
            if len(source_data) < 2:
                continue
            
            # Calculate pairwise agreement
            models = source_data['Model'].values
            scores = source_data['Macro_F1'].values
            
            pairwise_agreements = []
            for i in range(len(models)):
                for j in range(i+1, len(models)):
                    # Agreement based on performance similarity
                    score_diff = abs(scores[i] - scores[j])
                    agreement = 1 - (score_diff / (max(scores[i], scores[j]) + 1e-10))
                    pairwise_agreements.append(agreement)
            
            if pairwise_agreements:
                agreement_results[source] = {
                    'avg_pairwise_agreement': np.mean(pairwise_agreements),
                    'agreement_std': np.std(pairwise_agreements),
                    'min_agreement': np.min(pairwise_agreements),
                    'max_agreement': np.max(pairwise_agreements),
                    'high_agreement_pairs': sum(1 for x in pairwise_agreements if x > 0.8)
                }
                
                print(f"  {source.title()}:")
                print(f"    Avg agreement: {np.mean(pairwise_agreements):.4f}")
                print(f"    High agreement pairs: {sum(1 for x in pairwise_agreements if x > 0.8)}")
        
        self.enhanced_results['model_agreement'] = agreement_results
    
    def performance_gap_analysis(self):
        """Analyze performance gaps between models"""
        print("\n7. PERFORMANCE GAP ANALYSIS")
        print("-" * 40)
        
        gap_results = {}
        
        for source in ['reddit', 'x', 'news', 'meeting_minutes']:
            source_data = self.f1_data[self.f1_data['Source'] == source]
            
            if len(source_data) < 2:
                continue
            
            # Sort by performance
            sorted_data = source_data.sort_values('Macro_F1', ascending=False)
            
            # Calculate gaps between consecutive models
            gaps = []
            for i in range(len(sorted_data) - 1):
                gap = sorted_data.iloc[i]['Macro_F1'] - sorted_data.iloc[i+1]['Macro_F1']
                gaps.append(gap)
            
            if gaps:
                gap_results[source] = {
                    'avg_gap': np.mean(gaps),
                    'max_gap': np.max(gaps),
                    'min_gap': np.min(gaps),
                    'gap_std': np.std(gaps),
                    'large_gaps': sum(1 for x in gaps if x > 0.05),
                    'total_gaps': len(gaps)
                }
                
                print(f"  {source.title()}:")
                print(f"    Avg gap: {np.mean(gaps):.4f}")
                print(f"    Max gap: {np.max(gaps):.4f}")
                print(f"    Large gaps (>0.05): {sum(1 for x in gaps if x > 0.05)}")
        
        self.enhanced_results['performance_gaps'] = gap_results
    
    def generate_enhanced_visualizations(self):
        """Generate enhanced visualizations"""
        print("\n8. GENERATING ENHANCED VISUALIZATIONS")
        print("-" * 40)
        
        # Create output directory
        os.makedirs(f'{self.results_dir}/enhanced_plots', exist_ok=True)
        
        # 1. Model Similarity Heatmap
        self._plot_model_similarity_heatmap()
        
        # 2. Performance Clustering Visualization
        self._plot_performance_clustering()
        
        # 3. Cross-Source Consistency Plot
        self._plot_cross_source_consistency()
        
        # 4. Shot Type Impact Analysis
        self._plot_shot_type_impact()
        
        print(f"Enhanced visualizations saved to {self.results_dir}/enhanced_plots/")
    
    def _plot_model_similarity_heatmap(self):
        """Plot model similarity heatmap"""
        if 'model_similarity' not in self.enhanced_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        sources = ['reddit', 'x', 'news', 'meeting_minutes']
        
        for i, source in enumerate(sources):
            if source in self.enhanced_results['model_similarity']:
                data = self.enhanced_results['model_similarity'][source]
                corr_matrix = np.array(data['similarity_matrix'])
                model_names = data['model_names']
                
                # Shorten model names for display
                short_names = [name.replace('_zero_shot', '_0').replace('_few_shot', '_F') 
                             for name in model_names]
                
                sns.heatmap(corr_matrix, xticklabels=short_names, yticklabels=short_names,
                           annot=True, cmap='coolwarm', center=0, ax=axes[i],
                           fmt='.2f', cbar_kws={'shrink': 0.8})
                axes[i].set_title(f'{source.title()} - Model Similarity')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/enhanced_plots/model_similarity_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_clustering(self):
        """Plot performance clustering visualization"""
        if 'performance_clustering' not in self.enhanced_results:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Overall clustering
        if 'overall' in self.enhanced_results['performance_clustering']:
            cluster_data = self.enhanced_results['performance_clustering']['overall']
            
            # Create scatter plot of performance by cluster
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            
            for cluster_id, analysis in cluster_data['cluster_analysis'].items():
                cluster_num = int(cluster_id.split('_')[1])
                models = analysis['models']
                avg_score = analysis['avg_score']
                
                # Plot cluster
                ax1.scatter([cluster_num] * len(models), [avg_score] * len(models),
                           c=colors[cluster_num % len(colors)], alpha=0.7, s=100,
                           label=f'Cluster {cluster_num}')
            
            ax1.set_xlabel('Cluster ID')
            ax1.set_ylabel('Average F1 Score')
            ax1.set_title('Performance Clustering Results')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Silhouette analysis
        silhouette_scores = []
        n_clusters_range = range(2, 6)
        
        for n_clusters in n_clusters_range:
            if len(self.f1_data) >= n_clusters:
                scores_array = np.array(self.f1_data['Macro_F1']).reshape(-1, 1)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(scores_array)
                
                if len(set(cluster_labels)) > 1:
                    silhouette = silhouette_score(scores_array, cluster_labels)
                    silhouette_scores.append(silhouette)
                else:
                    silhouette_scores.append(0)
        
        ax2.plot(n_clusters_range, silhouette_scores, 'bo-')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/enhanced_plots/performance_clustering.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cross_source_consistency(self):
        """Plot cross-source consistency analysis"""
        if 'cross_source_consistency' not in self.enhanced_results:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        consistency_data = self.enhanced_results['cross_source_consistency']
        
        # Extract data for plotting
        models = list(consistency_data.keys())
        consistency_scores = [consistency_data[model]['consistency_score'] for model in models]
        performance_stds = [consistency_data[model]['performance_std'] for model in models]
        
        # Sort by consistency score
        sorted_data = sorted(zip(models, consistency_scores, performance_stds), 
                           key=lambda x: x[1], reverse=True)
        
        sorted_models, sorted_scores, sorted_stds = zip(*sorted_data)
        
        # Plot consistency scores
        colors = ['green' if score > 0.8 else 'orange' if score > 0.6 else 'red' 
                 for score in sorted_scores]
        
        ax1.barh(range(len(sorted_models)), sorted_scores, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(sorted_models)))
        ax1.set_yticklabels([m.replace('_', ' ').title() for m in sorted_models])
        ax1.set_xlabel('Consistency Score')
        ax1.set_title('Model Consistency Across Sources')
        ax1.grid(True, alpha=0.3)
        
        # Plot performance standard deviation
        ax2.barh(range(len(sorted_models)), sorted_stds, alpha=0.7, color='purple')
        ax2.set_yticks(range(len(sorted_models)))
        ax2.set_yticklabels([m.replace('_', ' ').title() for m in sorted_models])
        ax2.set_xlabel('Performance Standard Deviation')
        ax2.set_title('Performance Variability Across Sources')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/enhanced_plots/cross_source_consistency.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_shot_type_impact(self):
        """Plot shot type impact analysis"""
        if 'shot_type_impact' not in self.enhanced_results:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        sources = list(self.enhanced_results['shot_type_impact'].keys())
        
        # Extract data
        zero_shot_vars = [self.enhanced_results['shot_type_impact'][s]['zero_shot_variance'] for s in sources]
        few_shot_vars = [self.enhanced_results['shot_type_impact'][s]['few_shot_variance'] for s in sources]
        avg_improvements = [self.enhanced_results['shot_type_impact'][s]['avg_improvement'] for s in sources]
        improvement_consistencies = [self.enhanced_results['shot_type_impact'][s]['improvement_consistency'] for s in sources]
        
        # Plot variance comparison
        x = np.arange(len(sources))
        width = 0.35
        
        ax1.bar(x - width/2, zero_shot_vars, width, label='Zero-shot', alpha=0.8, color='lightblue')
        ax1.bar(x + width/2, few_shot_vars, width, label='Few-shot', alpha=0.8, color='lightcoral')
        ax1.set_xlabel('Data Sources')
        ax1.set_ylabel('Performance Variance')
        ax1.set_title('Performance Variance by Shot Type')
        ax1.set_xticks(x)
        ax1.set_xticklabels([s.title() for s in sources])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot average improvements
        colors = ['green' if x > 0 else 'red' for x in avg_improvements]
        ax2.bar(sources, avg_improvements, color=colors, alpha=0.7)
        ax2.set_xlabel('Data Sources')
        ax2.set_ylabel('Average F1 Improvement')
        ax2.set_title('Few-shot vs Zero-shot Improvement')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Plot improvement consistency
        ax3.bar(sources, improvement_consistencies, alpha=0.7, color='orange')
        ax3.set_xlabel('Data Sources')
        ax3.set_ylabel('Improvement Consistency')
        ax3.set_title('Consistency of Few-shot Improvements')
        ax3.grid(True, alpha=0.3)
        
        # Plot variance ratio
        variance_ratios = [few_shot_vars[i] / (zero_shot_vars[i] + 1e-10) for i in range(len(sources))]
        ax4.bar(sources, variance_ratios, alpha=0.7, color='purple')
        ax4.set_xlabel('Data Sources')
        ax4.set_ylabel('Variance Ratio (Few-shot / Zero-shot)')
        ax4.set_title('Variance Ratio by Source')
        ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Equal Variance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/enhanced_plots/shot_type_impact.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_enhanced_results(self):
        """Save enhanced results to JSON"""
        output_path = f'{self.results_dir}/enhanced_overlap_results.json'
        with open(output_path, 'w') as f:
            json.dump(self.enhanced_results, f, indent=2, default=str)
        
        print(f"Enhanced results saved to {output_path}")

def main():
    """Main function"""
    analyzer = EnhancedOverlapAnalyzer()
    analyzer.run_enhanced_analysis()

if __name__ == "__main__":
    main()
