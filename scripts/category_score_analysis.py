#!/usr/bin/env python3
"""
Category Score Analysis: Statistical comparison of comment scores by category

This script analyzes whether certain categories have statistically higher or lower
comment scores than others, using Bonferroni correction for multiple comparisons.

Usage:
    python scripts/category_score_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from pathlib import Path

# Set style
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('ggplot')
sns.set_palette("husl")

# Define all 16 categories (matching GPT4 output format)
ALL_CATEGORIES = [
    'Comment_ask a genuine question',
    'Comment_ask a rhetorical question',
    'Comment_provide a fact or claim',
    'Comment_provide an observation',
    'Comment_express their opinion',
    'Comment_express others opinions',
    'Critique_money aid allocation',
    'Critique_government critique',
    'Critique_societal critique',
    'Response_solutions/interventions',
    'Perception_personal interaction',
    'Perception_media portrayal',
    'Perception_not in my backyard',
    'Perception_harmful generalization',
    'Perception_deserving/undeserving',
    'Racist_Flag'
]

# Bonferroni correction for 16 categories
BONFERRONI_ALPHA = 0.05 / 16  # 0.003125

def load_and_merge_data():
    """Load Reddit data with scores and GPT4 category classifications"""
    print("Loading Reddit data...")
    
    # Load Reddit comments with scores
    reddit_file = 'complete_dataset/all_reddit_comments.csv'
    print(f"  Loading {reddit_file}...")
    reddit_df = pd.read_csv(reddit_file, low_memory=False)
    print(f"  Loaded {len(reddit_df)} Reddit comments")
    
    # Load GPT4 classified comments
    gpt4_file = 'output/reddit/gpt4/classified_comments_reddit_all_gpt4_reddit_flags.csv'
    print(f"  Loading {gpt4_file}...")
    gpt4_df = pd.read_csv(gpt4_file, low_memory=False)
    print(f"  Loaded {len(gpt4_df)} GPT4 classified comments")
    
    # Merge on Comment text
    # Reddit file uses 'Deidentified_Comment', GPT4 file uses 'Comment'
    print("  Merging datasets...")
    # Rename for merge
    reddit_df_merge = reddit_df.rename(columns={'Deidentified_Comment': 'Comment'})
    merged_df = reddit_df_merge.merge(
        gpt4_df[['Comment'] + ALL_CATEGORIES],
        on='Comment',
        how='inner',
        suffixes=('', '_gpt4')
    )
    print(f"  Merged dataset: {len(merged_df)} comments")
    
    # Check for Comment Score column
    if 'Comment Score' not in merged_df.columns:
        print("  ERROR: 'Comment Score' column not found!")
        print(f"  Available columns: {list(merged_df.columns)[:20]}")
        return None
    
    # Clean comment scores: remove NaN, convert to numeric
    merged_df['Comment Score'] = pd.to_numeric(merged_df['Comment Score'], errors='coerce')
    merged_df = merged_df.dropna(subset=['Comment Score'])
    
    # Convert category columns to binary (0/1)
    for cat in ALL_CATEGORIES:
        if cat in merged_df.columns:
            # Convert to binary: any non-zero/non-empty value = 1
            merged_df[cat] = pd.to_numeric(merged_df[cat], errors='coerce').fillna(0)
            merged_df[cat] = (merged_df[cat] > 0).astype(int)
        else:
            print(f"  Warning: Category {cat} not found, setting to 0")
            merged_df[cat] = 0
    
    print(f"  Final dataset: {len(merged_df)} comments with valid scores")
    print(f"  Score range: {merged_df['Comment Score'].min():.0f} to {merged_df['Comment Score'].max():.0f}")
    print(f"  Score mean: {merged_df['Comment Score'].mean():.2f}, median: {merged_df['Comment Score'].median():.2f}")
    
    return merged_df

def analyze_category_scores(df):
    """Perform statistical analysis comparing comment scores by category"""
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS: Comment Scores by Category")
    print("="*80)
    
    results = []
    all_scores = df['Comment Score'].values
    
    for category in ALL_CATEGORIES:
        # Get scores for comments with this category
        category_scores = df[df[category] == 1]['Comment Score'].values
        other_scores = df[df[category] == 0]['Comment Score'].values
        
        n_category = len(category_scores)
        n_other = len(other_scores)
        
        if n_category == 0:
            print(f"\n{category}:")
            print(f"  No comments with this category, skipping")
            continue
        
        # Calculate statistics
        mean_category = np.mean(category_scores)
        mean_other = np.mean(other_scores)
        median_category = np.median(category_scores)
        median_other = np.median(other_scores)
        
        # Perform statistical test
        # Use Mann-Whitney U test (non-parametric) since scores may not be normally distributed
        statistic, p_value = stats.mannwhitneyu(
            category_scores, 
            other_scores, 
            alternative='two-sided'
        )
        
        # Apply Bonferroni correction
        p_value_corrected = min(p_value * 16, 1.0)  # Multiply by number of tests
        is_significant = p_value_corrected < 0.05
        
        # Determine direction
        if is_significant:
            direction = "HIGHER" if mean_category > mean_other else "LOWER"
        else:
            direction = "NS"
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.var(category_scores, ddof=1) + np.var(other_scores, ddof=1)) / 2
        )
        if pooled_std > 0:
            cohens_d = (mean_category - mean_other) / pooled_std
        else:
            cohens_d = 0.0
        
        results.append({
            'Category': category.replace('Comment_', '').replace('Critique_', '').replace('Perception_', '').replace('Response_', ''),
            'Full_Category': category,
            'N_Category': n_category,
            'N_Other': n_other,
            'Mean_Category': mean_category,
            'Mean_Other': mean_other,
            'Median_Category': median_category,
            'Median_Other': median_other,
            'Difference': mean_category - mean_other,
            'P_Value': p_value,
            'P_Value_Corrected': p_value_corrected,
            'Significant': is_significant,
            'Direction': direction,
            'Cohens_D': cohens_d
        })
        
        # Print results
        print(f"\n{category}:")
        print(f"  N (category): {n_category:,}, N (other): {n_other:,}")
        print(f"  Mean (category): {mean_category:.2f}, Mean (other): {mean_other:.2f}")
        print(f"  Difference: {mean_category - mean_other:+.2f}")
        print(f"  P-value: {p_value:.6f}, P-value (Bonferroni corrected): {p_value_corrected:.6f}")
        print(f"  Significant: {'YES' if is_significant else 'NO'} ({direction})")
        print(f"  Effect size (Cohen's d): {cohens_d:+.3f}")
    
    results_df = pd.DataFrame(results)
    return results_df

def create_visualization(df, results_df):
    """Create visualization showing category score differences"""
    print("\n" + "="*80)
    print("CREATING VISUALIZATION")
    print("="*80)
    
    # Sort by difference (category mean - other mean)
    results_df = results_df.sort_values('Difference', ascending=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot 1: Bar plot of mean differences with significance indicators
    colors = ['red' if not sig else ('green' if diff > 0 else 'blue') 
              for sig, diff in zip(results_df['Significant'], results_df['Difference'])]
    
    bars = ax1.barh(range(len(results_df)), results_df['Difference'], color=colors, alpha=0.7)
    
    # Add significance markers
    for i, (sig, diff) in enumerate(zip(results_df['Significant'], results_df['Difference'])):
        if sig:
            marker = '***' if results_df.iloc[i]['P_Value_Corrected'] < 0.001 else '**' if results_df.iloc[i]['P_Value_Corrected'] < 0.01 else '*'
            ax1.text(diff + (0.1 if diff > 0 else -0.1), i, marker, 
                   ha='left' if diff > 0 else 'right', va='center', fontsize=12, fontweight='bold')
    
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax1.set_yticks(range(len(results_df)))
    ax1.set_yticklabels(results_df['Category'], fontsize=10)
    ax1.set_xlabel('Mean Score Difference (Category - Other)', fontsize=12, fontweight='bold')
    ax1.set_title('Comment Score Differences by Category\n(Bonferroni Corrected, * p<0.05, ** p<0.01, *** p<0.001)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Significantly Higher'),
        Patch(facecolor='blue', alpha=0.7, label='Significantly Lower'),
        Patch(facecolor='red', alpha=0.7, label='Not Significant')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    # Plot 2: Box plot comparison for top/bottom categories
    # Select top 4 and bottom 4 categories by difference
    top_categories = results_df.nlargest(4, 'Difference')
    bottom_categories = results_df.nsmallest(4, 'Difference')
    selected_categories = pd.concat([top_categories, bottom_categories])
    
    # Prepare data for box plot
    box_data = []
    box_labels = []
    for _, row in selected_categories.iterrows():
        category = row['Full_Category']
        category_scores = df[df[category] == 1]['Comment Score'].values
        other_scores = df[df[category] == 0]['Comment Score'].values
        
        box_data.append(category_scores)
        box_labels.append(f"{row['Category']}\n(category, n={len(category_scores)})")
        box_data.append(other_scores)
        box_labels.append(f"{row['Category']}\n(other, n={len(other_scores)})")
    
    bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True, 
                     showmeans=True, meanline=True)
    
    # Color boxes: green for category, gray for other
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor('lightgreen' if i % 2 == 0 else 'lightgray')
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Comment Score', fontsize=12, fontweight='bold')
    ax2.set_title('Score Distribution: Top 4 and Bottom 4 Categories\n(Green = Category, Gray = Other)', 
                  fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45, labelsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('output/charts')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'category_score_analysis.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved figure to: {output_file}")
    
    # Also save as PNG
    output_file_png = output_dir / 'category_score_analysis.png'
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure to: {output_file_png}")
    
    plt.close()

def save_results_table(results_df):
    """Save detailed results table"""
    output_dir = Path('output')
    output_file = output_dir / 'category_score_analysis_results.csv'
    
    # Format for readability
    results_df['P_Value'] = results_df['P_Value'].apply(lambda x: f"{x:.6f}")
    results_df['P_Value_Corrected'] = results_df['P_Value_Corrected'].apply(lambda x: f"{x:.6f}")
    results_df['Mean_Category'] = results_df['Mean_Category'].apply(lambda x: f"{x:.2f}")
    results_df['Mean_Other'] = results_df['Mean_Other'].apply(lambda x: f"{x:.2f}")
    results_df['Difference'] = results_df['Difference'].apply(lambda x: f"{x:+.2f}")
    results_df['Cohens_D'] = results_df['Cohens_D'].apply(lambda x: f"{x:+.3f}")
    
    results_df.to_csv(output_file, index=False)
    print(f"✓ Saved results table to: {output_file}")

def main():
    print("="*80)
    print("CATEGORY SCORE ANALYSIS")
    print("Statistical comparison of comment scores by category")
    print("Using Bonferroni correction (α = 0.05/16 = 0.003125)")
    print("="*80)
    
    # Load and merge data
    df = load_and_merge_data()
    if df is None:
        print("ERROR: Failed to load data")
        return
    
    # Perform statistical analysis
    results_df = analyze_category_scores(df)
    
    # Create visualization (before formatting)
    create_visualization(df, results_df)
    
    # Summary (before formatting)
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    n_significant = results_df['Significant'].sum()
    n_higher = ((results_df['Significant']) & (results_df['Direction'] == 'HIGHER')).sum()
    n_lower = ((results_df['Significant']) & (results_df['Direction'] == 'LOWER')).sum()
    
    print(f"Total categories analyzed: {len(results_df)}")
    print(f"Significantly different (Bonferroni corrected): {n_significant}")
    print(f"  - Significantly HIGHER scores: {n_higher}")
    print(f"  - Significantly LOWER scores: {n_lower}")
    print(f"  - Not significant: {len(results_df) - n_significant}")
    
    if n_significant > 0:
        print("\nCategories with significantly different scores:")
        sig_results = results_df[results_df['Significant']].sort_values('Difference', ascending=False)
        for _, row in sig_results.iterrows():
            print(f"  {row['Category']}: {row['Direction']} (diff={row['Difference']:+.2f}, p={row['P_Value_Corrected']:.6f})")
    
    # Save results (formats the dataframe)
    save_results_table(results_df)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == '__main__':
    main()
