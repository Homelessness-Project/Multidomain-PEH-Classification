#!/usr/bin/env python3
"""
Category Pattern Analysis: Comprehensive research analysis of category patterns

This script performs multiple complementary analyses:
1. Category co-occurrence patterns (which categories appear together)
2. Category prevalence by score ranges (high vs low scoring comments)
3. Category distribution by city size
4. Category interaction effects
5. Category distribution by submission score vs comment score
6. Category prevalence summary statistics

All results are saved as CSV files for future reference.

Usage:
    python scripts/category_pattern_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
from itertools import combinations
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('ggplot')
sns.set_palette("husl")

# Robust summaries for heavy-tailed engagement (no normality assumption)
def median_iqr(values):
    """Return (median, q25, q75) for numeric array-like, ignoring NaNs."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return 0.0, 0.0, 0.0
    med = float(np.median(v))
    q25 = float(np.quantile(v, 0.25))
    q75 = float(np.quantile(v, 0.75))
    return med, q25, q75

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

# City size groupings
LARGE_CITIES = ['san francisco', 'portland', 'buffalo', 'baltimore', 'el paso']
SMALL_CITIES = ['kalamazoo', 'south bend', 'rockford', 'scranton', 'fayetteville']

def load_and_prepare_data():
    """Load Reddit data with scores and GPT4 category classifications"""
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load Reddit comments with scores
    reddit_file = 'complete_dataset/all_reddit_comments.csv'
    print(f"Loading {reddit_file}...")
    reddit_df = pd.read_csv(reddit_file, low_memory=False)
    print(f"  Loaded {len(reddit_df)} Reddit comments")
    
    # Load GPT4 classified comments
    gpt4_file = 'output/reddit/gpt4/classified_comments_reddit_all_gpt4_reddit_flags.csv'
    print(f"Loading {gpt4_file}...")
    gpt4_df = pd.read_csv(gpt4_file, low_memory=False)
    print(f"  Loaded {len(gpt4_df)} GPT4 classified comments")
    
    # Merge on Comment text
    print("Merging datasets...")
    reddit_df_merge = reddit_df.rename(columns={'Deidentified_Comment': 'Comment'})
    merged_df = reddit_df_merge.merge(
        gpt4_df[['Comment', 'City'] + ALL_CATEGORIES],
        on='Comment',
        how='inner',
        suffixes=('', '_gpt4')
    )
    print(f"  Merged dataset: {len(merged_df)} comments")
    
    # Clean and prepare data
    merged_df['Comment Score'] = pd.to_numeric(merged_df['Comment Score'], errors='coerce')
    merged_df['Submission Score'] = pd.to_numeric(merged_df['Submission Score'], errors='coerce')
    merged_df = merged_df.dropna(subset=['Comment Score'])
    
    # Convert category columns to binary (0/1)
    for cat in ALL_CATEGORIES:
        if cat in merged_df.columns:
            merged_df[cat] = pd.to_numeric(merged_df[cat], errors='coerce').fillna(0)
            merged_df[cat] = (merged_df[cat] > 0).astype(int)
        else:
            merged_df[cat] = 0
    
    # Add city size grouping
    merged_df['City_Lower'] = merged_df['City'].str.lower()
    merged_df['City_Size'] = merged_df['City_Lower'].apply(
        lambda x: 'Large' if x in LARGE_CITIES else ('Small' if x in SMALL_CITIES else 'Unknown')
    )
    
    # Add score quartiles
    merged_df['Comment_Score_Quartile'] = pd.qcut(
        merged_df['Comment Score'], 
        q=4, 
        labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)']
    )
    
    if 'Submission Score' in merged_df.columns:
        merged_df['Submission_Score_Quartile'] = pd.qcut(
            merged_df['Submission Score'], 
            q=4, 
            labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)']
        )
    
    print(f"  Final dataset: {len(merged_df)} comments with valid scores")
    print(f"  Score range: {merged_df['Comment Score'].min():.0f} to {merged_df['Comment Score'].max():.0f}")
    print(f"  Cities: {merged_df['City'].nunique()} unique cities")
    print(f"  Large cities: {len(merged_df[merged_df['City_Size'] == 'Large'])} comments")
    print(f"  Small cities: {len(merged_df[merged_df['City_Size'] == 'Small'])} comments")
    
    return merged_df

def analyze_category_prevalence(df):
    """Analyze overall category prevalence"""
    print("\n" + "="*80)
    print("ANALYSIS 1: CATEGORY PREVALENCE")
    print("="*80)
    
    results = []
    total_comments = len(df)
    
    for category in ALL_CATEGORIES:
        category_name = category.replace('Comment_', '').replace('Critique_', '').replace('Perception_', '').replace('Response_', '').replace('Racist_Flag', 'racist')
        count = df[category].sum()
        percentage = (count / total_comments) * 100
        
        # Calculate mean score for this category
        category_scores = df[df[category] == 1]['Comment Score'].values
        mean_score = np.mean(category_scores) if len(category_scores) > 0 else 0
        median_score = np.median(category_scores) if len(category_scores) > 0 else 0
        
        results.append({
            'Category': category_name,
            'Full_Category': category,
            'Count': count,
            'Percentage': percentage,
            'Mean_Comment_Score': mean_score,
            'Median_Comment_Score': median_score
        })
    
    results_df = pd.DataFrame(results).sort_values('Percentage', ascending=False)
    
    print("\nCategory Prevalence (sorted by frequency):")
    print("-" * 80)
    for _, row in results_df.iterrows():
        print(f"  {row['Category']:40s}: {row['Count']:6,} ({row['Percentage']:5.2f}%) | Mean Score: {row['Mean_Comment_Score']:6.2f}")
    
    return results_df

def analyze_category_importance(df):
    """Analyze category importance: Prevalence * Mean Score (what's common AND highly upvoted)"""
    print("\n" + "="*80)
    print("ANALYSIS 2: CATEGORY IMPORTANCE (Prevalence × Score)")
    print("="*80)
    
    total_comments = len(df)
    results = []
    
    for category in ALL_CATEGORIES:
        category_name = category.replace('Comment_', '').replace('Critique_', '').replace('Perception_', '').replace('Response_', '').replace('Racist_Flag', 'racist')
        
        # Get comments with this category
        category_mask = (df[category] == 1)
        category_count = category_mask.sum()
        category_prevalence = (category_count / total_comments) * 100
        
        # Calculate mean score for this category
        category_scores = df[category_mask]['Comment Score'].values
        mean_score = np.mean(category_scores) if len(category_scores) > 0 else 0
        median_score = np.median(category_scores) if len(category_scores) > 0 else 0
        
        # Importance metric: Prevalence × Mean Score
        # This highlights categories that are both common AND highly upvoted
        importance_score = category_prevalence * mean_score
        
        results.append({
            'Category': category_name,
            'Full_Category': category,
            'Count': category_count,
            'Prevalence_Percent': category_prevalence,
            'Mean_Score': mean_score,
            'Median_Score': median_score,
            'Importance_Score': importance_score  # Prevalence × Mean Score
        })
    
    importance_df = pd.DataFrame(results).sort_values('Importance_Score', ascending=False)
    
    print("\nCategory Importance (Prevalence × Mean Score) - What's Common AND Highly Upvoted:")
    print("-" * 80)
    for _, row in importance_df.iterrows():
        print(f"  {row['Category']:40s}: Importance={row['Importance_Score']:8.2f} (Prevalence={row['Prevalence_Percent']:5.2f}% × Mean Score={row['Mean_Score']:6.2f})")
    
    return importance_df

def analyze_category_cooccurrence(df):
    """Analyze which categories co-occur together with importance metric (Prevalence × Score)"""
    print("\n" + "="*80)
    print("ANALYSIS 3: CATEGORY CO-OCCURRENCE PATTERNS WITH IMPORTANCE")
    print("="*80)
    
    total_comments = len(df)
    cooccurrence_pairs = []
    
    for i, cat1 in enumerate(ALL_CATEGORIES):
        for j, cat2 in enumerate(ALL_CATEGORIES):
            if i != j:
                cat1_name = cat1.replace('Comment_', '').replace('Critique_', '').replace('Perception_', '').replace('Response_', '').replace('Racist_Flag', 'racist')
                cat2_name = cat2.replace('Comment_', '').replace('Critique_', '').replace('Perception_', '').replace('Response_', '').replace('Racist_Flag', 'racist')
                
                # Get comments with both categories
                both_mask = ((df[cat1] == 1) & (df[cat2] == 1))
                cat1_mask = (df[cat1] == 1)
                cat2_mask = (df[cat2] == 1)
                union_mask = (cat1_mask | cat2_mask)
                
                both_count = both_mask.sum()
                cat1_count = cat1_mask.sum()
                cat2_count = cat2_mask.sum()
                union_count = union_mask.sum()
                
                # Co-occurrence prevalence (percentage of all comments)
                cooccurrence_prevalence = (both_count / total_comments) * 100
                
                # Mean score when both categories appear
                both_scores = df[both_mask]['Comment Score'].values
                mean_score_both = np.mean(both_scores) if len(both_scores) > 0 else 0
                
                # IMPORTANCE METRIC: Co-occurrence Prevalence × Mean Score
                # This shows which pairs are both common together AND highly upvoted
                cooccurrence_importance = cooccurrence_prevalence * mean_score_both
                
                # Unweighted Jaccard similarity
                jaccard_unweighted = both_count / union_count if union_count > 0 else 0
                
                # Weighted Jaccard (using scores as weights)
                min_score = df['Comment Score'].min()
                score_offset = abs(min_score) + 1 if min_score < 0 else 0
                df['Score_Weight'] = df['Comment Score'] + score_offset
                
                both_weighted = df[both_mask]['Score_Weight'].sum()
                union_weighted = df[union_mask]['Score_Weight'].sum()
                jaccard_weighted = both_weighted / union_weighted if union_weighted > 0 else 0
                
                # Additional statistics
                only_cat1_scores = df[cat1_mask & ~cat2_mask]['Comment Score'].values
                only_cat2_scores = df[cat2_mask & ~cat1_mask]['Comment Score'].values
                
                cooccurrence_pairs.append({
                    'Category_1': cat1_name,
                    'Category_2': cat2_name,
                    'Full_Category_1': cat1,
                    'Full_Category_2': cat2,
                    'Both_Count': both_count,
                    'Cooccurrence_Prevalence_Percent': cooccurrence_prevalence,
                    'Mean_Score_When_Both': mean_score_both,
                    'Importance_Score': cooccurrence_importance,  # Prevalence × Mean Score
                    'Category_1_Count': cat1_count,
                    'Category_2_Count': cat2_count,
                    'Union_Count': union_count,
                    'Jaccard_Unweighted': jaccard_unweighted,
                    'Jaccard_Weighted': jaccard_weighted,
                    'Jaccard_Difference': jaccard_weighted - jaccard_unweighted,
                    'Mean_Score_Only_Cat1': np.mean(only_cat1_scores) if len(only_cat1_scores) > 0 else 0,
                    'Mean_Score_Only_Cat2': np.mean(only_cat2_scores) if len(only_cat2_scores) > 0 else 0,
                })
    
    cooccurrence_df = pd.DataFrame(cooccurrence_pairs)
    
    # Find top pairs by importance (prevalence × score)
    top_importance = cooccurrence_df.nlargest(20, 'Importance_Score')
    
    print("\nTop 20 Category Pairs by Importance (Co-occurrence Prevalence × Mean Score):")
    print("-" * 80)
    for _, row in top_importance.iterrows():
        print(f"  {row['Category_1']:35s} <-> {row['Category_2']:35s}: Importance={row['Importance_Score']:8.2f} (Prevalence={row['Cooccurrence_Prevalence_Percent']:5.2f}% × Score={row['Mean_Score_When_Both']:6.2f}, N={row['Both_Count']:6,})")
    
    # Also show top by Jaccard for comparison
    top_jaccard = cooccurrence_df.nlargest(10, 'Jaccard_Unweighted')
    
    print("\nTop 10 Category Pairs by Jaccard Similarity (for comparison):")
    print("-" * 80)
    for _, row in top_jaccard.iterrows():
        print(f"  {row['Category_1']:35s} <-> {row['Category_2']:35s}: Jaccard={row['Jaccard_Unweighted']:.4f}, Importance={row['Importance_Score']:8.2f}")
    
    return cooccurrence_df
    
    cooccurrence_pairs = []
    
    for i, cat1 in enumerate(ALL_CATEGORIES):
        for j, cat2 in enumerate(ALL_CATEGORIES):
            if i != j:
                cat1_name = cat1.replace('Comment_', '').replace('Critique_', '').replace('Perception_', '').replace('Response_', '').replace('Racist_Flag', 'racist')
                cat2_name = cat2.replace('Comment_', '').replace('Critique_', '').replace('Perception_', '').replace('Response_', '').replace('Racist_Flag', 'racist')
                
                # UNWEIGHTED (weight=1 for all comments)
                both_mask = ((df[cat1] == 1) & (df[cat2] == 1))
                cat1_mask = (df[cat1] == 1)
                cat2_mask = (df[cat2] == 1)
                union_mask = (cat1_mask | cat2_mask)
                
                both_count = both_mask.sum()
                cat1_count = cat1_mask.sum()
                cat2_count = cat2_mask.sum()
                union_count = union_mask.sum()
                
                # Unweighted Jaccard similarity
                jaccard_unweighted = both_count / union_count if union_count > 0 else 0
                
                # WEIGHTED (by comment score)
                both_weighted = df[both_mask]['Score_Weight'].sum()
                cat1_weighted = df[cat1_mask]['Score_Weight'].sum()
                cat2_weighted = df[cat2_mask]['Score_Weight'].sum()
                union_weighted = df[union_mask]['Score_Weight'].sum()
                
                # Weighted Jaccard similarity (weighted intersection / weighted union)
                jaccard_weighted = both_weighted / union_weighted if union_weighted > 0 else 0
                
                # Difference between weighted and unweighted
                jaccard_difference = jaccard_weighted - jaccard_unweighted
                
                # Statistical test: Mann-Whitney U test comparing scores of comments with both categories
                # vs comments with only one category
                both_scores = df[both_mask]['Comment Score'].values
                only_cat1_scores = df[cat1_mask & ~cat2_mask]['Comment Score'].values
                only_cat2_scores = df[cat2_mask & ~cat1_mask]['Comment Score'].values
                
                if len(both_scores) > 0 and len(only_cat1_scores) > 0:
                    stat1, p_value1 = stats.mannwhitneyu(both_scores, only_cat1_scores, alternative='two-sided')
                else:
                    p_value1 = 1.0
                
                if len(both_scores) > 0 and len(only_cat2_scores) > 0:
                    stat2, p_value2 = stats.mannwhitneyu(both_scores, only_cat2_scores, alternative='two-sided')
                else:
                    p_value2 = 1.0
                
                # Use minimum p-value (more conservative)
                p_value = min(p_value1, p_value2)
                
                cooccurrence_pairs.append({
                    'Category_1': cat1_name,
                    'Category_2': cat2_name,
                    'Full_Category_1': cat1,
                    'Full_Category_2': cat2,
                    'Both_Count': both_count,
                    'Category_1_Count': cat1_count,
                    'Category_2_Count': cat2_count,
                    'Union_Count': union_count,
                    'Jaccard_Unweighted': jaccard_unweighted,
                    'Jaccard_Weighted': jaccard_weighted,
                    'Jaccard_Difference': jaccard_difference,
                    'Both_Weighted_Sum': both_weighted,
                    'Category_1_Weighted_Sum': cat1_weighted,
                    'Category_2_Weighted_Sum': cat2_weighted,
                    'Union_Weighted_Sum': union_weighted,
                    'Mean_Score_Both': np.mean(both_scores) if len(both_scores) > 0 else 0,
                    'Mean_Score_Only_Cat1': np.mean(only_cat1_scores) if len(only_cat1_scores) > 0 else 0,
                    'Mean_Score_Only_Cat2': np.mean(only_cat2_scores) if len(only_cat2_scores) > 0 else 0,
                    'P_Value_Score_Difference': p_value,
                    'Significant_Score_Diff': p_value < 0.05
                })
    
    cooccurrence_df = pd.DataFrame(cooccurrence_pairs)
    
    # Find top co-occurring pairs (unweighted)
    top_pairs_unweighted = cooccurrence_df.nlargest(20, 'Jaccard_Unweighted')
    
    print("\nTop 20 Co-occurring Category Pairs (Unweighted, by Jaccard Similarity):")
    print("-" * 80)
    for _, row in top_pairs_unweighted.iterrows():
        print(f"  {row['Category_1']:35s} <-> {row['Category_2']:35s}: Jaccard={row['Jaccard_Unweighted']:.4f}, Both={row['Both_Count']:6,}")
    
    # Find top co-occurring pairs (weighted)
    top_pairs_weighted = cooccurrence_df.nlargest(20, 'Jaccard_Weighted')
    
    print("\nTop 20 Co-occurring Category Pairs (Score-Weighted, by Jaccard Similarity):")
    print("-" * 80)
    for _, row in top_pairs_weighted.iterrows():
        print(f"  {row['Category_1']:35s} <-> {row['Category_2']:35s}: Jaccard={row['Jaccard_Weighted']:.4f}, Both={row['Both_Count']:6,}, Mean Score={row['Mean_Score_Both']:.2f}")
    
    # Find pairs with largest difference (weighted > unweighted)
    top_differences = cooccurrence_df.nlargest(20, 'Jaccard_Difference')
    
    print("\nTop 20 Pairs with Largest Increase in Co-occurrence (Weighted vs Unweighted):")
    print("-" * 80)
    for _, row in top_differences.iterrows():
        sig_marker = "***" if row['Significant_Score_Diff'] else ""
        print(f"  {row['Category_1']:35s} <-> {row['Category_2']:35s}: Diff={row['Jaccard_Difference']:+.4f} {sig_marker} (Unw={row['Jaccard_Unweighted']:.4f}, Wgt={row['Jaccard_Weighted']:.4f}, p={row['P_Value_Score_Difference']:.4f})")
    
    return cooccurrence_df

def analyze_by_score_quartiles(df):
    """Analyze category prevalence by comment score quartiles"""
    print("\n" + "="*80)
    print("ANALYSIS 4: CATEGORY PREVALENCE BY SCORE QUARTILES")
    print("="*80)
    
    results = []
    
    for quartile in ['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)']:
        quartile_df = df[df['Comment_Score_Quartile'] == quartile]
        quartile_total = len(quartile_df)
        
        if quartile_total == 0:
            continue
        
        for category in ALL_CATEGORIES:
            category_name = category.replace('Comment_', '').replace('Critique_', '').replace('Perception_', '').replace('Response_', '').replace('Racist_Flag', 'racist')
            count = quartile_df[category].sum()
            percentage = (count / quartile_total) * 100
            
            results.append({
                'Quartile': quartile,
                'Category': category_name,
                'Full_Category': category,
                'Count': count,
                'Percentage': percentage,
                'Quartile_Total': quartile_total
            })
    
    results_df = pd.DataFrame(results)
    
    # Create pivot table for easier viewing
    pivot_df = results_df.pivot(index='Category', columns='Quartile', values='Percentage')
    pivot_df = pivot_df.sort_values('Q4 (Highest)', ascending=False)
    
    print("\nCategory Prevalence by Score Quartile (%):")
    print("-" * 80)
    print(pivot_df.to_string())
    
    return results_df, pivot_df

def analyze_by_city_size(df):
    """Analyze category prevalence by city size"""
    print("\n" + "="*80)
    print("ANALYSIS 5: CATEGORY PREVALENCE BY CITY SIZE")
    print("="*80)
    
    results = []
    
    for city_size in ['Large', 'Small']:
        size_df = df[df['City_Size'] == city_size]
        size_total = len(size_df)
        
        if size_total == 0:
            continue
        
        for category in ALL_CATEGORIES:
            category_name = category.replace('Comment_', '').replace('Critique_', '').replace('Perception_', '').replace('Response_', '').replace('Racist_Flag', 'racist')
            count = size_df[category].sum()
            percentage = (count / size_total) * 100
            
            # Statistical test: compare to overall prevalence
            overall_count = df[category].sum()
            overall_total = len(df)
            overall_pct = (overall_count / overall_total) * 100
            
            # Chi-square test
            contingency = [[count, size_total - count],
                          [overall_count - count, overall_total - overall_count - (size_total - count)]]
            try:
                chi2, p_value, dof, expected = chi2_contingency(contingency)
            except:
                p_value = 1.0
            
            results.append({
                'City_Size': city_size,
                'Category': category_name,
                'Full_Category': category,
                'Count': count,
                'Percentage': percentage,
                'Size_Total': size_total,
                'Overall_Percentage': overall_pct,
                'Difference': percentage - overall_pct,
                'P_Value': p_value,
                'Significant': p_value < 0.05
            })
    
    results_df = pd.DataFrame(results)
    
    # Create pivot table
    pivot_df = results_df.pivot(index='Category', columns='City_Size', values='Percentage')
    pivot_df['Difference'] = pivot_df['Large'] - pivot_df['Small']
    pivot_df = pivot_df.sort_values('Difference', key=abs, ascending=False)
    
    print("\nCategory Prevalence by City Size (%):")
    print("-" * 80)
    print(pivot_df[['Large', 'Small', 'Difference']].to_string())
    
    # Show significant differences
    sig_results = results_df[results_df['Significant'] == True]
    if len(sig_results) > 0:
        print("\nSignificant Differences (p < 0.05):")
        print("-" * 80)
        for _, row in sig_results.iterrows():
            print(f"  {row['Category']:40s} ({row['City_Size']:5s}): {row['Percentage']:5.2f}% vs {row['Overall_Percentage']:5.2f}% overall (diff={row['Difference']:+.2f}%, p={row['P_Value']:.4f})")
    
    return results_df, pivot_df

def analyze_submission_vs_comment_score(df):
    """Analyze category patterns by submission score vs comment score"""
    print("\n" + "="*80)
    print("ANALYSIS 6: CATEGORY PATTERNS BY SUBMISSION VS COMMENT SCORE")
    print("="*80)
    
    if 'Submission Score' not in df.columns or df['Submission Score'].isna().all():
        print("  Submission Score data not available, skipping this analysis")
        return None, None
    
    # Remove rows with missing submission scores
    df_clean = df.dropna(subset=['Submission Score'])
    
    # Calculate correlation between submission score and comment score for each category
    results = []
    
    for category in ALL_CATEGORIES:
        category_name = category.replace('Comment_', '').replace('Critique_', '').replace('Perception_', '').replace('Response_', '').replace('Racist_Flag', 'racist')
        
        category_df = df_clean[df_clean[category] == 1]
        other_df = df_clean[df_clean[category] == 0]
        
        if len(category_df) == 0:
            continue
        
        # Correlation between submission and comment scores
        cat_corr, cat_p = stats.pearsonr(
            category_df['Submission Score'], 
            category_df['Comment Score']
        ) if len(category_df) > 1 else (0, 1)
        
        other_corr, other_p = stats.pearsonr(
            other_df['Submission Score'], 
            other_df['Comment Score']
        ) if len(other_df) > 1 else (0, 1)
        
        # Mean scores
        cat_mean_sub = category_df['Submission Score'].mean()
        cat_mean_com = category_df['Comment Score'].mean()
        other_mean_sub = other_df['Submission Score'].mean()
        other_mean_com = other_df['Comment Score'].mean()
        
        results.append({
            'Category': category_name,
            'Full_Category': category,
            'N_Category': len(category_df),
            'N_Other': len(other_df),
            'Category_Sub_Mean': cat_mean_sub,
            'Category_Com_Mean': cat_mean_com,
            'Other_Sub_Mean': other_mean_sub,
            'Other_Com_Mean': other_mean_com,
            'Category_Correlation': cat_corr,
            'Category_Corr_P_Value': cat_p,
            'Other_Correlation': other_corr,
            'Other_Corr_P_Value': other_p
        })
    
    results_df = pd.DataFrame(results)
    
    print("\nSubmission vs Comment Score Correlations by Category:")
    print("-" * 80)
    for _, row in results_df.iterrows():
        print(f"  {row['Category']:40s}: Cat Corr={row['Category_Correlation']:+.3f}, Other Corr={row['Other_Correlation']:+.3f}")
    
    return results_df, None

def create_visualizations(df, prevalence_df, importance_df, cooccurrence_df, quartile_df, city_size_df):
    # This will be set in the function
    global importance_errors_df
    """Create visualizations for key findings"""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    output_dir = Path('output/charts')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 0. Prevalence and Mean Score Comparison (Before combined importance chart)
    # Robust overall summary for skewed vote distributions
    overall_median_score, _, _ = median_iqr(df['Comment Score'].values)
    overall_prevalence = 100.0  # All comments are in the dataset
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Sort by prevalence percentage (left chart) for consistency
    importance_sorted = importance_df.sort_values('Prevalence_Percent', ascending=False).copy()
    
    categories = importance_sorted['Category'].values
    prevalences = importance_sorted['Prevalence_Percent'].values
    median_scores = importance_sorted['Median_Score'].values
    
    # Calculate standard errors for prevalence and robust spread for scores (IQR)
    se_prevalences = []
    score_err_low = []
    score_err_high = []
    for idx, row in importance_sorted.iterrows():
        category = row['Full_Category']
        category_mask = (df[category] == 1)
        category_scores = df[category_mask]['Comment Score'].values
        total_comments = len(df)
        category_count = category_mask.sum()
        
        # SE for prevalence
        prevalence = (category_count / total_comments) * 100
        se_prevalence = np.sqrt((prevalence / 100) * (1 - prevalence / 100) / total_comments) * 100
        se_prevalences.append(se_prevalence)
        
        # Robust spread for score: IQR around the median (asymmetric error bar)
        med, q25, q75 = median_iqr(category_scores)
        score_err_low.append(max(0.0, med - q25))
        score_err_high.append(max(0.0, q75 - med))
    
    # Plot 1: Prevalence vs Overall Average (100%)
    colors_prevalence = ['#3498db' if p > overall_prevalence else '#e74c3c' for p in prevalences]
    bars1 = ax1.bar(range(len(categories)), prevalences, yerr=se_prevalences,
                    color=colors_prevalence, alpha=0.7, edgecolor='black', linewidth=0.5,
                    capsize=3, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
    ax1.axhline(y=overall_prevalence, color='black', linestyle='--', linewidth=2, label='Overall Average (100%)')
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Prevalence (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Category Prevalence on Reddit\n(All 16 Categories)', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_ylim([0, max(prevalences) * 1.15])
    
    # Plot 2: Median score with IQR error bars (robust to heavy tails)
    colors_score = ['#2ecc71' if s > overall_median_score else '#e74c3c' for s in median_scores]
    yerr = np.vstack([score_err_low, score_err_high])
    bars2 = ax2.bar(range(len(categories)), median_scores, yerr=yerr,
                    color=colors_score, alpha=0.7, edgecolor='black', linewidth=0.5,
                    capsize=3, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
    ax2.axhline(y=overall_median_score, color='black', linestyle='--', linewidth=2,
                label=f'Overall Median ({overall_median_score:.2f})')
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Median Comment Score', fontsize=12, fontweight='bold')
    ax2.set_title('Median Comment Score by Category\n(All 16 Categories, error bars = IQR)', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_ylim([0, max(median_scores) * 1.15])
    
    plt.suptitle('Category Analysis: Prevalence and Mean Score (Before Combined Importance)', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'category_prevalence_and_score_separate.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'category_prevalence_and_score_separate.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved separate prevalence and median score charts (IQR error bars)")
    
    # 1. Category Importance Bar Chart (All 16 categories with error bars and Bonferroni correction)
    # Calculate standard errors for importance scores
    importance_with_errors = []
    for _, row in importance_df.iterrows():
        category = row['Full_Category']
        category_mask = (df[category] == 1)
        category_scores = df[category_mask]['Comment Score'].values
        total_comments = len(df)
        category_count = category_mask.sum()
        
        # Robust spread for score around median (IQR)
        med, q25, q75 = median_iqr(category_scores)
        iqr_half = (q75 - q25) / 2.0 if len(category_scores) > 0 else 0.0
        
        # Calculate standard error for prevalence
        prevalence = (category_count / total_comments) * 100
        se_prevalence = np.sqrt((prevalence / 100) * (1 - prevalence / 100) / total_comments) * 100
        
        # Importance now based on prevalence × median score (robust)
        median_score = row['Median_Score']
        importance_score = prevalence * median_score
        # Approximate uncertainty for product using robust spread for score
        se_importance = abs(prevalence) * iqr_half + abs(median_score) * se_prevalence
        
        importance_with_errors.append({
            'Category': row['Category'],
            'Importance_Score': importance_score,
            'SE_Importance': se_importance,
            'Median_Score': median_score,
            'IQR_HalfWidth_Score': iqr_half,
            'Prevalence': prevalence,
            'SE_Prevalence': se_prevalence
        })
    
    importance_errors_df = pd.DataFrame(importance_with_errors).sort_values('Importance_Score', ascending=False)
    
    # Statistical tests: Compare each category's importance to all others
    # Using Bonferroni correction for 16 categories
    bonferroni_alpha = 0.05 / 16  # 0.003125
    
    significance_results = []
    all_importance_scores = importance_errors_df['Importance_Score'].values
    
    for idx, row in importance_errors_df.iterrows():
        category_importance = row['Importance_Score']
        category_name = row['Category']
        
        # Get scores for this category vs all others
        full_category_name = importance_df[importance_df['Category'] == category_name]['Full_Category'].iloc[0]
        category_mask = (df[full_category_name] == 1)
        category_scores = df[category_mask]['Comment Score'].values
        other_scores = df[~category_mask]['Comment Score'].values
        
        # Mann-Whitney U test to see if this category's scores are significantly different
        if len(category_scores) > 0 and len(other_scores) > 0:
            statistic, p_value = stats.mannwhitneyu(category_scores, other_scores, alternative='two-sided')
            p_value_corrected = min(p_value * 16, 1.0)  # Bonferroni correction
            is_significant = p_value_corrected < 0.05
        else:
            p_value_corrected = 1.0
            is_significant = False
        
        significance_results.append({
            'Category': category_name,
            'P_Value': p_value if len(category_scores) > 0 and len(other_scores) > 0 else 1.0,
            'P_Value_Corrected': p_value_corrected,
            'Significant': is_significant
        })
    
    significance_df = pd.DataFrame(significance_results)
    importance_errors_df = importance_errors_df.merge(significance_df, on='Category')
    
    # Create bar chart with error bars
    fig, ax = plt.subplots(figsize=(14, 8))
    
    categories = importance_errors_df['Category'].values
    importance_scores = importance_errors_df['Importance_Score'].values
    errors = importance_errors_df['SE_Importance'].values
    is_significant = importance_errors_df['Significant'].values
    
    # Color bars: green for significant, gray for not significant
    colors = ['#2ecc71' if sig else '#95a5a6' for sig in is_significant]
    
    bars = ax.bar(range(len(categories)), importance_scores, yerr=errors, 
                  color=colors, alpha=0.7, edgecolor='black', linewidth=0.5,
                  capsize=5, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
    
    # Add significance markers
    for i, (sig, score, error) in enumerate(zip(is_significant, importance_scores, errors)):
        if sig:
            # Add asterisk for significance
            p_val = importance_errors_df.iloc[i]['P_Value_Corrected']
            if p_val < 0.001:
                marker = '***'
            elif p_val < 0.01:
                marker = '**'
            else:
                marker = '*'
            ax.text(i, score + error + max(importance_scores) * 0.02, marker, 
                   ha='center', va='bottom', fontsize=12, fontweight='bold', color='red')
    
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Importance Score\n(Prevalence × Median Score)', fontsize=12, fontweight='bold')
    ax.set_title('Category Importance on Reddit (All 16 Categories)\n' + 
                'Error bars ≈ prevalence×(IQR/2) + median×SE(prevalence); * p<0.05, ** p<0.01, *** p<0.001 (Bonferroni corrected)',
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.7, label='Significantly Different (Bonferroni corrected)'),
        Patch(facecolor='#95a5a6', alpha=0.7, label='Not Significant')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'category_importance_all16_with_errors.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'category_importance_all16_with_errors.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved category importance chart (all 16 with error bars and Bonferroni correction)")
    
    # 2. Category Prevalence Bar Chart (for comparison)
    fig, ax = plt.subplots(figsize=(12, 8))
    top_categories = prevalence_df.nlargest(12, 'Percentage')
    ax.barh(range(len(top_categories)), top_categories['Percentage'], color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(top_categories)))
    ax.set_yticklabels(top_categories['Category'], fontsize=10)
    ax.set_xlabel('Prevalence (%)', fontsize=12, fontweight='bold')
    ax.set_title('Top 12 Most Common Categories on Reddit', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'category_prevalence_top12.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'category_prevalence_top12.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved category prevalence chart")
    
    # 3. Co-occurrence Importance Heatmap (Prevalence × Score)
    if len(cooccurrence_df) > 0:
        top_cats = importance_df.nlargest(10, 'Importance_Score')['Full_Category'].tolist()
        cooccur_subset = cooccurrence_df[
            cooccurrence_df['Full_Category_1'].isin(top_cats) & 
            cooccurrence_df['Full_Category_2'].isin(top_cats)
        ]
        
        if len(cooccur_subset) > 0:
            # Importance heatmap (Prevalence × Score)
            pivot_importance = cooccur_subset.pivot(
                index='Category_1', 
                columns='Category_2', 
                values='Importance_Score'
            )
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(pivot_importance, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, 
                       cbar_kws={'label': 'Importance (Prevalence × Mean Score)'})
            ax.set_title('Category Co-occurrence Importance\n(Prevalence × Mean Score when co-occurring)\nTop 10 Most Important Categories', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / 'category_cooccurrence_importance_heatmap.pdf', dpi=300, bbox_inches='tight')
            plt.savefig(output_dir / 'category_cooccurrence_importance_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Saved co-occurrence importance heatmap (Prevalence × Score)")
            
            # Also create comparison: Importance vs Jaccard
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Importance
            sns.heatmap(pivot_importance, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax1, 
                       cbar_kws={'label': 'Importance Score'})
            ax1.set_title('Importance (Prevalence × Score)', fontsize=12, fontweight='bold')
            
            # Jaccard for comparison
            pivot_jaccard = cooccur_subset.pivot(
                index='Category_1', 
                columns='Category_2', 
                values='Jaccard_Unweighted'
            )
            sns.heatmap(pivot_jaccard, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax2, 
                       cbar_kws={'label': 'Jaccard Similarity'}, vmin=0, vmax=1)
            ax2.set_title('Jaccard Similarity (for comparison)', fontsize=12, fontweight='bold')
            
            plt.suptitle('Co-occurrence: Importance vs Jaccard (Top 10 Categories)', 
                        fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(output_dir / 'category_cooccurrence_importance_vs_jaccard.pdf', dpi=300, bbox_inches='tight')
            plt.savefig(output_dir / 'category_cooccurrence_importance_vs_jaccard.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Saved importance vs Jaccard comparison")
    
    print(f"\nAll visualizations saved to: {output_dir}/")
    
    return importance_errors_df

def save_all_results(prevalence_df, importance_df, cooccurrence_df, quartile_df, quartile_pivot, 
                    city_size_df, city_size_pivot, submission_df, importance_errors_df=None):
    """Save all results to CSV files"""
    print("\n" + "="*80)
    print("SAVING RESULTS TO CSV")
    print("="*80)
    
    output_dir = Path('output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each analysis
    files_saved = []
    
    prevalence_file = output_dir / 'category_pattern_analysis_prevalence.csv'
    prevalence_df.to_csv(prevalence_file, index=False)
    files_saved.append(prevalence_file)
    print(f"  ✓ Saved category prevalence: {prevalence_file}")
    
    importance_file = output_dir / 'category_pattern_analysis_importance.csv'
    importance_df.to_csv(importance_file, index=False)
    files_saved.append(importance_file)
    print(f"  ✓ Saved category importance (Prevalence × Score): {importance_file}")
    
    # Also save importance with errors and significance
    if importance_errors_df is not None:
        importance_errors_file = output_dir / 'category_pattern_analysis_importance_with_errors.csv'
        importance_errors_df.to_csv(importance_errors_file, index=False)
        files_saved.append(importance_errors_file)
        print(f"  ✓ Saved category importance with errors and significance: {importance_errors_file}")
    
    cooccurrence_file = output_dir / 'category_pattern_analysis_cooccurrence.csv'
    cooccurrence_df.to_csv(cooccurrence_file, index=False)
    files_saved.append(cooccurrence_file)
    print(f"  ✓ Saved co-occurrence patterns with importance: {cooccurrence_file}")
    
    quartile_file = output_dir / 'category_pattern_analysis_by_quartile.csv'
    quartile_df.to_csv(quartile_file, index=False)
    files_saved.append(quartile_file)
    print(f"  ✓ Saved quartile analysis: {quartile_file}")
    
    quartile_pivot_file = output_dir / 'category_pattern_analysis_by_quartile_pivot.csv'
    quartile_pivot.to_csv(quartile_pivot_file)
    files_saved.append(quartile_pivot_file)
    print(f"  ✓ Saved quartile pivot table: {quartile_pivot_file}")
    
    city_size_file = output_dir / 'category_pattern_analysis_by_city_size.csv'
    city_size_df.to_csv(city_size_file, index=False)
    files_saved.append(city_size_file)
    print(f"  ✓ Saved city size analysis: {city_size_file}")
    
    city_size_pivot_file = output_dir / 'category_pattern_analysis_by_city_size_pivot.csv'
    city_size_pivot.to_csv(city_size_pivot_file)
    files_saved.append(city_size_pivot_file)
    print(f"  ✓ Saved city size pivot table: {city_size_pivot_file}")
    
    if submission_df is not None:
        submission_file = output_dir / 'category_pattern_analysis_submission_scores.csv'
        submission_df.to_csv(submission_file, index=False)
        files_saved.append(submission_file)
        print(f"  ✓ Saved submission score analysis: {submission_file}")
    
    print(f"\n✓ All results saved! Total files: {len(files_saved)}")
    return files_saved

def main():
    print("="*80)
    print("CATEGORY PATTERN ANALYSIS")
    print("Comprehensive research analysis of category patterns on Reddit")
    print("="*80)
    
    # Load data
    df = load_and_prepare_data()
    
    # Perform analyses
    prevalence_df = analyze_category_prevalence(df)
    importance_df = analyze_category_importance(df)
    cooccurrence_df = analyze_category_cooccurrence(df)
    quartile_df, quartile_pivot = analyze_by_score_quartiles(df)
    city_size_df, city_size_pivot = analyze_by_city_size(df)
    submission_df, _ = analyze_submission_vs_comment_score(df)
    
    # Create visualizations (returns importance_errors_df)
    importance_errors_df = create_visualizations(df, prevalence_df, importance_df, cooccurrence_df, quartile_df, city_size_df)
    
    # Save all results
    save_all_results(prevalence_df, importance_df, cooccurrence_df, quartile_df, quartile_pivot,
                    city_size_df, city_size_pivot, submission_df, importance_errors_df)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total comments analyzed: {len(df):,}")
    print(f"Total categories: {len(ALL_CATEGORIES)}")
    print(f"Most common category: {prevalence_df.iloc[0]['Category']} ({prevalence_df.iloc[0]['Percentage']:.2f}%)")
    print(f"Least common category: {prevalence_df.iloc[-1]['Category']} ({prevalence_df.iloc[-1]['Percentage']:.2f}%)")
    print(f"Most important category (Prevalence × Score): {importance_df.iloc[0]['Category']} (Importance={importance_df.iloc[0]['Importance_Score']:.2f})")
    
    if len(cooccurrence_df) > 0:
        top_importance_pair = cooccurrence_df.nlargest(1, 'Importance_Score').iloc[0]
        print(f"Most important co-occurrence (Prevalence × Score): {top_importance_pair['Category_1']} <-> {top_importance_pair['Category_2']} (Importance={top_importance_pair['Importance_Score']:.2f})")
        top_pair = cooccurrence_df.nlargest(1, 'Jaccard_Unweighted').iloc[0]
        print(f"Strongest co-occurrence (Jaccard): {top_pair['Category_1']} <-> {top_pair['Category_2']} (Jaccard={top_pair['Jaccard_Unweighted']:.4f})")
    
    print("\n" + "="*80)
    print("Analysis complete! All results saved to CSV files for future reference.")
    print("="*80)

if __name__ == '__main__':
    main()
