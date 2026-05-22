#!/usr/bin/env python3
"""
Unified Category Pattern Analysis: Reddit and X
Comprehensive research analysis with bar charts and error bars

This script performs the same analyses for both Reddit (using comment scores) 
and X (formerly Twitter, using like rates), creating publication-ready bar charts with error bars.

Statistical Approach:
- Each category is compared to the overall average using one-sample tests
- Bonferroni correction (α = 0.05/16 = 0.003125) for multiple comparisons
- Non-parametric tests (Wilcoxon signed-rank) with parametric fallback (t-test)

Usage:
    python scripts/category_pattern_analysis_unified.py --source reddit
    python scripts/category_pattern_analysis_unified.py --source twitter
    python scripts/category_pattern_analysis_unified.py --source both
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS
# ============================================================================

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

# Negative Bias Frame (5 labels; subset of ALL_CATEGORIES)
NEGATIVE_BIAS_FRAME_CATEGORIES = [
    'Comment_ask a rhetorical question',
    'Perception_not in my backyard',
    'Perception_harmful generalization',
    'Perception_deserving/undeserving',
    'Racist_Flag',
]

# Statistical constants (16-category analyses)
BONFERRONI_ALPHA = 0.05 / len(ALL_CATEGORIES)  # 0.003125
N_COMPARISONS = len(ALL_CATEGORIES)  # 16

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_category_name(category):
    """Convert full category name to display name"""
    return (category.replace('Comment_', '')
                   .replace('Critique_', '')
                   .replace('Perception_', '')
                   .replace('Response_', '')
                   .replace('Racist_Flag', 'racist'))

def convert_categories_to_binary(df):
    """Convert all category columns to binary (0/1)"""
    for cat in ALL_CATEGORIES:
        if cat in df.columns:
            df[cat] = pd.to_numeric(df[cat], errors='coerce').fillna(0)
            df[cat] = (df[cat] > 0).astype(int)
        else:
            df[cat] = 0
    return df

def add_city_size_grouping(df):
    """Add city size grouping (Large/Small/Unknown)"""
    # Find city column
    city_col = None
    if 'city' in df.columns:
        city_col = 'city'
    elif 'City' in df.columns:
        city_col = 'City'
    
    if city_col:
        df['City_Lower'] = df[city_col].str.lower()
        df['City_Size'] = df['City_Lower'].apply(
            lambda x: 'Large' if x in LARGE_CITIES else ('Small' if x in SMALL_CITIES else 'Unknown')
        )
        if city_col == 'city':
            df['City'] = df['city']
    
    return df

def one_sample_test(category_metrics, overall_mean, overall_std):
    """
    Perform one-sample test comparing category mean to overall mean.
    Returns p-value (more conservative of Wilcoxon and t-test).
    """
    n = len(category_metrics)
    if n == 0:
        return 1.0
    
    mean_category = np.mean(category_metrics)
    
    # One-sample Wilcoxon signed-rank test (non-parametric)
    differences = category_metrics - overall_mean
    try:
        _, p_wilcoxon = stats.wilcoxon(differences, alternative='two-sided')
    except (ValueError, RuntimeWarning):
        # If all differences are zero or too few observations
        p_wilcoxon = 1.0
    
    # One-sample t-test (parametric)
    if n > 1 and overall_std > 0:
        t_statistic = (mean_category - overall_mean) / (overall_std / np.sqrt(n))
        p_ttest = 2 * (1 - stats.t.cdf(abs(t_statistic), df=n-1))
    else:
        p_ttest = 1.0
    
    # Use more conservative p-value
    return max(p_wilcoxon, p_ttest)

def calculate_standard_errors(df, category, metric_col, total_posts):
    """Calculate standard errors for prevalence and mean engagement"""
    category_mask = (df[category] == 1)
    category_count = category_mask.sum()
    category_metrics = df[category_mask][metric_col].values
    
    # SE for prevalence
    prevalence = (category_count / total_posts) * 100
    se_prevalence = np.sqrt((prevalence / 100) * (1 - prevalence / 100) / total_posts) * 100
    
    # SE for mean engagement
    se_mean_engagement = stats.sem(category_metrics) if len(category_metrics) > 1 else 0
    
    return se_prevalence, se_mean_engagement

def median_iqr(values):
    """Return (median, q25, q75) ignoring NaNs."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return 0.0, 0.0, 0.0
    med = float(np.median(v))
    q25 = float(np.quantile(v, 0.25))
    q75 = float(np.quantile(v, 0.75))
    return med, q25, q75

def setup_plotting_style():
    """Setup matplotlib and seaborn styles"""
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('ggplot')
    sns.set_palette("husl")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_reddit_data():
    """Load Reddit data with scores and GPT4 category classifications"""
    print("="*80)
    print("LOADING REDDIT DATA")
    print("="*80)
    
    reddit_file = 'complete_dataset/all_reddit_comments.csv'
    print(f"Loading {reddit_file}...")
    reddit_df = pd.read_csv(reddit_file, low_memory=False)
    print(f"  Loaded {len(reddit_df)} Reddit comments")
    
    gpt4_file = 'output/reddit/gpt4/classified_comments_reddit_all_gpt4_reddit_flags.csv'
    print(f"Loading {gpt4_file}...")
    gpt4_df = pd.read_csv(gpt4_file, low_memory=False)
    print(f"  Loaded {len(gpt4_df)} GPT4 classified comments")
    
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
    merged_df = merged_df.dropna(subset=['Comment Score'])
    
    # Convert categories and add city size
    merged_df = convert_categories_to_binary(merged_df)
    merged_df = add_city_size_grouping(merged_df)
    
    # Set engagement metric
    merged_df['Engagement_Metric'] = merged_df['Comment Score']
    merged_df['Source'] = 'Reddit'
    merged_df['Metric_Name'] = 'Comment Score'
    
    print(f"  Final dataset: {len(merged_df)} comments with valid scores")
    print(f"  Score range: {merged_df['Comment Score'].min():.0f} to {merged_df['Comment Score'].max():.0f}")
    print(f"  Score mean: {merged_df['Comment Score'].mean():.2f}, median: {merged_df['Comment Score'].median():.2f}")
    
    return merged_df

def load_twitter_data():
    """Load X data with GPT4 categories. Keeps posts with impressions > 0 and like_count >= 1 so
    like-rate and impression-weighted prevalence/importance exclude zero-like posts."""
    print("="*80)
    print("LOADING X DATA")
    print("="*80)
    
    twitter_file = 'complete_dataset/all_twitter_posts_merged_with_details.csv'
    print(f"Loading {twitter_file}...")
    twitter_df = pd.read_csv(twitter_file, low_memory=False)
    print(f"  Loaded {len(twitter_df)} X posts")
    
    gpt4_file = 'output/x/gpt4/classified_comments_x_all_gpt4_x_flags.csv'
    print(f"Loading {gpt4_file}...")
    gpt4_df = pd.read_csv(gpt4_file, low_memory=False)
    print(f"  Loaded {len(gpt4_df)} GPT4 classified posts")
    
    print("Merging datasets...")
    twitter_df_merge = twitter_df.rename(columns={'Deidentified_text': 'Comment'})
    merged_df = twitter_df_merge.merge(
        gpt4_df[['Comment', 'City'] + ALL_CATEGORIES],
        on='Comment',
        how='inner',
        suffixes=('', '_gpt4')
    )
    print(f"  Merged dataset: {len(merged_df)} posts")
    
    # Clean and prepare engagement metrics
    merged_df['like_count'] = pd.to_numeric(merged_df['like_count'], errors='coerce').fillna(0)
    merged_df['impression_count'] = pd.to_numeric(merged_df['impression_count'], errors='coerce').fillna(0)
    
    # Calculate Like Rate = likes / impressions
    merged_df['Like_Rate'] = merged_df['like_count'] / merged_df['impression_count'].replace(0, np.nan)
    merged_df['Like_Rate'] = merged_df['Like_Rate'].fillna(0)
    
    # Remove rows with no impressions
    merged_df = merged_df[merged_df['impression_count'] > 0]
    merged_df = merged_df[merged_df['Like_Rate'].notna()]
    # Like rate and impression-weighted stats: only posts with ≥1 like (avoids zero-inflated medians)
    _n = len(merged_df)
    merged_df = merged_df[merged_df['like_count'] >= 1]
    print(f"  Filtered to ≥1 like: {len(merged_df):,} posts (excluded {_n - len(merged_df):,} with 0 likes)")
    
    # Convert categories and add city size
    merged_df = convert_categories_to_binary(merged_df)
    merged_df = add_city_size_grouping(merged_df)
    
    # Set engagement metric
    merged_df['Engagement_Metric'] = merged_df['Like_Rate']
    merged_df['Source'] = 'X'
    merged_df['Metric_Name'] = 'Like Rate'
    
    print(f"  Final dataset: {len(merged_df)} posts with valid engagement metrics")
    print(f"  Like Rate range: {merged_df['Like_Rate'].min():.6f} to {merged_df['Like_Rate'].max():.6f}")
    print(f"  Like Rate mean: {merged_df['Like_Rate'].mean():.6f}, median: {merged_df['Like_Rate'].median():.6f}")
    
    return merged_df

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_category_prevalence(df, source_name):
    """Analyze overall category prevalence
    
    For Twitter/X: Prevalence is weighted by impressions (sum of impressions with category / total impressions)
    For Reddit: Prevalence is simple count (count with category / total posts)
    """
    print("\n" + "="*80)
    print(f"ANALYSIS 1: CATEGORY PREVALENCE ({source_name.upper()})")
    print("="*80)
    
    results = []
    metric_col = 'Engagement_Metric'
    metric_name = df['Metric_Name'].iloc[0]
    
    # Check if this is Twitter/X (has impression_count column)
    is_twitter = 'impression_count' in df.columns
    
    if is_twitter:
        # Twitter/X: Weight by impressions
        total_impressions = df['impression_count'].sum()
        print(f"Calculating impression-weighted prevalence (total impressions: {total_impressions:,})")
    else:
        # Reddit: Simple count
        total_posts = len(df)
        print(f"Calculating count-based prevalence (total posts: {total_posts:,})")
    
    for category in ALL_CATEGORIES:
        category_name = clean_category_name(category)
        category_mask = (df[category] == 1)
        
        if is_twitter:
            # Twitter/X: Prevalence = (sum of impressions with category) / (total impressions) * 100
            category_impressions = df[category_mask]['impression_count'].sum()
            percentage = (category_impressions / total_impressions) * 100 if total_impressions > 0 else 0
            count = category_mask.sum()  # Still track count for reference
        else:
            # Reddit: Prevalence = (count with category) / (total posts) * 100
            count = category_mask.sum()
            percentage = (count / total_posts) * 100
        
        # Calculate mean engagement metric for this category
        category_metrics = df[category_mask][metric_col].values
        mean_metric = np.mean(category_metrics) if len(category_metrics) > 0 else 0
        median_metric = np.median(category_metrics) if len(category_metrics) > 0 else 0
        
        results.append({
            'Category': category_name,
            'Full_Category': category,
            'Count': count,
            'Percentage': percentage,
            'Mean_Engagement': mean_metric,
            'Median_Engagement': median_metric
        })
    
    results_df = pd.DataFrame(results).sort_values('Percentage', ascending=False)
    
    print("\nCategory Prevalence (sorted by frequency):")
    print("-" * 80)
    prev_type = "impression-weighted" if is_twitter else "count-based"
    for _, row in results_df.iterrows():
        print(f"  {row['Category']:40s}: {row['Count']:6,} posts ({row['Percentage']:5.2f}% {prev_type}) | Mean {metric_name}: {row['Mean_Engagement']:8.6f}")
    
    return results_df

def analyze_category_importance(df, source_name):
    """Analyze category importance
    
    For Twitter/X: Importance = Total impressions for that category
    For Reddit: Importance = Prevalence × Mean Engagement (as before)
    """
    print("\n" + "="*80)
    print(f"ANALYSIS 2: CATEGORY IMPORTANCE ({source_name.upper()})")
    print("="*80)
    
    results = []
    metric_col = 'Engagement_Metric'
    metric_name = df['Metric_Name'].iloc[0]
    
    # Check if this is Twitter/X (has impression_count column)
    is_twitter = 'impression_count' in df.columns
    
    if is_twitter:
        # Twitter/X: Weight by impressions
        total_impressions = df['impression_count'].sum()
    else:
        # Reddit: Simple count
        total_posts = len(df)
    
    for category in ALL_CATEGORIES:
        category_name = clean_category_name(category)
        category_mask = (df[category] == 1)
        category_count = category_mask.sum()
        
        if is_twitter:
            # Twitter/X: Importance = Total impressions for that category
            category_impressions = df[category_mask]['impression_count'].sum()
            importance_score = category_impressions
            category_prevalence = (category_impressions / total_impressions) * 100 if total_impressions > 0 else 0
        else:
            # Reddit: Importance = Prevalence × Mean Engagement
            category_prevalence = (category_count / total_posts) * 100
            category_metrics = df[category_mask][metric_col].values
            mean_metric = np.mean(category_metrics) if len(category_metrics) > 0 else 0
            importance_score = category_prevalence * mean_metric
        
        category_metrics = df[category_mask][metric_col].values
        mean_metric = np.mean(category_metrics) if len(category_metrics) > 0 else 0
        median_metric = np.median(category_metrics) if len(category_metrics) > 0 else 0
        
        results.append({
            'Category': category_name,
            'Full_Category': category,
            'Count': category_count,
            'Prevalence_Percent': category_prevalence,
            'Mean_Engagement': mean_metric,
            'Median_Engagement': median_metric,
            'Importance_Score': importance_score
        })
    
    importance_df = pd.DataFrame(results).sort_values('Importance_Score', ascending=False)
    
    if is_twitter:
        print(f"\nCategory Importance (Total Impressions per Category):")
        print("-" * 80)
        for _, row in importance_df.iterrows():
            print(f"  {row['Category']:40s}: Importance={row['Importance_Score']:12,.0f} impressions ({row['Prevalence_Percent']:5.2f}% of total)")
    else:
        print(f"\nCategory Importance (Prevalence × Mean {metric_name}):")
        print("-" * 80)
        for _, row in importance_df.iterrows():
            print(f"  {row['Category']:40s}: Importance={row['Importance_Score']:10.6f} (Prevalence={row['Prevalence_Percent']:5.2f}% × {metric_name}={row['Mean_Engagement']:8.6f})")
    
    return importance_df

def analyze_engagement_differences(df, source_name, categories=None, analysis_label='CATEGORY'):
    """Analyze engagement metric differences by category (robust + nonparametric).

    Uses median differences (category - overall) and Mann–Whitney U test against the
    complement set (category==1 vs category==0). This avoids normality assumptions.
    """
    categories = categories or ALL_CATEGORIES
    n_comparisons = len(categories)

    print("\n" + "="*80)
    print(f"ANALYSIS 3: ENGAGEMENT DIFFERENCES BY {analysis_label} ({source_name.upper()})")
    print("="*80)
    print(
        f"Comparing each of {n_comparisons} categories to all others "
        f"(Mann–Whitney U; Bonferroni α = 0.05/{n_comparisons})"
    )
    
    results = []
    metric_col = 'Engagement_Metric'
    metric_name = df['Metric_Name'].iloc[0]
    all_metrics = df[metric_col].values
    overall_median, overall_q25, overall_q75 = median_iqr(all_metrics)
    
    print(f"\nOverall median {metric_name}: {overall_median:.6f} (IQR: {overall_q25:.6f}–{overall_q75:.6f})")
    
    for category in categories:
        category_metrics = df[df[category] == 1][metric_col].values
        other_metrics = df[df[category] == 0][metric_col].values
        n_category = len(category_metrics)
        
        if n_category == 0:
            continue
        
        median_category, q25_category, q75_category = median_iqr(category_metrics)
        
        # Two-sample nonparametric test: category vs others
        if len(other_metrics) > 0:
            try:
                _, p_value = stats.mannwhitneyu(category_metrics, other_metrics, alternative='two-sided')
            except Exception:
                p_value = 1.0
        else:
            p_value = 1.0
        
        # Bonferroni correction
        p_value_corrected = min(p_value * n_comparisons, 1.0)
        is_significant = p_value_corrected < 0.05
        
        if is_significant:
            direction = "HIGHER" if median_category > overall_median else "LOWER"
        else:
            direction = "NS"
        
        # Cohen's d is mean/SD based; keep for backward compatibility but set to NaN (not appropriate here)
        cohens_d = np.nan
        
        category_name = clean_category_name(category)
        diff_point = median_category - overall_median
        
        results.append({
            'Category': category_name,
            'Full_Category': category,
            'N_Category': n_category,
            'Mean_Category': float(np.mean(category_metrics)) if len(category_metrics) else 0.0,
            'Overall_Mean': float(np.mean(all_metrics)) if len(all_metrics) else 0.0,
            'Median_Category': median_category,
            'Q25_Category': q25_category,
            'Q75_Category': q75_category,
            'Overall_Median': overall_median,
            'Difference': diff_point,
            'P_Value': p_value,
            'P_Value_Corrected': p_value_corrected,
            'Significant': is_significant,
            'Direction': direction,
            'Cohens_D': cohens_d
        })
        
        print(f"\n{category_name}:")
        print(f"  N (category): {n_category:,}")
        print(f"  Median (category): {median_category:.8f} (IQR: {q25_category:.8f}–{q75_category:.8f})")
        print(f"  Overall median: {overall_median:.8f}")
        print(f"  Difference from overall median: {median_category - overall_median:+.8f}")
        print(f"  P-value (Bonferroni corrected): {p_value_corrected:.6f}")
        print(f"  Significant: {'YES' if is_significant else 'NO'} ({direction})")
    
    results_df = pd.DataFrame(results)
    return results_df

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_prevalence_engagement_chart(df, importance_df, display_name, output_dir, metric_name, file_source):
    """Create two-panel chart: Prevalence and Mean Engagement
    
    Args:
        display_name: Name for titles (e.g., "X" or "Reddit")
        file_source: Name for file paths (e.g., "x" or "reddit")
    """
    overall_mean = df['Engagement_Metric'].mean()
    overall_prevalence = 100.0
    
    # Check if this is Twitter/X (has impression_count column)
    is_twitter = 'impression_count' in df.columns
    
    # Sort by prevalence
    importance_sorted = importance_df.sort_values('Prevalence_Percent', ascending=False).copy()
    categories = importance_sorted['Category'].values
    prevalences = importance_sorted['Prevalence_Percent'].values
    mean_engagements = importance_sorted['Mean_Engagement'].values
    
    # Calculate standard errors
    metric_col = 'Engagement_Metric'
    se_prevalences = []
    se_mean_engagements = []
    
    if is_twitter:
        # Twitter/X: SE for impression-weighted prevalence
        total_impressions = df['impression_count'].sum()
        for _, row in importance_sorted.iterrows():
            category = row['Full_Category']
            category_mask = (df[category] == 1)
            category_impressions = df[category_mask]['impression_count'].sum()
            prevalence = (category_impressions / total_impressions) * 100 if total_impressions > 0 else 0
            
            # SE for impression-weighted prevalence (using binomial approximation)
            se_prevalence = np.sqrt((prevalence / 100) * (1 - prevalence / 100) / total_impressions) * 100 * np.sqrt(total_impressions)
            se_prevalences.append(se_prevalence)
            
            # SE for mean engagement
            category_metrics = df[category_mask][metric_col].values
            se_mean_engagement = stats.sem(category_metrics) if len(category_metrics) > 1 else 0
            se_mean_engagements.append(se_mean_engagement)
    else:
        # Reddit: SE for count-based prevalence
        total_posts = len(df)
        for _, row in importance_sorted.iterrows():
            se_prev, se_mean = calculate_standard_errors(df, row['Full_Category'], metric_col, total_posts)
            se_prevalences.append(se_prev)
            se_mean_engagements.append(se_mean)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Prevalence (no error bars, just percentages)
    colors_prevalence = ['#3498db' if p > overall_prevalence else '#e74c3c' for p in prevalences]
    ax1.bar(range(len(categories)), prevalences,
            color=colors_prevalence, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.axhline(y=overall_prevalence, color='black', linestyle='--', linewidth=2)
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Prevalence (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Category Prevalence on {display_name}\n(All 16 Categories)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, max(prevalences) * 1.15])
    
    # Plot 2: Mean Engagement
    colors_engagement = ['#2ecc71' if s > overall_mean else '#e74c3c' for s in mean_engagements]
    ax2.bar(range(len(categories)), mean_engagements, yerr=se_mean_engagements,
            color=colors_engagement, alpha=0.7, edgecolor='black', linewidth=0.5,
            capsize=3, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
    ax2.axhline(y=overall_mean, color='black', linestyle='--', linewidth=2)
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel(f'Mean {metric_name}', fontsize=12, fontweight='bold')
    ax2.set_title(f'Mean {metric_name} by Category\n(All 16 Categories)', 
                  fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, max(mean_engagements) * 1.15])
    
    plt.suptitle(f'Category Analysis: Prevalence and Mean {metric_name} ({display_name})\n' + 
                'Green = Above Overall Average, Red = Below Overall Average',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filename = f'{file_source.lower()}_category_prevalence_and_engagement_separate'
    plt.savefig(output_dir / f'{filename}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved separate prevalence and engagement charts")

def create_importance_chart(df, importance_df, display_name, output_dir, metric_name, file_source):
    """Create category importance chart with error bars and Bonferroni correction
    
    Args:
        display_name: Name for titles (e.g., "X" or "Reddit")
        file_source: Name for file paths (e.g., "x" or "reddit")
    """
    metric_col = 'Engagement_Metric'
    overall_mean = df[metric_col].mean()
    overall_std = df[metric_col].std(ddof=1)
    
    # Check if this is Twitter/X (has impression_count column)
    is_twitter = 'impression_count' in df.columns
    
    if is_twitter:
        total_impressions = df['impression_count'].sum()
    else:
        total_posts = len(df)
    
    # Calculate errors and significance
    importance_with_errors = []
    significance_results = []
    
    for _, row in importance_df.iterrows():
        category = row['Full_Category']
        category_mask = (df[category] == 1)
        category_metrics = df[category_mask][metric_col].values
        category_count = category_mask.sum()
        
        # Standard errors
        # Use prevalence from importance_df (already impression-weighted for Twitter/X)
        prevalence = row['Prevalence_Percent']
        
        # Standard errors - use impression-weighted for Twitter/X
        if is_twitter:
            # Twitter/X: SE for impression-weighted prevalence (using binomial approximation)
            se_prevalence = np.sqrt((prevalence / 100) * (1 - prevalence / 100) / total_impressions) * 100 * np.sqrt(total_impressions)
        else:
            # Reddit: SE for count-based prevalence
            se_prev, _ = calculate_standard_errors(df, category, metric_col, total_posts)
            se_prevalence = se_prev
        
        # SE for mean engagement
        se_mean = stats.sem(category_metrics) if len(category_metrics) > 1 else 0
        mean_engagement = row['Mean_Engagement']
        se_importance = abs(prevalence) * se_mean + abs(mean_engagement) * se_prevalence
        
        # Statistical test
        if len(category_metrics) > 0:
            p_value = one_sample_test(category_metrics, overall_mean, overall_std)
            p_value_corrected = min(p_value * N_COMPARISONS, 1.0)
            is_significant = p_value_corrected < 0.05
        else:
            p_value = 1.0
            p_value_corrected = 1.0
            is_significant = False
        
        if is_twitter:
            # Twitter/X: Importance = Total impressions (no error bars needed)
            importance_with_errors.append({
                'Category': row['Category'],
                'Importance_Score': row['Importance_Score'],
                'SE_Importance': 0,  # No error bars for impression counts
                'Mean_Engagement': row['Mean_Engagement'],
                'SE_Mean_Engagement': stats.sem(category_metrics) if len(category_metrics) > 1 else 0,
                'Prevalence': row['Prevalence_Percent'],
                'SE_Prevalence': 0
            })
        else:
            # Reddit: Importance = Prevalence × Mean Engagement (with error bars)
            importance_with_errors.append({
                'Category': row['Category'],
                'Importance_Score': row['Importance_Score'],
                'SE_Importance': se_importance,
                'Mean_Engagement': mean_engagement,
                'SE_Mean_Engagement': se_mean,
                'Prevalence': prevalence,
                'SE_Prevalence': se_prev
            })
        
        significance_results.append({
            'Category': row['Category'],
            'P_Value': p_value,
            'P_Value_Corrected': p_value_corrected,
            'Significant': is_significant
        })
    
    importance_errors_df = pd.DataFrame(importance_with_errors).sort_values('Importance_Score', ascending=False)
    significance_df = pd.DataFrame(significance_results)
    importance_errors_df = importance_errors_df.merge(significance_df, on='Category')
    
    # Create chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    categories = importance_errors_df['Category'].values
    importance_scores = importance_errors_df['Importance_Score'].values
    errors = importance_errors_df['SE_Importance'].values
    is_significant = importance_errors_df['Significant'].values
    
    colors = ['#2ecc71' if sig else '#95a5a6' for sig in is_significant]
    
    # Only show error bars for Reddit (not for Twitter/X impression counts)
    if is_twitter:
        ax.bar(range(len(categories)), importance_scores, 
               color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    else:
        ax.bar(range(len(categories)), importance_scores, yerr=errors, 
               color=colors, alpha=0.7, edgecolor='black', linewidth=0.5,
               capsize=5, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
    
    # Add significance markers
    for i, (sig, score) in enumerate(zip(is_significant, importance_scores)):
        if sig:
            p_val = importance_errors_df.iloc[i]['P_Value_Corrected']
            marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
            error_val = errors[i] if not is_twitter else 0
            ax.text(i, score + error_val + max(importance_scores) * 0.02, marker, 
                   ha='center', va='bottom', fontsize=12, fontweight='bold', color='red')
    
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    if is_twitter:
        # Format y-axis in millions for Twitter/X
        from matplotlib.ticker import FuncFormatter
        def millions_formatter(x, p):
            return f'{x/1e6:.1f}M'
        ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
        ylabel = 'Importance Score (Total Impressions per Category, in millions)'
        title_suffix = '* p<0.05, ** p<0.01, *** p<0.001 (Bonferroni corrected)'
    else:
        ylabel = f'Importance Score\n(Prevalence × Mean {metric_name})'
        title_suffix = 'Error bars = Standard Error, * p<0.05, ** p<0.01, *** p<0.001 (Bonferroni corrected)'
    
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(f'Category Importance on {display_name} (All 16 Categories)\n' + title_suffix,
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.7, label='Significantly Different (Bonferroni corrected)'),
        Patch(facecolor='#95a5a6', alpha=0.7, label='Not Significant')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    filename = f'{file_source.lower()}_category_importance_all16_with_errors'
    plt.savefig(output_dir / f'{filename}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved category importance chart (all 16 with error bars and Bonferroni correction)")
    
    return importance_errors_df

def create_avg_impressions_chart(df, display_name, output_dir, file_source):
    """Median impressions vs overall + Mann–Whitney (same framework as engagement differences).

    Skewed impression counts: use medians; compare each category to its complement with
    Mann–Whitney U and Bonferroni correction. Plot median difference (category − overall median),
    no error bars.
    """
    if 'impression_count' not in df.columns:
        return None  # Only for Twitter/X
    
    metric_col = 'impression_count'
    all_vals = df[metric_col].values
    overall_median, overall_q25, overall_q75 = median_iqr(all_vals)
    
    results = []
    for category in ALL_CATEGORIES:
        category_name = clean_category_name(category)
        category_mask = (df[category] == 1)
        category_impressions = df[category_mask][metric_col].values
        other_impressions = df[df[category] == 0][metric_col].values
        n_category = len(category_impressions)
        
        if n_category == 0:
            continue
        
        med_cat, q25_c, q75_c = median_iqr(category_impressions)
        mean_impressions = float(np.mean(category_impressions))
        
        if len(other_impressions) > 0:
            try:
                _, p_value = stats.mannwhitneyu(category_impressions, other_impressions, alternative='two-sided')
            except Exception:
                p_value = 1.0
        else:
            p_value = 1.0
        
        p_value_corrected = min(p_value * N_COMPARISONS, 1.0)
        is_significant = p_value_corrected < 0.05
        diff = med_cat - overall_median
        if is_significant:
            direction = "HIGHER" if med_cat > overall_median else "LOWER"
        else:
            direction = "NS"
        
        results.append({
            'Category': category_name,
            'Full_Category': category,
            'N_Category': n_category,
            'Mean_Impressions': mean_impressions,
            'Median_Impressions': med_cat,
            'Q25_Impressions': q25_c,
            'Q75_Impressions': q75_c,
            'Overall_Median_Impressions': overall_median,
            'Overall_Q25_Impressions': overall_q25,
            'Overall_Q75_Impressions': overall_q75,
            'Difference': diff,
            'P_Value': p_value,
            'P_Value_Corrected': p_value_corrected,
            'Significant': is_significant,
            'Direction': direction,
        })
    
    if len(results) == 0:
        return None
    
    results_df = pd.DataFrame(results).sort_values('Difference', ascending=True)
    
    # Horizontal bar chart (aligned with engagement-differences chart)
    fig, ax = plt.subplots(figsize=(14, 8))
    
    categories = results_df['Category'].values
    differences = results_df['Difference'].values
    is_significant = results_df['Significant'].values
    
    colors = []
    for sig, diff in zip(is_significant, differences):
        if sig:
            colors.append('green' if diff > 0 else 'red')
        else:
            colors.append('gray')
    
    ax.barh(range(len(categories)), differences,
            color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    for i, (sig, diff) in enumerate(zip(is_significant, differences)):
        if sig:
            p_val = results_df.iloc[i]['P_Value_Corrected']
            marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
            ad = np.abs(differences)
            offset = abs(diff) * 0.1 if diff != 0 else (float(np.max(ad)) * 0.02 if ad.size else 1.0)
            ax.text(diff + offset if diff > 0 else diff - offset, i, marker,
                    ha='left' if diff > 0 else 'right', va='center',
                    fontsize=12, fontweight='bold')
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=10)
    
    from matplotlib.ticker import FuncFormatter
    
    def _fmt_impr(x, _p):
        ax_abs = abs(x)
        if ax_abs >= 1e6:
            return f'{x/1e6:.2f}M'
        if ax_abs >= 1e3:
            return f'{x/1e3:.0f}K'
        return f'{x:,.0f}'
    
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_impr))
    ax.set_xlabel(
        'Median impressions difference (category − overall median)',
        fontsize=12, fontweight='bold',
    )
    ax.set_title(
        f'Impressions by category on {display_name} (median-based)\n'
        'Comparing each category to all others (Mann–Whitney U)\n'
        '* p<0.05, ** p<0.01, *** p<0.001 (Bonferroni corrected)',
        fontsize=14, fontweight='bold',
    )
    ax.grid(axis='x', alpha=0.3)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Significantly Higher'),
        Patch(facecolor='red', alpha=0.7, label='Significantly Lower'),
        Patch(facecolor='gray', alpha=0.7, label='Not Significant'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    filename = f'{file_source.lower()}_category_avg_impressions'
    plt.savefig(output_dir / f'{filename}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved average impressions per tweet chart (median difference, Mann–Whitney, Bonferroni)")
    
    return results_df

def create_engagement_differences_chart(
    df,
    engagement_diff_df,
    display_name,
    output_dir,
    metric_name,
    file_source,
    filename_stem=None,
    chart_title=None,
    n_comparisons=None,
    fig_height_per_category=1.2,
    bold_y_labels=False,
):
    """Create engagement differences chart (horizontal bar chart)
    
    Args:
        display_name: Name for titles (e.g., "X" or "Reddit")
        file_source: Name for file paths (e.g., "x" or "reddit")
        filename_stem: Output base name without extension (default: {source}_category_engagement_differences)
        chart_title: Optional override for main title line
        n_comparisons: Bonferroni family size for subtitle (default: 16)
    """
    if len(engagement_diff_df) == 0:
        return
    
    n_comparisons = n_comparisons or len(ALL_CATEGORIES)
    engagement_diff_sorted = engagement_diff_df.sort_values('Difference', ascending=True)
    
    fig_height = max(2.5, fig_height_per_category * len(engagement_diff_sorted))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    
    categories = engagement_diff_sorted['Category'].values
    differences = engagement_diff_sorted['Difference'].values
    is_significant = engagement_diff_sorted['Significant'].values
    
    # Color scheme: Higher = green, Lower = red, Not significant = gray
    colors = []
    for sig, diff in zip(is_significant, differences):
        if sig:
            colors.append('green' if diff > 0 else 'red')
        else:
            colors.append('gray')  # Gray for not significant
    
    ax.barh(range(len(categories)), differences,
            color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add significance markers
    for i, (sig, diff) in enumerate(zip(is_significant, differences)):
        if sig:
            p_val = engagement_diff_sorted.iloc[i]['P_Value_Corrected']
            marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
            offset = abs(diff) * 0.1 if diff != 0 else max(abs(differences)) * 0.02
            ax.text(diff + offset if diff > 0 else diff - offset, i, marker, 
                   ha='left' if diff > 0 else 'right', va='center', 
                   fontsize=12, fontweight='bold')
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_yticks(range(len(categories)))
    y_label_kw = {'fontsize': 10}
    if bold_y_labels:
        y_label_kw['fontweight'] = 'bold'
    ax.set_yticklabels(categories, **y_label_kw)
    ax.set_xlabel(f'Median {metric_name} Difference (Category - Overall Median)', 
                  fontsize=12, fontweight='bold')
    title_line = chart_title or f'{metric_name} Differences by Category on {display_name}'
    ax.set_title(
        title_line + '\n'
        'Comparing each category to all others (Mann–Whitney U)\n'
        f'* p<0.05, ** p<0.01, *** p<0.001 (Bonferroni corrected, α = 0.05/{n_comparisons})',
        fontsize=14,
        fontweight='bold',
    )
    ax.grid(axis='x', alpha=0.3)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Significantly Higher'),
        Patch(facecolor='red', alpha=0.7, label='Significantly Lower'),
        Patch(facecolor='gray', alpha=0.7, label='Not Significant')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    filename = filename_stem or f'{file_source.lower()}_category_engagement_differences'
    plt.savefig(output_dir / f'{filename}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved engagement differences chart: {filename}.pdf")

def create_visualizations(df, prevalence_df, importance_df, engagement_diff_df, display_name, file_source):
    """Create all bar chart visualizations with error bars
    
    Args:
        display_name: Name for titles (e.g., "X" or "Reddit")
        file_source: Name for file paths (e.g., "x" or "reddit")
    """
    print("\n" + "="*80)
    print(f"CREATING VISUALIZATIONS ({display_name})")
    print("="*80)
    
    output_dir = Path('output/charts') / file_source.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metric_name = df['Metric_Name'].iloc[0]
    
    # Create all charts (use display_name for titles, file_source for file paths)
    create_prevalence_engagement_chart(df, importance_df, display_name, output_dir, metric_name, file_source)
    importance_errors_df = create_importance_chart(df, importance_df, display_name, output_dir, metric_name, file_source)
    create_engagement_differences_chart(df, engagement_diff_df, display_name, output_dir, metric_name, file_source)

    bias_engagement_diff_df = analyze_engagement_differences(
        df,
        display_name,
        categories=NEGATIVE_BIAS_FRAME_CATEGORIES,
        analysis_label='NEGATIVE BIAS FRAME',
    )
    create_engagement_differences_chart(
        df,
        bias_engagement_diff_df,
        display_name,
        output_dir,
        metric_name,
        file_source,
        filename_stem=f'{file_source.lower()}_negative_bias_frame_engagement_differences',
        chart_title=f'{metric_name} Differences by Negative Bias Frame Category on {display_name}',
        n_comparisons=len(NEGATIVE_BIAS_FRAME_CATEGORIES),
        fig_height_per_category=0.6,
        bold_y_labels=True,
    )
    
    # Create average impressions chart for X/Twitter
    avg_impressions_df = None
    if 'impression_count' in df.columns:
        avg_impressions_df = create_avg_impressions_chart(df, display_name, output_dir, file_source)
    
    print(f"\nAll visualizations saved to: {output_dir}/")
    
    return importance_errors_df, avg_impressions_df, bias_engagement_diff_df

# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_all_results(
    prevalence_df,
    importance_df,
    engagement_diff_df,
    importance_errors_df,
    source_name,
    avg_impressions_df=None,
    bias_engagement_diff_df=None,
):
    """Save all results to CSV files in source-specific folder"""
    print("\n" + "="*80)
    print(f"SAVING RESULTS TO CSV ({source_name.upper()})")
    print("="*80)
    
    # Create source-specific output directory
    output_dir = Path('output') / source_name.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files_saved = []
    
    # Save each dataframe (no prefix needed since files are in source-specific folder)
    files = [
        (prevalence_df, 'category_pattern_analysis_prevalence.csv', 'category prevalence'),
        (importance_df, 'category_pattern_analysis_importance.csv', 'category importance'),
        (engagement_diff_df, 'category_engagement_differences.csv', 'engagement differences analysis')
    ]
    
    for df_to_save, filename, description in files:
        filepath = output_dir / filename
        df_to_save.to_csv(filepath, index=False)
        files_saved.append(filepath)
        print(f"  ✓ Saved {description}: {filepath}")

    if bias_engagement_diff_df is not None and len(bias_engagement_diff_df) > 0:
        filepath = output_dir / 'negative_bias_frame_engagement_differences.csv'
        bias_engagement_diff_df.to_csv(filepath, index=False)
        files_saved.append(filepath)
        print(f"  ✓ Saved negative bias frame engagement differences: {filepath}")
    
    if importance_errors_df is not None:
        filepath = output_dir / 'category_pattern_analysis_importance_with_errors.csv'
        importance_errors_df.to_csv(filepath, index=False)
        files_saved.append(filepath)
        print(f"  ✓ Saved category importance with errors and significance: {filepath}")
    
    # Save average impressions data if available (for X/Twitter)
    if avg_impressions_df is not None:
        filepath = output_dir / 'category_avg_impressions.csv'
        avg_impressions_df.to_csv(filepath, index=False)
        files_saved.append(filepath)
        print(f"  ✓ Saved average impressions per category: {filepath}")
    
    print(f"\n✓ All results saved! Total files: {len(files_saved)}")
    return files_saved

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to run analysis for Reddit and/or Twitter"""
    parser = argparse.ArgumentParser(
        description='Unified Category Pattern Analysis for Reddit and X (formerly Twitter)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/category_pattern_analysis_unified.py
  python scripts/category_pattern_analysis_unified.py --source reddit
  python scripts/category_pattern_analysis_unified.py --source x
        """
    )
    parser.add_argument('--source', type=str, choices=['reddit', 'x', 'twitter', 'both'], 
                       default='both', help='Data source to analyze: reddit, x (or twitter), or both (default: both)')
    args = parser.parse_args()
    
    # Setup plotting style
    setup_plotting_style()
    
    # Determine sources to process (normalize 'twitter' to 'x')
    if args.source == 'both':
        sources_to_process = ['reddit', 'x']
    elif args.source == 'twitter':
        sources_to_process = ['x']  # Normalize 'twitter' to 'x'
    else:
        sources_to_process = [args.source]
    
    for source in sources_to_process:
        print("\n" + "="*80)
        # Display name: "X" for titles, but use "x" for file paths
        display_name = 'X' if source == 'x' else source.capitalize()
        print(f"PROCESSING {display_name}")
        print("="*80)
        
        try:
            # Load data
            if source == 'reddit':
                df = load_reddit_data()
            else:  # x or twitter
                df = load_twitter_data()
            
            # Use display name for titles, but 'x' for file paths
            display_name = 'X' if source == 'x' else source.capitalize()
            
            # Perform analyses (use display name for titles)
            prevalence_df = analyze_category_prevalence(df, display_name)
            importance_df = analyze_category_importance(df, display_name)
            engagement_diff_df = analyze_engagement_differences(df, display_name)
            
            # Create visualizations (use display name for titles, source for file paths)
            importance_errors_df, avg_impressions_df, bias_engagement_diff_df = create_visualizations(
                df, prevalence_df, importance_df, engagement_diff_df, display_name, source
            )
            
            # Save all results (use source for file paths)
            save_all_results(
                prevalence_df,
                importance_df,
                engagement_diff_df,
                importance_errors_df,
                source,
                avg_impressions_df,
                bias_engagement_diff_df,
            )
            
            # Summary
            print("\n" + "="*80)
            print(f"SUMMARY ({display_name})")
            print("="*80)
            print(f"Total posts analyzed: {len(df):,}")
            print(f"Total categories: {len(ALL_CATEGORIES)}")
            print(f"Most common category: {prevalence_df.iloc[0]['Category']} ({prevalence_df.iloc[0]['Percentage']:.2f}%)")
            print(f"Least common category: {prevalence_df.iloc[-1]['Category']} ({prevalence_df.iloc[-1]['Percentage']:.2f}%)")
            # Format importance score based on source type
            is_twitter_source = 'impression_count' in df.columns if 'df' in locals() else source == 'x'
            if is_twitter_source:
                print(f"Most important category: {importance_df.iloc[0]['Category']} (Importance={importance_df.iloc[0]['Importance_Score']:,.0f} impressions)")
            else:
                print(f"Most important category: {importance_df.iloc[0]['Category']} (Importance={importance_df.iloc[0]['Importance_Score']:.6f})")
            
            if len(engagement_diff_df) > 0:
                n_significant = engagement_diff_df['Significant'].sum()
                n_higher = ((engagement_diff_df['Significant']) & 
                           (engagement_diff_df['Direction'] == 'HIGHER')).sum()
                n_lower = ((engagement_diff_df['Significant']) & 
                          (engagement_diff_df['Direction'] == 'LOWER')).sum()
                print(f"Significantly different engagement (Bonferroni corrected): {n_significant}")
                print(f"  - Significantly HIGHER: {n_higher}")
                print(f"  - Significantly LOWER: {n_lower}")
            
            print("\n" + "="*80)
            print(f"Analysis complete for {display_name}! All results saved.")
            print("="*80)
            
        except FileNotFoundError as e:
            print(f"ERROR: File not found: {e}")
            print(f"Skipping {display_name}...")
            continue
        except Exception as e:
            print(f"ERROR processing {display_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == '__main__':
    main()
