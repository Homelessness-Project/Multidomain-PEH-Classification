#!/usr/bin/env python3
"""
Category Pattern Analysis for Twitter/X: Comprehensive research analysis using engagement metrics

This script performs the same analyses as category_pattern_analysis.py but for Twitter/X data,
using engagement metrics (like_count, impression_count, like_rate, etc.) instead of comment scores.

Usage:
    python scripts/category_pattern_analysis_twitter.py
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
    """Load Twitter/X data with engagement metrics and GPT4 category classifications"""
    print("="*80)
    print("LOADING TWITTER/X DATA")
    print("="*80)
    
    # Load Twitter posts with engagement metrics
    twitter_file = 'complete_dataset/all_twitter_posts_merged_with_details.csv'
    print(f"Loading {twitter_file}...")
    twitter_df = pd.read_csv(twitter_file, low_memory=False)
    print(f"  Loaded {len(twitter_df)} Twitter posts")
    
    # Load GPT4 classified posts
    gpt4_file = 'output/x/gpt4/classified_comments_x_all_gpt4_x_flags.csv'
    print(f"Loading {gpt4_file}...")
    gpt4_df = pd.read_csv(gpt4_file, low_memory=False)
    print(f"  Loaded {len(gpt4_df)} GPT4 classified posts")
    
    # Merge on text (Twitter uses 'Deidentified_text', GPT4 uses 'Comment')
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
    # Convert to numeric
    merged_df['like_count'] = pd.to_numeric(merged_df['like_count'], errors='coerce').fillna(0)
    merged_df['impression_count'] = pd.to_numeric(merged_df['impression_count'], errors='coerce').fillna(0)
    merged_df['retweet_count'] = pd.to_numeric(merged_df['retweet_count'], errors='coerce').fillna(0)
    merged_df['reply_count'] = pd.to_numeric(merged_df['reply_count'], errors='coerce').fillna(0)
    merged_df['quote_count'] = pd.to_numeric(merged_df['quote_count'], errors='coerce').fillna(0)
    
    # Calculate engagement metrics
    # Like Rate = likes / impressions (best metric for engagement relative to visibility)
    merged_df['Like_Rate'] = merged_df['like_count'] / merged_df['impression_count'].replace(0, np.nan)
    merged_df['Like_Rate'] = merged_df['Like_Rate'].fillna(0)
    
    # Total Engagement = likes + retweets + replies + quotes
    merged_df['Total_Engagement'] = (merged_df['like_count'] + merged_df['retweet_count'] + 
                                     merged_df['reply_count'] + merged_df['quote_count'])
    
    # Engagement Rate = total engagement / impressions
    merged_df['Engagement_Rate'] = merged_df['Total_Engagement'] / merged_df['impression_count'].replace(0, np.nan)
    merged_df['Engagement_Rate'] = merged_df['Engagement_Rate'].fillna(0)
    
    # Use Like Rate as primary metric (similar to comment score on Reddit)
    # Remove rows with no impressions (can't calculate rate)
    merged_df = merged_df[merged_df['impression_count'] > 0]
    merged_df = merged_df[merged_df['Like_Rate'].notna()]
    
    # Convert category columns to binary (0/1)
    for cat in ALL_CATEGORIES:
        if cat in merged_df.columns:
            merged_df[cat] = pd.to_numeric(merged_df[cat], errors='coerce').fillna(0)
            merged_df[cat] = (merged_df[cat] > 0).astype(int)
        else:
            merged_df[cat] = 0
    
    # Add city size grouping
    if 'city' in merged_df.columns:
        merged_df['City_Lower'] = merged_df['city'].str.lower()
        merged_df['City_Size'] = merged_df['City_Lower'].apply(
            lambda x: 'Large' if x in LARGE_CITIES else ('Small' if x in SMALL_CITIES else 'Unknown')
        )
    elif 'City' in merged_df.columns:
        merged_df['City_Lower'] = merged_df['City'].str.lower()
        merged_df['City_Size'] = merged_df['City_Lower'].apply(
            lambda x: 'Large' if x in LARGE_CITIES else ('Small' if x in SMALL_CITIES else 'Unknown')
        )
    
    # Add engagement quartiles
    try:
        merged_df['Like_Rate_Quartile'] = pd.qcut(
            merged_df['Like_Rate'], 
            q=4, 
            labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)'],
            duplicates='drop'
        )
    except ValueError:
        # If qcut fails (not enough unique values), use cut instead
        merged_df['Like_Rate_Quartile'] = pd.cut(
            merged_df['Like_Rate'], 
            bins=4, 
            labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)'],
            duplicates='drop'
        )
    
    print(f"  Final dataset: {len(merged_df)} posts with valid engagement metrics")
    print(f"  Like Rate range: {merged_df['Like_Rate'].min():.6f} to {merged_df['Like_Rate'].max():.6f}")
    print(f"  Like Rate mean: {merged_df['Like_Rate'].mean():.6f}, median: {merged_df['Like_Rate'].median():.6f}")
    print(f"  Total Engagement range: {merged_df['Total_Engagement'].min():.0f} to {merged_df['Total_Engagement'].max():.0f}")
    if 'City' in merged_df.columns or 'city' in merged_df.columns:
        print(f"  Cities: {merged_df['City'].nunique() if 'City' in merged_df.columns else merged_df['city'].nunique()} unique cities")
    
    return merged_df

def analyze_category_prevalence(df):
    """Analyze overall category prevalence"""
    print("\n" + "="*80)
    print("ANALYSIS 1: CATEGORY PREVALENCE")
    print("="*80)
    
    results = []
    total_posts = len(df)
    
    for category in ALL_CATEGORIES:
        category_name = category.replace('Comment_', '').replace('Critique_', '').replace('Perception_', '').replace('Response_', '').replace('Racist_Flag', 'racist')
        count = df[category].sum()
        percentage = (count / total_posts) * 100
        
        # Calculate mean like rate for this category
        category_rates = df[df[category] == 1]['Like_Rate'].values
        mean_rate = np.mean(category_rates) if len(category_rates) > 0 else 0
        median_rate = np.median(category_rates) if len(category_rates) > 0 else 0
        
        results.append({
            'Category': category_name,
            'Full_Category': category,
            'Count': count,
            'Percentage': percentage,
            'Mean_Like_Rate': mean_rate,
            'Median_Like_Rate': median_rate
        })
    
    results_df = pd.DataFrame(results).sort_values('Percentage', ascending=False)
    
    print("\nCategory Prevalence (sorted by frequency):")
    print("-" * 80)
    for _, row in results_df.iterrows():
        print(f"  {row['Category']:40s}: {row['Count']:6,} ({row['Percentage']:5.2f}%) | Mean Like Rate: {row['Mean_Like_Rate']:8.6f}")
    
    return results_df

def analyze_category_importance(df):
    """Analyze category importance: Prevalence * Mean Like Rate (what's common AND highly engaged)"""
    print("\n" + "="*80)
    print("ANALYSIS 2: CATEGORY IMPORTANCE (Prevalence × Like Rate)")
    print("="*80)
    
    total_posts = len(df)
    results = []
    
    for category in ALL_CATEGORIES:
        category_name = category.replace('Comment_', '').replace('Critique_', '').replace('Perception_', '').replace('Response_', '').replace('Racist_Flag', 'racist')
        
        # Get posts with this category
        category_mask = (df[category] == 1)
        category_count = category_mask.sum()
        category_prevalence = (category_count / total_posts) * 100
        
        # Calculate mean like rate for this category
        category_rates = df[category_mask]['Like_Rate'].values
        mean_rate = np.mean(category_rates) if len(category_rates) > 0 else 0
        median_rate = np.median(category_rates) if len(category_rates) > 0 else 0
        
        # Importance metric: Prevalence × Mean Like Rate
        # This highlights categories that are both common AND highly engaged
        importance_score = category_prevalence * mean_rate
        
        results.append({
            'Category': category_name,
            'Full_Category': category,
            'Count': category_count,
            'Prevalence_Percent': category_prevalence,
            'Mean_Like_Rate': mean_rate,
            'Median_Like_Rate': median_rate,
            'Importance_Score': importance_score  # Prevalence × Mean Like Rate
        })
    
    importance_df = pd.DataFrame(results).sort_values('Importance_Score', ascending=False)
    
    print("\nCategory Importance (Prevalence × Mean Like Rate) - What's Common AND Highly Engaged:")
    print("-" * 80)
    for _, row in importance_df.iterrows():
        print(f"  {row['Category']:40s}: Importance={row['Importance_Score']:10.6f} (Prevalence={row['Prevalence_Percent']:5.2f}% × Like Rate={row['Mean_Like_Rate']:8.6f})")
    
    return importance_df

def analyze_category_cooccurrence(df):
    """Analyze which categories co-occur together with importance metric (Prevalence × Like Rate)"""
    print("\n" + "="*80)
    print("ANALYSIS 3: CATEGORY CO-OCCURRENCE PATTERNS WITH IMPORTANCE")
    print("="*80)
    
    total_posts = len(df)
    cooccurrence_pairs = []
    
    for i, cat1 in enumerate(ALL_CATEGORIES):
        for j, cat2 in enumerate(ALL_CATEGORIES):
            if i != j:
                cat1_name = cat1.replace('Comment_', '').replace('Critique_', '').replace('Perception_', '').replace('Response_', '').replace('Racist_Flag', 'racist')
                cat2_name = cat2.replace('Comment_', '').replace('Critique_', '').replace('Perception_', '').replace('Response_', '').replace('Racist_Flag', 'racist')
                
                # Get posts with both categories
                both_mask = ((df[cat1] == 1) & (df[cat2] == 1))
                cat1_mask = (df[cat1] == 1)
                cat2_mask = (df[cat2] == 1)
                union_mask = (cat1_mask | cat2_mask)
                
                both_count = both_mask.sum()
                cat1_count = cat1_mask.sum()
                cat2_count = cat2_mask.sum()
                union_count = union_mask.sum()
                
                # Co-occurrence prevalence (percentage of all posts)
                cooccurrence_prevalence = (both_count / total_posts) * 100
                
                # Mean like rate when both categories appear
                both_rates = df[both_mask]['Like_Rate'].values
                mean_rate_both = np.mean(both_rates) if len(both_rates) > 0 else 0
                
                # IMPORTANCE METRIC: Co-occurrence Prevalence × Mean Like Rate
                cooccurrence_importance = cooccurrence_prevalence * mean_rate_both
                
                # Unweighted Jaccard similarity
                jaccard_unweighted = both_count / union_count if union_count > 0 else 0
                
                # Weighted Jaccard (using like rates as weights)
                min_rate = df['Like_Rate'].min()
                rate_offset = abs(min_rate) + 1e-8 if min_rate < 0 else 1e-8
                df['Rate_Weight'] = df['Like_Rate'] + rate_offset
                
                both_weighted = df[both_mask]['Rate_Weight'].sum()
                union_weighted = df[union_mask]['Rate_Weight'].sum()
                jaccard_weighted = both_weighted / union_weighted if union_weighted > 0 else 0
                
                # Additional statistics
                only_cat1_rates = df[cat1_mask & ~cat2_mask]['Like_Rate'].values
                only_cat2_rates = df[cat2_mask & ~cat1_mask]['Like_Rate'].values
                
                cooccurrence_pairs.append({
                    'Category_1': cat1_name,
                    'Category_2': cat2_name,
                    'Full_Category_1': cat1,
                    'Full_Category_2': cat2,
                    'Both_Count': both_count,
                    'Cooccurrence_Prevalence_Percent': cooccurrence_prevalence,
                    'Mean_Like_Rate_When_Both': mean_rate_both,
                    'Importance_Score': cooccurrence_importance,  # Prevalence × Mean Like Rate
                    'Category_1_Count': cat1_count,
                    'Category_2_Count': cat2_count,
                    'Union_Count': union_count,
                    'Jaccard_Unweighted': jaccard_unweighted,
                    'Jaccard_Weighted': jaccard_weighted,
                    'Jaccard_Difference': jaccard_weighted - jaccard_unweighted,
                    'Mean_Like_Rate_Only_Cat1': np.mean(only_cat1_rates) if len(only_cat1_rates) > 0 else 0,
                    'Mean_Like_Rate_Only_Cat2': np.mean(only_cat2_rates) if len(only_cat2_rates) > 0 else 0,
                })
    
    cooccurrence_df = pd.DataFrame(cooccurrence_pairs)
    
    # Find top pairs by importance (prevalence × like rate)
    top_importance = cooccurrence_df.nlargest(20, 'Importance_Score')
    
    print("\nTop 20 Category Pairs by Importance (Co-occurrence Prevalence × Mean Like Rate):")
    print("-" * 80)
    for _, row in top_importance.iterrows():
        print(f"  {row['Category_1']:35s} <-> {row['Category_2']:35s}: Importance={row['Importance_Score']:10.6f} (Prevalence={row['Cooccurrence_Prevalence_Percent']:5.2f}% × Rate={row['Mean_Like_Rate_When_Both']:8.6f}, N={row['Both_Count']:6,})")
    
    # Also show top by Jaccard for comparison
    top_jaccard = cooccurrence_df.nlargest(10, 'Jaccard_Unweighted')
    
    print("\nTop 10 Category Pairs by Jaccard Similarity (for comparison):")
    print("-" * 80)
    for _, row in top_jaccard.iterrows():
        print(f"  {row['Category_1']:35s} <-> {row['Category_2']:35s}: Jaccard={row['Jaccard_Unweighted']:.4f}, Importance={row['Importance_Score']:10.6f}")
    
    return cooccurrence_df

def analyze_by_engagement_quartiles(df):
    """Analyze category prevalence by like rate quartiles"""
    print("\n" + "="*80)
    print("ANALYSIS 4: CATEGORY PREVALENCE BY LIKE RATE QUARTILES")
    print("="*80)
    
    results = []
    
    for quartile in ['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)']:
        quartile_df = df[df['Like_Rate_Quartile'] == quartile]
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
    if 'Q4 (Highest)' in pivot_df.columns:
        pivot_df = pivot_df.sort_values('Q4 (Highest)', ascending=False)
    
    print("\nCategory Prevalence by Like Rate Quartile (%):")
    print("-" * 80)
    print(pivot_df.to_string())
    
    return results_df, pivot_df

def analyze_by_city_size(df):
    """Analyze category prevalence by city size"""
    print("\n" + "="*80)
    print("ANALYSIS 5: CATEGORY PREVALENCE BY CITY SIZE")
    print("="*80)
    
    if 'City_Size' not in df.columns or df['City_Size'].value_counts().get('Unknown', 0) == len(df):
        print("  City size data not available, skipping this analysis")
        return pd.DataFrame(), pd.DataFrame()
    
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
    
    if len(results_df) > 0:
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
    else:
        pivot_df = pd.DataFrame()
    
    return results_df, pivot_df

def analyze_like_rate_differences(df):
    """Analyze like rate differences by category (similar to score analysis)"""
    print("\n" + "="*80)
    print("ANALYSIS 6: LIKE RATE DIFFERENCES BY CATEGORY")
    print("="*80)
    
    results = []
    all_rates = df['Like_Rate'].values
    
    for category in ALL_CATEGORIES:
        # Get rates for posts with this category
        category_rates = df[df[category] == 1]['Like_Rate'].values
        other_rates = df[df[category] == 0]['Like_Rate'].values
        
        n_category = len(category_rates)
        n_other = len(other_rates)
        
        if n_category == 0:
            continue
        
        # Calculate statistics
        mean_category = np.mean(category_rates)
        mean_other = np.mean(other_rates)
        median_category = np.median(category_rates)
        median_other = np.median(other_rates)
        
        # Perform statistical test
        # Use Mann-Whitney U test (non-parametric) since rates may not be normally distributed
        statistic, p_value = stats.mannwhitneyu(
            category_rates, 
            other_rates, 
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
            (np.var(category_rates, ddof=1) + np.var(other_rates, ddof=1)) / 2
        )
        if pooled_std > 0:
            cohens_d = (mean_category - mean_other) / pooled_std
        else:
            cohens_d = 0.0
        
        category_name = category.replace('Comment_', '').replace('Critique_', '').replace('Perception_', '').replace('Response_', '').replace('Racist_Flag', 'racist')
        
        results.append({
            'Category': category_name,
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
        print(f"\n{category_name}:")
        print(f"  N (category): {n_category:,}, N (other): {n_other:,}")
        print(f"  Mean (category): {mean_category:.8f}, Mean (other): {mean_other:.8f}")
        print(f"  Difference: {mean_category - mean_other:+.8f}")
        print(f"  P-value: {p_value:.6f}, P-value (Bonferroni corrected): {p_value_corrected:.6f}")
        print(f"  Significant: {'YES' if is_significant else 'NO'} ({direction})")
        print(f"  Effect size (Cohen's d): {cohens_d:+.3f}")
    
    results_df = pd.DataFrame(results)
    return results_df

def create_visualizations(df, prevalence_df, importance_df, cooccurrence_df, quartile_df, city_size_df, like_rate_diff_df):
    """Create visualizations for key findings"""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    output_dir = Path('output/charts/twitter')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 0. Prevalence and Mean Like Rate Comparison (Before combined importance chart)
    overall_mean_rate = df['Like_Rate'].mean()
    overall_prevalence = 100.0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Sort by prevalence percentage (left chart)
    importance_sorted = importance_df.sort_values('Prevalence_Percent', ascending=False).copy()
    
    categories = importance_sorted['Category'].values
    prevalences = importance_sorted['Prevalence_Percent'].values
    mean_rates = importance_sorted['Mean_Like_Rate'].values
    
    # Calculate standard errors
    se_prevalences = []
    se_mean_rates = []
    for idx, row in importance_sorted.iterrows():
        category = row['Full_Category']
        category_mask = (df[category] == 1)
        category_rates = df[category_mask]['Like_Rate'].values
        total_posts = len(df)
        category_count = category_mask.sum()
        
        # SE for prevalence
        prevalence = (category_count / total_posts) * 100
        se_prevalence = np.sqrt((prevalence / 100) * (1 - prevalence / 100) / total_posts) * 100
        se_prevalences.append(se_prevalence)
        
        # SE for mean rate
        if len(category_rates) > 1:
            se_mean_rate = stats.sem(category_rates)
        else:
            se_mean_rate = 0
        se_mean_rates.append(se_mean_rate)
    
    # Plot 1: Prevalence
    colors_prevalence = ['#3498db' if p > overall_prevalence else '#e74c3c' for p in prevalences]
    bars1 = ax1.bar(range(len(categories)), prevalences, yerr=se_prevalences,
                    color=colors_prevalence, alpha=0.7, edgecolor='black', linewidth=0.5,
                    capsize=3, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
    ax1.axhline(y=overall_prevalence, color='black', linestyle='--', linewidth=2, label='Overall Average (100%)')
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Prevalence (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Category Prevalence on Twitter/X\n(All 16 Categories)', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_ylim([0, max(prevalences) * 1.15])
    
    # Plot 2: Mean Like Rate
    colors_rate = ['#2ecc71' if s > overall_mean_rate else '#e74c3c' for s in mean_rates]
    bars2 = ax2.bar(range(len(categories)), mean_rates, yerr=se_mean_rates,
                    color=colors_rate, alpha=0.7, edgecolor='black', linewidth=0.5,
                    capsize=3, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
    ax2.axhline(y=overall_mean_rate, color='black', linestyle='--', linewidth=2, 
                label=f'Overall Average ({overall_mean_rate:.6f})')
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Mean Like Rate (Likes / Impressions)', fontsize=12, fontweight='bold')
    ax2.set_title('Mean Like Rate by Category\n(All 16 Categories)', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_ylim([0, max(mean_rates) * 1.15])
    
    plt.suptitle('Category Analysis: Prevalence and Mean Like Rate (Before Combined Importance)', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'twitter_category_prevalence_and_rate_separate.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'twitter_category_prevalence_and_rate_separate.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved separate prevalence and like rate charts")
    
    # 1. Category Importance Bar Chart (All 16 categories with error bars and Bonferroni correction)
    importance_with_errors = []
    for _, row in importance_df.iterrows():
        category = row['Full_Category']
        category_mask = (df[category] == 1)
        category_rates = df[category_mask]['Like_Rate'].values
        total_posts = len(df)
        category_count = category_mask.sum()
        
        # Calculate standard error for mean rate
        if len(category_rates) > 1:
            se_mean_rate = stats.sem(category_rates)
        else:
            se_mean_rate = 0
        
        # Calculate standard error for prevalence
        prevalence = (category_count / total_posts) * 100
        se_prevalence = np.sqrt((prevalence / 100) * (1 - prevalence / 100) / total_posts) * 100
        
        # Approximate standard error for importance
        mean_rate = row['Mean_Like_Rate']
        se_importance = abs(prevalence) * se_mean_rate + abs(mean_rate) * se_prevalence
        
        importance_with_errors.append({
            'Category': row['Category'],
            'Importance_Score': row['Importance_Score'],
            'SE_Importance': se_importance,
            'Mean_Like_Rate': mean_rate,
            'SE_Mean_Rate': se_mean_rate,
            'Prevalence': prevalence,
            'SE_Prevalence': se_prevalence
        })
    
    importance_errors_df = pd.DataFrame(importance_with_errors).sort_values('Importance_Score', ascending=False)
    
    # Statistical tests with Bonferroni correction
    significance_results = []
    for idx, row in importance_errors_df.iterrows():
        category_name = row['Category']
        full_category_name = importance_df[importance_df['Category'] == category_name]['Full_Category'].iloc[0]
        category_mask = (df[full_category_name] == 1)
        category_rates = df[category_mask]['Like_Rate'].values
        other_rates = df[~category_mask]['Like_Rate'].values
        
        if len(category_rates) > 0 and len(other_rates) > 0:
            statistic, p_value = stats.mannwhitneyu(category_rates, other_rates, alternative='two-sided')
            p_value_corrected = min(p_value * 16, 1.0)
            is_significant = p_value_corrected < 0.05
        else:
            p_value_corrected = 1.0
            is_significant = False
        
        significance_results.append({
            'Category': category_name,
            'P_Value': p_value if len(category_rates) > 0 and len(other_rates) > 0 else 1.0,
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
    
    colors = ['#2ecc71' if sig else '#95a5a6' for sig in is_significant]
    
    bars = ax.bar(range(len(categories)), importance_scores, yerr=errors, 
                  color=colors, alpha=0.7, edgecolor='black', linewidth=0.5,
                  capsize=5, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
    
    # Add significance markers
    for i, (sig, score, error) in enumerate(zip(is_significant, importance_scores, errors)):
        if sig:
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
    ax.set_ylabel('Importance Score\n(Prevalence × Mean Like Rate)', fontsize=12, fontweight='bold')
    ax.set_title('Category Importance on Twitter/X (All 16 Categories)\n' + 
                'Error bars = Standard Error, * p<0.05, ** p<0.01, *** p<0.001 (Bonferroni corrected)',
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.7, label='Significantly Different (Bonferroni corrected)'),
        Patch(facecolor='#95a5a6', alpha=0.7, label='Not Significant')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'twitter_category_importance_all16_with_errors.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'twitter_category_importance_all16_with_errors.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved category importance chart (all 16 with error bars and Bonferroni correction)")
    
    # 2. Like Rate Differences Chart (similar to score analysis)
    if len(like_rate_diff_df) > 0:
        like_rate_diff_sorted = like_rate_diff_df.sort_values('Difference', ascending=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot 1: Bar plot of mean differences
        colors = ['red' if not sig else ('green' if diff > 0 else 'blue') 
                  for sig, diff in zip(like_rate_diff_sorted['Significant'], like_rate_diff_sorted['Difference'])]
        
        bars = ax1.barh(range(len(like_rate_diff_sorted)), like_rate_diff_sorted['Difference'], color=colors, alpha=0.7)
        
        # Add significance markers
        for i, (sig, diff) in enumerate(zip(like_rate_diff_sorted['Significant'], like_rate_diff_sorted['Difference'])):
            if sig:
                marker = '***' if like_rate_diff_sorted.iloc[i]['P_Value_Corrected'] < 0.001 else '**' if like_rate_diff_sorted.iloc[i]['P_Value_Corrected'] < 0.01 else '*'
                ax1.text(diff + (abs(diff) * 0.1 if diff != 0 else 0.00001), i, marker, 
                       ha='left' if diff > 0 else 'right', va='center', fontsize=12, fontweight='bold')
        
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax1.set_yticks(range(len(like_rate_diff_sorted)))
        ax1.set_yticklabels(like_rate_diff_sorted['Category'], fontsize=10)
        ax1.set_xlabel('Mean Like Rate Difference (Category - Other)', fontsize=12, fontweight='bold')
        ax1.set_title('Like Rate Differences by Category\n(Bonferroni Corrected, * p<0.05, ** p<0.01, *** p<0.001)', 
                      fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Significantly Higher'),
            Patch(facecolor='blue', alpha=0.7, label='Significantly Lower'),
            Patch(facecolor='red', alpha=0.7, label='Not Significant')
        ]
        ax1.legend(handles=legend_elements, loc='lower right')
        
        # Plot 2: Box plot for top/bottom categories
        top_categories = like_rate_diff_sorted.nlargest(4, 'Difference')
        bottom_categories = like_rate_diff_sorted.nsmallest(4, 'Difference')
        selected_categories = pd.concat([top_categories, bottom_categories])
        
        box_data = []
        box_labels = []
        for _, row in selected_categories.iterrows():
            category = row['Full_Category']
            category_rates = df[df[category] == 1]['Like_Rate'].values
            other_rates = df[df[category] == 0]['Like_Rate'].values
            
            box_data.append(category_rates)
            box_labels.append(f"{row['Category']}\n(category, n={len(category_rates)})")
            box_data.append(other_rates)
            box_labels.append(f"{row['Category']}\n(other, n={len(other_rates)})")
        
        bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True, 
                         showmeans=True, meanline=True)
        
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor('lightgreen' if i % 2 == 0 else 'lightgray')
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Like Rate (Likes / Impressions)', fontsize=12, fontweight='bold')
        ax2.set_title('Like Rate Distribution: Top 4 and Bottom 4 Categories\n(Green = Category, Gray = Other)', 
                      fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45, labelsize=9)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'twitter_category_like_rate_differences.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'twitter_category_like_rate_differences.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved like rate differences chart")
    
    # 3. Co-occurrence Importance Heatmap
    if len(cooccurrence_df) > 0:
        top_cats = importance_df.nlargest(10, 'Importance_Score')['Full_Category'].tolist()
        cooccur_subset = cooccurrence_df[
            cooccurrence_df['Full_Category_1'].isin(top_cats) & 
            cooccurrence_df['Full_Category_2'].isin(top_cats)
        ]
        
        if len(cooccur_subset) > 0:
            pivot_importance = cooccur_subset.pivot(
                index='Category_1', 
                columns='Category_2', 
                values='Importance_Score'
            )
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(pivot_importance, annot=True, fmt='.6f', cmap='YlOrRd', ax=ax, 
                       cbar_kws={'label': 'Importance (Prevalence × Mean Like Rate)'})
            ax.set_title('Category Co-occurrence Importance on Twitter/X\n(Prevalence × Mean Like Rate when co-occurring)\nTop 10 Most Important Categories', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / 'twitter_category_cooccurrence_importance_heatmap.pdf', dpi=300, bbox_inches='tight')
            plt.savefig(output_dir / 'twitter_category_cooccurrence_importance_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✓ Saved co-occurrence importance heatmap")
    
    print(f"\nAll visualizations saved to: {output_dir}/")
    
    return importance_errors_df

def save_all_results(prevalence_df, importance_df, cooccurrence_df, quartile_df, quartile_pivot, 
                    city_size_df, city_size_pivot, like_rate_diff_df, importance_errors_df=None):
    """Save all results to CSV files"""
    print("\n" + "="*80)
    print("SAVING RESULTS TO CSV")
    print("="*80)
    
    output_dir = Path('output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files_saved = []
    
    prevalence_file = output_dir / 'twitter_category_pattern_analysis_prevalence.csv'
    prevalence_df.to_csv(prevalence_file, index=False)
    files_saved.append(prevalence_file)
    print(f"  ✓ Saved category prevalence: {prevalence_file}")
    
    importance_file = output_dir / 'twitter_category_pattern_analysis_importance.csv'
    importance_df.to_csv(importance_file, index=False)
    files_saved.append(importance_file)
    print(f"  ✓ Saved category importance (Prevalence × Like Rate): {importance_file}")
    
    if importance_errors_df is not None:
        importance_errors_file = output_dir / 'twitter_category_pattern_analysis_importance_with_errors.csv'
        importance_errors_df.to_csv(importance_errors_file, index=False)
        files_saved.append(importance_errors_file)
        print(f"  ✓ Saved category importance with errors and significance: {importance_errors_file}")
    
    cooccurrence_file = output_dir / 'twitter_category_pattern_analysis_cooccurrence.csv'
    cooccurrence_df.to_csv(cooccurrence_file, index=False)
    files_saved.append(cooccurrence_file)
    print(f"  ✓ Saved co-occurrence patterns with importance: {cooccurrence_file}")
    
    quartile_file = output_dir / 'twitter_category_pattern_analysis_by_quartile.csv'
    quartile_df.to_csv(quartile_file, index=False)
    files_saved.append(quartile_file)
    print(f"  ✓ Saved quartile analysis: {quartile_file}")
    
    if len(quartile_pivot) > 0:
        quartile_pivot_file = output_dir / 'twitter_category_pattern_analysis_by_quartile_pivot.csv'
        quartile_pivot.to_csv(quartile_pivot_file)
        files_saved.append(quartile_pivot_file)
        print(f"  ✓ Saved quartile pivot table: {quartile_pivot_file}")
    
    if len(city_size_df) > 0:
        city_size_file = output_dir / 'twitter_category_pattern_analysis_by_city_size.csv'
        city_size_df.to_csv(city_size_file, index=False)
        files_saved.append(city_size_file)
        print(f"  ✓ Saved city size analysis: {city_size_file}")
        
        if len(city_size_pivot) > 0:
            city_size_pivot_file = output_dir / 'twitter_category_pattern_analysis_by_city_size_pivot.csv'
            city_size_pivot.to_csv(city_size_pivot_file)
            files_saved.append(city_size_pivot_file)
            print(f"  ✓ Saved city size pivot table: {city_size_pivot_file}")
    
    like_rate_diff_file = output_dir / 'twitter_category_like_rate_analysis_results.csv'
    like_rate_diff_df.to_csv(like_rate_diff_file, index=False)
    files_saved.append(like_rate_diff_file)
    print(f"  ✓ Saved like rate differences analysis: {like_rate_diff_file}")
    
    print(f"\n✓ All results saved! Total files: {len(files_saved)}")
    return files_saved

def main():
    print("="*80)
    print("CATEGORY PATTERN ANALYSIS FOR TWITTER/X")
    print("Comprehensive research analysis using engagement metrics (Like Rate)")
    print("="*80)
    
    # Load data
    df = load_and_prepare_data()
    
    # Perform analyses
    prevalence_df = analyze_category_prevalence(df)
    importance_df = analyze_category_importance(df)
    cooccurrence_df = analyze_category_cooccurrence(df)
    quartile_df, quartile_pivot = analyze_by_engagement_quartiles(df)
    city_size_df, city_size_pivot = analyze_by_city_size(df)
    like_rate_diff_df = analyze_like_rate_differences(df)
    
    # Create visualizations
    importance_errors_df = create_visualizations(df, prevalence_df, importance_df, cooccurrence_df, 
                                                 quartile_df, city_size_df, like_rate_diff_df)
    
    # Save all results
    save_all_results(prevalence_df, importance_df, cooccurrence_df, quartile_df, quartile_pivot,
                    city_size_df, city_size_pivot, like_rate_diff_df, importance_errors_df)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total posts analyzed: {len(df):,}")
    print(f"Total categories: {len(ALL_CATEGORIES)}")
    print(f"Most common category: {prevalence_df.iloc[0]['Category']} ({prevalence_df.iloc[0]['Percentage']:.2f}%)")
    print(f"Least common category: {prevalence_df.iloc[-1]['Category']} ({prevalence_df.iloc[-1]['Percentage']:.2f}%)")
    print(f"Most important category (Prevalence × Like Rate): {importance_df.iloc[0]['Category']} (Importance={importance_df.iloc[0]['Importance_Score']:.6f})")
    
    if len(cooccurrence_df) > 0:
        top_importance_pair = cooccurrence_df.nlargest(1, 'Importance_Score').iloc[0]
        print(f"Most important co-occurrence (Prevalence × Like Rate): {top_importance_pair['Category_1']} <-> {top_importance_pair['Category_2']} (Importance={top_importance_pair['Importance_Score']:.6f})")
    
    if len(like_rate_diff_df) > 0:
        n_significant = like_rate_diff_df['Significant'].sum()
        n_higher = ((like_rate_diff_df['Significant']) & (like_rate_diff_df['Direction'] == 'HIGHER')).sum()
        n_lower = ((like_rate_diff_df['Significant']) & (like_rate_diff_df['Direction'] == 'LOWER')).sum()
        print(f"Significantly different like rates (Bonferroni corrected): {n_significant}")
        print(f"  - Significantly HIGHER rates: {n_higher}")
        print(f"  - Significantly LOWER rates: {n_lower}")
    
    print("\n" + "="*80)
    print("Analysis complete! All results saved to CSV files for future reference.")
    print("="*80)

if __name__ == '__main__':
    main()
