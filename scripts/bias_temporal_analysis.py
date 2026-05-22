#!/usr/bin/env python3
"""
Bias Temporal Analysis: Track bias categories over time by source and city groups

This script:
1. Loads data from all four sources (Twitter/X, Reddit, News, Meeting Minutes)
2. Merges with category annotations
3. Groups bias categories together (NIMBY, harmful generalization, deserving/undeserving, racist)
4. Calculates average bias by year and 6-month periods
5. Analyzes by city groups (large vs small cities)
6. Creates visualizations

Usage:
    python scripts/bias_temporal_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
from scipy import stats
from itertools import combinations
warnings.filterwarnings('ignore')

# Set plotting style
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('ggplot')
sns.set_palette("husl")

# ============================================================================
# CONSTANTS
# ============================================================================

# Define all 16 categories
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

# Bias categories to group together
BIAS_CATEGORIES = [
    'Comment_ask a rhetorical question',
    'Perception_not in my backyard',
    'Perception_harmful generalization',
    'Perception_deserving/undeserving',
    'Racist_Flag'
]

# Indicator categories (non-bias)
INDICATOR_CATEGORIES = [
    'Comment_provide a fact or claim',
    'Comment_provide an observation',
    'Comment_express their opinion',
    'Comment_express others opinions'
]

# City size groupings
LARGE_CITIES = ['san francisco', 'portland', 'buffalo', 'baltimore', 'el paso']
SMALL_CITIES = ['kalamazoo', 'south bend', 'rockford', 'scranton', 'fayetteville']


def _title_case_city(city: str) -> str:
    return ' '.join(word.capitalize() for word in city.split())


LARGE_CITY_CLUSTER_CITIES = ', '.join(_title_case_city(c) for c in LARGE_CITIES)
SMALL_CITY_CLUSTER_CITIES = ', '.join(_title_case_city(c) for c in SMALL_CITIES)
LARGE_CITY_CLUSTER_LEGEND = f'Large City Cluster ({LARGE_CITY_CLUSTER_CITIES})'
SMALL_CITY_CLUSTER_LEGEND = f'Small City Cluster ({SMALL_CITY_CLUSTER_CITIES})'
CITY_CLUSTER_PANEL_TITLES = {
    'Large': f'Large City Cluster\n({LARGE_CITY_CLUSTER_CITIES})',
    'Small': f'Small City Cluster\n({SMALL_CITY_CLUSTER_CITIES})',
}

# Y-axis label for bias_score (= count of active Negative Bias Frame labels, 0-5)
BIAS_SCORE_YLABEL = "Mean Negative Bias Frame Score (0-5)"

# Source mappings
SOURCE_FILES = {
    'twitter': {
        'data': 'complete_dataset/all_twitter_posts_merged_with_details.csv',
        'categories': 'output/x/gpt4/classified_comments_x_all_gpt4_x_flags.csv',
        'date_col': 'created_at',
        'text_col': 'Deidentified_text',
        'text_col_cat': 'Comment'
    },
    'reddit': {
        'data': 'complete_dataset/all_reddit_comments.csv',
        'categories': 'output/reddit/gpt4/classified_comments_reddit_all_gpt4_reddit_flags.csv',
        'date_col': 'Comment Timestamp',
        'text_col': 'Deidentified_Comment',
        'text_col_cat': 'Comment'
    },
    'news': {
        'data': 'complete_dataset/all_newspaper_articles.csv',
        'categories': 'output/news/gpt4/classified_comments_news_all_gpt4_news_flags.csv',
        'date_col': 'article_date',
        'text_col': 'Deidentified_paragraph_text',
        'text_col_cat': 'Comment'
    },
    'meeting_minutes': {
        'data': 'complete_dataset/all_meeting_minutes.csv',
        'categories': 'output/meeting_minutes/gpt4/classified_comments_meeting_minutes_all_gpt4_meeting_minutes_flags.csv',
        'date_col': 'date',
        'text_col': 'Deidentified_paragraph',
        'text_col_cat': 'Comment'
    }
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_source_data(source_name, source_config, filter_zero_category_rows=True):
    """Load data for a single source and merge with categories"""
    print(f"\n{'='*80}")
    print(f"Loading {source_name.upper()} data")
    print(f"{'='*80}")
    
    # Load main data file
    data_file = source_config['data']
    print(f"Loading data from {data_file}...")
    if not Path(data_file).exists():
        print(f"  WARNING: File not found: {data_file}")
        return None
    
    df_data = pd.read_csv(data_file, low_memory=False)
    print(f"  Loaded {len(df_data)} rows")
    
    # Load category annotations
    cat_file = source_config['categories']
    print(f"Loading categories from {cat_file}...")
    if not Path(cat_file).exists():
        print(f"  WARNING: File not found: {cat_file}")
        return None
    
    df_cats = pd.read_csv(cat_file, low_memory=False)
    print(f"  Loaded {len(df_cats)} category annotations")
    
    # Prepare merge keys
    text_col_data = source_config['text_col']
    text_col_cat = source_config['text_col_cat']
    
    # Normalize text for merging
    df_data['text_normalized'] = df_data[text_col_data].astype(str).str.strip()
    df_cats['text_normalized'] = df_cats[text_col_cat].astype(str).str.strip()
    
    # Merge on normalized text and city
    print("Merging data with categories...")
    if 'city' in df_data.columns and 'City' in df_cats.columns:
        df_data['city_normalized'] = df_data['city'].astype(str).str.lower().str.strip()
        df_cats['city_normalized'] = df_cats['City'].astype(str).str.lower().str.strip()
        merged = df_data.merge(
            df_cats[['text_normalized', 'city_normalized'] + ALL_CATEGORIES],
            on=['text_normalized', 'city_normalized'],
            how='inner',
            suffixes=('', '_cat')
        )
    else:
        merged = df_data.merge(
            df_cats[['text_normalized'] + ALL_CATEGORIES],
            on='text_normalized',
            how='inner',
            suffixes=('', '_cat')
        )
    
    print(f"  Merged dataset: {len(merged)} rows")
    
    # Convert category columns to binary
    for cat in ALL_CATEGORIES:
        if cat in merged.columns:
            merged[cat] = pd.to_numeric(merged[cat], errors='coerce').fillna(0)
            merged[cat] = (merged[cat] > 0).astype(int)
        else:
            merged[cat] = 0
    
    # Parse date
    date_col = source_config['date_col']
    if date_col in merged.columns:
        print(f"Parsing dates from {date_col}...")
        
        # Special handling for meeting minutes date format (MM_DD_YYYY)
        if source_name == 'meeting_minutes':
            # Convert MM_DD_YYYY format to YYYY-MM-DD
            merged['date_parsed'] = merged[date_col].apply(
                lambda x: pd.to_datetime(x.replace('_', '/'), format='%m/%d/%Y', errors='coerce', utc=True) 
                if pd.notna(x) and isinstance(x, str) and '_' in str(x) 
                else pd.to_datetime(x, errors='coerce', utc=True)
            )
        else:
            merged['date_parsed'] = pd.to_datetime(merged[date_col], errors='coerce', utc=True)
        
        merged = merged.dropna(subset=['date_parsed'])
        print(f"  {len(merged)} rows with valid dates")
    else:
        print(f"  WARNING: Date column '{date_col}' not found")
        return None
    
    # Add city size grouping
    if 'city' in merged.columns:
        merged['city_lower'] = merged['city'].astype(str).str.lower().str.strip()
        merged['city_size'] = merged['city_lower'].apply(
            lambda x: 'Large' if x in LARGE_CITIES else ('Small' if x in SMALL_CITIES else 'Unknown')
        )
    elif 'City' in merged.columns:
        merged['city_lower'] = merged['City'].astype(str).str.lower().str.strip()
        merged['city_size'] = merged['city_lower'].apply(
            lambda x: 'Large' if x in LARGE_CITIES else ('Small' if x in SMALL_CITIES else 'Unknown')
        )
    else:
        merged['city_size'] = 'Unknown'
    
    # Add source name
    merged['source'] = source_name
    
    # Calculate bias score (sum of bias categories)
    merged['bias_score'] = merged[BIAS_CATEGORIES].sum(axis=1)
    
    # Filter to only rows with at least one category
    if filter_zero_category_rows:
        merged = merged[merged[ALL_CATEGORIES].sum(axis=1) > 0]
    
    print(f"  Final dataset: {len(merged)} rows")
    print(f"  Date range: {merged['date_parsed'].min()} to {merged['date_parsed'].max()}")
    print(f"  Bias score range: {merged['bias_score'].min()} to {merged['bias_score'].max()}")
    print(f"  Average bias score: {merged['bias_score'].mean():.3f}")
    
    return merged

def load_all_data(filter_zero_category_rows=True):
    """Load data from all sources"""
    all_data = []
    
    for source_name, source_config in SOURCE_FILES.items():
        df = load_source_data(
            source_name,
            source_config,
            filter_zero_category_rows=filter_zero_category_rows
        )
        if df is not None:
            all_data.append(df)
    
    if not all_data:
        raise ValueError("No data loaded from any source!")
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n{'='*80}")
    print(f"COMBINED DATASET")
    print(f"{'='*80}")
    print(f"Total rows: {len(combined)}")
    print(f"Sources: {combined['source'].unique()}")
    print(f"Date range: {combined['date_parsed'].min()} to {combined['date_parsed'].max()}")
    
    return combined

# ============================================================================
# TEMPORAL ANALYSIS
# ============================================================================

def create_time_periods(df):
    """Create year and 6-month period columns"""
    df = df.copy()
    
    # Year
    df['year'] = df['date_parsed'].dt.year
    
    # 6-month periods (H1: Jan-Jun, H2: Jul-Dec)
    df['half_year'] = df['date_parsed'].dt.month.apply(lambda x: 'H1' if x <= 6 else 'H2')
    df['year_half'] = df['year'].astype(str) + '-' + df['half_year']
    
    # For easier sorting, create a period number
    df['period_num'] = (df['year'] - df['year'].min()) * 2 + (df['half_year'] == 'H2').astype(int)
    
    return df

def _sem(series):
    """Standard error of the mean for post-level scores in a bin."""
    n = len(series)
    if n <= 1:
        return 0.0
    return float(series.std(ddof=1) / np.sqrt(n))


def _aggregate_bias_by_group(df, group_cols):
    """Mean bias per group with SEM for error bars (post-level scores in each bin)."""
    results = (
        df.groupby(group_cols)['bias_score']
        .agg(
            mean_bias='mean',
            sem=_sem,
            count='count',
        )
        .reset_index()
    )
    return results


def calculate_bias_by_period(df, period_col='year'):
    """Mean bias by time period and source (one series per source)."""
    return _aggregate_bias_by_group(df, [period_col, 'source'])


def calculate_bias_by_period_city(df, period_col='year'):
    """Mean bias by time period, source, and city size."""
    results = _aggregate_bias_by_group(df, [period_col, 'source', 'city_size'])
    return results[results['city_size'] != 'Unknown']


def _plot_mean_sem(ax, x, period_data, style):
    """Line/scatter of mean with SEM error bars (uncertainty of the mean, not post-level SD)."""
    y = period_data['mean_bias']
    yerr = period_data['sem']
    marker_size = 10 if len(period_data) == 1 else 6
    if len(period_data) == 1:
        ax.scatter(
            x, y,
            marker=style['marker'], color=style['color'], label=style['name'],
            s=150, zorder=5, edgecolors='black', linewidths=1.5,
        )
    else:
        ax.plot(
            x, y,
            marker=style['marker'], color=style['color'], label=style['name'],
            linewidth=2, markersize=marker_size, linestyle='-',
        )
    ax.errorbar(
        x, y, yerr=yerr, fmt='none',
        color=style['color'], alpha=0.4, capsize=3, zorder=4,
    )

# ============================================================================
# VISUALIZATIONS
# ============================================================================

# Define distinct colors and markers for each source
SOURCE_STYLE = {
    'twitter': {'color': '#1DA1F2', 'marker': 'o', 'name': 'X(Twitter)'},  # Twitter blue, circle
    'reddit': {'color': '#FF4500', 'marker': '^', 'name': 'Reddit'},    # Reddit orange, triangle
    'news': {'color': '#2E7D32', 'marker': 's', 'name': 'News'},         # Green, square
    'meeting_minutes': {'color': '#9C27B0', 'marker': 'D', 'name': 'Meeting Minutes'}  # Purple, diamond
}


def _set_fig_title_subtitle(fig, title, subtitle=None, top=None):
    """Title and subtitle on separate lines above axes; tight vertical spacing."""
    title_lines = title.count('\n') + 1
    y_title = 0.97
    line_h = 0.026  # figure-fraction per title line (fontsize 14)
    y_after_title = y_title - title_lines * line_h

    fig.text(
        0.5, y_title, title,
        transform=fig.transFigure,
        ha='center', va='top', fontsize=14, fontweight='bold', linespacing=1.15,
    )
    if subtitle:
        y_sub = y_after_title - 0.008
        fig.text(
            0.5, y_sub, subtitle,
            transform=fig.transFigure,
            ha='center', va='top', fontsize=9, style='italic', color='#444444',
        )
        if top is None:
            top = y_sub - 0.032
    elif top is None:
        top = y_after_title - 0.015

    fig.subplots_adjust(top=top)


def _save_figure(fig, output_path, right=0.96, bottom=0.14, left=0.09, top=None):
    """Save without bbox_inches='tight' so figure titles are not clipped or overlapped."""
    output_path = Path(output_path)
    adjust = {'bottom': bottom, 'right': right, 'left': left}
    if top is not None:
        adjust['top'] = top
    fig.subplots_adjust(**adjust)
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix('.png'), dpi=300)


def plot_bias_by_year(df, output_dir='output'):
    """Plot mean Negative Bias Frame score by year (four source-specific series)."""
    year_data = calculate_bias_by_period(df, period_col='year')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for source in year_data['source'].unique():
        source_data = year_data[year_data['source'] == source].sort_values('year')
        style = SOURCE_STYLE.get(source, {'color': 'gray', 'marker': 'o', 'name': source.capitalize()})
        _plot_mean_sem(ax, source_data['year'], source_data, style)
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel(BIAS_SCORE_YLABEL, fontsize=12, fontweight='bold')
    _set_fig_title_subtitle(
        fig,
        'Mean Negative Bias Frame Score (0-5) by Year Across Four Sources',
        'Error bars: SEM (std / sqrt(n)) of post-level scores within each year x source bin',
    )
    ax.legend(title='Source', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    output_path = Path(output_dir) / 'bias_by_year_all_sources.pdf'
    _save_figure(fig, output_path)
    print(f"Saved: {output_path}")
    plt.close()

def plot_bias_by_half_year(df, output_dir='output'):
    """Plot mean Negative Bias Frame score by 6-month period (four source-specific series)."""
    half_year_data = calculate_bias_by_period(df, period_col='year_half')
    
    # Sort by period number for proper ordering
    df_with_period = df.copy()
    df_with_period = create_time_periods(df_with_period)
    period_order = df_with_period.groupby('year_half')['period_num'].first().sort_values().index.tolist()
    
    # Create mapping from period to x-axis position
    period_to_pos = {period: idx for idx, period in enumerate(period_order)}
    
    fig, ax = plt.subplots(figsize=(14, 5.2))
    
    for source in half_year_data['source'].unique():
        source_data = half_year_data[half_year_data['source'] == source].copy()
        
        # Map periods to x-axis positions
        source_data['x_pos'] = source_data['year_half'].map(period_to_pos)
        source_data = source_data.dropna(subset=['x_pos']).sort_values('x_pos')
        
        if len(source_data) > 0:
            style = SOURCE_STYLE.get(source, {'color': 'gray', 'marker': 'o', 'name': source.capitalize()})
            _plot_mean_sem(ax, source_data['x_pos'], source_data, style)
    
    ax.set_xticks(range(len(period_order)))
    ax.set_xticklabels(period_order, rotation=45, ha='right')
    ax.set_xlabel('6-Month Period', fontsize=12, fontweight='bold')
    ax.set_ylabel(BIAS_SCORE_YLABEL, fontsize=12, fontweight='bold')
    _set_fig_title_subtitle(
        fig,
        'Mean Negative Bias Frame Score (0-5) by 6-Month Period\nAcross Four Sources',
        'Error bars: SEM (std / sqrt(n)) of post-level scores within each 6-month x source bin',
    )
    ax.legend(title='Source', fontsize=9, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    output_path = Path(output_dir) / 'bias_by_half_year_all_sources.pdf'
    _save_figure(fig, output_path, right=0.98, bottom=0.15, left=0.07)
    print(f"Saved: {output_path}")
    plt.close()

def plot_bias_by_year_city(df, output_dir='output'):
    """Plot average bias by year for each city size group"""
    year_city_data = calculate_bias_by_period_city(df, period_col='year')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    _set_fig_title_subtitle(
        fig,
        'Mean Negative Bias Frame Score (0-5) by Year',
        'Error bars: SEM (std / sqrt(n)) within each year x source x city-cluster bin',
        top=0.88,
    )
    
    for idx, city_size in enumerate(['Large', 'Small']):
        ax = axes[idx]
        city_data = year_city_data[year_city_data['city_size'] == city_size]
        
        for source in city_data['source'].unique():
            source_data = city_data[city_data['source'] == source].sort_values('year')
            style = SOURCE_STYLE.get(source, {'color': 'gray', 'marker': 'o', 'name': source.capitalize()})
            _plot_mean_sem(ax, source_data['year'], source_data, style)
        
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel(BIAS_SCORE_YLABEL, fontsize=12, fontweight='bold')
        ax.set_title(CITY_CLUSTER_PANEL_TITLES[city_size], fontsize=11, fontweight='bold')
        ax.legend(title='Source', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    output_path = Path(output_dir) / 'bias_by_year_by_city_size.pdf'
    _save_figure(fig, output_path, right=0.88)
    print(f"Saved: {output_path}")
    plt.close()

def plot_bias_by_half_year_city(df, output_dir='output'):
    """Plot average bias by 6-month periods for each city size group"""
    half_year_city_data = calculate_bias_by_period_city(df, period_col='year_half')
    
    # Get period order
    df_with_period = df.copy()
    df_with_period = create_time_periods(df_with_period)
    period_order = df_with_period.groupby('year_half')['period_num'].first().sort_values().index.tolist()
    
    # Create mapping from period to x-axis position
    period_to_pos = {period: idx for idx, period in enumerate(period_order)}
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    _set_fig_title_subtitle(
        fig,
        'Mean Negative Bias Frame Score (0-5) by 6-Month Period',
        'Error bars: SEM (std / sqrt(n)) within each 6-month x source x city-cluster bin',
        top=0.88,
    )
    
    for idx, city_size in enumerate(['Large', 'Small']):
        ax = axes[idx]
        city_data = half_year_city_data[half_year_city_data['city_size'] == city_size].copy()
        
        for source in city_data['source'].unique():
            source_data = city_data[city_data['source'] == source].copy()
            
            # Map periods to x-axis positions
            source_data['x_pos'] = source_data['year_half'].map(period_to_pos)
            source_data = source_data.dropna(subset=['x_pos']).sort_values('x_pos')
            
            if len(source_data) > 0:
                style = SOURCE_STYLE.get(source, {'color': 'gray', 'marker': 'o', 'name': source.capitalize()})
                _plot_mean_sem(ax, source_data['x_pos'], source_data, style)
        
        ax.set_xticks(range(len(period_order)))
        ax.set_xticklabels(period_order, rotation=45, ha='right')
        ax.set_xlabel('6-Month Period', fontsize=12, fontweight='bold')
        ax.set_ylabel(BIAS_SCORE_YLABEL, fontsize=12, fontweight='bold')
        ax.set_title(CITY_CLUSTER_PANEL_TITLES[city_size], fontsize=11, fontweight='bold')
        ax.legend(title='Source', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    output_path = Path(output_dir) / 'bias_by_half_year_by_city_size.pdf'
    _save_figure(fig, output_path, right=0.88, bottom=0.22)
    print(f"Saved: {output_path}")
    plt.close()

def plot_bias_by_city_size_bar_chart(df, output_dir='output'):
    """Create bar chart comparing Large vs Small cities for each source, with significance testing"""
    
    # Filter to only Large and Small cities
    df_filtered = df[df['city_size'].isin(['Large', 'Small'])].copy()
    
    # Mean bias by source and city size (bar heights; no error bars on this figure)
    summary = df_filtered.groupby(['source', 'city_size'])['bias_score'].agg(['mean', 'count']).reset_index()
    summary.columns = ['source', 'city_size', 'mean_bias', 'count']
    
    # Perform statistical tests with Bonferroni correction
    # Compare Large vs Small for each source: 4 comparisons
    n_comparisons = 4
    bonferroni_alpha = 0.05 / n_comparisons
    
    print(f"\nStatistical Testing: Large vs Small Cities (Bonferroni corrected α = {bonferroni_alpha:.6f})")
    print("="*80)
    
    # Store significance results
    significance_results = []
    
    # Compare Large vs Small for each source
    source_order = ['twitter', 'reddit', 'news', 'meeting_minutes']
    
    for source in source_order:
        source_data = df_filtered[df_filtered['source'] == source]
        large_data = source_data[source_data['city_size'] == 'Large']['bias_score']
        small_data = source_data[source_data['city_size'] == 'Small']['bias_score']
        
        if len(large_data) > 0 and len(small_data) > 0:
            t_stat, p_value = stats.ttest_ind(large_data, small_data, equal_var=False)
            
            is_significant = p_value < bonferroni_alpha
            significance_results.append({
                'source': source,
                'large_mean': large_data.mean(),
                'small_mean': small_data.mean(),
                'large_count': len(large_data),
                'small_count': len(small_data),
                'p_value': p_value,
                'significant': is_significant,
                't_statistic': t_stat,
            })
            
            sig_marker = "***" if is_significant else ""
            print(
                f"{source.capitalize()}: Large mean={large_data.mean():.3f} vs "
                f"Small mean={small_data.mean():.3f}, p={p_value:.6f} {sig_marker}"
            )
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for grouped bars
    x = np.arange(len(source_order))
    width = 0.35
    
    large_means = []
    large_counts = []
    small_means = []
    small_counts = []
    
    for source in source_order:
        large_row = summary[(summary['source'] == source) & (summary['city_size'] == 'Large')]
        small_row = summary[(summary['source'] == source) & (summary['city_size'] == 'Small')]
        
        if len(large_row) > 0:
            large_means.append(large_row.iloc[0]['mean_bias'])
            large_counts.append(large_row.iloc[0]['count'])
        else:
            large_means.append(0)
            large_counts.append(0)
        
        if len(small_row) > 0:
            small_means.append(small_row.iloc[0]['mean_bias'])
            small_counts.append(small_row.iloc[0]['count'])
        else:
            small_means.append(0)
            small_counts.append(0)
    
    # Get colors for each source
    colors = [SOURCE_STYLE.get(s, {'color': 'gray'})['color'] for s in source_order]
    labels = [SOURCE_STYLE.get(s, {'name': s.capitalize()})['name'] for s in source_order]
    
    # Create bars (no error bars; significance markers already shown with Bonferroni correction)
    bars_large = ax.bar(
        x - width / 2,
        large_means,
        width,
        label=LARGE_CITY_CLUSTER_LEGEND,
        color=colors,
        alpha=0.7,
        edgecolor='black',
        linewidth=1.5,
    )
    
    bars_small = ax.bar(
        x + width / 2,
        small_means,
        width,
        label=SMALL_CITY_CLUSTER_LEGEND,
        color=colors,
        alpha=0.4,
        edgecolor='black',
        linewidth=1.5,
        hatch='///',
    )
    
    # Add significance markers
    max_bar_height = max(large_means + small_means) if (large_means or small_means) else 0
    bracket_height = 0.05
    bracket_y_base = max_bar_height + 0.08
    
    for i, source in enumerate(source_order):
        sig_result = next((s for s in significance_results if s['source'] == source), None)
        if sig_result and sig_result['significant']:
            # Draw bracket between Large and Small bars
            bracket_y = max(large_means[i], small_means[i]) + 0.05
            ax.plot([x[i] - width/2, x[i] - width/2, x[i] + width/2, x[i] + width/2],
                   [bracket_y, bracket_y + bracket_height, bracket_y + bracket_height, bracket_y],
                   'k-', linewidth=1.5)
            # Add asterisk
            ax.text(x[i], bracket_y + bracket_height + 0.02, '*', 
                   ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    # Add sample size annotations
    for i, (bar_l, bar_s, count_l, count_s) in enumerate(zip(bars_large, bars_small, large_counts, small_counts)):
        if count_l > 0:
            height_l = bar_l.get_height()
            ax.text(bar_l.get_x() + bar_l.get_width()/2., height_l + 0.02,
                   f'n={count_l}', ha='center', va='bottom', fontsize=8, fontstyle='italic')
        if count_s > 0:
            height_s = bar_s.get_height()
            ax.text(bar_s.get_x() + bar_s.get_width()/2., height_s + 0.02,
                   f'n={count_s}', ha='center', va='bottom', fontsize=8, fontstyle='italic')
    
    ax.set_xlabel('Source', fontsize=12, fontweight='bold')
    ax.set_ylabel(BIAS_SCORE_YLABEL, fontsize=12, fontweight='bold')
    bias_categories_text = (
        'Categories: rhetorical question, not in my backyard, harmful generalization, '
        'deserving/undeserving, racist (sum of active labels per post, 0-5)'
    )
    _set_fig_title_subtitle(
        fig,
        'Mean Negative Bias Frame Score (0-5): Large City Cluster vs Small City Cluster by Source',
        bias_categories_text,
        top=0.82,
    )
    
    # Add Bonferroni correction note
    bonferroni_note = '* p < 0.0125 (Bonferroni corrected, α = 0.05/4 comparisons)'
    fig.text(0.5, 0.02, bonferroni_note, ha='center', fontsize=9, style='italic')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='center')
    
    # Set y-axis limit before adding legend to ensure proper spacing
    y_max_with_markers = max_bar_height + 0.25  # Extra space for significance markers
    ax.set_ylim(bottom=0, top=y_max_with_markers)
    
    # Position legend to avoid overlap with significance markers
    ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    ax.grid(True, alpha=0.3, axis='y')
    
    output_path = Path(output_dir) / 'bias_by_source_city_size_bar_chart.pdf'
    _save_figure(fig, output_path, right=0.92, bottom=0.18)
    print(f"\nSaved: {output_path}")
    plt.close()
    
    # Save significance results
    sig_df = pd.DataFrame(significance_results)
    sig_output = Path(output_dir) / 'bias_significance_tests.csv'
    sig_df.to_csv(sig_output, index=False)
    print(f"Saved significance tests: {sig_output}")
    
    return summary, sig_df

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main analysis function"""
    print("="*80)
    print("BIAS TEMPORAL ANALYSIS")
    print("="*80)
    
    # Create output directory
    output_dir = Path('output/bias_temporal_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all data
    df = load_all_data(filter_zero_category_rows=True)
    
    # Create time periods
    df = create_time_periods(df)
    
    # Calculate summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    print("\nBias Score by Source:")
    print(df.groupby('source')['bias_score'].agg(['mean', 'std', 'count']))
    
    print("\nBias Score by City Size:")
    print(df[df['city_size'] != 'Unknown'].groupby('city_size')['bias_score'].agg(['mean', 'std', 'count']))
    
    print("\nBias Score by Source and City Size:")
    print(df[df['city_size'] != 'Unknown'].groupby(['source', 'city_size'])['bias_score'].agg(['mean', 'std', 'count']))
    
    # Calculate temporal statistics
    print(f"\n{'='*80}")
    print("TEMPORAL ANALYSIS")
    print(f"{'='*80}")
    
    # By year
    year_data = calculate_bias_by_period(df, period_col='year')
    year_output = output_dir / 'bias_by_year.csv'
    year_data.to_csv(year_output, index=False)
    print(f"\nSaved year data: {year_output}")
    print(year_data.head(20))
    
    # By 6-month periods
    half_year_data = calculate_bias_by_period(df, period_col='year_half')
    half_year_output = output_dir / 'bias_by_half_year.csv'
    half_year_data.to_csv(half_year_output, index=False)
    print(f"\nSaved 6-month period data: {half_year_output}")
    print(half_year_data.head(20))
    
    # By year and city size
    year_city_data = calculate_bias_by_period_city(df, period_col='year')
    year_city_output = output_dir / 'bias_by_year_by_city_size.csv'
    year_city_data.to_csv(year_city_output, index=False)
    print(f"\nSaved year by city size data: {year_city_output}")
    print(year_city_data.head(20))
    
    # By 6-month periods and city size
    half_year_city_data = calculate_bias_by_period_city(df, period_col='year_half')
    half_year_city_output = output_dir / 'bias_by_half_year_by_city_size.csv'
    half_year_city_data.to_csv(half_year_city_output, index=False)
    print(f"\nSaved 6-month period by city size data: {half_year_city_output}")
    print(half_year_city_data.head(20))
    
    # Create visualizations
    print(f"\n{'='*80}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    plot_bias_by_year(df, output_dir=str(output_dir))
    plot_bias_by_half_year(df, output_dir=str(output_dir))
    plot_bias_by_year_city(df, output_dir=str(output_dir))
    plot_bias_by_half_year_city(df, output_dir=str(output_dir))
    
    # Create bar chart with significance testing
    print(f"\n{'='*80}")
    print("BAR CHART WITH SIGNIFICANCE TESTING")
    print(f"{'='*80}")
    bar_summary, sig_results = plot_bias_by_city_size_bar_chart(df, output_dir=str(output_dir))
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"All outputs saved to: {output_dir}")

if __name__ == '__main__':
    main()
