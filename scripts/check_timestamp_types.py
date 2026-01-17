#!/usr/bin/env python3
"""
Check Timestamp Types by Source: Analyze different timestamp columns for Twitter and News

This script examines:
1. All timestamp/date columns in Twitter and News datasets
2. Their formats and data types
3. Sample values and ranges
4. Missing data patterns

Usage:
    python scripts/check_timestamp_types.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_timestamps(df, source_name, date_columns):
    """Analyze timestamp columns for a given dataset"""
    print(f"\n{'='*80}")
    print(f"{source_name.upper()} TIMESTAMP ANALYSIS")
    print(f"{'='*80}")
    
    print(f"\nTotal rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    # Find all potential date/timestamp columns
    all_date_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['date', 'time', 'timestamp', 'created', 'published']):
            all_date_cols.append(col)
    
    print(f"\nPotential date/timestamp columns found: {all_date_cols}")
    
    # Analyze each date column
    for col in all_date_cols:
        print(f"\n{'-'*80}")
        print(f"Column: {col}")
        print(f"{'-'*80}")
        
        # Basic info
        non_null_count = df[col].notna().sum()
        null_count = df[col].isna().sum()
        print(f"Non-null values: {non_null_count} ({non_null_count/len(df)*100:.2f}%)")
        print(f"Null values: {null_count} ({null_count/len(df)*100:.2f}%)")
        
        if non_null_count == 0:
            print("  WARNING: All values are null!")
            continue
        
        # Data type
        print(f"Data type: {df[col].dtype}")
        
        # Sample values
        print(f"\nSample values (first 10 non-null):")
        sample_values = df[col].dropna().head(10)
        for idx, val in enumerate(sample_values, 1):
            print(f"  {idx}. {val} (type: {type(val).__name__})")
        
        # Try to parse as datetime
        print(f"\nAttempting to parse as datetime...")
        try:
            parsed_dates = pd.to_datetime(df[col], errors='coerce', utc=True)
            valid_dates = parsed_dates.notna().sum()
            print(f"  Successfully parsed: {valid_dates} values ({valid_dates/len(df)*100:.2f}%)")
            
            if valid_dates > 0:
                print(f"  Date range:")
                print(f"    Min: {parsed_dates.min()}")
                print(f"    Max: {parsed_dates.max()}")
                print(f"    Span: {(parsed_dates.max() - parsed_dates.min()).days} days")
                
                # Check for timezone info
                sample_parsed = parsed_dates.dropna().head(5)
                if len(sample_parsed) > 0:
                    print(f"  Sample parsed dates:")
                    for val in sample_parsed:
                        print(f"    {val} (tz: {val.tz})")
        except Exception as e:
            print(f"  ERROR parsing dates: {e}")
        
        # Unique value count
        unique_count = df[col].nunique()
        print(f"\nUnique values: {unique_count}")
        if unique_count < 20:
            print(f"  Unique values: {sorted(df[col].dropna().unique())}")
        
        # Value length analysis (for string dates)
        if df[col].dtype == 'object':
            lengths = df[col].dropna().astype(str).str.len()
            print(f"\nString length statistics:")
            print(f"  Min: {lengths.min()}")
            print(f"  Max: {lengths.max()}")
            print(f"  Mean: {lengths.mean():.2f}")
            print(f"  Most common length: {lengths.mode().iloc[0] if len(lengths.mode()) > 0 else 'N/A'}")
    
    # Check for date columns specified in config
    if date_columns:
        print(f"\n{'-'*80}")
        print(f"SPECIFIED DATE COLUMNS ANALYSIS")
        print(f"{'-'*80}")
        for date_col in date_columns:
            if date_col in df.columns:
                print(f"\n{date_col}:")
                non_null = df[date_col].notna().sum()
                print(f"  Non-null: {non_null} ({non_null/len(df)*100:.2f}%)")
                if non_null > 0:
                    parsed = pd.to_datetime(df[date_col], errors='coerce', utc=True)
                    valid = parsed.notna().sum()
                    print(f"  Valid dates: {valid} ({valid/len(df)*100:.2f}%)")
                    if valid > 0:
                        print(f"  Range: {parsed.min()} to {parsed.max()}")
            else:
                print(f"\n{date_col}: NOT FOUND in columns")

def main():
    """Main analysis function"""
    print("="*80)
    print("TIMESTAMP TYPE ANALYSIS FOR TWITTER AND NEWS")
    print("="*80)
    
    # Load Twitter data
    twitter_file = 'complete_dataset/all_twitter_posts_merged_with_details.csv'
    print(f"\nLoading Twitter data from {twitter_file}...")
    if Path(twitter_file).exists():
        df_twitter = pd.read_csv(twitter_file, low_memory=False, nrows=10000)  # Sample for faster analysis
        print(f"Loaded {len(df_twitter)} rows (sampled)")
        analyze_timestamps(df_twitter, 'Twitter', ['created_at', 'author_created_at'])
    else:
        print(f"ERROR: File not found: {twitter_file}")
    
    # Load News data
    news_file = 'complete_dataset/all_newspaper_articles.csv'
    print(f"\nLoading News data from {news_file}...")
    if Path(news_file).exists():
        df_news = pd.read_csv(news_file, low_memory=False)
        print(f"Loaded {len(df_news)} rows")
        analyze_timestamps(df_news, 'News', ['article_date'])
    else:
        print(f"ERROR: File not found: {news_file}")
    
    # Comparison summary
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    if Path(twitter_file).exists() and Path(news_file).exists():
        df_twitter = pd.read_csv(twitter_file, low_memory=False)
        df_news = pd.read_csv(news_file, low_memory=False)
        
        # Twitter timestamps
        if 'created_at' in df_twitter.columns:
            twitter_dates = pd.to_datetime(df_twitter['created_at'], errors='coerce', utc=True)
            twitter_valid = twitter_dates.notna().sum()
            print(f"\nTwitter 'created_at':")
            print(f"  Total rows: {len(df_twitter)}")
            print(f"  Valid dates: {twitter_valid} ({twitter_valid/len(df_twitter)*100:.2f}%)")
            if twitter_valid > 0:
                print(f"  Range: {twitter_dates.min()} to {twitter_dates.max()}")
        
        # News timestamps
        if 'article_date' in df_news.columns:
            news_dates = pd.to_datetime(df_news['article_date'], errors='coerce', utc=True)
            news_valid = news_dates.notna().sum()
            print(f"\nNews 'article_date':")
            print(f"  Total rows: {len(df_news)}")
            print(f"  Valid dates: {news_valid} ({news_valid/len(df_news)*100:.2f}%)")
            if news_valid > 0:
                print(f"  Range: {news_dates.min()} to {news_dates.max()}")
        
        # Format comparison
        print(f"\nFormat comparison:")
        if 'created_at' in df_twitter.columns:
            twitter_sample = df_twitter['created_at'].dropna().head(3)
            print(f"  Twitter sample formats:")
            for val in twitter_sample:
                print(f"    {val}")
        
        if 'article_date' in df_news.columns:
            news_sample = df_news['article_date'].dropna().head(3)
            print(f"  News sample formats:")
            for val in news_sample:
                print(f"    {val}")

if __name__ == '__main__':
    main()
