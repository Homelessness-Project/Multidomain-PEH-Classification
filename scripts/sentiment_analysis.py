#!/usr/bin/env python3
"""
Comprehensive Sentiment Analysis by Category, City, and City Grouping

Analyzes sentiment on GPT-classified comments and compares sentiment across:
- All 16 categories
- Individual cities
- City size groupings (Large vs Small)
- Data sources (Reddit, News, X, Meeting Minutes)
- Statistical significance testing

Usage:
    python scripts/sentiment_analysis_by_category.py
"""

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
from textblob import TextBlob
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
warnings.filterwarnings('ignore')

# Try to import VADER (better for social media) and transformers
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("VADER not available. Install with: pip install vaderSentiment")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Install with: pip install transformers torch")

# Global sentiment analyzer instances (lazy loading)
_vader_analyzer = None
_transformer_pipeline = None

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')
sns.set_palette("husl")

# Define all 16 categories
ALL_CATEGORIES = [
    # Comment Types (6)
    'Comment_ask a genuine question',
    'Comment_ask a rhetorical question',
    'Comment_provide a fact or claim',
    'Comment_provide an observation',
    'Comment_express their opinion',
    'Comment_express others opinions',
    # Critique Categories (3)
    'Critique_money aid allocation',
    'Critique_government critique',
    'Critique_societal critique',
    # Response Categories (1)
    'Response_solutions/interventions',
    # Perception Types (5)
    'Perception_personal interaction',
    'Perception_media portrayal',
    'Perception_not in my backyard',
    'Perception_harmful generalization',
    'Perception_deserving/undeserving',
    # Racist Classification (1)
    'Racist_Flag'
]

# Category groups for analysis
CATEGORY_GROUPS = {
    'Comment Types': [
        'Comment_ask a genuine question',
        'Comment_ask a rhetorical question',
        'Comment_provide a fact or claim',
        'Comment_provide an observation',
        'Comment_express their opinion',
        'Comment_express others opinions'
    ],
    'Critique Categories': [
        'Critique_money aid allocation',
        'Critique_government critique',
        'Critique_societal critique'
    ],
    'Response Categories': [
        'Response_solutions/interventions'
    ],
    'Perception Types': [
        'Perception_personal interaction',
        'Perception_media portrayal',
        'Perception_not in my backyard',
        'Perception_harmful generalization',
        'Perception_deserving/undeserving'
    ],
    'Racist Classification': [
        'Racist_Flag'
    ]
}

# City size groupings (based on user specification)
LARGE_CITIES = ['san francisco', 'portland', 'buffalo', 'baltimore', 'el paso']
SMALL_CITIES = ['kalamazoo', 'south bend', 'rockford', 'scranton', 'fayetteville']

# Bonferroni correction for 16 categories
BONFERRONI_ALPHA = 0.05 / 16  # 0.003125

def get_sentiment_analyzer(method='vader'):
    """Get or initialize sentiment analyzer based on method."""
    global _vader_analyzer, _transformer_pipeline
    
    if method == 'vader' and VADER_AVAILABLE:
        if _vader_analyzer is None:
            _vader_analyzer = SentimentIntensityAnalyzer()
        return _vader_analyzer
    elif method == 'transformer' and TRANSFORMERS_AVAILABLE:
        if _transformer_pipeline is None:
            # Use a model specifically trained for sentiment on social media
            # cardiffnlp/twitter-roberta-base-sentiment-latest is excellent for Twitter/Reddit
            try:
                _transformer_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if torch.cuda.is_available() else -1,
                    return_all_scores=True
                )
            except Exception as e:
                print(f"Warning: Could not load transformer model: {e}")
                print("Falling back to VADER or TextBlob")
                return None
        return _transformer_pipeline
    return None

def get_textblob_sentiment(text):
    """Get TextBlob sentiment score: 1 (positive), 0 (neutral), -1 (negative).
    Returns tuple: (discrete_score, raw_polarity)
    """
    if not text or pd.isna(text):
        return (0, 0.0)
    
    text = str(text).strip()
    if not text:
        return (0, 0.0)
    
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            discrete = 1
        elif polarity < -0.1:
            discrete = -1
        else:
            discrete = 0
        return (discrete, polarity)
    except:
        return (0, 0.0)

def get_vader_sentiment(text):
    """Get VADER sentiment score: 1 (positive), 0 (neutral), -1 (negative).
    Returns tuple: (discrete_score, raw_compound)
    """
    if not text or pd.isna(text):
        return (0, 0.0)
    
    text = str(text).strip()
    if not text:
        return (0, 0.0)
    
    if not VADER_AVAILABLE:
        return (0, 0.0)
    
    try:
        analyzer = get_sentiment_analyzer('vader')
        if analyzer is not None:
            scores = analyzer.polarity_scores(text)
            compound = scores['compound']
            
            # VADER compound score: > 0.05 positive, < -0.05 negative, else neutral
            if compound > 0.05:
                discrete = 1
            elif compound < -0.05:
                discrete = -1
            else:
                discrete = 0
            return (discrete, compound)
    except:
        return (0, 0.0)
    
    return (0, 0.0)

def analyze_sentiment_both(text):
    """
    Analyze sentiment using both TextBlob and VADER.
    Returns a dict with both scores.
    Uses VADER for the main Sentiment (for analysis), TextBlob for reference.
    """
    if not text or pd.isna(text):
        return {'Sentiment': 0, 'TextBlob_Sentiment': 0, 'VADER_Sentiment': 0}
    
    text = str(text).strip()
    if not text:
        return {'Sentiment': 0, 'TextBlob_Sentiment': 0, 'VADER_Sentiment': 0}
    
    # Get both scores
    textblob_score = get_textblob_sentiment(text)
    
    if VADER_AVAILABLE:
        vader_score = get_vader_sentiment(text)
        # Use VADER for main Sentiment (for analysis)
        main_sentiment = vader_score
        vader_sentiment_col = vader_score
    else:
        # If VADER not available, use TextBlob for main sentiment
        vader_score = 0  # Placeholder
        main_sentiment = textblob_score
        vader_sentiment_col = textblob_score  # Use TextBlob value if VADER not available
    
    return {
        'Sentiment': main_sentiment,
        'TextBlob_Sentiment': textblob_score,
        'VADER_Sentiment': vader_sentiment_col
    }

def analyze_sentiment(text, method='vader'):
    """
    Analyze sentiment and return 1 (positive), 0 (neutral), -1 (negative).
    
    Methods (in order of accuracy):
    - 'transformer': Uses RoBERTa model fine-tuned on Twitter (most accurate)
    - 'vader': VADER sentiment analyzer (better than TextBlob for social media)
    - 'textblob': TextBlob (fallback, less accurate)
    
    Uses homelessness-specific keywords first, then model-based sentiment as fallback.
    """
    if not text or pd.isna(text):
        return 0
    
    text = str(text).strip()
    if not text:
        return 0
    
    # Homelessness-specific keywords (check first)
    text_lower = text.lower()
    
    positive_keywords = [
        "help", "support", "assist", "aid", "care", "compassion", "empathy",
        "solution", "housing", "shelter", "program", "initiative", "positive",
        "improve", "better", "hope", "recovery", "rehabilitation", "deserve",
        "rights", "dignity", "respect", "understanding"
    ]
    
    negative_keywords = [
        "problem", "issue", "crisis", "burden", "nuisance", "annoying",
        "dangerous", "threat", "criminal", "lazy", "drug addict", "alcoholic",
        "dirty", "filthy", "disgusting", "hate", "despise", "get rid of",
        "remove", "arrest", "jail", "criminal", "scary", "fear"
    ]
    
    positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
    negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
    
    # If keywords strongly indicate sentiment, use them
    if positive_count > negative_count + 1:
        return 1
    elif negative_count > positive_count + 1:
        return -1
    
    # Otherwise, use model-based sentiment
    try:
        if method == 'transformer' and TRANSFORMERS_AVAILABLE:
            analyzer = get_sentiment_analyzer('transformer')
            if analyzer is not None:
                # Truncate to model's max length (usually 512 tokens)
                result = analyzer(text[:512])[0]
                # Get the label with highest score
                scores = {item['label']: item['score'] for item in result}
                
                # Map labels to sentiment
                # Model uses: LABEL_0 (negative), LABEL_1 (neutral), LABEL_2 (positive)
                if 'LABEL_2' in scores and scores['LABEL_2'] > 0.5:
                    return 1
                elif 'LABEL_0' in scores and scores['LABEL_0'] > 0.5:
                    return -1
                else:
                    return 0
        
        elif method == 'vader' and VADER_AVAILABLE:
            return get_vader_sentiment(text)
        
        # Fallback to TextBlob
        return get_textblob_sentiment(text)
            
    except Exception as e:
        # If any error occurs, fall back to keyword-based or TextBlob
        if positive_count > negative_count:
            return 1
        elif negative_count > positive_count:
            return -1
        else:
            # Final fallback to TextBlob
            return get_textblob_sentiment(text)

def load_existing_sentiment(output_dir=None):
    """Load existing sentiment scores from CSV files if available.
    Checks for:
    1. all_sentiment_scores.csv in output_dir (new format)
    2. Individual source files in old locations (backward compatibility)
    """
    
    all_sentiment = []
    
    # First, check for the new consolidated file in output_dir
    if output_dir is not None:
        consolidated_file = Path(output_dir) / 'all_sentiment_scores.csv'
        if consolidated_file.exists():
            print(f"Loading existing sentiment from {consolidated_file}...")
            try:
                sentiment_df = pd.read_csv(consolidated_file)
                # Standardize column names
                if 'comment' in sentiment_df.columns:
                    sentiment_df = sentiment_df.rename(columns={'comment': 'Comment'})
                if 'city' in sentiment_df.columns:
                    sentiment_df = sentiment_df.rename(columns={'city': 'City'})
                # Normalize city names for matching
                sentiment_df['City_Normalized'] = sentiment_df['City'].astype(str).str.lower().str.strip()
                sentiment_df['Comment_Normalized'] = sentiment_df['Comment'].astype(str).str.strip()
                
                # Load TextBlob_Sentiment and VADER_Sentiment if available (discrete and raw)
                cols_to_keep = ['Comment', 'City', 'Source', 'Comment_Normalized', 'City_Normalized']
                if 'TextBlob_Sentiment' in sentiment_df.columns:
                    cols_to_keep.append('TextBlob_Sentiment')
                if 'TextBlob_Sentiment_Raw' in sentiment_df.columns:
                    cols_to_keep.append('TextBlob_Sentiment_Raw')
                if 'VADER_Sentiment' in sentiment_df.columns:
                    cols_to_keep.append('VADER_Sentiment')
                if 'VADER_Sentiment_Raw' in sentiment_df.columns:
                    cols_to_keep.append('VADER_Sentiment_Raw')
                
                all_sentiment.append(sentiment_df[cols_to_keep])
                print(f"  Loaded {len(sentiment_df)} sentiment scores from consolidated file")
                if all_sentiment:
                    combined_sentiment = pd.concat(all_sentiment, ignore_index=True)
                    print(f"\nTotal existing sentiment scores: {len(combined_sentiment)}")
                    return combined_sentiment
            except Exception as e:
                print(f"  WARNING: Could not load consolidated sentiment file: {e}")
    
    # Fall back to checking individual source files (backward compatibility)
    sentiment_files = {
        'reddit': 'output/reddit/nlp/sentiment.csv',
        'news': 'output/news/nlp/sentiment.csv',
        'x': 'output/x/nlp/sentiment.csv',
        'meeting_minutes': 'output/meeting_minutes/nlp/sentiment.csv'
    }
    
    for source, filepath in sentiment_files.items():
        if os.path.exists(filepath):
            print(f"Loading existing sentiment for {source} from {filepath}...")
            try:
                sentiment_df = pd.read_csv(filepath)
                # Standardize column names
                if 'comment' in sentiment_df.columns:
                    sentiment_df = sentiment_df.rename(columns={'comment': 'Comment'})
                if 'city' in sentiment_df.columns:
                    sentiment_df = sentiment_df.rename(columns={'city': 'City'})
                sentiment_df['Source'] = source
                # Normalize city names for matching
                sentiment_df['City_Normalized'] = sentiment_df['City'].astype(str).str.lower().str.strip()
                sentiment_df['Comment_Normalized'] = sentiment_df['Comment'].astype(str).str.strip()
                
                # Load TextBlob_Sentiment and VADER_Sentiment if available (discrete and raw)
                cols_to_keep = ['Comment', 'City', 'Source', 'Comment_Normalized', 'City_Normalized']
                if 'TextBlob_Sentiment' in sentiment_df.columns:
                    cols_to_keep.append('TextBlob_Sentiment')
                if 'TextBlob_Sentiment_Raw' in sentiment_df.columns:
                    cols_to_keep.append('TextBlob_Sentiment_Raw')
                if 'VADER_Sentiment' in sentiment_df.columns:
                    cols_to_keep.append('VADER_Sentiment')
                if 'VADER_Sentiment_Raw' in sentiment_df.columns:
                    cols_to_keep.append('VADER_Sentiment_Raw')
                # Also check for old 'Sentiment' column for backward compatibility
                if 'Sentiment' in sentiment_df.columns and 'VADER_Sentiment' not in sentiment_df.columns:
                    # Treat old Sentiment as VADER_Sentiment for backward compatibility
                    sentiment_df['VADER_Sentiment'] = sentiment_df['Sentiment']
                    cols_to_keep.append('VADER_Sentiment')
                
                all_sentiment.append(sentiment_df[cols_to_keep])
                print(f"  Loaded {len(sentiment_df)} sentiment scores")
            except Exception as e:
                print(f"  WARNING: Could not load sentiment file: {e}")
        else:
            print(f"  No existing sentiment file found for {source}")
    
    if all_sentiment:
        combined_sentiment = pd.concat(all_sentiment, ignore_index=True)
        print(f"\nTotal existing sentiment scores: {len(combined_sentiment)}")
        return combined_sentiment
    else:
        print("\nNo existing sentiment files found. Will calculate sentiment for all comments.")
        return None

def load_original_data():
    """Load original data files from complete_dataset."""
    
    data_files = {
        'reddit': {
            'file': 'complete_dataset/all_reddit_comments.csv',
            'comment_col': 'Deidentified_Comment'
        },
        'news': {
            'file': 'complete_dataset/all_newspaper_articles.csv',
            'comment_col': 'Deidentified_paragraph_text'
        },
        'x': {
            'file': 'complete_dataset/all_twitter_posts.csv',
            'comment_col': 'Deidentified_text'
        },
        'meeting_minutes': {
            'file': 'complete_dataset/all_meeting_minutes.csv',
            'comment_col': 'Deidentified_paragraph'
        }
    }
    
    all_data = []
    
    for source, config in data_files.items():
        filepath = config['file']
        comment_col = config['comment_col']
        
        print(f"Loading {source} data from {filepath}...")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['Source'] = source
            # Rename comment column to standard 'Comment'
            if comment_col in df.columns:
                df = df.rename(columns={comment_col: 'Comment'})
            # Ensure City column exists
            if 'city' in df.columns:
                df = df.rename(columns={'city': 'City'})
            elif 'City' not in df.columns:
                df['City'] = ''
            all_data.append(df)
            print(f"  Loaded {len(df)} rows")
        else:
            print(f"  WARNING: File not found: {filepath}")
    
    if not all_data:
        raise ValueError("No data files found!")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal combined rows: {len(combined_df)}")
    
    return combined_df

def load_gpt_categories():
    """Load GPT classification categories to merge with sentiment data. Memory optimized."""
    
    gpt_files = {
        'reddit': 'output/reddit/gpt4/classified_comments_reddit_all_gpt4_reddit_flags.csv',
        'news': 'output/news/gpt4/classified_comments_news_all_gpt4_news_flags.csv',
        'x': 'output/x/gpt4/classified_comments_x_all_gpt4_x_flags.csv',
        'meeting_minutes': 'output/meeting_minutes/gpt4/classified_comments_meeting_minutes_all_gpt4_meeting_minutes_flags.csv'
    }
    
    all_data = []
    
    for source, filepath in gpt_files.items():
        if os.path.exists(filepath):
            print(f"Loading GPT categories for {source} from {filepath}...")
            # Read only the columns we need to save memory
            df = pd.read_csv(filepath, low_memory=False)
            df['Source'] = source
            
            # Keep only Comment, City, Source, and category flag columns
            category_cols = [col for col in df.columns if col in ALL_CATEGORIES or 
                           col in ['Comment', 'City', 'Source'] or 
                           col.startswith('Comment_') or col.startswith('Critique_') or 
                           col.startswith('Response_') or col.startswith('Perception_') or 
                           col == 'Racist_Flag']
            df = df[category_cols]
            
            # Optimize data types to save memory
            # Convert category columns to bool/int8 if they're flags
            for col in category_cols:
                if col not in ['Comment', 'City', 'Source']:
                    if df[col].dtype == 'object':
                        # Try to convert string booleans to actual booleans
                        df[col] = df[col].astype(str).str.lower().isin(['yes', 'true', '1', 'y']).astype('int8')
                    elif df[col].dtype in ['float64', 'float32']:
                        df[col] = df[col].fillna(0).astype('int8')
                    elif df[col].dtype in ['int64', 'int32']:
                        df[col] = df[col].astype('int8')
            
            all_data.append(df)
            print(f"  Loaded {len(df)} rows with categories")
            
            # Force garbage collection after each file
            gc.collect()
        else:
            print(f"  WARNING: GPT file not found: {filepath}")
    
    if all_data:
        print("  Combining all GPT category files...")
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal GPT category rows: {len(combined_df)}")
        
        # Final garbage collection
        del all_data
        gc.collect()
        
        return combined_df
    else:
        print("\nNo GPT category files found. Will proceed without categories.")
        return None

def add_sentiment_and_groupings(df, existing_sentiment=None, output_dir=None):
    """Add sentiment scores and grouping columns to dataframe.
    Runs TextBlob first (checkpoint every 5000), then VADER (checkpoint every 1000).
    Two independent models - no overall sentiment column.
    """
    
    # Normalize comment and city for matching
    df['Comment_Normalized'] = df['Comment'].astype(str).str.strip()
    df['City_Normalized'] = df['City'].astype(str).str.lower().str.strip()
    
    # Try to merge with existing sentiment
    if existing_sentiment is not None:
        print("\nMatching comments with existing sentiment scores...")
        # Remove any existing sentiment columns to avoid conflicts
        sentiment_cols_to_drop = []
        if 'TextBlob_Sentiment' in df.columns:
            sentiment_cols_to_drop.append('TextBlob_Sentiment')
        if 'TextBlob_Sentiment_Raw' in df.columns:
            sentiment_cols_to_drop.append('TextBlob_Sentiment_Raw')
        if 'VADER_Sentiment' in df.columns:
            sentiment_cols_to_drop.append('VADER_Sentiment')
        if 'VADER_Sentiment_Raw' in df.columns:
            sentiment_cols_to_drop.append('VADER_Sentiment_Raw')
        if sentiment_cols_to_drop:
            df = df.drop(columns=sentiment_cols_to_drop)
        
        # Merge on normalized comment and city
        # Include sentiment columns if they exist (discrete and raw)
        merge_cols = ['Comment_Normalized', 'City_Normalized', 'Source']
        if 'TextBlob_Sentiment' in existing_sentiment.columns:
            merge_cols.append('TextBlob_Sentiment')
        if 'TextBlob_Sentiment_Raw' in existing_sentiment.columns:
            merge_cols.append('TextBlob_Sentiment_Raw')
        if 'VADER_Sentiment' in existing_sentiment.columns:
            merge_cols.append('VADER_Sentiment')
        if 'VADER_Sentiment_Raw' in existing_sentiment.columns:
            merge_cols.append('VADER_Sentiment_Raw')
        
        df = df.merge(
            existing_sentiment[merge_cols],
            on=['Comment_Normalized', 'City_Normalized', 'Source'],
            how='left'
        )
        
        textblob_matched = df['TextBlob_Sentiment'].notna().sum() if 'TextBlob_Sentiment' in df.columns else 0
        vader_matched = df['VADER_Sentiment'].notna().sum() if 'VADER_Sentiment' in df.columns else 0
        print(f"  Matched {textblob_matched} comments with TextBlob sentiment ({textblob_matched/len(df)*100:.1f}%)")
        print(f"  Matched {vader_matched} comments with VADER sentiment ({vader_matched/len(df)*100:.1f}%)")
    
    # Initialize sentiment columns if they don't exist (discrete and raw)
    if 'TextBlob_Sentiment' not in df.columns:
        df['TextBlob_Sentiment'] = np.nan
    if 'TextBlob_Sentiment_Raw' not in df.columns:
        df['TextBlob_Sentiment_Raw'] = np.nan
    if 'VADER_Sentiment' not in df.columns:
        df['VADER_Sentiment'] = np.nan
    if 'VADER_Sentiment_Raw' not in df.columns:
        df['VADER_Sentiment_Raw'] = np.nan
    
    # Check what needs to be calculated
    needs_textblob = df['TextBlob_Sentiment'].isna().sum()
    needs_vader = df['VADER_Sentiment'].isna().sum()
    
    # Determine which scores to calculate
    calculate_textblob = needs_textblob > 0
    calculate_vader = needs_vader > 0 and VADER_AVAILABLE
    
    # STEP 1: Run TextBlob first with checkpointing every 5000
    if calculate_textblob:
        print(f"\n{'='*80}")
        print("STEP 1: CALCULATING TEXTBLOB SENTIMENT")
        print(f"{'='*80}")
        print(f"  Need to calculate TextBlob for {needs_textblob} comments")
        print("  Checkpointing every 5,000 comments...")
        
        # Checkpoint file for TextBlob
        textblob_checkpoint_file = None
        if output_dir is not None:
            textblob_checkpoint_file = Path(output_dir) / 'textblob_checkpoint.csv'
        textblob_start_idx = 0
        
        # Load existing TextBlob checkpoint if available
        if textblob_checkpoint_file and textblob_checkpoint_file.exists():
            try:
                checkpoint_df = pd.read_csv(textblob_checkpoint_file)
                textblob_start_idx = len(checkpoint_df)
                print(f"  Found TextBlob checkpoint with {textblob_start_idx} processed comments. Resuming...")
                # Merge checkpoint data
                checkpoint_cols = ['Comment_Normalized', 'City_Normalized', 'Source', 'TextBlob_Sentiment']
                if 'TextBlob_Sentiment_Raw' in checkpoint_df.columns:
                    checkpoint_cols.append('TextBlob_Sentiment_Raw')
                df = df.merge(
                    checkpoint_df[checkpoint_cols],
                    on=['Comment_Normalized', 'City_Normalized', 'Source'],
                    how='left',
                    suffixes=('', '_checkpoint')
                )
                # Use checkpoint values where original is NaN
                for col in ['TextBlob_Sentiment', 'TextBlob_Sentiment_Raw']:
                    checkpoint_col = f'{col}_checkpoint'
                    if checkpoint_col in df.columns:
                        df.loc[df[col].isna() & df[checkpoint_col].notna(), col] = \
                            df.loc[df[col].isna() & df[checkpoint_col].notna(), checkpoint_col]
                        df = df.drop(columns=[checkpoint_col])
            except Exception as e:
                print(f"  Warning: Could not load TextBlob checkpoint: {e}. Starting fresh.")
                textblob_start_idx = 0
        
        # Find rows that need TextBlob calculation
        missing_textblob_indices = df[df['TextBlob_Sentiment'].isna()].index[textblob_start_idx:]
        total_textblob_to_process = len(missing_textblob_indices)
        
        if total_textblob_to_process > 0:
            print(f"  Processing {total_textblob_to_process} comments for TextBlob (starting from index {textblob_start_idx})...")
            
            textblob_checkpoint_data = []
            for i, idx in enumerate(tqdm(missing_textblob_indices, desc="TextBlob sentiment")):
                text = df.loc[idx, 'Comment']
                
                # Calculate TextBlob (returns tuple: discrete, raw)
                textblob_discrete, textblob_raw = get_textblob_sentiment(text)
                df.loc[idx, 'TextBlob_Sentiment'] = textblob_discrete
                df.loc[idx, 'TextBlob_Sentiment_Raw'] = textblob_raw
                
                # Save checkpoint data
                textblob_checkpoint_data.append({
                    'Comment_Normalized': df.loc[idx, 'Comment_Normalized'],
                    'City_Normalized': df.loc[idx, 'City_Normalized'],
                    'Source': df.loc[idx, 'Source'],
                    'TextBlob_Sentiment': textblob_discrete,
                    'TextBlob_Sentiment_Raw': textblob_raw
                })
                
                # Save checkpoint every 5,000 comments
                if (i + 1) % 5000 == 0:
                    if textblob_checkpoint_file:
                        checkpoint_df = pd.DataFrame(textblob_checkpoint_data)
                        if textblob_checkpoint_file.exists():
                            existing_checkpoint = pd.read_csv(textblob_checkpoint_file)
                            checkpoint_df = pd.concat([existing_checkpoint, checkpoint_df], ignore_index=True)
                        # Ensure directory exists
                        textblob_checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                        checkpoint_df.to_csv(textblob_checkpoint_file, index=False)
                        print(f"  TextBlob checkpoint saved: {len(checkpoint_df)} comments processed")
                    textblob_checkpoint_data = []
            
            # Save final TextBlob checkpoint
            if textblob_checkpoint_data and textblob_checkpoint_file:
                checkpoint_df = pd.DataFrame(textblob_checkpoint_data)
                if textblob_checkpoint_file.exists():
                    existing_checkpoint = pd.read_csv(textblob_checkpoint_file)
                    checkpoint_df = pd.concat([existing_checkpoint, checkpoint_df], ignore_index=True)
                # Ensure directory exists
                textblob_checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_df.to_csv(textblob_checkpoint_file, index=False)
                print(f"  Final TextBlob checkpoint saved: {len(checkpoint_df)} total comments processed")
            
            # Clean up TextBlob checkpoint file after successful completion
            if textblob_checkpoint_file and textblob_checkpoint_file.exists():
                try:
                    textblob_checkpoint_file.unlink()
                    print("  TextBlob checkpoint file cleaned up")
                except:
                    pass
        
        print(f"  TextBlob sentiment calculation complete: {df['TextBlob_Sentiment'].notna().sum()} comments processed")
    
    # STEP 2: Run VADER second with checkpointing every 1000
    if calculate_vader:
        print(f"\n{'='*80}")
        print("STEP 2: CALCULATING VADER SENTIMENT")
        print(f"{'='*80}")
        print(f"  Need to calculate VADER for {needs_vader} comments")
        print("  Checkpointing every 1,000 comments...")
        
        # Checkpoint file for VADER
        vader_checkpoint_file = None
        if output_dir is not None:
            vader_checkpoint_file = Path(output_dir) / 'vader_checkpoint.csv'
        vader_start_idx = 0
        
        # Load existing VADER checkpoint if available
        if vader_checkpoint_file and vader_checkpoint_file.exists():
            try:
                checkpoint_df = pd.read_csv(vader_checkpoint_file)
                vader_start_idx = len(checkpoint_df)
                print(f"  Found VADER checkpoint with {vader_start_idx} processed comments. Resuming...")
                # Merge checkpoint data
                checkpoint_cols = ['Comment_Normalized', 'City_Normalized', 'Source', 'VADER_Sentiment']
                if 'VADER_Sentiment_Raw' in checkpoint_df.columns:
                    checkpoint_cols.append('VADER_Sentiment_Raw')
                df = df.merge(
                    checkpoint_df[checkpoint_cols],
                    on=['Comment_Normalized', 'City_Normalized', 'Source'],
                    how='left',
                    suffixes=('', '_checkpoint')
                )
                # Use checkpoint values where original is NaN
                for col in ['VADER_Sentiment', 'VADER_Sentiment_Raw']:
                    checkpoint_col = f'{col}_checkpoint'
                    if checkpoint_col in df.columns:
                        df.loc[df[col].isna() & df[checkpoint_col].notna(), col] = \
                            df.loc[df[col].isna() & df[checkpoint_col].notna(), checkpoint_col]
                        df = df.drop(columns=[checkpoint_col])
            except Exception as e:
                print(f"  Warning: Could not load VADER checkpoint: {e}. Starting fresh.")
                vader_start_idx = 0
        
        # Find rows that need VADER calculation
        missing_vader_indices = df[df['VADER_Sentiment'].isna()].index[vader_start_idx:]
        total_vader_to_process = len(missing_vader_indices)
        
        if total_vader_to_process > 0:
            print(f"  Processing {total_vader_to_process} comments for VADER (starting from index {vader_start_idx})...")
            
            vader_checkpoint_data = []
            for i, idx in enumerate(tqdm(missing_vader_indices, desc="VADER sentiment")):
                text = df.loc[idx, 'Comment']
                
                # Calculate VADER (returns tuple: discrete, raw)
                vader_discrete, vader_raw = get_vader_sentiment(text)
                df.loc[idx, 'VADER_Sentiment'] = vader_discrete
                df.loc[idx, 'VADER_Sentiment_Raw'] = vader_raw
                
                # Save checkpoint data
                vader_checkpoint_data.append({
                    'Comment_Normalized': df.loc[idx, 'Comment_Normalized'],
                    'City_Normalized': df.loc[idx, 'City_Normalized'],
                    'Source': df.loc[idx, 'Source'],
                    'VADER_Sentiment': vader_discrete,
                    'VADER_Sentiment_Raw': vader_raw
                })
                
                # Save checkpoint every 1,000 comments
                if (i + 1) % 1000 == 0:
                    if vader_checkpoint_file:
                        checkpoint_df = pd.DataFrame(vader_checkpoint_data)
                        if vader_checkpoint_file.exists():
                            existing_checkpoint = pd.read_csv(vader_checkpoint_file)
                            checkpoint_df = pd.concat([existing_checkpoint, checkpoint_df], ignore_index=True)
                        # Ensure directory exists
                        vader_checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                        checkpoint_df.to_csv(vader_checkpoint_file, index=False)
                        print(f"  VADER checkpoint saved: {len(checkpoint_df)} comments processed")
                    vader_checkpoint_data = []
            
            # Save final VADER checkpoint
            if vader_checkpoint_data and vader_checkpoint_file:
                checkpoint_df = pd.DataFrame(vader_checkpoint_data)
                if vader_checkpoint_file.exists():
                    existing_checkpoint = pd.read_csv(vader_checkpoint_file)
                    checkpoint_df = pd.concat([existing_checkpoint, checkpoint_df], ignore_index=True)
                # Ensure directory exists
                vader_checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_df.to_csv(vader_checkpoint_file, index=False)
                print(f"  Final VADER checkpoint saved: {len(checkpoint_df)} total comments processed")
            
            # Clean up VADER checkpoint file after successful completion
            if vader_checkpoint_file and vader_checkpoint_file.exists():
                try:
                    vader_checkpoint_file.unlink()
                    print("  VADER checkpoint file cleaned up")
                except:
                    pass
        
        print(f"  VADER sentiment calculation complete: {df['VADER_Sentiment'].notna().sum()} comments processed")
    
    # Ensure all sentiment columns are numeric
    if 'TextBlob_Sentiment' in df.columns:
        df['TextBlob_Sentiment'] = pd.to_numeric(df['TextBlob_Sentiment'], errors='coerce').fillna(0).astype(int)
    if 'TextBlob_Sentiment_Raw' in df.columns:
        df['TextBlob_Sentiment_Raw'] = pd.to_numeric(df['TextBlob_Sentiment_Raw'], errors='coerce').fillna(0.0).astype(float)
    if 'VADER_Sentiment' in df.columns:
        df['VADER_Sentiment'] = pd.to_numeric(df['VADER_Sentiment'], errors='coerce').fillna(0).astype(int)
    if 'VADER_Sentiment_Raw' in df.columns:
        df['VADER_Sentiment_Raw'] = pd.to_numeric(df['VADER_Sentiment_Raw'], errors='coerce').fillna(0.0).astype(float)
    
    # Add city size grouping
    df['City_Size'] = 'Small'
    df.loc[df['City_Normalized'].isin([c.lower() for c in LARGE_CITIES]), 'City_Size'] = 'Large'
    
    return df

def merge_with_gpt_categories(df, gpt_categories):
    """Merge sentiment data with GPT categories. Optimized for memory efficiency."""
    
    if gpt_categories is None:
        return df
    
    print("\nMerging with GPT categories...")
    print(f"  Main dataframe: {len(df)} rows, {len(df.columns)} columns")
    print(f"  GPT categories: {len(gpt_categories)} rows, {len(gpt_categories.columns)} columns")
    
    # Get category columns from GPT data first (before any operations)
    category_cols = [col for col in gpt_categories.columns 
                    if col not in ['Comment', 'City', 'Source'] and 
                    (col in ALL_CATEGORIES or 
                     col.startswith('Comment_') or col.startswith('Critique_') or 
                     col.startswith('Response_') or col.startswith('Perception_') or 
                     col == 'Racist_Flag')]
    
    print(f"  Category columns to merge: {len(category_cols)}")
    
    # Create a minimal merge key dataframe from GPT categories
    # Only keep what we need for merging and the category columns
    print("  Preparing GPT categories for merge...")
    gpt_merge_cols = ['Comment', 'City', 'Source'] + category_cols
    gpt_subset = gpt_categories[gpt_merge_cols].copy()
    
    # Normalize GPT categories for matching (do this on the subset to save memory)
    print("  Normalizing GPT category keys...")
    gpt_subset['Comment_Normalized'] = gpt_subset['Comment'].astype(str).str.strip()
    gpt_subset['City_Normalized'] = gpt_subset['City'].astype(str).str.lower().str.strip()
    
    # Drop original Comment and City from gpt_subset to save memory (we'll use normalized versions)
    gpt_subset = gpt_subset.drop(columns=['Comment', 'City'])
    
    # Normalize main dataframe for matching (only if needed)
    print("  Normalizing main dataframe keys...")
    if 'Comment_Normalized' not in df.columns:
        df['Comment_Normalized'] = df['Comment'].astype(str).str.strip()
    if 'City_Normalized' not in df.columns:
        df['City_Normalized'] = df['City'].astype(str).str.lower().str.strip()
    
    # Create merge key columns list
    merge_keys = ['Comment_Normalized', 'City_Normalized', 'Source']
    
    # Use memory-efficient merge: only merge the category columns we need
    print("  Performing merge (this may take a moment for large datasets)...")
    try:
        # Merge only the category columns
        df_merged = df.merge(
            gpt_subset[merge_keys + category_cols],
            on=merge_keys,
            how='left',
            suffixes=('', '_gpt')
        )
        
        # Remove any duplicate columns that might have been created
        cols_to_drop = [col for col in df_merged.columns if col.endswith('_gpt')]
        if cols_to_drop:
            df_merged = df_merged.drop(columns=cols_to_drop)
        
    except MemoryError:
        print("  WARNING: Memory error during merge. Attempting chunked merge...")
        # Fall back to chunked processing
        chunk_size = 10000
        df_chunks = []
        
        for i in range(0, len(df), chunk_size):
            print(f"    Processing chunk {i//chunk_size + 1}/{(len(df)-1)//chunk_size + 1}...")
            df_chunk = df.iloc[i:i+chunk_size].copy()
            
            # Merge this chunk
            df_chunk_merged = df_chunk.merge(
                gpt_subset[merge_keys + category_cols],
                on=merge_keys,
                how='left'
            )
            
            df_chunks.append(df_chunk_merged)
            
            # Force garbage collection periodically
            if i % (chunk_size * 5) == 0:
                gc.collect()
        
        print("    Combining chunks...")
        df_merged = pd.concat(df_chunks, ignore_index=True)
        del df_chunks
        import gc
        gc.collect()
    
    matched_count = df_merged[category_cols[0]].notna().sum() if category_cols else 0
    print(f"  Matched {matched_count} comments with GPT categories ({matched_count/len(df_merged)*100:.1f}%)")
    
    # Clean up temporary columns
    print("  Cleaning up temporary columns...")
    df_merged = df_merged.drop(columns=['Comment_Normalized', 'City_Normalized'])
    
    return df_merged

def get_category_assignments(df, sentiment_col='VADER_Sentiment'):
    """Create a long-format dataframe with one row per comment-category pair.
    Uses VADER_Sentiment by default, but can use TextBlob_Sentiment if specified.
    """
    
    category_rows = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing categories"):
        comment = row['Comment']
        city = row['City']
        city_size = row['City_Size']
        source = row['Source']
        sentiment = row[sentiment_col] if sentiment_col in row else 0
        
        # Check each category flag
        for category in ALL_CATEGORIES:
            if category in df.columns:
                is_present = row[category]
                # Handle different possible formats (0/1, True/False, Yes/No)
                if pd.notna(is_present):
                    if isinstance(is_present, (int, float)):
                        is_present = bool(is_present)
                    elif isinstance(is_present, str):
                        is_present = is_present.lower() in ['yes', 'true', '1', 'y']
                    else:
                        is_present = bool(is_present)
                else:
                    is_present = False
                
                if is_present:
                    category_rows.append({
                        'Comment': comment,
                        'City': city,
                        'City_Size': city_size,
                        'Source': source,
                        'Category': category,
                        'Sentiment': sentiment
                    })
    
    return pd.DataFrame(category_rows)

def calculate_sentiment_stats(df, group_by_cols, sentiment_col='VADER_Sentiment'):
    """Calculate sentiment statistics grouped by specified columns.
    Uses VADER_Sentiment by default, but can use TextBlob_Sentiment if specified.
    """
    
    stats_list = []
    
    # Handle empty group_by_cols - calculate overall statistics
    if not group_by_cols or len(group_by_cols) == 0:
        sentiments = df[sentiment_col].values
        n = len(sentiments)
        
        if n > 0:
            positive_pct = (sentiments == 1).sum() / n * 100
            neutral_pct = (sentiments == 0).sum() / n * 100
            negative_pct = (sentiments == -1).sum() / n * 100
            mean_sentiment = sentiments.mean()
            
            stats_list.append({
                'N': n,
                'Positive_%': positive_pct,
                'Neutral_%': neutral_pct,
                'Negative_%': negative_pct,
                'Mean_Sentiment': mean_sentiment
            })
    else:
        # Group by specified columns
        for group_key, group_df in df.groupby(group_by_cols):
            if isinstance(group_key, tuple):
                group_dict = dict(zip(group_by_cols, group_key))
            else:
                group_dict = {group_by_cols[0]: group_key}
            
            sentiments = group_df[sentiment_col].values
            n = len(sentiments)
            
            if n == 0:
                continue
            
            positive_pct = (sentiments == 1).sum() / n * 100
            neutral_pct = (sentiments == 0).sum() / n * 100
            negative_pct = (sentiments == -1).sum() / n * 100
            mean_sentiment = sentiments.mean()
            
            stats_list.append({
                **group_dict,
                'N': n,
                'Positive_%': positive_pct,
                'Neutral_%': neutral_pct,
                'Negative_%': negative_pct,
                'Mean_Sentiment': mean_sentiment
            })
    
    return pd.DataFrame(stats_list)

def test_significance(group1_sentiments, group2_sentiments):
    """Test statistical significance between two sentiment distributions."""
    
    if len(group1_sentiments) < 2 or len(group2_sentiments) < 2:
        return None, None
    
    # Mann-Whitney U test (non-parametric, good for ordinal data)
    try:
        statistic, p_value = stats.mannwhitneyu(group1_sentiments, group2_sentiments, alternative='two-sided')
        return statistic, p_value
    except:
        return None, None

def analyze_by_category(category_df, sentiment_col='VADER_Sentiment'):
    """Analyze sentiment by individual category.
    Uses VADER_Sentiment by default, but can use TextBlob_Sentiment if specified.
    """
    
    print("\n" + "="*80)
    print("SENTIMENT ANALYSIS BY CATEGORY")
    print("="*80)
    
    results = []
    
    for category in ALL_CATEGORIES:
        category_data = category_df[category_df['Category'] == category]
        
        if len(category_data) == 0:
            continue
        
        sentiments = category_data[sentiment_col].values
        n = len(sentiments)
        
        positive_pct = (sentiments == 1).sum() / n * 100
        neutral_pct = (sentiments == 0).sum() / n * 100
        negative_pct = (sentiments == -1).sum() / n * 100
        mean_sentiment = sentiments.mean()
        
        results.append({
            'Category': category,
            'N': n,
            'Positive_%': positive_pct,
            'Neutral_%': neutral_pct,
            'Negative_%': negative_pct,
            'Mean_Sentiment': mean_sentiment
        })
    
    results_df = pd.DataFrame(results).sort_values('Mean_Sentiment', ascending=False)
    
    print("\nTop 5 Most Positive Categories:")
    print(results_df.head(5).to_string(index=False))
    
    print("\nTop 5 Most Negative Categories:")
    print(results_df.tail(5).to_string(index=False))
    
    return results_df

def analyze_by_city(category_df, sentiment_col='VADER_Sentiment'):
    """Analyze sentiment by city.
    Uses VADER_Sentiment by default, but can use TextBlob_Sentiment if specified.
    """
    
    print("\n" + "="*80)
    print("SENTIMENT ANALYSIS BY CITY")
    print("="*80)
    
    city_stats = calculate_sentiment_stats(category_df, ['City'], sentiment_col=sentiment_col)
    city_stats = city_stats.sort_values('Mean_Sentiment', ascending=False)
    
    print("\nSentiment by City (sorted by mean sentiment):")
    print(city_stats.to_string(index=False))
    
    return city_stats

def analyze_by_city_size(category_df, sentiment_col='VADER_Sentiment'):
    """Analyze sentiment by city size grouping.
    Uses VADER_Sentiment by default, but can use TextBlob_Sentiment if specified.
    """
    
    print("\n" + "="*80)
    print("SENTIMENT ANALYSIS BY CITY SIZE")
    print("="*80)
    
    size_stats = calculate_sentiment_stats(category_df, ['City_Size'], sentiment_col=sentiment_col)
    
    print("\nSentiment by City Size:")
    print(size_stats.to_string(index=False))
    
    # Statistical test
    large_sentiments = category_df[category_df['City_Size'] == 'Large'][sentiment_col].values
    small_sentiments = category_df[category_df['City_Size'] == 'Small'][sentiment_col].values
    
    statistic, p_value = test_significance(large_sentiments, small_sentiments)
    
    if p_value is not None:
        print(f"\nStatistical Test (Large vs Small Cities):")
        print(f"  Mann-Whitney U statistic: {statistic:.2f}")
        print(f"  p-value: {p_value:.4f}")
        if p_value < 0.05:
            print(f"  *** SIGNIFICANT (p < 0.05) ***")
        else:
            print(f"  Not significant (p >= 0.05)")
    
    return size_stats

def analyze_by_category_and_city_size(category_df, sentiment_col='VADER_Sentiment'):
    """Analyze sentiment by category and city size.
    Uses VADER_Sentiment by default, but can use TextBlob_Sentiment if specified.
    """
    
    print("\n" + "="*80)
    print("SENTIMENT ANALYSIS BY CATEGORY AND CITY SIZE")
    print("="*80)
    
    results = []
    
    for category in ALL_CATEGORIES:
        category_data = category_df[category_df['Category'] == category]
        
        if len(category_data) == 0:
            continue
        
        for city_size in ['Large', 'Small']:
            size_data = category_data[category_data['City_Size'] == city_size]
            
            if len(size_data) == 0:
                continue
            
            sentiments = size_data[sentiment_col].values
            n = len(sentiments)
            
            positive_pct = (sentiments == 1).sum() / n * 100
            negative_pct = (sentiments == -1).sum() / n * 100
            mean_sentiment = sentiments.mean()
            
            results.append({
                'Category': category,
                'City_Size': city_size,
                'N': n,
                'Positive_%': positive_pct,
                'Negative_%': negative_pct,
                'Mean_Sentiment': mean_sentiment
            })
    
    results_df = pd.DataFrame(results)
    
    # Test significance for each category
    print("\nSignificance Tests (Large vs Small Cities) by Category:")
    print("-" * 80)
    
    significant_categories = []
    
    for category in ALL_CATEGORIES:
        cat_data = category_df[category_df['Category'] == category]
        
        if len(cat_data) == 0:
            continue
        
        large_sentiments = cat_data[cat_data['City_Size'] == 'Large'][sentiment_col].values
        small_sentiments = cat_data[cat_data['City_Size'] == 'Small'][sentiment_col].values
        
        if len(large_sentiments) >= 2 and len(small_sentiments) >= 2:
            statistic, p_value = test_significance(large_sentiments, small_sentiments)
            
            if p_value is not None:
                # Bonferroni correction: p < 0.05/16 = 0.003125
                bonf_sig = p_value < BONFERRONI_ALPHA
                uncorrected_sig = p_value < 0.05
                
                if bonf_sig:
                    sig_marker = "*** (Bonf)"
                elif uncorrected_sig:
                    sig_marker = "* (uncorrected)"
                else:
                    sig_marker = ""
                
                print(f"{category:50s} p={p_value:.4f} {sig_marker}")
                
                # Store all categories with their p-values for comprehensive chart
                significant_categories.append({
                    'Category': category,
                    'p_value': p_value,
                    'Large_Mean': large_sentiments.mean(),
                    'Small_Mean': small_sentiments.mean(),
                    'Bonferroni_Significant': bonf_sig,
                    'Uncorrected_Significant': uncorrected_sig
                })
    
    bonf_count = sum(1 for cat in significant_categories if cat.get('Bonferroni_Significant', False))
    uncorrected_count = sum(1 for cat in significant_categories if cat.get('Uncorrected_Significant', False))
    
    if significant_categories:
        print(f"\n{bonf_count} categories show significant differences (Bonferroni corrected: p < {BONFERRONI_ALPHA:.6f})")
        print(f"{uncorrected_count} categories show significant differences (uncorrected: p < 0.05)")
    
    return results_df, significant_categories

def analyze_by_source(category_df, sentiment_col='VADER_Sentiment'):
    """Analyze sentiment by data source.
    Uses VADER_Sentiment by default, but can use TextBlob_Sentiment if specified.
    """
    
    print("\n" + "="*80)
    print("SENTIMENT ANALYSIS BY DATA SOURCE")
    print("="*80)
    
    source_stats = calculate_sentiment_stats(category_df, ['Source'], sentiment_col=sentiment_col)
    source_stats = source_stats.sort_values('Mean_Sentiment', ascending=False)
    
    print("\nSentiment by Source:")
    print(source_stats.to_string(index=False))
    
    return source_stats

def analyze_by_category_group(category_df, sentiment_col='VADER_Sentiment'):
    """Analyze sentiment by category groups.
    Uses VADER_Sentiment by default, but can use TextBlob_Sentiment if specified.
    """
    
    print("\n" + "="*80)
    print("SENTIMENT ANALYSIS BY CATEGORY GROUP")
    print("="*80)
    
    results = []
    
    for group_name, categories in CATEGORY_GROUPS.items():
        group_data = category_df[category_df['Category'].isin(categories)]
        
        if len(group_data) == 0:
            continue
        
        sentiments = group_data[sentiment_col].values
        n = len(sentiments)
        
        positive_pct = (sentiments == 1).sum() / n * 100
        neutral_pct = (sentiments == 0).sum() / n * 100
        negative_pct = (sentiments == -1).sum() / n * 100
        mean_sentiment = sentiments.mean()
        
        results.append({
            'Category_Group': group_name,
            'N': n,
            'Positive_%': positive_pct,
            'Neutral_%': neutral_pct,
            'Negative_%': negative_pct,
            'Mean_Sentiment': mean_sentiment
        })
    
    results_df = pd.DataFrame(results).sort_values('Mean_Sentiment', ascending=False)
    
    print("\nSentiment by Category Group:")
    print(results_df.to_string(index=False))
    
    return results_df

def calculate_standard_errors(df, group_col, value_col='VADER_Sentiment'):
    """Calculate standard errors for sentiment by group."""
    se_dict = {}
    for group_key, group_df in df.groupby(group_col):
        sentiments = group_df[value_col].values
        if len(sentiments) > 1:
            se = np.std(sentiments, ddof=1) / np.sqrt(len(sentiments))
        else:
            se = 0
        se_dict[group_key] = se
    return se_dict

def create_visualizations(output_dir, category_results, city_results, city_size_results, 
                        category_city_size_results, source_results, category_group_results,
                        significant_categories, category_df=None, sentiment_col='VADER_Sentiment'):
    """Create PDF charts for all sentiment analyses with error bars and significance.
    Uses VADER_Sentiment by default, but can use TextBlob_Sentiment if specified.
    """
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Create charts directory
    charts_dir = output_dir / 'charts'
    charts_dir.mkdir(exist_ok=True)

    def _save_pdf_png(path_no_ext: Path, dpi: int = 300):
        """Save current matplotlib figure as both PDF and PNG."""
        plt.savefig(path_no_ext.with_suffix('.pdf'), dpi=dpi, bbox_inches='tight')
        plt.savefig(path_no_ext.with_suffix('.png'), dpi=dpi, bbox_inches='tight')
    
    # 1. Sentiment by Category
    print("\nCreating sentiment by category chart...")
    fig, ax = plt.subplots(figsize=(14, 8))
    category_results_sorted = category_results.sort_values('Mean_Sentiment', ascending=True)
    categories_short = [cat.replace('Comment_', '').replace('Critique_', '').replace('Response_', '')
                       .replace('Perception_', '').replace('Racist_Flag', 'Racist') 
                       for cat in category_results_sorted['Category']]
    
    colors = ['#2ecc71' if x > 0 else '#e74c3c' if x < 0 else '#95a5a6' 
              for x in category_results_sorted['Mean_Sentiment']]
    
    # Calculate standard errors if we have raw data
    if category_df is not None:
        se_dict = calculate_standard_errors(category_df, 'Category', value_col=sentiment_col)
        errors = [se_dict.get(cat, 0) for cat in category_results_sorted['Category']]
        bars = ax.barh(range(len(category_results_sorted)), category_results_sorted['Mean_Sentiment'], 
                       color=colors, alpha=0.7, edgecolor='black', linewidth=0.5,
                       xerr=errors, capsize=3)
    else:
        bars = ax.barh(range(len(category_results_sorted)), category_results_sorted['Mean_Sentiment'], 
                       color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(category_results_sorted)))
    ax.set_yticklabels(categories_short, fontsize=9)
    ax.set_xlabel('Mean Sentiment Score', fontsize=12, fontweight='bold')
    ax.set_title('Sentiment Analysis by Category', fontsize=14, fontweight='bold', pad=20)
    # Set x-axis limits to show full range
    min_sent = category_results_sorted['Mean_Sentiment'].min()
    max_sent = category_results_sorted['Mean_Sentiment'].max()
    max_err = max(errors) if category_df is not None and len(errors) > 0 and all(e is not None for e in errors) else 0
    ax.set_xlim(min(min_sent - 0.1 - max_err, -1), max(max_sent + 0.1 + max_err, 1))
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='x', alpha=0.5, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add value labels (positioned to avoid error bars)
    for i, (idx, row) in enumerate(category_results_sorted.iterrows()):
        err_val = errors[i] if category_df is not None and i < len(errors) else 0
        # Add extra spacing to avoid overlapping with error bars
        spacing = max(0.05, err_val * 1.5) if err_val > 0 else 0.05
        label_x = row['Mean_Sentiment'] + spacing if row['Mean_Sentiment'] >= 0 else row['Mean_Sentiment'] - spacing
        ax.text(label_x, i, f" {row['Mean_Sentiment']:.3f}", 
               va='center', fontsize=8)
    
    plt.tight_layout()
    _save_pdf_png(charts_dir / 'sentiment_by_category', dpi=300)
    plt.close()
    print("  Saved: sentiment_by_category.pdf and sentiment_by_category.png")
    
    # 2. Sentiment by City (color coded by city size)
    print("Creating sentiment by city chart...")
    fig, ax = plt.subplots(figsize=(12, 6))
    city_results_sorted = city_results.sort_values('Mean_Sentiment', ascending=True)
    
    # Color code by city size
    city_colors = []
    city_labels = []
    for city in city_results_sorted['City']:
        city_lower = str(city).lower().strip()
        if city_lower in [c.lower() for c in LARGE_CITIES]:
            city_colors.append('#3498db')  # Blue for large cities
            city_labels.append('Large')
        elif city_lower in [c.lower() for c in SMALL_CITIES]:
            city_colors.append('#e67e22')  # Orange for small cities
            city_labels.append('Small')
        else:
            city_colors.append('#95a5a6')  # Gray for unknown
            city_labels.append('Unknown')
    
    # Calculate standard errors if we have raw data
    if category_df is not None:
        se_dict = calculate_standard_errors(category_df, 'City', value_col=sentiment_col)
        errors = [se_dict.get(city, 0) for city in city_results_sorted['City']]
        bars = ax.barh(range(len(city_results_sorted)), city_results_sorted['Mean_Sentiment'], 
                       color=city_colors, alpha=0.7, edgecolor='black', linewidth=0.5,
                       xerr=errors, capsize=3)
    else:
        bars = ax.barh(range(len(city_results_sorted)), city_results_sorted['Mean_Sentiment'], 
                       color=city_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(range(len(city_results_sorted)))
    ax.set_yticklabels(city_results_sorted['City'], fontsize=10)
    ax.set_xlabel('Mean Sentiment Score', fontsize=12, fontweight='bold')
    ax.set_title('Sentiment Analysis by City (Color Coded by City Size)', fontsize=14, fontweight='bold', pad=20)
    # Set x-axis limits
    min_sent = city_results_sorted['Mean_Sentiment'].min()
    max_sent = city_results_sorted['Mean_Sentiment'].max()
    max_err = max(errors) if category_df is not None and len(errors) > 0 and all(e is not None for e in errors) else 0
    ax.set_xlim(min(min_sent - 0.05 - max_err, -1), max(max_sent + 0.05 + max_err, 1))
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='x', alpha=0.5, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', alpha=0.7, edgecolor='black', label='Large Cities'),
        Patch(facecolor='#e67e22', alpha=0.7, edgecolor='black', label='Small Cities')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=10)
    
    plt.tight_layout()
    _save_pdf_png(charts_dir / 'sentiment_by_city', dpi=300)
    plt.close()
    print("  Saved: sentiment_by_city.pdf and sentiment_by_city.png")
    
    # 3. Sentiment by City Size
    print("Creating sentiment by city size chart...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Mean sentiment comparison with error bars and significance
    ax1 = axes[0]
    city_size_results_sorted = city_size_results.sort_values('Mean_Sentiment', ascending=True)
    colors = ['#2ecc71' if x > 0.4 else '#e67e22' for x in city_size_results_sorted['Mean_Sentiment']]
    
    # Calculate standard errors if we have raw data
    if category_df is not None:
        se_dict = calculate_standard_errors(category_df, 'City_Size', value_col=sentiment_col)
        errors = [se_dict.get(size, 0) for size in city_size_results_sorted['City_Size']]
        bars = ax1.barh(city_size_results_sorted['City_Size'], city_size_results_sorted['Mean_Sentiment'],
                       color=colors, alpha=0.7, edgecolor='black', linewidth=1,
                       xerr=errors, capsize=5)
        
        # Test significance between groups
        if len(city_size_results) == 2:
            large_sentiments = category_df[category_df['City_Size'] == 'Large'][sentiment_col].values
            small_sentiments = category_df[category_df['City_Size'] == 'Small'][sentiment_col].values
            if len(large_sentiments) >= 2 and len(small_sentiments) >= 2:
                statistic, p_value = test_significance(large_sentiments, small_sentiments)
                if p_value is not None:
                    sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    ax1.text(0.5, 0.02, f'Large vs Small: p={p_value:.4f} {sig_marker}', 
                           transform=ax1.transAxes, fontsize=10, fontweight='bold',
                           ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    else:
        bars = ax1.barh(city_size_results_sorted['City_Size'], city_size_results_sorted['Mean_Sentiment'],
                       color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    ax1.set_xlabel('Mean Sentiment Score', fontsize=12, fontweight='bold')
    ax1.set_title('Mean Sentiment by City Size', fontsize=13, fontweight='bold')
    # Set x-axis limits
    min_sent = city_size_results_sorted['Mean_Sentiment'].min()
    max_sent = city_size_results_sorted['Mean_Sentiment'].max()
    ax1.set_xlim(min(min_sent - 0.05, -1), max(max_sent + 0.05, 1))
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.grid(axis='x', alpha=0.5, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)
    
    # Percentage breakdown with error bars
    ax2 = axes[1]
    x = np.arange(len(city_size_results))
    width = 0.25
    
    # Calculate standard errors for percentages if we have raw data
    if category_df is not None:
        pos_errors = []
        neu_errors = []
        neg_errors = []
        for size in city_size_results['City_Size']:
            subset = category_df[category_df['City_Size'] == size]
            n = len(subset)
            if n > 1:
                pos_pct = (subset[sentiment_col] == 1).sum() / n
                neu_pct = (subset[sentiment_col] == 0).sum() / n
                neg_pct = (subset[sentiment_col] == -1).sum() / n
                # Standard error for proportion
                pos_se = np.sqrt(pos_pct * (1 - pos_pct) / n) * 100
                neu_se = np.sqrt(neu_pct * (1 - neu_pct) / n) * 100
                neg_se = np.sqrt(neg_pct * (1 - neg_pct) / n) * 100
                pos_errors.append(pos_se)
                neu_errors.append(neu_se)
                neg_errors.append(neg_se)
            else:
                pos_errors.append(0)
                neu_errors.append(0)
                neg_errors.append(0)
        
        ax2.bar(x - width, city_size_results['Positive_%'], width, label='Positive', 
               color='#2ecc71', alpha=0.7, edgecolor='black', yerr=pos_errors, capsize=3)
        ax2.bar(x, city_size_results['Neutral_%'], width, label='Neutral', 
               color='#95a5a6', alpha=0.7, edgecolor='black', yerr=neu_errors, capsize=3)
        ax2.bar(x + width, city_size_results['Negative_%'], width, label='Negative', 
               color='#e74c3c', alpha=0.7, edgecolor='black', yerr=neg_errors, capsize=3)
    else:
        ax2.bar(x - width, city_size_results['Positive_%'], width, label='Positive', 
               color='#2ecc71', alpha=0.7, edgecolor='black')
        ax2.bar(x, city_size_results['Neutral_%'], width, label='Neutral', 
               color='#95a5a6', alpha=0.7, edgecolor='black')
        ax2.bar(x + width, city_size_results['Negative_%'], width, label='Negative', 
               color='#e74c3c', alpha=0.7, edgecolor='black')
    
    ax2.set_xlabel('City Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Sentiment Distribution by City Size', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(city_size_results['City_Size'])
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.5, linestyle='-', linewidth=0.5)
    ax2.set_axisbelow(True)
    
    plt.tight_layout()
    _save_pdf_png(charts_dir / 'sentiment_by_city_size', dpi=300)
    plt.close()
    print("  Saved: sentiment_by_city_size.pdf and sentiment_by_city_size.png")
    
    # 4. Sentiment by Source
    print("Creating sentiment by source chart...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Mean sentiment with error bars
    ax1 = axes[0]
    source_results_sorted = source_results.sort_values('Mean_Sentiment', ascending=True)
    colors = ['#9b59b6', '#3498db', '#e67e22', '#1abc9c']
    
    # Calculate standard errors if we have raw data
    if category_df is not None:
        se_dict = calculate_standard_errors(category_df, 'Source', value_col=sentiment_col)
        errors = [se_dict.get(source, 0) for source in source_results_sorted['Source']]
        bars = ax1.barh(source_results_sorted['Source'], source_results_sorted['Mean_Sentiment'],
                       color=colors[:len(source_results_sorted)], alpha=0.7, edgecolor='black', linewidth=1,
                       xerr=errors, capsize=5)
    else:
        bars = ax1.barh(source_results_sorted['Source'], source_results_sorted['Mean_Sentiment'],
                       color=colors[:len(source_results_sorted)], alpha=0.7, edgecolor='black', linewidth=1)
    
    ax1.set_xlabel('Mean Sentiment Score', fontsize=12, fontweight='bold')
    ax1.set_title('Mean Sentiment by Data Source', fontsize=13, fontweight='bold')
    # Set x-axis limits
    min_sent = source_results_sorted['Mean_Sentiment'].min()
    max_sent = source_results_sorted['Mean_Sentiment'].max()
    ax1.set_xlim(min(min_sent - 0.05, -1), max(max_sent + 0.05, 1))
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.grid(axis='x', alpha=0.5, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)
    
    # Percentage breakdown with error bars
    ax2 = axes[1]
    x = np.arange(len(source_results))
    width = 0.25
    
    # Calculate standard errors for percentages if we have raw data
    if category_df is not None:
        pos_errors = []
        neu_errors = []
        neg_errors = []
        for source in source_results['Source']:
            subset = category_df[category_df['Source'] == source]
            n = len(subset)
            if n > 1:
                pos_pct = (subset[sentiment_col] == 1).sum() / n
                neu_pct = (subset[sentiment_col] == 0).sum() / n
                neg_pct = (subset[sentiment_col] == -1).sum() / n
                # Standard error for proportion
                pos_se = np.sqrt(pos_pct * (1 - pos_pct) / n) * 100
                neu_se = np.sqrt(neu_pct * (1 - neu_pct) / n) * 100
                neg_se = np.sqrt(neg_pct * (1 - neg_pct) / n) * 100
                pos_errors.append(pos_se)
                neu_errors.append(neu_se)
                neg_errors.append(neg_se)
            else:
                pos_errors.append(0)
                neu_errors.append(0)
                neg_errors.append(0)
        
        ax2.bar(x - width, source_results['Positive_%'], width, label='Positive', 
               color='#2ecc71', alpha=0.7, edgecolor='black', yerr=pos_errors, capsize=3)
        ax2.bar(x, source_results['Neutral_%'], width, label='Neutral', 
               color='#95a5a6', alpha=0.7, edgecolor='black', yerr=neu_errors, capsize=3)
        ax2.bar(x + width, source_results['Negative_%'], width, label='Negative', 
               color='#e74c3c', alpha=0.7, edgecolor='black', yerr=neg_errors, capsize=3)
    else:
        ax2.bar(x - width, source_results['Positive_%'], width, label='Positive', 
               color='#2ecc71', alpha=0.7, edgecolor='black')
        ax2.bar(x, source_results['Neutral_%'], width, label='Neutral', 
               color='#95a5a6', alpha=0.7, edgecolor='black')
        ax2.bar(x + width, source_results['Negative_%'], width, label='Negative', 
               color='#e74c3c', alpha=0.7, edgecolor='black')
    
    ax2.set_xlabel('Data Source', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Sentiment Distribution by Data Source', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(source_results['Source'], rotation=15, ha='right')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.5, linestyle='-', linewidth=0.5)
    ax2.set_axisbelow(True)
    
    plt.tight_layout()
    _save_pdf_png(charts_dir / 'sentiment_by_source', dpi=300)
    plt.close()
    print("  Saved: sentiment_by_source.pdf and sentiment_by_source.png")
    
    # 5. Sentiment by Category Group
    print("Creating sentiment by category group chart...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Mean sentiment with error bars
    ax1 = axes[0]
    category_group_results_sorted = category_group_results.sort_values('Mean_Sentiment', ascending=True)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(category_group_results_sorted)))
    
    # Calculate standard errors if we have raw data
    if category_df is not None:
        # Map category groups to individual categories for error calculation
        group_to_cats = {}
        for group_name, categories in CATEGORY_GROUPS.items():
            group_to_cats[group_name] = categories
        
        errors = []
        for group in category_group_results_sorted['Category_Group']:
            if group in group_to_cats:
                # Get all data for categories in this group
                group_data = category_df[category_df['Category'].isin(group_to_cats[group])]
                if len(group_data) > 1:
                    se = np.std(group_data[sentiment_col].values, ddof=1) / np.sqrt(len(group_data))
                else:
                    se = 0
                errors.append(se)
            else:
                errors.append(0)
        
        bars = ax1.barh(category_group_results_sorted['Category_Group'], 
                       category_group_results_sorted['Mean_Sentiment'],
                       color=colors, alpha=0.7, edgecolor='black', linewidth=1,
                       xerr=errors, capsize=5)
    else:
        bars = ax1.barh(category_group_results_sorted['Category_Group'], 
                       category_group_results_sorted['Mean_Sentiment'],
                       color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    ax1.set_xlabel('Mean Sentiment Score', fontsize=12, fontweight='bold')
    ax1.set_title('Mean Sentiment by Category Group', fontsize=13, fontweight='bold')
    # Set x-axis limits
    min_sent = category_group_results_sorted['Mean_Sentiment'].min()
    max_sent = category_group_results_sorted['Mean_Sentiment'].max()
    ax1.set_xlim(min(min_sent - 0.05, -1), max(max_sent + 0.05, 1))
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.grid(axis='x', alpha=0.5, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)
    
    # Percentage breakdown with error bars
    ax2 = axes[1]
    x = np.arange(len(category_group_results))
    width = 0.25
    
    if category_df is not None:
        pos_errors = []
        neu_errors = []
        neg_errors = []
        for group in category_group_results['Category_Group']:
            if group in CATEGORY_GROUPS:
                group_data = category_df[category_df['Category'].isin(CATEGORY_GROUPS[group])]
                n = len(group_data)
                if n > 1:
                    pos_pct = (group_data[sentiment_col] == 1).sum() / n
                    neu_pct = (group_data[sentiment_col] == 0).sum() / n
                    neg_pct = (group_data[sentiment_col] == -1).sum() / n
                    pos_se = np.sqrt(pos_pct * (1 - pos_pct) / n) * 100
                    neu_se = np.sqrt(neu_pct * (1 - neu_pct) / n) * 100
                    neg_se = np.sqrt(neg_pct * (1 - neg_pct) / n) * 100
                    pos_errors.append(pos_se)
                    neu_errors.append(neu_se)
                    neg_errors.append(neg_se)
                else:
                    pos_errors.append(0)
                    neu_errors.append(0)
                    neg_errors.append(0)
            else:
                pos_errors.append(0)
                neu_errors.append(0)
                neg_errors.append(0)
        
        ax2.bar(x - width, category_group_results['Positive_%'], width, label='Positive', 
               color='#2ecc71', alpha=0.7, edgecolor='black', yerr=pos_errors, capsize=3)
        ax2.bar(x, category_group_results['Neutral_%'], width, label='Neutral', 
               color='#95a5a6', alpha=0.7, edgecolor='black', yerr=neu_errors, capsize=3)
        ax2.bar(x + width, category_group_results['Negative_%'], width, label='Negative', 
               color='#e74c3c', alpha=0.7, edgecolor='black', yerr=neg_errors, capsize=3)
    else:
        ax2.bar(x - width, category_group_results['Positive_%'], width, label='Positive', 
               color='#2ecc71', alpha=0.7, edgecolor='black')
        ax2.bar(x, category_group_results['Neutral_%'], width, label='Neutral', 
               color='#95a5a6', alpha=0.7, edgecolor='black')
        ax2.bar(x + width, category_group_results['Negative_%'], width, label='Negative', 
               color='#e74c3c', alpha=0.7, edgecolor='black')
    
    ax2.set_xlabel('Category Group', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Sentiment Distribution by Category Group', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(category_group_results['Category_Group'], rotation=15, ha='right')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.5, linestyle='-', linewidth=0.5)
    ax2.set_axisbelow(True)
    
    plt.tight_layout()
    _save_pdf_png(charts_dir / 'sentiment_by_category_group', dpi=300)
    plt.close()
    print("  Saved: sentiment_by_category_group.pdf and sentiment_by_category_group.png")
    
    # 6. Category x City Size Heatmap
    print("Creating category x city size heatmap...")
    if len(category_city_size_results) > 0:
        try:
            pivot_data = category_city_size_results.pivot(index='Category', columns='City_Size', 
                                                          values='Mean_Sentiment')
            # Shorten category names for display
            pivot_data.index = [cat.replace('Comment_', '').replace('Critique_', '').replace('Response_', '')
                               .replace('Perception_', '').replace('Racist_Flag', 'Racist')[:30]
                               for cat in pivot_data.index]
            
            if len(pivot_data) > 0 and len(pivot_data.columns) > 0:
                fig, ax = plt.subplots(figsize=(10, max(8, len(pivot_data) * 0.3)))
                sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                           cbar_kws={'label': 'Mean Sentiment'}, ax=ax, linewidths=0.5)
                ax.set_title('Sentiment by Category and City Size', fontsize=14, fontweight='bold', pad=20)
                ax.set_xlabel('City Size', fontsize=12, fontweight='bold')
                ax.set_ylabel('Category', fontsize=12, fontweight='bold')
                
                plt.tight_layout()
                _save_pdf_png(charts_dir / 'sentiment_category_city_size_heatmap', dpi=300)
                plt.close()
                print("  Saved: sentiment_category_city_size_heatmap.pdf and sentiment_category_city_size_heatmap.png")
        except Exception as e:
            print(f"  Warning: Could not create heatmap: {e}")
    
    # 7. Sentiment by City Grouping (Overall)
    print("Creating sentiment by city grouping chart...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Mean sentiment comparison with error bars and significance
    ax1 = axes[0]
    city_size_results_sorted = city_size_results.sort_values('Mean_Sentiment', ascending=False)
    colors = ['#2ecc71' if x > 0.4 else '#e67e22' for x in city_size_results_sorted['Mean_Sentiment']]
    
    # Calculate standard errors and test significance
    if category_df is not None:
        se_dict = calculate_standard_errors(category_df, 'City_Size', value_col=sentiment_col)
        errors = [se_dict.get(size, 0) for size in city_size_results_sorted['City_Size']]
        bars = ax1.bar(city_size_results_sorted['City_Size'], city_size_results_sorted['Mean_Sentiment'],
                       color=colors, alpha=0.7, edgecolor='black', linewidth=1,
                       yerr=errors, capsize=5)
        
        # Test significance between groups
        if len(city_size_results) == 2:
            large_sentiments = category_df[category_df['City_Size'] == 'Large'][sentiment_col].values
            small_sentiments = category_df[category_df['City_Size'] == 'Small'][sentiment_col].values
            if len(large_sentiments) >= 2 and len(small_sentiments) >= 2:
                statistic, p_value = test_significance(large_sentiments, small_sentiments)
                if p_value is not None:
                    sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    # Add significance bracket
                    max_err_val = max(errors) if len(errors) > 0 and all(e is not None for e in errors) else 0
                    max_val = max(city_size_results_sorted['Mean_Sentiment'].max() + max_err_val, 
                                 city_size_results_sorted['Mean_Sentiment'].max() + 0.1)
                    ax1.plot([0, 1], [max_val, max_val], 'k-', linewidth=1.5)
                    ax1.plot([0, 0], [max_val - 0.02, max_val], 'k-', linewidth=1.5)
                    ax1.plot([1, 1], [max_val - 0.02, max_val], 'k-', linewidth=1.5)
                    ax1.text(0.5, max_val + 0.02, f'p={p_value:.4f} {sig_marker}', 
                           ha='center', fontsize=10, fontweight='bold')
    else:
        bars = ax1.bar(city_size_results_sorted['City_Size'], city_size_results_sorted['Mean_Sentiment'],
                       color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    ax1.set_ylabel('Mean Sentiment Score', fontsize=12, fontweight='bold')
    ax1.set_xlabel('City Size', fontsize=12, fontweight='bold')
    ax1.set_title('Mean Sentiment by City Grouping', fontsize=13, fontweight='bold')
    # Set y-axis limits
    min_sent = city_size_results_sorted['Mean_Sentiment'].min()
    max_sent = city_size_results_sorted['Mean_Sentiment'].max()
    max_err = max(errors) if category_df is not None and len(errors) > 0 and all(e is not None for e in errors) else 0
    ax1.set_ylim(min(min_sent - 0.05 - max_err, -1), max(max_sent + 0.05 + max_err, 1))
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.grid(axis='y', alpha=0.5, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(city_size_results_sorted.iterrows()):
        err_val = errors[i] if category_df is not None and i < len(errors) else 0
        label_y = row['Mean_Sentiment'] + err_val + 0.01 if row['Mean_Sentiment'] >= 0 else row['Mean_Sentiment'] - err_val - 0.01
        ax1.text(i, label_y, f"{row['Mean_Sentiment']:.3f}", ha='center', 
                va='bottom' if row['Mean_Sentiment'] >= 0 else 'top',
                fontsize=11, fontweight='bold')
    
    # Percentage breakdown with error bars
    ax2 = axes[1]
    x = np.arange(len(city_size_results))
    width = 0.25
    
    # Calculate standard errors for percentages if we have raw data
    if category_df is not None:
        pos_errors = []
        neu_errors = []
        neg_errors = []
        for size in city_size_results['City_Size']:
            subset = category_df[category_df['City_Size'] == size]
            n = len(subset)
            if n > 1:
                pos_pct = (subset[sentiment_col] == 1).sum() / n
                neu_pct = (subset[sentiment_col] == 0).sum() / n
                neg_pct = (subset[sentiment_col] == -1).sum() / n
                pos_se = np.sqrt(pos_pct * (1 - pos_pct) / n) * 100
                neu_se = np.sqrt(neu_pct * (1 - neu_pct) / n) * 100
                neg_se = np.sqrt(neg_pct * (1 - neg_pct) / n) * 100
                pos_errors.append(pos_se)
                neu_errors.append(neu_se)
                neg_errors.append(neg_se)
            else:
                pos_errors.append(0)
                neu_errors.append(0)
                neg_errors.append(0)
        
        ax2.bar(x - width, city_size_results['Positive_%'], width, label='Positive', 
               color='#2ecc71', alpha=0.7, edgecolor='black', yerr=pos_errors, capsize=3)
        ax2.bar(x, city_size_results['Neutral_%'], width, label='Neutral', 
               color='#95a5a6', alpha=0.7, edgecolor='black', yerr=neu_errors, capsize=3)
        ax2.bar(x + width, city_size_results['Negative_%'], width, label='Negative', 
               color='#e74c3c', alpha=0.7, edgecolor='black', yerr=neg_errors, capsize=3)
    else:
        ax2.bar(x - width, city_size_results['Positive_%'], width, label='Positive', 
               color='#2ecc71', alpha=0.7, edgecolor='black')
        ax2.bar(x, city_size_results['Neutral_%'], width, label='Neutral', 
               color='#95a5a6', alpha=0.7, edgecolor='black')
        ax2.bar(x + width, city_size_results['Negative_%'], width, label='Negative', 
               color='#e74c3c', alpha=0.7, edgecolor='black')
    
    ax2.set_xlabel('City Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Sentiment Distribution by City Grouping', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(city_size_results['City_Size'])
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.5, linestyle='-', linewidth=0.5)
    ax2.set_axisbelow(True)
    
    plt.tight_layout()
    _save_pdf_png(charts_dir / 'sentiment_by_city_grouping', dpi=300)
    plt.close()
    print("  Saved: sentiment_by_city_grouping.pdf and sentiment_by_city_grouping.png")
    
    # 8. Sentiment for All 16 Labels Chart
    print("Creating sentiment for all 16 labels chart...")
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Top: Mean sentiment bars with error bars
    ax1 = axes[0]
    category_results_sorted = category_results.sort_values('Mean_Sentiment', ascending=True)
    categories_short = [cat.replace('Comment_', '').replace('Critique_', '').replace('Response_', '')
                       .replace('Perception_', '').replace('Racist_Flag', 'Racist') 
                       for cat in category_results_sorted['Category']]
    
    colors = ['#2ecc71' if x > 0.3 else '#e67e22' if x > 0 else '#e74c3c' 
              for x in category_results_sorted['Mean_Sentiment']]
    
    # Calculate standard errors if we have raw data
    if category_df is not None:
        se_dict = calculate_standard_errors(category_df, 'Category', value_col=sentiment_col)
        errors = [se_dict.get(cat, 0) for cat in category_results_sorted['Category']]
        bars = ax1.barh(range(len(category_results_sorted)), category_results_sorted['Mean_Sentiment'], 
                       color=colors, alpha=0.7, edgecolor='black', linewidth=0.8,
                       xerr=errors, capsize=3)
    else:
        bars = ax1.barh(range(len(category_results_sorted)), category_results_sorted['Mean_Sentiment'], 
                       color=colors, alpha=0.7, edgecolor='black', linewidth=0.8)
    
    ax1.set_yticks(range(len(category_results_sorted)))
    ax1.set_yticklabels(categories_short, fontsize=10)
    ax1.set_xlabel('Mean Sentiment Score', fontsize=12, fontweight='bold')
    ax1.set_title('Sentiment Analysis: All 16 Category Labels (Mean Sentiment)', 
                 fontsize=14, fontweight='bold', pad=20)
    # Set x-axis limits
    min_sent = category_results_sorted['Mean_Sentiment'].min()
    max_sent = category_results_sorted['Mean_Sentiment'].max()
    max_err = max(errors) if category_df is not None and len(errors) > 0 and all(e is not None for e in errors) else 0
    ax1.set_xlim(min(min_sent - 0.1 - max_err, -1), max(max_sent + 0.1 + max_err, 1))
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.grid(axis='x', alpha=0.5, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)
    
    # Add value labels (positioned to avoid error bars)
    for i, (idx, row) in enumerate(category_results_sorted.iterrows()):
        err_val = errors[i] if category_df is not None and i < len(errors) else 0
        # Add extra spacing to avoid overlapping with error bars
        spacing = max(0.05, err_val * 1.5) if err_val > 0 else 0.05
        label_x = row['Mean_Sentiment'] + spacing if row['Mean_Sentiment'] >= 0 else row['Mean_Sentiment'] - spacing
        ax1.text(label_x, i, f" {row['Mean_Sentiment']:.3f} (n={row['N']:,})", 
               va='center', fontsize=9, fontweight='bold')
    
    # Bottom: Percentage breakdown with error bars
    ax2 = axes[1]
    x = np.arange(len(category_results_sorted))
    width = 0.25
    
    # Sort by mean sentiment for consistency
    category_results_for_pct = category_results.sort_values('Mean_Sentiment', ascending=True)
    
    # Calculate standard errors for percentages if we have raw data
    if category_df is not None:
        pos_errors = []
        neu_errors = []
        neg_errors = []
        for cat in category_results_for_pct['Category']:
            subset = category_df[category_df['Category'] == cat]
            n = len(subset)
            if n > 1:
                pos_pct = (subset[sentiment_col] == 1).sum() / n
                neu_pct = (subset[sentiment_col] == 0).sum() / n
                neg_pct = (subset[sentiment_col] == -1).sum() / n
                pos_se = np.sqrt(pos_pct * (1 - pos_pct) / n) * 100
                neu_se = np.sqrt(neu_pct * (1 - neu_pct) / n) * 100
                neg_se = np.sqrt(neg_pct * (1 - neg_pct) / n) * 100
                pos_errors.append(pos_se)
                neu_errors.append(neu_se)
                neg_errors.append(neg_se)
            else:
                pos_errors.append(0)
                neu_errors.append(0)
                neg_errors.append(0)
        
        bars1 = ax2.bar(x - width, category_results_for_pct['Positive_%'], width, 
                       label='Positive', color='#2ecc71', alpha=0.7, edgecolor='black',
                       yerr=pos_errors, capsize=2)
        bars2 = ax2.bar(x, category_results_for_pct['Neutral_%'], width, 
                       label='Neutral', color='#95a5a6', alpha=0.7, edgecolor='black',
                       yerr=neu_errors, capsize=2)
        bars3 = ax2.bar(x + width, category_results_for_pct['Negative_%'], width, 
                       label='Negative', color='#e74c3c', alpha=0.7, edgecolor='black',
                       yerr=neg_errors, capsize=2)
    else:
        bars1 = ax2.bar(x - width, category_results_for_pct['Positive_%'], width, 
                       label='Positive', color='#2ecc71', alpha=0.7, edgecolor='black')
        bars2 = ax2.bar(x, category_results_for_pct['Neutral_%'], width, 
                       label='Neutral', color='#95a5a6', alpha=0.7, edgecolor='black')
        bars3 = ax2.bar(x + width, category_results_for_pct['Negative_%'], width, 
                       label='Negative', color='#e74c3c', alpha=0.7, edgecolor='black')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories_short, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Sentiment Distribution: All 16 Category Labels (Percentage Breakdown)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.5, linestyle='-', linewidth=0.5)
    ax2.set_axisbelow(True)
    
    plt.tight_layout()
    _save_pdf_png(charts_dir / 'sentiment_all_16_labels', dpi=300)
    plt.close()
    print("  Saved: sentiment_all_16_labels.pdf and sentiment_all_16_labels.png")
    
    # 9. Significant Differences Chart (only Bonferroni significant)
    if significant_categories and len(significant_categories) > 0:
        print("Creating significant differences chart (Bonferroni corrected)...")
        sig_df = pd.DataFrame(significant_categories)
        # Filter to only Bonferroni significant
        bonf_sig_df = sig_df[sig_df.get('Bonferroni_Significant', False)].copy()
        
        if len(bonf_sig_df) > 0:
            bonf_sig_df = bonf_sig_df.sort_values('p_value', ascending=True)
            
            fig, ax = plt.subplots(figsize=(12, max(6, len(bonf_sig_df) * 0.4)))
            
            categories_short = [cat.replace('Comment_', '').replace('Critique_', '').replace('Response_', '')
                               .replace('Perception_', '').replace('Racist_Flag', 'Racist')[:40]
                               for cat in bonf_sig_df['Category']]
            
            x = np.arange(len(bonf_sig_df))
            width = 0.35
            
            bars1 = ax.barh(x - width/2, bonf_sig_df['Large_Mean'], width, label='Large Cities', 
                           color='#3498db', alpha=0.7, edgecolor='black')
            bars2 = ax.barh(x + width/2, bonf_sig_df['Small_Mean'], width, label='Small Cities', 
                           color='#e67e22', alpha=0.7, edgecolor='black')
            
            ax.set_yticks(x)
            ax.set_yticklabels(categories_short, fontsize=9)
            ax.set_xlabel('Mean Sentiment Score', fontsize=12, fontweight='bold')
            ax.set_title(f'Significant Differences: Large vs Small Cities (Bonferroni corrected, p < {BONFERRONI_ALPHA:.6f})', 
                        fontsize=13, fontweight='bold', pad=20)
            # Set x-axis limits
            all_means = list(bonf_sig_df['Large_Mean']) + list(bonf_sig_df['Small_Mean'])
            min_sent = min(all_means)
            max_sent = max(all_means)
            ax.set_xlim(min(min_sent - 0.05, -1), max(max_sent + 0.05, 1))
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax.legend()
            ax.grid(axis='x', alpha=0.5, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            
            # Add p-value annotations
            for i, (idx, row) in enumerate(bonf_sig_df.iterrows()):
                ax.text(max(row['Large_Mean'], row['Small_Mean']) + 0.05, i, 
                       f"p={row['p_value']:.4f}", va='center', fontsize=8)
            
            plt.tight_layout()
            _save_pdf_png(charts_dir / 'significant_differences', dpi=300)
            plt.close()
            print("  Saved: significant_differences.pdf and significant_differences.png")
        else:
            print("  No Bonferroni-corrected significant differences found")
    
    # 10. All Categories Comparison Chart (showing all with significance indicators)
    if significant_categories and len(significant_categories) > 0:
        print("Creating all categories comparison chart...")
        all_cats_df = pd.DataFrame(significant_categories)
        all_cats_df = all_cats_df.sort_values('p_value', ascending=True)
        
        fig, ax = plt.subplots(figsize=(14, max(8, len(all_cats_df) * 0.4)))
        
        categories_short = [cat.replace('Comment_', '').replace('Critique_', '').replace('Response_', '')
                           .replace('Perception_', '').replace('Racist_Flag', 'Racist')[:40]
                           for cat in all_cats_df['Category']]
        
        x = np.arange(len(all_cats_df))
        width = 0.35
        
        bars1 = ax.barh(x - width/2, all_cats_df['Large_Mean'], width, label='Large Cities', 
                       color='#3498db', alpha=0.7, edgecolor='black')
        bars2 = ax.barh(x + width/2, all_cats_df['Small_Mean'], width, label='Small Cities', 
                       color='#e67e22', alpha=0.7, edgecolor='black')
        
        ax.set_yticks(x)
        ax.set_yticklabels(categories_short, fontsize=9)
        ax.set_xlabel('Mean Sentiment Score', fontsize=12, fontweight='bold')
        ax.set_title('All Categories: Large vs Small Cities Comparison (with Significance Indicators)', 
                    fontsize=13, fontweight='bold', pad=20)
        # Set x-axis limits
        all_means = list(all_cats_df['Large_Mean']) + list(all_cats_df['Small_Mean'])
        min_sent = min(all_means)
        max_sent = max(all_means)
        ax.set_xlim(min(min_sent - 0.05, -1), max(max_sent + 0.05, 1))
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.legend()
        ax.grid(axis='x', alpha=0.5, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Add p-value annotations with significance markers
        for i, (idx, row) in enumerate(all_cats_df.iterrows()):
            bonf_sig = row.get('Bonferroni_Significant', False)
            uncorr_sig = row.get('Uncorrected_Significant', False)
            
            if bonf_sig:
                sig_marker = "***"
                color = 'red'
            elif uncorr_sig:
                sig_marker = "*"
                color = 'orange'
            else:
                sig_marker = "ns"
                color = 'gray'
            
            ax.text(max(row['Large_Mean'], row['Small_Mean']) + 0.05, i, 
                   f"p={row['p_value']:.4f} {sig_marker}", va='center', fontsize=8, color=color, fontweight='bold')
        
        # Add significance legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements2 = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='*** Bonferroni significant'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='* Uncorrected significant'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='ns Not significant')
        ]
        # Create combined legend
        first_legend = ax.legend(handles=[bars1, bars2], loc='upper left', fontsize=9)
        ax.add_artist(first_legend)
        ax.legend(handles=legend_elements2, loc='lower right', fontsize=8, title='Significance')
        
        plt.tight_layout()
        _save_pdf_png(charts_dir / 'all_categories_comparison', dpi=300)
        plt.close()
        print("  Saved: all_categories_comparison.pdf and all_categories_comparison.png")

    # 11. Per-category charts (one chart per label)
    if category_df is not None:
        print("Creating per-category charts (one per label)...")
        per_category_dir = charts_dir / 'per_category'
        per_category_dir.mkdir(exist_ok=True)

        def _safe_filename(name: str) -> str:
            safe = str(name).strip().lower()
            safe = safe.replace('/', '_')
            safe = safe.replace(' ', '_')
            safe = safe.replace('__', '_')
            safe = ''.join(ch for ch in safe if ch.isalnum() or ch in {'_', '-', '.'})
            return safe or 'category'

        for category in ALL_CATEGORIES:
            cat_data = category_df[category_df['Category'] == category]
            if len(cat_data) == 0:
                continue

            size_stats = calculate_sentiment_stats(cat_data, ['City_Size'], sentiment_col=sentiment_col)
            if len(size_stats) == 0:
                continue

            # Enforce consistent ordering (Large then Small when present)
            size_order = ['Large', 'Small']
            size_stats['City_Size'] = pd.Categorical(size_stats['City_Size'], categories=size_order, ordered=True)
            size_stats = size_stats.sort_values('City_Size')

            fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

            short_name = (category.replace('Comment_', '')
                          .replace('Critique_', '')
                          .replace('Response_', '')
                          .replace('Perception_', '')
                          .replace('Racist_Flag', 'Racist'))

            # Left: mean sentiment comparison with error bars
            ax1 = axes[0]
            colors = ['#3498db' if s == 'Large' else '#e67e22' for s in size_stats['City_Size']]

            se_dict = calculate_standard_errors(cat_data, 'City_Size', value_col=sentiment_col)
            mean_errors = [se_dict.get(size, 0) for size in size_stats['City_Size']]

            ax1.bar(size_stats['City_Size'].astype(str), size_stats['Mean_Sentiment'],
                    color=colors, alpha=0.75, edgecolor='black', linewidth=1,
                    yerr=mean_errors, capsize=5)
            ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.6)
            ax1.set_ylabel('Mean Sentiment Score', fontsize=11, fontweight='bold')
            ax1.set_xlabel('City Size', fontsize=11, fontweight='bold')
            ax1.set_title('Mean sentiment', fontsize=12, fontweight='bold')
            ax1.set_ylim(-1, 1)
            ax1.grid(axis='y', alpha=0.35, linestyle='-', linewidth=0.5)
            ax1.set_axisbelow(True)

            # Significance test (Large vs Small) if both exist
            try:
                large_vals = cat_data[cat_data['City_Size'] == 'Large'][sentiment_col].values
                small_vals = cat_data[cat_data['City_Size'] == 'Small'][sentiment_col].values
                if len(large_vals) >= 2 and len(small_vals) >= 2:
                    _, p_value = test_significance(large_vals, small_vals)
                    if p_value is not None:
                        sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                        ax1.text(0.5, 0.03, f'p={p_value:.4g} {sig_marker}',
                                 transform=ax1.transAxes, ha='center', va='bottom',
                                 fontsize=10, fontweight='bold',
                                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            except Exception:
                pass

            # Right: sentiment distribution
            ax2 = axes[1]
            x = np.arange(len(size_stats))
            width = 0.25

            pos_errors = []
            neu_errors = []
            neg_errors = []
            for size in size_stats['City_Size'].astype(str):
                subset = cat_data[cat_data['City_Size'] == size]
                n = len(subset)
                if n > 1:
                    pos_pct = (subset[sentiment_col] == 1).sum() / n
                    neu_pct = (subset[sentiment_col] == 0).sum() / n
                    neg_pct = (subset[sentiment_col] == -1).sum() / n
                    pos_errors.append(np.sqrt(pos_pct * (1 - pos_pct) / n) * 100)
                    neu_errors.append(np.sqrt(neu_pct * (1 - neu_pct) / n) * 100)
                    neg_errors.append(np.sqrt(neg_pct * (1 - neg_pct) / n) * 100)
                else:
                    pos_errors.append(0)
                    neu_errors.append(0)
                    neg_errors.append(0)

            ax2.bar(x - width, size_stats['Positive_%'], width, label='Positive',
                    color='#2ecc71', alpha=0.75, edgecolor='black', yerr=pos_errors, capsize=3)
            ax2.bar(x, size_stats['Neutral_%'], width, label='Neutral',
                    color='#95a5a6', alpha=0.75, edgecolor='black', yerr=neu_errors, capsize=3)
            ax2.bar(x + width, size_stats['Negative_%'], width, label='Negative',
                    color='#e74c3c', alpha=0.75, edgecolor='black', yerr=neg_errors, capsize=3)

            ax2.set_xticks(x)
            ax2.set_xticklabels(size_stats['City_Size'].astype(str))
            ax2.set_ylim(0, 100)
            ax2.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
            ax2.set_xlabel('City Size', fontsize=11, fontweight='bold')
            ax2.set_title('Distribution', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.35, linestyle='-', linewidth=0.5)
            ax2.set_axisbelow(True)

            fig.suptitle(f'Sentiment by city size: {short_name} (n={len(cat_data):,})',
                         fontsize=13, fontweight='bold', y=1.02)
            plt.tight_layout()

            out_name = f"sentiment_{_safe_filename(category)}_by_city_size.pdf"
            out_base = (per_category_dir / out_name).with_suffix('')
            _save_pdf_png(out_base, dpi=300)
            plt.close()
    
    print(f"\nAll charts saved to: {charts_dir}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive sentiment analysis by category, city, and grouping')
    parser.add_argument('--output_dir', type=str, default='output/charts/sentiment_analysis',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("COMPREHENSIVE SENTIMENT ANALYSIS")
    print("="*80)
    
    # Load existing sentiment if available (check in output_dir first)
    existing_sentiment = load_existing_sentiment(output_dir=output_dir)
    
    # Load original data files
    df = load_original_data()
    
    # Add sentiment and groupings (will use existing sentiment if available)
    df = add_sentiment_and_groupings(df, existing_sentiment, output_dir=output_dir)
    
    # Load GPT categories separately
    gpt_categories = load_gpt_categories()
    
    # Merge with GPT categories to create new combined file
    print("\nMemory status before merge:")
    import sys
    print(f"  DataFrame size: {sys.getsizeof(df) / 1024 / 1024:.2f} MB")
    if gpt_categories is not None:
        print(f"  GPT categories size: {sys.getsizeof(gpt_categories) / 1024 / 1024:.2f} MB")
    
    df = merge_with_gpt_categories(df, gpt_categories)
    
    # Clean up GPT categories after merge to free memory
    del gpt_categories
    gc.collect()
    print("  Memory cleaned up after merge")
    
    # Save combined file with sentiment and categories
    print("\nSaving combined sentiment and categories file...")
    combined_output_path = output_dir / 'sentiment_with_categories.csv'
    df.to_csv(combined_output_path, index=False)
    print(f"  Saved {len(df)} rows to {combined_output_path}")
    
    # Save sentiment scores for future reference (include both TextBlob and VADER, discrete and raw)
    print("\nSaving sentiment scores for future reference...")
    sentiment_cols = ['Comment', 'City', 'Source']
    if 'TextBlob_Sentiment' in df.columns:
        sentiment_cols.append('TextBlob_Sentiment')
    if 'TextBlob_Sentiment_Raw' in df.columns:
        sentiment_cols.append('TextBlob_Sentiment_Raw')
    if 'VADER_Sentiment' in df.columns:
        sentiment_cols.append('VADER_Sentiment')
    if 'VADER_Sentiment_Raw' in df.columns:
        sentiment_cols.append('VADER_Sentiment_Raw')
    
    sentiment_output = df[sentiment_cols].copy()
    sentiment_output.to_csv(output_dir / 'all_sentiment_scores.csv', index=False)
    print(f"  Saved {len(sentiment_output)} sentiment scores to all_sentiment_scores.csv")
    print(f"  Columns: TextBlob_Sentiment (discrete), TextBlob_Sentiment_Raw (continuous),")
    print(f"           VADER_Sentiment (discrete), VADER_Sentiment_Raw (continuous)")
    
    # Create category assignments (long format) - using VADER_Sentiment for analysis
    print("\nCreating category assignments...")
    # Use VADER_Sentiment from df, but it will be stored as 'Sentiment' in category_df
    category_df = get_category_assignments(df, sentiment_col='VADER_Sentiment')
    print(f"Total category assignments: {len(category_df)}")
    
    # Perform analyses using 'Sentiment' column (which contains VADER_Sentiment values)
    sentiment_col = 'Sentiment'  # Column name in category_df
    category_results = analyze_by_category(category_df, sentiment_col=sentiment_col)
    city_results = analyze_by_city(category_df, sentiment_col=sentiment_col)
    city_size_results = analyze_by_city_size(category_df, sentiment_col=sentiment_col)
    category_city_size_results, significant_categories = analyze_by_category_and_city_size(category_df, sentiment_col=sentiment_col)
    source_results = analyze_by_source(category_df, sentiment_col=sentiment_col)
    category_group_results = analyze_by_category_group(category_df, sentiment_col=sentiment_col)
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    category_results.to_csv(output_dir / 'sentiment_by_category.csv', index=False)
    city_results.to_csv(output_dir / 'sentiment_by_city.csv', index=False)
    city_size_results.to_csv(output_dir / 'sentiment_by_city_size.csv', index=False)
    category_city_size_results.to_csv(output_dir / 'sentiment_by_category_and_city_size.csv', index=False)
    source_results.to_csv(output_dir / 'sentiment_by_source.csv', index=False)
    category_group_results.to_csv(output_dir / 'sentiment_by_category_group.csv', index=False)
    
    if significant_categories:
        sig_df = pd.DataFrame(significant_categories)
        sig_df.to_csv(output_dir / 'significant_differences_by_category.csv', index=False)
        bonf_count = sum(1 for cat in significant_categories if cat.get('Bonferroni_Significant', False))
        print(f"\nSaved {len(significant_categories)} category comparisons")
        print(f"  {bonf_count} Bonferroni-corrected significant (p < {BONFERRONI_ALPHA:.6f})")
        print(f"  {len(significant_categories) - bonf_count} not significant after correction")
    
    # Overall summary
    overall_stats = calculate_sentiment_stats(category_df, [], sentiment_col=sentiment_col)
    if len(overall_stats) > 0:
        overall_stats.to_csv(output_dir / 'sentiment_overall_summary.csv', index=False)
    
    # Create visualizations (pass category_df for error bar calculations)
    create_visualizations(output_dir, category_results, city_results, city_size_results,
                         category_city_size_results, source_results, category_group_results,
                         significant_categories, category_df=category_df, sentiment_col=sentiment_col)
    
    print(f"\nAll results saved to: {output_dir}")
    print("\nFiles created:")
    print("  - sentiment_with_categories.csv (NEW: combined sentiment + GPT categories)")
    print("  - all_sentiment_scores.csv (all sentiment scores for future reference)")
    print("  - sentiment_by_category.csv")
    print("  - sentiment_by_city.csv")
    print("  - sentiment_by_city_size.csv")
    print("  - sentiment_by_category_and_city_size.csv")
    print("  - sentiment_by_source.csv")
    print("  - sentiment_by_category_group.csv")
    if significant_categories:
        print("  - significant_differences_by_category.csv")
    print("  - sentiment_overall_summary.csv")
    print("\nCharts (PDFs) created in charts/ subdirectory:")
    print("  - sentiment_by_category.pdf")
    print("  - sentiment_by_city.pdf")
    print("  - sentiment_by_city_size.pdf")
    print("  - sentiment_by_source.pdf")
    print("  - sentiment_by_category_group.pdf")
    print("  - sentiment_by_city_grouping.pdf")
    print("  - sentiment_all_16_labels.pdf")
    print("  - sentiment_category_city_size_heatmap.pdf")
    if significant_categories:
        print("  - significant_differences.pdf (Bonferroni-corrected only)")
        print("  - all_categories_comparison.pdf (all categories with significance indicators)")

if __name__ == "__main__":
    main()

