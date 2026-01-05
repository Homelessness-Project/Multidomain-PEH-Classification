#!/usr/bin/env python3
"""
Correlation Analysis: Bias vs Census Variables (Racial Fractionalization, etc.)
by City Size Grouping

Tests correlations between bias metrics (racist, harmful generalization, 
deserving/undeserving) and census variables (racial fractionalization,
GINI, poverty, etc.) separately for small and large cities.

Usage:
    python scripts/bias_census_correlation.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Try to import langfair for bias measurement
try:
    import langfair as lf
    LANGFAIR_AVAILABLE = True
except ImportError:
    LANGFAIR_AVAILABLE = False
    print("WARNING: langfair package not available. Install with: pip install langfair")
    print("  Continuing with annotation-based bias measurement only.")

# City to County mapping (based on user specification)
# Format: (County Name, Full State Name, State Abbreviation)
CITY_COUNTY_MAP = {
    # Small Cities
    'south bend': ('St. Joseph County', 'Indiana', 'IN'),
    'rockford': ('Winnebago County', 'Illinois', 'IL'),
    'kalamazoo': ('Kalamazoo County', 'Michigan', 'MI'),
    'scranton': ('Lackawanna County', 'Pennsylvania', 'PA'),
    'fayetteville': ('Washington County', 'Arkansas', 'AR'),
    # Large Cities
    'san francisco': ('San Francisco County', 'California', 'CA'),
    'portland': ('Multnomah County', 'Oregon', 'OR'),
    'buffalo': ('Erie County', 'New York', 'NY'),
    'baltimore': ('Baltimore County', 'Maryland', 'MD'),
    'el paso': ('El Paso County', 'Texas', 'TX')
}

# Expected format in census file: "County Name, State Name"
def get_expected_county_string(county_name, state_name):
    """Get the expected format string for county in census file."""
    return f"{county_name}, {state_name}"

# City size groupings
LARGE_CITIES = ['san francisco', 'portland', 'buffalo', 'baltimore', 'el paso']
SMALL_CITIES = ['kalamazoo', 'south bend', 'rockford', 'scranton', 'fayetteville']

# Bias categories to analyze
# Map from display name to possible column names in data files
BIAS_CATEGORIES = {
    'racist': ['Racist_Flag', 'Racist', 'racist'],
    'harmful generalization': ['Perception_harmful generalization', 'harmful generalization'],
    'deserving/undeserving': ['Perception_deserving/undeserving', 'deserving/undeserving']
}

def load_census_data():
    """Load census data with city/county statistics."""
    
    # Try the full data file first (has all counties), then the summary file
    census_files = [
        'census_data/all_states_2023.csv',
        'output/census_table_data.csv'
    ]
    
    for census_file in census_files:
        if Path(census_file).exists():
            print(f"Loading census data from {census_file}...")
            df = pd.read_csv(census_file)
            print(f"  Loaded {len(df)} counties")
            print(f"  Columns: {list(df.columns)[:10]}...")
            return df
    
    raise FileNotFoundError(f"Census data file not found. Tried: {census_files}")

def measure_bias_with_langfair(combined_df, checkpoint_dir=None):
    """Use langfair to measure bias in text data with progress tracking and checkpointing."""
    
    if not LANGFAIR_AVAILABLE:
        return combined_df
    
    print("\n  Measuring bias using langfair...")
    
    # Set up checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = Path('langfair_biased_results/checkpoints')
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_file = checkpoint_dir / 'langfair_bias_scores.parquet'
    
    try:
        from langfair.metrics import toxicity, stereotype
        
        # Create a copy for langfair analysis
        langfair_data = combined_df[['Comment', 'City']].copy()
        langfair_data = langfair_data.dropna(subset=['Comment'])
        
        if len(langfair_data) == 0:
            print("    No text data available for langfair analysis")
            return combined_df
        
        print(f"    Analyzing {len(langfair_data)} comments with langfair...")
        
        # Check for existing checkpoint (try parquet first, then pickle)
        checkpoint_df = None
        checkpoint_file_pkl = checkpoint_file.with_suffix('.pkl')
        
        if checkpoint_file.exists():
            print(f"    Loading checkpoint from {checkpoint_file}...")
            try:
                checkpoint_df = pd.read_parquet(checkpoint_file)
            except:
                try:
                    checkpoint_df = pd.read_pickle(checkpoint_file)
                except Exception as e:
                    print(f"    Could not load checkpoint: {e}")
        elif checkpoint_file_pkl.exists():
            print(f"    Loading checkpoint from {checkpoint_file_pkl}...")
            try:
                checkpoint_df = pd.read_pickle(checkpoint_file_pkl)
            except Exception as e:
                print(f"    Could not load checkpoint: {e}")
        
        if checkpoint_df is not None:
            # Merge checkpoint data with combined_df
            if 'index' in checkpoint_df.columns:
                checkpoint_df = checkpoint_df.set_index('index')
            elif checkpoint_df.index.name != 'index':
                # If index is not named 'index', assume it's the row index
                checkpoint_df.index.name = 'index'
            
            for col in ['langfair_toxicity', 'langfair_stereotype', 'langfair_bias_score']:
                if col in checkpoint_df.columns:
                    # Only update rows that exist in both dataframes
                    common_indices = checkpoint_df.index.intersection(combined_df.index)
                    combined_df.loc[common_indices, col] = checkpoint_df.loc[common_indices, col]
            print(f"    Loaded checkpoint with {len(checkpoint_df)} processed comments")
        
        # Find which indices still need processing
        if checkpoint_df is not None and 'langfair_bias_score' in checkpoint_df.columns:
            processed_indices = set(checkpoint_df.index)
            remaining_data = langfair_data[~langfair_data.index.isin(processed_indices)]
            print(f"    Resuming: {len(remaining_data)} comments remaining out of {len(langfair_data)}")
        else:
            remaining_data = langfair_data
            processed_indices = set()
        
        if len(remaining_data) == 0:
            print("    All comments already processed!")
            return combined_df
        
        # Initialize langfair metrics
        try:
            print("    Initializing langfair metrics...")
            toxicity_metrics = toxicity.ToxicityMetrics(batch_size=250)
            stereotype_metrics = stereotype.StereotypeMetrics()
        except Exception as e:
            print(f"    Could not initialize langfair metrics: {e}")
            return combined_df
        
        # Process in batches with progress bar
        batch_size = 250
        texts = remaining_data['Comment'].astype(str).tolist()
        indices = remaining_data.index.tolist()
        
        toxicity_scores = []
        stereotype_scores = []
        
        # Initialize with existing scores if checkpoint exists
        if checkpoint_df is not None:
            for idx in langfair_data.index:
                if idx in processed_indices:
                    toxicity_scores.append(checkpoint_df.loc[idx, 'langfair_toxicity'])
                    stereotype_scores.append(checkpoint_df.loc[idx, 'langfair_stereotype'])
                else:
                    toxicity_scores.append(None)
                    stereotype_scores.append(None)
        else:
            toxicity_scores = [None] * len(langfair_data)
            stereotype_scores = [None] * len(langfair_data)
        
        # Process toxicity scores in batches
        print("    Calculating toxicity scores...")
        remaining_texts = [texts[i] for i in range(len(texts))]
        remaining_indices = [indices[i] for i in range(len(indices))]
        
        with tqdm(total=len(remaining_texts), desc="      Toxicity", unit="comments") as pbar:
            for i in range(0, len(remaining_texts), batch_size):
                batch_texts = remaining_texts[i:i+batch_size]
                batch_indices = remaining_indices[i:i+batch_size]
                
                try:
                    batch_toxicity = toxicity_metrics.get_toxicity_scores(batch_texts)
                    # Map scores back to original indices
                    for j, idx in enumerate(batch_indices):
                        pos = langfair_data.index.get_loc(idx)
                        toxicity_scores[pos] = batch_toxicity[j]
                except Exception as e:
                    print(f"\n      Error in batch {i//batch_size + 1}: {e}")
                    for idx in batch_indices:
                        pos = langfair_data.index.get_loc(idx)
                        toxicity_scores[pos] = 0.0
                
                pbar.update(len(batch_texts))
                
                # Save checkpoint every 10 batches
                if (i // batch_size + 1) % 10 == 0:
                    _save_langfair_checkpoint(langfair_data, toxicity_scores, stereotype_scores, checkpoint_file)
        
        # Process stereotype scores in batches
        print("    Calculating stereotype scores...")
        with tqdm(total=len(remaining_texts), desc="      Stereotype", unit="comments") as pbar:
            for i in range(0, len(remaining_texts), batch_size):
                batch_texts = remaining_texts[i:i+batch_size]
                batch_indices = remaining_indices[i:i+batch_size]
                
                try:
                    result = stereotype_metrics.evaluate(batch_texts, return_data=True)
                    if isinstance(result, dict) and 'data' in result:
                        data = result['data']
                        # Extract scores - may need to handle different formats
                        if isinstance(data, dict):
                            # Try different possible keys
                            if 'Stereotype Classifier' in data:
                                batch_stereotype = data['Stereotype Classifier']
                            elif 'scores' in data:
                                batch_stereotype = data['scores']
                            else:
                                # Get first list/array value
                                batch_stereotype = [0.0] * len(batch_texts)
                                for key, val in data.items():
                                    if isinstance(val, (list, np.ndarray)) and len(val) == len(batch_texts):
                                        batch_stereotype = list(val)
                                        break
                            if not isinstance(batch_stereotype, list):
                                batch_stereotype = [0.0] * len(batch_texts)
                        else:
                            batch_stereotype = [0.0] * len(batch_texts)
                    else:
                        batch_stereotype = [0.0] * len(batch_texts)
                    
                    # Map scores back to original indices
                    for j, idx in enumerate(batch_indices):
                        pos = langfair_data.index.get_loc(idx)
                        stereotype_scores[pos] = batch_stereotype[j] if j < len(batch_stereotype) else 0.0
                        
                except Exception as e:
                    print(f"\n      Error in batch {i//batch_size + 1}: {e}")
                    import traceback
                    traceback.print_exc()
                    for idx in batch_indices:
                        pos = langfair_data.index.get_loc(idx)
                        stereotype_scores[pos] = 0.0
                
                pbar.update(len(batch_texts))
                
                # Save checkpoint every 10 batches
                if (i // batch_size + 1) % 10 == 0:
                    _save_langfair_checkpoint(langfair_data, toxicity_scores, stereotype_scores, checkpoint_file)
        
        # Final checkpoint save
        _save_langfair_checkpoint(langfair_data, toxicity_scores, stereotype_scores, checkpoint_file)
        
        # Add langfair bias scores to combined_df
        toxicity_array = np.array([s if s is not None else 0.0 for s in toxicity_scores])
        stereotype_array = np.array([s if s is not None else 0.0 for s in stereotype_scores])
        
        combined_df.loc[langfair_data.index, 'langfair_toxicity'] = toxicity_array
        combined_df.loc[langfair_data.index, 'langfair_stereotype'] = stereotype_array
        # Combined bias score (average of toxicity and stereotype)
        combined_df.loc[langfair_data.index, 'langfair_bias_score'] = (toxicity_array + stereotype_array) / 2.0
        
        print(f"\n    ✓ Calculated langfair bias scores for {len(langfair_data)} comments")
        
    except Exception as e:
        print(f"    Error using langfair: {e}")
        import traceback
        traceback.print_exc()
        print("    Falling back to annotation-based bias measurement")
    
    return combined_df

def _save_langfair_checkpoint(langfair_data, toxicity_scores, stereotype_scores, checkpoint_file):
    """Save checkpoint of langfair calculations."""
    try:
        # Only save non-None scores
        valid_indices = []
        valid_toxicity = []
        valid_stereotype = []
        
        for i, idx in enumerate(langfair_data.index):
            if toxicity_scores[i] is not None and stereotype_scores[i] is not None:
                valid_indices.append(idx)
                valid_toxicity.append(toxicity_scores[i])
                valid_stereotype.append(stereotype_scores[i])
        
        if len(valid_indices) == 0:
            return
        
        checkpoint_data = {
            'index': valid_indices,
            'langfair_toxicity': valid_toxicity,
            'langfair_stereotype': valid_stereotype
        }
        checkpoint_data['langfair_bias_score'] = (
            np.array(valid_toxicity) + np.array(valid_stereotype)
        ) / 2.0
        
        checkpoint_df = pd.DataFrame(checkpoint_data)
        
        # Try parquet first, fall back to pickle
        try:
            checkpoint_df.to_parquet(checkpoint_file, index=False)
        except:
            # Fall back to pickle if parquet not available
            checkpoint_file_pkl = checkpoint_file.with_suffix('.pkl')
            checkpoint_df.to_pickle(checkpoint_file_pkl)
            
    except Exception as e:
        print(f"      Warning: Could not save checkpoint: {e}")

def load_bias_data():
    """Load bias data by city from classification results and langfair."""
    
    print("\nLoading bias data from classification results...")
    
    # Try to load from existing aggregated file first
    bias_file = 'langfair_biased_results/bias_by_city.csv'
    if Path(bias_file).exists():
        print(f"Loading existing bias data from {bias_file}...")
        df = pd.read_csv(bias_file)
        df['City_Normalized'] = df['City'].str.lower().str.strip()
        print(f"  Loaded {len(df)} cities")
        return df
    
    # Otherwise, aggregate from complete dataset and classification results
    print("  Aggregating bias data from complete dataset...")
    
    # Load complete dataset files
    data_files = {
        'reddit': {
            'file': 'complete_dataset/all_reddit_comments.csv',
            'comment_col': 'Deidentified_Comment',
            'city_col': 'city'
        },
        'news': {
            'file': 'complete_dataset/all_newspaper_articles.csv',
            'comment_col': 'Deidentified_paragraph_text',
            'city_col': 'city'
        },
        'x': {
            'file': 'complete_dataset/all_twitter_posts.csv',
            'comment_col': 'Deidentified_text',
            'city_col': 'city'
        },
        'meeting_minutes': {
            'file': 'complete_dataset/all_meeting_minutes.csv',
            'comment_col': 'Deidentified_paragraph',
            'city_col': 'city'
        }
    }
    
    all_data = []
    
    for source, config in data_files.items():
        filepath = config['file']
        comment_col = config['comment_col']
        city_col = config['city_col']
        
        if not Path(filepath).exists():
            print(f"  WARNING: File not found: {filepath}")
            continue
        
        print(f"  Loading {source} data...")
        df = pd.read_csv(filepath)
        df['Source'] = source
        df['Comment'] = df[comment_col]
        
        # Normalize city column
        if city_col in df.columns:
            df['City'] = df[city_col]
        elif 'City' not in df.columns:
            print(f"  WARNING: No city column found in {filepath}")
            continue
        
        all_data.append(df[['Comment', 'City', 'Source']])
    
    if not all_data:
        raise ValueError("No data files found to aggregate bias from!")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"  Total comments loaded: {len(combined_df)}")
    
    # Use langfair to measure bias if available
    combined_df = measure_bias_with_langfair(combined_df, checkpoint_dir=Path('langfair_biased_results/checkpoints'))
    
    # Load classification results - try to find files with bias categories
    # Check NLP outputs for classification results
    nlp_outputs_dir = Path('nlp_outputs')
    classification_files = []
    
    for source_dir in ['reddit', 'x', 'news', 'meeting_minutes']:
        source_path = nlp_outputs_dir / source_dir
        if source_path.exists():
            # Look for CSV files with predictions
            for csv_file in source_path.glob('*.csv'):
                if 'bert' in csv_file.name.lower() or 'roberta' in csv_file.name.lower():
                    classification_files.append((source_dir, csv_file))
    
    # If no classification files found, try loading from gold standard or annotation files
    if not classification_files:
        print("  WARNING: No classification result files found in nlp_outputs/")
        print("  Attempting to load from annotation files...")
        
        annotation_files = {
            'reddit': 'annotation/reddit_raw_scores.csv',
            'x': 'annotation/x_raw_scores.csv',
            'news': 'annotation/news_raw_scores.csv',
            'meeting_minutes': 'annotation/meeting_minutes_raw_scores.csv'
        }
        
        for source, ann_file in annotation_files.items():
            if Path(ann_file).exists():
                print(f"  Loading {source} annotations...")
                try:
                    ann_df = pd.read_csv(ann_file)
                    # Check if bias columns exist (try all possible column names)
                    bias_cols_found = []
                    for bias_name, possible_cols in BIAS_CATEGORIES.items():
                        for col_name in possible_cols:
                            if col_name in ann_df.columns:
                                bias_cols_found.append((bias_name, col_name))
                                break  # Use first match
                    
                    if bias_cols_found:
                        # Merge with combined_df on Comment and City
                        # Normalize city names by removing spaces (e.g., 'south bend' -> 'southbend')
                        ann_df['City_Normalized'] = ann_df.get('City', '').astype(str).str.lower().str.strip().str.replace(' ', '')
                        combined_df['City_Normalized'] = combined_df['City'].astype(str).str.lower().str.strip().str.replace(' ', '')
                        combined_df['Comment_Normalized'] = combined_df['Comment'].astype(str).str.strip()
                        
                        comment_col = 'Deidentified_Comment' if 'Deidentified_Comment' in ann_df.columns else 'Comment'
                        if comment_col in ann_df.columns:
                            ann_df['Comment_Normalized'] = ann_df[comment_col].astype(str).str.strip()
                            
                            print(f"    Merging {source} annotations...")
                            print(f"      Annotation cities: {ann_df['City_Normalized'].unique()[:5].tolist()}")
                            print(f"      Combined cities: {combined_df['City_Normalized'].unique()[:5].tolist()}")
                            
                            merged = combined_df.merge(
                                ann_df[['Comment_Normalized', 'City_Normalized'] + [col for _, col in bias_cols_found]],
                                on=['Comment_Normalized', 'City_Normalized'],
                                how='left',
                                suffixes=('', '_ann')
                            )
                            
                            # Copy bias columns to combined_df (use standardized name)
                            # IMPORTANT: Merge preserves left index, so indices align perfectly
                            for bias_name, col_name in bias_cols_found:
                                # Store with standardized name
                                std_col_name = f'BIAS_{bias_name}'
                                if col_name in merged.columns:
                                    # Get merged values (fill NaN with 0 for this source)
                                    # Since merge preserves left index, new_values has same index as combined_df
                                    new_values = merged[col_name].fillna(0)
                                    
                                    # If column already exists, preserve existing non-zero values
                                    if std_col_name in combined_df.columns:
                                        # Only update where new value is non-zero, or where old value is 0/NaN
                                        # Both Series have same index from merge, so boolean operations align
                                        existing_values = combined_df[std_col_name].fillna(0)
                                        update_mask = (new_values > 0) | (existing_values == 0)
                                        combined_df.loc[update_mask, std_col_name] = new_values[update_mask]
                                    else:
                                        # First time creating this column - assign directly
                                        combined_df[std_col_name] = new_values
                                    
                                    # Count how many matches we got
                                    matches = (combined_df[std_col_name] > 0).sum()
                                    print(f"      Found {matches} total matches for {bias_name} (out of {len(combined_df)} total)")
                                else:
                                    print(f"      WARNING: Column {col_name} not found in merged dataframe")
                except Exception as e:
                    print(f"    Error loading {ann_file}: {e}")
    
    # Aggregate bias metrics by city
    print("\n  Aggregating bias metrics by city...")
    # Normalize city names by removing spaces (e.g., 'south bend' -> 'southbend')
    combined_df['City_Normalized'] = combined_df['City'].astype(str).str.lower().str.strip().str.replace(' ', '')
    
    city_bias_stats = []
    
    for city_normalized in combined_df['City_Normalized'].unique():
        city_data = combined_df[combined_df['City_Normalized'] == city_normalized]
        
        # Get original city name (first occurrence)
        city_name = city_data['City'].iloc[0] if len(city_data) > 0 else city_normalized
        
        stats_dict = {'City': city_name, 'N': len(city_data)}
        
        # Calculate bias percentages for each category
        for bias_name, possible_cols in BIAS_CATEGORIES.items():
            # Try standardized column name first, then try all possible names
            std_col_name = f'BIAS_{bias_name}'
            col_name = None
            
            if std_col_name in city_data.columns:
                col_name = std_col_name
            else:
                for possible_col in possible_cols:
                    if possible_col in city_data.columns:
                        col_name = possible_col
                        break
            
            if col_name and col_name in city_data.columns:
                # Check if column is binary (0/1) or continuous
                bias_values = city_data[col_name].dropna()
                if len(bias_values) > 0:
                    # If values are 0/1, calculate percentage
                    if bias_values.max() <= 1 and bias_values.min() >= 0:
                        # For annotation files, values might be 0-3 (scores), treat >0 as positive
                        if bias_values.max() > 1:
                            bias_pct = (bias_values > 0).sum() / len(bias_values) * 100
                            mean_bias = bias_values.mean() / bias_values.max() if bias_values.max() > 0 else 0
                        else:
                            bias_pct = (bias_values == 1).sum() / len(bias_values) * 100
                            mean_bias = bias_values.mean()
                    else:
                        # Continuous values - use mean
                        bias_pct = bias_values.mean() * 100
                        mean_bias = bias_values.mean()
                    
                    stats_dict[f'{bias_name}_%'] = bias_pct
                    stats_dict[f'Mean_{bias_name}'] = mean_bias
                else:
                    stats_dict[f'{bias_name}_%'] = 0.0
                    stats_dict[f'Mean_{bias_name}'] = 0.0
            else:
                stats_dict[f'{bias_name}_%'] = 0.0
                stats_dict[f'Mean_{bias_name}'] = 0.0
        
        # Add langfair bias scores if available
        for langfair_col in ['langfair_bias_score', 'langfair_toxicity', 'langfair_stereotype']:
            if langfair_col in city_data.columns:
                langfair_scores = city_data[langfair_col].dropna()
                if len(langfair_scores) > 0:
                    stats_dict[f'{langfair_col}_mean'] = langfair_scores.mean()
                    stats_dict[f'{langfair_col}_%'] = (langfair_scores > 0.5).sum() / len(langfair_scores) * 100
                else:
                    stats_dict[f'{langfair_col}_mean'] = 0.0
                    stats_dict[f'{langfair_col}_%'] = 0.0
        
        city_bias_stats.append(stats_dict)
    
    bias_df = pd.DataFrame(city_bias_stats)
    # Normalize city names by removing spaces (e.g., 'south bend' -> 'southbend')
    bias_df['City_Normalized'] = bias_df['City'].str.lower().str.strip().str.replace(' ', '')
    
    # Save for future use
    output_dir = Path('langfair_biased_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    bias_df.to_csv(output_dir / 'bias_by_city.csv', index=False)
    print(f"  Saved bias data to {output_dir / 'bias_by_city.csv'}")
    
    return bias_df

def match_cities_to_counties(bias_df, census_df):
    """Match cities to their county census data."""
    
    print("\nMatching cities to counties...")
    
    # Determine which columns are available
    if 'County' in census_df.columns:
        # Using summary file (census_table_data.csv)
        county_col = 'County'
        rfi_col = 'RFI'
    elif 'entity_name' in census_df.columns:
        # Using full file (all_states_2023.csv)
        county_col = 'entity_name'
        rfi_col = 'fragmentation_index'
    else:
        raise ValueError("Cannot determine county column in census data")
    
    # Normalize county names for matching
    census_df['County_Normalized'] = census_df[county_col].astype(str).str.lower().str.strip()
    
    matched_data = []
    
    for city_lower, (county_name, state_name, state_abbr) in CITY_COUNTY_MAP.items():
        # Find bias data
        city_bias = bias_df[bias_df['City_Normalized'] == city_lower]
        
        if len(city_bias) == 0:
            print(f"  WARNING: No bias data for {city_lower}")
            continue
        
        # Find census data - use exact expected format first
        expected_format = get_expected_county_string(county_name, state_name).lower()
        
        # Try exact match first (most reliable)
        county_census = census_df[
            census_df['County_Normalized'] == expected_format
        ]
        
        # If no exact match, try contains with both county and state
        if len(county_census) == 0:
            county_pattern = county_name.lower()
            state_pattern = state_name.lower()
            
            county_census = census_df[
                (census_df['County_Normalized'].str.contains(county_pattern, na=False, case=False, regex=False)) &
                (census_df['County_Normalized'].str.contains(state_pattern, na=False, case=False, regex=False))
            ]
        
        # If still no match, try with state abbreviation
        if len(county_census) == 0:
            county_pattern = county_name.lower()
            state_abbr_pattern = state_abbr.lower()
            
            county_census = census_df[
                (census_df['County_Normalized'].str.contains(county_pattern, na=False, case=False, regex=False)) &
                (census_df['County_Normalized'].str.contains(state_abbr_pattern, na=False, case=False, regex=False))
            ]
        
        # If still no match, try just county name (but prefer the right state if multiple matches)
        if len(county_census) == 0:
            county_pattern = county_name.lower().replace('.', '').replace('st ', 'st. ')
            county_census = census_df[
                census_df['County_Normalized'].str.contains(county_pattern, na=False, case=False, regex=False)
            ]
            
            # If multiple matches, prefer the one with the state name
            if len(county_census) > 1:
                state_matches = county_census[
                    county_census['County_Normalized'].str.contains(state_name.lower(), na=False, case=False, regex=False) |
                    county_census['County_Normalized'].str.contains(state_abbr.lower(), na=False, case=False, regex=False)
                ]
                if len(state_matches) > 0:
                    county_census = state_matches
        
        # Debug output
        if len(county_census) == 0:
            print(f"  WARNING: No census data for {county_name}, {state_name}")
            print(f"    Expected format: '{expected_format}'")
            # Show what counties are available that might match
            potential_matches = census_df[
                census_df['County_Normalized'].str.contains(county_name.lower().split()[0], na=False, case=False, regex=False)
            ]['County_Normalized'].head(5).tolist()
            if potential_matches:
                print(f"    Potential matches found: {potential_matches}")
            continue
        elif len(county_census) > 1:
            print(f"  NOTE: Multiple matches for {county_name}, {state_name}, using first one")
            print(f"    Matches: {county_census['County_Normalized'].tolist()}")
        
        # Use first match
        county_row = county_census.iloc[0]
        city_row = city_bias.iloc[0]
        
        # Extract values with fallbacks
        rfi = county_row.get(rfi_col, county_row.get('fragmentation_index', np.nan))
        population = county_row.get('Population', county_row.get('total_population', np.nan))
        gini = county_row.get('GINI', county_row.get('GINI', np.nan))
        
        combined = {
            'City': city_row['City'],
            'City_Size': 'Large' if city_lower in LARGE_CITIES else 'Small',
            'County': county_row[county_col] if county_col in county_row else county_name,
            'N': city_row.get('N', 0),
            'RFI': rfi,
            'Population': population,
            'GINI': gini,
            'RPP': county_row.get('RPP', np.nan),
            'RPA': county_row.get('RPA', np.nan),
            'Homelessness': county_row.get('Homelessness', county_row.get('Homelessness_By_County', np.nan))
        }
        
        # Add bias metrics
        for bias_name in BIAS_CATEGORIES.keys():
            combined[f'{bias_name}_%'] = city_row.get(f'{bias_name}_%', 0.0)
            combined[f'Mean_{bias_name}'] = city_row.get(f'Mean_{bias_name}', 0.0)
        
        # Add langfair metrics if available
        for langfair_col in ['langfair_bias_score_%', 'langfair_toxicity_%', 'langfair_stereotype_%']:
            combined[langfair_col] = city_row.get(langfair_col, 0.0)
        
        matched_data.append(combined)
        print(f"  Matched: {city_row['City']} -> {county_row[county_col] if county_col in county_row else county_name}")
    
    matched_df = pd.DataFrame(matched_data)
    print(f"\nSuccessfully matched {len(matched_df)} cities")
    
    return matched_df

def calculate_correlations(matched_df, city_size=None):
    """Calculate correlations between bias metrics and census variables."""
    
    if city_size:
        subset = matched_df[matched_df['City_Size'] == city_size].copy()
        title_suffix = f" ({city_size} Cities)"
    else:
        subset = matched_df.copy()
        title_suffix = " (All Cities)"
    
    if len(subset) < 3:
        print(f"\nWARNING: Not enough data for {city_size} cities (need at least 3)")
        return None
    
    print(f"\n{'='*80}")
    print(f"CORRELATION ANALYSIS{title_suffix}")
    print(f"{'='*80}")
    print(f"Number of cities: {len(subset)}")
    
    # Variables to test
    census_vars = ['RFI', 'Population', 'RPP', 'RPA', 'Homelessness', 'GINI']
    bias_vars = [f'{bias_name}_%' for bias_name in BIAS_CATEGORIES.keys()]
    
    # Add langfair bias metrics if available
    for langfair_col in ['langfair_bias_score_%', 'langfair_toxicity_%', 'langfair_stereotype_%']:
        if langfair_col in subset.columns:
            bias_vars.append(langfair_col)
    
    results = []
    
    for bias_var in bias_vars:
        for census_var in census_vars:
            # Remove missing values
            data = subset[[bias_var, census_var]].dropna()
            
            if len(data) < 3:
                continue
            
            x = data[census_var].values
            y = data[bias_var].values
            
            # Calculate correlation
            r, p_value = stats.pearsonr(x, y)
            
            results.append({
                'Bias_Variable': bias_var,
                'Census_Variable': census_var,
                'Correlation': r,
                'P_Value': p_value,
                'N': len(data),
                'City_Size': city_size if city_size else 'All'
            })
            
            sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"{bias_var:30s} vs {census_var:15s}: r={r:7.4f}, p={p_value:.4f} {sig_marker}")
    
    return pd.DataFrame(results)

def create_correlation_visualizations(matched_df, output_dir):
    """Create visualization charts for correlations."""
    
    print("\n" + "="*80)
    print("CREATING CORRELATION VISUALIZATIONS")
    print("="*80)
    
    charts_dir = output_dir / 'charts'
    charts_dir.mkdir(exist_ok=True)
    
    # 1. RFI vs Bias by City Size (for each bias category + langfair metrics)
    print("\nCreating RFI vs Bias scatter plots...")
    
    # Determine which bias metrics to plot
    bias_metrics_to_plot = []
    for bias_name in BIAS_CATEGORIES.keys():
        bias_col = f'{bias_name}_%'
        if bias_col in matched_df.columns and matched_df[bias_col].sum() > 0:
            bias_metrics_to_plot.append((bias_col, bias_name.replace("_", " ").title()))
    
    # Add langfair metrics
    langfair_metrics = [
        ('langfair_bias_score_%', 'Langfair Bias Score'),
        ('langfair_toxicity_%', 'Langfair Toxicity'),
        ('langfair_stereotype_%', 'Langfair Stereotype')
    ]
    for langfair_col, langfair_name in langfair_metrics:
        if langfair_col in matched_df.columns and matched_df[langfair_col].sum() > 0:
            bias_metrics_to_plot.append((langfair_col, langfair_name))
    
    if len(bias_metrics_to_plot) == 0:
        print("  WARNING: No bias metrics with non-zero values to plot")
        return
    
    n_bias = len(bias_metrics_to_plot)
    fig, axes = plt.subplots(1, n_bias, figsize=(6*n_bias, 6))
    
    if n_bias == 1:
        axes = [axes]
    
    for idx, (bias_col, bias_name) in enumerate(bias_metrics_to_plot):
        ax = axes[idx]
        
        for city_size in ['Small', 'Large']:
            subset = matched_df[matched_df['City_Size'] == city_size]
            
            if len(subset) < 2:
                continue
            
            color = '#e67e22' if city_size == 'Small' else '#3498db'
            label = city_size
            
            x = subset['RFI'].values
            y = subset[bias_col].values
            mask = ~(np.isnan(x) | np.isnan(y))
            
            if mask.sum() >= 2:
                x_clean = x[mask]
                y_clean = y[mask]
                
                ax.scatter(x_clean, y_clean, s=200, alpha=0.7, color=color, 
                          edgecolor='black', linewidth=1.5, label=label)
                
                # Add city labels
                for i, (_, row) in enumerate(subset[mask].iterrows()):
                    ax.annotate(row['City'], (x_clean[i], y_clean[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Racial Fragmentation Index (RFI)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{bias_name} (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'RFI vs {bias_name}', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        if len(bias_metrics_to_plot) > 1:
            ax.legend()
    
    plt.tight_layout()
    chart_path = charts_dir / 'rfi_bias_correlation_by_city_size.pdf'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {chart_path}")
    print(f"    Full path: {chart_path.absolute()}")
    
    # 2. Correlation heatmap
    print("Creating correlation heatmap...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, city_size in enumerate(['All', 'Small', 'Large']):
        ax = axes[idx] if idx < 3 else None
        if ax is None:
            break
        
        if city_size == 'All':
            subset = matched_df.copy()
        else:
            subset = matched_df[matched_df['City_Size'] == city_size]
        
        if len(subset) < 2:
            continue
        
        # Select numeric columns - include langfair metrics
        bias_cols = [f'{bias_name}_%' for bias_name in BIAS_CATEGORIES.keys()]
        langfair_cols = ['langfair_bias_score_%', 'langfair_toxicity_%', 'langfair_stereotype_%']
        numeric_cols = bias_cols + langfair_cols + ['RFI', 'Population', 'RPP', 'RPA', 'Homelessness', 'GINI']
        available_cols = [col for col in numeric_cols if col in subset.columns]
        corr_data = subset[available_cols].corr()
        
        # Create heatmap
        sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                   vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Correlation'}, 
                   square=True, linewidths=0.5)
        ax.set_title(f'{city_size} Cities\nCorrelation Matrix', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    chart_path = charts_dir / 'census_bias_correlation_heatmap.pdf'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {chart_path}")
    print(f"    Full path: {chart_path.absolute()}")
    
    # 3. Summary comparison chart (Small vs Large cities)
    print("Creating summary comparison chart...")
    
    # Check which variables have data
    available_vars = []
    for var in ['RFI', 'GINI', 'Homelessness', 'Population']:
        if var in matched_df.columns:
            if matched_df[var].notna().sum() >= 2:
                available_vars.append(var)
    
    # Use first 3 available variables
    census_vars = available_vars[:3] if len(available_vars) >= 3 else available_vars
    
    if len(census_vars) == 0:
        print("  WARNING: No census variables with data available for comparison chart")
        return
    
    # Determine which bias metrics to plot (prioritize langfair if annotation-based are all zeros)
    bias_metrics_to_plot = []
    for bias_name in BIAS_CATEGORIES.keys():
        bias_col = f'{bias_name}_%'
        if bias_col in matched_df.columns and matched_df[bias_col].sum() > 0:
            bias_metrics_to_plot.append((bias_col, bias_name.replace("_", " ").title()))
    
    # Add langfair metrics
    langfair_metrics = [
        ('langfair_bias_score_%', 'Langfair Bias Score'),
        ('langfair_toxicity_%', 'Langfair Toxicity'),
        ('langfair_stereotype_%', 'Langfair Stereotype')
    ]
    for langfair_col, langfair_name in langfair_metrics:
        if langfair_col in matched_df.columns and matched_df[langfair_col].sum() > 0:
            bias_metrics_to_plot.append((langfair_col, langfair_name))
    
    if len(bias_metrics_to_plot) == 0:
        print("  WARNING: No bias metrics with non-zero values to plot")
        return
    
    # Grid size: 2 rows (Small, Large) x n_cols x n_bias
    n_cols = len(census_vars)
    n_bias = len(bias_metrics_to_plot)
    fig, axes = plt.subplots(2, n_cols * n_bias, figsize=(6*n_cols*n_bias, 10))
    
    # Handle case where there's only one column
    if n_cols * n_bias == 1:
        axes = axes.reshape(-1, 1)
    else:
        axes = axes.reshape(2, n_cols * n_bias)
    
    # Process each row: Small, Large
    for row_idx, city_size in enumerate(['Small', 'Large']):
        subset = matched_df[matched_df['City_Size'] == city_size]
        color = '#e67e22' if city_size == 'Small' else '#3498db'
        
        for bias_idx, (bias_col, bias_name) in enumerate(bias_metrics_to_plot):
            
            for col_idx, census_var in enumerate(census_vars):
                ax_idx = bias_idx * n_cols + col_idx
                ax = axes[row_idx, ax_idx]
                
                if census_var not in subset.columns:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{city_size} Cities\n{census_var} vs {bias_name}', 
                               fontsize=10, fontweight='bold')
                    continue
                
                x = subset[census_var].values
                y = subset[bias_col].values
                mask = ~(np.isnan(x) | np.isnan(y))
                
                if mask.sum() < 2:
                    ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{city_size} Cities\n{census_var} vs {bias_name}', 
                               fontsize=10, fontweight='bold')
                    ax.grid(alpha=0.3)
                    continue
                
                x_clean = x[mask]
                y_clean = y[mask]
                
                ax.scatter(x_clean, y_clean, s=200, alpha=0.7, color=color, 
                          edgecolor='black', linewidth=1.5)
                
                # Add city labels
                for i, (_, row) in enumerate(subset[mask].iterrows()):
                    ax.annotate(row['City'], (x_clean[i], y_clean[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=7)
                
                # Regression line
                if len(x_clean) >= 2:
                    z = np.polyfit(x_clean, y_clean, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                    ax.plot(x_line, p(x_line), "--", color='gray', alpha=0.5, linewidth=2)
                    
                    r, p_val = stats.pearsonr(x_clean, y_clean)
                    sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    ax.text(0.05, 0.95, f"r={r:.3f}\np={p_val:.3f}{sig_marker}", 
                           transform=ax.transAxes, fontsize=9, fontweight='bold',
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                ax.set_xlabel(census_var, fontsize=10, fontweight='bold')
                if col_idx == 0 and bias_idx == 0:
                    ax.set_ylabel(f'{bias_name} (%)', fontsize=10, fontweight='bold')
                ax.set_title(f'{city_size} Cities\n{census_var} vs {bias_name}', 
                           fontsize=10, fontweight='bold')
                ax.grid(alpha=0.3)
                ax.set_axisbelow(True)
    
    plt.tight_layout()
    chart_path = charts_dir / 'census_variables_bias_comparison.pdf'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {chart_path}")
    print(f"    Full path: {chart_path.absolute()}")

def main():
    output_dir = Path('langfair_biased_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("BIAS vs CENSUS VARIABLES CORRELATION ANALYSIS")
    print("="*80)
    
    # Load data
    census_df = load_census_data()
    bias_df = load_bias_data()
    
    # Match cities to counties
    matched_df = match_cities_to_counties(bias_df, census_df)
    
    # Save matched data
    matched_output_path = output_dir / 'city_census_bias_matched.csv'
    matched_df.to_csv(matched_output_path, index=False)
    print(f"\nSaved matched data to: {matched_output_path}")
    print(f"  Full path: {matched_output_path.absolute()}")
    
    # Calculate correlations for each group
    all_correlations = []
    
    for city_size in [None, 'Small', 'Large']:
        corr_results = calculate_correlations(matched_df, city_size)
        if corr_results is not None:
            all_correlations.append(corr_results)
    
    if all_correlations:
        combined_correlations = pd.concat(all_correlations, ignore_index=True)
        corr_output_path = output_dir / 'bias_census_correlations.csv'
        combined_correlations.to_csv(corr_output_path, index=False)
        print(f"\nSaved correlation results to: {corr_output_path}")
        print(f"  Full path: {corr_output_path.absolute()}")
    else:
        print("\nWARNING: No correlation results to save")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS BY CITY SIZE")
    print("="*80)
    
    for city_size in ['Small', 'Large']:
        subset = matched_df[matched_df['City_Size'] == city_size]
        print(f"\n{city_size} Cities:")
        print(f"  Number of cities: {len(subset)}")
        if 'RFI' in subset.columns:
            print(f"  RFI: Mean={subset['RFI'].mean():.4f}, Std={subset['RFI'].std():.4f}")
        for bias_name in BIAS_CATEGORIES.keys():
            bias_col = f'{bias_name}_%'
            if bias_col in subset.columns:
                print(f"  {bias_name}: Mean={subset[bias_col].mean():.2f}%, Std={subset[bias_col].std():.2f}%")
    
    # Create visualizations
    create_correlation_visualizations(matched_df, output_dir)
    
    print(f"\n{'='*80}")
    print("OUTPUT FILES SUMMARY")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"\nCSV Files:")
    print(f"  1. city_census_bias_matched.csv")
    print(f"  2. bias_census_correlations.csv")
    print(f"  3. bias_by_city.csv")
    print(f"\nChart Files (in charts/ subdirectory):")
    print(f"  1. rfi_bias_correlation_by_city_size.pdf")
    print(f"  2. census_bias_correlation_heatmap.pdf")
    print(f"  3. census_variables_bias_comparison.pdf")
    print(f"\nAll results saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()

