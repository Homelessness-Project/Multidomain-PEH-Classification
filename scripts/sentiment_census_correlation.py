#!/usr/bin/env python3
"""
Correlation Analysis: Sentiment vs Census Variables (Racial Fractionalization, etc.)
by City Size Grouping

Tests correlations between sentiment and census variables (racial fractionalization,
GINI, poverty, etc.) separately for small and large cities.

Usage:
    python scripts/sentiment_census_correlation.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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

def load_sentiment_data():
    """Load sentiment data by city."""
    
    sentiment_file = 'output/charts/sentiment_analysis/sentiment_by_city.csv'
    print(f"\nLoading sentiment data from {sentiment_file}...")
    
    if not Path(sentiment_file).exists():
        raise FileNotFoundError(f"Sentiment data file not found: {sentiment_file}")
    
    df = pd.read_csv(sentiment_file)
    # Normalize city names
    df['City_Normalized'] = df['City'].str.lower().str.strip()
    print(f"  Loaded {len(df)} cities")
    
    return df

def match_cities_to_counties(sentiment_df, census_df):
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
        # Find sentiment data
        city_sentiment = sentiment_df[sentiment_df['City_Normalized'] == city_lower]
        
        if len(city_sentiment) == 0:
            print(f"  WARNING: No sentiment data for {city_lower}")
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
        city_row = city_sentiment.iloc[0]
        
        # Extract values with fallbacks
        rfi = county_row.get(rfi_col, county_row.get('fragmentation_index', np.nan))
        population = county_row.get('Population', county_row.get('total_population', np.nan))
        gini = county_row.get('GINI', county_row.get('GINI', np.nan))
        
        combined = {
            'City': city_row['City'],
            'City_Size': 'Large' if city_lower in LARGE_CITIES else 'Small',
            'County': county_row[county_col] if county_col in county_row else county_name,
            'Mean_Sentiment': city_row['Mean_Sentiment'],
            'Positive_%': city_row['Positive_%'],
            'Neutral_%': city_row['Neutral_%'],
            'Negative_%': city_row['Negative_%'],
            'RFI': rfi,
            'Population': population,
            'GINI': gini,
            'RPP': county_row.get('RPP', np.nan),
            'RPA': county_row.get('RPA', np.nan),
            'Homelessness': county_row.get('Homelessness', county_row.get('Homelessness_By_County', np.nan))
        }
        
        matched_data.append(combined)
        print(f"  Matched: {city_row['City']} -> {county_row[county_col] if county_col in county_row else county_name}")
    
    matched_df = pd.DataFrame(matched_data)
    print(f"\nSuccessfully matched {len(matched_df)} cities")
    
    return matched_df

def calculate_correlations(matched_df, city_size=None):
    """Calculate correlations between sentiment and census variables."""
    
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
    sentiment_vars = ['Mean_Sentiment', 'Positive_%', 'Negative_%']
    
    results = []
    
    for sentiment_var in sentiment_vars:
        for census_var in census_vars:
            # Remove missing values
            data = subset[[sentiment_var, census_var]].dropna()
            
            if len(data) < 3:
                continue
            
            x = data[census_var].values
            y = data[sentiment_var].values
            
            # Calculate correlation
            r, p_value = stats.pearsonr(x, y)
            
            results.append({
                'Sentiment_Variable': sentiment_var,
                'Census_Variable': census_var,
                'Correlation': r,
                'P_Value': p_value,
                'N': len(data),
                'City_Size': city_size if city_size else 'All'
            })
            
            sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"{sentiment_var:20s} vs {census_var:15s}: r={r:7.4f}, p={p_value:.4f} {sig_marker}")
    
    return pd.DataFrame(results)

def create_correlation_visualizations(matched_df, output_dir):
    """Create visualization charts for correlations."""
    
    print("\n" + "="*80)
    print("CREATING CORRELATION VISUALIZATIONS")
    print("="*80)
    
    charts_dir = output_dir / 'charts'
    charts_dir.mkdir(exist_ok=True)
    
    # 1. RFI vs Sentiment by City Size
    print("\nCreating RFI vs Sentiment scatter plot...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, city_size in enumerate(['Small', 'Large']):
        ax = axes[idx]
        subset = matched_df[matched_df['City_Size'] == city_size]
        
        if len(subset) < 2:
            continue
        
        colors = '#e67e22' if city_size == 'Small' else '#3498db'
        
        ax.scatter(subset['RFI'], subset['Mean_Sentiment'], 
                  s=200, alpha=0.7, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add city labels
        for _, row in subset.iterrows():
            ax.annotate(row['City'], (row['RFI'], row['Mean_Sentiment']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # Calculate and plot regression line
        if len(subset) >= 2:
            x = subset['RFI'].values
            y = subset['Mean_Sentiment'].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() >= 2:
                x_clean = x[mask]
                y_clean = y[mask]
                z = np.polyfit(x_clean, y_clean, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                ax.plot(x_line, p(x_line), "--", color='gray', alpha=0.5, linewidth=2)
                
                # Calculate correlation
                r, p_val = stats.pearsonr(x_clean, y_clean)
                sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                ax.text(0.05, 0.95, f"r={r:.3f}, p={p_val:.3f}{sig_marker}", 
                       transform=ax.transAxes, fontsize=11, fontweight='bold',
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Racial Fragmentation Index (RFI)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Sentiment Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{city_size} Cities: RFI vs Sentiment', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    chart_path = charts_dir / 'rfi_sentiment_correlation_by_city_size.pdf'
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
        
        # Select numeric columns
        numeric_cols = ['Mean_Sentiment', 'Positive_%', 'Negative_%', 
                       'RFI', 'Population', 'RPP', 'RPA', 'Homelessness', 'GINI']
        available_cols = [col for col in numeric_cols if col in subset.columns]
        corr_data = subset[available_cols].corr()
        
        # Create heatmap
        sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                   vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Correlation'}, 
                   square=True, linewidths=0.5)
        ax.set_title(f'{city_size} Cities\nCorrelation Matrix', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    chart_path = charts_dir / 'census_sentiment_correlation_heatmap.pdf'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {chart_path}")
    print(f"    Full path: {chart_path.absolute()}")
    
    # 3. Summary comparison chart (Small vs Large cities only - maintaining cluster design)
    print("Creating summary comparison chart...")
    print("  Note: Keeping Small and Large cities separate to maintain cluster/matching design")
    
    # Check which variables have data
    available_vars = []
    for var in ['RFI', 'GINI', 'Homelessness', 'Population']:
        if var in matched_df.columns:
            # Check if variable has non-null data
            if matched_df[var].notna().sum() >= 2:
                available_vars.append(var)
    
    # Use first 3 available variables
    census_vars = available_vars[:3] if len(available_vars) >= 3 else available_vars
    
    if len(census_vars) == 0:
        print("  WARNING: No census variables with data available for comparison chart")
        return
    
    # Grid size: 2 rows (Small, Large) x n_cols - maintaining cluster separation
    n_cols = len(census_vars)
    fig, axes = plt.subplots(2, n_cols, figsize=(6*n_cols, 10))
    
    # Handle case where there's only one column
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Process each row: Small, Large (maintaining cluster design)
    for row_idx, city_size in enumerate(['Small', 'Large']):
        subset = matched_df[matched_df['City_Size'] == city_size]
        color = '#e67e22' if city_size == 'Small' else '#3498db'
        
        for col_idx, census_var in enumerate(census_vars):
            ax = axes[row_idx, col_idx]
            
            if census_var not in subset.columns:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{city_size} Cities\n{census_var}', fontsize=11, fontweight='bold')
                continue
            
            x = subset[census_var].values
            y = subset['Mean_Sentiment'].values
            mask = ~(np.isnan(x) | np.isnan(y))
            
            if mask.sum() < 2:
                ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{city_size} Cities\n{census_var}', fontsize=11, fontweight='bold')
                ax.grid(alpha=0.3)
                continue
            
            x_clean = x[mask]
            y_clean = y[mask]
            
            ax.scatter(x_clean, y_clean, s=200, alpha=0.7, color=color, 
                      edgecolor='black', linewidth=1.5)
            
            # Add city labels with values
            for i, (_, row) in enumerate(subset[mask].iterrows()):
                city_name = row['City']
                var_value = x_clean[i]
                sentiment_value = y_clean[i]
                
                # Format the value based on variable type
                if census_var == 'Population':
                    var_str = f"{var_value:,.0f}"  # Format with commas
                elif census_var in ['RFI', 'GINI']:
                    var_str = f"{var_value:.3f}"  # 3 decimal places
                else:
                    var_str = f"{var_value:.2f}"  # 2 decimal places
                
                # Create label with city name and values
                label = f"{city_name}\n({var_str}, {sentiment_value:.3f})"
                ax.annotate(label, (x_clean[i], y_clean[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=7,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))
            
            # Regression line
            if len(x_clean) >= 2:
                z = np.polyfit(x_clean, y_clean, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                ax.plot(x_line, p(x_line), "--", color='gray', alpha=0.5, linewidth=2)
                
                r, p_val = stats.pearsonr(x_clean, y_clean)
                sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                ax.text(0.05, 0.95, f"r={r:.3f}\np={p_val:.3f}{sig_marker}", 
                       transform=ax.transAxes, fontsize=10, fontweight='bold',
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel(census_var, fontsize=11, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel('Mean Sentiment', fontsize=11, fontweight='bold')
            ax.set_title(f'{city_size} Cities\n{census_var}', fontsize=11, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.set_axisbelow(True)
    
    plt.tight_layout()
    chart_path = charts_dir / 'census_variables_sentiment_comparison.pdf'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {chart_path}")
    print(f"    Full path: {chart_path.absolute()}")

def main():
    output_dir = Path('output/charts/sentiment_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("SENTIMENT vs CENSUS VARIABLES CORRELATION ANALYSIS")
    print("="*80)
    
    # Load data
    census_df = load_census_data()
    sentiment_df = load_sentiment_data()
    
    # Match cities to counties
    matched_df = match_cities_to_counties(sentiment_df, census_df)
    
    # Save matched data
    matched_output_path = output_dir / 'city_census_sentiment_matched.csv'
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
        corr_output_path = output_dir / 'sentiment_census_correlations.csv'
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
        if 'Mean_Sentiment' in subset.columns:
            print(f"  Mean Sentiment: Mean={subset['Mean_Sentiment'].mean():.4f}, Std={subset['Mean_Sentiment'].std():.4f}")
    
    # Create visualizations
    create_correlation_visualizations(matched_df, output_dir)
    
    print(f"\n{'='*80}")
    print("OUTPUT FILES SUMMARY")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"\nCSV Files:")
    print(f"  1. city_census_sentiment_matched.csv")
    print(f"  2. sentiment_census_correlations.csv")
    print(f"\nChart Files (in charts/ subdirectory):")
    print(f"  1. rfi_sentiment_correlation_by_city_size.pdf")
    print(f"  2. census_sentiment_correlation_heatmap.pdf")
    print(f"  3. census_variables_sentiment_comparison.pdf")
    print(f"\nAll results saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()

