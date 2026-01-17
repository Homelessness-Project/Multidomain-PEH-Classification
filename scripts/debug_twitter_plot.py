#!/usr/bin/env python3
"""Debug script to check why Twitter isn't visible in plots"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the data
half_year_data = pd.read_csv('output/bias_temporal_analysis/bias_by_half_year.csv')
print("Twitter data:")
print(half_year_data[half_year_data['source'] == 'twitter'])

# Create period order
periods = sorted(half_year_data['year_half'].unique())
period_to_pos = {period: idx for idx, period in enumerate(periods)}

print(f"\nPeriod order (first 5, last 5):")
print(periods[:5])
print("...")
print(periods[-5:])
print(f"\n2024-H2 position: {period_to_pos.get('2024-H2', 'NOT FOUND')}")

# Test plotting
fig, ax = plt.subplots(figsize=(16, 6))

SOURCE_STYLE = {
    'twitter': {'color': '#1DA1F2', 'marker': 'o', 'name': 'Twitter'},
    'reddit': {'color': '#FF4500', 'marker': '^', 'name': 'Reddit'},
    'news': {'color': '#2E7D32', 'marker': 's', 'name': 'News'},
    'meeting_minutes': {'color': '#9C27B0', 'marker': 'D', 'name': 'Meeting Minutes'}
}

for source in half_year_data['source'].unique():
    source_data = half_year_data[half_year_data['source'] == source].copy()
    source_data['x_pos'] = source_data['year_half'].map(period_to_pos)
    source_data = source_data.dropna(subset=['x_pos']).sort_values('x_pos')
    
    if len(source_data) > 0:
        style = SOURCE_STYLE.get(source, {'color': 'gray', 'marker': 'o', 'name': source.capitalize()})
        print(f"\n{source}: {len(source_data)} points")
        print(f"  X positions: {source_data['x_pos'].tolist()}")
        print(f"  Y values: {source_data['avg_bias'].tolist()}")
        
        ax.plot(source_data['x_pos'], source_data['avg_bias'], 
                marker=style['marker'], color=style['color'], label=style['name'], 
                linewidth=2, markersize=10, linestyle='-', alpha=0.8)
        ax.errorbar(source_data['x_pos'], source_data['avg_bias'], 
                   yerr=source_data['std_bias'], color=style['color'], alpha=0.3, capsize=3)

ax.set_xticks(range(len(periods)))
ax.set_xticklabels(periods, rotation=45, ha='right')
ax.set_xlabel('6-Month Period', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Bias Score', fontsize=12, fontweight='bold')
ax.set_title('Debug: Average Bias Score by 6-Month Period', fontsize=14, fontweight='bold')
ax.legend(title='Source', fontsize=10)
ax.grid(True, alpha=0.3)

# Highlight 2024-H2 position
if '2024-H2' in period_to_pos:
    pos = period_to_pos['2024-H2']
    ax.axvline(x=pos, color='red', linestyle='--', alpha=0.5, label='2024-H2 marker')
    print(f"\n2024-H2 is at x-position: {pos}")

plt.tight_layout()
output_path = Path('output/bias_temporal_analysis/debug_twitter_plot.pdf')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved debug plot to: {output_path}")
plt.close()
