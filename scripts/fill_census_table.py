#!/usr/bin/env python3
"""
Script to extract census data for specific counties and fill in the table.
"""

import pandas as pd
import numpy as np

def extract_county_data():
    """Extract data for the specific counties mentioned in the table."""
    
    # Load the census data
    df = pd.read_csv('census_data/choose_counties_2023_census.csv')
    
    # Define the counties we need to find
    counties = [
        ('Erie', 'New York'),
        ('Baltimore', 'Maryland'), 
        ('El Paso', 'Texas'),
        ('Winnebago', 'Illinois'),
        ('Washington', 'Arkansas')
    ]
    
    results = {}
    
    for county_name, state in counties:
        # Find the county in the data
        mask = (df['NAME'] == county_name) & (df['state_name'] == state)
        county_data = df[mask]
        
        if not county_data.empty:
            row = county_data.iloc[0]
            
            # Extract the required data
            rfi = row['fragmentation_index']  # RFI
            population = row['total_population']  # Population
            poverty_rate = row['below_poverty_level'] / row['total_population'] * 10000  # RPP (per 10k)
            public_assistance_rate = row['public_assistance_with'] / row['total_population'] * 10000  # RPA (per 10k)
            homelessness_rate = row['Homelessness_By_County'] / row['total_population'] * 10000  # Homelessness (per 10k)
            gini = row['GINI']  # GINI coefficient
            
            results[f"{county_name} County, {state}"] = {
                'RFI': rfi,
                'Population': population,
                'RPP': poverty_rate,
                'RPA': public_assistance_rate,
                'Homelessness': homelessness_rate,
                'GINI': gini
            }
            
            print(f"Found {county_name}, {state}:")
            print(f"  RFI: {rfi:.2f}")
            print(f"  Population: {population:,}")
            print(f"  RPP (per 10k): {poverty_rate:.0f}")
            print(f"  RPA (per 10k): {public_assistance_rate:.0f}")
            print(f"  Homelessness (per 10k): {homelessness_rate:.2f}")
            print(f"  GINI: {gini:.3f}")
            print()
        else:
            print(f"Could not find {county_name}, {state}")
            print()
    
    # Add Lackawanna County data manually (from all_states_2023.csv)
    # For Lackawanna, I need to calculate the correct RPA from the raw data
    # From the grep results: public_assistance_with = 5137, total_population = 215672
    lackawanna_rpa = 5137 / 215672 * 10000
    results['Lackawanna County, Pennsylvania'] = {
        'RFI': 0.38,
        'Population': 215672,
        'RPP': 1252,  # calculated from below_poverty_level / total_population * 10000
        'RPA': lackawanna_rpa,   # calculated from public_assistance_with / total_population * 10000
        'Homelessness': 8.4,  # from the data we found
        'GINI': 0.456
    }
    
    return results

def generate_latex_table():
    """Generate the LaTeX table with the filled data."""
    
    data = extract_county_data()
    
    latex_table = r"""\begin{{table*}}
\begin{{tabular}}{{@{{}}cp{{6cm}}cccccc@{{}}}}
\toprule
\textbf{{Map Key}} & \textbf{{County, State (City Within County)}} & \textbf{{RFI$^\mathbf{{\ast}}$}} & \textbf{{Population}} & \textbf{{RPP$^\mathbf{{\dag}}$}} & \textbf{{RPA$^\mathbf{{\ddag}}$}} & \textbf{{Homelessness$^\mathbf{{\nabla}}$}} & \textbf{{GINI$^\mathbf{{\times}}$}} \\ \midrule \addlinespace[3pt]
\multicolumn{{8}}{{c}}{{\textbf{{Counties / Cities Comparable to San Francisco County (San Francisco, CA, USA)}}}} \\ \addlinespace[4pt]
1 & San Francisco County, California (San Francisco) & 0.75 & 851,036 & 1032 & 131 & 98 & 0.52 \\
2 & Multnomah County, Oregon (Portland) & 0.56 & 808,098 & 1198 & 237 & 91 & 0.47 \\
3 & Erie County, New York (Buffalo) & {:.2f} & {:,} & {:.0f} & {:.0f} & {:.2f} & {:.3f} \\
4 & Baltimore County, Maryland (Baltimore) & {:.2f} & {:,} & {:.0f} & {:.0f} & {:.2f} & {:.3f} \\
5 & El Paso County, Texas (El Paso) & {:.2f} & {:,} & {:.0f} & {:.0f} & {:.2f} & {:.3f} \\ \midrule \addlinespace[4pt]
\multicolumn{{8}}{{c}}{{\textbf{{Counties / Cities Comparable to St. Joseph County (South Bend, IN, USA)}}}} \\ \addlinespace[4pt]
A & St. Joseph County, Indiana (South Bend) & 0.52 & 272,388 & 1378 & 97 & 8.00 & 0.47 \\
B & Winnebago County, Illinois (Rockford) & {:.2f} & {:,} & {:.0f} & {:.0f} & {:.2f} & {:.3f} \\
C & Kalamazoo County, Michigan (Kalamazoo) & 0.43 & 261,426 & 1297 & 83 & 25.00 & 0.46 \\
D & Lackawanna County, Pennsylvania (Scranton) & {:.2f} & {:,} & {:.0f} & {:.0f} & {:.2f} & {:.3f} \\
E & Washington County, Arkansas () & {:.2f} & {:,} & {:.0f} & {:.0f} & {:.2f} & {:.3f} \\ \bottomrule
\end{{tabular}}
{{\raggedright
$^\mathrm{{\ast}}$RFI: Racial Fractionalization Index \\
$^\mathrm{{\dag}}$RPP: Rate of People Below Poverty Line (per 10k) \\
$^\mathrm{{\ddag}}$RPA: Rate of People With Public Assistance (per 10k)\\
$^\mathrm{{\nabla}}$Homelessness: Homelessness Rate (per 10k) \\
$^\mathrm{{\times}}$GINI: Income Inequality (GINI)\\
}}
\end{{table*}}""".format(
        data['Erie County, New York']['RFI'],
        data['Erie County, New York']['Population'],
        data['Erie County, New York']['RPP'],
        data['Erie County, New York']['RPA'],
        data['Erie County, New York']['Homelessness'],
        data['Erie County, New York']['GINI'],
        data['Baltimore County, Maryland']['RFI'],
        data['Baltimore County, Maryland']['Population'],
        data['Baltimore County, Maryland']['RPP'],
        data['Baltimore County, Maryland']['RPA'],
        data['Baltimore County, Maryland']['Homelessness'],
        data['Baltimore County, Maryland']['GINI'],
        data['El Paso County, Texas']['RFI'],
        data['El Paso County, Texas']['Population'],
        data['El Paso County, Texas']['RPP'],
        data['El Paso County, Texas']['RPA'],
        data['El Paso County, Texas']['Homelessness'],
        data['El Paso County, Texas']['GINI'],
        data['Winnebago County, Illinois']['RFI'],
        data['Winnebago County, Illinois']['Population'],
        data['Winnebago County, Illinois']['RPP'],
        data['Winnebago County, Illinois']['RPA'],
        data['Winnebago County, Illinois']['Homelessness'],
        data['Winnebago County, Illinois']['GINI'],
        data['Lackawanna County, Pennsylvania']['RFI'],
        data['Lackawanna County, Pennsylvania']['Population'],
        data['Lackawanna County, Pennsylvania']['RPP'],
        data['Lackawanna County, Pennsylvania']['RPA'],
        data['Lackawanna County, Pennsylvania']['Homelessness'],
        data['Lackawanna County, Pennsylvania']['GINI'],
        data['Washington County, Arkansas']['RFI'],
        data['Washington County, Arkansas']['Population'],
        data['Washington County, Arkansas']['RPP'],
        data['Washington County, Arkansas']['RPA'],
        data['Washington County, Arkansas']['Homelessness'],
        data['Washington County, Arkansas']['GINI']
    )
    
    # Save the LaTeX table
    with open('output/census_table_filled.tex', 'w') as f:
        f.write(latex_table)
    
    print("LaTeX table saved to output/census_table_filled.tex")
    
    # Also save the raw data as CSV
    df_output = pd.DataFrame([
        {
            'County': 'Erie County, New York',
            'RFI': data['Erie County, New York']['RFI'],
            'Population': data['Erie County, New York']['Population'],
            'RPP': data['Erie County, New York']['RPP'],
            'RPA': data['Erie County, New York']['RPA'],
            'Homelessness': data['Erie County, New York']['Homelessness'],
            'GINI': data['Erie County, New York']['GINI']
        },
        {
            'County': 'Baltimore County, Maryland',
            'RFI': data['Baltimore County, Maryland']['RFI'],
            'Population': data['Baltimore County, Maryland']['Population'],
            'RPP': data['Baltimore County, Maryland']['RPP'],
            'RPA': data['Baltimore County, Maryland']['RPA'],
            'Homelessness': data['Baltimore County, Maryland']['Homelessness'],
            'GINI': data['Baltimore County, Maryland']['GINI']
        },
        {
            'County': 'El Paso County, Texas',
            'RFI': data['El Paso County, Texas']['RFI'],
            'Population': data['El Paso County, Texas']['Population'],
            'RPP': data['El Paso County, Texas']['RPP'],
            'RPA': data['El Paso County, Texas']['RPA'],
            'Homelessness': data['El Paso County, Texas']['Homelessness'],
            'GINI': data['El Paso County, Texas']['GINI']
        },
        {
            'County': 'Winnebago County, Illinois',
            'RFI': data['Winnebago County, Illinois']['RFI'],
            'Population': data['Winnebago County, Illinois']['Population'],
            'RPP': data['Winnebago County, Illinois']['RPP'],
            'RPA': data['Winnebago County, Illinois']['RPA'],
            'Homelessness': data['Winnebago County, Illinois']['Homelessness'],
            'GINI': data['Winnebago County, Illinois']['GINI']
        },
        {
            'County': 'Lackawanna County, Pennsylvania',
            'RFI': data['Lackawanna County, Pennsylvania']['RFI'],
            'Population': data['Lackawanna County, Pennsylvania']['Population'],
            'RPP': data['Lackawanna County, Pennsylvania']['RPP'],
            'RPA': data['Lackawanna County, Pennsylvania']['RPA'],
            'Homelessness': data['Lackawanna County, Pennsylvania']['Homelessness'],
            'GINI': data['Lackawanna County, Pennsylvania']['GINI']
        },
        {
            'County': 'Washington County, Arkansas',
            'RFI': data['Washington County, Arkansas']['RFI'],
            'Population': data['Washington County, Arkansas']['Population'],
            'RPP': data['Washington County, Arkansas']['RPP'],
            'RPA': data['Washington County, Arkansas']['RPA'],
            'Homelessness': data['Washington County, Arkansas']['Homelessness'],
            'GINI': data['Washington County, Arkansas']['GINI']
        }
    ])
    
    df_output.to_csv('output/census_table_data.csv', index=False)
    print("Data saved to output/census_table_data.csv")

if __name__ == "__main__":
    generate_latex_table() 