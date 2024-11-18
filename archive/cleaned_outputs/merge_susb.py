import pandas as pd
import numpy as np

def standardize_industry_names(name):
    mapping = {
        # Administrative Services
        'Administrative and Support and Waste Management Services': 'Administrative and Support Services',
        'Administrative and Support and Waste Management and Remediation Services': 'Administrative and Support Services',
        
        # Agriculture
        'Agriculture, Forestry, Fishing, and Hunting': 'Agriculture, Forestry, Fishing and Hunting',
        
        # Other Services
        'Other Services, Except Public Administration': 'Other Services (except Public Administration)',
        
        # Remove or map special categories
        'Other': None,
        'Total': None,
        'Military': None,
        'Unknown': None
    }
    return mapping.get(name, name)

def merge_state_industry_data():
    # Read the datasets
    industry_analysis = pd.read_csv('cleaned_outputs/all_states_industry_analysis.csv')
    establishment_data = pd.read_csv('establishment_size/merged_2007_2021.csv')
    
    # Clean up industry names
    industry_analysis['industry_name'] = industry_analysis['industry_name'].str.strip().apply(standardize_industry_names)
    establishment_data['industry_name'] = establishment_data['industry_name'].str.strip().apply(standardize_industry_names)
    
    # Remove rows where industry_name is None
    industry_analysis = industry_analysis[industry_analysis['industry_name'].notna()]
    establishment_data = establishment_data[establishment_data['industry_name'].notna()]
    
    # Print unique industries from both datasets to compare
    print("\nIndustries in industry_analysis after standardization:")
    print(sorted(industry_analysis['industry_name'].unique()))
    print("\nIndustries in establishment_data after standardization:")
    print(sorted(establishment_data['industry_name'].unique()))
    
    # Merge the datasets
    merged_df = pd.merge(
        industry_analysis,
        establishment_data,
        on=['state_name', 'industry_name', 'YEAR'],
        how='outer'
    )
    
    # Remove 'United States' from state_name if present
    merged_df = merged_df[merged_df['state_name'] != 'United States']
    
    # Sort the data
    merged_df = merged_df.sort_values(['state_name', 'industry_name', 'YEAR'])
    
    # Save the merged dataset
    merged_df.to_csv('merged_state_industry_data.csv', index=False)
    
    # Print merge diagnostics
    print("\nMerge Diagnostics:")
    print(f"Rows in industry_analysis: {len(industry_analysis)}")
    print(f"Rows in establishment_data: {len(establishment_data)}")
    print(f"Rows in merged dataset: {len(merged_df)}")
    print(f"Years covered: {sorted(merged_df['YEAR'].unique())}")
    print(f"States covered: {len(merged_df['state_name'].unique())}")
    print(f"Industries covered: {len(merged_df['industry_name'].unique())}")
    
    return merged_df

if __name__ == "__main__":
    merged_data = merge_state_industry_data() 