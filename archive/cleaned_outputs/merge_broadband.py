import pandas as pd
import os

def load_and_prepare_broadband():
    """Load cleaned broadband data and prepare for merging"""
    df = pd.read_csv('clean_broadband_data.csv')
    print("\nBroadband columns:")
    print(df.columns.tolist())
    
    # Convert state codes to match format in industry data
    df['state'] = df['state'].astype(str).str.zfill(2)
    
    # Ensure year column is uppercase to match industry data
    df = df.rename(columns={'year': 'YEAR'})
    
    return df

def load_and_prepare_industry():
    """Load industry data and prepare for merging"""
    df = pd.read_csv('../merged_state_industry_data.csv')
    print("\nIndustry columns:")
    print(df.columns.tolist())
    
    # Extract state codes
    state_codes = {
        'Alabama': '01', 'Alaska': '02', 'Arizona': '04', 'Arkansas': '05',
        'California': '06', 'Colorado': '08', 'Connecticut': '09', 'Delaware': '10',
        'District of Columbia': '11', 'Florida': '12', 'Georgia': '13', 'Hawaii': '15',
        'Idaho': '16', 'Illinois': '17', 'Indiana': '18', 'Iowa': '19',
        'Kansas': '20', 'Kentucky': '21', 'Louisiana': '22', 'Maine': '23',
        'Maryland': '24', 'Massachusetts': '25', 'Michigan': '26', 'Minnesota': '27',
        'Mississippi': '28', 'Missouri': '29', 'Montana': '30', 'Nebraska': '31',
        'Nevada': '32', 'New Hampshire': '33', 'New Jersey': '34', 'New Mexico': '35',
        'New York': '36', 'North Carolina': '37', 'North Dakota': '38', 'Ohio': '39',
        'Oklahoma': '40', 'Oregon': '41', 'Pennsylvania': '42', 'Rhode Island': '44',
        'South Carolina': '45', 'South Dakota': '46', 'Tennessee': '47', 'Texas': '48',
        'Utah': '49', 'Vermont': '50', 'Virginia': '51', 'Washington': '53',
        'West Virginia': '54', 'Wisconsin': '55', 'Wyoming': '56'
    }
    
    df['state'] = df['state_name'].map(state_codes)
    return df

def merge_data():
    """Merge broadband and industry data"""
    print("Loading broadband data...")
    broadband_df = load_and_prepare_broadband()
    print(f"Broadband years: {sorted(broadband_df['YEAR'].unique())}")
    
    print("\nLoading industry data...")
    industry_df = load_and_prepare_industry()
    print(f"Industry years: {sorted(industry_df['YEAR'].unique())}")
    
    print("\nBefore merge:")
    print(f"Broadband data shape: {broadband_df.shape}")
    print(f"Industry data shape: {industry_df.shape}")
    
    # Merge on state and YEAR
    merged_df = pd.merge(
        industry_df,
        broadband_df,
        on=['state', 'YEAR'],
        how='left'
    ).fillna(pd.NA)
    
    print("\nAfter merge:")
    print(f"Merged data shape: {merged_df.shape}")
    print("\nMerged columns:")
    print(merged_df.columns.tolist())
    
    # Check for any states that didn't merge
    unmatched_states = industry_df[~industry_df['state'].isin(broadband_df['state'])]['state_name'].unique()
    if len(unmatched_states) > 0:
        print("\nWarning: Some states did not match:")
        print(unmatched_states)
    
    # Check for missing values in key columns
    print("\nMissing values in key columns:")
    key_cols = ['state_name', 'YEAR', 'industry_name', 'workers', 'LTE', '5G', 'total_coverage']
    print(merged_df[key_cols].isnull().sum())
    
    # Save merged file
    output_path = '../cleaned_outputs/merged_state_industry_broadband.csv'
    os.makedirs('../cleaned_outputs', exist_ok=True)
    merged_df.to_csv(output_path, index=False, na_rep='NA')
    print(f"\nMerged data saved to: {output_path}")
    
    return merged_df

if __name__ == "__main__":
    merge_data() 