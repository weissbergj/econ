import pandas as pd
import glob
import os

def load_and_clean_file(filepath):
    """Load and clean individual broadband file"""
    print(f"\nProcessing {filepath}")
    df = pd.read_csv(filepath)
    year = int(filepath.split('_')[1])
    print(f"Columns found: {df.columns.tolist()}")
    
    # Convert percentages to decimals
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if col != 'FIPS':
            df[col] = df[col] / 100
    
    df['state'] = df['FIPS'].astype(str).str.zfill(15).str[:2]
    print(f"Unique states found: {df['state'].nunique()}")
    
    if year >= 2019:
        print("Using post-2019 format")
        if '83' in df.columns:  # LTE
            df['LTE'] = df['83'].fillna(0)
            print("Found LTE (83)")
        if '85' in df.columns and '86' in df.columns:  # Legacy
            df['legacy_tech'] = df[['85', '86']].fillna(0).max(axis=1)
            print("Found legacy tech (85+86)")
        if '89' in df.columns:  # 5G
            df['5G'] = df['89'].fillna(0)
            print("Found 5G (89)")
        else:
            df['5G'] = 0
    else:
        print("Using pre-2019 format")
        if '4G_LTE' in df.columns:
            df['LTE'] = df['4G_LTE'].fillna(0)
            print("Found LTE (4G_LTE)")
        if '2G' in df.columns and '3G' in df.columns:
            df['legacy_tech'] = df[['2G', '3G']].fillna(0).max(axis=1)
            print("Found legacy tech (2G+3G)")
        df['5G'] = 0  # No 5G in pre-2019
    
    # Ensure all required columns exist
    if 'LTE' not in df.columns:
        df['LTE'] = 0
    if 'legacy_tech' not in df.columns:
        df['legacy_tech'] = 0
    if '5G' not in df.columns:
        df['5G'] = 0
    
    print("\nBefore aggregation:")
    print(df[['state', 'legacy_tech', 'LTE', '5G', 'ANY']].head())
    
    # Create state-level aggregates
    agg_df = df.groupby('state').agg({
        'legacy_tech': 'mean',
        'LTE': 'mean',
        '5G': 'mean',
        'ANY': 'mean'
    }).reset_index()
    
    print("\nAfter aggregation:")
    print(agg_df.head())
    
    agg_df = agg_df.rename(columns={'ANY': 'total_coverage'})
    agg_df['year'] = year
    
    # Calculate ratios
    for tech in ['LTE', 'legacy_tech', '5G']:
        agg_df[f'{tech}_ratio'] = agg_df[tech] / agg_df['total_coverage'].replace(0, 1)
    
    return agg_df

def clean_broadband_data():
    """Process all broadband files in current directory"""
    files = glob.glob('F477_*.csv')
    dfs = []
    
    for file in sorted(files):  # Sort to process in chronological order
        try:
            print(f"Processing {file}...")
            df = load_and_clean_file(file)
            dfs.append(df)
            print(f"Successfully processed {file}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    if not dfs:
        print("No files were processed successfully!")
        return None
        
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df.sort_values(['state', 'year'])

if __name__ == "__main__":
    # Change to script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("Starting broadband data cleaning...")
    clean_data = clean_broadband_data()
    
    if clean_data is not None:
        print("\nData Summary:")
        print(f"Years covered: {sorted(clean_data['year'].unique())}")
        print(f"Number of states: {clean_data['state'].nunique()}")
        
        # Show technology evolution
        for year in sorted(clean_data['year'].unique()):
            year_data = clean_data[clean_data['year'] == year]
            print(f"\nYear {year} averages:")
            print(f"Legacy: {year_data['legacy_tech'].mean():.2%}")
            print(f"LTE: {year_data['LTE'].mean():.2%}")
            print(f"5G: {year_data['5G'].mean():.2%}")
            print(f"Total Coverage: {year_data['total_coverage'].mean():.2%}")
        
        output_file = "../final_merge/clean_broadband_data.csv"
        clean_data.to_csv(output_file, index=False)
        print(f"\nCleaned data saved to {output_file}")