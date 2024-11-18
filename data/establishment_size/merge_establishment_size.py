import pandas as pd
import numpy as np
from pathlib import Path
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def map_naics_to_sector(naics_code):
    # Dictionary mapping 2-digit NAICS codes to broad sectors
    sector_mapping = {
        '11': 'Agriculture, Forestry, Fishing and Hunting',
        '21': 'Mining, Quarrying, and Oil and Gas Extraction',
        '22': 'Utilities',
        '23': 'Construction',
        '31': 'Manufacturing',
        '32': 'Manufacturing',
        '33': 'Manufacturing',
        '42': 'Wholesale Trade',
        '44': 'Retail Trade',
        '45': 'Retail Trade',
        '48': 'Transportation and Warehousing',
        '49': 'Transportation and Warehousing',
        '51': 'Information',
        '52': 'Finance and Insurance',
        '53': 'Real Estate and Rental and Leasing',
        '54': 'Professional, Scientific, and Technical Services',
        '55': 'Management of Companies and Enterprises',
        '56': 'Administrative and Support and Waste Management and Remediation Services',
        '61': 'Educational Services',
        '62': 'Health Care and Social Assistance',
        '71': 'Arts, Entertainment, and Recreation',
        '72': 'Accommodation and Food Services',
        '81': 'Other Services (except Public Administration)',
        '92': 'Public Administration'
    }
    
    # Handle special cases
    if pd.isna(naics_code) or naics_code == '--':
        return 'Total'
    
    # Get first two digits of NAICS code
    naics_prefix = str(naics_code)[:2]
    
    # Return mapped sector or original if no mapping exists
    return sector_mapping.get(naics_prefix, 'Other')

def clean_dataset(file_path, year):
    # Read the CSV file
    df = pd.read_csv(file_path, encoding='latin1')
    
    # Map NAICS codes to broad sectors
    df['industry_name'] = df['NAICS'].apply(map_naics_to_sector)
    
    # Different mapping based on year (2012 and before vs after 2012)
    if year <= 2012:
        size_mapping = {
            'Total': '01: Total',
            '0-4': '02: <5',
            '5-9': '03: 5-9',
            '10-19': '04: 10-19',
            '<20': '05: <20',
            '20-99': '06: 20-99',
            '100-499': '07: 100-499',
            '<500': '08: <500',
            '500+': '09: 500+'
        }
        df['size_category'] = df['ENTRSIZEDSCR'].map(size_mapping)
    elif year >= 2019:
        # Use ENTRSIZEDSCR directly for 2019 onwards
        df['size_category'] = df['ENTRSIZEDSCR']
    else:
        # For years 2013-2018, use numeric ENTRSIZE mapping
        size_mapping = {
            1: '01: Total',
            2: '02: <5',
            3: '03: 5-9',
            4: '04: 10-19',
            5: '05: <20',
            6: '06: 20-99',
            7: '07: 100-499',
            8: '08: <500',
            9: '09: 500+'
        }
        df['size_category'] = df['ENTRSIZE'].map(size_mapping)
    
    # Remove rows where size_category is None
    df = df[df['size_category'].notna()]
    
    # Debug prints
    print(f"\nUnique size_category values after mapping for {year}:")
    print(sorted(df['size_category'].unique()))
    
    # Group by state, mapped industry, and enterprise size
    df = df.groupby(['STATEDSCR', 'industry_name', 'size_category']).agg({
        'FIRM': 'sum',
        'ESTB': 'sum',
        'EMPL': 'sum',
        'PAYR': 'sum'
    }).reset_index()
    
    # Update size_order to match new categories
    size_order = [
        '01: Total',
        '02: <5',
        '03: 5-9',
        '04: 10-19',
        '05: <20',
        '06: 20-99',
        '07: 100-499',
        '08: <500',
        '09: 500+'
    ]
    
    # Create pivot tables for each metric
    metrics = {
        'FIRM': 'firms',
        'ESTB': 'establishments',
        'EMPL': 'employment',
        'PAYR': 'annual_payroll'
    }
    
    # Start with base columns and rename immediately
    result_df = df[['STATEDSCR', 'industry_name']].drop_duplicates()
    result_df = result_df.rename(columns={
        'STATEDSCR': 'state_name'
    })
    result_df['YEAR'] = year
    
    # For each metric, create columns for each enterprise size
    for old_col, new_col in metrics.items():
        pivot_df = df.pivot_table(
            index=['STATEDSCR', 'industry_name'],
            columns='size_category',
            values=old_col,
            aggfunc='first'
        )
        
        # Add missing columns with zeros
        for size in size_order:
            if size not in pivot_df.columns:
                pivot_df[size] = 0
        
        # Reorder columns according to size_order
        pivot_df = pivot_df[size_order]
        
        # Rename columns to include metric name
        new_columns = {
            size: f"{new_col}_{size}" 
            for size in size_order
        }
        pivot_df = pivot_df.rename(columns=new_columns)
        
        # Reset index to prepare for merge
        pivot_df = pivot_df.reset_index()
        
        # Rename the index columns to match result_df
        pivot_df = pivot_df.rename(columns={
            'STATEDSCR': 'state_name'
        })
        
        # Join with result
        result_df = result_df.merge(pivot_df, on=['state_name', 'industry_name'])
    
    # Format numeric columns
    for col in result_df.columns:
        if col not in ['state_name', 'industry_name', 'YEAR']:
            # Convert to numeric, replacing any errors with NaN
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
            
            # Format annual_payroll columns as integers (they're currently in thousands)
            if 'annual_payroll' in col:
                result_df[col] = result_df[col] * 1000
                result_df[col] = result_df[col].fillna(0).astype(np.int64)
            else:
                # For other numeric columns, round to integers
                result_df[col] = result_df[col].fillna(0).astype(np.int64)
    
    print("\nFinal dataframe Shape:", result_df.shape)
    print("\nFirst few rows:")
    print(result_df.head(2))
    
    # Save to CSV
    output_file = f'cleaned_{year}.csv'
    result_df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")
    
    return result_df

def merge_all_years():
    # Initialize empty list to store dataframes
    all_dfs = []
    
    # Process each year from 2007 to 2021
    for year in range(2007, 2022):
        file_path = f'{year}.csv'
        
        # Check if file exists
        if Path(file_path).exists():
            print(f"\nProcessing {year}...")
            try:
                df = clean_dataset(file_path, year)
                all_dfs.append(df)
            except Exception as e:
                print(f"Error processing {year}: {e}")
        else:
            print(f"\nWarning: File not found for {year}")
    
    # Merge all dataframes
    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        
        # Sort by year, state, and industry
        merged_df = merged_df.sort_values(['YEAR', 'state_name', 'industry_name'])
        
        # Save merged dataset
        output_file = '../final_merge/merged_establishment_size_2007_2021.csv'
        merged_df.to_csv(output_file, index=False)
        print(f"\nSaved merged data to {output_file}")
        
        # Print summary
        print("\nMerged data summary:")
        print(f"Total rows: {len(merged_df)}")
        print(f"Years included: {sorted(merged_df['YEAR'].unique())}")
        print(f"States included: {len(merged_df['state_name'].unique())}")
        
        return merged_df
    else:
        print("No data files were processed successfully")
        return None

if __name__ == "__main__":
    merged_data = merge_all_years()