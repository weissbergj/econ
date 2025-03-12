import pandas as pd
import re

def clean_industry_name(name):
    """
    Remove commas and quotes, strip whitespace, convert to lowercase.
    """
    if not isinstance(name, str):
        return ''
    cleaned = re.sub(r'[",]+', '', name)
    return cleaned.strip().lower()

def main():
    # 1) Read the BIG CSV
    big_df = pd.read_csv("merged_final.csv")  # adjust filename
    unique_big = big_df['industry_name'].unique()
    print("\n=== UNIQUE INDUSTRY NAMES IN BIG CSV ===")
    for ind in unique_big:
        print("  -", ind)
    
    # 2) Read the SMALL CSV
    small_df = pd.read_csv("digital_scores.csv")  # adjust filename
    unique_small = small_df['Industry group'].unique()
    print("\n=== UNIQUE INDUSTRY NAMES IN SMALL CSV ===")
    for ind in unique_small:
        print("  -", ind)
    
    # 3) Create a mapping dictionary from small -> big naming
    mapping_dict = {
        "Media": "Information",
        "Health Care Services and Hospitals": "Health Care and Social Assistance",
        "Information and Communications Technology": "Information",
        "Oil & Gas Extraction": "Mining, Quarrying, and Oil and Gas Extraction",
        "Advanced Manufacturing": "Manufacturing",
        "Basic Goods Manufacturing": "Manufacturing",
        "Administrative and Support and Waste Management and Remediation Services": "Administrative and Support Services",
        "Nursing and Residential Care Facilities, and Social Assistance": "Health Care and Social Assistance",
        "Mining (Except Oil and Gas)": "Mining, Quarrying, and Oil and Gas Extraction"
    }
    
    # 4) Apply the dictionary to the SMALL CSV
    small_df['mapped_industry'] = small_df['Industry group'].replace(mapping_dict)
    
    # 5) Clean both sets of names
    big_df['industry_clean'] = big_df['industry_name'].apply(clean_industry_name)
    small_df['industry_clean'] = small_df['mapped_industry'].apply(clean_industry_name)
    
    # 6) Merge (INNER JOIN)
    merged_df = pd.merge(big_df, small_df, how='inner', on='industry_clean')
    
    # 7A) Find unmatched from the SMALL CSV
    unmatched_small = small_df[~small_df['industry_clean'].isin(merged_df['industry_clean'])]
    if not unmatched_small.empty:
        print("\n=== UNMATCHED INDUSTRIES FROM SMALL CSV AFTER MAPPING ===")
        for missing in unmatched_small['Industry group'].unique():
            print("  -", missing)
    else:
        print("\nNo unmatched industries in the small CSV.")
    
    # 7B) Also find unmatched from the BIG CSV
    unmatched_big = big_df[~big_df['industry_clean'].isin(merged_df['industry_clean'])]
    if not unmatched_big.empty:
        print("\n=== UNMATCHED INDUSTRIES FROM BIG CSV ===")
        for missing in unmatched_big['industry_name'].unique():
            print("  -", missing)
    else:
        print("\nNo unmatched industries in the big CSV.")
    
    # 8) Write the merged output
    merged_df.to_csv("digital_merged_data.csv", index=False)
    print("\nMerged data written to 'digital_merged_data.csv'.")

if __name__ == "__main__":
    main()
