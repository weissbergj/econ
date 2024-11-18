import pandas as pd

# Read the datasets with different encoding
min_wage_df = pd.read_csv('Minimum Wage Data.csv', encoding='latin1')  # or try 'iso-8859-1'
merged_df = pd.read_csv('../cleaned_outputs/merged_state_industry_broadband.csv')

# Get unique years from merged_df to filter min_wage_df
years = merged_df['YEAR'].unique()

# Select relevant columns and filter for years in merged_df
min_wage_clean = min_wage_df[[
    'Year', 
    'State', 
    'Effective.Minimum.Wage', 
    'Effective.Minimum.Wage.2020.Dollars',
    'Federal.Minimum.Wage',
    'Federal.Minimum.Wage.2020.Dollars'
]]
min_wage_clean = min_wage_clean[min_wage_clean['Year'].isin(years)]

# Rename columns for clarity
min_wage_clean = min_wage_clean.rename(columns={
    'Year': 'YEAR',
    'State': 'state_name',
    'Effective.Minimum.Wage': 'min_wage_nominal',
    'Effective.Minimum.Wage.2020.Dollars': 'min_wage_real_2020',
    'Federal.Minimum.Wage': 'federal_min_wage_nominal',
    'Federal.Minimum.Wage.2020.Dollars': 'federal_min_wage_real_2020'
})

# Merge the datasets
final_df = pd.merge(
    merged_df,
    min_wage_clean,
    on=['YEAR', 'state_name'],
    how='left'
)

# Save the merged dataset
final_df.to_csv('../cleaned_outputs/merged_final.csv', index=False)