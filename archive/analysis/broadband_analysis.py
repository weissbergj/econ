import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels import PanelOLS
import statsmodels.api as sm
from scipy import stats

def load_and_prepare_data():
    """Load and prepare data for broadband analysis"""
    df = pd.read_csv('../../data/merged_final.csv')
    
    # Create state_industry identifier
    df['state_industry'] = df['state_name'] + "_" + df['industry_name']
    
    # Create digital industry indicator
    digital_industries = [
        'Information',
        'Finance and Insurance',
        'Professional, Scientific, and Technical Services'
    ]
    df['digital'] = df['industry_name'].isin(digital_industries).astype(int)
    
    # Create wage inequality measure
    df['wage_inequality'] = df['mean_wage'] - df['median_wage']
    
    # Limit to 2015-2021 period for cleaner analysis
    df = df[(df['YEAR'] >= 2015) & (df['YEAR'] <= 2021)].copy()
    
    # Create 5G treatment indicator (post-2019)
    df['post_5g'] = (df['YEAR'] >= 2019).astype(int)
    
    # Create high_broadband indicator (above median LTE coverage in 2018)
    df_2018 = df[df['YEAR'] == 2018].copy()
    median_lte = df_2018['LTE'].median()
    state_high_broadband = df_2018[df_2018['LTE'] > median_lte]['state'].unique()
    df['high_broadband'] = df['state'].isin(state_high_broadband).astype(int)
    
    print("\nDataset Overview:")
    print(f"Years covered: {df['YEAR'].min()} to {df['YEAR'].max()}")
    print(f"Number of states: {df['state_name'].nunique()}")
    print(f"Number of industries: {df['industry_name'].nunique()}")
    print(f"Number of digital industries: {df['digital'].sum() / df['YEAR'].nunique()}")
    
    return df

def verify_parallel_trends(df):
    """Test parallel trends assumption pre-2019"""
    pre_2019 = df[df['YEAR'] < 2019].copy()
    
    # Create year dummies (using 2015 as base year)
    years = sorted(pre_2019['YEAR'].unique())[1:]  # Exclude 2015 as base
    for year in years:
        pre_2019[f'year_{year}'] = (pre_2019['YEAR'] == year).astype(int)
        pre_2019[f'digital_x_{year}'] = pre_2019['digital'] * pre_2019[f'year_{year}']
    
    # Set up variables for regression
    year_dummies = [f'year_{year}' for year in years]
    interactions = [f'digital_x_{year}' for year in years]
    
    # Run regression with absorbed effects
    panel_data = pre_2019.set_index(['state_industry', 'YEAR'])
    model = PanelOLS(
        dependent=panel_data['wage_inequality'],
        exog=panel_data[year_dummies + interactions],  # Remove constant and base terms
        entity_effects=True,
        drop_absorbed=True  # Add this parameter
    )
    
    results = model.fit(cov_type='clustered', cluster_entity=True)
    return results

def run_triple_diff_analysis(df):
    """Run triple difference analysis"""
    # Create interaction terms
    df['digital_post'] = df['digital'] * df['post_5g']
    df['digital_high'] = df['digital'] * df['high_broadband']
    df['post_high'] = df['post_5g'] * df['high_broadband']
    df['triple_int'] = df['digital'] * df['post_5g'] * df['high_broadband']
    
    # Set up panel data
    panel_data = df.set_index(['state_industry', 'YEAR'])
    
    # Run regression
    model = PanelOLS(
        dependent=panel_data['wage_inequality'],
        exog=panel_data[[
            'digital_post', 'digital_high', 'post_high',
            'triple_int', 'workers'
        ]],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True
    )
    
    results = model.fit(cov_type='clustered', cluster_entity=True)
    return results

def plot_event_study(df):
    """Create event study plot"""
    event_data = df.groupby(['YEAR', 'digital', 'high_broadband'])['wage_inequality'].mean().reset_index()
    
    plt.figure(figsize=(12, 8))
    
    for digital in [0, 1]:
        for high_bb in [0, 1]:
            data = event_data[
                (event_data['digital'] == digital) & 
                (event_data['high_broadband'] == high_bb)
            ]
            label = f"{'Digital' if digital else 'Non-Digital'}, {'High' if high_bb else 'Low'} Broadband"
            plt.plot(data['YEAR'], data['wage_inequality'], marker='o', label=label)
    
    plt.axvline(x=2019, color='red', linestyle='--', alpha=0.5, label='5G Introduction')
    plt.title('Wage Inequality Evolution by Industry Type and Broadband Access')
    plt.xlabel('Year')
    plt.ylabel('Wage Inequality')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('event_study_plot.png', bbox_inches='tight', dpi=300)
    plt.close()

def run_causal_analysis(df):
    """Enhanced causal analysis with better controls and robustness"""
    
    # 1. Create state-level controls
    df['log_total_employment'] = np.log(df.groupby(['state_name', 'YEAR'])['workers'].transform('sum'))
    df['industry_concentration'] = df.groupby(['state_name', 'YEAR'])['workers'].transform('std')
    
    # 2. Add industry-specific time trends
    for ind in df['industry_name'].unique():
        df[f'trend_{ind}'] = (df['industry_name'] == ind) * df['YEAR']
    
    # 3. Create size quartiles
    df['size_quartile'] = pd.qcut(df['workers'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    # 4. Exclude COVID period for clean identification
    df_clean = df[df['YEAR'] < 2020].copy()
    
    # 5. Run matched DiD
    def get_matched_sample(data):
        # Match on pre-treatment characteristics
        pre_treatment = data[data['YEAR'] < 2019].groupby('state_industry').agg({
            'workers': 'mean',
            'wage_inequality': 'mean',
            'total_wage_bill': 'mean'
        }).reset_index()
        # Add matching logic here
        return pre_treatment['state_industry'].tolist()
    
    matched_sample = get_matched_sample(df_clean)
    df_matched = df_clean[df_clean['state_industry'].isin(matched_sample)]
    
    # 6. Run regression with enhanced controls
    panel_data = df_matched.set_index(['state_industry', 'YEAR'])
    
    model = PanelOLS(
        dependent=panel_data['wage_inequality'],
        exog=panel_data[[
            'digital_post', 'digital_high', 'post_high', 'triple_int',
            'log_total_employment', 'industry_concentration',
            'workers'
        ]].join(pd.get_dummies(df_matched['size_quartile'], drop_first=True)),
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True
    )
    
    return model.fit(cov_type='clustered', cluster_entity=True)

def run_placebo_tests(df):
    """Run placebo tests with fake treatment years"""
    results = {}
    for placebo_year in [2016, 2017, 2018]:
        df[f'post_{placebo_year}'] = (df['YEAR'] >= placebo_year).astype(int)
        df[f'digital_post_{placebo_year}'] = df['digital'] * df[f'post_{placebo_year}']
        # Run same specification with placebo treatment
        # Store results
    return results

def test_heterogeneous_effects(df):
    """Test for heterogeneous effects by firm size"""
    # Split sample analyses
    results = {}
    for quartile in df['size_quartile'].unique():
        subset = df[df['size_quartile'] == quartile]
        # Run main specification on each subset
        # Store results
    return results

def run_staggered_analysis(df):
    """Run staggered adoption event study"""
    # Define treatment based on 5G adoption
    df['has_5G'] = df['5G'].notna() & (df['5G'] > 0)
    
    # Get first 5G adoption year for each state
    first_5G = df[df['has_5G']].groupby('state')['YEAR'].min()
    df['treatment_year'] = df['state'].map(first_5G)
    
    # Calculate relative time to treatment
    df['rel_time'] = df['YEAR'] - df['treatment_year']
    
    # Create relative time dummies (from -3 to +2, excluding -1)
    for t in range(-3, 3):
        if t != -1:  # -1 is reference period
            df[f'rel_time_{t}'] = (df['rel_time'] == t).astype(int)
            df[f'digital_x_rel_time_{t}'] = df['digital'] * (df['rel_time'] == t).astype(int)
    
    # Set up panel data
    panel_data = df.set_index(['state_industry', 'YEAR'])
    
    # Variables for regression
    rel_time_vars = [col for col in df.columns if col.startswith('rel_time_')]
    interaction_vars = [col for col in df.columns if col.startswith('digital_x_rel_time_')]
    controls = ['workers', 'total_wage_bill']
    
    # Run regression
    model = PanelOLS(
        dependent=panel_data['wage_inequality'],
        exog=panel_data[rel_time_vars + interaction_vars + controls],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True
    )
    
    results = model.fit(cov_type='clustered', cluster_entity=True)
    return results, df  # Return both results and modified dataframe

def plot_staggered_event_study(df, results):
    """Create event study plot with confidence intervals for staggered adoption"""
    # Extract coefficients and standard errors
    coef = results.params
    se = results.std_errors
    
    # Create plot data
    plot_data = pd.DataFrame({
        'time': range(-3, 3),
        'coef': [coef.get(f'digital_x_rel_time_{t}', 0) if t != -1 else 0 for t in range(-3, 3)],
        'se': [se.get(f'digital_x_rel_time_{t}', 0) if t != -1 else 0 for t in range(-3, 3)]
    })
    
    plt.figure(figsize=(12, 8))
    plt.plot(plot_data['time'], plot_data['coef'], marker='o', color='blue', label='Coefficient')
    plt.fill_between(
        plot_data['time'],
        plot_data['coef'] - 1.96 * plot_data['se'],
        plot_data['coef'] + 1.96 * plot_data['se'],
        alpha=0.2,
        color='blue'
    )
    
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='5G Introduction')
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    plt.title('Effect of 5G on Wage Inequality: Staggered Adoption')
    plt.xlabel('Years Relative to State 5G Introduction')
    plt.ylabel('Effect on Wage Inequality')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('staggered_event_study.png', bbox_inches='tight', dpi=300)
    plt.close()

def run_robustness_checks(df):
    """Run comprehensive robustness checks"""
    results = {}
    
    # 1. Placebo tests with different years
    for placebo_year in [2016, 2017, 2018]:
        df_placebo = df.copy()
        df_placebo['post_5g'] = (df_placebo['YEAR'] >= placebo_year).astype(int)
        df_placebo['digital_post'] = df_placebo['digital'] * df_placebo['post_5g']
        df_placebo['digital_high'] = df_placebo['digital'] * df_placebo['high_broadband']
        df_placebo['post_high'] = df_placebo['post_5g'] * df_placebo['high_broadband']
        df_placebo['triple_int'] = df_placebo['digital'] * df_placebo['post_5g'] * df_placebo['high_broadband']
        
        panel_data = df_placebo.set_index(['state_industry', 'YEAR'])
        model = PanelOLS(
            dependent=panel_data['wage_inequality'],
            exog=panel_data[[
                'digital_post', 'digital_high', 'post_high',
                'triple_int', 'workers'
            ]],
            entity_effects=True,
            time_effects=True,
            drop_absorbed=True
        )
        results[f'placebo_{placebo_year}'] = model.fit(cov_type='clustered', cluster_entity=True)
    
    # 2. Sample restrictions
    # Exclude COVID period
    df_no_covid = df[df['YEAR'] < 2020].copy()
    panel_data = df_no_covid.set_index(['state_industry', 'YEAR'])
    model = PanelOLS(
        dependent=panel_data['wage_inequality'],
        exog=panel_data[[
            'digital_post', 'digital_high', 'post_high',
            'triple_int', 'workers'
        ]],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True
    )
    results['no_covid'] = model.fit(cov_type='clustered', cluster_entity=True)
    
    # 3. Alternative treatment definition
    # Using continuous 5G measure instead of binary
    df_continuous = df.copy()
    df_continuous['5G_intensity'] = df_continuous['5G'].fillna(0)
    df_continuous['digital_5G'] = df_continuous['digital'] * df_continuous['5G_intensity']
    
    panel_data = df_continuous.set_index(['state_industry', 'YEAR'])
    model = PanelOLS(
        dependent=panel_data['wage_inequality'],
        exog=panel_data[[
            'digital_5G', 'workers'
        ]],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True
    )
    results['continuous_5G'] = model.fit(cov_type='clustered', cluster_entity=True)
    
    # 4. Alternative industry definitions
    # Using broader digital industry definition
    broader_digital = [
        'Information',
        'Finance and Insurance',
        'Professional, Scientific, and Technical Services',
        'Management of Companies and Enterprises',
        'Administrative and Support Services'
    ]
    
    df_broad = df.copy()
    df_broad['digital'] = df_broad['industry_name'].isin(broader_digital).astype(int)
    df_broad['digital_post'] = df_broad['digital'] * df_broad['post_5g']
    df_broad['digital_high'] = df_broad['digital'] * df_broad['high_broadband']
    df_broad['post_high'] = df_broad['post_5g'] * df_broad['high_broadband']
    df_broad['triple_int'] = df_broad['digital'] * df_broad['post_5g'] * df_broad['high_broadband']
    
    panel_data = df_broad.set_index(['state_industry', 'YEAR'])
    model = PanelOLS(
        dependent=panel_data['wage_inequality'],
        exog=panel_data[[
            'digital_post', 'digital_high', 'post_high',
            'triple_int', 'workers'
        ]],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True
    )
    results['broader_digital'] = model.fit(cov_type='clustered', cluster_entity=True)
    
    return results

def main():
    # Load and prepare data
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    
    # Verify parallel trends
    print("\nTesting parallel trends assumption...")
    parallel_trends = verify_parallel_trends(df)
    print("\nParallel Trends Test Results:")
    print(parallel_trends.summary.tables[1])
    
    # Run main triple-diff analysis
    print("\nRunning triple-difference analysis...")
    triple_diff = run_triple_diff_analysis(df)
    print("\nTriple-Difference Results:")
    print(triple_diff.summary.tables[1])
    
    # Create event study plot
    print("\nCreating event study plot...")
    plot_event_study(df)
    
    # Run robustness checks
    print("\nRunning robustness checks...")
    robustness = run_robustness_checks(df)
    for name, results in robustness.items():
        print(f"\n{name} results:")
        print(results.summary.tables[1])
    
    # Add staggered adoption analysis
    print("\nRunning staggered adoption analysis...")
    staggered_results, df_staggered = run_staggered_analysis(df)
    print("\nStaggered Adoption Results:")
    print(staggered_results.summary.tables[1])
    
    # Create staggered event study plot
    print("\nCreating staggered event study plot...")
    plot_staggered_event_study(df_staggered, staggered_results)
    
    # Print adoption timing information
    print("\n5G Adoption Timing:")
    adoption_years = df_staggered[df_staggered['treatment_year'].notna()]['treatment_year'].value_counts().sort_index()
    print(adoption_years)

if __name__ == "__main__":
    main() 