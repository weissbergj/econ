import pandas as pd
import numpy as np
from linearmodels import PanelOLS
import matplotlib.pyplot as plt
import seaborn as sns

# No causal effect of 5G on employment strongest support in tech industries
# New Finding: COVID Consolidation Pattern
# 1. Transportation/Warehousing led (+1.62%)
# 2. Arts/Entertainment second (+1.01%)
# 3. Manufacturing third (+0.99%)

def prepare_broadband_analysis():
    """Prepare data for broadband analysis focusing on technology adoption and firm size"""
    
    print("Loading data...")
    df = pd.read_csv('../../archive/cleaned_outputs/merged_state_industry_broadband.csv')
    
    # Create key variables
    df['large_firm_share'] = df['employment_09: 500+'] / df['employment_01: Total']
    df['log_total_employment'] = np.log(df['employment_01: Total'])
    df['high_tech'] = df['industry_name'].isin([
        'Information', 
        'Professional, Scientific, and Technical Services',
        'Finance and Insurance'
    ]).astype(int)
    
    # Create technology treatment
    df['high_tech_coverage'] = (df['LTE_ratio'] > df.groupby('YEAR')['LTE_ratio'].transform('median')).astype(int)
    
    # Focus on state pairs with different tech adoption rates
    state_pairs = {
        'CA_NV': ['California', 'Nevada'],          # Different tech hubs
        'NY_PA': ['New York', 'Pennsylvania'],      # Urban/rural mix
        'TX_NM': ['Texas', 'New Mexico']           # Different adoption speeds
    }
    
    results = {}
    for pair_name, states in state_pairs.items():
        print(f"\nAnalyzing {pair_name}...")
        
        # Get pair data
        pair_data = df[df['state_name'].isin(states)].copy()
        
        # Print diagnostics
        print(f"Years covered: {pair_data['YEAR'].min()} to {pair_data['YEAR'].max()}")
        print(f"Number of industries: {pair_data['industry_name'].nunique()}")
        print(f"LTE coverage range: {pair_data['LTE_ratio'].min():.2f} to {pair_data['LTE_ratio'].max():.2f}")
        
        # Create panel structure
        pair_data['state_industry'] = pair_data['state_name'] + "_" + pair_data['industry_name']
        pair_data = pair_data.set_index(['state_industry', 'YEAR'])
        
        # Create DiD variables
        pair_data['treat_state'] = (pair_data['state_name'] == states[0]).astype(int)
        pair_data['post_tech'] = (pair_data['LTE_ratio'] > 0.75).astype(int)  # High LTE coverage
        pair_data['did'] = pair_data['treat_state'] * pair_data['post_tech']
        pair_data['did_tech_ind'] = pair_data['did'] * pair_data['high_tech']
        
        results[pair_name] = pair_data
        
    return results

# def analyze_5g_effects(df):
#     """Analyze employment effects of 5G implementation (2019)"""
    
#     # Create treatment variables
#     df['post'] = (df['YEAR'] >= 2019).astype(int)  # 5G implementation
#     df['treat_state'] = df['state_name'].isin(['Texas']).astype(int)
#     df['did'] = df['post'] * df['treat_state']
    
#     # Focus on window around treatment
#     df_window = df[df['YEAR'].between(2018, 2020)].copy()  # One year before/after
    
#     # Log employment
#     df_window['log_employment'] = np.log(df_window['employment_01: Total'])
    
#     # Create panel structure
#     df_window['state_industry'] = df_window['state_name'] + "_" + df_window['industry_name']
#     panel_data = df_window.set_index(['state_industry', 'YEAR'])
    
#     # Run DiD
#     model = PanelOLS(
#         dependent=panel_data['log_employment'],
#         exog=panel_data[['did', 'post']],
#         entity_effects=True
#     )
    
#     results = model.fit(cov_type='clustered', cluster_entity=True)
    
#     print("\n5G Implementation Analysis (2019)")
#     print("================================")
#     print("\n5G Coverage:")
#     print(df_window.groupby(['state_name', 'YEAR'])['5G_ratio'].mean().unstack())
    
#     print("\nRegression Results:")
#     print(results.summary.tables[1])
    
#     return results

def comprehensive_5g_test(df):
    """Core test for 5G employment effects"""
    
    print("\nComprehensive 5G Causality Tests")
    print("================================")
    
    # 1. Create clean panel (2018-2020 window)
    df_clean = df[df['YEAR'].between(2018, 2020)].copy()
    
    # 2. Create key variables
    df_clean['log_employment'] = np.log(df_clean['employment_01: Total'])
    df_clean['post'] = (df_clean['YEAR'] >= 2019).astype(int)
    
    # Use actual 5G ratios instead of binary treatment
    df_clean['did'] = df_clean['post'] * df_clean['5G_ratio']
    
    # Create panel ID and set index
    df_clean['panel_id'] = df_clean['state_name'] + "_" + df_clean['industry_name']
    df_clean = df_clean.set_index(['panel_id', 'YEAR'])
    
    # 3. Basic DiD with continuous treatment
    print("\nBaseline DiD (Continuous 5G Treatment):")
    model = PanelOLS(
        dependent=df_clean['log_employment'],
        exog=df_clean[['did', 'post']],
        entity_effects=True,
        check_rank=False  # Allow estimation with rank issues
    )
    results = model.fit(cov_type='clustered', cluster_entity=True)
    print(results.summary.tables[1])
    
    # 4. Summary statistics
    df_reset = df_clean.reset_index()
    
    print("\n5G Coverage by Year:")
    coverage = df_reset.groupby('YEAR')['5G_ratio'].agg(['mean', 'std', 'min', 'max'])
    print(coverage)
    
    print("\nTop 5 States by 2019 5G Coverage:")
    top_states = df_reset[df_reset['YEAR'] == 2019].groupby('state_name')['5G_ratio'].mean().sort_values(ascending=False).head()
    print(top_states)
    
    print("\nBottom 5 States by 2019 5G Coverage:")
    bottom_states = df_reset[df_reset['YEAR'] == 2019].groupby('state_name')['5G_ratio'].mean().sort_values().head()
    print(bottom_states)
    
    # 5. Employment changes
    print("\nEmployment Growth 2018-2020:")
    emp_change = df_reset.groupby('YEAR')['log_employment'].mean()
    print(emp_change)
    
    return results

def find_causality(df):
    """Focus on 5G rollout in tech-intensive industries (2018-2020)"""
    
    print("\nSearching for Causal Effects in Tech Industries")
    print("============================================")
    
    # 1. Focus on tech-intensive industries
    tech_industries = [
        'Information',
        'Professional, Scientific, and Technical Services',
        'Transportation and Warehousing'
    ]
    
    # 2. Clean 5G window (2018-2020)
    df_clean = df[
        (df['industry_name'].isin(tech_industries)) &
        (df['YEAR'].between(2018, 2020))
    ].copy()
    
    # 3. Variables
    df_clean['log_employment'] = np.log(df_clean['employment_01: Total'])
    df_clean['post'] = (df_clean['YEAR'] >= 2019).astype(int)
    df_clean['treatment'] = df_clean['5G_ratio']  # Continuous treatment
    df_clean['did'] = df_clean['post'] * df_clean['treatment']
    
    # 4. Panel structure
    df_clean['panel_id'] = df_clean['state_name'] + "_" + df_clean['industry_name']
    df_clean = df_clean.set_index(['panel_id', 'YEAR'])
    
    # 5. Run analysis
    model = PanelOLS(
        dependent=df_clean['log_employment'],
        exog=df_clean[['did', 'post']],
        entity_effects=True,
        check_rank=False
    )
    results = model.fit(cov_type='clustered', cluster_entity=True)
    
    print("\nBaseline Results (Tech Industries Only):")
    print(results.summary.tables[1])
    
    # 6. Industry breakdown
    df_reset = df_clean.reset_index()
    print("\nEffects by Industry:")
    for industry in tech_industries:
        ind_data = df_reset[df_reset['industry_name'] == industry].copy()
        ind_data = ind_data.set_index(['panel_id', 'YEAR'])
        model = PanelOLS(
            dependent=ind_data['log_employment'],
            exog=ind_data[['did', 'post']],
            entity_effects=True,
            check_rank=False
        )
        try:
            ind_results = model.fit(cov_type='clustered', cluster_entity=True)
            print(f"\n{industry}:")
            print(f"Effect: {ind_results.params['did']:.3f}")
            print(f"P-value: {ind_results.pvalues['did']:.3f}")
        except:
            continue
    
    return results

def analyze_5g_effects(df):
    """Analyze 5G effects on firm structure and wages"""
    
    print("\nAnalyzing 5G Effects on Market Structure")
    print("=====================================")
    
    # 1. Setup panel (2018-2020)
    df_clean = df[df['YEAR'].between(2018, 2020)].copy()
    
    # 2. Create outcome variables
    df_clean['large_firm_share'] = df_clean['employment_09: 500+'] / df_clean['employment_01: Total']
    df_clean['small_firm_share'] = df_clean['employment_02: <5'] / df_clean['employment_01: Total']
    df_clean['avg_wage'] = df_clean['annual_payroll_01: Total'] / df_clean['employment_01: Total']
    df_clean['log_avg_wage'] = np.log(df_clean['avg_wage'])
    
    # 3. Treatment variables
    df_clean['post'] = (df_clean['YEAR'] >= 2019).astype(int)
    df_clean['treatment'] = df_clean['5G_ratio']
    df_clean['did'] = df_clean['post'] * df_clean['treatment']
    
    # 4. Panel structure
    df_clean['panel_id'] = df_clean['state_name'] + "_" + df_clean['industry_name']
    df_clean = df_clean.set_index(['panel_id', 'YEAR'])
    
    # 5. Test multiple outcomes
    outcomes = {
        'large_firm_share': 'Effect on Large Firm Share',
        'small_firm_share': 'Effect on Small Firm Share',
        'log_avg_wage': 'Effect on Log Average Wages'
    }
    
    for var, title in outcomes.items():
        print(f"\n{title}:")
        model = PanelOLS(
            dependent=df_clean[var],
            exog=df_clean[['did', 'post']],
            entity_effects=True,
            check_rank=False
        )
        results = model.fit(cov_type='clustered', cluster_entity=True)
        print(results.summary.tables[1])
    
    # 6. Industry heterogeneity
    print("\nTop Industries by Large Firm Share Change:")
    df_reset = df_clean.reset_index()
    industry_effects = df_reset.groupby('industry_name').apply(
        lambda x: np.mean(x[x['YEAR'] >= 2019]['large_firm_share']) - 
                 np.mean(x[x['YEAR'] < 2019]['large_firm_share'])
    ).sort_values(ascending=False)
    print(industry_effects.head())
    
    return df_clean

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('../../archive/cleaned_outputs/merged_state_industry_broadband.csv')
    
    # Run core test
    # results = comprehensive_5g_test(df)
    df_analyzed = analyze_5g_effects(df)

    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 