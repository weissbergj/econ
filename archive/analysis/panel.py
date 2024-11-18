import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS

def mega_panel_analysis(df):
    """Panel analysis of 5G effects on industry structure"""
    
    print("\nMEGA PANEL ANALYSIS")
    print("==================")
    
    # 1. Create core outcome variables (with safety checks)
    df['large_firm_share'] = (df['employment_09: 500+'] / df['employment_01: Total']).fillna(0)
    df['wage_premium'] = (df['annual_payroll_01: Total'] / df['employment_01: Total']).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 2. Create treatment variables
    df['post_2019'] = (df['YEAR'] >= 2019).astype(int)
    df['high_5g'] = (df['5G_ratio'] > df.groupby('YEAR')['5G_ratio'].transform('median')).astype(int)
    df['treatment'] = df['post_2019'] * df['high_5g']
    
    # 3. Industry groups
    tech_industries = ['Information', 'Professional, Scientific, and Technical Services']
    service_industries = ['Retail Trade', 'Accommodation and Food Services']
    df['is_tech'] = df['industry_name'].isin(tech_industries).astype(int)
    df['is_service'] = df['industry_name'].isin(service_industries).astype(int)
    
    # 4. Data quality report
    print("\nData Summary:")
    print(f"Years: {df['YEAR'].min()}-{df['YEAR'].max()}")
    print(f"States: {df['state_name'].nunique()}")
    print(f"Industries: {df['industry_name'].nunique()}")
    print("\nMean Values by Period:")
    print(df.groupby('post_2019')[['large_firm_share', 'wage_premium']].mean())
    
    # 5. Panel setup
    df['entity'] = df['state_name'] + "_" + df['industry_name']
    df = df.set_index(['entity', 'YEAR'])
    
    # 6. Run regressions for each outcome
    outcomes = ['wage_premium', 'large_firm_share']
    samples = {
        'All Industries': df,
        'Tech Industries': df[df['is_tech'] == 1],
        'Service Industries': df[df['is_service'] == 1]
    }
    
    for outcome in outcomes:
        print(f"\n=== {outcome.replace('_', ' ').title()} Analysis ===")
        
        for sample_name, sample_df in samples.items():
            if len(sample_df) > 0:
                try:
                    # Standardize outcome for easier interpretation
                    y = (sample_df[outcome] - sample_df[outcome].mean()) / sample_df[outcome].std()
                    
                    model = PanelOLS(
                        dependent=y,
                        exog=sample_df[['treatment']],
                        entity_effects=True,
                        time_effects=True
                    )
                    
                    results = model.fit(cov_type='clustered', cluster_entity=True)
                    
                    print(f"\n{sample_name}:")
                    print(f"Treatment Effect: {results.params['treatment']:.3f}")
                    print(f"Standard Error: {results.std_errors['treatment']:.3f}")
                    print(f"P-value: {results.pvalues['treatment']:.3f}")
                    print(f"N: {len(sample_df)}")
                    
                except Exception as e:
                    print(f"Error in {sample_name}: {str(e)}")
    
    # 7. State-level effects
    print("\nTop 5 States by Treatment Effect (2019+):")
    recent = df[df['post_2019'] == 1].reset_index()
    state_effects = recent.groupby('state_name').agg({
        'treatment': 'mean',
        'wage_premium': 'mean',
        'large_firm_share': 'mean'
    }).round(3)
    print(state_effects.sort_values('treatment', ascending=False).head())
    
    return df

def explore_panel(df):
    """Explore all variables in our panel"""
    
    print("\nPANEL DATA CONTENTS")
    print("==================")
    
    # 1. Employment Variables
    print("\nEmployment Measures:")
    emp_cols = [col for col in df.columns if 'employment' in col.lower()]
    for col in emp_cols:
        print(f"- {col}")
    
    # 2. Firm Size Categories
    print("\nFirm Size Categories:")
    firm_cols = [col for col in df.columns if 'firms' in col.lower()]
    for col in firm_cols:
        print(f"- {col}")
        
    # 3. Establishment Data
    print("\nEstablishment Measures:")
    est_cols = [col for col in df.columns if 'establishments' in col.lower()]
    for col in est_cols:
        print(f"- {col}")
    
    # 4. Wage/Payroll Data
    print("\nWage/Payroll Measures:")
    wage_cols = [col for col in df.columns if 'payroll' in col.lower() or 'wage' in col.lower()]
    for col in wage_cols:
        print(f"- {col}")
    
    # 5. Technology Variables
    print("\nTechnology Variables:")
    tech_cols = ['LTE', '5G', 'legacy_tech', 'total_coverage', 'LTE_ratio', '5G_ratio']
    for col in tech_cols:
        print(f"- {col}")
    
    # 6. Policy Variables
    print("\nPolicy Variables:")
    policy_cols = [col for col in df.columns if 'min_wage' in col.lower()]
    for col in policy_cols:
        print(f"- {col}")
    
    # 7. Time Coverage
    print("\nTime Coverage:")
    print(f"Years: {df['YEAR'].min()} to {df['YEAR'].max()}")
    print(f"Number of years: {df['YEAR'].nunique()}")
    
    # 8. Geographic Coverage
    print("\nGeographic Coverage:")
    print(f"Number of states: {df['state_name'].nunique()}")
    print("Top 5 states by total employment:")
    top_states = df.groupby('state_name')['employment_01: Total'].mean().sort_values(ascending=False).head()
    print(top_states)
    
    # 9. Industry Coverage
    print("\nIndustry Coverage:")
    print(f"Number of industries: {df['industry_name'].nunique()}")
    print("\nIndustries:")
    print(df['industry_name'].unique())
    
    return df

def main():
    print("Loading panel data...")
    df = pd.read_csv('../../data/merged_final.csv')
    df_analyzed = mega_panel_analysis(df)
    df_explored = explore_panel(df)
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()