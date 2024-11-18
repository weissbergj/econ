import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels import PanelOLS
import statsmodels.api as sm

def load_and_prepare_data():
    """Load and prepare data for concentration analysis"""
    df = pd.read_csv('../../archive/cleaned_outputs/merged_state_industry_broadband.csv')
    
    # Create concentration measures
    df['large_firm_share'] = df['firms_07: 100-499'] / df['firms_01: Total']
    df['very_large_firm_share'] = df['firms_09: 500+'] / df['firms_01: Total']
    
    # Create wage inequality measures
    df['wage_inequality'] = df['mean_wage'] - df['median_wage']
    df['wage_ratio'] = df['mean_wage'] / df['median_wage']
    
    # Create digital industry indicator
    digital_industries = [
        'Information',
        'Finance and Insurance',
        'Professional, Scientific, and Technical Services'
    ]
    df['digital'] = df['industry_name'].isin(digital_industries).astype(int)
    
    # Create state-industry identifier
    df['state_industry'] = df['state_name'] + "_" + df['industry_name']
    
    # Calculate employment share in large firms
    df['large_emp_share'] = (df['employment_07: 100-499'] + df['employment_09: 500+']) / df['employment_01: Total']
    
    print("\nSummary Statistics:")
    print(df.groupby('digital')[['large_firm_share', 'large_emp_share', 'wage_inequality']].describe().round(2))
    
    return df

def run_concentration_analysis(df):
    """Analyze relationship between concentration and wage inequality"""
    # Create interaction terms
    df['concentration_digital'] = df['large_emp_share'] * df['digital']
    
    # Panel regression
    panel_data = df.set_index(['state_industry', 'YEAR'])
    
    model = PanelOLS(
        dependent=panel_data['wage_inequality'],
        exog=panel_data[[
            'large_emp_share',
            'concentration_digital'
        ]],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True
    )
    
    return model.fit(cov_type='clustered', cluster_entity=True)

def plot_relationships(df):
    """Create visualizations of key relationships"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Concentration vs Wage Inequality by Digital Status
    sns.scatterplot(
        data=df,
        x='large_emp_share',
        y='wage_inequality',
        hue='digital',
        alpha=0.5,
        ax=ax1
    )
    ax1.set_title('Wage Inequality vs Large Firm Employment Share')
    ax1.set_xlabel('Share of Employment in Large Firms')
    ax1.set_ylabel('Wage Inequality (Mean - Median)')
    
    # Plot 2: Time Trends
    yearly_data = df.groupby(['YEAR', 'digital'])[['large_emp_share', 'wage_inequality']].mean().reset_index()
    
    sns.lineplot(
        data=yearly_data,
        x='YEAR',
        y='wage_inequality',
        hue='digital',
        style='digital',
        markers=True,
        ax=ax2
    )
    ax2.set_title('Wage Inequality Trends Over Time')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Wage Inequality')
    
    plt.tight_layout()
    plt.savefig('concentration_inequality_analysis.png')
    plt.close()

def analyze_by_state_size(df):
    """Analyze effects by state size"""
    # Create state size indicator based on total employment
    state_size = df.groupby('state_name')['employment_01: Total'].mean()
    df['large_state'] = df['state_name'].map(state_size > state_size.median())
    
    results = {}
    for is_large in [True, False]:
        subset = df[df['large_state'] == is_large]
        panel_data = subset.set_index(['state_industry', 'YEAR'])
        
        model = PanelOLS(
            dependent=panel_data['wage_inequality'],
            exog=panel_data[[
                'large_emp_share',
                'concentration_digital'
            ]],
            entity_effects=True,
            time_effects=True,
            drop_absorbed=True
        )
        results[f'{"large" if is_large else "small"}_states'] = model.fit(cov_type='clustered', cluster_entity=True)
    
    return results

def main():
    # Load and prepare data
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    
    # Run main analysis
    print("\nRunning concentration analysis...")
    results = run_concentration_analysis(df)
    print("\nMain Results:")
    print(results.summary.tables[1])
    
    # Analyze by state size
    print("\nAnalyzing by state size...")
    state_results = analyze_by_state_size(df)
    for name, res in state_results.items():
        print(f"\n{name.title()} Results:")
        print(res.summary.tables[1])
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_relationships(df)

if __name__ == "__main__":
    main() 