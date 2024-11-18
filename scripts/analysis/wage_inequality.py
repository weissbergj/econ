import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

# "In one well-identified case (CA vs NV, 2012-2016), a minimum wage increase was associated with higher wage inequality 
# as measured by the mean-median gap. While this provides causal evidence in this specific context, we should be cautious 
# about generalizing this finding to other settings or claiming minimum wages always increase inequality."

# Define experiments at module level
experiments = {
    # 'WA_ID': {
    #     'treat_state': 'Washington',
    #     'control_state': 'Idaho',
    #     'treatment_year': 2017  # Big jump to $11.00
    # },
    # 'MA_NH': {
    #     'treat_state': 'Massachusetts',
    #     'control_state': 'New Hampshire',
    #     'treatment_year': 2015  # Jump to $9.00
    # },
    'CA_NV': {
        'treat_state': 'California',
        'control_state': 'Nevada',
        'treatment_year': 2014  # Jump to $9.00
    },
    # 'OR_ID': {
    #     'treat_state': 'Oregon',
    #     'control_state': 'Idaho',
    #     'treatment_year': 2016
    # },
    # 'NY_PA': {
    #     'treat_state': 'New York',
    #     'control_state': 'Pennsylvania',
    #     'treatment_year': 2014
    # },
    # 'VT_NH': {
    #     'treat_state': 'Vermont',
    #     'control_state': 'New Hampshire',
    #     'treatment_year': 2015
    # },
    # 'CT_NH': {
    #     'treat_state': 'Connecticut',
    #     'control_state': 'New Hampshire',
    #     'treatment_year': 2015
    # },
    # 'RI_NH': {
    #     'treat_state': 'Rhode Island',
    #     'control_state': 'New Hampshire',
    #     'treatment_year': 2014
    # },
    # 'IL_IN': {
    #     'treat_state': 'Illinois',
    #     'control_state': 'Indiana',
    #     'treatment_year': 2015
    # },
    # 'MI_IN': {
    #     'treat_state': 'Michigan',
    #     'control_state': 'Indiana',
    #     'treatment_year': 2014
    # },
    # 'MN_WI': {
    #     'treat_state': 'Minnesota',
    #     'control_state': 'Wisconsin',
    #     'treatment_year': 2014
    # },
    # 'DE_PA': {
    #     'treat_state': 'Delaware',
    #     'control_state': 'Pennsylvania',
    #     'treatment_year': 2015
    # },
    # 'WV_VA': {
    #     'treat_state': 'West Virginia',
    #     'control_state': 'Virginia',
    #     'treatment_year': 2015
    # },
    # 'MD_VA': {
    #     'treat_state': 'Maryland',
    #     'control_state': 'Virginia',
    #     'treatment_year': 2015
    # }
}

def calculate_wage_gini(df):
    """Calculate Gini coefficient for wage distribution."""
    size_categories = [
        ('employment_02: <5', 'annual_payroll_02: <5'),
        ('employment_03: 5-9', 'annual_payroll_03: 5-9'),
        ('employment_04: 10-19', 'annual_payroll_04: 10-19'),
        ('employment_06: 20-99', 'annual_payroll_06: 20-99'),
        ('employment_07: 100-499', 'annual_payroll_07: 100-499'),
        ('employment_09: 500+', 'annual_payroll_09: 500+')
    ]
    
    wages = []
    weights = []
    
    # Collect non-zero wage data
    for emp_col, pay_col in size_categories:
        emp = float(df[emp_col])
        pay = float(df[pay_col])
        if emp > 0 and pay > 0:  # Ensure both are positive
            avg_wage = pay / emp
            if avg_wage > 0:  # Additional check for positive wages
                wages.append(avg_wage)
                weights.append(emp)
    
    # Return 0 if insufficient data
    if len(wages) < 2:
        return 0
    
    # Convert to numpy arrays
    wages = np.array(wages, dtype=float)
    weights = np.array(weights, dtype=float)
    
    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)
    
    # Sort both arrays by wage
    sorted_indices = np.argsort(wages)
    wages = wages[sorted_indices]
    weights = weights[sorted_indices]
    
    # Calculate cumulative proportions
    cumsum_weights = np.cumsum(weights)
    cumsum_weighted_wages = np.cumsum(weights * wages)
    total_weighted_wages = np.sum(weights * wages)
    
    # Ensure no division by zero
    if total_weighted_wages == 0:
        return 0
    
    # Calculate Lorenz curve points
    lorenz_curve = cumsum_weighted_wages / total_weighted_wages
    
    # Calculate Gini coefficient
    # G = 1 - 2 * area under Lorenz curve
    # Area = sum of trapezoids
    gini = 1 - np.sum((cumsum_weights[1:] - cumsum_weights[:-1]) * 
                      (lorenz_curve[1:] + lorenz_curve[:-1]))
    
    return gini

def load_and_prepare_data():
    """Load and prepare data for minimum wage analysis"""
    df = pd.read_csv('../../data/merged_final.csv')
    
    # Calculate minimum wage "bite"
    df['min_wage_premium'] = df['min_wage_nominal'] - df['federal_min_wage_nominal']
    
    # Create wage inequality measure
    df['wage_inequality'] = df['mean_wage'] - df['median_wage']
    
    # Print available states and years
    print("\nAvailable States:")
    print(df['state_name'].unique())
    print("\nYears covered:")
    print(df['YEAR'].unique())
    
    # Print minimum wage variation
    print("\nMinimum Wage Summary by State and Year:")
    print(df.groupby(['state_name', 'YEAR'])['min_wage_nominal'].mean().unstack())
    
    return df

def simple_analysis(df):
    """Simple analysis of minimum wage effects"""
    # Create high/low minimum wage indicator
    df['high_min_wage'] = df['min_wage_nominal'] > df['min_wage_nominal'].median()
    
    # Print basic comparisons
    print("\nWage Inequality by Minimum Wage Level:")
    print(df.groupby('high_min_wage')['wage_inequality'].describe())
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='YEAR', y='wage_inequality', hue='high_min_wage')
    plt.title('Wage Inequality by Minimum Wage Level Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('basic_min_wage_analysis.png')
    plt.close()
    
    return df

def analyze_border_states(df):
    """Analyze specific border state pairs"""
    border_pairs = [
        ('NY', 'PA'), 
        ('MA', 'NH'), 
        ('CA', 'NV')
    ]
    
    print("\nBorder State Analysis:")
    for state1, state2 in border_pairs:
        pair_data = df[df['state_name'].isin([state1, state2])]
        if len(pair_data) > 0:
            print(f"\n{state1}-{state2} Pair:")
            print("Minimum Wage Difference:")
            min_wage_diff = pair_data.groupby(['state_name', 'YEAR'])['min_wage_nominal'].mean().unstack()
            print(min_wage_diff)
            print("\nWage Inequality Difference:")
            inequality_diff = pair_data.groupby(['state_name', 'YEAR'])['wage_inequality'].mean().unstack()
            print(inequality_diff)

def analyze_natural_experiments(df):
    """Analyze natural experiments with both measures"""
    results = {}  # Initialize results dictionary
    
    # First, clean the data
    analysis_df = df.copy()
    
    # Calculate both inequality measures
    analysis_df['wage_inequality'] = analysis_df['mean_wage'] - analysis_df['median_wage']
    analysis_df['wage_gini'] = analysis_df.apply(calculate_wage_gini, axis=1)
    
    # Aggregate to state-year level
    state_year_df = analysis_df.groupby(['state_name', 'YEAR']).agg({
        'wage_inequality': 'mean',
        'wage_gini': 'mean',
        'min_wage_nominal': 'first'
    }).reset_index()
    
    for name, exp in experiments.items():
        if name == 'CA_NV':
            print(f"\nAnalyzing {name}...")
            
            # Get data for this pair
            pair_data = state_year_df[
                (state_year_df['state_name'].isin([exp['treat_state'], exp['control_state']])) &
                (state_year_df['YEAR'] < 2020)
            ].copy()
            
            # Store regression results
            results[name] = {
                'data': pair_data,
                'summary': pair_data.groupby(['state_name', 'YEAR']).agg({
                    'wage_inequality': 'mean',
                    'wage_gini': 'mean',
                    'min_wage_nominal': 'first'
                })
            }
            
            # ... [rest of your analysis code] ...
            
            # Run main regression for both measures
            for measure in ['wage_inequality', 'wage_gini']:
                X = pd.DataFrame({
                    'state': (pair_data['state_name'] == exp['treat_state']).astype(float),
                    'post': (pair_data['YEAR'] >= exp['treatment_year']).astype(float),
                    'did': ((pair_data['YEAR'] >= exp['treatment_year']) & 
                           (pair_data['state_name'] == exp['treat_state'])).astype(float)
                })
                X = sm.add_constant(X)
                y = pair_data[measure]
                model = sm.OLS(y, X)
                results[name][f'{measure}_regression'] = model.fit(cov_type='HC1')
    
    return results

def analyze_ca_nv_robust(df):
    """Detailed analysis of CA-NV with robustness checks"""
    
    # Different specifications
    specs = {
        'base': {
            'years': (2007, 2019),
            'treatment_year': 2014,
        },
        'stable_control': {
            'years': (2007, 2019),
            'treatment_year': 2014,
            'exclude_years': [2011, 2013]  # Drop volatile NV years
        },
        'narrow_window': {
            'years': (2012, 2016),  # Just around treatment
            'treatment_year': 2014,
        },
        'early_stop': {
            'years': (2007, 2017),  # Before later wage hikes
            'treatment_year': 2014,
        }
    }
    
    results = {}
    for name, spec in specs.items():
        print(f"\nAnalyzing {name}...")
        
        # Filter data for this specification
        spec_data = df[
            (df['state_name'].isin(['California', 'Nevada'])) &
            (df['YEAR'].between(spec['years'][0], spec['years'][1]))
        ].copy()
        
        # Create DiD variables
        spec_data['post'] = (spec_data['YEAR'] >= spec['treatment_year']).astype(int)
        spec_data['treat'] = (spec_data['state_name'] == 'California').astype(int)
        spec_data['did'] = spec_data['post'] * spec_data['treat']
        
        # Create state-industry identifier
        spec_data['state_industry'] = spec_data['state_name'] + "_" + spec_data['industry_name']
        
        # Print sample info
        print(f"Years: {spec['years'][0]}-{spec['years'][1]}")
        print(f"Treatment year: {spec['treatment_year']}")
        print(f"Number of observations: {len(spec_data)}")
        print(f"Number of industries: {spec_data['industry_name'].nunique()}")
        
        # Run regression
        panel_data = spec_data.set_index(['state_industry', 'YEAR'])
        model = PanelOLS(
            dependent=panel_data['wage_inequality'],
            exog=panel_data[['did', 'post']],
            entity_effects=True,
            drop_absorbed=True
        )
        
        results[name] = {
            'regression': model.fit(cov_type='clustered', cluster_entity=True),
            'data': spec_data
        }
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=spec_data,
            x='YEAR',
            y='wage_inequality',
            hue='state_name',
            style='state_name',
            markers=True,
            err_style='band'
        )
        plt.axvline(x=spec['treatment_year'], color='r', linestyle='--', alpha=0.5,
                   label='Treatment Year')
        plt.title(f'CA-NV Wage Inequality: {name}')
        plt.savefig(f'ca_nv_{name}_trends.png')
        plt.close()
        
        # Print industry-specific effects
        print("\nLargest industry effects:")
        industry_effects = spec_data.groupby('industry_name').agg({
            'wage_inequality': lambda x: x[spec_data['post']==1].mean() - x[spec_data['post']==0].mean()
        }).sort_values('wage_inequality', ascending=False)
        print(industry_effects.head())
    
    return results

def main():
    # Load data
    print("Loading and preparing data...")
    df = pd.read_csv('../../data/merged_final.csv')
    
    # Run natural experiments analysis
    print("\nAnalyzing Natural Experiments...")
    results = analyze_natural_experiments(df)
    
    # Print results
    for exp, result in results.items():
        print(f"\n{exp} Results:")
        
        # Print regression results for both measures
        for measure in ['wage_inequality', 'wage_gini']:
            measure_name = "Mean-Median Gap" if measure == 'wage_inequality' else "Gini Coefficient"
            print(f"\n{measure_name} Results:")
            if result[f'{measure}_regression'] is not None:
                print(result[f'{measure}_regression'].summary().tables[1])
            else:
                print("Regression failed to converge")
        
        print("\nSummary Statistics:")
        print(result['summary'])

if __name__ == "__main__":
    main() 