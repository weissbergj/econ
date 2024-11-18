import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

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

def run_did_regression(df, start_year, end_year, treatment_year, outcome_var):
    """Run DiD regression for specified window and outcome."""
    window_data = df[
        (df['YEAR'] >= start_year) & 
        (df['YEAR'] <= end_year) &
        (df['state_name'].isin(['California', 'Nevada']))
    ].copy()
    
    window_data['state'] = (window_data['state_name'] == 'California').astype(float)
    window_data['post'] = (window_data['YEAR'] >= treatment_year).astype(float)
    window_data['did'] = window_data['state'] * window_data['post']
    
    X = sm.add_constant(window_data[['state', 'post', 'did']])
    y = window_data[outcome_var]
    
    model = sm.OLS(y, X)
    results = model.fit(cov_type='HC1')
    
    return results

def calculate_treatment_effects(df, outcome_var):
    """Calculate actual changes in treatment and control states"""
    pre_data = df[df['YEAR'] < 2014]
    post_data = df[df['YEAR'] >= 2014]
    
    changes = {}
    for state in ['California', 'Nevada']:
        state_pre = pre_data[pre_data['state_name'] == state][outcome_var].mean()
        state_post = post_data[post_data['state_name'] == state][outcome_var].mean()
        changes[state] = state_post - state_pre
    
    return {
        'Treatment change': changes['California'],
        'Control change': changes['Nevada'],
        'DiD': changes['California'] - changes['Nevada']
    }

def run_detailed_window_analysis(df, outcome_var):
    """Run multiple window specifications"""
    windows = [
        (2012, 2016, "Focused"),    # Clean identification
        (2011, 2017, "Medium"),     # Broader window
        (2007, 2015, "Early"),      # Pre-period focus
        (2014, 2019, "Late"),       # Post-period focus
        (2013, 2015, "Immediate"),  # Short-term effect
        (2016, 2019, "Long-term")   # Long-term effect
    ]
    
    results = {}
    for start, end, label in windows:
        model_results = run_did_regression(df, start, end, 2014, outcome_var)
        results[label] = {
            'coef': model_results.params['did'],
            'pvalue': model_results.pvalues['did'],
            'window': f"{start}-{end}"
        }
    return results

def analyze_pre_trends(df, outcome_var):
    """Detailed analysis of pre-treatment trends"""
    windows = [
        (2007, 2015, "Full pre-period"),
        (2012, 2016, "Focused window"),
        (2011, 2017, "Medium window")
    ]
    
    trends = {}
    for start, end, label in windows:
        pre_data = df[
            (df['YEAR'] >= start) & 
            (df['YEAR'] < 2014)
        ].copy()
        
        # Calculate annual changes
        for state in ['California', 'Nevada']:
            state_data = pre_data[pre_data['state_name'] == state]
            annual_change = np.polyfit(state_data['YEAR'], 
                                     state_data[outcome_var], 1)[0]
            trends[f"{label}_{state}"] = annual_change
            
    return trends

def run_comprehensive_placebos(df, outcome_var):
    """Run placebo tests for multiple years"""
    placebo_years = [2012, 2013, 2015, 2016]
    results = {}
    
    for year in placebo_years:
        model = run_did_regression(df, 2007, 2019, year, outcome_var)
        results[year] = {
            'coef': model.params['did'],
            'pvalue': model.pvalues['did']
        }
        
    # Check how many placebos are significant
    sig_count = sum(1 for r in results.values() if r['pvalue'] < 0.05)
    
    return results, sig_count

def compare_inequality_measures(df):
    """Compare Gini and Mean-Median gap measures for CA-NV"""
    
    # Calculate both measures
    analysis_df = df.copy()
    analysis_df['wage_inequality'] = analysis_df['mean_wage'] - analysis_df['median_wage']
    analysis_df['wage_gini'] = analysis_df.apply(calculate_wage_gini, axis=1)
    
    # Filter for CA-NV
    pair_data = analysis_df[
        (analysis_df['state_name'].isin(['California', 'Nevada'])) &
        (analysis_df['YEAR'] < 2020)
    ].copy()
    
    # 1. Basic Correlation Analysis
    print("\nCorrelation between measures:")
    corr = pair_data[['wage_inequality', 'wage_gini']].corr()
    print(corr)
    
    # 2. Year-by-Year Analysis
    print("\nYear-by-Year Comparison:")
    yearly = pair_data.groupby(['state_name', 'YEAR']).agg({
        'wage_inequality': 'mean',
        'wage_gini': 'mean',
        'min_wage_nominal': 'first'
    }).reset_index()
    
    for state in ['California', 'Nevada']:
        state_data = yearly[yearly['state_name'] == state]
        print(f"\n{state}:")
        print("Year  Min Wage    Gap      Gini")
        print("-" * 35)
        for _, row in state_data.iterrows():
            print(f"{row['YEAR']}  ${row['min_wage_nominal']:6.2f}  ${row['wage_inequality']:7.0f}  {row['wage_gini']:.3f}")
    
    # 3. DiD Analysis for both measures
    print("\nDiD Analysis:")
    for measure in ['wage_inequality', 'wage_gini']:
        X = pd.DataFrame({
            'state': (pair_data['state_name'] == 'California').astype(float),
            'post': (pair_data['YEAR'] >= 2014).astype(float),
            'did': ((pair_data['YEAR'] >= 2014) & 
                   (pair_data['state_name'] == 'California')).astype(float)
        })
        X = sm.add_constant(X)
        y = pair_data[measure]
        model = sm.OLS(y, X)
        results = model.fit(cov_type='HC1')
        
        measure_name = "Mean-Median Gap" if measure == 'wage_inequality' else "Gini"
        print(f"\n{measure_name} Results:")
        print("\nCoefficients:")
        print("-------------")
        for var, coef, se, p in zip(results.model.exog_names, 
                                   results.params, 
                                   results.HC1_se, 
                                   results.pvalues):
            print(f"{var:>8}: {coef:10.4f} (SE: {se:8.4f}) p={p:.4f}")
    
    # Add robustness checks
    for measure in ['wage_inequality', 'wage_gini']:
        print(f"\nAnalyzing {measure}")
        
        # Calculate treatment effects
        effects = calculate_treatment_effects(pair_data, measure)
        print("\nTreatment Effects:")
        for key, value in effects.items():
            print(f"{key}: ${value:,.2f}")
        
        # Window analysis
        window_results = run_detailed_window_analysis(pair_data, measure)
        print("\nWindow Analysis:")
        for window, results in window_results.items():
            print(f"{window} ({results['window']}): ${results['coef']:,.2f} (p={results['pvalue']:.3f})")
        
        # Pre-trends
        trends = analyze_pre_trends(pair_data, measure)
        print("\nPre-trends (Annual Changes):")
        for key, value in trends.items():
            print(f"{key}: ${value:,.2f}")
        
        # Placebo tests
        placebo_results, sig_placebos = run_comprehensive_placebos(pair_data, measure)
        print(f"\nPlacebo Tests ({sig_placebos}/4 significant):")
        for year, results in placebo_results.items():
            print(f"Placebo {year}: ${results['coef']:,.2f} (p={results['pvalue']:.3f})")
    
    # 4. Visualizations
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Mean-Median Gap
    sns.lineplot(data=pair_data, x='YEAR', y='wage_inequality', 
                hue='state_name', style='state_name', markers=True, ax=ax1)
    ax1.axvline(x=2014, color='r', linestyle='--', alpha=0.5)
    ax1.set_title('Mean-Median Wage Gap')
    ax1.set_ylabel('Dollars')
    
    # Gini
    sns.lineplot(data=pair_data, x='YEAR', y='wage_gini', 
                hue='state_name', style='state_name', markers=True, ax=ax2)
    ax2.axvline(x=2014, color='r', linestyle='--', alpha=0.5)
    ax2.set_title('Gini Coefficient')
    ax2.set_ylabel('Gini')
    
    plt.tight_layout()
    plt.savefig('inequality_measures_comparison.png')
    plt.close()
    
    return pair_data

def main():
    print("Loading and preparing data...")
    df = pd.read_csv('../../data/merged_final.csv')
    
    print("\nComparing inequality measures...")
    results = compare_inequality_measures(df)

if __name__ == "__main__":
    main()