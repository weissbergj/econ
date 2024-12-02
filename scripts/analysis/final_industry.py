import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from scipy import stats
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import YearLocator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ast

# Import all functions from final_analysis.py

from industry_gini import (
    # Section 1: Base Analysis
    load_and_prepare_data,
    clean_data,
    calculate_wage_gini,
    run_all_regressions,
    print_results,
    verify_regression_assumptions,
    
    # Section 2: Industry Dynamics
    run_extended_regressions,
    print_extended_results,
    run_nonlinear_analysis,
    print_nonlinear_results,
    analyze_industry_characteristics,
    
    # Section 3: Geographic and Temporal Variations
    test_regional_differences,
    create_region_mapping,
    analyze_time_trends,
    run_time_effects,
    analyze_post2015_drivers,
    
    # Section 4: Structural Changes and Digital Transformation
    analyze_digital_transformation,
    analyze_digital_transformation_robust,
    plot_digital_transformation_effect,
    run_event_study,
    verify_event_study_data,
    plot_event_study_results,
    print_regression_results,
    calculate_actual_inequality,
    
    # Section 5: ML Analysis and Robustness
    analyze_heterogeneity,
    analyze_mechanisms,
    run_placebo_tests,
    # run_robust_model,
    identify_outliers,
    run_section_five,
    print_section_five_results,
    
    # Utility Functions
    create_summary_statistics,
    format_summary_tables,
    create_table,
    create_main_visualization,
    calculate_controlled_differences,
    plot_controlled_event_study_results,
    run_diff_in_diff,
    run_robustness,
    main
)

def run_section_one(df, industry_dummies):
    """
    Section 1: Base Analysis
    - Two-way fixed effects model for establishment size and wage relationships
    """
    print("\n=== Section 1: Base Analysis ===")
    results = {}
    
    # 1. Base regressions
    print("\n1. Base Analysis Results")
    base_results = run_all_regressions(df, industry_dummies)
    results['base'] = base_results
    
    # 2. Verify assumptions
    print("\n2. Regression Assumptions")
    assumption_results = verify_regression_assumptions(df, base_results['size_inequality'])
    results['assumptions'] = assumption_results
    
    return results

def run_section_two(df):
    """
    Section 2: Industry Dynamics
    - Industry interactions and non-linear relationships
    """
    print("\n=== Section 2: Industry Dynamics ===")
    results = {}
    
    # 1. Extended regressions
    print("\n1. Extended Regression Results")
    extended_results = run_extended_regressions(df)
    results['extended'] = extended_results
    
    # 2. Non-linear analysis
    print("\n2. Non-linear Analysis")
    nonlinear_results = run_nonlinear_analysis(df)
    results['nonlinear'] = nonlinear_results
    
    # 3. Industry characteristics
    print("\n3. Industry Characteristics")
    characteristics_results = analyze_industry_characteristics(df)
    results['characteristics'] = characteristics_results
    
    return results

def run_section_three(df):
    """
    Section 3: Geographic and Temporal Variations
    - Regional analysis
    - Time trends
    - Post-2015 analysis
    """
    print("\n=== Section 3: Geographic and Temporal Variations ===")
    results = {}
    
    # 1. Regional Analysis
    print("\n1. Regional Differences")
    # Create region mapping
    regions = {
        'Northeast': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'],
        'Midwest': ['OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
        'South': ['DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA', 'OK', 'TX'],
        'West': ['MT', 'ID', 'WY', 'CO', 'NM', 'AZ', 'UT', 'NV', 'WA', 'OR', 'CA', 'AK', 'HI']
    }
    df['region'] = df['state'].map({state: region for region, states in regions.items() for state in states})
    
    # Run regional analysis for each region separately
    regional_results = {}
    for region in regions.keys():
        region_data = df[df['region'] == region]
        if len(region_data) > 0:
            regional_results[region] = run_all_regressions(region_data)
    results['regional'] = regional_results
    
    # 2. Time Trends
    print("\n2. Time Trends Analysis")
    time_results = analyze_time_trends(df)
    results['time'] = time_results
    
    # 3. Post-2015 Analysis
    print("\n3. Post-2015 Analysis")
    post2015_results = analyze_post2015_drivers(df)
    results['post2015'] = post2015_results
    
    return results

def run_section_four(df):
    """
    Section 4: Structural Changes and Digital Transformation
    - Digital transformation analysis
    - Event study (for pre-trends and post-2015 effects)
    - Triple difference design
    """
    print("\n=== Section 4: Structural Changes and Digital Transformation ===")
    results = {}
    
    # 1. Event Study
    print("\n1. Event Study Analysis")
    event_results = run_event_study(df)
    results['event'] = event_results
    
    # 2. Digital Transformation (existing triple-diff analysis)
    print("\n2. Digital Transformation Analysis")
    dig_results, trend_results = analyze_digital_transformation(df)
    results['digital'] = dig_results
    results['trends'] = trend_results
    
    # # 3. Robust Digital Analysis
    # print("\n3. Robust Digital Analysis")
    # robust_results = analyze_digital_transformation_robust(df)
    # results['robust'] = robust_results
    
    # # 4. Calculate Actual Effects
    # print("\n4. Actual Inequality Effects")
    # actual_results = calculate_actual_inequality(df)
    # results['actual'] = actual_results
    
    return results

def print_section_one(results):
    """Print results from section one analysis"""
    print("\n=== Detailed Section 1 Results ===")
    print_results(results['base'])
    print("\nRegression Assumptions:")
    print(results['assumptions'])
    
def print_section_two(results):
    """Print results from section two analysis"""
    print("\n=== Detailed Section 2 Results ===")
    print_extended_results(results['extended'])
    print_nonlinear_results(results['nonlinear'])
    
def print_section_three(results):
    """Print results from section three analysis"""
    print("\n=== Detailed Section 3 Results ===")
    
    # Print regional results
    if 'regional' in results:
        print("\nRegional Analysis Results:")
        for region, result in results['regional'].items():
            print(f"\n{region} Results:")
            if 'size_inequality' in result:
                print(result['size_inequality'].summary.tables[1])
    
    # Print time trends
    if 'time' in results:
        print("\nTime Trends Results:")
        if hasattr(results['time'], 'summary'):
            print(results['time'].summary.tables[1])
        else:
            print(results['time'])  # Print DataFrame directly
    
    # Print post-2015 analysis
    if 'post2015' in results:
        print("\nPost-2015 Analysis Results:")
        if hasattr(results['post2015'], 'summary'):
            print(results['post2015'].summary.tables[1])
        else:
            print(results['post2015'])  # Print DataFrame directly
    
def print_section_four(results):
    """Print results from section four analysis"""
    print("\n=== Detailed Section 4 Results ===")
    print("\nDigital Transformation Results:")
    print(results['digital'].summary.tables[1])
    print("\nParallel Trends Results:")
    print(results['trends'].summary.tables[1])

def main():
    """Main execution function"""
    # Load and prepare data
    state_industry, industry_dummies = load_and_prepare_data()
    
    # Run all sections
    section_one_results = run_section_one(state_industry, industry_dummies)
    print_section_one(section_one_results)
    
    section_two_results = run_section_two(state_industry)
    print_section_two(section_two_results)
    
    section_three_results = run_section_three(state_industry)
    print_section_three(section_three_results)
    
    section_four_results = run_section_four(state_industry)
    print_section_four(section_four_results)
    
    section_five_results = run_section_five(state_industry)
    print_section_five_results(section_five_results)
    
    # Store all results in a dictionary
    all_results = {
        'section_one': section_one_results,
        'section_two': section_two_results,
        'section_three': section_three_results,
        'section_four': section_four_results,
        'section_five': section_five_results
    }
    
    return all_results

if __name__ == "__main__":
    results = main()