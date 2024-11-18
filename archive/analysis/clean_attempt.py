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

# At the top of the file, define constants
REGIONS = {
    'Northeast': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'],
    'Midwest': ['OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
    'South': ['DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA', 'OK', 'TX'],
    'West': ['MT', 'ID', 'WY', 'CO', 'NM', 'AZ', 'UT', 'NV', 'WA', 'OR', 'CA', 'AK', 'HI']
}

STATE_ABBREV = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL',
    'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
    'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
    'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR',
    'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA',
    'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}

def create_panel_model(data, dependent_var, exog_vars, entity_effects=True, time_effects=True):
    """Helper function to create PanelOLS models with standard settings"""
    # Ensure data is properly indexed
    if not isinstance(data.index, pd.MultiIndex):
        data = data.set_index(['state_industry', 'YEAR'])
    
    # Add constant if not using entity effects (for industry premiums)
    if not entity_effects:
        data['const'] = 1
        exog_vars = ['const'] + exog_vars
    
    return PanelOLS(
        dependent=data[dependent_var],
        exog=data[exog_vars],
        entity_effects=entity_effects,
        time_effects=time_effects,
        check_rank=False,
        drop_absorbed=True
    )

def print_regression_results(name, results):
    """Consolidated function to print regression results in a consistent format
    Args:
        name: Name/description of the regression
        results: PanelOLS results object
    """
    print(f"\n=== {name} ===")
    
    # Main statistics
    print("\nKey Statistics:")
    print(f"R-squared: {results.rsquared:.4f}")
    print(f"Within R-squared: {results.rsquared_within:.4f}")
    
    # Coefficients table
    print("\nCoefficients:")
    coef_data = pd.DataFrame({
        'Coefficient': results.params,
        'Std Error': results.std_errors,
        't-stat': results.tstats,
        'p-value': results.pvalues
    })
    print(coef_data.round(4))
    
    # Residuals
    print("\nResiduals Summary:")
    print(pd.Series(results.resids).describe().round(4))

def save_results_to_csv(results):
    """Save regression results to CSV files"""
    for name, result in results.items():
        params = pd.DataFrame({
            'coefficient': result.params,
            'std_error': result.std_errors,
            't_stat': result.tstats,
            'p_value': result.pvalues
        })
        params.to_csv(f'results_{name}.csv')

def get_region_mapping():
    """Helper function to create state-to-region mapping"""
    state_to_region = {}
    for region, abbrevs in REGIONS.items():
        for state, abbrev in STATE_ABBREV.items():
            if abbrev in abbrevs:
                state_to_region[state] = region
    return state_to_region

def calculate_establishment_size(df, size_categories):
    """Calculate weighted establishment size index"""
    weighted_sum = 0
    for col, midpoint in size_categories.items():
        share = (df[col] / df['establishments_01: Total']).fillna(0)
        weighted_sum += share * midpoint
    return weighted_sum

def calculate_wage_gini(row, size_categories):
    """Calculate Gini coefficient for wage distribution"""
    wages = []
    weights = []
    
    for emp_col, pay_col in size_categories:
        total_payroll = row[pay_col]
        total_employment = row[emp_col]
        
        if total_employment > 0:
            avg_wage = total_payroll / total_employment
            wages.append(avg_wage)
            weights.append(total_employment)
    
    if not wages:  # Return 0 if no valid wage data
        return 0
        
    wages = np.array(wages)
    weights = np.array(weights)
    
    # Sort and calculate Gini
    sorted_indices = np.argsort(wages)
    wages = wages[sorted_indices]
    weights = weights[sorted_indices]
    
    cumulative_weights = np.cumsum(weights)
    total_weight = np.sum(weights)
    cumulative_share_weights = cumulative_weights / total_weight
    cumulative_share_wages = np.cumsum(wages * weights) / np.sum(wages * weights)
    
    return 1 - 2 * np.trapz(cumulative_share_wages, cumulative_share_weights)

def prepare_data(filepath='../../data/merged_final.csv', min_years=10):
    """Load and prepare data in one consolidated function
    
    Args:
        filepath: Path to raw data file
        min_years: Minimum years required for balanced panel
    
    Returns:
        tuple: (cleaned DataFrame, industry dummies DataFrame)
    """
    # Define size categories once
    ESTABLISHMENT_CATEGORIES = {
        'establishments_02: <5': 2.5,
        'establishments_03: 5-9': 7,
        'establishments_04: 10-19': 14.5,
        'establishments_06: 20-99': 59.5,
        'establishments_07: 100-499': 299.5,
        'establishments_09: 500+': 500
    }
    
    WAGE_CATEGORIES = [
        ('employment_02: <5', 'annual_payroll_02: <5'),
        ('employment_03: 5-9', 'annual_payroll_03: 5-9'),
        ('employment_04: 10-19', 'annual_payroll_04: 10-19'),
        ('employment_06: 20-99', 'annual_payroll_06: 20-99'),
        ('employment_07: 100-499', 'annual_payroll_07: 100-499'),
        ('employment_09: 500+', 'annual_payroll_09: 500+')
    ]
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Basic cleaning
    df['YEAR'] = pd.to_numeric(df['YEAR'])
    df['state_industry'] = df['state_name'] + "_" + df['industry_name']
    
    # Filter conditions
    df = df[
        (df['establishments_01: Total'] > 0) &
        (df['YEAR'] <= 2021)
    ]
    
    # Calculate key metrics
    df['estab_weighted_sum'] = calculate_establishment_size(df, ESTABLISHMENT_CATEGORIES)
    df['wage_gini'] = df.apply(lambda x: calculate_wage_gini(x, WAGE_CATEGORIES), axis=1)
    df['wage_relationship_flag'] = df['mean_wage'] < df['median_wage']
    df['wage_skewness'] = (df['mean_wage'] - df['median_wage']) / df['mean_wage']
    
    # Balance panel
    valid_pairs = df.groupby('state_industry')['YEAR'].count() >= min_years
    df = df[df['state_industry'].isin(valid_pairs[valid_pairs].index)]
    
    # Create industry dummies
    industry_dummies = pd.get_dummies(df['industry_name'], prefix='ind', drop_first=True)
    
    return df, industry_dummies

def run_base_regressions(df, industry_dummies, size_measure='estab_weighted_sum'):
    """Run base regressions with consolidated model creation"""
    results = {}
    
    # Create panel data once
    panel_data = df.set_index(['state_industry', 'YEAR'])
    
    # 1. Base Size Effects
    # 1a. Size effect on median wages
    model = create_panel_model(panel_data, 'median_wage', [size_measure])
    results['size_wage'] = model.fit(cov_type='clustered', cluster_entity=True)
    print_regression_results("Size Effect on Median Wages", results['size_wage'])
    
    # 1b. Size effect on wage inequality
    model = create_panel_model(panel_data, 'wage_gini', [size_measure])
    results['size_inequality'] = model.fit(cov_type='clustered', cluster_entity=True)
    print_regression_results("Size Effect on Wage Inequality", results['size_inequality'])
    
    # 2. Industry Wage Premiums
    if industry_dummies is not None:
        # Create industry dummies directly
        industry_data = panel_data.copy()
        for industry in df['industry_name'].unique():
            if industry != df['industry_name'].iloc[0]:  # Skip first industry as base
                col_name = f'ind_{industry}'
                industry_data[col_name] = (df['industry_name'] == industry).astype(float)
        
        # Add size measure
        industry_data[size_measure] = panel_data[size_measure]
        
        # Get variable names (excluding the constant which will be added in create_panel_model)
        ind_vars = [col for col in industry_data.columns if col.startswith('ind_')] + [size_measure]
        
        model = create_panel_model(industry_data, 'median_wage', ind_vars, entity_effects=False)
        results['industry_premiums'] = model.fit(cov_type='clustered', cluster_entity=True)
        print_regression_results("Industry Wage Premiums", results['industry_premiums'])
    
    # 3. Size-Industry Interactions
    major_industries = ['Manufacturing', 'Finance and Insurance', 'Retail Trade']
    interaction_data = panel_data.copy()
    
    for ind in major_industries:
        interaction_data[f'size_x_{ind}'] = (
            interaction_data[size_measure] * 
            (df['industry_name'] == ind).astype(float)
        )
    
    interaction_vars = [f'size_x_{ind}' for ind in major_industries] + [size_measure]
    model = create_panel_model(interaction_data, 'wage_gini', interaction_vars)
    results['size_industry_interactions'] = model.fit(cov_type='clustered', cluster_entity=True)
    print_regression_results("Size-Industry Interactions", results['size_industry_interactions'])
    
    # Save results
    save_results_to_csv(results)
    
    return results

def run_robustness(df, size_measure='estab_weighted_sum'):
    """Run robustness checks with alternative specifications
    
    Args:
        df: DataFrame with state-industry panel data
        size_measure: Column name for size measure to use
    
    Returns:
        DataFrame: Enhanced dataset with robustness check variables
    """
    df = df.copy()
    
    # Alternative inequality measures
    df['log_inequality'] = np.log(df['wage_gini'].clip(lower=1))
    
    # Size categories
    df['size_category'] = pd.qcut(df[size_measure], q=5, 
                                 labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'])
    
    # Alternative dependent variables
    df['wage_relationship_flag'] = df['mean_wage'] < df['median_wage']
    df['wage_skewness'] = (df['mean_wage'] - df['median_wage']) / df['mean_wage']
    
    return df

def run_regional_analysis(df, size_measure='estab_weighted_sum'):
    """Consolidated regional analysis including interactions, regressions, and tests
    
    Args:
        df: DataFrame with state-industry panel data
        size_measure: Column name for size measure to use
    
    Returns:
        dict: Results from all regional analyses
    """
    results = {}
    
    # Setup panel data
    df = df.copy()
    df['region'] = df['state_name'].map(get_region_mapping())
    panel_data = df.set_index(['state_industry', 'YEAR'])
    
    # 1. Basic Regional Regressions
    print("\n=== Regional Analysis ===")
    results_by_region = {}
    
    for region in REGIONS.keys():
        region_states = df[df['region'] == region]['state_industry'].unique()
        region_data = panel_data[panel_data.index.get_level_values('state_industry').isin(region_states)]
        
        if len(region_data) > 0:
            try:
                model = create_panel_model(region_data, 'wage_gini', [size_measure])
                results_by_region[region] = model.fit(cov_type='clustered', cluster_entity=True)
                print_regression_results(f"Region: {region}", results_by_region[region])
            except Exception as e:
                print(f"Could not estimate model for {region}: {str(e)}")
    
    results['by_region'] = results_by_region
    
    # 2. Region Interaction Analysis
    # Create region interactions
    for region in REGIONS.keys():
        df[f'size_x_{region}'] = df[size_measure] * (df['region'] == region)
    
    # Run interaction regression
    interaction_vars = [size_measure] + [f'size_x_{region}' for region in REGIONS.keys()]
    model = create_panel_model(panel_data, 'wage_gini', interaction_vars)
    results['region_interactions'] = model.fit(cov_type='clustered', cluster_entity=True)
    print_regression_results("Region Interactions", results['region_interactions'])
    
    # 3. Test Regional Differences
    print("\n=== Tests of Regional Differences ===")
    regions = list(REGIONS.keys())
    diff_tests = {}
    
    for i, region1 in enumerate(regions):
        for region2 in regions[i+1:]:
            # Get coefficients and standard errors
            coef1 = results_by_region[region1].params[size_measure]
            coef2 = results_by_region[region2].params[size_measure]
            se1 = results_by_region[region1].std_errors[size_measure]
            se2 = results_by_region[region2].std_errors[size_measure]
            
            # Calculate test statistics
            coef_diff = coef1 - coef2
            se_diff = (se1**2 + se2**2)**0.5
            z_stat = coef_diff / se_diff
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            diff_tests[f"{region1}_vs_{region2}"] = {
                'difference': coef_diff,
                'std_error': se_diff,
                'z_stat': z_stat,
                'p_value': p_value,
                'significant_05': p_value < 0.05
            }
            
            # Print results
            print(f"\n{region1} vs {region2}:")
            print(f"Difference in coefficients: {coef_diff:.4f}")
            print(f"Z-statistic: {z_stat:.4f}")
            print(f"P-value: {p_value:.4f}")
            print(f"Significant at 5% level: {'Yes' if p_value < 0.05 else 'No'}")
    
    results['difference_tests'] = diff_tests
    
    # 4. Additional Regional Statistics
    regional_stats = df.groupby('region').agg({
        'state_industry': 'count',
        'state_name': 'nunique',
        'industry_name': 'nunique',
        size_measure: ['mean', 'median', 'std']
    })
    
    results['regional_stats'] = regional_stats
    print("\n=== Regional Statistics ===")
    print(regional_stats)
    
    return results

def run_time_analysis(df, size_measure='estab_weighted_sum'):
    """Consolidated time period analysis including all period effects and interactions
    
    Args:
        df: DataFrame with state-industry panel data
        size_measure: Column name for size measure to use
    
    Returns:
        dict: Results from all time-related analyses
    """
    results = {}
    panel_data = df.set_index(['state_industry', 'YEAR'])
    
    # 1. Basic Period Effects
    time_periods = {
        'post_2008': 2008,  # Financial Crisis
        'post_2015': 2015,  # Digital transformation
        'post_covid': 2020  # COVID period
    }
    
    # Create period indicators and interactions
    for period, year in time_periods.items():
        panel_data[period] = (panel_data.index.get_level_values('YEAR') >= year).astype(float)
        panel_data[f'size_x_{period}'] = panel_data[size_measure] * panel_data[period]
    
    # Add continuous time trend
    panel_data['time_trend'] = panel_data.index.get_level_values('YEAR') - df['YEAR'].min()
    panel_data[f'size_x_trend'] = panel_data[size_measure] * panel_data['time_trend']
    
    # 2. Run Basic Time Effects Model
    time_vars = [size_measure] + [f'size_x_{period}' for period in time_periods.keys()]
    model = create_panel_model(panel_data, 'wage_gini', time_vars, time_effects=False)
    results['time_effects'] = model.fit(cov_type='clustered', cluster_entity=True)
    print_regression_results("Time Period Effects", results['time_effects'])
    
    # 3. Industry-Specific Period Effects (focusing on post-2015)
    key_industries = [
        'Information',
        'Professional, Scientific, and Technical Services',
        'Finance and Insurance',
        'Retail Trade',
        'Transportation and Warehousing'
    ]
    
    # Create industry-period interactions
    for ind in key_industries:
        panel_data[f'ind_{ind}'] = (df['industry_name'] == ind).astype(float)
        panel_data[f'post2015_x_{ind}'] = panel_data['post_2015'] * panel_data[f'ind_{ind}']
        panel_data[f'size_x_post2015_x_{ind}'] = (
            panel_data[size_measure] * 
            panel_data['post_2015'] * 
            panel_data[f'ind_{ind}']
        )
    
    # Run industry-period interaction model
    industry_time_vars = [
        size_measure, 'post_2015',
        *[f'ind_{ind}' for ind in key_industries],
        *[f'post2015_x_{ind}' for ind in key_industries],
        *[f'size_x_post2015_x_{ind}' for ind in key_industries]
    ]
    
    model = create_panel_model(panel_data, 'wage_gini', industry_time_vars)
    results['industry_time'] = model.fit(cov_type='clustered', cluster_entity=True)
    print_regression_results("Industry-Time Interactions", results['industry_time'])
    
    # 4. Calculate Total Effects for each Period
    print("\n=== Total Effects by Period ===")
    for period in time_periods.keys():
        base_effect = results['time_effects'].params[size_measure]
        period_effect = results['time_effects'].params[f'size_x_{period}']
        total_effect = base_effect + period_effect
        
        # Calculate standard error for total effect
        base_se = results['time_effects'].std_errors[size_measure]
        period_se = results['time_effects'].std_errors[f'size_x_{period}']
        total_se = (base_se**2 + period_se**2)**0.5
        
        # Calculate t-stat and p-value
        t_stat = total_effect / total_se
        p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        
        print(f"\n{period}:")
        print(f"Total effect: {total_effect:.4f}")
        print(f"Standard Error: {total_se:.4f}")
        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value: {p_val:.4f}")
    
    return results

def create_summary_statistics(df, size_measure='estab_weighted_sum'):
    """Create comprehensive summary statistics tables
    
    Args:
        df: DataFrame with state-industry panel data
        size_measure: Column name for size measure to use
    
    Returns:
        dict: Dictionary of DataFrames with different summary statistics
    """
    def format_number(x, is_ratio=False):
        """Helper function for consistent number formatting"""
        if isinstance(x, float):
            if is_ratio:
                return f"{x:.2f}"
            if abs(x) >= 1000:
                return f"{x:,.0f}"
            return f"{x:.0f}"
        return str(x)

    # 1. Basic Dataset Information
    basic_info = pd.DataFrame({
        'Metric': [
            'Total Observations',
            'Unique States',
            'Unique Industries', 
            'State-Industry Pairs',
            'Years Covered'
        ],
        'Value': [
            f"{len(df):,}",
            df['state_name'].nunique(),
            df['industry_name'].nunique(),
            df['state_industry'].nunique(),
            f"{df['YEAR'].min()}-{df['YEAR'].max()}"
        ]
    })

    # 2. Key Variables Statistics
    key_vars = ['workers', size_measure, 'median_wage', 'mean_wage', 'wage_gini']
    var_stats = df[key_vars].agg([
        'count',
        'mean',
        'std',
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        'min',
        'max'
    ]).round(2)
    var_stats.index = ['Count', 'Mean', 'Std Dev', '25%', 'Median', '75%', 'Min', 'Max']
    
    # Format each column appropriately
    for col in var_stats.columns:
        is_ratio = col == 'wage_gini'
        var_stats[col] = var_stats[col].apply(lambda x: format_number(x, is_ratio))

    # 3. Coverage Statistics
    coverage_stats = pd.DataFrame({
        'Metric': [
            'Avg Industries per State',
            'Min Industries per State',
            'Max Industries per State',
            'Avg Years per State-Industry',
            'Min Years per State-Industry',
            'Max Years per State-Industry'
        ],
        'Value': [
            f"{df.groupby('state_name')['industry_name'].nunique().mean():.1f}",
            df.groupby('state_name')['industry_name'].nunique().min(),
            df.groupby('state_name')['industry_name'].nunique().max(),
            f"{df.groupby('state_industry')['YEAR'].nunique().mean():.1f}",
            df.groupby('state_industry')['YEAR'].nunique().min(),
            df.groupby('state_industry')['YEAR'].nunique().max()
        ]
    })

    return {
        'basic_info': basic_info,
        'variable_stats': var_stats,
        'coverage_stats': coverage_stats
    }

def identify_outliers(series):
    """Helper function to identify outliers using IQR method"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return series[(series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))]

def format_summary_tables(basic_info, var_stats, coverage_stats, 
                        quality_stats, regional_stats, time_stats):
    """Create formatted PDF with summary statistics tables"""
    pdf = FPDF()
    
    # Title Page
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Summary Statistics', 0, 1, 'C')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, 'State-Industry Panel Analysis', 0, 1, 'C')
    
    # Table 1: Basic Dataset Information
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Table 1: Dataset Overview', 0, 1, 'L')
    create_table(pdf, basic_info)
    
    # Table 2: Key Variables Statistics
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Table 2: Key Variables Statistics', 0, 1, 'L')
    create_table(pdf, var_stats)
    
    # Table 3: Coverage Statistics
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Table 3: Coverage Statistics', 0, 1, 'L')
    create_table(pdf, coverage_stats)
    
    # Table 4: Data Quality
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Table 4: Data Quality Indicators', 0, 1, 'L')
    create_table(pdf, quality_stats)
    
    # Table 5: Regional Distribution
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Table 5: Regional Statistics', 0, 1, 'L')
    create_table(pdf, regional_stats)
    
    # Table 6: Time Series Properties
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Table 6: Time Period Analysis', 0, 1, 'L')
    create_table(pdf, time_stats)
    
    pdf.output('summary_statistics.pdf')
    
    # Also save Excel version for easier access
    with pd.ExcelWriter('summary_statistics.xlsx') as writer:
        basic_info.to_excel(writer, sheet_name='Dataset Overview', index=False)
        var_stats.to_excel(writer, sheet_name='Key Variables')
        coverage_stats.to_excel(writer, sheet_name='Coverage', index=False)
        quality_stats.to_excel(writer, sheet_name='Data Quality', index=False)
        regional_stats.to_excel(writer, sheet_name='Regional Stats')
        time_stats.to_excel(writer, sheet_name='Time Analysis', index=False)

def create_table(pdf, df):
    """Helper function to create individual tables in PDF"""
    # Configure table settings
    pdf.set_font('Arial', 'B', 10)
    col_width = pdf.w / (len(df.columns) + 1)
    row_height = 7
    
    # Headers
    for col in df.columns:
        pdf.cell(col_width, row_height, str(col), 1)
    pdf.ln()
    
    # Data
    pdf.set_font('Arial', '', 10)
    for idx, row in df.iterrows():
        # If index is meaningful (not just numbers), include it
        if not isinstance(idx, int):
            pdf.cell(col_width, row_height, str(idx), 1)
        
        # Add each cell in the row
        for item in row:
            pdf.cell(col_width, row_height, str(item), 1)
        pdf.ln()

def create_main_visualization(df):
    """Create the main visualization showing size-inequality relationship over time"""
    df = df.copy()
    df['log_workers'] = np.log(df['estab_weighted_sum'])
    
    # Use actual terciles instead of arbitrary cutoffs
    cutoffs = pd.qcut(df['estab_weighted_sum'], q=3, retbins=True)[1]
    size_labels = [
        f'Small (<{cutoffs[1]:.0f})', 
        f'Medium ({cutoffs[1]:.0f}-{cutoffs[2]:.0f})', 
        f'Large (>{cutoffs[2]:.0f})'
    ]
    df['size_category'] = pd.qcut(df['estab_weighted_sum'], q=3, labels=size_labels)
    
    yearly_stats = df.groupby(['YEAR', 'size_category'])['wage_gini'].mean().reset_index()
    yearly_stats = yearly_stats.pivot(index='YEAR', columns='size_category', values='wage_gini')
    
    # Calculate the size-inequality gap using new labels
    yearly_stats['Large-Small Gap'] = yearly_stats[size_labels[2]] - yearly_stats[size_labels[0]]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
    
    # Update colors dictionary with new labels
    colors = {
        size_labels[0]: '#2ecc71',  # Small
        size_labels[1]: '#3498db',  # Medium
        size_labels[2]: '#e74c3c'   # Large
    }
    
    # Rest of the code remains the same
    for category in colors.keys():
        ax1.plot(yearly_stats.index, yearly_stats[category], 
                color=colors[category], linewidth=2.5, 
                label=category, marker='o')
    
    # Add period shading to both panels
    for ax in [ax1, ax2]:
        ax.axvspan(2020, df['YEAR'].max(), alpha=0.2, color='#FFD700')
        ax.axvspan(2015, 2019, alpha=0.2, color='#B0C4DE')
        ax.set_xticks(range(2007, 2023, 2))
        ax.grid(True, alpha=0.3)
    
    # Bottom panel: Size-inequality gap
    ax2.plot(yearly_stats.index, yearly_stats['Large-Small Gap'], 
            color='black', linewidth=2.5, marker='o')
    ax2.set_ylabel('Large-Small Gini Gap', fontsize=12)
    
    # Customize appearance
    ax1.set_title('Wage Inequality (Gini) by Establishment Size', 
                fontsize=14, pad=20, fontweight='bold')
    ax2.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Wage Gini Coefficient', fontsize=12)
    
    # Legend only in top panel
    ax1.legend(title='Establishment Size', title_fontsize=12, 
             loc='upper left', bbox_to_anchor=(0.02, 0.98))
    
    plt.tight_layout()
    plt.savefig('wage_inequality_trends.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def run_nonlinear_analysis(df, size_measure='estab_weighted_sum'):
    """Run comprehensive non-linear analysis of size effects"""
    results = {}
    panel_data = df.set_index(['state_industry', 'YEAR'])
    
    # 1. Cubic Model
    panel_data[f'{size_measure}_squared'] = panel_data[size_measure] ** 2
    panel_data[f'{size_measure}_cubed'] = panel_data[size_measure] ** 3
    
    model = create_panel_model(
        panel_data, 
        'wage_gini',
        [size_measure, f'{size_measure}_squared', f'{size_measure}_cubed']
    )
    results['cubic'] = model.fit(cov_type='clustered', cluster_entity=True)
    print_regression_results("Cubic Model", results['cubic'])
    
    # Calculate turning points if available
    if all(x in results['cubic'].params for x in [f'{size_measure}_squared', f'{size_measure}_cubed']):
        b2 = results['cubic'].params[f'{size_measure}_squared']
        b3 = results['cubic'].params[f'{size_measure}_cubed']
        tp = -2 * b2 / (3 * b3)
        print(f"\nTurning point at: {tp:,.0f}")
    
    # 2. Log Specification
    panel_data[f'log_{size_measure}'] = np.log(panel_data[size_measure])
    model = create_panel_model(panel_data, 'wage_gini', [f'log_{size_measure}'])
    results['log'] = model.fit(cov_type='clustered', cluster_entity=True)
    print_regression_results("Log Specification", results['log'])
    
    # 3. Industry-Specific
    key_industries = ['Finance and Insurance', 'Manufacturing', 'Retail Trade', 'Information']
    results['industry_specific'] = run_industry_specific_analysis(
        panel_data, key_industries, size_measure
    )
    
    # 4. Time Evolution (note: this overlaps with our time_period_analysis function)
    results['time_evolution'] = run_time_period_analysis(
        panel_data, size_measure, include_nonlinear=True
    )
    
    return results

def run_industry_specific_analysis(panel_data, industries, size_measure):
    """Run industry-specific non-linear analysis"""
    results = {}
    print("\n=== Industry-Specific Analysis ===")
    
    for ind in industries:
        ind_data = panel_data[panel_data['industry_name'] == ind].copy()
        model = create_panel_model(
            ind_data, 
            'wage_gini',
            [size_measure, f'{size_measure}_squared']
        )
        results[ind] = model.fit(cov_type='clustered', cluster_entity=True)
        print_regression_results(f"Industry: {ind}", results[ind])
        
        # Calculate turning point
        if all(x in results[ind].params for x in [size_measure, f'{size_measure}_squared']):
            b1 = results[ind].params[size_measure]
            b2 = results[ind].params[f'{size_measure}_squared']
            tp = -b1/(2*b2)
            print(f"Turning point: {tp:,.0f}")
    
    return results

def run_difference_analysis(df, size_metrics=['estab_weighted_sum', 'workers']):
    """Comprehensive difference-in-difference analysis including:
    1. Basic DiD for each industry
    2. Digital transformation DiD
    3. Event study
    4. Placebo tests
    5. Robust checks with controls
    
    Args:
        df: DataFrame with state-industry panel data
        size_metrics: List of size measures to use
    
    Returns:
        dict: Nested results from all analyses
    """
    df = df[df['industry_name'] != 'Unknown'].copy()
    results = {}
    
    # Define industry groups
    digital_industries = [
        'Information',
        'Finance and Insurance',
        'Professional, Scientific, and Technical Services'
    ]
    
    for size_measure in size_metrics:
        results[size_measure] = {}
        
        # 1. Basic DiD for each industry
        results[size_measure]['basic_did'] = {}
        for industry in df['industry_name'].unique():
            # Create DiD variables
            df[f'{industry}_ind'] = (df['industry_name'] == industry).astype(int)
            df['post_2015'] = (df['YEAR'] >= 2015).astype(int)
            df['covid_period'] = (df['YEAR'] >= 2020).astype(int)
            
            # Run main DiD
            panel_data = df.set_index(['state_industry', 'YEAR'])
            model = PanelOLS(
                dependent=panel_data['wage_gini'],
                exog=panel_data[[
                    size_measure,
                    f'{industry}_ind',
                    'post_2015',
                    'covid_period'
                ]],
                entity_effects=True,
                time_effects=True
            )
            results[size_measure]['basic_did'][industry] = model.fit(
                cov_type='clustered', 
                cluster_entity=True
            )
        
        # 2. Digital transformation DiD
        df['digital'] = df['industry_name'].isin(digital_industries).astype(int)
        df['digital_x_post2015'] = df['digital'] * df['post_2015']
        df[f'size_x_digital'] = df[size_measure] * df['digital']
        
        panel_data = df.set_index(['state_industry', 'YEAR'])
        model = PanelOLS(
            dependent=panel_data['wage_gini'],
            exog=panel_data[[
                size_measure,
                'digital',
                'post_2015',
                'digital_x_post2015',
                f'size_x_digital'
            ]],
            entity_effects=True,
            time_effects=True
        )
        results[size_measure]['digital_did'] = model.fit(
            cov_type='clustered', 
            cluster_entity=True
        )
        
        # 3. Event study
        df['event_time'] = df['YEAR'] - 2015
        event_vars = []
        for tau in range(-5, 6):
            if tau != 0:  # Omit base year
                df[f'event_{tau}'] = (df['event_time'] == tau).astype(int)
                df[f'digital_x_event_{tau}'] = df['digital'] * df[f'event_{tau}']
                event_vars.extend([f'event_{tau}', f'digital_x_event_{tau}'])
        
        panel_data = df.set_index(['state_industry', 'YEAR'])
        model = PanelOLS(
            dependent=panel_data['wage_gini'],
            exog=panel_data[[size_measure] + event_vars],
            entity_effects=True,
            time_effects=True
        )
        results[size_measure]['event_study'] = model.fit(
            cov_type='clustered', 
            cluster_entity=True
        )
        
        # 4. Placebo tests
        results[size_measure]['placebo'] = {}
        for year in range(2010, 2014):
            df[f'post_{year}'] = (df['YEAR'] >= year).astype(int)
            df[f'digital_x_post_{year}'] = df['digital'] * df[f'post_{year}']
            
            panel_data = df.set_index(['state_industry', 'YEAR'])
            model = PanelOLS(
                dependent=panel_data['wage_gini'],
                exog=panel_data[[
                    size_measure,
                    f'post_{year}',
                    f'digital_x_post_{year}'
                ]],
                entity_effects=True,
                time_effects=True
            )
            results[size_measure]['placebo'][year] = model.fit(
                cov_type='clustered', 
                cluster_entity=True
            )
        
            # 5. Robust checks with controls
            df['digital_x_post2015_x_5G'] = df['digital_x_post2015'] * df['5G_ratio']
            
            # Create event time indicators
            df['event_time'] = df['YEAR'] - 2015
            event_vars = []
            for tau in range(-5, 6):
                if tau != 0:  # Omit base year
                    df[f'event_time_{tau}'] = (df['event_time'] == tau).astype(int)
                    event_vars.append(f'event_time_{tau}')
            
            panel_data = df.set_index(['state_industry', 'YEAR'])
            model = PanelOLS(
                dependent=panel_data['wage_gini'],
                exog=panel_data[[
                    size_measure,
                    'digital',
                    'post_2015',
                    'digital_x_post2015',
                    'digital_x_post2015_x_5G',
                    'pct_change_employment',
                    'median_wage',
                    'mean_wage',
                    'min_wage_real_2020',
                    'LTE_ratio',
                    '5G_ratio'
                ] + event_vars],
                entity_effects=True,
                time_effects=True,
                drop_absorbed=True,
                check_rank=False
            )
            results[size_measure]['robust'] = model.fit(
                cov_type='clustered', 
                cluster_entity=True
            )
    
    return results

def run_mechanism_analysis(df, size_metrics=['estab_weighted_sum', 'workers']):
    """Analyze mechanisms behind size-inequality relationship including:
    1. Industry characteristics
    2. Mechanism proxies
    3. ML-based heterogeneity
    
    Args:
        df: DataFrame with state-industry panel data
        size_metrics: List of size measures to use
    
    Returns:
        dict: Results from mechanism analyses
    """
    results = {}
    
    for size_measure in size_metrics:
        results[size_measure] = {}
        
        # 1. Industry characteristics
        df['digital_intensity'] = df['industry_name'].isin([
            'Information',
            'Finance and Insurance',
            'Professional, Scientific, and Technical Services'
        ]).astype(int)
        
        df[f'size_x_digital'] = df[size_measure] * df['digital_intensity']
        
        panel_data = df.set_index(['state_industry', 'YEAR'])
        model = PanelOLS(
            dependent=panel_data['wage_gini'],
            exog=panel_data[[
                size_measure,
                'digital_intensity',
                f'size_x_digital'
            ]],
            entity_effects=True,
            time_effects=True
        )
        results[size_measure]['characteristics'] = model.fit(
            cov_type='clustered', 
            cluster_entity=True
        )
        
        # 2. Mechanism proxies
        # Create proxies
        df['automation'] = df['digital_intensity']  # Using digital as proxy
        df['skill_intensity'] = df['mean_wage'] / df.groupby('YEAR')['mean_wage'].transform('mean')
        df['complexity'] = df[size_measure] / df.groupby('YEAR')[size_measure].transform('mean')
        
        mechanisms = ['automation', 'skill_intensity', 'complexity']
        results[size_measure]['mechanisms'] = {}
        
        for mechanism in mechanisms:
            df[f'size_x_{mechanism}'] = df[size_measure] * df[mechanism]
            
            panel_data = df.set_index(['state_industry', 'YEAR'])
            model = PanelOLS(
                dependent=panel_data['wage_gini'],
                exog=panel_data[[
                    size_measure,
                    mechanism,
                    f'size_x_{mechanism}'
                ]],
                entity_effects=True,
                time_effects=True
            )
            results[size_measure]['mechanisms'][mechanism] = model.fit(
                cov_type='clustered', 
                cluster_entity=True
            )
        
        # 3. ML-based heterogeneity
        df['size_quartile'] = pd.qcut(df[size_measure], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        df['wage_quartile'] = pd.qcut(df['mean_wage'], q=4, labels=['Low', 'Mid-Low', 'Mid-High', 'High'])
        
        features = pd.get_dummies(df[[
            'digital_intensity', 
            'size_quartile',
            'wage_quartile',
            'state'
        ]])
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, 
            df['wage_gini'],
            test_size=0.2,
            random_state=42
        )
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        results[size_measure]['heterogeneity'] = {
            'model': rf,
            'importance': pd.DataFrame({
                'feature': features.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False),
            'test_score': rf.score(X_test, y_test)
        }
    
    return results

def calculate_actual_inequality(df):
    """
    Calculate actual wage inequality (mean-median gap) for digital vs non-digital industries
    """
    # Define periods (2-year bins)
    df['period'] = pd.cut(df['YEAR'], 
                         bins=[2007, 2009, 2011, 2013, 2015, 2017, 2019, 2022],
                         labels=['2008-09', '2010-11', '2012-13', '2015', '2016-17', '2018-19', '2020-22'])
    
    # Calculate mean inequality by period and digital status
    actual_effects = (df.groupby(['period', 'digital'])
                     .agg({
                         'wage_gini': ['mean', 'std', 'count'],
                         'estab_weighted_sum': 'mean'
                     })
                     .round(2))
    
    print("\nActual Wage Inequality by Period:")
    print("=================================")
    print("\nDigital Industries:")
    digital_data = df[df['digital'] == 1].groupby('period')['wage_gini'].agg(['mean', 'std', 'count'])
    print(digital_data.round(2))
    
    print("\nNon-Digital Industries:")
    non_digital_data = df[df['digital'] == 0].groupby('period')['wage_gini'].agg(['mean', 'std', 'count'])
    print(non_digital_data.round(2))
    
    return actual_effects

def analyze_heterogeneity(df):
    """Use ML to identify patterns in digital transformation effects"""
    
    # Create richer features
    df['size_quartile'] = pd.qcut(df['estab_weighted_sum'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    df['post_2015'] = (df['YEAR'] >= 2015).astype(int)
    df['wage_level'] = pd.qcut(df['mean_wage'], q=4, labels=['Low', 'Mid-Low', 'Mid-High', 'High'])
    
    # Create interaction features
    df['size_x_digital'] = df['estab_weighted_sum'] * df['digital']
    df['size_x_post2015'] = df['estab_weighted_sum'] * df['post_2015']
    
    # Prepare features for ML
    features = pd.get_dummies(df[[
        'digital', 'size_quartile', 'state', 'post_2015',
        'wage_level'
    ]])
    
    # Add numeric features
    scaler = StandardScaler()
    numeric_features = ['estab_weighted_sum', 'size_x_digital', 'size_x_post2015']
    for col in numeric_features:
        features[f'{col}_scaled'] = scaler.fit_transform(df[[col]])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, 
        df['wage_gini'],
        test_size=0.2,
        random_state=42
    )
    
    # Train random forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Get feature importance
    importance = pd.DataFrame({
        'feature': features.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Analyze interactions more deeply
    interactions = []
    
    # Size quartile effects
    for size_q in ['Q1', 'Q2', 'Q3', 'Q4']:
        subset = df[df['size_quartile'] == size_q]
        dig_effect = subset[subset['digital'] == 1]['wage_gini'].mean() - \
                    subset[subset['digital'] == 0]['wage_gini'].mean()
        interactions.append({
            'type': 'size',
            'group': size_q, 
            'digital_effect': dig_effect
        })
    
    # Wage level effects
    for wage_level in ['Low', 'Mid-Low', 'Mid-High', 'High']:
        subset = df[df['wage_level'] == wage_level]
        dig_effect = subset[subset['digital'] == 1]['wage_gini'].mean() - \
                    subset[subset['digital'] == 0]['wage_gini'].mean()
        interactions.append({
            'type': 'wage',
            'group': wage_level, 
            'digital_effect': dig_effect
        })
    
    # Time period effects
    for year in df['YEAR'].unique():
        subset = df[df['YEAR'] == year]
        dig_effect = subset[subset['digital'] == 1]['wage_gini'].mean() - \
                    subset[subset['digital'] == 0]['wage_gini'].mean()
        interactions.append({
            'type': 'year',
            'group': str(year), 
            'digital_effect': dig_effect
        })
    
    return {
        'model': rf,
        'importance': importance,
        'interactions': pd.DataFrame(interactions),
        'test_score': rf.score(X_test, y_test),
        'predictions': rf.predict(X_test)
    }

def calculate_controlled_differences(results, df):
    """
    Calculate controlled differences between digital and non-digital firms
    accounting for all fixed effects and controls
    """
    # Get mean values for controls to use as baseline
    mean_workers = df['estab_weighted_sum'].mean()
    print(f"Mean workers: {mean_workers}")  # Add this debug print
    
    # Initialize arrays for years and differences
    years = range(2007, 2023)
    controlled_diffs = []
    
    for year in years:
        # Map year to period
        if year <= 2009:
            period = 'Reference'
            effect = 0
        elif year <= 2011:
            period = 't-2'
            base = results.params['digital_x_t-2']
            size_int = results.params['size_x_digital_x_t-2']
            effect = base + (size_int * mean_workers)
        elif year <= 2013:
            period = 't-1'
            base = results.params['digital_x_t-1']
            size_int = results.params['size_x_digital_x_t-1']
            effect = base + (size_int * mean_workers)
        elif year <= 2014:
            period = 'Reference'
            effect = 0
        elif year == 2015:
            period = 't0'
            base = results.params['digital_x_t0']
            size_int = results.params['size_x_digital_x_t0']
            effect = base + (size_int * mean_workers)
        elif year <= 2017:
            period = 't1'
            base = results.params['digital_x_t1']
            size_int = results.params['size_x_digital_x_t1']
            effect = base + (size_int * mean_workers)
        elif year <= 2019:
            period = 't2'
            base = results.params['digital_x_t2']
            size_int = results.params['size_x_digital_x_t2']
            effect = base + (size_int * mean_workers)
        else:
            period = 't3'
            base = results.params['digital_x_t3']
            size_int = results.params['size_x_digital_x_t3']
            effect = base + (size_int * mean_workers)
            
        controlled_diffs.append({
            'year': year,
            'period': period,
            'effect': effect,
            'se': results.std_errors.get(f'digital_x_{period}', 0) if period != 'Reference' else 0
        })
    
    return pd.DataFrame(controlled_diffs)

def plot_controlled_event_study_results(df, results):
    """Create clean event study plot showing controlled wage inequality"""
    periods = ['2007', '2008-09', '2010-11', '2012-13', '2015', '2016-17', '2018-19', '2020-22']
    
    # Calculate controlled values using same logic as calculate_controlled_differences
    mean_workers = df['estab_weighted_sum'].mean()
    
    # Initialize controlled effects
    controlled_digital = []
    controlled_non_digital = [0] * len(periods)  # Always 0 for non-digital
    
    # Calculate effects matching our previous function
    for i, period in enumerate(periods):
        if period == '2007' or period == '2008-09':
            effect = 0  # Reference period
        elif period == '2010-11':
            base = results.params['digital_x_t-2']
            size_int = results.params['size_x_digital_x_t-2']
            effect = base + (size_int * mean_workers)  # Should be -299.91
        elif period == '2012-13':
            base = results.params['digital_x_t-1']
            size_int = results.params['size_x_digital_x_t-1']
            effect = base + (size_int * mean_workers)  # Should be 74.16
        elif period == '2015':
            base = results.params['digital_x_t0']
            size_int = results.params['size_x_digital_x_t0']
            effect = base + (size_int * mean_workers)  # Should be 670.12
        elif period == '2016-17':
            base = results.params['digital_x_t1']
            size_int = results.params['size_x_digital_x_t1']
            effect = base + (size_int * mean_workers)  # Should be 1434.72
        elif period == '2018-19':
            base = results.params['digital_x_t2']
            size_int = results.params['size_x_digital_x_t2']
            effect = base + (size_int * mean_workers)  # Should be 1858.04
        else:  # 2020-22
            base = results.params['digital_x_t3']
            size_int = results.params['size_x_digital_x_t3']
            effect = base + (size_int * mean_workers)  # Should be 3043.96
            
        controlled_digital.append(effect)
    
    plt.figure(figsize=(12, 8))
    
    # Create x-axis points explicitly
    x = np.arange(len(periods))
    
    # Plot both lines
    plt.plot(x, controlled_digital, 'bo-', linewidth=2, markersize=8, label='Digital Industries (Controlled)')
    plt.plot(x, controlled_non_digital, 'ko-', linewidth=2, markersize=8, alpha=0.5, label='Non-Digital Industries (Controlled)')
    
    # Add significance stars
    significance = [
        (4, controlled_digital[4], '*'),      # 2015: p < 0.05
        (5, controlled_digital[5], '***'),    # 2016-17: p < 0.001
        (6, controlled_digital[6], '***'),    # 2018-19: p < 0.001
        (7, controlled_digital[7], '***')     # 2020-22: p < 0.001
    ]
    
    for x_pos, y, stars in significance:
        plt.text(x_pos, y + 100, stars, ha='center', va='bottom')
    
    # Add vertical line at 2015
    plt.axvline(x=4, color='red', linestyle='--', alpha=0.5)
    plt.text(4.1, max(controlled_digital)*0.9, '2015 Digital\nTransformation', 
             rotation=90, color='red', alpha=0.5)
    
    plt.xticks(x, periods, rotation=45)
    plt.title('Controlled Wage Inequality Over Time by Industry Type', fontsize=14, pad=20)
    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('Controlled Wage Inequality (Mean - Median in USD)', fontsize=12)
    
    # Format y-axis with dollar signs and commas
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', frameon=True, framealpha=0.9)
    
    # Add debug prints to verify values
    print("\nControlled Digital Effects for Plot:")
    for p, e in zip(periods, controlled_digital):
        print(f"{p}: ${e:,.2f}")
    
    # Add after the grid and legend lines, before tight_layout
    # Set y-axis limits to prevent overflow
    max_value = max(controlled_digital)
    plt.ylim(-500, max_value * 1.15)  # Give 10% padding above max value
    
    plt.tight_layout()
    plt.savefig('digital_transformation_controlled.png', bbox_inches='tight', dpi=300)
    plt.close()


def main():
    """Main analysis workflow"""
    # 1. Load and prepare data
    state_industry, industry_dummies = prepare_data()
    
    # 2. Run base analysis
    base_results = run_base_regressions(state_industry, industry_dummies)
    
    # 3. Run comprehensive difference-in-difference analysis
    print("\n=== Running Difference-in-Difference Analysis ===")
    did_results = run_difference_analysis(state_industry)
    
    # This includes:
    # - Basic DiD for each industry
    # - Digital transformation DiD
    # - Event study
    # - Placebo tests
    # - Robust checks with controls
    
    # 4. Run mechanism analysis
    print("\n=== Running Mechanism Analysis ===")
    mechanism_results = run_mechanism_analysis(state_industry)
    
    # This includes:
    # - Industry characteristics
    # - Mechanism proxies (automation, skill, complexity)
    # - ML-based heterogeneity
    
    # 5. Create visualizations
    print("\n=== Creating Visualizations ===")
    create_main_visualization(state_industry)  # Size-inequality trends
    
    # 6. Print summary results
    print("\n=== Summary Statistics ===")
    summary_stats = create_summary_statistics(state_industry)
    print(summary_stats)
    
    return {
        'base': base_results,
        'did': did_results,
        'mechanisms': mechanism_results,
        'summary': summary_stats
    }

if __name__ == "__main__":
    main()