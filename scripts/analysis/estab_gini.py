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

def calculate_wage_gini(df):
    """Calculate Gini coefficient for wage distribution."""
    df = df.copy()
    
    # Wage distribution calculation by size categories
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
    
    for emp_col, pay_col in size_categories:
        # Total payroll and employment for the category
        total_payroll = df[pay_col]
        total_employment = df[emp_col]
        
        if total_employment > 0:  # Avoid division by zero
            # Average wage in the category
            avg_wage = total_payroll / total_employment
            
            # Append wage and corresponding weight (number of workers)
            wages.append(avg_wage)
            weights.append(total_employment)
    
    # Combine wages and weights into a single array for sorting
    wages = np.array(wages)
    weights = np.array(weights)
    
    # Sort wages and weights
    sorted_indices = np.argsort(wages)
    wages = wages[sorted_indices]
    weights = weights[sorted_indices]
    
    # Cumulative proportions for Gini calculation
    cumulative_weights = np.cumsum(weights)
    total_weight = np.sum(weights)
    cumulative_share_weights = cumulative_weights / total_weight
    cumulative_share_wages = np.cumsum(wages * weights) / np.sum(wages * weights)
    
    # Calculate Gini coefficient using the Lorenz curve
    gini = 1 - 2 * np.trapz(cumulative_share_wages, cumulative_share_weights)
    
    return gini

# def calculate_wage_gini(df):
#     """Calculate Gini coefficient for wage distribution."""
#     size_categories = [
#         ('employment_02: <5', 'annual_payroll_02: <5'),
#         ('employment_03: 5-9', 'annual_payroll_03: 5-9'),
#         ('employment_04: 10-19', 'annual_payroll_04: 10-19'),
#         ('employment_06: 20-99', 'annual_payroll_06: 20-99'),
#         ('employment_07: 100-499', 'annual_payroll_07: 100-499'),
#         ('employment_09: 500+', 'annual_payroll_09: 500+')
#     ]
    
#     wages = []
#     weights = []
    
#     # Collect non-zero wage data
#     for emp_col, pay_col in size_categories:
#         emp = float(df[emp_col])
#         pay = float(df[pay_col])
#         if emp > 0 and pay > 0:  # Ensure both are positive
#             avg_wage = pay / emp
#             if avg_wage > 0:  # Additional check for positive wages
#                 wages.append(avg_wage)
#                 weights.append(emp)
    
#     # Return 0 if insufficient data
#     if len(wages) < 2:
#         return 0
    
#     # Convert to numpy arrays
#     wages = np.array(wages, dtype=float)
#     weights = np.array(weights, dtype=float)
    
#     # Normalize weights to sum to 1
#     weights = weights / np.sum(weights)
    
#     # Sort both arrays by wage
#     sorted_indices = np.argsort(wages)
#     wages = wages[sorted_indices]
#     weights = weights[sorted_indices]
    
#     # Calculate cumulative proportions
#     cumsum_weights = np.cumsum(weights)
#     cumsum_weighted_wages = np.cumsum(weights * wages)
#     total_weighted_wages = np.sum(weights * wages)
    
#     # Ensure no division by zero
#     if total_weighted_wages == 0:
#         return 0
    
#     # Calculate Lorenz curve points
#     lorenz_curve = cumsum_weighted_wages / total_weighted_wages
    
#     # Calculate Gini coefficient
#     # G = 1 - 2 * area under Lorenz curve
#     # Area = sum of trapezoids
#     gini = 1 - np.sum((cumsum_weights[1:] - cumsum_weights[:-1]) * 
#                       (lorenz_curve[1:] + lorenz_curve[:-1]))
    
#     return gini

def clean_data(df):
    """Clean data and return a copy"""
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Remove zero establishments
    df = df[df['establishments_01: Total'] > 0]

    # Remove 2022 data if present
    df = df[df['YEAR'] <= 2021]
    
    # Calculate establishment shares for each size category
    size_categories = {
        'establishments_02: <5': 2.5,      # midpoint of 0-4
        'establishments_03: 5-9': 7,       # midpoint of 5-9
        'establishments_04: 10-19': 14.5,  # midpoint of 10-19
        'establishments_06: 20-99': 59.5,  # midpoint of 20-99
        'establishments_07: 100-499': 299.5, # midpoint of 100-499
        'establishments_09: 500+': 500     # minimum of 500+ category
    }
    
    # Calculate weighted size index
    weighted_sum = 0
    for col, midpoint in size_categories.items():
        share = (df[col] / df['establishments_01: Total']).fillna(0)
        weighted_sum += share * midpoint
    
    # Store the weighted size index
    df['estab_weighted_sum'] = weighted_sum
    
    # Flag impossible wage relationships
    df.loc[:, 'wage_relationship_flag'] = df['mean_wage'] < df['median_wage']
    
    # Balance panel (optional)
    min_years = 10  # or another threshold
    valid_pairs = df.groupby('state_industry')['YEAR'].count() >= min_years
    df = df[df['state_industry'].isin(valid_pairs[valid_pairs].index)]

    df['wage_gini'] = df.apply(calculate_wage_gini, axis=1)

    return df

def load_and_prepare_data():
    """Load and prepare data with verification"""
    state_industry = pd.read_csv('../../data/merged_final.csv')
    state_industry['YEAR'] = pd.to_numeric(state_industry['YEAR'])
    state_industry['state_industry'] = state_industry['state_name'] + "_" + state_industry['industry_name']
    state_industry = clean_data(state_industry)
    state_industry['wage_gini'] = state_industry['wage_gini']
    industry_dummies = pd.get_dummies(state_industry['industry_name'], prefix='ind', drop_first=True)
    
    return state_industry, industry_dummies

def run_all_regressions(state_industry, industry_dummies):
    """Run all base regressions with rank check disabled"""
    results = {}
    panel_data = state_industry.set_index(['state_industry', 'YEAR'])
    
    # 1a. Size effect on median wages
    model = PanelOLS(
        dependent=panel_data['median_wage'],
        exog=sm.add_constant(panel_data[['estab_weighted_sum']]),
        entity_effects=True,
        time_effects=True,
        check_rank=False
    )
    results['size_wage'] = model.fit(cov_type='clustered', cluster_entity=True)
    
    # 1b. Size effect on wage inequality
    model = PanelOLS(
        dependent=panel_data['wage_gini'],
        exog=sm.add_constant(panel_data[['estab_weighted_sum']]),
        entity_effects=True,
        time_effects=True,
        check_rank=False
    )
    results['size_inequality'] = model.fit(cov_type='clustered', cluster_entity=True)
    
    # 2. Industry Wage Premiums
    exog_data = pd.concat([industry_dummies, state_industry[['workers']]], axis=1)
    exog_data = sm.add_constant(exog_data)
    exog_data = exog_data.set_index(panel_data.index)
    
    model = PanelOLS(
        dependent=panel_data['median_wage'],
        exog=exog_data,
        entity_effects=False,
        time_effects=True,
        check_rank=False
    )
    results['industry_premiums'] = model.fit(cov_type='clustered', cluster_entity=True)
    
    # 3. Size-Industry Interactions
    major_industries = ['Manufacturing', 'Finance and Insurance', 'Retail Trade']
    for ind in major_industries:
        state_industry[f'size_x_{ind}'] = (
            state_industry['estab_weighted_sum'] * 
            (state_industry['industry_name'] == ind)
        )
    
    interaction_data = sm.add_constant(
        state_industry[[f'size_x_{ind}' for ind in major_industries] + ['estab_weighted_sum']]
    )
    interaction_data = interaction_data.set_index(panel_data.index)
    
    model = PanelOLS(
        dependent=panel_data['wage_gini'],
        exog=interaction_data,
        entity_effects=True,
        time_effects=True,
        check_rank=False
    )
    results['size_industry_interactions'] = model.fit(cov_type='clustered', cluster_entity=True)
    
    return results

def print_results(results):
    """Print results in a clear, organized way"""
    print("\n=== 1. Establishment Size Effects ===")
    print("\na) Effect on Median Wages:")
    print(results['size_wage'].summary)
    
    print("\nb) Effect on Wage Inequality:")
    print(results['size_inequality'].summary)
    
    print("\n=== 2. Industry Wage Premiums ===")
    print(results['industry_premiums'].summary)
    
    print("\n=== 3. Size-Industry Interactions ===")
    print(results['size_industry_interactions'].summary)
    
    # Save detailed results to CSV
    for name, result in results.items():
        params = pd.DataFrame({
            'coefficient': result.params,
            'std_error': result.std_errors,
            't_stat': result.tstats,
            'p_value': result.pvalues
        })
        params.to_csv(f'results_{name}.csv')

def verify_regression_assumptions(model, results):
    """Verify regression assumptions and print diagnostics"""
    print("\n=== Regression Diagnostics ===")
    
    # 1. Check R-squared and model fit
    print("\nModel Fit:")
    print(f"R-squared: {results.rsquared:.4f}")
    print(f"Within R-squared: {results.rsquared_within:.4f}")
    
    # 2. Check residuals
    residuals = results.resids
    print("\nResiduals Summary:")
    print(pd.Series(residuals).describe())
    
    # 3. Check for serial correlation
    # Add appropriate test here
    
    return True

def analyze_time_trends(df):
    """Analyze how size effects change over time"""
    # Create a copy to avoid warnings
    df = df.copy()
    
    # Pre/Post Financial Crisis
    df['post_2008'] = df['YEAR'] >= 2008
    df['size_x_post2008'] = df['estab_weighted_sum'] * df['post_2008']
    
    # Pre/Post COVID
    df['post_covid'] = df['YEAR'] >= 2020
    df['size_x_postcovid'] = df['estab_weighted_sum'] * df['post_covid']
    
    # Time trend
    df['time_trend'] = df['YEAR'] - df['YEAR'].min()
    df['size_x_trend'] = df['estab_weighted_sum'] * df['time_trend']
    
    return df

def run_robustness(df):
    """Run robustness checks"""
    df = df.copy()
    
    # Non-linear size effects
    df['workers_squared'] = df['estab_weighted_sum'] ** 2
    df['workers_cubed'] = df['estab_weighted_sum'] ** 3
    
    # Alternative inequality measures
    df['log_inequality'] = np.log(df['wage_gini'].clip(lower=1))
    
    # Size categories
    df['size_category'] = pd.qcut(df['estab_weighted_sum'], q=5, labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'])
    
    return df

def test_regional_differences(results):
    """Test if differences between regional coefficients are statistically significant"""
    print("\n=== Tests of Regional Differences ===")
    
    regions = ['Midwest', 'Northeast', 'West', 'South']
    
    # Get coefficients and standard errors
    coef_data = {
        region: {
            'coef': results['regional'][region].params['estab_weighted_sum'],
            'se': results['regional'][region].std_errors['estab_weighted_sum']
        }
        for region in regions
    }
    
    # Perform pairwise tests
    for i, region1 in enumerate(regions):
        for region2 in regions[i+1:]:
            # Calculate z-statistic for difference
            coef_diff = coef_data[region1]['coef'] - coef_data[region2]['coef']
            se_diff = (coef_data[region1]['se']**2 + coef_data[region2]['se']**2)**0.5
            z_stat = coef_diff / se_diff
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # Two-tailed test
            
            print(f"\n{region1} vs {region2}:")
            print(f"Difference in coefficients: {coef_diff:.4f}")
            print(f"Z-statistic: {z_stat:.4f}")
            print(f"P-value: {p_value:.4f}")
            print(f"Significant at 5% level: {'Yes' if p_value < 0.05 else 'No'}")

def run_extended_regressions(df):
    """Run additional regression specifications"""
    results = {}
    
    # First, create region mapping
    regions = {
        'Northeast': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'],
        'Midwest': ['OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
        'South': ['DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA', 'OK', 'TX'],
        'West': ['MT', 'ID', 'WY', 'CO', 'NM', 'AZ', 'UT', 'NV', 'WA', 'OR', 'CA', 'AK', 'HI']
    }
    
    # Create state to region mapping using full state names
    state_name_to_abbrev = {
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
    
    # Create state to region mapping
    state_to_region = {}
    for region, abbrevs in regions.items():
        for state, abbrev in state_name_to_abbrev.items():
            if abbrev in abbrevs:
                state_to_region[state] = region
    
    # Add region to dataframe
    df = df.copy()
    df['region'] = df['state_name'].map(state_to_region)
    
    # Create panel data
    panel_data = df.set_index(['state_industry', 'YEAR'])
    
    # 1. Base Model
    model = PanelOLS(
        dependent=panel_data['wage_gini'],
        exog=sm.add_constant(panel_data[['estab_weighted_sum']]),
        entity_effects=True,
        time_effects=True,
        check_rank=False
    )
    results['base_model'] = model.fit(cov_type='clustered', cluster_entity=True)
    
    # 2. Time periods analysis
    periods_data = panel_data.copy()
    periods_data['post_2015'] = (periods_data.index.get_level_values('YEAR') >= 2015).astype(float)
    periods_data['size_post2015'] = periods_data['estab_weighted_sum'] * periods_data['post_2015']
    
    model = PanelOLS(
        dependent=periods_data['wage_gini'],
        exog=sm.add_constant(periods_data[['estab_weighted_sum', 'size_post2015']]),
        entity_effects=True,
        time_effects=False,
        check_rank=False
    )
    results['time_effects'] = model.fit(cov_type='clustered', cluster_entity=True)
    
    # 3. Non-linear effects
    nonlinear_data = panel_data.copy()
    nonlinear_data['workers_squared'] = nonlinear_data['estab_weighted_sum'] ** 2
    
    model = PanelOLS(
        dependent=nonlinear_data['wage_gini'],
        exog=sm.add_constant(nonlinear_data[['estab_weighted_sum', 'workers_squared']]),
        entity_effects=True,
        time_effects=True,
        check_rank=False
    )
    results['nonlinear'] = model.fit(cov_type='clustered', cluster_entity=True)
    
    # 4. Regional Analysis
    results_by_region = {}
    print("\nDebugging Regional Analysis:")
    print("Available regions:", df['region'].unique())
    
    for region in df['region'].unique():
        print(f"\nProcessing region: {region}")
        # Get state_industry values for this region
        region_states = df[df['region'] == region]['state_industry'].unique()
        print(f"Number of state-industries in region: {len(region_states)}")
        
        # Filter panel data using index
        region_data = panel_data[panel_data.index.get_level_values('state_industry').isin(region_states)]
        print(f"Number of observations in region: {len(region_data)}")
        
        if len(region_data) > 0:
            try:
                model = PanelOLS(
                    dependent=region_data['wage_gini'],
                    exog=sm.add_constant(region_data[['estab_weighted_sum']]),
                    entity_effects=True,
                    time_effects=True,
                    check_rank=False
                )
                results_by_region[region] = model.fit(cov_type='clustered', cluster_entity=True)
                print(f"Successfully estimated model for {region}")
            except Exception as e:
                print(f"Could not estimate model for region {region}: {str(e)}")
        else:
            print(f"No data available for region {region}")
    
    results['regional'] = results_by_region
    print(f"\nTotal regions processed: {len(results_by_region)}")
    
    if len(results_by_region) > 0:
        test_regional_differences(results)
    
    return results

def print_extended_results(results):
    """Print results from extended analysis"""
    print("\n=== Extended Analysis Results ===")
    
    print("\n1. Base Model:")
    print(results['base_model'].summary.tables[1])
    
    print("\n2. Time Effects:")
    print(results['time_effects'].summary.tables[1])
    
    print("\n3. Non-linear Effects:")
    print(results['nonlinear'].summary.tables[1])
    
    print("\n4. Regional Analysis:")
    if 'regional' in results:
        for region, result in results['regional'].items():
            print(f"\nRegion: {region}")
            if result is not None:
                print(result.summary.tables[1])
            else:
                print("No results available for this region")
    else:
        print("No regional results available")

def run_time_effects(panel_data):
    """Run specific analysis of COVID impact on size-inequality relationship"""
    print("\n=== COVID Period Analysis ===")
    
    # Create COVID period indicator
    panel_data = panel_data.copy()
    panel_data['post_covid'] = (panel_data.index.get_level_values('YEAR') >= 2020).astype(float)
    panel_data['size_x_covid'] = panel_data['estab_weighted_sum'] * panel_data['post_covid']
    
    # Run regression with COVID interaction
    model = PanelOLS(
        dependent=panel_data['wage_gini'],
        exog=sm.add_constant(panel_data[['estab_weighted_sum', 'size_x_covid']]),
        entity_effects=True,
        time_effects=True,
        check_rank=False
    )
    results = model.fit(cov_type='clustered', cluster_entity=True)
    
    print("\nEffect of establishment size on wage inequality during COVID:")
    print(results.summary.tables[1])
    
    # Calculate total effect during COVID
    covid_total_effect = results.params['estab_weighted_sum'] + results.params['size_x_covid']
    covid_se = (results.std_errors['estab_weighted_sum']**2 + results.std_errors['size_x_covid']**2)**0.5
    t_stat = covid_total_effect / covid_se
    p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    
    print("\nTotal effect during COVID period:")
    print(f"Coefficient: {covid_total_effect:.4f}")
    print(f"Standard Error: {covid_se:.4f}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_val:.4f}")
    
    return results

def analyze_post2015_drivers(df):
    """Analyze what drove the post-2015 change"""
    # Create interaction terms for key industries
    industries_to_check = [
        'Information',  # Tech sector
        'Professional, Scientific, and Technical Services',
        'Finance and Insurance',
        'Retail Trade',
        'Transportation and Warehousing'  # Gig economy effects
    ]
    
    for ind in industries_to_check:
        df[f'post2015_x_{ind}'] = (
            (df['YEAR'] >= 2015) * 
            (df['industry_name'] == ind)
        )
    
    return df

def create_summary_statistics(df):
    """Create comprehensive summary statistics tables"""
    
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
    key_vars = ['workers', 'estab_weighted_sum', 'median_wage', 'mean_wage', 'wage_gini']  # Added 'workers'
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
        is_ratio = col == 'wage_gini'  # Only wage_gini is a ratio
        var_stats[col] = var_stats[col].apply(lambda x: format_number(x, is_ratio))

    # 3. Coverage Statistics
    industries_per_state = df.groupby('state_name')['industry_name'].nunique()
    years_per_pair = df.groupby('state_industry')['YEAR'].nunique()
    pairs_per_year = df.groupby('YEAR')['state_industry'].nunique()
    
    potential_obs = df['state_name'].nunique() * df['industry_name'].nunique() * df['YEAR'].nunique()
    
    coverage_stats = pd.DataFrame({
        'Metric': [
            'Avg Industries per State',
            'Min Industries per State',
            'Max Industries per State',
            'Avg Years per State-Industry',
            'Avg State-Industry Pairs per Year',
            'Balanced Panels (%)',
            'Data Completeness (%)'
        ],
        'Value': [
            f"{industries_per_state.mean():.1f}",
            str(industries_per_state.min()),
            str(industries_per_state.max()),
            f"{years_per_pair.mean():.1f}",
            f"{pairs_per_year.mean():.1f}",
            f"{(years_per_pair == years_per_pair.max()).mean()*100:.1f}%",
            f"{(len(df)/potential_obs)*100:.1f}%"
        ]
    })

    # 4. Data Quality Indicators
    missing_by_col = df[key_vars].isnull().sum()  # Only check key variables
    total_missing = missing_by_col.sum()
    rows_with_missing = df[key_vars].isnull().any(axis=1).sum()
    
    quality_stats = pd.DataFrame({
        'Metric': [
            'Rows with Missing Values',
            'Total Missing Values',
            'Mean < Median Cases',
            'Worker Variable Outliers',
            'Zero/Negative Wages'
        ],
        'Count': [
            rows_with_missing,
            total_missing,
            (df['mean_wage'] < df['median_wage']).sum(),
            len(identify_outliers(df['estab_weighted_sum'])),
            (df['median_wage'] <= 0).sum()
        ]
    })
    # Calculate percentages correctly
    quality_stats['Percentage'] = [
        f"{(rows_with_missing / len(df) * 100):.2f}%",  # Percentage of rows with any missing
        f"{(total_missing / (len(df) * len(key_vars)) * 100):.2f}%",  # Percentage of total possible values
        f"{((df['mean_wage'] < df['median_wage']).sum() / len(df) * 100):.2f}%",
        f"{(len(identify_outliers(df['estab_weighted_sum'])) / len(df) * 100):.2f}%",
        f"{((df['median_wage'] <= 0).sum() / len(df) * 100):.2f}%"
    ]

    # 5. Regional Statistics (updated)
    regional_stats = df.groupby('region').agg({
        'state_industry': ['count', lambda x: f"{len(x)/len(df)*100:.1f}%"],
        'state_name': 'nunique',
        'industry_name': 'nunique',
        'workers': 'mean',  # Just the mean worker count as additional info
        'estab_weighted_sum': ['mean', 'median']  # Main size metric
    })
    regional_stats.columns = ['Observations', 'Share', 'States', 'Industries', 
                            'Mean Workers',  # Single worker metric
                            'Mean Estab Size', 'Median Estab Size']  # Size metrics
    regional_stats = regional_stats.applymap(lambda x: format_number(x) if isinstance(x, (int, float)) and not isinstance(x, str) else x)

    # 6. Time Series Properties
    time_periods = {
        'Pre-2015': df['YEAR'] < 2015,
        'Post-2015': (df['YEAR'] >= 2015) & (df['YEAR'] < 2020),
        'COVID Period': df['YEAR'] >= 2020
    }
    
    time_stats = []
    for period, mask in time_periods.items():
        period_data = df[mask]
        time_stats.append({
            'Period': period,
            'Years': f"{period_data['YEAR'].min()}-{period_data['YEAR'].max()}",
            'Observations': len(period_data),
            'Share': f"{len(period_data)/len(df)*100:.1f}%",
            'Mean Workers': format_number(period_data['workers'].mean()),
            'Mean Estab Size': format_number(period_data['estab_weighted_sum'].mean()),
            'Mean Wage Gini': format_number(period_data['wage_gini'].mean(), is_ratio=True),  # Added is_ratio=True
            'Median Wage Gini': format_number(period_data['wage_gini'].median(), is_ratio=True)  # Added is_ratio=True
        })
    
    time_stats = pd.DataFrame(time_stats)

    return format_summary_tables(basic_info, var_stats, coverage_stats, 
                               quality_stats, regional_stats, time_stats)

def create_region_mapping():
    """Helper function to create state to region mapping"""
    region_dict = {
        'Northeast': ['Maine', 'New Hampshire', 'Vermont', 'Massachusetts', 'Rhode Island', 
                     'Connecticut', 'New York', 'New Jersey', 'Pennsylvania'],
        'Midwest': ['Ohio', 'Indiana', 'Illinois', 'Michigan', 'Wisconsin', 'Minnesota', 
                   'Iowa', 'Missouri', 'North Dakota', 'South Dakota', 'Nebraska', 'Kansas'],
        'South': ['Delaware', 'Maryland', 'District of Columbia', 'Virginia', 'West Virginia', 
                 'North Carolina', 'South Carolina', 'Georgia', 'Florida', 'Kentucky', 
                 'Tennessee', 'Alabama', 'Mississippi', 'Arkansas', 'Louisiana', 'Oklahoma', 'Texas'],
        'West': ['Montana', 'Idaho', 'Wyoming', 'Colorado', 'New Mexico', 'Arizona', 'Utah', 
                'Nevada', 'Washington', 'Oregon', 'California', 'Alaska', 'Hawaii']
    }
    return {state: region for region, states in region_dict.items() for state in states}

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
    
    # Calculate widths based on whether index should be included
    has_meaningful_index = not all(isinstance(idx, int) for idx in df.index)
    total_cols = len(df.columns) + (1 if has_meaningful_index else 0)
    col_width = (pdf.w - 20) / total_cols  # -20 for margins
    row_height = 7
    
    # Headers
    if has_meaningful_index:
        pdf.cell(col_width, row_height, 'Category', 1)  # Index header
    for col in df.columns:
        pdf.cell(col_width, row_height, str(col), 1)
    pdf.ln()
    
    # Data
    pdf.set_font('Arial', '', 10)
    for idx, row in df.iterrows():
        # Add index if meaningful
        if has_meaningful_index:
            pdf.cell(col_width, row_height, str(idx), 1)
        
        # Add each cell in the row
        for item in row:
            # Handle potential NaN values
            value = str(item) if pd.notna(item) else ''
            pdf.cell(col_width, row_height, value, 1)
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

def run_nonlinear_analysis(df):
    """Run comprehensive non-linear analysis of size effects"""
    results = {}
    panel_data = df.set_index(['state_industry', 'YEAR'])
    
    # 1. Cubic Model
    panel_data['workers_squared'] = panel_data['estab_weighted_sum'] ** 2
    panel_data['workers_cubed'] = panel_data['estab_weighted_sum'] ** 3
    
    model = PanelOLS(
        dependent=panel_data['wage_gini'],
        exog=sm.add_constant(panel_data[['estab_weighted_sum', 'workers_squared', 'workers_cubed']]),
        entity_effects=True,
        time_effects=True,
        check_rank=False,
        drop_absorbed=True
    )
    results['cubic'] = model.fit(cov_type='clustered', cluster_entity=True)
    
    # 2. Log Specification
    panel_data['log_workers'] = np.log(panel_data['estab_weighted_sum'])
    model = PanelOLS(
        dependent=panel_data['wage_gini'],
        exog=sm.add_constant(panel_data[['log_workers']]),
        entity_effects=True,
        time_effects=True,
        check_rank=False,
        drop_absorbed=True
    )
    results['log'] = model.fit(cov_type='clustered', cluster_entity=True)
    
    # 3. Industry-Specific
    key_industries = ['Finance and Insurance', 'Manufacturing', 'Retail Trade', 'Information']
    industry_results = {}
    
    for ind in key_industries:
        ind_data = panel_data[panel_data['industry_name'] == ind].copy()
        model = PanelOLS(
            dependent=ind_data['wage_gini'],
            exog=sm.add_constant(ind_data[['estab_weighted_sum', 'workers_squared']]),
            entity_effects=True,
            time_effects=True,
            check_rank=False,
            drop_absorbed=True
        )
        industry_results[ind] = model.fit(cov_type='clustered', cluster_entity=True)
    
    results['industry_specific'] = industry_results
    
    # 4. Time Evolution
    periods = {
        'pre_2015': panel_data.index.get_level_values('YEAR') < 2015,
        'post_2015': (panel_data.index.get_level_values('YEAR') >= 2015) & 
                     (panel_data.index.get_level_values('YEAR') < 2020),
        'covid': panel_data.index.get_level_values('YEAR') >= 2020
    }
    
    period_results = {}
    for period_name, mask in periods.items():
        period_data = panel_data[mask]
        model = PanelOLS(
            dependent=period_data['wage_gini'],
            exog=sm.add_constant(period_data[['estab_weighted_sum', 'workers_squared']]),
            entity_effects=True,
            time_effects=True,
            check_rank=False,
            drop_absorbed=True
        )
        period_results[period_name] = model.fit(cov_type='clustered', cluster_entity=True)
    
    results['time_evolution'] = period_results
    
    return results

def print_nonlinear_results(results):
    """Print comprehensive non-linear analysis results"""
    print("\n=== Non-linear Analysis Results ===")
    
    # 1. Cubic Model
    print("\n1. Cubic Model Results:")
    print(results['cubic'].summary.tables[1])
    
    # Only use available coefficients
    if 'workers_squared' in results['cubic'].params and 'workers_cubed' in results['cubic'].params:
        b2 = results['cubic'].params['workers_squared']
        b3 = results['cubic'].params['workers_cubed']
        # For cubic with no linear term, turning point is at x = 0 and x = -2b2/(3b3)
        tp = -2 * b2 / (3 * b3)
        print(f"\nTurning point at: {tp:,.0f} workers")
    
    # 2. Log Model
    print("\n2. Log Specification Results:")
    print(results['log'].summary.tables[1])
    
    # 3. Industry-Specific Results
    print("\n3. Industry-Specific Non-linear Patterns:")
    for ind, res in results['industry_specific'].items():
        print(f"\n{ind}:")
        print(res.summary.tables[1])
        # Calculate turning point only if both coefficients exist
        if 'workers_squared' in res.params:
            if 'estab_weighted_sum' in res.params:
                b1 = res.params['estab_weighted_sum']
                b2 = res.params['workers_squared']
                tp = -b1/(2*b2)
                print(f"Turning point: {tp:,.0f} workers")
            else:
                print("Linear term absorbed by fixed effects")
    
    # 4. Time Evolution
    print("\n4. Time Evolution of Non-linear Effects:")
    for period, res in results['time_evolution'].items():
        print(f"\n{period.replace('_', ' ').title()}:")
        print(res.summary.tables[1])
        # Calculate turning point only if both coefficients exist
        if 'workers_squared' in res.params:
            if 'estab_weighted_sum' in res.params:
                b1 = res.params['estab_weighted_sum']
                b2 = res.params['workers_squared']
                tp = -b1/(2*b2)
                print(f"Turning point: {tp:,.0f} workers")
            else:
                print("Linear term absorbed by fixed effects")

def run_diff_in_diff(df):
    """Run difference-in-differences analysis for each industry"""
    
    # Drop unknown industry
    df = df[df['industry_name'] != 'Unknown'].copy()
    
    results = {}
    
    # Run DiD for each industry
    for industry in df['industry_name'].unique():
        print(f"\n=== DiD Analysis for {industry} ===")
        
        # Create DiD variables
        df[f'{industry}_ind'] = (df['industry_name'] == industry).astype(int)
        df['post_2015'] = (df['YEAR'] >= 2015).astype(int)
        df['covid_period'] = (df['YEAR'] >= 2020).astype(int)
        df[f'ind_post2015'] = df[f'{industry}_ind'] * df['post_2015']
        df[f'ind_covid'] = df[f'{industry}_ind'] * df['covid_period']
        df[f'size_x_ind_post2015'] = df['estab_weighted_sum'] * df[f'ind_post2015']
        df[f'size_x_ind_covid'] = df['estab_weighted_sum'] * df[f'ind_covid']
        
        # Create year interactions for parallel trends
        for year in range(2010, 2015):
            df[f'year_{year}'] = (df['YEAR'] == year).astype(int)
            df[f'size_x_ind_{year}'] = df['estab_weighted_sum'] * df[f'{industry}_ind'] * df[f'year_{year}']
        
        # Set up panel data
        panel_data = df.set_index(['state_industry', 'YEAR'])
        
        # Main DiD with COVID
        model = PanelOLS(
            dependent=panel_data['wage_gini'],
            exog=sm.add_constant(panel_data[[
                'estab_weighted_sum', 
                f'{industry}_ind', 
                'post_2015', 
                'covid_period',
                f'size_x_ind_post2015', 
                f'size_x_ind_covid'
            ]]),
            entity_effects=True,
            time_effects=True,
            check_rank=False,
            drop_absorbed=True
        )
        main_results = model.fit(cov_type='clustered', cluster_entity=True)
        
        # Parallel trends test
        pre_trend_vars = [f'size_x_ind_{year}' for year in range(2010, 2015)]
        model_trends = PanelOLS(
            dependent=panel_data['wage_gini'],
            exog=sm.add_constant(panel_data[['estab_weighted_sum', f'{industry}_ind'] + pre_trend_vars]),
            entity_effects=True,
            time_effects=True,
            check_rank=False,
            drop_absorbed=True
        )
        trend_results = model_trends.fit(cov_type='clustered', cluster_entity=True)
        
        # Store results
        results[industry] = {
            'main': main_results,
            'trends': trend_results
        }
        
        # Print results
        print("\nMain DiD Results:")
        print(main_results.summary.tables[1])
        print("\nParallel Trends Test:")
        print(trend_results.summary.tables[1])
    
    return results


def run_INDUSTRY_diff_in_diff(df):
    """Run difference-in-differences analysis for each industry USING INDUSTRY SIZE"""
    
    # Drop unknown industry
    df = df[df['industry_name'] != 'Unknown'].copy()
    
    results = {}
    
    # Run DiD for each industry
    for industry in df['industry_name'].unique():
        print(f"\n=== DiD Analysis for {industry} ===")
        
        # Create DiD variables
        df[f'{industry}_ind'] = (df['industry_name'] == industry).astype(int)
        df['post_2015'] = (df['YEAR'] >= 2015).astype(int)
        df['covid_period'] = (df['YEAR'] >= 2020).astype(int)
        df[f'ind_post2015'] = df[f'{industry}_ind'] * df['post_2015']
        df[f'ind_covid'] = df[f'{industry}_ind'] * df['covid_period']
        df[f'size_x_ind_post2015'] = df['workers'] * df[f'ind_post2015']
        df[f'size_x_ind_covid'] = df['workers'] * df[f'ind_covid']
        
        # Create year interactions for parallel trends
        for year in range(2010, 2015):
            df[f'year_{year}'] = (df['YEAR'] == year).astype(int)
            df[f'size_x_ind_{year}'] = df['workers'] * df[f'{industry}_ind'] * df[f'year_{year}']
        
        # Set up panel data
        panel_data = df.set_index(['state_industry', 'YEAR'])
        
        # Main DiD with COVID
        model = PanelOLS(
            dependent=panel_data['wage_gini'],
            exog=sm.add_constant(panel_data[[
                'workers', 
                f'{industry}_ind', 
                'post_2015', 
                'covid_period',
                f'size_x_ind_post2015', 
                f'size_x_ind_covid'
            ]]),
            entity_effects=True,
            time_effects=True,
            check_rank=False,
            drop_absorbed=True
        )
        main_results = model.fit(cov_type='clustered', cluster_entity=True)
        
        # Parallel trends test
        pre_trend_vars = [f'size_x_ind_{year}' for year in range(2010, 2015)]
        model_trends = PanelOLS(
            dependent=panel_data['wage_gini'],
            exog=sm.add_constant(panel_data[['workers', f'{industry}_ind'] + pre_trend_vars]),
            entity_effects=True,
            time_effects=True,
            check_rank=False,
            drop_absorbed=True
        )
        trend_results = model_trends.fit(cov_type='clustered', cluster_entity=True)
        
        # Store results
        results[industry] = {
            'main': main_results,
            'trends': trend_results
        }
        
        # Print results
        print("\nMain DiD Results:")
        print(main_results.summary.tables[1])
        print("\nParallel Trends Test:")
        print(trend_results.summary.tables[1])
    
    return results

def run_event_study(df):
    """Run event study for key industries with optimized performance"""
    base_year = 2014
    years = sorted(df['YEAR'].unique())
    key_industries = [
        'Information',
        'Finance and Insurance',
        'Professional, Scientific, and Technical Services',
        'Manufacturing',
        'Retail Trade'
    ]
    
    # Pre-allocate all columns at once
    new_cols = {}
    for year in years:
        if year != base_year:
            new_cols[f'year_{year}'] = (df['YEAR'] == year).astype(int)
            for industry in key_industries:
                new_cols[f'{industry}_x_{year}'] = new_cols[f'year_{year}'] * (df['industry_name'] == industry).astype(int)
                new_cols[f'size_x_{industry}_x_{year}'] = new_cols[f'{industry}_x_{year}'] * df['estab_weighted_sum']
    
    # Add all columns at once
    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)
    
    results = {}
    print("\nEvent Study Results:")
    print("=" * 50)
    
    for industry in key_industries:
        print(f"\n{industry}:")
        panel_data = df.set_index(['state_industry', 'YEAR'])
        year_interactions = [f'size_x_{industry}_x_{year}' for year in years if year != base_year]
        
        model = PanelOLS(
            dependent=panel_data['wage_gini'],
            exog=sm.add_constant(panel_data[['estab_weighted_sum'] + year_interactions]),
            entity_effects=True,
            time_effects=True
        )
        results[industry] = model.fit(cov_type='clustered', cluster_entity=True)
        
        # Print key coefficients
        post_2015_coefs = {year: results[industry].params[f'size_x_{industry}_x_{year}'] 
                          for year in range(2015, 2022)}
        print("\nPost-2015 Effects:")
        for year, coef in post_2015_coefs.items():
            pval = results[industry].pvalues[f'size_x_{industry}_x_{year}']
            stars = '*' * sum([pval < 0.1, pval < 0.05, pval < 0.01])
            print(f"{year}: {coef:.3f} {stars}")
    
    return results

def run_placebo_tests(df):
    """Run placebo tests using random treatment years"""
    placebo_years = range(2010, 2014)
    results = {}
    
    for year in placebo_years:
        # Create placebo treatment
        df[f'post_{year}'] = (df['YEAR'] >= year).astype(int)
        df[f'size_x_post_{year}'] = df['estab_weighted_sum'] * df[f'post_{year}']
        
        # Run base regression with placebo
        panel_data = df.set_index(['state_industry', 'YEAR'])
        model = PanelOLS(
            dependent=panel_data['wage_gini'],
            exog=sm.add_constant(panel_data[[
                'estab_weighted_sum', 
                f'size_x_post_{year}'
            ]]),
            entity_effects=True,
            time_effects=True,
            drop_absorbed=True  # Add this to handle collinearity
        )
        results[year] = model.fit(cov_type='clustered', cluster_entity=True)
    
    return results

def analyze_industry_characteristics(df):
    """Analyze how industry characteristics affect size-inequality relationship"""
    # Add industry characteristics
    digital_industries = [
        'Information',
        'Finance and Insurance',
        'Professional, Scientific, and Technical Services'
    ]
    
    # Create digital intensity dummy and interactions
    df['digital_intensity'] = df['industry_name'].isin(digital_industries).astype(int)
    df['workers_x_digital_intensity'] = df['estab_weighted_sum'] * df['digital_intensity']
    df['post_2015'] = (df['YEAR'] >= 2015).astype(int)
    df['post2015_x_digital_intensity'] = df['post_2015'] * df['digital_intensity']
    
    # Run interaction analysis
    panel_data = df.set_index(['state_industry', 'YEAR'])
    model = PanelOLS(
        dependent=panel_data['wage_gini'],
        exog=sm.add_constant(panel_data[[
            'estab_weighted_sum',
            'workers_x_digital_intensity',
            'post2015_x_digital_intensity'
        ]]),
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True
    )
    
    return model.fit(cov_type='clustered', cluster_entity=True)

def analyze_mechanisms(df):
    """Analyze potential mechanisms driving size-inequality relationship"""
    # Create proxy variables
    
    # 1. Automation exposure proxy (based on industry digital intensity)
    digital_industries = [
        'Information',
        'Finance and Insurance',
        'Professional, Scientific, and Technical Services'
    ]
    df['automation_exposure'] = df['industry_name'].isin(digital_industries).astype(float)
    
    # 2. Skill intensity proxy (based on mean wages)
    df['skill_intensity'] = (df['mean_wage'] / df.groupby('YEAR')['mean_wage'].transform('mean')).fillna(0)
    
    # 3. Organizational complexity proxy (based on establishment size)
    df['organizational_complexity'] = (df['estab_weighted_sum'] / df.groupby('YEAR')['estab_weighted_sum'].transform('mean')).fillna(0)
    
    # Create interactions
    mechanisms = ['automation_exposure', 'skill_intensity', 'organizational_complexity']
    for mechanism in mechanisms:
        df[f'workers_x_{mechanism}'] = df['estab_weighted_sum'] * df[mechanism]
    
    # Create panel data
    panel_data = df.set_index(['state_industry', 'YEAR'])
    
    results = {}
    for mechanism in mechanisms:
        model = PanelOLS(
            dependent=panel_data['wage_gini'],
            exog=sm.add_constant(panel_data[[
                'estab_weighted_sum',
                f'workers_x_{mechanism}'
            ]]),
            entity_effects=True,
            time_effects=True,
            drop_absorbed=True
        )
        results[mechanism] = model.fit(cov_type='clustered', cluster_entity=True)
    
    return results

def analyze_digital_transformation_robust(df):
    """
    More robust version using:
    1. Dynamic event study specification
    2. Two-year bins to reduce noise
    3. Controls for industry-specific trends
    """
    # Define digital industries first!
    digital_industries = [
        'Information',
        'Finance and Insurance',
        'Professional, Scientific, and Technical Services'
    ]
    
    # Create digital indicator
    df['digital'] = df['industry_name'].isin(digital_industries).astype(int)
    
    # Create two-year bins (except 2015 as break year)
    df['period'] = pd.cut(df['YEAR'], 
                         bins=[2007, 2009, 2011, 2013, 2015, 2017, 2019, 2022],
                         labels=['2008-09', '2010-11', '2012-13', '2015', '2016-17', '2018-19', '2020-22'])
    
    # Create event-time dummies (using bins)
    df['event_time'] = df['period'].map({
        '2008-09': -3, '2010-11': -2, '2012-13': -1, 
        '2015': 0, '2016-17': 1, '2018-19': 2, '2020-22': 3
    })
    
    # Add industry-specific linear trends
    df['ind_trend'] = df.groupby('industry_name').cumcount()
    
    # Create interactions
    interactions = []
    for t in df['event_time'].unique():
        if t != -3:  # Use 2008-09 as reference period
            df[f'digital_x_t{t}'] = (df['event_time'] == t) * df['digital']
            df[f'size_x_digital_x_t{t}'] = df['estab_weighted_sum'] * df[f'digital_x_t{t}']
            interactions.extend([f'digital_x_t{t}', f'size_x_digital_x_t{t}'])
    
    # Run regression
    panel_data = df.set_index(['state_industry', 'YEAR'])
    model = PanelOLS(
        dependent=panel_data['wage_gini'],
        exog=sm.add_constant(panel_data[['estab_weighted_sum', 'ind_trend'] + interactions]),
        entity_effects=True,
        time_effects=True,
        check_rank=False,
        drop_absorbed=True
    )
    
    results = model.fit(cov_type='clustered', cluster_entity=True)
    print("\nRobust Event Study Results:")
    print(results.summary.tables[1])
    
    return results

def analyze_digital_transformation(df):
    """
    Run triple-difference analysis of digital transformation's effect on size-inequality relationship
    """
    # Define digital industries
    digital_industries = [
        'Information',
        'Finance and Insurance',
        'Professional, Scientific, and Technical Services'
    ]
    
    # Create key variables
    df['digital'] = df['industry_name'].isin(digital_industries).astype(int)
    df['post_2015'] = (df['YEAR'] >= 2015).astype(int)
    
    # Create interactions
    df['size_x_digital'] = df['estab_weighted_sum'] * df['digital']
    df['size_x_post2015'] = df['estab_weighted_sum'] * df['post_2015']
    df['digital_x_post2015'] = df['digital'] * df['post_2015']
    df['size_x_digital_x_post2015'] = df['estab_weighted_sum'] * df['digital'] * df['post_2015']
    
    # Run triple-diff regression
    panel_data = df.set_index(['state_industry', 'YEAR'])
    model = PanelOLS(
        dependent=panel_data['wage_gini'],
        exog=sm.add_constant(panel_data[[
            'estab_weighted_sum',
            'size_x_digital',
            'size_x_post2015',
            'digital_x_post2015',
            'size_x_digital_x_post2015'
        ]]),
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True
    )
    
    results = model.fit(cov_type='clustered', cluster_entity=True)
    
    # Add parallel trends test
    pre_2015 = df[df['YEAR'] < 2015].copy()
    pre_2015['event_time'] = pre_2015['YEAR'] - 2015
    
    # Create event-time interactions
    for year in range(-5, 0):  # 2010-2014
        pre_2015[f'digital_x_t{year}'] = (pre_2015['event_time'] == year) * pre_2015['digital']
        pre_2015[f'size_x_digital_x_t{year}'] = pre_2015['estab_weighted_sum'] * pre_2015[f'digital_x_t{year}']
    
    # Run parallel trends regression
    panel_pre = pre_2015.set_index(['state_industry', 'YEAR'])
    trend_vars = ([f'size_x_digital_x_t{t}' for t in range(-5, 0)] + 
                 [f'digital_x_t{t}' for t in range(-5, 0)])
    
    trend_model = PanelOLS(
        dependent=panel_pre['wage_gini'],
        exog=sm.add_constant(panel_pre[['estab_weighted_sum'] + trend_vars]),
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True
    )
    
    trend_results = trend_model.fit(cov_type='clustered', cluster_entity=True)
    
    return results, trend_results

def plot_digital_transformation_effect(df, results):
    """
    Create simplified visualization of digital transformation effect
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate average effects by year and digital status
    yearly_effects = df.groupby(['YEAR', 'digital'])['wage_gini'].mean().reset_index()
    
    # Plot two lines: digital and non-digital firms
    for is_digital in [0, 1]:
        data = yearly_effects[yearly_effects['digital'] == is_digital]
        label = 'Digital Industries' if is_digital else 'Non-Digital Industries'
        color = 'red' if is_digital else 'blue'
        
        plt.plot(data['YEAR'], data['wage_gini'], 
                color=color, label=label, linewidth=2)
    
    # Add vertical line at 2015
    plt.axvline(x=2015, color='gray', linestyle='--', alpha=0.5)
    plt.text(2015.2, plt.gca().get_ylim()[0], '2015 Digital\nTransformation', 
             rotation=90, verticalalignment='bottom')
    
    # Annotate key findings
    plt.text(2016, plt.gca().get_ylim()[1]*0.9,
             'Post-2015:\n+$1,732 level shift\n+$0.38 per worker',
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
    
    plt.title('Digital Transformation and Wage Inequality', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Mean-Median Wage Inequality ($)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.2)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('digital_transformation_effect.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_event_study_results(df):
    """Create clean event study plot showing actual wage inequality"""
    periods = ['2007', '2008-09', '2010-11', '2012-13', '2015', '2016-17', '2018-19', '2020-22']
    
    # Create period bins including 2007
    df['period'] = pd.cut(df['YEAR'], 
                         bins=[2006, 2007, 2009, 2011, 2013, 2015, 2017, 2019, 2022],
                         labels=periods,
                         include_lowest=True)
    
    # Get actual values
    digital_effects = df[df['digital'] == 1].groupby('period', observed=True)['wage_gini'].mean()
    non_digital_effects = df[df['digital'] == 0].groupby('period', observed=True)['wage_gini'].mean()
    
    plt.figure(figsize=(12, 8))
    
    # Create x-axis points explicitly
    x = np.arange(len(periods))  # Use numpy array instead of range
    
    # Plot both lines
    plt.plot(x, digital_effects, 'bo-', linewidth=2, markersize=8, label='Digital Industries')
    plt.plot(x, non_digital_effects, 'ko-', linewidth=2, markersize=8, alpha=0.5, label='Non-Digital Industries')
    
    # Add significance stars
    significance = [
        (4, digital_effects.iloc[4], '*'),      # 2015: p < 0.05
        (5, digital_effects.iloc[5], '***'),    # 2016-17: p < 0.001
        (6, digital_effects.iloc[6], '***'),    # 2018-19: p < 0.001
        (7, digital_effects.iloc[7], '***')     # 2020-22: p < 0.001
    ]
    
    for x_pos, y, stars in significance:
        plt.text(x_pos, y + 500, stars, ha='center', va='bottom')
    
    # Set y-axis limits to match our data
    plt.ylim(7000, 23000)
    
    # Add vertical line at 2015
    plt.axvline(x=4, color='red', linestyle='--', alpha=0.5)
    plt.text(4.1, 20000, '2015 Digital\nTransformation', rotation=90, color='red', alpha=0.5)
    
    plt.xticks(x, periods, rotation=45)
    plt.title('Wage Inequality Over Time by Industry Type', fontsize=14, pad=20)
    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('Wage Inequality (Mean - Median in USD)', fontsize=12)
    
    # Format y-axis with dollar signs and commas
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', frameon=True, framealpha=0.9)
    
    # Add significance note
    # plt.figtext(0.99, 0.01, '* p<0.05, *** p<0.001', ha='right', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('digital_transformation_actual.png', bbox_inches='tight', dpi=300)
    plt.close()

def print_regression_results(results):
    """
    Print key coefficients for verification
    """
    print("\nKey Coefficients for Plotting:")
    print("==============================")
    print("\nBase Effect (Non-Digital):")
    print(f"workers coefficient: {results.params['estab_weighted_sum']:.4f} (p={results.pvalues['estab_weighted_sum']:.4f})")
    
    print("\nDigital Industry Effects by Period:")
    for t in [-2, -1, 0, 1, 2, 3]:
        coef = results.params[f'digital_x_t{t}']
        pval = results.pvalues[f'digital_x_t{t}']
        print(f"t{t}: {coef:.2f} (p={pval:.4f})")
    
    print("\nSize x Digital Interactions by Period:")
    for t in [-2, -1, 0, 1, 2, 3]:
        coef = results.params[f'size_x_digital_x_t{t}']
        pval = results.pvalues[f'size_x_digital_x_t{t}']
        print(f"t{t}: {coef:.4f} (p={pval:.4f})")

def verify_event_study_data(results, state_industry):
    """
    Print comprehensive verification of data for event study plot
    """
    print("\nVerifying Event Study Data:")
    print("==========================")
    
    # 1. Basic regression coefficients
    print("\n1. Base Effects:")
    print(f"Worker effect: {results.params['estab_weighted_sum']:.4f}")
    
    # Only print industry trend if it exists
    if 'ind_trend' in results.params:
        print(f"Industry trend: {results.params['ind_trend']:.4f}")
    
    # 2. Digital effects by period
    print("\n2. Digital Effects by Period:")
    for t in [-2, -1, 0, 1, 2, 3]:
        if f'digital_x_t{t}' in results.params:
            base = results.params[f'digital_x_t{t}']
            size_int = results.params[f'size_x_digital_x_t{t}']
            print(f"\nPeriod t{t}:")
            print(f"Base digital effect: {base:.4f}")
            print(f"Size interaction: {size_int:.4f}")
    
    # 3. Sample statistics
    print("\n3. Sample Statistics:")
    print("\nMean workers by group:")
    print(state_industry.groupby('digital')['estab_weighted_sum'].mean())
    
    # 4. Calculated effects
    print("\n4. Total Effects at Mean Size:")
    mean_workers = state_industry['estab_weighted_sum'].mean()
    print(f"Mean establishment size: {mean_workers:.1f}")
    
    for t in [-2, -1, 0, 1, 2, 3]:
        if f'digital_x_t{t}' in results.params:
            base = results.params[f'digital_x_t{t}']
            size_int = results.params[f'size_x_digital_x_t{t}']
            total = base + (size_int * mean_workers)
            print(f"\nt{t} total effect at mean size: {total:.4f}")

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

def run_robust_model(df):
    """Run robust model with digital industry interactions and controls"""
    df = df.copy()
    
    # 1. Define digital industries
    digital_industries = [
        'Information',
        'Finance and Insurance',
        'Professional, Scientific, and Technical Services'
    ]
    
    # 2. Create necessary variables
    df['digital_industry'] = df['industry_name'].isin(digital_industries).astype(int)
    df['post2015'] = (df['YEAR'] >= 2015).astype(int)
    df['digital_x_post2015'] = df['digital_industry'] * df['post2015']
    df['digital_x_post2015_x_5G'] = df['digital_x_post2015'] * df['5G_ratio']
    
    # 3. Create event time indicators (-5 to +5 years around 2015)
    df['event_time'] = df['YEAR'] - 2015
    for tau in range(-5, 6):
        df[f'event_time_{tau}'] = (df['event_time'] == tau).astype(int)
    
    # 4. Set up panel data
    panel_data = df.set_index(['state_industry', 'YEAR'])
    
    # 5. Create model with all variables
    model = PanelOLS(
        dependent=panel_data['wage_gini'],
        exog=panel_data[[
            'estab_weighted_sum',
            'digital_industry',
            'post2015',
            'digital_x_post2015',
            'digital_x_post2015_x_5G',
            'pct_change_employment',
            'median_wage',
            'mean_wage',
            'min_wage_real_2020',
            'LTE_ratio',
            '5G_ratio'
        ] + [f'event_time_{tau}' for tau in range(-5, 6) if tau != 0]],  # Omit tau=0 as base
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
        check_rank=False
    )
    
    # 6. Fit model with clustered standard errors
    results = model.fit(cov_type='clustered', 
                       cluster_entity=True)
    
    # 7. Print detailed results
    print("\n=== Robust Model Results ===")
    print("\nMain Effects:")
    main_vars = ['estab_weighted_sum', 'digital_industry', 'post2015', 
                 'digital_x_post2015', 'digital_x_post2015_x_5G']
    for var in main_vars:
        if var in results.params:
            coef = results.params[var]
            se = results.std_errors[var]
            t_stat = results.tstats[var]
            p_val = results.pvalues[var]
            print(f"\n{var}:")
            print(f"Coefficient: {coef:.4f}")
            print(f"Std Error: {se:.4f}")
            print(f"T-statistic: {t_stat:.4f}")
            print(f"P-value: {p_val:.4f}")
    
    # 8. Print event study coefficients
    print("\nEvent Study Coefficients:")
    event_vars = [f'event_time_{tau}' for tau in range(-5, 6) if tau != 0]
    for var in event_vars:
        if var in results.params:
            print(f"{var}: {results.params[var]:.4f} (p={results.pvalues[var]:.4f})")
    
    return results


def run_section_five(df):
    """
    Section 5: Machine Learning Analysis and Robustness Tests
    """
    print("\n=== Section 5: ML Analysis and Robustness Tests ===")
    results = {}
    
    # 1. Original DDD Results (baseline for comparison)
    print("\n1. Baseline Digital Transformation Results")
    baseline_results, parallel_trends = analyze_digital_transformation(df)
    results['baseline'] = baseline_results
    results['parallel_trends'] = parallel_trends
    
    # 2. ML Analysis
    print("\n2. Machine Learning Analysis")
    ml_results = analyze_heterogeneity(df)
    results['ml'] = ml_results
    
    # 3. Mechanism Analysis
    print("\n3. Mechanism Analysis")
    mechanism_results = analyze_mechanisms(df)
    results['mechanisms'] = mechanism_results
    
    # 4. Placebo Tests
    print("\n4. Placebo Tests")
    placebo_results = run_placebo_tests(df)
    results['placebo'] = placebo_results
    
    # 5. Alternative Inequality Measures
    print("\n5. Alternative Inequality Measures")
    df = df.copy()
    
    # Calculate alternative inequality measures with safeguards
    # Mean/Median ratio
    df['wage_ratio'] = df['mean_wage'] / df['median_wage']
    
    # Top/Bottom ratio (with safeguards against division by zero)
    df['top_wage'] = df['annual_payroll_09: 500+'].replace(0, np.nan) / \
                     df['employment_09: 500+'].replace(0, np.nan)
    df['bottom_wage'] = df['annual_payroll_02: <5'].replace(0, np.nan) / \
                       df['employment_02: <5'].replace(0, np.nan)
    df['top_bottom_ratio'] = df['top_wage'] / df['bottom_wage']
    
    # Mid-size to small ratio (with safeguards)
    df['mid_wage'] = df['annual_payroll_06: 20-99'].replace(0, np.nan) / \
                     df['employment_06: 20-99'].replace(0, np.nan)
    df['small_wage'] = df['annual_payroll_02: <5'].replace(0, np.nan) / \
                      df['employment_02: <5'].replace(0, np.nan)
    df['mid_small_ratio'] = df['mid_wage'] / df['small_wage']
    
    # Remove infinite values and clean the ratios
    for col in ['wage_ratio', 'top_bottom_ratio', 'mid_small_ratio']:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Remove extreme outliers (more than 3 IQRs from median)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 3*IQR, upper=Q3 + 3*IQR)
    
    alt_measures = {}
    for measure in ['wage_ratio', 'top_bottom_ratio', 'mid_small_ratio']:
        # Drop NaN values for this specific measure
        valid_data = df.dropna(subset=[measure, 'estab_weighted_sum', 'digital', 
                                     'post_2015', 'size_x_digital_x_post2015'])
        
        panel_data = valid_data.set_index(['state_industry', 'YEAR'])
        model = PanelOLS(
            dependent=panel_data[measure],
            exog=sm.add_constant(panel_data[[
                'estab_weighted_sum',
                'digital',
                'post_2015',
                'size_x_digital_x_post2015'
            ]]),
            entity_effects=True,
            time_effects=True,
            drop_absorbed=True
        )
        alt_measures[measure] = model.fit(cov_type='clustered', cluster_entity=True)
    results['alternative_measures'] = alt_measures
    
    # 6. Different Standard Error Clustering
    print("\n6. Alternative Standard Error Clustering")
    clustering_results = {}
    panel_data = df.set_index(['state_industry', 'YEAR'])
    base_model = PanelOLS(
        dependent=panel_data['wage_gini'],
        exog=sm.add_constant(panel_data[[
            'estab_weighted_sum',
            'digital',
            'post_2015',
            'size_x_digital_x_post2015'
        ]]),
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True
    )
    
    clustering_results['entity'] = base_model.fit(cov_type='clustered', cluster_entity=True)
    clustering_results['time'] = base_model.fit(cov_type='clustered', cluster_time=True)
    clustering_results['two_way'] = base_model.fit(cov_type='clustered', 
                                                 cluster_entity=True,
                                                 cluster_time=True)
    results['clustering'] = clustering_results
    
    # 7. Outlier Exclusion
    print("\n7. Outlier Analysis")
    df_no_outliers = df.copy()
    
    # Remove size outliers
    size_q1 = df['estab_weighted_sum'].quantile(0.25)
    size_q3 = df['estab_weighted_sum'].quantile(0.75)
    size_iqr = size_q3 - size_q1
    size_upper = size_q3 + (1.5 * size_iqr)
    size_lower = size_q1 - (1.5 * size_iqr)
    df_no_outliers = df_no_outliers[
        (df_no_outliers['estab_weighted_sum'] >= size_lower) & 
        (df_no_outliers['estab_weighted_sum'] <= size_upper)
    ]
    
    # Remove inequality outliers
    gini_q1 = df['wage_gini'].quantile(0.25)
    gini_q3 = df['wage_gini'].quantile(0.75)
    gini_iqr = gini_q3 - gini_q1
    gini_upper = gini_q3 + (1.5 * gini_iqr)
    gini_lower = gini_q1 - (1.5 * gini_iqr)
    df_no_outliers = df_no_outliers[
        (df_no_outliers['wage_gini'] >= gini_lower) & 
        (df_no_outliers['wage_gini'] <= gini_upper)
    ]
    
    # Run main regression without outliers
    no_outliers_results, _ = analyze_digital_transformation(df_no_outliers)
    results['no_outliers'] = no_outliers_results
    
    # 8. Sensitivity Analysis
    print("\n8. Sensitivity Analysis")
    sensitivity_results = {}
    
    # Different fixed effects specifications
    panel_data = df.set_index(['state_industry', 'YEAR'])
    
    # No fixed effects
    model_no_fe = PanelOLS(
        dependent=panel_data['wage_gini'],
        exog=sm.add_constant(panel_data[[
            'estab_weighted_sum',
            'digital',
            'post_2015',
            'size_x_digital_x_post2015'
        ]]),
        drop_absorbed=True
    )
    sensitivity_results['no_fe'] = model_no_fe.fit(cov_type='clustered', cluster_entity=True)
    
    # Only entity fixed effects
    model_entity_fe = PanelOLS(
        dependent=panel_data['wage_gini'],
        exog=sm.add_constant(panel_data[[
            'estab_weighted_sum',
            'digital',
            'post_2015',
            'size_x_digital_x_post2015'
        ]]),
        entity_effects=True,
        drop_absorbed=True
    )
    sensitivity_results['entity_fe'] = model_entity_fe.fit(cov_type='clustered', cluster_entity=True)
    
    # Only time fixed effects
    model_time_fe = PanelOLS(
        dependent=panel_data['wage_gini'],
        exog=sm.add_constant(panel_data[[
            'estab_weighted_sum',
            'digital',
            'post_2015',
            'size_x_digital_x_post2015'
        ]]),
        time_effects=True,
        drop_absorbed=True
    )
    sensitivity_results['time_fe'] = model_time_fe.fit(cov_type='clustered', cluster_entity=True)
    
    # Additional controls
    model_controls = PanelOLS(
        dependent=panel_data['wage_gini'],
        exog=sm.add_constant(panel_data[[
            'estab_weighted_sum',
            'digital',
            'post_2015',
            'size_x_digital_x_post2015',
            'mean_wage',
            'median_wage',
            'workers'
        ]]),
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True
    )
    sensitivity_results['additional_controls'] = model_controls.fit(cov_type='clustered', cluster_entity=True)
    
    results['sensitivity'] = sensitivity_results
    
    return results

def print_section_five_results(results):
    """Print detailed results from section five analysis"""
    print("\n=== Detailed Section 5 Results ===")
    
    # 1. ML Results
    print("\n1. Machine Learning Feature Importance:")
    print(results['ml']['importance'].head(10))
    print(f"ML R-squared: {results['ml'].get('test_score', 'N/A')}")
    
    # 2. Mechanism Results
    print("\n2. Mechanism Analysis:")
    for mechanism, result in results['mechanisms'].items():
        print(f"\n{mechanism}:")
        print(result.summary.tables[1])
        print(f"R-squared: {result.rsquared:.4f}")
        print(f"Within R-squared: {result.rsquared_within:.4f}")
    
    # 3. Placebo Tests
    print("\n3. Placebo Test Results:")
    for year, result in results['placebo'].items():
        print(f"\nPlacebo Year {year}:")
        print(result.summary.tables[1])
        print(f"R-squared: {result.rsquared:.4f}")
        print(f"Within R-squared: {result.rsquared_within:.4f}")
    
    # 4. Alternative Measures
    print("\n4. Alternative Inequality Measures:")
    for measure, result in results['alternative_measures'].items():
        print(f"\n{measure}:")
        print(result.summary.tables[1])
        print(f"R-squared: {result.rsquared:.4f}")
        print(f"Within R-squared: {result.rsquared_within:.4f}")
    
    # 5. Clustering Results
    print("\n5. Alternative Standard Error Clustering:")
    for cluster_type, result in results['clustering'].items():
        print(f"\n{cluster_type} clustering:")
        print(result.summary.tables[1])
        print(f"R-squared: {result.rsquared:.4f}")
        print(f"Within R-squared: {result.rsquared_within:.4f}")
    
    # 6. Sensitivity Analysis
    print("\n6. Sensitivity Analysis:")
    for spec, result in results['sensitivity'].items():
        print(f"\n{spec}:")
        print(result.summary.tables[1])
        print(f"R-squared: {result.rsquared:.4f}")
        print(f"Within R-squared: {result.rsquared_within:.4f}")


def main():
    # Load and prepare initial data
    state_industry, industry_dummies = load_and_prepare_data()
    
    # Add region mapping
    regions = {
        'Northeast': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'],
        'Midwest': ['OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
        'South': ['DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA', 'OK', 'TX'],
        'West': ['MT', 'ID', 'WY', 'CO', 'NM', 'AZ', 'UT', 'NV', 'WA', 'OR', 'CA', 'AK', 'HI']
    }
    
    # Create state to region mapping using full state names
    state_name_to_abbrev = {
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
    
    # Create state to region mapping
    state_to_region = {}
    for region, abbrevs in regions.items():
        for state, abbrev in state_name_to_abbrev.items():
            if abbrev in abbrevs:
                state_to_region[state] = region
    
    # Add region to dataframe
    state_industry['region'] = state_industry['state_name'].map(state_to_region)
    
    # Run base regressions
    base_results = run_all_regressions(state_industry, industry_dummies)
    # print_results(base_results)
    
    # # Add extensions
    # state_industry = analyze_time_trends(state_industry)
    # # state_industry = analyze_geographic_variation(state_industry)
    # state_industry = run_robustness(state_industry)
    
    # # Add post-2015 industry analysis
    # state_industry = analyze_post2015_drivers(state_industry)
    
    # # Create panel data for analysis
    # panel_data = state_industry.set_index(['state_industry', 'YEAR'])
    
    # # Run post-2015 industry regression
    # model = PanelOLS(
    #     dependent=panel_data['wage_gini'],
    #     exog=sm.add_constant(panel_data[[
    #         'estab_weighted_sum',
    #         'post2015_x_Information',
    #         'post2015_x_Professional, Scientific, and Technical Services',
    #         'post2015_x_Finance and Insurance',
    #         'post2015_x_Retail Trade',
    #         'post2015_x_Transportation and Warehousing'
    #     ]]),
    #     entity_effects=True,
    #     time_effects=True,
    #     check_rank=False
    # )
    # post2015_results = model.fit(cov_type='clustered', cluster_entity=True)
    
    # print("\n=== Post-2015 Industry Analysis ===")
    # print(post2015_results.summary.tables[1])
    
    # # Run COVID analysis
    # covid_results = run_time_effects(panel_data)
    
    # Run and print extended analysis
    # extended_results = run_extended_regressions(state_industry)
    # print_extended_results(extended_results)
    
    # # Create summary statistics
    # summary_stats = create_summary_statistics(state_industry)
    # print(summary_stats)
    
    # # Add visualization
    # create_main_visualization(state_industry)
    
    # # Add non-linear analysis
    # print("\n=== Running Non-linear Analysis ===")
    # nonlinear_results = run_nonlinear_analysis(state_industry)
    # print_nonlinear_results(nonlinear_results)
    
    # Add DiD analysis
    # did_results = run_diff_in_diff(state_industry)
    # did_industry_results = run_INDUSTRY_diff_in_diff(state_industry)
    
    # # print(summary_stats)

    # # Add placebo tests
    # print("\n=== Running Placebo Tests ===")
    # placebo_results = run_placebo_tests(state_industry)
    # print("\nPlacebo Test Results:")
    # for year, result in placebo_results.items():
    #     print(f"\nPlacebo Year {year}:")
    #     print(result.summary.tables[1])
    
    # # Add industry characteristics analysis
    # print("\n=== Analyzing Industry Characteristics ===")
    # characteristics_results = analyze_industry_characteristics(state_industry)
    # print("\nIndustry Characteristics Results:")
    # print(characteristics_results.summary.tables[1])
    
    # # Add mechanism analysis
    # print("\n=== Analyzing Potential Mechanisms ===")
    # mechanism_results = analyze_mechanisms(state_industry)
    # print("\nMechanism Analysis Results:")
    # for mechanism, result in mechanism_results.items():
    #     print(f"\n{mechanism.replace('_', ' ').title()}:")
    #     print(result.summary.tables[1])
    
    # # # Add digital transformation analysis THIS IS THE MAIN DDDDD!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # print("\n=== Digital Transformation Analysis ===")
    # dig_results, trend_results = analyze_digital_transformation(state_industry)
    # print("\nTriple-Difference Results:")
    # print(dig_results.summary.tables[1])
    # print("\nParallel Trends Test:")
    # print(trend_results.summary.tables[1])
    
    # # Add visualization
    # plot_digital_transformation_effect(state_industry, dig_results)

    # print("\n=== Digital Transformation Analysis ROBUST===")
    # results = analyze_digital_transformation_robust(state_industry)
    # verify_event_study_data(results, state_industry)
    # print_regression_results(results)
    # print(results.summary.tables[1])
    # plot_event_study_results(state_industry)

    # actual_effects = calculate_actual_inequality(state_industry)
    
    # # Add ML heterogeneity analysis
    # print("\n=== ML Heterogeneity Analysis ===")
    # ml_results = analyze_heterogeneity(state_industry)
    # print("\nTop 10 Most Important Features:")
    # print(ml_results['importance'].head(10))
    # print("\nDigital Effect by Size Quartile:")
    # print(ml_results['interactions'])
    # print("\nModel R-squared:", ml_results['test_score'])
    
    # # Could also add visualization of ML results
    # plt.figure(figsize=(10, 6))
    # top_features = ml_results['importance'].head(10)
    # plt.barh(top_features['feature'], top_features['importance'])
    # plt.title('Top 10 Features Predicting Wage Inequality')
    # plt.xlabel('Feature Importance')
    # plt.tight_layout()
    # plt.savefig('feature_importance.png', bbox_inches='tight', dpi=300)
    # plt.close()
    
    # # Add controlled difference calculations and plots
    # controlled_diffs = calculate_controlled_differences(results, state_industry)
    # print("\nControlled Digital Premium by Year:")
    # print(controlled_diffs)
    
    # # Create both visualizations
    # plot_event_study_results(state_industry)  # Original raw differences
    # plot_controlled_event_study_results(state_industry, results)  # New controlled differences

    # # Run robust model
    # print("\nRunning robust model analysis...")
    # robust_results = run_robust_model(state_industry)

    # SECTION 5:
    section_five_results = run_section_five(state_industry)
    print_section_five_results(section_five_results)


if __name__ == "__main__":
    main()