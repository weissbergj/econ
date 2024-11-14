import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from scipy import stats
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import YearLocator

def verify_data_quality(df):
    """Verify data quality and print diagnostic information"""
    print("\n=== Data Quality Checks ===")
    
    # 1. Basic Dataset Info
    print("\nDataset Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    
    # 2. Missing Values
    print("\nMissing Values:")
    print(df.isnull().sum().sum())
    
    # 3. Key Variables Summary
    print("\nKey Variables Summary:")
    for var in ['workers', 'median_wage', 'mean_wage', 'total_wage_bill']:
        print(f"\n{var}:")
        print(df[var].describe())
    
    # 4. Industry Coverage
    print("\nIndustries per State:")
    print(df.groupby('state_name')['industry_name'].nunique())
    
    # 5. Time Series Completeness
    print("\nYears per State-Industry:")
    years_per_pair = df.groupby(['state_name', 'industry_name'])['YEAR'].count()
    print(years_per_pair.value_counts().sort_index())
    
    # 6. Check for Suspicious Values
    print("\nSuspicious Values Check:")
    print("Negative wages:", (df['median_wage'] <= 0).sum())
    print("Negative workers:", (df['workers'] <= 0).sum())
    print("Mean < Median wage:", (df['mean_wage'] < df['median_wage']).sum())
    
    # 7. Check for Outliers
    print("\nPotential Outliers (workers):")
    q1 = df['workers'].quantile(0.25)
    q3 = df['workers'].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df['workers'] < (q1 - 1.5 * iqr)) | (df['workers'] > (q3 + 1.5 * iqr))]
    print(f"Number of outliers: {len(outliers)}")
    
    return True

def clean_data(df):
    """Clean data and return a copy"""
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Remove zero wages
    df = df[df['median_wage'] > 0]
    
    # Flag impossible wage relationships
    df.loc[:, 'wage_relationship_flag'] = df['mean_wage'] < df['median_wage']
    
    # Balance panel (optional)
    min_years = 10  # or another threshold
    valid_pairs = df.groupby('state_industry')['YEAR'].count() >= min_years
    df = df[df['state_industry'].isin(valid_pairs[valid_pairs].index)]
    
    return df

def load_and_prepare_data():
    """Load and prepare data with verification"""
    # Read the data
    state_industry = pd.read_csv('cleaned_outputs/all_states_industry_analysis.csv')
    
    # Basic cleaning
    state_industry['YEAR'] = pd.to_numeric(state_industry['YEAR'])
    state_industry['state_industry'] = state_industry['state_name'] + "_" + state_industry['industry_name']
    
    # Clean data first
    state_industry = clean_data(state_industry)
    
    # Calculate inequality measures
    state_industry['wage_inequality'] = state_industry['mean_wage'] - state_industry['median_wage']
    state_industry['wage_inequality_ratio'] = state_industry['mean_wage'] / state_industry['median_wage']
    
    # Create industry dummies after cleaning
    industry_dummies = pd.get_dummies(state_industry['industry_name'], prefix='ind', drop_first=True)
    
    # Verify data quality
    verify_data_quality(state_industry)
    
    return state_industry, industry_dummies

def run_all_regressions(state_industry, industry_dummies):
    """Run all base regressions with rank check disabled"""
    results = {}
    panel_data = state_industry.set_index(['state_industry', 'YEAR'])
    
    # 1a. Size effect on median wages
    model = PanelOLS(
        dependent=panel_data['median_wage'],
        exog=sm.add_constant(panel_data[['workers']]),
        entity_effects=True,
        time_effects=True,
        check_rank=False
    )
    results['size_wage'] = model.fit(cov_type='clustered', cluster_entity=True)
    
    # 1b. Size effect on wage inequality
    model = PanelOLS(
        dependent=panel_data['wage_inequality'],
        exog=sm.add_constant(panel_data[['workers']]),
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
            state_industry['workers'] * 
            (state_industry['industry_name'] == ind)
        )
    
    interaction_data = sm.add_constant(
        state_industry[[f'size_x_{ind}' for ind in major_industries] + ['workers']]
    )
    interaction_data = interaction_data.set_index(panel_data.index)
    
    model = PanelOLS(
        dependent=panel_data['wage_inequality'],
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
    df['size_x_post2008'] = df['workers'] * df['post_2008']
    
    # Pre/Post COVID
    df['post_covid'] = df['YEAR'] >= 2020
    df['size_x_postcovid'] = df['workers'] * df['post_covid']
    
    # Time trend
    df['time_trend'] = df['YEAR'] - df['YEAR'].min()
    df['size_x_trend'] = df['workers'] * df['time_trend']
    
    return df

def analyze_geographic_variation(df):
    """Analyze geographic patterns"""
    # Define regions
    regions = {
        'Northeast': ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA'],
        'Midwest': ['OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
        'South': ['DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA', 'OK', 'TX'],
        'West': ['MT', 'ID', 'WY', 'CO', 'NM', 'AZ', 'UT', 'NV', 'WA', 'OR', 'CA', 'AK', 'HI']
    }
    
    # Create region mapping
    state_to_region = {}
    for region, states in regions.items():
        for state in states:
            state_to_region[state] = region
    
    # Add region
    df = df.copy()
    df['region'] = df['state_name'].map(state_to_region)
    
    # Create region interactions
    for region in regions.keys():
        df[f'size_x_{region}'] = df['workers'] * (df['region'] == region)
    
    return df

def run_robustness(df):
    """Run robustness checks"""
    df = df.copy()
    
    # Non-linear size effects
    df['workers_squared'] = df['workers'] ** 2
    df['workers_cubed'] = df['workers'] ** 3
    
    # Alternative inequality measures
    df['log_inequality'] = np.log(df['wage_inequality'].clip(lower=1))
    df['relative_inequality'] = df['wage_inequality'] / df['median_wage']
    
    # Size categories
    df['size_category'] = pd.qcut(df['workers'], q=5, labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'])
    
    return df

def test_regional_differences(results):
    """Test if differences between regional coefficients are statistically significant"""
    print("\n=== Tests of Regional Differences ===")
    
    regions = ['Midwest', 'Northeast', 'West', 'South']
    
    # Get coefficients and standard errors
    coef_data = {
        region: {
            'coef': results['regional'][region].params['workers'],
            'se': results['regional'][region].std_errors['workers']
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
    
    # Debug state names
    # print("\nDebugging State Names:")
    # print("Unique state names in data:", df['state_name'].unique())
    
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
    
    # Debug regions
    # print("\nDebugging Regions:")
    # print("Number of states with assigned regions:", df['region'].notna().sum())
    # print("Unique regions:", df['region'].unique())
    
    # Create panel data
    panel_data = df.set_index(['state_industry', 'YEAR'])
    
    # 1. Base Model
    model = PanelOLS(
        dependent=panel_data['wage_inequality'],
        exog=sm.add_constant(panel_data[['workers']]),
        entity_effects=True,
        time_effects=True,
        check_rank=False
    )
    results['base_model'] = model.fit(cov_type='clustered', cluster_entity=True)
    
    # 2. Time periods analysis
    periods_data = panel_data.copy()
    periods_data['post_2015'] = (periods_data.index.get_level_values('YEAR') >= 2015).astype(float)
    periods_data['size_post2015'] = periods_data['workers'] * periods_data['post_2015']
    
    model = PanelOLS(
        dependent=periods_data['wage_inequality'],
        exog=sm.add_constant(periods_data[['workers', 'size_post2015']]),
        entity_effects=True,
        time_effects=False,
        check_rank=False
    )
    results['time_effects'] = model.fit(cov_type='clustered', cluster_entity=True)
    
    # 3. Non-linear effects
    nonlinear_data = panel_data.copy()
    nonlinear_data['workers_squared'] = nonlinear_data['workers'] ** 2
    
    model = PanelOLS(
        dependent=nonlinear_data['wage_inequality'],
        exog=sm.add_constant(nonlinear_data[['workers', 'workers_squared']]),
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
                    dependent=region_data['wage_inequality'],
                    exog=sm.add_constant(region_data[['workers']]),
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
    
    # 5. Alternative measure
    model = PanelOLS(
        dependent=panel_data['relative_inequality'],
        exog=sm.add_constant(panel_data[['workers']]),
        entity_effects=True,
        time_effects=True,
        check_rank=False
    )
    results['alternative_measure'] = model.fit(cov_type='clustered', cluster_entity=True)
    
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
    
    print("\n5. Alternative Measure:")
    print(results['alternative_measure'].summary.tables[1])

def run_time_effects(panel_data):
    """Run specific analysis of COVID impact on size-inequality relationship"""
    print("\n=== COVID Period Analysis ===")
    
    # Create COVID period indicator
    panel_data = panel_data.copy()
    panel_data['post_covid'] = (panel_data.index.get_level_values('YEAR') >= 2020).astype(float)
    panel_data['size_x_covid'] = panel_data['workers'] * panel_data['post_covid']
    
    # Run regression with COVID interaction
    model = PanelOLS(
        dependent=panel_data['wage_inequality'],
        exog=sm.add_constant(panel_data[['workers', 'size_x_covid']]),
        entity_effects=True,
        time_effects=True,
        check_rank=False
    )
    results = model.fit(cov_type='clustered', cluster_entity=True)
    
    print("\nEffect of establishment size on wage inequality during COVID:")
    print(results.summary.tables[1])
    
    # Calculate total effect during COVID
    covid_total_effect = results.params['workers'] + results.params['size_x_covid']
    covid_se = (results.std_errors['workers']**2 + results.std_errors['size_x_covid']**2)**0.5
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
    key_vars = ['workers', 'median_wage', 'mean_wage', 'wage_inequality', 'wage_inequality_ratio']
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
        is_ratio = col == 'wage_inequality_ratio'
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
    missing_by_col = df.isnull().sum()
    total_missing = missing_by_col.sum()
    rows_with_missing = df.isnull().any(axis=1).sum()
    
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
            len(identify_outliers(df['workers'])),
            (df['median_wage'] <= 0).sum()
        ]
    })
    quality_stats['Percentage'] = (quality_stats['Count'] / len(df) * 100).round(2).astype(str) + '%'

    # 5. Regional Statistics
    df['region'] = df['state_name'].map(create_region_mapping())
    states_per_region = df.groupby('region')['state_name'].nunique()
    
    regional_stats = df.groupby('region').agg({
        'state_industry': ['count', lambda x: f"{len(x)/len(df)*100:.1f}%"],
        'state_name': 'nunique',
        'industry_name': 'nunique',
        'workers': ['mean', 'median']
    })
    regional_stats.columns = ['Observations', 'Share', 'States', 'Industries', 'Mean Size', 'Median Size']
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
            'Mean Wage Inequality': format_number(period_data['wage_inequality'].mean()),
            'Median Wage Inequality': format_number(period_data['wage_inequality'].median())
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
    df['log_workers'] = np.log(df['workers'])
    df['size_category'] = pd.qcut(df['log_workers'], q=3, 
                                 labels=['Small (<250)', 'Medium (250-1000)', 'Large (>1000)'])
    
    yearly_stats = df.groupby(['YEAR', 'size_category'])['wage_inequality'].mean().reset_index()
    yearly_stats = yearly_stats.pivot(index='YEAR', columns='size_category', values='wage_inequality')
    
    # Calculate the size-inequality gap
    yearly_stats['Large-Small Gap'] = yearly_stats['Large (>1000)'] - yearly_stats['Small (<250)']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
    
    colors = {'Small (<250)': '#2ecc71', 
             'Medium (250-1000)': '#3498db', 
             'Large (>1000)': '#e74c3c'}
    
    # Top panel: Original lines
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
    ax2.set_ylabel('Large-Small Gap', fontsize=12)
    
    # Customize appearance
    ax1.set_title('Wage Inequality by Establishment Size', 
                fontsize=14, pad=20, fontweight='bold')
    ax2.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Wage Inequality (Mean - Median)', fontsize=12)
    
    # Legend only in top panel
    ax1.legend(title='Establishment Size', title_fontsize=12, 
             loc='upper left', bbox_to_anchor=(0.02, 0.98))
    
    plt.tight_layout()
    plt.savefig('wage_inequality_trends.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Load and prepare initial data
    state_industry, industry_dummies = load_and_prepare_data()
    
    # Run base regressions
    base_results = run_all_regressions(state_industry, industry_dummies)
    print_results(base_results)
    
    # Add extensions
    state_industry = analyze_time_trends(state_industry)
    state_industry = analyze_geographic_variation(state_industry)
    state_industry = run_robustness(state_industry)
    
    # Add post-2015 industry analysis
    state_industry = analyze_post2015_drivers(state_industry)
    
    # Create panel data for analysis
    panel_data = state_industry.set_index(['state_industry', 'YEAR'])
    
    # Run post-2015 industry regression
    model = PanelOLS(
        dependent=panel_data['wage_inequality'],
        exog=sm.add_constant(panel_data[[
            'workers',
            'post2015_x_Information',
            'post2015_x_Professional, Scientific, and Technical Services',
            'post2015_x_Finance and Insurance',
            'post2015_x_Retail Trade',
            'post2015_x_Transportation and Warehousing'
        ]]),
        entity_effects=True,
        time_effects=True,
        check_rank=False
    )
    post2015_results = model.fit(cov_type='clustered', cluster_entity=True)
    
    print("\n=== Post-2015 Industry Analysis ===")
    print(post2015_results.summary.tables[1])
    
    # Run COVID analysis
    covid_results = run_time_effects(panel_data)
    
    # Run and print extended analysis
    extended_results = run_extended_regressions(state_industry)
    print_extended_results(extended_results)
    
    # Create summary statistics
    summary_stats = create_summary_statistics(state_industry)
    
    # Add visualization
    create_main_visualization(state_industry)
    
    print(summary_stats)

if __name__ == "__main__":
    main()