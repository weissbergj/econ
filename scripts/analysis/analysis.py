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
    state_industry = pd.read_csv('../../data/merged_final.csv')
    
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
    # verify_data_quality(state_industry)
    
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

def run_nonlinear_analysis(df):
    """Run comprehensive non-linear analysis of size effects"""
    results = {}
    panel_data = df.set_index(['state_industry', 'YEAR'])
    
    # 1. Cubic Model
    panel_data['workers_squared'] = panel_data['workers'] ** 2
    panel_data['workers_cubed'] = panel_data['workers'] ** 3
    
    model = PanelOLS(
        dependent=panel_data['wage_inequality'],
        exog=sm.add_constant(panel_data[['workers', 'workers_squared', 'workers_cubed']]),
        entity_effects=True,
        time_effects=True,
        check_rank=False,
        drop_absorbed=True
    )
    results['cubic'] = model.fit(cov_type='clustered', cluster_entity=True)
    
    # 2. Log Specification
    panel_data['log_workers'] = np.log(panel_data['workers'])
    model = PanelOLS(
        dependent=panel_data['wage_inequality'],
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
            dependent=ind_data['wage_inequality'],
            exog=sm.add_constant(ind_data[['workers', 'workers_squared']]),
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
            dependent=period_data['wage_inequality'],
            exog=sm.add_constant(period_data[['workers', 'workers_squared']]),
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
            if 'workers' in res.params:
                b1 = res.params['workers']
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
            if 'workers' in res.params:
                b1 = res.params['workers']
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
            dependent=panel_data['wage_inequality'],
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
            dependent=panel_data['wage_inequality'],
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

def examine_industries(df):
    """Print all industries and their sample sizes"""
    industry_counts = df['industry_name'].value_counts()
    print("\nAll Industries and Observations:")
    print(industry_counts)
    
    # Basic stats
    print(f"\nTotal Industries: {len(industry_counts)}")
    print("\nMean observations per industry:", industry_counts.mean())
    print("Min observations:", industry_counts.min())
    print("Max observations:", industry_counts.max())

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
                new_cols[f'size_x_{industry}_x_{year}'] = new_cols[f'{industry}_x_{year}'] * df['workers']
    
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
            dependent=panel_data['wage_inequality'],
            exog=sm.add_constant(panel_data[['workers'] + year_interactions]),
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
        
        # Plot results
        # plot_event_study(results[industry], industry, base_year)
    
    return results

def plot_event_study(result, industry, base_year):
    """Create event study plot for an industry"""
    coef = result.params[2:]  # Skip constant and workers
    se = result.std_errors[2:]
    years = [int(col.split('_')[-1]) for col in result.params.index[2:]]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(years, coef, yerr=1.96*se, fmt='o', capsize=5)
    plt.axvline(x=base_year, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=2015, color='g', linestyle='--', alpha=0.5)
    plt.axvline(x=2020, color='b', linestyle='--', alpha=0.5)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.title(f'Event Study: {industry}')
    plt.xlabel('Year')
    plt.ylabel('Effect on Wage Inequality')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'event_study_{industry.replace(" ", "_")}.png')
    plt.close()

def run_placebo_tests(df):
    """Run placebo tests using random treatment years"""
    placebo_years = range(2010, 2014)
    results = {}
    
    for year in placebo_years:
        # Create placebo treatment
        df[f'post_{year}'] = (df['YEAR'] >= year).astype(int)
        df[f'size_x_post_{year}'] = df['workers'] * df[f'post_{year}']
        
        # Run base regression with placebo
        panel_data = df.set_index(['state_industry', 'YEAR'])
        model = PanelOLS(
            dependent=panel_data['wage_inequality'],
            exog=sm.add_constant(panel_data[[
                'workers', 
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
    df['workers_x_digital_intensity'] = df['workers'] * df['digital_intensity']
    df['post_2015'] = (df['YEAR'] >= 2015).astype(int)
    df['post2015_x_digital_intensity'] = df['post_2015'] * df['digital_intensity']
    
    # Run interaction analysis
    panel_data = df.set_index(['state_industry', 'YEAR'])
    model = PanelOLS(
        dependent=panel_data['wage_inequality'],
        exog=sm.add_constant(panel_data[[
            'workers',
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
    df['organizational_complexity'] = (df['workers'] / df.groupby('YEAR')['workers'].transform('mean')).fillna(0)
    
    # Create interactions
    mechanisms = ['automation_exposure', 'skill_intensity', 'organizational_complexity']
    for mechanism in mechanisms:
        df[f'workers_x_{mechanism}'] = df['workers'] * df[mechanism]
    
    # Create panel data
    panel_data = df.set_index(['state_industry', 'YEAR'])
    
    results = {}
    for mechanism in mechanisms:
        model = PanelOLS(
            dependent=panel_data['wage_inequality'],
            exog=sm.add_constant(panel_data[[
                'workers',
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
            df[f'size_x_digital_x_t{t}'] = df['workers'] * df[f'digital_x_t{t}']
            interactions.extend([f'digital_x_t{t}', f'size_x_digital_x_t{t}'])
    
    # Run regression
    panel_data = df.set_index(['state_industry', 'YEAR'])
    model = PanelOLS(
        dependent=panel_data['wage_inequality'],
        exog=sm.add_constant(panel_data[['workers', 'ind_trend'] + interactions]),
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
    df['size_x_digital'] = df['workers'] * df['digital']
    df['size_x_post2015'] = df['workers'] * df['post_2015']
    df['digital_x_post2015'] = df['digital'] * df['post_2015']
    df['size_x_digital_x_post2015'] = df['workers'] * df['digital'] * df['post_2015']
    
    # Run triple-diff regression
    panel_data = df.set_index(['state_industry', 'YEAR'])
    model = PanelOLS(
        dependent=panel_data['wage_inequality'],
        exog=sm.add_constant(panel_data[[
            'workers',
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
        pre_2015[f'size_x_digital_x_t{year}'] = pre_2015['workers'] * pre_2015[f'digital_x_t{year}']
    
    # Run parallel trends regression
    panel_pre = pre_2015.set_index(['state_industry', 'YEAR'])
    trend_vars = ([f'size_x_digital_x_t{t}' for t in range(-5, 0)] + 
                 [f'digital_x_t{t}' for t in range(-5, 0)])
    
    trend_model = PanelOLS(
        dependent=panel_pre['wage_inequality'],
        exog=sm.add_constant(panel_pre[['workers'] + trend_vars]),
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
    yearly_effects = df.groupby(['YEAR', 'digital'])['wage_inequality'].mean().reset_index()
    
    # Plot two lines: digital and non-digital firms
    for is_digital in [0, 1]:
        data = yearly_effects[yearly_effects['digital'] == is_digital]
        label = 'Digital Industries' if is_digital else 'Non-Digital Industries'
        color = 'red' if is_digital else 'blue'
        
        plt.plot(data['YEAR'], data['wage_inequality'], 
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
    digital_effects = df[df['digital'] == 1].groupby('period', observed=True)['wage_inequality'].mean()
    non_digital_effects = df[df['digital'] == 0].groupby('period', observed=True)['wage_inequality'].mean()
    
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
    print(f"workers coefficient: {results.params['workers']:.4f} (p={results.pvalues['workers']:.4f})")
    
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
    print(f"Worker effect: {results.params['workers']:.4f}")
    print(f"Industry trend: {results.params['ind_trend']:.4f}")
    
    # 2. Digital effects by period
    print("\n2. Digital Effects by Period:")
    for t in [-2, -1, 0, 1, 2, 3]:
        base = results.params[f'digital_x_t{t}']
        size_int = results.params[f'size_x_digital_x_t{t}']
        print(f"\nPeriod t{t}:")
        print(f"Base digital effect: {base:.2f}")
        print(f"Size interaction: {size_int:.4f}")
    
    # 3. Sample statistics
    print("\n3. Sample Statistics:")
    print("\nMean workers by group:")
    print(state_industry.groupby('digital')['workers'].mean())
    
    # 4. Calculated effects
    print("\n4. Total Effects at Mean Size:")
    mean_workers = state_industry['workers'].mean()
    print(f"Mean establishment size: {mean_workers:.1f}")
    
    for t in [-2, -1, 0, 1, 2, 3]:
        base = results.params[f'digital_x_t{t}']
        size_int = results.params[f'size_x_digital_x_t{t}']
        total = base + (size_int * mean_workers)
        print(f"\nt{t} total effect at mean size: {total:.2f}")

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
                         'wage_inequality': ['mean', 'std', 'count'],
                         'workers': 'mean'
                     })
                     .round(2))
    
    print("\nActual Wage Inequality by Period:")
    print("=================================")
    print("\nDigital Industries:")
    digital_data = df[df['digital'] == 1].groupby('period')['wage_inequality'].agg(['mean', 'std', 'count'])
    print(digital_data.round(2))
    
    print("\nNon-Digital Industries:")
    non_digital_data = df[df['digital'] == 0].groupby('period')['wage_inequality'].agg(['mean', 'std', 'count'])
    print(non_digital_data.round(2))
    
    return actual_effects

def verify_raw_data(df):
    """Verify raw data calculations"""
    print("\nRaw Data Verification:")
    print("=====================")
    
    # 1. Check wage inequality calculation
    print("\n1. Wage Inequality Formula:")
    print("mean_wage - median_wage values for first few rows:")
    print(df[['mean_wage', 'median_wage', 'wage_inequality']].head())
    
    # 2. Check specific industries
    print("\n2. Sample from each group:")
    for is_digital in [0, 1]:
        print(f"\n{'Digital' if is_digital else 'Non-Digital'} Industry Sample:")
        sample = df[df['digital'] == is_digital].groupby('industry_name').first()
        print(sample[['mean_wage', 'median_wage', 'wage_inequality']].head())
    
    # 3. Count by industry
    print("\n3. Observations by Industry:")
    print(df.groupby('industry_name').size().sort_values(ascending=False))

def verify_time_coverage(df):
    """Verify data coverage by year"""
    print("\nData Coverage by Year:")
    print("=====================")
    print("\nObservations per year:")
    year_counts = df.groupby('YEAR').size()
    print(year_counts.sort_index())
    
    print("\nWage inequality by year and digital status:")
    yearly_stats = df.groupby(['YEAR', 'digital'])['wage_inequality'].agg(['mean', 'count', 'std'])
    print(yearly_stats.round(2))

def analyze_heterogeneity(df):
    """Use ML to identify patterns in digital transformation effects"""
    
    # Create richer features
    df['size_quartile'] = pd.qcut(df['workers'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    df['post_2015'] = (df['YEAR'] >= 2015).astype(int)
    df['wage_level'] = pd.qcut(df['mean_wage'], q=4, labels=['Low', 'Mid-Low', 'Mid-High', 'High'])
    
    # Create interaction features
    df['size_x_digital'] = df['workers'] * df['digital']
    df['size_x_post2015'] = df['workers'] * df['post_2015']
    
    # Prepare features for ML
    features = pd.get_dummies(df[[
        'digital', 'size_quartile', 'state', 'post_2015',
        'wage_level'
    ]])
    
    # Add numeric features
    scaler = StandardScaler()
    numeric_features = ['workers', 'size_x_digital', 'size_x_post2015']
    for col in numeric_features:
        features[f'{col}_scaled'] = scaler.fit_transform(df[[col]])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, 
        df['wage_inequality'],
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
        dig_effect = subset[subset['digital'] == 1]['wage_inequality'].mean() - \
                    subset[subset['digital'] == 0]['wage_inequality'].mean()
        interactions.append({
            'type': 'size',
            'group': size_q, 
            'digital_effect': dig_effect
        })
    
    # Wage level effects
    for wage_level in ['Low', 'Mid-Low', 'Mid-High', 'High']:
        subset = df[df['wage_level'] == wage_level]
        dig_effect = subset[subset['digital'] == 1]['wage_inequality'].mean() - \
                    subset[subset['digital'] == 0]['wage_inequality'].mean()
        interactions.append({
            'type': 'wage',
            'group': wage_level, 
            'digital_effect': dig_effect
        })
    
    # Time period effects
    for year in df['YEAR'].unique():
        subset = df[df['YEAR'] == year]
        dig_effect = subset[subset['digital'] == 1]['wage_inequality'].mean() - \
                    subset[subset['digital'] == 0]['wage_inequality'].mean()
        interactions.append({
            'type': 'year',
            'group': str(year), 
            'digital_effect': dig_effect
        })
    
    # # Create visualizations
    # plt.figure(figsize=(15, 10))
    
    # # Plot 1: Size effects
    # plt.subplot(2, 2, 1)
    # size_effects = pd.DataFrame([i for i in interactions if i['type'] == 'size'])
    # plt.bar(size_effects['group'], size_effects['digital_effect'])
    # plt.title('Digital Effect by Firm Size')
    # plt.ylabel('Wage Inequality Increase ($)')
    
    # # Plot 2: Wage level effects
    # plt.subplot(2, 2, 2)
    # wage_effects = pd.DataFrame([i for i in interactions if i['type'] == 'wage'])
    # plt.bar(wage_effects['group'], wage_effects['digital_effect'])
    # plt.title('Digital Effect by Wage Level')
    
    # # Plot 3: Time trends
    # plt.subplot(2, 2, 3)
    # time_effects = pd.DataFrame([i for i in interactions if i['type'] == 'year'])
    # plt.plot(time_effects['group'], time_effects['digital_effect'], marker='o')
    # plt.title('Digital Effect Over Time')
    # plt.xticks(rotation=45)
    
    # # Plot 4: Top 10 features
    # plt.subplot(2, 2, 4)
    # top_features = importance.head(10)
    # plt.barh(top_features['feature'], top_features['importance'])
    # plt.title('Top 10 Predictive Features')
    
    # plt.tight_layout()
    # plt.savefig('heterogeneity_analysis.png', bbox_inches='tight', dpi=300)
    # plt.close()
    
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
    mean_workers = df['workers'].mean()
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
    mean_workers = df['workers'].mean()
    
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































def load_and_prepare_data_establishment():
    """Load and prepare data from merged_state_industry_data.csv with verification"""
    # Read the data
    df = pd.read_csv('../../data/merged_final.csv')
    
    # Basic cleaning
    df['YEAR'] = pd.to_numeric(df['YEAR'])
    df['state_industry'] = df['state_name'] + "_" + df['industry_name']
    
    # Calculate wage inequality measures
    df['wage_inequality'] = df['mean_wage'] - df['median_wage']
    df['wage_inequality_ratio'] = df['mean_wage'] / df['median_wage']
    
    # Calculate establishment size measures
    df['large_estab_share'] = df['establishments_09: 500+'] / df['establishments_01: Total']
    
    # Print verification info
    print("\nData Verification:")
    print("=================")
    print(f"Years covered: {df['YEAR'].min()} to {df['YEAR'].max()}")
    print(f"Number of states: {df['state_name'].nunique()}")
    print(f"Number of industries: {df['industry_name'].nunique()}")
    print("\nWage statistics:")
    print(f"Median wage range: ${df['median_wage'].min():,.0f} to ${df['median_wage'].max():,.0f}")
    print(f"Mean wage range: ${df['mean_wage'].min():,.0f} to ${df['mean_wage'].max():,.0f}")
    print("\nEstablishment size statistics:")
    print("Average share of large establishments (500+):")
    print(df.groupby('industry_name')['large_estab_share'].mean().sort_values(ascending=False).head())
    
    # Create industry dummies
    industry_dummies = pd.get_dummies(df['industry_name'], prefix='ind', drop_first=True)
    
    return df, industry_dummies


def run_triple_diff(df):
    """Run robust triple difference regression with event study specification"""
    df = df.copy()
    
    # Define digital industries
    digital_industries = [
        'Information',
        'Professional, Scientific, and Technical Services',
        'Finance and Insurance'
    ]
    df['digital'] = df['industry_name'].isin(digital_industries).astype(int)
    
    # Create two-year bins (UPDATED to end at 2021)
    df['period'] = pd.cut(df['YEAR'], 
                         bins=[2007, 2009, 2011, 2013, 2015, 2017, 2019, 2021],  # Changed 2022 to 2021
                         labels=['2008-09', '2010-11', '2012-13', '2015', '2016-17', '2018-19', '2020-21'])  # Updated label
    
    # Create event-time dummies (with explicit handling of NaN)
    period_map = {
        '2008-09': -3, '2010-11': -2, '2012-13': -1, 
        '2015': 0, '2016-17': 1, '2018-19': 2, '2020-21': 3  # Updated label
    }
    df['event_time'] = df['period'].map(period_map)
    
    # Add industry trends
    df['ind_trend'] = df.groupby('industry_name').cumcount()
    
    # Calculate establishment size measure
    df['large_estab_intensity'] = (
        df['establishments_09: 500+'] / 
        df['establishments_01: Total']
    )
    
    # Create interactions
    interactions = []
    for t in df['event_time'].unique():
        if t != -3:  # Use 2008-09 as reference
            df[f'digital_x_t{t}'] = (df['event_time'] == t) * df['digital']
            df[f'size_x_digital_x_t{t}'] = df['large_estab_intensity'] * df[f'digital_x_t{t}']
            interactions.extend([f'digital_x_t{t}', f'size_x_digital_x_t{t}'])
    
    # Run regression
    panel_data = df.set_index(['state_industry', 'YEAR'])
    model = PanelOLS(
        dependent=panel_data['wage_inequality'],
        exog=sm.add_constant(panel_data[['large_estab_intensity', 'ind_trend'] + interactions]),
        entity_effects=True,
        time_effects=True,
        check_rank=False,
        drop_absorbed=True
    )
    
    results = model.fit(cov_type='clustered', cluster_entity=True)
    print(results)
    
    return results

def analyze_heterogeneity_establishment(df):
    """Analyze heterogeneous effects of digital transformation using ML"""
    df = df.copy()
    
    # Calculate establishment size measure first
    df['large_estab_intensity'] = (
        df['establishments_09: 500+'] / 
        df['establishments_01: Total']
    ).fillna(0)
    
    # Define digital industries (if not already defined)
    digital_industries = [
        'Information',
        'Professional, Scientific, and Technical Services',
        'Finance and Insurance'
    ]
    df['digital'] = df['industry_name'].isin(digital_industries).astype(int)
    
    # Drop rows with NaN in wage_inequality (target variable)
    df = df.dropna(subset=['wage_inequality'])
    
    # Create key features
    df['size_quartile'] = pd.qcut(df['large_estab_intensity'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    df['post_2015'] = (df['YEAR'] >= 2015).astype(int)
    df['wage_level'] = pd.qcut(df['mean_wage'], q=4, labels=['Low', 'Mid-Low', 'Mid-High', 'High'])
    
    # Create interaction features
    df['size_x_digital'] = df['large_estab_intensity'] * df['digital']
    df['size_x_post2015'] = df['large_estab_intensity'] * df['post_2015']
    
    # Prepare features for ML
    features = pd.get_dummies(df[[
        'digital', 'size_quartile', 'state_name', 'post_2015',
        'wage_level'
    ]])
    
    # Add numeric features
    scaler = StandardScaler()
    numeric_features = ['large_estab_intensity', 'size_x_digital', 'size_x_post2015']
    for col in numeric_features:
        features[f'{col}_scaled'] = scaler.fit_transform(df[[col]])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, 
        df['wage_inequality'],
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
    
    # Print key findings
    print("\nHeterogeneity Analysis Results:")
    print("==============================")
    
    # 1. Top 10 most important features
    print("\nTop 10 Most Important Features:")
    print(importance.head(10))
    
    # 2. Size quartile effects
    print("\nDigital Effect by Size Quartile:")
    for size_q in ['Q1', 'Q2', 'Q3', 'Q4']:
        subset = df[df['size_quartile'] == size_q]
        dig_effect = subset[subset['digital'] == 1]['wage_inequality'].mean() - \
                    subset[subset['digital'] == 0]['wage_inequality'].mean()
        print(f"{size_q}: ${dig_effect:,.2f}")
    
    # 3. Wage level effects
    print("\nDigital Effect by Wage Level:")
    for wage_level in ['Low', 'Mid-Low', 'Mid-High', 'High']:
        subset = df[df['wage_level'] == wage_level]
        dig_effect = subset[subset['digital'] == 1]['wage_inequality'].mean() - \
                    subset[subset['digital'] == 0]['wage_inequality'].mean()
        print(f"{wage_level}: ${dig_effect:,.2f}")
    
    # 4. Model performance
    print("\nModel Performance:")
    print(f"R-squared (test set): {rf.score(X_test, y_test):.4f}")
    
    return {
        'importance': importance,
        'test_score': rf.score(X_test, y_test)
    }


# def main():
#     # Load and prepare initial data
#     state_industry, industry_dummies = load_and_prepare_data()
    
#     # Run base regressions
#     base_results = run_all_regressions(state_industry, industry_dummies)
#     print_results(base_results)
    
#     # Add extensions
#     state_industry = analyze_time_trends(state_industry)
#     state_industry = analyze_geographic_variation(state_industry)
#     state_industry = run_robustness(state_industry)
    
#     # Add post-2015 industry analysis
#     state_industry = analyze_post2015_drivers(state_industry)
    
#     # Create panel data for analysis
#     panel_data = state_industry.set_index(['state_industry', 'YEAR'])
    
#     # Run post-2015 industry regression
#     model = PanelOLS(
#         dependent=panel_data['wage_inequality'],
#         exog=sm.add_constant(panel_data[[
#             'workers',
#             'post2015_x_Information',
#             'post2015_x_Professional, Scientific, and Technical Services',
#             'post2015_x_Finance and Insurance',
#             'post2015_x_Retail Trade',
#             'post2015_x_Transportation and Warehousing'
#         ]]),
#         entity_effects=True,
#         time_effects=True,
#         check_rank=False
#     )
#     post2015_results = model.fit(cov_type='clustered', cluster_entity=True)
    
#     print("\n=== Post-2015 Industry Analysis ===")
#     print(post2015_results.summary.tables[1])
    
#     # Run COVID analysis
#     covid_results = run_time_effects(panel_data)
    
#     # Run and print extended analysis
#     extended_results = run_extended_regressions(state_industry)
#     print_extended_results(extended_results)
    
#     # # Create summary statistics
#     summary_stats = create_summary_statistics(state_industry)
#     print(summary_stats)
    
#     # # Add visualization
#     # create_main_visualization(state_industry)
    
#     # Add non-linear analysis
#     print("\n=== Running Non-linear Analysis ===")
#     nonlinear_results = run_nonlinear_analysis(state_industry)
#     print_nonlinear_results(nonlinear_results)
    
#     # Add DiD analysis
#     did_results = run_diff_in_diff(state_industry)
    
#     # Examine industries
#     print("\nExamining all industries:")
#     examine_industries(state_industry)
#     print("\n=== Running Event Study ===")
#     event_study = run_event_study(state_industry)
    
#     print(summary_stats)

#     # Add placebo tests
#     print("\n=== Running Placebo Tests ===")
#     placebo_results = run_placebo_tests(state_industry)
#     print("\nPlacebo Test Results:")
#     for year, result in placebo_results.items():
#         print(f"\nPlacebo Year {year}:")
#         print(result.summary.tables[1])
    
#     # Add industry characteristics analysis
#     print("\n=== Analyzing Industry Characteristics ===")
#     characteristics_results = analyze_industry_characteristics(state_industry)
#     print("\nIndustry Characteristics Results:")
#     print(characteristics_results.summary.tables[1])
    
#     # Add mechanism analysis
#     print("\n=== Analyzing Potential Mechanisms ===")
#     mechanism_results = analyze_mechanisms(state_industry)
#     print("\nMechanism Analysis Results:")
#     for mechanism, result in mechanism_results.items():
#         print(f"\n{mechanism.replace('_', ' ').title()}:")
#         print(result.summary.tables[1])
    
#     # Add digital transformation analysis
#     print("\n=== Digital Transformation Analysis ===")
#     dig_results, trend_results = analyze_digital_transformation(state_industry)
#     print("\nTriple-Difference Results:")
#     print(dig_results.summary.tables[1])
#     print("\nParallel Trends Test:")
#     print(trend_results.summary.tables[1])
    
#     # Add visualization
#     # plot_digital_transformation_effect(state_industry, dig_results)

#     results = analyze_digital_transformation_robust(state_industry)
#     verify_event_study_data(results, state_industry)
#     print_regression_results(results)
#     # plot_event_study_results(state_industry)

#     # actual_effects = calculate_actual_inequality(state_industry)

#     # Add raw data verification
#     # verify_raw_data(state_industry)
#     # verify_time_coverage(state_industry)
    
#     # Add ML heterogeneity analysis
#     # print("\n=== ML Heterogeneity Analysis ===")
#     # ml_results = analyze_heterogeneity(state_industry)
#     # print("\nTop 10 Most Important Features:")
#     # print(ml_results['importance'].head(10))
#     # print("\nDigital Effect by Size Quartile:")
#     # print(ml_results['interactions'])
#     # print("\nModel R-squared:", ml_results['test_score'])
    
#     # Could also add visualization of ML results
#     # plt.figure(figsize=(10, 6))
#     # top_features = ml_results['importance'].head(10)
#     # plt.barh(top_features['feature'], top_features['importance'])
#     # plt.title('Top 10 Features Predicting Wage Inequality')
#     # plt.xlabel('Feature Importance')
#     # plt.tight_layout()
#     # plt.savefig('feature_importance.png', bbox_inches='tight', dpi=300)
#     # plt.close()
    
#     # Add controlled difference calculations and plots
#     # controlled_diffs = calculate_controlled_differences(results, state_industry)
#     # print("\nControlled Digital Premium by Year:")
#     # print(controlled_diffs)
    
#     # # Create both visualizations
#     # plot_event_study_results(state_industry)  # Original raw differences
#     # plot_controlled_event_study_results(state_industry, results)  # New controlled differences



#     state_industry, industry_dummies = load_and_prepare_data_establishment()
#     triple_diff_results = run_triple_diff(state_industry)

#     heterogeneity_results_establishment = analyze_heterogeneity_establishment(state_industry)



# if __name__ == "__main__":
#     main()




def main():
    # 1. Load and prepare initial data
    state_industry, industry_dummies = load_and_prepare_data()
    
    # 2. Run robustness checks to create additional variables
    state_industry = run_robustness(state_industry)
    
    # 3. Base Establishment Size Analysis
    print("\n=== Base Establishment Size Analysis ===")
    panel_data = state_industry.set_index(['state_industry', 'YEAR'])
    
    # Base model with mean-median ratio
    model_mw = PanelOLS(
        dependent=panel_data['wage_inequality_ratio'],
        exog=sm.add_constant(panel_data[['workers']]),
        entity_effects=True,
        time_effects=True
    )
    results_mw = model_mw.fit(cov_type='clustered', cluster_entity=True)
    print("\nMean-Median Ratio Results:")
    print(results_mw.summary.tables[1])
    
    # Base model with wage inequality
    model_wi = PanelOLS(
        dependent=panel_data['wage_inequality'],
        exog=sm.add_constant(panel_data[['workers']]),
        entity_effects=True,
        time_effects=True
    )
    results_wi = model_wi.fit(cov_type='clustered', cluster_entity=True)
    print("\nWage Inequality Results:")
    print(results_wi.summary.tables[1])
    
    # 3. Industry Analysis
    print("\n=== Industry Analysis ===")
    industry_results = run_all_regressions(state_industry, industry_dummies)
    print_results(industry_results)
    
    # 4. Geographic Analysis
    print("\n=== Geographic Analysis ===")
    state_industry = analyze_geographic_variation(state_industry)
    
    # 5. Time Trend Analysis
    print("\n=== Time Trend Analysis ===")
    state_industry = analyze_time_trends(state_industry)
    
    # 6. Non-Linear Effects
    print("\n=== Non-Linear Analysis ===")
    nonlinear_results = run_nonlinear_analysis(state_industry)
    print_nonlinear_results(nonlinear_results)
    
    # 7. Mechanism Validation
    print("\n=== Mechanism Analysis ===")
    mechanism_results = analyze_mechanisms(state_industry)
    print("\nMechanism Analysis Results:")
    for mechanism, result in mechanism_results.items():
        print(f"\n{mechanism.replace('_', ' ').title()}:")
        print(result.summary.tables[1])
    
    # 8. Digital Transformation Analysis
    print("\n=== Digital Transformation Analysis ===")
    dig_results, trend_results = analyze_digital_transformation(state_industry)
    print("\nTriple-Difference Results:")
    print(dig_results.summary.tables[1])
    print("\nParallel Trends Test:")
    print(trend_results.summary.tables[1])
    
    # 9. Extended Analysis
    print("\n=== Extended Analysis ===")
    extended_results = run_extended_regressions(state_industry)
    print_extended_results(extended_results)

    print('\n=== Triple Dif ===')
    run_triple_diff(state_industry)

if __name__ == "__main__":
    main()