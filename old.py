# import pandas as pd
# import numpy as np
# import statsmodels.api as sm
# from linearmodels.panel import PanelOLS

# # Read the data
# state_industry = pd.read_csv('cleaned_outputs/all_states_industry_analysis.csv', low_memory=False)

# # Ensure YEAR is numeric
# state_industry['YEAR'] = pd.to_numeric(state_industry['YEAR'])

# # Create industry dummies before setting index
# industry_dummies = pd.get_dummies(state_industry['industry_name'], prefix='ind', drop_first=True)

# # Print unique industries to verify
# print("\nUnique Industries:")
# print(state_industry['industry_name'].unique())

# # Create panel identifiers
# state_industry['state_industry'] = state_industry['state_name'] + "_" + state_industry['industry_name']

# def run_industry_wage_regression():
#     """Industry Wage Premium Analysis"""
#     # Get industry dummy columns
#     industry_cols = [col for col in industry_dummies.columns]  # Get all industry dummy columns
#     print(f"\nNumber of industry dummies: {len(industry_cols)}")
#     print("Industry dummy columns:", industry_cols[:5], "...")  # Print first 5 for verification
    
#     # Prepare exogenous variables with the same index
#     exog_data = pd.concat([industry_dummies, state_industry[['workers']]], axis=1)
#     exog_data = sm.add_constant(exog_data)
    
#     # Set the same index for both dependent and independent variables
#     panel_data = state_industry.set_index(['state_industry', 'YEAR'])
#     exog_data = exog_data.set_index([state_industry['state_industry'], state_industry['YEAR']])
    
#     model = PanelOLS(
#         dependent=panel_data['median_wage'],
#         exog=exog_data,
#         entity_effects=False,
#         time_effects=True
#     )
#     results = model.fit(cov_type='clustered', cluster_entity=True)
#     print("\nRegression: Industry Wage Premiums")
#     print(results.summary)
    
#     # Save and display results
#     params = pd.DataFrame({
#         'coefficient': results.params,
#         'std_error': results.std_errors,
#         't_stat': results.tstats,
#         'p_value': results.pvalues
#     })
    
#     # Drop the constant and workers terms
#     industry_effects = params.loc[params.index.str.startswith('ind_')].copy()
    
#     # Clean up industry names and sort by coefficient
#     industry_effects.index = industry_effects.index.str.replace('ind_', '')
#     industry_effects = industry_effects.sort_values('coefficient', ascending=False)
    
#     # Save full results
#     industry_effects.to_csv('industry_wage_premiums.csv')
    
#     print("\nTop 5 Highest-Paying Industries (relative to base industry):")
#     print(industry_effects.head())
#     print("\nBottom 5 Lowest-Paying Industries (relative to base industry):")
#     print(industry_effects.tail())
    
#     return results

# if __name__ == "__main__":
#     run_industry_wage_regression()


# import pandas as pd
# import numpy as np
# import statsmodels.api as sm
# from linearmodels.panel import PanelOLS

# # Read the data
# state_industry = pd.read_csv('cleaned_outputs/all_states_industry_analysis.csv', low_memory=False)

# # Create panel identifiers and set multi-index
# state_industry['state_industry'] = state_industry['state_name'] + "_" + state_industry['industry_name']
# state_industry = state_industry.set_index(['state_industry', 'YEAR'])

# # Calculate wage inequality measure (we have mean and median wages)
# state_industry['wage_inequality'] = state_industry['mean_wage'] - state_industry['median_wage']

# def run_regression():
#     """Relationship between Employment Size and Wage Inequality"""
#     exog_vars = sm.add_constant(state_industry[['workers']])
#     model = PanelOLS(
#         dependent=state_industry['wage_inequality'],
#         exog=exog_vars,
#         entity_effects=True,  # Control for state-industry fixed effects
#         time_effects=True     # Control for year fixed effects
#     )
#     results = model.fit(cov_type='clustered', cluster_entity=True)
#     print("\nRegression: Effect of Employment Size on Wage Inequality")
#     print(results.summary)
    
#     # Save results to CSV
#     params = pd.DataFrame({
#         'coefficient': results.params,
#         'std_error': results.std_errors,
#         't_stat': results.tstats,
#         'p_value': results.pvalues
#     })
#     params.to_csv('regression_results.csv')
#     return results

# if __name__ == "__main__":
#     run_regression()




# import pandas as pd
# import numpy as np
# import statsmodels.api as sm
# from linearmodels.panel import PanelOLS

# # Read the data with low_memory=False to handle mixed types
# labor_prod = pd.read_csv('cleaned_outputs/formatted_labor_productivity.csv', low_memory=False)
# state_industry = pd.read_csv('cleaned_outputs/all_states_industry_analysis.csv', low_memory=False)

# # Prepare national labor productivity data
# labor_prod = labor_prod[labor_prod['Measure'] == 'Labor productivity']
# labor_prod = labor_prod[['Year', 'Value']].rename(columns={'Value': 'nat_labor_prod'})

# # Merge national productivity data with state-industry data
# df = state_industry.merge(labor_prod, left_on='YEAR', right_on='Year', how='left')

# # Create panel identifiers and set multi-index
# df['state_industry'] = df['state_name'] + "_" + df['industry_name']
# df = df.set_index(['state_industry', 'YEAR'])

# # Calculate wage inequality measure
# df['wage_inequality'] = df['mean_wage'] - df['median_wage']
# df['wage_inequality_ratio'] = df['mean_wage'] / df['median_wage']

# # Calculate year-over-year growth rates
# df['median_wage_growth'] = df.groupby('state_industry')['median_wage'].pct_change()

# # Create dummy variables for industries and states
# industry_dummies = pd.get_dummies(df['industry_name'], prefix='ind')
# state_dummies = pd.get_dummies(df['state_name'], prefix='state')
# df = pd.concat([df, industry_dummies, state_dummies], axis=1)

# def run_regression_1():
#     """Effect of National Labor Productivity on State-Level Wages"""
#     exog_vars = sm.add_constant(df[['nat_labor_prod', 'workers']])
#     model = PanelOLS(
#         dependent=df['median_wage'],
#         exog=exog_vars,
#         entity_effects=True,
#         time_effects=True
#     )
#     results = model.fit(cov_type='clustered', cluster_entity=True)
#     print("\nRegression 1: National Labor Productivity Effect on Wages")
#     print(results.summary)
#     return results

# def run_regression_2():
#     """Impact of National Economic Output on State-Level Employment"""
#     exog_vars = sm.add_constant(df[['nat_labor_prod']])
#     model = PanelOLS(
#         dependent=df['pct_change_employment'],
#         exog=exog_vars,
#         entity_effects=True,
#         time_effects=True
#     )
#     results = model.fit(cov_type='clustered', cluster_entity=True)
#     print("\nRegression 2: National Output Effect on Employment")
#     print(results.summary)
#     return results

# def run_regression_3():
#     """Wage Inequality Analysis"""
#     exog_vars = sm.add_constant(df[['workers']])
#     model = PanelOLS(
#         dependent=df['wage_inequality'],
#         exog=exog_vars,
#         entity_effects=True,
#         time_effects=True
#     )
#     results = model.fit(cov_type='clustered', cluster_entity=True)
#     print("\nRegression 3: Wage Inequality Analysis")
#     print(results.summary)
#     return results

# def run_regression_4():
#     """Determinants of State-Level Employment Growth"""
#     exog_vars = sm.add_constant(df[['median_wage_growth', 'total_wage_bill']])
#     model = PanelOLS(
#         dependent=df['pct_change_employment'],
#         exog=exog_vars,
#         entity_effects=True,
#         time_effects=True
#     )
#     results = model.fit(cov_type='clustered', cluster_entity=True)
#     print("\nRegression 4: Employment Growth Determinants")
#     print(results.summary)
#     return results

# def run_regression_5():
#     """Effect of National Labor Productivity on Wage Inequality"""
#     exog_vars = sm.add_constant(df[['nat_labor_prod']])
#     model = PanelOLS(
#         dependent=df['wage_inequality'],
#         exog=exog_vars,
#         entity_effects=True,
#         time_effects=True
#     )
#     results = model.fit(cov_type='clustered', cluster_entity=True)
#     print("\nRegression 5: Labor Productivity Effect on Wage Inequality")
#     print(results.summary)
#     return results

# def run_regression_6():
#     """Industry Characteristics Effect on Wage Levels"""
#     # Select industry dummy variables (excluding one to avoid perfect multicollinearity)
#     industry_cols = [col for col in df.columns if col.startswith('ind_')][1:]
#     exog_vars = sm.add_constant(df[industry_cols])
    
#     model = PanelOLS(
#         dependent=df['median_wage'],
#         exog=exog_vars,
#         entity_effects=False,  # Already including industry dummies
#         time_effects=True
#     )
#     results = model.fit(cov_type='clustered', cluster_entity=True)
#     print("\nRegression 6: Industry Effects on Wages")
#     print(results.summary)
#     return results

# def main():
#     # Run all regressions and store results
#     results = {}
#     # results['reg1'] = run_regression_1() #no results
#     # results['reg2'] = run_regression_2() #no results
#     # results['reg3'] = run_regression_3() #p low
#     # results['reg4'] = run_regression_4() #broke
#     # results['reg5'] = run_regression_5()
#     # results['reg6'] = run_regression_6()
    
#     # Optional: Save results to CSV files
#     # for name, result in results.items():
#     #     params = pd.DataFrame({
#     #         'coefficient': result.params,
#     #         'std_error': result.std_errors,
#     #         't_stat': result.tstats,
#     #         'p_value': result.pvalues
#     #     })
#     #     params.to_csv(f'regression_results_{name}.csv')

# if __name__ == "__main__":
#     main()