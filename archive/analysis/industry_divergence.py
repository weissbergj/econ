import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

class IndustryDivergenceAnalysis:
    def __init__(self, data_path):
        """Initialize with full panel dataset"""
        self.df = pd.read_csv(data_path)
        # Filter to complete years
        self.df = self.df[self.df['YEAR'] <= 2021]
        self.setup_categories()
        self.create_metrics()
        
        if not os.path.exists('figures'):
            os.makedirs('figures')
            
    def setup_categories(self):
        """Define industry categories and time periods"""
        self.tech_industries = ['Information', 'Professional, Scientific, and Technical Services']
        self.service_industries = ['Retail Trade', 'Accommodation and Food Services']
        self.manufacturing_industries = ['Manufacturing', 'Transportation and Warehousing']
        
        self.periods = {
            'Pre-Crisis': (2007, 2009),
            'Recovery': (2010, 2015),
            'Tech Boom': (2016, 2019),
            'COVID': (2020, 2021)
        }
        
    def create_metrics(self):
        """Create all analysis metrics with data cleaning"""
        # Clean denominators
        self.df['employment_01: Total'] = self.df['employment_01: Total'].replace(0, np.nan)
        
        # Create metrics with safety checks
        self.df['large_firm_share'] = (
            self.df['employment_09: 500+'] / self.df['employment_01: Total']
        ).fillna(0)
        
        self.df['avg_wage'] = (
            self.df['annual_payroll_01: Total'] / self.df['employment_01: Total']
        ).fillna(0)
        
        # Industry categories
        self.df['is_tech'] = self.df['industry_name'].isin(self.tech_industries)
        self.df['is_service'] = self.df['industry_name'].isin(self.service_industries)
        
        # State categories
        self.df['high_wage_state'] = (
            self.df.groupby('state_name')['avg_wage']
            .transform('mean') > self.df['avg_wage'].mean()
        )
    
    def analyze_trends(self):
        """Analyze key trends by period"""
        trends = []
        
        for period_name, (start_year, end_year) in self.periods.items():
            period_data = self.df[
                (self.df['YEAR'] >= start_year) & 
                (self.df['YEAR'] <= end_year)
            ]
            
            # Calculate period averages
            period_stats = period_data.groupby('industry_name').agg({
                'large_firm_share': 'mean',
                'avg_wage': 'mean'
            }).round(3)
            
            trends.append({
                'period': period_name,
                'stats': period_stats
            })
            
            print(f"\n{period_name} ({start_year}-{end_year}):")
            print("Top 5 Industries by Concentration:")
            print(period_stats.sort_values('large_firm_share', ascending=False).head())
        
        return trends
    
    def create_figures(self):
        """Generate key figures for paper"""
        print("\nGenerating figures...")
        
        try:
            # Figure 1: Industry Evolution
            plt.figure(figsize=(12, 6))
            for ind in self.tech_industries + self.service_industries:
                ind_data = self.df[self.df['industry_name'] == ind]
                yearly_avg = ind_data.groupby('YEAR')['large_firm_share'].mean()
                plt.plot(yearly_avg.index, yearly_avg.values, label=ind)
            
            plt.title('Industry Concentration Evolution (2007-2021)')
            plt.xlabel('Year')
            plt.ylabel('Large Firm Share')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig('figures/industry_evolution.png')
            plt.close()
            
            # Figure 2: State Patterns
            plt.figure(figsize=(12, 6))
            state_data = self.df.groupby(['YEAR', 'high_wage_state'])['large_firm_share'].mean().unstack()
            state_data.plot(title='Concentration by State Wage Level')
            plt.xlabel('Year')
            plt.ylabel('Large Firm Share')
            plt.legend(['Low Wage States', 'High Wage States'])
            plt.tight_layout()
            plt.savefig('figures/state_patterns.png')
            plt.close()
            
            print("Figures saved successfully!")
            
        except Exception as e:
            print(f"Error generating figures: {str(e)}")
    
    def analyze_complex_patterns(self):
        """Analyze deeper patterns and interactions"""
        
        # 1. Size Distribution Changes
        self.df['small_firm_share'] = self.df['employment_02: <5'] / self.df['employment_01: Total']
        self.df['mid_firm_share'] = (
            (self.df['employment_06: 20-99'] + self.df['employment_07: 100-499']) / 
            self.df['employment_01: Total']
        )
        
        # 2. Wage Inequality
        self.df['wage_gap'] = (
            self.df['annual_payroll_09: 500+'] / self.df['employment_09: 500+'] -
            self.df['annual_payroll_02: <5'] / self.df['employment_02: <5']
        )
        
        # 3. Industry Transitions
        transitions = []
        for year in range(2007, 2021):
            year_data = self.df[self.df['YEAR'] == year]
            next_year = self.df[self.df['YEAR'] == year + 1]
            
            # Track changes in firm size distribution
            for industry in year_data['industry_name'].unique():
                curr = year_data[year_data['industry_name'] == industry]
                future = next_year[next_year['industry_name'] == industry]
                
                if not curr.empty and not future.empty:
                    small_change = future['small_firm_share'].values[0] - curr['small_firm_share'].values[0]
                    large_change = future['large_firm_share'].values[0] - curr['large_firm_share'].values[0]
                    
                    transitions.append({
                        'year': year,
                        'industry': industry,
                        'small_change': small_change,
                        'large_change': large_change
                    })
        
        transitions_df = pd.DataFrame(transitions)
        
        # 4. Industry Groupings by Pattern
        patterns = self.df.groupby('industry_name').agg({
            'small_firm_share': ['mean', 'std'],
            'large_firm_share': ['mean', 'std'],
            'wage_gap': ['mean', 'std'],
            'avg_wage': ['mean', 'std']
        })
        
        # 5. Find interesting patterns
        patterns['volatility'] = patterns[('large_firm_share', 'std')] / patterns[('large_firm_share', 'mean')]
        patterns['wage_inequality'] = patterns[('wage_gap', 'mean')] / patterns[('avg_wage', 'mean')]
        
        # Print key findings
        print("\nIndustry Patterns:")
        print("\n1. Most Volatile Industries (Size Distribution Changes):")
        print(patterns.sort_values('volatility', ascending=False).head())
        
        print("\n2. Largest Wage Gaps:")
        print(patterns.sort_values('wage_inequality', ascending=False).head())
        
        print("\n3. Interesting Transitions:")
        interesting = transitions_df[
            (abs(transitions_df['small_change']) > transitions_df['small_change'].std()) |
            (abs(transitions_df['large_change']) > transitions_df['large_change'].std())
        ]
        print(interesting.sort_values('large_change', ascending=False).head())
        
        return patterns, transitions_df
    
    def analyze_employment_dynamics(self):
        """Analyze employment shifts and wage patterns"""
        
        # 1. Create size categories with CORRECT column names
        size_categories = {
            'tiny': 'employment_02: <5',
            'small': 'employment_03: 5-9',
            'small_med': 'employment_04: 10-19',
            'medium': 'employment_05: 20-49',  # This was wrong
            'med_large': 'employment_06: 50-99',  # This was wrong
            'large': 'employment_07: 100-499',
            'very_large': 'employment_09: 500+'
        }
        
        # Let's verify columns first
        print("\nAvailable employment columns:")
        emp_cols = [col for col in self.df.columns if 'employment' in col]
        print(emp_cols)
        
        # 2. Calculate employment share changes
        base_year = 2007
        end_years = [2009, 2015, 2019, 2021]
        
        changes = []
        for end_year in end_years:
            base = self.df[self.df['YEAR'] == base_year]
            current = self.df[self.df['YEAR'] == end_year]
            
            for industry in self.df['industry_name'].unique():
                base_ind = base[base['industry_name'] == industry]
                curr_ind = current[current['industry_name'] == industry]
                
                if not base_ind.empty and not curr_ind.empty:
                    # Use only columns that exist in the data
                    for size, col in size_categories.items():
                        if col in self.df.columns:  # Check if column exists
                            try:
                                base_share = base_ind[col].values[0] / base_ind['employment_01: Total'].values[0]
                                curr_share = curr_ind[col].values[0] / curr_ind['employment_01: Total'].values[0]
                                
                                changes.append({
                                    'period_end': end_year,
                                    'industry': industry,
                                    'size_category': size,
                                    'share_change': curr_share - base_share,
                                    'base_share': base_share,
                                    'end_share': curr_share
                                })
                            except Exception as e:
                                print(f"Error processing {industry} for {size}: {str(e)}")
        
        changes_df = pd.DataFrame(changes)
        
        if not changes_df.empty:
            # 3. Find interesting patterns
            print("\nMost Dramatic Employment Shifts:")
            dramatic = changes_df[abs(changes_df['share_change']) > changes_df['share_change'].std() * 2]
            print(dramatic.sort_values('share_change', ascending=False).head(10))
            
            # 4. Industry Employment Evolution
            print("\nIndustry Employment Evolution:")
            for industry in self.df['industry_name'].unique():
                ind_data = self.df[self.df['industry_name'] == industry]
                try:
                    total_emp_change = (
                        ind_data[ind_data['YEAR'] == 2021]['employment_01: Total'].values[0] /
                        ind_data[ind_data['YEAR'] == 2007]['employment_01: Total'].values[0] - 1
                    )
                    print(f"{industry}: {total_emp_change:.1%}")
                except Exception as e:
                    print(f"Error calculating change for {industry}: {str(e)}")
        else:
            print("No valid employment changes calculated")
        
        return changes_df
    
    def analyze_tech_transformation(self):
        """Analyze how technology adoption relates to industry structure"""
        
        # 1. Create tech adoption measures
        self.df['tech_period'] = pd.cut(self.df['YEAR'], 
                                       bins=[2006, 2015, 2018, 2021],
                                       labels=['Pre-5G', 'Early-5G', 'Post-5G'])
        
        # 2. Calculate tech intensity
        self.df['tech_intensity'] = (
            self.df['5G_ratio'] + self.df['LTE_ratio']
        ) / 2
        
        # 3. Industry-level analysis
        industry_tech = self.df.groupby(['industry_name', 'tech_period']).agg({
            'tech_intensity': 'mean',
            'avg_wage': 'mean',
            'large_firm_share': 'mean',
            'employment_01: Total': 'sum'
        }).reset_index()
        
        # 4. Calculate changes
        for metric in ['tech_intensity', 'avg_wage', 'large_firm_share']:
            industry_tech[f'{metric}_change'] = industry_tech.groupby('industry_name')[metric].diff()
        
        # 5. Find interesting patterns
        print("\nTechnology Adoption and Industry Changes:")
        
        # High-tech adopters
        print("\nTop Tech Adopters and Their Changes:")
        high_tech = industry_tech[
            industry_tech['tech_period'] == 'Post-5G'
        ].sort_values('tech_intensity', ascending=False)
        print(high_tech[['industry_name', 'tech_intensity', 'avg_wage_change', 'large_firm_share_change']].head())
        
        # Wage growth leaders
        print("\nTop Wage Growth Industries:")
        wage_growth = industry_tech[
            industry_tech['tech_period'] == 'Post-5G'
        ].sort_values('avg_wage_change', ascending=False)
        print(wage_growth[['industry_name', 'tech_intensity', 'avg_wage_change']].head())
        
        # Structure changes
        print("\nBiggest Structure Changes:")
        structure_change = industry_tech[
            industry_tech['tech_period'] == 'Post-5G'
        ].sort_values('large_firm_share_change', ascending=False)
        print(structure_change[['industry_name', 'tech_intensity', 'large_firm_share_change']].head())
        
        # 6. Correlation analysis
        correlations = industry_tech[
            industry_tech['tech_period'] == 'Post-5G'
        ][['tech_intensity', 'avg_wage_change', 'large_firm_share_change']].corr()
        
        print("\nCorrelations between Tech and Changes:")
        print(correlations['tech_intensity'])
        
        return industry_tech
    
    def analyze_industry_mobility(self):
        """Analyze industry rank changes and mobility patterns"""
        
        # 1. Create rankings for each year
        rankings = []
        metrics = ['large_firm_share', 'avg_wage']
        
        for year in sorted(self.df['YEAR'].unique()):
            year_data = self.df[self.df['YEAR'] == year]
            
            for metric in metrics:
                # Get rankings
                ranked = year_data.sort_values(metric, ascending=False)
                
                # Store top 5 for each year
                for rank, row in enumerate(ranked.itertuples(), 1):
                    if rank <= 5:
                        rankings.append({
                            'year': year,
                            'industry': row.industry_name,
                            'metric': metric,
                            'rank': rank,
                            'value': getattr(row, metric)
                        })
        
        rankings_df = pd.DataFrame(rankings)
        
        # 2. Analyze mobility
        print("\nIndustry Mobility Analysis:")
        
        # Track industries that entered/exited top 5
        for metric in metrics:
            metric_rankings = rankings_df[rankings_df['metric'] == metric]
            
            print(f"\n{metric} Changes:")
            
            # Early years (2007-2009)
            early = set(metric_rankings[
                metric_rankings['year'].isin(range(2007, 2010))
            ]['industry'].unique())
            
            # Recent years (2019-2021)
            recent = set(metric_rankings[
                metric_rankings['year'].isin(range(2019, 2022))
            ]['industry'].unique())
            
            # Find changes
            entered = recent - early
            exited = early - recent
            stayed = early & recent
            
            print("\nEntered Top 5:")
            print(entered)
            print("\nExited Top 5:")
            print(exited)
            print("\nStayed in Top 5:")
            print(stayed)
            
            # Calculate stability score
            stability = len(stayed) / len(early)
            print(f"\nStability Score: {stability:.2f}")
        
        return rankings_df
    
    def run_statistical_tests(self):
        """Run comprehensive statistical tests on key findings"""
        from scipy import stats
        from statsmodels.stats.multitest import multipletests
        import statsmodels.api as sm
        import numpy as np
        
        print("\n=== STATISTICAL TESTS ===")
        
        # Store all p-values for multiple testing correction
        all_pvals = []
        
        # A. UTILITIES TRANSFORMATION TESTS
        print("\nA. UTILITIES TRANSFORMATION TESTS")
        utilities = self.df[self.df['industry_name'] == 'Utilities']
        other = self.df[self.df['industry_name'] != 'Utilities']
        
        # Employment growth test (fixed)
        u_emp = utilities.groupby('YEAR')['employment_01: Total'].mean()
        o_emp = other.groupby('YEAR')['employment_01: Total'].mean()
        u_growth = u_emp.pct_change(fill_method=None)
        o_growth = o_emp.pct_change(fill_method=None)
        
        t_stat, p_val = stats.ttest_ind(u_growth.dropna(), o_growth.dropna())
        print(f"Employment Growth t-test: t={t_stat:.3f}, p={p_val:.3f}")
        all_pvals.append(p_val)
        
        # B. INDUSTRY DIVERGENCE TESTS (fixed)
        print("\nB. INDUSTRY DIVERGENCE TESTS")
        
        # Calculate growth rates for each industry
        industry_growths = {}
        for ind in self.df['industry_name'].unique():
            ind_data = self.df[self.df['industry_name'] == ind]
            yearly_emp = ind_data.groupby('YEAR')['employment_01: Total'].mean()
            growth = yearly_emp.pct_change(fill_method=None).dropna()
            if len(growth) > 0:  # Only include if we have data
                industry_growths[ind] = growth.values
        
        # ANOVA on growth rates
        growth_data = [v for v in industry_growths.values() if len(v) > 0]
        if len(growth_data) > 1:  # Make sure we have enough data
            f_stat, p_val = stats.f_oneway(*growth_data)
            print(f"Industry Growth ANOVA: F={f_stat:.3f}, p={p_val:.3f}")
            all_pvals.append(p_val)
        
        # C. WAGE INEQUALITY TESTS (new)
        print("\nC. WAGE INEQUALITY TESTS")
        
        # Test if wage gaps are increasing over time
        early_years = self.df[self.df['YEAR'] <= 2015]['avg_wage']
        late_years = self.df[self.df['YEAR'] > 2015]['avg_wage']
        t_stat, p_val = stats.ttest_ind(early_years, late_years)
        print(f"Wage Gap t-test: t={t_stat:.3f}, p={p_val:.3f}")
        all_pvals.append(p_val)
        
        # D. CONCENTRATION DYNAMICS (new)
        print("\nD. CONCENTRATION DYNAMICS TESTS")
        
        # Test for trend in concentration
        years = self.df['YEAR'].values
        conc = self.df['large_firm_share'].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, conc)
        print(f"Concentration Trend: slope={slope:.4f}, p={p_value:.3f}, R²={r_value**2:.3f}")
        all_pvals.append(p_value)
        
        # E. MULTIPLE TESTING CORRECTION (fixed)
        print("\nE. MULTIPLE TESTING CORRECTION")
        
        # Apply multiple testing corrections
        rejected, p_corrected, alphacSidak, alphacBonf = multipletests(
            all_pvals, 
            alpha=0.05, 
            method='bonferroni'
        )
        
        print("\nMultiple Testing Summary:")
        print("Test              Original p    Corrected p    Significant")
        print("-" * 60)
        test_names = [
            "Employment Growth",
            "Industry ANOVA",
            "Wage Gap",
            "Concentration Trend"
        ]
        
        for test, orig_p, corr_p, rej in zip(test_names, all_pvals, p_corrected, rejected):
            print(f"{test:<20} {orig_p:8.3f}     {corr_p:8.3f}     {'Yes' if rej else 'No'}")
        
        # F. ROBUSTNESS CHECKS (new)
        print("\nF. ROBUSTNESS CHECKS")
        
        # Non-parametric tests
        stat, p_val = stats.kruskal(*growth_data)
        print(f"Kruskal-Wallis test: H={stat:.3f}, p={p_val:.3f}")
        
        # Test for normality of growth rates
        stat, p_val = stats.normaltest(np.concatenate(growth_data))
        print(f"Normality test: stat={stat:.3f}, p={p_val:.3f}")
        
        return {
            'utilities_tests': {'t_stat': t_stat, 'p_val': p_val},
            'industry_tests': {'f_stat': f_stat, 'p_val': p_val},
            'wage_tests': {'t_stat': t_stat, 'p_val': p_val},
            'concentration_tests': {'slope': slope, 'p_val': p_value, 'r_squared': r_value**2},
            'multiple_testing': {'rejected': rejected, 'corrected_pvals': p_corrected}
        }

def main():
    print("Loading data and initializing analysis...")
    
    try:
        # Initialize analysis
        analysis = IndustryDivergenceAnalysis('../../archive/cleaned_outputs/merged_state_industry_broadband.csv')
        
        # 1. Basic Trend Analysis
        print("\n=== BASIC TREND ANALYSIS ===")
        trends = analysis.analyze_trends()
        
        # 2. Complex Pattern Analysis
        print("\n=== COMPLEX PATTERN ANALYSIS ===")
        patterns, transitions = analysis.analyze_complex_patterns()
        
        # 3. Employment Dynamics
        print("\n=== EMPLOYMENT DYNAMICS ===")
        employment_changes = analysis.analyze_employment_dynamics()
        
        # 4. Technology Transformation
        print("\n=== TECHNOLOGY TRANSFORMATION ===")
        industry_tech = analysis.analyze_tech_transformation()
        
        # 5. Industry Mobility
        print("\n=== INDUSTRY MOBILITY ===")
        rankings = analysis.analyze_industry_mobility()
        
        # 6. Generate Figures
        print("\n=== GENERATING FIGURES ===")
        analysis.create_figures()
        
        # 7. Run Statistical Tests
        print("\n=== STATISTICAL TESTS ===")
        analysis.run_statistical_tests()
        
        print("\nAnalysis complete! Check the figures directory for visualizations.")
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 