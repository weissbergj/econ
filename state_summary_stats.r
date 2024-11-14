# Load required libraries
library(dplyr)
library(readr)
library(tidyr)

# Read the CSV files
industry_emp <- read_csv("output/state_industry_employment.csv", show_col_types = FALSE)
wages <- read_csv("output/state_wage_bill.csv", show_col_types = FALSE)
emp_changes <- read_csv("output/state_employment_changes.csv", show_col_types = FALSE)

# 1. Panel Summary by State-Industry
panel_summary <- wages %>%
  group_by(state_name, industry_name) %>%
  summarise(
    # Time coverage
    years_covered = n_distinct(YEAR),
    start_year = min(YEAR),
    end_year = max(YEAR),
    
    # Employment
    mean_employment = mean(total_workers),
    sd_employment = sd(total_workers),
    min_employment = min(total_workers),
    max_employment = max(total_workers),
    employment_growth = (last(total_workers) - first(total_workers)) / first(total_workers) * 100,
    employment_volatility = sd(total_workers) / mean(total_workers) * 100,
    
    # Wages
    mean_wage = mean(mean_wage),
    median_wage = mean(median_wage),
    sd_wage = sd(mean_wage),
    min_wage = min(mean_wage),
    max_wage = max(mean_wage),
    wage_growth = (last(mean_wage) - first(mean_wage)) / first(mean_wage) * 100,
    wage_volatility = sd(mean_wage) / mean(mean_wage) * 100,
    
    # Total wage bill
    mean_wage_bill = mean(total_wage_bill),
    wage_bill_growth = (last(total_wage_bill) - first(total_wage_bill)) / first(total_wage_bill) * 100,
    
    # Crisis periods
    emp_2009 = total_workers[YEAR == 2009],
    emp_2020 = total_workers[YEAR == 2020],
    emp_2022 = total_workers[YEAR == 2022],
    wage_2009 = mean_wage[YEAR == 2009],
    wage_2020 = mean_wage[YEAR == 2020],
    wage_2022 = mean_wage[YEAR == 2022],
    
    # Employment changes
    mean_annual_change = mean(emp_changes$pct_change[
      emp_changes$state_name == first(state_name) & 
      emp_changes$industry_name == first(industry_name)], na.rm = TRUE),
    sd_annual_change = sd(emp_changes$pct_change[
      emp_changes$state_name == first(state_name) & 
      emp_changes$industry_name == first(industry_name)], na.rm = TRUE),
    
    .groups = 'drop'
  ) %>%
  # Add state-level shares
  group_by(state_name) %>%
  mutate(
    emp_share_2022 = emp_2022 / sum(emp_2022, na.rm = TRUE) * 100,
    wage_share_2022 = (emp_2022 * wage_2022) / sum(emp_2022 * wage_2022, na.rm = TRUE) * 100,
    industry_rank_by_emp = rank(-emp_2022, ties.method = "min"),
    industry_rank_by_wage = rank(-wage_2022, ties.method = "min")
  ) %>%
  ungroup()

# Save single comprehensive CSV
write_csv(panel_summary, "output/state_summary_stats.csv")

# Print preview
cat("\n=== Summary Statistics Preview (Top 10 rows) ===\n")
print(head(panel_summary, 10))

cat("\nComprehensive summary statistics saved to 'comprehensive_summary_stats.csv'\n") 