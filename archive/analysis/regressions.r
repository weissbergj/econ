# Load required libraries
suppressMessages({
  if (!require(rmarkdown)) install.packages("rmarkdown")
  library(dplyr)
  library(broom)
  library(fixest)
  library(modelsummary)
  library(ggplot2)
  library(scales)
  library(readr)
})

# Read and process data
wages <- read_csv("output/all_states_wage_bill.csv", show_col_types = FALSE)
emp_changes <- read_csv("output/all_states_employment_changes.csv", show_col_types = FALSE)
productivity <- read_csv("output/state_productivity.csv", show_col_types = FALSE)
value_added <- read_csv("output/state_value_added.csv", show_col_types = FALSE)

# Add after reading value_added data (around line 15):
# print("Value added data check:")
# print("Number of rows in value_added:")
# print(nrow(value_added))
# print("States in value_added:")
# print(unique(value_added$state_name))
# print("Years in value_added:")
# print(unique(value_added$YEAR))
# print("Sample of value_added data:")
# print(head(value_added))

# Create regression datasets
covid_reg <- wages %>%
  mutate(covid_period = ifelse(YEAR >= 2020, 1, 0))

concentration_reg <- wages %>%
  group_by(state_name, YEAR) %>%
  mutate(industry_share = total_workers / sum(total_workers)) %>%
  ungroup()

wage_premium_reg <- wages %>%
  group_by(state_name, YEAR) %>%
  mutate(
    state_avg_wage = mean(mean_wage),
    wage_premium = mean_wage / state_avg_wage,
    lag_premium = lag(wage_premium)
  ) %>%
  ungroup()

growth_reg <- wages %>%
  group_by(state_name, industry_name) %>%
  arrange(YEAR) %>%
  mutate(
    pct_change = (total_workers - lag(total_workers)) / lag(total_workers),
    lag_pct_change = lag(pct_change),
    industry_size = log(lag(total_workers))
  ) %>%
  ungroup()

inequality_reg <- wages %>%
  group_by(state_name, YEAR) %>%
  summarise(
    wage_ratio = max(mean_wage) / min(mean_wage),
    employment_concentration = max(total_workers) / sum(total_workers),
    total_employment = sum(total_workers),
    .groups = 'drop'
  )

# Run regressions
reg1 <- feols(log(total_workers) ~ covid_period | state_name + industry_name, data = covid_reg)
reg2 <- feols(log(mean_wage) ~ covid_period | state_name + industry_name, data = covid_reg)
reg3 <- feols(log(mean_wage) ~ industry_share | state_name + industry_name + YEAR, data = concentration_reg)
reg4 <- feols(wage_premium ~ lag_premium | state_name + industry_name, data = wage_premium_reg)
reg5 <- feols(pct_change ~ lag_pct_change + industry_size | state_name + industry_name + YEAR, data = growth_reg)
reg6 <- feols(log(wage_ratio) ~ log(employment_concentration) + log(total_employment) | state_name + YEAR, data = inequality_reg)

# Additional regressions for long-term analysis

# Long-term growth trends
reg_growth <- feols(log(total_workers) ~ year_trend | state_name + industry_name, 
                   data = wages %>% mutate(year_trend = YEAR - 2007))

reg_wage_growth <- feols(log(mean_wage) ~ year_trend | state_name + industry_name, 
                        data = wages %>% mutate(year_trend = YEAR - 2007))

# Financial crisis effects
reg_crisis_emp <- feols(log(total_workers) ~ post_crisis | state_name + industry_name, 
                       data = wages %>% mutate(post_crisis = ifelse(YEAR >= 2009, 1, 0)))

reg_crisis_wage <- feols(log(mean_wage) ~ post_crisis | state_name + industry_name, 
                        data = wages %>% mutate(post_crisis = ifelse(YEAR >= 2009, 1, 0)))

# Industry concentration over time
reg_concentration_trend <- feols(industry_share ~ year_trend | state_name + industry_name, 
                               data = concentration_reg %>% mutate(year_trend = YEAR - 2007))

# Wage inequality trends
reg_inequality_trend <- feols(log(wage_ratio) ~ year_trend + log(total_employment) | state_name, 
                            data = inequality_reg %>% mutate(year_trend = YEAR - 2007))

# Industry size and wage premium
reg_premium_trend <- feols(wage_premium ~ industry_size * year_trend | state_name + industry_name, 
                         data = wage_premium_reg %>% mutate(
                           year_trend = YEAR - 2007,
                           industry_size = log(total_workers)
                         ))

# Productivity regressions for the four states (CA, NY, MI, TX)
reg_prod_trend <- feols(log(labor_productivity) ~ year_trend | state_name, 
                       data = productivity %>% 
                         mutate(year_trend = YEAR - 2007))

reg_prod_covid <- feols(log(labor_productivity) ~ covid_period | state_name, 
                       data = productivity %>% 
                         mutate(covid_period = ifelse(YEAR >= 2020, 1, 0)))

# Productivity growth determinants for the four major states
reg_prod_growth <- feols(pct_change ~ lag(pct_change) + log(labor_productivity) | state_name, 
                        data = productivity)

# Industry composition effect on productivity (four-state analysis)
# Create a mapping for state names
state_mapping <- c(
  "California" = "CA",
  "New York" = "NY",
  "Michigan" = "MI",
  "Texas" = "TX"
)

prod_ind_comp <- wages %>%
  filter(state_name %in% names(state_mapping)) %>%  # Filter for four states
  mutate(state_name = state_mapping[state_name]) %>%  # Convert to abbreviations
  group_by(state_name, YEAR) %>%
  summarise(
    employment_concentration = max(total_workers) / sum(total_workers),
    avg_wage = mean(mean_wage),
    total_employment = sum(total_workers),
    .groups = 'drop'
  )

# print("After initial summarise:")
# print(head(prod_ind_comp))

prod_ind_comp <- prod_ind_comp %>%
  # Join with productivity data
  left_join(
    productivity %>% select(state_name, YEAR, labor_productivity),
    by = c("state_name", "YEAR")
  )

# print("After join:")
# print(head(prod_ind_comp))

prod_ind_comp <- prod_ind_comp %>%
  # Remove any NA values
  filter(!is.na(labor_productivity), 
         !is.na(employment_concentration),
         !is.na(avg_wage))

# print("Final dataset:")
# print(head(prod_ind_comp))

reg_prod_composition <- feols(log(labor_productivity) ~ log(employment_concentration) + 
                              log(avg_wage) | state_name + YEAR, 
                            data = prod_ind_comp)

# Create wage bill determinants dataset for four states
wage_bill_reg <- wages %>%
  # Filter for four states and convert state names
  filter(state_name %in% c("California", "New York", "Michigan", "Texas")) %>%
  mutate(state_name = case_when(
    state_name == "California" ~ "CA",
    state_name == "New York" ~ "NY", 
    state_name == "Michigan" ~ "MI",
    state_name == "Texas" ~ "TX"
  )) %>%
  # Calculate wage bill changes
  group_by(state_name, YEAR) %>%
  summarise(
    total_wage_bill = sum(total_workers * mean_wage, na.rm = TRUE),
    total_employment = sum(total_workers, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  # Join with employment changes data
  left_join(
    emp_changes %>%
      filter(state_name %in% c("California", "New York", "Michigan", "Texas")) %>%
      group_by(state_name, YEAR) %>%
      summarise(avg_emp_change = mean(pct_change, na.rm = TRUE), .groups = 'drop') %>%
      mutate(state_name = case_when(
        state_name == "California" ~ "CA",
        state_name == "New York" ~ "NY",
        state_name == "Michigan" ~ "MI", 
        state_name == "Texas" ~ "TX"
      )),
    by = c("state_name", "YEAR")
  ) %>%
  group_by(state_name) %>%
  arrange(state_name, YEAR) %>%
  mutate(
    pct_change_wage_bill = (total_wage_bill - lag(total_wage_bill)) / lag(total_wage_bill) * 100
  ) %>%
  ungroup()

# Create a mapping for state names
state_mapping <- c(
  "California" = "CA",
  "New York" = "NY",
  "Michigan" = "MI",
  "Texas" = "TX"
)

# First aggregate value_added by state and year
value_added_agg <- value_added %>%
  mutate(state_name = state_mapping[state_name]) %>%
  group_by(state_name, YEAR) %>%
  summarise(state_value_added = sum(state_value_added, na.rm = TRUE),
            .groups = 'drop')

# Join with productivity and aggregated value_added
wage_bill_reg <- wage_bill_reg %>%
  left_join(productivity %>% select(state_name, YEAR, labor_productivity), 
            by = c("state_name", "YEAR")) %>%
  left_join(value_added_agg,
            by = c("state_name", "YEAR"))

wage_bill_reg <- wage_bill_reg %>%
  # Log transform variables
  mutate(
    log_productivity = log(labor_productivity),
    log_employment = log(total_employment),
    log_value_added = log(state_value_added)
  ) %>%
  # Remove NA values
  filter(!is.na(pct_change_wage_bill),
         !is.na(log_productivity),
         !is.na(log_employment),
         !is.na(log_value_added))

# print("Final dataset:")
# print(nrow(wage_bill_reg))
# print("Final years:")
# print(unique(wage_bill_reg$YEAR))
# print(head(wage_bill_reg))

# Modified regression with employment changes
reg_wage_bill <- feols(pct_change_wage_bill ~ 
                        log_productivity + 
                        log_employment +
                        log_value_added +
                        avg_emp_change | # Using the averaged employment change
                        state_name + YEAR, 
                      data = wage_bill_reg)

# Use existing wage_bill_reg dataset and add growth calculations
employment_analysis <- wage_bill_reg %>%
  group_by(state_name) %>%
  arrange(state_name, YEAR) %>%
  mutate(
    productivity_growth = (labor_productivity - lag(labor_productivity)) / lag(labor_productivity),
    value_added_growth = (state_value_added - lag(state_value_added)) / lag(state_value_added),
    employment_growth = (total_employment - lag(total_employment)) / lag(total_employment),
    prod_value_interaction = productivity_growth * value_added_growth
  ) %>%
  ungroup() %>%
  # Remove NA values from growth calculations
  filter(!is.na(employment_growth),
         !is.na(productivity_growth),
         !is.na(value_added_growth))

# Run the regression
reg_employment_changes <- feols(employment_growth ~ 
                                productivity_growth + 
                                value_added_growth + 
                                prod_value_interaction | 
                                state_name + YEAR,
                              data = employment_analysis)

# Print detailed results to terminal
cat("\n=== Employment Changes Regression Results ===\n")
summary(reg_employment_changes)
cat("\nNumber of observations:", nobs(reg_employment_changes))
cat("\nNumber of states:", length(unique(employment_analysis$state_name)))
cat("\nYears covered:", paste(range(employment_analysis$YEAR), collapse=" - "), "\n\n")

# Create separate model lists for COVID and long-term analysis
models_covid <- list(
  "Employment\n(COVID)" = reg1,
  "Wages\n(COVID)" = reg2,
  "Wages\n(Concentration)" = reg3
)

models_industry <- list(
  "Wage\nPremium" = reg4,
  "Employment\nGrowth" = reg5,
  "Wage\nInequality" = reg6
)

models_trends1 <- list(
  "Employment\nGrowth" = reg_growth,
  "Wage\nGrowth" = reg_wage_growth,
  "Employment\n(Crisis)" = reg_crisis_emp,
  "Wages\n(Crisis)" = reg_crisis_wage
)

models_trends2 <- list(
  "Industry\nConcentration" = reg_concentration_trend,
  "Wage\nInequality" = reg_inequality_trend,
  "Wage\nPremium" = reg_premium_trend
)

models_productivity <- list(
  "Productivity\nTrend" = reg_prod_trend,
  "COVID\nImpact" = reg_prod_covid,
  "Growth\nDynamics" = reg_prod_growth,
  "Industry\nComposition" = reg_prod_composition,
  "Wage Bill\nDeterminants" = reg_wage_bill,
  "Employment\nChanges" = reg_employment_changes
)

# Create state summary statistics
state_summary <- wages %>%
  group_by(state_name) %>%
  summarise(
    avg_employment = mean(total_workers, na.rm = TRUE),
    avg_wage = mean(mean_wage, na.rm = TRUE),
    n_industries = n_distinct(industry_name),
    wage_inequality = mean(max(mean_wage) / min(mean_wage), na.rm = TRUE)
  ) %>%
  arrange(desc(avg_employment))

# Create COVID impact results
state_covid_results <- wages %>%
  mutate(covid_period = ifelse(YEAR >= 2020, 1, 0)) %>%
  group_by(state_name) %>%
  summarise(
    pre_covid_emp = mean(total_workers[YEAR == 2019], na.rm = TRUE),
    post_covid_emp = mean(total_workers[YEAR >= 2020], na.rm = TRUE),
    employment_change = (post_covid_emp / pre_covid_emp - 1) * 100
  ) %>%
  arrange(employment_change)

# Create wage inequality trends
state_inequality <- wages %>%
  group_by(state_name, YEAR) %>%
  summarise(
    wage_ratio = max(mean_wage, na.rm = TRUE) / min(mean_wage, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  group_by(state_name) %>%
  summarise(
    pre_covid_inequality = mean(wage_ratio[YEAR < 2020], na.rm = TRUE),
    post_covid_inequality = mean(wage_ratio[YEAR >= 2020], na.rm = TRUE),
    inequality_change = (post_covid_inequality / pre_covid_inequality - 1) * 100
  )








# # Create R Markdown content
# rmd_content <- '---
# title: "Economic Analysis Results"
# output:
#   pdf_document:
#     toc: true
#     toc_depth: 2
#     latex_engine: xelatex
# header-includes:
#   - \\usepackage{booktabs}
#   - \\usepackage{longtable}
#   - \\usepackage{float}
#   - \\usepackage{unicode-math}
#   - \\usepackage{geometry}
#   - \\usepackage{caption}
#   - \\usepackage{lscape}
#   - \\usepackage{tabularray}
#   - \\usepackage{graphicx}
#   - \\usepackage{codehigh}
#   - \\usepackage[normalem]{ulem}
#   - \\geometry{margin=1in}
#   - \\captionsetup{width=0.9\\textwidth, font=small}
#   - \\renewcommand{\\arraystretch}{1.2}
#   - \\UseTblrLibrary{booktabs}
#   - \\UseTblrLibrary{siunitx}
#   - \\usepackage{threeparttable}
# ---

# # Summary Statistics
# ```{r, echo=FALSE}
# options(knitr.kable.NA = "")
# knitr::kable(state_summary,
#   format = "latex",
#   booktabs = TRUE,
#   longtable = TRUE,
#   digits = 2,
#   caption = "State Summary Statistics",
#   col.names = c("State", "Avg Employment", "Avg Wage", "N Industries", "Wage Inequality"),
#   align = c("l", rep("r", 4)))
# ```

# \\newpage
# # Regression Models

# ## COVID Impact Models
# 1. Employment COVID Impact:
#    \\[ \\log(Employment_{ist}) = \\beta_0 + \\beta_1 COVID_t + \\alpha_s + \\gamma_i + \\epsilon_{ist} \\]
#    - Shows how employment levels changed during COVID (2020 onwards)

# 2. Wage COVID Impact:
#    \\[ \\log(Wage_{ist}) = \\beta_0 + \\beta_1 COVID_t + \\alpha_s + \\gamma_i + \\epsilon_{ist} \\]
#    - Shows how wages changed during COVID (2020 onwards)

# 3. Industry Share and Wages:
#    \\[ \\log(Wage_{ist}) = \\beta_0 + \\beta_1 Share_{ist} + \\alpha_s + \\gamma_i + \\delta_t + \\epsilon_{ist} \\]
#    - Tests if industries with larger employment shares pay higher wages

# 4. Wage Premium Persistence:
#    \\[ Premium_{ist} = \\beta_0 + \\beta_1 Premium_{is,t-1} + \\alpha_s + \\gamma_i + \\epsilon_{ist} \\]
#    - Examines if industries that paid above-average wages continue to do so

# 5. Industry Size and Growth:
#    \\[ \\Delta Employment_{ist} = \\beta_0 + \\beta_1 \\Delta Employment_{is,t-1} + \\beta_2 \\log(Size_{ist}) + \\alpha_s + \\gamma_i + \\delta_t + \\epsilon_{ist} \\]
#    - Tests if larger industries have different employment growth patterns

# 6. Employment Concentration and Inequality:
#    \\[ \\log(WageRatio_{st}) = \\beta_0 + \\beta_1 \\log(Concentration_{st}) + \\beta_2 \\log(TotalEmp_{st}) + \\alpha_s + \\delta_t + \\epsilon_{st} \\]
#    - Examines if states with more concentrated employment have higher wage inequality

# ## Long-term Growth Models
# 7. Employment Trends:
#    \\[ \\log(Employment_{ist}) = \\beta_0 + \\beta_1 (Year_t - 2007) + \\alpha_s + \\gamma_i + \\epsilon_{ist} \\]
#    - Shows long-term employment growth trends since 2007

# 8. Wage Trends:
#    \\[ \\log(Wage_{ist}) = \\beta_0 + \\beta_1 (Year_t - 2007) + \\alpha_s + \\gamma_i + \\epsilon_{ist} \\]
#    - Shows long-term wage growth trends since 2007

# ## Financial Crisis Models
# 9. Crisis Employment Impact:
#    \\[ \\log(Employment_{ist}) = \\beta_0 + \\beta_1 PostCrisis_t + \\alpha_s + \\gamma_i + \\epsilon_{ist} \\]
#    - Shows employment changes after 2008-09 financial crisis

# 10. Crisis Wage Impact:
#     \\[ \\log(Wage_{ist}) = \\beta_0 + \\beta_1 PostCrisis_t + \\alpha_s + \\gamma_i + \\epsilon_{ist} \\]
#     - Shows wage changes after 2008-09 financial crisis

# ## Industry Structure Models
# 11. Industry Concentration Trends:
#     \\[ Share_{ist} = \\beta_0 + \\beta_1 (Year_t - 2007) + \\alpha_s + \\gamma_i + \\epsilon_{ist} \\]
#     - Shows if industries are becoming more or less concentrated

# 12. Wage Inequality Trends:
#     \\[ \\log(WageRatio_{st}) = \\beta_0 + \\beta_1 (Year_t - 2007) + \\beta_2 \\log(TotalEmp_{st}) + \\alpha_s + \\epsilon_{st} \\]
#     - Shows if wage inequality within states is changing over time

# 13. Size-Premium Relationship:
#     \\[ Premium_{ist} = \\beta_0 + \\beta_1 \\log(Size_{ist}) + \\beta_2 (Year_t - 2007) + \\beta_3 [\\log(Size_{ist}) \\times (Year_t - 2007)] + \\alpha_s + \\gamma_i + \\epsilon_{ist} \\]
#     - Tests if larger industries wage premiums evolve differently

# ## Productivity Models (Four-State Analysis: CA, NY, MI, TX)
# 14. Long-term Productivity Trends:
#     \\[ \\log(Productivity_{st}) = \\beta_0 + \\beta_1 (Year_t - 2007) + \\alpha_s + \\epsilon_{st} \\]
#     - Shows long-term productivity growth trends since 2007 for major states
#     - Captures state-specific productivity trajectories

# 15. COVID Impact on Productivity:
#     \\[ \\log(Productivity_{st}) = \\beta_0 + \\beta_1 COVID_t + \\alpha_s + \\epsilon_{st} \\]
#     - Shows productivity changes during COVID period across major states
#     - Identifies differential impacts across the four largest state economies

# 16. Productivity Growth Dynamics:
#     \\[ \\Delta Productivity_{st} = \\beta_0 + \\beta_1 \\Delta Productivity_{s,t-1} + \\beta_2 \\log(Productivity_{st}) + \\alpha_s + \\epsilon_{st} \\]
#     - Tests for persistence in productivity growth in major state economies
#     - Examines convergence patterns across states

# 17. Industry Structure and Productivity:
#     \\[ \\log(Productivity_{st}) = \\beta_0 + \\beta_1 \\log(Concentration_{st}) + \\beta_2 \\log(Wage_{st}) + \\alpha_s + \\delta_t + \\epsilon_{st} \\]
#     - Examines if industry concentration affects productivity in major states
#     - Controls for wage levels and state-specific factors

# 18. Wage Bill Determinants (Four-State Analysis):
#     \\[ \\Delta WageBill_{st} = \\beta_0 + \\beta_1 Productivity_{st} + \\beta_2 Employment_{ist} + \\beta_3 ValueAdded_{ist} + \\beta_4 \\Delta Employment_{st} + \\alpha_s + \\delta_t + \\epsilon_{ist} \\]
#     - Examines determinants of changes in state-level wage bills
#     - Tests relationships between productivity, employment levels, employment growth, and value added
#     - Includes average employment changes across industries
#     - Focuses on the four major state economies

# Where:
# - \\(\\alpha_s\\) represents state fixed effects
# - \\(\\gamma_i\\) represents industry fixed effects
# - \\(\\delta_t\\) represents year fixed effects
# - \\(\\epsilon_{ist}\\) is the error term
# - Subscripts: i = industry, s = state, t = year

# # Regression Results

# ```{r, echo=FALSE}
# options("modelsummary_format_numeric_latex" = "plain")
# modelsummary(models_covid,
#   stars = TRUE,
#   gof_map = c("nobs", "r.squared"),
#   fmt = "%.3f",
#   estimate = "{estimate}{stars}",
#   statistic = "({std.error})",
#   coef_map = c(
#     "covid_period" = "COVID Period",
#     "industry_share" = "Industry Share"
#   ),
#   title = "COVID Impact Results",
#   notes = list("Significance levels: * p<0.1, ** p<0.05, *** p<0.01",
#     "Standard errors in parentheses"),
#   output = "latex")
# ```

# ```{r, echo=FALSE}
# modelsummary(models_industry,
#   stars = TRUE,
#   gof_map = c("nobs", "r.squared"),
#   fmt = "%.3f",
#   estimate = "{estimate}{stars}",
#   statistic = "({std.error})",
#   coef_map = c(
#     "lag_premium" = "Lagged Premium",
#     "lag_pct_change" = "Lagged % Change",
#     "industry_size" = "Industry Size",
#     "log(employment_concentration)" = "Log Emp Concentration",
#     "log(total_employment)" = "Log Total Employment"
#   ),
#   title = "Industry Structure Results",
#   notes = list("Significance levels: * p<0.1, ** p<0.05, *** p<0.01",
#     "Standard errors in parentheses"),
#   output = "latex",
#   float = FALSE)
# ```

# ```{r, echo=FALSE}
# modelsummary(models_trends1,
#   stars = TRUE,
#   gof_map = c("nobs", "r.squared"),
#   fmt = "%.3f",
#   estimate = "{estimate}{stars}",
#   statistic = "({std.error})",
#   coef_map = c(
#     "year_trend" = "Year Trend",
#     "post_crisis" = "Post-Crisis Period"
#   ),
#   title = "Long-term Growth Results",
#   notes = list("Significance levels: * p<0.1, ** p<0.05, *** p<0.01",
#     "Standard errors in parentheses"),
#   output = "latex",
#   float = FALSE)
# ```
# \\newpage
# ```{r, echo=FALSE}
# modelsummary(models_trends2,
#   stars = TRUE,
#   gof_map = c("nobs", "r.squared"),
#   fmt = "%.3f",
#   estimate = "{estimate}{stars}",
#   statistic = "({std.error})",
#   coef_map = c(
#     "year_trend" = "Year Trend",
#     "industry_size" = "Industry Size",
#     "industry_size:year_trend" = "Industry Size × Year Trend",
#     "log(total_employment)" = "Log Total Employment"
#   ),
#   title = "Industry Evolution Results",
#   notes = list("Significance levels: * p<0.1, ** p<0.05, *** p<0.01",
#     "Standard errors in parentheses"),
#   output = "latex",
#   float = FALSE)
# ```

# ```{r, echo=FALSE}
# modelsummary(reg_prod_trend,
#   stars = TRUE,
#   gof_map = c("nobs", "r.squared"),
#   fmt = "%.3f",
#   estimate = "{estimate}{stars}",
#   statistic = "({std.error})",
#   coef_map = c(
#     "year_trend" = "Year Trend"
#   ),
#   title = "Productivity Trends",
#   notes = list("Significance levels: * p<0.1, ** p<0.05, *** p<0.01",
#     "Standard errors in parentheses"),
#   output = "latex",
#   float = FALSE)
# ```

# ```{r, echo=FALSE}
# modelsummary(reg_prod_covid,
#   stars = TRUE,
#   gof_map = c("nobs", "r.squared"),
#   fmt = "%.3f",
#   estimate = "{estimate}{stars}",
#   statistic = "({std.error})",
#   coef_map = c(
#     "covid_period" = "COVID Period"
#   ),
#   title = "COVID Productivity Impact",
#   notes = list("Significance levels: * p<0.1, ** p<0.05, *** p<0.01",
#     "Standard errors in parentheses"),
#   output = "latex",
#   float = FALSE)
# ```

# ```{r, echo=FALSE}
# modelsummary(reg_prod_growth,
#   stars = TRUE,
#   gof_map = c("nobs", "r.squared"),
#   fmt = "%.3f",
#   estimate = "{estimate}{stars}",
#   statistic = "({std.error})",
#   coef_map = c(
#     "lag(pct_change)" = "Lagged % Change",
#     "log(labor_productivity)" = "Log Labor Productivity"
#   ),
#   title = "Productivity Growth Determinants",
#   notes = list("Significance levels: * p<0.1, ** p<0.05, *** p<0.01",
#     "Standard errors in parentheses"),
#   output = "latex",
#   float = FALSE)
# ```

# ```{r, echo=FALSE}
# modelsummary(reg_prod_composition,
#   stars = TRUE,
#   gof_map = c("nobs", "r.squared"),
#   fmt = "%.3f",
#   estimate = "{estimate}{stars}",
#   statistic = "({std.error})",
#   coef_map = c(
#     "log(employment_concentration)" = "Log Emp Concentration",
#     "log(avg_wage)" = "Log Avg Wage"
#   ),
#   title = "Industry Composition Effect on Productivity",
#   notes = list("Significance levels: * p<0.1, ** p<0.05, *** p<0.01",
#     "Standard errors in parentheses"),
#   output = "latex",
#   float = FALSE)
# ```

# ```{r, echo=FALSE}
# modelsummary(reg_wage_bill,
#   stars = TRUE,
#   gof_map = c("nobs", "r.squared"),
#   fmt = "%.3f",
#   estimate = "{estimate}{stars}",
#   statistic = "({std.error})",
#   coef_map = c(
#     "log_productivity" = "Log Productivity",
#     "log_employment" = "Log Employment",
#     "log_value_added" = "Log Value Added",
#     "avg_emp_change" = "Avg Employment Change"
#   ),
#   title = "Wage Bill Determinants (Four Major States)",
#   notes = list(
#     "Analysis covers California, New York, Michigan, and Texas",
#     "Dependent variable: % Change in State Wage Bill",
#     "Significance levels: * p<0.1, ** p<0.05, *** p<0.01",
#     "Standard errors in parentheses"
#   ),
#   output = "latex",
#   float = FALSE)
# ```

# ```{r, echo=FALSE}
# modelsummary(models_productivity,
#   stars = TRUE,
#   gof_map = c("nobs", "r.squared"),
#   fmt = "%.3f",
#   estimate = "{estimate}{stars}",
#   statistic = "({std.error})",
#   coef_map = c(
#     "year_trend" = "Year Trend",
#     "covid_period" = "COVID Period",
#     "lag(pct_change)" = "Lagged % Change",
#     "log(labor_productivity)" = "Log Productivity",
#     "log(employment_concentration)" = "Log Emp Concentration",
#     "log(avg_wage)" = "Log Average Wage",
#     "log_productivity" = "Log Productivity",
#     "log_employment" = "Log Employment",
#     "log_value_added" = "Log Value Added"
#   ),
#   title = "Productivity Analysis Results (Four Major States)",
#   notes = list(
#     "Analysis covers California, New York, Michigan, and Texas",
#     "Significance levels: * p<0.1, ** p<0.05, *** p<0.01",
#     "Standard errors in parentheses"
#   ),
#   output = "latex",
#   float = FALSE)
# ```'

# # Write Rmd file
# writeLines(rmd_content, "output/analysis.Rmd")

# # Render to PDF
# rmarkdown::render("output/analysis.Rmd", 
#                  output_file = "regressions.pdf",
#                  output_dir = "output")

# cat("Analysis complete! Check output/regressions.pdf for results\n")
# cat("Analysis complete! Check output/regressions.pdf for results\n")