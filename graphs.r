# Load required libraries
library(dplyr)
library(ggplot2)
library(readr)
library(scales)
library(zoo)  # For rolling calculations

# Read the CSV files
industry_emp <- read_csv("output/all_states_industry_employment.csv", show_col_types = FALSE)
wages <- read_csv("output/all_states_wage_bill.csv", show_col_types = FALSE)
emp_changes <- read_csv("output/all_states_employment_changes.csv", show_col_types = FALSE)

# Create clean theme
clean_theme <- theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 10, hjust = 0.5),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    legend.background = element_rect(fill = "white", color = NA),
    legend.position = "bottom",
    legend.box = "horizontal",
    axis.text = element_text(size = 9),
    axis.title = element_text(size = 10)
  )

# Helper functions to identify interesting states
get_most_volatile_states <- function(data, n = 5) {
  data %>%
    group_by(state_name) %>%
    arrange(YEAR) %>%
    mutate(
      rolling_vol = rollapply(pct_change, 
                            width = 4, 
                            FUN = function(x) sd(x, na.rm = TRUE),
                            fill = NA, align = "right")
    ) %>%
    summarise(
      volatility = mean(rolling_vol, na.rm = TRUE),
      n_obs = n()
    ) %>%
    filter(n_obs >= 20) %>%
    arrange(desc(volatility)) %>%
    head(n) %>%
    pull(state_name)
}

get_highest_wage_inequality_states <- function(data, n = 5) {
  data %>%
    group_by(YEAR, state_name) %>%
    summarise(
      wage_ratio = max(mean_wage, na.rm = TRUE) / min(mean_wage, na.rm = TRUE),
      n_industries = n(),
      .groups = 'drop'
    ) %>%
    filter(n_industries >= 5) %>%  # Ensure enough industries
    group_by(state_name) %>%
    summarise(avg_ratio = mean(wage_ratio)) %>%
    arrange(desc(avg_ratio)) %>%
    head(n) %>%
    pull(state_name)
}

get_largest_covid_impact_states <- function(data, n = 5) {
  data %>%
    filter(YEAR == 2020) %>%
    group_by(state_name) %>%
    summarise(
      max_drop = min(pct_change, na.rm = TRUE),
      n_industries = n()
    ) %>%
    filter(n_industries >= 5) %>%  # Ensure enough industries
    arrange(max_drop) %>%
    head(n) %>%
    pull(state_name)
}

# Get interesting states for different visualizations
volatile_states <- get_most_volatile_states(emp_changes)
inequality_states <- get_highest_wage_inequality_states(wages)
covid_impact_states <- get_largest_covid_impact_states(emp_changes)

# Create PDF
pdf("output/graphs.pdf", width = 11, height = 8.5)

# 1. Employment Volatility Analysis
cat("Creating employment volatility plot...\n")

# Debug prints
print("Volatile states:")
print(volatile_states)

print("Sample of emp_changes data:")
print(head(emp_changes))

filtered_data <- emp_changes %>%
  filter(state_name %in% volatile_states) %>%
  group_by(state_name, industry_name) %>%
  filter(n() >= 10) %>%
  summarise(
    volatility = sd(pct_change, na.rm = TRUE),
    mean_size = mean(workers, na.rm = TRUE),
    n_obs = n(),
    .groups = 'drop'
  ) %>%
  filter(
    mean_size > 100,  # Lowered from 1000 to 100
    volatility > 0,
    !is.na(volatility)
  )

print("Final data for plotting:")
print(head(filtered_data))

ggplot(filtered_data, aes(x = mean_size/1000, y = volatility)) +
  geom_point(aes(color = state_name), alpha = 0.6) +
  geom_text(aes(label = ifelse(volatility > 15, industry_name, "")), 
            size = 3, hjust = -0.1) +
  scale_x_log10(labels = comma) +
  scale_color_brewer(palette = "Set1") +
  clean_theme +
  labs(title = "Employment Volatility vs Industry Size in Most Volatile States",
       subtitle = "Top 5 states by employment volatility across industries",
       y = "Employment Volatility (SD of % changes)",
       x = "Average Industry Size (thousands)",
       color = "State")

# 2. Wage Inequality Analysis
cat("Creating wage inequality plot...\n")
wages %>%
  filter(state_name %in% inequality_states) %>%
  group_by(YEAR, state_name) %>%
  summarise(
    wage_ratio = max(mean_wage) / min(mean_wage),
    n_industries = n(),
    .groups = 'drop'
  ) %>%
  filter(n_industries >= 5) %>%  # Ensure enough industries
  ggplot(aes(x = YEAR, y = wage_ratio, color = state_name)) +
  geom_line(linewidth = 1) +
  scale_y_continuous(labels = comma) +
  scale_color_brewer(palette = "Set1") +
  clean_theme +
  labs(title = "Wage Inequality Trends in States with Highest Disparities",
       subtitle = "Ratio of highest to lowest industry mean wages",
       y = "Wage Ratio (highest/lowest)",
       x = "Year",
       color = "State")

# Median Wage Inequality Analysis
wages %>%
  filter(state_name %in% inequality_states) %>%
  group_by(YEAR, state_name) %>%
  summarise(
    wage_ratio = max(median_wage) / min(median_wage),
    n_industries = n(),
    .groups = 'drop'
  ) %>%
  filter(n_industries >= 5) %>%  # Ensure enough industries
  ggplot(aes(x = YEAR, y = wage_ratio, color = state_name)) +
  geom_line(linewidth = 1) +
  scale_y_continuous(labels = comma) +
  scale_color_brewer(palette = "Set1") +
  clean_theme +
  labs(title = "Median Wage Inequality Trends in States with Highest Disparities",
       subtitle = "Ratio of highest to lowest industry median wages",
       y = "Wage Ratio (highest/lowest)",
       x = "Year",
       color = "State")

# 3. COVID Impact Analysis
cat("Creating COVID impact plot...\n")
emp_changes %>%
  filter(
    state_name %in% covid_impact_states,
    YEAR >= 2019, 
    YEAR <= 2021
  ) %>%
  group_by(state_name, industry_name) %>%
  summarise(
    covid_drop = min(pct_change),
    recovery = max(pct_change),
    net_change = sum(pct_change),
    n_obs = n(),
    .groups = 'drop'
  ) %>%
  filter(
    abs(net_change) > 5,
    n_obs >= 3  # Ensure data for all COVID period
  ) %>%
  ggplot(aes(x = covid_drop, y = recovery, color = state_name)) +
  geom_point(aes(size = abs(net_change)), alpha = 0.6) +
  geom_text(aes(label = industry_name), size = 3, hjust = -0.1) +
  scale_size_continuous(name = "Absolute Net Change (%)") +
  scale_color_brewer(palette = "Set1") +
  clean_theme +
  labs(title = "COVID Impact and Recovery in Most Affected States",
       subtitle = "Comparing worst drop to best recovery (2019-2021)",
       y = "Strongest Recovery (%)",
       x = "Largest Drop (%)",
       color = "State")

# 4. Industry Concentration Analysis
cat("Creating industry concentration plot...\n")
concentration_states <- wages %>%
  group_by(YEAR, state_name) %>%
  summarise(
    hhi = sum((total_workers / sum(total_workers))^2),
    n_industries = n(),
    .groups = 'drop'
  ) %>%
  filter(n_industries >= 5) %>%
  group_by(state_name) %>%
  summarise(
    hhi_change = last(hhi) - first(hhi)
  ) %>%
  arrange(desc(abs(hhi_change))) %>%
  head(5) %>%
  pull(state_name)

wages %>%
  filter(state_name %in% concentration_states) %>%
  group_by(YEAR, state_name) %>%
  summarise(
    hhi = sum((total_workers / sum(total_workers))^2),
    .groups = 'drop'
  ) %>%
  ggplot(aes(x = YEAR, y = hhi, color = state_name)) +
  geom_line(linewidth = 1) +
  scale_y_continuous(labels = scales::percent) +
  scale_color_brewer(palette = "Set1") +
  clean_theme +
  labs(title = "Industry Concentration Trends",
       subtitle = "States with largest changes in employment concentration (HHI)",
       y = "Employment Concentration (HHI)",
       x = "Year",
       color = "State")

# 5. Industry Growth Analysis
cat("Creating industry growth trends plot...\n")
growth_states <- emp_changes %>%
  filter(YEAR >= 2015) %>%
  group_by(state_name, YEAR) %>%
  summarise(
    total_growth = sum(pct_change * workers) / sum(workers),
    n_industries = n(),
    .groups = 'drop'
  ) %>%
  filter(n_industries >= 5) %>%
  group_by(state_name) %>%
  summarise(
    avg_growth = mean(total_growth, na.rm = TRUE)
  ) %>%
  arrange(desc(abs(avg_growth))) %>%
  head(5) %>%
  pull(state_name)

emp_changes %>%
  filter(
    state_name %in% growth_states,
    YEAR >= 2015
  ) %>%
  group_by(state_name, YEAR) %>%
  summarise(
    weighted_growth = sum(pct_change * workers) / sum(workers),
    .groups = 'drop'
  ) %>%
  ggplot(aes(x = YEAR, y = weighted_growth, color = state_name)) +
  geom_line(linewidth = 1) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  scale_color_brewer(palette = "Set1") +
  clean_theme +
  labs(title = "Employment Growth Trends",
       subtitle = "States with most volatile employment growth (2015-present)",
       y = "Employment Growth Rate (%)",
       x = "Year",
       color = "State")

# 5. Wage Premium Analysis
cat("Creating wage premium plot...\n")
premium_states <- wages %>%
  group_by(state_name, YEAR) %>%
  mutate(
    state_avg_wage = mean(mean_wage, na.rm = TRUE),
    state_median_wage = median(median_wage, na.rm = TRUE),
    wage_premium_mean = mean_wage / state_avg_wage * 100,
    wage_premium_median = median_wage / state_median_wage * 100
  ) %>%
  group_by(state_name, industry_name) %>%
  summarise(
    premium_volatility = mean(c(
      sd(wage_premium_mean, na.rm = TRUE),
      sd(wage_premium_median, na.rm = TRUE)
    )),
    avg_workers = mean(total_workers, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  filter(avg_workers > 5000) %>%
  group_by(state_name) %>%
  summarise(
    avg_volatility = mean(premium_volatility, na.rm = TRUE)
  ) %>%
  arrange(desc(avg_volatility)) %>%
  head(5) %>%
  pull(state_name)

wages %>%
  filter(
    state_name %in% premium_states,
    industry_name %in% (wages %>%
                       filter(YEAR == max(YEAR)) %>%
                       group_by(industry_name) %>%
                       summarise(total = sum(total_workers)) %>%
                       arrange(desc(total)) %>%
                       head(8) %>%
                       pull(industry_name))
  ) %>%
  group_by(state_name, YEAR) %>%
  mutate(
    state_avg_wage = mean(mean_wage, na.rm = TRUE),
    state_median_wage = median(median_wage, na.rm = TRUE),
    wage_premium_mean = mean_wage / state_avg_wage * 100,
    wage_premium_median = median_wage / state_median_wage * 100
  ) %>%
  ggplot(aes(x = YEAR, y = wage_premium_mean, color = industry_name)) +
  geom_line(linewidth = 1) +
  geom_hline(yintercept = 100, linetype = "dashed", color = "gray50") +
  facet_wrap(~state_name) +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  scale_color_brewer(palette = "Set2") +
  clean_theme +
  labs(title = "Industry Wage Premium Patterns",
       subtitle = "Industry mean wage relative to state average (100 = state average)",
       y = "Relative Wage (%)",
       x = "Year",
       color = "Industry")

# Median Wage Premium Analysis
wages %>%
  filter(
    state_name %in% premium_states,
    industry_name %in% (wages %>%
                       filter(YEAR == max(YEAR)) %>%
                       group_by(industry_name) %>%
                       summarise(total = sum(total_workers)) %>%
                       arrange(desc(total)) %>%
                       head(8) %>%
                       pull(industry_name))
  ) %>%
  group_by(state_name, YEAR) %>%
  mutate(
    state_median_wage = median(median_wage, na.rm = TRUE),
    wage_premium = median_wage / state_median_wage * 100
  ) %>%
  ggplot(aes(x = YEAR, y = wage_premium, color = industry_name)) +
  geom_line(linewidth = 1) +
  geom_hline(yintercept = 100, linetype = "dashed", color = "gray50") +
  facet_wrap(~state_name) +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  scale_color_brewer(palette = "Set2") +
  clean_theme +
  labs(title = "Industry Median Wage Premium Patterns",
       subtitle = "Industry median wage relative to state median (100 = state median)",
       y = "Relative Wage (%)",
       x = "Year",
       color = "Industry")

dev.off()

cat("\nDone! Check output/graphs.pdf for the visualizations.\n")