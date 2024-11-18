# Load required libraries
library(dplyr)
library(ggplot2)
library(readr)
library(scales)     # for nice number formatting

# Read the CSV files (silencing messages)
industry_emp <- read_csv("output/state_industry_employment.csv", show_col_types = FALSE)
wages <- read_csv("output/state_wage_bill.csv", show_col_types = FALSE)
emp_changes <- read_csv("output/state_employment_changes.csv", show_col_types = FALSE)
productivity <- read_csv("output/state_productivity.csv", show_col_types = FALSE)

# Add percentage calculations
wages <- wages %>%
  group_by(YEAR, state_name) %>%
  mutate(
    pct_of_state_employment = total_workers / sum(total_workers) * 100,
    relative_wage = mean_wage / mean(mean_wage) * 100,
    relative_median_wage = median_wage / median(median_wage) * 100
  )

# Common theme for all plots
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

# Create PDF with multiple pages
pdf("output/state_graphs.pdf", width = 11, height = 8.5)  # Standard letter size

# 1. Employment Trends
ggplot(industry_emp %>%
       group_by(state_name, industry_name) %>%
       filter(industry_name %in% (industry_emp %>%
                                filter(YEAR == max(YEAR)) %>%
                                group_by(industry_name) %>%
                                summarise(total = sum(workers)) %>%
                                arrange(desc(total)) %>%
                                head(5) %>%
                                pull(industry_name)))) +
  geom_line(aes(x = YEAR, y = workers/1000, color = industry_name), size = 1) +
  facet_wrap(~state_name, scales = "free_y") +
  scale_y_continuous(labels = comma) +
  scale_color_brewer(palette = "Set2") +
  clean_theme +
  labs(title = "Employment Trends in Top 5 Industries (2007-2022)",
       subtitle = "Workers (thousands) by state and industry",
       y = "Workers (thousands)",
       x = "Year",
       color = "Industry")

# 2. Wage Comparison
ggplot(wages %>% 
       filter(YEAR == 2022) %>%
       filter(industry_name %in% (wages %>%
                                filter(YEAR == 2022) %>%
                                group_by(industry_name) %>%
                                summarise(total = sum(total_workers)) %>%
                                arrange(desc(total)) %>%
                                head(5) %>%
                                pull(industry_name)))) +
  geom_col(aes(x = reorder(industry_name, mean_wage), 
               y = mean_wage/1000, 
               fill = state_name),
          position = "dodge") +
  coord_flip() +
  scale_y_continuous(labels = comma) +
  scale_fill_brewer(palette = "Set1") +
  clean_theme +
  labs(title = "Mean Annual Wages by Industry and State (2022)",
       subtitle = "Top 5 industries by employment",
       x = "Industry",
       y = "Mean Annual Wage (thousands $)",
       fill = "State")

# Median Wage Comparison
ggplot(wages %>% 
       filter(YEAR == 2022) %>%
       filter(industry_name %in% (wages %>%
                                filter(YEAR == 2022) %>%
                                group_by(industry_name) %>%
                                summarise(total = sum(total_workers)) %>%
                                arrange(desc(total)) %>%
                                head(5) %>%
                                pull(industry_name)))) +
  geom_col(aes(x = reorder(industry_name, median_wage), 
               y = median_wage/1000, 
               fill = state_name),
          position = "dodge") +
  coord_flip() +
  scale_y_continuous(labels = comma) +
  scale_fill_brewer(palette = "Set1") +
  clean_theme +
  labs(title = "Median Annual Wages by Industry and State (2022)",
       subtitle = "Top 5 industries by employment",
       x = "Industry",
       y = "Median Annual Wage (thousands $)",
       fill = "State")

# 3. Employment Changes Heatmap
ggplot(emp_changes %>%
       filter(!is.na(pct_change))) +
  geom_tile(aes(x = YEAR, y = reorder(industry_name, pct_change), fill = pct_change)) +
  facet_wrap(~state_name) +
  scale_fill_gradient2(low = "#FF9999", mid = "white", high = "#99FF99", 
                      midpoint = 0, limits = c(-20, 20)) +
  clean_theme +
  theme(axis.text.y = element_text(size = 8)) +
  labs(title = "Employment Changes by Industry (2007-2022)",
       subtitle = "Percentage change year-over-year",
       x = "Year",
       y = "Industry",
       fill = "% Change")

# 4. Wage Trends (new plot)
# First plot: Mean Wages
ggplot(wages %>%
       filter(industry_name %in% (wages %>%
                                filter(YEAR == 2022) %>%
                                group_by(industry_name) %>%
                                summarise(total = sum(total_workers)) %>%
                                arrange(desc(total)) %>%
                                head(5) %>%
                                pull(industry_name)))) +
  geom_line(aes(x = YEAR, y = mean_wage/1000, color = industry_name), size = 1) +
  facet_wrap(~state_name, scales = "free_y") +
  scale_y_continuous(labels = comma) +
  scale_color_brewer(palette = "Set2") +
  clean_theme +
  labs(title = "Mean Wage Trends by Industry (2007-2022)",
       subtitle = "Mean annual wages (thousands $) for top 5 industries",
       y = "Mean Annual Wage (thousands $)",
       x = "Year",
       color = "Industry")

# Median Wage Trends
ggplot(wages %>%
       filter(industry_name %in% (wages %>%
                                filter(YEAR == 2022) %>%
                                group_by(industry_name) %>%
                                summarise(total = sum(total_workers)) %>%
                                arrange(desc(total)) %>%
                                head(5) %>%
                                pull(industry_name)))) +
  geom_line(aes(x = YEAR, y = median_wage/1000, color = industry_name), size = 1) +
  facet_wrap(~state_name, scales = "free_y") +
  scale_y_continuous(labels = comma) +
  scale_color_brewer(palette = "Set2") +
  clean_theme +
  labs(title = "Median Wage Trends by Industry (2007-2022)",
       subtitle = "Median annual wages (thousands $) for top 5 industries",
       y = "Median Annual Wage (thousands $)",
       x = "Year",
       color = "Industry")

# 5. Total Wage Bill Trends (another new plot)
ggplot(wages %>%
       filter(industry_name %in% (wages %>%
                                filter(YEAR == 2022) %>%
                                group_by(industry_name) %>%
                                summarise(total = sum(total_workers)) %>%
                                arrange(desc(total)) %>%
                                head(5) %>%
                                pull(industry_name)))) +
  geom_line(aes(x = YEAR, y = total_wage_bill/1e9, color = industry_name), size = 1) +
  facet_wrap(~state_name, scales = "free_y") +
  scale_y_continuous(labels = comma) +
  scale_color_brewer(palette = "Set2") +
  clean_theme +
  labs(title = "Total Wage Bill Trends by Industry (2007-2022)",
       subtitle = "Total annual wages (billions $) for top 5 industries",
       y = "Total Wage Bill (billions $)",
       x = "Year",
       color = "Industry")

# 1. Industry Composition Changes
ggplot(wages %>%
       filter(industry_name %in% (wages %>%
                                filter(YEAR == 2022) %>%
                                group_by(industry_name) %>%
                                summarise(total = sum(pct_of_state_employment)) %>%
                                arrange(desc(total)) %>%
                                head(8) %>%
                                pull(industry_name)))) +
  geom_line(aes(x = YEAR, y = pct_of_state_employment, color = industry_name), size = 1) +
  facet_wrap(~state_name) +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  scale_color_brewer(palette = "Set2") +
  clean_theme +
  labs(title = "Industry Share of State Employment (2007-2022)",
       subtitle = "Percentage of state's total employment by industry",
       y = "Share of State Employment (%)",
       x = "Year",
       color = "Industry")

# 2. Relative Wage Premium
ggplot(wages %>%
       filter(industry_name %in% (wages %>%
                                filter(YEAR == 2022) %>%
                                group_by(industry_name) %>%
                                summarise(total = sum(total_workers)) %>%
                                arrange(desc(total)) %>%
                                head(8) %>%
                                pull(industry_name)))) +
  geom_line(aes(x = YEAR, y = relative_wage, color = industry_name), size = 1) +
  geom_hline(yintercept = 100, linetype = "dashed", color = "gray50") +
  facet_wrap(~state_name) +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  scale_color_brewer(palette = "Set2") +
  clean_theme +
  labs(title = "Industry Wage Premium (2007-2022)",
       subtitle = "Industry mean wage relative to state average (100 = state average)",
       y = "Relative Wage (%)",
       x = "Year",
       color = "Industry")

# Relative Median Wage Premium
ggplot(wages %>%
       filter(industry_name %in% (wages %>%
                                filter(YEAR == 2022) %>%
                                group_by(industry_name) %>%
                                summarise(total = sum(total_workers)) %>%
                                arrange(desc(total)) %>%
                                head(8) %>%
                                pull(industry_name)))) +
  geom_line(aes(x = YEAR, y = relative_median_wage, color = industry_name), size = 1) +
  geom_hline(yintercept = 100, linetype = "dashed", color = "gray50") +
  facet_wrap(~state_name) +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  scale_color_brewer(palette = "Set2") +
  clean_theme +
  labs(title = "Industry Median Wage Premium (2007-2022)",
       subtitle = "Industry median wage relative to state median (100 = state median)",
       y = "Relative Median Wage (%)",
       x = "Year",
       color = "Industry")

# 3. Wage Inequality Trends
wages %>%
  group_by(YEAR, state_name) %>%
  summarise(
    wage_ratio = max(mean_wage) / min(mean_wage),
    .groups = 'drop'
  ) %>%
  ggplot(aes(x = YEAR, y = wage_ratio, color = state_name)) +
  geom_line(size = 1) +
  scale_y_continuous(labels = comma) +
  scale_color_brewer(palette = "Set1") +
  clean_theme +
  labs(title = "Wage Inequality Trends (2007-2022)",
       subtitle = "Ratio of highest to lowest industry mean wages",
       y = "Wage Ratio (highest/lowest)",
       x = "Year",
       color = "State")

# Median Wage Inequality Trends
wages %>%
  group_by(YEAR, state_name) %>%
  summarise(
    wage_ratio = max(median_wage) / min(median_wage),
    .groups = 'drop'
  ) %>%
  ggplot(aes(x = YEAR, y = wage_ratio, color = state_name)) +
  geom_line(size = 1) +
  scale_y_continuous(labels = comma) +
  scale_color_brewer(palette = "Set1") +
  clean_theme +
  labs(title = "Median Wage Inequality Trends (2007-2022)",
       subtitle = "Ratio of highest to lowest industry median wages",
       y = "Wage Ratio (highest/lowest)",
       x = "Year",
       color = "State")

# 4. Employment Volatility
emp_changes %>%
  group_by(state_name, industry_name) %>%
  summarise(
    volatility = sd(pct_change, na.rm = TRUE),
    mean_size = mean(workers, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  filter(mean_size > 1000) %>%  # Focus on significant industries
  ggplot(aes(x = mean_size/1000, y = volatility)) +
  geom_point(aes(color = state_name), alpha = 0.6) +
  geom_text(aes(label = ifelse(volatility > 15, industry_name, "")), 
            size = 3, hjust = -0.1) +
  scale_x_log10(labels = comma) +
  scale_color_brewer(palette = "Set1") +
  clean_theme +
  labs(title = "Employment Volatility vs Industry Size",
       subtitle = "Standard deviation of year-over-year changes vs average employment",
       y = "Employment Volatility (SD of % changes)",
       x = "Average Industry Size (thousands)",
       color = "State")

# 5. State Specialization Index
wages %>%
  group_by(YEAR, industry_name) %>%
  mutate(national_share = total_workers / sum(total_workers)) %>%
  group_by(YEAR, state_name, industry_name) %>%
  mutate(specialization = (total_workers / sum(total_workers)) / national_share) %>%
  filter(industry_name %in% (wages %>%
                           filter(YEAR == 2022) %>%
                           group_by(industry_name) %>%
                           summarise(total = sum(total_workers)) %>%
                           arrange(desc(total)) %>%
                           head(8) %>%
                           pull(industry_name))) %>%
  ggplot(aes(x = YEAR, y = specialization, color = industry_name)) +
  geom_line(size = 1) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "gray50") +
  facet_wrap(~state_name) +
  scale_color_brewer(palette = "Set2") +
  clean_theme +
  labs(title = "State Industry Specialization (2007-2022)",
       subtitle = "Values > 1 indicate higher concentration than national average",
       y = "Specialization Index",
       x = "Year",
       color = "Industry")

# 6. COVID Impact Analysis
emp_changes %>%
  filter(YEAR >= 2019, YEAR <= 2021) %>%
  group_by(state_name, industry_name) %>%
  summarise(
    covid_drop = min(pct_change),
    recovery = max(pct_change),
    net_change = sum(pct_change),
    .groups = 'drop'
  ) %>%
  filter(abs(net_change) > 5) %>%  # Focus on significantly affected industries
  ggplot(aes(x = covid_drop, y = recovery, color = state_name)) +
  geom_point(aes(size = abs(net_change)), alpha = 0.6) +
  geom_text(aes(label = industry_name), size = 3, hjust = -0.1) +
  scale_size_continuous(name = "Absolute Net Change (%)") +
  scale_color_brewer(palette = "Set1") +
  clean_theme +
  labs(title = "COVID Impact and Recovery by Industry (2019-2021)",
       subtitle = "Comparing worst drop to best recovery",
       y = "Strongest Recovery (%)",
       x = "Largest Drop (%)",
       color = "State")

# 7. Industry Concentration Over Time
wages %>%
  group_by(YEAR, state_name) %>%
  arrange(desc(total_workers)) %>%
  mutate(
    cumulative_share = cumsum(total_workers) / sum(total_workers) * 100,
    top_3 = row_number() <= 3
  ) %>%
  filter(top_3) %>%
  group_by(YEAR, state_name) %>%
  summarise(
    top_3_share = sum(total_workers) / sum(wages$total_workers[wages$YEAR == YEAR & 
                                                              wages$state_name == first(state_name)]) * 100,
    .groups = 'drop'
  ) %>%
  ggplot(aes(x = YEAR, y = top_3_share, color = state_name)) +
  geom_line(size = 1) +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  scale_color_brewer(palette = "Set1") +
  clean_theme +
  labs(title = "Industry Concentration Trends (2007-2022)",
       subtitle = "Share of employment in top 3 industries",
       y = "Share of Total Employment (%)",
       x = "Year",
       color = "State")

# 1. Overall Productivity Trends
ggplot(productivity) +
  geom_line(aes(x = YEAR, y = labor_productivity, color = state_name), size = 1) +
  scale_y_continuous(labels = comma) +
  scale_color_brewer(palette = "Set1") +
  clean_theme +
  labs(title = "Labor Productivity Trends by State (2007-2023)",
       subtitle = "Index: 2017 = 100",
       y = "Labor Productivity Index",
       x = "Year",
       color = "State")

# 2. Productivity Growth Rates
ggplot(productivity %>% filter(!is.na(pct_change))) +
  geom_col(aes(x = YEAR, y = pct_change, fill = state_name), 
           position = "dodge", width = 0.8) +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  scale_fill_brewer(palette = "Set1") +
  clean_theme +
  labs(title = "Annual Productivity Growth Rates by State",
       subtitle = "Year-over-year percentage change",
       y = "Productivity Growth (%)",
       x = "Year",
       fill = "State")

# 3. Productivity Volatility Analysis
productivity %>%
  filter(!is.na(pct_change)) %>%
  group_by(state_name) %>%
  summarise(
    mean_growth = mean(pct_change),
    volatility = sd(pct_change),
    .groups = 'drop'
  ) %>%
  ggplot(aes(x = mean_growth, y = volatility, color = state_name)) +
  geom_point(size = 3) +
  geom_text(aes(label = state_name), hjust = -0.2, size = 4) +
  scale_color_brewer(palette = "Set1") +
  clean_theme +
  labs(title = "Productivity Growth: Risk vs. Return by State",
       subtitle = "Average growth rate vs. standard deviation (2007-2023)",
       x = "Average Annual Growth Rate (%)",
       y = "Growth Rate Volatility (Standard Deviation)",
       color = "State")

# 4. COVID Impact on Productivity
productivity %>%
  filter(YEAR >= 2019, YEAR <= 2022) %>%
  ggplot(aes(x = YEAR, y = labor_productivity, color = state_name)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  scale_y_continuous(labels = comma) +
  scale_color_brewer(palette = "Set1") +
  clean_theme +
  labs(title = "Productivity During COVID-19 Period",
       subtitle = "Labor productivity index (2019-2022)",
       y = "Labor Productivity Index",
       x = "Year",
       color = "State")

dev.off()