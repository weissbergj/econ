# preprocess_bea.R
library(dplyr)
library(readr)
library(tidyr)

# Read the raw BEA data, skipping the header rows and setting column names manually
bea_raw <- read_csv("output/bea_import.csv", skip = 4, col_names = FALSE) %>%
  # Remove any completely empty rows or footnote rows
  filter(!is.na(X1), !grepl("^Note\\.", X1), !grepl("^Legend", X1))

# Create proper column names
col_names <- c("Line", "Industry")
years <- 2006:2023
names(bea_raw) <- c(col_names, as.character(years))

# Create industry name mapping
industry_mapping <- c(
  "Agriculture, forestry, fishing, and hunting" = "Agriculture, Forestry, Fishing, and Hunting",
  "Mining" = "Mining, Quarrying, and Oil and Gas Extraction",
  "Utilities" = "Utilities",
  "Construction" = "Construction",
  "Manufacturing" = "Manufacturing",
  "Wholesale trade" = "Wholesale Trade",
  "Retail trade" = "Retail Trade",
  "Transportation and warehousing" = "Transportation and Warehousing",
  "Information" = "Information",
  "Finance and insurance" = "Finance and Insurance",
  "Real estate and rental and leasing" = "Real Estate and Rental and Leasing",
  "Professional, scientific, and technical services" = "Professional, Scientific, and Technical Services",
  "Management of companies and enterprises" = "Management of Companies and Enterprises",
  "Administrative and waste management services" = "Administrative and Support and Waste Management Services",
  "Educational services" = "Educational Services",
  "Health care and social assistance" = "Health Care and Social Assistance",
  "Arts, entertainment, and recreation" = "Arts, Entertainment, and Recreation",
  "Accommodation and food services" = "Accommodation and Food Services",
  "Other services, except government" = "Other Services, Except Public Administration"
)

# Clean and transform the BEA data
bea_clean <- bea_raw %>%
  # Remove line numbers and leading spaces
  mutate(Industry = gsub("^Line\\s+\\d+\\s+", "", Industry)) %>%
  mutate(Industry = trimws(Industry)) %>%
  # Keep only main industry categories that match employment data
  filter(Industry %in% names(industry_mapping)) %>%
  # Map industry names to match employment data
  mutate(Industry = industry_mapping[Industry]) %>%
  # Convert to long format
  pivot_longer(cols = as.character(years),
               names_to = "YEAR",
               values_to = "value_added") %>%
  # Convert Year to numeric and Value to billions
  mutate(
    YEAR = as.numeric(YEAR),
    value_added = value_added * 1000  # Convert to millions to match scale with employment data
  ) %>%
  # Remove Line column and rows with missing values
  select(-Line) %>%
  filter(!is.na(value_added)) %>%
  # Rename Industry column to match employment data
  rename(industry_name = Industry) %>%
  # Sort data
  arrange(industry_name, YEAR)

# Read state employment data
state_emp <- read_csv("output/state_industry_employment.csv")

# Calculate national employment by industry and year
national_emp <- state_emp %>%
  group_by(industry_name, YEAR) %>%
  summarize(national_workers = sum(workers), .groups = 'drop')

# Calculate state shares and value added
state_value_added <- state_emp %>%
  # Keep only the four states we want
  filter(state_name %in% c("California", "Michigan", "New York", "Texas")) %>%
  # Join with national totals
  left_join(national_emp, by = c("industry_name", "YEAR")) %>%
  # Calculate share
  mutate(state_share = workers / national_workers) %>%
  # Join with BEA value added
  left_join(bea_clean, by = c("industry_name", "YEAR")) %>%
  # Calculate state-level value added
  mutate(state_value_added = value_added * state_share) %>%
  # Keep relevant columns and remove any NA rows
  select(state_name, industry_name, YEAR, state_value_added) %>%
  filter(!is.na(state_value_added))

# Save both datasets
write_csv(bea_clean, "output/bea_clean.csv")
write_csv(state_value_added, "output/state_value_added.csv")

# Print summary to verify
cat("\nUnique industries in BEA data:\n")
print(unique(bea_clean$industry_name))
cat("\nYear range:", min(bea_clean$YEAR), "to", max(bea_clean$YEAR), "\n")

cat("\nStates in value added data:\n")
print(unique(state_value_added$state_name))