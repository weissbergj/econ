# This was used to clean datafile.dat which is now called industry_wage_employment.dat in the data/industry_wage_employment folder

# Load required libraries
library(dplyr)
library(readr)

# Set the path to the data file
data_file <- "datafile.dat"

# Define column positions
column_positions <- fwf_positions(
  start = c(1,    55,  91,    84,   87),
  end =   c(4,    56,  96,    84,   90),
  col_names = c("YEAR", "STATEFIP", "INCWAGE", "EMPSTAT", "IND")
)

# Create state lookup table
state_names <- tribble(
  ~STATEFIP, ~state_name,
  "01", "Alabama",
  "02", "Alaska",
  "04", "Arizona",
  "05", "Arkansas",
  "06", "California",
  "08", "Colorado",
  "09", "Connecticut",
  "10", "Delaware",
  "11", "District of Columbia",
  "12", "Florida",
  "13", "Georgia",
  "15", "Hawaii",
  "16", "Idaho",
  "17", "Illinois",
  "18", "Indiana",
  "19", "Iowa",
  "20", "Kansas",
  "21", "Kentucky",
  "22", "Louisiana",
  "23", "Maine",
  "24", "Maryland",
  "25", "Massachusetts",
  "26", "Michigan",
  "27", "Minnesota",
  "28", "Mississippi",
  "29", "Missouri",
  "30", "Montana",
  "31", "Nebraska",
  "32", "Nevada",
  "33", "New Hampshire",
  "34", "New Jersey",
  "35", "New Mexico",
  "36", "New York",
  "37", "North Carolina",
  "38", "North Dakota",
  "39", "Ohio",
  "40", "Oklahoma",
  "41", "Oregon",
  "42", "Pennsylvania",
  "44", "Rhode Island",
  "45", "South Carolina",
  "46", "South Dakota",
  "47", "Tennessee",
  "48", "Texas",
  "49", "Utah",
  "50", "Vermont",
  "51", "Virginia",
  "53", "Washington",
  "54", "West Virginia",
  "55", "Wisconsin",
  "56", "Wyoming",
  "72", "Puerto Rico"
)

# Define focus states
focus_states <- c("California", "Texas", "New York", "Michigan")

# Read data FIRST
cat("[1/5] Reading data file...\n")
data <- read_fwf(
  data_file, 
  col_positions = column_positions,
  col_types = cols(
    YEAR = col_integer(),
    STATEFIP = col_character(),
    INCWAGE = col_integer(),
    EMPSTAT = col_integer(),
    IND = col_character()
  )
) %>%
  filter(YEAR >= 2007) %>%
  left_join(state_names, by = "STATEFIP") %>%
  filter(state_name %in% focus_states)  # Filter to 4 states EARLY

# THEN do industry lookup on smaller dataset
cat("[2/5] Adding industry classifications...\n")
data <- data %>%
  mutate(
    IND_num = as.numeric(IND),
    industry_name = case_when(
      IND_num == 0 ~ "N/A",
      IND_num >= 170 & IND_num <= 290 ~ "Agriculture, Forestry, Fishing, and Hunting",
      IND_num >= 370 & IND_num <= 490 ~ "Mining, Quarrying, and Oil and Gas Extraction",
      IND_num >= 570 & IND_num <= 690 ~ "Utilities",
      IND_num == 770 ~ "Construction",
      IND_num >= 1070 & IND_num <= 3990 ~ "Manufacturing",
      IND_num >= 4070 & IND_num <= 4590 ~ "Wholesale Trade",
      IND_num >= 4670 & IND_num <= 5790 ~ "Retail Trade",
      IND_num >= 6070 & IND_num <= 6390 ~ "Transportation and Warehousing",
      IND_num >= 6470 & IND_num <= 6780 ~ "Information",
      IND_num >= 6870 & IND_num <= 6992 ~ "Finance and Insurance",
      IND_num >= 7071 & IND_num <= 7190 ~ "Real Estate and Rental and Leasing",
      IND_num >= 7270 & IND_num <= 7490 ~ "Professional, Scientific, and Technical Services",
      IND_num == 7570 ~ "Management of Companies and Enterprises",
      IND_num >= 7580 & IND_num <= 7790 ~ "Administrative and Support and Waste Management Services",
      IND_num >= 7860 & IND_num <= 7890 ~ "Educational Services",
      IND_num >= 7970 & IND_num <= 8470 ~ "Health Care and Social Assistance",
      IND_num >= 8561 & IND_num <= 8590 ~ "Arts, Entertainment, and Recreation",
      IND_num >= 8660 & IND_num <= 8690 ~ "Accommodation and Food Services",
      IND_num >= 8770 & IND_num <= 9290 ~ "Other Services, Except Public Administration",
      IND_num >= 9370 & IND_num <= 9590 ~ "Public Administration",
      IND_num >= 9670 & IND_num <= 9870 ~ "Military",
      IND_num == 9920 ~ "Unemployed or Never Worked",
      TRUE ~ "Unknown"
    )
  )

# Calculate employment by industry
cat("[3/5] Calculating employment by industry...\n")
industry_employment <- data %>%
  filter(EMPSTAT == 1) %>%
  group_by(state_name, industry_name, YEAR) %>%
  summarise(
    workers = n(),
    .groups = 'drop'
  )

# Calculate wages by industry
cat("[4/5] Calculating wage analysis by industry...\n")
wage_analysis <- data %>%
  filter(EMPSTAT == 1) %>%
  group_by(state_name, industry_name, YEAR) %>%
  summarise(
    total_workers = n(),
    median_wage = median(INCWAGE[INCWAGE < 999998], na.rm = TRUE),
    mean_wage = mean(INCWAGE[INCWAGE < 999998], na.rm = TRUE),
    total_wage_bill = sum(INCWAGE[INCWAGE < 999998], na.rm = TRUE),
    .groups = 'drop'
  )

# Calculate employment changes by industry
cat("[5/5] Calculating employment changes by industry...\n")
employment_changes <- industry_employment %>%
  arrange(state_name, industry_name, YEAR) %>%
  group_by(state_name, industry_name) %>%
  mutate(
    pct_change = (workers - lag(workers)) / lag(workers) * 100
  )

# Print results
cat("\n=== 1. EMPLOYMENT BY INDUSTRY ===\n")
print(industry_employment %>% 
      filter(YEAR == max(YEAR)) %>%
      arrange(state_name, desc(workers)))

cat("\n=== 2. WAGE ANALYSIS BY INDUSTRY ===\n")
print(wage_analysis %>% 
      filter(YEAR == max(YEAR)) %>%
      arrange(state_name, desc(total_workers)))

cat("\n=== 3. EMPLOYMENT CHANGES BY INDUSTRY ===\n")
print(employment_changes %>%
      arrange(state_name, industry_name, YEAR))

# Save results
write_csv(industry_employment, "output/state_industry_employment.csv")
write_csv(wage_analysis, "output/state_wage_bill.csv")
write_csv(employment_changes, "output/state_employment_changes.csv")

cat("\nDone! Check the current directory for CSV files.\n")