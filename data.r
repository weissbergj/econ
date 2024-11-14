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

# Process data by year
years <- 2007:2022  # adjust end year as needed
all_data <- NULL

for(current_year in years) {
  cat(sprintf("\nProcessing year %d...\n", current_year))
  
  # Read data for just one year
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
    filter(YEAR == current_year) %>%
    left_join(state_names, by = "STATEFIP")

  # Process industry classifications for this year
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
      ),
      broad_sector = case_when(
        industry_name %in% c("Agriculture, Forestry, Fishing, and Hunting",
                           "Mining, Quarrying, and Oil and Gas Extraction",
                           "Construction",
                           "Transportation and Warehousing",
                           "Utilities") ~ "Nonfarm business sector",
        
        industry_name %in% c("Wholesale Trade",
                           "Retail Trade",
                           "Information",
                           "Finance and Insurance",
                           "Real Estate and Rental and Leasing",
                           "Professional, Scientific, and Technical Services",
                           "Management of Companies and Enterprises",
                           "Administrative and Support and Waste Management Services",
                           "Educational Services",
                           "Health Care and Social Assistance",
                           "Arts, Entertainment, and Recreation",
                           "Accommodation and Food Services",
                           "Other Services, Except Public Administration") ~ "Business sector",
        
        industry_name == "Manufacturing" ~ "Manufacturing sector",
        
        industry_name %in% c("Public Administration",
                           "Military") ~ "Nonfinancial corporate sector",
        
        TRUE ~ "Other"
      )
    )

  # Append to main dataset
  if(is.null(all_data)) {
    all_data <- data
  } else {
    all_data <- bind_rows(all_data, data)
  }
  
  # Clean up
  rm(data)
  gc()
}

# Combine all metrics into one comprehensive dataset
cat("[3/5] Creating comprehensive industry analysis...\n")
comprehensive_analysis <- all_data %>%
  filter(EMPSTAT == 1) %>%
  group_by(state_name, industry_name, broad_sector, YEAR) %>%
  summarise(
    workers = n(),
    median_wage = median(INCWAGE[INCWAGE < 999998], na.rm = TRUE),
    mean_wage = mean(INCWAGE[INCWAGE < 999998], na.rm = TRUE),
    total_wage_bill = sum(INCWAGE[INCWAGE < 999998], na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  # Add employment changes
  arrange(state_name, industry_name, YEAR) %>%
  group_by(state_name, industry_name) %>%
  mutate(
    pct_change_employment = (workers - lag(workers)) / lag(workers) * 100
  ) %>%
  ungroup()

# Save single comprehensive file
write_csv(comprehensive_analysis, "cleaned_outputs/all_states_industry_analysis.csv")

cat("\nDone! Check the output directory for the comprehensive CSV file.\n")