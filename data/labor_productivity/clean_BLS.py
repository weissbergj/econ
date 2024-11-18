# import pandas as pd

# # Load the dataset
# data = pd.read_excel("productivity.xlsx")

# # Combine 'Measure' and 'Units' to form the new column titles
# data["Measure_Unit"] = data["Measure"] + " (" + data["Units"] + ")"

# # Pivot the dataset: Use 'Year' as columns and 'Measure_Unit' as the values
# pivoted_data = data.pivot_table(
#     index=["Sector", "NAICS", "Industry"],  # Keep these as rows
#     columns=["Measure_Unit", "Year"],       # Measures and Years as column headers
#     values="Value",                        # The values for each Measure_Unit
#     aggfunc="first"                        # In case of duplicates
# )

# # Flatten the column MultiIndex into single strings
# pivoted_data.columns = [f"{measure}_{year}" for measure, year in pivoted_data.columns]

# # Reset the index for a clean output
# pivoted_data.reset_index(inplace=True)

# # Save the result to a CSV
# pivoted_data.to_csv("formatted_labor_productivity.csv", index=False)

import pandas as pd

# Load the dataset
data = pd.read_excel("productivity.xlsx")

# Drop the 'Digit' and 'Basis' columns
data = data.drop(columns=["Digit", "Basis"])

# Save the result to a CSV
data.to_csv("formatted_labor_productivity.csv", index=False)
