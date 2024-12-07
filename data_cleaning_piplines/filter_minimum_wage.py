import pandas as pd
import numpy as np

# Indicate the start of the script
print("********** Start Filtering Minimum Wage Data **********")

# Step 1: Read the original CSV data
print("Reading the input CSV file...")
df = pd.read_csv(r"..\datasets\raw-data\minimum-wage\general_historical_minimum_wage.csv")
print("Input data loaded successfully.")

# Step 2: Convert 'Effective Date' to datetime format
print("Converting 'Effective Date' column to datetime format...")
df['Effective Date'] = pd.to_datetime(df['Effective Date'], format='%d-%b-%y', errors='coerce')
print("Conversion complete.")

# Step 3: Clean and process the 'Minimum Wage' column
print("Processing 'Minimum Wage' column...")
df['Minimum Wage'] = df['Minimum Wage'].replace('NA', np.nan)  # Replace 'NA' with NaN
df['Minimum Wage'] = df['Minimum Wage'].str.replace('$', '', regex=False)  # Remove dollar signs
df['Minimum Wage'] = pd.to_numeric(df['Minimum Wage'], errors='coerce')  # Convert to numeric
print("Processing complete.")

# Step 4: Filter data to include only valid dates
print("Filtering rows with valid 'Effective Date' values...")
df = df.dropna(subset=['Effective Date'])
print(f"Filtered data contains {len(df)} rows.")

# Step 5: Create a monthly date range from 2013-01-01 to 2023-12-01
print("Creating a monthly date range for the specified period...")
monthly_index = pd.date_range(start='2013-01-01', end='2023-12-01', freq='MS')
print("Monthly date range created.")

# Step 6: Process data for each jurisdiction
print("Reorganizing data by jurisdiction...")
jurisdictions = df['Jurisdiction'].unique()
jurisdiction_dfs = []

for jur in jurisdictions:
    print(f"Processing jurisdiction: {jur}...")
    
    # Filter data for the current jurisdiction
    jur_data = df[df['Jurisdiction'] == jur].copy()
    jur_data = jur_data.sort_values(by='Effective Date')  # Sort by date
    jur_data = jur_data.set_index('Effective Date')  # Set index to 'Effective Date'
    
    # Create a DataFrame with monthly dates and fill missing values
    monthly_df = pd.DataFrame(index=monthly_index)
    monthly_df = pd.merge_asof(
        monthly_df.reset_index().rename(columns={'index': 'Month'}),
        jur_data.reset_index().rename(columns={'Effective Date': 'Date'}),
        left_on='Month', right_on='Date', direction='backward'
    ).set_index('Month')
    
    # Forward fill missing 'Minimum Wage' values
    monthly_df['Minimum Wage'] = monthly_df['Minimum Wage'].ffill()
    
    # Rename the column for the current jurisdiction
    monthly_df = monthly_df[['Minimum Wage']].rename(columns={'Minimum Wage': jur})
    jurisdiction_dfs.append(monthly_df)
    
    print(f"Finished processing jurisdiction: {jur}.")

# Step 7: Combine all jurisdictions into a single DataFrame
print("Combining data from all jurisdictions...")
final_df = pd.concat(jurisdiction_dfs, axis=1)
print(f"Final DataFrame shape: {final_df.shape}")

# Step 8: Save the processed data to a CSV file
print("Saving the processed data to a CSV file...")
output_path = "../datasets/filtered_data/minimum-wage/historical_minimum_wage.csv"
final_df.to_csv(output_path, index_label='Month')
print(f"Data successfully saved to {output_path}.")

# Indicate the end of the script
print("********** Finished Filtering Minimum Wage Data **********")
