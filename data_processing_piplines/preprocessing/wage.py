import os
import pandas as pd

def process_wage_data(base_path, output_path):

    # Load the uploaded file
    minimum_wage_data = pd.read_csv(base_path)

    # Check and clean Effective Date column
    def parse_effective_date(date):
        try:
            return pd.to_datetime(date, errors='coerce')
        except Exception:
            return pd.NaT

    minimum_wage_data['Effective Date'] = minimum_wage_data['Effective Date'].apply(parse_effective_date)
    
    # Drop rows with invalid Effective Date
    minimum_wage_data = minimum_wage_data.dropna(subset=['Effective Date'])

    # # Filter out rows where Effective Date is outside 2013-2024 range
    # minimum_wage_data = minimum_wage_data[(minimum_wage_data['Effective Date'].dt.year >= 2013) & 
    #                                       (minimum_wage_data['Effective Date'].dt.year <= 2024)]

    # Replace 'Jurisdiction' column with 'GEO' and map the names
    jurisdiction_mapping = {
        'NL': 'Newfoundland and Labrador',
        'PEI': 'Prince Edward Island',
        'NS': 'Nova Scotia',
        'NB': 'New Brunswick',
        'QC': 'Quebec',
        'ON': 'Ontario',
        'MB': 'Manitoba',
        'SK': 'Saskatchewan',
        'AB': 'Alberta',
        'BC': 'British Columbia',
        'YT': 'Yukon',
        'FJ': 'Canada'
    }

    minimum_wage_data.rename(columns={'Jurisdiction': 'GEO'}, inplace=True)
    minimum_wage_data['GEO'] = minimum_wage_data['GEO'].map(jurisdiction_mapping)

    # Drop rows where GEO is null
    minimum_wage_data = minimum_wage_data.dropna(subset=['GEO'])

    # Extract year and month from Effective Date
    minimum_wage_data['Effective Date'] = minimum_wage_data['Effective Date'].dt.to_period('M').astype(str)

    # Remove '$' symbol from Minimum Wage and convert to numeric
    minimum_wage_data['Minimum Wage'] = minimum_wage_data['Minimum Wage'].replace({'\$': ''}, regex=True).astype(float)

    # Drop the 'Note' column if it exists
    if 'Note' in minimum_wage_data.columns:
        minimum_wage_data = minimum_wage_data.drop(columns=['Note'])

    # Sort by Effective Date
    minimum_wage_data = minimum_wage_data.sort_values(by='Effective Date')

    # Save the cleaned data
    minimum_wage_data.to_csv(output_path, index=False)

    print(f"Preprocess [Wage] data successfully")

    return minimum_wage_data