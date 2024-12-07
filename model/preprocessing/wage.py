import os
import pandas as pd

def process_wage_data(base_path, output_path):

    # Load the uploaded file
    minimum_wage_data = pd.read_csv(base_path)

    # Ensure Effective Date is a datetime
    minimum_wage_data['Effective Date'] = pd.to_datetime(minimum_wage_data['Effective Date'], errors='coerce')

    # Filter out rows where Effective Date is before 2013
    minimum_wage_data = minimum_wage_data[minimum_wage_data['Effective Date'].dt.year >= 2013]

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
        'YT': 'Yukon'
    }

    minimum_wage_data.rename(columns={'Jurisdiction': 'GEO'}, inplace=True)
    minimum_wage_data['GEO'] = minimum_wage_data['GEO'].map(jurisdiction_mapping)

    # Save the cleaned data
    minimum_wage_data.to_csv(output_path, index=False)

    print(f"Preprocess [Wage] data successfully")

    return minimum_wage_data


    # import ace_tools as tools; tools.display_dataframe_to_user(name="Cleaned Minimum Wage Data", dataframe=minimum_wage_data)
