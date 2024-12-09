import os
import pandas as pd

from .cpi import process_cpi_data
from .gdp import process_gdp_data
from .unemployment import process_unemployment_data
from .wage import process_wage_data

def data_preprocess():

    output_folder = './datasets/filtered_data/preprocessed-data'

    cpi_base_path = './datasets/raw-data/cpi' 
    cpi_output_path = './datasets/filtered_data/preprocessed-data/cpi_data.csv'

    gdp_base_path = './datasets/raw-data/gdp'
    gdp_output_path = './datasets/filtered_data/preprocessed-data/gdp_data.csv' 

    unemployment_base_path = './datasets/raw-data/historical-unemployment-rate'
    unemployment_output_path = './datasets/filtered_data/preprocessed-data/unemployment_data.csv' 

    wage_base_path = './datasets/raw-data/minimum-wage/general_historical_minimum_wage.csv'
    wage_output_path = './datasets/filtered_data/preprocessed-data/wage_data.csv' 

    os.makedirs(output_folder, exist_ok=True)

    print(f'preprocessing data...')

    # preprocess each raw data
    cpi_data = process_cpi_data(cpi_base_path,cpi_output_path)
    gdp_data = process_gdp_data(gdp_base_path,gdp_output_path)
    unemployment_data = process_unemployment_data(unemployment_base_path, unemployment_output_path)
    wage_data = process_wage_data(wage_base_path,wage_output_path)

    print(f'mergering data...')

    # 1) merge cpi & gdp data
    data_c_g = pd.merge(cpi_data, gdp_data, on=["REF_DATE", "GEO"], how="inner")
    data_c_g['Year'] = data_c_g['REF_DATE']

    # data_c_g.to_csv('./datasets/filtered_data/data_c_g.csv', index=False)
    print(f'mergering data[CPI&GDP]')

    # 2) merge cpi & gdp data with unemployee data
    data_c_g_u = pd.merge(
        unemployment_data,
        data_c_g,
        left_on=["Year", "GEO"],
        right_on=["Year", "GEO"],
        how="left"
    )

    # fill the year data to mounthly
    if data_c_g_u.isnull().any().any():
        # data_c_g = data_c_g.sort_values(by=["GEO", "Year"])
        for index, row in data_c_g_u[data_c_g_u.isnull().any(axis=1)].iterrows():
            nearest = data_c_g[
                (data_c_g["GEO"] == row["GEO"]) &
                (data_c_g["Year"] <= row["Year"])
            ].sort_values(by="Year", ascending=False).head(1)
            
            if not nearest.empty:
                for col in nearest.columns:
                    if col not in ["Year", "GEO", "REF_DATE"]:
                        data_c_g_u.loc[index, col] = nearest.iloc[0][col]

    data_c_g_u.drop(columns=["REF_DATE_y"], inplace=True)
    data_c_g_u.rename(columns={"REF_DATE_x": "REF_DATE"}, inplace=True)

    # data_c_g_u.to_csv('./datasets/filtered_data/data_c_g_u.csv', index=False)
    print(f'mergering data[CPI&GDP&Uncemployemnt]')

    # 3) merge cpi & gdp & unemployee data with wage data
    data_c_g_u['REF_DATE'] = pd.to_datetime(data_c_g_u['REF_DATE'], errors='coerce').dt.strftime('%Y-%m')
    wage_data['Effective Date'] = pd.to_datetime(wage_data['Effective Date'], errors='coerce').dt.strftime('%Y-%m')

    final_merged_data = pd.merge(
        data_c_g_u,
        wage_data,
        left_on=['GEO', 'REF_DATE'],
        right_on=['GEO', 'Effective Date'],
        how='left'
    )

    # final_merged_data.sort_values(by=['GEO', 'REF_DATE'], inplace=True) 

    # Define the function to get the most recent wage data before REF_DATE
    def get_latest_wage(row, wage_data):
        relevant_wages = wage_data[
            (wage_data['GEO'] == row['GEO']) & (wage_data['Effective Date'] <= row['REF_DATE'])
        ].sort_values(by='Effective Date')
        if not relevant_wages.empty:
            return relevant_wages.iloc[-1]['Minimum Wage']
        return None

    # Apply the function to fill missing Minimum Wage values
    final_merged_data['Minimum Wage'] = final_merged_data.apply(
        lambda row: row['Minimum Wage'] if not pd.isna(row['Minimum Wage']) else get_latest_wage(row, wage_data), axis=1
    )
    final_merged_data = final_merged_data.drop(columns=['Effective Date'])
    final_merged_data = final_merged_data.drop(columns=['Year'])

    # Save the merged data
    final_merged_data.to_csv('./datasets/filtered_data/final_merged_data.csv', index=False)
    print(f'mergering data[CPI&GDP&Uncemployemnt&wage]')

    print(f'All data have been merged!')
    
    return final_merged_data