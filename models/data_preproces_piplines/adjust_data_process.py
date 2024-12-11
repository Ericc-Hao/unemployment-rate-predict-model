import os
import pandas as pd

from .adjust_cpi import adjust_process_cpi_data
from .adjust_gdp import adjust_process_gdp_data
from .adjust_unemployment import adjust_process_unemployment_data
# from adjust_wage import adjust_process_wage_data

def adjust_data_process():

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
    cpi_data = adjust_process_cpi_data(cpi_base_path,cpi_output_path)
    gdp_data = adjust_process_gdp_data(gdp_base_path,gdp_output_path)
    unemployment_data = adjust_process_unemployment_data(unemployment_base_path, unemployment_output_path)

    print(f'mergering data...')

    # 1) merge cpi & gdp data
    data_c_g = pd.merge(cpi_data, gdp_data, on=["REF_DATE"], how="inner")
    data_c_g.rename(columns={('REF_DATE', ''): 'REF_DATE'}, inplace=True)

    # data_c_g.to_csv('./datasets/filtered_data/data_c_g.csv', index=False)
    print("Merging CPI and GDP data completed.")


    # 2) merge cpi & gdp data with unemployee data
    data_c_g_u = pd.merge(
        unemployment_data,
        data_c_g,
        left_on=["Year"],
        right_on=["REF_DATE"],
        how="left"
    )

    # Fill missing values by finding the nearest year data
    if data_c_g_u.isnull().any().any():
        for index, row in data_c_g_u[data_c_g_u.isnull().any(axis=1)].iterrows():
            nearest = data_c_g[
                (data_c_g["REF_DATE"] <= row["Year"])
            ].sort_values(by="REF_DATE", ascending=False).head(1)

            if not nearest.empty:
                for col in nearest.columns:
                    if col not in ["Year", "REF_DATE"]:
                        data_c_g_u.loc[index, col] = nearest.iloc[0][col]

    data_c_g_u.drop(columns=["REF_DATE"], inplace=True)
    data_c_g_u.rename(columns={('REF_DATE', ''): 'REF_DATE'}, inplace=True)


    # Save merged data
    data_c_g_u.to_csv('./datasets/filtered_data/final_processed_data.csv', index=False)
    print("Merging CPI, GDP, and Unemployment data completed.")

    return data_c_g_u