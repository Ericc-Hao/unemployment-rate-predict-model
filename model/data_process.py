import os
import pandas as pd

from preprocessing.cpi import process_cpi_data 
from preprocessing.gdp import process_gdp_data 
from preprocessing.unemployment import process_unemployment_data 
from preprocessing.wage import process_wage_data



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

cpi_data = process_cpi_data(cpi_base_path,cpi_output_path)
gdp_data = process_gdp_data(gdp_base_path,gdp_output_path)
unemployment_data = process_unemployment_data(unemployment_base_path, unemployment_output_path)
wage_data = process_wage_data(wage_base_path,wage_output_path)

# merge cpi % gdp
merged_cpi_gdp_data = pd.merge(cpi_data, gdp_data, on=["REF_DATE", "GEO"], how="inner")
merged_cpi_gdp_data.sort_values(by="REF_DATE", inplace=True)
merged_cpi_gdp_data.to_csv('./datasets/filtered_data/merged_cpi_gdp_data.csv', index=False)


# merge all three
# 加载上传的文件

# Ensure the correct data types for matching
unemployment_data['Year'] = unemployment_data['Year'].astype(int)
merged_cpi_gdp_data['REF_DATE'] = pd.to_datetime(merged_cpi_gdp_data['REF_DATE'])
merged_cpi_gdp_data['Year'] = merged_cpi_gdp_data['REF_DATE'].dt.year

# 初步合并
extended_data = pd.merge(
    unemployment_data,
    merged_cpi_gdp_data,
    left_on=["Year", "GEO"],
    right_on=["Year", "GEO"],
    how="left"
)

# 对于未匹配的记录，尝试按最近的年份填充
if extended_data.isnull().any().any():
    # 先对 cpi_gdp_data 按 GEO 和年份排序
    merged_cpi_gdp_data = merged_cpi_gdp_data.sort_values(by=["GEO", "Year"])
    
    # 使用最近年份值填充
    for index, row in extended_data[extended_data.isnull().any(axis=1)].iterrows():
        nearest = merged_cpi_gdp_data[
            (merged_cpi_gdp_data["GEO"] == row["GEO"]) &
            (merged_cpi_gdp_data["Year"] <= row["Year"])
        ].sort_values(by="Year", ascending=False).head(1)
        
        if not nearest.empty:
            for col in nearest.columns:
                if col not in ["Year", "GEO", "REF_DATE"]:
                    extended_data.loc[index, col] = nearest.iloc[0][col]

extended_data.drop(columns=["REF_DATE_y"], inplace=True)
extended_data.rename(columns={"REF_DATE_x": "REF_DATE"}, inplace=True)

output_extended_path = './datasets/filtered_data/extended_unemployment_data_filled.csv'
extended_data.to_csv(output_extended_path, index=False)
