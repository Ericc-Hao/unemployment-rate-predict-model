
import os
import pandas as pd

def adjust_process_gdp_data(base_path, output_path):

    # Init Empty frame to store data
    final_gdp_data = pd.DataFrame()

    # go through all files under the data path
    for file_name in os.listdir('./datasets/raw-data/gdp'):
        if file_name.endswith('.csv'):
            file_path = os.path.join('./datasets/raw-data/gdp', file_name)

            # read data
            gdp_data = pd.read_csv(file_path)

            # filter data
            filtered_gdp = gdp_data[gdp_data["Estimates"].isin([
                "Gross domestic product at market prices", 
                "Gross fixed capital formation"
            ])]

            # keep needed list
            filtered_gdp = filtered_gdp[["REF_DATE", "GEO", "Estimates", "VALUE"]]

            # make sure the value type is numeric
            filtered_gdp["VALUE"] = pd.to_numeric(filtered_gdp["VALUE"], errors="coerce")

            # drop the empty
            filtered_gdp = filtered_gdp.dropna(subset=["VALUE"])

            # reconstruct data
            pivoted_gdp = filtered_gdp.pivot_table(
                index=["REF_DATE"],
                columns=["GEO", "Estimates"],
                values="VALUE"
            ).reset_index()

            # Flatten multi-level columns
            pivoted_gdp.columns = [
                f"{geo}_GDP" if "Gross domestic product at market prices" in estimate 
                else f"{geo}_GFCF" if "Gross fixed capital formation" in estimate 
                else f"{geo}" if isinstance(geo, str) 
                else geo
                for geo, estimate in pivoted_gdp.columns
            ]

            # Separate Canada and other GEOs
            canada_data = pivoted_gdp[[col for col in pivoted_gdp.columns if "Canada" in col or col == "REF_DATE"]]
            other_geo_data = pivoted_gdp[[col for col in pivoted_gdp.columns if "Canada" not in col and col != "REF_DATE"]]

            # Merge Canada and other GEO features
            final_gdp_data = pd.concat([final_gdp_data, canada_data, other_geo_data], axis=1, ignore_index=False)

            # Remove duplicate REF_DATE columns, keeping only the first one
            if 'REF_DATE' in final_gdp_data.columns:
                final_gdp_data = final_gdp_data.loc[:, ~final_gdp_data.columns.duplicated()]

    # save to a csv file
    final_gdp_data.to_csv('./datasets/filtered_data/preprocessed-data/gdp_data.csv', index=False)

    print(f"Preprocess [GDP] data successfully")

    return final_gdp_data


