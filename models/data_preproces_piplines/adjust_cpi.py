import os
import pandas as pd

def adjust_process_cpi_data(base_path, output_path):

    # Init Empty frame to store data
    final_cpi_data = pd.DataFrame()

    # go through all files under the data path
    for file_name in os.listdir('./datasets/raw-data/cpi'):
        if file_name.endswith('.csv'):
            file_path = os.path.join('./datasets/raw-data/cpi', file_name)

            # read data
            cpi_data = pd.read_csv(file_path)

            # filter data
            filtered_cpi = cpi_data[cpi_data["Products and product groups"] == "All-items"]

            # keep needed list
            filtered_cpi = filtered_cpi[["REF_DATE", "GEO", "VALUE"]]

            # make sure the value type is numeric
            filtered_cpi["VALUE"] = pd.to_numeric(filtered_cpi["VALUE"], errors="coerce")

            # drop rows with empty values
            filtered_cpi = filtered_cpi.dropna(subset=["VALUE"])

            # pivot the data to separate GEO and values
            pivoted_cpi = filtered_cpi.pivot_table(
                index=["REF_DATE"],
                columns=["GEO"],
                values="VALUE"
            ).reset_index()

            # Flatten multi-level columns
            pivoted_cpi.columns = [
                f"{geo}_CPI" if geo != "REF_DATE" else geo
                for geo in pivoted_cpi.columns
            ]

            # Separate Canada and other GEOs
            canada_data = pivoted_cpi[[col for col in pivoted_cpi.columns if "Canada" in col or col == "REF_DATE"]]
            other_geo_data = pivoted_cpi[[col for col in pivoted_cpi.columns if "Canada" not in col and col != "REF_DATE"]]

            # Merge Canada and other GEO features
            final_cpi_data = pd.concat([final_cpi_data, canada_data, other_geo_data], axis=1, ignore_index=False)

            # Remove duplicate REF_DATE columns, keeping only the first one
            if 'REF_DATE' in final_cpi_data.columns:
                final_cpi_data = final_cpi_data.loc[:, ~final_cpi_data.columns.duplicated()]

    # Save to a CSV file
    final_cpi_data.to_csv('./datasets/filtered_data/preprocessed-data/cpi_data.csv', index=False)

    print(f"Preprocess [CPI] data successfully")

    return final_cpi_data

