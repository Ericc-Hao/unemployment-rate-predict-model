import os
import pandas as pd

def adjust_process_unemployment_data(base_path, output_path):
    # Init Empty frame to store data
    final_data = pd.DataFrame()

    # go through all year folders under the data path
    for year in range(2012, 2025): 
        year_path = os.path.join(base_path, str(year))
        if not os.path.exists(year_path):
            continue

        # go through all files under the data path
        for file_name in os.listdir(year_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(year_path, file_name)

                # read data
                data = pd.read_csv(file_path)

                # filter data
                filtered_data = data.loc[
                    data["Labour force characteristics"].isin(
                        ["Population", "Unemployment rate", "Participation rate"]
                    )
                ]

                # make sure the value type is numeric
                filtered_data = data.loc[
                    data["Labour force characteristics"].isin(
                        ["Population", "Unemployment rate", "Participation rate"]
                    )
                ].copy()

                filtered_data["VALUE"] = pd.to_numeric(filtered_data["VALUE"], errors="coerce")


                # drop rows with missing values
                filtered_data = filtered_data.dropna(subset=["VALUE"])

                # reconstruct data
                pivoted_data = filtered_data.pivot_table(
                    index=["REF_DATE"],
                    columns=["GEO", "Labour force characteristics"],
                    values="VALUE"
                ).reset_index()

                # Flatten multi-level columns and rename them
                pivoted_data.columns = [
                    f"{geo}-{characteristic}" if geo and characteristic else col
                    for col, geo, characteristic in zip(
                        pivoted_data.columns, 
                        pivoted_data.columns.get_level_values(0), 
                        pivoted_data.columns.get_level_values(1)
                    )
                ]

                # Add a column for the year
                pivoted_data["Year"] = year

                # concat final data
                final_data = pd.concat([final_data, pivoted_data], ignore_index=True)

    # Ensure unique REF_DATE and drop duplicates
    if 'REF_DATE' in final_data.columns:
        final_data = final_data.loc[:, ~final_data.columns.duplicated()]

    # save to a csv file
    final_data.to_csv(output_path, index=False)

    print(f"Preprocess [Unemployment] data successfully")

    return final_data

# Example usage
# adjust_process_unemployment_data('./datasets/raw-data/historical-unemployment-rate','./datasets/filtered_data/preprocessed-data/unemployment_data.csv')
