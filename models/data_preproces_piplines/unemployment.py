import os
import pandas as pd

def process_unemployment_data(base_path, output_path):
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
                filtered_data.loc[:, "VALUE"] = pd.to_numeric(filtered_data["VALUE"], errors="coerce")

                # drop the empty
                filtered_data = filtered_data.dropna(subset=["VALUE", "SCALAR_FACTOR"])

                # make sure "SCALAR_FACTOR" is in string type
                filtered_data.loc[:, "SCALAR_FACTOR"] = filtered_data["SCALAR_FACTOR"].astype(str)

                # reconstruct data
                pivoted_data = filtered_data.pivot_table(
                    index=["REF_DATE", "GEO"],
                    columns="Labour force characteristics",
                    values=["VALUE", "SCALAR_FACTOR"],
                    aggfunc="first"
                ).reset_index()

                # rename the column
                pivoted_data.columns = [
                    f"{col[1]}_{col[0]}" if col[1] else col[0] for col in pivoted_data.columns
                ]
                pivoted_data.rename(
                    columns={
                        "REF_DATE": "REF_DATE",
                        "GEO": "GEO",
                        "Population_VALUE": "Population",
                        "Unemployment rate_VALUE": "Unemployment Rate",
                        "Participation rate_VALUE": "Participation Rate",
                        "Population_SCALAR_FACTOR": "Population_SCALAR_FACTOR",
                        "Unemployment rate_SCALAR_FACTOR": "Unemployment Rate_SCALAR_FACTOR",
                        "Participation rate_SCALAR_FACTOR": "Participation Rate_SCALAR_FACTOR"
                    },
                    inplace=True
                )

                # add Year column
                pivoted_data["Year"] = int(year)

                # concat final data
                final_data = pd.concat([final_data, pivoted_data], ignore_index=True)

    # save to a csv file
    final_data.to_csv(output_path, index=False)

    print(f"Preprocess [Unemployment] data successfully")

    return final_data
