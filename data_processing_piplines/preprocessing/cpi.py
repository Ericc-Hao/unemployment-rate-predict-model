import os
import pandas as pd


def process_cpi_data(base_path, output_path):
    # Init Empty frame to store data
    final_cpi_data = pd.DataFrame()

    # go through all files under the data path
    for file_name in os.listdir(base_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(base_path, file_name)

            # read data
            cpi_data = pd.read_csv(file_path)

            # filter data
            filtered_cpi = cpi_data[cpi_data["Products and product groups"] == "All-items"]

            # keep needed list
            filtered_cpi = filtered_cpi[["REF_DATE", "GEO", "VALUE", "SCALAR_FACTOR"]]

            # make sure the value type is numeric
            filtered_cpi["VALUE"] = pd.to_numeric(filtered_cpi["VALUE"], errors="coerce")

            # drop the empty
            filtered_cpi = filtered_cpi.dropna(subset=["VALUE", "SCALAR_FACTOR"])

            # rename the column
            filtered_cpi.rename(
                columns={
                    "VALUE": "CPI", 
                    "SCALAR_FACTOR": "CPI_SCALAR"
                },
                inplace=True
            )

            # concat final data
            final_cpi_data = pd.concat([final_cpi_data, filtered_cpi], ignore_index=True)

    # save to a csv file
    final_cpi_data.to_csv(output_path, index=False)

    print(f"Preprocess [CPI] data successfully")

    return final_cpi_data
