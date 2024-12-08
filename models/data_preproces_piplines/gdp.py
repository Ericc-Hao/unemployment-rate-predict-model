import os
import pandas as pd

def process_gdp_data(base_path, output_path):

    # Init Empty frame to store data
    final_gdp_data = pd.DataFrame()

    # go through all files under the data path
    for file_name in os.listdir(base_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(base_path, file_name)

            # read data
            gdp_data = pd.read_csv(file_path)

            # filter data
            filtered_gdp = gdp_data[gdp_data["Estimates"].isin([
                "Gross domestic product at market prices", 
                "Gross fixed capital formation"
            ])]

            # keep needed list
            filtered_gdp = filtered_gdp[["REF_DATE", "GEO", "Estimates", "VALUE", "SCALAR_FACTOR"]]

            # make sure the value type is numeric
            filtered_gdp["VALUE"] = pd.to_numeric(filtered_gdp["VALUE"], errors="coerce")

            # drop the empty
            filtered_gdp = filtered_gdp.dropna(subset=["VALUE", "SCALAR_FACTOR"])

            # reconstruct data
            pivoted_gdp = filtered_gdp.pivot_table(
                index=["REF_DATE", "GEO"],
                columns="Estimates",
                values="VALUE"
            ).reset_index()

            # add scaler column
            scalar_gdp = filtered_gdp[filtered_gdp["Estimates"] == "Gross domestic product at market prices"]
            scalar_gdp = scalar_gdp.drop_duplicates(subset=["REF_DATE", "GEO"])[["REF_DATE", "GEO", "SCALAR_FACTOR"]]
            scalar_gdp.rename(columns={"SCALAR_FACTOR": "GDP_SCALE"}, inplace=True)

            scalar_gcgf = filtered_gdp[filtered_gdp["Estimates"] == "Gross fixed capital formation"]
            scalar_gcgf = scalar_gcgf.drop_duplicates(subset=["REF_DATE", "GEO"])[["REF_DATE", "GEO", "SCALAR_FACTOR"]]
            scalar_gcgf.rename(columns={"SCALAR_FACTOR": "GCGF_SCALE"}, inplace=True)

            final_gdp = pd.merge(pivoted_gdp, scalar_gdp, on=["REF_DATE", "GEO"], how="left")
            final_gdp = pd.merge(final_gdp, scalar_gcgf, on=["REF_DATE", "GEO"], how="left")

            # concat ro final data
            final_gdp_data = pd.concat([final_gdp_data, final_gdp], ignore_index=True)

    # save to a csv file
    final_gdp_data.to_csv(output_path, index=False)

    print(f"Preprocess [GDP] data successfully")

    return final_gdp_data
