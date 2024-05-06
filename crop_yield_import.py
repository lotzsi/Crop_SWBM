#crop yield import
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

file_path = 'data/crop_yield/estat_apro_cpshr.tsv' # Crop production in EU standard humidity by NUTS 2 regions (apro_cpshr)

# Load the shapefile containing all NUTS regions in Europe
shapefile_path = 'data/EU_regions/NUTS_RG_60M_2021_3035.shp/NUTS_RG_60M_2021_3035.shp'
nuts_regions = gpd.read_file(shapefile_path)

#only keep NUTS_ID and geometry
nuts_regions = nuts_regions[['NUTS_ID', 'geometry']]

# Load the text file containing the NUTS region codes
crop_yield = pd.read_csv(file_path, sep='\t', na_values=':')

# Extract the desired columns and remove the unwanted characters
crop_yield['NUTS_ID'] = crop_yield['freq,crops,strucpro,geo\TIME_PERIOD'].str.split(',').str[-1]
crop_yield['strucpro'] = crop_yield['freq,crops,strucpro,geo\TIME_PERIOD'].str.split(',').str[-2]
crop_yield['crops'] = crop_yield['freq,crops,strucpro,geo\TIME_PERIOD'].str.split(',').str[-3]
crop_yield['freq'] = crop_yield['freq,crops,strucpro,geo\TIME_PERIOD'].str.split(',').str[-4]

# Drop the original column
crop_yield.drop(columns=['freq,crops,strucpro,geo\TIME_PERIOD'], inplace=True)
print(crop_yield.head())

# Select only the year columns and the "NUTS_ID" column
year_columns = [col for col in crop_yield.columns if col[:4].isdigit() and 2000 <= int(col[:4]) <= 2024]
filtered_data = crop_yield[['NUTS_ID'] + year_columns]

# Group the DataFrame by "NUTS_ID" and sum up the values for year columns
summed_data = filtered_data.groupby('NUTS_ID').sum().reset_index()

print(summed_data.head())
summed_data.to_csv('summed.csv', index = False)

# Select only the year columns and the "NUTS_ID" column
#year_columns = [str(year) for year in range(2000, 2025)]
#filtered_data = crop_yield[['NUTS_ID'] + year_columns]

# Group the DataFrame by "NUTS_ID" and sum up the values for year columns
#summed_data = filtered_data.groupby('NUTS_ID', as_index=False)[year_columns].sum()

#print(summed_data.head())

# Perform the join based on the common column (NUTS region code)
joined_data = pd.merge(nuts_regions, crop_yield, on='NUTS_ID', how='inner')
filter_DE = joined_data[joined_data["NUTS_ID"].str.startswith("DE")]

# Now `joined_data` contains both the geometries from the shapefile and the data from the text file joined together

import dictionary_states as ds #import dictionary with keys for study regions

# Get the list of NUTS IDs from the dictionary keys
nuts_ids_to_keep = list(ds.states.keys())

# Filter the DataFrame to keep only the rows where the "NUTS_ID" column matches the keys in the dictionary
filtered_data = joined_data[joined_data["NUTS_ID"].isin(nuts_ids_to_keep)]
filtered_data = filtered_data.drop_duplicates()

