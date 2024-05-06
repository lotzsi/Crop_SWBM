#crop yield import
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

file_path = 'data/crop_yield/apro_cpshr_clean.txt' # Crop production in EU standard humidity by NUTS 2 regions (apro_cpshr)

# Load the shapefile containing all NUTS regions in Europe
shapefile_path = 'data/EU_regions/NUTS_RG_60M_2021_3035.shp/NUTS_RG_60M_2021_3035.shp'
nuts_regions = gpd.read_file(shapefile_path)

#only keep NUTS_ID and geometry
nuts_regions = nuts_regions[['NUTS_ID', 'geometry']]

# Load the text file containing the NUTS region codes
crop_yield = pd.read_csv(file_path, sep = ",", header = 0, decimal=".")
print(crop_yield.keys())

# Extract the desired columns and remove the unwanted characters
crop_yield['NUTS_ID'] = crop_yield['geo']


# Drop the original column
crop_yield.drop(columns=['geo'], inplace=True)
print(crop_yield.head())

# Exclude the last three columns from the DataFrame
crop_yield_subset = crop_yield.iloc[:, 3:]

# Group the subset DataFrame by 'NUTS_ID' and sum up the values for each year column
summed_data = crop_yield_subset.groupby('NUTS_ID').sum().reset_index()

print(summed_data.head())
summed_data.to_csv('summed.csv', index = False)

# Perform the join based on the common column (NUTS region code)
joined_data = pd.merge(nuts_regions, summed_data, on='NUTS_ID', how='inner')
filter_DE = joined_data[joined_data["NUTS_ID"].str.startswith("DE")]

# Now `joined_data` contains both the geometries from the shapefile and the data from the text file joined together

import dictionary_states as ds #import dictionary with keys for study regions

# Get the list of NUTS IDs from the dictionary keys
nuts_ids_to_keep = list(ds.states.keys())

# Filter the DataFrame to keep only the rows where the "NUTS_ID" column matches the keys in the dictionary
filtered_data = joined_data[joined_data["NUTS_ID"].isin(nuts_ids_to_keep)]

print(filtered_data.head())

filtered_data.to_file('data/crop_yield/crop_yield_DE.shp', index = False)


