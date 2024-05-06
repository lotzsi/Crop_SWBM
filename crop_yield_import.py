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


# Load the text file containing the NUTS region codes
crop_yield = pd.read_csv(file_path, sep='\t')

# Extract the desired columns and remove the unwanted characters
crop_yield['geo'] = df['geo\\TIME_PERIOD'].str.split(',').str[-1]
crop_yield['strucpro'] = df['geo\\TIME_PERIOD'].str.split(',').str[-2]
crop_yield['crops'] = df['geo\\TIME_PERIOD'].str.split(',').str[-3]
crop_yield['freq'] = df['geo\\TIME_PERIOD'].str.split(',').str[-4]

# Drop the original column
crop_yield.drop(columns=['geo\\TIME_PERIOD'], inplace=True)

print(crop_yield.head())

# Perform the join based on the common column (NUTS region code)
joined_data = pd.merge(nuts_regions, crop_yield, on='NUTS_ID', how='inner')

# Now `joined_data` contains both the geometries from the shapefile and the data from the text file joined together
print(joined_data.head())


data=pd.read_csv(file_path2 ,sep='\t')
print(data.head())


