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
crop_yield = pd.read_csv(file_path, delimiter='\t')  # Adjust delimiter as per your file format

# Perform the join based on the common column (NUTS region code)
joined_data = pd.merge(nuts_regions, crop_yield, on='NUTS_CODE', how='inner')

# Now `joined_data` contains both the geometries from the shapefile and the data from the text file joined together
print(joined_data.head())


data=pd.read_csv(file_path2 ,sep='\t')
print(data.head())


