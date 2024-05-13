import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
import matplotlib as mpl
import pandas as pd

# Load the geometries of German counties
crops_counties = gpd.read_file("data/crop_yield/crop_yield_DE.shp")
crops_counties = crops_counties.to_crs(epsg=4326)

# Open the NetCDF file

precipitation_file = "data/total_precipitation/tp.daily.calc.era5.0d50_CentralEurope.2000.nc"
nc_file = nc.Dataset(precipitation_file)

lon = nc_file.variables['lon'][:]
lat = nc_file.variables['lat'][:]

# Create a grid of lon-lat combinations
lon_grid, lat_grid = np.meshgrid(lon, lat)
lon_flat, lat_flat = lon_grid.flatten(), lat_grid.flatten()

# Create a GeoDataFrame with Point objects for each lon-lat combination
grid_gdf = gpd.GeoDataFrame(geometry=[Point(lon_val, lat_val) for lon_val, lat_val in zip(lon_flat, lat_flat)], crs='EPSG:4326')
grid_gdf.to_crs(crops_counties.crs, inplace=True)

# Define a function to find the matching county for each point
def find_matching_county(point):
    for index, county_geometry in crops_counties.geometry.items():
        if point.within(county_geometry):
            return crops_counties.loc[index, 'NUTS_ID']  # Return the NUTS_ID of the matching county


# Plot the polygons
crops_counties.plot(color='lightgrey', edgecolor='black', alpha=0.5, figsize=(10, 10))

import dictionary_states as ds

# Plot the points with colors corresponding to the counties
for index, row in grid_gdf.iterrows():
    point = row['geometry']
    county = find_matching_county(point)  # Find the corresponding county for each point
    color = ds.colors.get(county, 'none')  # Get the color for the county, default to grey if not found
    plt.plot(point.x, point.y, marker='s', markersize= 15, color=color, alpha = 0.5)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Grid Points Colored by County')
plt.savefig('crop_yield/Figures/grid_points_colored_by_county.png', transparent = True)
plt.show()



#TRY2 Adding the WSI to the counties! 
# Load the geometries of German counties
crops_counties = gpd.read_file("data/crop_yield/crop_yield_DE.shp")
crops_counties = crops_counties.to_crs(epsg=4326)
#like this there is just a geometry column in the grid_gdf now I need to append the year values of the water stress index.

#here load the DF from PIA
df = pd.read_csv('crop_yield/maximum_waterstress.csv', header = 0, na_values= 'NaN')
# Fill NaN values with 0s
df.fillna(0, inplace=True)

print(df.tail())
# Assuming you have a DataFrame named df with 'lon' and 'lat' columns
geometry = [Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])]

# Create a GeoDataFrame
grid_gdf = gpd.GeoDataFrame(df.iloc[:, 2:], geometry=geometry, crs='EPSG:4326')
# Optionally, transform the GeoDataFrame to match the coordinate reference system (CRS) of crops_counties
grid_gdf = grid_gdf.to_crs(crops_counties.crs)
print(grid_gdf.tail())
grid_gdf.fillna(0, inplace=True)


# Load the geometries of German counties
crops_counties = gpd.read_file("data/crop_yield/crop_yield_DE.shp")
crops_counties = crops_counties.to_crs(epsg=4326)

# Define a function to find the matching county for each point
def find_matching_county(point):
    for index, county_geometry in crops_counties.geometry.items():
        if point.within(county_geometry):
            return crops_counties.loc[index, 'NUTS_ID']  # Return the NUTS_ID of the matching county

# Create an empty dictionary to store the soil moisture values for each county and each year
county_soil_moisture = {county: {str(year): [] for year in range(2000, 2022)} for county in crops_counties['NUTS_ID']}

# Iterate over each grid cell
for index, grid_cell in grid_gdf.iterrows():
    # Find the matching county for the current grid cell
    county_name = find_matching_county(grid_cell.geometry)
    
    if county_name:
        # Retrieve the soil moisture values for each year for the current grid cell
        soil_moisture_values = [grid_cell[str(year)] for year in range(2000, 2022)]
        
        # Append the soil moisture values to the corresponding county and year
        for year, value in zip(range(2000, 2022), soil_moisture_values):
            county_soil_moisture[county_name][str(year)].append(value)

# Calculate the mean soil moisture value for each county for each year
county_mean_soil_moisture = {county: {year: np.nanmean(values) for year, values in year_values.items()} for county, year_values in county_soil_moisture.items()}

# Convert the dictionary to a DataFrame
mean_soil_moisture_df = pd.DataFrame(county_mean_soil_moisture).T
mean_soil_moisture_df.index.name = 'NUTS_ID'

print(mean_soil_moisture_df)

# Drop the 'crops' column
crops_counties_cleaned = crops_counties.drop('crops', axis=1)
# Define the range of columns to drop
columns_to_drop = [str(year) for year in range(2000, 2025)]
# Drop the columns from the crops_counties DataFrame
crops_counties_cleaned = crops_counties_cleaned.drop(columns=columns_to_drop)

# Keep only unique rows based on 'NUTS_ID'
crops_counties_unique = crops_counties_cleaned.drop_duplicates(subset=['NUTS_ID'])

# Merge the mean soil moisture values with the crops_counties GeoDataFrame
crops_counties_with_WSI = crops_counties_unique.merge(mean_soil_moisture_df, left_on='NUTS_ID', right_index=True, how='left')
crops_counties_with_WSI.fillna(0, inplace=True)
crops_counties_with_WSI.to_file("data/crop_yield/crop_yield_DE_WSI.shp")
# Plot the polygons
crops_counties_with_WSI.plot(column='2018', cmap='hot', edgecolor='black', alpha=0.5, legend=True, figsize=(10, 10))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Mean Soil Moisture by County for Year 2000')
plt.savefig('crop_yield/Figures/mean_WSI_by_county_2000.png', transparent=True)
plt.show()