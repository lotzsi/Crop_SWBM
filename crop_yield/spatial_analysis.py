import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
import matplotlib as mpl

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

# Create an empty list to store the results
grid_polygons = []

# Iterate over each grid cell center point
for index, center_point in grid_gdf.iterrows():
    found_match = False  # Flag to indicate if a match is found
    for index, county_geometry in crops_counties.geometry.items():
        # Check if the center point is within the county polygon
        if center_point.geometry.within(county_geometry):
            # If it is, append the county polygon to the list of grid polygons
            grid_polygons.append((center_point.geometry.x, center_point.geometry.y, county_geometry))
            found_match = True  # Set the flag to True
            break  # Exit the loop once a matching polygon is found
    if not found_match:
        print(f"No matching polygon found for grid cell center point Lon: {center_point.geometry.x}, Lat: {center_point.geometry.y}")

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

# Example DataFrame containing lon, lat, and soil moisture values
# Replace this with your actual DataFrame
# Example DataFrame
#data = {
 #   'lon': [4.75, 5.25, 5.75, 6.25, 6.75],
  #  'lat': [54.75, 54.25, 53.75, 53.25, 52.75],
   # 'soil_moisture': [0.1, 0.2, 0.3, 0.4, 0.5]
#}
#grid_df = pd.DataFrame(data)

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

#like this there is just a geometry column in the grid_gdf now I need to append the year values of the water stress index.

# Iterate over each year
for year in range(2000, 2022):
    # Extract the water stress index for the current year from the NetCDF file
    water_stress_index = nc_file.variables[f'WSI_{year}'][:]  # Replace 'water_stress_index' with the actual variable name
    
    # Add the water stress index as a new column to the GeoDataFrame
    grid_gdf[f'WSI_{year}'] = water_stress_index.flatten()

# Close the NetCDF file
nc_file.close()

# Load the geometries of German counties
crops_counties = gpd.read_file("data/crop_yield/crop_yield_DE.shp")
crops_counties = crops_counties.to_crs(epsg=4326)

# Define a function to find the matching county for each point
def find_matching_county(point):
    for index, county_geometry in crops_counties.geometry.items():
        if point.within(county_geometry):
            return crops_counties.loc[index, 'NUTS_ID']  # Return the NUTS_ID of the matching county

# Create an empty dictionary to store the mean soil moisture values for each county and each year
county_mean_WSI = {county: {str(year): [] for year in range(2000, 2022)} for county in crops_counties['NUTS_ID']}

# Iterate over each year
for year in range(2000, 2022):
    # Iterate over each grid cell center point for the current year
    for index, center_point in grid_gdf.iterrows():
        found_match = False  # Flag to indicate if a match is found
        # Iterate over each county polygon
        for index, county_geometry in crops_counties.geometry.items():
            # Check if the center point is within the county polygon
            if center_point.geometry.within(county_geometry):
                # If it is, append the soil moisture value to the corresponding county and year
                county_name = crops_counties.loc[index, 'NUTS_ID']
                WSI_value = center_point['year'] # Replace ... with your method to retrieve soil moisture value for the grid point
                WSI_value[county_name][str(year)].append(WSI_value)
                found_match = True  # Set the flag to True
                break  # Exit the loop once a matching polygon is found
        if not found_match:
            print(f"No matching polygon found for grid cell center point Lon: {center_point.geometry.x}, Lat: {center_point.geometry.y}")

# Calculate the mean soil moisture value for each county for each year
for county_name, year_values in county_mean_WSI.items():
    for year, WSI_values in year_values.items():
        mean_WSI = np.mean(WSI_values)
        county_mean_WSI[county_name][year] = mean_WSI

# Convert the dictionary to a DataFrame
mean_WSI_df = pd.DataFrame(county_mean_WSI).T

# Merge the mean soil moisture values with the crops_counties GeoDataFrame
crops_counties_with_WSI = crops_counties.merge(mean_WSI_df, left_on='NUTS_ID', right_index=True, how='left') 

# Plot the polygons
crops_counties_with_WSI.plot(column='2000', cmap='viridis', edgecolor='black', alpha=0.5, legend=True, figsize=(10, 10))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Mean Soil Moisture by County for Year 2000')
plt.savefig('crop_yield/Figures/mean_WSI_by_county_2000.png', transparent=True)
plt.show()


# Create an empty list to store the grid cell center points
grid_centerpoints = []

# Iterate over each lon and lat value and create Point objects
for lon_val, lat_val in zip(grid_df['lon'], grid_df['lat']):
    grid_centerpoints.append(Point(lon_val, lat_val))

# Create an empty dictionary to store soil moisture values for each county
soil_moisture_by_county = {county_name: [] for county_name in crops_counties['NUTS_ID']}

# Iterate over each grid cell center point
for center_point, soil_moisture in zip(grid_centerpoints, grid_df['soil_moisture']):
    # Iterate over each county polygon
    for index, county in crops_counties.iterrows():
        # Check if the center point is within the county polygon
        if center_point.within(county.geometry):
            # If it is, append the soil moisture value to the corresponding county
            county_name = county['NUTS_ID']
            soil_moisture_by_county[county_name].append(soil_moisture)
            break  # Exit the loop once a matching polygon is found

# Add soil moisture values to the crops_counties GeoDataFrame
for county_name, soil_moisture_values in soil_moisture_by_county.items():
    crops_counties.loc[crops_counties['NUTS_ID'] == county_name, 'soil_moisture_values'] = soil_moisture_values

# Print the crops_counties GeoDataFrame with soil moisture values
print(crops_counties)
print(crops_counties.head())   