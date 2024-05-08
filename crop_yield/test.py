import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

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
    # Iterate over each county polygon
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
    color = ds.colors.get(county, 'grey')  # Get the color for the county, default to grey if not found
    plt.plot(point.x, point.y, marker='s', markersize= 15, color=color, alpha = 0.5)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Grid Points Colored by County')
plt.show()


# Load the geometries of German counties
crops_counties = gpd.read_file("data/crop_yield/crop_yield_DE.shp")
crops_counties = crops_counties.to_crs(epsg=3035)  # Change to a projected CRS (Equal Earth)

# Create a grid of lon-lat combinations
lon, lat = np.meshgrid(np.linspace(crops_counties.total_bounds[0], crops_counties.total_bounds[2], 100),
                       np.linspace(crops_counties.total_bounds[1], crops_counties.total_bounds[3], 100))

# Create a GeoDataFrame with Point objects for each lon-lat combination
grid_gdf = gpd.GeoDataFrame(geometry=[Point(lon_val, lat_val) for lon_val, lat_val in zip(lon.flatten(), lat.flatten())],
                             crs='EPSG:4326')  # Use EPSG:4326 for lon-lat points

# Project the grid points to the same CRS as the counties
grid_gdf = grid_gdf.to_crs(crops_counties.crs)

# Create a Basemap instance
m = Basemap(projection='cyl', llcrnrlon=crops_counties.total_bounds[0], llcrnrlat=crops_counties.total_bounds[1],
            urcrnrlon=crops_counties.total_bounds[2], urcrnrlat=crops_counties.total_bounds[3], resolution='l')

# Plot the basemap
m.drawcountries(linewidth=0.5)
m.drawcoastlines(linewidth=0.5)

# Plot the grid points with colors corresponding to the counties
for index, row in grid_gdf.iterrows():
    point = row['geometry']
    color = 'grey'  # Default color if county not found
    for _, county in crops_counties.iterrows():
        if point.within(county.geometry):
            color = 'r'  # Set color based on county
            break
    x, y = m(point.x, point.y)
    m.scatter(x, y, color=color, s=10)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Grid Points Colored by County')
plt.show()