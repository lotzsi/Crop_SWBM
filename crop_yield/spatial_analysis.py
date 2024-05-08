# spatial analysis

import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import Polygon
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry import box
from geopandas import overlay

precipitation_file = "data/total_precipitation/tp.daily.calc.era5.0d50_CentralEurope.2000.nc"
# Open the NetCDF file
nc_file = nc.Dataset(precipitation_file)

lon = nc_file.variables['lon'][:]
lat = nc_file.variables['lat'][:]

# Load the geometries of German counties
crops_counties = gpd.read_file("data/crop_yield/crop_yield_DE.shp")
crops_counties = crops_counties.to_crs(epsg=4326)

# Create a grid of lon-lat combinations
lon_grid, lat_grid = np.meshgrid(lon, lat)
lon_flat, lat_flat = lon_grid.flatten(), lat_grid.flatten()

# Create a GeoDataFrame with Point objects for each lon-lat combination
grid_gdf = gpd.GeoDataFrame(geometry=[Point(lon_val, lat_val) for lon_val, lat_val in zip(lon_flat, lat_flat)], crs='EPSG:4326')


#grid_gdf = gpd.GeoDataFrame(geometry=[Point(lon_val, lat_val) for lon_val, lat_val in zip(lon, lat)], crs='EPSG:4326')
grid_gdf.to_crs(crops_counties.crs, inplace=True)
print(grid_gdf)

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

# Print the results
for lon_val, lat_val, polygon in grid_polygons:
    print(f"Grid Cell Center Point Lon: {lon_val}, Lat: {lat_val} is inside the polygon: {polygon}")

# Plot the polygons
crops_counties.plot(edgecolor='black', figsize=(10, 10))
plt.title('Polygons of Counties in Germany')

# Plot the center points
for lon_val, lat_val, _ in grid_polygons:
    plt.scatter(lon_val, lat_val, color='red')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
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