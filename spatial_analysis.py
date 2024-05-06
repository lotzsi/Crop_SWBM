# spatial analysis

import geopandas as gpd
from shapely.geometry import Point
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

# Create a grid from the latitude and longitude coordinates
# Assuming 'lon', 'lat', 'crops_counties' are defined
# Create a grid of points
lons, lats = np.meshgrid(lon, lat)
points = [Point(lon, lat) for lon, lat in zip(lons.ravel(), lats.ravel())]

# Create a GeoDataFrame with points
grid_gdf = gpd.GeoDataFrame(geometry=points, crs=crops_counties.crs)

# Define the cell size for the grid
cell_size = 1.0  # Adjust as needed

# Create grid cells around the points
min_x, min_y, max_x, max_y = grid_gdf.total_bounds
x_coords = np.arange(min_x, max_x + cell_size, cell_size)
y_coords = np.arange(min_y, max_y + cell_size, cell_size)
grid_cells = []
for x in x_coords[:-1]:
    for y in y_coords[:-1]:
        cell = box(x, y, x + cell_size, y + cell_size)
        grid_cells.append(cell)

# Create a GeoDataFrame with grid cells
grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs=crops_counties.crs)

# Plot the grid cells
grid_gdf.plot()
plt.show()

# Perform the intersection
intersection = overlay(grid_gdf, crops_counties, how='intersection')

# Plot the result
intersection.plot()
plt.show()


