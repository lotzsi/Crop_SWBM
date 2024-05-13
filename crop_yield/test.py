cmap = mpl.colors.ListedColormap(county_colors)
bounds = np.arange(len(county_colors) + 1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# Create an empty 2D array for the grid data
grid_data = np.zeros((len(lat), len(lon)))

# Plot the points with colors corresponding to the counties
for index, row in grid_gdf.iterrows():
    point = row['geometry']
    county = find_matching_county(point)  # Find the corresponding county for each point
    color = ds.colors.get(county, 'none')  # Get the color for the county, default to grey if not found
    plt.plot(point.x, point.y, marker='s', markersize= 15, color=color, alpha=0.5)

# Plot the grid cells with colors corresponding to the counties
plt.imshow(grid_data, extent=[lon.min(), lon.max(), lat.min(), lat.max()], cmap=cmap, norm=norm)

# Add a colorbar to show the correspondence between colors and counties
cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=np.arange(len(county_colors)))
cb.ax.set_yticklabels(ds.counties.values())
cb.set_label('Counties')

# Add grid lines
plt.grid(True)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Grid Cells Colored by County')
plt.show()