import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pandas import DataFrame
import geopandas as gpd

# Read the data
data = pd.read_csv("filtered_data.csv")
print(data.head())

# Assuming your DataFrame is named df and contains the crop yield data for one crop type
# Select one row (one crop type) from your DataFrame
crop_row = data.iloc[42, 3:26].astype(float)  # Assuming the crop yield data starts from the third column

# Compute the moving average using lowess filter
window_size = int(0.20 * len(crop_row))  # Window size is 20% of the entire time series
smoothed = lowess(crop_row, np.arange(len(crop_row)), frac=1, return_sorted=False)
print(window_size)
print(window_size/len(crop_row))

# Detrend the crop yield by subtracting the moving average
detrended = crop_row - smoothed

# Plot original and detrended crop yield
years = range(2000, 2023)  # Assuming the years are from 2000 to 2024
plt.figure(figsize=(10, 6))
plt.plot(years, crop_row, label='Original Crop Yield')
plt.plot(years, smoothed, label='Smoothed (Moving Average)')
plt.plot(years, detrended, label='Detrended Crop Yield')
plt.xlabel('Year')
plt.ylabel('Crop Yield')
plt.title('Detrended Crop Yield for One Crop Type')
plt.legend()
plt.grid(True)

plt.savefig('crop_yield/Figures/detrended_crop_yield_timeseries.png', transparent=True)

plt.show()

# Fit a linear regression model to the smoothed data
regression_model = LinearRegression()
regression_model.fit(np.arange(len(crop_row)).reshape(-1, 1), smoothed)

# Get the slope of the regression line
slope = regression_model.coef_[0]

# Subtract the slope from the original data
detrended = crop_row - slope * np.arange(len(crop_row))

# Plot the detrended time series
plt.plot(years, detrended, label='Detrended Time Series')

# Add labels and legend
plt.xlabel('Year')
plt.ylabel('Detrended Crop Yield')
plt.title('Detrended Time Series')
plt.legend()
plt.savefig('crop_yield/Figures/detrended.png', transparent=True)
# Show plot
plt.show()

# Plot the original and detrended time series
plt.plot(years, crop_row, label='Original Data')
plt.plot(years, detrended, label='Detrended Data')

# Add labels and legend
plt.xlabel('Year')
plt.ylabel('Crop Yield')
plt.title('Original vs. Detrended Time Series')
plt.legend()
plt.savefig('crop_yield/Figures/detrended_time_seriesvsoriginal.png', transparent=True)

# Show plot
plt.show()

# Assuming your DataFrame is named data and contains the crop yield data for multiple crop types
# Initialize an empty DataFrame to store detrended data
detrended_data = pd.DataFrame()

# Iterate over each row in the DataFrame
for index, row in data.iterrows():
    crop_row = row[3:26].astype(float)  # Assuming the crop yield data starts from the third column

    # Apply LOWESS smoothing to the crop yield data
    smoothed = lowess(crop_row, np.arange(len(crop_row)), frac=1, return_sorted=False)

    # Fit a linear regression model to the smoothed data
    regression_model = LinearRegression()
    regression_model.fit(np.arange(len(crop_row)).reshape(-1, 1), smoothed)

    # Get the slope of the regression line
    slope = regression_model.coef_[0]

    # Subtract the slope from the original data to detrend
    detrended = crop_row - slope * np.arange(len(crop_row))

    # Append detrended data to the detrended_data DataFrame
    detrended_data = pd.concat([detrended_data, pd.DataFrame(detrended).T])
    #This line of code concatenates the existing detrended_data DataFrame with a new DataFrame created from the detrended data (pd.DataFrame(detrended).T). pd.DataFrame(detrended).T creates a DataFrame from the detrended data and transposes it to ensure that the detrended data is added as a row.
# Rename columns of the detrended_data DataFrame to match original column names
detrended_data.columns = data.columns[3:26]
# Add the first three columns of the original data to detrended_data
detrended_data = pd.concat([data.iloc[:, :3], detrended_data], axis=1)

detrended_data.to_csv('crop_yield/detrended_data.csv', index=False)
# Creating a GeoDataFrame
gdf = gpd.GeoDataFrame(detrended_data, geometry=geometry)

# Exporting as shapefile
gdf.to_file("data/crop_yield/crop_yield_DE_detrended_filtered.shp", driver='ESRI Shapefile')

# Now detrended_data contains detrended data for each crop type


# Choose a random row index
random_row_index = np.random.randint(0, len(data))

# Select the crop yield data for the random row
crop_row = data.iloc[random_row_index, 3:26].astype(float)

# Apply LOWESS smoothing to the crop yield data
smoothed = lowess(crop_row, np.arange(len(crop_row)), frac=1, return_sorted=False)

# Fit a linear regression model to the smoothed data
regression_model = LinearRegression()
regression_model.fit(np.arange(len(crop_row)).reshape(-1, 1), smoothed)

# Get the slope of the regression line
slope = regression_model.coef_[0]

# Subtract the slope from the original data to detrend
detrended = crop_row - slope * np.arange(len(crop_row))

# Plot the original and detrended time series
years = np.arange(len(crop_row))  # Assuming each data point corresponds to a year
plt.plot(years, crop_row, label='Original Data')
plt.plot(years, detrended, label='Detrended Data')

# Add labels and legend
plt.xlabel('Year')
plt.ylabel('Crop Yield')
plt.title('Original vs. Detrended Time Series')
plt.legend()

# Save plot
#plt.savefig('crop_yield/Figures/detrended_time_series_vs_original.png', transparent=True)

# Show plot
plt.show()