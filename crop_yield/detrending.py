import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt

# Read the data
data = pd.read_csv("filtered_data.csv")


# Assuming your DataFrame is named df and contains the crop yield data for one crop type
# Select one row (one crop type) from your DataFrame
crop_row = data.iloc[42, 3:].astype(float)  # Assuming the crop yield data starts from the third column

# Compute the moving average using lowess filter
window_size = int(0.20 * len(crop_row))  # Window size is 20% of the entire time series
smoothed = lowess(crop_row, np.arange(len(crop_row)), frac=window_size/len(crop_row), return_sorted=False)

# Detrend the crop yield by subtracting the moving average
detrended = crop_row - smoothed

# Plot original and detrended crop yield
years = range(2000, 2025)  # Assuming the years are from 2000 to 2024
plt.figure(figsize=(10, 6))
plt.plot(years, crop_row, label='Original Crop Yield')
plt.plot(years, smoothed, label='Smoothed (Moving Average)')
plt.plot(years, detrended, label='Detrended Crop Yield')
plt.xlabel('Year')
plt.ylabel('Crop Yield')
plt.title('Detrended Crop Yield for One Crop Type')
plt.legend()
plt.grid(True)
plt.show()

# Compute the moving average using lowess filter
window_size = int(0.20 * len(row_df.columns))  # 20% of the entire time series
detrended_row = row_df.apply(lambda x: x - lowess(x, np.arange(len(x)), frac=window_size/len(x), return_sorted=False)[:, 1])

# Plot the original and detrended crop yield for Crop_Type_A in Region_X
plt.figure(figsize=(10, 6))
plt.plot(row_df.columns.astype(int), row_df.values[0], label='Original')
plt.plot(row_df.columns.astype(int), detrended_row.values[0], label='Detrended')
plt.xlabel('Year')
plt.ylabel('Crop Yield')
plt.title('Original and Detrended Crop Yield for Crop_Type_A in Region_X')
plt.legend()
plt.grid(True)
plt.show()



# Exclude the geometry column and select only crop yield and years columns
data = data.iloc[:, 2:]

# Group by crop type
grouped_data = data.groupby('crops')

# Compute the moving average using lowess filter for each crop type
window_size = int(0.20 * len(data.columns))  # 20% of the entire time series
detrended_data = grouped_data.apply(lambda x: x.apply(lambda y: y - lowess(y, np.arange(len(y)), frac=window_size/len(y), return_sorted=False)[:, 1]))

# Plot the detrended data for each crop type
plt.figure(figsize=(10, 6))
for crop, group in detrended_data.groupby(level=0):
    for column in group.columns:
        plt.plot(np.arange(2000, 2025), group[column], label=crop + ': ' + column)
plt.xlabel('Year')
plt.ylabel('Detrended Crop Yield')
plt.title('Detrended Crop Yield Over Time')
plt.legend()
plt.grid(True)
plt.show()