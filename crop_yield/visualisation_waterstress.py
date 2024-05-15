import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import netCDF4 as nc
import datetime as dt
import matplotlib.dates as mdates
from datetime import timedelta

file_path1 = 'results/model_output_1715580762.1522522/soil_moisture.nc'
nc_file = nc.Dataset(file_path1)
longitude_values = np.arange(0,22,1)
latitude_values = np.arange(0,22,1)
lati = nc_file.variables['lat'][:]
loni = nc_file.variables['lon'][:]
years = np.arange(2000,2024,1)
full_data = np.zeros((8766,len(latitude_values),len(longitude_values)))
water_stress_value = 300
water_stress = np.zeros((8766,len(latitude_values),len(longitude_values)))

for i, lat in enumerate(latitude_values):
    for j, lon in enumerate(longitude_values):
        dates = nc_file.variables['time'][:]
        full_data[:,i,j] = nc_file.variables['soil_moisture'][:,i,j]

for i, lat in enumerate(latitude_values):
    for j, lon in enumerate(longitude_values):
        for day in range(1,8766):
            water_stress[day,i,j] = water_stress[day-1,i,j] + (water_stress_value - full_data[day,i,j])
            if water_stress[day,i,j] < 0:
                water_stress[day,i,j] = 0

base = dt.datetime(2000,1,1)
date_list = [base + dt.timedelta(days=x) for x in range(dates[-1]+1)]
dates = pd.to_datetime(date_list)

years_str = years.astype(str).tolist()

# Create DataFrame with columns for latitude, longitude, and years
max_valuedf = pd.DataFrame(columns=['lat', 'lon'] + years_str)

# Create an empty array to store maximum values
max_value = np.zeros((len(years), len(latitude_values), len(longitude_values)))

# Create a list to store dictionaries of data
data_list = []

for i, lat in enumerate(lati):
    for j, lon in enumerate(loni):
        # Create DataFrame for water stress data
        df = pd.DataFrame({'date': dates, 'waterstress': water_stress[:, i, j]})
        # Group by year and find maximum water stress
        max_values_per_year = df.groupby(df['date'].dt.year)['waterstress'].max()
        # Store maximum values in max_value array
        max_value[:, i, j] = max_values_per_year.values
        # Append data to data_list
        data_list.append([lat, lon, *max_values_per_year])


# Convert data_list to DataFrame
columns = ['lat', 'lon'] + years_str
max_valuedf = pd.DataFrame(data_list, columns=columns)
max_valuedf.to_csv('crop_yield/maximum_waterstress.csv', index=False)

fig, ax = plt.subplots()
ax.set_xlabel('Date')
ax.set_ylabel('Soil Moisture [mm]')
lon_index = 10
lat_index = 10
a = 6600
b = 7100
soil_moisture = full_data[a:b,lat_index,lon_index]
dates = dates[a:b]
water_stress = water_stress[a:b,lat_index,lon_index]
max_stress = int(np.where(water_stress == np.max(water_stress))[0])
# Find the index where the water_stress changes from zero to non-zero
nonzero_to_zero = int(np.where((water_stress == 0) & (np.roll(water_stress, 1) != 0))[0])
# Find the index where the water_stress changes from non-zero to zero
zero_to_nonzero = int(np.where((water_stress != 0) & (np.roll(water_stress, 1) == 0))[0])
"""seasons = ['Winter', 'Spring', 'Summer', 'Autumn', 'Winter', 'Spring', 'Summer']
middles = ['2018-01-15', '2018-04-15', '2018-07-15', '2018-10-15', '2019-01-15', '2019-04-15', '2019-07-15']
middles = [np.datetime64(date) for date in middles]
date_ticks = [np.datetime64(date) for date in dates]

# Initialize an empty array to store the corresponding season labels
season_ticks = []

# Iterate through each date in the original date ticks array
for date in date_ticks:
    # Find the index of the closest middle date
    idx = np.abs(np.array(middles) - date).argmin()
    # Use the corresponding season label for the found middle date
    season_ticks.append(seasons[idx])


ax.set_xticks(middles, season_ticks)"""
ax.fill_between(dates[zero_to_nonzero:max_stress], soil_moisture[zero_to_nonzero:max_stress], water_stress_value, color='red', alpha=0.5)
ax.fill_between(dates[max_stress:nonzero_to_zero], soil_moisture[max_stress:nonzero_to_zero], water_stress_value, color='green', alpha=0.5)
ax.plot(dates, soil_moisture, 'darkblue', label='Soil Moisture [mm]')
ax.vlines(dates[max_stress], 0, 500, colors='black', linestyles='dashed', label='Maximum Water Stress')
#ax.set_xlim(np.min(soil_moisture), np.max(soil_moisture))
ax.hlines(water_stress_value, dates[0], dates[-1] , colors='red', linestyles='dashed', label='Water Stress Value')
ax1 = ax.twinx()
ax1.plot(dates, water_stress, 'r')
ax1.set_ylabel('Water Stress [mm]')
plt.savefig('crop_yield/Figures/soil_moisture.png', transparent=True)
plt.show()



