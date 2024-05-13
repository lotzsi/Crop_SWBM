import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import netCDF4 as nc
import datetime as dt

def calc_averages(core, variable, begin, end):
    ' data should be read in from nc files'
    data_list = []
    years = np.arange(2000,2024,1)
    years_str = years.astype(str).tolist()
    columns = ['lat', 'lon'] + years_str
    nc_file = nc.Dataset('data/total_precipitation/tp.daily.calc.era5.0d50_CentralEurope.2000.nc')
    lati = nc_file.variables['lat'][:]
    loni = nc_file.variables['lon'][:]
    full_data = np.zeros((8766, len(lati), len(loni)))
    if core == 'daily_average_temperature':
        variable1 = 't2m'
    else:
        variable1 = variable

    for i, lat in enumerate(lati):
        for j, lon in enumerate(loni):
            data = []
            # get radiation, temperature and precipitation data from netCDF files
            for year in years:
                file_path1 = 'data/'+core+'/'+variable+'.daily.0d50_CentralEurope.' + str(year) + '.nc'
                nc_file = nc.Dataset(file_path1)
                # 7,8 is the grid cell of interest for the respective catchment area
                dates = nc_file.variables['time'][:]
                data.append(nc_file.variables[variable1][:, i, j])
            full_data[:, i, j] = np.concatenate(data)
    if core == 'total_precipitation':
            full_data = full_data * 10 ** 3  # from m/day to mm/day
    if core == 'net_radiation':
        conv = 1 / 2260000  # from J/day/m**2 to mm/day
        full_data = full_data * conv
    base = dt.datetime(2000, 1, 1)
    date_list = [base + dt.timedelta(days=x) for x in range(len(full_data[:,0,0]))]
    dates = pd.to_datetime(date_list)
    for i, lat in enumerate(lati):
        for j, lon in enumerate(loni):
            # Create DataFrame for water stress data
            df = pd.DataFrame({'date': dates, variable: full_data[:, i, j]})
            df['year'] = df['date'].dt.year
            # Filter the DataFrame to include only the months of April to September
            df_filtered = df[(df['date'].dt.month >= begin) & (df['date'].dt.month <= end)]
            average_values = df_filtered.groupby('year')[variable].mean()
            data_list.append([lat, lon, *average_values])
    averaged_valuedf = pd.DataFrame(data_list, columns=columns)
    averaged_valuedf.to_csv('crop_yield/averaged/'+core+str(begin)+str(end)+'.csv', index=False)
    return averaged_valuedf

# Example usage
#list_of_cores = ['total_precipitation', 'net_radiation', 'daily_average_temperature', 'lai']
#list_of_variables = ['tp', 'nr', 't2m_mean', 'lai']
list_of_cores = ['lai']
list_of_variables = ['lai']
for i in range(len(list_of_cores)):
    calc_averages(list_of_cores[i], list_of_variables[i], 3, 9)

