import numpy as np
import pandas as pd
import netCDF4 as nc
import datetime as dt
import geopandas as gpd
from shapely.geometry import Point
import dictionary_states as ds
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def calc_water_stress(water_stress_value):
    file_path1 = 'results/model_output_1715580762.1522522/soil_moisture.nc'
    nc_file = nc.Dataset(file_path1)
    longitude_values = np.arange(0,22,1)
    latitude_values = np.arange(0,22,1)
    lati = nc_file.variables['lat'][:]
    loni = nc_file.variables['lon'][:]
    years = np.arange(2000,2024,1)
    full_data = np.zeros((8766,len(latitude_values),len(longitude_values)))
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
    max_valuedf.to_csv(f'crop_yield/compareWSI/maximum_waterstress_{water_stress_value}.csv', index=False)
    return
# Define a function to find the matching county for each point
def find_matching_county(point):
    crops_counties = gpd.read_file("data/crop_yield/crop_yield_DE.shp")
    crops_counties = crops_counties.to_crs(epsg=4326)
    for index, county_geometry in crops_counties.geometry.items():
        if point.within(county_geometry):
            return crops_counties.loc[index, 'NUTS_ID']  # Return the NUTS_ID of the matching county
        
def map_to_state(water_stress_value):
    file_path = f'crop_yield/compareWSI/maximum_waterstress_{water_stress_value}'
    #TRY2 Adding the WSI to the counties! 
    # Load the geometries of German counties
    crops_counties = gpd.read_file("data/crop_yield/crop_yield_DE.shp")
    crops_counties = crops_counties.to_crs(epsg=4326)
    #like this there is just a geometry column in the grid_gdf now I need to append the year values of the water stress index.

    #here load the DF from PIA
    df = pd.read_csv(f'{file_path}.csv', header = 0, na_values= 'NaN')
    # Fill NaN values with 0s
    df.fillna(0, inplace=True)

    print(df.tail())
    # Assuming you have a DataFrame named df with 'lon' and 'lat' columns
    geometry = [Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])]

    # Create a GeoDataFrame
    grid_gdf = gpd.GeoDataFrame(df.iloc[:, 2:], geometry=geometry, crs='EPSG:4326')
    # Optionally, transform the GeoDataFrame to match the coordinate reference system (CRS) of crops_counties
    grid_gdf = grid_gdf.to_crs(crops_counties.crs)
    grid_gdf.fillna(0, inplace=True)

    # Create an empty dictionary to store the soil moisture values for each county and each year
    county_soil_moisture = {county: {str(year): [] for year in range(2000, 2023)} for county in crops_counties['NUTS_ID']}

    # Iterate over each grid cell
    for index, grid_cell in grid_gdf.iterrows():
        # Find the matching county for the current grid cell
        county_name = find_matching_county(grid_cell.geometry)
        
        if county_name:
            # Retrieve the soil moisture values for each year for the current grid cell
            soil_moisture_values = [grid_cell[str(year)] for year in range(2000, 2023)]
            
            # Append the soil moisture values to the corresponding county and year
            for year, value in zip(range(2000, 2023), soil_moisture_values):
                county_soil_moisture[county_name][str(year)].append(value)

    # Calculate the mean soil moisture value for each county for each year
    county_mean_soil_moisture = {county: {year: np.nanmean(values) for year, values in year_values.items()} for county, year_values in county_soil_moisture.items()}

    # Convert the dictionary to a DataFrame
    mean_soil_moisture_df = pd.DataFrame(county_mean_soil_moisture).T
    mean_soil_moisture_df.index.name = 'NUTS_ID'


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
    crops_counties_with_WSI.to_file(f"{file_path}_mapped.shp")
    crops_counties_with_WSI.to_csv(f"{file_path}_mapped.csv")



def find_relevant_years(water_stress_value, state):
    height = 500
    file_path = f'crop_yield/compareWSI/maximum_waterstress_{water_stress_value}_mapped.csv'
    df = pd.read_csv(file_path)
    df = df[df['NUTS_ID'] == state]
    years_cols = df.columns[3:]  # Assuming the first three columns are not years

    # Select only numeric columns
    numeric_cols = df[years_cols].select_dtypes(include=[float, int])

    # Select columns where all values are less than 10
    dropped_cols = numeric_cols.columns[(numeric_cols < height).all()]

    # Drop selected columns
    df.drop(labels=dropped_cols, axis=1, inplace=True)

    # Get the keys of the dropped columns
    dropped_cols_keys = dropped_cols.tolist()

    #print("Dropped columns keys:", dropped_cols_keys)
    return df, dropped_cols_keys

def drop_crop(state, crop, dropped_cols_keys):
    crop_data = pd.read_csv('crop_yield/detrended_data.csv')
    filtered_crop_data = crop_data[crop_data['crops'].str.contains(crop)].copy()
    filtered_crop_data_state = filtered_crop_data[filtered_crop_data['NUTS_ID'].str.contains(state)].copy()
    filtered_crop_data_state.drop(labels=dropped_cols_keys, axis=1, inplace=True)
    return filtered_crop_data_state


def plot_comparison():
    sns.set_theme()
    plt.rcParams.update({'font.size': 14})
    df = pd.read_csv('crop_yield/compareWSI/results.csv')
    # Create empty lists to store mean and standard deviation for each water stress index
    # Convert 'Correlation' column to numeric
    df['Pearson Correlation'] = pd.to_numeric(df['Pearson Correlation'], errors='coerce')
    # Group by 'crop_type' and 'Water Stress Index'
    nan_counts = df.groupby(['crop_type', 'Water Stress Index']).apply(lambda x: (x['Pearson Correlation'].isnull().sum() <= 5)).reset_index(name='Less Than 5 NaN')

    filtered_nan_counts = nan_counts[nan_counts['Less Than 5 NaN']]

    # Group by 'crop_type' and 'Water Stress Index' and calculate the mean
    mean_correlation = df.groupby(['crop_type', 'Water Stress Index'])['Pearson Correlation'].mean()
    #print('here')
    #print(df.groupby(['crop_type', 'Water Stress Index'])['Pearson Correlation'].values)
    meantolist = np.array(mean_correlation.tolist())
    std_correlation = df.groupby(['crop_type', 'Water Stress Index'])['Pearson Correlation'].std()
    stdtolist = np.array(std_correlation.tolist())
    i = 0
    color = 'black'
    for crop in ds.crop_types:
        water_stress_values = filtered_nan_counts[filtered_nan_counts['crop_type'] == crop]['Water Stress Index'].unique().tolist()
        missing_waterstess = len(water_stress_values)-2
        fig = plt.figure(figsize=(10, 6))
        plt.plot(water_stress_values, meantolist[i+missing_waterstess:i+14], marker='o', linestyle='-', color=color, label='Mean over all states')
        plt.fill_between(water_stress_values, meantolist[i+missing_waterstess:i+14]-stdtolist[i+missing_waterstess:i+14], meantolist[i+missing_waterstess:i+14]+stdtolist[i+missing_waterstess:i+14], alpha=0.2, label = r'1 $\sigma$ evironment')#, color=color)
        plt.hlines(-ds.crop_types_correlation[crop], 290, 390, color=color, linestyle='--', label='Neg. correlation for SM', alpha=0.5)
        for state in ds.important_states:
            df_state = df[df['NUTS_ID'] == state]
            df_crop = df_state[df_state['crop_type'] == crop]
            plt.plot(df_crop['Water Stress Index'], df_crop['Pearson Correlation'], label=ds.short_states[state], alpha=0.5, color=ds.colors[state])

        i += 14
        plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=5)
        plt.ylim(-0.7, 0.3)
        plt.xlim(315,395)
        plt.xlabel('Water Stress Index')
        plt.ylabel('Correlation')
        plt.title(f'Correlation for {ds.crop_short[crop]}', fontsize=25)
        plt.tight_layout()
        plt.savefig(f'crop_yield/compareWSI/correlationcomp_{crop}.png', dpi = 500)#, transparent = True)
        plt.show()

def plot_comp_comparison():
    sns.set_theme()
    plt.rcParams.update({'font.size': 14})
    df = pd.read_csv('crop_yield/compareWSI/results.csv')
    # Create empty lists to store mean and standard deviation for each water stress index
    # Convert 'Correlation' column to numeric
    df['Pearson Correlation'] = pd.to_numeric(df['Pearson Correlation'], errors='coerce')
    # Group by 'crop_type' and 'Water Stress Index'
    nan_counts = df.groupby(['crop_type', 'Water Stress Index']).apply(lambda x: (x['Pearson Correlation'].isnull().sum() <= 5)).reset_index(name='Less Than 5 NaN')

    filtered_nan_counts = nan_counts[nan_counts['Less Than 5 NaN']]

    # Group by 'crop_type' and 'Water Stress Index' and calculate the mean
    mean_correlation = df.groupby(['crop_type', 'Water Stress Index'])['Pearson Correlation'].mean()
    #print('here')
    #print(df.groupby(['crop_type', 'Water Stress Index'])['Pearson Correlation'].values)
    meantolist = np.array(mean_correlation.tolist())
    std_correlation = df.groupby(['crop_type', 'Water Stress Index'])['Pearson Correlation'].std()
    stdtolist = np.array(std_correlation.tolist())
    i = 0
    fig = plt.figure(figsize=(10, 6))
    colors = ['green', 'red', 'purple']
    cmap = LinearSegmentedColormap.from_list("mycmap", colors)
    for crop in ds.filal_plot_crops:
        water_stress_values = filtered_nan_counts[filtered_nan_counts['crop_type'] == crop]['Water Stress Index'].unique().tolist()
        missing_waterstess = len(water_stress_values)-2
        plt.plot(water_stress_values, meantolist[i+14+missing_waterstess:i+14+14], marker='o', linestyle='-', color=ds.crop_types_colors[crop], label=f'{ds.crop_short[crop]}')
        plt.fill_between(water_stress_values, meantolist[i+missing_waterstess+14:i+14+14]-stdtolist[i+14+missing_waterstess:i+14+14], meantolist[i+14+missing_waterstess:i+14+14]+stdtolist[i+missing_waterstess+14:i+14+14], alpha=0.2, color=ds.crop_types_colors[crop])
        i += 14
        plt.hlines(-ds.crop_types_correlation[crop], water_stress_values[0], 390, linestyle='--',  color=ds.crop_types_colors[crop]),#label=f'Average correlation for soilmoisture {ds.crop_short[crop]}')
    #plt.plot(water_stress_values, 390*np.ones_like(water_stress_values), linestyle='--', label='Average correlation for soilmoisture', color='grey')
    plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3)
    plt.ylim(-0.7, 0.3)
    plt.xlabel('Water Stress Index')
    plt.xlim(315,395)
    plt.ylabel('Correlation')
    plt.title(f'Correlation for remaining crops', fontsize=25)
    plt.tight_layout()
    plt.savefig(f'crop_yield/compareWSI/correlationcompcomp_{crop}.png', dpi = 500)#, transparent = True)
    plt.show()

    
WSI = np.arange(260,400, 10)

"""for i in WSI:
    calc_water_stress(i)
    map_to_state(i)
"""
"""results = pd.DataFrame(columns=['NUTS_ID', 'crop_type', 'Water Stress Index',  'Correlation'])
for state in ds.important_states:
    for crop in ds.crop_types:
        print(crop)
        for i in WSI:
            water_stressdf, dropped_keys = find_relevant_years(i, state)
            crop_data = drop_crop(state, crop, dropped_keys)
            wcorr = water_stressdf.iloc[0, 3:].values
            cropcorr = crop_data.iloc[0, 3:].values
            if len(wcorr) < 10:
                pearson_corr = None
                p = None
            else:
                pearson_corr, p = pearsonr(wcorr, cropcorr)
            print(pearson_corr, p)
            results = results._append({'NUTS_ID': state, 'crop_type': crop, 'Water Stress Index': i, 'Correlation': pearson_corr}, ignore_index=True)

results.to_csv('crop_yield/compareWSI/results.csv')"""


plot_comp_comparison()
plot_comparison()