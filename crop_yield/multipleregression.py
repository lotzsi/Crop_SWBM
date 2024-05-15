import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from scipy.stats import pearsonr
import dictionary_states as ds
import seaborn as sns
#what do i want: a pandas dataframe with for each state and crop type: year, yiel, (soil moisture, temperature, precipitation, radiation) average and water stress index
results_df = pd.DataFrame(columns=['NUTS_ID', 'crop', 'Temperature', 'Precipitation', 'Radiation'])
crop = 'C0000'
state = 'DE4'
df = pd.DataFrame(columns=['year', 'yield', 'soil moisture', 'temperature', 'precipitation', 'radiation', 'water stress index'])
df['year'] = np.arange(2000, 2023, 1)
years = np.arange(2000, 2024, 1)
yield_data = pd.read_csv('crop_yield/detrended_data.csv')
soil_moisture_data = pd.read_csv('crop_yield/averaged/crop_yield_processed/soil_moisutre39.csv')
temperature_data = pd.read_csv('crop_yield/averaged/crop_yield_processed/daily_average_temperature39.csv')
precipitation_data = pd.read_csv('crop_yield/averaged/crop_yield_processed/total_precipitation39.csv')
radiation_data = pd.read_csv('crop_yield/averaged/crop_yield_processed/net_radiation39.csv')
wsi_data = pd.read_csv('crop_yield/averaged/crop_yield_processed/crop_yield_DE_WSI.csv')

for state in ds.important_states:
    for crop in ds.crop_types:
        yield_values = yield_data[(yield_data['NUTS_ID'] == state) & (yield_data['crops'] == crop)].loc[:, '2000':'2022'].values
        soil_moisture_values = soil_moisture_data[(soil_moisture_data['NUTS_ID'] == state)].loc[:, '2000':'2022'].values
        temperature_values = temperature_data[(temperature_data['NUTS_ID'] == state)].loc[:, '2000':'2022'].values
        precipitation_values = precipitation_data[(precipitation_data['NUTS_ID'] == state)].loc[:, '2000':'2022'].values
        radiation_values = radiation_data[(radiation_data['NUTS_ID'] == state)].loc[:, '2000':'2022'].values
        wsi_values = wsi_data[(wsi_data['NUTS_ID'] == state)].loc[:, '2000':'2022'].values
        """print('T,SM', pearsonr(temperature_values.flatten(), soil_moisture_values.flatten()))   
        print('T, P', pearsonr(temperature_values.flatten(), precipitation_values.flatten()))
        print('T, R', pearsonr(temperature_values.flatten(), radiation_values.flatten()))
        print('T, WSI', pearsonr(temperature_values.flatten(), wsi_values.flatten()))
        print('SM, P', pearsonr(soil_moisture_values.flatten(), precipitation_values.flatten()))
        print('SM, R', pearsonr(soil_moisture_values.flatten(), radiation_values.flatten()))
        print('SM, WSI', pearsonr(soil_moisture_values.flatten(), wsi_values.flatten()))
        print('P, R', pearsonr(precipitation_values.flatten(), radiation_values.flatten()))
        print('P, WSI', pearsonr(precipitation_values.flatten(), wsi_values.flatten()))
        print('R, WSI', pearsonr(radiation_values.flatten(), wsi_values.flatten()))"""


        df['yield'] = yield_values.flatten()
        df['soil moisture'] = soil_moisture_values.flatten()
        df['temperature'] = temperature_values.flatten()
        df['precipitation'] = precipitation_values.flatten()
        df['radiation'] = radiation_values.flatten()
        df['water stress index'] = wsi_values.flatten()



        x = df[['temperature', 'precipitation', 'radiation']]#, 'soil moisture']]#, 'water stress index']]
        y = df["yield"]
        # Using sklearn
        regression = linear_model.LinearRegression()
        regression.fit(x, y)
        predictions_sklearn = regression.predict(x)
        # Using statsmodels
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        predictions_statsmodels = model.predict(x)
        summary = model.summary()
        t_values = model.tvalues
        res = {'NUTS_ID': state, 'crop': crop, 'Temperature': t_values['temperature'], 'Precipitation': t_values['precipitation'], 'Radiation': t_values['radiation']}
        new_row_df = pd.DataFrame(res, index=[0])  # Assuming the index should start from 0
        # Append the new row to the existing DataFrame
        results_df = results_df._append(new_row_df, ignore_index=True)



print(results_df)

df_melted = pd.melt(results_df, id_vars=['NUTS_ID', 'crop'], var_name='Variable', value_name='t_value')

# Plot
custom_palette = {"Temperature": "#7ABA78", "Precipitation": "#F3CA52", "Radiation": "#F6E9B2"}
plt.figure(figsize=(14, 8))
plt.rcParams.update({'font.size': 14})
plt.hlines(0, -1, 4, color='grey', linestyle='--', alpha=0.3)
sns.boxplot(data=df_melted, x='crop', y='t_value', hue='Variable', palette=custom_palette)
my_ticks = ['Cereals', 'Wheat and Spelt', 'Potatoes', 'Sugar beet']
plt.xticks(ticks=[0, 1, 2, 3], labels=my_ticks)
#plt.xticks(rotation=45)
plt.title("Multiple Linear Regression t-values for each crop type and variable")
plt.xlabel('')

plt.ylabel('t-value')
plt.legend(title='Variable')
plt.tight_layout()
plt.savefig('crop_yield/Figures/t_values.png', dpi=500, transparent=True)
plt.show()