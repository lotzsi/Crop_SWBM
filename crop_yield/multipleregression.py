import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

#what do i want: a pandas dataframe with for each state and crop type: year, yiel, (soil moisture, temperature, precipitation, radiation) average and water stress index

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


yield_values = yield_data[(yield_data['NUTS_ID'] == state) & (yield_data['crops'] == crop)].loc[:, '2000':'2022'].values
soil_moisture_values = soil_moisture_data[(soil_moisture_data['NUTS_ID'] == state)].loc[:, '2000':'2022'].values
temperature_values = temperature_data[(temperature_data['NUTS_ID'] == state)].loc[:, '2000':'2022'].values
precipitation_values = precipitation_data[(precipitation_data['NUTS_ID'] == state)].loc[:, '2000':'2022'].values
radiation_values = radiation_data[(radiation_data['NUTS_ID'] == state)].loc[:, '2000':'2022'].values
wsi_values = wsi_data[(wsi_data['NUTS_ID'] == state)].loc[:, '2000':'2022'].values

df['yield'] = yield_values.flatten()
df['soil moisture'] = soil_moisture_values.flatten()
df['temperature'] = temperature_values.flatten()
df['precipitation'] = precipitation_values.flatten()
df['radiation'] = radiation_values.flatten()
df['water stress index'] = wsi_values.flatten()

x = df[['soil moisture', 'temperature', 'precipitation', 'radiation']]#, 'water stress index']]
y = df["yield"]
print(y)
# Using sklearn
regression = linear_model.LinearRegression()
regression.fit(x, y)
predictions_sklearn = regression.predict(x)
print("Intercept: \n", regression.intercept_)
print("Coefficients: \n", regression.coef_)

# Using statsmodels
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
predictions_statsmodels = model.predict(x)
summary = model.summary()
print(summary)



