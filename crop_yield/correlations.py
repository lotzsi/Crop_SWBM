from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dictionary_states as ds

# Assuming 'crop_data' and 'soil_moisture' are your DataFrames

years = np.arange(2000, 2022, 1)
years_str = years.astype(str).tolist()

# Read crop_data and soil_moisture from CSV files
crop_data = pd.read_csv('crop_yield/detrended_data.csv')
soil_moisture = pd.read_csv('crop_yield/averaged/crop_yield_processed/soil_moisutre39.csv')

# Filter crop_data based on specific_string
specific_string = 'C0000'
filtered_crop_data = crop_data[crop_data['crops'].str.contains(specific_string)]
for state in ds.states:
# Filter crop_data further based on another_string
    another_string = state
    crop_data_filtered = filtered_crop_data[filtered_crop_data['NUTS_ID'].str.contains(another_string)]

    # Filter soil_moisture based on another_string
    soil_moisture_filtered = soil_moisture[soil_moisture['NUTS_ID'].str.contains(another_string)]
    y = crop_data_filtered[years_str].values.flatten()
    X = soil_moisture_filtered[years_str].values.flatten()
    # Create a scatter plot
    plt.scatter(crop_data_filtered[years_str].values.flatten(), soil_moisture_filtered[years_str].values.flatten())

    # Customize the plot
    plt.xlabel('Crop Data')
    plt.ylabel('soil moisture Data')
    plt.title('Crop Data vs soil moisture Data')

    # Show the plot
    #plt.show()

    print(ds.states[state],pearsonr(X, y)) 