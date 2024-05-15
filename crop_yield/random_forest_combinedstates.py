from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
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
column_names = [str(year) for year in years]


# Read crop_data and soil_moisture from CSV files
crop_data = pd.read_csv('crop_yield/detrended_data.csv')
soil_moisture = pd.read_csv('crop_yield/averaged/crop_yield_processed/soil_moisutre39.csv')
T = pd.read_csv('crop_yield/averaged/crop_yield_processed/daily_average_temperature39.csv')
P = pd.read_csv('crop_yield/averaged/crop_yield_processed/total_precipitation39.csv')
R = pd.read_csv('crop_yield/averaged/crop_yield_processed/net_radiation39.csv')
L = pd.read_csv('crop_yield/averaged/crop_yield_processed/lai39.csv')


# Filter crop_data based on specific_string
specific_string = 'C0000'
filtered_crop_data = crop_data[crop_data['crops'].str.contains(specific_string)]
crop_data_filtered = filtered_crop_data[column_names].values.flatten()
# Filter soil_moisture based on another_string
soil_moisture_filtered = soil_moisture[column_names].values.flatten()
T_filtered = T[column_names].values.flatten()
P_filtered = P[column_names].values.flatten()
R_filtered = R[column_names].values.flatten()
all_data = np.concatenate([T_filtered, P_filtered, R_filtered])#, L_filtered])#([soil_moisture_filtered,T_filtered, P_filtered, R_filtered])#, L_filtered])
plt.plot(crop_data_filtered, soil_moisture_filtered, 'o')
plt.xlabel('Crop Data')
plt.ylabel('soil moisture Data')
plt.title('Crop Data vs soil moisture Data')
plt.show()
y = crop_data_filtered
X = all_data.reshape(-1, 3)

    
    # Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

y_pred_cv = cross_val_predict(model, X, y, cv=10)

    # Calculate R squared
r2 = r2_score(y, y_pred_cv)
print("R2:", r2)
model.fit(X,y)
    # Get feature importances
importances = model.feature_importances_

    # Get indices of features sorted by importance
indices = np.argsort(importances)[::-1]
features = ['SM', 'T', 'P', 'R']#, 'L']
    # Print feature ranking
print("Feature ranking for")
for f in range(X.shape[1]):
    print("%d. Feature %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

