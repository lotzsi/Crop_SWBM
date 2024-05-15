from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.inspection import PartialDependenceDisplay

# Assuming 'crop_data' and 'soil_moisture' are your DataFrames

years = np.arange(2000, 2022, 1)
years_str = years.astype(str).tolist()

# Read crop_data and soil_moisture from CSV files
crop_data = pd.read_csv('crop_yield/detrended_data.csv')
water_stress = pd.read_csv('data\crop_yield\crop_yield_DE_WSI.csv')
soil_moisture = pd.read_csv('crop_yield/averaged/crop_yield_processed/soil_moisutre39.csv')
T = pd.read_csv('crop_yield/averaged/crop_yield_processed/daily_average_temperature39.csv')
P = pd.read_csv('crop_yield/averaged/crop_yield_processed/total_precipitation39.csv')
R = pd.read_csv('crop_yield/averaged/crop_yield_processed/net_radiation39.csv')
L = pd.read_csv('crop_yield/averaged/crop_yield_processed/lai39.csv')

#results = pd.DataFrame(columns=['state', 'crop_type', 'R2', 'T', 'P', 'R'])
results = pd.DataFrame(columns=['state', 'crop_type', 'R2'])#, 'SM', 'T', 'P', 'R'])#, 'L'])
# Filter crop_data based on specific_string

for state in ds.states:
    for crop_type in ds.crop_types:
    # Filter crop data based on crop type
        filtered_crop_data = crop_data[crop_data['crops'].str.contains(crop_type)]
    # Filter crop_data further based on another_string
        another_string = state
        crop_data_filtered = filtered_crop_data[filtered_crop_data['NUTS_ID'].str.contains(another_string)]

        # Filter soil_moisture based on another_string
        soil_moisture_filtered = soil_moisture[soil_moisture['NUTS_ID'].str.contains(another_string)][years_str].values.flatten()
        water_stress_filtered = water_stress[water_stress['NUTS_ID'].str.contains(another_string)][years_str].values.flatten()
        T_filtered = T[T['NUTS_ID'].str.contains(another_string)][years_str].values.flatten()
        P_filtered = P[P['NUTS_ID'].str.contains(another_string)][years_str].values.flatten()
        R_filtered = R[R['NUTS_ID'].str.contains(another_string)][years_str].values.flatten()
        #L_filtered = L[L['NUTS_ID'].str.contains(another_string)][years_str].values.flatten()
        all_data = np.concatenate([soil_moisture_filtered])#,T_filtered, P_filtered, R_filtered])#, L_filtered])
        #all_data = np.concatenate([T_filtered, P_filtered, R_filtered])#, L_filtered])#([soil_moisture_filtered,T_filtered, P_filtered, R_filtered])#, L_filtered])
        y = crop_data_filtered[years_str].values.flatten()
        #X = all_data.reshape(-1, 3)
        X = all_data.reshape(-1, 1)



        
        # Initialize the Random Forest Regressor
        #model = RandomForestRegressor(n_estimators=100, random_state=42)
        model = RandomForestClassifier(min_samples_split=2, random_state=0)


        model.fit(X, y)

        # Compute cross-validated predictions
        y_pred_cv = cross_val_predict(model, X, y, cv=10)
        plt.scatter(y, y_pred_cv)
        plt.title(ds.states[state] + ' ' + ds.crop_types[crop_type] + ' Random Forest')
        plt.show()

        # Calculate R squared
        r2 = r2_score(y, y_pred_cv)
        # Get feature importances

        new_row = {'state': state, 'crop_type':crop_type,'R2': r2}#, 'SM': importances[0])#, 'T': importances[1], 'P': importances[2], 'R': importances[3]}
        #new_row = {'state': state, 'crop_type':crop_type,'R2': r2, 'T': importances[0], 'P': importances[1], 'R': importances[2]}
        results = results._append(new_row, ignore_index=True)



print(results)
results.to_csv('crop_yield/R2_justSM.csv', index=False)