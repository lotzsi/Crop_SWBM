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
results = pd.DataFrame(columns=['state', 'crop_type', 'R2', 'SM', 'T', 'P', 'R'])#, 'L'])
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
        all_data = np.concatenate([soil_moisture_filtered,T_filtered, P_filtered, R_filtered])#, L_filtered])
        #all_data = np.concatenate([T_filtered, P_filtered, R_filtered])#, L_filtered])#([soil_moisture_filtered,T_filtered, P_filtered, R_filtered])#, L_filtered])
        y = crop_data_filtered[years_str].values.flatten()
        #X = all_data.reshape(-1, 3)
        X = all_data.reshape(-1, 4)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        plt.title(ds.states[state] + ' ' + ds.crop_types[crop_type])

        # Plot Crop Data on the primary y-axis (left)
        ax1.plot(years, crop_data_filtered[years_str].values.flatten(), label='Crop Data', color='blue')
        ax1.set_xlabel('Years')
        ax1.set_ylabel('Crop Data', color='blue')

        # Create a twin Axes sharing the same x-axis
        ax2 = ax1.twinx()

        # Plot Soil Moisture Data on the secondary y-axis (right)
        ax2.plot(years, soil_moisture_filtered, label='Soil Moisture Data', color='red')
        ax2.plot(years, water_stress_filtered/np.max(water_stress_filtered)*np.max(soil_moisture_filtered), label='Water Stress Data (normalized)', color='black')
        ax2.set_ylabel('Soil Moisture Data', color='red')

        # Plot the legend outside the plot
        plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1))
        plt.tight_layout()
        plt.savefig('crop_yield/comparison_cysm/' + ds.states[state] + '_' + ds.crop_types[crop_type] + '.png', dpi=500, transparent=True)
        plt.clf()
        #plt.show()


        
        # Initialize the Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)

        model.fit(X, y)

        # Compute cross-validated predictions
        y_pred_cv = cross_val_predict(model, X, y, cv=10)
        """
        # Specify the features for which you want to compute partial dependence
        features = [0, 1, 2]  # Example features: feature 0, feature 1, feature 2

        # Generate partial dependence plots
        display = PartialDependenceDisplay.from_estimator(model, X, features)
        display.plot()
        plt.tight_layout()
        plt.show()"""


        # Calculate R squared
        r2 = r2_score(y, y_pred_cv)
        #print("R2:", r2)
        model.fit(X,y)
        # Get feature importances
        importances = model.feature_importances_

        # Get indices of features sorted by importance
        indices = np.argsort(importances)[::-1]
        new_row = {'state': state, 'crop_type':crop_type,'R2': r2, 'SM': importances[0], 'T': importances[1], 'P': importances[2], 'R': importances[3]}
        #new_row = {'state': state, 'crop_type':crop_type,'R2': r2, 'T': importances[0], 'P': importances[1], 'R': importances[2]}
        results = results._append(new_row, ignore_index=True)
        features = ['SM', 'T', 'P', 'R']#, 'L']
        #features = ['T', 'P', 'R']#, 'L']
        """# Print feature ranking
        print("Feature ranking for %s:" %(ds.states[state]))
        for f in range(X.shape[1]):
            print("%d. Feature %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))"""


print(results)
results.to_csv('crop_yield/feature_importance_wSM.csv', index=False)