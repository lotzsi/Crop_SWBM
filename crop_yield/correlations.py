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
import seaborn as sns

# Assuming 'crop_data' and 'soil_moisture' are your DataFrames

years = np.arange(2000, 2022, 1)
years_str = years.astype(str).tolist()

crop_data = pd.read_csv('crop_yield/detrended_data.csv')
soil_moisture = pd.read_csv('crop_yield/averaged/crop_yield_processed/soil_moisutre39.csv')

# Create an empty DataFrame to store correlation results
correlation_results = pd.DataFrame(columns=[crops for crops in ds.crop_types])

for crop_type in ds.crop_types:
    # Filter crop data based on crop type
    filtered_crop_data = crop_data[crop_data['crops'].str.contains(crop_type)]
    
    for state in ds.states:
        # Filter crop data and soil moisture based on state
        filtered_crop_data_state = filtered_crop_data[filtered_crop_data['NUTS_ID'].str.contains(state)]
        soil_moisture_state = soil_moisture[soil_moisture['NUTS_ID'].str.contains(state)]
        
        # Extract data for correlation
        y = filtered_crop_data_state[years_str].values.flatten()
        X = soil_moisture_state[years_str].values.flatten()
        
        # Calculate Pearson correlation coefficient
        pearson_corr, _ = pearsonr(X, y)
        
        # Fill the correlation result into the DataFrame
        correlation_results.loc[state, crop_type] = pearson_corr

# Reset the index to make 'state' a column
correlation_results.reset_index(inplace=True)
correlation_results.rename(columns={'index': 'state'}, inplace=True)
correlation_results.dropna(inplace=True)

print(correlation_results)
correlation_results.set_index('state', inplace=True)


ig, ax = plt.subplots(figsize=(16, 8))

# Number of states and crop types
num_states = len(correlation_results)
num_crop_types = len(correlation_results.columns)

# Width of each bar
bar_width = 0.25

# Spacing between groups
spacing = 0.0001

# Generate x positions for each group
x_positions = np.arange(num_states)*2
sns.set_theme('paper')
# Iterate over crop types
for i, crop_type in enumerate(correlation_results.columns):
    # Calculate x positions for bars in this crop type group
    x_positions_crop = x_positions + (i - num_crop_types // 2) * (bar_width + spacing)
    
    # Plot bars for each state
    ax.bar(x_positions_crop, correlation_results[crop_type], width=bar_width, label=ds.crop_types[crop_type])

# Set x-axis ticks and labels
ax.set_xticks(x_positions)
print(correlation_results.index.values)
ax.set_xticklabels([ds.states[state] for state in correlation_results.index.values], rotation=90)


# Add legend
ax.legend(title='Crop Type', loc='upper left', bbox_to_anchor=(1, 1))

# Add labels and title
ax.set_xlabel('State')
ax.set_ylabel('Correlation Value')
ax.set_title('Correlation Values by Crop Type and State')

plt.tight_layout()
plt.savefig('crop_yield/Figures/correlation_values.png', dpi=500, transparent=True)
plt.show()