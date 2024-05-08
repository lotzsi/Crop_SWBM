import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dictionary_states as ds


file_path = 'crop_yield/crop_yield_DE.csv'
data = pd.read_csv(file_path)

years = np.arange(2000, 2023, 1)

# Increase the size of the window
plt.figure(figsize=(10, 6))

for i in range(len(data['NUTS_ID'])):
    if data['NUTS_ID'][i] in ds.south_states:
        plt.plot(years,data.iloc[i, 2:-2], label=ds.states[data['NUTS_ID'][i]], color=ds.south_states[data['NUTS_ID'][i]])
    if data['NUTS_ID'][i] in ds.middle_states:
        plt.plot(years,data.iloc[i, 2:-2], label=ds.states[data['NUTS_ID'][i]], color=ds.middle_states[data['NUTS_ID'][i]]) 
    if data['NUTS_ID'][i] in ds.north_states:
        plt.plot(years,data.iloc[i, 2:-2], label=ds.states[data['NUTS_ID'][i]], color=ds.north_states[data['NUTS_ID'][i]])
    if data['NUTS_ID'][i] in ds.city_state:
        plt.plot(years,data.iloc[i, 2:-2], label=ds.states[data['NUTS_ID'][i]], color=ds.city_state[data['NUTS_ID'][i]])


# Place legend to the right of the plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Rotate x-axis ticks by 45 degrees
plt.xticks(rotation=45)



plt.show()



