import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dictionary_states as ds


file_path = 'crop_yield/crop_yield_DE.csv'
data = pd.read_csv(file_path)

years = np.arange(2000, 2023, 1)

# Increase the size of the window
plt.figure(figsize=(12, 6))

for i in range(len(data['NUTS_ID'])):
        plt.plot(years,data.iloc[i, 2:-2], label=ds.states[data['NUTS_ID'][i]], color=ds.colors[data['NUTS_ID'][i]])


# Place legend to the right of the plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Rotate x-axis ticks by 45 degrees
plt.xticks(rotation=45)
plt.xlabel('Year')
plt.ylabel('Combined Crop Yield (tonnes per hectare)')

plt.savefig('Figures/states.png', bbox_inches='tight', transparent=True)
plt.show()



