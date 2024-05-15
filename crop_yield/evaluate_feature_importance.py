import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import dictionary_states as ds

results = pd.read_csv('crop_yield/feature_importance.csv')
heatmap_data = results.pivot(index='state', columns='crop_type', values='R2')

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', linewidths=0.5, linecolor='black', xticklabels=[ds.crop_types[type] for type in heatmap_data.columns], yticklabels=[ds.states[st] for st in heatmap_data.index])
plt.title('R2 Values by State and Crop Type')
plt.xlabel('Crop Type')
plt.ylabel('State')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('crop_yield/Figures/feature_importance_heatmap.png', dpi=500, transparent=True)
plt.show()


features = ['T', 'P', 'R']#, 'L']
for crop_type in ds.crop_types:
    # Filter results based on crop type
    results_crop = results[results['crop_type'] == crop_type]
    
    # Get the feature importances for this crop type
    feature_importances = results_crop.set_index('state')[features]
    
    # Plot the feature importances
    plt.figure(figsize=(10, 8))
    feature_importances.plot(kind='bar', stacked=True, figsize=(10, 8))
    my_tocks = [ds.states[state] for state in feature_importances.index]
    plt.xticks(np.arange(len(feature_importances.index)), my_tocks)
    plt.title(f'Feature Importances for {ds.crop_types[crop_type]}')
    plt.xlabel('State')
    plt.ylabel('Feature Importance')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'crop_yield/Figures/feature_importance_{crop_type}.png', dpi=500, transparent=True)
    plt.show()