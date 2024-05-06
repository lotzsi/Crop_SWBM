#crop yield import
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = 'data/crop_yield/estat_apro_cpsh1.tsv' # Crop production in EU standard humidity (apro_cpsh1)
file_path2 = 'data/crop_yield/estat_apro_cpshr.tsv' # Crop production in EU standard humidity by NUTS 2 regions (apro_cpshr)

data=pd.read_csv(file_path2 ,sep='\t')
print(data.head())

