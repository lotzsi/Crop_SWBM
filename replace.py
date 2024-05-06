with open('data/crop_yield/estat_apro_cpshr.tsv', "rt") as fin:
    with open('data/crop_yield/apro_cpshr.txt', "wt") as fout:
        for i, line in enumerate(fin):
            # Remove '\TIME_PERIOD' only from the first column (i.e., the header row)
            if i == 0:
                line = line.replace('\\TIME_PERIOD', '')
            # Replace ':' with '0' in all columns
            line = line.replace(':', '0')
            line = line.replace(' ', ';')
            line = line.replace(',', ';')
            line = line.replace(';', ',')
            fout.write(line)



import pandas as pd
import numpy as np

# Define the data file path
file_path = "data/crop_yield/apro_cpshr.txt"

# Read the CSV file into a list of lines
with open(file_path, 'r') as file:
    lines = file.readlines()

# Remove newline characters and split each line into a list of values
data = [line.strip().split(',') for line in lines]

# Replace 'e' and 'b' values with NaN in the year columns (index 4 onwards)
fo