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
for i in range(len(data)):
    for j in range(4, len(data[i])):
        if data[i][j] == 'e' or data[i][j] == 'b' or data[i][j] == 'p' or data[i][j] == 'd' or data[i][j] == 'u' or data[i][j] == 'c' or data[i][j] == 'cd':
            data[i][j] = np.nan

# Join the processed data back into lines, converting NaN values to empty strings
processed_lines = [','.join(str(val) if not pd.isna(val) else '' for val in row) for row in data]

# Write the processed data back to a new file
processed_file_path = "data/crop_yield/apro_cpshr_clean.txt"
with open(processed_file_path, 'w') as file:
    file.write('\n'.join(processed_lines))

# Open the file and read its contents
with open("data/crop_yield/apro_cpshr_clean.txt", 'r') as file:
    file_content = file.read()

# Replace ',,' with ',' in the file content
file_content = file_content.replace(',,', ',')

# Write the modified content back to the file
with open("data/crop_yield/apro_cpshr_clean.txt", 'w') as file:
    file.write(file_content)

# Open the file and read its contents
with open("data/crop_yield/apro_cpshr_clean.txt", 'r') as file:
    file_content = file.readlines()

# Replace ',,' with ',' in the file content
file_content = [line.rstrip(',\n') + '\n' if line.strip() else line for line in file_content]

# Write the modified content back to the file
with open("data/crop_yield/apro_cpshr_clean.txt", 'w') as file:
    file.writelines(file_content)
