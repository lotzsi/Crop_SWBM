import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from scipy.stats import pearsonr
import dictionary_states as ds
import seaborn as sns
#what do i want: a pandas dataframe with for each state and crop type: year, yiel, (soil moisture, temperature, precipitation, radiation) average and water stress index
def with_soil_moisture():
    results_df = pd.DataFrame(columns=['NUTS_ID', 'crop', 'Temperature', 'Precipitation', 'Radiation', 'Soil Moisture'])
    p_results = pd.DataFrame(columns=['NUTS_ID', 'crop', 'Temperature', 'Precipitation', 'Radiation', 'Soil Moisture'])
    crop = 'C0000'
    state = 'DE4'
    df = pd.DataFrame(columns=['year', 'yield', 'soil moisture', 'temperature', 'precipitation', 'radiation', 'water stress index'])
    df['year'] = np.arange(2000, 2023, 1)
    years = np.arange(2000, 2024, 1)
    yield_data = pd.read_csv('crop_yield/detrended_data.csv')
    soil_moisture_data = pd.read_csv('crop_yield/averaged/crop_yield_processed/soil_moisutre39.csv')
    temperature_data = pd.read_csv('crop_yield/averaged/crop_yield_processed/daily_average_temperature39.csv')
    precipitation_data = pd.read_csv('crop_yield/averaged/crop_yield_processed/total_precipitation39.csv')
    radiation_data = pd.read_csv('crop_yield/averaged/crop_yield_processed/net_radiation39.csv')
    wsi_data = pd.read_csv('crop_yield/averaged/crop_yield_processed/crop_yield_DE_WSI.csv')

    for state in ds.important_states:
        for crop in ds.crop_types:
            yield_values = yield_data[(yield_data['NUTS_ID'] == state) & (yield_data['crops'] == crop)].loc[:, '2000':'2022'].values
            soil_moisture_values = soil_moisture_data[(soil_moisture_data['NUTS_ID'] == state)].loc[:, '2000':'2022'].values
            temperature_values = temperature_data[(temperature_data['NUTS_ID'] == state)].loc[:, '2000':'2022'].values
            precipitation_values = precipitation_data[(precipitation_data['NUTS_ID'] == state)].loc[:, '2000':'2022'].values
            radiation_values = radiation_data[(radiation_data['NUTS_ID'] == state)].loc[:, '2000':'2022'].values
            wsi_values = wsi_data[(wsi_data['NUTS_ID'] == state)].loc[:, '2000':'2022'].values
            """print('T,SM', pearsonr(temperature_values.flatten(), soil_moisture_values.flatten()))   
            print('T, P', pearsonr(temperature_values.flatten(), precipitation_values.flatten()))
            print('T, R', pearsonr(temperature_values.flatten(), radiation_values.flatten()))
            print('T, WSI', pearsonr(temperature_values.flatten(), wsi_values.flatten()))
            print('SM, P', pearsonr(soil_moisture_values.flatten(), precipitation_values.flatten()))
            print('SM, R', pearsonr(soil_moisture_values.flatten(), radiation_values.flatten()))
            print('SM, WSI', pearsonr(soil_moisture_values.flatten(), wsi_values.flatten()))
            print('P, R', pearsonr(precipitation_values.flatten(), radiation_values.flatten()))
            print('P, WSI', pearsonr(precipitation_values.flatten(), wsi_values.flatten()))
            print('R, WSI', pearsonr(radiation_values.flatten(), wsi_values.flatten()))"""


            df['yield'] = yield_values.flatten()
            df['soil moisture'] = soil_moisture_values.flatten()
            df['temperature'] = temperature_values.flatten()
            df['precipitation'] = precipitation_values.flatten()
            df['radiation'] = radiation_values.flatten()
            df['water stress index'] = wsi_values.flatten()



            x = df[['temperature', 'precipitation', 'radiation', 'soil moisture']]#, 'water stress index']]
            y = df["yield"]
            # Using sklearn
            regression = linear_model.LinearRegression()
            regression.fit(x, y)
            predictions_sklearn = regression.predict(x)
            # Using statsmodels
            x = sm.add_constant(x)
            model = sm.OLS(y, x).fit()
            predictions_statsmodels = model.predict(x)
            summary = model.summary()
            t_values = model.tvalues
            p_values = model.pvalues
            res = {'NUTS_ID': state, 'crop': crop, 'Temperature': t_values['temperature'], 'Precipitation': t_values['precipitation'], 'Radiation': t_values['radiation'], 'Soil Moisture': t_values['soil moisture']}
            res_p = {'NUTS_ID': state, 'crop': crop, 'Temperature': p_values['temperature'], 'Precipitation': p_values['precipitation'], 'Radiation': p_values['radiation'], 'Soil Moisture': p_values['soil moisture']}
            new_row_df = pd.DataFrame(res, index=[0])  # Assuming the index should start from 0
            new_row_df_p = pd.DataFrame(res_p, index=[0])
            # Append the new row to the existing DataFrame
            results_df = results_df._append(new_row_df, ignore_index=True)
            p_results = p_results._append(new_row_df_p, ignore_index=True)


    # Group by 'crop' and calculate the mean and standard deviation
    # Group by 'crop' and calculate the mean and standard deviation
    mean_std_per_crop = p_results.groupby('crop').agg({'Temperature': ['mean', 'std'], 
                                                    'Precipitation': ['mean', 'std'], 
                                                    'Radiation': ['mean', 'std'], 
                                                    'Soil Moisture': ['mean', 'std']})

    # Flatten the multi-level columns
    mean_std_per_crop.columns = [' '.join(col).strip() for col in mean_std_per_crop.columns.values]

    # Format mean and standard deviation together
    for col in ['Temperature', 'Precipitation', 'Radiation', 'Soil Moisture']:
        mean_std_per_crop[col] = mean_std_per_crop[col + ' mean'].map("{:.2f}".format) + r" $\pm$ " + mean_std_per_crop[col + ' std'].map("{:.2f}".format)
        mean_std_per_crop.drop(columns=[col + ' mean', col + ' std'], inplace=True)

    # Convert mean and standard deviation to LaTeX format
    mean_std_latex = mean_std_per_crop.to_latex(column_format='|c|c|c|c|', 
                                                bold_rows=True, 
                                                escape=False)

    # Write the LaTeX code into the file
    with open("mean_std_table_wSM.tex", 'w') as f:
        f.write(mean_std_latex)
    df_melted = pd.melt(results_df, id_vars=['NUTS_ID', 'crop'], var_name='Variable', value_name='t_value')
    sns.set_theme()
    # Plot
    custom_palette = {"Temperature": "#7ABA78", "Precipitation": "#F3CA52", "Radiation": "#F6E9B2", "Soil Moisture": '#0A6847'}
    plt.figure(figsize=(14, 8))
    plt.rcParams.update({'font.size': 14})
    plt.hlines(0, -1, 4, color='grey', linestyle='--', alpha=0.3)
    sns.boxplot(data=df_melted, x='crop', y='t_value', hue='Variable', palette=custom_palette)
    my_ticks = ['Cereals', 'Wheat and Spelt', 'Potatoes', 'Sugar beet']
    plt.xticks(ticks=[0, 1, 2, 3], labels=my_ticks)
    #plt.xticks(rotation=45)
    plt.title("Multiple Linear Regression t-values for each crop type and variable")
    plt.xlabel('')

    plt.ylabel('t-value')
    #plt.legend(title='Variable')
    plt.legend(title='Variable', bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=4)
    plt.tight_layout()
    plt.savefig('crop_yield/Figures/t_values_withSM.png', dpi=500, transparent=True)
    plt.show()

#what do i want: a pandas dataframe with for each state and crop type: year, yiel, (soil moisture, temperature, precipitation, radiation) average and water stress index
def without_soil_moisture():
    results_df = pd.DataFrame(columns=['NUTS_ID', 'crop', 'Temperature', 'Precipitation', 'Radiation'])
    p_results = pd.DataFrame(columns=['NUTS_ID', 'crop', 'Temperature', 'Precipitation', 'Radiation'])
    crop = 'C0000'
    state = 'DE4'
    df = pd.DataFrame(columns=['year', 'yield', 'soil moisture', 'temperature', 'precipitation', 'radiation', 'water stress index'])
    df['year'] = np.arange(2000, 2023, 1)
    years = np.arange(2000, 2024, 1)
    yield_data = pd.read_csv('crop_yield/detrended_data.csv')
    soil_moisture_data = pd.read_csv('crop_yield/averaged/crop_yield_processed/soil_moisutre39.csv')
    temperature_data = pd.read_csv('crop_yield/averaged/crop_yield_processed/daily_average_temperature39.csv')
    precipitation_data = pd.read_csv('crop_yield/averaged/crop_yield_processed/total_precipitation39.csv')
    radiation_data = pd.read_csv('crop_yield/averaged/crop_yield_processed/net_radiation39.csv')
    wsi_data = pd.read_csv('crop_yield/averaged/crop_yield_processed/crop_yield_DE_WSI.csv')

    for state in ds.important_states:
        for crop in ds.crop_types:
            yield_values = yield_data[(yield_data['NUTS_ID'] == state) & (yield_data['crops'] == crop)].loc[:, '2000':'2022'].values
            soil_moisture_values = soil_moisture_data[(soil_moisture_data['NUTS_ID'] == state)].loc[:, '2000':'2022'].values
            temperature_values = temperature_data[(temperature_data['NUTS_ID'] == state)].loc[:, '2000':'2022'].values
            precipitation_values = precipitation_data[(precipitation_data['NUTS_ID'] == state)].loc[:, '2000':'2022'].values
            radiation_values = radiation_data[(radiation_data['NUTS_ID'] == state)].loc[:, '2000':'2022'].values
            wsi_values = wsi_data[(wsi_data['NUTS_ID'] == state)].loc[:, '2000':'2022'].values
            """print('T,SM', pearsonr(temperature_values.flatten(), soil_moisture_values.flatten()))   
            print('T, P', pearsonr(temperature_values.flatten(), precipitation_values.flatten()))
            print('T, R', pearsonr(temperature_values.flatten(), radiation_values.flatten()))
            print('T, WSI', pearsonr(temperature_values.flatten(), wsi_values.flatten()))
            print('SM, P', pearsonr(soil_moisture_values.flatten(), precipitation_values.flatten()))
            print('SM, R', pearsonr(soil_moisture_values.flatten(), radiation_values.flatten()))
            print('SM, WSI', pearsonr(soil_moisture_values.flatten(), wsi_values.flatten()))
            print('P, R', pearsonr(precipitation_values.flatten(), radiation_values.flatten()))
            print('P, WSI', pearsonr(precipitation_values.flatten(), wsi_values.flatten()))
            print('R, WSI', pearsonr(radiation_values.flatten(), wsi_values.flatten()))"""


            df['yield'] = yield_values.flatten()
            df['soil moisture'] = soil_moisture_values.flatten()
            df['temperature'] = temperature_values.flatten()
            df['precipitation'] = precipitation_values.flatten()
            df['radiation'] = radiation_values.flatten()
            df['water stress index'] = wsi_values.flatten()


            x = df[['temperature', 'precipitation', 'radiation']]#, 'water stress index']]
            y = df["yield"]
            # Using sklearn
            regression = linear_model.LinearRegression()
            regression.fit(x, y)
            predictions_sklearn = regression.predict(x)
            # Using statsmodels
            x = sm.add_constant(x)
            model = sm.OLS(y, x).fit()
            predictions_statsmodels = model.predict(x)
            summary = model.summary()
            t_values = model.tvalues
            p_values = model.pvalues
            res = {'NUTS_ID': state, 'crop': crop, 'Temperature': t_values['temperature'], 'Precipitation': t_values['precipitation'], 'Radiation': t_values['radiation']}
            res_p = {'NUTS_ID': state, 'crop': crop, 'Temperature': p_values['temperature'], 'Precipitation': p_values['precipitation'], 'Radiation': p_values['radiation']}
            new_row_df = pd.DataFrame(res, index=[0])  # Assuming the index should start from 0
            new_row_df_p = pd.DataFrame(res_p, index=[0])
            # Append the new row to the existing DataFrame
            results_df = results_df._append(new_row_df, ignore_index=True)
            p_results = p_results._append(new_row_df_p, ignore_index=True)


    # Group by 'crop' and calculate the mean and standard deviation
    # Group by 'crop' and calculate the mean and standard deviation
    mean_std_per_crop = p_results.groupby('crop').agg({'Temperature': ['mean', 'std'], 
                                                    'Precipitation': ['mean', 'std'], 
                                                    'Radiation': ['mean', 'std']})

    # Flatten the multi-level columns
    mean_std_per_crop.columns = [' '.join(col).strip() for col in mean_std_per_crop.columns.values]

    # Format mean and standard deviation together
    for col in ['Temperature', 'Precipitation', 'Radiation']:
        mean_std_per_crop[col] = mean_std_per_crop[col + ' mean'].map("{:.2f}".format) + r" $\pm$ " + mean_std_per_crop[col + ' std'].map("{:.2f}".format)
        mean_std_per_crop.drop(columns=[col + ' mean', col + ' std'], inplace=True)

    # Convert mean and standard deviation to LaTeX format
    mean_std_latex = mean_std_per_crop.to_latex(column_format='|c|c|c|c|', 
                                                bold_rows=True, 
                                                escape=False)

    # Write the LaTeX code into the file
    with open("mean_std_table_woSM.tex", 'w') as f:
        f.write(mean_std_latex)
    df_melted = pd.melt(results_df, id_vars=['NUTS_ID', 'crop'], var_name='Variable', value_name='t_value')

    # Plot
    sns.set_theme()
    custom_palette = {"Temperature": "#7ABA78", "Precipitation": "#F3CA52", "Radiation": "#F6E9B2", "Soil Moisture": '#0A6847'}
    plt.figure(figsize=(14, 8))
    plt.rcParams.update({'font.size': 14})
    plt.hlines(0, -1, 4, color='grey', linestyle='--', alpha=0.3)
    sns.boxplot(data=df_melted, x='crop', y='t_value', hue='Variable', palette=custom_palette)
    my_ticks = ['Cereals', 'Wheat and Spelt', 'Potatoes', 'Sugar beet']
    plt.xticks(ticks=[0, 1, 2, 3], labels=my_ticks)
    #plt.xticks(rotation=45)
    plt.title("Multiple Linear Regression t-values for each crop type and variable")
    plt.xlabel('')

    plt.ylabel('t-value')
    #plt.legend(title='Variable')
    plt.legend(title='Variable', bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=4)
    plt.tight_layout()
    plt.savefig('crop_yield/Figures/t_values_woSM.png', dpi=500, transparent=True)
    plt.show()

def correlation():
    
    results_df = pd.DataFrame(columns=['NUTS_ID', 'crop', 'Temperature', 'Precipitation', 'Radiation', 'Soil Moisture'])
    crop = 'C0000'
    state = 'DE4'
    df = pd.DataFrame(columns=['year', 'yield', 'soil moisture', 'temperature', 'precipitation', 'radiation', 'water stress index'])
    df['year'] = np.arange(2000, 2023, 1)
    years = np.arange(2000, 2024, 1)
    yield_data = pd.read_csv('crop_yield/detrended_data.csv')
    soil_moisture_data = pd.read_csv('crop_yield/averaged/crop_yield_processed/soil_moisutre39.csv')
    temperature_data = pd.read_csv('crop_yield/averaged/crop_yield_processed/daily_average_temperature39.csv')
    precipitation_data = pd.read_csv('crop_yield/averaged/crop_yield_processed/total_precipitation39.csv')
    radiation_data = pd.read_csv('crop_yield/averaged/crop_yield_processed/net_radiation39.csv')
    wsi_data = pd.read_csv('crop_yield/averaged/crop_yield_processed/crop_yield_DE_WSI.csv')
    correlation_matrix = np.zeros((4, 4))
    for state in ds.important_states:
        for crop in ds.crop_types:
            yield_values = yield_data[(yield_data['NUTS_ID'] == state) & (yield_data['crops'] == crop)].loc[:, '2000':'2022'].values
            soil_moisture_values = soil_moisture_data[(soil_moisture_data['NUTS_ID'] == state)].loc[:, '2000':'2022'].values
            temperature_values = temperature_data[(temperature_data['NUTS_ID'] == state)].loc[:, '2000':'2022'].values
            precipitation_values = precipitation_data[(precipitation_data['NUTS_ID'] == state)].loc[:, '2000':'2022'].values
            radiation_values = radiation_data[(radiation_data['NUTS_ID'] == state)].loc[:, '2000':'2022'].values
            wsi_values = wsi_data[(wsi_data['NUTS_ID'] == state)].loc[:, '2000':'2022'].values
            data = {
                'Soil Moisture': soil_moisture_values.flatten(),
                'Precipitation': precipitation_values.flatten(),
                'Temperature': temperature_values.flatten(),
                'Radiation': radiation_values.flatten(),


            }
            df = pd.DataFrame(data)

            # Compute the correlation matrix
            correlation_matrix = correlation_matrix + df.corr()
    correlation_matrix = correlation_matrix / len(ds.important_states)/len(ds.crop_types)
    correlation_matrix_sum = correlation_matrix.abs().sum()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # Plot the correlation matrix with masked upper triangle
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Matrix")
    plt.xlabel("Variables")
    plt.ylabel("Variables")
    plt.show()
    print(correlation_matrix)
    print(correlation_matrix_sum)
with_soil_moisture()
without_soil_moisture()
#correlation()