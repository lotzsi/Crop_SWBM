import matplotlib.pyplot as plt
import numpy as np
'Use as helper to test snow function'


def snow_function(Snow_n, P_n, T_n,
        c_m):  # Returns 1. Amount of snow after a time step and 2. Amount of water coming into the soil (liquid rain and/or melting snow)
    if T_n <= 273.15:  # Snow stays on the ground
        return Snow_n + P_n, 0
    elif Snow_n < 0.001:  # no accumulated snow and temperature above 0 degrees -> return "0 accumulated snow" and treat precipitation as rain
        return 0, P_n
    else:  # Snow is melting (if there was snow)
        SnowMelt = c_m * (T_n - 273.15)  # Amount of snow melting (if there was)
        if SnowMelt > Snow_n:  # Is the amount of snow that would melt larger than the existing amount of snow?
            return 0, Snow_n + P_n  # no snow remains, all existing snow melts
        else:
            return Snow_n - SnowMelt, SnowMelt + P_n  # Some snow remains, some snow melts
        

def snow_function(Snow_n, P_n, T_n, c_m):
    # Element-wise comparison for temperature below freezing
    more_snow_condition = T_n <= 273.15
    melting_snow_condition = np.logical_and(T_n <= 273.15, Snow_n < 0.001)

    # Element-wise comparison for negligible snow accumulation and temperature above freezing
    no_snow_condition = np.logical_and(Snow_n < 0.001, ~snow_condition)

    # Element-wise snow melting calculation
    SnowMelt = c_m * (T_n - 273.15)

    # Conditions for snow melting
    snow_melting_condition = ~no_snow_condition#, SnowMelt > Snow_n)
    if np.any(snow_melting_condition):
        print('snow_melting_condition\n', snow_melting_condition)

    # Calculate results based on conditions
    #snow_result = np.where(snow_melting_condition, 0, np.where(no_snow_condition, 0, np.maximum(Snow_n - SnowMelt, 0)))
    snow_result = np.where(snow_melting_condition, 0, np.where(snow_condition, 0, np.maximum(Snow_n - SnowMelt, 0)))

    water_infiltration = np.where(snow_melting_condition, Snow_n + P_n, np.where(no_snow_condition, P_n, SnowMelt + P_n))
    if np.any(snow_condition):
        print('should be new snow \n', snow_condition)
        print('snow results\n', snow_result)
    return snow_result, water_infiltration

P_n = np.array([[5, 5], [5, 5]])
T_n = np.array([[273-2, 273-2], [273+2, 273+5]])
Snow_n = np.array([[0, 5], [0, 5]])
c_m = 0.5

snow_function(Snow_n, P_n, T_n, c_m)

