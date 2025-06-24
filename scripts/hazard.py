# exposure.py ---> hazard.py ---> vulnerability-landd-weight.py  ---> governmentresponse.py 

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np
import os
import warnings
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import anderson


# Suppress all warnings
warnings.filterwarnings("ignore")
path = os.getcwd() + r"/flood-data-ecosystem-Odisha"

# Load data
master_variables = pd.read_csv(os.getcwd()+'/data/MASTER_VARIABLES.csv')
hazard_vars = ['inundation_intensity_mean_nonzero', 'inundation_intensity_sum', 'mean_rain', 'max_rain','drainage_density', 'Sum_Runoff', 'Peak_Runoff','slope_mean','elevation_mean','distance_from_river','distance_from_sea']
hazard_df = master_variables[hazard_vars + ['timeperiod', 'object_id']]
hazard_df_months = []


# Define categories for hazard levels
categories = [1, 2, 3, 4, 5]
def custom_binning(df, var):
    conditions = [
        (df[var] == 0),
        (df[var] > 0) & (df[var] <= df[var].quantile(0.25)),
        (df[var] > df[var].quantile(0.25)) & (df[var] <= df[var].quantile(0.5)),
        (df[var] > df[var].quantile(0.5)) & (df[var] <= df[var].quantile(0.75)),
        (df[var] > df[var].quantile(0.75))
    ]
    return np.select(conditions, categories, default=0)

reversed_categories = [5,4,3,2,1]
def custom_binning_reversed(df, var):
    conditions = [
        (df[var] < df[var].quantile(0.25)),  # Below the 25th percentile
        (df[var] >= df[var].quantile(0.25)) & (df[var] < df[var].quantile(0.5)),  # 25th to 50th percentile
        (df[var] >= df[var].quantile(0.5)) & (df[var] < df[var].quantile(0.75)),  # 50th to 75th percentile
        (df[var] >= df[var].quantile(0.75)) & (df[var] < df[var].quantile(1.0)),  # 75th to 100th percentile (excluding max)
        (df[var] >= df[var].quantile(1.0))  # Including max value
    ]
    return np.select(conditions, categories, default=0)


# Method 2: Log Transformation with Quantile Binning
def log_quantile_binning(df, var):
    # Add a small constant to avoid log(0)
    log_var = np.log1p(df[var])
    return pd.qcut(log_var, q=5, labels=categories, duplicates='drop')

# Processing monthly data
for month in tqdm(hazard_df.timeperiod.unique()):
    hazard_df_month = hazard_df[hazard_df.timeperiod == month]

    # Apply custom binning based on value ranges
    hazard_df_month['inundation_intensity_sum_binned'] = custom_binning(hazard_df_month, 'inundation_intensity_sum')
    hazard_df_month['inundation_intensity_mean_nonzero_binned'] = custom_binning(hazard_df_month, 'inundation_intensity_mean_nonzero')
    hazard_df_month['drainage_density_binned'] = custom_binning(hazard_df_month, 'drainage_density')
    hazard_df_month['mean_rain_binned'] = custom_binning(hazard_df_month, 'mean_rain')
    hazard_df_month['max_rain_binned'] = custom_binning(hazard_df_month, 'max_rain')
    hazard_df_month['Sum_Runoff_binned'] = custom_binning(hazard_df_month, 'Sum_Runoff')
    hazard_df_month['Peak_Runoff_binned'] = custom_binning(hazard_df_month, 'Peak_Runoff')
    hazard_df_month['slope_mean_binned'] = custom_binning(hazard_df_month, 'slope_mean')
    hazard_df_month['elevation_mean_binned'] = custom_binning_reversed(hazard_df_month, 'elevation_mean')
    hazard_df_month['distance_from_river_mean_binned'] = custom_binning_reversed(hazard_df_month, 'distance_from_river')
    
    # Average hazard score
    hazard_df_month['flood-hazard'] = (hazard_df_month[['inundation_intensity_sum_binned','inundation_intensity_mean_nonzero_binned','drainage_density_binned', 'mean_rain_binned', 
                                                        'max_rain_binned', 'Sum_Runoff_binned',
                                                        'Peak_Runoff_binned','slope_mean_binned','elevation_mean_binned','distance_from_river_mean_binned']]
                                       .astype(float).mean(axis=1))
    hazard_df_month['flood-hazard'] = round(hazard_df_month['flood-hazard'])

    hazard_df_months.append(hazard_df_month)

# Compile results
hazard = pd.concat(hazard_df_months)
master_variables = master_variables.merge(hazard[['timeperiod', 'object_id', 'flood-hazard']], on=['timeperiod', 'object_id'])


# Save the final results
master_variables.to_csv(os.getcwd() + r'/data/factor_scores_l1_hazard.csv', index=False)
