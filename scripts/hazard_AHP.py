import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm 
import os
import warnings

# Note - comment out the "weights" 
# Pairwise comparison matrix
'''
pairwise_matrix = np.array([
    [1, 2, 3, 5, 7],
    [0.5, 1, 3, 5, 7],
    [0.33, 0.33, 1, 2, 5],
    [0.2, 0.2, 0.5, 1, 5],
    [0.14, 0.14, 0.2, 0.2, 1]
])

# Step 1: Normalize the pairwise matrix
normalized_matrix = pairwise_matrix / pairwise_matrix.sum(axis=0)

# Step 2: Calculate the priority vector (weights)
priority_vector = normalized_matrix.mean(axis=1)
weights = {
    'elevation_mean': priority_vector[0],
    'slope_mean': priority_vector[1],
    'distance-from-river-mean': priority_vector[2],
    'mean_rain': priority_vector[3],
    'Mean_Daily_Runoff': priority_vector[4]
}

# Normalize weights to ensure they sum to 1
total_weight = sum(weights.values())
weights = {k: v / total_weight for k, v in weights.items()}

print("AHP-derived weights:", weights)
'''

# Suppress all warnings
warnings.filterwarnings("ignore")
path = os.getcwd() + r"/flood-data-ecosystem-Odisha"

master_variables = pd.read_csv(path+'/RiskScoreModel/data/MASTER_VARIABLES.csv')

hazard_vars = ['sum_rain', 'Mean_Daily_Runoff','elevation_mean','distance_from_sea']#'slope_mean','distance_from_river',

hazard_df = master_variables[hazard_vars + ['timeperiod', 'object_id']]
weights = {
    'elevation_mean': 0.22,
    #'slope_mean': 0.19,
    'distance_from_sea': 0.11,
    #'distance_from_river': 0.11,

    'sum_rain': 0.08,
    'Mean_Daily_Runoff': 0.03  # Adjust to match the variable name if different
}

total_weight = sum(weights.values())
weights = {k: v / total_weight for k, v in weights.items()}

hazard_df_months = []
for month in tqdm(hazard_df.timeperiod.unique()):
    scaler = MinMaxScaler()
    hazard_df = master_variables[hazard_vars + ['timeperiod', 'object_id']]
    hazard_df_month = hazard_df[hazard_df.timeperiod == month]
    hazard_df_month[hazard_vars] = scaler.fit_transform(hazard_df_month[hazard_vars])

    hazard_df_month['elevation_mean'] = 1 - hazard_df_month['elevation_mean']
    #hazard_df_month['slope_mean'] = 1 - hazard_df_month['slope_mean']
    hazard_df_month['distance_from_sea'] = 1 - hazard_df_month['distance_from_sea']

    # Calculate weighted hazard scores
    hazard_df_month['flood_hazard_level'] = hazard_df_month[hazard_vars].apply(
        lambda row: sum(row[var] * weights[var] for var in hazard_vars), axis=1
    )

   # Categorize the flood hazard into levels (1 to 5)
    categories = [1, 2, 3, 4, 5]
    hazard_df_month['flood-hazard'] = pd.cut(
        hazard_df_month['flood_hazard_level'], 
        bins=np.linspace(0, 1, 6),  # Divide into 5 equal intervals
        labels=categories,
        include_lowest=True
    )
    
    hazard_df_months.append(hazard_df_month)

    #hazard_df_month['flood_hazard'] = hazard_df_month['flood_hazard_level']

hazard = pd.concat(hazard_df_months)


master_variables = master_variables.merge(hazard[['timeperiod', 'object_id', 'flood-hazard']],
                       on = ['timeperiod', 'object_id'],how='left')
print(master_variables.columns)
master_variables.to_csv(path+r'/RiskScoreModel/data/factor_scores_l1_flood-hazard.csv', index=False)

# Normalize data using MinMaxScaler
