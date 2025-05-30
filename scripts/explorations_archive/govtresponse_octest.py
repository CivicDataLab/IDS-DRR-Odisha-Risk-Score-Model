import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm 
import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

master_variables = pd.read_csv(os.getcwd()+'/IDS-DRR-Assam/RiskScoreModel/data/MASTER_VARIABLES.csv')

master_variables = master_variables.sort_values(['object_id', 'timeperiod'])

# Calculate historical tenders
master_variables['historical_tenders'] = (
    master_variables
    .groupby(['timeperiod','object_id'])['total_tender_awarded_value']
    .cumsum()
)

def get_financial_year(timeperiod):
    if int(timeperiod.split('_')[1]) >= 4:
        return str(int(timeperiod.split('_')[0]))+'-'+str(int(timeperiod.split('_')[0])+1)
    else:
        return str(int(timeperiod.split('_')[0]) - 1)+'-'+str(int(timeperiod.split('_')[0]))
    
# Apply the function to create the 'FinancialYear' column
master_variables['FinancialYear'] = master_variables['timeperiod'].apply(lambda x: get_financial_year(x))

#INPUT VARS
government_response_vars = ["total_tender_awarded_value",
                            "SDRF_sanctions_awarded_value",
                       "SOPD_tenders_awarded_value",
                       "SDRF_tenders_awarded_value",
                       "RIDF_tenders_awarded_value",
                       "LTIF_tenders_awarded_value",
                       "CIDF_tenders_awarded_value",
                       "Preparedness Measures_tenders_awarded_value",
                       "Immediate Measures_tenders_awarded_value",
                       "Others_tenders_awarded_value",
                      ]

# Find cumsum in each FY of the government response vars
for var in government_response_vars:
    #var.astype(int)
    master_variables[var]=master_variables.groupby(['object_id','FinancialYear'])[var].cumsum()


# HISTORICAL CUMSUM
for var in government_response_vars:
    #var.astype(int)
    master_variables[var]=master_variables.groupby(['object_id','FinancialYear'])[var].cumsum()

govtresponse_df = master_variables[government_response_vars + ['timeperiod', 'object_id']]

#INPUT VARS
government_response_model_vars = ["total_tender_awarded_value",
                            "SDRF_sanctions_awarded_value",
                       "Others_tenders_awarded_value"
                      ]

govtresponse_df_months = []
for month in tqdm(govtresponse_df.timeperiod.unique()):
    govtresponse_df_month = govtresponse_df[govtresponse_df.timeperiod == month]
    # Initialize MinMaxScaler
    scaler = MinMaxScaler()
    # Fit scaler to the data and transform it
    govtresponse_df_month[government_response_model_vars] = scaler.fit_transform(govtresponse_df_month[government_response_model_vars])
    
    # Sum scaled exposure vars
    govtresponse_df_month['sum'] = govtresponse_df_month[government_response_model_vars].sum(axis=1)

    # Ensure mean and std are calculated from 'sum' column
    mean = govtresponse_df_month['sum'].mean()
    std = govtresponse_df_month['sum'].std()

    # First categorization (increasing risk)
    conditions_1 = [
        (govtresponse_df_month['sum'] <= mean),
        (govtresponse_df_month['sum'] > mean) & (govtresponse_df_month['sum'] <= mean + std),
        (govtresponse_df_month['sum'] > mean + std) & (govtresponse_df_month['sum'] <= mean + 2 * std),
        (govtresponse_df_month['sum'] > mean + 2 * std) & (govtresponse_df_month['sum'] <= mean + 3 * std),
        (govtresponse_df_month['sum'] > mean + 3 * std)
    ]

    categories_1 = [5, 4, 3, 2, 1]  # Risk categories: 5 is very low, 1 is very high risk

    # Second categorization (decreasing risk)
    conditions_2 = [
        (govtresponse_df_month['sum'] >= mean),
        (govtresponse_df_month['sum'] < mean) & (govtresponse_df_month['sum'] >= mean - std),
        (govtresponse_df_month['sum'] < mean - std) & (govtresponse_df_month['sum'] >= mean - 2 * std),
        (govtresponse_df_month['sum'] < mean - 2 * std) & (govtresponse_df_month['sum'] >= mean - 3 * std),
        (govtresponse_df_month['sum'] < mean - 3 * std)
    ]
    govtresponse_df_month['government-response_1'] = np.select(conditions_1, categories_1, default='outlier')

    quantiles = govtresponse_df_month['sum'].quantile([0.2, 0.4, 0.6, 0.8])
    categories_2 = [1, 2, 3, 4, 5]

    conditions_2 = [
        (govtresponse_df_month['sum'] >= quantiles[0.8]),
        (govtresponse_df_month['sum'] < quantiles[0.8]) & (govtresponse_df_month['sum'] >= quantiles[0.6]),
        (govtresponse_df_month['sum'] < quantiles[0.6]) & (govtresponse_df_month['sum'] >= quantiles[0.4]),
        (govtresponse_df_month['sum'] < quantiles[0.4]) & (govtresponse_df_month['sum'] >= quantiles[0.2]),
        (govtresponse_df_month['sum'] < quantiles[0.2])
    ]

    govtresponse_df_month['government-response_2'] = np.select(conditions_2, categories_2, default='outlier')


    govtresponse_df_months.append(govtresponse_df_month)

govtresponse = pd.concat(govtresponse_df_months)

# Merge to include the updated government response variables and avoid duplicating columns
#govtresponse_df_updated = govtresponse[['timeperiod', 'object_id','government-response'] + government_response_vars ]
govtresponse_df_updated = govtresponse_df.merge(govtresponse[['timeperiod', 'object_id', 'government-response_1','government-response_2','sum']], on = ['timeperiod', 'object_id'])

master_variables = master_variables.drop(columns=government_response_vars)


#master_variables = master_variables.merge(govtresponse_df[['timeperiod', 'object_id', 'government-response']+ government_response_vars],
#                       on = ['timeperiod', 'object_id'])

# Merge the updated govtresponse_df with the master_variables to update the existing columns
master_variables = master_variables.merge(govtresponse_df_updated, on=['timeperiod', 'object_id'], how='left')

master_variables.to_csv(os.getcwd()+'/IDS-DRR-Assam/RiskScoreModel/data/factor_scores_l1_government-response_octtest.csv', index=False)