import numpy as np
import pandas as pd
from scipy.optimize import linprog

vul_master = pd.read_csv(r'D:\CivicDataLabs_IDS-DRR\IDS-DRR_Github\IDS-DRR-Assam\RiskScoreModel\data\factor_scores_l1_vulnerability.csv')

vulnerability_vars = ["mean_sexratio",
                      "sum_aged_population",
                      "schools_count",
                      "health_centres_count",
                      "rail_length",
                      "road_length",
                      "Embankment breached",
                      "net_sown_area_in_hac",
                      "avg_electricity",
                      "rc_nosanitation_hhds_pct",
                      "rc_piped_hhds_pct"]

#OUTPUT VARS
damage_vars = ["Total_Animal_Affected","Population_affected_Total",
                "Crop_Area","Total_House_Fully_Damaged","Embankments affected",
                 "Roads","Bridge"]

Biswanath_Aug23 = vul_master[vul_master['revenue_ci_x'].isin(['Biswanath'])]

def dea_model(inputs, outputs):
    """
    Data Envelopment Analysis (DEA) model with Slack variables.

    Parameters:
    - inputs: 2D array representing input data for each decision-making unit.
    - outputs: 1D array representing output data for each decision-making unit.

    Returns:
    - efficiency_scores: Array containing efficiency scores for each unit.
    - slack_variables: Array containing the slack variables associated with the constraints.
    """

    num_units, num_inputs = inputs.shape

    # Objective function coefficients (c)
    c = np.zeros(num_inputs + num_units)

    # Inequality matrix (A) and right-hand side (b) for constraints
    A_ineq = np.hstack((inputs, -np.eye(num_units)))
    b_ineq = outputs

    # Equality matrix (A_eq) and right-hand side (b_eq) for constraints
    A_eq = np.zeros((num_units, num_inputs + num_units))
    A_eq[:, :num_inputs] = -inputs
    A_eq[:, num_inputs:] = np.eye(num_units)
    b_eq = np.zeros(num_units)

    # Bounds for variables (x >= 0)
    bounds = [(0, None)] * num_inputs + [(None, None)] * num_units

    # Solve the linear programming problem
    result = linprog(c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    # Extract results
    efficiency_scores = result['x'][:num_inputs]
    slack_variables = result['x'][num_inputs:]

    return efficiency_scores, slack_variables

# Example usage:
#inputs = np.array([[2, 3], [4, 5], [6, 7]])
#outputs = np.array([10, 10, 10])

inputs = np.array(Biswanath_Aug23[vulnerability_vars])
outputs = np.array(Biswanath_Aug23[damage_vars])

efficiency_scores, slack_variables = dea_model(inputs, outputs)

print("Efficiency Scores:", efficiency_scores)
print("Slack Variables:", slack_variables)
