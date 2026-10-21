import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings("ignore")
path = os.getcwd()  # + r"/flood-data-ecosystem-Odisha"

master_variables = pd.read_csv(path + '/data/MASTER_VARIABLES.csv')


# ===========================================================================
# STEP 1: AHP WEIGHT DERIVATION (pairwise comparison + consistency check)
# ===========================================================================
def ahp_weights(matrix, factor_names):
    """
    Computes AHP priority vector (weights) and Consistency Ratio (CR)
    from a Saaty pairwise comparison matrix.
    """
    n = matrix.shape[0]

    col_sums = matrix.sum(axis=0)
    normalized_matrix = matrix / col_sums
    priority_vector = normalized_matrix.mean(axis=1)

    weighted_sum = matrix @ priority_vector
    lambda_vals = weighted_sum / priority_vector
    lambda_max = lambda_vals.mean()

    CI = (lambda_max - n) / (n - 1)

    RI_table = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
                6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI = RI_table[n]
    CR = CI / RI if RI != 0 else 0

    weights = dict(zip(factor_names, priority_vector))

    print(f"Priority vector (weights): {weights}")
    print(f"Lambda_max: {lambda_max:.4f}")
    print(f"Consistency Index (CI): {CI:.4f}")
    print(f"Consistency Ratio (CR): {CR:.4f}",
          "(ACCEPTABLE, CR < 0.10)" if CR < 0.10 else "(INCONSISTENT - revise matrix)")

    return weights, CR


# ---------------------------------------------------------------------------
# Factors, following the Kosi River Basin structure: SAR/Bhuvan-derived
# inundation is treated as a top-tier AHP input (comparable in importance to
# elevation).
# ---------------------------------------------------------------------------
factor_names = ['elevation_mean', 'inundation', 'distance_from_river',
                 'distance_from_sea', 'sum_rain', 'Mean_Daily_Runoff']

# Saaty pairwise comparison matrix.
#   - elevation vs inundation = 1 (equal): both top-tier -- elevation is the
#     strongest physical conditioning factor in the literature, inundation
#     is your strongest empirical/observed factor (Kosi treats SAR
#     inundation as comparably important to elevation/slope)
#   - elevation & inundation vs distance_from_river = 2 (slightly more important)
#   - elevation & inundation vs distance_from_sea = 3 (moderately more important)
#   - elevation & inundation vs rain/runoff = 5 (strongly more important)
#   - distance_from_river vs distance_from_sea = 2
#   - distance_from_river vs rain/runoff = 4
#   - distance_from_sea vs rain/runoff = 2
#   - rain vs runoff = 1 (equal, no literature precedent to separate them)

pairwise_matrix = np.array([
    [1,    1,    2,    3,    5,    5   ],  # elevation_mean
    [1,    1,    2,    3,    5,    5   ],  # inundation
    [1/2,  1/2,  1,    2,    4,    4   ],  # distance_from_river
    [1/3,  1/3,  1/2,  1,    2,    2   ],  # distance_from_sea
    [1/5,  1/5,  1/4,  1/2,  1,    1   ],  # sum_rain
    [1/5,  1/5,  1/4,  1/2,  1,    1   ],  # Mean_Daily_Runoff
])

group_weights, cr = ahp_weights(pairwise_matrix, factor_names)

if cr >= 0.10:
    raise ValueError(f"Consistency Ratio {cr:.4f} exceeds 0.10 -- revise the pairwise matrix before proceeding.")

# Split the combined "inundation" weight evenly between your two inundation
# columns. NOTE: these two are almost certainly correlated (one is a
# per-cell mean, the other a sum) -- splitting the weight avoids double
# counting the same underlying signal twice at full strength. Adjust the
# split (e.g. 70/30) if you have reason to weight one more than the other.
inundation_weight = group_weights.pop('inundation')
weights = {
    **group_weights,
    'inundation_intensity_mean_nonzero': inundation_weight / 2,
    'inundation_intensity_sum': inundation_weight / 2,
}

print("\nFinal per-column weights (after splitting inundation):")
print(weights)


# ===========================================================================
# STEP 2: HAZARD SCORING
# ===========================================================================
hazard_vars = list(weights.keys())
# ['elevation_mean', 'distance_from_river', 'distance_from_sea', 'sum_rain',
#  'Mean_Daily_Runoff', 'inundation_intensity_mean_nonzero', 'inundation_intensity_sum']

# Variables where a HIGHER raw/scaled value means LOWER hazard, inverted (1-x).
# Inundation is NOT inverted -- higher observed inundation directly means higher hazard.
inverse_vars = ['elevation_mean', 'distance_from_sea', 'distance_from_river']

hazard_df = master_variables[hazard_vars + ['timeperiod', 'object_id']]

hazard_df_months = []
for month in tqdm(hazard_df.timeperiod.unique()):
    scaler = MinMaxScaler()
    hazard_df = master_variables[hazard_vars + ['timeperiod', 'object_id']]
    hazard_df_month = hazard_df[hazard_df.timeperiod == month].copy()
    hazard_df_month[hazard_vars] = scaler.fit_transform(hazard_df_month[hazard_vars])

    for var in inverse_vars:
        hazard_df_month[var] = 1 - hazard_df_month[var]

    hazard_df_month['flood_hazard_level'] = hazard_df_month[hazard_vars].apply(
        lambda row: sum(row[var] * weights[var] for var in hazard_vars), axis=1
    )

    # Quantile-based bins so every class (1-5) actually gets populated,
    # rather than equal-width bins that can leave a class empty if your
    # hazard scores aren't uniformly distributed across 0-1.
    hazard_df_month['flood-hazard'] = pd.qcut(
        hazard_df_month['flood_hazard_level'],
        q=5,
        labels=[1, 2, 3, 4, 5],
        duplicates='drop'
    )

    hazard_df_months.append(hazard_df_month)

hazard = pd.concat(hazard_df_months)

master_variables = master_variables.merge(
    hazard[['timeperiod', 'object_id', 'flood-hazard']],
    on=['timeperiod', 'object_id'], how='left'
)

print(master_variables.columns)
master_variables.to_csv(path + '/data/factor_scores_l1_flood-hazard.csv', index=False)


# ===========================================================================
# IMPORTANT CAVEAT ON VALIDATION
# ===========================================================================
# Because inundation_intensity_mean_nonzero and inundation_intensity_sum are
# now INPUTS to flood_hazard_level, you can no longer validate this hazard
# that would be circular. This mirrors how the Kosi paper
# validated its combined AHP+SAR risk map against a SEPARATE, independent
# product (Bhuvan's Flood Hazard Atlas), not against the same SAR inundation
# layer it used as an input.
#
# To validate this version properly, you need an independent ground truth
# NOT used above, for example:
#   - historical flood-affected village/district records (SFDRS or state
#     disaster management reports)
#   - a separate Bhuvan product (e.g. their Flood Hazard Zonation Atlas,
#     which is distinct from the raw inundation intensity layers used here)

#
#