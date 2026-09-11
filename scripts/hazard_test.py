import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import spearmanr
from tqdm import tqdm
import os
import warnings
import matplotlib.pyplot as plt

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

    matrix        : n x n numpy array, Saaty 1-9 scale pairwise comparisons
    factor_names  : list of n factor names matching matrix row/column order
    """
    n = matrix.shape[0]

    # Normalize columns, then average rows -> priority vector (the weights)
    col_sums = matrix.sum(axis=0)
    normalized_matrix = matrix / col_sums
    priority_vector = normalized_matrix.mean(axis=1)

    # Consistency check
    weighted_sum = matrix @ priority_vector
    lambda_vals = weighted_sum / priority_vector
    lambda_max = lambda_vals.mean()

    CI = (lambda_max - n) / (n - 1)

    # Saaty's Random Index (RI) table
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


# Conditioning factors ONLY (physical susceptibility drivers).
# inundation_intensity_* is deliberately excluded here -- it's used later as
# an independent validation variable (Bhuvan/SAR-derived, not a model input).
factor_names = ['elevation_mean', 'distance_from_river', 'distance_from_sea',
                 'sum_rain', 'Mean_Daily_Runoff']

# Saaty pairwise comparison matrix. Judgments informed by AHP flood-hazard
# literature (Nowshera, Diyala, Kosi River Basin, Chalakudy, Cheliff-Ghrib):
#   - elevation vs rain/runoff = 5 (strongly more important)
#     elevation routinely outweighing rainfall by 2-3x
#   - elevation vs distance_from_river = 2 (slightly more important): both
#     are top-tier drivers, elevation has the widest/highest reported range
#   - distance_from_river vs distance_from_sea = 2: river proximity is the
#     standard fluvial-hazard factor across nearly every paper
#   - rain vs runoff = 1 (equal): no literature precedent to rank one above
#     the other as standalone factors
pairwise_matrix = np.array([
    [1,    2,    3,    5,    5   ],  # elevation_mean
    [1/2,  1,    2,    4,    4   ],  # distance_from_river
    [1/3,  1/2,  1,    2,    2   ],  # distance_from_sea
    [1/5,  1/4,  1/2,  1,    1   ],  # sum_rain
    [1/5,  1/4,  1/2,  1,    1   ],  # Mean_Daily_Runoff
])

weights, cr = ahp_weights(pairwise_matrix, factor_names)

if cr >= 0.10:
    raise ValueError(f"Consistency Ratio {cr:.4f} exceeds 0.10 -- revise the pairwise matrix before proceeding.")


# ===========================================================================
# STEP 2: HAZARD SCORING (MinMax scale -> invert distance/elevation -> weighted sum -> classes)
# ===========================================================================
hazard_vars = factor_names  # elevation_mean, distance_from_river, distance_from_sea, sum_rain, Mean_Daily_Runoff

# Variables where a HIGHER raw/scaled value means LOWER hazard, inverted (1-x)
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

    categories = [1, 2, 3, 4, 5]
    hazard_df_month['flood-hazard'] = pd.cut(
        hazard_df_month['flood_hazard_level'],
        bins=np.linspace(0, 1, 6),
        labels=categories,
        include_lowest=True
    )

    hazard_df_months.append(hazard_df_month)

hazard = pd.concat(hazard_df_months)

master_variables = master_variables.merge(
    hazard[['timeperiod', 'object_id', 'flood-hazard']],
    on=['timeperiod', 'object_id'], how='left'
)

print(master_variables.columns)
#master_variables.to_csv(path + '/data/factor_scores_l1_flood-hazard.csv', index=False)


# ===========================================================================
# STEP 3: VALIDATION against Bhuvan/SAR observed inundation (independent data)
# ===========================================================================
validation_vars = ['inundation_intensity_mean_nonzero', 'inundation_intensity_sum']

validation_df = master_variables[
    ['timeperiod', 'object_id', 'flood-hazard'] + validation_vars
].dropna(subset=['flood-hazard']).copy()

# Binarize ground truth: any nonzero observed inundation = "flooded"
validation_df['observed_flooded'] = (validation_df['inundation_intensity_sum'] > 0).astype(int)

print("\nObserved flood prevalence (ground truth):")
print(validation_df['observed_flooded'].value_counts(normalize=True))

# --- ROC-AUC using the continuous (pre-binning) hazard score as predictor ---
merged = hazard[['timeperiod', 'object_id', 'flood_hazard_level']].merge(
    validation_df[['timeperiod', 'object_id', 'observed_flooded']],
    on=['timeperiod', 'object_id'], how='inner'
).dropna()

fpr, tpr, thresholds = roc_curve(merged['observed_flooded'], merged['flood_hazard_level'])
auc_score = roc_auc_score(merged['observed_flooded'], merged['flood_hazard_level'])

print(f"\nAUC-ROC: {auc_score:.3f}")
if auc_score >= 0.9:
    print("Outstanding fit")
elif auc_score >= 0.8:
    print("Excellent fit (comparable to Kosi River Basin paper's 0.837)")
elif auc_score >= 0.7:
    print("Acceptable fit")
else:
    print("Poor fit -- reconsider AHP pairwise matrix / weights")

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'AHP hazard index (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Success Rate)')
plt.title('AHP Flood Hazard Index Validation (ROC Curve)')
plt.legend()
plt.tight_layout()
plt.savefig(path + '/data/ahp_validation_roc_curve.png', dpi=150)
plt.close()

# --- Spearman correlation (finer-grained secondary check) ---
corr_df = hazard[['timeperiod', 'object_id', 'flood_hazard_level']].merge(
    master_variables[['timeperiod', 'object_id'] + validation_vars],
    on=['timeperiod', 'object_id'], how='left'
).dropna()

for var in validation_vars:
    corr, pval = spearmanr(corr_df['flood_hazard_level'], corr_df[var])
    print(f"Spearman correlation (flood_hazard_level vs {var}): rho={corr:.3f}, p={pval:.4g}")

# --- Class-wise summary: confirms observed inundation increases monotonically 1->5 ---
class_summary = validation_df.groupby('flood-hazard').agg(
    inundation_mean=('inundation_intensity_mean_nonzero', 'mean'),
    inundation_sum_mean=('inundation_intensity_sum', 'mean'),
    pct_flooded=('observed_flooded', 'mean'),
    n=('observed_flooded', 'count')
)
print("\nObserved inundation & flood prevalence by AHP hazard class:")
print(class_summary)

#class_summary.to_csv(path + '/data/validation_hazard_vs_inundation.csv')