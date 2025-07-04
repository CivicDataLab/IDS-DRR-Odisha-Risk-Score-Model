#!/usr/bin/env python3
# government-response-weight.py
#
# Build “government-response” scores:
#   • cumulative tender values (per FY and per district)
#   • month-wise Min-Max scaling
#   • 5-level risk buckets (1 = least response, 5 = strongest response)
# ---------------------------------------------------------------------------

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)   # mute pandas / sklearn chatter

# ---------------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------------
DATA_DIR  = Path("data/bihar")
IN_FILE   ='/home/prajna/civicdatalab/ids-drr/bihar/flood-data-ecosystem-Bihar/MASTER_VARIABLES.csv'
OUT_FILE  = DATA_DIR / "factor_scores_l1_government-response.csv"

# columns
GOV_RESPONSE_VARS = [
    "total_tender_awarded_value",
    # "SDRF_sanctions_awarded_value",
    # "RIDF_tenders_awarded_value",
    # "Preparedness Measures_tenders_awarded_value",
    # "Immediate Measures_tenders_awarded_value",
    # "Others_tenders_awarded_value",
]

MODEL_VARS = [                 # used for Min–Max scaling + sum
    "total_tender_awarded_value",
    # "SDRF_sanctions_awarded_value",
    # "Others_tenders_awarded_value",
]

# ---------------------------------------------------------------------------
# 2. HELPERS
# ---------------------------------------------------------------------------
def to_fin_year(tp: str) -> str:
    """Convert 'YYYY_MM' timeperiod → 'YYYY-YY' FY string."""
    yr, m = map(int, tp.split("_"))
    return f"{yr if m >= 4 else yr-1}-{(yr+1)%100 if m >= 4 else yr%100:02d}"

def scale_0_1(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Column-wise Min-Max scaling (returns a *copy*)."""
    scaler = MinMaxScaler()
    df_loc = df.copy()
    df_loc[cols] = scaler.fit_transform(df_loc[cols].astype("float64"))
    return df_loc

def bucket_response(month_df: pd.DataFrame) -> pd.Series:
    """
    5-level score:
        1 = ≤ mean       (weakest response)
        2 = (mean-1σ , mean]
        3 = (mean-2σ , mean-1σ]
        4 = (mean-3σ , mean-2σ]
        5 = < mean-3σ    (strongest response)
    """
    s = month_df["sum"]
    mean, std = s.mean(), s.std(ddof=0)

    cond = [
        s >= mean,
        (s < mean) & (s >= mean - std),
        (s < mean - std) & (s >= mean - 2 * std),
        (s < mean - 2 * std) & (s >= mean - 3 * std),
        s < mean - 3 * std,
    ]

    return np.select(cond, [1, 2, 3, 4, 5], default=np.nan).astype(int)

# ---------------------------------------------------------------------------
# 3. LOAD & PREP MASTER
# ---------------------------------------------------------------------------
master = (
    pd.read_csv(IN_FILE)
      .loc[:, ~pd.read_csv(IN_FILE).columns.duplicated()]   # drop dup headers
      .sort_values(["object_id", "timeperiod"])
      .assign(FinancialYear=lambda d: d["timeperiod"].map(to_fin_year))
)

# cumulative “historical_tenders” per district (across months)
master["historical_tenders"] = (
    master.groupby("object_id")["total_tender_awarded_value"].cumsum()
)

# FY-wise cumulative sums for each response var
for col in GOV_RESPONSE_VARS:
    master[col] = (
        master.groupby(["object_id", "FinancialYear"])[col]
              .cumsum()
              .astype("float64")          # ensure numeric for scaling
    )

# ---------------------------------------------------------------------------
# 4. MONTH-WISE SCORING
# ---------------------------------------------------------------------------
response_frames = []
for month in tqdm(master["timeperiod"].unique(), desc="government-response"):
    mdf = master.loc[master["timeperiod"] == month,
                     ["object_id", "timeperiod", *MODEL_VARS]].copy()

    mdf = scale_0_1(mdf, MODEL_VARS)
    mdf["sum"] = mdf[MODEL_VARS].sum(axis=1)

    mdf["government-response"] = bucket_response(mdf)
    response_frames.append(mdf[["object_id", "timeperiod", "government-response"]])

gov_response = pd.concat(response_frames, ignore_index=True)

# ---------------------------------------------------------------------------
# 5. MERGE BACK & SAVE
# ---------------------------------------------------------------------------
out = (
    master.drop(columns=GOV_RESPONSE_VARS)        # avoid duplicates
          .merge(gov_response, on=["object_id", "timeperiod"], how="left")
)

out.to_csv(OUT_FILE, index=False)
print(f"✓  Saved {OUT_FILE.relative_to(Path.cwd())}")
