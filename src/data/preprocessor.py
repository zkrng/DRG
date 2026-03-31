"""
src/data/preprocessor.py
========================
Cleans, encodes, and splits the raw episode dataset.

Key decisions (literature-grounded):
  - Target encoding for drg_code (high cardinality) — per Jain et al. 2024
  - Log-transform episode_cost (right-skewed) — standard in health economics
  - Stratified split on readmitted_30d (imbalanced target)
  - Separate feature sets for each model task
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, OrdinalEncoder
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import SEED, TEST_SIZE, VAL_SIZE


# ── Column groups ─────────────────────────────────────────────────────────────
CHARLSON_COLS = [c for c in [
    "cm_mi", "cm_chf", "cm_peripheral_vd", "cm_cerebrovascular",
    "cm_dementia", "cm_copd", "cm_connective_td", "cm_peptic_ulcer",
    "cm_liver_mild", "cm_diabetes_unc", "cm_diabetes_comp",
    "cm_hemiplegia", "cm_renal_mod_sev", "cm_cancer",
    "cm_liver_mod_sev", "cm_metastatic", "cm_aids",
]]

# Features to DROP before modelling (leakage or admin fields)
DROP_COLS = [
    "episode_id", "admission_date", "drg_description",
    "_is_upcoded", "_upcoding_factor", "_provider_cost_factor",
    "provider_id",     # too many levels — provider-level features used instead
]


def _target_encode(df: pd.DataFrame, col: str, target: str,
                   smoothing: float = 20.0) -> pd.Series:
    """
    Bayesian target encoding with smoothing.
    Replaces each category with: (count * cat_mean + smoothing * global_mean)
                                  / (count + smoothing)
    Prevents overfitting on rare categories.
    """
    global_mean = df[target].mean()
    stats = df.groupby(col)[target].agg(["count", "mean"])
    smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / \
               (stats["count"] + smoothing)
    return df[col].map(smoothed)


def engineer_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features before the main encoding step.
    """
    df = df.copy()

    # Temporal features
    df["admission_month"]   = df["admission_date"].dt.month
    df["admission_quarter"] = df["admission_date"].dt.quarter
    df["admission_dow"]     = df["admission_date"].dt.dayofweek
    df["is_weekend_admit"]  = (df["admission_dow"] >= 5).astype(int)

    # Cost-related ratios
    df["cost_per_los"]      = df["episode_cost"] / df["actual_los"].clip(lower=1)
    df["los_vs_geo_mean"]   = df["actual_los"] - df["geo_mean_los"]
    df["los_ratio"]         = df["actual_los"] / df["geo_mean_los"].clip(lower=0.5)
    df["weighted_drg_cost"] = df["drg_weight"] * 6500   # expected base cost

    # Age bands (clinical groupings)
    df["age_band"] = pd.cut(
        df["age"],
        bins=[0, 40, 60, 75, 120],
        labels=["<40", "40-60", "60-75", "75+"]
    ).astype(str)

    # Complexity score combining severity and Charlson
    df["complexity_score"] = df["severity"] * (1 + 0.2 * df["charlson_score"])

    # High-severity flag
    df["is_high_severity"] = (df["severity"] >= 3).astype(int)

    # Previous admission risk
    df["high_prev_admit"] = (df["prev_admissions"] >= 2).astype(int)

    # Log-transforms (right-skewed distributions)
    df["log_episode_cost"] = np.log1p(df["episode_cost"])
    df["log_charlson"]     = np.log1p(df["charlson_score"])

    return df


def preprocess(df: pd.DataFrame) -> dict:
    """
    Full preprocessing pipeline. Returns a dict of train/val/test DataFrames
    for each of the three modelling tasks.

    Tasks:
      A. cost_regression   — predict log(episode_cost)
      B. readmission_clf   — predict readmitted_30d (binary)
      C. anomaly_detection — unsupervised, full feature set

    Returns
    -------
    {
      'cost':       {'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test', 'feature_names'},
      'readmit':    { ... same ... },
      'full_df':    df with all engineered features,
      'encoders':   dict of fitted encoders for later use,
    }
    """
    df = engineer_base_features(df)

    # ── Categorical encoding ──────────────────────────────────────────────────
    cat_cols = ["gender", "payer", "admission_type", "disposition",
                "volume_tier", "region", "specialty", "accreditation",
                "age_band"]

    le_encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
            le_encoders[col] = le

    # Target-encode DRG code separately for each target
    df["drg_code_enc_cost"]    = _target_encode(df, "drg_code", "episode_cost")
    df["drg_code_enc_readmit"] = _target_encode(df, "drg_code", "readmitted_30d")

    # ── Feature sets ──────────────────────────────────────────────────────────
    numeric_base = [
        "age", "severity", "actual_los", "geo_mean_los",
        "drg_weight", "charlson_score", "prev_admissions",
        "los_vs_geo_mean", "los_ratio", "complexity_score",
        "admission_month", "admission_quarter", "is_weekend_admit",
        "is_high_severity", "high_prev_admit", "log_charlson",
    ]
    cat_encoded = [f"{c}_enc" for c in cat_cols if f"{c}_enc" in df.columns]
    charlson_present = [c for c in CHARLSON_COLS if c in df.columns]

    # Task A: cost prediction features
    cost_features = (
        numeric_base
        + cat_encoded
        + charlson_present
        + ["drg_code_enc_cost", "weighted_drg_cost"]
    )

    # Task B: readmission features (LoS and cost are available at discharge)
    readmit_features = (
        numeric_base
        + cat_encoded
        + charlson_present
        + ["drg_code_enc_readmit", "log_episode_cost", "cost_per_los"]
    )

    # Drop any features that might not exist
    cost_features   = [f for f in cost_features   if f in df.columns]
    readmit_features= [f for f in readmit_features if f in df.columns]

    # ── Train / Val / Test split ───────────────────────────────────────────────
    def split(features, target_col):
        X = df[features].fillna(0)
        y = df[target_col]

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=TEST_SIZE,
            stratify=(y if y.nunique() == 2 else None),
            random_state=SEED
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=VAL_SIZE / (1 - TEST_SIZE),
            stratify=(y_trainval if y_trainval.nunique() == 2 else None),
            random_state=SEED
        )
        return {
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val":  y_val, "y_test":  y_test,
            "feature_names": features,
        }

    cost_data   = split(cost_features,    "log_episode_cost")
    readmit_data= split(readmit_features, "readmitted_30d")

    print("\n── Preprocessing summary ──────────────────────────────")
    print(f"  Total episodes:        {len(df):,}")
    print(f"  Cost features:         {len(cost_features)}")
    print(f"  Readmission features:  {len(readmit_features)}")
    print(f"  Train: {len(cost_data['X_train']):,}  "
          f"Val: {len(cost_data['X_val']):,}  "
          f"Test: {len(cost_data['X_test']):,}")
    print(f"  Readmission rate (train): {readmit_data['y_train'].mean():.1%}")

    return {
        "cost":     cost_data,
        "readmit":  readmit_data,
        "full_df":  df,
        "encoders": le_encoders,
    }
