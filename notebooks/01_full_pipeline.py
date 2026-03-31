# %% [markdown]
# # DRG Analytics — End-to-End Data Science Pipeline
# ### Health Insurance Provider: Cost Efficiency & Provider Benchmarking
#
# **Author:** Data Science Division  
# **Dataset:** 5,000 synthetic hospital episodes (30 DRG codes, 40 providers)
#
# ---
# ## What you will learn
# 1. Generating realistic synthetic clinical data with controlled distributions
# 2. Feature engineering: Charlson comorbidity scoring, target encoding, log-transforms
# 3. Gradient Boosting regression for episode cost prediction
# 4. Binary classification with probability calibration and threshold optimisation
# 5. Unsupervised anomaly detection (Isolation Forest) for upcoding detection
# 6. K-means provider benchmarking with percentile ranking
# 7. Permutation feature importance / SHAP explainability
# 8. Proper train/val/test splits and evaluation metrics

# %% [markdown]
# ## 0. Setup & Imports

# %%
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "..")   # adjust if running from notebooks/ subfolder

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    DRG_TABLE, N_EPISODES, N_PROVIDERS,
    COST_MODEL_PARAMS, READMISSION_MODEL_PARAMS,
    PLOT_DIR, DATA_DIR, SEED
)

print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
import sklearn; print("scikit-learn:", sklearn.__version__)

# %%
# ─────────────────────────────────────────────────────────────
# STEP 1: Generate synthetic data
# ─────────────────────────────────────────────────────────────

# %% [markdown]
# ## 1. Synthetic Data Generation
#
# We simulate a realistic DRG episode dataset.  
# Key design decisions grounded in the literature:
# - **Log-normal costs** — right-skewed, consistent with real claims data (Jain et al. 2024)
# - **Negative binomial LoS** — overdispersed count data around DRG geometric mean
# - **Charlson comorbidity** — sampled from age/severity-dependent Bernoulli distributions
# - **Upcoding** — 7–10% of episodes have an inflated DRG weight (NHCAA estimate)
# - **Provider random effects** — each hospital has a log-normal cost efficiency factor

# %%
from src.data.generator import generate_dataset

df, providers_df = generate_dataset(n_episodes=N_EPISODES, seed=SEED)

print(f"\nShape: {df.shape}")
print(f"\nColumn groups:")
print(f"  Patient cols:    age, gender, payer")
print(f"  Clinical cols:   drg_code, severity, actual_los, charlson_score")
print(f"  Provider cols:   provider_id, specialty, volume_tier, region")
print(f"  Outcomes:        episode_cost, readmitted_30d")
print(f"  Ground truth:    _is_upcoded (synthetic label, not available in real data)")

df.head()

# %%
# Quick distributional checks
fig, axes = plt.subplots(2, 3, figsize=(14, 7))

axes[0,0].hist(df["episode_cost"], bins=60, color="#185FA5", alpha=0.8, edgecolor="white")
axes[0,0].set_title("Episode Cost Distribution"); axes[0,0].set_xlabel("Cost ($)")

axes[0,1].hist(np.log1p(df["episode_cost"]), bins=50, color="#0F6E56", alpha=0.8, edgecolor="white")
axes[0,1].set_title("Log(Cost) — much more normal"); axes[0,1].set_xlabel("log(Cost)")

axes[0,2].hist(df["actual_los"], bins=40, color="#534AB7", alpha=0.8, edgecolor="white")
axes[0,2].set_title("Length of Stay (days)"); axes[0,2].set_xlabel("Days")

axes[1,0].hist(df["charlson_score"], bins=20, color="#BA7517", alpha=0.8, edgecolor="white")
axes[1,0].set_title("Charlson Comorbidity Score"); axes[1,0].set_xlabel("Score")

cost_by_drg = df.groupby("drg_code")["episode_cost"].mean().sort_values(ascending=False).head(10)
axes[1,1].barh(cost_by_drg.index, cost_by_drg.values, color="#185FA5", alpha=0.85)
axes[1,1].set_title("Top 10 DRG Codes by Mean Cost"); axes[1,1].set_xlabel("Mean Cost ($)")

readmit_by_payer = df.groupby("payer")["readmitted_30d"].mean()
axes[1,2].bar(readmit_by_payer.index, readmit_by_payer.values, color="#993C1D", alpha=0.85)
axes[1,2].set_title("Readmission Rate by Payer"); axes[1,2].set_ylabel("Rate"); axes[1,2].tick_params(axis="x", rotation=20)

for ax in axes.flatten():
    ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(PLOT_DIR / "01_eda_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print("EDA plot saved.")

# %%
# ─────────────────────────────────────────────────────────────
# STEP 2: Feature Engineering & Preprocessing
# ─────────────────────────────────────────────────────────────

# %% [markdown]
# ## 2. Feature Engineering
#
# Key engineering steps:
# - **Target encoding** for `drg_code` — Bayesian smoothed mean of target per category.
#   This outperforms one-hot encoding for high-cardinality categoricals (Jain et al. 2024).
# - **Log-transform** on `episode_cost` — converts a right-skewed target into an
#   approximately normal distribution, making RMSE meaningful.
# - **Complexity score** = severity × (1 + 0.2 × charlson) — single composite clinical risk feature.
# - **Temporal features** — month, quarter, weekend admission flag.

# %%
from src.data.preprocessor import preprocess, engineer_base_features

# Inspect engineered features before encoding
df_eng = engineer_base_features(df.copy())
print("New features added:")
new_cols = [c for c in df_eng.columns if c not in df.columns]
print(new_cols)

# %%
# Show the effect of target encoding on drg_code
print("\nSample target-encoded DRG codes (cost target):")
from src.data.preprocessor import _target_encode
te = _target_encode(df, "drg_code", "episode_cost").rename("drg_enc_cost")
pd.concat([df["drg_code"], df["episode_cost"], te], axis=1).groupby("drg_code").first().sort_values("drg_enc_cost", ascending=False).head(8)

# %%
# Run full preprocessing
data = preprocess(df)
cost_data    = data["cost"]
readmit_data = data["readmit"]
full_df      = data["full_df"]

print("\nFeature matrix shapes:")
print(f"  Cost task     X_train: {cost_data['X_train'].shape}   X_test: {cost_data['X_test'].shape}")
print(f"  Readmit task  X_train: {readmit_data['X_train'].shape}   X_test: {readmit_data['X_test'].shape}")

# %%
# Check class balance
print(f"\nReadmission class balance:")
print(readmit_data["y_train"].value_counts(normalize=True).round(3))
print("\nThis is ~80/20 — moderately imbalanced.")
print("We use scale_pos_weight (XGBoost) or class_weight (sklearn) to handle this.")

# %%
# ─────────────────────────────────────────────────────────────
# STEP 3: Task A — Episode Cost Prediction (Regression)
# ─────────────────────────────────────────────────────────────

# %% [markdown]
# ## 3. Task A: Episode Cost Prediction
#
# **Goal:** Predict `log(episode_cost)` from DRG code, patient features, and provider characteristics.
#
# **Why log-transform the target?**
# - Raw cost is right-skewed (some episodes cost 10× the median)
# - RMSE on raw cost would be dominated by expensive outliers
# - Log-scale RMSE ≈ "proportional error" — meaningful for all cost levels
#
# **Model:** `HistGradientBoostingRegressor` (sklearn) = equivalent to XGBoost locally.
# Swap to `XGBRegressor` with identical results when `xgboost` is installed.

# %%
from src.models.cost_predictor import DRGCostPredictor

cost_model = DRGCostPredictor()
cost_model.fit(
    cost_data["X_train"], cost_data["y_train"],
    cost_data["X_val"],   cost_data["y_val"],
)

# %%
cost_metrics = cost_model.evaluate(
    cost_data["X_train"], cost_data["y_train"],
    cost_data["X_test"],  cost_data["y_test"],
)

# %%
# Actual vs predicted plot
y_pred_log = cost_model.predict_log(cost_data["X_test"])
y_true_log = cost_data["y_test"].values

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(y_pred_log, y_true_log, alpha=0.25, s=8, color="#185FA5")
lo, hi = min(y_pred_log.min(), y_true_log.min()), max(y_pred_log.max(), y_true_log.max())
axes[0].plot([lo,hi],[lo,hi], "r--", lw=1.5, label="Perfect fit")
axes[0].set_xlabel("Predicted log(cost)"); axes[0].set_ylabel("Actual log(cost)")
axes[0].set_title(f"Cost Prediction  (Test R² = {cost_metrics['test_r2']:.3f})")
axes[0].legend(); axes[0].spines[["top","right"]].set_visible(False)

residuals = y_true_log - y_pred_log
axes[1].hist(residuals, bins=50, color="#0F6E56", alpha=0.85, edgecolor="white")
axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
axes[1].set_xlabel("Residual (actual − predicted)"); axes[1].set_title("Residual Distribution")
axes[1].spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig(PLOT_DIR / "02_cost_prediction_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Cost variance flagging — identify high-cost outlier episodes
y_test_actual_cost = np.expm1(cost_data["y_test"])
cost_results = cost_model.predict_with_variance(cost_data["X_test"], y_test_actual_cost)

print(f"\nCost variance summary (test set):")
print(f"  Episodes with actual > 35% above predicted: {cost_results['is_cost_outlier'].sum()} "
      f"({cost_results['is_cost_outlier'].mean():.1%})")
print(f"\nTop 5 most expensive outliers:")
cost_results.sort_values("variance_pct", ascending=False).head()

# %%
# ─────────────────────────────────────────────────────────────
# STEP 4: Task B — 30-day Readmission Classification
# ─────────────────────────────────────────────────────────────

# %% [markdown]
# ## 4. Task B: 30-day Readmission Risk Classification
#
# **Goal:** Predict probability of 30-day readmission at point of discharge.
#
# **Key ML concepts demonstrated:**
# 1. **Probability calibration** — raw model probabilities are often poorly calibrated
#    (not matching true frequencies). Platt scaling fixes this, critical for risk stratification.
# 2. **Threshold optimisation** — default threshold 0.5 is rarely optimal for imbalanced data.
#    We find the threshold maximising F1 on the validation set.
# 3. **AUC-ROC vs AUC-PR** — for imbalanced classes, AUC-PR (Precision-Recall) is more
#    informative than AUC-ROC (Saito & Rehmsmeier 2015).

# %%
from src.models.readmission_classifier import ReadmissionClassifier

clf = ReadmissionClassifier(calibrate=True)
clf.fit(
    readmit_data["X_train"], readmit_data["y_train"],
    readmit_data["X_val"],   readmit_data["y_val"],
)
print(f"\nOptimised decision threshold: {clf.threshold:.3f}")

# %%
readmit_metrics = clf.evaluate(readmit_data["X_test"], readmit_data["y_test"])

# %%
# ROC + Precision-Recall curves
from sklearn.metrics import roc_curve, precision_recall_curve, auc

proba_test = clf.predict_proba(readmit_data["X_test"])
y_test_bin = readmit_data["y_test"].values

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# ROC
fpr, tpr, _ = roc_curve(y_test_bin, proba_test)
roc_auc = auc(fpr, tpr)
axes[0].plot(fpr, tpr, color="#185FA5", lw=2, label=f"AUC-ROC = {roc_auc:.3f}")
axes[0].plot([0,1],[0,1], "k--", lw=1); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].set_title("ROC Curve"); axes[0].legend(); axes[0].spines[["top","right"]].set_visible(False)

# PR curve
prec, rec, _ = precision_recall_curve(y_test_bin, proba_test)
pr_auc = auc(rec, prec)
axes[1].plot(rec, prec, color="#0F6E56", lw=2, label=f"AUC-PR = {pr_auc:.3f}")
axes[1].axhline(y_test_bin.mean(), color="r", linestyle="--", lw=1,
                label=f"Baseline = {y_test_bin.mean():.3f}")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve"); axes[1].legend(); axes[1].spines[["top","right"]].set_visible(False)

# Calibration curve
from sklearn.calibration import calibration_curve
frac_pos, mean_pred = calibration_curve(y_test_bin, proba_test, n_bins=10)
axes[2].plot(mean_pred, frac_pos, "s-", color="#185FA5", lw=2, label="Model")
axes[2].plot([0,1],[0,1], "k--", lw=1, label="Perfect calibration")
axes[2].set_xlabel("Mean predicted prob"); axes[2].set_ylabel("Fraction positives")
axes[2].set_title("Calibration Curve (Reliability Diagram)"); axes[2].legend()
axes[2].spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig(PLOT_DIR / "03_readmission_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Risk stratification — this is what clinical teams actually use
risk_results = clf.predict_with_risk_bands(readmit_data["X_test"])
print("Risk band distribution (test set):")
print(risk_results["risk_band"].value_counts())

print("\nActual readmission rate by risk band:")
merged = risk_results.copy()
merged["actual"] = readmit_data["y_test"].values
merged.groupby("risk_band")["actual"].mean().sort_values(ascending=False)

# %%
# ─────────────────────────────────────────────────────────────
# STEP 5: Task C — Upcoding Anomaly Detection
# ─────────────────────────────────────────────────────────────

# %% [markdown]
# ## 5. Task C: Upcoding Detection (Isolation Forest)
#
# **Goal:** Detect episodes where DRG code assignment appears inconsistent with
# clinical features — a potential indicator of upcoding fraud.
#
# **Why unsupervised?** In real deployments, labels are almost never available.
# Insurance companies must detect anomalies *without* confirmed fraud cases.
#
# **Isolation Forest intuition:**
# - Builds random trees; anomalous points are *isolated in fewer splits*
# - Score = average path length (shorter = more anomalous)
# - Contamination parameter = expected fraud rate (~8% based on NHCAA estimates)
#
# **Features used:** DRG weight, actual LoS vs geometric mean LoS, severity,
# Charlson score, cost-per-day — all signals that should be internally consistent.

# %%
from src.models.anomaly_detector import UpcodeAnomalyDetector

detector = UpcodeAnomalyDetector(contamination=0.08)
detector.fit(full_df)

# %%
anomaly_results = detector.detect(full_df, threshold=0.60)

print("Anomaly detection summary:")
print(f"  Total episodes:  {len(anomaly_results):,}")
print(f"  Flagged:         {anomaly_results['is_flagged'].sum():,}  ({anomaly_results['is_flagged'].mean():.1%})")

# %%
# Evaluate against synthetic ground truth (only possible because we generated the data)
print("\n[Synthetic ground truth evaluation — not possible in real deployments]")
anomaly_metrics = detector.evaluate_against_labels(
    anomaly_results, full_df["_is_upcoded"]
)

# Note: a low AUC here is expected! The upcoding simulation adds noise, and
# Isolation Forest with no labels is working purely from distributional signals.
# In practice, combining it with rule-based pre-filters significantly improves results.

# %%
# Visualise anomaly scores
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Score distribution by true label
normal_scores  = anomaly_results.loc[full_df["_is_upcoded"]==0, "anomaly_score"]
upcoded_scores = anomaly_results.loc[full_df["_is_upcoded"]==1, "anomaly_score"]
axes[0].hist(normal_scores,  bins=50, alpha=0.7, color="#0F6E56", label=f"Normal (n={len(normal_scores):,})")
axes[0].hist(upcoded_scores, bins=50, alpha=0.7, color="#E24B4A", label=f"Upcoded (n={len(upcoded_scores):,})")
axes[0].axvline(0.60, color="black", linestyle="--", lw=1.5, label="Flag threshold = 0.60")
axes[0].set_xlabel("Anomaly Score"); axes[0].set_ylabel("Count")
axes[0].set_title("Anomaly Score Distribution by True Label"); axes[0].legend()
axes[0].spines[["top","right"]].set_visible(False)

# DRG weight vs LoS, coloured by anomaly flag
flagged = anomaly_results["is_flagged"] == 1
axes[1].scatter(full_df.loc[~flagged, "drg_weight"], full_df.loc[~flagged, "actual_los"],
                alpha=0.15, s=5, color="#185FA5", label="Normal")
axes[1].scatter(full_df.loc[flagged, "drg_weight"], full_df.loc[flagged, "actual_los"],
                alpha=0.6, s=15, color="#E24B4A", label="Flagged", zorder=5)
axes[1].set_xlabel("DRG Relative Weight"); axes[1].set_ylabel("Actual LoS (days)")
axes[1].set_title("DRG Weight vs LoS — Anomaly Flags"); axes[1].legend()
axes[1].spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig(PLOT_DIR / "04_anomaly_detection.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Provider-level upcoding risk report
provider_risk = detector.provider_risk_report(full_df, anomaly_results)
print("\nTop 10 highest-risk providers:")
provider_risk[["provider_id","specialty","volume_tier","n_episodes",
               "n_flagged","flag_rate_pct","mean_anomaly_score","provider_risk"]].head(10)

# %%
# ─────────────────────────────────────────────────────────────
# STEP 6: Provider Benchmarking
# ─────────────────────────────────────────────────────────────

# %% [markdown]
# ## 6. Provider Benchmarking
#
# **Goal:** Create fair, peer-adjusted provider scorecards.
#
# **Pipeline:**
# 1. **K-means clustering** segments providers into peer groups by volume, specialty, and case mix.
#    Clustering first prevents unfair comparison (a 50-bed rural hospital vs a 500-bed academic centre).
# 2. **Percentile ranking** within each peer group on: mean cost, LoS, readmission rate, cost efficiency.
# 3. **Composite score** = mean percentile across KPIs.
# 4. **Outlier flags** = providers at or above 90th percentile on any KPI.
#
# **Why not just rank all providers together?** Case-mix differences make direct comparison misleading.
# A provider treating mostly high-severity sepsis will always look expensive — that's not inefficiency.

# %%
from src.models.provider_benchmarker import ProviderBenchmarker

# Generate cost predictions for the full dataset
all_cost_features = [f for f in cost_data["feature_names"] if f in full_df.columns]
X_all = full_df[all_cost_features].fillna(0)
all_pred_costs = cost_model.predict_cost(X_all)

benchmarker = ProviderBenchmarker(n_clusters=6)
benchmarker.fit(full_df, pd.Series(all_pred_costs, index=full_df.index))
scorecard = benchmarker.score(full_df, pd.Series(all_pred_costs, index=full_df.index))

# %%
# Show scorecard
print("\nProvider Benchmarking Scorecard (top 10 by composite risk rank):")
scorecard[["provider_id","specialty","peer_group_label","composite_rank",
           "performance_tier","cost_outlier","los_outlier","readmit_outlier",
           "n_outlier_flags","mean_episode_cost","readmission_rate"]].head(10)

# %%
# Visualise peer groups
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

colors = plt.cm.Set2(np.linspace(0, 1, 6))
for pg in range(6):
    subset = scorecard[scorecard["peer_group"] == pg]
    axes[0].scatter(subset["n_episodes"], subset["mean_episode_cost"],
                    s=60, alpha=0.85, color=colors[pg],
                    label=f"Peer Group {pg+1} (n={len(subset)})")
axes[0].set_xlabel("Annual Episode Volume"); axes[0].set_ylabel("Mean Episode Cost ($)")
axes[0].set_title("Providers by Volume & Cost — Peer Groups")
axes[0].legend(fontsize=8); axes[0].spines[["top","right"]].set_visible(False)

# Composite rank bar chart — top 20
top20 = scorecard.head(20)
flag_colors = top20["n_outlier_flags"].map({0:"#0F6E56", 1:"#BA7517", 2:"#E24B4A", 3:"#A32D2D"})
bars = axes[1].barh(top20["provider_id"][::-1], top20["composite_rank"][::-1],
                     color=flag_colors[::-1].values, alpha=0.85)
axes[1].axvline(90, color="black", linestyle="--", lw=1.2, label="90th pct threshold")
axes[1].set_xlabel("Composite Percentile Rank (within peer group)")
axes[1].set_title("Top 20 Providers by Risk Composite Score"); axes[1].legend(fontsize=9)
axes[1].spines[["top","right"]].set_visible(False)

legend_labels = {"#0F6E56":"0 flags", "#BA7517":"1 flag",
                 "#E24B4A":"2 flags", "#A32D2D":"3 flags"}
from matplotlib.patches import Patch
patches = [Patch(color=c, label=l) for c, l in legend_labels.items()]
axes[1].legend(handles=patches, fontsize=8, loc="lower right")

plt.tight_layout()
plt.savefig(PLOT_DIR / "05_provider_benchmarking.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# ─────────────────────────────────────────────────────────────
# STEP 7: Feature Importance & Explainability
# ─────────────────────────────────────────────────────────────

# %% [markdown]
# ## 7. Feature Importance & Explainability
#
# **Permutation importance** (used here locally) works by:
# 1. Computing baseline model score on test set
# 2. Randomly shuffling one feature at a time
# 3. Measuring score degradation — larger drop = more important feature
#
# **When SHAP is installed** (`pip install shap`), the `FeatureExplainer` class
# automatically switches to SHAP TreeExplainer, which gives:
# - *Direction* of impact (not just magnitude)
# - Individual episode explanations (force plots)
# - Interaction effects (SHAP interaction values)
#
# **Literature:** SHAP is now the regulatory standard for explainable AI in healthcare
# (NHS AI Lab, 2023; FDA AI/ML Action Plan, 2024).

# %%
from src.evaluation.metrics import FeatureExplainer

# Cost model explainer
cost_explainer = FeatureExplainer(
    cost_model.model,
    cost_data["X_train"].sample(200, random_state=SEED),
    model_name="Cost Prediction"
)
cost_explainer.compute(cost_data["X_test"].sample(500, random_state=SEED))
fig_cost_imp = cost_explainer.plot_importance_bar(top_n=15)
plt.show()

print("\nTop 10 cost prediction features:")
print(cost_explainer.global_importance(10).to_string(index=False))

# %%
# Readmission model explainer
readmit_explainer = FeatureExplainer(
    clf.base_model,
    readmit_data["X_train"].sample(200, random_state=SEED),
    model_name="Readmission Risk"
)
readmit_explainer.compute(readmit_data["X_test"].sample(500, random_state=SEED))
fig_readmit_imp = readmit_explainer.plot_importance_bar(top_n=15)
plt.show()

print("\nTop 10 readmission risk features:")
print(readmit_explainer.global_importance(10).to_string(index=False))

# %%
# Individual episode explanation — explain why a specific patient was flagged as high-risk
sample_500 = readmit_data["X_test"].sample(500, random_state=SEED)
high_risk = risk_results.loc[risk_results["risk_band"].isin(["High","Very High"])].index
common_idx = [i for i in high_risk if i in sample_500.index]

if common_idx:
    pos = sample_500.index.get_loc(common_idx[0])
    episode_explanation = readmit_explainer.explain_episode(pos)
    print(f"\nIndividual episode explanation (high readmission risk):")
    print(episode_explanation[["feature","value","shap_value","direction"]].to_string(index=False))
    print("\nInterpretation:")
    print("  Positive values → increase readmission probability")
    print("  Negative values → decrease readmission probability")

# %%
# ─────────────────────────────────────────────────────────────
# STEP 8: Summary & Next Steps
# ─────────────────────────────────────────────────────────────

# %% [markdown]
# ## 8. Pipeline Summary
#
# All four models have been fitted, evaluated, and saved.
# The summary report is at `outputs/reports/model_summary.md`.

# %%
print("=" * 60)
print("  PIPELINE RESULTS SUMMARY")
print("=" * 60)
print(f"\n  Dataset: {len(df):,} episodes  |  {N_PROVIDERS} providers  |  30 DRG codes")
print(f"\n  Task A — Cost Prediction (Gradient Boosting Regression)")
print(f"    Test R²:      {cost_metrics['test_r2']:.4f}")
print(f"    Test RMSE:    {cost_metrics['test_rmse']:.4f}  (log-scale)")
print(f"    Cost outliers flagged:  {cost_results['is_cost_outlier'].sum()} episodes  ({cost_results['is_cost_outlier'].mean():.1%})")
print(f"\n  Task B — Readmission Classification (HGBC + Calibration)")
print(f"    AUC-ROC:      {readmit_metrics['auc_roc']:.4f}")
print(f"    AUC-PR:       {readmit_metrics['auc_pr']:.4f}")
print(f"    F1 (opt thr): {readmit_metrics['f1']:.4f}")
print(f"\n  Task C — Upcoding Detection (Isolation Forest, unsupervised)")
print(f"    Flagged:      {anomaly_results['is_flagged'].sum():,} episodes  ({anomaly_results['is_flagged'].mean():.1%})")
print(f"    AUC-ROC vs ground truth: {anomaly_metrics['auc_roc']:.4f}")
print(f"\n  Task D — Provider Benchmarking (K-means + Percentile Ranking)")
print(f"    Providers ranked:  {len(scorecard)}")
print(f"    Peer groups:       {benchmarker.n_clusters}")
print(f"    Multi-flag outliers: {(scorecard['n_outlier_flags'] >= 2).sum()}")

# %% [markdown]
# ## 9. Learning Extensions
#
# Once you are comfortable with this pipeline, try these improvements:
#
# ### Modelling
# - **XGBoost / LightGBM:** `pip install xgboost lightgbm` — swap the backend by changing the import.
#   The `_make_regressor` / `_make_classifier` functions handle this automatically.
# - **Hyperparameter tuning:** Replace fixed params with `Optuna` or `sklearn GridSearchCV`
# - **SMOTE:** Add `from imblearn.over_sampling import SMOTE` for the readmission task before fitting
# - **Stacking:** Use cost prediction and anomaly score as *input features* to the readmission model
#
# ### Features
# - **NLP on diagnosis codes:** Use ICD-10 code embeddings (Med2Vec, ClinicalBERT) as features
# - **Time series features:** Rolling readmission rates, cost trends per provider over time
# - **Network features:** Provider referral patterns as graph features
#
# ### Explainability
# - **SHAP:** `pip install shap` — the `FeatureExplainer` class automatically activates SHAP
# - **SHAP interaction values:** `shap.TreeExplainer(model).shap_interaction_values(X)`
# - **Partial dependence plots:** `sklearn.inspection.PartialDependenceDisplay`
#
# ### Production
# - **MLflow tracking:** Wrap `model.fit()` in `with mlflow.start_run():`
# - **Feature drift:** Add `evidently` or `nannyml` to monitor input distributions over time
# - **REST API:** Serve any saved `.joblib` model with FastAPI in ~20 lines of Python
