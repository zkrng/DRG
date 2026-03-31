"""
src/models/anomaly_detector.py
================================
Isolation Forest for upcoding and cost anomaly detection.

Literature basis:
  - ScienceDirect systematic review (2024): 41 unsupervised studies use Isolation Forest
    and LOF as primary algorithms for healthcare fraud detection
  - NHCAA: ~$100B annual US healthcare fraud; upcoding is the #1 billing fraud type
  - Bauder et al.: Upcoding review — unsupervised methods essential as labels are scarce

Design:
  - Features: cost variance, LoS variance, DRG weight vs peer group median,
    complexity-adjusted expected cost ratio
  - Isolation Forest with contamination = estimated fraud rate (~8%)
  - Anomaly scores mapped to [0,1] for interpretability
  - Provider-level aggregation: flag providers with > 30% episode anomaly rate
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import ANOMALY_CONTAMINATION, MODEL_DIR, SEED


# Features specifically designed to detect upcoding
ANOMALY_FEATURES = [
    "drg_weight",           # Inflated weight is the primary upcoding signal
    "actual_los",           # Unusually short LoS for a high-weight DRG = red flag
    "los_ratio",            # actual_los / geo_mean_los
    "complexity_score",     # Combined severity × Charlson
    "charlson_score",
    "severity",
    "cost_per_los",         # High cost-per-day relative to DRG
    "drg_weight",
    "los_vs_geo_mean",
    "age",
]


class UpcodeAnomalyDetector:
    """
    Detects potential upcoding and billing anomalies using Isolation Forest.

    Usage
    -----
    >>> detector = UpcodeAnomalyDetector()
    >>> detector.fit(df_train_features)
    >>> results = detector.detect(df_features, df_meta)
    """

    def __init__(self, contamination: float = ANOMALY_CONTAMINATION):
        self.contamination = contamination
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("iso_forest", IsolationForest(
                n_estimators=200,
                contamination=contamination,
                max_samples="auto",
                random_state=SEED,
                n_jobs=-1,
            ))
        ])
        self.feature_names = None
        self.is_fitted = False

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select anomaly detection features that exist in the dataframe."""
        available = [f for f in ANOMALY_FEATURES if f in df.columns]
        # Remove duplicates while preserving order
        seen, unique = set(), []
        for f in available:
            if f not in seen:
                unique.append(f)
                seen.add(f)
        self.feature_names = unique
        return df[unique].fillna(0)

    def fit(self, df: pd.DataFrame) -> "UpcodeAnomalyDetector":
        X = self._select_features(df)
        self.pipeline.fit(X)
        self.is_fitted = True
        print(f"✓ Anomaly detector fitted on {len(X):,} episodes  "
              f"(contamination={self.contamination:.0%})")
        return self

    def anomaly_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns anomaly scores in [0, 1].
        Higher = more anomalous. Isolation Forest raw scores are negative;
        we flip and normalise to [0, 1].
        """
        X = df[self.feature_names].fillna(0)
        raw = self.pipeline.named_steps["iso_forest"].score_samples(
            self.pipeline.named_steps["scaler"].transform(X)
        )
        # Flip and scale to [0, 1]
        score = (raw * -1)
        score = (score - score.min()) / (score.max() - score.min() + 1e-9)
        return score

    def detect(self, df: pd.DataFrame, threshold: float = 0.60) -> pd.DataFrame:
        """
        Run anomaly detection on a DataFrame.

        Returns DataFrame with anomaly scores and flags.
        threshold: anomaly_score above this is flagged
        """
        X = df[self.feature_names].fillna(0)
        iso = self.pipeline.named_steps["iso_forest"]
        scaler = self.pipeline.named_steps["scaler"]

        labels = iso.predict(scaler.transform(X))   # -1 = anomaly, 1 = normal
        scores = self.anomaly_score(df)

        result = pd.DataFrame({
            "anomaly_score":    scores.round(4),
            "is_anomaly":       (labels == -1).astype(int),
            "is_flagged":       (scores >= threshold).astype(int),
        }, index=df.index)

        return result

    def provider_risk_report(self, df: pd.DataFrame,
                             anomaly_results: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate episode-level anomaly flags to provider level.
        Providers with >30% flagged episodes are high-risk.
        """
        merged = df[["provider_id", "specialty", "volume_tier", "region",
                     "episode_cost", "drg_weight", "actual_los"]].copy()
        merged["anomaly_score"] = anomaly_results["anomaly_score"].values
        merged["is_flagged"]    = anomaly_results["is_flagged"].values

        report = merged.groupby("provider_id").agg(
            n_episodes          = ("episode_cost", "count"),
            n_flagged           = ("is_flagged", "sum"),
            flag_rate           = ("is_flagged", "mean"),
            mean_anomaly_score  = ("anomaly_score", "mean"),
            mean_episode_cost   = ("episode_cost", "mean"),
            mean_drg_weight     = ("drg_weight", "mean"),
            mean_los            = ("actual_los", "mean"),
            specialty           = ("specialty", "first"),
            volume_tier         = ("volume_tier", "first"),
            region              = ("region", "first"),
        ).reset_index()

        report["flag_rate_pct"] = (report["flag_rate"] * 100).round(1)
        report["provider_risk"] = pd.cut(
            report["flag_rate"],
            bins=[-0.01, 0.10, 0.20, 0.30, 1.01],
            labels=["Low", "Medium", "High", "Critical"]
        ).astype(str)

        report = report.sort_values("flag_rate", ascending=False)

        n_high = (report["provider_risk"].isin(["High", "Critical"])).sum()
        print(f"\n── Upcoding Detection Report ───────────────────────")
        print(f"  Episodes scanned:   {len(df):,}")
        print(f"  Flagged episodes:   {anomaly_results['is_flagged'].sum():,}  "
              f"({anomaly_results['is_flagged'].mean():.1%})")
        print(f"  High-risk providers:{n_high} of {len(report)}")

        return report

    def evaluate_against_labels(self, anomaly_results: pd.DataFrame,
                                  true_labels: pd.Series) -> dict:
        """
        If we have ground truth (synthetic data), evaluate detection quality.
        In real deployments this is unavailable.
        """
        from sklearn.metrics import roc_auc_score, average_precision_score

        auc_roc = roc_auc_score(true_labels, anomaly_results["anomaly_score"])
        auc_pr  = average_precision_score(true_labels, anomaly_results["anomaly_score"])

        tp = ((anomaly_results["is_flagged"] == 1) & (true_labels == 1)).sum()
        fp = ((anomaly_results["is_flagged"] == 1) & (true_labels == 0)).sum()
        fn = ((anomaly_results["is_flagged"] == 0) & (true_labels == 1)).sum()

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)

        print(f"\n── Anomaly Detector vs Ground Truth ───────────────")
        print(f"  AUC-ROC:    {auc_roc:.4f}")
        print(f"  AUC-PR:     {auc_pr:.4f}")
        print(f"  Precision:  {precision:.4f}")
        print(f"  Recall:     {recall:.4f}")

        return {"auc_roc": auc_roc, "auc_pr": auc_pr,
                "precision": precision, "recall": recall}

    def save(self, path=None):
        path = path or MODEL_DIR / "anomaly_detector.joblib"
        joblib.dump(self, path)
        print(f"  Saved anomaly detector → {path}")

    @classmethod
    def load(cls, path=None):
        path = path or MODEL_DIR / "anomaly_detector.joblib"
        return joblib.load(path)
