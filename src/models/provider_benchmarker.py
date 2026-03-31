"""
src/models/provider_benchmarker.py
====================================
K-means peer group segmentation + percentile ranking for provider benchmarking.

Literature basis:
  - BMC Medical Informatics (2021): ML-based DRG grouping with k-means for peer cohorts
  - DRGKB Oxford Academic (2024): Random Forest best for cost/LoS benchmarking
  - Frontiers in AI (2025): Peer comparison essential for DRG cost management

Pipeline:
  1. Aggregate episode data to provider level
  2. K-means cluster providers into peer groups (by specialty, volume, case mix)
  3. Within each peer group, compute percentile ranks on key KPIs
  4. Flag outlier providers (> 90th percentile on cost, LoS, or readmission rate)
  5. Generate composite benchmarking scorecard
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import stats
import joblib

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import KMEANS_CLUSTERS, OUTLIER_PERCENTILE, MODEL_DIR, SEED


# KPIs used for benchmarking (all higher = worse, except quality metrics)
BENCHMARK_KPIS = {
    "mean_episode_cost":    "Cost",
    "mean_actual_los":      "Length of Stay",
    "readmission_rate":     "30-day Readmission",
    "mean_drg_weight":      "Case Mix Index",
    "cost_per_drg_weight":  "Cost Efficiency",
}


class ProviderBenchmarker:
    """
    Segments providers into peer groups and computes benchmarking scorecards.

    Usage
    -----
    >>> bench = ProviderBenchmarker()
    >>> bench.fit(episodes_df)
    >>> scorecard = bench.score(episodes_df, cost_predictions=pred_costs)
    """

    def __init__(self, n_clusters: int = KMEANS_CLUSTERS,
                 outlier_pct: int = OUTLIER_PERCENTILE):
        self.n_clusters   = n_clusters
        self.outlier_pct  = outlier_pct
        self.pipeline     = Pipeline([
            ("scaler",  StandardScaler()),
            ("kmeans",  KMeans(n_clusters=n_clusters, random_state=SEED,
                               n_init=20, max_iter=500)),
        ])
        self.provider_summary = None
        self.is_fitted = False

    def _build_provider_summary(self, df: pd.DataFrame,
                                 cost_predictions: pd.Series = None) -> pd.DataFrame:
        """
        Aggregate episode-level data to one row per provider with all KPIs.
        """
        agg = df.groupby("provider_id").agg(
            n_episodes         = ("episode_cost", "count"),
            mean_episode_cost  = ("episode_cost", "mean"),
            median_episode_cost= ("episode_cost", "median"),
            mean_actual_los    = ("actual_los", "mean"),
            median_los         = ("actual_los", "median"),
            readmission_rate   = ("readmitted_30d", "mean"),
            mean_drg_weight    = ("drg_weight", "mean"),
            mean_charlson      = ("charlson_score", "mean"),
            mean_severity      = ("severity", "mean"),
            specialty          = ("specialty", "first"),
            volume_tier        = ("volume_tier", "first"),
            region             = ("region", "first"),
        ).reset_index()

        agg["case_mix_index"]     = agg["mean_drg_weight"]
        agg["cost_per_drg_weight"]= agg["mean_episode_cost"] / agg["mean_drg_weight"].clip(0.1)

        # If cost predictions available, compute cost efficiency vs expected
        if cost_predictions is not None:
            expected = df.copy()
            expected["pred_cost"] = cost_predictions.values
            exp_agg = expected.groupby("provider_id")["pred_cost"].mean()
            agg = agg.merge(exp_agg.rename("mean_predicted_cost"),
                            on="provider_id", how="left")
            agg["cost_vs_expected_pct"] = (
                (agg["mean_episode_cost"] - agg["mean_predicted_cost"])
                / agg["mean_predicted_cost"].clip(1) * 100
            ).round(1)

        return agg

    def fit(self, df: pd.DataFrame,
            cost_predictions: pd.Series = None) -> "ProviderBenchmarker":
        """Fit K-means clustering on provider aggregates."""
        self.provider_summary = self._build_provider_summary(df, cost_predictions)

        # Clustering features: volume, cost, LoS, case mix, readmission
        cluster_features = [
            "n_episodes", "mean_episode_cost", "mean_actual_los",
            "readmission_rate", "case_mix_index", "mean_charlson",
        ]
        X = self.provider_summary[cluster_features].fillna(0)
        self.pipeline.fit(X)

        self.provider_summary["peer_group"] = self.pipeline.predict(X)
        self.provider_summary["peer_group_label"] = (
            "Peer Group " + (self.provider_summary["peer_group"] + 1).astype(str)
        )

        self.is_fitted = True
        print(f"✓ Provider benchmarker fitted: {len(self.provider_summary)} providers "
              f"→ {self.n_clusters} peer groups")
        return self

    def score(self, df: pd.DataFrame,
              cost_predictions: pd.Series = None) -> pd.DataFrame:
        """
        Compute benchmarking scorecard for all providers.
        Returns provider-level DataFrame with percentile ranks within peer groups.
        """
        summary = self._build_provider_summary(df, cost_predictions)

        # Assign peer groups to new data
        cluster_features = [
            "n_episodes", "mean_episode_cost", "mean_actual_los",
            "readmission_rate", "case_mix_index", "mean_charlson",
        ]
        X = summary[cluster_features].fillna(0)
        summary["peer_group"] = self.pipeline.predict(X)
        summary["peer_group_label"] = (
            "Peer Group " + (summary["peer_group"] + 1).astype(str)
        )

        # Percentile ranks within each peer group
        for kpi in BENCHMARK_KPIS:
            if kpi not in summary.columns:
                continue
            summary[f"{kpi}_pct_rank"] = summary.groupby("peer_group")[kpi].transform(
                lambda x: x.rank(pct=True) * 100
            ).round(1)

        # Composite score (mean percentile across KPIs — higher = worse)
        rank_cols = [f"{kpi}_pct_rank" for kpi in BENCHMARK_KPIS
                     if f"{kpi}_pct_rank" in summary.columns]
        summary["composite_rank"] = summary[rank_cols].mean(axis=1).round(1)

        # Outlier flags
        summary["cost_outlier"] = (
            summary["mean_episode_cost_pct_rank"] >= self.outlier_pct
        ).astype(int)
        summary["los_outlier"] = (
            summary.get("mean_actual_los_pct_rank", pd.Series(0, index=summary.index)) >= self.outlier_pct
        ).astype(int)
        summary["readmit_outlier"] = (
            summary.get("readmission_rate_pct_rank", pd.Series(0, index=summary.index)) >= self.outlier_pct
        ).astype(int)
        summary["n_outlier_flags"] = (
            summary["cost_outlier"]
            + summary["los_outlier"]
            + summary["readmit_outlier"]
        )

        # Overall performance tier
        summary["performance_tier"] = pd.cut(
            summary["composite_rank"],
            bins=[-1, 25, 50, 75, 101],
            labels=["Top Quartile", "Above Average", "Below Average", "Bottom Quartile"]
        ).astype(str)

        n_flagged = (summary["n_outlier_flags"] >= 2).sum()
        print(f"\n── Provider Benchmarking Summary ───────────────────")
        print(f"  Providers scored:   {len(summary)}")
        print(f"  Peer groups:        {self.n_clusters}")
        print(f"  Multi-flag outliers:{n_flagged}")

        peer_summary = summary.groupby("peer_group_label").agg(
            n=("provider_id", "count"),
            mean_cost=("mean_episode_cost", "mean"),
            mean_los=("mean_actual_los", "mean"),
        )
        print(f"\n  Peer group breakdown:")
        print(peer_summary.to_string())

        return summary.sort_values("composite_rank", ascending=False)

    def save(self, path=None):
        path = path or MODEL_DIR / "provider_benchmarker.joblib"
        joblib.dump(self, path)
        print(f"  Saved provider benchmarker → {path}")

    @classmethod
    def load(cls, path=None):
        path = path or MODEL_DIR / "provider_benchmarker.joblib"
        return joblib.load(path)
