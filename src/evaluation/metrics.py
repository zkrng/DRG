"""
src/evaluation/metrics.py
==========================
Feature importance, explainability, and evaluation metrics.

Uses SHAP when installed; falls back to sklearn permutation importance.
Literature basis:
  - Luo et al. Scientific Reports (2024): SHAP for readmission explanation
  - Jain et al. BMC Health Services Research (2024): SHAP for LoS feature importance
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import PLOT_DIR, REPORT_DIR


class FeatureExplainer:
    """
    Feature importance explainer — uses SHAP TreeExplainer when available,
    falls back to sklearn permutation importance otherwise.
    """

    def __init__(self, model, X_background: pd.DataFrame, model_name: str = "model"):
        self.model_name  = model_name
        self.model       = model
        self.X_background = X_background
        self.shap_values = None
        self.perm_imp    = None

        if SHAP_AVAILABLE:
            try:
                self.explainer = shap.TreeExplainer(model)
                self._mode = "shap"
            except Exception:
                self._mode = "permutation"
        else:
            self._mode = "permutation"

        print(f"  Explainer mode: {self._mode}")

    def compute(self, X: pd.DataFrame):
        if self._mode == "shap":
            self.shap_values = self.explainer.shap_values(X)
            self.X_computed  = X
            return self.shap_values
        else:
            # Permutation importance on background sample
            from sklearn.inspection import permutation_importance
            from sklearn.metrics import r2_score
            y_bg = self.X_background.iloc[:, 0]   # dummy — use model itself
            result = permutation_importance(
                self.model, X, self.model.predict(X),
                n_repeats=10, random_state=42, n_jobs=-1,
            )
            self.perm_imp   = result
            self.X_computed = X
            return result.importances_mean

    def global_importance(self, top_n: int = 15) -> pd.DataFrame:
        features = self.X_computed.columns.tolist()
        if self._mode == "shap" and self.shap_values is not None:
            mean_abs = np.abs(self.shap_values).mean(axis=0)
        elif self.perm_imp is not None:
            mean_abs = self.perm_imp.importances_mean
        elif hasattr(self.model, "feature_importances_"):
            mean_abs = self.model.feature_importances_
        else:
            return pd.DataFrame()

        df = pd.DataFrame({"feature": features, "importance": mean_abs})
        df = df.sort_values("importance", ascending=False).head(top_n)
        total = df["importance"].abs().sum()
        df["importance_pct"] = (df["importance"].abs() / (total + 1e-9) * 100).round(1)
        df["rank"] = range(1, len(df) + 1)
        return df.reset_index(drop=True)

    def plot_importance_bar(self, top_n: int = 15, save: bool = True) -> plt.Figure:
        imp = self.global_importance(top_n)
        if imp.empty:
            return plt.figure()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(imp["feature"][::-1], imp["importance"][::-1].abs(),
                color="#185FA5", alpha=0.85)
        mode_label = "SHAP |mean|" if self._mode == "shap" else "Permutation importance"
        ax.set_xlabel(mode_label, fontsize=11)
        ax.set_title(f"Feature Importance — {self.model_name}", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        if save:
            fname = self.model_name.lower().replace(" ", "_")
            fig.savefig(PLOT_DIR / f"feature_importance_{fname}.png",
                        bbox_inches="tight", dpi=150)
        return fig

    def plot_summary(self, X=None, max_display: int = 15, save: bool = True):
        if self._mode == "shap" and self.shap_values is not None:
            X_plot = X if X is not None else self.X_computed
            fig, _ = plt.subplots(figsize=(9, 6))
            shap.summary_plot(self.shap_values, X_plot,
                              max_display=max_display, show=False, plot_size=None)
            fig = plt.gcf()
            fig.suptitle(f"SHAP Summary — {self.model_name}", y=1.01, fontsize=12)
            plt.tight_layout()
            if save:
                fname = self.model_name.lower().replace(" ", "_")
                fig.savefig(PLOT_DIR / f"shap_summary_{fname}.png",
                            bbox_inches="tight", dpi=150)
            return fig
        else:
            return self.plot_importance_bar(save=save)

    def explain_episode(self, idx: int, feature_names: list = None) -> pd.DataFrame:
        if self._mode == "shap" and self.shap_values is not None:
            sv = self.shap_values[idx]
        else:
            sv = self.perm_imp.importances_mean if self.perm_imp else np.zeros(len(self.X_computed.columns))

        features = feature_names or self.X_computed.columns.tolist()
        values   = self.X_computed.iloc[idx].values
        df = pd.DataFrame({
            "feature":    features,
            "value":      values,
            "shap_value": sv,
            "direction":  ["↑ risk" if s > 0 else "↓ risk" for s in sv],
        })
        df["abs_shap"] = np.abs(df["shap_value"])
        return df.sort_values("abs_shap", ascending=False).head(10).reset_index(drop=True)


# Keep old name as alias for backward compatibility
SHAPExplainer = FeatureExplainer


def plot_cost_predictions(y_true_log, y_pred_log, title="Cost Prediction",
                          save: bool = True) -> plt.Figure:
    """Actual vs predicted scatter plot in log scale."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Scatter: actual vs predicted
    ax = axes[0]
    ax.scatter(y_pred_log, y_true_log, alpha=0.3, s=8, color="#185FA5")
    lo, hi = min(y_pred_log.min(), y_true_log.min()), max(y_pred_log.max(), y_true_log.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit")
    ax.set_xlabel("Predicted log(cost)", fontsize=10)
    ax.set_ylabel("Actual log(cost)", fontsize=10)
    ax.set_title(f"{title} — Actual vs Predicted", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Residuals histogram
    ax = axes[1]
    residuals = y_true_log - y_pred_log
    ax.hist(residuals, bins=40, color="#0F6E56", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", lw=1.5)
    ax.set_xlabel("Residual (actual − predicted)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Residual Distribution", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    if save:
        fig.savefig(PLOT_DIR / "cost_prediction_diagnostics.png", bbox_inches="tight", dpi=150)
    return fig


def plot_roc_pr(y_true, y_proba, model_name="Readmission",
                save: bool = True) -> plt.Figure:
    """ROC curve and Precision-Recall curve side by side."""
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color="#185FA5", lw=2, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0,1],[0,1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"{model_name} — ROC Curve"); axes[0].legend()
    axes[0].spines[["top","right"]].set_visible(False)

    # PR curve
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(rec, prec)
    axes[1].plot(rec, prec, color="#0F6E56", lw=2, label=f"AUC-PR = {pr_auc:.3f}")
    axes[1].axhline(y_true.mean(), color="r", linestyle="--", lw=1,
                    label=f"Baseline = {y_true.mean():.3f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title(f"{model_name} — Precision-Recall Curve"); axes[1].legend()
    axes[1].spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    if save:
        fig.savefig(PLOT_DIR / f"roc_pr_{model_name.lower().replace(' ','_')}.png",
                    bbox_inches="tight", dpi=150)
    return fig


def plot_calibration(frac_pos, mean_pred, model_name="Readmission",
                     save: bool = True) -> plt.Figure:
    """Reliability diagram for classifier calibration."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, frac_pos, "s-", color="#185FA5", lw=2, label="Model")
    ax.plot([0,1],[0,1], "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
    ax.set_title(f"{model_name} — Calibration Curve")
    ax.legend(); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    if save:
        fig.savefig(PLOT_DIR / "calibration_curve.png", bbox_inches="tight", dpi=150)
    return fig


def plot_provider_scorecard(scorecard_df: pd.DataFrame,
                             top_n: int = 20, save: bool = True) -> plt.Figure:
    """Horizontal bar chart of top providers by composite risk rank."""
    top = scorecard_df.head(top_n)[["provider_id", "composite_rank",
                                     "performance_tier", "n_outlier_flags"]].copy()
    colors = top["n_outlier_flags"].map({0:"#0F6E56", 1:"#BA7517", 2:"#E24B4A", 3:"#A32D2D"})

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top["provider_id"][::-1], top["composite_rank"][::-1],
                   color=colors[::-1].values, alpha=0.85)
    ax.axvline(90, color="red", linestyle="--", lw=1.2, label="90th pct threshold")
    ax.set_xlabel("Composite Percentile Rank (within peer group)", fontsize=10)
    ax.set_title("Provider Benchmarking Scorecard — Top Risk Providers", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    if save:
        fig.savefig(PLOT_DIR / "provider_scorecard.png", bbox_inches="tight", dpi=150)
    return fig


def plot_anomaly_distribution(anomaly_results: pd.DataFrame, episodes_df: pd.DataFrame,
                               save: bool = True) -> plt.Figure:
    """Anomaly score distribution with flagged episodes highlighted."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Score distribution
    ax = axes[0]
    normal  = anomaly_results[anomaly_results["is_flagged"] == 0]["anomaly_score"]
    flagged = anomaly_results[anomaly_results["is_flagged"] == 1]["anomaly_score"]
    ax.hist(normal,  bins=40, alpha=0.7, color="#0F6E56", label="Normal")
    ax.hist(flagged, bins=40, alpha=0.7, color="#E24B4A", label="Flagged")
    ax.set_xlabel("Anomaly Score"); ax.set_ylabel("Count")
    ax.set_title("Anomaly Score Distribution"); ax.legend()
    ax.spines[["top","right"]].set_visible(False)

    # Cost vs DRG weight scatter, anomalies highlighted
    ax = axes[1]
    normal_idx  = anomaly_results[anomaly_results["is_flagged"]==0].index
    flagged_idx = anomaly_results[anomaly_results["is_flagged"]==1].index
    ax.scatter(episodes_df.loc[normal_idx,  "drg_weight"],
               episodes_df.loc[normal_idx,  "episode_cost"],
               alpha=0.2, s=6, color="#185FA5", label="Normal")
    ax.scatter(episodes_df.loc[flagged_idx, "drg_weight"],
               episodes_df.loc[flagged_idx, "episode_cost"],
               alpha=0.6, s=12, color="#E24B4A", label="Flagged", zorder=5)
    ax.set_xlabel("DRG Relative Weight"); ax.set_ylabel("Episode Cost ($)")
    ax.set_title("Cost vs DRG Weight — Anomaly Detection"); ax.legend()
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    if save:
        fig.savefig(PLOT_DIR / "anomaly_distribution.png", bbox_inches="tight", dpi=150)
    return fig


def save_summary_report(metrics: dict, output_path=None):
    """Save all metrics to a markdown report."""
    path = output_path or REPORT_DIR / "model_summary.md"
    lines = ["# DRG DS Project — Model Summary Report\n"]
    for section, vals in metrics.items():
        lines.append(f"\n## {section}\n")
        if isinstance(vals, dict):
            for k, v in vals.items():
                lines.append(f"- **{k}**: {v}")
        else:
            lines.append(str(vals))
    path.write_text("\n".join(lines))
    print(f"\n  Report saved → {path}")
