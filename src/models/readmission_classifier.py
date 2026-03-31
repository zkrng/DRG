"""
src/models/readmission_classifier.py
=====================================
Gradient Boosting binary classifier for 30-day readmission risk.

Uses sklearn HistGradientBoostingClassifier locally; swap to XGBClassifier
when xgboost is installed — identical API.

Literature basis:
  - Luo et al. Scientific Reports (2024): SHAP + XGBoost for readmission
  - PMC study (2025): XGBoost AUC-ROC 0.667 on UCI diabetes readmission
  - ICU readmission: XGBoost + Bayesian opt. AUROC 0.92 (PMC 2023)
"""

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    def _make_classifier(params):
        p = {k: v for k, v in params.items()
             if k not in ("use_label_encoder", "eval_metric", "n_jobs")}
        return xgb.XGBClassifier(**p)
    BACKEND = "xgboost"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    def _make_classifier(params):
        return HistGradientBoostingClassifier(
            max_iter      = params.get("n_estimators", 300),
            max_depth     = params.get("max_depth", 5),
            learning_rate = params.get("learning_rate", 0.05),
            random_state  = params.get("random_state", 42),
        )
    BACKEND = "sklearn"

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve,
)
import joblib

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import READMISSION_MODEL_PARAMS, MODEL_DIR, SEED


class ReadmissionClassifier:
    """
    Predicts 30-day hospital readmission risk with probability calibration.

    Usage
    -----
    >>> clf = ReadmissionClassifier()
    >>> clf.fit(X_train, y_train, X_val, y_val)
    >>> results = clf.predict_with_risk_bands(X_test)
    """

    RISK_BANDS = {
        "Low":      (0.00, 0.15),
        "Moderate": (0.15, 0.30),
        "High":     (0.30, 0.50),
        "Very High":(0.50, 1.01),
    }

    def __init__(self, params: dict = None, calibrate: bool = True):
        self.params    = params or READMISSION_MODEL_PARAMS
        self.calibrate = calibrate
        self.base_model  = _make_classifier(self.params)
        self.model       = None
        self.threshold   = 0.50
        self.is_fitted   = False
        print(f"  Using backend: {BACKEND}")

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "ReadmissionClassifier":
        """Train base model, then calibrate, then optimise threshold."""
        self.base_model.fit(X_train, y_train)

        if self.calibrate and X_val is not None:
            # sklearn 1.2+ dropped cv="prefit"; use cv=None with a pre-fitted estimator
            # by wrapping validation data as a single held-out fold
            try:
                self.model = CalibratedClassifierCV(
                    self.base_model, method="sigmoid", cv=None
                )
                self.model.fit(X_val, y_val)
            except Exception:
                # Last-resort fallback: use 3-fold on val set
                self.model = CalibratedClassifierCV(
                    self.base_model, method="sigmoid", cv=3
                )
                self.model.fit(X_val, y_val)
            print("  ✓ Probability calibration applied (Platt scaling)")
        else:
            self.model = self.base_model

        if X_val is not None:
            self.threshold = self._optimise_threshold(X_val, y_val)

        self.is_fitted = True
        print("✓ Readmission classifier fitted")
        return self

    def _optimise_threshold(self, X_val, y_val) -> float:
        """Find threshold maximising F1 on validation set."""
        probs = self.model.predict_proba(X_val)[:, 1]
        prec, rec, thresholds = precision_recall_curve(y_val, probs)
        f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
        best_idx  = np.argmax(f1_scores[:-1])
        best_thr  = float(thresholds[best_idx])
        print(f"  Optimal threshold: {best_thr:.3f}  (val F1={f1_scores[best_idx]:.3f})")
        return best_thr

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def predict_with_risk_bands(self, X) -> pd.DataFrame:
        """Return probabilities and risk band labels."""
        probs = self.predict_proba(X)
        bands = pd.cut(
            probs,
            bins=[0, 0.15, 0.30, 0.50, 1.01],
            labels=["Low", "Moderate", "High", "Very High"],
            right=True,
        )
        return pd.DataFrame({
            "readmit_probability": probs.round(4),
            "risk_band":           bands.astype(str),
            "predicted_readmit":   (probs >= self.threshold).astype(int),
        }, index=X.index)

    def evaluate(self, X_test, y_test) -> dict:
        """Full classification evaluation suite."""
        probs = self.predict_proba(X_test)
        preds = self.predict(X_test)

        auc_roc = roc_auc_score(y_test, probs)
        auc_pr  = average_precision_score(y_test, probs)
        cm      = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            "auc_roc":    round(auc_roc, 4),
            "auc_pr":     round(auc_pr, 4),
            "threshold":  round(self.threshold, 3),
            "f1":         round(f1_score(y_test, preds), 4),
            "sensitivity":round(tp / (tp + fn + 1e-9), 4),
            "specificity":round(tn / (tn + fp + 1e-9), 4),
            "ppv":        round(tp / (tp + fp + 1e-9), 4),
            "npv":        round(tn / (tn + fn + 1e-9), 4),
        }

        print("\n── Readmission Classifier Evaluation ──────────────")
        print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
        print(f"  AUC-PR:      {metrics['auc_pr']:.4f}")
        print(f"  F1:          {metrics['f1']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  PPV:         {metrics['ppv']:.4f}")
        print(f"\n{classification_report(y_test, preds, target_names=['No readmit','Readmit'])}")

        return metrics

    def get_calibration_data(self, X_test, y_test, n_bins=10):
        """Return calibration curve data for plotting."""
        probs = self.predict_proba(X_test)
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=n_bins)
        return frac_pos, mean_pred

    def save(self, path=None):
        path = path or MODEL_DIR / "readmission_classifier.joblib"
        joblib.dump(self, path)
        print(f"  Saved readmission classifier → {path}")

    @classmethod
    def load(cls, path=None):
        path = path or MODEL_DIR / "readmission_classifier.joblib"
        return joblib.load(path)
