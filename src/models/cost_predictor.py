"""
src/models/cost_predictor.py
=============================
Gradient Boosting regression model to predict log(episode_cost).

Uses sklearn GradientBoostingRegressor locally.
Swap in XGBoost by replacing the import and params — identical API.

Literature basis:
  - Faradisa et al. (2025): XGBoost outperforms LR and RF on APR-DRG cost prediction
  - Jain et al. (2024): Target encoding + gradient boosting best for LoS/cost prediction
"""

import numpy as np
import pandas as pd
try:
    import xgboost as xgb
    def _make_regressor(params):
        p = {k: v for k, v in params.items() if k != "n_jobs"}
        return xgb.XGBRegressor(**p)
    BACKEND = "xgboost"
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
    def _make_regressor(params):
        return HistGradientBoostingRegressor(
            max_iter        = params.get("n_estimators", 300),
            max_depth       = params.get("max_depth", 6),
            learning_rate   = params.get("learning_rate", 0.05),
            l2_regularization = params.get("reg_lambda", 1.0),
            random_state    = params.get("random_state", 42),
        )
    BACKEND = "sklearn"

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import COST_MODEL_PARAMS, COST_VARIANCE_THRESHOLD, MODEL_DIR, SEED


class DRGCostPredictor:
    """
    Predicts expected episode cost from DRG and patient features.

    Usage
    -----
    >>> model = DRGCostPredictor()
    >>> model.fit(X_train, y_train, X_val, y_val)
    >>> results = model.predict_with_variance(X_test, y_test_actual_cost)
    """

    def __init__(self, params: dict = None):
        self.params = params or COST_MODEL_PARAMS
        self.model  = _make_regressor(self.params)
        self.feature_names = None
        self.is_fitted = False
        print(f"  Using backend: {BACKEND}")

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "DRGCostPredictor":
        self.feature_names = list(X_train.columns)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        print("✓ Cost prediction model fitted")
        return self

    def predict_log(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_cost(self, X) -> np.ndarray:
        return np.expm1(self.model.predict(X))

    def predict_with_variance(self, X_test, y_actual_cost: pd.Series) -> pd.DataFrame:
        pred_cost = self.predict_cost(X_test)
        actual    = y_actual_cost.values
        variance_pct = (actual - pred_cost) / pred_cost.clip(min=1)
        return pd.DataFrame({
            "predicted_cost":  pred_cost.round(2),
            "actual_cost":     actual.round(2),
            "variance_pct":    (variance_pct * 100).round(1),
            "abs_error":       np.abs(actual - pred_cost).round(2),
            "is_cost_outlier": (variance_pct > COST_VARIANCE_THRESHOLD).astype(int),
        }, index=X_test.index)

    def evaluate(self, X_train, y_train_log, X_test, y_test_log) -> dict:
        train_pred = self.predict_log(X_train)
        test_pred  = self.predict_log(X_test)
        metrics = {
            "train_rmse": round(float(np.sqrt(mean_squared_error(y_train_log, train_pred))), 4),
            "test_rmse":  round(float(np.sqrt(mean_squared_error(y_test_log,  test_pred))),  4),
            "train_mae":  round(float(mean_absolute_error(y_train_log, train_pred)), 4),
            "test_mae":   round(float(mean_absolute_error(y_test_log,  test_pred)),  4),
            "train_r2":   round(float(r2_score(y_train_log, train_pred)), 4),
            "test_r2":    round(float(r2_score(y_test_log,  test_pred)),  4),
        }
        print("\n── Cost Predictor Evaluation ───────────────────────")
        print(f"  Train RMSE: {metrics['train_rmse']:.4f}  |  Test RMSE: {metrics['test_rmse']:.4f}")
        print(f"  Train MAE:  {metrics['train_mae']:.4f}  |  Test MAE:  {metrics['test_mae']:.4f}")
        print(f"  Train R²:   {metrics['train_r2']:.4f}  |  Test R²:   {metrics['test_r2']:.4f}")
        return metrics

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
            names = self.feature_names or [f"f{i}" for i in range(len(imp))]
            df = pd.DataFrame({"feature": names, "importance": imp})
            df = df.sort_values("importance", ascending=False).head(top_n)
            df["importance_pct"] = (df["importance"] / df["importance"].sum() * 100).round(1)
            return df.reset_index(drop=True)
        return pd.DataFrame()

    def save(self, path=None):
        path = path or MODEL_DIR / "cost_predictor.joblib"
        joblib.dump(self, path)
        print(f"  Saved cost predictor → {path}")

    @classmethod
    def load(cls, path=None):
        path = path or MODEL_DIR / "cost_predictor.joblib"
        return joblib.load(path)
