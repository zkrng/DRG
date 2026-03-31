"""
pipeline.py — Main orchestrator for the DRG Analytics DS project.

Run this file to execute the full end-to-end pipeline:
  1. Generate synthetic data
  2. Preprocess & feature-engineer
  3. Train cost prediction model (XGBoost regression)
  4. Train readmission classifier (XGBoost + calibration)
  5. Train anomaly / upcoding detector (Isolation Forest)
  6. Run provider benchmarking (K-means + percentile ranking)
  7. SHAP explainability for all models
  8. Generate all evaluation plots
  9. Save models + summary report

Usage:
  python pipeline.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from config import DATA_DIR, OUTPUT_DIR, SEED

# ── Module imports ────────────────────────────────────────────────────────────
from src.data.generator       import generate_dataset
from src.data.preprocessor    import preprocess
from src.models.cost_predictor          import DRGCostPredictor
from src.models.readmission_classifier  import ReadmissionClassifier
from src.models.anomaly_detector        import UpcodeAnomalyDetector
from src.models.provider_benchmarker    import ProviderBenchmarker
from src.evaluation.metrics import (
    FeatureExplainer,
    SHAPExplainer,
    plot_cost_predictions,
    plot_roc_pr,
    plot_calibration,
    plot_provider_scorecard,
    plot_anomaly_distribution,
    save_summary_report,
)


def run_pipeline():
    print("=" * 60)
    print("  DRG Analytics DS Pipeline")
    print("=" * 60)

    # ── STEP 1: Generate synthetic data ──────────────────────────────────────
    print("\n[1/8] Generating synthetic data...")
    # episodes_df, providers_df = generate_dataset()
    episodes_df = pd.read_csv(DATA_DIR / "episodes_raw.csv")
    providers_df = pd.read_csv(DATA_DIR / "providers.csv")

    print(f"Loaded {len(episodes_df)} episodes and {len(providers_df)} providers")

    # ── STEP 2: Preprocess ───────────────────────────────────────────────────
    print("\n[2/8] Preprocessing & feature engineering...")
    data = preprocess(episodes_df)

    cost_data   = data["cost"]
    readmit_data= data["readmit"]
    full_df     = data["full_df"]

    # ── STEP 3: Cost prediction model ────────────────────────────────────────
    print("\n[3/8] Training cost prediction model (XGBoost Regressor)...")
    cost_model = DRGCostPredictor()
    cost_model.fit(
        cost_data["X_train"], cost_data["y_train"],
        cost_data["X_val"],   cost_data["y_val"],
    )
    cost_metrics = cost_model.evaluate(
        cost_data["X_train"], cost_data["y_train"],
        cost_data["X_test"],  cost_data["y_test"],
    )

    # Generate cost variance results on test set
    # y_test is log-scale; convert back for dollar-level variance analysis
    y_test_actual_cost = np.expm1(cost_data["y_test"])
    cost_results = cost_model.predict_with_variance(
        cost_data["X_test"], y_test_actual_cost
    )
    n_cost_outliers = cost_results["is_cost_outlier"].sum()
    print(f"  Cost outliers flagged: {n_cost_outliers}  "
          f"({n_cost_outliers/len(cost_results):.1%} of test set)")

    # Plot: actual vs predicted
    plot_cost_predictions(
        cost_data["y_test"].values,
        cost_model.predict_log(cost_data["X_test"]),
        title="DRG Episode Cost Prediction"
    )

    # ── STEP 4: Readmission classifier ───────────────────────────────────────
    print("\n[4/8] Training readmission classifier (XGBoost + Calibration)...")
    clf = ReadmissionClassifier(calibrate=True)
    clf.fit(
        readmit_data["X_train"], readmit_data["y_train"],
        readmit_data["X_val"],   readmit_data["y_val"],
    )
    readmit_metrics = clf.evaluate(readmit_data["X_test"], readmit_data["y_test"])

    # ROC + PR curves
    proba_test = clf.predict_proba(readmit_data["X_test"])
    plot_roc_pr(readmit_data["y_test"].values, proba_test, "30-day Readmission")

    # Calibration curve
    frac_pos, mean_pred = clf.get_calibration_data(
        readmit_data["X_test"], readmit_data["y_test"]
    )
    plot_calibration(frac_pos, mean_pred, "30-day Readmission")

    # Risk band distribution
    risk_results = clf.predict_with_risk_bands(readmit_data["X_test"])
    print(f"\n  Risk band distribution (test set):")
    print(risk_results["risk_band"].value_counts().to_string())

    # ── STEP 5: Anomaly / Upcoding detection ─────────────────────────────────
    print("\n[5/8] Training anomaly detector (Isolation Forest)...")
    detector = UpcodeAnomalyDetector()
    detector.fit(full_df)

    anomaly_results = detector.detect(full_df)
    provider_risk   = detector.provider_risk_report(full_df, anomaly_results)

    # Evaluate against synthetic ground truth
    anomaly_metrics = detector.evaluate_against_labels(
        anomaly_results, full_df["_is_upcoded"]
    )

    # Anomaly plot
    plot_anomaly_distribution(anomaly_results, full_df)

    # ── STEP 6: Provider benchmarking ────────────────────────────────────────
    print("\n[6/8] Running provider benchmarking (K-means + percentile ranking)...")

    # Use all-data cost predictions for benchmarking
    all_cost_features = [f for f in cost_data["feature_names"] if f in full_df.columns]
    X_all = full_df[all_cost_features].fillna(0)
    all_pred_costs = cost_model.predict_cost(X_all)

    benchmarker = ProviderBenchmarker()
    benchmarker.fit(full_df, pd.Series(all_pred_costs, index=full_df.index))
    scorecard = benchmarker.score(full_df, pd.Series(all_pred_costs, index=full_df.index))

    # Plot provider scorecard
    plot_provider_scorecard(scorecard, top_n=20)

    # Save scorecards
    scorecard.to_csv(OUTPUT_DIR / "reports" / "provider_scorecard.csv", index=False)
    provider_risk.to_csv(OUTPUT_DIR / "reports" / "provider_upcoding_risk.csv", index=False)

    # ── STEP 7: Feature importance / Explainability ───────────────────────────
    print("\n[7/8] Computing feature importance...")

    # Cost model explainer
    print("  → Cost prediction importance...")
    cost_explainer = FeatureExplainer(
        cost_model.model,
        cost_data["X_train"].sample(min(200, len(cost_data["X_train"])), random_state=SEED),
        model_name="Cost Prediction"
    )
    cost_explainer.compute(cost_data["X_test"].sample(min(500, len(cost_data["X_test"])), random_state=SEED))
    cost_explainer.plot_importance_bar(top_n=15)
    print(cost_explainer.global_importance(10).to_string(index=False))

    # Readmission model explainer — use base_model (pre-calibration)
    print("\n  → Readmission importance...")
    readmit_explainer = FeatureExplainer(
        clf.base_model,
        readmit_data["X_train"].sample(min(200, len(readmit_data["X_train"])), random_state=SEED),
        model_name="Readmission Risk"
    )
    readmit_explainer.compute(
        readmit_data["X_test"].sample(min(500, len(readmit_data["X_test"])), random_state=SEED)
    )
    readmit_explainer.plot_importance_bar(top_n=15)
    print(readmit_explainer.global_importance(10).to_string(index=False))

    # Example: explain one high-risk episode
    high_risk_idx = risk_results[risk_results["risk_band"] == "Very High"].index
    if len(high_risk_idx) > 0:
        sample_X = readmit_data["X_test"].sample(min(500, len(readmit_data["X_test"])), random_state=SEED)
        common   = [i for i in high_risk_idx if i in sample_X.index]
        if common:
            pos = sample_X.index.get_loc(common[0])
            explanation = readmit_explainer.explain_episode(pos)
            print(f"\n  Episode explanation (Very High readmission risk):")
            print(explanation[["feature", "value", "shap_value", "direction"]].to_string(index=False))

    # ── STEP 8: Save models + summary report ─────────────────────────────────
    print("\n[8/8] Saving models and generating report...")
    cost_model.save()
    clf.save()
    detector.save()
    benchmarker.save()

    all_metrics = {
        "Cost Prediction (XGBoost Regressor)": cost_metrics,
        "Readmission Classifier (XGBoost + Calibration)": readmit_metrics,
        "Anomaly Detector (Isolation Forest)": anomaly_metrics,
        "Provider Benchmarking": {
            "n_providers":     len(scorecard),
            "peer_groups":     benchmarker.n_clusters,
            "cost_outliers":   int(scorecard["cost_outlier"].sum()),
            "los_outliers":    int(scorecard["los_outlier"].sum()),
            "readmit_outliers":int(scorecard["readmit_outlier"].sum()),
        }
    }
    save_summary_report(all_metrics)

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  Models   → outputs/models/")
    print(f"  Plots    → outputs/plots/")
    print(f"  Reports  → outputs/reports/")
    print("=" * 60)

    return {
        "episodes_df":    episodes_df,
        "scorecard":      scorecard,
        "cost_results":   cost_results,
        "risk_results":   risk_results,
        "anomaly_results":anomaly_results,
        "provider_risk":  provider_risk,
        "metrics":        all_metrics,
    }


if __name__ == "__main__":
    results = run_pipeline()
