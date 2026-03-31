# DRG DS Project — Model Summary Report


## Cost Prediction (XGBoost Regressor)

- **train_rmse**: 0.0624
- **test_rmse**: 0.1697
- **train_mae**: 0.0464
- **test_mae**: 0.1357
- **train_r2**: 0.9873
- **test_r2**: 0.8985

## Readmission Classifier (XGBoost + Calibration)

- **auc_roc**: 0.7802
- **auc_pr**: 0.4662
- **threshold**: 0.462
- **f1**: 0.3359
- **sensitivity**: 0.2353
- **specificity**: 0.9619
- **ppv**: 0.5867
- **npv**: 0.8454

## Anomaly Detector (Isolation Forest)

- **auc_roc**: 0.4981425543936625
- **auc_pr**: 0.08042780324374374
- **precision**: 0.08571428571367348
- **recall**: 0.030534351144960472

## Provider Benchmarking

- **n_providers**: 40
- **peer_groups**: 6
- **cost_outliers**: 7
- **los_outliers**: 7
- **readmit_outliers**: 7