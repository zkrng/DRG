# DRG Analytics — Data Science Project
### Cost Efficiency & Provider Benchmarking for Health Insurance Providers

---

## 📚 Literature Review & Market Research

### 1. Problem Context

Diagnosis-Related Groups (DRGs) are a prospective hospital payment system that classifies
inpatient episodes by clinical similarity and expected resource use. Introduced in the US
in 1983, they are now used across Europe, Australia, and increasingly Asia. From a health
insurer's perspective, DRG data unlocks three high-value analytics opportunities:

| Opportunity | Description | Business Value |
|---|---|---|
| **Cost prediction** | Predict expected episode cost before discharge | Actuarial pricing, reserve estimation |
| **Provider benchmarking** | Compare providers on cost/quality within peer groups | Network optimisation, contract negotiation |
| **Upcoding detection** | Detect inflated DRG code assignment | Fraud prevention, $100B+ annual US losses |

---

### 2. Literature Review Summary

#### 2.1 DRG Cost Prediction with ML

**Key finding (npj Digital Medicine, 2021):** A deep learning NLP model achieved macro-AUC
of 0.871 (MS-DRG) and 0.884 (APR-DRG) predicting DRG codes from clinical notes on the
*first day* of ICU admission — enabling *prospective* cost management rather than
retrospective coding.

**Key finding (BMC Health Services Research, 2024):** Using 2.3M patient records from
NY SPARCS, XGBoost with target-encoded features outperformed linear models for Length of
Stay (LoS) prediction. Top SHAP features: APR DRG Code, Severity of Illness Code, Patient
Disposition, CCS Procedure Code.

**Key finding (Atlantis Press, 2025):** XGBoost and Random Forest both outperformed
Linear Regression for predicting APR-DRG inpatient costs using demographics, diagnoses,
and facility utilisation features from claims data.

**Methodological consensus across studies:**
- Gradient-boosted trees (XGBoost, LightGBM) consistently outperform linear baselines
- SHAP values are the standard explainability approach in clinical settings
- Class imbalance requires SMOTE or class_weight adjustments
- Target encoding outperforms one-hot for high-cardinality categoricals (e.g. DRG codes)

#### 2.2 Provider Benchmarking

**Key finding (BMC Medical Informatics, 2021):** ML-based DRG grouping using decision
trees, random forests, SVM, and neural networks showed kappa coefficients comparable to
expert-oriented grouping — with the advantage of continuous, low-cost updates from new
data. Random Forest performed best overall.

**Key finding (DRGKB, Oxford Academic, 2024):** A worldwide DRG knowledgebase review
found that ML-based cost calculation methods (particularly Random Forest) offer
"advantages of high transparency and efficiency for cost interpretation and LoS prediction"
vs. expert rule systems.

**Provider benchmarking approach:** Peer group segmentation using K-means clustering on
specialty + volume + case-mix complexity, followed by percentile ranking within cohort.
Isolation Forest reliably detects outlier providers.

#### 2.3 Upcoding & Fraud Detection

**Key finding (ScienceDirect systematic review, 2024):** Reviewed 147 ML papers on
healthcare fraud detection (2000–2024). Of these, 94 used supervised, 41 used
unsupervised, and 12 used hybrid approaches. Ensemble methods dominate; SHAP is critical
for regulators.

**Key finding (Frontiers in AI, 2025):** For cardiovascular DRG management, deep learning
models that predict high-cost complications allow proactive intervention — preventing
cost overruns beyond the fixed DRG reimbursement ceiling.

**Key finding (JMIR Medical Informatics, 2022):** XGBoost with regularisation and deep
learning models are most effective for readmission charge prediction — the metric most
correlated with upcoding risk.

---

### 3. Market Research

#### 3.1 Healthcare Fraud Detection Market

| Metric | Value |
|---|---|
| US annual healthcare fraud losses | >$100 billion (NHCAA) |
| Global fraud detection market (2024) | ~$4.8 billion |
| Projected market size (2032) | $15.36 billion |
| CAGR | ~21% |
| CMS fraud control budget (FY2025) | $941 million |

Source: SNS Insider (2025), Mordor Intelligence (2025)

#### 3.2 Anomaly Detection Market (broad)

| Metric | Value |
|---|---|
| Global market size (2024) | $6.18–6.31 billion |
| Projected market size (2032) | ~$20 billion |
| CAGR | ~16% |
| Dominant vertical | BFSI (29%), Healthcare growing fast |
| Fastest-growing tech segment | ML/AI anomaly detection (18.9% CAGR) |

Source: Verified Market Research, Precedence Research (2025)

#### 3.3 Competitive Landscape

| Vendor | Approach | Strength |
|---|---|---|
| IBM Watson Health | NLP + rules + ML | EHR integration |
| SAS Fraud Framework | Statistical + supervised ML | Regulatory compliance |
| Optum | Claims + clinical data fusion | Payer-native |
| Cotiviti | DRG auditing + benchmarking | Insurer focus |
| Health Catalyst | Data warehouse + ML | Analytics platform |

#### 3.4 Key Trends (2024–2026)

1. **LLM-assisted DRG coding** — DRG-LLaMA (fine-tuned LLaMA) now predicts DRG from
   clinical notes with strong accuracy; reduces manual coding effort
2. **Value-Based Care integration** — DRG payments increasingly linked to outcome metrics,
   not just episode classification
3. **Asia-Pacific expansion** — China, Japan, South Korea actively deploying DRG systems;
   18–21% CAGR in APAC fraud detection
4. **Real-time scoring** — Shift from monthly batch benchmarking to real-time claim-level
   scoring at point of submission
5. **Explainability mandates** — Regulators increasingly require SHAP-level explanations
   for any claim denial or provider flag

---

## 🏗️ Project Structure

```
drg_ds_project/
├── README.md                        ← You are here
├── requirements.txt                 ← All dependencies
├── config.py                        ← Central config / constants
│
├── data/
│   └── (generated by pipeline)
│
├── src/
│   ├── data/
│   │   ├── generator.py             ← Synthetic data generator (5,000 episodes)
│   │   └── preprocessor.py         ← Cleaning, encoding, train/test split
│   │
│   ├── features/
│   │   ├── feature_engineer.py      ← Feature engineering (Charlson, peer groups, etc.)
│   │   └── feature_store.py         ← Simple in-memory feature cache
│   │
│   ├── models/
│   │   ├── cost_predictor.py        ← XGBoost regression (expected episode cost)
│   │   ├── readmission_classifier.py← XGBoost binary classifier (30-day readmission)
│   │   ├── anomaly_detector.py      ← Isolation Forest (upcoding / outlier detection)
│   │   └── provider_benchmarker.py  ← K-means clustering + percentile ranking
│   │
│   ├── evaluation/
│   │   └── metrics.py               ← All evaluation metrics + reports
│   │
│   └── visualization/
│       └── plots.py                 ← EDA + model result plots
│
├── notebooks/
│   └── 01_full_pipeline.ipynb       ← End-to-end walkthrough notebook
│
├── outputs/                         ← Saved models, plots, reports
│
├── pipeline.py                      ← Main orchestrator — run this!
└── tests/
    └── test_pipeline.py             ← Basic smoke tests
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python pipeline.py

# 3. Outputs saved to outputs/
```

---

## 🎓 Learning Objectives

This project is designed to teach a mid-level data scientist:

1. **Synthetic data generation** that reflects real clinical distributions
2. **Multi-target ML** — regression, binary classification, and unsupervised anomaly detection in one project
3. **Feature engineering** — Charlson comorbidity scoring, target encoding, peer group assignment
4. **Model stacking** — using XGBoost predictions as features for the anomaly detector
5. **SHAP explainability** — global feature importance + individual episode explanations
6. **Calibration** — probability calibration for the readmission classifier
7. **Class imbalance** — SMOTE + class_weight strategies
8. **Provider benchmarking** — K-means segmentation + percentile ranking within cohorts
9. **Pipeline orchestration** — sklearn Pipelines for reproducible preprocessing
10. **Model evaluation** — RMSE, MAE, AUC-ROC, calibration curves, confusion matrices
