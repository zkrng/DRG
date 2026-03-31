"""
config.py — Central configuration for the DRG Analytics DS project.
All constants, paths, and hyperparameters live here.
"""
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
DATA_DIR   = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR  = OUTPUT_DIR / "models"
PLOT_DIR   = OUTPUT_DIR / "plots"
REPORT_DIR = OUTPUT_DIR / "reports"

for d in [DATA_DIR, MODEL_DIR, PLOT_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Random seed ───────────────────────────────────────────────────────────────
SEED = 30

# ── Data generation ───────────────────────────────────────────────────────────
N_EPISODES   = 5_000   # total synthetic episodes
N_PROVIDERS  = 40      # unique providers in the network
N_DRG_CODES  = 30      # DRG codes simulated

# MS-DRG codes with relative weights (simplified subset)
# Format: {code: (description, relative_weight, geometric_mean_los)}
DRG_TABLE = {
    "470": ("Major joint replacement",         2.10, 2.7),
    "291": ("Heart failure w MCC",             2.74, 5.6),
    "292": ("Heart failure w CC",              1.52, 4.1),
    "293": ("Heart failure w/o CC/MCC",        0.98, 2.8),
    "392": ("Esophagitis/gastroenteritis w MCC",1.32, 4.0),
    "603": ("Cellulitis w MCC",                1.64, 5.1),
    "641": ("Nutritional disorders w MCC",     1.21, 4.8),
    "683": ("Renal failure w MCC",             1.98, 5.3),
    "690": ("UTI w MCC",                       1.42, 4.5),
    "194": ("Simple pneumonia w MCC",          1.95, 5.2),
    "195": ("Simple pneumonia w CC",           1.18, 3.9),
    "247": ("Perc cardiovasc proc w DES",      3.86, 2.3),
    "460": ("Spinal fusion except cervical",   4.21, 2.5),
    "064": ("Intracranial haem w MCC",         4.15, 6.1),
    "065": ("Intracranial haem w CC",          2.43, 4.7),
    "871": ("Septicaemia w MV >96hrs",         7.65, 9.2),
    "872": ("Septicaemia w/o MV w MCC",        2.10, 5.8),
    "189": ("Pulmonary oedema w MCC",          1.87, 5.0),
    "312": ("Syncope & collapse",              0.88, 2.4),
    "378": ("GI haemorrhage w MCC",            2.01, 4.8),
    "536": ("Fractures of hip & pelvis w MCC", 1.55, 4.2),
    "637": ("Diabetes w MCC",                  1.41, 4.3),
    "638": ("Diabetes w CC",                   0.93, 3.1),
    "177": ("Respiratory infections w MCC",    2.55, 6.4),
    "418": ("Laparoscopic cholecystectomy",    2.44, 2.0),
    "552": ("Medical back problems w MCC",     1.30, 4.0),
    "101": ("Seizures w MCC",                  1.82, 4.6),
    "308": ("Cardiac arrhythmia w MCC",        1.76, 4.1),
    "280": ("Acute MI discharged alive w MCC", 2.93, 5.0),
    "945": ("Rehabilitation w CC/MCC",         1.44, 9.8),
}

# Specialty mapping for providers
SPECIALTIES = [
    "Cardiology", "Orthopaedics", "General Surgery",
    "Neurology", "Respiratory", "Nephrology",
    "General Medicine", "Gastroenterology",
]

# Volume tiers (annual episodes)
VOLUME_TIERS = {
    "High":   (800, 1200),
    "Medium": (300, 799),
    "Low":    (50,  299),
}

# Charlson comorbidity conditions and their weights
CHARLSON_CONDITIONS = {
    "mi":            1,   # Myocardial infarction
    "chf":           1,   # Congestive heart failure
    "peripheral_vd": 1,   # Peripheral vascular disease
    "cerebrovascular":1,  # Cerebrovascular disease
    "dementia":      1,
    "copd":          1,   # Chronic obstructive pulmonary disease
    "connective_td": 1,   # Connective tissue disease
    "peptic_ulcer":  1,
    "liver_mild":    1,   # Mild liver disease
    "diabetes_unc":  1,   # Diabetes uncomplicated
    "diabetes_comp": 2,   # Diabetes with complications
    "hemiplegia":    2,
    "renal_mod_sev": 2,   # Moderate/severe renal disease
    "cancer":        2,
    "liver_mod_sev": 3,
    "metastatic":    6,
    "aids":          6,
}

# ── Model hyperparameters ─────────────────────────────────────────────────────
COST_MODEL_PARAMS = {
    "n_estimators":   400,
    "max_depth":      6,
    "learning_rate":  0.05,
    "subsample":      0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":      0.1,
    "reg_lambda":     1.0,
    "random_state":   SEED,
    "n_jobs":        -1,
}

READMISSION_MODEL_PARAMS = {
    "n_estimators":   300,
    "max_depth":      5,
    "learning_rate":  0.05,
    "subsample":      0.8,
    "colsample_bytree": 0.7,
    "scale_pos_weight": 4,   # handles class imbalance (~20% readmission rate)
    "use_label_encoder": False,
    "eval_metric":    "auc",
    "random_state":   SEED,
    "n_jobs":        -1,
}

ANOMALY_CONTAMINATION = 0.08   # estimated ~8% upcoding rate in dataset

KMEANS_CLUSTERS = 6            # peer group clusters

# ── Train/test split ──────────────────────────────────────────────────────────
TEST_SIZE   = 0.20
VAL_SIZE    = 0.10     # of training set

# ── Cost bands for flagging ───────────────────────────────────────────────────
COST_VARIANCE_THRESHOLD = 0.35   # flag if actual > 35% above predicted
LOS_VARIANCE_THRESHOLD  = 2.0    # flag if LoS > 2 days above DRG geometric mean
OUTLIER_PERCENTILE      = 90     # flag providers above 90th percentile in peer group
