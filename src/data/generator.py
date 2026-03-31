"""
src/data/generator.py
=====================
Generates realistic synthetic DRG episode data for the insurance DS project.

Design principles (grounded in literature):
  - DRG cost follows a log-normal distribution (right-skewed, as in real claims data)
  - LoS is Negative Binomial distributed around the DRG geometric mean
  - Charlson comorbidity index increases cost and LoS multiplicatively
  - ~8% of episodes contain simulated upcoding (DRG weight inflated)
  - ~20% 30-day readmission rate (consistent with US Medicare averages)
  - Provider-level random effects (some hospitals consistently cost more)

References:
  - Jain et al. BMC Health Services Research (2024) — LoS distributions
  - Faradisa et al. Atlantis Press (2025) — XGBoost on claims data
  - NHCAA estimates on healthcare fraud rates
"""

import numpy as np
import pandas as pd
import sys
import os
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):          # graceful fallback
        total = kwargs.get("total", "?")
        for i, item in enumerate(iterable):
            if i % 1000 == 0:
                print(f"  ... {i} episodes generated", flush=True)
            yield item
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import (
    SEED, N_EPISODES, N_PROVIDERS, DRG_TABLE,
    SPECIALTIES, VOLUME_TIERS, CHARLSON_CONDITIONS, DATA_DIR
)


def _charlson_score(rng: np.random.Generator, age: int, severity: int) -> tuple[dict, int]:
    """
    Sample a Charlson comorbidity profile and compute the total index score.
    Probability of each condition increases with age and severity.

    Returns (condition_dict, total_score)
    """
    base_p = 0.05 + (age - 18) / 500 + (severity - 1) * 0.06
    base_p = min(base_p, 0.45)

    conditions = {}
    for cond, weight in CHARLSON_CONDITIONS.items():
        # Some conditions are more common than others
        p = base_p * (1.5 if weight >= 2 else 1.0)
        conditions[cond] = int(rng.random() < p)

    total = sum(conditions[c] * w for c, w in CHARLSON_CONDITIONS.items())
    return conditions, int(total)


def _generate_providers(rng: np.random.Generator) -> pd.DataFrame:
    """
    Build a provider reference table with specialty, volume tier, region,
    and a random cost efficiency factor (some hospitals are just more expensive).
    """
    rows = []
    for pid in range(1, N_PROVIDERS + 1):
        specialty = rng.choice(SPECIALTIES)
        tier_name = rng.choice(list(VOLUME_TIERS.keys()), p=[0.25, 0.45, 0.30])
        vol_lo, vol_hi = VOLUME_TIERS[tier_name]
        volume = int(rng.integers(vol_lo, vol_hi))
        region = rng.choice(["North", "South", "East", "West", "Central"])
        # Provider cost efficiency: 1.0 = average, >1 = expensive, <1 = lean
        cost_factor = float(rng.lognormal(mean=0.0, sigma=0.18))
        # Upcoding propensity: small fraction of providers upcode regularly
        upcode_propensity = float(rng.beta(1, 12))   # most near 0, a few > 0.3

        rows.append({
            "provider_id":        f"PRV{pid:03d}",
            "specialty":          specialty,
            "volume_tier":        tier_name,
            "annual_volume":      volume,
            "region":             region,
            "cost_factor":        cost_factor,
            "upcode_propensity":  upcode_propensity,
            "accreditation":      rng.choice(["Level I", "Level II", "Level III"],
                                             p=[0.20, 0.50, 0.30]),
        })
    return pd.DataFrame(rows)


def generate_dataset(n_episodes: int = N_EPISODES, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate the full synthetic episode dataset and provider reference table.

    Returns
    -------
    episodes_df : pd.DataFrame  — one row per hospital episode
    providers_df : pd.DataFrame — one row per provider
    """
    rng = np.random.default_rng(seed)
    providers_df = _generate_providers(rng)
    provider_ids = providers_df["provider_id"].tolist()

    drg_codes = list(DRG_TABLE.keys())
    drg_weights = np.array([DRG_TABLE[c][1] for c in drg_codes])
    # Weight sampling: higher-weight DRGs are less common
    drg_sampling_p = 1 / (drg_weights ** 0.5)
    drg_sampling_p /= drg_sampling_p.sum()

    rows = []
    print(f"Generating {n_episodes:,} synthetic DRG episodes...")

    for ep_id in tqdm(range(1, n_episodes + 1)):
        # --- Patient demographics ---
        age    = int(rng.integers(18, 90))
        gender = rng.choice(["M", "F"])
        payer  = rng.choice(
            ["Medicare", "Medicaid", "Private", "Self-pay"],
            p=[0.38, 0.18, 0.38, 0.06]
        )

        # --- Clinical episode ---
        drg_code = rng.choice(drg_codes, p=drg_sampling_p)
        drg_desc, drg_weight, geo_mean_los = DRG_TABLE[drg_code]
        severity = int(rng.choice([1, 2, 3, 4], p=[0.30, 0.40, 0.22, 0.08]))
        admission_type = rng.choice(
            ["Emergency", "Elective", "Urgent", "Newborn"],
            p=[0.45, 0.35, 0.18, 0.02]
        )

        # Charlson comorbidity
        charlson_conditions, charlson_score = _charlson_score(rng, age, severity)

        # --- Provider assignment ---
        provider_id = rng.choice(provider_ids)
        prov = providers_df[providers_df["provider_id"] == provider_id].iloc[0]

        # --- Length of Stay ---
        # Base LoS from DRG geometric mean, adjusted by severity and comorbidity
        los_mean = geo_mean_los * (1 + 0.15 * (severity - 1)) * (1 + 0.06 * charlson_score)
        los_mean *= prov["cost_factor"] ** 0.5   # provider effect on LoS
        # Negative binomial: mean = los_mean, overdispersion ~0.5
        los_r = 3.0
        los_p = los_r / (los_r + los_mean)
        actual_los = int(rng.negative_binomial(n=los_r, p=los_p)) + 1

        # --- Upcoding simulation ---
        # A provider's upcode_propensity determines chance of inflated DRG weight
        is_upcoded = rng.random() < prov["upcode_propensity"]
        upcoding_factor = float(rng.uniform(1.15, 1.45)) if is_upcoded else 1.0
        effective_weight = drg_weight * upcoding_factor

        # --- Episode cost ---
        # Base cost = national base rate × effective DRG weight × comorbidity × provider
        # National base rate ~$6,500 (approximate US Medicare FY2024)
        base_rate = 6500
        cost_multiplier = (
            effective_weight
            * (1 + 0.08 * charlson_score)
            * (1 + 0.05 * (severity - 1))
            * prov["cost_factor"]
        )
        # Log-normal noise: real costs are right-skewed
        noise = float(rng.lognormal(mean=0.0, sigma=0.12))
        episode_cost = round(base_rate * cost_multiplier * noise, 2)

        # --- 30-day readmission ---
        # Driven by severity, comorbidity, LoS, age, and payer
        readmit_logit = (
            -4.8
            + 0.02 * age
            + 0.18 * charlson_score
            + 0.12 * severity
            + 0.02 * actual_los
            + (0.30 if payer == "Medicaid" else 0)
            + (-0.25 if admission_type == "Elective" else 0)
        )
        readmit_p = 1 / (1 + np.exp(-readmit_logit))
        readmitted_30d = int(rng.random() < readmit_p)

        # --- Discharge disposition ---
        disposition = rng.choice(
            ["Home", "Skilled nursing", "Rehab", "Expired", "AMA"],
            p=[0.60, 0.20, 0.12, 0.05, 0.03]
        )

        # --- Previous admissions (last 12 months) ---
        prev_admissions = int(rng.negative_binomial(n=1, p=0.7))

        # --- Build row ---
        row = {
            "episode_id":          f"EP{ep_id:06d}",
            "admission_date":      pd.Timestamp("2023-01-01") + pd.Timedelta(
                                       days=int(rng.integers(0, 730))
                                   ),
            # Patient
            "age":                 age,
            "gender":              gender,
            "payer":               payer,
            # Clinical
            "drg_code":            drg_code,
            "drg_description":     drg_desc,
            "drg_weight":          drg_weight,
            "severity":            severity,
            "admission_type":      admission_type,
            "disposition":         disposition,
            "actual_los":          actual_los,
            "geo_mean_los":        geo_mean_los,
            "los_variance":        actual_los - geo_mean_los,
            # Comorbidity
            "charlson_score":      charlson_score,
            **{f"cm_{k}": v for k, v in charlson_conditions.items()},
            # Provider
            "provider_id":         provider_id,
            "specialty":           prov["specialty"],
            "volume_tier":         prov["volume_tier"],
            "region":              prov["region"],
            # Outcomes
            "episode_cost":        episode_cost,
            "readmitted_30d":      readmitted_30d,
            "prev_admissions":     prev_admissions,
            # Ground truth labels (not available in real data!)
            "_is_upcoded":         int(is_upcoded),
            "_upcoding_factor":    round(upcoding_factor, 3),
            "_provider_cost_factor": round(prov["cost_factor"], 3),
        }
        rows.append(row)

    episodes_df = pd.DataFrame(rows)

    # Save raw data (CSV fallback if parquet engine not available)
    try:
        episodes_df.to_parquet(DATA_DIR / "episodes_raw.parquet", index=False)
        providers_df.to_parquet(DATA_DIR / "providers.parquet", index=False)
    except ImportError:
        episodes_df.to_csv(DATA_DIR / "episodes_raw.csv", index=False)
        providers_df.to_csv(DATA_DIR / "providers.csv", index=False)

    print(f"\n✓ Generated {len(episodes_df):,} episodes across {N_PROVIDERS} providers")
    print(f"  Readmission rate:  {episodes_df['readmitted_30d'].mean():.1%}")
    print(f"  Upcoding rate:     {episodes_df['_is_upcoded'].mean():.1%}")
    print(f"  Mean episode cost: ${episodes_df['episode_cost'].mean():,.0f}")
    print(f"  Median LoS:        {episodes_df['actual_los'].median():.1f} days")

    return episodes_df, providers_df


if __name__ == "__main__":
    df, prov = generate_dataset()
    print(df.head())
