"""
DRG Specialized Analysis Tool
==============================
Upload any CMS Medicare Inpatient CSV and get:
  1. Scatter plot — submitted charges vs Medicare payment (outlier detection)
  2. Charge-to-payment ratio distribution
  3. Peer-group outlier flagging (within same MDC)
  4. Payment efficiency heatmap
  5. Summary outlier report (CSV export)

Required columns (CMS Geography & Service dataset):
  DRG_Desc, Tot_Dschrgs, Avg_Submtd_Cvrd_Chrg,
  Avg_Tot_Pymt_Amt, Avg_Mdcr_Pymt_Amt

Usage:
  pip install pandas matplotlib seaborn scipy numpy
  python drg_analysis.py                          # prompts for file
  python drg_analysis.py --file your_data.csv    # direct path
  python drg_analysis.py --file data.csv --out ./reports
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

# ── Colour palette (matches CMS/healthcare conventions) ──────────────────────
C_NORMAL  = "#378ADD"
C_OUTLIER = "#E24B4A"
C_LOWEFF  = "#BA7517"
C_REF     = "#1D9E75"
C_BG      = "#F8F8F6"
C_GRID    = "#E8E8E4"


# ── Column aliases ────────────────────────────────────────────────────────────
COL_ALIASES = {
    "drg_desc":    ["drg_desc", "drg description", "description", "ms-drg description", "DRG Definition"],
    "discharges":  ["tot_dschrgs", "total discharges", "discharges", "tot discharges", "Total Discharges"],
    "charges":     ["avg_submtd_cvrd_chrg", "avg submitted covered charges",
                    "average covered charges", "avg_cvrd_chrg", "Average Covered Charges"],
    "total_pay":   ["avg_tot_pymt_amt", "avg total payments", "average total payments", "Average Total Payments"],
    "medicare_pay":["avg_mdcr_pymt_amt", "avg medicare payments", "average medicare payments", "Average Medicare Payments"],
}

def resolve_columns(df: pd.DataFrame) -> dict:
    """Map canonical names to actual column names, case-insensitive."""
    lower_cols = {c.lower().strip(): c for c in df.columns}
    mapping = {}
    for canonical, aliases in COL_ALIASES.items():
        for alias in aliases:
            key = alias.lower().strip()
            if key in lower_cols:
                mapping[canonical] = lower_cols[key]
                break
        if canonical not in mapping:
            raise KeyError(
                f"Could not find column for '{canonical}'.\n"
                f"Expected one of: {aliases}\n"
                f"Found columns: {list(df.columns)}"
            )
    return mapping


def clean_money(series: pd.Series) -> pd.Series:
    """Strip $, commas, spaces from currency columns and convert to float."""
    return (series.astype(str)
                  .str.replace(r"[\$,\s]", "", regex=True)
                  .str.replace(r"[^\d\.]", "", regex=True)
                  .replace("", np.nan)
                  .astype(float))


def extract_mdc(drg_desc: pd.Series) -> pd.Series:
    """
    Derive a rough MDC label from the DRG description.
    Falls back to first 4 words as a grouping key.
    """
    system_keywords = {
        "CARDIAC": "Circulatory", "HEART": "Circulatory", "CORONARY": "Circulatory",
        "AMI": "Circulatory", "VASCULAR": "Circulatory",
        "PULMONARY": "Respiratory", "RESPIRATORY": "Respiratory", "PNEUMONIA": "Respiratory",
        "COPD": "Respiratory", "BRONCHITIS": "Respiratory",
        "RENAL": "Kidney/Urinary", "KIDNEY": "Kidney/Urinary", "URINARY": "Kidney/Urinary",
        "HEPATIC": "Hepatobiliary", "LIVER": "Hepatobiliary", "PANCREA": "Hepatobiliary",
        "CRANIOTOMY": "Nervous System", "INTRACRANIAL": "Nervous System",
        "NEUROLOGICAL": "Nervous System", "SEIZURE": "Nervous System", "STROKE": "Nervous System",
        "TRACHEOSTOMY": "Respiratory", "ECMO": "Respiratory",
        "HIP": "Musculoskeletal", "KNEE": "Musculoskeletal", "SPINE": "Musculoskeletal",
        "JOINT": "Musculoskeletal", "FRACTURE": "Musculoskeletal",
        "SEPTIC": "Infectious", "INFECTION": "Infectious",
        "TRANSPLANT": "Transplant", "BONE MARROW": "Transplant",
        "DIABETES": "Endocrine", "NUTRITIONAL": "Endocrine",
        "MENTAL": "Mental Health", "PSYCHO": "Mental Health",
        "OBSTETRIC": "Pregnancy/Childbirth", "CESAREAN": "Pregnancy/Childbirth",
        "NEOPLASM": "Oncology", "LYMPHOMA": "Oncology", "LEUKEMIA": "Oncology",
    }
    def _classify(desc):
        d = str(desc).upper()
        for kw, mdc in system_keywords.items():
            if kw in d:
                return mdc
        words = str(desc).split()
        return " ".join(words[:3]) if len(words) >= 3 else str(desc)[:30]
    return drg_desc.map(_classify)


def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag each DRG with three outlier categories:
      - 'outlier'  : charge-to-Medicare ratio > 3.5×
      - 'loweff'   : Medicare covers < 30 % of submitted charges
      - 'normal'   : neither
    Also adds peer-group Z-score within same MDC.
    """
    df = df.copy()
    df["ratio"]   = df["charges"] / df["medicare_pay"]
    df["mc_pct"]  = df["medicare_pay"] / df["charges"] * 100
    df["pt_pay"]  = df["total_pay"] - df["medicare_pay"]

    df["flag"] = "normal"
    df.loc[df["ratio"] > 3.5, "flag"]  = "outlier"
    df.loc[df["mc_pct"] < 30, "flag"]  = "loweff"

    # Peer-group Z-score (ratio within MDC)
    df["ratio_zscore"] = df.groupby("mdc")["ratio"].transform(
        lambda x: stats.zscore(x, ddof=0) if len(x) > 1 else pd.Series([0]*len(x), index=x.index)
    )
    df["peer_outlier"] = df["ratio_zscore"].abs() > 2.0
    return df


# ── Plot 1: Scatter — charges vs Medicare payment ────────────────────────────
def plot_scatter(df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(11, 7), facecolor=C_BG)
    ax.set_facecolor(C_BG)

    color_map = {"normal": C_NORMAL, "outlier": C_OUTLIER, "loweff": C_LOWEFF}

    for flag, grp in df.groupby("flag"):
        sizes = np.clip(np.sqrt(grp["discharges"]) * 8, 30, 400)
        ax.scatter(
            grp["charges"] / 1000, grp["medicare_pay"] / 1000,
            s=sizes, c=color_map[flag], alpha=0.72,
            edgecolors="white", linewidths=0.6, label=flag, zorder=3
        )

    # Reference line (1:1)
    lim = max(df["charges"].max(), df["medicare_pay"].max()) / 1000 * 1.05
    ax.plot([0, lim], [0, lim], color=C_REF, linestyle="--",
            linewidth=1.2, alpha=0.5, label="1:1 reference", zorder=2)

    # Annotate outliers
    top_outliers = df[df["flag"] == "outlier"].nlargest(6, "ratio")
    for _, row in top_outliers.iterrows():
        label = str(row["drg_desc"])[:35] + "…" if len(str(row["drg_desc"])) > 35 else str(row["drg_desc"])
        ax.annotate(
            label,
            xy=(row["charges"]/1000, row["medicare_pay"]/1000),
            xytext=(12, 6), textcoords="offset points",
            fontsize=7, color=C_OUTLIER, alpha=0.9,
            arrowprops=dict(arrowstyle="-", color=C_OUTLIER, alpha=0.4, lw=0.7)
        )

    ax.set_xlabel("Avg submitted charges ($000)", fontsize=11, color="#555")
    ax.set_ylabel("Avg Medicare payment ($000)", fontsize=11, color="#555")
    ax.set_title("Submitted charges vs Medicare payment\nBubble size = discharge volume",
                 fontsize=13, fontweight="bold", pad=14)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}k"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}k"))
    ax.grid(True, color=C_GRID, linewidth=0.7, zorder=1)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)

    patches = [
        mpatches.Patch(color=C_NORMAL,  label="Normal"),
        mpatches.Patch(color=C_OUTLIER, label="High ratio outlier (>3.5×)"),
        mpatches.Patch(color=C_LOWEFF,  label="Low efficiency (<30% coverage)"),
        plt.Line2D([0],[0], color=C_REF, linestyle="--", label="1:1 reference"),
    ]
    ax.legend(handles=patches, fontsize=9, framealpha=0.9, loc="upper left")

    fig.tight_layout()
    path = os.path.join(out_dir, "01_scatter_charges_vs_payment.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 2: Ratio distribution ───────────────────────────────────────────────
def plot_ratio_distribution(df: pd.DataFrame, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=C_BG)

    # Histogram of charge-to-Medicare ratio
    ax = axes[0]
    ax.set_facecolor(C_BG)
    n, bins, patches_ = ax.hist(df["ratio"], bins=30, color=C_NORMAL, edgecolor="white",
                                linewidth=0.5, alpha=0.85)
    # Colour outlier bins
    for patch, left in zip(patches_, bins[:-1]):
        if left >= 3.5:
            patch.set_facecolor(C_OUTLIER)
    ax.axvline(3.5, color=C_OUTLIER, linestyle="--", linewidth=1.4, label="Outlier threshold (3.5×)")
    ax.axvline(df["ratio"].median(), color=C_REF, linestyle=":", linewidth=1.4,
               label=f"Median ({df['ratio'].median():.1f}×)")
    ax.set_xlabel("Charge-to-Medicare ratio", fontsize=11)
    ax.set_ylabel("Number of DRGs", fontsize=11)
    ax.set_title("Distribution of charge-to-payment ratio", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", color=C_GRID, linewidth=0.7)

    # Box plot by MDC
    ax2 = axes[1]
    ax2.set_facecolor(C_BG)
    mdc_counts = df["mdc"].value_counts()
    top_mdcs = mdc_counts[mdc_counts >= 3].index.tolist()[:10]
    sub = df[df["mdc"].isin(top_mdcs)]
    order = sub.groupby("mdc")["ratio"].median().sort_values(ascending=False).index
    sns.boxplot(data=sub, x="ratio", y="mdc", order=order, ax=ax2,
                color=C_NORMAL, flierprops={"marker":"o","markersize":4,
                                             "markerfacecolor":C_OUTLIER,"alpha":0.6},
                linewidth=0.8, width=0.55)
    ax2.axvline(3.5, color=C_OUTLIER, linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.set_xlabel("Charge-to-Medicare ratio", fontsize=11)
    ax2.set_ylabel("")
    ax2.set_title("Ratio distribution by MDC group\n(top groups by count)", fontsize=12, fontweight="bold")
    ax2.grid(axis="x", color=C_GRID, linewidth=0.7)

    fig.tight_layout(pad=2)
    path = os.path.join(out_dir, "02_ratio_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 3: Payment efficiency heatmap ───────────────────────────────────────
def plot_efficiency_heatmap(df: pd.DataFrame, out_dir: str):
    """Medicare coverage % by MDC group — shows which clinical areas are underpriced."""
    mdc_counts = df["mdc"].value_counts()
    top_mdcs = mdc_counts[mdc_counts >= 2].index.tolist()[:15]
    sub = df[df["mdc"].isin(top_mdcs)].copy()

    pivot = sub.groupby("mdc").agg(
        avg_mc_pct   = ("mc_pct",     "mean"),
        avg_ratio    = ("ratio",      "mean"),
        n_drgs       = ("drg_desc",   "count"),
        n_outliers   = ("flag",       lambda x: (x == "outlier").sum()),
        avg_discharges=("discharges", "mean"),
    ).round(1).sort_values("avg_mc_pct")

    fig, ax = plt.subplots(figsize=(9, max(5, len(pivot)*0.55 + 1.5)), facecolor=C_BG)
    ax.set_facecolor(C_BG)

    y_pos = np.arange(len(pivot))
    bars = ax.barh(y_pos, pivot["avg_mc_pct"], color=[
        C_OUTLIER if v < 30 else (C_LOWEFF if v < 40 else C_NORMAL)
        for v in pivot["avg_mc_pct"]
    ], edgecolor="white", linewidth=0.5, height=0.65, alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.axvline(30, color=C_OUTLIER, linestyle="--", linewidth=1.2, alpha=0.7, label="Low efficiency (30%)")
    ax.axvline(40, color=C_LOWEFF,  linestyle=":",  linewidth=1.0, alpha=0.7, label="Caution zone (40%)")

    for i, (_, row) in enumerate(pivot.iterrows()):
        ax.text(row["avg_mc_pct"] + 0.5, i,
                f"{row['avg_mc_pct']:.0f}%  (ratio {row['avg_ratio']:.1f}×, {row['n_outliers']} outliers)",
                va="center", fontsize=8.5, color="#444")

    ax.set_xlabel("Avg Medicare payment as % of submitted charges", fontsize=11)
    ax.set_title("Payment efficiency by clinical group\n(lower % = bigger gap between charges and Medicare payment)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, max(pivot["avg_mc_pct"]) * 1.35)
    ax.grid(axis="x", color=C_GRID, linewidth=0.7)

    fig.tight_layout()
    path = os.path.join(out_dir, "03_efficiency_by_mdc.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 4: Peer-group outliers ───────────────────────────────────────────────
def plot_peer_outliers(df: pd.DataFrame, out_dir: str):
    """Within-MDC Z-score: highlights DRGs that are outliers even among peers."""
    peer_outs = df[df["peer_outlier"]].nlargest(20, "ratio_zscore")
    if peer_outs.empty:
        print("  No peer-group outliers found (Z > 2.0 within MDC) — skipping plot 4.")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(peer_outs)*0.5 + 1.5)), facecolor=C_BG)
    ax.set_facecolor(C_BG)

    labels = [str(d)[:45]+"…" if len(str(d))>45 else str(d)
              for d in peer_outs["drg_desc"]]
    colors = [C_OUTLIER if z > 3 else C_LOWEFF for z in peer_outs["ratio_zscore"]]

    ax.barh(range(len(peer_outs)), peer_outs["ratio_zscore"],
            color=colors, edgecolor="white", linewidth=0.4, height=0.7, alpha=0.85)
    ax.set_yticks(range(len(peer_outs)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(2.0, color="#888", linestyle="--", linewidth=1, alpha=0.6, label="Z = 2.0 threshold")

    for i, (_, row) in enumerate(peer_outs.iterrows()):
        ax.text(row["ratio_zscore"] + 0.05, i,
                f"Z={row['ratio_zscore']:.1f}  MDC: {row['mdc']}",
                va="center", fontsize=8, color="#444")

    ax.set_xlabel("Within-MDC Z-score of charge-to-Medicare ratio", fontsize=11)
    ax.set_title("Peer-group outliers — DRGs with anomalous ratios within their clinical group",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=9); ax.grid(axis="x", color=C_GRID, linewidth=0.7)
    fig.tight_layout()
    path = os.path.join(out_dir, "04_peer_group_outliers.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Report: CSV export ────────────────────────────────────────────────────────
def export_report(df: pd.DataFrame, out_dir: str):
    report_cols = [
        "drg_desc", "mdc", "discharges",
        "charges", "total_pay", "medicare_pay", "pt_pay",
        "ratio", "mc_pct", "ratio_zscore", "flag", "peer_outlier"
    ]
    report = df[report_cols].copy()
    report.columns = [
        "DRG Description", "MDC Group", "Total Discharges",
        "Avg Submitted Charges", "Avg Total Payment", "Avg Medicare Payment",
        "Avg Patient/Other Payment", "Charge:Medicare Ratio", "Medicare Coverage %",
        "Peer Z-Score", "Outlier Flag", "Peer Group Outlier"
    ]
    report = report.sort_values("Charge:Medicare Ratio", ascending=False)
    path = os.path.join(out_dir, "drg_outlier_report.csv")
    report.to_csv(path, index=False, float_format="%.2f")
    print(f"  Saved: {path}")

    # Print summary
    print("\n── Summary ───────────────────────────────────────────")
    print(f"  Total DRGs analysed   : {len(df)}")
    print(f"  High-ratio outliers   : {(df['flag']=='outlier').sum()}  (ratio > 3.5×)")
    print(f"  Low-efficiency DRGs   : {(df['flag']=='loweff').sum()}  (Medicare < 30%)")
    print(f"  Peer-group outliers   : {df['peer_outlier'].sum()}  (Z > 2.0 within MDC)")
    print(f"  Avg charge-to-pay ratio: {df['ratio'].mean():.2f}×")
    print(f"  Median coverage %     : {df['mc_pct'].median():.1f}%")
    print("\n── Top 5 High-Ratio Outliers ─────────────────────────")
    top5 = df[df["flag"]=="outlier"].nlargest(5, "ratio")[
        ["drg_desc","charges","medicare_pay","ratio","discharges"]]
    for _, r in top5.iterrows():
        print(f"  {str(r['drg_desc'])[:55]:<56} ratio={r['ratio']:.1f}×  "
              f"charges=${r['charges']:,.0f}  medicare=${r['medicare_pay']:,.0f}  n={r['discharges']:.0f}")
    print("──────────────────────────────────────────────────────")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DRG Outlier Analysis Tool")
    parser.add_argument("--file", "-f", type=str, default=None,
                        help="Path to CMS Medicare Inpatient CSV file")
    parser.add_argument("--out", "-o", type=str, default="./drg_output",
                        help="Output directory for plots and report")
    args = parser.parse_args()

    # ── File selection ────────────────────────────────────────────────────────
    if args.file:
        csv_path = args.file
    else:
        print("\nDRG Specialized Analysis Tool")
        print("─" * 40)
        csv_path = input("Enter path to your CMS Medicare Inpatient CSV: ").strip().strip("\"'")

    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    # ── Output directory ──────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    print(f"\nOutput directory: {os.path.abspath(args.out)}")

    # ── Load & clean ──────────────────────────────────────────────────────────
    print(f"\nLoading: {csv_path}")
    try:
        df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)

    print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    col_map = resolve_columns(df)
    print(f"  Column mapping: {col_map}")

    # Rename to canonical
    df = df.rename(columns={v: k for k, v in col_map.items()})

    # Clean money columns
    for col in ["charges", "total_pay", "medicare_pay"]:
        df[col] = clean_money(df[col])
    df["discharges"] = pd.to_numeric(
        df["discharges"].astype(str).str.replace(",",""), errors="coerce")

    # Drop rows with missing key values
    df = df.dropna(subset=["charges","total_pay","medicare_pay","discharges"])
    df = df[df["medicare_pay"] > 0]
    print(f"  Clean rows: {len(df):,}")

    # Derive MDC grouping
    df["mdc"] = extract_mdc(df["drg_desc"])

    # Flag outliers
    df = flag_outliers(df)

    # ── Generate plots ────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_scatter(df, args.out)
    plot_ratio_distribution(df, args.out)
    plot_efficiency_heatmap(df, args.out)
    plot_peer_outliers(df, args.out)

    # ── Export report ─────────────────────────────────────────────────────────
    print("\nExporting outlier report...")
    export_report(df, args.out)

    print(f"\nDone. All outputs saved to: {os.path.abspath(args.out)}")
    print("Open the PNG files to view charts, or drg_outlier_report.csv for the full data.")


if __name__ == "__main__":
    main()
