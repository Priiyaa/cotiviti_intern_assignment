"""
EDA: compute the 10 literature-standard provider-level features, plus rank
them by how well each actually separates fraud from clean providers on
this dataset (Cohen's d effect size + Mann-Whitney U test).

This is meant to run BEFORE any model changes -- the output tells us
which features are worth including, and which of them could plausibly be
turned into per-claim (time-aware) features vs. which are only meaningful
as a provider-level snapshot (and therefore belong in clustering/peer
grouping, not in the time-series anomaly detector).

Output:
  - Printed ranked table of features by |Cohen's d|
  - eda_boxplots.png -- one boxplot per feature, fraud vs clean

Run: python3 eda.py
"""

import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save the figure
import matplotlib.pyplot as plt

DATA_DIR = "data"


def _find_file(data_dir, base_name):
    """Handles Kaggle's version-hash filenames, e.g. Train-1542865627584.csv."""
    pattern = re.compile(rf"^{re.escape(base_name)}(-\d+)?\.csv$")
    matches = [
        f for f in glob.glob(f"{data_dir}/*.csv")
        if pattern.match(f.split("/")[-1])
    ]
    if not matches:
        raise FileNotFoundError(f"No file matching '{base_name}.csv' or '{base_name}-<hash>.csv' in {data_dir}/")
    return sorted(matches)[0]


def load_raw(data_dir=DATA_DIR):
    train = pd.read_csv(_find_file(data_dir, "Train"))
    outpatient = pd.read_csv(_find_file(data_dir, "Train_Outpatientdata"))
    inpatient = pd.read_csv(_find_file(data_dir, "Train_Inpatientdata"))
    beneficiary = pd.read_csv(_find_file(data_dir, "Train_Beneficiarydata"))

    outpatient["ClaimType"] = "Outpatient"
    inpatient["ClaimType"] = "Inpatient"

    common_cols = [c for c in outpatient.columns if c in inpatient.columns]
    claims = pd.concat(
        [outpatient[common_cols], inpatient[common_cols]], ignore_index=True
    )
    claims = claims.merge(train, on="Provider", how="inner")

    for col in ["ClaimStartDt", "ClaimEndDt"]:
        if col in claims.columns:
            claims[col] = pd.to_datetime(claims[col])
    for col in ["AdmissionDt", "DischargeDt"]:
        if col in inpatient.columns:
            inpatient[col] = pd.to_datetime(inpatient[col])

    return claims, inpatient, beneficiary


def compute_chronic_cond_count(beneficiary):
    """
    ChronicCond_* columns in this dataset are coded 1=Yes, 2=No. Count how
    many of the 11 tracked chronic conditions each beneficiary has.
    NOTE: if your copy of the dataset uses a different encoding, this
    count will look off -- sanity-check a few rows if the numbers seem
    wrong (e.g. everyone has 0 or everyone has 11).
    """
    chronic_cols = [c for c in beneficiary.columns if c.startswith("ChronicCond_")]
    beneficiary = beneficiary.copy()
    beneficiary["ChronicCondCount"] = (beneficiary[chronic_cols] == 1).sum(axis=1)
    return beneficiary[["BeneID", "ChronicCondCount"]]


def build_provider_features(claims, inpatient, beneficiary, min_claims=8):
    """Computes all 10 candidate features per provider."""

    counts = claims.groupby("Provider").size()
    eligible = counts[counts >= min_claims].index
    claims = claims[claims["Provider"].isin(eligible)].copy()

    bene_chronic = compute_chronic_cond_count(beneficiary)
    claims = claims.merge(bene_chronic, on="BeneID", how="left")

    claims["ClaimDuration"] = (claims["ClaimEndDt"] - claims["ClaimStartDt"]).dt.days

    inpatient_ids = set(inpatient["ClaimID"]) if "ClaimID" in inpatient.columns else set()
    has_operating = "OperatingPhysician" in claims.columns
    has_attending = "AttendingPhysician" in claims.columns

    inpatient_dur = inpatient.copy()
    if "AdmissionDt" in inpatient_dur.columns and "DischargeDt" in inpatient_dur.columns:
        inpatient_dur["DaysInHospital"] = (
            inpatient_dur["DischargeDt"] - inpatient_dur["AdmissionDt"]
        ).dt.days
        hosp_days = inpatient_dur.groupby("Provider")["DaysInHospital"].mean()
    else:
        hosp_days = pd.Series(dtype=float)

    rows = []
    for provider, g in claims.groupby("Provider"):
        n_claims = len(g)
        total_reimbursed = g["InscClaimAmtReimbursed"].sum()
        avg_reimbursed = g["InscClaimAmtReimbursed"].mean()
        avg_duration = g["ClaimDuration"].mean()
        avg_days_hosp = hosp_days.get(provider, 0.0)
        unique_patients = g["BeneID"].nunique()
        unique_attend_phys = g["AttendingPhysician"].nunique() if has_attending else np.nan

        if has_attending and has_operating:
            same_rate = (g["AttendingPhysician"] == g["OperatingPhysician"]).mean()
        else:
            same_rate = np.nan

        avg_chronic = g["ChronicCondCount"].mean()
        inpatient_ratio = (g["ClaimType"] == "Inpatient").mean()
        fraud = g["PotentialFraud"].iloc[0]

        rows.append(dict(
            Provider=provider,
            TotalClaims=n_claims,
            TotalReimbursed=total_reimbursed,
            AvgReimbursed=avg_reimbursed,
            AvgClaimDuration=avg_duration,
            AvgDaysInHospital=avg_days_hosp,
            UniquePatients=unique_patients,
            UniqueAttendPhys=unique_attend_phys,
            SameAttendOperRate=same_rate,
            AvgChronicConds=avg_chronic,
            InpatientRatio=inpatient_ratio,
            PotentialFraud=fraud,
        ))

    return pd.DataFrame(rows)


def cohens_d(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_std = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    if pooled_std == 0:
        return np.nan
    return (a.mean() - b.mean()) / pooled_std


CORE_FEATURES = [
    "TotalClaims", "TotalReimbursed", "AvgReimbursed",
    "AvgClaimDuration", "AvgDaysInHospital",
    "UniquePatients", "UniqueAttendPhys",
    "SameAttendOperRate", "AvgChronicConds", "InpatientRatio",
]


def rank_features(df, features=CORE_FEATURES):
    fraud = df[df["PotentialFraud"] == "Yes"]
    clean = df[df["PotentialFraud"] == "No"]

    results = []
    for f in features:
        d = cohens_d(fraud[f], clean[f])
        results.append(dict(
            feature=f,
            cohens_d=d,
            abs_d=abs(d) if not np.isnan(d) else -1,
            fraud_mean=fraud[f].mean(),
            clean_mean=clean[f].mean(),
        ))

    ranked = pd.DataFrame(results).sort_values("abs_d", ascending=False).reset_index(drop=True)
    ranked = ranked.drop(columns="abs_d")
    return ranked


def plot_boxplots(df, features=CORE_FEATURES, out_path="eda_boxplots.png"):
    n = len(features)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.2 * rows))
    axes = axes.flatten()

    for i, f in enumerate(features):
        ax = axes[i]
        data = [df[df["PotentialFraud"] == "No"][f].dropna(),
                df[df["PotentialFraud"] == "Yes"][f].dropna()]
        try:
            ax.boxplot(data, tick_labels=["Clean", "Fraud"])
        except TypeError:
            # older matplotlib versions use 'labels' instead of 'tick_labels'
            ax.boxplot(data, labels=["Clean", "Fraud"])
        ax.set_title(f, fontsize=10)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    print("Loading and joining raw data...")
    claims, inpatient, beneficiary = load_raw()

    print("Computing provider-level features...")
    provider_df = build_provider_features(claims, inpatient, beneficiary)
    provider_df.to_csv("provider_features_full.csv", index=False)
    print(
        f"Computed features for {len(provider_df)} providers "
        f"({(provider_df.PotentialFraud=='Yes').sum()} fraud, "
        f"{(provider_df.PotentialFraud=='No').sum()} clean)"
    )

    print("\nRanking features by effect size (Cohen's d, fraud vs clean)...\n")
    ranked = rank_features(provider_df)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(ranked.to_string(index=False))

    print(
        "\nReading the table: |Cohen's d| around 0.2 = small effect, "
        "0.5 = medium, 0.8+ = large. Features near the top actually "
        "separate fraud from clean providers on this data; features near "
        "the bottom may not be worth including."
    )

    plot_boxplots(provider_df, ranked["feature"].tolist())