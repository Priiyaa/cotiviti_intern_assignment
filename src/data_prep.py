"""
Loads the Kaggle "Healthcare Provider Fraud Detection Analysis" dataset
(rohitrox/healthcare-provider-fraud-detection-analysis) and builds a
manageable, balanced sample of providers for the demo.

Expected files in ./data/ (download from Kaggle and place here):
  Train.csv                 -> Provider, PotentialFraud
  Train_Outpatientdata.csv
  Train_Inpatientdata.csv
  Train_Beneficiarydata.csv (not required for the current features, kept
                              here in case you want to extend the feature set)
"""

import glob
import re
import pandas as pd
import numpy as np


def _find_file(data_dir, base_name):
    """Matches Kaggle's downloaded filenames, which often carry a version
    hash suffix, e.g. Train-1542865627584.csv instead of Train.csv, or
    Train_Outpatientdata-1542865627584.csv. Matches base_name followed
    only by an optional hash and '.csv' -- so base_name="Train" matches
    "Train.csv" / "Train-123.csv" but NOT "Train_Outpatientdata-123.csv"
    or any Test_* file.
    """
    pattern = re.compile(rf"^{re.escape(base_name)}(-\d+)?\.csv$")
    matches = [
        f for f in glob.glob(f"{data_dir}/*.csv")
        if pattern.match(f.split("/")[-1])
    ]
    if not matches:
        raise FileNotFoundError(
            f"No file matching '{base_name}.csv' or '{base_name}-<hash>.csv' "
            f"found in {data_dir}/. Make sure you downloaded the Train_* CSVs "
            f"from the Kaggle dataset (not the Test_* files)."
        )
    return sorted(matches)[0]


def load_and_join(data_dir="data"):
    train = pd.read_csv(_find_file(data_dir, "Train"))
    outpatient = pd.read_csv(_find_file(data_dir, "Train_Outpatientdata"))
    inpatient = pd.read_csv(_find_file(data_dir, "Train_Inpatientdata"))

    # Outpatient and inpatient have mostly overlapping columns. Keep only
    # the columns both share so we can stack them into one claims table.
    common_cols = [c for c in outpatient.columns if c in inpatient.columns]
    claims = pd.concat(
        [outpatient[common_cols], inpatient[common_cols]],
        ignore_index=True,
    )

    claims = claims.merge(train, on="Provider", how="inner")

    claims["ClaimStartDt"] = pd.to_datetime(claims["ClaimStartDt"])
    claims["InscClaimAmtReimbursed"] = claims["InscClaimAmtReimbursed"].astype(float)

    return claims


def build_provider_sample(claims, n_fraud=20, n_clean=25, min_claims=8, seed=42):
    """
    Pull a bigger, balanced sample of providers than a tiny handful, so the
    anomaly score has enough signal to evaluate meaningfully. Filters out
    providers with too few claims to form a usable time series.
    """
    counts = claims.groupby("Provider").size()
    eligible = counts[counts >= min_claims].index
    df = claims[claims["Provider"].isin(eligible)]

    fraud_providers = df[df["PotentialFraud"] == "Yes"]["Provider"].unique()
    clean_providers = df[df["PotentialFraud"] == "No"]["Provider"].unique()

    rng = np.random.default_rng(seed)
    sel_fraud = rng.choice(
        fraud_providers, size=min(n_fraud, len(fraud_providers)), replace=False
    )
    sel_clean = rng.choice(
        clean_providers, size=min(n_clean, len(clean_providers)), replace=False
    )

    sample_providers = list(sel_fraud) + list(sel_clean)
    sample = df[df["Provider"].isin(sample_providers)].sort_values(
        ["Provider", "ClaimStartDt"]
    )
    return sample.reset_index(drop=True)


if __name__ == "__main__":
    claims = load_and_join("data")
    sample = build_provider_sample(claims)
    sample.to_csv("provider_sample.csv", index=False)
    print(
        f"Saved {len(sample)} claims across {sample['Provider'].nunique()} providers "
        f"({(sample.drop_duplicates('Provider')['PotentialFraud']=='Yes').sum()} fraud, "
        f"{(sample.drop_duplicates('Provider')['PotentialFraud']=='No').sum()} clean)"
    )