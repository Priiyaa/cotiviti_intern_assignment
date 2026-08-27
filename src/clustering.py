"""
Provider clustering, using the features EDA showed actually separate
fraud from clean providers (Cohen's d ranking on the real dataset):
TotalReimbursed (d=2.16), UniquePatients (d=1.27), TotalClaims (d=1.20),
InpatientRatio (d=0.98), UniqueAttendPhys (d=0.81), AvgChronicConds (d=0.50).

This replaces the earlier tercile split on average claim amount with a
real KMeans fit -- this is the "Clustering" focus area from the
assignment topic, done properly rather than as a placeholder for the
agent's peer-benchmark lookup.

SameAttendOperRate was dropped (d=-0.068, no real separation on this data).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

CLUSTER_FEATURE_COLS = [
    "TotalReimbursed", "UniquePatients", "TotalClaims",
    "InpatientRatio", "UniqueAttendPhys", "AvgChronicConds",
]

N_CLUSTERS = 3


def fit_kmeans(train_provider_df, n_clusters=N_CLUSTERS, feature_cols=CLUSTER_FEATURE_COLS, seed=42):
    X = train_provider_df[feature_cols].fillna(0).values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(Xs)
    return scaler, kmeans


def assign_clusters(provider_df, scaler, kmeans, feature_cols=CLUSTER_FEATURE_COLS):
    X = provider_df[feature_cols].fillna(0).values
    Xs = scaler.transform(X)
    labels = kmeans.predict(Xs)
    return pd.Series(labels, index=provider_df["Provider"].values, name="Cluster")


def compute_peer_benchmarks(train_provider_df, cluster_labels, feature_cols=CLUSTER_FEATURE_COLS):
    """
    Real per-cluster feature averages, computed from TRAIN providers only
    (never test, to avoid leaking test-set information into the agent's
    tool during a live demo evaluation).
    """
    df = train_provider_df.set_index("Provider").copy()
    df["Cluster"] = cluster_labels

    benchmarks = {}
    for c, g in df.groupby("Cluster"):
        benchmarks[str(int(c))] = {
            "n_providers_in_cluster": int(len(g)),
            "avg_total_reimbursed": round(float(g["TotalReimbursed"].mean()), 2),
            "avg_total_claims": round(float(g["TotalClaims"].mean()), 1),
            "avg_unique_patients": round(float(g["UniquePatients"].mean()), 1),
            "avg_inpatient_ratio": round(float(g["InpatientRatio"].mean()), 3),
            "avg_unique_attend_phys": round(float(g["UniqueAttendPhys"].mean()), 1),
            "avg_chronic_conds": round(float(g["AvgChronicConds"].mean()), 2),
        }
    return benchmarks