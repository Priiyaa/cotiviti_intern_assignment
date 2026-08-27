"""
Full pipeline, run offline (not inside Streamlit):

  1. Load the FULL dataset (all eligible providers, not a small sample).
  2. Split providers into train/val/test (60/20/20, stratified by fraud
     label so each split has a realistic fraud rate).
  3. Fit the PCA-AE anomaly model on TRAIN claims from CLEAN providers
     ONLY. This is standard practice for reconstruction-error anomaly
     detection: the model needs to learn what NORMAL looks like, so that
     fraud (which it never saw) reconstructs poorly. Fitting on a mix of
     fraud+clean claims lets the dominant component absorb the fraud
     pattern too, making fraud claims reconstruct just as well as clean
     ones -- which is exactly backwards, and is what caused an
     early version of this pipeline to score AUC-ROC < 0.5 (worse than
     random) on the real dataset.
  4. Choose aggregation (mean vs max) and threshold on VAL, optimizing F2
     among thresholds that keep precision above a floor (prevents F2
     from degenerating into "flag almost everyone").
  5. Fit KMeans clustering on TRAIN providers (all of them, fraud+clean --
     clustering groups providers by size/case-mix for peer benchmarking,
     it isn't the anomaly signal itself, so this doesn't need the same
     clean-only restriction).
  6. Report the FINAL result -- metrics + confusion matrix -- on TEST,
     which the model and threshold never saw. This is the number that's
     actually honest to put in a report or say out loud in a demo.
  7. Save everything the Streamlit app needs to disk, so the app just
     loads results instead of re-fitting live on every run.

Run: python3 pipeline.py
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eda import load_raw, build_provider_features
from anomaly_model import (
    FEATURE_COLS,
    engineer_features,
    fit_pca_ae,
    reconstruction_error,
    score_providers,
    evaluate,
    best_threshold_evaluate,
    threshold_independent_scores,
    fbeta,
)
from clustering import fit_kmeans, assign_clusters, compute_peer_benchmarks
from classifier import fit_random_forest, score_providers as rf_score_providers, get_feature_importances, CLASSIFIER_FEATURE_COLS

MODEL_DIR = "models"
MIN_CLAIMS = 8
MIN_PRECISION = 0.30  # threshold sweep only considers cutoffs that keep precision >= this


def split_providers(provider_df, train_frac=0.6, val_frac=0.2, seed=42):
    train_val, test = train_test_split(
        provider_df,
        test_size=round(1 - train_frac - val_frac, 4),
        stratify=provider_df["PotentialFraud"],
        random_state=seed,
    )
    val_size = val_frac / (train_frac + val_frac)
    train, val = train_test_split(
        train_val, test_size=val_size,
        stratify=train_val["PotentialFraud"], random_state=seed,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def choose_agg_and_threshold(val_claims_scored, val_labels, metric="f2", min_precision=MIN_PRECISION):
    """Tries both mean and max aggregation on VAL, keeps whichever + whatever
    threshold gives the best F2 -- all decided on val, never on test.
    Restricted to thresholds keeping precision >= min_precision, so the
    result stays operationally usable rather than flagging everything."""
    best = None
    for agg in ["max", "mean"]:
        scores = score_providers(val_claims_scored, agg=agg)
        result = best_threshold_evaluate(scores, val_labels, metric=metric, min_precision=min_precision)
        result["agg"] = agg
        if best is None or result[metric] > best[metric]:
            best = result
    return best


def confusion_metrics(scores, labels, threshold):
    """Shared TP/FP/FN/TN + precision/recall/F1/F2 computation, used for
    both the PCA-AE and Random Forest test-set evaluations so the two
    are computed identically and stay comparable."""
    y_true = (labels.loc[scores.index] == "Yes")
    y_pred = scores >= threshold
    tp = int((y_pred & y_true).sum())
    fp = int((y_pred & ~y_true).sum())
    fn = int((~y_pred & y_true).sum())
    tn = int((~y_pred & ~y_true).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    f2 = fbeta(precision, recall, beta=2)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1, f2=f2, threshold=float(threshold))


def plot_confusion_matrix(cm_dict, out_path, title="Test Set Confusion Matrix (held out, final result)"):
    cm = np.array([[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, cm[i, j], ha="center", va="center", fontsize=15,
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: Clean", "Pred: Fraud"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual: Clean", "Actual: Fraud"])
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def run_pipeline(data_dir="data"):
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading raw data...")
    claims, inpatient, beneficiary = load_raw(data_dir)

    print("Engineering claim-level (time-series) features...")
    claims = engineer_features(claims)

    print("Building provider-level (clustering) features + filtering by min_claims...")
    provider_df = build_provider_features(claims, inpatient, beneficiary, min_claims=MIN_CLAIMS)
    eligible = set(provider_df["Provider"])
    claims = claims[claims["Provider"].isin(eligible)]

    print(
        f"{len(provider_df)} eligible providers "
        f"({(provider_df.PotentialFraud=='Yes').sum()} fraud, "
        f"{(provider_df.PotentialFraud=='No').sum()} clean)"
    )

    train_p, val_p, test_p = split_providers(provider_df)
    print(f"Split -> train: {len(train_p)}, val: {len(val_p)}, test: {len(test_p)}")

    train_claims = claims[claims["Provider"].isin(train_p["Provider"])]
    val_claims = claims[claims["Provider"].isin(val_p["Provider"])]
    test_claims = claims[claims["Provider"].isin(test_p["Provider"])]

    print("\nFitting PCA-AE anomaly model on TRAIN claims from CLEAN providers only...")
    print(
        "  (Standard practice for reconstruction-error anomaly detection: the model "
        "must learn what NORMAL looks like. Fitting on a mix of fraud+clean claims "
        "lets the dominant component absorb the fraud pattern too, so fraud claims "
        "end up reconstructed just as well as clean ones -- exactly backwards.)"
    )
    train_clean_providers = train_p[train_p["PotentialFraud"] == "No"]["Provider"]
    train_claims_clean = train_claims[train_claims["Provider"].isin(train_clean_providers)]
    print(f"  Fitting on {len(train_claims_clean)} claims from {len(train_clean_providers)} clean providers")
    scaler, pca = fit_pca_ae(train_claims_clean, feature_cols=FEATURE_COLS)

    val_scored = reconstruction_error(val_claims, scaler, pca, feature_cols=FEATURE_COLS)
    test_scored = reconstruction_error(test_claims, scaler, pca, feature_cols=FEATURE_COLS)

    val_labels = val_p.set_index("Provider")["PotentialFraud"]
    test_labels = test_p.set_index("Provider")["PotentialFraud"]

    print("Choosing aggregation (mean/max) + threshold on VAL, optimizing F2...")
    best = choose_agg_and_threshold(val_scored, val_labels)
    agg_mode, threshold, percentile = best["agg"], best["threshold"], best["percentile"]
    print(f"  Chosen: agg={agg_mode}, percentile={percentile}, val F2={best['f2']*100:.1f}%")

    print("\nFitting KMeans clustering on TRAIN providers...")
    cluster_scaler, kmeans = fit_kmeans(train_p)
    train_clusters = assign_clusters(train_p, cluster_scaler, kmeans)
    peer_benchmarks = compute_peer_benchmarks(train_p, train_clusters)

    print("\nScoring TEST set (final, held-out result -- model + threshold never saw this data)...")
    test_provider_scores = score_providers(test_scored, agg=agg_mode)
    test_metrics = confusion_metrics(test_provider_scores, test_labels, threshold)

    auc_scores = threshold_independent_scores(test_provider_scores, test_labels)
    test_metrics.update(auc_scores)

    print("\n=== PCA-AE (unsupervised) FINAL TEST SET RESULT ===")
    print(f"Precision: {test_metrics['precision']*100:.1f}%  Recall: {test_metrics['recall']*100:.1f}%  F1: {test_metrics['f1']*100:.1f}%  F2: {test_metrics['f2']*100:.1f}%")
    print(f"TP={test_metrics['tp']}  FP={test_metrics['fp']}  FN={test_metrics['fn']}  TN={test_metrics['tn']}")
    print(f"AUC-ROC: {auc_scores['auc_roc']:.3f}  AUC-PR: {auc_scores['auc_pr']:.3f}  (threshold-independent)")

    plot_confusion_matrix(test_metrics, "confusion_matrix_pca_ae.png", title="PCA-AE (unsupervised) -- Test Set")
    print("Saved confusion_matrix_pca_ae.png")

    print("\nFitting Random Forest classifier on TRAIN providers (supervised, uses fraud label directly)...")
    rf_model = fit_random_forest(train_p)

    val_rf_scores = rf_score_providers(rf_model, val_p)
    rf_best = best_threshold_evaluate(val_rf_scores, val_labels, metric="f2", min_precision=MIN_PRECISION)
    rf_threshold = rf_best["threshold"]
    print(f"  Chosen RF threshold: percentile={rf_best['percentile']}, val F2={rf_best['f2']*100:.1f}%")

    test_rf_scores = rf_score_providers(rf_model, test_p)
    rf_test_metrics = confusion_metrics(test_rf_scores, test_labels, rf_threshold)
    rf_auc = threshold_independent_scores(test_rf_scores, test_labels)
    rf_test_metrics.update(rf_auc)

    print("\n=== Random Forest (supervised) FINAL TEST SET RESULT ===")
    print(f"Precision: {rf_test_metrics['precision']*100:.1f}%  Recall: {rf_test_metrics['recall']*100:.1f}%  F1: {rf_test_metrics['f1']*100:.1f}%  F2: {rf_test_metrics['f2']*100:.1f}%")
    print(f"TP={rf_test_metrics['tp']}  FP={rf_test_metrics['fp']}  FN={rf_test_metrics['fn']}  TN={rf_test_metrics['tn']}")
    print(f"AUC-ROC: {rf_auc['auc_roc']:.3f}  AUC-PR: {rf_auc['auc_pr']:.3f}  (threshold-independent)")

    plot_confusion_matrix(rf_test_metrics, "confusion_matrix_rf.png", title="Random Forest (supervised) -- Test Set")
    print("Saved confusion_matrix_rf.png")

    feat_importance = get_feature_importances(rf_model)
    print("\nRandom Forest feature importances:")
    print(feat_importance.to_string())

    test_clusters = assign_clusters(test_p, cluster_scaler, kmeans)

    joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")
    joblib.dump(pca, f"{MODEL_DIR}/pca.joblib")
    joblib.dump(cluster_scaler, f"{MODEL_DIR}/cluster_scaler.joblib")
    joblib.dump(kmeans, f"{MODEL_DIR}/kmeans.joblib")
    joblib.dump(rf_model, f"{MODEL_DIR}/random_forest.joblib")

    config = dict(
        pca_ae=dict(
            agg_mode=agg_mode, threshold=float(threshold), percentile=percentile,
            val_f2=best["f2"], min_precision_floor=MIN_PRECISION,
            test_metrics=test_metrics, feature_cols=FEATURE_COLS,
        ),
        random_forest=dict(
            threshold=float(rf_threshold), percentile=rf_best["percentile"],
            val_f2=rf_best["f2"], min_precision_floor=MIN_PRECISION,
            test_metrics=rf_test_metrics, feature_cols=CLASSIFIER_FEATURE_COLS,
            feature_importances=feat_importance.round(4).to_dict(),
        ),
        n_train=len(train_p), n_val=len(val_p), n_test=len(test_p),
    )
    with open(f"{MODEL_DIR}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(f"{MODEL_DIR}/peer_benchmarks.json", "w") as f:
        json.dump(peer_benchmarks, f, indent=2)

    test_provider_out = test_p.copy()
    test_provider_out["AnomalyScore"] = test_provider_out["Provider"].map(test_provider_scores)
    test_provider_out["Flagged"] = test_provider_out["AnomalyScore"] >= threshold
    test_provider_out["RF_Score"] = test_provider_out["Provider"].map(test_rf_scores)
    test_provider_out["RF_Flagged"] = test_provider_out["RF_Score"] >= rf_threshold
    test_provider_out["Cluster"] = test_provider_out["Provider"].map(test_clusters)
    test_provider_out.to_csv("test_providers_scored.csv", index=False)

    keep_cols = [c for c in [
        "Provider", "ClaimID", "BeneID", "ClaimStartDt", "ClaimEndDt",
        "InscClaimAmtReimbursed", "claim_gap_days", "claim_duration",
        "recon_error", "PotentialFraud",
    ] if c in test_scored.columns]
    test_scored[keep_cols].to_csv("test_claims_scored.csv", index=False)

    print("\nSaved models/ artifacts, test_providers_scored.csv, test_claims_scored.csv")
    return config


if __name__ == "__main__":
    run_pipeline()