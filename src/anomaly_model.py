"""
PCA-based Autoencoder anomaly detection, applied to per-provider claim
time series. This is the same core method from the NAPS 2026 paper
(PCA-AE for smart-grid FDIA detection), transplanted to healthcare claims:

  1. Treat each provider's claims, ordered by date, as a time series.
  2. Fit PCA on the (mostly normal) claim feature vectors -> this is the
     "autoencoder": PCA compresses to a lower dimension, then reconstructs.
  3. Reconstruction error = how well a claim fits the learned normal
     pattern. Large error = anomalous claim.
  4. Aggregate per-provider: MEAN dilutes a single spike across all of a
     provider's claims; MAX preserves it. This mean-vs-max distinction is
     the key finding from the original paper -> max catches short, sharp
     anomalies that mean-aggregation washes out.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score

FEATURE_COLS = ["InscClaimAmtReimbursed", "claim_gap_days", "claim_duration"]


def engineer_features(df):
    """
    Per-claim time-series features:
      - claim amount
      - claim_gap_days: days since this provider's previous claim
      - claim_duration: days between claim start and end

    claim_duration replaces the earlier procedure_diversity feature --
    EDA (Cohen's d on the real dataset) showed AvgClaimDuration has a
    large effect size (d=0.809) separating fraud from clean providers,
    and unlike most of the literature-standard features, it's genuinely
    computable per-claim, so it fits the time-series framing instead of
    being a provider-level total.
    """
    df = df.sort_values(["Provider", "ClaimStartDt"]).copy()
    df["claim_gap_days"] = (
        df.groupby("Provider")["ClaimStartDt"].diff().dt.days.fillna(0)
    )
    df["claim_duration"] = (df["ClaimEndDt"] - df["ClaimStartDt"]).dt.days.fillna(0)
    return df


def fit_pca_ae(df, feature_cols=FEATURE_COLS, n_components=1):
    X = df[feature_cols].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    pca.fit(Xs)
    return scaler, pca


def reconstruction_error(df, scaler, pca, feature_cols=FEATURE_COLS):
    X = df[feature_cols].values
    Xs = scaler.transform(X)
    Xp = pca.inverse_transform(pca.transform(Xs))
    errors = np.sum((Xs - Xp) ** 2, axis=1)
    df = df.copy()
    df["recon_error"] = errors
    return df


def score_providers(df, agg="max"):
    if agg == "max":
        scores = df.groupby("Provider")["recon_error"].max()
    else:
        scores = df.groupby("Provider")["recon_error"].mean()
    return scores.sort_values(ascending=False)


def evaluate(scores, fraud_labels, threshold_percentile=70):
    """
    Unsupervised threshold: flag anything above the Nth percentile of
    scores in this batch. Chosen by statistics, not by peeking at labels.
    """
    threshold = np.percentile(scores.values, threshold_percentile)
    predicted_fraud = scores >= threshold
    y_true = fraud_labels.loc[scores.index] == "Yes"

    tp = int((predicted_fraud & y_true).sum())
    fp = int((predicted_fraud & ~y_true).sum())
    fn = int((~predicted_fraud & y_true).sum())
    tn = int((~predicted_fraud & ~y_true).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    f2 = fbeta(precision, recall, beta=2)

    return dict(
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=precision, recall=recall, f1=f1, f2=f2,
        threshold=float(threshold),
    )


def fbeta(precision, recall, beta=2):
    """
    F-beta score. beta=1 weights precision and recall equally (F1).
    beta=2 (F2) weights recall twice as heavily as precision -- the right
    choice for fraud/anomaly screening, where missing an actual fraud
    provider (false negative) is more costly than flagging a clean one
    for a reviewer to clear (false positive).
    """
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * precision * recall / (b2 * precision + recall) if (b2 * precision + recall) else 0.0


def threshold_independent_scores(scores, fraud_labels):
    """
    AUC-ROC and AUC-PR (average precision) don't depend on picking any
    single cutoff -- they measure how well the anomaly score itself
    ranks fraud providers above clean ones, across all possible
    thresholds at once. This is the fairest single number for an
    unsupervised detector, since it sidesteps the "which threshold did
    you pick" question entirely.
    """
    y_true = (fraud_labels.loc[scores.index] == "Yes").astype(int)
    auc_roc = roc_auc_score(y_true, scores.values)
    auc_pr = average_precision_score(y_true, scores.values)
    return dict(auc_roc=auc_roc, auc_pr=auc_pr)


def best_threshold_evaluate(scores, fraud_labels, percentiles=range(30, 96, 5), metric="f2", min_precision=None):
    """
    Sweeps a range of percentile cutoffs and returns the metrics for
    whichever one maximizes the chosen metric (default F2, recall-weighted)
    on this batch.

    min_precision, if set, restricts the sweep to thresholds that keep
    precision at or above that floor. Without this, pure F2 optimization
    can degenerate on imbalanced data: recall keeps climbing as the
    threshold drops, and F2's heavy recall weighting can make that look
    like an improvement even as precision collapses and the model ends up
    flagging half the population. A precision floor keeps the result
    operationally usable -- a review team can't act on a list where most
    entries are false positives.

    This IS using the labels to pick the threshold -- that's disclosed,
    not hidden. It answers "what's the best this unsupervised score can
    do if a human picks the operating point afterward," which is how
    anomaly detection is normally evaluated and reported: the model
    never trains on labels, but the deployment threshold is a business
    decision made with label/outcome knowledge, exactly like it would be
    in production.
    """
    best = None
    for p in percentiles:
        result = evaluate(scores, fraud_labels, threshold_percentile=p)
        if min_precision is not None and result["precision"] < min_precision:
            continue
        result["percentile"] = p
        if best is None or result[metric] > best[metric]:
            best = result

    if best is None:
        # Nothing met the precision floor -- fall back to the single
        # threshold with the highest precision available, so the
        # function still returns something usable rather than None.
        results = [evaluate(scores, fraud_labels, threshold_percentile=p) for p in percentiles]
        best = max(results, key=lambda r: r["precision"])
        best["percentile"] = list(percentiles)[results.index(best)]

    return best