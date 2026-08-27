"""
Supervised Random Forest classifier, trained directly on the fraud label,
using the provider-level features EDA showed have real separation. This
is the "Classification" focus area, distinct from the unsupervised
PCA-AE anomaly detector ("Time-Series Anomaly Detection" focus area).

Unlike the anomaly detector, this model is SUPPOSED to use the label
during training -- that's the whole point of "supervised." It will
generally outperform the unsupervised approach on standard metrics,
because it's allowed to directly learn "what does fraud look like,"
rather than inferring it from what's statistically abnormal.

The honest trade-off worth stating in the report/demo: a classifier like
this is strong at catching fraud that resembles PAST fraud in the
training data. The unsupervised detector is better suited to catching
NOVEL patterns that never appeared in any training example. Showing both,
on the same held-out test set, tells a more complete story than either
model alone.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# SameAttendOperRate dropped -- EDA showed no real signal (Cohen's d = -0.068)
CLASSIFIER_FEATURE_COLS = [
    "TotalClaims", "TotalReimbursed", "AvgReimbursed",
    "AvgClaimDuration", "AvgDaysInHospital",
    "UniquePatients", "UniqueAttendPhys",
    "AvgChronicConds", "InpatientRatio",
]


def fit_random_forest(train_provider_df, feature_cols=CLASSIFIER_FEATURE_COLS, seed=42):
    X = train_provider_df[feature_cols].fillna(0).values
    y = (train_provider_df["PotentialFraud"] == "Yes").astype(int).values
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",  # fraud is ~12% of providers -- don't let the majority class dominate
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def score_providers(model, provider_df, feature_cols=CLASSIFIER_FEATURE_COLS):
    X = provider_df[feature_cols].fillna(0).values
    proba = model.predict_proba(X)[:, 1]
    return pd.Series(proba, index=provider_df["Provider"].values).sort_values(ascending=False)


def get_feature_importances(model, feature_cols=CLASSIFIER_FEATURE_COLS):
    return pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)