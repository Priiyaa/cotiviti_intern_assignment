import json
import joblib
import pandas as pd
import streamlit as st

from agent import investigate

st.set_page_config(page_title="Provider Anomaly Review", layout="wide")

st.title("Provider Payment Anomaly Review")
st.caption(
    "PCA-Autoencoder time-series anomaly detection + Random Forest classification + "
    "KMeans clustering + agentic investigation, on real CMS Medicare claims data"
)


@st.cache_data
def load_artifacts():
    with open("models/config.json") as f:
        config = json.load(f)
    with open("models/peer_benchmarks.json") as f:
        peer_benchmarks = json.load(f)
    test_providers = pd.read_csv("test_providers_scored.csv")
    test_claims = pd.read_csv("test_claims_scored.csv", parse_dates=["ClaimStartDt", "ClaimEndDt"])
    return config, peer_benchmarks, test_providers, test_claims


try:
    config, peer_benchmarks, test_providers, test_claims = load_artifacts()
except FileNotFoundError:
    st.error(
        "No model artifacts found. Run `python3 pipeline.py` first -- it "
        "fits the model on a train split, tunes the threshold on a "
        "validation split, and saves the held-out test-set result here."
    )
    st.stop()

st.divider()
st.subheader("Final result (held-out test set) -- two complementary models")
st.caption(
    f"Trained on {config['n_train']} providers, thresholds tuned on {config['n_val']} providers, "
    f"reported here on {config['n_test']} providers neither model saw during fitting or tuning."
)

tab_pca, tab_rf = st.tabs(["Unsupervised: PCA-AE (time-series anomaly detection)", "Supervised: Random Forest (classification)"])

with tab_pca:
    pca_cfg = config["pca_ae"]
    m = pca_cfg["test_metrics"]
    st.caption(
        f"Aggregation: **{pca_cfg['agg_mode']}** (chosen on validation, optimizing F2). "
        "Never sees the fraud label at prediction time -- flags claims that don't fit the "
        "learned normal pattern. Best suited to catching **novel** fraud patterns."
    )
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Precision", f"{m['precision']*100:.1f}%")
    col2.metric("Recall", f"{m['recall']*100:.1f}%")
    col3.metric("F1", f"{m['f1']*100:.1f}%")
    col4.metric("F2", f"{m['f2']*100:.1f}%")
    col5.metric("Flagged", int(test_providers["Flagged"].sum()))
    col6, col7 = st.columns(2)
    col6.metric("AUC-ROC", f"{m['auc_roc']:.3f}")
    col7.metric("AUC-PR", f"{m['auc_pr']:.3f}")

    cm_df = pd.DataFrame(
        [[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
        index=["Actual: Clean", "Actual: Fraud"],
        columns=["Predicted: Clean", "Predicted: Fraud"],
    )
    st.table(cm_df)

with tab_rf:
    rf_cfg = config["random_forest"]
    m = rf_cfg["test_metrics"]
    st.caption(
        "Trains directly on the fraud label using the EDA-validated provider features. "
        "Best suited to catching fraud that resembles **known** patterns in the training data."
    )
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Precision", f"{m['precision']*100:.1f}%")
    col2.metric("Recall", f"{m['recall']*100:.1f}%")
    col3.metric("F1", f"{m['f1']*100:.1f}%")
    col4.metric("F2", f"{m['f2']*100:.1f}%")
    col5.metric("Flagged", int(test_providers["RF_Flagged"].sum()))
    col6, col7 = st.columns(2)
    col6.metric("AUC-ROC", f"{m['auc_roc']:.3f}")
    col7.metric("AUC-PR", f"{m['auc_pr']:.3f}")

    cm_df = pd.DataFrame(
        [[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
        index=["Actual: Clean", "Actual: Fraud"],
        columns=["Predicted: Clean", "Predicted: Fraud"],
    )
    st.table(cm_df)

    imp_df = pd.DataFrame(
        list(rf_cfg["feature_importances"].items()), columns=["Feature", "Importance"]
    ).sort_values("Importance", ascending=False)
    st.bar_chart(imp_df.set_index("Feature"))

with st.expander("Why two models, and how the numbers were produced"):
    st.markdown(
        f"""
Both use the same honest train/validation/test split -- {config['n_train']} providers to fit,
{config['n_val']} to tune the threshold (optimizing F2, restricted to thresholds keeping
precision above a floor so the result stays operationally usable), {config['n_test']} to report
the final number on, untouched during fitting or tuning.

**Why both, instead of just the better-scoring one:** they answer different questions.
The unsupervised PCA-AE never uses the fraud label to learn what fraud looks like -- it only
learns what *normal* looks like, then flags deviations. That makes it well-suited to catching
fraud schemes that don't resemble anything in the training data. The supervised Random Forest
directly learns "what does labeled fraud look like" -- it will generally score better on these
metrics, precisely because it's allowed to fit the label directly, but it can only recognize
fraud that resembles what it was shown. A real payment-integrity pipeline benefits from both:
the classifier catches known patterns efficiently, the anomaly detector is a safety net for
patterns nobody has seen yet.
        """
    )

st.divider()
st.subheader("Provider ranking (test set)")

rank_df = test_providers[[
    "Provider", "AnomalyScore", "Flagged", "RF_Score", "RF_Flagged", "PotentialFraud", "Cluster",
]].copy()
rank_df = rank_df.sort_values("AnomalyScore", ascending=False)
rank_df.columns = ["Provider", "PCA_AE_Score", "PCA_AE_Flagged", "RF_Score", "RF_Flagged", "ActualLabel", "Cluster"]
st.dataframe(rank_df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Investigate a flagged provider")
st.caption("Uses the PCA-AE flagged list (the anomaly detector this demo centers on).")

flagged_providers = rank_df[rank_df["PCA_AE_Flagged"]]["Provider"].tolist()

if not flagged_providers:
    st.info("No providers flagged at the current threshold.")
else:
    selected_provider = st.selectbox("Choose a provider", flagged_providers)

    prov_claims = test_claims[test_claims["Provider"] == selected_provider].sort_values("ClaimStartDt")

    st.line_chart(prov_claims.set_index("ClaimStartDt")["recon_error"], height=250)

    if st.button("Run agent investigation"):
        cluster_id = str(int(rank_df.set_index("Provider").loc[selected_provider, "Cluster"]))
        anomaly_score = float(rank_df.set_index("Provider").loc[selected_provider, "PCA_AE_Score"])

        stats = {
            "provider_id": selected_provider,
            "cluster": cluster_id,
            "anomaly_score": anomaly_score,
            "aggregation_method": config["pca_ae"]["agg_mode"],
            "num_claims": int(len(prov_claims)),
            "avg_claim_amount": round(float(prov_claims["InscClaimAmtReimbursed"].mean()), 2),
            "max_claim_amount": round(float(prov_claims["InscClaimAmtReimbursed"].max()), 2),
            "avg_gap_days": round(float(prov_claims["claim_gap_days"].mean()), 1),
            "avg_claim_duration": round(float(prov_claims["claim_duration"].mean()), 1),
        }

        with st.spinner("Agent investigating..."):
            result = investigate(stats, peer_benchmarks)

        st.markdown(result)

st.divider()
with st.expander("How this works"):
    st.markdown(
        """
1. **Time-series anomaly detection** (unsupervised): each provider's claims are ordered by
   date and treated as a sequence. A PCA model, fit only on known-clean training claims,
   learns the normal pattern of claim amount, days-between-claims, and claim duration.
   Reconstruction error (how far a claim deviates from that pattern) is the anomaly signal --
   the same method used in a published IEEE paper on smart-grid anomaly detection, applied
   here to healthcare claims.
2. **Mean vs. max aggregation**: a provider's overall anomaly score is either the average
   or the maximum of their claim-level errors. Max preserves a single sharp spike;
   mean can dilute it across many normal claims. The validation split picks whichever wins.
3. **Classification** (supervised): a Random Forest trained directly on the fraud label,
   using provider-level features (billing volume, patient count, case mix) that EDA showed
   have strong real separation between fraud and clean providers.
4. **Clustering**: a real KMeans fit on the same EDA-validated provider features, grouping
   providers into peer clusters the agent can compare against.
5. **Agentic investigation**: clicking "Run agent investigation" sends only the
   precomputed stats (never raw claim data) to Claude, which can call a
   `lookup_peer_benchmark` tool if it decides peer context would help, then reasons through
   explicit steps (chain reasoning) before rendering a verdict.
        """
    )