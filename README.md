# Provider Payment Anomaly Review — Cotiviti POC

Time-series anomaly detection (PCA-Autoencoder) + agentic AI investigation,
on real CMS Medicare claims data.

## Setup

1. **Get the data.** Download these 4 files from Kaggle and put them in a
   `data/` folder next to `app.py`:
   - https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis
   - `Train.csv`
   - `Train_Outpatientdata.csv`
   - `Train_Inpatientdata.csv`
   - (`Train_Beneficiarydata.csv` is not required by the current code, but
     grab it too if you want to extend the feature set later)

   Your folder should look like:
   ```
   cotiviti_poc/
     app.py
     data_prep.py
     anomaly_model.py
     agent.py
     requirements.txt
     data/
       Train.csv
       Train_Outpatientdata.csv
       Train_Inpatientdata.csv
       Train_Beneficiarydata.csv
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Set your API key** (the agent step needs this):
   ```
   export ANTHROPIC_API_KEY=your_key_here      # Mac/Linux
   set ANTHROPIC_API_KEY=your_key_here          # Windows cmd
   ```

4. **Run it:**
   ```
   streamlit run app.py
   ```

## What each file does

- `data_prep.py` — loads and joins the raw CSVs, samples ~45 providers
  (20 fraud, 25 clean) with at least 8 claims each, so there's enough
  history per provider to form a usable time series.
- `anomaly_model.py` — the PCA-AE method: fits PCA on claim features,
  scores anomalies by reconstruction error, aggregates per provider by
  mean or max, and evaluates against the real fraud labels with an
  unsupervised (label-blind) threshold.
- `agent.py` — the tool-calling agent. Claude gets a flagged provider's
  stats and a `lookup_peer_benchmark` tool, and decides on its own
  whether to call it before rendering a verdict.
- `app.py` — the Streamlit UI tying it together. Default Streamlit
  widgets, no custom styling — deliberately plain.

## If precision/recall look mediocre

That's expected and fine to say out loud. This is an unsupervised
statistical method run on a small sample, not a trained classifier —
the honest result is a legitimate finding, not a bug. If you want to
push the numbers, the two easiest levers are:
- Increasing `n_fraud` / `n_clean` in `data_prep.py` (bigger sample = less noise)
- Adding features to `FEATURE_COLS` in `anomaly_model.py` (currently just
  claim amount and days-since-last-claim — procedure code diversity or
  diagnosis code count would likely help)
