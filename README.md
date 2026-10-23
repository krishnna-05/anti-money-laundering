# 🛡️ Anti-Money Laundering (AML) Data Warehousing & Mining Platform

An end-to-end Data Warehousing and Data Mining platform for detecting, clustering and predicting money-laundering
activity on the **SAML-D** dataset.

Topic: **Anti-money laundering** · Team: **Code hunters**

---

## 📋 Rubric Requirement Checklist & Implementation Mapping

| Rubric Item | Status | Implementation Details | File Reference |
| :--- | :---: | :--- | :--- |
| **Topic** | ✅ | Anti-money laundering | Full repository |
| **Dataset Description** | ✅ | SAML-D (Kaggle): 9.5M transactions, 28 typologies, 7 payment types, 18 bank locations, 13 currencies | [`data/saml_loader.py`](data/saml_loader.py), dashboard **📖 Dataset** page |
| **DW to be built** | ✅ | Star schema: `Fact_Transactions` + `Dim_Date`, `Dim_Account`, `Dim_Channel`, `Dim_Location`, `Dim_Currency` (DuckDB & SQLite) | [`warehouse/schema.sql`](warehouse/schema.sql), [`warehouse/etl_pipeline.py`](warehouse/etl_pipeline.py) |
| **Sample OLAP operations** | ✅ | Roll-up, Drill-down, Slice, Dice, Pivot | [`warehouse/olap_queries.sql`](warehouse/olap_queries.sql), [`warehouse/olap_engine.py`](warehouse/olap_engine.py) |
| **Classification** | ✅ | 5 classifiers compared (Logistic Regression, Naive Bayes, Decision Tree, Random Forest, HistGradientBoosting) | [`models/1_classification.py`](models/1_classification.py) |
| **Clustering** | ✅ | 7 algorithms (K-Means, Mini-Batch K-Means, Agglomerative, Gaussian Mixture, BIRCH, DBSCAN, HDBSCAN), K selection, comparative study | [`models/2_clustering.py`](models/2_clustering.py) |
| **Association Rules** | ✅ | FP-Growth rules predicting suspicious outcomes / typologies | [`models/3_association.py`](models/3_association.py) |
| **Prediction** | ✅ | Calibrated 0–100 AML risk score, trained on earlier months, forecast on later months | [`models/4_prediction.py`](models/4_prediction.py) |
| **Dashboard** | ✅ | Streamlit app, 10 pages (see below) | [`dashboard/app.py`](dashboard/app.py) |
| **Demo** | ✅ | One-command pipeline `python demo.py` | [`demo.py`](demo.py) |

---

## 📦 Dataset: SAML-D

- **Source**: Kaggle [`berkanoztas/synthetic-transaction-monitoring-dataset-aml`](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml)
- **Paper**: Oztas et al., *Enhancing Anti-Money Laundering: Development of a Synthetic Transaction Monitoring Dataset*, IEEE ICEBE 2023.
- **Size**: 9,504,852 transactions (Oct 2022 – Aug 2023), 9,873 laundering (0.104%).
- **Columns**: `Time, Date, Sender_account, Receiver_account, Amount, Payment_currency, Received_currency,
  Sender_bank_location, Receiver_bank_location, Payment_type, Is_laundering, Laundering_type`.
- **Typologies**: 11 normal (e.g. Normal Fan Out, Normal Periodical) and 17 suspicious (Structuring, Smurfing,
  Layered Fan In/Out, Cycle, Scatter-Gather, Gather-Scatter, Bipartite, Deposit-Send, Over-Invoicing, …).

Public AML datasets are all synthetic: real bank transactions with laundering labels cannot be published.

`data/saml_loader.py` downloads the CSV with `kagglehub` (public dataset, no API key needed) and:
1. Keeps **all 9,873 laundering transactions** plus a random sample of **300,000 normal** ones (≈3.2% suspicious)
   so the warehouse and dashboard stay fast. Change `SAMPLE_NORMAL_TX` in `config.py` to load more.
2. Builds **account behaviour profiles over the full 9.5M rows** (fan-in, fan-out, cash share, cross-border share,
   structuring share, activity) for clustering and OLAP.
3. Computes **point-in-time features** for every transaction: the sender's and receiver's history strictly *before*
   that transaction. Models never see future information.

> **Why point-in-time?** Profiles aggregated over the whole period let a classifier "see the future" (for example,
> how many laundering senders a receiver will eventually have) and score a meaningless ROC-AUC of 1.0. With
> point-in-time features and a split by account, the comparison between models is realistic.

## 💱 Currency

All amounts are in **GBP**, the dataset's native currency. Currency and regulatory wording live in one place,
[`config.py`](config.py). To present everything in INR (₹, Lakh/Crore formatting, ₹10 L PMLA threshold, FIU-IND STR wording):

```bash
AML_CURRENCY=INR python demo.py              # or set ACTIVE_PROFILE = "INR" in config.py
AML_CURRENCY=INR streamlit run dashboard/app.py
```

The loader converts amounts at a fixed illustrative rate (₹100 per £), which also maps the UK £10,000 reference
threshold onto PMLA's ₹10 Lakh. The source data stays UK-based either way.

---

## 🚀 Quickstart

```bash
source .venv/bin/activate
pip install -r requirements.txt

python demo.py                       # download (first run, ~190 MB) → load → ETL → OLAP → train all models
streamlit run dashboard/app.py       # launch the dashboard
```

The first run downloads the dataset to `~/.cache/kagglehub`. The whole pipeline takes about 2 minutes.

---

## 🏛️ Data Warehouse (Star Schema)

```
                     +----------------+
                     |    Dim_Date    |
                     +-------+--------+
                             |
+----------------+   +-------+-------------+   +-----------------+
|  Dim_Account   |<--+  Fact_Transactions  +-->|   Dim_Channel   |
| behaviour      |   | amount, flags,      |   | 7 payment types |
| profile (full  |   | typology, label,    |   +-----------------+
| 9.5M rows)     |   | point-in-time       |
+----------------+   | history measures    +-->+-----------------+
                     +-------+-------------+   |  Dim_Currency   |
                             |                 | currency pairs  |
                     +-------v--------+        +-----------------+
                     |  Dim_Location  |
                     | corridor, FATF |
                     | grey-list risk |
                     +----------------+
```

High-risk locations are those on the FATF grey list during the dataset period: UAE, Turkey, Nigeria, Morocco, Albania.

---

## 🧠 Data Mining Modules

### 1. Classification: suspicious vs normal
Five classifiers trained on the same split, **grouped by sender account** so test accounts are unseen.
Representative hold-out results:

| Model | ROC-AUC | PR-AUC |
| :--- | :---: | :---: |
| HistGradientBoosting | 0.999 | 0.982 |
| Random Forest | 0.999 | 0.979 |
| Decision Tree | 0.975 | 0.950 |
| Logistic Regression | 0.925 | 0.419 |
| Naive Bayes | 0.857 | 0.153 |

The strongest signals are network behaviour: the receiver's fan-in, whether the receiver is a pass-through account,
and the sender's fan-out.

### 2. Clustering: account behaviour groups
- **7 algorithms**, each with its own tunable parameters.
- **Choosing K**: elbow (inertia), silhouette, Davies-Bouldin and Calinski-Harabasz across a K range, an automatic
  elbow suggestion, a silhouette plot, and BIC/AIC for Gaussian Mixtures. A k-distance plot helps choose DBSCAN's eps.
- **Comparative study**: metrics table, per-metric bar charts, side-by-side PCA projections, Adjusted Rand Index
  agreement matrix, and K-sensitivity curves for every K-based algorithm.
- **AML usefulness**: the laundering label is used only *after* clustering, to measure how strongly each clustering
  concentrates suspicious accounts (best-cluster lift, share of suspicious accounts in hotspot clusters).

### 3. Association rules
FP-Growth over transaction itemsets (channel, amount band relative to the reporting threshold, route, timing,
sender/receiver history). Only rules with observable antecedents and a suspicious or typology consequent are kept,
for example `Channel=Cash Withdrawal AND Corridor=High-risk location ⇒ SUSPICIOUS` (confidence 1.0, lift ≈ 31).

### 4. Prediction: AML risk score (0–100)
An isotonic-calibrated gradient-boosting model is trained on Oct 2022 – May 2023 and forecasts Jun – Aug 2023.
On this synthetic data, separation is near-perfect once counterparty history is known (PR-AUC ≈ 0.999).
Scores ≥ 70 are high risk (file a SAR) and 30–70 medium risk (Enhanced Due Diligence).

---

## 📊 Dashboard Pages

1. **📖 Dataset**: source, columns, sampling and typology distributions.
2. **📊 Surveillance Overview**: KPIs, monthly volume and laundering trends, channel risk, transaction explorer.
3. **🏛️ Data Warehouse & OLAP**: interactive Roll-up (day/month/quarter/year), Drill-down, Slice, Dice, Pivot heatmap.
4. **🎯 Classification**: model comparison, ROC and PR curves, confusion matrix with adjustable threshold,
   recall per typology, permutation feature importance.
5. **🧩 Clustering Lab**:
   - *K-Means & choosing K*: K-range sweep with four diagnostics, K slider, silhouette plot.
   - *Algorithm explorer*: any of the 7 algorithms, each with its own parameter controls.
   - *Comparative study*: run several algorithms side by side and compare them.
6. **🔗 Association Rules**: filter by confidence, lift and consequent, plus a support-vs-confidence scatter.
7. **📈 Risk Prediction**: ROC and PR curves with the operating point, score distribution, calibration curve,
   threshold trade-off, predicted vs actual alerts per month, detection by typology, risk by channel,
   top-risk accounts.
8. **⚡ Live Risk Scoring**: load a real transaction of any typology, or edit the inputs, and score it live.
9. **🕸️ Money Flow Network**: real laundering networks from the data, with nodes coloured by source, intermediary
   or sink role.
10. **📋 SAR Case Generator**: draft and download a report for a flagged transaction.
