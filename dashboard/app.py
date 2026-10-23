"""
AML Intelligence Platform - SAML-D Edition
Data warehouse, OLAP, classification, clustering lab, association rules and risk prediction
on the SAML-D anti-money-laundering dataset. Currency and regulatory wording come from config.py.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import duckdb
import joblib
import networkx as nx
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix, silhouette_samples, adjusted_rand_score,
    precision_score, recall_score, f1_score,
)

from config import (
    CUR, STRUCTURING_LOW, STRUCTURING_HIGH, HIGH_RISK_LOCATIONS, SAMPLE_NORMAL_TX, KAGGLE_DATASET,
    DUCKDB_PATH, format_money, money_unit,
)
from warehouse.olap_engine import AML_OLAP_Engine
from models import clustering_module as clu
from models.prediction_module import explain_transaction_risk, HIGH_RISK_SCORE, MEDIUM_RISK_SCORE
from models.features import CHANNELS

def ensure_duckdb():
    if not os.path.exists(DUCKDB_PATH):
        parts_dir = os.path.join("data", "processed", "warehouse_parts")
        if os.path.exists(parts_dir):
            import glob, gzip, shutil
            parts = sorted(glob.glob(os.path.join(parts_dir, "part_*")))
            if parts:
                os.makedirs(os.path.dirname(DUCKDB_PATH), exist_ok=True)
                gz_temp = DUCKDB_PATH + ".temp.gz"
                with open(gz_temp, "wb") as out_f:
                    for p in parts:
                        with open(p, "rb") as in_f:
                            shutil.copyfileobj(in_f, out_f)
                with gzip.open(gz_temp, "rb") as gz_in, open(DUCKDB_PATH, "wb") as db_out:
                    shutil.copyfileobj(gz_in, db_out)
                if os.path.exists(gz_temp):
                    os.remove(gz_temp)

ensure_duckdb()

st.set_page_config(page_title="AML Intelligence Platform", page_icon=":material/shield:", layout="wide",
                   initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- palette
# Categorical slots in fixed order (validated reference palette); status colours reserved for risk tiers.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
EXTRA = px.colors.qualitative.Alphabet  # only when a clustering produces more than 8 clusters
NEUTRAL = "#898781"
SOFT_BLUE = "#9ec5f4"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b"}
SEQ_BLUE = ["#eef5fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0, "#2a78d6"], [0.5, "#f0efec"], [1, "#e34948"]]
GRID = "rgba(137,135,129,0.22)"
DIV, UNIT = money_unit()
SYM = CUR["symbol"]
RPT = CUR["report_name"]

st.markdown("""
<style>
    .block-container {padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1360px;}
    h1 {font-size: 1.85rem !important; font-weight: 700 !important; letter-spacing: -0.015em;}
    h2 {font-size: 1.3rem !important; font-weight: 650 !important; margin-top: 0.6rem !important;}
    h3 {font-size: 1.08rem !important; font-weight: 650 !important;}
    .eyebrow {text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.09em; font-weight: 650;
              color: #2a78d6; margin-bottom: -0.55rem;}
    .page-sub {opacity: 0.72; font-size: 0.95rem; margin-top: -0.4rem; margin-bottom: 1.1rem; max-width: 900px;}
    .chart-title {font-weight: 650; font-size: 0.98rem; margin: 0.35rem 0 0 0;}
    .chart-caption {font-size: 0.8rem; opacity: 0.68; margin: 0.05rem 0 0 0;}
    [data-testid="stMetricValue"] {font-size: 1.55rem;}
    [data-testid="stMetricLabel"] p {font-size: 0.82rem; opacity: 0.8;}
    .brand {font-weight: 700; font-size: 1.05rem; letter-spacing: -0.01em;}
    .brand-sub {font-size: 0.78rem; opacity: 0.65; margin-top: -0.2rem;}
    .facts {font-size: 0.8rem; line-height: 1.55; opacity: 0.85;}
    .facts b {font-weight: 600;}
    .score-card {border: 1px solid rgba(137,135,129,0.35); border-radius: 12px; padding: 1.2rem 1.4rem;}
    .score-num {font-size: 3.4rem; font-weight: 700; line-height: 1; letter-spacing: -0.03em;}
    .score-of {font-size: 1rem; opacity: 0.6; font-weight: 500;}
    .tier-pill {display: inline-block; padding: 0.18rem 0.65rem; border-radius: 999px; font-size: 0.78rem;
                font-weight: 650; color: #fff; margin-top: 0.6rem;}
    .score-track {position: relative; height: 10px; border-radius: 999px; margin-top: 1rem;
                  background: linear-gradient(90deg, rgba(12,163,12,0.35) 0%, rgba(12,163,12,0.35) VAR_MED%,
                  rgba(250,178,25,0.45) VAR_MED%, rgba(250,178,25,0.45) VAR_HIGH%, rgba(208,59,59,0.45) VAR_HIGH%);}
    .score-marker {position: absolute; top: -5px; width: 4px; height: 20px; border-radius: 2px;
                   background: currentColor; transform: translateX(-2px);}
    .score-scale {display: flex; justify-content: space-between; font-size: 0.72rem; opacity: 0.6; margin-top: 0.3rem;}
    .flag-list {margin: 0.3rem 0 0 0; padding-left: 1.1rem;}
    .flag-list li {margin-bottom: 0.3rem;}
</style>
""".replace("VAR_MED", str(MEDIUM_RISK_SCORE)).replace("VAR_HIGH", str(HIGH_RISK_SCORE)), unsafe_allow_html=True)


# --------------------------------------------------------------------------- UI helpers
def page_header(eyebrow, title, subtitle=None):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.markdown(f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)


def chart(fig, title=None, caption=None, height=360, legend=True):
    if title:
        st.markdown(f'<div class="chart-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="chart-caption">{caption}</div>', unsafe_allow_html=True)
    fig.update_layout(
        title=None, height=height, margin=dict(l=8, r=8, t=30 if legend else 12, b=8),
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12.5),
        showlegend=legend, hoverlabel=dict(font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text=""),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, title_font_size=12)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title_font_size=12)
    st.plotly_chart(fig, width="stretch")


def kpis(items, border=True, widths=None):
    cols = st.columns(widths or len(items))
    for col, item in zip(cols, items):
        label, value = item[0], item[1]
        help_text = item[2] if len(item) > 2 else None
        col.metric(label, value, help=help_text, border=border)


def pretty_cols(df, rename=None):
    rename = rename or {}
    return df.rename(columns={c: rename.get(c, c.replace("_", " ").capitalize()) for c in df.columns})


def money_config(*cols):
    return {c: st.column_config.NumberColumn(c, format="compact") for c in cols}


def cluster_color_map(labels):
    uniq = sorted(set(labels))
    palette = SERIES if len([u for u in uniq if u != -1]) <= len(SERIES) else EXTRA
    cmap, i = {}, 0
    for u in uniq:
        if u == -1:
            cmap["Noise"] = NEUTRAL
        else:
            cmap[f"C{u}"] = palette[i % len(palette)]
            i += 1
    return cmap


def cluster_names(labels):
    return np.array(["Noise" if l == -1 else f"C{l}" for l in labels])


def cluster_order(cmap):
    return sorted(cmap.keys(), key=lambda s: (s == "Noise", int(s[1:]) if s != "Noise" else 0))


def month_label(ym):
    return pd.to_datetime(ym + "-01").strftime("%b %y")


# --------------------------------------------------------------------------- cached data
@st.cache_resource
def get_olap_engine():
    return AML_OLAP_Engine()


@st.cache_data
def load_all_transactions():
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    df = con.execute("""
        SELECT f.transaction_id, f.timestamp, d.year_month, sa.account_id AS sender_account,
               ra.account_id AS receiver_account, sa.account_segment, c.channel_name, l.corridor,
               l.sender_location, l.receiver_location, l.is_cross_border, l.is_high_risk_corridor,
               cu.payment_currency, cu.received_currency, f.amount, f.is_structuring,
               f.is_suspicious, f.laundering_typology, f.sar_flag
        FROM Fact_Transactions f
        JOIN Dim_Date d ON f.date_key = d.date_key
        JOIN Dim_Account sa ON f.sender_account_key = sa.account_key
        JOIN Dim_Account ra ON f.receiver_account_key = ra.account_key
        JOIN Dim_Channel c ON f.channel_key = c.channel_key
        JOIN Dim_Location l ON f.location_key = l.location_key
        JOIN Dim_Currency cu ON f.currency_key = cu.currency_key
        ORDER BY f.timestamp DESC
    """).df()
    con.close()
    return df


@st.cache_resource
def load_artifact(name):
    path = f"models/saved/{name}.joblib"
    return joblib.load(path) if os.path.exists(path) else None


@st.cache_data
def cl_profiles(n, seed):
    return clu.load_account_profiles(n, seed)


@st.cache_data
def cl_matrix(n, seed, features):
    X, _ = clu.prepare_matrix(cl_profiles(n, seed), list(features))
    return X


@st.cache_data
def cl_pca(n, seed, features):
    return clu.pca_project(cl_matrix(n, seed, features), 3)


@st.cache_data(show_spinner=False)
def cl_run(name, params_items, n, seed, features):
    X = cl_matrix(n, seed, features)
    res = clu.run_clustering(name, X, dict(params_items))
    y = cl_profiles(n, seed)["ever_suspicious"].values
    metrics = clu.evaluate_clustering(X, res["labels"], y)
    return {"labels": res["labels"], "seconds": res["seconds"], "metrics": metrics,
            "inertia": res.get("inertia"), "bic": res.get("bic"), "aic": res.get("aic")}


@st.cache_data(show_spinner=False)
def cl_sweep(name, k_min, k_max, params_items, n, seed, features):
    X = cl_matrix(n, seed, features)
    y = cl_profiles(n, seed)["ever_suspicious"].values
    return clu.k_sweep(name, X, range(k_min, k_max + 1), dict(params_items), y)


@st.cache_data(show_spinner=False)
def cl_kdist(n, seed, features, min_samples):
    return clu.k_distance(cl_matrix(n, seed, features), min_samples)


@st.cache_data(show_spinner=False)
def cl_silhouette_samples(name, params_items, n, seed, features):
    X = cl_matrix(n, seed, features)
    labels = cl_run(name, params_items, n, seed, features)["labels"]
    mask = labels != -1
    if len(set(labels[mask])) < 2:
        return None
    return labels[mask], silhouette_samples(X[mask], labels[mask])


olap_engine = get_olap_engine()
df_all_tx = load_all_transactions()
cls_art = load_artifact("classifier")
asc_art = load_artifact("association_rules")
pred_art = load_artifact("risk_predictor")


def missing_artifact():
    st.warning("Model artifacts not found. Run `python demo.py` (or `python models/train_all.py`) first.")


# =========================================================================== DATASET
def page_dataset():
    page_header("Overview", "Dataset",
                "SAML-D, the Synthetic Anti-Money Laundering Dataset (Oztas et al., IEEE ICEBE 2023), "
                "built with AML specialists to mirror real transaction-monitoring data.")
    kpis([("Source transactions", "9.50M"), ("Laundering in source", "9,873", "0.104% of all transactions"),
          ("Typologies", "28", "11 normal, 17 suspicious"), ("Period", "Oct 22 – Aug 23")])

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("How the data is prepared")
        conv = (f" Amounts are converted from GBP at {SYM}{CUR['rate_from_gbp']:g} per £."
                if CUR["rate_from_gbp"] != 1 else "")
        st.markdown(f"""
- **Source**: Kaggle `{KAGGLE_DATASET}`, downloaded by `data/saml_loader.py` (no API key needed).
- **Sample**: all 9,873 laundering transactions plus {SAMPLE_NORMAL_TX:,} randomly sampled normal ones
  ({len(df_all_tx):,} rows, {df_all_tx['is_suspicious'].mean() * 100:.1f}% suspicious instead of 0.1%).
- **Account profiles** (fan-in, fan-out, cash share, cross-border share) are computed over the **full 9.5M rows**.
- **Point-in-time features**: each transaction carries the sender's and receiver's history *before* that moment,
  so no model can see the future.
- **Currency**: {CUR['code']}.{conv}
- **High-risk locations**: FATF grey list during the period: {', '.join(sorted(HIGH_RISK_LOCATIONS))}.

No public dataset of real bank transactions with laundering labels exists; banks cannot release customer data.
""")
    with right:
        st.subheader("Original columns")
        st.dataframe(pd.DataFrame([
            ("Time, Date", "Transaction timestamp"),
            ("Sender / Receiver account", "Anonymised account IDs"),
            ("Amount", "Amount in GBP"),
            ("Payment / Received currency", "13 currencies"),
            ("Sender / Receiver bank location", "18 countries"),
            ("Payment type", "7 payment channels"),
            ("Is laundering", "Label (1 = laundering)"),
            ("Laundering type", "Typology, 28 classes"),
        ], columns=["Column", "Description"]), hide_index=True, width="stretch")

    c1, c2 = st.columns(2, gap="large")
    for col, flag, title, color in [(c1, 1, "Laundering typologies", SERIES[7]), (c2, 0, "Normal typologies", SERIES[0])]:
        with col:
            typ = (df_all_tx[df_all_tx["is_suspicious"] == flag]["laundering_typology"].value_counts()
                   .rename_axis("Typology").reset_index(name="Transactions"))
            fig = px.bar(typ, x="Transactions", y="Typology", orientation="h", color_discrete_sequence=[color])
            fig.update_yaxes(autorange="reversed", title=None)
            chart(fig, title, "Transactions in the loaded sample", height=500, legend=False)


# =========================================================================== OVERVIEW
def page_overview():
    page_header("Overview", "Surveillance overview",
                f"{len(df_all_tx):,} transactions in the warehouse, Oct 2022 to Aug 2023.")
    total_vol = df_all_tx["amount"].sum()
    sus = df_all_tx[df_all_tx["is_suspicious"] == 1]
    kpis([("Transactions monitored", f"{len(df_all_tx):,}"),
          ("Total volume", format_money(total_vol)),
          ("Laundering transactions", f"{len(sus):,}", f"{len(sus) / len(df_all_tx) * 100:.1f}% of the loaded sample"),
          ("Laundering volume", format_money(sus["amount"].sum()),
           f"{sus['amount'].sum() / total_vol * 100:.1f}% of total volume")])

    roll = olap_engine.rollup("month")
    roll["Month"] = pd.to_datetime(roll["year_month"] + "-01")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig = go.Figure()
        fig.add_bar(x=roll["Month"], y=(roll["total_volume"] - roll["illicit_volume"]) / DIV, name="Normal",
                    marker_color=SERIES[0])
        fig.add_bar(x=roll["Month"], y=roll["illicit_volume"] / DIV, name="Laundering", marker_color=SERIES[7])
        fig.update_layout(barmode="stack", yaxis_title=UNIT, bargap=0.25)
        fig.update_xaxes(tickformat="%b %y", dtick="M2")
        chart(fig, f"Monthly volume ({UNIT})")
    with c2:
        fig = px.line(roll, x="Month", y="suspicious_tx_count", markers=True, color_discrete_sequence=[SERIES[7]],
                      labels={"Month": "", "suspicious_tx_count": "Transactions"})
        fig.update_traces(line_width=2, marker_size=8)
        fig.update_yaxes(rangemode="tozero")
        fig.update_xaxes(tickformat="%b %y", dtick="M2")
        chart(fig, "Monthly laundering transactions", legend=False)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        ch = df_all_tx.groupby("channel_name")["is_suspicious"].agg(["mean", "sum"]).reset_index()
        ch["rate"] = ch["mean"] * 100
        fig = px.bar(ch.sort_values("rate"), x="rate", y="channel_name", orientation="h",
                     color_discrete_sequence=[SERIES[1]], hover_data={"sum": True},
                     labels={"rate": "Suspicious rate (%)", "channel_name": "", "sum": "Laundering txns"})
        chart(fig, "Suspicious rate by payment channel", legend=False)
    with c4:
        typo = sus["laundering_typology"].value_counts().head(10).rename_axis("Typology").reset_index(name="Count")
        fig = px.bar(typo, x="Count", y="Typology", orientation="h", color_discrete_sequence=[SERIES[7]])
        fig.update_yaxes(autorange="reversed", title=None)
        chart(fig, "Top 10 laundering typologies", legend=False)

    st.subheader("Transaction explorer")
    f1, f2, f3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    channels = sorted(df_all_tx["channel_name"].unique())
    with f1:
        channel_filter = st.multiselect("Payment channels", channels, placeholder="All channels")
    with f2:
        typology_filter = st.multiselect("Typologies", sorted(sus["laundering_typology"].unique()),
                                         placeholder="All typologies")
    with f3:
        only_sus = st.toggle("Laundering only", value=True)
    view = df_all_tx
    if channel_filter:
        view = view[view["channel_name"].isin(channel_filter)]
    if typology_filter:
        view = view[view["laundering_typology"].isin(typology_filter)]
    if only_sus:
        view = view[view["is_suspicious"] == 1]
    st.caption(f"{len(view):,} matching transactions, showing the latest 500")
    disp = view.head(500)[["transaction_id", "timestamp", "sender_account", "receiver_account", "channel_name",
                           "corridor", "amount", "laundering_typology"]]
    st.dataframe(pretty_cols(disp, {"transaction_id": "Transaction", "channel_name": "Channel",
                                    "amount": f"Amount ({SYM})", "laundering_typology": "Typology"}),
                 hide_index=True, width="stretch", height=380,
                 column_config={f"Amount ({SYM})": st.column_config.NumberColumn(format="%.2f")})


# =========================================================================== OLAP
def page_olap():
    page_header("Data warehouse", "OLAP explorer",
                "Star schema in DuckDB: Fact_Transactions with Date, Account, Channel, Location and Currency "
                "dimensions. Each tab runs one OLAP operation against the cube.")
    t1, t2, t3, t4, t5 = st.tabs(["Roll-up", "Drill-down", "Slice", "Dice", "Pivot"])

    with t1:
        st.caption("Aggregate up the time hierarchy: day → month → quarter → year.")
        level = st.segmented_control("Level", ["day", "month", "quarter", "year"], default="month",
                                     format_func=str.capitalize) or "month"
        df_roll = olap_engine.rollup(level)
        x_col = {"day": "full_date", "month": "year_month", "quarter": "quarter_name", "year": "year_number"}[level]
        fig = px.bar(df_roll, x=x_col, y="suspicious_tx_count", color_discrete_sequence=[SERIES[7]],
                     labels={x_col: "", "suspicious_tx_count": "Laundering transactions"})
        if level in ("year", "quarter"):
            fig.update_xaxes(type="category")
        chart(fig, f"Laundering transactions by {level}", height=320, legend=False)
        st.dataframe(pretty_cols(df_roll), hide_index=True, width="stretch", height=320,
                     column_config=money_config("Total volume", "Illicit volume", "Avg ticket"))

    with t2:
        st.caption("Drill from corridor to receiver location to payment channel.")
        hr = st.toggle("High-risk corridors only (otherwise all cross-border)", value=True)
        df_drill = olap_engine.drilldown(high_risk_only=hr)
        fig = px.bar(df_drill.head(12), x="suspicious_count", y="corridor", color="channel_name", orientation="h",
                     color_discrete_sequence=SERIES, labels={"suspicious_count": "Laundering transactions", "corridor": ""})
        fig.update_yaxes(autorange="reversed")
        chart(fig, "Top corridors by laundering transactions, split by channel", height=400)
        st.dataframe(pretty_cols(df_drill), hide_index=True, width="stretch", column_config=money_config("Total volume"))

    with t3:
        st.caption("Fix one dimension member (a payment channel) and inspect the remaining sub-cube.")
        channels = olap_engine.channels()
        channel_pick = st.selectbox("Payment channel", channels, index=channels.index("Cross-border"))
        df_sl = olap_engine.slice(channel=channel_pick)
        fig = px.bar(df_sl, x="quarter_name", y="flagged_suspicious", color="account_segment", barmode="group",
                     color_discrete_sequence=SERIES, labels={"flagged_suspicious": "Laundering transactions",
                                                             "quarter_name": ""})
        chart(fig, f"{channel_pick}: laundering transactions by quarter and account segment")
        st.dataframe(pretty_cols(df_sl), hide_index=True, width="stretch", column_config=money_config("Total volume"))

    with t4:
        st.caption("Restrict several dimensions at once: amount, quarters and corridor risk.")
        d1, d2, d3 = st.columns([1, 2, 1], vertical_alignment="bottom")
        with d1:
            min_amt = st.number_input(f"Minimum amount ({SYM})", min_value=0.0, value=float(STRUCTURING_LOW),
                                      step=float(STRUCTURING_HIGH / 10))
        with d2:
            qs = st.multiselect("Quarters", olap_engine.quarters(), default=olap_engine.quarters())
        with d3:
            hr_only = st.toggle("High-risk corridors only", value=False)
        df_dc = olap_engine.dice(min_amount=min_amt, quarters=tuple(qs) if qs else None, high_risk_only=hr_only)
        st.dataframe(pretty_cols(df_dc), hide_index=True, width="stretch", column_config=money_config("Diced volume"))

    with t5:
        st.caption("Rotate the cube: payment channel (rows) against account segment (columns).")
        measure = st.segmented_control("Measure", ["suspicious", "count", "volume"], default="suspicious",
                                       format_func={"suspicious": "Laundering transactions", "count": "All transactions",
                                                    "volume": "Volume"}.get) or "suspicious"
        df_pv = olap_engine.pivot(measure)
        mat = df_pv.set_index("channel_name")
        fig = px.imshow(mat, text_auto=".3s" if measure == "volume" else ",.0f", aspect="auto",
                        color_continuous_scale=SEQ_BLUE, labels=dict(x="", y="", color=""))
        chart(fig, "Channel × account segment", height=420, legend=False)


# =========================================================================== CLASSIFICATION
def page_classification():
    page_header("Data mining", "Classification",
                "Five classifiers predict whether a transaction is laundering. All share one split in which "
                "test transactions come from sender accounts never seen in training.")
    if cls_art is None:
        return missing_artifact()
    metrics, tf, best = cls_art["metrics"], cls_art["test_frame"], cls_art["model_name"]
    bm = metrics[best]
    kpis([("Best model (by PR-AUC)", best), ("PR-AUC", f"{bm['pr_auc']:.3f}"), ("ROC-AUC", f"{bm['roc_auc']:.3f}"),
          ("Train / test transactions", f"{cls_art['train_size'] // 1000}k / {cls_art['test_size'] // 1000}k")],
         widths=[1.6, 1, 1, 1.2])

    mt = pd.DataFrame([{"Model": n, "ROC-AUC": m["roc_auc"], "PR-AUC": m["pr_auc"], "Precision": m["precision"],
                        "Recall": m["recall"], "F1": m["f1"], "Train time (s)": m["train_seconds"]}
                       for n, m in metrics.items()]).sort_values("PR-AUC", ascending=False)
    st.subheader("Model comparison")
    st.caption("Precision, recall and F1 at a 0.5 threshold. PR-AUC is the headline metric because laundering is rare.")
    st.dataframe(mt.style.format({c: "{:.3f}" for c in mt.columns if c != "Model"} | {"Train time (s)": "{:.1f}"})
                 .highlight_max(subset=["ROC-AUC", "PR-AUC", "F1"], color="rgba(42,120,214,0.18)"),
                 hide_index=True, width="stretch")

    names = list(metrics.keys())
    cmap = {n: SERIES[i] for i, n in enumerate(names)}
    y_true = tf["y_true"].values
    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig = go.Figure()
        for n in names:
            fpr, tpr, _ = roc_curve(y_true, tf[f"proba_{n}"])
            fig.add_scatter(x=fpr, y=tpr, mode="lines", name=n, line=dict(color=cmap[n], width=2))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color=NEUTRAL, dash="dot", width=1))
        fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
        chart(fig, "ROC curves", height=400)
    with c2:
        fig = go.Figure()
        for n in names:
            p, r, _ = precision_recall_curve(y_true, tf[f"proba_{n}"])
            fig.add_scatter(x=r, y=p, mode="lines", name=n, line=dict(color=cmap[n], width=2))
        fig.add_hline(y=y_true.mean(), line_dash="dot", line_color=NEUTRAL, annotation_text="Base rate",
                      annotation_position="bottom left")
        fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
        chart(fig, "Precision-recall curves", height=400)

    st.subheader("Inspect a model")
    s1, s2 = st.columns(2)
    with s1:
        pick = st.selectbox("Model", names, index=names.index(best))
    with s2:
        thr = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)
    pred = (tf[f"proba_{pick}"].values >= thr).astype(int)
    kpis([("Precision", f"{precision_score(y_true, pred, zero_division=0):.3f}"),
          ("Recall", f"{recall_score(y_true, pred):.3f}"), ("F1", f"{f1_score(y_true, pred):.3f}"),
          ("Alerts raised", f"{pred.sum():,}")])
    c3, c4 = st.columns([2, 3], gap="large")
    with c3:
        cm = confusion_matrix(y_true, pred)
        fig = px.imshow(pd.DataFrame(cm, index=["Actual normal", "Actual laundering"],
                                     columns=["Predicted normal", "Predicted laundering"]),
                        text_auto=",", color_continuous_scale=SEQ_BLUE, aspect="auto")
        fig.update_layout(coloraxis_showscale=False)
        chart(fig, "Confusion matrix", height=380, legend=False)
    with c4:
        rec = (tf.assign(pred=pred)[tf["y_true"] == 1].groupby("laundering_typology")["pred"].mean().mul(100)
               .sort_values().reset_index())
        fig = px.bar(rec, x="pred", y="laundering_typology", orientation="h", color_discrete_sequence=[SERIES[2]],
                     labels={"pred": "Detected (%)", "laundering_typology": ""})
        fig.update_xaxes(range=[0, 100])
        chart(fig, "Recall by laundering typology", height=380, legend=False)

    fi = cls_art["feature_importances"].head(12)
    fig = px.bar(fi, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[SERIES[0]])
    fig.update_yaxes(autorange="reversed", title=None)
    chart(fig, f"Feature importance: {best}", "Permutation importance: drop in PR-AUC when the feature is shuffled",
          height=420, legend=False)


# =========================================================================== CLUSTERING LAB
def page_clustering():
    page_header("Data mining", "Clustering lab",
                "Group sending accounts by behaviour: fan-in, fan-out, cash share, cross-border share and amounts. "
                "The laundering label is used only afterwards, to judge how useful each clustering is.")

    with st.expander("Data settings", icon=":material/tune:"):
        d1, d2 = st.columns([1, 2])
        with d1:
            n_accounts = st.slider("Accounts to cluster", 1000, 8000, 4000, 500,
                                   help="Agglomerative clustering and silhouette scores grow quadratically with this.")
            seed = st.number_input("Sample seed", 0, 9999, 42)
        with d2:
            feats = st.multiselect("Behaviour features", clu.FEATURE_NAMES, default=clu.FEATURE_NAMES,
                                   format_func=lambda f: clu.FEATURE_LABELS[f])
    if len(feats) < 2:
        st.error("Select at least two features.")
        return
    features = tuple(feats)
    prof = cl_profiles(n_accounts, seed)
    X = cl_matrix(n_accounts, seed, features)
    coords, evr = cl_pca(n_accounts, seed, features)
    y_sus = prof["ever_suspicious"].values
    st.caption(f"{len(prof):,} accounts · {y_sus.mean() * 100:.1f}% have at least one laundering transaction · "
               f"PCA explains {evr[:2].sum() * 100:.0f}% (2D) and {evr.sum() * 100:.0f}% (3D) of the variance")

    view = st.segmented_control("View", ["K-Means and choosing K", "Algorithm explorer", "Comparative study"],
                                default="K-Means and choosing K", label_visibility="collapsed") or "K-Means and choosing K"

    def param_widgets(algo, prefix, skip=()):
        params = {}
        spec = {p: s for p, s in clu.ALGORITHMS[algo]["params"].items() if p not in skip}
        if not spec:
            st.caption("No further parameters.")
            return params
        cols = st.columns(min(len(spec), 4))
        for i, (p, s) in enumerate(spec.items()):
            with cols[i % len(cols)]:
                key = f"{prefix}_{algo}_{p}"
                if s["type"] == "int":
                    params[p] = st.slider(s["label"], s["min"], s["max"], s["default"], s.get("step", 1), key=key)
                elif s["type"] == "float":
                    params[p] = st.slider(s["label"], float(s["min"]), float(s["max"]), float(s["default"]),
                                          float(s.get("step", 0.1)), key=key)
                else:
                    params[p] = st.selectbox(s["label"], s["options"], index=s["options"].index(s["default"]), key=key)
        return params

    def fmt(v, spec="{:.3f}"):
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else spec.format(v)

    def render_results(algo, params, key):
        res = cl_run(algo, tuple(sorted(params.items())), n_accounts, seed, features)
        labels, m = res["labels"], res["metrics"]
        kpis([("Clusters", str(m["n_clusters"])), ("Noise points", f"{m['noise_pct']:.1f}%"),
              ("Silhouette", fmt(m["silhouette"]), "Higher is better (range -1 to 1)"),
              ("Davies-Bouldin", fmt(m["davies_bouldin"]), "Lower is better"),
              ("Best cluster lift", f"{m['max_suspicious_lift']:.1f}×",
               "Suspicious rate of the most suspicious cluster divided by the overall rate"),
              ("Runtime", f"{res['seconds'] * 1000:.0f} ms")])

        cmap = cluster_color_map(labels)
        order = cluster_order(cmap)
        plot_df = prof.assign(cluster=cluster_names(labels), pc1=coords[:, 0], pc2=coords[:, 1], pc3=coords[:, 2],
                              suspicious=np.where(y_sus == 1, "Yes", "No"))
        hover = {"account_id": True, "out_tx_count": True, "distinct_receivers": True, "distinct_senders": True,
                 "cash_ratio": ":.2f", "suspicious": True, "pc1": False, "pc2": False, "pc3": False}
        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            dim = st.segmented_control("Projection", ["2D", "3D"], default="2D", key=f"{key}_dim") or "2D"
            if dim == "3D":
                fig = px.scatter_3d(plot_df, x="pc1", y="pc2", z="pc3", color="cluster", color_discrete_map=cmap,
                                    category_orders={"cluster": order}, hover_data=hover, opacity=0.75)
                fig.update_traces(marker_size=3)
            else:
                fig = px.scatter(plot_df, x="pc1", y="pc2", color="cluster", color_discrete_map=cmap,
                                 category_orders={"cluster": order}, hover_data=hover, opacity=0.75,
                                 labels={"pc1": "PC1", "pc2": "PC2"})
                fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color="white")))
            chart(fig, f"{algo}: accounts in PCA space", height=470)
        with c2:
            rates = plot_df.groupby("cluster").agg(accounts=("account_id", "size"), rate=("ever_suspicious", "mean"))
            rates = rates.reset_index().assign(rate=lambda d: d["rate"] * 100)
            fig = px.bar(rates, x="cluster", y="rate", color="cluster", color_discrete_map=cmap,
                         category_orders={"cluster": order}, hover_data={"accounts": True},
                         labels={"rate": "Suspicious accounts (%)", "cluster": ""})
            fig.add_hline(y=y_sus.mean() * 100, line_dash="dot", line_color=NEUTRAL, annotation_text="overall",
                          annotation_position="top right")
            chart(fig, "Suspicious rate per cluster", "Dotted line: rate across all sampled accounts",
                  height=500, legend=False)

        prof_tbl = clu.profile_clusters(prof, X, labels, list(features))
        st.markdown('<div class="chart-title">Cluster profiles</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-caption">Each cluster is named after its two most distinctive features</div>',
                    unsafe_allow_html=True)
        st.dataframe(prof_tbl[["cluster_name", "accounts", "share_pct", "suspicious_rate_pct"]],
                     hide_index=True, width="stretch",
                     column_config={"cluster_name": st.column_config.TextColumn("Cluster", width="large"),
                                    "accounts": st.column_config.NumberColumn("Accounts"),
                                    "share_pct": st.column_config.NumberColumn("Share (%)", format="%.1f"),
                                    "suspicious_rate_pct": st.column_config.ProgressColumn(
                                        "Suspicious (%)", format="%.1f", min_value=0, max_value=100)})
        z = clu.cluster_zscores(X, labels, list(features))
        z.index = [("Noise" if i == -1 else f"C{i}") for i in z.index]
        lim = float(np.nanmax(np.abs(z.values))) or 1
        fig = px.imshow(z.T, color_continuous_scale=DIVERGING, zmin=-lim, zmax=lim, aspect="auto", text_auto=".1f",
                        labels=dict(color="z-score", x="", y=""))
        fig.update_xaxes(side="top")
        chart(fig, "Feature profile per cluster",
              "Standardised mean of each feature: red is above the overall average, blue is below",
              height=80 + 30 * len(features), legend=False)
        return res

    if view == "K-Means and choosing K":
        st.subheader("1. Sweep K and read the diagnostics")
        a1, a2, a3 = st.columns([2, 1, 1])
        with a1:
            k_range = st.slider("K range", 2, 15, (2, 10))
        with a2:
            init = st.selectbox("Initialisation", ["k-means++", "random"])
        with a3:
            n_init = st.slider("Restarts (n_init)", 1, 30, 10)
        if k_range[1] - k_range[0] < 2:
            st.warning("Choose a range spanning at least 3 values of K.")
            return
        base = {"init": init, "n_init": n_init}
        with st.spinner("Running K-Means for each K"):
            sweep = cl_sweep("K-Means", k_range[0], k_range[1], tuple(sorted(base.items())), n_accounts, seed, features)
        elbow_k = clu.find_elbow(sweep["k"], sweep["inertia"])
        sil_k = int(sweep.loc[sweep["silhouette"].idxmax(), "k"])
        db_k = int(sweep.loc[sweep["davies_bouldin"].idxmin(), "k"])
        kpis([("Elbow method suggests", f"K = {elbow_k}"), ("Best silhouette", f"K = {sil_k}"),
              ("Best Davies-Bouldin", f"K = {db_k}")])

        k_sel = st.slider("2. Choose K", k_range[0], k_range[1], elbow_k)
        diag = [("inertia", "Elbow: within-cluster sum of squares", elbow_k, "Lower is tighter"),
                ("silhouette", "Silhouette score", sil_k, "Higher means better-separated clusters"),
                ("davies_bouldin", "Davies-Bouldin index", db_k, "Lower is better"),
                ("calinski_harabasz", "Calinski-Harabasz index", None, "Higher is better")]
        cols = st.columns(2, gap="large")
        for i, (col, title, mark_k, hint) in enumerate(diag):
            with cols[i % 2]:
                fig = px.line(sweep, x="k", y=col, markers=True, color_discrete_sequence=[SERIES[0]],
                              labels={"k": "K", col: ""})
                fig.update_traces(line_width=2, marker_size=8)
                fig.add_vline(x=k_sel, line_dash="dot", line_color=SERIES[1])
                if mark_k is not None:
                    row = sweep[sweep["k"] == mark_k].iloc[0]
                    fig.add_scatter(x=[mark_k], y=[row[col]], mode="markers", name="suggested",
                                    marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(width=2, color=SERIES[7])))
                fig.update_xaxes(dtick=1)
                chart(fig, title, f"{hint} · orange line = chosen K"
                      + (" · red ring = suggested K" if mark_k is not None else ""), height=280, legend=False)

        st.subheader(f"3. K-Means result with K = {k_sel}")
        params = {**clu.default_params("K-Means"), **base, "n_clusters": k_sel}
        render_results("K-Means", params, "km")

        sil = cl_silhouette_samples("K-Means", tuple(sorted(params.items())), n_accounts, seed, features)
        if sil is not None:
            labs, vals = sil
            cmap = cluster_color_map(labs)
            fig = go.Figure()
            y0 = 0
            for c in sorted(set(labs)):
                v = np.sort(vals[labs == c])
                fig.add_scatter(x=v, y=np.arange(y0, y0 + len(v)), fill="tozerox", mode="lines",
                                line=dict(width=0.5, color=cmap[f"C{c}"]), name=f"C{c}")
                y0 += len(v) + 25
            fig.add_vline(x=float(vals.mean()), line_dash="dot", line_color=SERIES[7],
                          annotation_text=f"mean {vals.mean():.3f}", annotation_position="top right")
            fig.update_layout(xaxis_title="Silhouette coefficient", yaxis=dict(showticklabels=False, title=""))
            chart(fig, "Silhouette plot",
                  "One band per cluster. Wide bands past the mean line are well separated; "
                  "negative values sit closer to another cluster.", height=420)

    elif view == "Algorithm explorer":
        algo = st.selectbox("Algorithm", list(clu.ALGORITHMS.keys()),
                            format_func=lambda a: f"{a}  ({clu.ALGORITHMS[a]['family']})")
        with st.container(border=True):
            params = param_widgets(algo, "exp")
        if algo == "DBSCAN":
            with st.expander("Help choosing eps: k-distance plot"):
                kd = cl_kdist(n_accounts, seed, features, params["min_samples"])
                fig = px.line(x=np.arange(len(kd)), y=kd, color_discrete_sequence=[SERIES[0]],
                              labels={"x": "Accounts sorted by distance",
                                      "y": f"Distance to neighbour #{params['min_samples']}"})
                fig.add_hline(y=params["eps"], line_dash="dot", line_color=SERIES[1],
                              annotation_text=f"eps = {params['eps']}")
                chart(fig, None, "Pick eps near the knee, where the curve bends sharply upwards", height=300, legend=False)
        if algo == "Gaussian Mixture (EM)":
            with st.expander("Help choosing components: BIC and AIC"):
                gs = cl_sweep(algo, 2, 12, tuple(sorted({k: v for k, v in params.items() if k != "n_clusters"}.items())),
                              n_accounts, seed, features)
                long = gs.melt(id_vars="k", value_vars=["bic", "aic"], var_name="Criterion", value_name="value")
                long["Criterion"] = long["Criterion"].str.upper()
                fig = px.line(long, x="k", y="value", color="Criterion", markers=True, color_discrete_sequence=SERIES[:2],
                              labels={"k": "Components", "value": ""})
                fig.add_vline(x=params["n_clusters"], line_dash="dot", line_color=NEUTRAL)
                chart(fig, None, "Lower is better for both criteria", height=300)
        render_results(algo, params, "exp")

    else:
        st.subheader("Configure")
        chosen = st.multiselect("Algorithms to compare", list(clu.ALGORITHMS.keys()), default=list(clu.ALGORITHMS.keys()))
        if not chosen:
            st.info("Select at least one algorithm.")
            return
        common_k = st.slider("Number of clusters for the K-based algorithms", 2, 15, 5)
        configs = {}
        with st.container(border=True):
            st.caption("Algorithm-specific parameters")
            for tab, algo in zip(st.tabs(chosen), chosen):
                with tab:
                    st.caption(clu.ALGORITHMS[algo]["family"])
                    p = param_widgets(algo, "cmp", skip=("n_clusters",))
                    if clu.ALGORITHMS[algo]["uses_k"]:
                        p["n_clusters"] = common_k
                    configs[algo] = {**clu.default_params(algo), **p}

        with st.spinner("Running all algorithms"):
            runs = {a: cl_run(a, tuple(sorted(p.items())), n_accounts, seed, features) for a, p in configs.items()}

        cmp_df = pd.DataFrame([{
            "Algorithm": a, "Family": clu.ALGORITHMS[a]["family"], "Clusters": r["metrics"]["n_clusters"],
            "Noise %": r["metrics"]["noise_pct"], "Silhouette": r["metrics"]["silhouette"],
            "Davies-Bouldin": r["metrics"]["davies_bouldin"], "Calinski-Harabasz": r["metrics"]["calinski_harabasz"],
            "Best cluster lift": r["metrics"]["max_suspicious_lift"],
            "Suspicious in hotspots %": r["metrics"]["suspicious_in_hotspots_pct"],
            "Runtime (ms)": r["seconds"] * 1000} for a, r in runs.items()])

        valid = cmp_df.dropna(subset=["Silhouette"])
        st.subheader("Results")
        if len(valid):
            kpis([("Best separation (silhouette)", valid.loc[valid["Silhouette"].idxmax(), "Algorithm"]),
                  ("Most compact (Davies-Bouldin)", valid.loc[valid["Davies-Bouldin"].idxmin(), "Algorithm"]),
                  ("Strongest AML hotspot (lift)", cmp_df.loc[cmp_df["Best cluster lift"].idxmax(), "Algorithm"])])
        st.dataframe(
            cmp_df.drop(columns=["Family"]).style.format({"Silhouette": "{:.3f}", "Davies-Bouldin": "{:.3f}", "Noise %": "{:.1f}",
                                 "Calinski-Harabasz": "{:.0f}", "Best cluster lift": "{:.1f}×",
                                 "Suspicious in hotspots %": "{:.1f}", "Runtime (ms)": "{:.0f}"}, na_rep="n/a")
            .highlight_max(subset=["Silhouette", "Calinski-Harabasz", "Best cluster lift", "Suspicious in hotspots %"],
                           color="rgba(42,120,214,0.18)")
            .highlight_min(subset=["Davies-Bouldin", "Runtime (ms)"], color="rgba(42,120,214,0.18)"),
            hide_index=True, width="stretch")
        st.caption("Highlighted cells are the best in each column. Best cluster lift: suspicious rate of the most "
                   "suspicious cluster divided by the overall rate. Suspicious in hotspots: share of suspicious accounts "
                   "in clusters with at least twice the overall rate. Density methods' internal scores exclude noise.")

        metric_specs = [("Silhouette", True, "Higher is better"), ("Davies-Bouldin", False, "Lower is better"),
                        ("Calinski-Harabasz", True, "Higher is better"), ("Best cluster lift", True, "Higher is better"),
                        ("Suspicious in hotspots %", True, "Higher is better"), ("Runtime (ms)", False, "Lower is better")]
        value_fmt = {"Silhouette": ".3f", "Davies-Bouldin": ".3f", "Calinski-Harabasz": ",.0f",
                     "Best cluster lift": ".1f", "Suspicious in hotspots %": ".1f", "Runtime (ms)": ",.0f"}
        grid = st.columns(2, gap="large")
        for i, (mcol, higher, hint) in enumerate(metric_specs):
            with grid[i % 2]:
                d = cmp_df[["Algorithm", mcol]].dropna()
                best_val = d[mcol].max() if higher else d[mcol].min()
                d = d.assign(best=np.where(d[mcol] == best_val, "Best", "Other"))
                fig = px.bar(d, x=mcol, y="Algorithm", orientation="h", color="best", text=mcol,
                             color_discrete_map={"Best": SERIES[0], "Other": SOFT_BLUE}, labels={mcol: "", "Algorithm": ""})
                fig.update_traces(texttemplate=f"%{{text:{value_fmt[mcol]}}}", textposition="outside", cliponaxis=False)
                fig.update_yaxes(categoryorder="array", categoryarray=list(cmp_df["Algorithm"])[::-1])
                fig.update_xaxes(showticklabels=False, range=[0, d[mcol].max() * 1.18])
                fig.update_layout(bargap=0.35)
                chart(fig, mcol, f"{hint} · darker bar = best", height=270, legend=False)

        st.subheader("Side-by-side clusterings")
        st.caption("Every algorithm plotted on the same PCA projection of the same accounts")
        grid = st.columns(3, gap="medium")
        for i, (a, r) in enumerate(runs.items()):
            with grid[i % 3]:
                cmap = cluster_color_map(r["labels"])
                fig = px.scatter(x=coords[:, 0], y=coords[:, 1], color=cluster_names(r["labels"]),
                                 color_discrete_map=cmap, category_orders={"color": cluster_order(cmap)},
                                 opacity=0.75, labels={"x": "PC1", "y": "PC2", "color": ""})
                fig.update_traces(marker=dict(size=4))
                chart(fig, a, f"{r['metrics']['n_clusters']} clusters"
                      + (f", {r['metrics']['noise_pct']:.0f}% noise" if r["metrics"]["noise_pct"] else ""),
                      height=300, legend=False)

        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            names_list = list(runs.keys())
            ari = pd.DataFrame([[adjusted_rand_score(runs[a]["labels"], runs[b]["labels"]) for b in names_list]
                                for a in names_list], index=names_list, columns=names_list)
            fig = px.imshow(ari, text_auto=".2f", color_continuous_scale=SEQ_BLUE, zmin=0, zmax=1, aspect="auto",
                            labels=dict(x="", y="", color="ARI"))
            chart(fig, "Agreement between algorithms",
                  "Adjusted Rand Index: 1 = identical partitions, 0 = chance-level agreement", height=440, legend=False)
        with c2:
            k_algos = [a for a in chosen if clu.ALGORITHMS[a]["uses_k"]]
            if k_algos:
                kr = st.slider("K range for the sensitivity sweep", 2, 15, (2, 10), key="cmp_krange")
                frames = []
                with st.spinner("Sweeping K for each algorithm"):
                    for a in k_algos:
                        base = {k: v for k, v in configs[a].items() if k != "n_clusters"}
                        s = cl_sweep(a, kr[0], kr[1], tuple(sorted(base.items())), n_accounts, seed, features)
                        frames.append(s.assign(Algorithm=a))
                sw = pd.concat(frames)
                algo_cmap = {a: SERIES[i] for i, a in enumerate(k_algos)}
                fig = px.line(sw, x="k", y="silhouette", color="Algorithm", markers=True, color_discrete_map=algo_cmap,
                              labels={"k": "K", "silhouette": "Silhouette"})
                fig.update_traces(line_width=2, marker_size=7)
                fig.update_xaxes(dtick=1)
                chart(fig, "Sensitivity to K", "Silhouette score of each K-based algorithm as K changes", height=360)


# =========================================================================== ASSOCIATION RULES
def page_rules():
    page_header("Data mining", "Association rules",
                "FP-Growth over transaction itemsets: channel, amount band, route, timing and sender/receiver history. "
                "Only rules whose outcome is a suspicious flag or a laundering typology are kept.")
    if asc_art is None:
        return missing_artifact()
    rules_df, p = asc_art["rules_df"], asc_art["params"]
    kpis([("Rules mined", f"{len(rules_df):,}"), ("Min support", f"{p['min_support']}"),
          ("Min confidence", f"{p['min_confidence']}"), ("Max itemset length", str(p["max_len"]))])
    c1, c2, c3 = st.columns(3)
    with c1:
        min_c = st.slider("Minimum confidence", 0.2, 1.0, 0.6, 0.05)
    with c2:
        min_l = st.slider("Minimum lift", 1.0, float(max(2.0, np.ceil(rules_df["lift"].max()))), 5.0, 1.0)
    with c3:
        target = st.selectbox("Rule outcome", ["Any", "Suspicious flag only", "Typology only"])
    fr = rules_df[(rules_df["confidence"] >= min_c) & (rules_df["lift"] >= min_l)]
    if target == "Suspicious flag only":
        fr = fr[fr["consequents_str"] == "Outcome=SUSPICIOUS"]
    elif target == "Typology only":
        fr = fr[fr["consequents_str"].str.contains("Typology=")]
    st.caption(f"{len(fr)} rules match the filters")
    st.dataframe(fr[["antecedents_str", "consequents_str", "support", "confidence", "lift"]], hide_index=True,
                 width="stretch", height=400,
                 column_config={"antecedents_str": st.column_config.TextColumn("If", width="large"),
                                "consequents_str": st.column_config.TextColumn("Then", width="large"),
                                "support": st.column_config.NumberColumn("Support", format="%.4f"),
                                "confidence": st.column_config.ProgressColumn("Confidence", format="%.2f",
                                                                              min_value=0, max_value=1),
                                "lift": st.column_config.NumberColumn("Lift", format="%.1f")})
    if len(fr):
        fig = px.scatter(fr, x="support", y="confidence", size="lift", color="lift", color_continuous_scale=SEQ_BLUE[2:],
                         hover_data={"antecedents_str": True, "consequents_str": True},
                         labels={"antecedents_str": "If", "consequents_str": "Then", "support": "Support",
                                 "confidence": "Confidence", "lift": "Lift"})
        chart(fig, "Support vs confidence", "Bubble size and colour show lift", height=420, legend=False)


# =========================================================================== RISK PREDICTION
def page_prediction():
    if pred_art is None:
        page_header("Data mining", "Risk prediction")
        return missing_artifact()
    m, tf = pred_art["metrics"], pred_art["test_frame"]
    y, s = tf["y_true"].values, tf["risk_score"].values
    page_header("Data mining", "Risk prediction",
                f"A calibrated gradient-boosting model scores each transaction from 0 to 100. It is trained on "
                f"{month_label(m['train_months'][0])} – {month_label(m['train_months'][-1])} "
                f"({m['train_size']:,} transactions) and evaluated on {month_label(m['test_months'][0])} – "
                f"{month_label(m['test_months'][-1])} ({m['test_size']:,} transactions), which it has never seen.")
    kpis([("Forecast transactions", f"{m['test_size']:,}"), ("Laundering in forecast", f"{int(y.sum()):,}"),
          ("ROC-AUC", f"{m['roc_auc']:.4f}"), ("PR-AUC", f"{m['pr_auc']:.4f}"),
          ("Brier score", f"{m['brier']:.4f}", "Mean squared error of the probabilities; lower is better")])
    st.info("SAML-D's simulated laundering networks are highly separable once counterparty history is known, so scores "
            "cluster near 0 and 100. Every feature uses only information available before the transaction, and the "
            "forecast months are never used in training.", icon=":material/info:")

    thr = st.slider("Alert threshold (risk score)", 1, 99, HIGH_RISK_SCORE)
    pred = (s >= thr).astype(int)
    kpis([("Alerts raised", f"{pred.sum():,}"), ("Precision", f"{precision_score(y, pred, zero_division=0):.3f}"),
          ("Recall", f"{recall_score(y, pred):.3f}"), ("False positives", f"{int(((pred == 1) & (y == 0)).sum()):,}"),
          ("Missed laundering", f"{int(((pred == 0) & (y == 1)).sum()):,}")])

    marker = dict(size=12, color=SERIES[1], line=dict(width=2, color="white"))
    c1, c2 = st.columns(2, gap="large")
    with c1:
        fpr, tpr, _ = roc_curve(y, s)
        fig = go.Figure()
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name="Model", line=dict(color=SERIES[0], width=2))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color=NEUTRAL, dash="dot", width=1))
        fig.add_scatter(x=[((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)],
                        y=[((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)],
                        mode="markers", name=f"Threshold {thr}", marker=marker)
        fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
        chart(fig, "ROC curve", height=380)
    with c2:
        pr, rc, _ = precision_recall_curve(y, s)
        fig = go.Figure()
        fig.add_scatter(x=rc, y=pr, mode="lines", name="Model", line=dict(color=SERIES[0], width=2))
        fig.add_scatter(x=[recall_score(y, pred)], y=[precision_score(y, pred, zero_division=0)], mode="markers",
                        name=f"Threshold {thr}", marker=marker)
        fig.add_hline(y=y.mean(), line_dash="dot", line_color=NEUTRAL, annotation_text="Base rate",
                      annotation_position="bottom left")
        fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
        chart(fig, "Precision-recall curve", height=380)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        dist = pd.DataFrame({"Risk score": s, "Class": np.where(y == 1, "Laundering", "Normal")})
        fig = px.histogram(dist, x="Risk score", color="Class", nbins=50, barmode="overlay", log_y=True, opacity=0.75,
                           color_discrete_map={"Normal": SERIES[0], "Laundering": SERIES[7]})
        fig.add_vline(x=thr, line_dash="dot", line_color=SERIES[1])
        fig.update_layout(yaxis_title="Transactions (log scale)")
        chart(fig, "Risk score distribution", "Orange line = alert threshold", height=380)
    with c4:
        cal = pred_art["calibration"]
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                        line=dict(color=NEUTRAL, dash="dot", width=1))
        fig.add_scatter(x=cal["mean_predicted"], y=cal["fraction_positive"], mode="lines+markers", name="Model",
                        line=dict(color=SERIES[0], width=2), marker=dict(size=8))
        fig.update_layout(xaxis_title="Predicted probability", yaxis_title="Observed laundering rate")
        chart(fig, "Calibration curve", "Points on the diagonal mean the score can be read as a probability", height=380)

    c5, c6 = st.columns(2, gap="large")
    with c5:
        ths = np.arange(1, 100)
        alerts = s[None, :] >= ths[:, None]
        tp = (alerts & (y[None, :] == 1)).sum(axis=1)
        n_alerts = alerts.sum(axis=1)
        prec = np.divide(tp, n_alerts, out=np.zeros(len(ths)), where=n_alerts > 0)
        rec = tp / max(y.sum(), 1)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(len(ths)), where=(prec + rec) > 0)
        long = pd.DataFrame({"Threshold": ths, "Precision": prec, "Recall": rec, "F1": f1}).melt(
            id_vars="Threshold", var_name="Metric", value_name="Value")
        fig = px.line(long, x="Threshold", y="Value", color="Metric", color_discrete_sequence=SERIES[:3])
        fig.update_traces(line_width=2)
        fig.add_vline(x=thr, line_dash="dot", line_color=NEUTRAL)
        fig.update_layout(yaxis_range=[0, 1.02], yaxis_title="")
        chart(fig, "Threshold trade-off", height=380)
    with c6:
        monthly = tf.assign(alert=pred).groupby("year_month").agg(Actual=("y_true", "sum"),
                                                                    Predicted=("alert", "sum")).reset_index()
        monthly["Month"] = monthly["year_month"].map(month_label)
        long = monthly.melt(id_vars="Month", value_vars=["Actual", "Predicted"], var_name="Series", value_name="Transactions")
        long["Series"] = long["Series"].map({"Actual": "Actual laundering", "Predicted": "Alerts raised"})
        fig = px.bar(long, x="Month", y="Transactions", color="Series", barmode="group",
                     color_discrete_map={"Actual laundering": SERIES[7], "Alerts raised": SERIES[0]}, labels={"Month": ""})
        chart(fig, "Forecast months: alerts vs actual laundering", height=380)

    c7, c8 = st.columns(2, gap="large")
    with c7:
        typ = (tf[tf["y_true"] == 1].assign(alert=pred[y == 1]).groupby("laundering_typology")
               .agg(avg_score=("risk_score", "mean"), detected=("alert", "mean")).reset_index())
        typ["detected"] *= 100
        fig = px.bar(typ.sort_values("detected"), x="detected", y="laundering_typology", orientation="h",
                     color_discrete_sequence=[SERIES[2]], hover_data={"avg_score": ":.1f"},
                     labels={"detected": "Detected at threshold (%)", "laundering_typology": "", "avg_score": "Avg score"})
        fig.update_xaxes(range=[0, 100])
        chart(fig, "Detection rate by typology", height=470, legend=False)
    with c8:
        ch = tf.groupby(["channel_name", "y_true"])["risk_score"].mean().reset_index()
        ch["Class"] = np.where(ch["y_true"] == 1, "Laundering", "Normal")
        fig = px.bar(ch, x="risk_score", y="channel_name", color="Class", barmode="group", orientation="h",
                     color_discrete_map={"Normal": SERIES[0], "Laundering": SERIES[7]}, text="risk_score",
                     labels={"channel_name": "", "risk_score": "Average risk score"})
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
        fig.update_xaxes(range=[0, 112])
        chart(fig, "Average risk score by channel", height=470)

    st.subheader("Highest-risk sender accounts in the forecast window")
    acc = (tf.groupby("sender_account").agg(max_score=("risk_score", "max"), avg_score=("risk_score", "mean"),
                                            transactions=("transaction_id", "size"),
                                            alerts=("risk_score", lambda v: int((v >= thr).sum())),
                                            actual=("y_true", "sum"))
           .sort_values(["max_score", "alerts"], ascending=False).head(25).reset_index())
    st.dataframe(acc, hide_index=True, width="stretch",
                 column_config={"sender_account": "Sender account",
                                "max_score": st.column_config.ProgressColumn("Max score", format="%.0f", min_value=0, max_value=100),
                                "avg_score": st.column_config.NumberColumn("Avg score", format="%.1f"),
                                "transactions": "Transactions", "alerts": "Alerts", "actual": "Actual laundering"})


# =========================================================================== LIVE SCORING
def page_live():
    page_header("Investigation", "Live risk scoring",
                "Load a real transaction from the forecast window, or edit the inputs, and the model re-scores it instantly.")
    if pred_art is None:
        return missing_artifact()
    pool = pred_art["example_pool"]
    defaults = {"amount": float(STRUCTURING_LOW * 1.05), "channel_name": "Cash Deposit", "hour_of_day": 14,
                "day_of_week": 3, "is_cross_border": False, "is_high_risk_corridor": False, "is_fx_conversion": False,
                "s_prior_out_count": 20, "s_prior_in_count": 5, "s_prior_out_avg": float(STRUCTURING_HIGH * 0.3),
                "s_prior_distinct_receivers": 5, "s_prior_cash_ratio": 0.1, "s_prior_cross_border_ratio": 0.0,
                "s_prior_high_risk_ratio": 0.0, "s_secs_since_last_out": 86400, "r_prior_in_count": 3,
                "r_prior_out_count": 0, "r_prior_distinct_senders": 2}
    for k, v in defaults.items():
        st.session_state.setdefault(f"sim_{k}", v)

    left, right = st.columns([3, 2], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**Load a real example**")
            typologies = ["Normal (any)"] + sorted(pool.loc[pool["is_suspicious"] == 1, "laundering_typology"].unique())
            e1, e2 = st.columns([3, 2], vertical_alignment="bottom")
            with e1:
                typ_pick = st.selectbox("Typology", typologies, index=typologies.index("Structuring")
                                        if "Structuring" in typologies else 0)
            with e2:
                if st.button("Load random transaction", width="stretch", icon=":material/shuffle:"):
                    cand = (pool[pool["is_suspicious"] == 0] if typ_pick == "Normal (any)"
                            else pool[pool["laundering_typology"] == typ_pick])
                    row = cand.sample(1).iloc[0]
                    for k in defaults:
                        st.session_state[f"sim_{k}"] = type(defaults[k])(row[k])
                    st.session_state["sim_loaded"] = f"{row['transaction_id']} ({row['laundering_typology']})"
                    st.rerun()
            if "sim_loaded" in st.session_state:
                st.caption(f"Loaded {st.session_state['sim_loaded']}")

        st.markdown("**Transaction**")
        c1, c2 = st.columns(2)
        with c1:
            amount = st.number_input(f"Amount ({SYM})", min_value=1.0, max_value=float(STRUCTURING_HIGH * 2000),
                                     step=float(STRUCTURING_HIGH / 20), key="sim_amount")
            st.selectbox("Payment channel", CHANNELS, key="sim_channel_name")
            st.toggle("Cross-border", key="sim_is_cross_border")
            st.toggle("Currency conversion", key="sim_is_fx_conversion")
        with c2:
            st.slider("Hour of day", 0, 23, key="sim_hour_of_day")
            st.select_slider("Day of week", options=[1, 2, 3, 4, 5, 6, 7],
                             format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d - 1],
                             key="sim_day_of_week")
            st.toggle("High-risk corridor", key="sim_is_high_risk_corridor",
                      help=f"Touches an FATF grey-list location: {', '.join(sorted(HIGH_RISK_LOCATIONS))}")

        with st.expander("Sender and receiver history (before this transaction)", expanded=True):
            h1, h2 = st.columns(2)
            with h1:
                st.number_input("Sender: past outgoing payments", 0, 100000, step=1, key="sim_s_prior_out_count")
                st.number_input("Sender: past incoming payments", 0, 100000, step=1, key="sim_s_prior_in_count")
                st.number_input(f"Sender: past average amount ({SYM})", 0.0, 1e10, step=100.0, key="sim_s_prior_out_avg")
                st.number_input("Sender: past distinct receivers", 0, 100000, step=1, key="sim_s_prior_distinct_receivers")
                st.number_input("Sender: seconds since last payment (-1 = none)", -1, 10**9, step=60,
                                key="sim_s_secs_since_last_out")
            with h2:
                st.slider("Sender: past cash share", 0.0, 1.0, step=0.05, key="sim_s_prior_cash_ratio")
                st.slider("Sender: past cross-border share", 0.0, 1.0, step=0.05, key="sim_s_prior_cross_border_ratio")
                st.slider("Sender: past high-risk share", 0.0, 1.0, step=0.05, key="sim_s_prior_high_risk_ratio")
                st.number_input("Receiver: past incoming payments", 0, 100000, step=1, key="sim_r_prior_in_count")
                st.number_input("Receiver: past outgoing payments", 0, 100000, step=1, key="sim_r_prior_out_count")
                st.number_input("Receiver: past distinct senders", 0, 100000, step=1, key="sim_r_prior_distinct_senders")

    tx = {k: st.session_state[f"sim_{k}"] for k in defaults}
    for k in ("is_cross_border", "is_high_risk_corridor", "is_fx_conversion"):
        tx[k] = int(tx[k])
    out = explain_transaction_risk(tx)
    score = out["predicted_score"]
    if score >= HIGH_RISK_SCORE:
        tier, color = f"High risk: file a {RPT}", STATUS["critical"]
        action = (f"File a {CUR['report_long']} with the {CUR['authority']} under the {CUR['law']}.")
    elif score >= MEDIUM_RISK_SCORE:
        tier, color = "Medium risk: enhanced due diligence", "#b7791f"
        action = "Route to a compliance analyst for Enhanced Due Diligence (EDD)."
    else:
        tier, color = "Low risk: straight-through processing", STATUS["good"]
        action = "No action needed; process normally."

    with right:
        flags = "".join(f"<li>{f}</li>" for f in out["key_factors"])
        st.markdown(f"""
<div class="score-card">
  <div style="font-size:0.8rem;opacity:0.7;font-weight:600;letter-spacing:0.04em;text-transform:uppercase">AML risk score</div>
  <div style="margin-top:0.4rem"><span class="score-num">{score:.0f}</span><span class="score-of"> / 100</span></div>
  <span class="tier-pill" style="background:{color}">{tier}</span>
  <div class="score-track"><div class="score-marker" style="left:{min(max(score, 0), 100)}%"></div></div>
  <div class="score-scale"><span>0</span><span>{MEDIUM_RISK_SCORE}</span><span>{HIGH_RISK_SCORE}</span><span>100</span></div>
  <div style="margin-top:1.1rem;font-weight:600;font-size:0.9rem">Red flags</div>
  <ul class="flag-list">{flags}</ul>
  <div style="margin-top:0.8rem;font-weight:600;font-size:0.9rem">Recommended action</div>
  <div style="font-size:0.9rem;margin-top:0.2rem">{action}</div>
  <div style="margin-top:0.8rem;font-size:0.78rem;opacity:0.6">Amount: {format_money(amount)}</div>
</div>
""", unsafe_allow_html=True)


# =========================================================================== MONEY FLOW NETWORK
def page_network():
    page_header("Investigation", "Money flow network",
                "Real laundering transactions from SAML-D, grouped into connected networks of accounts.")
    sus = df_all_tx[df_all_tx["is_suspicious"] == 1]
    typs = sorted(sus["laundering_typology"].unique())
    c1, c2 = st.columns([1, 2])
    with c1:
        typ_pick = st.selectbox("Typology", typs, index=typs.index("Fan In") if "Fan In" in typs else 0)
    edges = sus[sus["laundering_typology"] == typ_pick]
    G = nx.MultiDiGraph()
    for r in edges.itertuples():
        G.add_edge(r.sender_account, r.receiver_account, amount=r.amount)
    comps = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    comps = [c for c in comps if len(c) >= 3] or comps

    def comp_label(i):
        sub = G.subgraph(comps[i])
        vol = sum(d["amount"] for _, _, d in sub.edges(data=True))
        return f"Network {i + 1}: {len(comps[i])} accounts, {sub.number_of_edges()} payments, {format_money(vol)}"
    with c2:
        idx = st.selectbox("Network", range(len(comps)), format_func=comp_label)
    st.caption(f"{len(edges):,} {typ_pick} transactions form {len(comps)} networks of three or more accounts")

    nodes = list(comps[idx])[:60]
    H = G.subgraph(nodes)
    pos = nx.spring_layout(nx.DiGraph(H), seed=7, k=1.4 / np.sqrt(max(len(nodes), 1)))
    indeg, outdeg = dict(H.in_degree()), dict(H.out_degree())
    role = {n: ("Source" if indeg[n] == 0 else "Sink" if outdeg[n] == 0 else "Intermediary") for n in H.nodes}
    role_color = {"Source": SERIES[0], "Intermediary": SERIES[3], "Sink": SERIES[7]}
    size = {n: 14 + 2.5 * min(indeg[n] + outdeg[n], 10) for n in H.nodes}

    fig = go.Figure()
    for u, v in {(u, v) for u, v in H.edges()}:
        fig.add_annotation(x=pos[v][0], y=pos[v][1], ax=pos[u][0], ay=pos[u][1], xref="x", yref="y", axref="x",
                           ayref="y", showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=1.4,
                           arrowcolor="rgba(137,135,129,0.85)", standoff=size[v] / 2 + 2, startstandoff=size[u] / 2)
    for rname, color in role_color.items():
        ns = [n for n in H.nodes if role[n] == rname]
        if not ns:
            continue
        sent = {n: sum(d["amount"] for _, _, d in H.out_edges(n, data=True)) for n in ns}
        recv = {n: sum(d["amount"] for _, _, d in H.in_edges(n, data=True)) for n in ns}
        fig.add_scatter(x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns], mode="markers", name=rname,
                        marker=dict(size=[size[n] for n in ns], color=color, line=dict(width=2, color="white")),
                        hovertext=[f"Account {n}<br>{rname}<br>Sent {format_money(sent[n])}<br>Received {format_money(recv[n])}"
                                   for n in ns], hoverinfo="text")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    chart(fig, None, "Source accounts only send, sinks only receive, intermediaries do both. "
          "Larger nodes have more connections.", height=560)

    tbl = edges[edges["sender_account"].isin(nodes) & edges["receiver_account"].isin(nodes)].sort_values("timestamp")
    st.dataframe(tbl[["timestamp", "sender_account", "receiver_account", "channel_name", "corridor", "amount"]],
                 hide_index=True, width="stretch",
                 column_config={"timestamp": "Timestamp", "sender_account": "Sender", "receiver_account": "Receiver",
                                "channel_name": "Channel", "corridor": "Corridor",
                                "amount": st.column_config.NumberColumn(f"Amount ({SYM})", format="%.2f")})


# =========================================================================== CASE GENERATOR
def page_case():
    page_header("Investigation", f"{RPT} case generator",
                f"Draft a {CUR['report_long']} for the {CUR['authority']} from a high-risk transaction.")
    if pred_art is not None:
        flagged = pred_art["test_frame"].sort_values("risk_score", ascending=False)
        flagged = flagged[flagged["risk_score"] >= HIGH_RISK_SCORE]
        score_by_id = flagged.set_index("transaction_id")["risk_score"].to_dict()
    else:
        flagged = df_all_tx[df_all_tx["is_suspicious"] == 1]
        score_by_id = {}
    selected_id = st.selectbox("Flagged transaction", flagged["transaction_id"].head(200),
                               format_func=lambda t: f"{t}  (risk {score_by_id[t]:.0f})" if t in score_by_id else t)
    r_tx = df_all_tx[df_all_tx["transaction_id"] == selected_id].iloc[0]
    score = score_by_id.get(selected_id)
    year = str(r_tx["timestamp"])[:4]

    kpis([("Amount", format_money(r_tx["amount"])), ("Channel", r_tx["channel_name"]),
          ("Corridor", r_tx["corridor"]), ("Risk score", f"{score:.0f}" if score is not None else "n/a")])

    narrative = f"""{CUR['report_long'].upper()} ({RPT})
===========================================================================
Filing reference : {RPT}/{year}/{selected_id}
Submitted to     : {CUR['authority']}
Legal basis      : {CUR['law']}
Transaction date : {r_tx['timestamp']}
Subject account  : {r_tx['sender_account']} ({r_tx['account_segment']})
Counterparty     : {r_tx['receiver_account']}
Channel          : {r_tx['channel_name']}
Corridor         : {r_tx['corridor']}
Currencies       : {r_tx['payment_currency']} -> {r_tx['received_currency']}
Amount           : {format_money(r_tx['amount'])} ({CUR['code']} {r_tx['amount']:,.2f})
Model risk score : {f'{score:.1f} / 100' if score is not None else 'n/a'}
Suspected pattern: {r_tx['laundering_typology']}

NARRATIVE:
On {r_tx['timestamp']}, transaction {selected_id} was flagged by the automated transaction-monitoring
system. Its pattern is consistent with the '{r_tx['laundering_typology']}' laundering typology.
{'The amount falls just below the ' + format_money(STRUCTURING_HIGH) + ' reference threshold, which may indicate structuring. ' if r_tx['is_structuring'] else ''}{'The corridor involves a jurisdiction under increased FATF monitoring. ' if r_tx['is_high_risk_corridor'] else ''}
The matter is referred for review and onward reporting.
==========================================================================="""
    st.markdown('<div class="chart-title">Report text</div>', unsafe_allow_html=True)
    st.code(narrative, language=None, wrap_lines=True)
    st.download_button(f"Download {RPT} (TXT)", data=narrative, file_name=f"{RPT}_{selected_id}.txt", mime="text/plain",
                       icon=":material/download:")


# =========================================================================== NAVIGATION
pages = {
    "Overview": [
        st.Page(page_overview, title="Surveillance overview", icon=":material/monitoring:", url_path="overview", default=True),
        st.Page(page_dataset, title="Dataset", icon=":material/dataset:", url_path="dataset"),
    ],
    "Data warehouse": [
        st.Page(page_olap, title="OLAP explorer", icon=":material/view_in_ar:", url_path="olap"),
    ],
    "Data mining": [
        st.Page(page_classification, title="Classification", icon=":material/rule:", url_path="classification"),
        st.Page(page_clustering, title="Clustering lab", icon=":material/bubble_chart:", url_path="clustering"),
        st.Page(page_rules, title="Association rules", icon=":material/link:", url_path="association-rules"),
        st.Page(page_prediction, title="Risk prediction", icon=":material/trending_up:", url_path="prediction"),
    ],
    "Investigation": [
        st.Page(page_live, title="Live risk scoring", icon=":material/speed:", url_path="live-scoring"),
        st.Page(page_network, title="Money flow network", icon=":material/hub:", url_path="network"),
        st.Page(page_case, title=f"{RPT} case generator", icon=":material/description:", url_path="case"),
    ],
}

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
st.logo(os.path.join(ASSETS, "logo.svg"), size="large", icon_image=os.path.join(ASSETS, "icon.svg"))

nav = st.navigation(pages)

with st.sidebar:
    st.markdown('<div class="brand-sub">Team Code Hunters · Data Warehousing & Mining</div>', unsafe_allow_html=True)
    conv = " (converted from GBP)" if CUR["rate_from_gbp"] != 1 else ""
    st.markdown(f"""<div class="facts">
<b>Dataset</b> SAML-D, {len(df_all_tx):,} transactions<br>
<b>Currency</b> {CUR['code']} ({SYM}){conv}<br>
<b>Reference threshold</b> {format_money(STRUCTURING_HIGH)}<br>
<b>Reporting</b> {RPT} to {CUR['authority']}
</div>""", unsafe_allow_html=True)

nav.run()
