"""
Shared transaction-level feature engineering for the classification and prediction modules.
Each transaction is enriched with the sender's and receiver's behaviour profile (Dim_Account).
Labels (is_suspicious, ever_suspicious, typology) are never used as features.
"""

import os
import sys
import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from config import DUCKDB_PATH

CHANNELS = ["ACH", "Cash Deposit", "Cash Withdrawal", "Cheque", "Credit card", "Cross-border", "Debit card"]

NUMERIC_FEATURES = [
    "amount_log", "amount_vs_sender_avg", "is_structuring", "hour_of_day", "day_of_week", "is_weekend",
    "is_cross_border", "is_high_risk_corridor", "is_fx_conversion",
    "s_prior_out_count_log", "s_prior_in_count_log", "s_prior_out_avg_log", "s_prior_distinct_receivers_log",
    "s_prior_cash_ratio", "s_prior_cross_border_ratio", "s_prior_high_risk_ratio", "s_hours_since_last_out_log",
    "s_is_first_out", "r_prior_in_count_log", "r_prior_out_count_log", "r_prior_distinct_senders_log",
]
FEATURE_COLS = NUMERIC_FEATURES + [f"channel_{c}" for c in CHANNELS]

FEATURE_LABELS = {
    "amount_log": "Amount (log)",
    "amount_vs_sender_avg": "Amount vs sender's past average",
    "is_structuring": "Just below reporting threshold",
    "hour_of_day": "Hour of day",
    "day_of_week": "Day of week",
    "is_weekend": "Weekend",
    "is_cross_border": "Cross-border",
    "is_high_risk_corridor": "High-risk corridor",
    "is_fx_conversion": "Currency conversion",
    "s_prior_out_count_log": "Sender: past outgoing txns",
    "s_prior_in_count_log": "Sender: past incoming txns",
    "s_prior_out_avg_log": "Sender: past avg amount",
    "s_prior_distinct_receivers_log": "Sender: past fan-out (distinct receivers)",
    "s_prior_cash_ratio": "Sender: past cash share",
    "s_prior_cross_border_ratio": "Sender: past cross-border share",
    "s_prior_high_risk_ratio": "Sender: past high-risk share",
    "s_hours_since_last_out_log": "Sender: hours since last payment",
    "s_is_first_out": "Sender: first-ever payment",
    "r_prior_in_count_log": "Receiver: past incoming txns",
    "r_prior_out_count_log": "Receiver: past outgoing txns",
    "r_prior_distinct_senders_log": "Receiver: past fan-in (distinct senders)",
}
for _c in CHANNELS:
    FEATURE_LABELS[f"channel_{_c}"] = f"Channel: {_c}"

PIT_COLS = [
    "s_prior_out_count", "s_prior_in_count", "s_prior_out_avg", "s_prior_distinct_receivers",
    "s_prior_cash_ratio", "s_prior_cross_border_ratio", "s_prior_high_risk_ratio", "s_secs_since_last_out",
    "r_prior_in_count", "r_prior_out_count", "r_prior_distinct_senders",
]


def load_transactions(db_path: str = DUCKDB_PATH) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    pit = ", ".join(f"f.{c}" for c in PIT_COLS)
    df = con.execute(f"""
        SELECT
            f.transaction_id, f.timestamp, d.year_month, sa.account_id AS sender_account,
            ra.account_id AS receiver_account, f.amount, f.is_structuring, f.hour_of_day,
            d.day_of_week, d.is_weekend, l.is_cross_border, l.is_high_risk_corridor, l.corridor,
            cu.is_fx_conversion, c.channel_name, {pit},
            f.is_suspicious, f.laundering_typology
        FROM Fact_Transactions f
        JOIN Dim_Date d ON f.date_key = d.date_key
        JOIN Dim_Account sa ON f.sender_account_key = sa.account_key
        JOIN Dim_Account ra ON f.receiver_account_key = ra.account_key
        JOIN Dim_Location l ON f.location_key = l.location_key
        JOIN Dim_Currency cu ON f.currency_key = cu.currency_key
        JOIN Dim_Channel c ON f.channel_key = c.channel_key
    """).df()
    con.close()
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw transaction + point-in-time history columns into the model matrix (FEATURE_COLS order)."""
    X = pd.DataFrame(index=df.index)
    X["amount_log"] = np.log1p(df["amount"])
    X["amount_vs_sender_avg"] = (df["amount"] / df["s_prior_out_avg"].replace(0, np.nan)).fillna(1.0).clip(0, 50)
    for col in ["is_structuring", "hour_of_day", "day_of_week", "is_weekend", "is_cross_border",
                "is_high_risk_corridor", "is_fx_conversion", "s_prior_cash_ratio",
                "s_prior_cross_border_ratio", "s_prior_high_risk_ratio"]:
        X[col] = df[col].astype(float)
    for col in ["s_prior_out_count", "s_prior_in_count", "s_prior_out_avg", "s_prior_distinct_receivers",
                "r_prior_in_count", "r_prior_out_count", "r_prior_distinct_senders"]:
        X[f"{col}_log"] = np.log1p(df[col].astype(float))
    secs = df["s_secs_since_last_out"].astype(float)
    X["s_is_first_out"] = (secs < 0).astype(float)
    X["s_hours_since_last_out_log"] = np.log1p(secs.clip(lower=0) / 3600.0)
    for ch in CHANNELS:
        X[f"channel_{ch}"] = (df["channel_name"] == ch).astype(float)
    return X[FEATURE_COLS]
