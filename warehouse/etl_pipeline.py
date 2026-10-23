"""
AML Data Warehouse ETL Pipeline (SAML-D)
Extracts raw transaction and account data, creates dimensions and facts, and loads into DuckDB & SQLite.
"""

import os
import sys
import duckdb
import sqlite3
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from config import (
    CUR, HIGH_RISK_LOCATIONS, CASH_CHANNELS, ELECTRONIC_CHANNELS, STRUCTURING_LOW, STRUCTURING_HIGH,
    RAW_TX_PATH, RAW_ACC_PATH, DUCKDB_PATH, SQLITE_PATH,
)

# Point-in-time behaviour measures (sender/receiver history strictly before each transaction)
PIT_COLS = [
    "s_prior_out_count", "s_prior_in_count", "s_prior_out_avg", "s_prior_distinct_receivers",
    "s_prior_cash_ratio", "s_prior_cross_border_ratio", "s_prior_high_risk_ratio", "s_secs_since_last_out",
    "r_prior_in_count", "r_prior_out_count", "r_prior_distinct_senders",
]

CHANNEL_CATEGORY = {
    "Credit card": "Card Payment",
    "Debit card": "Card Payment",
    "Cheque": "Paper Instrument",
    "ACH": "Domestic Bank Transfer",
    "Cross-border": "International Wire",
    "Cash Deposit": "Cash Inflow",
    "Cash Withdrawal": "Cash Outflow",
}


def run_etl(
    raw_tx_path: str = RAW_TX_PATH,
    raw_acc_path: str = RAW_ACC_PATH,
    duckdb_path: str = DUCKDB_PATH,
    sqlite_path: str = SQLITE_PATH
):
    print("=" * 60)
    print(f"Starting AML Data Warehouse ETL Pipeline ({CUR['code']})...")
    print("=" * 60)

    os.makedirs(os.path.dirname(duckdb_path), exist_ok=True)

    # 1. EXTRACT
    df_tx = pd.read_csv(raw_tx_path, dtype={"sender_account": str, "receiver_account": str})
    df_acc = pd.read_csv(raw_acc_path, dtype={"account_id": str})

    # 2. TRANSFORM
    # 2.1 Dim_Date
    df_tx["timestamp_dt"] = pd.to_datetime(df_tx["timestamp"])
    unique_dates = df_tx["timestamp_dt"].dt.normalize().drop_duplicates().sort_values()

    dim_date = pd.DataFrame({"full_date": unique_dates.values})
    dim_date["date_key"] = dim_date["full_date"].dt.strftime("%Y%m%d").astype(int)
    dim_date["day_of_week"] = dim_date["full_date"].dt.dayofweek + 1
    dim_date["day_name"] = dim_date["full_date"].dt.day_name()
    dim_date["day_of_month"] = dim_date["full_date"].dt.day
    dim_date["month_number"] = dim_date["full_date"].dt.month
    dim_date["month_name"] = dim_date["full_date"].dt.month_name()
    dim_date["year_month"] = dim_date["full_date"].dt.strftime("%Y-%m")
    dim_date["quarter_number"] = dim_date["full_date"].dt.quarter
    dim_date["quarter_name"] = dim_date["full_date"].dt.year.astype(str) + "-Q" + dim_date["quarter_number"].astype(str)
    dim_date["year_number"] = dim_date["full_date"].dt.year
    dim_date["is_weekend"] = dim_date["day_of_week"].isin([6, 7]).astype(int)

    # 2.2 Dim_Account (behaviour profile computed by the loader over the full dataset)
    dim_account = df_acc.copy().reset_index(drop=True)
    dim_account["account_key"] = dim_account.index + 1

    # 2.3 Dim_Channel
    dim_channel = pd.DataFrame({"channel_name": sorted(df_tx["payment_channel"].unique())})
    dim_channel["channel_key"] = dim_channel.index + 1
    dim_channel["channel_category"] = dim_channel["channel_name"].map(CHANNEL_CATEGORY).fillna("Other")
    dim_channel["is_electronic"] = dim_channel["channel_name"].isin(ELECTRONIC_CHANNELS).astype(int)
    dim_channel["is_cash"] = dim_channel["channel_name"].isin(CASH_CHANNELS).astype(int)

    # 2.4 Dim_Location (sender bank location -> receiver bank location corridor)
    locations = df_tx[["sender_location", "receiver_location"]].drop_duplicates().sort_values(
        ["sender_location", "receiver_location"]).reset_index(drop=True)
    locations["location_key"] = locations.index + 1
    locations["corridor"] = locations["sender_location"] + " ➔ " + locations["receiver_location"]
    locations["is_cross_border"] = (locations["sender_location"] != locations["receiver_location"]).astype(int)
    locations["is_high_risk_corridor"] = (
        locations["sender_location"].isin(HIGH_RISK_LOCATIONS) | locations["receiver_location"].isin(HIGH_RISK_LOCATIONS)
    ).astype(int)
    dim_location = locations[["location_key", "sender_location", "receiver_location", "corridor",
                              "is_cross_border", "is_high_risk_corridor"]]

    # 2.5 Dim_Currency (payment currency -> received currency pair)
    currencies = df_tx[["payment_currency", "received_currency"]].drop_duplicates().sort_values(
        ["payment_currency", "received_currency"]).reset_index(drop=True)
    currencies["currency_key"] = currencies.index + 1
    currencies["currency_pair"] = currencies["payment_currency"] + " ➔ " + currencies["received_currency"]
    currencies["is_fx_conversion"] = (currencies["payment_currency"] != currencies["received_currency"]).astype(int)
    dim_currency = currencies[["currency_key", "payment_currency", "received_currency", "currency_pair", "is_fx_conversion"]]

    # 2.6 Fact_Transactions
    account_key_map = dim_account.set_index("account_id")["account_key"]
    channel_key_map = dim_channel.set_index("channel_name")["channel_key"]
    location_key_map = dim_location.set_index(["sender_location", "receiver_location"])["location_key"]
    currency_key_map = dim_currency.set_index(["payment_currency", "received_currency"])["currency_key"]

    df_fact = df_tx.copy()
    df_fact["date_key"] = df_fact["timestamp_dt"].dt.strftime("%Y%m%d").astype(int)
    df_fact["hour_of_day"] = df_fact["timestamp_dt"].dt.hour
    df_fact["sender_account_key"] = df_fact["sender_account"].map(account_key_map)
    df_fact["receiver_account_key"] = df_fact["receiver_account"].map(account_key_map)
    df_fact["channel_key"] = df_fact["payment_channel"].map(channel_key_map)
    df_fact["location_key"] = location_key_map.reindex(
        pd.MultiIndex.from_frame(df_fact[["sender_location", "receiver_location"]])).values
    df_fact["currency_key"] = currency_key_map.reindex(
        pd.MultiIndex.from_frame(df_fact[["payment_currency", "received_currency"]])).values

    df_fact["transaction_key"] = df_fact.index + 1
    df_fact["timestamp"] = df_fact["timestamp_dt"]
    df_fact["amount_log"] = np.log1p(df_fact["amount"])
    # Structuring band: just below the reference reporting threshold
    df_fact["is_structuring"] = ((df_fact["amount"] >= STRUCTURING_LOW) & (df_fact["amount"] < STRUCTURING_HIGH)).astype(int)

    fact_transactions = df_fact[[
        "transaction_key", "transaction_id", "date_key", "sender_account_key", "receiver_account_key",
        "channel_key", "location_key", "currency_key", "timestamp", "hour_of_day", "amount", "amount_log",
        "is_structuring", "is_suspicious", "laundering_typology", "sar_flag"
    ] + PIT_COLS]

    # 3. LOAD into DuckDB and SQLite
    con_duck = duckdb.connect(duckdb_path)
    for name, frame in [("Dim_Date", dim_date), ("Dim_Account", dim_account), ("Dim_Channel", dim_channel),
                        ("Dim_Location", dim_location), ("Dim_Currency", dim_currency),
                        ("Fact_Transactions", fact_transactions)]:
        con_duck.register("frame_view", frame)
        con_duck.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM frame_view")
        con_duck.unregister("frame_view")
    con_duck.close()

    con_sqlite = sqlite3.connect(sqlite_path)
    dim_date.to_sql("Dim_Date", con_sqlite, if_exists="replace", index=False)
    dim_account.to_sql("Dim_Account", con_sqlite, if_exists="replace", index=False)
    dim_channel.to_sql("Dim_Channel", con_sqlite, if_exists="replace", index=False)
    dim_location.to_sql("Dim_Location", con_sqlite, if_exists="replace", index=False)
    dim_currency.to_sql("Dim_Currency", con_sqlite, if_exists="replace", index=False)
    fact_transactions.to_sql("Fact_Transactions", con_sqlite, if_exists="replace", index=False)
    con_sqlite.close()

    print(f"ETL completed! {len(fact_transactions):,} transactions, {len(dim_account):,} accounts, "
          f"{len(dim_location)} corridors, {len(dim_currency)} currency pairs loaded into DW.")


if __name__ == "__main__":
    run_etl()
