"""
SAML-D Dataset Loader (Kaggle: berkanoztas/synthetic-transaction-monitoring-dataset-aml)

SAML-D: 9.5M transactions, 28 typologies (11 normal, 17 suspicious), 7 payment types,
18 bank locations, 13 currencies. Oztas et al., "Enhancing Anti-Money Laundering:
Development of a Synthetic Transaction Monitoring Dataset", IEEE ICEBE 2023.

Steps:
1. Download the CSV with kagglehub (public dataset, no API key needed).
2. Build account behaviour profiles over the FULL 9.5M rows.
3. Keep every laundering transaction plus a random sample of normal transactions.
4. Convert amounts into the active currency (config.CUR) and write the raw CSVs
   consumed by the warehouse ETL.
"""

import os
import sys
import argparse
import duckdb

sys.path.insert(0, os.path.abspath("."))
from config import (
    CUR, KAGGLE_DATASET, SAMPLE_NORMAL_TX, RANDOM_SEED, HIGH_RISK_LOCATIONS,
    CASH_CHANNELS, STRUCTURING_LOW, STRUCTURING_HIGH, RAW_TX_PATH, RAW_ACC_PATH,
)


def locate_csv(csv_path: str = None) -> str:
    if csv_path:
        return csv_path
    import kagglehub
    folder = kagglehub.dataset_download(KAGGLE_DATASET)
    return os.path.join(folder, "SAML-D.csv")


def _sql_list(values) -> str:
    return ", ".join(f"'{v}'" for v in sorted(values))


def load_saml_d(csv_path: str = None, sample_normal: int = SAMPLE_NORMAL_TX, seed: int = RANDOM_SEED):
    print("=" * 60)
    print(f"Loading SAML-D dataset (amounts in {CUR['code']}, rate {CUR['rate_from_gbp']} per GBP)...")
    print("=" * 60)

    csv_path = locate_csv(csv_path)
    os.makedirs(os.path.dirname(RAW_TX_PATH), exist_ok=True)

    rate = CUR["rate_from_gbp"]
    high_risk = _sql_list(HIGH_RISK_LOCATIONS)
    cash = _sql_list(CASH_CHANNELS)

    con = duckdb.connect()
    con.execute(f"""
        CREATE TABLE raw AS
        SELECT row_number() OVER () AS rid, *, Amount * {rate} AS amt
        FROM read_csv_auto('{csv_path}')
    """)
    total, laundering = con.execute("SELECT COUNT(*), SUM(Is_laundering) FROM raw").fetchone()
    print(f"Full dataset: {total:,} transactions, {int(laundering):,} laundering ({laundering / total * 100:.3f}%)")

    # Account behaviour profiles over the full dataset (no sampling bias)
    con.execute(f"""
        CREATE TABLE out_prof AS
        SELECT
            Sender_account AS account_id,
            COUNT(*) AS out_tx_count,
            SUM(amt) AS out_total,
            AVG(amt) AS out_avg,
            COALESCE(STDDEV_SAMP(amt), 0) AS out_std,
            MAX(amt) AS out_max,
            COUNT(DISTINCT Receiver_account) AS distinct_receivers,
            AVG(CASE WHEN Payment_type IN ({cash}) THEN 1 ELSE 0 END) AS cash_ratio,
            AVG(CASE WHEN Sender_bank_location <> Receiver_bank_location THEN 1 ELSE 0 END) AS cross_border_ratio,
            AVG(CASE WHEN Sender_bank_location IN ({high_risk}) OR Receiver_bank_location IN ({high_risk})
                     THEN 1 ELSE 0 END) AS high_risk_ratio,
            AVG(CASE WHEN Payment_currency <> Received_currency THEN 1 ELSE 0 END) AS fx_ratio,
            AVG(CASE WHEN amt >= {STRUCTURING_LOW} AND amt < {STRUCTURING_HIGH} THEN 1 ELSE 0 END) AS structuring_ratio,
            COUNT(DISTINCT Date) AS active_days,
            MIN(Date) AS first_seen,
            MAX(Date) AS last_seen,
            MODE(Sender_bank_location) AS out_location,
            MAX(Is_laundering) AS out_suspicious
        FROM raw GROUP BY Sender_account
    """)
    con.execute("""
        CREATE TABLE in_prof AS
        SELECT
            Receiver_account AS account_id,
            COUNT(*) AS in_tx_count,
            SUM(amt) AS in_total,
            COUNT(DISTINCT Sender_account) AS distinct_senders,
            MODE(Receiver_bank_location) AS in_location,
            MAX(Is_laundering) AS in_suspicious
        FROM raw GROUP BY Receiver_account
    """)

    con.execute(f"""
        CREATE TABLE sampled AS
        SELECT * FROM raw WHERE Is_laundering = 1
        UNION ALL
        SELECT * FROM (SELECT * FROM raw WHERE Is_laundering = 0)
            USING SAMPLE reservoir({int(sample_normal)} ROWS) REPEATABLE ({int(seed)})
    """)

    # Point-in-time behaviour: for every sampled transaction, the sender's and receiver's
    # history strictly BEFORE that transaction (computed over the full dataset, so no
    # future information leaks into the features).
    con.execute("""
        CREATE TABLE events AS
        WITH accts AS (
            SELECT Sender_account AS acct FROM sampled UNION SELECT Receiver_account FROM sampled
        ),
        base AS (
            SELECT rid, CAST(Date AS DATE) + CAST(Time AS TIME) AS ts, Sender_account, Receiver_account, amt,
                   Payment_type, Sender_bank_location, Receiver_bank_location
            FROM raw
            WHERE Sender_account IN (SELECT acct FROM accts) OR Receiver_account IN (SELECT acct FROM accts)
        )
        SELECT rid, ts, Sender_account AS acct, 1 AS is_out, Receiver_account AS cp, amt,
               Payment_type, Sender_bank_location AS s_loc, Receiver_bank_location AS r_loc
        FROM base WHERE Sender_account IN (SELECT acct FROM accts)
        UNION ALL
        SELECT rid, ts, Receiver_account, 0, Sender_account, amt,
               Payment_type, Sender_bank_location, Receiver_bank_location
        FROM base WHERE Receiver_account IN (SELECT acct FROM accts)
    """)
    con.execute(f"""
        CREATE TABLE pit AS
        SELECT rid, is_out,
            SUM(is_out) OVER w AS prior_out_count,
            SUM(1 - is_out) OVER w AS prior_in_count,
            SUM(CASE WHEN is_out = 1 THEN amt END) OVER w AS prior_out_amount,
            COUNT(DISTINCT CASE WHEN is_out = 1 THEN cp END) OVER w AS prior_distinct_receivers,
            COUNT(DISTINCT CASE WHEN is_out = 0 THEN cp END) OVER w AS prior_distinct_senders,
            SUM(CASE WHEN is_out = 1 AND Payment_type IN ({cash}) THEN 1 ELSE 0 END) OVER w AS prior_cash_out,
            SUM(CASE WHEN is_out = 1 AND s_loc <> r_loc THEN 1 ELSE 0 END) OVER w AS prior_cross_border_out,
            SUM(CASE WHEN is_out = 1 AND (s_loc IN ({high_risk}) OR r_loc IN ({high_risk})) THEN 1 ELSE 0 END)
                OVER w AS prior_high_risk_out,
            MAX(CASE WHEN is_out = 1 THEN ts END) OVER w AS last_out_ts,
            ts
        FROM events
        WINDOW w AS (PARTITION BY acct ORDER BY ts, rid, is_out ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        QUALIFY rid IN (SELECT rid FROM sampled)
    """)

    con.execute(f"""
        COPY (
            SELECT
                'TXN' || lpad(CAST(sampled.rid AS VARCHAR), 8, '0') AS transaction_id,
                CAST(Date AS DATE) + CAST(Time AS TIME) AS timestamp,
                CAST(Sender_account AS VARCHAR) AS sender_account,
                CAST(Receiver_account AS VARCHAR) AS receiver_account,
                ROUND(amt, 2) AS amount,
                Payment_currency AS payment_currency,
                Received_currency AS received_currency,
                Payment_type AS payment_channel,
                Sender_bank_location AS sender_location,
                Receiver_bank_location AS receiver_location,
                CASE WHEN Sender_bank_location <> Receiver_bank_location THEN 1 ELSE 0 END AS is_cross_border,
                Is_laundering AS is_suspicious,
                replace(Laundering_type, '_', ' ') AS laundering_typology,
                Is_laundering AS sar_flag,
                COALESCE(so.prior_out_count, 0) AS s_prior_out_count,
                COALESCE(so.prior_in_count, 0) AS s_prior_in_count,
                COALESCE(so.prior_out_amount / NULLIF(so.prior_out_count, 0), 0) AS s_prior_out_avg,
                so.prior_distinct_receivers AS s_prior_distinct_receivers,
                COALESCE(so.prior_cash_out / NULLIF(so.prior_out_count, 0), 0) AS s_prior_cash_ratio,
                COALESCE(so.prior_cross_border_out / NULLIF(so.prior_out_count, 0), 0) AS s_prior_cross_border_ratio,
                COALESCE(so.prior_high_risk_out / NULLIF(so.prior_out_count, 0), 0) AS s_prior_high_risk_ratio,
                COALESCE(date_diff('second', so.last_out_ts, so.ts), -1) AS s_secs_since_last_out,
                COALESCE(ri.prior_in_count, 0) AS r_prior_in_count,
                COALESCE(ri.prior_out_count, 0) AS r_prior_out_count,
                ri.prior_distinct_senders AS r_prior_distinct_senders
            FROM sampled
            JOIN pit so ON so.rid = sampled.rid AND so.is_out = 1
            JOIN pit ri ON ri.rid = sampled.rid AND ri.is_out = 0
            ORDER BY timestamp
        ) TO '{RAW_TX_PATH}' (HEADER, DELIMITER ',')
    """)

    con.execute(f"""
        COPY (
            WITH ids AS (
                SELECT Sender_account AS account_id FROM sampled
                UNION SELECT Receiver_account FROM sampled
            ),
            joined AS (
                SELECT
                    ids.account_id,
                    COALESCE(o.out_location, i.in_location) AS primary_location,
                    COALESCE(o.out_tx_count, 0) AS out_tx_count,
                    COALESCE(o.out_total, 0) AS out_total,
                    COALESCE(o.out_avg, 0) AS out_avg,
                    COALESCE(o.out_std, 0) AS out_std,
                    COALESCE(o.out_max, 0) AS out_max,
                    COALESCE(o.distinct_receivers, 0) AS distinct_receivers,
                    COALESCE(o.cash_ratio, 0) AS cash_ratio,
                    COALESCE(o.cross_border_ratio, 0) AS cross_border_ratio,
                    COALESCE(o.high_risk_ratio, 0) AS high_risk_ratio,
                    COALESCE(o.fx_ratio, 0) AS fx_ratio,
                    COALESCE(o.structuring_ratio, 0) AS structuring_ratio,
                    COALESCE(o.active_days, 0) AS active_days,
                    COALESCE(date_diff('day', o.first_seen, o.last_seen), 0) AS tenure_days,
                    COALESCE(i.in_tx_count, 0) AS in_tx_count,
                    COALESCE(i.in_total, 0) AS in_total,
                    COALESCE(i.distinct_senders, 0) AS distinct_senders,
                    GREATEST(COALESCE(o.out_suspicious, 0), COALESCE(i.in_suspicious, 0)) AS ever_suspicious
                FROM ids
                LEFT JOIN out_prof o USING (account_id)
                LEFT JOIN in_prof i USING (account_id)
            )
            SELECT
                CAST(account_id AS VARCHAR) AS account_id,
                primary_location,
                CASE
                    WHEN out_tx_count + in_tx_count < 10 THEN 'Occasional (<10 txns)'
                    WHEN out_tx_count + in_tx_count <= 30 THEN 'Regular (10-30 txns)'
                    WHEN out_tx_count + in_tx_count <= 200 THEN 'Active (31-200 txns)'
                    ELSE 'High-Volume (>200 txns)'
                END AS account_segment,
                CASE WHEN primary_location IN ({high_risk}) THEN 1 ELSE 0 END AS is_high_risk_location,
                * EXCLUDE (account_id, primary_location)
            FROM joined
        ) TO '{RAW_ACC_PATH}' (HEADER, DELIMITER ',')
    """)

    n_tx, n_sus = con.execute(f"SELECT COUNT(*), SUM(is_suspicious) FROM '{RAW_TX_PATH}'").fetchone()
    n_acc = con.execute(f"SELECT COUNT(*) FROM '{RAW_ACC_PATH}'").fetchone()[0]
    con.close()
    print(f"Sampled {n_tx:,} transactions ({int(n_sus):,} laundering, {n_sus / n_tx * 100:.2f}%) across {n_acc:,} accounts.")
    print(f"Saved: {RAW_TX_PATH}, {RAW_ACC_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load SAML-D into the raw data folder")
    parser.add_argument("--csv", help="Path to an already-downloaded SAML-D.csv (skips kagglehub)")
    parser.add_argument("--sample-normal", type=int, default=SAMPLE_NORMAL_TX)
    args = parser.parse_args()
    load_saml_d(args.csv, args.sample_normal)
