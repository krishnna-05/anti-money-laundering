"""
Central configuration: currency, regulatory framing, and dataset settings.

The source dataset (SAML-D) is denominated in GBP. To present the whole platform in
another currency, change ACTIVE_PROFILE (or set the AML_CURRENCY env var) and re-run
`python demo.py` — the loader converts amounts once and every module reads from here.
"""

import os

PROFILES = {
    "GBP": {
        "code": "GBP",
        "symbol": "£",
        "rate_from_gbp": 1.0,
        "number_style": "western",           # k / M
        "reference_threshold": 10_000.0,     # UK £10k high-value cash reference point
        "report_name": "SAR",
        "report_long": "Suspicious Activity Report",
        "authority": "UK Financial Intelligence Unit (National Crime Agency)",
        "law": "Proceeds of Crime Act 2002 (POCA), Part 7",
        "jurisdiction_label": "United Kingdom",
    },
    "INR": {
        "code": "INR",
        "symbol": "₹",
        "rate_from_gbp": 100.0,              # fixed illustrative rate; amounts converted from GBP
        "number_style": "indian",            # Lakh / Crore
        "reference_threshold": 1_000_000.0,  # PMLA ₹10 Lakh reporting threshold
        "report_name": "STR",
        "report_long": "Suspicious Transaction Report",
        "authority": "Financial Intelligence Unit - India (FIU-IND)",
        "law": "Prevention of Money Laundering Act 2002 (PMLA), Section 12",
        "jurisdiction_label": "India",
    },
}

ACTIVE_PROFILE = os.environ.get("AML_CURRENCY", "GBP")
CUR = PROFILES[ACTIVE_PROFILE]

# Structuring band: amounts just below the reference threshold (88% - 100%)
STRUCTURING_LOW = 0.88 * CUR["reference_threshold"]
STRUCTURING_HIGH = CUR["reference_threshold"]

# Dataset: SAML-D (Oztas et al., IEEE ICEBE 2023) on Kaggle
KAGGLE_DATASET = "berkanoztas/synthetic-transaction-monitoring-dataset-aml"
# All laundering transactions are kept; normal transactions are randomly sampled
# so the warehouse and dashboard stay responsive.
SAMPLE_NORMAL_TX = 300_000
RANDOM_SEED = 42

# Bank locations on the FATF "increased monitoring" (grey) list during the dataset
# period (Oct 2022 - Aug 2023).
HIGH_RISK_LOCATIONS = {"UAE", "Turkey", "Nigeria", "Morocco", "Albania"}

CASH_CHANNELS = {"Cash Deposit", "Cash Withdrawal"}
ELECTRONIC_CHANNELS = {"Credit card", "Debit card", "ACH", "Cross-border"}

DUCKDB_PATH = "data/processed/aml_warehouse.duckdb"
SQLITE_PATH = "data/processed/aml_warehouse.db"
RAW_TX_PATH = "data/raw/aml_transactions.csv"
RAW_ACC_PATH = "data/raw/accounts_metadata.csv"


def format_money(val: float) -> str:
    sym = CUR["symbol"]
    if val is None:
        return "-"
    a = abs(val)
    if CUR["number_style"] == "indian":
        if a >= 1e7:
            return f"{sym}{val / 1e7:.2f} Cr"
        if a >= 1e5:
            return f"{sym}{val / 1e5:.2f} L"
        return f"{sym}{val:,.0f}"
    if a >= 1e9:
        return f"{sym}{val / 1e9:.2f}B"
    if a >= 1e6:
        return f"{sym}{val / 1e6:.2f}M"
    if a >= 1e3:
        return f"{sym}{val / 1e3:.1f}k"
    return f"{sym}{val:,.2f}"


def money_unit():
    """(divisor, label) used for chart axes, e.g. (1e6, '£ M') or (1e7, '₹ Cr')."""
    if CUR["number_style"] == "indian":
        return 1e7, f"{CUR['symbol']} Cr"
    return 1e6, f"{CUR['symbol']} M"
