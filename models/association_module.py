"""
AML Association Rule Mining: Suspicious Transaction Pattern Rules (SAML-D)
Each transaction becomes an itemset of discretised attributes (channel, amount band, route, timing,
point-in-time sender/receiver behaviour). FP-Growth finds frequent itemsets; we keep rules whose
consequent is the suspicious outcome or a laundering typology.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

sys.path.insert(0, os.path.abspath("."))
from models.features import load_transactions
from config import STRUCTURING_LOW, STRUCTURING_HIGH, format_money

OUTCOME_ITEM = "Outcome=SUSPICIOUS"


def build_itemsets(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot (boolean) item matrix, one row per transaction."""
    items = {}
    for ch in sorted(df["channel_name"].unique()):
        items[f"Channel={ch}"] = df["channel_name"] == ch

    amt = df["amount"]
    thr = STRUCTURING_HIGH
    items[f"Amount=Small (< {format_money(0.1 * thr)})"] = amt < 0.1 * thr
    items[f"Amount=Medium ({format_money(0.1 * thr)} - {format_money(STRUCTURING_LOW)})"] = (amt >= 0.1 * thr) & (amt < STRUCTURING_LOW)
    items[f"Amount=Structuring band ({format_money(STRUCTURING_LOW)} - {format_money(thr)})"] = (amt >= STRUCTURING_LOW) & (amt < thr)
    items[f"Amount=Large ({format_money(thr)} - {format_money(5 * thr)})"] = (amt >= thr) & (amt < 5 * thr)
    items[f"Amount=Very large (> {format_money(5 * thr)})"] = amt >= 5 * thr

    items["Route=Cross-border"] = df["is_cross_border"] == 1
    items["Route=Domestic"] = df["is_cross_border"] == 0
    items["Corridor=High-risk location"] = df["is_high_risk_corridor"] == 1
    items["FX=Currency conversion"] = df["is_fx_conversion"] == 1
    items["Time=Night (00-06h)"] = df["hour_of_day"] < 6
    items["Day=Weekend"] = df["is_weekend"] == 1

    # Point-in-time behaviour of the counterparties (known at the moment of the transaction)
    items["Sender=First-ever payment"] = df["s_prior_out_count"] == 0
    items["Sender=Paid again within 1h"] = (df["s_secs_since_last_out"] >= 0) & (df["s_secs_since_last_out"] < 3600)
    items["Sender=Mostly cash history"] = df["s_prior_cash_ratio"] >= 0.5
    items["Receiver=Fan-in hub (5+ senders)"] = df["r_prior_distinct_senders"] >= 5
    items["Receiver=Pass-through (50+ payments out)"] = df["r_prior_out_count"] >= 50
    items["Receiver=New account (no history)"] = (df["r_prior_in_count"] == 0) & (df["r_prior_out_count"] == 0)

    items[OUTCOME_ITEM] = df["is_suspicious"] == 1
    for typ in sorted(df.loc[df["is_suspicious"] == 1, "laundering_typology"].unique()):
        items[f"Typology={typ}"] = df["laundering_typology"] == typ

    return pd.DataFrame(items).astype(bool)


def mine_association_rules(min_support: float = 0.0005, min_confidence: float = 0.20, max_len: int = 4):
    print("=" * 60)
    print("Mining AML Suspicious Pattern Association Rules (FP-Growth)...")
    print("=" * 60)

    os.makedirs("models/saved", exist_ok=True)
    df = load_transactions()
    basket = build_itemsets(df)

    frequent_itemsets = fpgrowth(basket, min_support=min_support, use_colnames=True, max_len=max_len)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    def mentions_label(items):
        return any(x == OUTCOME_ITEM or x.startswith("Typology=") for x in items)

    # Keep rules that predict suspicion from observable attributes only (no labels in the antecedent)
    suspicious_rules = rules[rules["consequents"].apply(mentions_label) & ~rules["antecedents"].apply(mentions_label)]
    suspicious_rules = suspicious_rules.sort_values(by=["lift", "confidence"], ascending=False)

    rules_display = suspicious_rules.copy()
    rules_display["antecedents_str"] = rules_display["antecedents"].apply(lambda s: " AND ".join(sorted(s)))
    rules_display["consequents_str"] = rules_display["consequents"].apply(lambda s: " AND ".join(sorted(s)))
    rules_display["rule_length"] = rules_display["antecedents"].apply(len) + rules_display["consequents"].apply(len)

    artifacts = {
        "rules_df": rules_display,
        "all_rules": rules,
        "frequent_itemsets": frequent_itemsets,
        "item_support": basket.mean().sort_values(ascending=False),
        "params": {"min_support": min_support, "min_confidence": min_confidence, "max_len": max_len},
    }

    joblib.dump(artifacts, "models/saved/association_rules.joblib", compress=3)
    rules_display.drop(columns=["antecedents", "consequents"]).to_csv(
        "data/processed/suspicious_association_rules.csv", index=False)
    print(f"Association rules saved! {len(frequent_itemsets):,} frequent itemsets, "
          f"{len(suspicious_rules):,} suspicious-outcome rules.")
    return artifacts


if __name__ == "__main__":
    mine_association_rules()
