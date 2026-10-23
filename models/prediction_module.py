"""
AML Prediction Module: Predict a calibrated AML Risk Score (0 - 100) for future transactions (SAML-D)

Time-based evaluation: the model is trained on earlier months and scores later, unseen months,
the way a deployed monitoring system would. Probabilities are isotonic-calibrated and scaled to 0-100.
Note: normal transactions were down-sampled (see config.SAMPLE_NORMAL_TX), so scores reflect the
sampled base rate rather than the 0.1% population rate.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

sys.path.insert(0, os.path.abspath("."))
from models.features import load_transactions, build_features, FEATURE_COLS, CHANNELS
from config import RANDOM_SEED, STRUCTURING_LOW, STRUCTURING_HIGH, HIGH_RISK_LOCATIONS, CUR, format_money

TEST_FROM_MONTH = "2023-06"
HIGH_RISK_SCORE = 70
MEDIUM_RISK_SCORE = 30


def risk_tier(score: float):
    if score >= HIGH_RISK_SCORE:
        return f"HIGH RISK (File {CUR['report_name']})", "#DC2626"
    if score >= MEDIUM_RISK_SCORE:
        return "MEDIUM RISK (Enhanced Due Diligence)", "#D97706"
    return "LOW RISK (Straight-Through Processing)", "#059669"


def train_risk_predictor():
    print("=" * 60)
    print("Training AML Risk Score Predictor (time-based forecast)...")
    print("=" * 60)

    os.makedirs("models/saved", exist_ok=True)
    df = load_transactions()
    X = build_features(df)
    y = df["is_suspicious"].astype(int)

    train_mask = df["year_month"] < TEST_FROM_MONTH
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]

    base = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=RANDOM_SEED)
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    scores = np.round(proba * 100, 1)

    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="uniform")

    test_frame = df.loc[~train_mask, ["transaction_id", "timestamp", "year_month", "sender_account",
                                      "receiver_account", "channel_name", "corridor", "amount",
                                      "laundering_typology"]].copy()
    test_frame["y_true"] = y_test.values
    test_frame["risk_score"] = scores

    metrics = {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "brier": brier_score_loss(y_test, proba),
        "train_months": sorted(df.loc[train_mask, "year_month"].unique().tolist()),
        "test_months": sorted(df.loc[~train_mask, "year_month"].unique().tolist()),
        "train_size": int(train_mask.sum()),
        "test_size": int((~train_mask).sum()),
        "base_rate_test": float(y_test.mean()),
    }

    artifacts = {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "metrics": metrics,
        "calibration": pd.DataFrame({"mean_predicted": mean_pred, "fraction_positive": frac_pos}),
        "test_frame": test_frame,
        # Real test-period transactions used to pre-fill the dashboard simulator
        "example_pool": pd.concat([
            df.loc[~train_mask & (y == 1)],
            df.loc[~train_mask & (y == 0)].sample(n=5000, random_state=RANDOM_SEED),
        ]).reset_index(drop=True),
    }
    joblib.dump(artifacts, "models/saved/risk_predictor.joblib", compress=3)
    print(f"Risk predictor saved! Test months {metrics['test_months'][0]}..{metrics['test_months'][-1]}: "
          f"ROC-AUC {metrics['roc_auc']:.4f}, PR-AUC {metrics['pr_auc']:.4f}, Brier {metrics['brier']:.4f}")
    return artifacts


_MODEL_CACHE = {}


def _load_model():
    if "art" not in _MODEL_CACHE:
        _MODEL_CACHE["art"] = joblib.load("models/saved/risk_predictor.joblib")
    return _MODEL_CACHE["art"]


def explain_transaction_risk(tx: dict):
    """
    Score a single transaction. `tx` holds the raw inputs used by build_features:
    amount, channel_name, hour_of_day, day_of_week, is_cross_border, is_high_risk_corridor,
    is_fx_conversion and the sender/receiver point-in-time history (s_prior_*, r_prior_*).
    """
    art = _load_model()
    row = dict(tx)
    row["is_structuring"] = int(STRUCTURING_LOW <= row["amount"] < STRUCTURING_HIGH)
    row["is_weekend"] = int(row["day_of_week"] in (6, 7))
    X_input = build_features(pd.DataFrame([row]))
    score = float(art["model"].predict_proba(X_input)[0, 1] * 100)
    tier, color = risk_tier(score)

    factors = []
    if row["is_structuring"]:
        factors.append(f"Amount {format_money(row['amount'])} sits just below the "
                       f"{format_money(STRUCTURING_HIGH)} reporting threshold (structuring band).")
    if row["r_prior_distinct_senders"] >= 5:
        factors.append(f"Receiver has collected funds from {int(row['r_prior_distinct_senders'])} distinct senders (fan-in).")
    if row["r_prior_out_count"] >= 50:
        factors.append(f"Receiver is itself a busy pass-through account ({int(row['r_prior_out_count'])} past payments out).")
    if row["is_high_risk_corridor"]:
        factors.append(f"Corridor touches an FATF grey-list location ({', '.join(sorted(HIGH_RISK_LOCATIONS))}).")
    if row["is_fx_conversion"]:
        factors.append("Payment involves a currency conversion.")
    if row["channel_name"] in ("Cash Deposit", "Cash Withdrawal"):
        factors.append(f"Cash channel: {row['channel_name']}.")
    if row["s_secs_since_last_out"] >= 0 and row["s_secs_since_last_out"] < 3600:
        factors.append("Sender made another payment less than an hour ago (velocity).")
    if row["s_prior_out_count"] == 0:
        factors.append("First-ever outgoing payment from this sender.")
    if not factors:
        factors.append("No rule-based red flags; score driven by learned behavioural patterns.")

    return {
        "predicted_score": round(score, 1),
        "risk_tier": tier,
        "badge_color": color,
        "key_factors": factors,
    }


if __name__ == "__main__":
    train_risk_predictor()
