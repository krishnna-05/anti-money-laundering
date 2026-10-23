"""
AML Classification Module: Classify Suspicious vs Normal Transactions (SAML-D)
Trains and compares several classifiers on the same hold-out split (grouped by sender account).
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix, precision_score, recall_score, f1_score,
)

sys.path.insert(0, os.path.abspath("."))
from models.features import load_transactions, build_features, FEATURE_COLS, FEATURE_LABELS
from config import RANDOM_SEED


def get_classifiers():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=14, min_samples_leaf=3, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31, class_weight="balanced", random_state=RANDOM_SEED),
    }


def train_classifier():
    print("=" * 60)
    print("Training AML Transaction Classifiers (SAML-D)...")
    print("=" * 60)

    os.makedirs("models/saved", exist_ok=True)
    df = load_transactions()
    X = build_features(df)
    y = df["is_suspicious"].astype(int)

    # Split by sender account so test transactions come from accounts never seen in training
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED)
    pos_train, pos_test = next(splitter.split(X, y, groups=df["sender_account"]))
    idx_train, idx_test = df.index[pos_train], df.index[pos_test]
    X_train, X_test = X.loc[idx_train], X.loc[idx_test]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]

    results, probas, fitted = {}, {}, {}
    for name, model in get_classifiers().items():
        start = time.time()
        model.fit(X_train, y_train)
        train_secs = time.time() - start
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        results[name] = {
            "roc_auc": roc_auc_score(y_test, y_prob),
            "pr_auc": average_precision_score(y_test, y_prob),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "train_seconds": train_secs,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }
        probas[name] = y_prob
        fitted[name] = model
        print(f"  {name:22s} ROC-AUC {results[name]['roc_auc']:.4f}  PR-AUC {results[name]['pr_auc']:.4f}  "
              f"F1 {results[name]['f1']:.4f}  ({train_secs:.1f}s)")

    best_name = max(results, key=lambda n: results[n]["pr_auc"])
    best_model = fitted[best_name]

    # Model-agnostic importance for the best model on a stratified subsample of the test set
    sub_idx, _ = train_test_split(X_test.index, train_size=min(20000, len(X_test) - 1),
                                  random_state=RANDOM_SEED, stratify=y_test)
    perm = permutation_importance(best_model, X_test.loc[sub_idx], y_test.loc[sub_idx],
                                  scoring="average_precision", n_repeats=3, random_state=RANDOM_SEED, n_jobs=-1)
    feature_importances = pd.DataFrame({
        "Feature": [FEATURE_LABELS.get(c, c) for c in FEATURE_COLS],
        "Importance": perm.importances_mean,
    }).sort_values(by="Importance", ascending=False)

    test_frame = df.loc[idx_test, ["transaction_id", "laundering_typology", "channel_name", "amount"]].copy()
    test_frame["y_true"] = y_test.values
    for name, p in probas.items():
        test_frame[f"proba_{name}"] = p

    artifacts = {
        "model": best_model,
        "model_name": best_name,
        "feature_cols": FEATURE_COLS,
        "feature_importances": feature_importances,
        "metrics": results,
        "test_frame": test_frame,
        "train_size": len(idx_train),
        "test_size": len(idx_test),
    }
    joblib.dump(artifacts, "models/saved/classifier.joblib", compress=3)
    print(f"Classifier saved! Best: {best_name}, PR-AUC: {results[best_name]['pr_auc']:.4f}")
    return artifacts


if __name__ == "__main__":
    train_classifier()
