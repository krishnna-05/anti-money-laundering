"""
AML Clustering Module: Cluster Account Transaction-Behaviour Groups (SAML-D)

A small library used both by the batch pipeline and live by the dashboard:
- ALGORITHMS: seven clustering algorithms, each with a parameter spec (drives the dashboard widgets)
- run_clustering / evaluate_clustering: fit one algorithm and score it
- k_sweep / find_elbow: choose K for the K-based algorithms
- profile_clusters: describe each cluster by its distinguishing behaviour
Account profiles come from Dim_Account (aggregated over the full 9.5M-row dataset).
The ever_suspicious label is used only to evaluate clusters, never to fit them.
"""

import os
import sys
import time
import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, Birch, DBSCAN, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

sys.path.insert(0, os.path.abspath("."))
from config import DUCKDB_PATH, RANDOM_SEED

# (column, log-transform?, readable label)
CLUSTER_FEATURES = [
    ("out_tx_count", True, "Outgoing txn count"),
    ("out_avg", True, "Avg outgoing amount"),
    ("out_std", True, "Amount volatility"),
    ("out_max", True, "Largest payment"),
    ("distinct_receivers", True, "Fan-out (distinct receivers)"),
    ("in_tx_count", True, "Incoming txn count"),
    ("distinct_senders", True, "Fan-in (distinct senders)"),
    ("active_days", True, "Active days"),
    ("cash_ratio", False, "Cash share"),
    ("cross_border_ratio", False, "Cross-border share"),
    ("high_risk_ratio", False, "High-risk corridor share"),
    ("fx_ratio", False, "FX conversion share"),
    ("structuring_ratio", False, "Structuring-band share"),
]
FEATURE_NAMES = [c for c, _, _ in CLUSTER_FEATURES]
FEATURE_LABELS = {c: label for c, _, label in CLUSTER_FEATURES}

ALGORITHMS = {
    "K-Means": {
        "family": "Partitioning (centroid)",
        "uses_k": True,
        "params": {
            "n_clusters": {"type": "int", "min": 2, "max": 15, "default": 4, "label": "K (number of clusters)"},
            "init": {"type": "select", "options": ["k-means++", "random"], "default": "k-means++", "label": "Initialisation"},
            "n_init": {"type": "int", "min": 1, "max": 30, "default": 10, "label": "Number of restarts (n_init)"},
            "max_iter": {"type": "int", "min": 50, "max": 1000, "default": 300, "step": 50, "label": "Max iterations"},
        },
    },
    "Mini-Batch K-Means": {
        "family": "Partitioning (centroid, mini-batch)",
        "uses_k": True,
        "params": {
            "n_clusters": {"type": "int", "min": 2, "max": 15, "default": 4, "label": "K (number of clusters)"},
            "batch_size": {"type": "int", "min": 128, "max": 4096, "default": 1024, "step": 128, "label": "Batch size"},
            "n_init": {"type": "int", "min": 1, "max": 30, "default": 3, "label": "Number of restarts (n_init)"},
        },
    },
    "Agglomerative (Hierarchical)": {
        "family": "Hierarchical",
        "uses_k": True,
        "params": {
            "n_clusters": {"type": "int", "min": 2, "max": 15, "default": 4, "label": "Number of clusters"},
            "linkage": {"type": "select", "options": ["ward", "complete", "average", "single"], "default": "ward", "label": "Linkage"},
        },
    },
    "Gaussian Mixture (EM)": {
        "family": "Model-based (probabilistic)",
        "uses_k": True,
        "params": {
            "n_clusters": {"type": "int", "min": 2, "max": 15, "default": 4, "label": "Number of components"},
            "covariance_type": {"type": "select", "options": ["full", "tied", "diag", "spherical"], "default": "full", "label": "Covariance type"},
            "max_iter": {"type": "int", "min": 50, "max": 1000, "default": 200, "step": 50, "label": "Max EM iterations"},
        },
    },
    "BIRCH": {
        "family": "Hierarchical (CF-tree)",
        "uses_k": True,
        "params": {
            "n_clusters": {"type": "int", "min": 2, "max": 15, "default": 4, "label": "Number of clusters"},
            "threshold": {"type": "float", "min": 0.1, "max": 3.0, "default": 0.8, "step": 0.1, "label": "Sub-cluster radius threshold"},
            "branching_factor": {"type": "int", "min": 10, "max": 200, "default": 50, "step": 10, "label": "Branching factor"},
        },
    },
    "DBSCAN": {
        "family": "Density-based",
        "uses_k": False,
        "params": {
            "eps": {"type": "float", "min": 0.1, "max": 5.0, "default": 1.5, "step": 0.05, "label": "Neighbourhood radius (eps)"},
            "min_samples": {"type": "int", "min": 2, "max": 100, "default": 15, "label": "Min samples per core point"},
        },
    },
    "HDBSCAN": {
        "family": "Density-based (hierarchical)",
        "uses_k": False,
        "params": {
            "min_cluster_size": {"type": "int", "min": 5, "max": 500, "default": 40, "step": 5, "label": "Min cluster size"},
            "min_samples": {"type": "int", "min": 1, "max": 100, "default": 10, "label": "Min samples"},
        },
    },
}
K_ALGORITHMS = [name for name, spec in ALGORITHMS.items() if spec["uses_k"]]


def default_params(name: str) -> dict:
    return {p: spec["default"] for p, spec in ALGORITHMS[name]["params"].items()}


# --------------------------------------------------------------------------- data

def load_account_profiles(sample_size: int = 5000, seed: int = RANDOM_SEED, db_path: str = DUCKDB_PATH) -> pd.DataFrame:
    """Random sample of sending accounts with their full-period behaviour profile."""
    con = duckdb.connect(db_path, read_only=True)
    cols = ", ".join(FEATURE_NAMES)
    df = con.execute(f"""
        SELECT * FROM (
            SELECT account_id, primary_location, account_segment, out_total, {cols}, ever_suspicious
            FROM Dim_Account
            WHERE out_tx_count > 0
        ) senders
        USING SAMPLE reservoir({int(sample_size)} ROWS) REPEATABLE ({int(seed)})
    """).df()
    con.close()
    return df.sort_values("account_id").reset_index(drop=True)


def prepare_matrix(df: pd.DataFrame, features: list = None):
    """Log-transform skewed counts/amounts, then standardise. Returns (X_scaled, feature list)."""
    features = features or FEATURE_NAMES
    log_cols = {c for c, use_log, _ in CLUSTER_FEATURES if use_log}
    M = pd.DataFrame({c: np.log1p(df[c].astype(float)) if c in log_cols else df[c].astype(float) for c in features})
    return StandardScaler().fit_transform(M.values), features


def pca_project(X: np.ndarray, n_components: int = 3):
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    return pca.fit_transform(X), pca.explained_variance_ratio_


# --------------------------------------------------------------------------- algorithms

def build_model(name: str, params: dict):
    p = {**default_params(name), **(params or {})}
    if name == "K-Means":
        return KMeans(n_clusters=p["n_clusters"], init=p["init"], n_init=p["n_init"],
                      max_iter=p["max_iter"], random_state=RANDOM_SEED)
    if name == "Mini-Batch K-Means":
        return MiniBatchKMeans(n_clusters=p["n_clusters"], batch_size=p["batch_size"],
                               n_init=p["n_init"], random_state=RANDOM_SEED)
    if name == "Agglomerative (Hierarchical)":
        return AgglomerativeClustering(n_clusters=p["n_clusters"], linkage=p["linkage"])
    if name == "Gaussian Mixture (EM)":
        return GaussianMixture(n_components=p["n_clusters"], covariance_type=p["covariance_type"],
                               max_iter=p["max_iter"], random_state=RANDOM_SEED)
    if name == "BIRCH":
        return Birch(n_clusters=p["n_clusters"], threshold=p["threshold"], branching_factor=p["branching_factor"])
    if name == "DBSCAN":
        return DBSCAN(eps=p["eps"], min_samples=p["min_samples"], n_jobs=-1)
    if name == "HDBSCAN":
        return HDBSCAN(min_cluster_size=p["min_cluster_size"], min_samples=p["min_samples"], copy=True)
    raise ValueError(f"Unknown algorithm: {name}")


def run_clustering(name: str, X: np.ndarray, params: dict = None) -> dict:
    model = build_model(name, params)
    start = time.time()
    labels = model.fit_predict(X)
    seconds = time.time() - start
    extra = {}
    if hasattr(model, "inertia_"):
        extra["inertia"] = float(model.inertia_)
    if isinstance(model, GaussianMixture):
        extra["bic"] = float(model.bic(X))
        extra["aic"] = float(model.aic(X))
    return {"labels": np.asarray(labels), "seconds": seconds, "model": model, **extra}


def evaluate_clustering(X: np.ndarray, labels: np.ndarray, y_suspicious: np.ndarray = None,
                        silhouette_sample: int = 4000) -> dict:
    """Internal quality (silhouette, Davies-Bouldin, Calinski-Harabasz) + AML usefulness of the clusters."""
    labels = np.asarray(labels)
    clustered = labels != -1
    n_clusters = len(set(labels[clustered]))
    out = {
        "n_clusters": n_clusters,
        "noise_pct": float((~clustered).mean() * 100),
        "silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan,
    }
    if n_clusters >= 2 and clustered.sum() > n_clusters:
        Xc, lc = X[clustered], labels[clustered]
        out["silhouette"] = float(silhouette_score(
            Xc, lc, sample_size=min(silhouette_sample, len(lc)), random_state=RANDOM_SEED))
        out["davies_bouldin"] = float(davies_bouldin_score(Xc, lc))
        out["calinski_harabasz"] = float(calinski_harabasz_score(Xc, lc))
    if y_suspicious is not None:
        y = np.asarray(y_suspicious).astype(float)
        overall = y.mean() if y.mean() > 0 else np.nan
        rates = pd.Series(y).groupby(labels).mean()
        lift = rates / overall
        hotspots = lift[lift >= 2].index
        out["max_suspicious_rate"] = float(rates.max() * 100)
        out["max_suspicious_lift"] = float(lift.max())
        # Share of all suspicious accounts that land in "hotspot" clusters (rate >= 2x overall)
        out["suspicious_in_hotspots_pct"] = float(y[np.isin(labels, hotspots)].sum() / max(y.sum(), 1) * 100)
    return out


def k_sweep(name: str, X: np.ndarray, k_values, base_params: dict = None, y_suspicious=None) -> pd.DataFrame:
    rows = []
    for k in k_values:
        params = {**default_params(name), **(base_params or {}), "n_clusters": int(k)}
        res = run_clustering(name, X, params)
        metrics = evaluate_clustering(X, res["labels"], y_suspicious)
        rows.append({"k": int(k), "seconds": res["seconds"], "inertia": res.get("inertia"),
                     "bic": res.get("bic"), "aic": res.get("aic"), **metrics})
    return pd.DataFrame(rows)


def find_elbow(k_values, values) -> int:
    """Knee point: the k whose (normalised) point is farthest from the line joining the first and last points."""
    k = np.asarray(k_values, dtype=float)
    v = np.asarray(values, dtype=float)
    if len(k) < 3 or np.any(np.isnan(v)):
        return int(k[0])
    kn = (k - k.min()) / (k.max() - k.min())
    vn = (v - v.min()) / (v.max() - v.min() or 1)
    dx, dy = kn[-1] - kn[0], vn[-1] - vn[0]
    d = np.abs(dx * (vn - vn[0]) - dy * (kn - kn[0])) / np.hypot(dx, dy)
    return int(k[np.argmax(d)])


def k_distance(X: np.ndarray, min_samples: int) -> np.ndarray:
    """Sorted distance to the min_samples-th neighbour: the 'knee' suggests a DBSCAN eps."""
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
    dist, _ = nn.kneighbors(X)
    return np.sort(dist[:, -1])


# --------------------------------------------------------------------------- interpretation

def profile_clusters(df: pd.DataFrame, X: np.ndarray, labels: np.ndarray, features: list = None) -> pd.DataFrame:
    """Per-cluster size, suspicious rate, raw feature means and an auto-generated descriptive name."""
    features = features or FEATURE_NAMES
    labels = np.asarray(labels)
    Z = pd.DataFrame(X, columns=features)
    rows = []
    for c in sorted(set(labels)):
        mask = labels == c
        if c == -1:
            name = "Noise / Outliers"
        else:
            z = Z[mask].mean()
            top = z.abs().sort_values(ascending=False).index[:2]
            name = " · ".join(f"{'High' if z[f] > 0 else 'Low'} {FEATURE_LABELS[f].lower()}" for f in top)
        row = {"cluster": int(c), "cluster_name": f"C{c}: {name}" if c != -1 else name,
               "accounts": int(mask.sum()), "share_pct": mask.mean() * 100}
        if "ever_suspicious" in df:
            row["suspicious_rate_pct"] = df.loc[mask, "ever_suspicious"].mean() * 100
        for f in features:
            row[f] = df.loc[mask, f].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_zscores(X: np.ndarray, labels: np.ndarray, features: list = None) -> pd.DataFrame:
    """Mean standardised value of each feature per cluster (for heatmaps)."""
    features = features or FEATURE_NAMES
    Z = pd.DataFrame(X, columns=[FEATURE_LABELS[f] for f in features])
    Z["cluster"] = labels
    return Z.groupby("cluster").mean()


# --------------------------------------------------------------------------- batch pipeline

def compare_algorithms(X: np.ndarray, y_suspicious, configs: dict) -> pd.DataFrame:
    rows = []
    for name, params in configs.items():
        res = run_clustering(name, X, params)
        rows.append({"algorithm": name, "family": ALGORITHMS[name]["family"], "seconds": res["seconds"],
                     **evaluate_clustering(X, res["labels"], y_suspicious)})
    return pd.DataFrame(rows)


def train_behavioral_clustering(sample_size: int = 5000):
    print("=" * 60)
    print("Clustering Account Transaction-Behaviour Groups (SAML-D)...")
    print("=" * 60)

    os.makedirs("models/saved", exist_ok=True)
    df = load_account_profiles(sample_size)
    X, features = prepare_matrix(df)
    y = df["ever_suspicious"].values

    sweep = k_sweep("K-Means", X, range(2, 11), y_suspicious=y)
    elbow_k = find_elbow(sweep["k"], sweep["inertia"])
    best_sil_k = int(sweep.loc[sweep["silhouette"].idxmax(), "k"])
    print(f"K-Means sweep: elbow at K={elbow_k}, best silhouette at K={best_sil_k}")

    kmeans = run_clustering("K-Means", X, {"n_clusters": elbow_k})
    df["kmeans_cluster"] = kmeans["labels"]
    profiles = profile_clusters(df, X, kmeans["labels"], features)
    df["cluster_label"] = df["kmeans_cluster"].map(profiles.set_index("cluster")["cluster_name"])
    coords, _ = pca_project(X)
    df["pca_x"], df["pca_y"], df["pca_z"] = coords[:, 0], coords[:, 1], coords[:, 2]

    configs = {name: {**default_params(name), **({"n_clusters": elbow_k} if ALGORITHMS[name]["uses_k"] else {})}
               for name in ALGORITHMS}
    comparison = compare_algorithms(X, y, configs)
    print(comparison[["algorithm", "n_clusters", "noise_pct", "silhouette", "davies_bouldin",
                      "max_suspicious_lift", "seconds"]].round(3).to_string(index=False))

    artifacts = {
        "kmeans": kmeans["model"],
        "features": features,
        "k_sweep": sweep,
        "elbow_k": elbow_k,
        "best_silhouette_k": best_sil_k,
        "cluster_profiles": profiles,
        "comparison": comparison,
        "profiled_df": df,
    }
    joblib.dump(artifacts, "models/saved/clusterer.joblib", compress=3)
    df.to_csv("data/processed/account_behavior_clusters.csv", index=False)
    print("Clustering artifacts saved successfully!")
    return artifacts


if __name__ == "__main__":
    train_behavioral_clustering()
