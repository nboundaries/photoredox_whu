"""Cluster reactions by concatenated reactant and product VAE latent vectors."""

from __future__ import annotations

import argparse

import numpy as np

try:
    from .utils import ensure_dir, read_table, write_json
except ImportError:
    from utils import ensure_dir, read_table, write_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train K-Means on concatenated reactant/product latent vectors.")
    parser.add_argument("--reactant-latent", required=True)
    parser.add_argument("--product-latent", required=True)
    parser.add_argument("--metadata", required=True, help="Original metadata CSV/XLSX.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-clusters", type=int, default=15)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    import joblib
    from sklearn import metrics
    from sklearn.cluster import KMeans
    from sklearn.metrics import davies_bouldin_score, silhouette_score

    out_dir = ensure_dir(args.out_dir)
    reactant = np.load(args.reactant_latent)
    product = np.load(args.product_latent)
    if reactant.shape[0] != product.shape[0]:
        raise ValueError(f"Latent row counts differ: reactant={reactant.shape[0]}, product={product.shape[0]}")
    if args.n_clusters < 2:
        raise ValueError("--n-clusters must be at least 2.")
    if args.n_clusters > reactant.shape[0]:
        raise ValueError("--n-clusters cannot exceed the number of samples.")

    reaction_latent = np.concatenate([reactant, product], axis=1)
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.random_state, n_init=10)
    labels_zero = kmeans.fit_predict(reaction_latent)
    labels = labels_zero + 1
    centers = kmeans.cluster_centers_
    distances = np.linalg.norm(reaction_latent - centers[labels_zero], axis=1)

    center_indices = []
    for cluster_id in range(args.n_clusters):
        member_indices = np.where(labels_zero == cluster_id)[0]
        if len(member_indices) == 0:
            continue
        closest_member = member_indices[np.argmin(distances[member_indices])]
        center_indices.append(int(closest_member))

    metadata = read_table(args.metadata).iloc[: reaction_latent.shape[0]].reset_index(drop=True).copy()
    metadata["Cluster"] = labels
    metadata["is_cluster_center"] = False
    metadata.loc[center_indices, "is_cluster_center"] = True
    metadata["distance_to_center"] = distances

    metric_data = {
        "n_samples": int(reaction_latent.shape[0]),
        "latent_dim": int(reaction_latent.shape[1]),
        "n_clusters": int(args.n_clusters),
        "inertia": float(kmeans.inertia_),
        "calinski_harabasz": float(metrics.calinski_harabasz_score(reaction_latent, labels_zero)),
    }
    if args.n_clusters > 1 and reaction_latent.shape[0] > args.n_clusters:
        metric_data["silhouette"] = float(silhouette_score(reaction_latent, labels_zero))
        metric_data["davies_bouldin"] = float(davies_bouldin_score(reaction_latent, labels_zero))

    np.save(out_dir / "reaction_latent.npy", reaction_latent)
    metadata.to_csv(out_dir / "reactions_with_clusters.csv", index=False)
    joblib.dump(kmeans, out_dir / "kmeans_model.joblib")
    write_json(out_dir / "cluster_metrics.json", metric_data)
    print(f"Saved clustered reactions to {out_dir / 'reactions_with_clusters.csv'}.")


if __name__ == "__main__":
    main()
