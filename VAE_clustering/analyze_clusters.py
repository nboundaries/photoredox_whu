"""Create downstream summaries and visualizations for clustered reactions."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

try:
    from .utils import ensure_dir
except ImportError:
    from utils import ensure_dir


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze and plot reaction clustering results.")
    parser.add_argument("--clustered", required=True, help="reactions_with_clusters.csv from cluster_reactions.py.")
    parser.add_argument("--latent", required=True, help="reaction_latent.npy from cluster_reactions.py.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import TSNE

    out_dir = ensure_dir(args.out_dir)
    clustered = pd.read_csv(args.clustered)
    latent = np.load(args.latent)
    if len(clustered) != latent.shape[0]:
        raise ValueError("Clustered CSV and latent matrix have different row counts.")

    cluster_counts = clustered["Cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Count")
    cluster_counts.to_csv(out_dir / "cluster_counts.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=cluster_counts, x="Cluster", y="Count", color="#4C78A8")
    plt.tight_layout()
    plt.savefig(out_dir / "cluster_counts.png", dpi=300)
    plt.close()

    if latent.shape[0] > 2:
        perplexity = min(args.perplexity, max(1.0, latent.shape[0] - 1.0))
        tsne = TSNE(n_components=2, random_state=args.random_state, perplexity=perplexity, init="pca", learning_rate="auto")
        tsne_values = tsne.fit_transform(latent)
        tsne_df = pd.DataFrame(tsne_values, columns=["t-SNE1", "t-SNE2"])
        tsne_df["Cluster"] = clustered["Cluster"].values
        tsne_df.to_csv(out_dir / "tsne.csv", index=False)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=tsne_df, x="t-SNE1", y="t-SNE2", hue="Cluster", palette="tab20", s=45, alpha=0.75)
        plt.tight_layout()
        plt.savefig(out_dir / "tsne_clusters.png", dpi=300)
        plt.close()

    if "Title" in clustered.columns:
        title_counts = clustered.groupby(["Title", "Cluster"]).size().unstack(fill_value=0)
        title_counts.to_csv(out_dir / "cluster_frequencies_by_title.csv")

    reagent_col = pick_column(clustered, ["Reagents_Smiles", "Reagent", "Reagents", "reagents"])
    catalyst_col = pick_column(clustered, ["Catalysts_Smiles", "Catalyst", "Catalysts", "catalysts"])
    if reagent_col and catalyst_col:
        combo_counts = (
            clustered.groupby(["Cluster", reagent_col, catalyst_col])
            .size()
            .reset_index(name="Frequency")
            .sort_values(["Cluster", "Frequency"], ascending=[True, False])
        )
        combo_counts.to_csv(out_dir / "reagent_catalyst_frequency_by_cluster.csv", index=False)

    print(f"Saved analysis outputs to {out_dir}.")


if __name__ == "__main__":
    main()
