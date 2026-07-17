"""Build a fixed-width VAE input matrix from generated RDKit features."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

try:
    from .utils import ensure_dir, write_json
except ImportError:
    from utils import ensure_dir, write_json


def clean_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.apply(pd.to_numeric, errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(axis=1, how="all")
    return clean.fillna(0.0).astype(float)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a VAE-ready dataset from generated feature files.")
    parser.add_argument("--features-dir", required=True, help="Directory from generate_features.py.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--atom-pca-components", type=int, default=50, help="PCA components for atom+bond features.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    import joblib
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import StandardScaler

    features_dir = ensure_dir(args.features_dir)
    out_dir = ensure_dir(args.out_dir)

    atom = clean_numeric_frame(pd.read_csv(features_dir / "atom_features.csv"))
    bond = clean_numeric_frame(pd.read_csv(features_dir / "bond_features.csv"))
    descriptors = clean_numeric_frame(pd.read_csv(features_dir / "rdkit_2d_descriptors.csv"))

    if not (len(atom) == len(bond) == len(descriptors)):
        raise ValueError("Feature files have different row counts.")

    local_features = pd.concat([atom.add_prefix("A_"), bond.add_prefix("B_")], axis=1)
    local_scaler = StandardScaler()
    local_scaled = local_scaler.fit_transform(local_features)

    max_components = min(args.atom_pca_components, local_scaled.shape[0], local_scaled.shape[1])
    if max_components < 1:
        raise ValueError("No valid rows/features available for PCA.")
    local_pca = PCA(n_components=max_components)
    local_pca_values = local_pca.fit_transform(local_scaled)
    local_pca_df = pd.DataFrame(local_pca_values, columns=[f"PCA{i}" for i in range(1, max_components + 1)])

    descriptor_scaler = StandardScaler()
    descriptor_scaled = descriptor_scaler.fit_transform(descriptors)
    descriptor_df = pd.DataFrame(descriptor_scaled, columns=descriptors.columns)

    full_features = pd.concat([local_pca_df, descriptor_df], axis=1)
    selector = VarianceThreshold(0)
    dataset = selector.fit_transform(full_features).astype(np.float32)
    selected_columns = full_features.columns[selector.get_support()].tolist()

    pd.DataFrame(dataset, columns=selected_columns).to_csv(out_dir / "dataset.csv", index=False)
    np.save(out_dir / "dataset.npy", dataset)
    joblib.dump(local_scaler, out_dir / "local_feature_scaler.joblib")
    joblib.dump(local_pca, out_dir / "local_feature_pca.joblib")
    joblib.dump(descriptor_scaler, out_dir / "descriptor_scaler.joblib")
    joblib.dump(selector, out_dir / "variance_selector.joblib")
    write_json(
        out_dir / "preprocess_config.json",
        {
            "n_rows": int(dataset.shape[0]),
            "input_dim": int(dataset.shape[1]),
            "local_feature_dim": int(local_features.shape[1]),
            "descriptor_dim": int(descriptors.shape[1]),
            "atom_pca_components_requested": int(args.atom_pca_components),
            "atom_pca_components_used": int(max_components),
            "selected_columns": selected_columns,
        },
    )

    print(f"Saved dataset with shape {dataset.shape} to {out_dir}.")


if __name__ == "__main__":
    main()
