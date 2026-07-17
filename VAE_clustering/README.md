# VAE_clustering: Reaction Clustering with VAE Latent Space and K-Means

This directory provides a cleaned, command-line workflow for reaction clustering based on molecular feature generation, variational autoencoder (VAE) embedding, and K-Means clustering.

The workflow was adapted from the methodology and code structure of the original repository:

https://github.com/HamidHadipour/Deep-clustering-of-small-molecules-at-large-scale-via-variational-autoencoder-embedding-and-K-means

Compared with the original notebook-style scripts, this version separates the workflow into reusable Python entry points, removes hard-coded absolute paths, avoids global notebook variables, and focuses on the reaction-level task used in this project. In the current workflow, separate VAE models are trained for reactants and products only. The reactant and product latent vectors are then concatenated and clustered with K-Means.

## Workflow Overview

The full workflow is:

1. Generate molecular features for reactants and train a reactant VAE.
2. Generate molecular features for products and train a product VAE.
3. Concatenate reactant and product latent vectors and train K-Means.
4. Perform downstream cluster analysis, including t-SNE visualization and frequency statistics.

The VAE input dimension, `D_in`, is inferred automatically from the processed feature matrix. It does not need to be manually edited in the source code.

## Installation

It is recommended to use a clean Python or conda environment.

```bash
pip install -r VAE_clustering/requirements.txt
```

If `rdkit` cannot be installed through pip in your environment, install it with conda:

```bash
conda install -c conda-forge rdkit
```

## 1. Reactant Feature Generation and VAE Training

Assuming that a cleaned SciFinder reaction table named `scifinder_clean.csv` is available, the `SMILES` column is used as the reactant column in the example below.

```bash
python VAE_clustering/generate_features.py --input scifinder_clean.csv --smiles-col SMILES --entity reactant --out-dir VAE_clustering/outputs/reactant

python VAE_clustering/build_dataset.py --features-dir VAE_clustering/outputs/reactant --out-dir VAE_clustering/outputs/reactant

python VAE_clustering/train_vae.py --dataset VAE_clustering/outputs/reactant/dataset.npy --entity reactant --out-dir VAE_clustering/outputs/reactant
```

Main outputs:

- `atom_features.csv`
- `bond_features.csv`
- `rdkit_2d_descriptors.csv`
- `metadata.csv`
- `invalid_smiles.csv`, if invalid SMILES are detected
- `dataset.csv`
- `dataset.npy`
- `vae_model.pt`
- `latent.npy`
- `train_log.csv`
- `vae_config.json`

## 2. Product Feature Generation and VAE Training

For the same `scifinder_clean.csv` table, the `Products` column is used as the product column in the example below.

```bash
python VAE_clustering/generate_features.py --input scifinder_clean.csv --smiles-col Products --entity product --out-dir VAE_clustering/outputs/product

python VAE_clustering/build_dataset.py --features-dir VAE_clustering/outputs/product --out-dir VAE_clustering/outputs/product

python VAE_clustering/train_vae.py --dataset VAE_clustering/outputs/product/dataset.npy --entity product --out-dir VAE_clustering/outputs/product
```

Training parameters can be adjusted from the command line. For example:

```bash
python VAE_clustering/train_vae.py --dataset VAE_clustering/outputs/product/dataset.npy --entity product --out-dir VAE_clustering/outputs/product --epochs 1500 --batch-size 1024 --latent-dim 32
```

## 3. Reaction-Level K-Means Clustering

After the reactant and product VAEs have been trained, concatenate their latent vectors and train K-Means:

```bash
python VAE_clustering/cluster_reactions.py --reactant-latent VAE_clustering/outputs/reactant/latent.npy --product-latent VAE_clustering/outputs/product/latent.npy --metadata scifinder_clean.csv --n-clusters 15 --out-dir VAE_clustering/outputs/clustering
```

The reaction representation is constructed as:

```text
reaction_latent = concatenate([reactant_latent, product_latent], axis=1)
```

Main outputs:

- `reaction_latent.npy`: concatenated reactant/product latent vectors
- `kmeans_model.joblib`: trained K-Means model
- `reactions_with_clusters.csv`: original metadata plus `Cluster`, `is_cluster_center`, and `distance_to_center`
- `cluster_metrics.json`: internal clustering metrics, including Calinski-Harabasz, silhouette, and Davies-Bouldin scores

## 4. Downstream Cluster Analysis

```bash
python VAE_clustering/analyze_clusters.py --clustered VAE_clustering/outputs/clustering/reactions_with_clusters.csv --latent VAE_clustering/outputs/clustering/reaction_latent.npy --out-dir VAE_clustering/outputs/analysis
```

Main outputs:

- `tsne.csv`
- `tsne_clusters.png`
- `cluster_counts.csv`
- `cluster_counts.png`
- `cluster_frequencies_by_title.csv`, generated when the metadata contains a `Title` column
- `reagent_catalyst_frequency_by_cluster.csv`, generated when reagent and catalyst columns are available

## Input Data

The scripts are not tied to a fixed input filename. Input files and column names are provided through command-line arguments.

Key arguments for feature generation:

- `--input`: input CSV or Excel file
- `--smiles-col`: column containing the SMILES strings to process
- `--entity`: either `reactant` or `product`
- `--out-dir`: output directory

Supported SMILES formats include:

- a single SMILES string, for example `CCO`
- a dot-separated multi-molecule SMILES string, for example `CCO.O=C=O`
- a Python-list-like string, for example `['CCO', 'O=C=O']`

Rows that cannot be parsed by RDKit are written to `invalid_smiles.csv`. Valid rows are still processed.

## Preprocessing Details

`build_dataset.py` combines atom, bond, and RDKit 2D descriptor features into a VAE-ready matrix.

The preprocessing steps are:

1. Load atom, bond, and RDKit 2D descriptor feature files.
2. Replace invalid values such as NaN or infinity.
3. Standardize RDKit 2D descriptors.
4. Apply PCA to the concatenated atom and bond features.
5. Concatenate PCA-transformed atom/bond features with standardized RDKit descriptors.
6. Remove zero-variance columns with `VarianceThreshold(0)`.
7. Save both `dataset.csv` and `dataset.npy`.

The final VAE input dimension is inferred as:

```python
D_in = dataset.shape[1]
```

## VAE Model and Training

`train_vae.py` trains either a reactant VAE or a product VAE. It does not train reagent or catalyst VAEs.

The VAE consists of:

- an encoder made of fully connected layers with batch normalization and ReLU activation
- two parallel output heads for the latent mean, `mu`, and log variance, `logvar`
- a reparameterization step during training
- a decoder made of fully connected layers with batch normalization and ReLU activation

The latent representation saved for clustering is `mu`, rather than a randomly sampled latent vector.

Important command-line parameters:

- `--epochs`: number of training epochs
- `--batch-size`: mini-batch size
- `--latent-dim`: latent dimension, default `32`
- `--seed`: random seed
- `--lr`: Adam learning rate

The loss is:

```text
total loss = reconstruction MSE + KL divergence
```

## Recommended Usage Notes

Use the same original reaction table for reactant and product processing. After feature generation, check `invalid_smiles.csv` for both reactants and products. If the valid rows differ between the two sides, clean the input table first so that reactant and product samples remain aligned before clustering.

The expected order is:

1. Generate reactant features.
2. Build the reactant dataset.
3. Train the reactant VAE.
4. Generate product features.
5. Build the product dataset.
6. Train the product VAE.
7. Concatenate latent vectors and run K-Means.
8. Run downstream cluster analysis.
