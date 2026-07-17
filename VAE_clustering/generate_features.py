"""Generate RDKit features for one reaction entity column."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

try:
    from .utils import ensure_dir, normalize_smiles_series, read_table
except ImportError:
    from utils import ensure_dir, normalize_smiles_series, read_table


RDLogger.DisableLog("rdApp.*")


ATOM_FEATURES = {
    "atomic_num": list(range(118)),
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-1, -2, 1, 2, 0],
    "chiral_tag": [0, 1, 2, 3],
    "num_Hs": [0, 1, 2, 3, 4],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}


def one_hot_encoding(value, choices):
    encoding = [0] * len(choices)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def atom_features_onehot(atom) -> List[float]:
    return (
        one_hot_encoding(atom.GetAtomicNum() - 1, ATOM_FEATURES["atomic_num"])
        + one_hot_encoding(atom.GetTotalDegree(), ATOM_FEATURES["degree"])
        + one_hot_encoding(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
        + one_hot_encoding(int(atom.GetChiralTag()), ATOM_FEATURES["chiral_tag"])
        + one_hot_encoding(int(atom.GetTotalNumHs()), ATOM_FEATURES["num_Hs"])
        + one_hot_encoding(atom.GetHybridization(), ATOM_FEATURES["hybridization"])
        + [1 if atom.GetIsAromatic() else 0]
        + [atom.GetMass() * 0.01]
    )


def bond_features_onehot(bond) -> List[float]:
    bt = bond.GetBondType()
    return [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated() if bt is not None else 0,
        bond.IsInRing() if bt is not None else 0,
    ] + one_hot_encoding(int(bond.GetStereo()), list(range(6)))


def get_rdkit_2d_descriptors(mol) -> List[float]:
    values = []
    for _, func in Descriptors.descList:
        try:
            values.append(float(func(mol)))
        except Exception:
            values.append(np.nan)
    return values


def aggregate_local_features(mol, atom_dim: int, bond_dim: int) -> tuple[np.ndarray, np.ndarray]:
    atom_features = [atom_features_onehot(atom) for atom in mol.GetAtoms()]
    bond_features = [bond_features_onehot(bond) for bond in mol.GetBonds()]

    if atom_features:
        atom_vector = np.mean(np.asarray(atom_features, dtype=np.float32), axis=0)
    else:
        atom_vector = np.zeros(atom_dim, dtype=np.float32)

    if bond_features:
        bond_vector = np.mean(np.asarray(bond_features, dtype=np.float32), axis=0)
    else:
        bond_vector = np.zeros(bond_dim, dtype=np.float32)

    return atom_vector, bond_vector


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate RDKit features for reactant or product SMILES.")
    parser.add_argument("--input", required=True, help="Input CSV/XLSX file.")
    parser.add_argument("--smiles-col", required=True, help="Column containing SMILES for this entity.")
    parser.add_argument("--entity", required=True, choices=["reactant", "product"], help="Entity name for metadata.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = ensure_dir(args.out_dir)
    data = read_table(args.input)
    if args.smiles_col not in data.columns:
        raise ValueError(f"Column '{args.smiles_col}' not found in {args.input}")

    smiles_values = normalize_smiles_series(data[args.smiles_col])
    atom_dim = len(atom_features_onehot(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
    bond_dim = len(bond_features_onehot(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))
    descriptor_names = [name for name, _ in Descriptors.descList]

    atom_rows, bond_rows, descriptor_rows, valid_indices, invalid_rows = [], [], [], [], []

    for idx, smiles in enumerate(smiles_values):
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            invalid_rows.append({"row_index": idx, "smiles": smiles, "reason": "RDKit failed to parse SMILES"})
            continue
        mol = Chem.AddHs(mol)
        atom_vector, bond_vector = aggregate_local_features(mol, atom_dim, bond_dim)
        atom_rows.append(atom_vector)
        bond_rows.append(bond_vector)
        descriptor_rows.append(get_rdkit_2d_descriptors(mol))
        valid_indices.append(idx)

    atom_df = pd.DataFrame(atom_rows, columns=[f"atom_{i}" for i in range(atom_dim)])
    bond_df = pd.DataFrame(bond_rows, columns=[f"bond_{i}" for i in range(bond_dim)])
    desc_df = pd.DataFrame(descriptor_rows, columns=descriptor_names)
    metadata = data.iloc[valid_indices].reset_index(drop=True).copy()
    metadata.insert(0, f"{args.entity}_source_row", valid_indices)
    metadata[f"{args.entity}_smiles_normalized"] = [smiles_values[i] for i in valid_indices]

    atom_df.to_csv(out_dir / "atom_features.csv", index=False)
    bond_df.to_csv(out_dir / "bond_features.csv", index=False)
    desc_df.to_csv(out_dir / "rdkit_2d_descriptors.csv", index=False)
    metadata.to_csv(out_dir / "metadata.csv", index=False)
    pd.DataFrame(invalid_rows).to_csv(out_dir / "invalid_smiles.csv", index=False)

    print(f"Generated features for {len(valid_indices)} valid {args.entity} rows.")
    if invalid_rows:
        print(f"Skipped {len(invalid_rows)} invalid rows. See {out_dir / 'invalid_smiles.csv'}.")


if __name__ == "__main__":
    main()

