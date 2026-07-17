"""Shared utilities for the VAE + K-Means reaction clustering workflow."""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def write_json(path: str | Path, data: dict) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_smiles_cell(value) -> str:
    """Normalize one SMILES cell to a dot-joined string.

    Supports ordinary SMILES strings, dot-joined multi-molecule strings, and
    Python-list-like strings such as "['CCO', 'O=C=O']".
    """
    if pd.isna(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        return ".".join(str(item).strip() for item in value if str(item).strip())

    text = str(value).strip()
    if not text:
        return ""

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text
        if isinstance(parsed, (list, tuple, set)):
            return ".".join(str(item).strip() for item in parsed if str(item).strip())
    return text


def normalize_smiles_series(values: Iterable) -> List[str]:
    return [parse_smiles_cell(value) for value in values]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_numeric_matrix(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    return pd.read_csv(path).to_numpy(dtype=np.float32)


def align_metadata(metadata: pd.DataFrame, valid_indices: Sequence[int]) -> pd.DataFrame:
    return metadata.iloc[list(valid_indices)].reset_index(drop=True)

