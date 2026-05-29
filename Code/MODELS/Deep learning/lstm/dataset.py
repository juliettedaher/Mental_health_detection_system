"""
dataset.py
----------
Loads the French mental health CSV, encodes labels, and returns
train/test splits ready for vectorisation.

The label encoder is returned so predictor.py can inverse-transform
predictions back to human-readable strings ('Healthy' / 'Unhealthy').

Usage
-----
    from dataset import load_french_dataset

    X_train, X_test, y_train, y_test, le = load_french_dataset(
        path="data/french_cleaned.csv",
        text_col="text_nostop",
        label_col="mental_state",
        test_size=0.2,
        random_state=42,
    )
    # X_train / X_test : pd.Series of French strings
    # y_train / y_test : np.ndarray of integer class indices (0 or 1)
    # le               : fitted sklearn LabelEncoder
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_french_dataset(
    path: str,
    text_col: str = "text_nostop",
    label_col: str = "mental_state",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Load, validate, split, and encode the dataset.

    Parameters
    ----------
    path         : path to the CSV file
    text_col     : name of the column containing pre-processed French text
    label_col    : name of the column containing class labels
    test_size    : fraction of the dataset held out for testing
    random_state : random seed for reproducibility

    Returns
    -------
    X_train : pd.Series  — training texts
    X_test  : pd.Series  — test texts
    y_train : np.ndarray — integer-encoded training labels
    y_test  : np.ndarray — integer-encoded test labels
    le      : LabelEncoder fitted on training labels (use for inverse_transform)
    """
    # ── 1. Load ──────────────────────────────────────────────────────────
    print(f"[dataset] Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"[dataset] Raw shape: {df.shape}")

    # ── 2. Validate columns ──────────────────────────────────────────────
    for col in [text_col, label_col]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found. Available columns: {list(df.columns)}"
            )

    # ── 3. Drop nulls ────────────────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=[text_col, label_col])
    dropped = before - len(df)
    if dropped:
        print(f"[dataset] Dropped {dropped} rows with null values.")

    # ── 4. Class distribution ────────────────────────────────────────────
    print(f"[dataset] Class distribution:\n{df[label_col].value_counts().to_string()}")

    # ── 5. Encode labels ─────────────────────────────────────────────────
    le = LabelEncoder()
    y = le.fit_transform(df[label_col].values)
    print(f"[dataset] Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # ── 6. Split ─────────────────────────────────────────────────────────
    X = df[text_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(
        f"[dataset] Split → train: {len(X_train)} rows | test: {len(X_test)} rows"
    )

    return X_train, X_test, y_train, y_test, le
