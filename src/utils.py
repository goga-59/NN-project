"""Shared utilities: load_data, split, compute_metrics, find_best_threshold_f2."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def load_data(path: str = "data/raw/diabetic_data.csv") -> pd.DataFrame:
    return pd.read_csv(path, na_values="?", low_memory=False)


def split(
    df: pd.DataFrame,
    target_col: str = "target",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test = 60/20/20."""
    y = df[target_col]
    train, tmp = train_test_split(df, test_size=0.40, stratify=y, random_state=random_state)
    y_tmp = tmp[target_col]
    val, test = train_test_split(tmp, test_size=0.50, stratify=y_tmp, random_state=random_state)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def find_best_threshold_f2(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Return threshold that maximises F2 on the given split."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns one extra element (at recall=0), align
    f2 = (5 * precision[:-1] * recall[:-1]) / (4 * precision[:-1] + recall[:-1] + 1e-9)
    return float(thresholds[np.argmax(f2)])


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
    }
