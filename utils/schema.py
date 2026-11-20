"""Centralized Feature Schema, Normalization, and Validation.

Enforces strict exclusion of all targets, ground truth, and metadata columns
across dataset loading, scenario pool creation, training, prediction, and drift lab.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# Normalized (lowercase, stripped, spaces/hyphens -> underscores) names of prohibited columns
PROHIBITED_FEATURE_NAMES: frozenset[str] = frozenset({
    "label",
    "binary_target",
    "multiclass_target",
    "attack_cat",
    "attack_category",
    "attack_type",
    "class",
    "target",
    "y",
    "is_attack",
    "_attack_type",
    "_source_file",
    "source_file",
    "id",
    "row",
    "original_label",
    "final_verdict",
    "binary_prediction",
    "multiclass_prediction",
    "attack_probability",
    "binary_confidence",
    "multiclass_confidence",
    "models_agree",
    "alternative_categories",
    "prediction",
    "predicted_label",
    "ground_truth",
})

# Substrings that indicate target or metadata when appearing as complete tokens or suffixes
PROHIBITED_SUFFIXES: tuple[str, ...] = (
    "_target",
    "_label",
    "_prediction",
    "_probability",
    "_confidence",
    "_verdict",
)


def normalize_feature_name(name: str | object) -> str:
    """Normalize column names consistently: strip, lowercase, spaces/hyphens -> underscores."""
    s = str(name).strip().lower()
    return re.sub(r"[\s\-]+", "_", s)


def is_prohibited_feature(name: str | object) -> bool:
    """Check if a column name is a prohibited target, label, metadata, or app-generated output.

    Uses explicit schema matching rather than broad substring matching to avoid
    dropping legitimate network-flow features (e.g. Subflow_Fwd_Packets, Fwd_Header_Length).
    """
    norm = normalize_feature_name(name)
    if norm in PROHIBITED_FEATURE_NAMES:
        return True
    if norm.startswith("unnamed") or norm.startswith("_"):
        return True
    if any(norm.endswith(suffix) for suffix in PROHIBITED_SUFFIXES):
        return True
    return False


def filter_clean_features(columns: Iterable[str]) -> list[str]:
    """Return only legitimate predictor feature columns, preserving original order."""
    return [c for c in columns if not is_prohibited_feature(c)]


def validate_feature_schema(columns: Sequence[str], context: str = "") -> list[str]:
    """Validate that a list of feature columns contains no prohibited fields and no duplicates."""
    cols_list = list(columns)
    prohibited = [c for c in cols_list if is_prohibited_feature(c)]
    if prohibited:
        ctx_msg = f" in {context}" if context else ""
        raise ValueError(
            f"Prohibited target/metadata column(s) detected{ctx_msg}: {prohibited}. "
            "These columns must never be used as predictive features."
        )
    if len(cols_list) != len(set(cols_list)):
        from collections import Counter
        dupes = [col for col, count in Counter(cols_list).items() if count > 1]
        ctx_msg = f" in {context}" if context else ""
        raise ValueError(f"Duplicate feature columns detected{ctx_msg}: {dupes}")
    return cols_list


def validate_feature_columns_json(path_or_list: str | Path | Sequence[str]) -> list[str]:
    """Validate feature_columns.json file or list against the prohibited schema."""
    if isinstance(path_or_list, (str, Path)):
        p = Path(path_or_list)
        if not p.exists():
            raise FileNotFoundError(f"Feature columns file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = list(path_or_list)

    if not isinstance(data, list) or not data:
        raise ValueError("Feature columns must be a non-empty list of column names.")
    return validate_feature_schema(data, context="feature_columns.json")


def assert_clean_features(X: pd.DataFrame, expected_features: Sequence[str] | None = None) -> None:
    """Hard assertion immediately before every model fit and predict.

    Guarantees:
    1. X is a DataFrame.
    2. No prohibited target, label, or metadata columns exist in X.
    3. Feature names are unique.
    4. If expected_features is provided, columns match exactly in name and order.
    5. All values are numeric (no strings/objects).
    6. No NaN or infinite values exist in X.
    """
    if not isinstance(X, pd.DataFrame):
        raise AssertionError(f"Expected pandas DataFrame for features, got {type(X).__name__}")

    prohibited = [c for c in X.columns if is_prohibited_feature(c)]
    if prohibited:
        raise AssertionError(
            f"DATA LEAKAGE BLOCKED: Prohibited column(s) found in feature matrix X: {prohibited}. "
            "Targets, labels, and metadata must never enter model training or prediction."
        )

    if len(X.columns) != len(set(X.columns)):
        from collections import Counter
        dupes = [col for col, count in Counter(X.columns).items() if count > 1]
        raise AssertionError(f"Duplicate columns in feature matrix X: {dupes}")

    if expected_features is not None:
        expected_list = list(expected_features)
        if list(X.columns) != expected_list:
            missing = set(expected_list) - set(X.columns)
            extra = set(X.columns) - set(expected_list)
            raise AssertionError(
                f"Feature schema mismatch. Expected {len(expected_list)} features, got {len(X.columns)}. "
                f"Missing: {list(missing)[:5]}, Extra: {list(extra)[:5]}"
            )

    # Check numeric types
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise AssertionError(f"Non-numeric column(s) in feature matrix: {non_numeric[:5]}")

    # Check NaNs and Infs
    if X.isna().any().any():
        nan_cols = [c for c in X.columns if X[c].isna().any()]
        raise AssertionError(f"Feature matrix contains NaN values in columns: {nan_cols[:5]}")

    vals = X.to_numpy(dtype=float, copy=False)
    if np.isinf(vals).any():
        inf_cols = [X.columns[i] for i in np.flatnonzero(np.isinf(vals).any(axis=0))]
        raise AssertionError(f"Feature matrix contains infinite values in columns: {inf_cols[:5]}")


def audit_suspicious_features(
    df: pd.DataFrame, target: np.ndarray, threshold: float = 0.999
) -> list[dict]:
    """Inspect features for exact or near-exact copies of the target.

    Does not automatically delete legitimate features; returns a list of suspicious
    correlations for audit reporting.
    """
    suspicious = []
    y_num = target.astype(float)
    y_std = np.std(y_num)
    if y_std == 0:
        return suspicious

    for col in df.columns:
        if is_prohibited_feature(col):
            continue
        try:
            x_vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            if np.isnan(x_vals).all():
                continue
            valid = ~np.isnan(x_vals) & ~np.isnan(y_num)
            if valid.sum() < 2:
                continue
            x_v = x_vals[valid]
            y_v = y_num[valid]
            x_std = np.std(x_v)
            if x_std == 0:
                continue
            corr = np.abs(np.corrcoef(x_v, y_v)[0, 1])
            if np.isnan(corr):
                continue
            if corr >= threshold:
                suspicious.append({
                    "feature": col,
                    "correlation_with_target": float(corr),
                    "reason": f"Suspiciously high correlation (>= {threshold}) with target",
                })
        except Exception:
            continue
    return suspicious
