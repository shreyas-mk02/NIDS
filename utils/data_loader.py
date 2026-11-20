"""Dataset discovery, chunked loading and profiling (no UI code besides st.cache_data).

Large CSVs (CIC-IDS2017 is >2M rows) are scanned in chunks: exact record and class
counts are computed over the *whole* data, while feature-level analysis uses an
order-preserving random sample so memory stays bounded.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils.config import DATA_DIR
from utils.prediction import is_benign
from utils.preprocessing import normalize_columns
from utils.schema import assert_clean_features, filter_clean_features, is_prohibited_feature

DATASETS = {
    "CIC-IDS2017": {
        "key": "cicids",
        "match": ("cic",),
        "folder": "data/CIC-IDS2017/",
        "label_candidates": ["Attack Type", "Attack_Type", "attack_type", "Label", "label"],
        "description": (
            "Flow-based dataset from the Canadian Institute for Cybersecurity. Traffic was captured over "
            "five days and converted to bidirectional flows with CICFlowMeter; flows are labelled as benign "
            "or as one of several attack families (DoS, DDoS, brute force, web attacks, botnet, port scan…)."
        ),
        "files_hint": "The daily 'MachineLearningCVE' CSV files or the merged/cleaned CSV (e.g. cicids2017_cleaned.csv) "
        "with a 'Label' or 'Attack Type' column.",
    },
    "UNSW-NB15": {
        "key": "unsw",
        "match": ("unsw", "nb15"),
        "folder": "data/UNSW-NB15/",
        "label_candidates": ["attack_cat", "Attack_cat", "label", "Label"],
        "description": (
            "Dataset created by the Cyber Range Lab of UNSW Canberra using the IXIA PerfectStorm tool. It mixes "
            "real normal activity with synthetic attack behaviour from nine families (Fuzzers, Analysis, Backdoors, "
            "DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms)."
        ),
        "files_hint": "UNSW_NB15_training-set.csv / UNSW_NB15_testing-set.csv (files with a header row, "
        "containing 'attack_cat' and 'label').",
    },
}

LABEL_LIKE = {
    "label", "attack_cat", "attack_category", "attack type", "attack_type",
    "class", "id", "_attack_type", "_source_file", "binary_target", "multiclass_target", "target", "y"
}


@dataclass
class DatasetBundle:
    name: str
    source: str  # "disk" | "upload"
    files: list[str]
    sample: pd.DataFrame  # includes _attack_type; feature columns normalised
    total_records: int
    attack_counts: pd.Series  # exact counts per class (full data)
    feature_columns: list[str]
    numeric_columns: list[str]
    sampled: bool
    notes: list[str] = field(default_factory=list)

    @property
    def normal_records(self) -> int:
        return int(sum(v for k, v in self.attack_counts.items() if is_benign(k)))

    @property
    def malicious_records(self) -> int:
        return int(self.total_records - self.normal_records)

    @property
    def n_attack_classes(self) -> int:
        return int(sum(1 for k in self.attack_counts.index if not is_benign(k)))


# --------------------------------------------------------------------------- #
def find_files(dataset_name: str) -> list[Path]:
    """CSV files under data/ whose path mentions the dataset."""
    if not DATA_DIR.exists():
        return []
    tokens = DATASETS[dataset_name]["match"]
    files = [
        p for p in sorted(DATA_DIR.rglob("*.csv"))
        if any(t in str(p.relative_to(DATA_DIR)).lower() for t in tokens)
    ]
    return files


def _clean_label(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    s = s.str.replace(r"[–—ï¿½]+", "-", regex=True)
    s = s.str.replace(r"\s*-\s*", " - ", regex=True)
    return s


def attack_type_series(df: pd.DataFrame, dataset_name: str) -> pd.Series | None:
    """Per-row attack category, with benign traffic named consistently."""
    cands = DATASETS[dataset_name]["label_candidates"]
    present = {c.lower(): c for c in df.columns}
    if dataset_name == "UNSW-NB15":
        cat = next((present[c.lower()] for c in ("attack_cat",) if c.lower() in present), None)
        lab = present.get("label")
        if cat is not None:
            s = _clean_label(df[cat]).fillna("").replace("", "Normal")
            if lab is not None:
                lab_num = pd.to_numeric(df[lab], errors="coerce")
                s = s.where(~((lab_num == 0) & (s == "Normal")), "Normal")
            return s.astype(str)
        if lab is not None:
            lab_num = pd.to_numeric(df[lab], errors="coerce")
            return pd.Series(np.where(lab_num == 0, "Normal", "Attack"), index=df.index)
        return None
    col = next((present[c.lower()] for c in cands if c.lower() in present), None)
    if col is None:
        return None
    return _clean_label(df[col]).fillna("Unknown").astype(str)


def _count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            n += chunk.count(b"\n")
    return max(n - 1, 0)


def _feature_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return clean feature columns and numeric feature columns using centralized schema validation."""
    feats = filter_clean_features(df.columns)
    nums = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    return feats, nums


def signature(files: list[Path]) -> tuple:
    return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files)


# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=4)
def load_disk_dataset(dataset_name: str, sig: tuple, max_sample_rows: int = 150_000, seed: int = 42):
    """Scan all files of a dataset. Returns DatasetBundle or an error string."""
    files = [Path(s[0]) for s in sig]
    try:
        total_lines = sum(_count_lines(p) for p in files)
        frac = min(1.0, max_sample_rows / max(total_lines, 1))
        rng = np.random.default_rng(seed)
        counts: Counter = Counter()
        n_rows = 0
        samples: list[pd.DataFrame] = []
        for p in files:
            for chunk in pd.read_csv(p, chunksize=200_000, low_memory=False, encoding_errors="replace"):
                chunk = normalize_columns(chunk)
                at = attack_type_series(chunk, dataset_name)
                if at is None:
                    return f"'{p.name}' has no recognisable label column."
                counts.update(at.value_counts().to_dict())
                n_rows += len(chunk)
                mask = rng.random(len(chunk)) < frac
                if mask.any():
                    s = chunk.loc[mask].copy()
                    s["_attack_type"] = at.loc[mask].to_numpy()
                    s["_source_file"] = p.name
                    samples.append(s)
        if not samples:
            return "The dataset files contain no rows."
        sample = pd.concat(samples, ignore_index=True)
    except Exception as exc:  # noqa: BLE001 - surface a friendly message instead of a stack trace
        return f"Dataset could not be read ({type(exc).__name__}: {str(exc)[:140]})."

    feats, nums = _feature_cols(sample)
    notes = []
    if frac < 1.0:
        notes.append(
            f"Record and class counts are exact ({n_rows:,} rows); feature plots use an order-preserving "
            f"random sample of {len(sample):,} rows ({frac:.1%})."
        )
    return DatasetBundle(
        name=dataset_name, source="disk", files=[p.name for p in files], sample=sample,
        total_records=n_rows, attack_counts=pd.Series(counts).sort_values(ascending=False),
        feature_columns=feats, numeric_columns=nums, sampled=frac < 1.0, notes=notes,
    )


def bundle_from_frame(df: pd.DataFrame, dataset_name: str, filename: str) -> DatasetBundle | str:
    """Profile an uploaded dataset (whole file is used; counts are exact)."""
    df = normalize_columns(df)
    at = attack_type_series(df, dataset_name)
    if at is None:
        return "No label column found. Expected one of: " + ", ".join(DATASETS[dataset_name]["label_candidates"])
    df = df.copy()
    df["_attack_type"] = at.to_numpy()
    df["_source_file"] = filename
    feats, nums = _feature_cols(df)
    return DatasetBundle(
        name=dataset_name, source="upload", files=[filename], sample=df,
        total_records=len(df), attack_counts=df["_attack_type"].value_counts(),
        feature_columns=feats, numeric_columns=nums, sampled=False,
        notes=["Statistics are computed from the uploaded file only."],
    )


def derive_truth(series: pd.Series) -> np.ndarray:
    """Boolean ground truth (True = attack) from a label column (strings or 0/1)."""
    return np.array([not is_benign(v) for v in _clean_label(series).fillna("")], dtype=bool)


def clean_numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Numeric view with inf→NaN (for plotting/statistics only)."""
    clean_cols = filter_clean_features(cols)
    return df[clean_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def load_scenario_pool(dataset_name: str, sig: tuple, counts: dict, features: tuple, cap_per_class: int = 40_000, seed: int = 7):
    """Class-capped pool for the drift lab: *all* records of rare categories, a uniform sample of big ones.

    ``counts`` are the exact per-category counts of the dataset (from ``load_disk_dataset``). Returns a
    DataFrame (feature columns + ``_attack_type`` + ``_source_file``) or an error string.
    Guarantees no target or metadata columns are included in feature columns.
    """
    files = [Path(s[0]) for s in sig]
    rng = np.random.default_rng(seed)
    rate = {c: min(1.0, cap_per_class / max(int(n), 1)) for c, n in counts.items()}
    clean_req_features = filter_clean_features(features)
    parts: list[pd.DataFrame] = []
    try:
        for p in files:
            for chunk in pd.read_csv(p, chunksize=200_000, low_memory=False, encoding_errors="replace"):
                chunk = normalize_columns(chunk)
                at = attack_type_series(chunk, dataset_name)
                if at is None:
                    return f"'{p.name}' has no recognisable label column."
                keep = rng.random(len(chunk)) < at.map(rate).fillna(1.0).to_numpy()
                if keep.any():
                    cols = [f for f in clean_req_features if f in chunk.columns]
                    part = chunk.loc[keep, cols].apply(pd.to_numeric, errors="coerce").astype("float32")
                    part["_attack_type"] = at.loc[keep].astype(str).to_numpy()
                    part["_source_file"] = p.name
                    parts.append(part)
        if not parts:
            return "The dataset files contain no rows."
        pool = pd.concat(parts, ignore_index=True)
    except Exception as exc:  # noqa: BLE001
        return f"Dataset could not be read ({type(exc).__name__}: {str(exc)[:140]})."
    feats = [f for f in clean_req_features if f in pool.columns]
    pool[feats] = pool[feats].replace([np.inf, -np.inf], np.nan)
    pool = pool.dropna(subset=feats).reset_index(drop=True)
    # Hard assertion: pool features must be clean
    assert_clean_features(pool[feats], feats)
    return pool
