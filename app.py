from __future__ import annotations

import hashlib
import linecache
import sys
import types
from pathlib import Path

import streamlit as st

_BUNDLE_DIR = Path(__file__).resolve().parent

_DARK_THEME = """[theme]
base = "dark"
primaryColor = "#ff4fa3"
backgroundColor = "#070b1f"
secondaryBackgroundColor = "#0b1033"
textColor = "#eef2ff"
font = "sans serif"

[server]
maxUploadSize = 500

[browser]
gatherUsageStats = false
"""


def _ensure_dark_theme() -> None:
    """The design is built for Streamlit's dark base theme. Apply it for new sessions right away and, if the
    project has no .streamlit/config.toml yet, create one so it is also active from the next start."""
    try:
        for key, val in (("theme.base", "dark"), ("theme.primaryColor", "#ff4fa3"), ("theme.backgroundColor", "#070b1f"),
                         ("theme.secondaryBackgroundColor", "#0b1033"), ("theme.textColor", "#eef2ff")):
            st._config.set_option(key, val)
        cfg = _BUNDLE_DIR / ".streamlit" / "config.toml"
        if Path.cwd().resolve() == _BUNDLE_DIR:
            # create it, or refresh the file written by the previous (navy) version of this app
            if not cfg.exists() or any(m in cfg.read_text(encoding="utf-8") for m in ("#050816", "#1b0e52")):
                cfg.parent.mkdir(exist_ok=True)
                cfg.write_text(_DARK_THEME, encoding="utf-8")
    except Exception:  # noqa: BLE001 - purely cosmetic, never block the app
        pass


_ensure_dark_theme()

# --------------------------------------------------------------------------------------------- #
# Embedded source code
# --------------------------------------------------------------------------------------------- #
_MODULE_SOURCES = {
    "utils.config": r'''
"""Central configuration: project paths, expected artifacts and project metadata.

The project root defaults to the parent of the ``app/`` directory. It can be
overridden with the ``NIDS_PROJECT_ROOT`` environment variable (useful for
testing against a different artifact folder without touching the real one).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# ``_BUNDLE_DIR`` is injected when everything is bundled into a single root-level app.py.
APP_DIR = Path(globals().get("_BUNDLE_DIR") or Path(__file__).resolve().parents[1])
ROOT = Path(os.environ.get("NIDS_PROJECT_ROOT", APP_DIR if globals().get("_BUNDLE_DIR") else APP_DIR.parent)).resolve()

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
DATA_DIR = ROOT / "data"
PROJECT_INFO_PATH = ROOT / "project_info.json"

# Filled by the entry point with the registered st.Page objects (used for programmatic navigation).
NAV_PAGES: dict = {}

# Artifact registry: key -> (path, human label, what it is used for)
ARTIFACTS: dict[str, dict] = {
    "binary_model": {
        "path": MODELS_DIR / "cicids_binary_model.joblib",
        "label": "Binary model",
        "purpose": "Normal vs. attack prediction",
    },
    "multiclass_model": {
        "path": MODELS_DIR / "cicids_multiclass_model.joblib",
        "label": "Multiclass model",
        "purpose": "Attack-category prediction",
    },
    "features": {
        "path": MODELS_DIR / "feature_columns.json",
        "label": "Feature list",
        "purpose": "Feature names and order expected by the models",
    },
    "classes": {
        "path": MODELS_DIR / "multiclass_classes.json",
        "label": "Multiclass labels",
        "purpose": "Class names for the multiclass model",
    },
    "binary_report": {
        "path": REPORTS_DIR / "cicids_binary_results.csv",
        "label": "Binary report",
        "purpose": "Binary evaluation results",
    },
    "multiclass_report": {
        "path": REPORTS_DIR / "cicids_multiclass_results.csv",
        "label": "Multiclass report",
        "purpose": "Multiclass evaluation results",
    },
}

PROJECT_TITLE = (
    "An Explainable, Concept-Drift-Aware Machine Learning Framework "
    "for Real-Time Network Intrusion Detection"
)
APP_TITLE = "Explainable Network Intrusion Detection System"
APP_SUBTITLE = "Intelligent, Adaptive and Explainable AI for Detecting Evolving Cyber Threats."

_DEFAULT_INFO = {
    "developer": "SHREYAS_MK",
    "usn": "ENG24AM0289",
    "university": "Dayananda Sagar University, Bengaluru",   # from the paper; override in project_info.json
    "department": "Dept. of CSE (AI & ML)",
    "guide": "",
    "team_members": ["V. Rohith", "Shreyas M. K.", "Siddharth B", "R. K. Karthik Kashyap", "Ankitha Takkar"],
    "timeline": [],
}


def load_project_info() -> dict:
    """Merge defaults with ``project_info.json`` (if present). Never raises."""
    info = dict(_DEFAULT_INFO)
    try:
        if PROJECT_INFO_PATH.exists():
            with open(PROJECT_INFO_PATH, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                info.update({k: v for k, v in data.items() if v not in (None, "", [])})  # blanks keep the defaults
    except (OSError, ValueError):
        pass
    return info
''',
    "utils.prediction": r'''
"""Prediction logic (no UI code).

Works with whatever the saved models expose (``classes_``, ``predict_proba``)
and never fabricates outputs: if a model cannot provide probabilities the
confidence columns are NaN instead of invented numbers.
"""
from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

BENIGN_TOKENS = {
    "benign", "normal", "normal traffic", "0", "0.0", "false", "no attack", "legitimate", "legit", "none",
}


def is_benign(label: Any) -> bool:
    """True when a class label denotes normal / benign traffic."""
    return str(label).strip().lower() in BENIGN_TOKENS


@contextmanager
def _quiet_feature_name_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*feature names.*")
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        yield


# --------------------------------------------------------------------------- #
# Label handling
# --------------------------------------------------------------------------- #
def binary_values(model: Any) -> tuple[Any, Any]:
    """Return ``(benign_value, attack_value)`` as stored in ``model.classes_``."""
    classes = list(getattr(model, "classes_", []))
    if len(classes) != 2:
        raise ValueError(
            f"The binary model exposes {len(classes)} classes; exactly 2 are required."
        )
    flags = [is_benign(c) for c in classes]
    if sum(flags) == 1:
        b = flags.index(True)
        return classes[b], classes[1 - b]
    # Ambiguous naming: fall back to the conventional 0 = normal, 1 = attack ordering.
    return classes[0], classes[1]


def class_names(model: Any, json_classes: list[str] | None) -> list[str]:
    """Human-readable class names aligned with ``model.classes_`` (and predict_proba columns)."""
    raw = list(getattr(model, "classes_", []))
    if not raw:
        raise ValueError("The multiclass model does not expose classes_.")
    numeric = all(isinstance(c, (int, np.integer)) and not isinstance(c, (bool, np.bool_)) for c in raw)
    if numeric:
        if json_classes:
            if len(json_classes) != len(raw):
                raise ValueError(
                    f"multiclass_classes.json lists {len(json_classes)} classes but the model "
                    f"was trained with {len(raw)}."
                )
            if min(raw) < 0 or max(raw) >= len(json_classes):
                raise ValueError("Model class indices fall outside multiclass_classes.json.")
            return [str(json_classes[int(c)]) for c in raw]
        return [str(c) for c in raw]
    names = [str(c) for c in raw]
    if json_classes and set(names) != {str(c) for c in json_classes}:
        raise ValueError(
            "Class labels in the model do not match multiclass_classes.json "
            f"(model: {sorted(names)[:5]}…, json: {sorted(map(str, json_classes))[:5]}…)."
        )
    return names


def _proba(model: Any, X: pd.DataFrame) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    with _quiet_feature_name_warnings():
        return np.asarray(model.predict_proba(X))


# --------------------------------------------------------------------------- #
# Predictors
# --------------------------------------------------------------------------- #
def predict_binary(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    """Binary prediction. Columns: is_attack, binary_prediction, attack_probability, binary_confidence."""
    benign_v, attack_v = binary_values(model)
    classes = list(model.classes_)
    proba = _proba(model, X)
    if proba is not None:
        pred_vals = np.asarray(classes, dtype=object)[proba.argmax(axis=1)]
        attack_col = classes.index(attack_v)
        attack_p = proba[:, attack_col]
        confidence = proba.max(axis=1)
    else:
        with _quiet_feature_name_warnings():
            pred_vals = np.asarray(model.predict(X), dtype=object)
        attack_p = np.full(len(X), np.nan)
        confidence = np.full(len(X), np.nan)
    is_attack = np.array([v == attack_v for v in pred_vals], dtype=bool)
    return pd.DataFrame(
        {
            "is_attack": is_attack,
            "binary_prediction": np.where(is_attack, "Attack", "Normal"),
            "attack_probability": attack_p,
            "binary_confidence": confidence,
        },
        index=X.index,
    )


def predict_multiclass(
    model: Any, X: pd.DataFrame, json_classes: list[str] | None, top_k: int = 3
) -> tuple[pd.DataFrame, np.ndarray | None, list[str]]:
    """Multiclass prediction. Returns (table, probability matrix or None, class names)."""
    names = class_names(model, json_classes)
    proba = _proba(model, X)
    if proba is not None:
        if proba.shape[1] != len(names):
            raise ValueError("Probability matrix width does not match the number of classes.")
        idx = proba.argmax(axis=1)
        pred = np.asarray(names, dtype=object)[idx]
        conf = proba.max(axis=1)
        order = np.argsort(-proba, axis=1)[:, : top_k + 1]
        alts = []
        for r in range(len(X)):
            parts = [
                f"{names[c]} ({proba[r, c]:.1%})" for c in order[r] if c != idx[r]
            ][: top_k - 1]
            alts.append(", ".join(parts))
    else:
        with _quiet_feature_name_warnings():
            raw = np.asarray(model.predict(X))
        mapping = dict(zip(list(model.classes_), names))
        pred = np.asarray([mapping.get(v, str(v)) for v in raw], dtype=object)
        conf = np.full(len(X), np.nan)
        alts = [""] * len(X)
    table = pd.DataFrame(
        {
            "multiclass_prediction": pred,
            "multiclass_confidence": conf,
            "alternative_categories": alts,
        },
        index=X.index,
    )
    return table, proba, names


@dataclass
class PredictionResult:
    table: pd.DataFrame
    binary_proba: np.ndarray | None = None
    multi_proba: np.ndarray | None = None
    multi_names: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def predict_all(bundle: Any, X: pd.DataFrame) -> PredictionResult:
    """Run every available saved model on ``X`` (already validated, features in JSON order)."""
    parts: list[pd.DataFrame] = []
    notes: list[str] = []
    res = PredictionResult(table=pd.DataFrame(index=X.index))

    if bundle.binary_model is not None:
        Xb = X[bundle.order_for("binary")]
        b = predict_binary(bundle.binary_model, Xb)
        parts.append(b)
        res.binary_proba = _proba(bundle.binary_model, Xb)
    if bundle.multiclass_model is not None:
        Xm = X[bundle.order_for("multiclass")]
        m, proba, names = predict_multiclass(bundle.multiclass_model, Xm, bundle.classes)
        parts.append(m)
        res.multi_proba, res.multi_names = proba, names

    if not parts:
        raise RuntimeError("No usable saved model is available.")

    table = pd.concat(parts, axis=1)

    verdict, agree = [], []
    for i in range(len(table)):
        row = table.iloc[i]
        has_b = "is_attack" in table.columns
        has_m = "multiclass_prediction" in table.columns
        if has_b and has_m:
            m_attack = not is_benign(row["multiclass_prediction"])
            agree.append(bool(row["is_attack"]) == m_attack)
            if row["is_attack"]:
                verdict.append(row["multiclass_prediction"] if m_attack else "Attack (category undecided)")
            else:
                verdict.append("Normal")
        elif has_b:
            verdict.append("Attack" if row["is_attack"] else "Normal")
            agree.append(np.nan)
        else:
            verdict.append(row["multiclass_prediction"])
            agree.append(np.nan)
    table.insert(0, "final_verdict", verdict)
    if "is_attack" in table.columns and "multiclass_prediction" in table.columns:
        table["models_agree"] = agree
        n_dis = int((table["models_agree"] == False).sum())  # noqa: E712
        if n_dis:
            notes.append(
                f"{n_dis} record(s) received contradictory binary and multiclass outputs; "
                "the binary model decides normal vs. attack, the multiclass model supplies the category."
            )
    if bundle.binary_model is None:
        notes.append("Binary model unavailable - verdict derived from the multiclass model only.")
    if bundle.multiclass_model is None:
        notes.append("Multiclass model unavailable - attack category cannot be reported.")
    res.table, res.notes = table, notes
    return res
''',
    "utils.preprocessing": r'''
"""Input validation and cleaning for manual and CSV inputs (no UI code)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FLOAT32_MAX = float(np.finfo(np.float32).max)  # tree models cast to float32 internally


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from headers (CIC-IDS2017 files have leading spaces); drop duplicate headers."""
    out = df.copy()
    out.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in out.columns]
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()]
    return out


@dataclass
class PreparedData:
    X: pd.DataFrame = field(default_factory=pd.DataFrame)
    valid_rows: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))  # 0-based original rows
    dropped: pd.DataFrame = field(default_factory=pd.DataFrame)  # row number + reason
    n_input: int = 0
    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and len(self.X) > 0


def prepare_batch(df: pd.DataFrame, features: list[str]) -> PreparedData:
    """Validate a CSV frame against the expected features and return clean rows only."""
    res = PreparedData(n_input=len(df))
    if df is None or df.empty:
        res.errors.append("The uploaded file contains no rows.")
        return res

    df = normalize_columns(df)
    present = set(df.columns)
    missing = [f for f in features if f not in present]

    # Case-insensitive rescue for headers that differ only by capitalisation.
    if missing:
        lower_map = {c.lower(): c for c in df.columns}
        rename = {lower_map[m.lower()]: m for m in missing if m.lower() in lower_map}
        if rename:
            df = df.rename(columns=rename)
            res.messages.append(f"{len(rename)} column header(s) matched case-insensitively.")
            present = set(df.columns)
            missing = [f for f in features if f not in present]

    res.missing_columns = missing
    res.extra_columns = [c for c in df.columns if c not in set(features)]
    if missing:
        shown = ", ".join(missing[:8]) + (" …" if len(missing) > 8 else "")
        res.errors.append(f"{len(missing)} required column(s) are missing: {shown}")
        return res
    if res.extra_columns:
        res.messages.append(
            f"{len(res.extra_columns)} extra column(s) (e.g. labels) were ignored for prediction."
        )

    raw = df[features]
    orig_na = raw.isna()
    num = raw.apply(pd.to_numeric, errors="coerce")
    non_numeric = num.isna() & ~orig_na
    missing_val = orig_na
    inf = pd.DataFrame(np.isinf(num.to_numpy(dtype=float, na_value=np.nan)), index=num.index, columns=num.columns)
    oor = (num.abs() > FLOAT32_MAX) & ~inf

    bad = (non_numeric | missing_val | inf | oor)
    bad_rows = bad.any(axis=1).to_numpy()

    reasons = []
    for pos in np.flatnonzero(bad_rows)[:2000]:
        for label, mask in (
            ("non-numeric value", non_numeric),
            ("missing value", missing_val),
            ("infinite value", inf),
            ("out-of-range value", oor),
        ):
            row = mask.iloc[pos]
            if row.any():
                reasons.append({"row": int(pos) + 1, "reason": f"{label} in '{row.idxmax()}'"})
                break
    res.dropped = pd.DataFrame(reasons)

    keep = ~bad_rows
    res.valid_rows = np.flatnonzero(keep)
    res.X = num.loc[keep].astype("float64")
    res.X.index = res.valid_rows  # original 0-based row positions

    n_bad = int(bad_rows.sum())
    if n_bad:
        res.messages.append(
            f"{n_bad} of {len(df)} row(s) were excluded (missing, non-numeric, infinite or out-of-range values)."
        )
    if len(res.X) == 0:
        res.errors.append("No valid rows remain after validation.")
    return res


def prepare_manual(raw: dict[str, str], features: list[str]) -> tuple[pd.DataFrame | None, list[str]]:
    """Validate manually typed values. Returns (1-row frame in feature order, error list)."""
    errors: list[str] = []
    values: dict[str, float] = {}
    for f in features:
        txt = str(raw.get(f, "")).strip()
        if txt == "":
            errors.append(f"'{f}': value is missing.")
            continue
        try:
            v = float(txt.replace(",", ""))
        except ValueError:
            errors.append(f"'{f}': '{txt}' is not a valid number.")
            continue
        if np.isnan(v) or np.isinf(v):
            errors.append(f"'{f}': value must be finite.")
        elif abs(v) > FLOAT32_MAX:
            errors.append(f"'{f}': value is outside the supported numeric range.")
        else:
            values[f] = v
    if errors:
        return None, errors
    return pd.DataFrame([values], columns=features), []
''',
    "utils.metrics": r'''
"""Reading evaluation reports (tolerant of different column names) and computing metrics."""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

from utils.config import ARTIFACTS

# canonical column -> accepted (normalised) aliases
ALIASES: dict[str, list[str]] = {
    "model": ["model", "modelname", "classifier", "algorithm", "name", "estimator", "method"],
    "accuracy": ["accuracy", "acc", "testaccuracy", "accuracyscore"],
    "precision": ["precision", "prec", "precisionweighted", "weightedprecision", "precisionmacro", "macroprecision", "precisionscore"],
    "recall": ["recall", "tpr", "sensitivity", "recallweighted", "weightedrecall", "recallmacro", "macrorecall", "recallscore"],
    "f1": ["f1", "f1score", "fscore", "f1weighted", "weightedf1", "f1macro", "macrof1", "fmeasure"],
    "roc_auc": ["rocauc", "auc", "aucroc", "roc", "rocaucovr", "rocaucscore", "aucscore", "rocaucweighted", "auroc"],
    "fpr": ["fpr", "falsepositiverate", "falsealarmrate"],
    "mcc": ["mcc", "matthewscorrcoef", "matthewscorrelationcoefficient", "matthewscorrelation"],
    "train_time": ["traintime", "trainingtime", "trainingtimes", "fittime", "trainseconds", "trainingseconds", "traintimes", "traintimesec", "trainingtimesec"],
    "predict_time": ["predicttime", "predictiontime", "inferencetime", "predicttimes", "predictionseconds", "testtime", "predicttimesec", "predictiontimesec"],
    "tn": ["tn", "truenegative", "truenegatives"],
    "fp": ["fp", "falsepositive", "falsepositives"],
    "fn": ["fn", "falsenegative", "falsenegatives"],
    "tp": ["tp", "truepositive", "truepositives"],
    "confusion_matrix": ["confusionmatrix", "cm", "confmatrix", "confusion"],
}
METRIC_COLS = ["accuracy", "precision", "recall", "f1", "roc_auc", "fpr", "mcc"]
FRACTION_COLS = ["accuracy", "precision", "recall", "f1", "roc_auc", "fpr"]
METRIC_LABELS = {
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1 Score",
    "roc_auc": "ROC-AUC", "fpr": "False Positive Rate", "mcc": "MCC",
    "train_time": "Training time (s)", "predict_time": "Prediction time (s)",
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


@dataclass
class Report:
    df: pd.DataFrame | None = None
    error: str | None = None
    path: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.df is not None and not self.df.empty


def _standardise(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    df = raw.copy()
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed:.*") | (df.columns == df.columns[0])]

    # Transposed layout (metrics as rows, models as columns)?
    first = df.iloc[:, 0].astype(str).map(_norm)
    if not any(_norm(c) in ALIASES["accuracy"] for c in df.columns) and first.isin(ALIASES["accuracy"]).any():
        df = df.set_index(df.columns[0]).T.reset_index().rename(columns={"index": "model"})
        df.columns.name = None
        notes.append("Report was transposed (metrics were rows).")

    mapping: dict[str, str] = {}
    for canon, names in ALIASES.items():
        for col in df.columns:
            if col in mapping:
                continue
            if _norm(col) in names:
                mapping[col] = canon
                break
    df = df.rename(columns=mapping)
    df = df.loc[:, ~df.columns.duplicated()]

    if "model" not in df.columns:
        first_col = df.columns[0]
        if df[first_col].dtype == object:
            df = df.rename(columns={first_col: "model"})
        else:
            df.insert(0, "model", [f"Model {i + 1}" for i in range(len(df))])
            notes.append("No model-name column found; rows were numbered.")
    df["model"] = df["model"].astype(str).str.strip()

    for c in METRIC_COLS + ["train_time", "predict_time", "tn", "fp", "fn", "tp"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in FRACTION_COLS:
        if c in df.columns and df[c].notna().any() and df[c].max() > 1.0 and df[c].max() <= 100.0:
            df[c] = df[c] / 100.0
            notes.append(f"'{c}' looked like a percentage and was rescaled to 0–1.")

    # FPR is derived only when the confusion-matrix counts are really present.
    if "fpr" not in df.columns and {"fp", "tn"} <= set(df.columns):
        df["fpr"] = df["fp"] / (df["fp"] + df["tn"]).replace(0, np.nan)
        notes.append("False Positive Rate was computed from the FP and TN columns of the report.")

    df["is_adaptive"] = df["model"].str.contains(r"adaptive|drift|river|online", case=False, regex=True)
    return df.reset_index(drop=True), notes


def load_report(kind: str) -> Report:
    """kind: 'binary' | 'multiclass'."""
    path = ARTIFACTS[f"{kind}_report"]["path"]
    rep = Report(path=f"{path.parent.name}/{path.name}")
    if not path.exists():
        rep.error = f"{rep.path} was not found."
        return rep
    try:
        raw = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        rep.error = f"{rep.path} could not be read ({type(exc).__name__})."
        return rep
    if raw.empty:
        rep.error = f"{rep.path} is empty."
        return rep
    df, notes = _standardise(raw)
    if not any(c in df.columns for c in METRIC_COLS):
        rep.error = (
            f"{rep.path} has no recognisable metric columns "
            f"(found: {', '.join(map(str, raw.columns[:8]))})."
        )
        return rep
    rep.df, rep.notes = df, notes
    return rep


def confusion_from_row(row: pd.Series) -> tuple[np.ndarray, list[str]] | None:
    """Confusion matrix from a report row (TN/FP/FN/TP columns or a serialised matrix)."""
    if all(k in row.index and pd.notna(row[k]) for k in ("tn", "fp", "fn", "tp")):
        return np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=float), ["Normal", "Attack"]
    cell = row.get("confusion_matrix")
    if isinstance(cell, str) and cell.strip():
        for parser in (json.loads, ast.literal_eval):
            try:
                m = np.array(parser(re.sub(r"\s+", ",", cell.strip()) if parser is json.loads else cell), dtype=float)
                if m.ndim == 2 and m.shape[0] == m.shape[1]:
                    return m, [str(i) for i in range(m.shape[0])]
            except (ValueError, SyntaxError, TypeError):
                continue
    return None


# --------------------------------------------------------------------------- #
# Live evaluation on labelled data (clearly distinct from the reported experiments)
# --------------------------------------------------------------------------- #
def evaluate_predictions(
    y_true, y_pred, labels: list | None = None, proba: np.ndarray | None = None, positive=None
) -> dict:
    """Standard metrics. Binary: positive class metrics. Multiclass: weighted averages + macro FPR."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred), key=str)
    binary = len(classes) == 2
    out: dict = {"n": int(len(y_true)), "accuracy": float(accuracy_score(y_true, y_pred))}
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    if binary:
        pos = positive if positive is not None else classes[1]
        out["precision"] = float(precision_score(y_true, y_pred, pos_label=pos, zero_division=0))
        out["recall"] = float(recall_score(y_true, y_pred, pos_label=pos, zero_division=0))
        out["f1"] = float(f1_score(y_true, y_pred, pos_label=pos, zero_division=0))
        tn, fp, fn, tp = (cm.ravel() if pos == classes[1] else cm[::-1, ::-1].ravel())
        out["fpr"] = float(fp / (fp + tn)) if (fp + tn) else float("nan")
        if proba is not None and len(set(y_true)) == 2:
            try:
                out["roc_auc"] = float(roc_auc_score(np.asarray(y_true) == pos, proba))
            except ValueError:
                pass
    else:
        out["precision"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        out["recall"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        out["f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        fprs = []
        total = cm.sum()
        for i in range(len(classes)):
            fp = cm[:, i].sum() - cm[i, i]
            tn = total - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            if (fp + tn) > 0:
                fprs.append(fp / (fp + tn))
        out["fpr"] = float(np.mean(fprs)) if fprs else float("nan")
        if proba is not None:
            try:
                present = [c for c in classes if c in set(y_true)]
                if len(present) == len(classes):
                    out["roc_auc"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="weighted", labels=classes))
            except ValueError:
                pass
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    out["confusion_matrix"] = cm
    out["labels"] = [str(c) for c in classes]
    return out
''',
    "utils.data_loader": r'''
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

LABEL_LIKE = {"label", "attack_cat", "attack_category", "attack type", "attack_type", "class", "id", "_attack_type", "_source_file"}


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
    s = s.str.replace(r"[�–—ï¿½]+", "-", regex=True)
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
    feats = [c for c in df.columns if c.lower() not in LABEL_LIKE]
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
    return df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def load_scenario_pool(dataset_name: str, sig: tuple, counts: dict, features: tuple, cap_per_class: int = 40_000, seed: int = 7):
    """Class-capped pool for the drift lab: *all* records of rare categories, a uniform sample of big ones.

    ``counts`` are the exact per-category counts of the dataset (from ``load_disk_dataset``). Returns a
    DataFrame (feature columns + ``_attack_type`` + ``_source_file``) or an error string.
    """
    files = [Path(s[0]) for s in sig]
    rng = np.random.default_rng(seed)
    rate = {c: min(1.0, cap_per_class / max(int(n), 1)) for c, n in counts.items()}
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
                    cols = [f for f in features if f in chunk.columns]
                    part = chunk.loc[keep, cols].apply(pd.to_numeric, errors="coerce").astype("float32")
                    part["_attack_type"] = at.loc[keep].astype(str).to_numpy()
                    part["_source_file"] = p.name
                    parts.append(part)
        if not parts:
            return "The dataset files contain no rows."
        pool = pd.concat(parts, ignore_index=True)
    except Exception as exc:  # noqa: BLE001
        return f"Dataset could not be read ({type(exc).__name__}: {str(exc)[:140]})."
    feats = [f for f in features if f in pool.columns]
    pool[feats] = pool[feats].replace([np.inf, -np.inf], np.nan)
    pool = pool.dropna(subset=feats).reset_index(drop=True)
    return pool
''',
    "utils.sources": r'''
"""Turning dataset files / uploads into model-ready, label-aligned records (no UI code)."""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from utils.data_loader import DATASETS, attack_type_series, derive_truth
from utils.preprocessing import PreparedData, normalize_columns, prepare_batch

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


@dataclass
class Source:
    X: pd.DataFrame  # validated, model feature order, reset 0..n-1 index
    labels: pd.Series | None  # raw category per row (string), aligned with X
    name: str
    origin: str  # "disk" | "upload"
    notes: list[str] = field(default_factory=list)
    files: pd.Series | None = None  # per-row source file name (disk)

    @property
    def y_attack(self) -> np.ndarray | None:
        return None if self.labels is None else derive_truth(self.labels)


def infer_labels(df: pd.DataFrame) -> pd.Series | None:
    """Attack category per row from a label column (CIC-IDS 'Label' or UNSW 'attack_cat'/'label')."""
    cols = {c.lower() for c in df.columns}
    cic_cols = {c.lower() for c in DATASETS["CIC-IDS2017"]["label_candidates"]}
    if (cols & cic_cols) and "attack_cat" not in cols:
        s = attack_type_series(df, "CIC-IDS2017")
        if s is not None and not pd.to_numeric(s, errors="coerce").notna().all():
            return s
    if "attack_cat" in cols or "label" in cols:
        return attack_type_series(df, "UNSW-NB15")
    return None


def chronological_positions(files: pd.Series) -> np.ndarray | None:
    """Stable ordering by weekday found in file names (Monday..Friday); None if no weekday present."""
    def day(name: str) -> int:
        low = str(name).lower()
        return next((i for i, d in enumerate(WEEKDAYS) if d in low), -1)
    d = files.map(day).to_numpy()
    if (d < 0).all():
        return None
    d = np.where(d < 0, 99, d)
    return np.argsort(d, kind="stable")


def build_source(
    df: pd.DataFrame, features: list[str], name: str, origin: str, labels: pd.Series | None = None,
    files: pd.Series | None = None, chronological: bool = False, max_rows: int | None = None,
) -> tuple[Source | None, PreparedData]:
    """Validate ``df`` against the features and align labels with the surviving rows."""
    df = normalize_columns(df)
    if labels is None:
        labels = infer_labels(df)
    prep = prepare_batch(df, features)
    if not prep.ok:
        return None, prep
    idx = prep.valid_rows
    X = prep.X[features].reset_index(drop=True)
    lab = None if labels is None else labels.iloc[idx].reset_index(drop=True)
    fil = None if files is None else files.iloc[idx].reset_index(drop=True)
    notes = list(prep.messages)
    if chronological and fil is not None:
        order = chronological_positions(fil)
        if order is not None:
            X = X.iloc[order].reset_index(drop=True)
            lab = None if lab is None else lab.iloc[order].reset_index(drop=True)
            fil = fil.iloc[order].reset_index(drop=True)
            notes.append("Records ordered chronologically by weekday in the file names (Monday → Friday).")
    if max_rows and len(X) > max_rows:
        X = X.iloc[:max_rows]
        lab = None if lab is None else lab.iloc[:max_rows]
        fil = None if fil is None else fil.iloc[:max_rows]
        notes.append(f"Stream limited to the first {max_rows:,} valid records.")
    return Source(X=X, labels=lab, name=name, origin=origin, notes=notes, files=fil), prep


def read_uploaded_csv(raw: bytes) -> pd.DataFrame:
    """Parse uploaded CSV bytes. Raises ValueError with a friendly message on failure."""
    if not raw or not raw.strip():
        raise ValueError("The uploaded file is empty.")
    try:
        df = pd.read_csv(io.BytesIO(raw), low_memory=False, encoding_errors="replace")
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The uploaded file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError("The file could not be parsed as CSV (inconsistent rows or delimiter).") from exc
    except UnicodeDecodeError as exc:
        raise ValueError("The file is not a text CSV (encoding could not be decoded).") from exc
    if df.shape[1] < 2:
        raise ValueError("The file has fewer than 2 columns - is it a comma-separated CSV with a header row?")
    return df
''',
    "utils.model_loader": r'''
"""Loading and validation of saved artifacts (models + JSON metadata)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import streamlit as st

from utils.config import ARTIFACTS
from utils.prediction import binary_values, class_names


@dataclass
class ModelBundle:
    binary_model: Any = None
    multiclass_model: Any = None
    features: list[str] | None = None
    classes: list[str] | None = None
    binary_features: list[str] | None = None
    multiclass_features: list[str] | None = None
    problems: list[str] = field(default_factory=list)  # things that disable functionality
    warnings: list[str] = field(default_factory=list)  # things worth knowing

    @property
    def can_predict(self) -> bool:
        return bool(self.features) and (self.binary_model is not None or self.multiclass_model is not None)

    def order_for(self, kind: str) -> list[str]:
        """Exact feature order expected by the given model ('binary' | 'multiclass')."""
        order = self.binary_features if kind == "binary" else self.multiclass_features
        return order or list(self.features or [])

    def model_for(self, kind: str) -> Any:
        return self.binary_model if kind == "binary" else self.multiclass_model


# --------------------------------------------------------------------------- #
def artifact_status() -> list[dict]:
    """Presence of every expected artifact (for readiness panels)."""
    out = []
    for key, meta in ARTIFACTS.items():
        p: Path = meta["path"]
        out.append(
            {
                "key": key,
                "label": meta["label"],
                "purpose": meta["purpose"],
                "path": p,
                "rel": f"{p.parent.name}/{p.name}",
                "exists": p.exists(),
            }
        )
    return out


def _signature() -> tuple:
    """Changes whenever an artifact is added/modified so caches refresh automatically."""
    sig = []
    for meta in ARTIFACTS.values():
        p: Path = meta["path"]
        sig.append((p.name, p.stat().st_mtime_ns if p.exists() else None))
    return tuple(sig)


def read_json(path: Path) -> tuple[Any, str | None]:
    if not path.exists():
        return None, f"{path.parent.name}/{path.name} not found."
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh), None
    except (OSError, ValueError) as exc:
        return None, f"{path.name} could not be read ({type(exc).__name__})."


def _parse_features(data: Any) -> tuple[list[str] | None, str | None]:
    if isinstance(data, dict):
        data = data.get("features") or data.get("feature_columns") or data.get("columns")
    if not isinstance(data, list) or not data:
        return None, "feature_columns.json must contain a non-empty list of feature names."
    feats = [str(x).strip() for x in data]
    if len(set(feats)) != len(feats):
        return None, "feature_columns.json contains duplicate feature names."
    if any(not f for f in feats):
        return None, "feature_columns.json contains an empty feature name."
    return feats, None


def _parse_classes(data: Any) -> tuple[list[str] | None, str | None]:
    if isinstance(data, dict):
        try:
            items = sorted(data.items(), key=lambda kv: int(kv[0]))
            return [str(v) for _, v in items], None
        except (TypeError, ValueError):
            data = data.get("classes") or list(data.values())
    if not isinstance(data, list) or not data:
        return None, "multiclass_classes.json must contain a non-empty list (or index→name mapping)."
    return [str(x) for x in data], None


def _load_model(path: Path, label: str) -> tuple[Any, str | None]:
    if not path.exists():
        return None, f"{label} file not found ({path.parent.name}/{path.name})."
    try:
        model = joblib.load(path)
    except Exception as exc:  # unpickling can fail in many ways (version mismatch, missing module…)
        msg = str(exc).splitlines()[0][:160] if str(exc) else ""
        return None, (
            f"{label} could not be loaded ({type(exc).__name__}: {msg}). "
            "This usually means the model was saved with different library versions."
        )
    if not hasattr(model, "predict"):
        return None, f"{label} was loaded but has no predict() method (unsupported object type)."
    return model, None


def _check_features(model: Any, features: list[str], label: str) -> tuple[list[str] | None, str | None, str | None]:
    """Validate model input dimension / names. Returns (order, problem, warning)."""
    n_in = getattr(model, "n_features_in_", None)
    if n_in is not None and int(n_in) != len(features):
        return None, (
            f"{label} expects {int(n_in)} features but feature_columns.json lists {len(features)}."
        ), None
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        names = [str(n) for n in names]
        if set(names) != set(features):
            missing = sorted(set(names) - set(features))[:3]
            return None, f"{label} was trained on different feature names (e.g. {missing})."
        if names != features:
            return names, None, (
                f"{label}: feature order in feature_columns.json differs from the order the model "
                "was trained with; the model's own order is used."
            )
        return names, None, None
    return list(features), None, None


@st.cache_resource(show_spinner="Loading saved models…")
def _load_bundle(_sig: tuple) -> ModelBundle:
    b = ModelBundle()

    data, err = read_json(ARTIFACTS["features"]["path"])
    if err:
        b.problems.append(err)
    else:
        b.features, err = _parse_features(data)
        if err:
            b.problems.append(err)

    data, err = read_json(ARTIFACTS["classes"]["path"])
    if err:
        b.warnings.append(err)
    else:
        b.classes, err = _parse_classes(data)
        if err:
            b.problems.append(err)

    for kind in ("binary", "multiclass"):
        key = f"{kind}_model"
        label = f"{kind.capitalize()} model"
        model, err = _load_model(ARTIFACTS[key]["path"], label)
        if err:
            b.problems.append(err)
            continue
        if not b.features:
            b.problems.append(f"{label} loaded, but it cannot be used without a valid feature_columns.json.")
            continue
        order, prob, warn = _check_features(model, b.features, label)
        if prob:
            b.problems.append(prob)
            continue
        if warn:
            b.warnings.append(warn)
        try:
            if kind == "binary":
                binary_values(model)
            else:
                class_names(model, b.classes)
        except ValueError as exc:
            b.problems.append(f"{label}: {exc}")
            continue
        if kind == "binary":
            b.binary_model, b.binary_features = model, order
        else:
            b.multiclass_model, b.multiclass_features = model, order
    return b


def get_bundle() -> ModelBundle:
    """Cached bundle of models + metadata; refreshes when artifact files change."""
    return _load_bundle(_signature())
''',
    "utils.explainability": r'''
"""SHAP and LIME engines (no UI code, only st.cache_resource for explainers).

The explainer is chosen from the *actual* model architecture:

* tree models (RandomForest, DecisionTree, ExtraTrees, GradientBoosting, XGBoost) -> ``TreeExplainer``
* LogisticRegression (optionally behind a scaler pipeline) -> ``LinearExplainer``
* anything else exposing ``predict_proba`` -> model-agnostic ``KernelExplainer`` (slow, capped)

Every failure raises ``ExplainError`` with a human-readable message - the UI shows it instead of
a made-up explanation.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from utils.prediction import binary_values

try:  # optional, only used for type detection
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = ()  # type: ignore[assignment]

TREE_TYPES = tuple(t for t in (
    RandomForestClassifier, ExtraTreesClassifier, DecisionTreeClassifier, GradientBoostingClassifier, XGBClassifier
) if t)


class ExplainError(Exception):
    """Raised when an explanation cannot be produced honestly."""


def _final_estimator(model: Any):
    steps = getattr(model, "steps", None)
    return steps[-1][1] if steps else model


def _transform(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Apply all pipeline steps before the final estimator (identity for plain models)."""
    steps = getattr(model, "steps", None)
    if not steps or len(steps) == 1:
        return X.to_numpy(dtype=float)
    pre = model[:-1]
    out = pre.transform(X)
    out = out.toarray() if hasattr(out, "toarray") else np.asarray(out)
    if out.shape[1] != X.shape[1]:
        raise ExplainError(
            "The preprocessing pipeline changes the number of features, so SHAP values cannot be "
            "mapped back to the original network-flow features."
        )
    return out.astype(float)


def explainer_kind(model: Any) -> str:
    est = _final_estimator(model)
    if isinstance(est, TREE_TYPES):
        return "tree"
    if isinstance(est, LogisticRegression):
        return "linear"
    if hasattr(model, "predict_proba"):
        return "kernel"
    return "unsupported"


KIND_LABEL = {
    "tree": "TreeExplainer (exact, fast - tree ensembles)",
    "linear": "LinearExplainer (exact - linear model)",
    "kernel": "KernelExplainer (model-agnostic approximation, slower)",
}


@dataclass
class ShapEngine:
    kind: str
    explainer: Any
    model: Any
    feature_names: list[str]
    classes: list[Any]  # model.classes_

    @property
    def label(self) -> str:
        return KIND_LABEL.get(self.kind, self.kind)


@st.cache_resource(show_spinner=False, max_entries=6)
def _build_engine(_model: Any, kind_key: str, _background: pd.DataFrame, bg_sig: tuple, feature_names: tuple) -> ShapEngine:
    model = _model
    kind = explainer_kind(model)
    if kind == "unsupported":
        raise ExplainError("This model type has no predict_proba and is not supported by the explainers.")
    names = list(feature_names)
    classes = list(getattr(model, "classes_", []))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if kind == "tree":
                expl = shap.TreeExplainer(_final_estimator(model))
            elif kind == "linear":
                bg = _transform(model, _background.head(500))
                expl = shap.LinearExplainer(_final_estimator(model), bg)
            else:
                bg = shap.sample(_background[names], min(40, len(_background)), random_state=0)
                f = lambda a: model.predict_proba(pd.DataFrame(a, columns=names))  # noqa: E731
                expl = shap.KernelExplainer(f, bg)
    except Exception as exc:  # noqa: BLE001
        raise ExplainError(f"SHAP could not build an explainer for this model ({type(exc).__name__}: {str(exc)[:140]}).") from exc
    return ShapEngine(kind, expl, model, names, classes)


def get_engine(model: Any, kind_key: str, background: pd.DataFrame, features: list[str]) -> ShapEngine:
    sig = (len(background), int(pd.util.hash_pandas_object(background.head(50), index=False).sum() % (2**31)))
    return _build_engine(model, kind_key, background, sig, tuple(features))


def _pick_class(values: Any, class_idx: int, n_classes: int) -> np.ndarray:
    """Normalise SHAP's several output layouts to (n_samples, n_features) for one class."""
    if isinstance(values, list):
        return np.asarray(values[class_idx])
    arr = np.asarray(values)
    if arr.ndim == 3:
        return arr[:, :, class_idx]
    if arr.ndim == 2:
        # single-output models (e.g. XGBoost binary): values describe the positive (2nd) class
        return arr if (n_classes != 2 or class_idx == 1) else -arr
    raise ExplainError(f"Unexpected SHAP output shape {arr.shape}.")


def _base_value(expected: Any, class_idx: int, n_classes: int) -> float:
    e = np.atleast_1d(np.asarray(expected, dtype=float)).ravel()
    if len(e) == 1:
        return float(e[0] if (n_classes != 2 or class_idx == 1) else -e[0])
    return float(e[min(class_idx, len(e) - 1)])


def shap_values(engine: ShapEngine, X: pd.DataFrame, class_idx: int) -> tuple[np.ndarray, float]:
    """SHAP matrix (n, features) and base value for ``class_idx`` (index into model.classes_)."""
    n_classes = max(len(engine.classes), 2)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if engine.kind == "tree":
                try:
                    raw = engine.explainer.shap_values(_transform(engine.model, X), check_additivity=False)
                except TypeError:
                    raw = engine.explainer.shap_values(_transform(engine.model, X))
            elif engine.kind == "linear":
                raw = engine.explainer.shap_values(_transform(engine.model, X))
            else:
                raw = engine.explainer.shap_values(X[engine.feature_names], nsamples=200, silent=True)
        vals = _pick_class(raw, class_idx, n_classes)
        base = _base_value(engine.explainer.expected_value, class_idx, n_classes)
    except ExplainError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ExplainError(f"SHAP computation failed ({type(exc).__name__}: {str(exc)[:160]}).") from exc
    if vals.shape != (len(X), len(engine.feature_names)):
        raise ExplainError(
            f"SHAP output shape {vals.shape} does not match the data ({len(X)} x {len(engine.feature_names)})."
        )
    return vals, base


def global_importance(vals: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Mean |SHAP| per feature, sorted descending."""
    imp = np.abs(vals).mean(axis=0)
    return pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False).reset_index(drop=True)


def local_table(vals_row: np.ndarray, x_row: pd.Series, names: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({"feature": names, "value": x_row[names].to_numpy(float), "contribution": vals_row})
    return df.reindex(df["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def class_index(model: Any, label: Any) -> int:
    return list(model.classes_).index(label)


def binary_attack_index(model: Any) -> int:
    return list(model.classes_).index(binary_values(model)[1])


# --------------------------------------------------------------------------- #
# LIME
# --------------------------------------------------------------------------- #
def lime_explain(
    model: Any, background: pd.DataFrame, x_row: pd.Series, feature_names: list[str],
    class_labels: list[str], class_idx: int, num_features: int = 15, num_samples: int = 2000,
) -> pd.DataFrame:
    """LIME tabular explanation for one record. Returns rows (rule, weight) sorted by |weight|."""
    if not hasattr(model, "predict_proba"):
        raise ExplainError("LIME needs predict_proba, which this model does not provide.")
    if len(background) < 20:
        raise ExplainError("LIME needs at least 20 reference records to learn feature distributions.")
    try:
        from lime.lime_tabular import LimeTabularExplainer

        bg = background[feature_names].to_numpy(float)
        explainer = LimeTabularExplainer(
            bg, feature_names=feature_names, class_names=class_labels, mode="classification",
            discretize_continuous=True, random_state=0,
        )

        def predict_fn(a: np.ndarray) -> np.ndarray:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return model.predict_proba(pd.DataFrame(a, columns=feature_names))

        exp = explainer.explain_instance(
            x_row[feature_names].to_numpy(float), predict_fn, num_features=num_features,
            labels=(class_idx,), num_samples=num_samples,
        )
        items = exp.as_list(label=class_idx)
    except ExplainError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ExplainError(f"LIME failed ({type(exc).__name__}: {str(exc)[:160]}).") from exc
    df = pd.DataFrame(items, columns=["rule", "weight"])
    return df.reindex(df["weight"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# SHAP vs LIME agreement (paper Section IV-E)
# --------------------------------------------------------------------------- #
def _feature_of_rule(rule: str, names: list[str]) -> str | None:
    """LIME returns discretised rules ('Flow IAT Max <= 12.0'); recover the feature (longest name wins)."""
    hits = [n for n in names if n in rule]
    return max(hits, key=len) if hits else None


def agreement_study(model: Any, engine: ShapEngine, background: pd.DataFrame, X: pd.DataFrame, feature_names: list[str],
                    class_idx: int, k: int = 10, num_samples: int = 2000) -> pd.DataFrame:
    """For every row of ``X`` compare the top-k features of TreeSHAP and LIME (Jaccard index) and time both.

    The LIME explainer is built once (its set-up is not part of the per-alert cost), mirroring the paper's
    measurement of the cost of explaining one alert.
    """
    import time

    if not hasattr(model, "predict_proba"):
        raise ExplainError("LIME needs predict_proba, which this model does not provide.")
    try:
        from lime.lime_tabular import LimeTabularExplainer

        lime = LimeTabularExplainer(background[feature_names].to_numpy(float), feature_names=feature_names, mode="classification",
                                    discretize_continuous=True, random_state=0)
    except Exception as exc:  # noqa: BLE001
        raise ExplainError(f"LIME could not be initialised ({type(exc).__name__}: {str(exc)[:140]}).") from exc

    def predict_fn(a: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model.predict_proba(pd.DataFrame(a, columns=feature_names))

    rows = []
    for pos in range(len(X)):
        xr = X.iloc[[pos]]
        t0 = time.perf_counter()
        vals, _ = shap_values(engine, xr, class_idx)
        t_shap = (time.perf_counter() - t0) * 1000
        shap_top = [feature_names[j] for j in np.argsort(-np.abs(vals[0]))[:k]]
        t0 = time.perf_counter()
        try:
            exp = lime.explain_instance(xr.iloc[0][feature_names].to_numpy(float), predict_fn, num_features=k, labels=(class_idx,), num_samples=num_samples)
            items = exp.as_list(label=class_idx)
        except Exception as exc:  # noqa: BLE001
            raise ExplainError(f"LIME failed ({type(exc).__name__}: {str(exc)[:140]}).") from exc
        t_lime = (time.perf_counter() - t0) * 1000
        lime_top = [f for f in (_feature_of_rule(r, feature_names) for r, _ in items) if f][:k]
        a, b = set(shap_top), set(lime_top)
        rows.append({"record": int(X.index[pos]), "jaccard": len(a & b) / max(len(a | b), 1), "shap_ms": t_shap, "lime_ms": t_lime,
                     "shap_top": shap_top, "lime_top": lime_top, "shared": sorted(a & b)})
    return pd.DataFrame(rows)
''',
    "utils.drift_detection": r'''
"""Concept-drift monitoring (River ADWIN) and adaptive learning - no UI code.

Experiment design (prequential "test-then-train" on sequential batches)
-----------------------------------------------------------------------
For every batch of labelled flow records, in stream order:

1. **Incoming data -> prediction.** The *static* model (the saved, frozen binary model) and the
   *adaptive* model both predict the batch.
2. **Error monitoring.** The per-record error of the adaptive system's deployed model
   (1 = wrong, 0 = correct) is fed one value at a time to ``river.drift.ADWIN``.
3. **Drift alert.** When ADWIN reports a statistically significant change in the mean of
   the error stream, a drift event is logged (batch + position inside the batch).
4. **Adaptive retraining.** Strategy ``retrain``: a *fresh, unfitted clone* of the saved model's
   architecture (``sklearn.base.clone`` - no ``partial_fit`` assumed) is fitted on the most recent
   labelled window (optionally plus a replay sample of older labelled records, to limit forgetting)
   and replaces the deployed model; ADWIN is re-initialised. Only a *rise* of the error mean counts
   as drift - a fall means the model improved.
   Strategy ``river``: an incremental River learner (scaler + Hoeffding Adaptive Tree) is trained
   record-by-record (learn_one) after it predicted them, so it adapts continuously.
5. **Updated model evaluation.** Later batches are predicted by the updated model, so accuracy
   before / after each drift can be compared with the never-updated static model.

Assumption (stated in the UI): ground-truth labels arrive with the stream. Real deployments get
labels with delay (analyst feedback, honeypots, retrospective rules). This is a *simulated stream
over recorded flows*, not live packet capture.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from river import compose, drift, preprocessing, tree
from sklearn.base import clone

from utils.metrics import evaluate_predictions
from utils.prediction import binary_values, predict_binary


# --------------------------------------------------------------------------- #
# Synthetic drift (clearly labelled as such in the UI)
# --------------------------------------------------------------------------- #
def inject_synthetic_drift(
    X: pd.DataFrame, start_frac: float, n_features: int, magnitude: float, seed: int = 7
) -> tuple[pd.DataFrame, list[str]]:
    """Covariate shift: from ``start_frac`` of the stream onward, rescale+shift selected features.

    Features are picked among the most variable ones (so the shift matters to the model).
    Labels are *not* changed, only the feature distribution - the relationship between
    flows and labels stays learnable, so an adaptive model can recover.
    """
    rng = np.random.default_rng(seed)
    Xs = X.copy()
    start = int(len(Xs) * start_frac)
    spread = Xs.std(numeric_only=True).replace(0, np.nan).dropna()
    pool = spread.sort_values(ascending=False).index[: max(n_features * 3, n_features)]
    chosen = list(rng.choice(pool, size=min(n_features, len(pool)), replace=False))
    for c in chosen:
        sd = float(spread.get(c, 1.0))
        Xs.iloc[start:, Xs.columns.get_loc(c)] = Xs.iloc[start:][c] * magnitude + sd * (magnitude - 1.0) * 0.5
    return Xs, chosen


# --------------------------------------------------------------------------- #
@dataclass
class DriftConfig:
    batch_size: int = 1000
    max_batches: int = 40
    strategy: str = "retrain"  # "retrain" | "river"
    window_size: int = 4000  # recent labelled records used for retraining
    min_retrain_samples: int = 300
    adwin_delta: float = 0.002
    include_river_online: bool = False  # also run the incremental River learner as a third contender
    cooldown_batches: int = 1  # ignore alerts right after a retrain
    replay_fraction: float = 0.5  # older labelled records replayed in retraining (x window); 0 = recent window only


@dataclass
class DriftRun:
    batches: pd.DataFrame
    events: pd.DataFrame
    summary: dict
    config: DriftConfig
    provenance: str
    notes: list[str] = field(default_factory=list)
    error_stream: np.ndarray | None = None  # adaptive-system per-record error (0/1)
    drift_positions: list[int] = field(default_factory=list)  # global record index of alerts
    overall: dict = field(default_factory=dict)  # full-stream metrics per model (static / adaptive / online)


def _make_river_model():
    return compose.Pipeline(
        preprocessing.StandardScaler(),
        tree.HoeffdingAdaptiveTreeClassifier(grace_period=100, max_depth=12, seed=1),
    )


def _river_predict(model, X: np.ndarray, cols: list[str]) -> np.ndarray:
    out = np.zeros(len(X), dtype=bool)
    for i in range(len(X)):
        out[i] = bool(model.predict_one(dict(zip(cols, X[i]))) or False)
    return out


def _fit_clone(model: Any, X: pd.DataFrame, y_attack: np.ndarray):
    """Fit a fresh clone of the saved model's architecture on the recent labelled window."""
    benign_v, attack_v = binary_values(model)
    y = np.where(y_attack, attack_v, benign_v)
    new = clone(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new.fit(X, y)
    return new


def run_drift_experiment(
    model: Any,
    X: pd.DataFrame,
    y_attack: np.ndarray,
    cfg: DriftConfig,
    provenance: str,
    progress: Callable[[float, str], None] | None = None,
) -> DriftRun:
    """Run the prequential drift experiment. ``X`` is in model feature order, ``y_attack`` boolean."""
    n_total = min(len(X), cfg.batch_size * cfg.max_batches)
    X = X.iloc[:n_total].reset_index(drop=True)
    y_attack = np.asarray(y_attack[:n_total], dtype=bool)
    n_batches = int(np.ceil(n_total / cfg.batch_size))
    cols = list(X.columns)

    adaptive_model = model  # starts identical to the static model
    river_model = _make_river_model() if (cfg.strategy == "river" or cfg.include_river_online) else None
    adwin = drift.ADWIN(delta=cfg.adwin_delta)

    rows, events, notes = [], [], []
    err_stream: list[int] = []
    drift_positions: list[int] = []
    cooldown = 0
    last_retrain_batch = -1
    all_static, all_adapt, all_online = [], [], []
    t0 = time.perf_counter()

    for b in range(n_batches):
        lo, hi = b * cfg.batch_size, min((b + 1) * cfg.batch_size, n_total)
        Xb, yb = X.iloc[lo:hi], y_attack[lo:hi]

        # 1) predictions ---------------------------------------------------- #
        static_pred = predict_binary(model, Xb)["is_attack"].to_numpy()
        if cfg.strategy == "river":
            adapt_pred = _river_predict(river_model, Xb.to_numpy(float), cols)
        else:
            adapt_pred = predict_binary(adaptive_model, Xb)["is_attack"].to_numpy()
        online_pred = None
        if cfg.include_river_online and cfg.strategy != "river":
            online_pred = _river_predict(river_model, Xb.to_numpy(float), cols)

        all_static.append(static_pred)
        all_adapt.append(adapt_pred)
        if online_pred is not None:
            all_online.append(online_pred)
        static_err = (static_pred != yb).astype(int)
        adapt_err = (adapt_pred != yb).astype(int)

        # 2-3) error monitoring + ADWIN ------------------------------------- #
        drift_here: list[int] = []
        improved = 0
        for i, e in enumerate(adapt_err):
            prev_mean = adwin.estimation
            adwin.update(int(e))
            err_stream.append(int(e))
            if adwin.drift_detected and cooldown == 0:
                # ADWIN reports a change of the mean in either direction. Only a *rise* in error
                # signals degradation (drift); a fall means the model got better.
                if adwin.estimation > prev_mean:
                    drift_here.append(i)
                    drift_positions.append(lo + i)
                else:
                    improved += 1
        if cooldown > 0:
            cooldown -= 1

        retrained, retrain_note = False, ""
        if improved and not drift_here:
            events.append({"batch": b + 1, "record_index": lo, "type": "Error decrease",
                           "detail": "ADWIN saw the error rate fall (model improved) - not treated as drift."})
        if drift_here:
            events.append({
                "batch": b + 1, "record_index": lo + drift_here[0], "type": "Drift alert",
                "detail": (f"ADWIN flagged a change in the error stream (batch error "
                           f"{adapt_err.mean():.1%}; {len(drift_here)} alert(s) in batch)."),
            })
            # 4) adaptive response ------------------------------------------ #
            if cfg.strategy == "retrain":
                w_lo = max(0, hi - cfg.window_size)
                Xw, yw = X.iloc[w_lo:hi], y_attack[w_lo:hi]
                n_replay = int(cfg.window_size * cfg.replay_fraction)
                if cfg.replay_fraction > 0 and w_lo > 0 and n_replay > 0:
                    # replay: uniform sample of older labelled records to limit forgetting of earlier attack types
                    pick = np.sort(np.random.default_rng(b).choice(w_lo, size=min(n_replay, w_lo), replace=False))
                    Xw = pd.concat([X.iloc[pick], Xw])
                    yw = np.concatenate([y_attack[pick], yw])
                if len(yw) < cfg.min_retrain_samples:
                    retrain_note = f"skipped - only {len(yw)} labelled records in window"
                elif len(set(yw)) < 2:
                    retrain_note = "skipped - recent window contains a single class"
                else:
                    try:
                        adaptive_model = _fit_clone(model, Xw, yw)
                        retrained = True
                        last_retrain_batch = b + 1
                        retrain_note = f"retrained on {len(yw):,} labelled records (recent window{' + replay of older records' if len(yw) > hi - w_lo else ''})"
                    except Exception as exc:  # noqa: BLE001
                        retrain_note = f"retraining failed ({type(exc).__name__})"
                        notes.append(f"Retraining failed at batch {b + 1}: {str(exc)[:120]}")
                if retrained:
                    adwin = drift.ADWIN(delta=cfg.adwin_delta)  # fresh reference window
                    cooldown = cfg.cooldown_batches
            else:
                retrain_note = "River learner adapts continuously (Hoeffding Adaptive Tree)"
            events.append({
                "batch": b + 1, "record_index": hi, "type": "Adaptation", "detail": retrain_note,
            })

        # River learners: learn after predicting (test-then-train)
        if river_model is not None:
            Xnp = Xb.to_numpy(float)
            for i in range(len(Xnp)):
                river_model.learn_one(dict(zip(cols, Xnp[i])), bool(yb[i]))

        row = {
            "batch": b + 1, "records": hi - lo, "start": lo, "end": hi,
            "static_error": float(static_err.mean()),
            "adaptive_error": float(adapt_err.mean()),
            "static_accuracy": 1 - float(static_err.mean()),
            "adaptive_accuracy": 1 - float(adapt_err.mean()),
            "attack_share": float(yb.mean()),
            "drift_detected": bool(drift_here),
            "retrained": retrained,
            "adwin_width": float(adwin.width),
        }
        if online_pred is not None:
            row["online_error"] = float((online_pred != yb).mean())
            row["online_accuracy"] = 1 - row["online_error"]
        rows.append(row)
        if progress:
            progress((b + 1) / n_batches, f"Batch {b + 1}/{n_batches}")

    batches = pd.DataFrame(rows)
    events_df = pd.DataFrame(events, columns=["batch", "record_index", "type", "detail"])
    summary = _summarise(batches, events_df, cfg)
    summary["runtime_s"] = time.perf_counter() - t0
    summary["records"] = n_total
    overall = {}
    for name, parts in (("static", all_static), ("adaptive", all_adapt), ("online", all_online)):
        if parts:
            overall[name] = evaluate_predictions(y_attack, np.concatenate(parts), labels=[False, True], positive=True)
    return DriftRun(
        batches=batches, events=events_df, summary=summary, config=cfg, provenance=provenance,
        notes=notes, error_stream=np.asarray(err_stream), drift_positions=drift_positions, overall=overall,
    )


def _summarise(b: pd.DataFrame, ev: pd.DataFrame, cfg: DriftConfig) -> dict:
    w = b["records"]
    alerts = ev[ev["type"] == "Drift alert"] if len(ev) else ev
    out = {
        "n_batches": int(len(b)),
        "n_drift_events": int(len(alerts)),
        "n_retrains": int(b["retrained"].sum()),
        "static_accuracy": float(np.average(b["static_accuracy"], weights=w)),
        "adaptive_accuracy": float(np.average(b["adaptive_accuracy"], weights=w)),
        "status": "Drift detected" if len(alerts) else "Stable - no drift detected",
        "last_batch_status": "Drift alert" if bool(b["drift_detected"].iloc[-1]) else "Stable",
    }
    if "online_accuracy" in b:
        out["online_accuracy"] = float(np.average(b["online_accuracy"], weights=w))
    # before / after the first drift (4-batch windows)
    if len(alerts):
        first = int(alerts["batch"].iloc[0])
        k = 4
        before = b[(b["batch"] < first) & (b["batch"] >= first - k)]
        after = b[(b["batch"] > first) & (b["batch"] <= first + k)]
        out["first_drift_batch"] = first
        if len(before) and len(after):
            for name in ("static", "adaptive"):
                out[f"{name}_before"] = float(before[f"{name}_accuracy"].mean())
                out[f"{name}_after"] = float(after[f"{name}_accuracy"].mean())
    return out


def overall_metrics(run_X: pd.DataFrame, model: Any, y_attack: np.ndarray) -> dict:
    """Helper for a full-stream metric table of the static model (used in tests / reporting)."""
    pred = predict_binary(model, run_X)["is_attack"].to_numpy()
    return evaluate_predictions(y_attack, pred, labels=[False, True], positive=True)
''',
    "utils.drift_lab": r'''
"""Drift lab - the evaluation protocol of the paper (no UI code).

Implements, on whatever dataset the user provides:

* **Scenarios** (paper Section III-F): S1/S2 *unseen attack families* (the model is trained without some
  families that appear after the drift onset) and S3 *covariate shift* (inputs move, labels do not - a
  false-alarm test).
* **Algorithm 1** - prequential (test-then-train) evaluation with an ADWIN monitor on the per-flow error
  stream, an alert only when the error *rises*, a one-batch cool-down, and window + replay + anchor retraining.
* The four **policies**: static, periodic (every 5 batches), drift-triggered and an online Hoeffding Adaptive
  Tree updated after every flow.
* A **threshold study** (paper Section IV-F): re-tune the decision threshold on recent labelled traffic.

Nothing is simulated beyond the construction of the stream from real labelled records.
"""
from __future__ import annotations

import re
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from utils.prediction import is_benign

# ----------------------------------------------------------------------------------------------- #
# Paper settings (Section III-D / III-F). All of them can be overridden from the UI.
# ----------------------------------------------------------------------------------------------- #
BATCH_SIZE = 1_000
N_BATCHES = 40
N_TRAIN = 20_000
ONSET_BATCH = 15          # batches 1-15 are drift-free, drift starts with batch 16
RECENT, REPLAY, ANCHOR = 4_000, 2_000, 2_000
ADWIN_DELTA = 0.002
PERIODIC_EVERY = 5
MODEL_NAMES = ["XGBoost", "Random Forest", "Decision Tree", "Logistic Regression"]
POLICIES = ["Static", "Periodic", "Drift-triggered", "Online HAT"]


# ----------------------------------------------------------------------------------------------- #
# Models
# ----------------------------------------------------------------------------------------------- #
def make_model(name: str, seed: int = 42):
    """Standardise with training statistics, then one of the four learners (near-default settings)."""
    if name == "XGBoost":
        from xgboost import XGBClassifier

        clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3, tree_method="hist", n_jobs=2,
                            random_state=seed, eval_metric="logloss", verbosity=0)
    elif name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(n_estimators=100, n_jobs=2, random_state=seed)
    elif name == "Decision Tree":
        clf = DecisionTreeClassifier(random_state=seed)
    elif name == "Logistic Regression":
        clf = LogisticRegression(max_iter=300, random_state=seed)
    else:
        raise ValueError(f"Unknown model '{name}'.")
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def _fit(model, X: pd.DataFrame, y: np.ndarray):
    m = clone(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X, y.astype(int))
    return m


def _proba(model, X: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.predict_proba(X)[:, 1]


# ----------------------------------------------------------------------------------------------- #
# Scenario construction
# ----------------------------------------------------------------------------------------------- #
@dataclass
class ScenarioConfig:
    kind: str = "unseen"                       # "unseen" (S1/S2) | "covariate" (S3)
    unseen: list[str] = field(default_factory=list)
    n_train: int = N_TRAIN
    n_batches: int = N_BATCHES
    batch_size: int = BATCH_SIZE
    onset_batch: int = ONSET_BATCH             # number of drift-free batches
    unseen_share: float = 0.10                 # share of unseen-family flows in the post-onset stream
    shift_features: list[str] = field(default_factory=list)
    shift_scale: float = 3.0
    shift_sd: float = 1.0
    split_by_file: bool = False                # train from "*train*" files, stream from "*test*" files (paper S2)
    seed: int = 42


@dataclass
class Scenario:
    cfg: ScenarioConfig
    features: list[str]
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_stream: pd.DataFrame
    y_stream: np.ndarray
    cat_stream: np.ndarray
    onset_flow: int                            # 0-based index of the first drifted flow
    notes: list[str]
    n_unseen_flows: int = 0

    @property
    def n_flows(self) -> int:
        return len(self.X_stream)


def default_unseen(dataset: str, categories: list[str]) -> list[str]:
    """Which categories to hold out of training by default (follows the paper's S1 / S2)."""
    cats = [c for c in categories if not is_benign(c)]
    low = {c: str(c).lower() for c in cats}
    if "unsw" in dataset.lower():
        keep = ("generic", "exploit")
        return [c for c in cats if not any(k in low[c] for k in keep)]
    web = [c for c in cats if any(k in low[c] for k in ("web", "xss", "sql"))]
    return web or cats[-1:]


def default_shift_features(X: pd.DataFrame, k: int = 5) -> list[str]:
    """Five high-variance timing features (idle / inter-arrival times), as in the paper's S3."""
    var = X.var(numeric_only=True)
    timing = [c for c in var.index if re.search(r"idle|iat", str(c), re.I)]
    pool = timing if len(timing) >= 2 else list(var.index)
    return sorted(pool, key=lambda c: -var[c])[:k]


def build_scenario(pool: pd.DataFrame, features: list[str], cfg: ScenarioConfig) -> Scenario:
    """Turn a labelled pool (columns: features + ``_attack_type`` [+ ``_source_file``]) into train + stream.

    Pre-onset stream flows and the training set only contain *seen* categories. After the onset the stream is
    a random mix of the seen categories plus ``unseen_share`` flows of the held-out families (all that exist,
    up to that share). For the covariate scenario nothing is held out; instead ``shift_features`` are scaled
    and shifted from the onset on, while the labels stay untouched.
    """
    rng = np.random.default_rng(cfg.seed)
    notes: list[str] = []
    cat = pool["_attack_type"].astype(str).to_numpy()
    attack = np.array([not is_benign(c) for c in cat])
    unseen_mask = np.isin(cat, cfg.unseen) if cfg.kind == "unseen" else np.zeros(len(pool), bool)
    n_pre = cfg.onset_batch * cfg.batch_size
    n_total = cfg.n_batches * cfg.batch_size
    n_post = n_total - n_pre

    seen_idx = np.flatnonzero(~unseen_mask)
    unseen_idx = np.flatnonzero(unseen_mask)
    rng.shuffle(seen_idx)
    rng.shuffle(unseen_idx)

    train_pool, stream_pool = seen_idx, seen_idx
    if cfg.split_by_file and "_source_file" in pool:
        fn = pool["_source_file"].astype(str).str.lower().to_numpy()
        tr = seen_idx[np.char.find(fn[seen_idx].astype(str), "train") >= 0]
        te = seen_idx[np.char.find(fn[seen_idx].astype(str), "test") >= 0]
        if len(tr) and len(te):
            train_pool, stream_pool = tr, te
            unseen_idx = unseen_idx[np.char.find(fn[unseen_idx].astype(str), "test") >= 0]
            notes.append("Official split: training flows come from the *train* file(s), the stream from the *test* file(s).")
        else:
            notes.append("Official split requested but the files do not contain both 'train' and 'test' in their names - using a random disjoint split.")
    same_pool = train_pool is stream_pool

    n_u = min(len(unseen_idx), int(round(cfg.unseen_share * n_post))) if cfg.kind == "unseen" else 0
    if cfg.kind == "unseen" and n_u == 0:
        raise ValueError("None of the selected unseen categories exist in the pool - pick other categories.")
    n_seen_post = n_post - n_u

    # --- training set --------------------------------------------------------------------------
    need_stream_seen = n_pre + n_seen_post
    n_train = cfg.n_train
    if same_pool and len(train_pool) < n_train + need_stream_seen:
        # shrink training to leave the stream enough fresh records, then allow re-use if still short
        n_train = max(2_000, min(cfg.n_train, int(len(train_pool) * 0.35)))
        notes.append(f"Only {len(train_pool):,} seen-category records are available, so the training set was reduced to {n_train:,}.")
    tr_idx = train_pool[:n_train]
    rest = stream_pool[n_train:] if same_pool else stream_pool
    if len(rest) < need_stream_seen:
        notes.append(f"The stream needs {need_stream_seen:,} seen-category flows but only {len(rest):,} are left, so some flows are re-used (sampled with replacement).")
        rest = rng.choice(stream_pool if len(rest) == 0 else rest, size=need_stream_seen, replace=True)
    pre_idx = rest[:n_pre]
    post_seen_idx = rest[n_pre:n_pre + n_seen_post]
    post_unseen_idx = unseen_idx[:n_u]
    post = np.concatenate([post_seen_idx, post_unseen_idx])
    rng.shuffle(post)
    stream_idx = np.concatenate([pre_idx, post])

    X_train = pool.iloc[tr_idx][features].reset_index(drop=True)
    X_stream = pool.iloc[stream_idx][features].reset_index(drop=True).astype(float)
    y_train = attack[tr_idx]
    y_stream = attack[stream_idx]
    cat_stream = cat[stream_idx]

    if cfg.kind == "covariate":
        feats = [f for f in cfg.shift_features if f in features] or default_shift_features(X_train)
        sd = X_train[feats].std().replace(0, 1.0)
        X_stream.loc[X_stream.index >= n_pre, feats] = X_stream.loc[X_stream.index >= n_pre, feats] * cfg.shift_scale + sd * cfg.shift_sd
        notes.append(f"Covariate shift from flow {n_pre:,}: {', '.join(feats)} scaled ×{cfg.shift_scale:g} and shifted by {cfg.shift_sd:g} σ. Labels are untouched.")
    if y_train.all() or not y_train.any():
        raise ValueError("The training set contains only one class (all benign or all attack) - choose different unseen categories.")
    return Scenario(cfg, features, X_train, y_train, X_stream, y_stream, cat_stream, n_pre, notes, n_u)


# ----------------------------------------------------------------------------------------------- #
# Metrics
# ----------------------------------------------------------------------------------------------- #
def binary_metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    y, yhat = np.asarray(y, bool), np.asarray(yhat, bool)
    tp, fp = int((y & yhat).sum()), int((~y & yhat).sum())
    fn, tn = int((y & ~yhat).sum()), int((~y & ~yhat).sum())
    n = max(len(y), 1)
    return {"acc": (tp + tn) / n, "f1": 2 * tp / max(2 * tp + fp + fn, 1), "recall": tp / max(tp + fn, 1),
            "precision": tp / max(tp + fp, 1), "fpr": fp / max(fp + tn, 1), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def batch_accuracy(y: np.ndarray, yhat: np.ndarray, bs: int) -> np.ndarray:
    n = len(y) // bs
    return np.array([(y[i * bs:(i + 1) * bs] == yhat[i * bs:(i + 1) * bs]).mean() for i in range(n)])


def category_rates(cat: np.ndarray, y: np.ndarray, yhat: np.ndarray, start: int) -> pd.DataFrame:
    """Post-onset detection rate per true category (specificity for benign traffic)."""
    rows = []
    for c in pd.unique(cat[start:]):
        m = cat[start:] == c
        yy, pp = y[start:][m], yhat[start:][m]
        rows.append({"category": c, "flows": int(m.sum()), "rate": float((pp == yy).mean()),
                     "kind": "specificity" if not yy.any() else "detection rate"})
    return pd.DataFrame(rows).sort_values("category").reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- #
# Policies
# ----------------------------------------------------------------------------------------------- #
@dataclass
class PolicyResult:
    name: str
    yhat: np.ndarray
    proba: np.ndarray | None = None
    retrain_batches: list[int] = field(default_factory=list)   # 1-based batch numbers
    retrain_seconds: float = 0.0
    alerts: list[dict] = field(default_factory=list)
    runtime_seconds: float = 0.0
    final_model: object = None

    @property
    def n_retrains(self) -> int | None:
        return None if self.name == "Online HAT" else len(self.retrain_batches)


def _retrain_set(sc: Scenario, upto: int, rng: np.random.Generator, recent=RECENT, replay=REPLAY, anchor=ANCHOR):
    """(a) most recent labelled flows, (b) uniform replay of older stream flows, (c) anchor sample of D0."""
    lo = max(0, upto - recent)
    Xr, yr = sc.X_stream.iloc[lo:upto], sc.y_stream[lo:upto]
    parts_X, parts_y = [Xr], [yr]
    if lo > 0:
        k = min(replay, lo)
        ridx = rng.choice(lo, size=k, replace=False)
        parts_X.append(sc.X_stream.iloc[ridx]); parts_y.append(sc.y_stream[ridx])
    k = min(anchor, len(sc.X_train))
    aidx = rng.choice(len(sc.X_train), size=k, replace=False)
    parts_X.append(sc.X_train.iloc[aidx]); parts_y.append(sc.y_train[aidx])
    return pd.concat(parts_X, ignore_index=True), np.concatenate(parts_y)


def run_policy(sc: Scenario, f0, policy: str, delta: float = ADWIN_DELTA, seed: int = 42,
               recent: int = RECENT, replay: int = REPLAY, anchor: int = ANCHOR) -> PolicyResult:
    """Prequential evaluation of one update policy (paper Algorithm 1 for 'Drift-triggered')."""
    from river import drift as rdrift

    bs, nb = sc.cfg.batch_size, sc.cfg.n_batches
    rng = np.random.default_rng(seed)
    f = f0
    yhat = np.zeros(sc.n_flows, bool)
    proba = np.zeros(sc.n_flows)
    res = PolicyResult(policy, yhat, proba)
    adwin = rdrift.ADWIN(delta=delta)
    cooldown = 0
    t_run = time.perf_counter()
    for b in range(nb):
        lo, hi = b * bs, (b + 1) * bs
        Xb, yb = sc.X_stream.iloc[lo:hi], sc.y_stream[lo:hi]
        p = _proba(f, Xb)                                   # 1) test first
        proba[lo:hi] = p
        yhat[lo:hi] = p >= 0.5
        alert = False
        if policy == "Drift-triggered":
            err = (yhat[lo:hi] != yb).astype(int)
            for i, e in enumerate(err):                     # 2) labels revealed -> error stream -> ADWIN
                prev = adwin.estimation
                adwin.update(int(e))
                if adwin.drift_detected:
                    rose = adwin.estimation > prev
                    counted = rose and cooldown == 0
                    res.alerts.append({"flow": lo + i + 1, "batch": b + 1, "error_before": float(prev),
                                       "error_after": float(adwin.estimation), "rose": bool(rose),
                                       "in_cooldown": cooldown > 0, "triggered_retrain": bool(counted)})
                    alert = alert or counted
            cooldown = max(0, cooldown - 1)
        retrain = alert or (policy == "Periodic" and (b + 1) % PERIODIC_EVERY == 0)
        if retrain:                                         # 3) adapt; old model served until now
            t0 = time.perf_counter()
            Xs, ys = _retrain_set(sc, hi, rng, recent, replay, anchor)
            f = _fit(f0, Xs, ys)
            res.retrain_seconds += time.perf_counter() - t0
            res.retrain_batches.append(b + 1)
            if policy == "Drift-triggered":
                adwin = rdrift.ADWIN(delta=delta)
                cooldown = 1
    res.runtime_seconds = time.perf_counter() - t_run
    res.final_model = f
    return res


def run_online_hat(sc: Scenario, seed: int = 42, progress: Callable[[float], None] | None = None) -> PolicyResult:
    """River Hoeffding Adaptive Tree (grace 100, depth 12) pre-trained on D0, then test-then-train per flow."""
    from river import compose, preprocessing, tree

    cols = list(sc.features)
    model = compose.Pipeline(preprocessing.StandardScaler(), tree.HoeffdingAdaptiveTreeClassifier(grace_period=100, max_depth=12, seed=seed))
    t0 = time.perf_counter()
    Xt = sc.X_train.to_numpy(float)
    for i in range(len(Xt)):
        model.learn_one(dict(zip(cols, Xt[i])), bool(sc.y_train[i]))
    Xs = sc.X_stream.to_numpy(float)
    yhat = np.zeros(len(Xs), bool)
    for i in range(len(Xs)):
        x = dict(zip(cols, Xs[i]))
        yhat[i] = bool(model.predict_one(x) or False)
        model.learn_one(x, bool(sc.y_stream[i]))
        if progress and i % 2000 == 0:
            progress(i / len(Xs))
    return PolicyResult("Online HAT", yhat, None, [], 0.0, [], time.perf_counter() - t0, model)


def fit_initial(sc: Scenario, model_name: str):
    t0 = time.perf_counter()
    f0 = _fit(make_model(model_name), sc.X_train, sc.y_train)
    return f0, time.perf_counter() - t0


def summarise(sc: Scenario, results: dict[str, PolicyResult]) -> pd.DataFrame:
    """Paper Table IV for one scenario (computed, not copied)."""
    rows = []
    for name, r in results.items():
        m = binary_metrics(sc.y_stream, r.yhat)
        post = slice(sc.onset_flow, None)
        post_acc = float((sc.y_stream[post] == r.yhat[post]).mean())
        delay = None
        if r.alerts:
            first = [a["flow"] for a in r.alerts if a["triggered_retrain"] and a["flow"] > sc.onset_flow]
            delay = (min(first) - sc.onset_flow) if first else None
        rows.append({"Policy": name, "Acc (%)": 100 * m["acc"], "F1": m["f1"], "Rec (%)": 100 * m["recall"], "FPR (%)": 100 * m["fpr"],
                     "Post-onset acc (%)": 100 * post_acc, "Retrains": r.n_retrains,
                     "Retrain time (s)": None if name == "Online HAT" else r.retrain_seconds,
                     "Delay (flows)": delay if name == "Drift-triggered" else None})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- #
# Threshold study (paper Section IV-F)
# ----------------------------------------------------------------------------------------------- #
def threshold_study(sc: Scenario, proba: np.ndarray, tune_batches: int = 5) -> dict | None:
    """Re-tune the decision threshold on the first post-onset batches (labels revealed), evaluate on the rest."""
    start = sc.onset_flow
    cut = start + tune_batches * sc.cfg.batch_size
    if cut >= sc.n_flows - sc.cfg.batch_size:
        return None
    yt, pt = sc.y_stream[start:cut], proba[start:cut]
    ye, pe = sc.y_stream[cut:], proba[cut:]
    if not yt.any() or yt.all() or not ye.any() or ye.all():
        return None
    grid = np.linspace(0.02, 0.98, 97)
    f1s = [binary_metrics(yt, pt >= t)["f1"] for t in grid]
    t_star = float(grid[int(np.argmax(f1s))])
    fpr, tpr, _ = roc_curve(ye, pe)
    return {"threshold": t_star, "default": binary_metrics(ye, pe >= 0.5), "tuned": binary_metrics(ye, pe >= t_star),
            "grid": grid, "eval_curve": [binary_metrics(ye, pe >= t) for t in grid], "roc": (fpr, tpr),
            "tune_flows": int(cut - start), "eval_flows": int(len(ye))}
''',
    "utils.paper_data": r'''
"""Results and metadata published in the project paper.

"An Explainable, Concept-Drift-Aware Machine Learning Framework for Real-Time Network Intrusion Detection"
(V. Rohith, Shreyas M. K., Siddharth B, R. K. Karthik Kashyap, Ankitha Takkar, - Dayananda Sagar University).

These numbers are transcribed from the paper's tables and are shown as *published reference results*. The app never
presents them as something it computed; live results come from the user's own reports, data and the Drift Lab.
"""
from __future__ import annotations

import pandas as pd

PAPER_TITLE = "An Explainable, Concept-Drift-Aware Machine Learning Framework for Real-Time Network Intrusion Detection"
PAPER_AUTHORS = ["V. Rohith", "Shreyas M. K.", "Siddharth B", "R. K. Karthik Kashyap", "Ankitha Takkar"]
PAPER_AFFILIATION = "Dayananda Sagar University, Bengaluru, India"
PAPER_DEPARTMENT = "Dept. of CSE (AI & ML)"

ABSTRACT = (
    "Most machine-learning intrusion detection systems (IDS) are trained once and scored on a random split of a fixed benchmark. "
    "That hides two problems that matter in practice: traffic keeps changing, and analysts are rarely told why an alert fired. "
    "The framework pairs classical detectors with an error-driven drift monitor and post-hoc explanations. Logistic regression, a decision tree, "
    "random forest and XGBoost are compared on CIC-IDS2017 and UNSW-NB15. An ADWIN test watches the deployed model's prediction errors; when they "
    "rise significantly, the model is refitted on recent labelled traffic plus a replay sample of older flows, and SHAP and LIME explain its decisions. "
    "Everything runs on a two-core CPU."
)

CONTRIBUTIONS = [
    "A modular framework combining an offline-trained detector, an ADWIN monitor on the per-flow error stream, window-plus-replay retraining and SHAP/LIME explanations, served through a dashboard that runs on a laptop CPU.",
    "A comparison of four classical learners on CIC-IDS2017 and UNSW-NB15 for binary and multi-class detection, with a port-feature ablation and per-class analysis showing where aggregate metrics mislead.",
    "A prequential evaluation over three drift scenarios comparing static, periodic, drift-triggered and online Hoeffding Adaptive Tree policies on accuracy, detection delay and retraining cost.",
    "A measurement of how often SHAP and LIME agree on the same alert, and what each costs.",
]

RESEARCH_GAPS = [
    ("Explanations are an add-on", "Most papers show one SHAP plot and never check whether a second explainer would tell the analyst the same thing."),
    ("The pieces rarely meet", "Binary detection, attack-type classification, a fair comparison of learners and explanations live in separate papers rather than one reproducible workflow."),
    ("Random-split evaluation", "Nearly everyone tests on a random split where every attack family is already seen in training, so behaviour under drift and the cost of staying current go unmeasured; adaptive methods that do exist need far more compute than most security teams can spare."),
]

LIMITATIONS = [
    "Near-perfect CIC-IDS2017 scores come from a random split of two capture days in which DDoS is easy to spot, so they say little about another network.",
    "The prequential setup assumes labels arrive soon after prediction; in practice they arrive late and only for some flows, which will lengthen ADWIN's delay.",
    "Known labelling problems in CIC-IDS2017 and a TTL artefact in UNSW-NB15 deserve a re-run on corrected data.",
    "Default settings and a single seed: small gaps between tree models should not be over-read.",
    "Scenarios produce abrupt drift; gradual drift and cross-dataset transfer remain open.",
]

FUTURE_WORK = ["Handle delayed and partial labels", "Gradual drift", "Track how explanations change across retrained models", "Cross-dataset transfer (e.g. CSE-CIC-IDS2018)", "Unsupervised input-drift detectors as an early warning"]

# --- Table I ------------------------------------------------------------------------------------
TABLE_I = pd.DataFrame([
    ["FL-MalDrift [9]", "Android malware", "Federated NN", "ADWIN/DDM/EDDM/HDDM per client", "Drebin, CICMalDroid, AndroZoo", "–", "–"],
    ["Joshi & Purohit [8]", "Phishing", "Incremental ML", "Drift monitor, incremental update", "Public phishing sets", "–", "✓"],
    ["Ahmadi-Assalemi et al. [16]", "ICS", "Adaptive classifier", "Adaptive learning", "ICS testbed data", "–", "✓"],
    ["CyberAdaptAI [10]", "NIDS", "Weighted ensemble", "ADWIN + weight rebalancing", "CIC-IDS2017, UNSW-NB15", "SHAP", "–"],
    ["Ben Ncir et al. [13]", "NIDS", "LSTM ensemble", "None (static split)", "Public IDS benchmarks", "SHAP", "–"],
    ["REACT-D3QN [11]", "NIDS (edge)", "Dueling DDQN", "Asymmetric reward, transfer", "CIC-IDS2017 → CSE-CIC-IDS2018", "–", "–"],
    ["This work", "NIDS", "LR, DT, RF, XGBoost", "ADWIN on error stream + window/replay retraining", "CIC-IDS2017, UNSW-NB15, three drift scenarios", "SHAP, LIME", "✓"],
], columns=["Work", "Domain", "Learner", "Drift handling", "Data / evaluation", "XAI", "CPU-only"])

# --- Table II: binary --------------------------------------------------------------------------
TABLE_II = pd.DataFrame([
    ["CIC-IDS2017", "Logistic Regression", 97.90, 96.64, 96.76, 96.70, 99.82, 1.57, 2.8, 0.8],
    ["CIC-IDS2017", "Decision Tree", 99.95, 99.92, 99.92, 99.92, 99.94, 0.04, 3.2, 0.7],
    ["CIC-IDS2017", "Random Forest", 99.96, 99.97, 99.90, 99.93, 99.99, 0.01, 10.4, 3.3],
    ["CIC-IDS2017", "XGBoost", 99.99, 99.99, 99.97, 99.98, 100.0, 0.00, 2.0, 1.2],
    ["UNSW-NB15", "Logistic Regression", 81.20, 75.44, 97.65, 85.12, 95.58, 38.95, 6.7, 0.8],
    ["UNSW-NB15", "Decision Tree", 86.41, 82.44, 95.69, 88.57, 85.61, 24.97, 2.9, 1.0],
    ["UNSW-NB15", "Random Forest", 87.25, 82.00, 98.45, 89.47, 97.86, 26.48, 12.2, 6.9],
    ["UNSW-NB15", "XGBoost", 87.42, 82.33, 98.23, 89.58, 98.34, 25.82, 2.6, 2.1],
], columns=["Dataset", "Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1 (%)", "ROC-AUC (%)", "FPR (%)", "Train time (s)", "Inference (µs/flow)"])

# --- Table III: multiclass ----------------------------------------------------------------------
TABLE_III = pd.DataFrame([
    ["CIC-IDS2017 (4 classes)", "Logistic Regression", 99.13, 75.49, 76.21, 75.47, 99.14, 99.88],
    ["CIC-IDS2017 (4 classes)", "Decision Tree", 99.84, 90.52, 91.84, 91.11, 99.84, 95.89],
    ["CIC-IDS2017 (4 classes)", "Random Forest", 99.84, 90.94, 88.53, 89.60, 99.83, 99.98],
    ["CIC-IDS2017 (4 classes)", "XGBoost", 99.86, 90.88, 92.43, 91.58, 99.87, 99.35],
    ["UNSW-NB15 (10 classes)", "Logistic Regression", 69.93, 37.59, 38.44, 33.51, 70.71, 94.91],
    ["UNSW-NB15 (10 classes)", "Decision Tree", 73.72, 48.81, 52.72, 49.24, 76.35, 82.78],
    ["UNSW-NB15 (10 classes)", "Random Forest", 75.69, 53.70, 49.56, 47.03, 77.75, 94.37],
    ["UNSW-NB15 (10 classes)", "XGBoost", 76.72, 49.36, 54.36, 49.59, 78.37, 96.19],
], columns=["Dataset", "Model", "Accuracy (%)", "Macro precision (%)", "Macro recall (%)", "Macro F1 (%)", "Weighted F1 (%)", "Macro AUC (%)"])

# --- Table IV: prequential results ------------------------------------------------------------------
SCENARIO_NAMES = {"S1": "S1 · unseen web attacks (CIC-IDS2017)", "S2": "S2 · unseen families (UNSW-NB15)", "S3": "S3 · covariate shift (CIC-IDS2017)"}
TABLE_IV = pd.DataFrame([
    ["S1", "Static", 95.30, 0.867, 76.49, 0.00, 0, None],
    ["S1", "Periodic", 98.96, 0.973, 94.85, 0.02, 8, None],
    ["S1", "Drift-triggered", 99.61, 0.990, 98.10, 0.01, 1, 71],
    ["S1", "Online HAT", 97.15, 0.930, 94.89, 2.28, None, None],
    ["S2", "Static", 84.92, 0.870, 78.41, 3.47, 0, None],
    ["S2", "Periodic", 94.26, 0.954, 92.94, 3.39, 8, None],
    ["S2", "Drift-triggered", 94.11, 0.953, 92.33, 2.72, 1, 103],
    ["S2", "Online HAT", 89.24, 0.916, 92.05, 15.78, None, None],
    ["S3", "Static", 99.98, 1.000, 99.94, 0.01, 0, None],
    ["S3", "Periodic", 99.96, 0.999, 99.89, 0.01, 8, None],
    ["S3", "Drift-triggered", 99.98, 1.000, 99.94, 0.01, 0, None],
    ["S3", "Online HAT", 98.80, 0.980, 98.87, 1.23, None, None],
], columns=["Scenario", "Policy", "Acc (%)", "F1", "Rec (%)", "FPR (%)", "Retrains", "Delay (flows)"])

# --- Table V: explainer agreement and cost ----------------------------------------------------------
TABLE_V = pd.DataFrame([
    ["CIC-IDS2017", 0.35, 0.03, 57.5],
    ["UNSW-NB15", 0.07, 0.08, 53.9],
], columns=["Dataset", "Top-10 Jaccard", "TreeSHAP (ms)", "LIME (ms)"])

# --- Findings quoted in the text ----------------------------------------------------------------
FINDINGS = [
    ("Protocol shapes the headline", "XGBoost: 99.99 % accuracy on a random CIC-IDS2017 split vs 87.42 % (25.8 % FPR) on the official UNSW-NB15 split."),
    ("AUC stays high, threshold moves", "UNSW-NB15 ROC-AUC is 0.983 yet the default 0.5 threshold gives a 25.8 % FPR - re-tune the threshold on recent labelled traffic before a full retrain."),
    ("Port-feature ablation", "Removing Source/Destination Port changed XGBoost only from 99.988 % to 99.991 % and RF from 99.958 % to 99.949 % - the models rely on flow statistics, not ports."),
    ("Minority classes", "CIC-IDS2017 XGBoost: 99.86 % accuracy but 91.58 % macro-F1; XSS is often mistaken for brute force (F1 0.73 vs 0.93). UNSW-NB15 best macro-F1 is 49.6 %."),
    ("Drift response", "One drift-triggered retraining recovered most of the loss within ~100 flows; the periodic policy needed eight retrainings. On S2 retraining took 0.27 s vs 2.04 s."),
    ("False-alarm test", "On the harmless covariate shift (S3) ADWIN stayed silent while the periodic policy retrained eight times for no benefit."),
    ("Explanation artefact", "UNSW-NB15: source-to-destination TTL (sttl) carries about four times the weight of the next feature - more an artefact of how the data were generated than a transferable attack signature."),
]
''',
    "components.styles": r'''
"""Design system: glassmorphism CSS, icon helper and the shared Plotly theme."""
from __future__ import annotations

import streamlit as st

# ---- palette (single source of truth, mirrored in the CSS variables below) ----
COLORS = {
    "bg": "#070b1f", "navy": "#0b1033", "cyan": "#22e3ff", "blue": "#4f8cff", "violet": "#8b6bff",
    "green": "#2ee6a6", "red": "#ff4d6d", "amber": "#ffb547", "pink": "#ff4fa3", "orange": "#ff8a4c",
    "text": "#eef2ff", "muted": "rgba(226,232,255,.62)",
}
COLORWAY = ["#22e3ff", "#ff4fa3", "#8b6bff", "#ffb547", "#2ee6a6", "#4f8cff", "#ff8a4c", "#ff4d6d"]

ICONS = {  # semantic name -> Font Awesome 6 class
    "shield": "fa-solid fa-shield-halved", "network": "fa-solid fa-network-wired", "brain": "fa-solid fa-brain",
    "chart": "fa-solid fa-chart-column", "warning": "fa-solid fa-triangle-exclamation", "lock": "fa-solid fa-lock",
    "database": "fa-solid fa-database", "robot": "fa-solid fa-robot", "radar": "fa-solid fa-satellite-dish",
    "code": "fa-solid fa-code", "rocket": "fa-solid fa-rocket", "activity": "fa-solid fa-wave-square",
    "search": "fa-solid fa-magnifying-glass", "bug": "fa-solid fa-bug", "eye": "fa-solid fa-eye",
    "check": "fa-solid fa-circle-check", "bolt": "fa-solid fa-bolt", "layers": "fa-solid fa-layer-group",
    "flask": "fa-solid fa-flask", "book": "fa-solid fa-book-open", "users": "fa-solid fa-users",
    "clock": "fa-solid fa-clock", "gear": "fa-solid fa-gear", "file": "fa-solid fa-file-csv",
    "target": "fa-solid fa-bullseye", "sliders": "fa-solid fa-sliders", "arrow": "fa-solid fa-arrow-right",
}


def icon(name: str, color: str | None = None) -> str:
    cls = ICONS.get(name, name)
    style = f' style="color:{color}"' if color else ""
    return f'<i class="{cls}"{style}></i>'


def plotly_layout(height: int = 380, **extra) -> dict:
    """Shared dark / transparent Plotly layout so every chart matches the glass cards."""
    layout = dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#eef2ff", size=12),
        colorway=COLORWAY, margin=dict(l=10, r=10, t=46, b=10),
        title=dict(font=dict(family="Space Grotesk, Inter, sans-serif", size=15)),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.18),
        xaxis=dict(gridcolor="rgba(255,255,255,.08)", zerolinecolor="rgba(255,255,255,.2)", automargin=True),
        yaxis=dict(gridcolor="rgba(255,255,255,.08)", zerolinecolor="rgba(255,255,255,.2)", automargin=True),
        hoverlabel=dict(bgcolor="rgba(10,16,48,.96)", bordercolor="#22e3ff", font=dict(color="#eef2ff")),
    )
    layout.update(extra)
    return layout


_CSS = """
<style>
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

/* ===== Midnight Neon Glass: deep navy, neon cyan/pink/violet glows, frosted dark panels ===== */
:root{
  --bg:#070b1f; --navy:#0b1033; --cyan:#22e3ff; --blue:#4f8cff; --violet:#8b6bff; --pink:#ff4fa3; --orange:#ff8a4c;
  --green:#2ee6a6; --red:#ff4d6d; --amber:#ffb547; --text:#eef2ff; --muted:rgba(226,232,255,.64);
  --glass:linear-gradient(145deg,rgba(255,255,255,.10),rgba(255,255,255,.035));
  --glass-strong:linear-gradient(145deg,rgba(255,255,255,.16),rgba(255,255,255,.06));
  --border:rgba(255,255,255,.15); --border-hot:rgba(34,227,255,.6);
  --radius:20px; --shadow:0 12px 40px rgba(0,0,0,.5), inset 0 1px 0 rgba(255,255,255,.16);
  --grad:linear-gradient(135deg,#22e3ff 0%,#8b6bff 50%,#ff4fa3 100%);
  --grad-warm:linear-gradient(135deg,#ffb547 0%,#ff4fa3 100%);
}
html, body, [class*="css"], .stApp{font-family:'Inter',sans-serif; color:var(--text);}
.stApp{
  background:
    radial-gradient(900px 640px at 0% 100%, rgba(255,60,160,.34), transparent 62%),
    radial-gradient(900px 620px at 100% 0%, rgba(0,200,255,.26), transparent 60%),
    radial-gradient(760px 520px at 55% 48%, rgba(110,70,255,.18), transparent 64%),
    radial-gradient(600px 420px at 8% 4%, rgba(40,120,255,.20), transparent 62%),
    linear-gradient(180deg,#060919 0%,#0a1031 58%,#070a1c 100%);
  background-attachment:fixed;
}
.stApp{isolation:isolate;}
.stApp::before, .stApp::after{content:""; position:fixed; z-index:-1; pointer-events:none; border-radius:50%; filter:blur(80px); opacity:.38;}
.stApp::before{width:420px; height:420px; left:38%; top:14%; background:radial-gradient(circle,#ff4fa3,transparent 70%); animation:float1 20s ease-in-out infinite;}
.stApp::after{width:460px; height:460px; right:6%; bottom:4%; background:radial-gradient(circle,#22e3ff,transparent 70%); animation:float2 24s ease-in-out infinite;}
@keyframes float1{0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(90px,70px) scale(1.15)}}
@keyframes float2{0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(-110px,-60px) scale(1.12)}}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stMainBlockContainer"], .block-container{max-width:1280px; padding-top:2.2rem; padding-bottom:4rem; animation:pageIn .55s ease both;}
@keyframes pageIn{from{opacity:0; transform:translateY(14px)} to{opacity:1; transform:none}}
@keyframes fadeUp{from{opacity:0; transform:translateY(16px)} to{opacity:1; transform:none}}
@keyframes pulseGlow{0%,100%{box-shadow:0 0 0 0 rgba(34,211,238,.35)} 50%{box-shadow:0 0 0 10px rgba(34,211,238,0)}}
@keyframes sweep{from{transform:rotate(0)} to{transform:rotate(360deg)}}
@keyframes drift{0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)}}
@keyframes blink{0%,100%{opacity:1} 50%{opacity:.35}}

h1,h2,h3,h4{font-family:'Space Grotesk','Inter',sans-serif; letter-spacing:-.01em; color:var(--text);}
h1 a, h2 a, h3 a{display:none !important;}
p, li{color:#c3d0e6; line-height:1.65;}
a{color:var(--cyan);}
code{background:rgba(34,211,238,.10); color:#7ee7f7; border-radius:6px; padding:.1rem .4rem;}
hr{border-color:var(--border);}

/* ---------- reusable classes ---------- */
.grad-text{background:var(--grad); -webkit-background-clip:text; background-clip:text; color:transparent;}
.eyebrow{display:inline-flex; gap:.55rem; align-items:center; font-size:.74rem; font-weight:600; letter-spacing:.16em; text-transform:uppercase; color:var(--cyan);}
.page-title{font-family:'Space Grotesk',sans-serif; font-size:2.15rem; font-weight:700; margin:.2rem 0 .3rem; line-height:1.15;}
.page-sub{color:var(--muted); font-size:1.02rem; max-width:860px; margin-bottom:1.4rem;}
.section-title{display:flex; align-items:center; gap:.7rem; font-family:'Space Grotesk',sans-serif; font-size:1.25rem; font-weight:600; margin:2rem 0 .9rem;}
.section-title i{color:var(--cyan); font-size:1rem; width:34px; height:34px; display:inline-grid; place-items:center; border-radius:10px; background:rgba(34,211,238,.10); border:1px solid rgba(34,211,238,.25);}

.glass, [class*="st-key-glass"]{
  background:var(--glass); backdrop-filter:blur(16px) saturate(140%); -webkit-backdrop-filter:blur(16px) saturate(140%);
  border:1px solid var(--border); border-radius:var(--radius); box-shadow:var(--shadow);
  padding:1.2rem 1.3rem; transition:border-color .25s ease, transform .25s ease, box-shadow .25s ease;
  animation:fadeUp .6s ease both;
}
.glass:hover, [class*="st-key-glass"]:hover{border-color:var(--border-hot); box-shadow:0 14px 48px rgba(34,211,238,.10), var(--shadow);}
[class*="st-key-glass"]{margin-bottom:.9rem;}

.feature-card{height:100%;}
.feature-card .ico{width:50px; height:50px; border-radius:14px; display:grid; place-items:center; font-size:1.25rem; margin-bottom:.9rem; color:#fff; background:var(--grad); box-shadow:0 8px 24px rgba(59,130,246,.35);}
.feature-card h3{margin:.1rem 0 .4rem; font-size:1.2rem;}
.feature-card p{margin:0; color:var(--muted); font-size:.95rem;}
.feature-card:hover{transform:translateY(-4px);}

.metric-card{position:relative; overflow:hidden; text-align:left; min-height:118px; animation:fadeUp .6s ease both;}
.metric-card .lbl{font-size:.74rem; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); display:flex; gap:.5rem; align-items:center;}
.metric-card .val{font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700; margin:.35rem 0 .1rem; line-height:1.1;}
.metric-card .sub{font-size:.82rem; color:var(--muted);}
.metric-card::after{content:""; position:absolute; right:-40px; top:-40px; width:120px; height:120px; border-radius:50%; background:radial-gradient(circle,var(--accent,#22e3ff) 0%,transparent 70%); opacity:.22;}
.metric-card:hover{transform:translateY(-3px);}
.metric-card .lbl i{color:var(--accent,#22e3ff);}

.pill{display:inline-flex; align-items:center; gap:.45rem; padding:.28rem .8rem; border-radius:999px; font-size:.78rem; font-weight:600; border:1px solid var(--border); background:rgba(255,255,255,.04);}
.pill.ok{color:var(--green); border-color:rgba(52,211,153,.4); background:rgba(52,211,153,.08);}
.pill.bad{color:var(--red); border-color:rgba(244,63,94,.45); background:rgba(244,63,94,.08);}
.pill.warn{color:var(--amber); border-color:rgba(251,191,36,.4); background:rgba(251,191,36,.08);}
.pill.info{color:var(--cyan); border-color:rgba(34,211,238,.4); background:rgba(34,211,238,.08);}
.dot{width:8px; height:8px; border-radius:50%; background:currentColor; display:inline-block; animation:blink 1.8s infinite;}

.result-card{display:flex; gap:1.3rem; align-items:center; padding:1.6rem 1.8rem; border-radius:22px; border:1px solid; animation:fadeUp .5s ease both; backdrop-filter:blur(16px);}
.result-card .badge{width:74px; height:74px; flex:0 0 74px; border-radius:50%; display:grid; place-items:center; font-size:1.9rem;}
.result-card .kicker{font-size:.76rem; letter-spacing:.14em; text-transform:uppercase; font-weight:600;}
.result-card .verdict{font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700; line-height:1.15; margin:.15rem 0;}
.result-card .meta{color:#c3d0e6; font-size:.95rem;}
.result-card.normal{background:linear-gradient(135deg,rgba(52,211,153,.14),rgba(16,26,56,.6)); border-color:rgba(52,211,153,.5); box-shadow:0 0 40px rgba(52,211,153,.12);}
.result-card.normal .badge{background:rgba(52,211,153,.16); color:var(--green); animation:pulseGlow 2.4s infinite;}
.result-card.normal .kicker{color:var(--green);}
.result-card.attack{background:linear-gradient(135deg,rgba(244,63,94,.16),rgba(16,26,56,.6)); border-color:rgba(244,63,94,.55); box-shadow:0 0 40px rgba(244,63,94,.15);}
.result-card.attack .badge{background:rgba(244,63,94,.18); color:var(--red);}
.result-card.attack .kicker{color:var(--red);}

.hero{position:relative; overflow:hidden; padding:3rem 2.4rem 2.6rem; border-radius:28px; border:1px solid var(--border);
  background:linear-gradient(135deg,rgba(14,24,58,.78),rgba(22,16,56,.62)); backdrop-filter:blur(18px); box-shadow:var(--shadow); animation:fadeUp .6s ease both;}
.hero h1{font-size:clamp(2rem,4.6vw,3.3rem); line-height:1.08; margin:.7rem 0 .8rem; font-weight:700;}
.hero p.sub{font-size:1.18rem; color:#b6c6e2; max-width:660px; margin:0 0 .6rem;}
.hero p.intro{color:var(--muted); max-width:700px; margin:0;}
.radar{position:absolute; right:-70px; top:50%; margin-top:-210px; width:420px; height:420px; border-radius:50%; opacity:.9; pointer-events:none;
  background:
    radial-gradient(circle, transparent 0 28%, rgba(34,211,238,.20) 28.4% 29%, transparent 29.4% 52%, rgba(34,211,238,.16) 52.4% 53%, transparent 53.4% 76%, rgba(34,211,238,.12) 76.4% 77%, transparent 77.4%),
    linear-gradient(rgba(34,211,238,.14),rgba(34,211,238,.14)) center/1px 100% no-repeat,
    linear-gradient(rgba(34,211,238,.14),rgba(34,211,238,.14)) center/100% 1px no-repeat;}
.radar::before{content:""; position:absolute; inset:0; border-radius:50%; background:conic-gradient(from 0deg, rgba(34,211,238,.38), transparent 28%); animation:sweep 5s linear infinite;}
.radar .blip{position:absolute; width:9px; height:9px; border-radius:50%; background:var(--red); box-shadow:0 0 14px var(--red); animation:blink 2.4s infinite;}
.hero .shield-float{position:absolute; right:150px; top:50%; margin-top:-34px; font-size:4rem; color:rgba(255,255,255,.9); filter:drop-shadow(0 0 22px rgba(34,211,238,.7)); animation:drift 5s ease-in-out infinite;}
@media (max-width: 900px){ .radar{opacity:.22; right:-160px} .hero .shield-float{display:none} .hero{padding:2rem 1.3rem} }

.arch{display:flex; flex-wrap:wrap; gap:.6rem; align-items:stretch; justify-content:center;}
.arch .node{flex:1 1 130px; min-width:130px; text-align:center; padding:.9rem .6rem; border-radius:14px; background:rgba(255,255,255,.04); border:1px solid var(--border); transition:all .25s;}
.arch .node:hover{border-color:var(--border-hot); transform:translateY(-3px);}
.arch .node i{font-size:1.2rem; color:var(--cyan); margin-bottom:.4rem; display:block;}
.arch .node b{display:block; font-size:.85rem; color:var(--text);}
.arch .node span{font-size:.74rem; color:var(--muted);}
.arch .join{align-self:center; color:var(--violet); opacity:.8;}

.chips{display:flex; flex-wrap:wrap; gap:.5rem;}
.chip{padding:.35rem .8rem; border-radius:999px; font-size:.82rem; background:rgba(139,92,246,.12); border:1px solid rgba(139,92,246,.35); color:#d6c8ff;}
.chip.c{background:rgba(34,211,238,.10); border-color:rgba(34,211,238,.35); color:#9cecf8;}
.chip.g{background:rgba(52,211,153,.10); border-color:rgba(52,211,153,.35); color:#a7f3d0;}

.note{border-radius:14px; padding:.85rem 1.05rem; border:1px solid var(--border); background:rgba(255,255,255,.035); font-size:.92rem; color:#c3d0e6; margin:.5rem 0 1rem;}
.note i{margin-right:.55rem;}
.note.info{border-color:rgba(34,211,238,.35); background:rgba(34,211,238,.06);} .note.info i{color:var(--cyan);}
.note.warn{border-color:rgba(251,191,36,.4); background:rgba(251,191,36,.06);} .note.warn i{color:var(--amber);}
.note.bad{border-color:rgba(244,63,94,.45); background:rgba(244,63,94,.07);} .note.bad i{color:var(--red);}
.note.ok{border-color:rgba(52,211,153,.4); background:rgba(52,211,153,.06);} .note.ok i{color:var(--green);}
.tag-sim{display:inline-flex; gap:.45rem; align-items:center; padding:.25rem .7rem; border-radius:8px; font-size:.72rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--amber); border:1px dashed rgba(251,191,36,.6); background:rgba(251,191,36,.07);}
.tag-real{display:inline-flex; gap:.45rem; align-items:center; padding:.25rem .7rem; border-radius:8px; font-size:.72rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--green); border:1px solid rgba(52,211,153,.5); background:rgba(52,211,153,.07);}

.timeline{position:relative; margin:.4rem 0 0 .6rem; padding-left:1.6rem; border-left:2px solid rgba(34,211,238,.3);}
.timeline .item{position:relative; margin-bottom:1.1rem;}
.timeline .item::before{content:""; position:absolute; left:-2.02rem; top:.35rem; width:12px; height:12px; border-radius:50%; background:var(--cyan); box-shadow:0 0 12px var(--cyan);}
.timeline .when{font-size:.74rem; letter-spacing:.1em; text-transform:uppercase; color:var(--cyan);}
.timeline .what{font-weight:600; color:var(--text);} .timeline .desc{color:var(--muted); font-size:.9rem;}

/* ---------- Streamlit widgets ---------- */
[data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(7,12,32,.96),rgba(5,8,22,.98)); border-right:1px solid var(--border);}
[data-testid="stSidebar"] *{color:#c9d6ec;}
[data-testid="stSidebarNav"] a, [data-testid="stSidebarNavLink"]{border-radius:12px; margin:2px 0; transition:all .2s;}
[data-testid="stSidebarNavLink"]:hover{background:rgba(34,211,238,.09);}
[data-testid="stSidebarNavLink"][aria-current="page"]{background:linear-gradient(135deg,rgba(34,211,238,.18),rgba(139,92,246,.18)); border:1px solid rgba(34,211,238,.3);}
.brand{display:flex; align-items:center; gap:.7rem; padding:.3rem .2rem 1rem;}
.brand .logo{width:42px; height:42px; border-radius:12px; display:grid; place-items:center; background:var(--grad); color:#fff; font-size:1.15rem; box-shadow:0 6px 20px rgba(59,130,246,.45);}
.brand b{font-family:'Space Grotesk',sans-serif; font-size:1.05rem; color:#fff; display:block; line-height:1.1;}
.brand span{font-size:.72rem; color:var(--muted);}
.side-label{font-size:.7rem; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); margin:1.1rem 0 .4rem;}
.side-row{display:flex; justify-content:space-between; gap:.5rem; font-size:.8rem; padding:.22rem 0; color:#b6c6e2;}

.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button{
  border-radius:12px; border:1px solid var(--border); background:rgba(255,255,255,.05); color:var(--text); font-weight:600; padding:.55rem 1.2rem; transition:all .2s ease;}
.stButton > button:hover, .stDownloadButton > button:hover{border-color:var(--border-hot); transform:translateY(-2px); box-shadow:0 8px 24px rgba(34,211,238,.18); color:#fff;}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"], .stDownloadButton > button[kind="primary"]{
  background:var(--grad); border:none; color:#fff; box-shadow:0 8px 28px rgba(59,130,246,.38);}
.stButton > button[kind="primary"]:hover{filter:brightness(1.1); box-shadow:0 12px 34px rgba(139,92,246,.45);}
.stButton > button:disabled{opacity:.45;}

[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"], .stTextInput input, .stNumberInput input{
  background:rgba(8,14,36,.7) !important; border-radius:10px !important; border-color:var(--border) !important; color:var(--text) !important;}
.stTextInput input:focus, .stNumberInput input:focus{border-color:var(--cyan) !important; box-shadow:0 0 0 1px var(--cyan) !important;}
label, [data-testid="stWidgetLabel"] p{color:#b6c6e2 !important; font-size:.85rem !important;}

[data-testid="stTabs"] [role="tablist"]{gap:.3rem; border-bottom:1px solid var(--border);}
[data-testid="stTabs"] button[role="tab"]{border-radius:10px 10px 0 0; padding:.55rem 1rem; color:var(--muted);}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{color:#fff; background:rgba(34,211,238,.08);}
[data-testid="stExpander"]{border:1px solid var(--border) !important; border-radius:14px !important; background:rgba(16,26,56,.35);}
[data-testid="stDataFrame"]{border:1px solid var(--border); border-radius:14px; overflow:hidden;}
[data-testid="stFileUploader"] section{background:rgba(8,14,36,.55); border:1.5px dashed rgba(34,211,238,.35); border-radius:16px;}
[data-testid="stAlert"]{border-radius:14px; backdrop-filter:blur(8px);}
[data-testid="stDialog"] > div, div[role="dialog"]{background:rgba(9,15,38,.94) !important; backdrop-filter:blur(22px); border:1px solid var(--border-hot); border-radius:22px !important; box-shadow:0 30px 80px rgba(0,0,0,.65);}
[data-testid="stMetric"]{background:var(--glass); border:1px solid var(--border); border-radius:16px; padding:.8rem 1rem;}
.stProgress > div > div > div > div{background:var(--grad);}

/* info (ⓘ) buttons next to chart titles */
[class*="st-key-info_"] button{padding:.15rem .55rem; min-height:0; border-radius:999px; font-size:.8rem;}

@media (max-width: 768px){
  .page-title{font-size:1.6rem} .metric-card .val{font-size:1.55rem}
  .result-card{flex-direction:column; text-align:center; padding:1.3rem} .result-card .verdict{font-size:1.5rem}
  [data-testid="stMainBlockContainer"], .block-container{padding-left:1rem; padding-right:1rem;}
}
@media (prefers-reduced-motion: reduce){*{animation:none !important; transition:none !important;}}
/* ---------- Midnight Neon Glass overrides ---------- */
p, li{color:rgba(226,232,255,.84);}
a{color:#7defff;}
code{background:rgba(34,227,255,.12); color:#9cf0ff;}
.glass, [class*="st-key-glass"]{backdrop-filter:blur(22px) saturate(160%); -webkit-backdrop-filter:blur(22px) saturate(160%);}
.glass:hover, [class*="st-key-glass"]:hover{border-color:rgba(34,227,255,.55); box-shadow:0 16px 50px rgba(34,227,255,.14), var(--shadow);}
.page-title{font-size:2.3rem;}
.grad-text{background:linear-gradient(100deg,#ffffff 0%,#9cf0ff 45%,#ff9bd0 100%); -webkit-background-clip:text; background-clip:text; color:transparent;}
.eyebrow{color:#22e3ff;}
.section-title i{color:#fff; background:var(--grad); border:none; box-shadow:0 6px 18px rgba(139,107,255,.45);}
.metric-card{border-top:2px solid var(--accent,#22e3ff);}
.metric-card::after{opacity:.45; width:140px; height:140px;}
.metric-card .lbl, .metric-card .sub{color:rgba(226,232,255,.66);}
[data-testid="stColumn"]:nth-child(1) .feature-card .ico{background:linear-gradient(135deg,#22e3ff,#4f8cff);}
[data-testid="stColumn"]:nth-child(2) .feature-card .ico{background:linear-gradient(135deg,#8b6bff,#ff4fa3);}
[data-testid="stColumn"]:nth-child(3) .feature-card .ico{background:linear-gradient(135deg,#ffb547,#ff4fa3);}
.feature-card p{color:rgba(226,232,255,.72);}
.hero{background:linear-gradient(135deg,rgba(255,255,255,.11),rgba(255,255,255,.03)); backdrop-filter:blur(26px) saturate(170%); border:1px solid rgba(255,255,255,.2);
  box-shadow:0 20px 70px rgba(0,0,0,.55), inset 0 1px 0 rgba(255,255,255,.2);}
.hero p.sub{color:rgba(238,242,255,.92);} .hero p.intro{color:rgba(226,232,255,.7);}
.radar::before{background:conic-gradient(from 0deg, rgba(255,79,163,.5), transparent 28%);}
.radar .blip{background:#ffb547; box-shadow:0 0 16px #ffb547;}
.arch .node, .note{background:rgba(255,255,255,.06); border-color:rgba(255,255,255,.16); backdrop-filter:blur(12px);}
.arch .node:nth-child(4n+1) i{color:#22e3ff;} .arch .node:nth-child(4n+3) i{color:#ff4fa3;} .arch .node:nth-child(4n+5) i{color:#ffb547;} .arch .node:nth-child(4n+7) i{color:#2ee6a6;}
.arch .node span, .side-row, .note{color:rgba(226,232,255,.8);}
.chip{background:rgba(255,79,163,.14); border-color:rgba(255,79,163,.5); color:#ffd3e8;}
.chip.c{background:rgba(34,227,255,.12); border-color:rgba(34,227,255,.5); color:#c9f8ff;}
.chip.g{background:rgba(46,230,166,.12); border-color:rgba(46,230,166,.5); color:#cdfbe9;}
.timeline{border-left-color:rgba(255,255,255,.25);}
.timeline .item::before{background:#ff4fa3; box-shadow:0 0 14px #ff4fa3;}
.timeline .when{color:#22e3ff;}
.result-card{backdrop-filter:blur(22px) saturate(170%);}
.result-card.normal{background:linear-gradient(135deg,rgba(46,230,166,.24),rgba(255,255,255,.04)); border-color:rgba(46,230,166,.75);}
.result-card.attack{background:linear-gradient(135deg,rgba(255,77,109,.28),rgba(255,255,255,.04)); border-color:rgba(255,77,109,.8);}
.result-card .meta{color:rgba(238,242,255,.88);}
.pill{background:rgba(255,255,255,.08);}

/* ----- dashboard pieces (Overview page) ----- */
.kpi{position:relative; overflow:hidden; padding:1rem 1.1rem .9rem; min-height:186px; display:flex; flex-direction:column; gap:.1rem; animation:fadeUp .6s ease both;}
.kpi .top{display:flex; gap:.7rem; align-items:center;}
.kpi .ic{font-size:1.45rem; color:var(--accent); filter:drop-shadow(0 0 9px var(--accent));}
.kpi .lbl{font-size:.68rem; letter-spacing:.09em; text-transform:uppercase; color:var(--muted); line-height:1.25; font-weight:600;}
.kpi .val{font-family:'Space Grotesk',sans-serif; font-size:2.05rem; font-weight:700; line-height:1.1; margin:.35rem 0 .1rem;}
.kpi .bot{display:flex; justify-content:space-between; align-items:flex-end; gap:.5rem;}
.kpi .delta{font-size:.78rem; color:var(--muted); line-height:1.35; margin-top:auto;}
.kpi .delta b{color:var(--accent);}
.kpi svg{display:block; width:100%; height:34px; margin:.25rem 0 .2rem; overflow:visible;}
.panel-title{font-family:'Space Grotesk',sans-serif; font-size:.82rem; letter-spacing:.12em; text-transform:uppercase; color:rgba(238,242,255,.9); font-weight:600; margin:0 0 .5rem;}
.alerts-wrap{overflow-x:auto;}
.alerts{width:100%; border-collapse:collapse; font-size:.9rem;}
.alerts th{font-size:.64rem; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); font-weight:600; text-align:left; padding:.35rem .4rem; border-bottom:1px solid rgba(255,255,255,.14);}
.alerts td{padding:.55rem .4rem; border-bottom:1px solid rgba(255,255,255,.07); color:#eef2ff; white-space:nowrap;}
.alerts td:nth-child(4){white-space:normal; font-size:.82rem; line-height:1.3;}
.alerts tr:hover td{background:rgba(255,255,255,.05);}
.sev{display:inline-block; padding:.16rem .6rem; border-radius:999px; font-size:.7rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; border:1px solid; background:rgba(0,0,0,.18);}
.sev.critical{color:#ff6b86; border-color:rgba(255,77,109,.8); box-shadow:0 0 14px rgba(255,77,109,.35);}
.sev.high{color:#ff8fc4; border-color:rgba(255,79,163,.8); box-shadow:0 0 14px rgba(255,79,163,.3);}
.sev.medium{color:#ffc971; border-color:rgba(255,181,71,.8); box-shadow:0 0 14px rgba(255,181,71,.28);}
.bar-mini{display:inline-block; height:6px; border-radius:6px; background:var(--grad); vertical-align:middle; margin-right:.5rem;}
.info-grid{display:grid; grid-template-columns:auto 1fr; gap:.45rem 1rem; font-size:.9rem;}
.info-grid span:nth-child(odd){color:var(--muted);} .info-grid span:nth-child(even){font-weight:600; color:#eef2ff;}

[data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(255,255,255,.09),rgba(255,255,255,.025)) !important; backdrop-filter:blur(26px) saturate(170%); -webkit-backdrop-filter:blur(26px) saturate(170%); border-right:1px solid rgba(255,255,255,.16);}
[data-testid="stSidebar"] > div:first-child{background:transparent !important;}
[data-testid="stSidebar"] *{color:rgba(238,242,255,.9);}
[data-testid="stSidebarNavLink"]:hover{background:rgba(255,255,255,.09);}
[data-testid="stSidebarNavLink"][aria-current="page"]{background:linear-gradient(135deg,rgba(34,227,255,.22),rgba(255,79,163,.22)); border:1px solid rgba(255,255,255,.34); box-shadow:0 6px 20px rgba(255,79,163,.2);}
.brand .logo{box-shadow:0 8px 26px rgba(139,107,255,.5);}

.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button{background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.2); backdrop-filter:blur(12px);}
.stButton > button:hover, .stDownloadButton > button:hover{background:rgba(255,255,255,.14); border-color:rgba(34,227,255,.7); box-shadow:0 10px 28px rgba(34,227,255,.2);}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"], .stDownloadButton > button[kind="primary"]{background:var(--grad); box-shadow:0 10px 30px rgba(139,107,255,.42), inset 0 1px 0 rgba(255,255,255,.4);}
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"], .stTextInput input, .stNumberInput input{background:rgba(255,255,255,.07) !important; border-color:rgba(255,255,255,.2) !important; color:#fff !important; backdrop-filter:blur(10px);}
.stTextInput input:focus, .stNumberInput input:focus{border-color:#22e3ff !important; box-shadow:0 0 0 1px #22e3ff !important;}
label, [data-testid="stWidgetLabel"] p{color:rgba(226,232,255,.86) !important;}
[data-baseweb="popover"] > div, [data-baseweb="menu"], ul[role="listbox"]{background:rgba(12,18,54,.94) !important; backdrop-filter:blur(24px); border:1px solid rgba(255,255,255,.2); border-radius:14px;}
[data-testid="stTabs"] [role="tablist"]{border-bottom:1px solid rgba(255,255,255,.16);}
[data-testid="stTabs"] button[role="tab"]{color:rgba(226,232,255,.66);}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{color:#fff; background:rgba(255,255,255,.08);}
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{background:linear-gradient(90deg,#22e3ff,#ff4fa3) !important; height:3px !important;}
[data-testid="stExpander"]{background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.16) !important; backdrop-filter:blur(16px);}
[data-testid="stDataFrame"]{border:1px solid rgba(255,255,255,.16); background:rgba(255,255,255,.04);}
[data-testid="stFileUploader"] section{background:rgba(255,255,255,.05); border:1.5px dashed rgba(34,227,255,.5);}
[data-testid="stAlert"]{background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.2);}
[data-testid="stDialog"] > div, div[role="dialog"]{background:linear-gradient(135deg,rgba(24,22,86,.94),rgba(10,24,66,.94)) !important; backdrop-filter:blur(30px) saturate(170%); border:1px solid rgba(255,255,255,.3) !important;}
[data-testid="stMetric"]{background:var(--glass); backdrop-filter:blur(16px);}
[data-testid="stSlider"] [role="slider"]{background:#ff4fa3 !important; box-shadow:0 0 0 4px rgba(255,79,163,.28);}
[data-testid="stToolbar"], [data-testid="stHeader"] *{color:#fff;}
[class*="st-key-info_"] button, [class*="st-key-info_"] button *{white-space:nowrap !important;}
[data-testid="stColumn"]:not(:has([data-testid="stColumn"])):has([class*="st-key-info_"]){min-width:96px !important; flex:0 0 auto !important; width:auto !important;}
</style>
"""


def inject_css() -> None:
    """Inject the global stylesheet (call once per run from the entrypoint)."""
    css = "\n".join(line for line in _CSS.splitlines() if line.strip())  # blank lines would end the HTML block
    if hasattr(st, "html"):
        st.html(css)  # injected as-is, <style> is applied globally
    else:  # very old Streamlit
        st.markdown(css, unsafe_allow_html=True)
''',
    "components.cards": r'''
"""Reusable presentation building blocks (glass cards, metric cards, notices)."""
from __future__ import annotations

import html
from contextlib import contextmanager
from typing import Iterable

import streamlit as st

from components.styles import COLORS, icon

_E = html.escape


def reset_keys() -> None:
    """Called at the top of every run so generated widget/container keys stay deterministic."""
    st.session_state["_ui_counter"] = 0


def _next(prefix: str) -> str:
    n = st.session_state.get("_ui_counter", 0) + 1
    st.session_state["_ui_counter"] = n
    return f"{prefix}_{n}"


@contextmanager
def glass():
    """Frosted-glass container for any Streamlit content (charts, tables, forms)."""
    with st.container(key=_next("glass")):
        yield


def md(html_str: str) -> None:
    """Render raw HTML. Lines are stripped so Markdown never treats indentation as a code block."""
    st.markdown("\n".join(line.strip() for line in html_str.splitlines() if line.strip()), unsafe_allow_html=True)


def page_header(ico: str, eyebrow: str, title: str, subtitle: str = "") -> None:
    md(f"""<div class="eyebrow">{icon(ico)} {_E(eyebrow)}</div>
    <div class="page-title grad-text">{_E(title)}</div>
    <div class="page-sub">{_E(subtitle)}</div>""")


def section(title: str, ico: str = "chart") -> None:
    md(f'<div class="section-title">{icon(ico)}<span>{_E(title)}</span></div>')


def note(kind: str, text: str, ico: str | None = None) -> None:
    """kind: info | warn | bad | ok.  ``text`` may contain simple inline HTML (<b>, <code>)."""
    default = {"info": "search", "warn": "warning", "bad": "warning", "ok": "check"}[kind]
    md(f'<div class="note {kind}">{icon(ico or default)}{text}</div>')


def pill(text: str, kind: str = "info", dot: bool = False) -> str:
    d = '<span class="dot"></span>' if dot else ""
    return f'<span class="pill {kind}">{d}{_E(text)}</span>'


def provenance_tag(kind: str) -> str:
    """'real' = computed from the user's real data / reports; 'sim' = simulation or synthetic."""
    if kind == "sim":
        return f'<span class="tag-sim">{icon("flask")} Simulation / synthetic demonstration</span>'
    return f'<span class="tag-real">{icon("check")} Computed from your data</span>'


def feature_card(ico: str, title: str, body: str) -> str:
    return (f'<div class="glass feature-card"><div class="ico">{icon(ico)}</div>'
            f'<h3>{_E(title)}</h3><p>{_E(body)}</p></div>')


def metric_card_html(label: str, value: str, ico: str = "chart", sub: str = "", accent: str | None = None, delay: float = 0) -> str:
    acc = accent or COLORS["cyan"]
    return (f'<div class="glass metric-card" style="--accent:{acc}; animation-delay:{delay:.2f}s">'
            f'<div class="lbl">{icon(ico)} {_E(label)}</div><div class="val">{_E(str(value))}</div>'
            f'<div class="sub">{_E(sub) if sub else "&nbsp;"}</div></div>')


def metric_row(items: Iterable[dict]) -> None:
    """items: dicts with label, value, icon, sub, accent. One responsive column per item."""
    items = list(items)
    cols = st.columns(len(items))
    for i, (col, it) in enumerate(zip(cols, items)):
        with col:
            md(metric_card_html(it["label"], it["value"], it.get("icon", "chart"), it.get("sub", ""), it.get("accent"), i * 0.07))


def result_card(attack: bool, verdict: str, meta: str = "", kicker: str | None = None) -> None:
    kind = "attack" if attack else "normal"
    ico = "warning" if attack else "shield"
    kicker = kicker or ("Attack detected" if attack else "Normal traffic")
    md(f'<div class="result-card {kind}"><div class="badge">{icon(ico)}</div><div>'
       f'<div class="kicker">{_E(kicker)}</div><div class="verdict">{_E(verdict)}</div>'
       f'<div class="meta">{meta}</div></div></div>')


def chips(items: Iterable[str], style: str = "") -> None:
    md('<div class="chips">' + "".join(f'<span class="chip {style}">{_E(i)}</span>' for i in items) + "</div>")


def timeline(items: list[dict]) -> None:
    md('<div class="timeline">' + "".join(
        f'<div class="item"><div class="when">{_E(str(i.get("when", "")))}</div>'
        f'<div class="what">{_E(str(i.get("title", "")))}</div><div class="desc">{_E(str(i.get("desc", "")))}</div></div>'
        for i in items) + "</div>")


def architecture_flow(nodes: list[tuple[str, str, str]]) -> None:
    parts = []
    for i, (ico, title, sub) in enumerate(nodes):
        parts.append(f'<div class="node">{icon(ico)}<b>{_E(title)}</b><span>{_E(sub)}</span></div>')
        if i < len(nodes) - 1:
            parts.append(f'<div class="join">{icon("arrow")}</div>')
    md('<div class="arch">' + "".join(parts) + "</div>")


def missing_artifacts_panel(statuses: list[dict], needed: Iterable[str] | None = None, title: str = "Required files are missing") -> bool:
    """Show a friendly panel for missing artifacts. Returns True if something is missing."""
    needed = set(needed) if needed else None
    missing = [s for s in statuses if not s["exists"] and (needed is None or s["key"] in needed)]
    if not missing:
        return False
    rows = "".join(f'<li><code>{_E(s["rel"])}</code> - {_E(s["purpose"])}</li>' for s in missing)
    note("warn", f"<b>{_E(title)}.</b> Place these files in the project folder and refresh the page:<ul style='margin:.5rem 0 0 1rem'>{rows}</ul>")
    return True


# --------------------------------------------------------------------------- #
# dashboard cards
# --------------------------------------------------------------------------- #
def sparkline_svg(values, color: str = "#22e3ff", w: int = 108, h: int = 36) -> str:
    """Tiny inline-SVG trend line with a soft area fill (values must be real data)."""
    import numpy as np

    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if len(v) < 2:
        return ""
    lo, hi = float(v.min()), float(v.max())
    span = (hi - lo) or 1.0
    pts = [(i * w / (len(v) - 1), h - 4 - (x - lo) / span * (h - 8)) for i, x in enumerate(v)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"0,{h} {line} {w},{h}"
    gid = f"g{abs(hash((color, round(lo, 6), round(hi, 6), len(v)))) % 10**8}"
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none"><defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{color}" stop-opacity=".35"/><stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
            f'<polygon points="{area}" fill="url(#{gid})"/>'
            f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" '
            f'style="filter:drop-shadow(0 0 4px {color})"/></svg>')


def kpi_card_html(label: str, value: str, ico: str, detail: str = "", series=None, accent: str = "#22e3ff", delay: float = 0) -> str:
    spark = sparkline_svg(series, accent, 200, 34) if series is not None else '<div style="height:34px"></div>'
    return (f'<div class="glass kpi" style="--accent:{accent}; animation-delay:{delay:.2f}s">'
            f'<div class="top"><span class="ic">{icon(ico)}</span><span class="lbl">{_E(label)}</span></div>'
            f'<div class="val">{_E(str(value))}</div>{spark}'
            f'<div class="delta">{detail}</div></div>')


def kpi_row(items: Iterable[dict]) -> None:
    """items: dicts with label, value, icon, detail (may contain <b>), series, accent."""
    items = list(items)
    for i, (col, it) in enumerate(zip(st.columns(len(items)), items)):
        with col:
            md(kpi_card_html(it["label"], it["value"], it.get("icon", "chart"), it.get("detail", ""), it.get("series"), it.get("accent", "#22e3ff"), i * 0.07))


def severity(confidence: float) -> tuple[str, str]:
    """Confidence band used for the detections table (a band of model confidence, not a risk score)."""
    if confidence >= 0.99:
        return "critical", "Critical"
    if confidence >= 0.90:
        return "high", "High"
    return "medium", "Medium"


def detections_table(rows: list[dict]) -> None:
    """rows: dicts with id, category, confidence (0-1), features (str)."""
    body = ""
    for r in rows:
        cls, txt = severity(r["confidence"])
        body += (f'<tr><td>#{_E(str(r["id"]))}</td><td><b>{_E(str(r["category"]))}</b></td>'
                 f'<td><span class="bar-mini" style="width:{max(6, int(r["confidence"] * 40))}px"></span>{r["confidence"]:.1%}</td>'
                 f'<td style="color:rgba(226,232,255,.7)">{_E(str(r.get("note", "")))}</td>'
                 f'<td><span class="sev {cls}">{txt}</span></td></tr>')
    md(f'<div class="alerts-wrap"><table class="alerts"><thead><tr><th>Record</th><th>Predicted category</th><th>Confidence</th><th>Most unusual feature</th><th>Band</th></tr></thead>'
       f'<tbody>{body}</tbody></table></div>')
''',
    "components.charts": r'''
"""Plotly figure builders in the shared dark/glass theme (presentation only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.styles import COLORS, COLORWAY, plotly_layout

_CONFIG = {"displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def show(fig: go.Figure, key: str | None = None) -> None:
    """Render a figure responsively."""
    st.plotly_chart(fig, width="stretch", config=_CONFIG, key=key)


def _fig(title: str, height: int = 380, **extra) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**plotly_layout(height, title=dict(text=title), **extra))
    return fig


def bar(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None, horizontal: bool = False,
        height: int = 380, log: bool = False, text: bool = True, colors: list[str] | None = None) -> go.Figure:
    fig = _fig(title, height)
    groups = [(None, df)] if color is None else list(df.groupby(color, sort=False))
    for i, (name, g) in enumerate(groups):
        kw = dict(x=g[y], y=g[x], orientation="h") if horizontal else dict(x=g[x], y=g[y])
        fig.add_bar(name=None if name is None else str(name), showlegend=name is not None,
                    marker=dict(color=(colors or COLORWAY)[i % len(colors or COLORWAY)] if name is not None else
                                [(colors if colors and len(colors) > 1 else COLORWAY)[j % len(colors if colors and len(colors) > 1 else COLORWAY)]
                                 for j in range(len(g))],  # one vivid colour per bar
                                line=dict(width=0), cornerradius=6),
                    text=g[y].map(lambda v: f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}") if text else None,
                    textposition="outside", cliponaxis=False, **kw)
    if log:
        fig.update_yaxes(type="log") if not horizontal else fig.update_xaxes(type="log")
    if horizontal:
        fig.update_yaxes(autorange="reversed")
    fig.update_layout(barmode="group")
    return fig


def donut(labels: list[str], values: list[float], title: str, colors: list[str] | None = None, height: int = 380) -> go.Figure:
    fig = _fig(title, height)
    fig.add_pie(labels=labels, values=values, hole=0.62, sort=False,
                marker=dict(colors=colors or COLORWAY, line=dict(color="rgba(255,255,255,.55)", width=1.5)),
                textinfo="percent", hovertemplate="%{label}<br>%{value:,} records (%{percent})<extra></extra>")
    return fig


def heatmap(z: np.ndarray, x: list[str], y: list[str], title: str, colorscale=None, zmid=None,
            fmt: str = ".2f", height: int = 480, zmin=None, zmax=None, text: bool = True) -> go.Figure:
    fig = _fig(title, height)
    colorscale = colorscale or ([[0, "#22e3ff"], [.5, "#2a2470"], [1, "#ff4fa3"]] if zmid is not None  # diverging: cyan - violet - pink
                                else [[0, "#1a1f6b"], [.35, "#6a5cff"], [.7, "#ff4fa3"], [1, "#ffd166"]])
    fig.add_heatmap(z=z, x=x, y=y, colorscale=colorscale, zmid=zmid, zmin=zmin, zmax=zmax,
                    text=np.vectorize(lambda v: format(v, fmt))(z) if text else None,
                    texttemplate="%{text}" if text else None, textfont=dict(size=10),
                    hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>", colorbar=dict(thickness=12))
    fig.update_yaxes(autorange="reversed")
    return fig


def confusion(cm: np.ndarray, labels: list[str], title: str = "Confusion matrix", normalize: bool = False) -> go.Figure:
    cm = np.asarray(cm, dtype=float)
    z = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    txt = [[f"{cm[i, j]:,.0f}<br>({z[i, j]:.1%})" if normalize else f"{cm[i, j]:,.0f}" for j in range(cm.shape[1])] for i in range(cm.shape[0])]
    n = len(labels)
    fig = _fig(title, height=min(760, max(380, 60 * n + 160)))
    fig.add_heatmap(z=z, x=labels, y=labels, text=txt, texttemplate="%{text}", colorscale=[[0, "#1a1f6b"], [.35, "#6a5cff"], [.7, "#ff4fa3"], [1, "#ffd166"]],
                    hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{text}<extra></extra>", colorbar=dict(thickness=12), textfont=dict(size=11 if n < 10 else 9))
    fig.update_xaxes(title="Predicted", side="bottom")
    fig.update_yaxes(title="Actual", autorange="reversed")
    return fig


def contributions(df: pd.DataFrame, label_col: str, value_col: str, title: str, top: int = 15, height: int | None = None,
                  pos_name: str = "pushes toward predicted class", neg_name: str = "pushes away") -> go.Figure:
    """Diverging horizontal bars (green = positive contribution, red = negative)."""
    d = df.head(top).iloc[::-1]
    fig = _fig(title, height or max(340, 28 * len(d) + 120))
    colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in d[value_col]]
    fig.add_bar(y=d[label_col], x=d[value_col], orientation="h", marker=dict(color=colors, cornerradius=4),
                text=d[value_col].map(lambda v: f"{v:+.3f}"), textposition="outside", cliponaxis=False,
                hovertemplate="%{y}<br>%{x:+.4f}<extra></extra>", showlegend=False)
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,.35)", width=1))
    fig.update_layout(margin=dict(l=10, r=40, t=46, b=10))
    fig.update_xaxes(title=f"green: {pos_name} · red: {neg_name}")
    return fig


def importance(df: pd.DataFrame, title: str, top: int = 15) -> go.Figure:
    d = df.head(top).iloc[::-1]
    fig = _fig(title, max(340, 28 * len(d) + 120))
    fig.add_bar(y=d["feature"], x=d["importance"], orientation="h", showlegend=False,
                marker=dict(color=d["importance"], colorscale=[[0, "#3f8cff"], [.5, "#8b6bff"], [1, "#ff4fa3"]], cornerradius=4),
                text=d["importance"].map(lambda v: f"{v:.4f}"), textposition="outside", cliponaxis=False)
    fig.update_xaxes(title="mean |SHAP value|")
    return fig


def metric_bars(df: pd.DataFrame, metrics: list[str], labels: dict[str, str], title: str, height: int = 420) -> go.Figure:
    """Grouped bars: one group per model, one bar per metric."""
    fig = _fig(title, height)
    for i, m in enumerate(metrics):
        fig.add_bar(x=df["model"], y=df[m], name=labels.get(m, m), marker=dict(color=COLORWAY[i % len(COLORWAY)], cornerradius=5),
                    hovertemplate="%{x}<br>" + labels.get(m, m) + ": %{y:.4f}<extra></extra>")
    fig.update_layout(barmode="group")
    return fig


def radar(df: pd.DataFrame, metrics: list[str], labels: dict[str, str], title: str) -> go.Figure:
    fig = _fig(title, 440)
    cats = [labels.get(m, m) for m in metrics]
    for i, (_, r) in enumerate(df.iterrows()):
        vals = [float(r[m]) if pd.notna(r[m]) else 0 for m in metrics]
        fig.add_scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], name=r["model"], fill="toself", opacity=.55,
                             line=dict(color=COLORWAY[i % len(COLORWAY)]))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(gridcolor="rgba(255,255,255,.22)", range=[max(0, float(np.nanmin(df[metrics].to_numpy(float))) - .05), 1]),
                                 angularaxis=dict(gridcolor="rgba(255,255,255,.22)")))
    return fig


def histogram_compare(series: dict[str, pd.Series], feature: str, title: str, log_y: bool = False, bins: int = 50) -> go.Figure:
    fig = _fig(title, 400)
    for i, (name, s) in enumerate(series.items()):
        fig.add_histogram(x=s, name=name, opacity=.62, nbinsx=bins, histnorm="probability",
                          marker=dict(color=[COLORS["green"], COLORS["red"], COLORS["violet"], COLORS["amber"]][i % 4]))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title=feature)
    fig.update_yaxes(title="share of records", type="log" if log_y else "linear")
    return fig


def box_compare(frames: dict[str, pd.Series], feature: str, title: str, log_y: bool = False) -> go.Figure:
    fig = _fig(title, 400)
    for i, (name, s) in enumerate(frames.items()):
        fig.add_box(y=s, name=name, boxmean=True, marker=dict(color=[COLORS["green"], COLORS["red"], COLORS["violet"], COLORS["amber"]][i % 4]))
    fig.update_yaxes(title=feature, type="log" if log_y else "linear")
    return fig


def probabilities(names: list[str], probs: np.ndarray, title: str = "Class probabilities", top: int = 8, highlight: str | None = None) -> go.Figure:
    order = np.argsort(-probs)[:top][::-1]
    fig = _fig(title, max(240, 34 * len(order) + 110))
    fig.add_bar(y=[names[i] for i in order], x=[float(probs[i]) for i in order], orientation="h", showlegend=False,
                marker=dict(color=[COLORS["cyan"] if names[i] == highlight else "rgba(255,255,255,.38)" for i in order], cornerradius=4),
                text=[f"{probs[i]:.1%}" for i in order], textposition="outside", cliponaxis=False)
    fig.update_xaxes(range=[0, 1.12], tickformat=".0%")
    return fig


# ------------------------------- drift charts -------------------------------- #
def drift_error_chart(batches: pd.DataFrame, events: pd.DataFrame, show_online: bool = True) -> go.Figure:
    fig = _fig("Prediction error per batch (lower is better)", 420)
    fig.add_scatter(x=batches["batch"], y=batches["static_error"], name="Static model (never updated)", mode="lines+markers",
                    line=dict(color=COLORS["red"], width=2.5), marker=dict(size=6))
    fig.add_scatter(x=batches["batch"], y=batches["adaptive_error"], name="Adaptive drift-aware model", mode="lines+markers",
                    line=dict(color=COLORS["cyan"], width=2.5), marker=dict(size=6))
    if show_online and "online_error" in batches:
        fig.add_scatter(x=batches["batch"], y=batches["online_error"], name="River online learner", mode="lines",
                        line=dict(color=COLORS["violet"], width=2, dash="dot"))
    alerts = events[events["type"] == "Drift alert"] if len(events) else events
    for b in alerts["batch"].unique() if len(alerts) else []:
        fig.add_vline(x=int(b), line=dict(color=COLORS["amber"], width=1.5, dash="dash"))
    if len(alerts):
        ys = batches.set_index("batch").loc[alerts["batch"].unique(), "adaptive_error"]
        fig.add_scatter(x=ys.index, y=ys.values, mode="markers", name="ADWIN drift alert",
                        marker=dict(symbol="diamond", size=14, color=COLORS["amber"], line=dict(color="#fff", width=1)))
    fig.update_xaxes(title="Batch (sequential stream order)", dtick=max(1, len(batches) // 15))
    fig.update_yaxes(title="Error rate", tickformat=".0%", rangemode="tozero")
    return fig


def drift_timeline(batches: pd.DataFrame) -> go.Figure:
    fig = _fig("Drift detection timeline", 210)
    state = np.where(batches["retrained"], 2, np.where(batches["drift_detected"], 1, 0))
    names = {0: "Stable", 1: "Drift alert", 2: "Alert + retrained"}
    cols = {0: "rgba(52,211,153,.55)", 1: COLORS["amber"], 2: COLORS["violet"]}
    for s in (0, 1, 2):
        m = state == s
        if m.any():
            fig.add_bar(x=batches["batch"][m], y=np.ones(m.sum()), name=names[s], marker=dict(color=cols[s]),
                        hovertemplate="Batch %{x}<extra>" + names[s] + "</extra>")
    fig.update_layout(barmode="stack", bargap=0.08)
    fig.update_yaxes(visible=False)
    fig.update_xaxes(title="Batch", dtick=max(1, len(batches) // 15))
    return fig


def adwin_chart(err: np.ndarray, positions: list[int], window: int = 500) -> go.Figure:
    """Rolling error rate of the monitored stream with ADWIN alert positions."""
    fig = _fig("Monitored error stream (rolling mean) and ADWIN alerts", 340)
    roll = pd.Series(err).rolling(window, min_periods=max(10, window // 10)).mean()
    step = max(1, len(roll) // 3000)
    fig.add_scatter(x=np.arange(len(roll))[::step], y=roll.to_numpy()[::step], mode="lines", name=f"Rolling error ({window} records)",
                    line=dict(color=COLORS["cyan"], width=2), fill="tozeroy", fillcolor="rgba(56,225,255,.14)")
    for p in positions:
        fig.add_vline(x=p, line=dict(color=COLORS["amber"], width=1.4, dash="dash"))
    fig.update_xaxes(title="Record index")
    fig.update_yaxes(title="Error rate", tickformat=".0%", rangemode="tozero")
    return fig



# --------------------------------------------------------------------------- #
# dashboard charts
# --------------------------------------------------------------------------- #
def confidence_histogram(values: np.ndarray, title: str = "", bins: int = 24) -> go.Figure:
    """Distribution of the binary model's attack probability (0-1) on real records."""
    fig = _fig(title, 300)
    counts, edges = np.histogram(np.asarray(values, float), bins=bins, range=(0, 1))
    mids = (edges[:-1] + edges[1:]) / 2
    fig.add_bar(x=mids, y=counts, marker=dict(color=counts, colorscale=[[0, "#4f8cff"], [.5, "#22e3ff"], [1, "#ff4fa3"]], cornerradius=3),
                hovertemplate="P(attack) %{x:.2f}<br>%{y:,} records<extra></extra>", width=(1 / bins) * 0.86)
    fig.update_layout(showlegend=False, bargap=0.1, margin=dict(l=10, r=10, t=10, b=10))
    fig.update_xaxes(title="P(attack)", range=[0, 1])
    fig.update_yaxes(title="records (log scale)", type="log", tickvals=[1, 10, 100, 1000, 10000], ticktext=["1", "10", "100", "1k", "10k"])
    return fig


def model_lines(df: pd.DataFrame, metrics: list[str], labels: dict[str, str], title: str = "", height: int = 340) -> go.Figure:
    """One neon line per metric across the models of a real evaluation report."""
    fig = _fig(title, height)
    for i, m in enumerate(metrics):
        if m not in df.columns:
            continue
        fig.add_scatter(x=df["model"], y=df[m], mode="lines+markers", name=labels.get(m, m),
                        line=dict(width=3, color=COLORWAY[i % len(COLORWAY)], shape="spline", smoothing=.6), marker=dict(size=8, line=dict(color="#fff", width=1)))
    fig.update_layout(margin=dict(l=10, r=10, t=70, b=10), legend=dict(orientation="h", y=1.22, x=0))
    fig.update_yaxes(tickformat=".0%")
    return fig


def signature_heatmap(z: np.ndarray, classes: list[str], features: list[str], title: str = "") -> go.Figure:
    """Mean standardised feature value per traffic category (z-scores; real records)."""
    fig = _fig(title, max(470, 40 * len(classes) + 210))
    fig.add_heatmap(z=z, x=features, y=classes, zmid=0, colorscale=[[0, "#22e3ff"], [.5, "#15163f"], [1, "#ff4fa3"]],
                    hovertemplate="%{y}<br>%{x}: %{z:+.2f} σ<extra></extra>", colorbar=dict(thickness=10, title=dict(text="σ", side="right")))
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    fig.update_xaxes(tickangle=-35, showgrid=False); fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def beeswarm(vals: np.ndarray, X: np.ndarray, features: list[str], top: int = 15, title: str = "") -> go.Figure:
    """Native SHAP summary (beeswarm): x = SHAP value, colour = feature value (low blue -> high pink)."""
    vals = np.asarray(vals, float); X = np.asarray(X, float)
    order = np.argsort(-np.abs(vals).mean(axis=0))[:top]
    rng = np.random.default_rng(0)
    fig = _fig(title, max(360, 30 * len(order) + 140))
    xs, ys, cs, hv = [], [], [], []
    for rank, j in enumerate(order):
        col = X[:, j]
        lo, hi = np.nanpercentile(col, 5), np.nanpercentile(col, 95)
        norm = np.clip((col - lo) / (hi - lo), 0, 1) if hi > lo else np.full(len(col), .5)
        v = vals[:, j]
        # jitter proportional to local point density so the swarm shape is visible
        dens = np.histogram(v, bins=30)[0][np.clip(np.digitize(v, np.histogram(v, bins=30)[1]) - 1, 0, 29)]
        jit = rng.uniform(-1, 1, len(v)) * np.clip(dens / max(dens.max(), 1), .08, 1) * .38
        xs.append(v); ys.append(rank + jit); cs.append(norm); hv.append(np.array([f"{features[j]}<br>value {c:,.4g}" for c in col]))
    fig.add_scatter(x=np.concatenate(xs), y=np.concatenate(ys), mode="markers", text=np.concatenate(hv), hovertemplate="%{text}<br>SHAP %{x:+.4f}<extra></extra>",
                    marker=dict(size=6, opacity=.85, color=np.concatenate(cs), colorscale=[[0, "#3f8cff"], [.5, "#8b6bff"], [1, "#ff4fa3"]],
                                colorbar=dict(title=dict(text="Feature value", side="right"), tickvals=[0, 1], ticktext=["Low", "High"], thickness=10, len=.8)))
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(order))), ticktext=[features[j] for j in order], autorange="reversed", showgrid=True)
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,.4)", width=1))
    fig.update_xaxes(title="SHAP value (impact on model output)")
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    return fig


_POLICY_COLORS = {"Static": "#ff4d6d", "Periodic": "#ffb547", "Drift-triggered": "#22e3ff", "Online HAT": "#8b6bff"}


def policy_accuracy_chart(acc: dict[str, np.ndarray], onset_batch: int, retrains: dict[str, list[int]] | None = None,
                          alerts: list[int] | None = None, height: int = 420) -> go.Figure:
    """Per-batch accuracy of each update policy; onset line, retrain markers (diamonds) and ADWIN alerts (dashed)."""
    fig = _fig("", height, legend=dict(orientation="h", y=1.14, x=0))
    for name, a in acc.items():
        x = np.arange(1, len(a) + 1)
        fig.add_scatter(x=x, y=a, name=name, mode="lines", line=dict(color=_POLICY_COLORS.get(name, COLORS["cyan"]), width=2.4))
        rb = (retrains or {}).get(name) or []
        if rb:
            fig.add_scatter(x=rb, y=[a[b - 1] for b in rb], mode="markers", showlegend=False, name=f"{name} retrain",
                            marker=dict(symbol="diamond", size=11, color=_POLICY_COLORS.get(name), line=dict(color="#fff", width=1)))
    fig.add_vline(x=onset_batch + 0.5, line=dict(color="#ffffff", width=1.5, dash="dot"),
                  annotation_text="drift onset", annotation_position="bottom right", annotation_font_color="#cbd5ff")
    for b in alerts or []:
        fig.add_vline(x=b, line=dict(color=COLORS["amber"], width=1, dash="dash"))
    fig.update_xaxes(title="Batch (sequential stream order)")
    fig.update_yaxes(title="Accuracy", tickformat=".1%")
    return fig


def category_bars(df: pd.DataFrame, series: list[str], height: int = 360) -> go.Figure:
    """Grouped bars: detection rate per traffic category (df index = category, columns = series)."""
    fig = _fig("", height, barmode="group", legend=dict(orientation="h", y=1.14, x=0))
    for s in series:
        fig.add_bar(x=df.index.astype(str), y=df[s], name=s, marker=dict(color=_POLICY_COLORS.get(s, COLORS["cyan"]), cornerradius=5))
    fig.update_yaxes(title="Detected / correct (%)", range=[0, 105])
    return fig


def threshold_chart(grid: np.ndarray, f1: list[float], fpr: list[float], t_star: float, height: int = 340) -> go.Figure:
    fig = _fig("", height, legend=dict(orientation="h", y=1.14, x=0))
    fig.add_scatter(x=grid, y=f1, name="F1", mode="lines", line=dict(color=COLORS["cyan"], width=2.5))
    fig.add_scatter(x=grid, y=fpr, name="False-positive rate", mode="lines", line=dict(color=COLORS["red"], width=2.5))
    fig.add_vline(x=0.5, line=dict(color="#fff", width=1, dash="dot"), annotation_text="default 0.5")
    fig.add_vline(x=t_star, line=dict(color=COLORS["amber"], width=1.5, dash="dash"), annotation_text=f"tuned {t_star:.2f}", annotation_position="top")
    fig.update_xaxes(title="Decision threshold on P(attack)")
    fig.update_yaxes(title="Score on the evaluation stream", range=[0, 1.02])
    return fig
''',
    "components.modals": r'''
"""Popup dialogs (st.dialog) and the explanatory texts behind every (i) button."""
from __future__ import annotations

import streamlit as st

from components.cards import _next, chips, md, note
from utils.config import APP_SUBTITLE, NAV_PAGES, PROJECT_TITLE

# --------------------------------------------------------------------------- #
# Explanatory content
# --------------------------------------------------------------------------- #
METRIC_INFO: dict[str, dict[str, str]] = {
    "accuracy": {
        "title": "Accuracy",
        "measures": "The share of all records (normal and attack) that the model classified correctly.",
        "matters": "A quick overall health check. In intrusion detection it can be misleading: attacks are usually rare, so a model that calls everything 'normal' can still look very accurate.",
        "read": "Higher is better. Always read it together with Recall and False Positive Rate, especially on imbalanced data.",
    },
    "precision": {
        "title": "Precision",
        "measures": "Of all flows the model flagged as attacks, the fraction that really were attacks.",
        "matters": "Low precision means many false alarms, which overwhelm analysts and lead to alert fatigue.",
        "read": "Higher is better. Precision of 0.99 means roughly 1 in 100 alerts is a false alarm.",
    },
    "recall": {
        "title": "Recall (Detection rate)",
        "measures": "Of all real attacks, the fraction the model caught.",
        "matters": "Low recall means attacks slip through undetected - usually the costliest error for a security system.",
        "read": "Higher is better. Trade-off: pushing recall up often lowers precision.",
    },
    "f1": {
        "title": "F1 Score",
        "measures": "The harmonic mean of Precision and Recall - one number that balances false alarms and missed attacks.",
        "matters": "Useful for imbalanced data, where accuracy alone hides weak attack detection.",
        "read": "Higher is better (1.0 is perfect). It drops sharply if either precision or recall is low.",
    },
    "roc_auc": {
        "title": "ROC-AUC",
        "measures": "How well the model ranks attacks above normal traffic across all possible decision thresholds.",
        "matters": "Threshold-independent: shows whether a good operating point exists, even before you choose a cut-off.",
        "read": "0.5 is random guessing, 1.0 is perfect separation. Higher is better.",
    },
    "fpr": {
        "title": "False Positive Rate (FPR)",
        "measures": "The fraction of normal traffic wrongly flagged as an attack: FP / (FP + TN).",
        "matters": "Even a tiny FPR becomes thousands of false alerts on busy networks, so operations teams track it closely.",
        "read": "Lower is better. Values near 0 are ideal.",
    },
    "mcc": {
        "title": "Matthews Correlation Coefficient (MCC)",
        "measures": "A correlation between predicted and true labels that uses all four cells of the confusion matrix.",
        "matters": "One of the most reliable single scores for imbalanced classes - it is high only if the model is good on both classes.",
        "read": "-1 to +1: +1 perfect, 0 no better than chance, -1 total disagreement. Higher is better.",
    },
    "train_time": {
        "title": "Training time",
        "measures": "How long it took to fit the model on the training data (seconds).",
        "matters": "Determines how quickly a model can be retrained when traffic changes - important for adaptive systems.",
        "read": "Lower means faster retraining. Compare only on the same hardware and data.",
    },
    "predict_time": {
        "title": "Prediction time",
        "measures": "How long the model takes to predict the evaluation set (seconds).",
        "matters": "Detection must keep up with traffic volume; slow inference creates a backlog.",
        "read": "Lower is better.",
    },
}

CHART_INFO: dict[str, dict[str, str]] = {
    "binary_dist": {
        "title": "Binary class distribution",
        "purpose": "Shows how many records are normal versus malicious.",
        "read": "Strong imbalance means accuracy alone is unreliable and that resampling (e.g. SMOTE) or class weights may be needed. The exact counts come from the loaded data.",
    },
    "attack_dist": {
        "title": "Attack-type distribution",
        "purpose": "Shows how many records belong to each traffic category.",
        "read": "Rare attack classes (a few hundred records) are much harder to learn than frequent ones. Use the log scale to see small classes next to very large ones.",
    },
    "corr": {
        "title": "Correlation heatmap",
        "purpose": "Pearson correlation between the selected numeric features.",
        "read": "Values near +1/-1 mean two features carry nearly the same information (redundancy). Strongly correlated groups can split importance between them in explainability results.",
    },
    "dist": {
        "title": "Numerical feature distribution",
        "purpose": "Histogram of one feature, split by traffic class.",
        "read": "Network features are heavy-tailed, so the log axis and the percentile clip help. Where the normal and attack histograms separate, the feature is informative.",
    },
    "compare": {
        "title": "Normal vs. attack comparison",
        "purpose": "Box plots (with mean marker) and summary statistics of a feature for each class.",
        "read": "Large differences between medians suggest the feature helps distinguish attacks. Overlapping boxes suggest it is weak on its own.",
    },
    "model_bars": {
        "title": "Model comparison",
        "purpose": "Side-by-side comparison of the selected metrics for every model in the report.",
        "read": "Look for models that are strong on Recall and F1, not only Accuracy. Differences in the 3rd decimal are often within noise.",
    },
    "radar": {
        "title": "Performance profile",
        "purpose": "A radar view of all selected metrics per model.",
        "read": "A larger, rounder area means a better balanced model. Dents show the metric where a model is weak.",
    },
    "cm": {
        "title": "Confusion matrix",
        "purpose": "Counts of actual versus predicted classes.",
        "read": "Diagonal cells are correct decisions. Off-diagonal cells are mistakes: 'Actual attack, predicted normal' is a missed attack, 'Actual normal, predicted attack' is a false alarm.",
    },
    "bin_vs_multi": {
        "title": "Binary vs. multiclass",
        "purpose": "Compares each model's metrics between the two tasks.",
        "read": "Multiclass is harder - scores are normally lower and rare classes pull weighted averages down.",
    },
    "shap_global": {
        "title": "SHAP global importance",
        "purpose": "Mean absolute SHAP value per feature over the reference records.",
        "read": "Longer bar = the feature changes the model's output more on average. It says how much, not in which direction.",
    },
    "shap_summary": {
        "title": "SHAP summary (beeswarm)",
        "purpose": "Every dot is one record. Position = effect on the prediction, colour = the feature's value.",
        "read": "Red dots on the right mean high feature values push the prediction toward the explained class; blue dots on the right mean low values do.",
    },
    "drift_error": {
        "title": "Prediction error over time",
        "purpose": "Error rate of each model on every sequential batch.",
        "read": "A rising static-model curve signals that its learned patterns no longer match incoming traffic. The adaptive model should recover after retraining.",
    },
    "drift_timeline": {
        "title": "Drift detection timeline",
        "purpose": "Per-batch status: stable, ADWIN drift alert, or alert followed by retraining.",
        "read": "Clusters of alerts indicate sustained change; isolated alerts can be noise (try a smaller delta for more sensitivity or larger for fewer alerts).",
    },
}

XAI_INFO = {
    "SHAP": ("SHAP (SHapley Additive exPlanations) borrows an idea from game theory: the prediction is a 'payout' "
             "shared fairly among the features. Each feature gets a value showing how much it pushed this prediction "
             "up or down compared with an average prediction. Global importance is the average size of those pushes "
             "over many records; local explanations show one record at a time."),
    "LIME": ("LIME (Local Interpretable Model-agnostic Explanations) explains one record by creating many slightly "
             "perturbed copies of it, asking the real model about each, and fitting a small, simple model around that "
             "neighbourhood. The weights of that simple model are the explanation. It works with any model but is random-sampling "
             "based, so results can vary slightly between runs."),
    "When they differ": ("SHAP and LIME can disagree: SHAP uses the model's internals (for tree/linear models) and has "
                         "consistency guarantees, while LIME is a local approximation. Strong agreement between the two "
                         "increases confidence in an explanation; disagreement is a signal to look closer."),
}


# --------------------------------------------------------------------------- #
# Dialog helpers
# --------------------------------------------------------------------------- #
def _render_info(sections: list[tuple[str, str]]) -> None:
    for head, body in sections:
        st.markdown(f"**{head}**")
        st.write(body)


def open_info(title: str, sections: list[tuple[str, str]], width: str = "medium") -> None:
    """Open a dialog with titled paragraphs."""
    st.dialog(title, width=width)(lambda: _render_info(sections))()


def info_button(kind: str, key: str, label: str = "") -> None:
    """Small (i) button that opens the explanation dialog for a chart or metric."""
    uid = _next("i")  # unique per call so the same chart/metric can appear on several tabs
    with st.container(key=f"info_{kind}_{key}_{uid}"):
        if st.button(label or "Info", key=f"btn_{kind}_{key}_{uid}", icon=":material/info:", help="What does this show?"):
            if kind == "metric":
                d = METRIC_INFO[key]
                open_info(d["title"], [("What it measures", d["measures"]), ("Why it matters in intrusion detection", d["matters"]),
                                       ("How to interpret it", d["read"])])
            elif kind == "chart":
                d = CHART_INFO[key]
                open_info(d["title"], [("Purpose", d["purpose"]), ("How to interpret it", d["read"])])


def xai_info_dialog() -> None:
    open_info("SHAP and LIME in simple language", list(XAI_INFO.items()), width="large")


def notice(kind: str, title: str, message: str, details: list[str] | None = None) -> None:
    """Modal notice. kind: ok | warn | bad | info."""
    def body() -> None:
        note(kind, f"<b>{title}</b><br>{message}")
        if details:
            for d in details[:12]:
                st.markdown(f"- {d}")
            if len(details) > 12:
                st.caption(f"... and {len(details) - 12} more.")
    st.dialog(title, width="medium")(body)()


# --------------------------------------------------------------------------- #
# Home-page dialogs
# --------------------------------------------------------------------------- #
def start_detection_dialog() -> None:
    def body() -> None:
        st.markdown("Follow these steps to classify network traffic with the saved models:")
        steps = [
            ("1. Open the Predict page", "Choose **Manual Prediction** for one flow or **Batch Prediction** for a CSV file."),
            ("2. Provide network-flow features",
             "Enter values for each flow feature listed in `models/feature_columns.json` (e.g. flow duration, packet lengths, inter-arrival times, "
             "TCP flag counts). For batch mode upload a CSV whose headers match those feature names - extra columns such as `Label` are ignored."),
            ("3. Validation", "Missing, non-numeric, infinite or out-of-range values are rejected or excluded before anything reaches a model."),
            ("4. Read the result", "You get the Normal/Attack verdict, the predicted attack category, confidence and the most likely alternatives."),
            ("5. Ask 'why?'", "Open the **Explainability** page to see SHAP and LIME explanations for the same records."),
        ]
        for h, t in steps:
            st.markdown(f"**{h}**")
            st.write(t)
        if st.button("Go to Predict", type="primary", icon=":material/arrow_forward:", key="dlg_go_predict"):
            st.switch_page(NAV_PAGES.get("predict") or "pages/2_Predict.py")
    st.dialog("How to start detection", width="large", icon=":material/shield:")(body)()


def explore_project_dialog() -> None:
    def body() -> None:
        st.markdown(f"#### {PROJECT_TITLE}")
        t1, t2, t3, t4 = st.tabs(["Abstract", "Objectives", "Methodology", "Research gap"])
        with t1:
            st.write(
                "Network intrusion detection systems based on machine learning are usually trained once on a fixed snapshot of traffic "
                "and then deployed unchanged, while real traffic and attacker behaviour keep evolving. This project builds an intrusion "
                "detection framework that (i) separates normal from malicious flows, (ii) assigns malicious flows to an attack category, "
                "(iii) monitors the live error stream with the ADWIN change detector to recognise concept drift, (iv) adapts the model "
                "when drift is detected, and (v) explains individual and global model decisions with SHAP and LIME.")
        with t2:
            for o in ["Detect normal vs. malicious network flows (binary classification).",
                      "Identify the attack category (multiclass classification).",
                      "Detect changes in traffic behaviour (concept drift) with ADWIN.",
                      "Adapt the model after drift through retraining / online learning and compare it with static models.",
                      "Explain predictions with SHAP and LIME so analysts can trust and verify alerts."]:
                st.markdown(f"- {o}")
        with t3:
            for m in ["**Data:** CIC-IDS2017 and UNSW-NB15 flow records; cleaning of missing/infinite values; class-imbalance handling (imbalanced-learn).",
                      "**Models:** Logistic Regression, Decision Tree, Random Forest, XGBoost, evaluated with Accuracy, Precision, Recall, F1, ROC-AUC, FPR and MCC.",
                      "**Drift:** prequential evaluation on sequential batches; River ADWIN monitors the error stream.",
                      "**Adaptation:** drift-triggered retraining on the most recent labelled window, and an incremental River learner.",
                      "**Explainability:** SHAP (global + local) and LIME (local)."]:
                st.markdown(f"- {m}")
        with t4:
            st.write(
                "Static IDS models silently degrade when traffic patterns shift, and many high-accuracy models remain black boxes, "
                "which limits analyst trust. Work that combines **drift awareness, adaptation and explainability in one workflow** "
                "is comparatively scarce. This framework puts the three together and keeps the drift experiment honest: it runs on "
                "recorded flows in a simulated stream, not on live traffic.")
    st.dialog("Explore the project", width="large", icon=":material/travel_explore:")(body)()
''',
    "components.data_source": r'''
"""Shared UI for choosing the records used by Explainability / Drift / live evaluation."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components.cards import note
from utils.data_loader import DATASETS, find_files, load_disk_dataset, signature
from utils.sources import Source, build_source, read_uploaded_csv


def disk_datasets() -> dict[str, list]:
    return {n: f for n in DATASETS if (f := find_files(n))}


@st.cache_data(show_spinner=False, max_entries=2)
def _parse_upload(raw: bytes) -> pd.DataFrame:
    return read_uploaded_csv(raw)


def pick_source(key: str, features: list[str], need_labels: bool = False, allow_batch: bool = True,
                chronological_option: bool = False, max_rows: int | None = None) -> Source | None:
    """Let the user choose real records. Returns a validated ``Source`` or None (with guidance shown)."""
    options: list[str] = []
    disk = disk_datasets()
    options += [f"Dataset on disk · {n}" for n in disk]
    batch = st.session_state.get("last_batch")
    if allow_batch and batch is not None and not need_labels:
        options.append("Last batch from the Predict page")
    options.append("Upload a CSV file")
    choice = st.radio("Record source", options, key=f"{key}_src", horizontal=True)

    chrono = False
    if chronological_option:
        chrono = st.checkbox("Order records chronologically by weekday in file names (Monday → Friday)", value=True,
                             key=f"{key}_chrono", help="CIC-IDS2017 attacks differ per day, so this order contains natural traffic change.")

    if choice.startswith("Dataset on disk"):
        name = choice.split("· ")[1]
        files = disk[name]
        with st.spinner("Reading dataset files…"):
            bundle = load_disk_dataset(name, signature(files))
        if isinstance(bundle, str):
            note("bad", f"The dataset could not be used: {bundle}")
            return None
        df = bundle.sample
        src, prep = build_source(df, features, f"{name} (sample of {len(df):,} rows from {len(files)} file(s))", "disk",
                                 labels=df["_attack_type"], files=df["_source_file"], chronological=chrono, max_rows=max_rows)
        if src is None:
            note("bad", "; ".join(prep.errors) + " The dataset columns do not match <code>feature_columns.json</code>.")
            return None
        return src

    if choice.startswith("Last batch"):
        t = batch
        X = t["X"].reset_index(drop=True)
        return Source(X=X, labels=None, name="Last batch from the Predict page", origin="batch", notes=[])

    up = st.file_uploader("CSV with the model's feature columns" + (" and a label column (`Label` or `attack_cat`)" if need_labels else ""),
                          type=["csv"], key=f"{key}_upload")
    if up is None:
        note("info", "Upload a CSV to continue." + (" It must contain a ground-truth label column." if need_labels else ""))
        return None
    try:
        df = _parse_upload(up.getvalue())
    except ValueError as exc:
        note("bad", str(exc))
        return None
    src, prep = build_source(df, features, up.name, "upload", chronological=False, max_rows=max_rows)
    if src is None:
        note("bad", " ".join(prep.errors))
        return None
    if need_labels and src.labels is None:
        note("bad", "No label column found. Add a <code>Label</code> (CIC-IDS2017) or <code>attack_cat</code>/<code>label</code> (UNSW-NB15) column.")
        return None
    return src
''',
    "components.sidebar": r'''
"""Sidebar: brand, artifact readiness and session summary (shown on every page)."""
from __future__ import annotations

import streamlit as st

from components.cards import md
from components.styles import icon
from utils.model_loader import artifact_status


def render_sidebar() -> None:
    with st.sidebar:
        md(f"""<div class="brand"><div class="logo">{icon('shield')}</div>
        <div><b>NIDS Research</b><span>Drift-aware · Explainable</span></div></div>""")

        statuses = artifact_status()
        n_ok = sum(s["exists"] for s in statuses)
        md('<div class="side-label">Artifact readiness</div>')
        rows = "".join(
            f'<div class="side-row"><span>{s["label"]}</span>'
            f'<span style="color:{"#2ee6a6" if s["exists"] else "#ff4d6d"}">{icon("check" if s["exists"] else "warning")}</span></div>'
            for s in statuses)
        md(rows)
        st.caption(f"{n_ok}/{len(statuses)} expected files found")

        md('<div class="side-label">Session</div>')
        last = st.session_state.get("last_batch")
        if last is not None:
            n = len(last["table"])
            n_att = int((last["table"]["final_verdict"] != "Normal").sum())
            md(f'<div class="side-row"><span>Last batch</span><span>{n:,} flows · {n_att:,} attacks</span></div>')
        else:
            md('<div class="side-row"><span>Last batch</span><span>none yet</span></div>')
        if st.session_state.get("drift_run") is not None:
            r = st.session_state["drift_run"]
            md(f'<div class="side-row"><span>Drift run</span><span>{r.summary["n_drift_events"]} event(s)</span></div>')
        md('<div class="side-label">Scope</div>')
        st.caption("Offline analysis of recorded flow records. No live packet capture, no firewall control, no blocking.")
''',
}

_PAGE_SOURCES = {
    "home": r'''
"""Home - landing dashboard."""
import streamlit as st

from components import cards, modals
from components.cards import architecture_flow, chips, feature_card, md, metric_row, page_header, section
from components.styles import icon
from utils.config import APP_SUBTITLE, APP_TITLE
from utils.model_loader import artifact_status, get_bundle

md(f"""<div class="hero"><div class="radar"><span class="blip" style="left:68%;top:30%"></span><span class="blip" style="left:30%;top:64%;animation-delay:.8s"></span><span class="blip" style="left:74%;top:72%;animation-delay:1.5s"></span></div>
<div class="shield-float">{icon('shield')}</div>
<div class="eyebrow">{icon('radar')} Research platform · Final-year project</div>
<h1 class="grad-text">{APP_TITLE}</h1>
<p class="sub">{APP_SUBTITLE}</p>
<p class="intro">Classify network flows as normal or malicious, name the attack category, watch for concept drift with ADWIN,
adapt the model when traffic changes, and see exactly <i>why</i> every decision was made with SHAP and LIME.</p></div>""")

st.write("")
c1, c2, _ = st.columns([1, 1, 3])
if c1.button("Start Detection", type="primary", icon=":material/rocket_launch:", key="home_start", width="stretch"):
    modals.start_detection_dialog()
if c2.button("Explore Project", icon=":material/travel_explore:", key="home_explore", width="stretch"):
    modals.explore_project_dialog()

st.write("")
a, b, c = st.columns(3)
a.markdown(feature_card("shield", "Binary Detection", "Identify whether network traffic is normal or malicious."), unsafe_allow_html=True)
b.markdown(feature_card("bug", "Attack Classification", "Classify malicious network traffic into its corresponding attack category."), unsafe_allow_html=True)
c.markdown(feature_card("eye", "Model Explainability", "Understand why the model generated a particular prediction using SHAP and LIME."), unsafe_allow_html=True)

section("System status", "activity")
bundle = get_bundle()
stat = artifact_status()
metric_row([
    {"label": "Saved models", "value": f"{int(bundle.binary_model is not None) + int(bundle.multiclass_model is not None)} / 2",
     "icon": "brain", "sub": "binary + multiclass loaded", "accent": "#22e3ff"},
    {"label": "Input features", "value": f"{len(bundle.features):,}" if bundle.features else "-", "icon": "network",
     "sub": "from feature_columns.json", "accent": "#4f8cff"},
    {"label": "Attack classes", "value": f"{len(bundle.classes):,}" if bundle.classes else "-", "icon": "bug",
     "sub": "from multiclass_classes.json", "accent": "#8b6bff"},
    {"label": "Artifacts found", "value": f"{sum(s['exists'] for s in stat)} / {len(stat)}", "icon": "database",
     "sub": "models + reports", "accent": "#2ee6a6"},
])
if bundle.problems:
    cards.note("warn", "<b>Some artifacts are unavailable.</b> " + " ".join(bundle.problems[:3]) +
               " The related pages explain what is needed; the rest of the app keeps working.")

section("Project objectives", "target")
o1, o2 = st.columns(2)
with o1, cards.glass():
    st.markdown("**What the framework does**")
    for t in ["Detects normal vs. malicious traffic (binary classification).", "Identifies the specific attack category (multiclass).",
              "Explains predictions with global and local SHAP and LIME.", "Predicts single flows or whole CSV files."]:
        st.markdown(f"- {t}")
with o2, cards.glass():
    st.markdown("**What makes it drift-aware**")
    for t in ["Monitors the model's error stream with River ADWIN.", "Raises a drift alert when error behaviour changes significantly.",
              "Adapts by retraining on recent data or by online learning.", "Compares the adaptive model against static models."]:
        st.markdown(f"- {t}")

section("Key capabilities & technology", "layers")
with cards.glass():
    st.markdown("**Capabilities**")
    chips(["Binary detection", "Multiclass attack typing", "Batch CSV prediction", "SHAP global/local", "LIME local", "ADWIN drift detection",
           "Drift-triggered retraining", "Interactive dataset analysis"], "c")
    st.write("")
    st.markdown("**Technology**")
    chips(["Python", "Streamlit", "scikit-learn", "XGBoost", "River", "ADWIN", "SHAP", "LIME", "Pandas", "NumPy", "Imbalanced-learn", "Plotly", "Joblib"])

section("System architecture", "network")
with cards.glass():
    architecture_flow([
        ("database", "Flow data", "CIC-IDS2017 · UNSW-NB15"), ("gear", "Preprocessing", "validate · clean · order"),
        ("brain", "Models", "LR · DT · RF · XGBoost"), ("bug", "Detection", "binary + attack type"),
        ("activity", "Drift monitor", "River ADWIN"), ("robot", "Adaptation", "retrain / online"),
        ("eye", "Explain", "SHAP · LIME"),
    ])
    st.caption("Drift monitoring and adaptation run as a simulated stream over recorded flows - this is not live packet monitoring.")
''',
    "overview": r'''
"""Overview - a dashboard assembled only from real artifacts.

Every number comes from (a) the evaluation report CSVs, (b) the dataset files on disk and (c) the saved models
scoring a random sample of those recorded flows right now. Nothing is live traffic and nothing is simulated.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import streamlit as st

from components import cards, charts
from components.cards import detections_table, glass, kpi_row, md, note, page_header, provenance_tag, section
from utils.data_loader import DATASETS, find_files, load_disk_dataset, signature
from utils.metrics import METRIC_LABELS, load_report
from utils.model_loader import _signature as model_signature
from utils.model_loader import artifact_status, get_bundle
from utils.prediction import is_benign, predict_all, predict_binary
from utils.preprocessing import prepare_batch

DATASET = "CIC-IDS2017"
SCORE_N = 3000

page_header("activity", "Dashboard", "Overview",
            "Models, data and detections at a glance - computed from your report files, your recorded flows and your saved models.")
md(f'<div style="margin:-.4rem 0 1.1rem">{provenance_tag("real")} '
   f'<span style="color:var(--muted);font-size:.88rem">&nbsp;Recorded flows scored by the saved models just now - this is not live network monitoring.</span></div>')

bundle = get_bundle()
rep = load_report("binary")


def Scored(X, labels, sample_idx, table, chunk_attack_rate, chunk_confidence, chunk_latency_ms) -> SimpleNamespace:
    """Result of scoring the sample: X/labels = all prepared records, table = predictions for ``sample_idx``."""
    return SimpleNamespace(X=X, labels=labels, sample_idx=sample_idx, table=table, chunk_attack_rate=chunk_attack_rate,
                           chunk_confidence=chunk_confidence, chunk_latency_ms=chunk_latency_ms)


@st.cache_data(show_spinner=False, max_entries=1)
def _prepared(_sig: tuple, dataset_sig: tuple, features: tuple):
    """Dataset sample -> validated feature matrix + true categories (cached per files/model signature)."""
    res = load_disk_dataset(DATASET, dataset_sig)
    if isinstance(res, str):
        return res
    prep = prepare_batch(res.sample, list(features))
    if not prep.ok:
        return "; ".join(prep.errors) or "The dataset has no rows compatible with the model features."
    idx = prep.valid_rows
    X = prep.X[list(features)].reset_index(drop=True)
    lab = res.sample["_attack_type"].iloc[idx].reset_index(drop=True)
    return res, X, lab


def _score(X: pd.DataFrame, lab: pd.Series) -> SimpleNamespace:
    rng = np.random.default_rng(7)
    pos = np.sort(rng.choice(len(X), min(SCORE_N, len(X)), replace=False))
    Xs = X.iloc[pos]
    result = predict_all(bundle, Xs)
    rates, confs, lats = [], [], []
    if bundle.binary_model is not None:
        Xb = Xs[bundle.order_for("binary")]
        for chunk in np.array_split(np.arange(len(Xb)), 20):
            if len(chunk) == 0:
                continue
            t0 = time.perf_counter()
            b = predict_binary(bundle.binary_model, Xb.iloc[chunk])
            lats.append((time.perf_counter() - t0) * 1000 / len(chunk))
            rates.append(float(b["is_attack"].mean()))
            confs.append(float(b["binary_confidence"].mean()))
    return Scored(X, lab, pos, result.table, rates, confs, lats)


# ---- gather real inputs ---------------------------------------------------------------------
files = find_files(DATASET)
data = None
data_error = None
if files and bundle.can_predict:
    try:
        with st.spinner(f"Reading {len(files)} {DATASET} file(s) and scoring a sample - the first run on large files can take a minute…"):
            data = _prepared(model_signature(), signature(files), tuple(bundle.features))
    except Exception as exc:  # noqa: BLE001
        data_error = f"{type(exc).__name__}: {str(exc)[:140]}"
if isinstance(data, str):
    data_error, data = data, None

scored = None
if data is not None:
    ds, X_all, lab_all = data
    key = ("ov", model_signature(), signature(files), len(X_all))
    if st.session_state.get("ov_key") != key:
        try:
            with st.spinner("Scoring a random sample of recorded flows with the saved models…"):
                st.session_state["ov_scored"] = _score(X_all, lab_all)
            st.session_state["ov_key"] = key
        except Exception as exc:  # noqa: BLE001
            data_error = f"The saved models could not score the sample ({type(exc).__name__}: {str(exc)[:120]})."
    scored = st.session_state.get("ov_scored") if st.session_state.get("ov_key") == key else None

# ---- KPI row ----------------------------------------------------------------------------------
kpis = []
if rep.ok and "accuracy" in rep.df.columns:
    d = rep.df.dropna(subset=["accuracy"])
    best = d.loc[d["accuracy"].idxmax()]
    kpis.append(dict(label="Best model accuracy", value=f"{best['accuracy']:.2%}", icon="brain", accent="#ff4fa3", series=d["accuracy"].tolist(),
                     detail=f"<b>{best['model']}</b><br>best of {len(d)} models · binary report"))
else:
    kpis.append(dict(label="Best model accuracy", value="—", icon="brain", accent="#ff4fa3", detail="needs reports/cicids_binary_results.csv"))
if data is not None:
    kpis.append(dict(label=f"{DATASET} flows", value=f"{ds.total_records:,}", icon="network", accent="#22e3ff",
                     series=ds.attack_counts.sort_values().tolist(), detail=f"<b>{len(ds.attack_counts)}</b> traffic categories"))
else:
    kpis.append(dict(label=f"{DATASET} flows", value="—", icon="network", accent="#22e3ff", detail="needs dataset files in data/"))
if scored is not None and "is_attack" in scored.table:
    n_att = int(scored.table["is_attack"].sum())
    kpis.append(dict(label="Attacks flagged", value=f"{n_att:,}", icon="shield", accent="#ff4d6d", series=scored.chunk_attack_rate,
                     detail=f"<b>{n_att / len(scored.table):.1%}</b> of {len(scored.table):,} scored flows"))
    kpis.append(dict(label="Mean model confidence", value=f"{scored.table['binary_confidence'].mean():.1%}", icon="target", accent="#8b6bff",
                     series=scored.chunk_confidence, detail="binary model · sample sequence"))
    if scored.chunk_latency_ms:
        kpis.append(dict(label="Prediction latency", value=f"{np.mean(scored.chunk_latency_ms):.2f} ms", icon="bolt", accent="#ffb547",
                         series=scored.chunk_latency_ms, detail="per flow · measured now"))
else:
    for lbl, ic, ac in (("Attacks flagged", "shield", "#ff4d6d"), ("Mean model confidence", "target", "#8b6bff"), ("Prediction latency", "bolt", "#ffb547")):
        kpis.append(dict(label=lbl, value="—", icon=ic, accent=ac, detail="needs models + dataset"))
kpi_row(kpis)

if data_error:
    note("warn", f"<b>Some panels are unavailable.</b> {data_error}")
if not bundle.can_predict:
    cards.missing_artifacts_panel(artifact_status(), {"binary_model", "multiclass_model", "features", "classes"}, "Scoring panels need the saved models")
elif not files:
    note("warn", f"No {DATASET} files found in <code>{DATASETS[DATASET]['folder']}</code> - data panels are hidden. {DATASETS[DATASET]['files_hint']}")

# ---- row 2: signatures + distribution + confidence ------------------------------------------------
st.write("")
left, right = st.columns([5, 3])
with left:
    with glass():
        md('<div class="panel-title">Attack signatures · what each traffic category looks like</div>')
        if data is None:
            st.caption("Needs the dataset files and the saved models.")
        else:
            try:
                Xs, labs = data[1], data[2]
                counts = labs.value_counts()
                classes = [c for c in counts.index if counts[c] >= 20][:10]
                mask = labs.isin(classes).to_numpy()
                Xc = Xs[mask]
                lo, hi = Xc.quantile(0.01), Xc.quantile(0.99)
                Xc = Xc.clip(lo, hi, axis=1)  # robust to the heavy tails of flow features
                std = Xc.std().replace(0, np.nan)
                Z = ((Xc - Xc.mean()) / std).fillna(0)
                means = Z.groupby(labs[mask].to_numpy()).mean().loc[classes]
                feats = means.var().sort_values(ascending=False).head(12).index.tolist()
                charts.show(charts.signature_heatmap(means[feats].to_numpy(), classes, feats), key="ov_sig")
                st.caption("Average standardised value (σ above/below the overall mean) of the 12 features that differ most between categories - from the real, labelled records.")
            except Exception as exc:  # noqa: BLE001
                st.caption(f"Signature view unavailable ({type(exc).__name__}).")
with right:
    with glass():
        md('<div class="panel-title">Top 5 attack types</div>')
        if data is None:
            st.caption("Needs the dataset files.")
        else:
            atk = ds.attack_counts[[not is_benign(k) for k in ds.attack_counts.index]].head(5)
            if atk.empty:
                st.caption("No attack records in the dataset.")
            else:
                charts.show(charts.donut(atk.index.tolist(), atk.tolist(), "", None, 300), key="ov_donut")
    with glass():
        md('<div class="panel-title">Model confidence · P(attack)</div>')
        if scored is None or "attack_probability" not in scored.table or scored.table["attack_probability"].isna().all():
            st.caption("Needs a model with probability output and the dataset.")
        else:
            charts.show(charts.confidence_histogram(scored.table["attack_probability"].to_numpy()), key="ov_conf")

# ---- row 3: detections + performance --------------------------------------------------------------
left, right = st.columns([5, 4])
with left:
    with glass():
        md('<div class="panel-title">Highest-confidence detections · scored sample</div>')
        if scored is None or "is_attack" not in scored.table or not scored.table["is_attack"].any():
            st.caption("No attacks were flagged in the scored sample." if scored is not None else "Needs the saved models and the dataset files.")
        else:
            t = scored.table
            rows = []
            Xs = scored.X.iloc[scored.sample_idx]
            mu, sd = scored.X.mean(), scored.X.std().replace(0, np.nan)
            top = t[t["is_attack"]].sort_values("attack_probability", ascending=False, kind="stable").head(7)
            for ridx, r in top.iterrows():
                pos = t.index.get_loc(ridx)
                z = ((Xs.iloc[pos] - mu) / sd).fillna(0)
                f = z.abs().idxmax()
                rows.append(dict(id=int(scored.sample_idx[pos]), category=r["final_verdict"], confidence=float(r["attack_probability"]),
                                 note=f"{f} ({z[f]:+.1f}σ)"))
            detections_table(rows)
            st.caption("“Most unusual feature” = the feature furthest from its dataset average (not a SHAP explanation - see the Explainability page). "
                       "Bands are confidence thresholds (≥99 % Critical, ≥90 % High, else Medium), not a risk score.")
with right:
    with glass():
        md('<div class="panel-title">Model performance · binary report</div>')
        if not rep.ok:
            st.caption(rep.error or "reports/cicids_binary_results.csv is required.")
        else:
            mets = [m for m in ("accuracy", "precision", "recall", "f1", "roc_auc") if m in rep.df.columns]
            charts.show(charts.model_lines(rep.df, mets, METRIC_LABELS), key="ov_perf")
            st.caption(f"From {rep.path}. Lines connect models only to make them easier to compare - they are not a time series.")

with st.expander("Where does every number on this page come from?", icon=":material/info:"):
    st.markdown(
        "- **Best model accuracy / performance lines** - read from `reports/cicids_binary_results.csv`.\n"
        f"- **Flows, top attack types** - exact counts from the {DATASET} files in `data/` (full scan).\n"
        f"- **Attacks flagged, confidence, latency, detections** - the saved binary/multiclass models scoring a random sample of {SCORE_N:,} recorded flows "
        "(latency is wall-clock time measured on this machine just now).\n"
        "- **Attack signatures** - standardised feature averages per true label from the dataset sample.\n"
        "- Sparklines show the real sequence of the sample or the models of the report. Nothing here is live traffic, simulated or hard-coded.")
''',
    "predict": r'''
"""Predict - manual and batch prediction with the saved models."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from components import cards, charts, modals
from components.cards import glass, metric_row, note, page_header, result_card, section
from components.data_source import disk_datasets
from utils.data_loader import load_disk_dataset, signature
from utils.model_loader import artifact_status, get_bundle
from utils.prediction import PredictionResult, is_benign, predict_all
from utils.preprocessing import prepare_batch, prepare_manual
from utils.sources import build_source, read_uploaded_csv

page_header("radar", "Detection", "Predict", "Classify network flows with the saved binary and multiclass models - one flow at a time or a whole CSV.")

bundle = get_bundle()
if not bundle.can_predict:
    cards.missing_artifacts_panel(artifact_status(), {"binary_model", "multiclass_model", "features"}, "Cannot predict yet - model artifacts are missing")
    for p in bundle.problems:
        note("bad", p)
    st.stop()
for p in bundle.problems:
    note("warn", p)
for w in bundle.warnings:
    note("info", w)

features = bundle.features


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _pct(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.1%}"


def _group_features(names: list[str]) -> dict[str, list[int]]:
    """Group feature indices by name so a 78-field form stays navigable."""
    rules = [
        ("TCP flags", ("flag", "psh", "urg", "fin", "syn", "rst", "ack", "ece", "cwe")),
        ("Forward direction", ("fwd",)),
        ("Backward direction", ("bwd",)),
        ("Flow timing & rates", ("duration", "iat", "idle", "active", "/s", "rate", "time")),
        ("Packet lengths & sizes", ("length", "size", "segment", "packet", "bytes")),
    ]
    groups: dict[str, list[int]] = {}
    for i, n in enumerate(names):
        low = n.lower()
        g = next((g for g, keys in rules if any(k in low for k in keys)), "Other features")
        groups.setdefault(g, []).append(i)
    if len(groups) == 1:  # generic names: chunk instead
        groups = {f"Features {a + 1}-{min(a + 15, len(names))}": list(range(a, min(a + 15, len(names)))) for a in range(0, len(names), 15)}
    return groups


def _fill_from_record(row: pd.Series, label: str) -> None:
    for i, f in enumerate(features):
        st.session_state[f"man_{i}"] = f"{float(row[f]):.10g}"
    st.session_state["man_loaded_label"] = label


def _cb_load_record(dataset: str) -> None:
    files = disk_datasets()[dataset]
    ds = load_disk_dataset(dataset, signature(files))
    if isinstance(ds, str):
        st.session_state["man_fill_error"] = ds
        return
    src, prep = build_source(ds.sample, features, dataset, "disk", labels=ds.sample["_attack_type"])
    if src is None:
        st.session_state["man_fill_error"] = "; ".join(prep.errors)
        return
    idx = int(st.session_state.get("man_rec_idx", 0)) % len(src.X)
    _fill_from_record(src.X.iloc[idx], f"{dataset} sample record #{idx} - recorded label: {src.labels.iloc[idx]}")
    st.session_state.pop("man_fill_error", None)


def _cb_paste() -> None:
    txt = st.session_state.get("man_paste", "")
    parts = [p for p in pd.Series(txt.replace("\n", ",").replace("\t", ",").replace(";", ",").split(",")).str.strip() if p != ""]
    if len(parts) != len(features):
        st.session_state["man_fill_error"] = f"Expected {len(features)} comma-separated values in feature order, got {len(parts)}."
        return
    for i, v in enumerate(parts):
        st.session_state[f"man_{i}"] = v
    st.session_state["man_loaded_label"] = "pasted values"
    st.session_state.pop("man_fill_error", None)


def _cb_clear() -> None:
    for i in range(len(features)):
        st.session_state[f"man_{i}"] = ""
    st.session_state.pop("man_loaded_label", None)


def render_single(res: PredictionResult, pos: int = 0, show_probs: bool = True) -> None:
    """Large result card + class probabilities for one record."""
    row = res.table.iloc[pos]
    verdict = row["final_verdict"]
    attack = verdict != "Normal"
    meta_parts = []
    if "binary_confidence" in res.table and pd.notna(row.get("binary_confidence")):
        meta_parts.append(f"Binary model confidence: <b>{_pct(row['binary_confidence'])}</b>")
    if attack and "multiclass_confidence" in res.table and pd.notna(row.get("multiclass_confidence")):
        meta_parts.append(f"Category probability: <b>{_pct(row['multiclass_confidence'])}</b>")
    if not meta_parts:
        meta_parts.append("This model does not provide probabilities, so no confidence is shown.")
    result_card(attack, verdict, " · ".join(meta_parts))

    if not show_probs:
        return
    cols = st.columns(2)
    with cols[0], glass():
        st.markdown("**Binary decision**")
        if "attack_probability" in res.table and pd.notna(row.get("attack_probability")):
            ap = float(row["attack_probability"])
            charts.show(charts.probabilities(["Normal", "Attack"], np.array([1 - ap, ap]), "Binary probabilities", 2,
                                             highlight="Attack" if attack else "Normal"), key=f"pb_{pos}")
        else:
            st.caption("Binary probabilities are not available for this model.")
    with cols[1], glass():
        st.markdown("**Multiclass decision**")
        if res.multi_proba is not None:
            probs = res.multi_proba[pos]
            charts.show(charts.probabilities(res.multi_names, probs, "Top class probabilities", 6,
                                             highlight=row["multiclass_prediction"]), key=f"pm_{pos}")
            alt = row.get("alternative_categories", "")
            if alt:
                st.caption(f"Top alternatives: {alt}")
        elif "multiclass_prediction" in res.table:
            st.write(f"Predicted category: **{row['multiclass_prediction']}**")
            st.caption("Class probabilities are not available for this model.")
        else:
            st.caption("The multiclass model is not available.")
    if "models_agree" in res.table and row.get("models_agree") is False:
        note("warn", "The binary and multiclass models disagree on this record. The binary model decides normal vs. attack; "
                     "treat the category as uncertain.")


tab_manual, tab_batch = st.tabs([":material/edit_note:  Manual prediction", ":material/upload_file:  Batch prediction (CSV)"])

# --------------------------------------------------------------------------- #
# A. Manual
# --------------------------------------------------------------------------- #
with tab_manual:
    section("Network-flow features", "sliders")
    note("info", f"The models expect <b>{len(features)}</b> numeric features in the order defined by <code>models/feature_columns.json</code>. "
                 "Every value must be a finite number; nothing is auto-filled.")

    with st.expander("Quick fill with real data (optional)", icon=":material/bolt:"):
        disk = disk_datasets()
        f1, f2 = st.columns(2)
        with f1:
            st.markdown("**From a dataset record**")
            if disk:
                ds_name = st.selectbox("Dataset", list(disk), key="man_ds")
                st.number_input("Record index (in the loaded sample)", min_value=0, value=0, step=1, key="man_rec_idx")
                st.button("Load record", key="man_load", on_click=_cb_load_record, args=(ds_name,))
            else:
                st.caption("No dataset files found in `data/`. Add CIC-IDS2017 CSVs to enable this.")
        with f2:
            st.markdown("**Paste one row**")
            st.text_area(f"{len(features)} comma-separated values in feature order", key="man_paste", height=90)
            st.button("Apply pasted values", key="man_apply_paste", on_click=_cb_paste)
        if st.session_state.get("man_fill_error"):
            note("bad", st.session_state["man_fill_error"])
        if st.session_state.get("man_loaded_label"):
            note("ok", f"Loaded: {st.session_state['man_loaded_label']}")
        st.button("Clear all fields", key="man_clear", on_click=_cb_clear)

    groups = _group_features(features)
    with st.form("manual_form", border=False):
        for gname, idxs in groups.items():
            with st.expander(f"{gname}  ·  {len(idxs)} features", expanded=gname == next(iter(groups))):
                cols = st.columns(3)
                for k, i in enumerate(idxs):
                    cols[k % 3].text_input(features[i], key=f"man_{i}", placeholder="enter a number")
        submitted = st.form_submit_button("Predict this flow", type="primary", icon=":material/radar:")

    if submitted:
        raw = {f: st.session_state.get(f"man_{i}", "") for i, f in enumerate(features)}
        X, errors = prepare_manual(raw, features)
        if errors:
            note("bad", f"<b>{len(errors)} input problem(s) found.</b> Nothing was sent to the model.")
            modals.notice("bad", "Invalid input", "Please correct the following fields and try again.", errors)
        else:
            try:
                with st.spinner("Running the saved models…"):
                    res = predict_all(bundle, X)
                st.session_state["last_manual"] = {"X": X, "result": res}
            except Exception as exc:  # noqa: BLE001
                note("bad", f"Prediction failed ({type(exc).__name__}). The input is incompatible with the saved model.")

    lm = st.session_state.get("last_manual")
    if lm is not None:
        section("Result", "check")
        render_single(lm["result"], 0)
        for n in lm["result"].notes:
            note("info", n)

# --------------------------------------------------------------------------- #
# B. Batch
# --------------------------------------------------------------------------- #
with tab_batch:
    section("Upload network-flow records", "file")
    note("info", "Upload a CSV with a header row containing all required feature columns (extra columns such as <code>Label</code> are kept for reference but "
                 "not used). Rows with missing, non-numeric, infinite or out-of-range values are excluded and listed.")
    with st.expander("Required columns", icon=":material/list:"):
        st.code(", ".join(features), language=None)
    up = st.file_uploader("CSV file", type=["csv"], key="batch_upload")

    prep = None
    df_up = None
    if up is not None:
        try:
            df_up = read_uploaded_csv(up.getvalue())
        except ValueError as exc:
            note("bad", str(exc))
            if st.session_state.get("_last_batch_err") != up.file_id:
                st.session_state["_last_batch_err"] = up.file_id
                modals.notice("bad", "Unreadable file", str(exc))
        if df_up is not None:
            prep = prepare_batch(df_up, features)
            section("Preview", "eye")
            with glass():
                st.dataframe(df_up.head(10), width="stretch")
                st.caption(f"{len(df_up):,} rows × {df_up.shape[1]} columns")
            if prep.missing_columns:
                note("bad", f"<b>Incompatible file.</b> {prep.errors[0]}")
                if st.session_state.get("_last_incompat") != up.file_id:
                    st.session_state["_last_incompat"] = up.file_id
                    modals.notice("bad", "Incompatible CSV", f"{len(prep.missing_columns)} required column(s) are missing.", prep.missing_columns)
            else:
                for m in prep.messages:
                    note("warn", m)
                if prep.errors:
                    note("bad", " ".join(prep.errors))
                else:
                    note("ok", f"<b>{len(prep.X):,}</b> valid record(s) ready for prediction.")
                if len(prep.dropped):
                    with st.expander(f"Excluded rows ({len(prep.dropped):,})"):
                        st.dataframe(prep.dropped, width="stretch", hide_index=True)

    run = st.button("Run batch prediction", type="primary", icon=":material/play_arrow:", key="batch_run",
                    disabled=not (prep is not None and prep.ok))
    if run and prep is not None and prep.ok:
        try:
            with st.spinner(f"Predicting {len(prep.X):,} records with the saved models…"):
                res = predict_all(bundle, prep.X)
            out = res.table.copy()
            out.insert(0, "row", out.index.to_numpy() + 1)
            out = out.reset_index(drop=True)
            lab_col = next((c for c in df_up.columns if str(c).strip().lower() in ("label", "attack_cat")), None)
            if lab_col is not None:
                out["original_label"] = df_up[lab_col].iloc[prep.valid_rows].astype(str).to_numpy()
            st.session_state["last_batch"] = {"table": out, "X": prep.X, "name": up.name, "result": res,
                                              "excluded": int(prep.n_input - len(prep.X))}
            modals.notice("ok", "Prediction complete", f"{len(out):,} record(s) classified with the saved models.",
                          res.notes or None)
        except Exception as exc:  # noqa: BLE001
            note("bad", f"Prediction failed ({type(exc).__name__}). The file is incompatible with the saved models.")

    lb = st.session_state.get("last_batch")
    if lb is not None:
        t: pd.DataFrame = lb["table"]
        section(f"Results · {lb['name']}", "chart")
        n_att = int((t["final_verdict"] != "Normal").sum())
        metric_row([
            {"label": "Records classified", "value": f"{len(t):,}", "icon": "database", "sub": f"{lb.get('excluded', 0):,} excluded by validation", "accent": "#4f8cff"},
            {"label": "Normal traffic", "value": f"{len(t) - n_att:,}", "icon": "shield", "sub": f"{(len(t) - n_att) / len(t):.1%} of records", "accent": "#2ee6a6"},
            {"label": "Attacks detected", "value": f"{n_att:,}", "icon": "warning", "sub": f"{n_att / len(t):.1%} of records", "accent": "#ff4d6d"},
            {"label": "Attack categories", "value": f"{t.loc[t['final_verdict'] != 'Normal', 'final_verdict'].nunique():,}", "icon": "bug", "sub": "distinct in this file", "accent": "#8b6bff"},
        ])
        for n in lb["result"].notes:
            note("info", n)
        c1, c2 = st.columns([3, 2])
        counts = t["final_verdict"].value_counts().rename_axis("verdict").reset_index(name="records")
        with c1, glass():
            charts.show(charts.bar(counts, "verdict", "records", "Predicted traffic categories", height=360, log=len(counts) > 1 and counts["records"].max() > 50 * max(counts["records"].min(), 1)), key="batch_counts")
        with c2, glass():
            charts.show(charts.donut(["Normal", "Attack"], [len(t) - n_att, n_att], "Normal vs. attack", ["#2ee6a6", "#ff4d6d"], 360), key="batch_donut")

        with glass():
            st.markdown("**Prediction table**")
            only_att = st.checkbox("Show attacks only", key="batch_only_att")
            shown = t[t["final_verdict"] != "Normal"] if only_att else t
            fmt = {c: st.column_config.NumberColumn(format="percent") for c in ("attack_probability", "binary_confidence", "multiclass_confidence") if c in shown}
            st.dataframe(shown.head(2000), width="stretch", hide_index=True, column_config=fmt)
            if len(shown) > 2000:
                st.caption(f"Showing the first 2,000 of {len(shown):,} rows - the download contains everything.")
            st.download_button("Download results (CSV)", t.to_csv(index=False).encode("utf-8"), "nids_predictions.csv", "text/csv",
                               type="primary", icon=":material/download:", key="batch_dl")

        section("Inspect one record", "search")
        rec = st.number_input("Row number in the uploaded file", min_value=int(t["row"].min()), max_value=int(t["row"].max()),
                              value=int(t["row"].iloc[0]), step=1, key="batch_inspect")
        pos = t.index[t["row"] == rec]
        if len(pos):
            render_single(lb["result"], int(pos[0]))
            st.caption("To see why, open the Explainability page and pick this batch as the record source.")
        else:
            note("warn", "That row was excluded by validation or is not in the file.")
''',
    "insights": r'''
"""Dataset Insights - interactive analysis of CIC-IDS2017 / UNSW-NB15 (real files only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from components import cards, charts, modals
from components.cards import glass, metric_row, note, page_header, section
from utils.data_loader import (DATASETS, DatasetBundle, bundle_from_frame, clean_numeric_frame, find_files,
                               load_disk_dataset, signature)
from utils.prediction import is_benign
from utils.sources import read_uploaded_csv

page_header("database", "Data", "Dataset Insights",
            "Explore the real flow datasets: class balance, attack types, feature correlations and how normal and malicious traffic differ.")

name = st.radio("Dataset", list(DATASETS), horizontal=True, key="ds_name")
meta = DATASETS[name]
files = find_files(name)


def chart_head(title: str, info_key: str) -> None:
    a, b = st.columns([9, 1])
    a.markdown(f"**{title}**")
    with b:
        modals.info_button("chart", info_key)


# ---- obtain a dataset bundle: disk first, upload as the alternative ----------------------
bundle: DatasetBundle | None = None
if files:
    with st.spinner(f"Scanning {len(files)} {name} file(s) - the first scan of large datasets can take a minute…"):
        res = load_disk_dataset(name, signature(files))
    if isinstance(res, str):
        note("bad", f"<b>Dataset loading failed.</b> {res}")
    else:
        bundle = res
        note("ok", f"Using <b>{len(files)}</b> file(s) found in <code>{meta['folder']}</code>.")
else:
    note("warn", f"<b>No {name} files found.</b> Put the CSV files in <code>{meta['folder']}</code> (or anywhere under <code>data/</code> with "
                 f"“{meta['match'][0]}” in the path), or upload a file below. Required: {meta['files_hint']}")

with st.expander("Upload a dataset file instead" if not files else "Analyse a different file (upload)", icon=":material/upload_file:", expanded=not files):
    up = st.file_uploader(f"{name} CSV (needs a label column: {', '.join(meta['label_candidates'])})", type=["csv"], key=f"ds_up_{name}")
    if up is not None:
        try:
            frame = read_uploaded_csv(up.getvalue())
            ub = bundle_from_frame(frame, name, up.name)
            if isinstance(ub, str):
                note("bad", ub)
            else:
                bundle = ub
                note("ok", f"Analysing the uploaded file <b>{up.name}</b> instead of the files on disk.")
        except ValueError as exc:
            note("bad", str(exc))

if bundle is None:
    st.stop()

df = bundle.sample
types = list(bundle.attack_counts.index)

# ---- overview --------------------------------------------------------------------------
section("Overview", "layers")
metric_row([
    {"label": "Total records", "value": f"{bundle.total_records:,}", "icon": "database", "sub": f"{len(bundle.files)} file(s)", "accent": "#4f8cff"},
    {"label": "Total features", "value": f"{len(bundle.feature_columns):,}", "icon": "sliders", "sub": f"{len(bundle.numeric_columns)} numeric", "accent": "#22e3ff"},
    {"label": "Attack classes", "value": f"{bundle.n_attack_classes:,}", "icon": "bug", "sub": "excluding normal traffic", "accent": "#8b6bff"},
    {"label": "Normal records", "value": f"{bundle.normal_records:,}", "icon": "shield", "sub": f"{bundle.normal_records / bundle.total_records:.1%}", "accent": "#2ee6a6"},
    {"label": "Malicious records", "value": f"{bundle.malicious_records:,}", "icon": "warning", "sub": f"{bundle.malicious_records / bundle.total_records:.1%}", "accent": "#ff4d6d"},
])
with glass():
    st.markdown("**Dataset characteristics**")
    st.write(meta["description"])
    for n in bundle.notes:
        st.caption(n)
    st.caption("Files: " + ", ".join(bundle.files[:12]) + (" …" if len(bundle.files) > 12 else ""))

# ---- class distribution ------------------------------------------------------------------
section("Class distribution", "chart")
c1, c2 = st.columns([2, 3])
with c1, glass():
    chart_head("Binary class distribution", "binary_dist")
    charts.show(charts.donut(["Normal", "Malicious"], [bundle.normal_records, bundle.malicious_records], "Normal vs. malicious",
                             ["#2ee6a6", "#ff4d6d"], 380), key="ds_binary")
with c2, glass():
    chart_head("Attack-type distribution", "attack_dist")
    log = st.toggle("Logarithmic scale", value=True, key="ds_log")
    d = bundle.attack_counts.rename_axis("category").reset_index(name="records")
    fig = charts.bar(d, "category", "records", "Records per category", horizontal=True, log=log, height=max(380, 30 * len(d) + 110), colors=["#22e3ff"])
    charts.show(fig, key="ds_attacks")
with st.expander("Class counts table"):
    tbl = bundle.attack_counts.rename_axis("category").reset_index(name="records")
    tbl["share"] = tbl["records"] / bundle.total_records
    st.dataframe(tbl, width="stretch", hide_index=True, column_config={"share": st.column_config.NumberColumn(format="percent"), "records": st.column_config.NumberColumn(format="%d")})

# ---- feature analysis --------------------------------------------------------------------
section("Feature analysis", "search")
nums = bundle.numeric_columns
if not nums:
    note("warn", "No numeric feature columns were found in this dataset.")
    st.stop()

with glass():
    st.markdown("**Filters**")
    f1, f2 = st.columns([3, 2])
    sel_types = f1.multiselect("Traffic categories to include", types, default=types[: min(len(types), 8)], key="ds_types",
                               help="Filters the sampled records used by the feature charts below.")
    clip = f2.slider("Clip extreme values (percentile range)", 0, 10, 1, key="ds_clip",
                     help="Network features are heavy-tailed; clipping the extreme tails makes distributions readable.")
    if bundle.sampled:
        st.caption(f"Feature charts use a random sample of {len(df):,} of {bundle.total_records:,} records.")
view = df[df["_attack_type"].isin(sel_types)] if sel_types else df.iloc[0:0]
if view.empty:
    note("warn", "No records match the selected categories.")
    st.stop()
numeric = clean_numeric_frame(view, nums)
variances = numeric.var().replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
labels_bin = np.where(view["_attack_type"].map(is_benign), "Normal", "Attack")

# correlation
with glass():
    chart_head("Correlation heatmap", "corr")
    default_feats = list(variances.index[:12])
    cf = st.multiselect("Features (2-30)", nums, default=default_feats, key="ds_corr_feats")
    method = st.radio("Method", ["pearson", "spearman"], horizontal=True, key="ds_corr_method")
    if len(cf) >= 2:
        cm = numeric[cf[:30]].corr(method=method).fillna(0)
        charts.show(charts.heatmap(cm.to_numpy(), list(cm.columns), list(cm.index), f"{method.capitalize()} correlation",
                                   zmid=0, zmin=-1, zmax=1, height=max(420, 26 * len(cm) + 160), text=len(cm) <= 16), key="ds_corr")
    else:
        st.caption("Select at least two features.")

# distribution + comparison
feat = st.selectbox("Feature for distribution and comparison", nums, index=nums.index(variances.index[0]) if len(variances) else 0, key="ds_feat")
s = numeric[feat].dropna()
lo, hi = (s.quantile(clip / 100), s.quantile(1 - clip / 100)) if clip else (s.min(), s.max())
mask = numeric[feat].between(lo, hi)
group_by = st.radio("Split by", ["Normal vs. attack", "Attack category"], horizontal=True, key="ds_split")
grp = pd.Series(labels_bin if group_by.startswith("Normal") else view["_attack_type"].to_numpy(), index=numeric.index)
series = {g: numeric.loc[mask & (grp == g), feat].dropna() for g in grp.unique()}
series = {k: v for k, v in series.items() if len(v) > 1}
if len(series) > 8:
    series = dict(sorted(series.items(), key=lambda kv: -len(kv[1]))[:8])
    st.caption("Showing the 8 largest groups.")

d1, d2 = st.columns(2)
with d1, glass():
    chart_head("Numerical feature distribution", "dist")
    ly = st.toggle("Log y-axis", key="ds_ly")
    charts.show(charts.histogram_compare(series, feat, f"Distribution of {feat}", log_y=ly), key="ds_hist")
with d2, glass():
    chart_head("Normal vs. attack comparison", "compare")
    ly2 = st.toggle("Log y-axis", key="ds_ly2")
    charts.show(charts.box_compare(series, feat, f"{feat} by group", log_y=ly2), key="ds_box")
with glass():
    st.markdown("**Summary statistics (from the filtered, clipped sample)**")
    stats = pd.DataFrame({g: v.describe(percentiles=[.5]) for g, v in series.items()}).T
    st.dataframe(stats[["count", "mean", "std", "min", "50%", "max"]], width="stretch")
''',
    "performance": r'''
"""Model Performance - reads the real evaluation reports (never invents numbers)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import streamlit as st

from components import cards, charts, modals
from components.cards import glass, metric_card_html, md, note, page_header, section
from components.data_source import pick_source
from utils.metrics import METRIC_COLS, METRIC_LABELS, Report, confusion_from_row, evaluate_predictions, load_report
from utils.model_loader import artifact_status, get_bundle
from utils.prediction import predict_binary, predict_multiclass

page_header("chart", "Evaluation", "Model Performance",
            "Metrics loaded from your evaluation reports: compare models, inspect errors and see how binary and multiclass tasks differ.")

reports = {"binary": load_report("binary"), "multiclass": load_report("multiclass")}
bundle = get_bundle()

if not any(r.ok for r in reports.values()):
    cards.missing_artifacts_panel(artifact_status(), {"binary_report", "multiclass_report"}, "No evaluation reports found")
    for k, r in reports.items():
        if r.error:
            note("bad", r.error)
    note("info", "Each report is a CSV with one row per model and columns such as <code>model, accuracy, precision, recall, f1, roc_auc</code> "
                 "(optional: <code>fpr, mcc, train_time, predict_time, tn, fp, fn, tp</code>). Different column spellings are recognised. "
                 "You can still evaluate the saved models on your own labelled data in the <b>Live evaluation</b> tab below.")

CARD_ORDER = [("accuracy", "check", "#22e3ff"), ("precision", "target", "#4f8cff"), ("recall", "radar", "#8b6bff"), ("f1", "bolt", "#2ee6a6"),
              ("roc_auc", "chart", "#ffb547"), ("fpr", "warning", "#ff4d6d"), ("mcc", "layers", "#22e3ff"),
              ("train_time", "clock", "#4f8cff"), ("predict_time", "clock", "#8b6bff")]


def fmt(key: str, v: float) -> str:
    if pd.isna(v):
        return "n/a"
    return f"{v:.2f}s" if key.endswith("_time") else f"{v:.4f}"


def metric_cards(row: pd.Series, prefix: str) -> None:
    avail = [(k, i, a) for k, i, a in CARD_ORDER if k in row.index and pd.notna(row[k])]
    for start in range(0, len(avail), 4):
        cols = st.columns(4)
        for j, (k, ico, acc) in enumerate(avail[start:start + 4]):
            with cols[j]:
                md(metric_card_html(METRIC_LABELS[k], fmt(k, row[k]), ico, "", acc, (start + j) * 0.05))
                modals.info_button("metric", k, "About")


def show_report(kind: str) -> None:
    rep: Report = reports[kind]
    if not rep.ok:
        note("warn", f"<b>{kind.capitalize()} report unavailable.</b> {rep.error} Expected file: <code>reports/cicids_{kind}_results.csv</code>.")
        return
    df = rep.df
    for n in rep.notes:
        st.caption(n)
    note("ok", f"Loaded <code>{rep.path}</code> · {len(df)} model(s): {', '.join(df['model'])}")
    if not df["is_adaptive"].any():
        note("info", "No adaptive drift-aware model appears in this report. Its results are produced live on the <b>Concept Drift &amp; Adaptive Learning</b> page.")

    section("Selected model", "brain")
    model_name = st.selectbox("Model", list(df["model"]), key=f"perf_model_{kind}")
    row = df[df["model"] == model_name].iloc[0]
    metric_cards(row, kind)

    metrics_present = [m for m in METRIC_COLS if m in df.columns and df[m].notna().any()]
    section("Model comparison", "chart")
    with glass():
        a, b = st.columns([9, 1])
        a.markdown("**Metric comparison across models**")
        with b:
            modals.info_button("chart", "model_bars")
        default = [m for m in ("accuracy", "precision", "recall", "f1", "roc_auc") if m in metrics_present]
        chosen = st.multiselect("Metrics", metrics_present, default=default, format_func=lambda m: METRIC_LABELS[m], key=f"perf_metrics_{kind}")
        if chosen:
            charts.show(charts.metric_bars(df, chosen, METRIC_LABELS, f"{kind.capitalize()} classification - model comparison"), key=f"perf_bar_{kind}")
            if df[chosen].min().min() > 0.8:
                st.caption("The y-axis starts at 0; use the zoom tool or the table below to compare close scores.")
    c1, c2 = st.columns(2)
    radar_metrics = [m for m in ("accuracy", "precision", "recall", "f1", "roc_auc") if m in metrics_present]
    with c1, glass():
        a, b = st.columns([9, 1])
        a.markdown("**Performance profile**")
        with b:
            modals.info_button("chart", "radar")
        if len(radar_metrics) >= 3:
            charts.show(charts.radar(df.dropna(subset=radar_metrics), radar_metrics, METRIC_LABELS, "Metric profile per model"), key=f"perf_radar_{kind}")
        else:
            st.caption("Needs at least three of accuracy / precision / recall / F1 / ROC-AUC in the report.")
    with c2, glass():
        tcols = [c for c in ("train_time", "predict_time") if c in df.columns and df[c].notna().any()]
        if tcols:
            st.markdown("**Training & prediction time**")
            fig = charts.metric_bars(df, tcols, METRIC_LABELS, "Time (seconds, log scale)", 440)
            fig.update_yaxes(type="log")
            charts.show(fig, key=f"perf_time_{kind}")
        else:
            st.markdown("**Training & prediction time**")
            st.caption("No timing columns in this report.")

    section("Confusion matrix", "target")
    with glass():
        a, b = st.columns([9, 1])
        a.markdown(f"**{model_name}**")
        with b:
            modals.info_button("chart", "cm")
        cm = confusion_from_row(row)
        if cm is None:
            st.caption("The report has no confusion-matrix data for this model (needs TN/FP/FN/TP columns or a serialised matrix). "
                       "Use the Live evaluation tab to compute one from labelled data with the saved model.")
        else:
            norm = st.toggle("Normalise by actual class", key=f"perf_cmnorm_{kind}")
            charts.show(charts.confusion(cm[0], cm[1], f"Confusion matrix - {model_name}", norm), key=f"perf_cm_{kind}")

    with st.expander("Full report table"):
        show = df.drop(columns=["is_adaptive"]).copy()
        st.dataframe(show, width="stretch", hide_index=True)


def live_evaluation() -> None:
    section("Evaluate the saved models on labelled data", "flask")
    md(f'<div style="margin-bottom:.6rem">{cards.provenance_tag("real")}</div>')
    note("info", "This runs the <b>saved models</b> on records you provide, with their ground-truth labels. It is a fresh evaluation on that data - "
                 "not the original experiment from the report files. If the data overlaps the training set, scores will be optimistic.")
    if not bundle.can_predict:
        cards.missing_artifacts_panel(artifact_status(), {"binary_model", "multiclass_model", "features"}, "Saved models are required")
        return
    src = pick_source("perf_live", bundle.features, need_labels=True, max_rows=50_000)
    if src is None:
        return
    st.caption(f"{len(src.X):,} validated records from {src.name}")
    for n in src.notes:
        st.caption(n)
    if not st.button("Evaluate saved models", type="primary", key="perf_live_run", icon=":material/play_arrow:"):
        return
    y_attack = src.y_attack
    with st.spinner("Evaluating…"):
        try:
            if bundle.binary_model is not None:
                Xb = src.X[bundle.order_for("binary")]
                pb = predict_binary(bundle.binary_model, Xb)
                ev = evaluate_predictions(y_attack, pb["is_attack"].to_numpy(), labels=[False, True], positive=True,
                                          proba=pb["attack_probability"].to_numpy() if pb["attack_probability"].notna().all() else None)
                section("Binary model (live)", "shield")
                m = pd.Series({k: ev[k] for k in METRIC_COLS if k in ev})
                metric_cards(m, "live_bin")
                charts.show(charts.confusion(ev["confusion_matrix"], ["Normal", "Attack"], "Confusion matrix - binary (live)"), key="live_cm_bin")
            if bundle.multiclass_model is not None:
                Xm = src.X[bundle.order_for("multiclass")]
                tab, proba, names = predict_multiclass(bundle.multiclass_model, Xm, bundle.classes)
                norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())  # noqa: E731
                lookup = {norm(n): n for n in names}
                truth = src.labels.map(lambda v: lookup.get(norm(v)))
                known = truth.notna().to_numpy()
                section("Multiclass model (live)", "bug")
                if known.sum() == 0:
                    note("bad", "None of the dataset's labels match the saved class names, so multiclass metrics cannot be computed. "
                                f"Dataset labels (e.g. {', '.join(map(str, src.labels.unique()[:4]))}) vs. model classes (e.g. {', '.join(names[:4])}).")
                else:
                    if (~known).any():
                        note("warn", f"{int((~known).sum()):,} record(s) with labels unknown to the model were excluded from multiclass metrics.")
                    yt, yp = truth[known].to_numpy(), tab["multiclass_prediction"].to_numpy()[known]
                    evm = evaluate_predictions(yt, yp, labels=names, proba=proba[known] if proba is not None else None)
                    metric_cards(pd.Series({k: evm[k] for k in METRIC_COLS if k in evm}), "live_mc")
                    st.caption("Multiclass precision / recall / F1 / ROC-AUC are support-weighted averages; FPR is the macro-average over classes.")
                    present = [n for n in names if n in set(yt) | set(yp)]
                    sub = evaluate_predictions(yt, yp, labels=present)
                    charts.show(charts.confusion(sub["confusion_matrix"], present, "Confusion matrix - multiclass (live)", normalize=True), key="live_cm_mc")
        except Exception as exc:  # noqa: BLE001
            note("bad", f"Evaluation failed ({type(exc).__name__}). The data is incompatible with the saved models.")


tab_b, tab_m, tab_c, tab_l = st.tabs(["Binary", "Multiclass", "Binary vs. multiclass", "Live evaluation"])
with tab_b:
    show_report("binary")
with tab_m:
    show_report("multiclass")
with tab_c:
    rb, rm = reports["binary"], reports["multiclass"]
    if rb.ok and rm.ok:
        a, b = rb.df, rm.df
        common = [m for m in ("accuracy", "precision", "recall", "f1", "roc_auc", "mcc") if m in a.columns and m in b.columns]
        models = [x for x in a["model"] if x in set(b["model"])]
        if not models or not common:
            note("warn", "The two reports share no model names or metrics to compare.")
        else:
            with glass():
                ca, cb = st.columns([9, 1])
                ca.markdown("**Binary versus multiclass performance**")
                with cb:
                    modals.info_button("chart", "bin_vs_multi")
                metric = st.selectbox("Metric", common, format_func=lambda m: METRIC_LABELS[m], key="perf_cmp_metric")
                rows = pd.concat([a[a["model"].isin(models)][["model", metric]].assign(task="Binary"),
                                  b[b["model"].isin(models)][["model", metric]].assign(task="Multiclass")])
                fig = charts.bar(rows, "model", metric, f"{METRIC_LABELS[metric]} - binary vs. multiclass", color="task", colors=["#22e3ff", "#8b6bff"])
                charts.show(fig, key="perf_cmp")
            st.caption("Multiclass averages are weighted/macro depending on how your report was produced - check your notebook before comparing absolute values.")
    else:
        note("info", "Both <code>cicids_binary_results.csv</code> and <code>cicids_multiclass_results.csv</code> are needed for this comparison.")
with tab_l:
    live_evaluation()
''',
    "explain": r'''
"""Explainability - SHAP (global + local) and LIME (local) for the saved models."""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from components import cards, charts, modals
from components.cards import glass, metric_row, note, page_header, result_card, section
from components.data_source import pick_source
from utils import explainability as xai
from utils.model_loader import artifact_status, get_bundle
from utils.prediction import class_names, is_benign, predict_binary, predict_multiclass

matplotlib.use("Agg")

page_header("eye", "Explainable AI", "Explainability",
            "See which network-flow features drive a prediction - globally with SHAP, and record by record with SHAP and LIME.")
if st.button("SHAP & LIME in simple language", icon=":material/help:", key="xai_info"):
    modals.xai_info_dialog()

bundle = get_bundle()
if not bundle.can_predict:
    cards.missing_artifacts_panel(artifact_status(), {"binary_model", "multiclass_model", "features", "classes"}, "Explainability needs the saved models")
    for p in bundle.problems:
        note("bad", p)
    st.stop()

available = [k for k in ("binary", "multiclass") if bundle.model_for(k) is not None]
kind = st.radio("Model to explain", available, format_func=lambda k: {"binary": "Binary (normal vs. attack)", "multiclass": "Multiclass (attack category)"}[k],
                horizontal=True, key="xai_kind")
model = bundle.model_for(kind)
order = bundle.order_for(kind)
ek = xai.explainer_kind(model)
if ek == "unsupported":
    note("bad", "This model type has no <code>predict_proba</code> and cannot be explained by the available methods.")
    st.stop()
cls_names = ["Normal", "Attack"] if kind == "binary" else class_names(model, bundle.classes)
if kind == "binary":  # label order aligned with model.classes_
    from utils.prediction import binary_values
    atk = binary_values(model)[1]
    cls_names = ["Attack" if c == atk else "Normal" for c in model.classes_]
note("info", f"Explainer for this model: <b>{xai.KIND_LABEL[ek]}</b>.")

# ---------------- reference / record source ----------------
section("Records to explain", "database")
src = pick_source("xai", order)
if src is None:
    st.stop()
X_all = src.X[order]
labels_all = src.labels
if len(X_all) > 3000:
    take = np.sort(np.random.default_rng(0).choice(len(X_all), 3000, replace=False))
    X_all = X_all.iloc[take].reset_index(drop=True)
    labels_all = None if labels_all is None else labels_all.iloc[take].reset_index(drop=True)
    st.caption("A random sample of 3,000 records is used as the explanation pool.")
st.caption(f"Pool: {len(X_all):,} records from {src.name}. This pool also serves as the reference data (background) for SHAP and LIME.")
if len(X_all) < 20:
    note("warn", "Fewer than 20 records: LIME and global SHAP need a larger reference pool. Pick a larger source for reliable explanations.")

try:
    engine = xai.get_engine(model, kind, X_all.head(300), order)
except xai.ExplainError as exc:
    note("bad", f"<b>SHAP is not available for this model.</b> {exc}")
    engine = None


def predicted_for(i: int) -> tuple[str, int, float | None]:
    """(label shown, class index in model.classes_, probability of that class)."""
    row = X_all.iloc[[i]]
    if kind == "binary":
        p = predict_binary(model, row)
        is_atk = bool(p["is_attack"].iloc[0])
        idx = xai.binary_attack_index(model) if is_atk else 1 - xai.binary_attack_index(model)
        prob = p["attack_probability"].iloc[0]
        return ("Attack" if is_atk else "Normal"), idx, (None if pd.isna(prob) else float(prob if is_atk else 1 - prob))
    t, proba, names = predict_multiclass(model, row, bundle.classes)
    lab = t["multiclass_prediction"].iloc[0]
    return lab, names.index(lab), (None if proba is None else float(proba[0, names.index(lab)]))


def _model_name(m) -> str:
    final = m.steps[-1][1] if hasattr(m, "steps") and m.steps else m
    return type(final).__name__


with glass():
    cards.md('<div class="panel-title">Session info</div>')
    cards.md('<div class="info-grid">'
             f'<span>Model</span><span>{_model_name(model)}</span>'
             f'<span>Task</span><span>{"Binary detection" if kind == "binary" else "Attack classification"}</span>'
             f'<span>Explainer</span><span>{xai.KIND_LABEL[ek].split(" (")[0]}</span>'
             f'<span>Features</span><span>{len(order)}</span>'
             f'<span>Reference pool</span><span>{len(X_all):,} records · {src.name}</span></div>')

tab_g, tab_l, tab_lime, tab_agree = st.tabs(["SHAP · global", "SHAP · local", "LIME · local", "SHAP vs LIME"])

# --------------------------------------------------------------------------- #
# SHAP global
# --------------------------------------------------------------------------- #
def _tab_global() -> None:
    section("Which features matter most overall?", "chart")
    if engine is None:
        return
    g1, g2, g3 = st.columns(3)
    max_n = min(500 if ek != "kernel" else 40, len(X_all))
    n_s = g1.slider("Records used", 20, max_n, min(200 if ek != "kernel" else 30, max_n), key="xai_gn") if max_n > 20 else max_n
    top = g2.select_slider("Top features", [10, 15, 20], value=15, key="xai_top")
    if kind == "multiclass":
        g_cls = g3.selectbox("Class to explain", cls_names, index=0, key="xai_gcls")
        g_idx = cls_names.index(g_cls)
    else:
        g_cls, g_idx = "Attack", xai.binary_attack_index(model)
        g3.caption("Explaining the output for the **Attack** class.")
    sample = X_all.sample(min(n_s, len(X_all)), random_state=1)
    ckey = ("g", kind, src.name, len(X_all), n_s, g_idx)
    if st.button("Compute global SHAP", type="primary", key="xai_grun", icon=":material/play_arrow:") or st.session_state.get("xai_gkey") == ckey:
        if st.session_state.get("xai_gkey") != ckey:
            try:
                with st.spinner(f"Computing SHAP values for {len(sample)} records…"):
                    vals, base = xai.shap_values(engine, sample, g_idx)
                st.session_state.update(xai_gkey=ckey, xai_gvals=vals, xai_gsample=sample)
            except xai.ExplainError as exc:
                note("bad", str(exc))
                st.session_state.pop("xai_gkey", None)
        if st.session_state.get("xai_gkey") == ckey:
            vals, smp = st.session_state["xai_gvals"], st.session_state["xai_gsample"]
            imp = xai.global_importance(vals, order)
            with glass():
                a, b = st.columns([9, 1])
                a.markdown(f"**Top {top} features · class: {g_cls}**")
                with b:
                    modals.info_button("chart", "shap_global")
                charts.show(charts.importance(imp, f"Global feature importance (mean |SHAP|) · {g_cls}", top), key="xai_gimp")
            tops = imp.head(3)["feature"].tolist()
            note("info", f"<b>Reading this:</b> across the {len(smp)} records analysed, <b>{tops[0]}</b>, <b>{tops[1]}</b> and <b>{tops[2]}</b> change the model's "
                         f"output for <i>{g_cls}</i> the most. Importance shows how much a feature matters, not whether high values raise or lower the score - see the summary plot below.")
            with glass():
                a, b = st.columns([9, 1])
                a.markdown("**SHAP summary (beeswarm)**")
                with b:
                    modals.info_button("chart", "shap_summary")
                try:
                    charts.show(charts.beeswarm(vals, smp[order].to_numpy(float), order, top, ""), key="xai_swarm")
                    st.caption("Each dot is one record. Right of the centre line = the feature pushed the output up; colour shows whether that record's value was low (blue) or high (pink).")
                except Exception as exc:  # noqa: BLE001
                    st.caption(f"The summary plot is not available for this model ({type(exc).__name__}). The bar chart above is unaffected.")
            with st.expander("Importance table"):
                st.dataframe(imp.head(30), width="stretch", hide_index=True)
    else:
        note("info", "Press <b>Compute global SHAP</b> to analyse the selected records.")


with tab_g:
    _tab_global()

# --------------------------------------------------------------------------- #
# record selection shared by local tabs
# --------------------------------------------------------------------------- #
def select_record(prefix: str) -> tuple[int, str, int, float | None]:
    c1, c2 = st.columns([1, 3])
    i = int(c1.number_input("Record index in pool", 0, len(X_all) - 1, min(int(st.session_state.get("xai_rec", 0)), len(X_all) - 1), 1, key=f"{prefix}_rec"))
    st.session_state["xai_rec"] = i
    label, idx, prob = predicted_for(i)
    with c2:
        meta = [f"Model: <b>{kind}</b>"]
        if prob is not None:
            meta.append(f"Probability of this class: <b>{prob:.1%}</b>")
        if labels_all is not None:
            meta.append(f"Recorded label in the data: <b>{labels_all.iloc[i]}</b>")
        result_card(not is_benign(label), label, " · ".join(meta), "Model prediction")
    return i, label, idx, prob


def explain_sentence(tab: pd.DataFrame, value_col: str, label: str, name_col: str = "feature") -> str:
    pos = tab[tab[value_col] > 0].head(3)
    neg = tab[tab[value_col] < 0].head(2)
    p = ", ".join(f"<b>{r[name_col]}</b>" for _, r in pos.iterrows()) or "no single feature"
    s = f"The record is classified as <b>{label}</b> mainly because {p} pushed the model toward it"
    if len(neg):
        s += "; " + ", ".join(f"<b>{r[name_col]}</b>" for _, r in neg.iterrows()) + " pulled the other way"
    return s + "."


# --------------------------------------------------------------------------- #
# SHAP local
# --------------------------------------------------------------------------- #
def _tab_local() -> None:
    section("Why was this record classified this way?", "search")
    if engine is None:
        return
    i, label, pred_idx, _ = select_record("xai_l")
    if kind == "multiclass":
        e_cls = st.selectbox("Explain contributions toward class", cls_names, index=pred_idx, key="xai_lcls")
        e_idx = cls_names.index(e_cls)
    else:
        e_cls, e_idx = label, pred_idx
    try:
        vals, base = xai.shap_values(engine, X_all.iloc[[i]], e_idx)
        tab = xai.local_table(vals[0], X_all.iloc[i], order)
    except xai.ExplainError as exc:
        note("bad", str(exc))
        return

    with glass():
        charts.show(charts.contributions(tab, "feature", "contribution", f"SHAP contributions toward “{e_cls}”", 15,
                                         pos_name=f"toward {e_cls}", neg_name=f"away from {e_cls}"), key="xai_lbar")
        st.caption(f"Baseline (average model output): {base:.4f} · sum of all contributions: {vals[0].sum():+.4f} · "
                   "units are the model's raw output (probability for forests/trees, log-odds for XGBoost).")
    note("info", explain_sentence(tab, "contribution", e_cls))
    p1, p2 = st.columns(2)
    with p1, glass():
        st.markdown("**Top positive contributors**")
        st.dataframe(tab[tab["contribution"] > 0].head(8), width="stretch", hide_index=True,
                     column_config={"contribution": st.column_config.NumberColumn(format="%+.4f"), "value": st.column_config.NumberColumn(format="%.4g")})
    with p2, glass():
        st.markdown("**Top negative contributors**")
        st.dataframe(tab[tab["contribution"] < 0].head(8), width="stretch", hide_index=True,
                     column_config={"contribution": st.column_config.NumberColumn(format="%+.4f"), "value": st.column_config.NumberColumn(format="%.4g")})
    with st.expander("Original network-flow feature values"):
        st.dataframe(X_all.iloc[[i]].T.rename(columns={X_all.index[i]: "value"}), width="stretch")


with tab_l:
    _tab_local()

# --------------------------------------------------------------------------- #
# LIME
# --------------------------------------------------------------------------- #
def _tab_lime() -> None:
    section("LIME - an alternative, model-agnostic explanation", "flask")
    i2, label2, pred_idx2, _ = select_record("xai_m")
    l1, l2, l3 = st.columns(3)
    nf = l1.slider("Features in explanation", 5, 25, 15, key="lime_nf")
    ns = l2.select_slider("Perturbation samples", [500, 1000, 2000, 5000], value=2000, key="lime_ns", help="More samples = more stable but slower.")
    if kind == "multiclass":
        lime_cls = l3.selectbox("Explain class", cls_names, index=pred_idx2, key="lime_cls")
        lime_idx = cls_names.index(lime_cls)
    else:
        lime_cls, lime_idx = label2, pred_idx2
    lkey = ("lime", kind, src.name, i2, nf, ns, lime_idx)
    if st.button("Run LIME", type="primary", key="lime_run", icon=":material/play_arrow:"):
        try:
            with st.spinner("LIME is sampling the neighbourhood of this record…"):
                out = xai.lime_explain(model, X_all, X_all.iloc[i2], order, cls_names, lime_idx, nf, ns)
            st.session_state.update(lime_key=lkey, lime_out=out)
        except xai.ExplainError as exc:
            note("bad", str(exc))
            st.session_state.pop("lime_key", None)
    if st.session_state.get("lime_key") == lkey:
        out = st.session_state["lime_out"]
        with glass():
            charts.show(charts.contributions(out, "rule", "weight", f"LIME weights toward “{lime_cls}”", nf,
                                             pos_name=f"toward {lime_cls}", neg_name=f"away from {lime_cls}"), key="lime_bar")
            st.caption("Each bar is a value range (rule) of a feature that the local surrogate model found influential for this record.")
        note("info", explain_sentence(out, "weight", lime_cls, "rule"))
        with st.expander("LIME table"):
            st.dataframe(out, width="stretch", hide_index=True)
        if engine is not None and st.session_state.get("xai_rec") == i2:
            try:
                sv, _ = xai.shap_values(engine, X_all.iloc[[i2]], lime_idx)
                st_ = xai.local_table(sv[0], X_all.iloc[i2], order).head(nf)
                lime_feats = [f for f in order if any(r.startswith(f) or f in r for r in out["rule"].head(8))]
                shap_top = set(st_["feature"].head(8))
                overlap = len(shap_top & set(lime_feats))
                note("info", f"<b>Agreement check:</b> {overlap} of SHAP's top-8 features also appear among LIME's top-8 rules. "
                             "High overlap increases confidence in the explanation; low overlap is a reason to look closer.")
            except xai.ExplainError:
                pass
    else:
        note("info", "Press <b>Run LIME</b> to explain the selected record (it takes a few seconds).")


with tab_lime:
    _tab_lime()


# --------------------------------------------------------------------------- #
# SHAP vs LIME agreement (paper Section IV-E)
# --------------------------------------------------------------------------- #
def _tab_agree() -> None:
    section("Do SHAP and LIME tell the analyst the same story?", "balance-scale")
    note("info", "For each alert the app compares the <b>top-10 features</b> of TreeSHAP and LIME with the Jaccard index "
                 "(|A ∩ B| / |A ∪ B|; 1 = identical, 0 = nothing in common) and measures the time each explainer needs. "
                 "The paper found weak agreement (mean 0.35 on CIC-IDS2017, 0.07 on UNSW-NB15) and a cost gap of about three orders of magnitude.")
    if kind != "binary":
        note("warn", "This study is defined for the binary detector - switch <b>Model to explain</b> to Binary.")
        return
    if engine is None:
        note("bad", "SHAP is not available for this model, so the two explainers cannot be compared.")
        return
    c1, c2, c3 = st.columns(3)
    n_alert = c1.slider("Alerts to compare", 5, 50, 20, key="ag_n")
    k_top = c2.select_slider("Top-k features", [5, 10, 15], value=10, key="ag_k")
    n_samp = c3.select_slider("LIME perturbation samples", [500, 1000, 2000, 5000], value=2000, key="ag_s")
    pred = predict_binary(model, X_all)
    cand = np.flatnonzero(pred["is_attack"].to_numpy())
    basis = "flagged as attack by the binary model"
    if labels_all is not None:
        truth = np.array([not is_benign(v) for v in labels_all.astype(str)])
        both = np.flatnonzero(pred["is_attack"].to_numpy() & truth)
        if len(both):
            cand, basis = both, "correctly detected attack flows (true label = attack and prediction = attack)"
    if len(cand) < 3:
        note("warn", "Fewer than 3 attack alerts were found in this pool - choose a pool that contains attack traffic.")
        return
    pick = np.sort(np.random.default_rng(1).choice(cand, min(n_alert, len(cand)), replace=False))
    st.caption(f"{len(pick)} alerts, sampled at random from the pool's {len(cand):,} {basis}.")
    if st.button("Run agreement study", type="primary", icon=":material/compare_arrows:", key="ag_run"):
        try:
            with st.spinner(f"Explaining {len(pick)} alerts with TreeSHAP and LIME…"):
                st.session_state["ag_out"] = (xai.agreement_study(model, engine, X_all, X_all.iloc[pick], order, xai.binary_attack_index(model), k_top, n_samp), k_top, n_samp)
        except xai.ExplainError as exc:
            note("bad", str(exc))
    out = st.session_state.get("ag_out")
    if out is None:
        return
    df, k_used, s_used = out
    cards.kpi_row([
        dict(label=f"Mean top-{k_used} Jaccard", value=f"{df['jaccard'].mean():.2f}", icon="balance-scale", accent="#ff4fa3", series=df["jaccard"].tolist(),
             detail=f"<b>{(df['jaccard'] == 0).mean():.0%}</b> of alerts share no feature"),
        dict(label="SHAP cost / alert", value=f"{df['shap_ms'].mean():.2f} ms", icon="bolt", accent="#22e3ff", series=df["shap_ms"].tolist(), detail="exact TreeSHAP" if ek == "tree" else "measured now"),
        dict(label="LIME cost / alert", value=f"{df['lime_ms'].mean():.1f} ms", icon="clock", accent="#ffb547", series=df["lime_ms"].tolist(), detail=f"{s_used:,} perturbation samples"),
        dict(label="LIME ÷ SHAP", value=f"{df['lime_ms'].mean() / max(df['shap_ms'].mean(), 1e-9):,.0f}×", icon="chart", accent="#8b6bff", detail="measured on this machine"),
    ])
    with glass():
        fig = charts._fig("", 320)
        fig.add_bar(x=[f"#{r}" for r in df["record"]], y=df["jaccard"], marker=dict(color=df["jaccard"], colorscale=[[0, "#ff4d6d"], [.5, "#8b6bff"], [1, "#22e3ff"]], cornerradius=4),
                    hovertemplate="record %{x}<br>Jaccard %{y:.2f}<extra></extra>")
        fig.add_hline(y=float(df["jaccard"].mean()), line=dict(color="#ffb547", dash="dash"), annotation_text=f"mean {df['jaccard'].mean():.2f}")
        fig.update_yaxes(range=[0, 1], title=f"top-{k_used} Jaccard"); fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
        charts.show(fig, key="ag_bar")
    pick_rec = st.selectbox("Inspect one alert", df["record"].tolist(), format_func=lambda r: f"record #{r} - Jaccard {df.set_index('record').loc[r, 'jaccard']:.2f}", key="ag_rec")
    row = df.set_index("record").loc[pick_rec]
    left, right = st.columns(2)
    for col, ttl, feats in ((left, "TreeSHAP top features", row["shap_top"]), (right, "LIME top features", row["lime_top"])):
        with col, glass():
            cards.md(f'<div class="panel-title">{ttl}</div>')
            chips = "".join(f'<span class="chip {"g" if f in row["shared"] else ""}">{i + 1}. {f}</span> ' for i, f in enumerate(feats))
            cards.md(chips)
    st.caption("Green chips are features that both explainers rank in their top-k. Weak agreement means two analysts using different explainers could reach different conclusions about the same alert, "
               "so explanation stability should be measured, not assumed. TreeSHAP is the sensible real-time default; LIME works best as an on-demand second opinion.")


with tab_agree:
    _tab_agree()
''',
    "drift": r'''
"""Concept Drift & Adaptive Learning - River ADWIN monitoring and drift-triggered adaptation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from components import cards, charts, modals
from components.cards import architecture_flow, glass, md, metric_row, note, page_header, section
from components.data_source import pick_source
from utils.drift_detection import DriftConfig, inject_synthetic_drift, run_drift_experiment
from utils.metrics import METRIC_LABELS
from utils.model_loader import artifact_status, get_bundle

page_header("activity", "Core research contribution", "Concept Drift & Adaptive Learning",
            "Monitor a model's error stream with ADWIN, raise drift alerts, adapt the model, and compare it with a static one.")
md(f'<span class="tag-sim">{cards.icon("flask")} Simulated stream over recorded flows - not live network monitoring</span>')

# --------------------------------------------------------------------------- #
# A. explanation
# --------------------------------------------------------------------------- #
section("What is concept drift?", "book")
c = st.columns(4)
explain = [
    ("brain", "Concept drift", "The statistical relationship between network-flow features and the correct label changes over time, so what the model learned stops being true."),
    ("network", "Why traffic changes", "New applications and protocols, changed user behaviour, new attack tools and variants, and attackers deliberately mimicking normal traffic."),
    ("warning", "Why static models degrade", "A model trained once keeps applying yesterday's patterns. Its error creeps up silently - often unnoticed until attacks are missed."),
    ("robot", "Why adapt?", "Detecting the change and updating the model keeps detection rates high against evolving threats without waiting for a manual retraining cycle."),
]
for col, (ico, t, d) in zip(c, explain):
    col.markdown(cards.feature_card(ico, t, d), unsafe_allow_html=True)

with glass():
    st.markdown("**Workflow implemented on this page**")
    architecture_flow([
        ("database", "Incoming data", "sequential batches"), ("brain", "Prediction", "static + adaptive"),
        ("activity", "Error monitoring", "1 = wrong, 0 = right"), ("radar", "ADWIN", "River drift detector"),
        ("warning", "Drift alert", "logged per batch"), ("robot", "Adaptive retraining", "recent labelled window"),
        ("check", "Updated evaluation", "vs. static model"),
    ])
    st.caption("Assumption: ground-truth labels arrive with each batch (in practice they come with delay, e.g. from analyst feedback). "
               "The static model is the saved binary model and is never changed.")

# --------------------------------------------------------------------------- #
# B/C. experiment setup
# --------------------------------------------------------------------------- #
section("Run a drift experiment", "flask")
bundle = get_bundle()
if bundle.binary_model is None or not bundle.features:
    cards.missing_artifacts_panel(artifact_status(), {"binary_model", "features"}, "The drift experiment needs the saved binary model")
    for p in bundle.problems:
        note("bad", p)
    st.stop()
model = bundle.binary_model
order = bundle.order_for("binary")

src = pick_source("drift", order, need_labels=True, chronological_option=True, max_rows=100_000)
if src is None:
    note("info", "The experiment needs real labelled flow records (CIC-IDS2017 CSVs in <code>data/</code> or an uploaded CSV with a label column). "
                 "No synthetic data is generated for you.")
    st.stop()
for n in src.notes:
    st.caption(n)
y_all = src.y_attack
st.caption(f"{len(src.X):,} labelled records · {y_all.mean():.1%} attacks · source: {src.name}")
if len(set(y_all)) < 2:
    note("bad", "The records contain only one traffic class, so the experiment is not meaningful.")
    st.stop()

with glass():
    st.markdown("**Experiment settings**")
    a, b, c3 = st.columns(3)
    batch_size = a.slider("Batch size (records)", 200, 5000, 1000, 100, key="dr_bs")
    max_batches = b.slider("Maximum batches", 8, 100, min(40, max(8, len(src.X) // batch_size)), key="dr_nb")
    delta = c3.select_slider("ADWIN sensitivity δ", [0.0001, 0.0005, 0.002, 0.01, 0.05], value=0.002, key="dr_delta",
                             help="Confidence parameter of ADWIN. Smaller δ = fewer, more certain alerts; larger δ = more sensitive.")
    strategy = st.radio("Adaptive strategy", ["retrain", "river"], horizontal=True, key="dr_strategy",
                        format_func=lambda s: {"retrain": "Drift-triggered retraining of the saved model's architecture",
                                               "river": "River online learner (Hoeffding Adaptive Tree, learns every record)"}[s])
    d1, d2 = st.columns(2)
    if strategy == "retrain":
        window = d1.slider("Retraining window (recent labelled records)", 1000, 20000, 4000, 500, key="dr_win")
        with_online = d2.checkbox("Also run a River online learner as a third contender (slower)", value=False, key="dr_online")
        replay = d1.checkbox("Replay older labelled records when retraining", value=True, key="dr_replay",
                             help="Adds a random sample of earlier labelled records (0.5 × window) to the recent window so the retrained model does not forget attack types that are absent from the recent window.")
    else:
        window, with_online, replay = 4000, False, False
        d1.caption("The River learner predicts each record, then learns from its true label (test-then-train); ADWIN still monitors its error stream.")
        d2.caption("River processes records one by one in Python - 40k records take roughly a minute.")

    inject = st.toggle("Inject synthetic drift (demonstration)", value=False, key="dr_inject",
                       help="Recorded data may contain little drift. This optionally rescales features partway through the stream.")
    if inject:
        e1, e2, e3 = st.columns(3)
        start = e1.slider("Drift starts at (% of stream)", 20, 80, 50, 5, key="dr_start")
        nfeat = e2.slider("Features shifted", 1, 20, 8, key="dr_nfeat")
        mag = e3.slider("Shift magnitude (×)", 1.2, 5.0, 2.5, 0.1, key="dr_mag")
        note("warn", "<b>Synthetic drift:</b> feature values are rescaled after the chosen point; labels are unchanged. "
                     "Results are a <b>demonstration</b> of the mechanism, not evidence about real traffic change.")

if st.button("Run drift experiment", type="primary", icon=":material/play_arrow:", key="dr_run"):
    X, y = src.X, y_all
    shifted: list[str] = []
    n_used = min(len(X), batch_size * max_batches)
    X, y = X.iloc[:n_used].reset_index(drop=True), y[:n_used]
    if inject:
        X, shifted = inject_synthetic_drift(X, start / 100, nfeat, mag)
        prov = f"SYNTHETIC drift injected at {start}% of the stream into {src.name} (features: {', '.join(shifted[:5])}{'…' if len(shifted) > 5 else ''})"
    else:
        prov = f"Natural record order of {src.name} - no artificial change added"
    cfg = DriftConfig(batch_size=batch_size, max_batches=max_batches, strategy=strategy, window_size=window,
                      adwin_delta=delta, include_river_online=with_online, replay_fraction=0.5 if replay else 0.0)
    bar = st.progress(0.0, text="Starting…")
    try:
        run = run_drift_experiment(model, X[order], y, cfg, prov, progress=lambda f, t: bar.progress(f, text=t))
        st.session_state["drift_run"] = run
        st.session_state["drift_run_synthetic"] = bool(inject)
        bar.empty()
    except Exception as exc:  # noqa: BLE001
        bar.empty()
        note("bad", f"The experiment failed ({type(exc).__name__}: {str(exc)[:160]}). Check that the data matches the saved model.")

# --------------------------------------------------------------------------- #
# D. dashboard
# --------------------------------------------------------------------------- #
run = st.session_state.get("drift_run")
if run is None:
    note("info", "Configure the experiment above and press <b>Run drift experiment</b> to populate the dashboard.")
    st.stop()

section("Drift dashboard", "chart")
synthetic = st.session_state.get("drift_run_synthetic", False)
md(f'<div style="margin-bottom:.7rem">{cards.provenance_tag("sim" if synthetic else "real")} &nbsp; '
   f'<span style="color:#8fa3c0;font-size:.88rem">{run.provenance}</span></div>')
s = run.summary
metric_row([
    {"label": "Batches processed", "value": f"{s['n_batches']:,}", "icon": "layers", "sub": f"{s['records']:,} records · {run.config.batch_size} per batch", "accent": "#4f8cff"},
    {"label": "Drift events", "value": f"{s['n_drift_events']:,}", "icon": "warning", "sub": f"ADWIN δ = {run.config.adwin_delta}", "accent": "#ffb547"},
    {"label": "Detection status", "value": "Drift" if s["n_drift_events"] else "Stable", "icon": "radar",
     "sub": f"last batch: {s['last_batch_status']}", "accent": "#ff4d6d" if s["n_drift_events"] else "#2ee6a6"},
    {"label": "Adaptations", "value": f"{s['n_retrains']:,}" if run.config.strategy == "retrain" else "online",
     "icon": "robot", "sub": "model retrainings" if run.config.strategy == "retrain" else "continuous learning", "accent": "#8b6bff"},
])
for n in run.notes:
    note("warn", n)

with glass():
    a, b = st.columns([9, 1])
    a.markdown("**Prediction error over time**")
    with b:
        modals.info_button("chart", "drift_error")
    charts.show(charts.drift_error_chart(run.batches, run.events), key="dr_err")
with glass():
    a, b = st.columns([9, 1])
    a.markdown("**Drift detection timeline**")
    with b:
        modals.info_button("chart", "drift_timeline")
    charts.show(charts.drift_timeline(run.batches), key="dr_tl")
    if run.error_stream is not None and len(run.error_stream):
        charts.show(charts.adwin_chart(run.error_stream, run.drift_positions, window=max(100, run.config.batch_size // 2)), key="dr_adwin")

# before / after + static vs adaptive
section("Before and after drift", "clock")
if "first_drift_batch" in s and "static_before" in s:
    k = s["first_drift_batch"]
    fig = charts.bar(pd.DataFrame({
        "model": ["Static", "Static", "Adaptive", "Adaptive"], "period": ["Before drift", "After drift", "Before drift", "After drift"],
        "accuracy": [s["static_before"], s["static_after"], s["adaptive_before"], s["adaptive_after"]]}),
        "model", "accuracy", f"Accuracy in the 4 batches before / after the first alert (batch {k})", color="period", colors=["#22e3ff", "#ff4d6d"], height=360)
    fig.update_yaxes(tickformat=".0%", range=[max(0, min(s["static_after"], s["adaptive_after"], s["static_before"], s["adaptive_before"]) - .1), 1.02])
    with glass():
        charts.show(fig, key="dr_ba")
    note("info", f"After the first drift alert (batch {k}) the static model's accuracy changed from <b>{s['static_before']:.1%}</b> to <b>{s['static_after']:.1%}</b>; "
                 f"the adaptive model's from <b>{s['adaptive_before']:.1%}</b> to <b>{s['adaptive_after']:.1%}</b>. "
                 "Interpret with care: a short window and a single run are a demonstration, not a statistical test.")
elif s["n_drift_events"]:
    note("info", "ADWIN raised alerts, but there were not enough batches before and after the first alert for a comparison window. Use more batches.")
else:
    note("info", "No drift was detected in this run, so there is no before/after comparison. Try a smaller δ, more batches, chronological order, or the synthetic-drift option.")

section("Static vs. adaptive model", "layers")
rows = []
for key, name in (("static", "Static model (saved, never updated)"), ("adaptive", "Adaptive drift-aware model"), ("online", "River online learner")):
    if key in run.overall:
        ev = run.overall[key]
        rows.append({"model": name, **{m: ev.get(m) for m in ("accuracy", "precision", "recall", "f1", "fpr", "mcc")}})
cmp_df = pd.DataFrame(rows)
if len(cmp_df):
    g1, g2 = st.columns([3, 2])
    with g1, glass():
        charts.show(charts.metric_bars(cmp_df, ["accuracy", "precision", "recall", "f1"], METRIC_LABELS, "Whole-stream metrics (attack = positive class)", 400), key="dr_cmp")
    with g2, glass():
        st.markdown("**Metric table**")
        st.dataframe(cmp_df.set_index("model").rename(columns=METRIC_LABELS), width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.4f") for c in METRIC_LABELS.values()})
        st.caption("Whole-stream numbers include the period before the first drift, when both models are identical. "
                   "Adaptive retrainings use only labelled records seen earlier in the stream (no look-ahead).")

with st.expander("Drift & adaptation events", expanded=True):
    if len(run.events):
        st.dataframe(run.events, width="stretch", hide_index=True)
    else:
        st.caption("ADWIN did not report any change in the monitored error stream.")
with st.expander("Per-batch results"):
    st.dataframe(run.batches, width="stretch", hide_index=True)
    st.download_button("Download per-batch results (CSV)", run.batches.to_csv(index=False).encode(), "drift_batches.csv", "text/csv", key="dr_dl")
''',
    "lab": r'''
"""Drift Lab - the paper's prequential experiment (Algorithm 1) run on the user's own dataset files."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st

from components import cards, charts
from components.cards import glass, kpi_row, md, note, page_header, provenance_tag, section
from utils import drift_lab as lab
from utils.data_loader import DATASETS, find_files, load_disk_dataset, load_scenario_pool, signature
from utils.paper_data import SCENARIO_NAMES, TABLE_IV

page_header("flask", "Paper · Algorithm 1", "Drift Lab",
            "Replay the paper's experiment: train on seen traffic, stream 1,000-flow batches, let ADWIN watch the error and compare four update policies.")
md(f'<div style="margin:-.4rem 0 1rem">{provenance_tag("real")} <span style="color:var(--muted);font-size:.88rem">'
   '&nbsp;Computed now from your recorded dataset files. The stream is <b>constructed</b> from recorded flows (unseen families held out of training, or inputs shifted) - it is not live traffic.</span></div>')


@st.cache_data(show_spinner=False, max_entries=2)
def _counts(dataset: str, sig: tuple):
    return load_disk_dataset(dataset, sig)


@st.cache_data(show_spinner=False, max_entries=1)
def _pool(dataset: str, sig: tuple, counts: tuple, features: tuple, cap: int):
    return load_scenario_pool(dataset, sig, dict(counts), features, cap_per_class=cap)


def _default_shift(feats: list[str]) -> list[str]:
    """Timing features (idle / inter-arrival) when the dataset has them, otherwise the first few numeric features."""
    import re
    timing = [f for f in feats if re.search(r"idle|iat", str(f), re.I)]
    return (timing or feats)[:5]


# ---- 1. setup --------------------------------------------------------------------------------
section("1 · Choose a scenario", "sliders")
c1, c2 = st.columns([1, 2])
with c1:
    dataset = st.selectbox("Dataset", list(DATASETS), key="lab_ds")
files = find_files(dataset)
if not files:
    note("warn", f"No {dataset} files found in <code>{DATASETS[dataset]['folder']}</code>. {DATASETS[dataset]['files_hint']}")
    st.stop()
with c2:
    kind_label = st.radio("Drift scenario", ["Unseen attack families (S1 / S2)", "Covariate shift - inputs move, labels do not (S3)"], key="lab_kind")
kind = "unseen" if kind_label.startswith("Unseen") else "covariate"

with st.spinner("Counting categories in the dataset files - large files can take a minute the first time…"):
    ds = _counts(dataset, signature(files))
if isinstance(ds, str):
    note("error", ds)
    st.stop()
feats = list(ds.numeric_columns)
cats = [str(c) for c in ds.attack_counts.index]
attack_cats = [c for c in cats if c.lower() not in ("benign", "normal", "normal traffic")]

cfg = lab.ScenarioConfig(kind=kind)
if kind == "unseen":
    u1, u2 = st.columns([3, 1])
    with u1:
        cfg.unseen = st.multiselect("Attack families held out of training (appear only after the drift onset)", attack_cats,
                                    default=lab.default_unseen(dataset, cats), key=f"lab_unseen_{dataset}")
    with u2:
        cfg.unseen_share = st.slider("Share of unseen flows after onset", 0.02, 0.40, 0.10, 0.01, key="lab_share")
else:
    cfg.shift_features = st.multiselect("Features to shift after the onset", feats, default=_default_shift(feats), key=f"lab_shift_{dataset}")

m1, m2 = st.columns([1, 2])
with m1:
    model_name = st.selectbox("Base learner", lab.MODEL_NAMES, key="lab_model")
with m2:
    include_hat = st.toggle("Include the Online HAT policy (River; ~20 s per scenario)", value=True, key="lab_hat")

with st.expander("Advanced settings (defaults reproduce the paper)", icon=":material/tune:"):
    a1, a2, a3 = st.columns(3)
    delta = a1.number_input("ADWIN δ", 0.0001, 0.05, lab.ADWIN_DELTA, 0.0005, format="%.4f", key="lab_delta")
    cfg.batch_size = int(a2.number_input("Batch size (flows)", 200, 5000, lab.BATCH_SIZE, 100, key="lab_bs"))
    cfg.n_batches = int(a3.number_input("Number of batches", 20, 80, lab.N_BATCHES, 1, key="lab_nb"))
    b1, b2, b3 = st.columns(3)
    cfg.onset_batch = int(b1.number_input("Drift-free batches", 5, 30, lab.ONSET_BATCH, 1, key="lab_onset"))
    cfg.n_train = int(b2.number_input("Training flows", 5000, 100000, lab.N_TRAIN, 1000, key="lab_ntrain"))
    cfg.seed = int(b3.number_input("Seed", 0, 9999, 42, 1, key="lab_seed"))
    r1, r2, r3 = st.columns(3)
    recent = int(r1.number_input("Recent window", 500, 20000, lab.RECENT, 500, key="lab_recent"))
    replay = int(r2.number_input("Replay sample", 0, 20000, lab.REPLAY, 500, key="lab_replay"))
    anchor = int(r3.number_input("Anchor sample (initial data)", 0, 20000, lab.ANCHOR, 500, key="lab_anchor"))
    if kind == "covariate":
        s1, s2 = st.columns(2)
        cfg.shift_scale = float(s1.number_input("Shift scale ×", 1.0, 10.0, 3.0, 0.5, key="lab_sscale"))
        cfg.shift_sd = float(s2.number_input("Shift (σ)", 0.0, 5.0, 1.0, 0.5, key="lab_ssd"))
    else:
        cfg.split_by_file = st.checkbox("Train from files named *train*, stream from files named *test* (paper's S2 on UNSW-NB15)",
                                        value=False, key="lab_split")
    cap = int(st.number_input("Max flows read per category (memory guard)", 5000, 200000, 40000, 5000, key="lab_cap"))

if kind == "unseen" and not cfg.unseen:
    note("warn", "Pick at least one attack family to hold out.")
    st.stop()
if kind == "covariate" and not cfg.shift_features:
    note("warn", "Pick at least one feature to shift.")
    st.stop()

go = st.button("Run the experiment", type="primary", icon=":material/play_arrow:", key="lab_run")

# ---- 2. run ----------------------------------------------------------------------------------
if go:
    try:
        t_all = time.perf_counter()
        with st.status("Running the prequential experiment…", expanded=True) as status:
            st.write("Reading a class-capped pool of recorded flows…")
            pool = _pool(dataset, signature(files), tuple((str(k), int(v)) for k, v in ds.attack_counts.items()), tuple(feats), cap)
            if isinstance(pool, str):
                raise RuntimeError(pool)
            sc = lab.build_scenario(pool, feats, cfg)
            for n in sc.notes:
                st.write("• " + n)
            st.write(f"Fitting the initial {model_name} on {len(sc.X_train):,} flows…")
            f0, t0 = lab.fit_initial(sc, model_name)
            results = {}
            for pol in ("Static", "Periodic", "Drift-triggered"):
                st.write(f"Running policy: {pol}…")
                results[pol] = lab.run_policy(sc, f0, pol, delta=delta, seed=cfg.seed, recent=recent, replay=replay, anchor=anchor)
            if include_hat:
                st.write("Running policy: Online HAT (River)…")
                bar = st.progress(0.0)
                results["Online HAT"] = lab.run_online_hat(sc, seed=cfg.seed, progress=lambda p: bar.progress(min(1.0, float(p))))
            status.update(label=f"Done in {time.perf_counter() - t_all:.0f} s", state="complete", expanded=False)
        st.session_state["lab"] = dict(sc=sc, results=results, base_fit=t0, model=model_name, dataset=dataset, kind=kind, delta=delta)
    except Exception as exc:  # noqa: BLE001
        st.session_state.pop("lab", None)
        note("error", f"<b>The experiment could not run.</b> {type(exc).__name__}: {str(exc)[:240]}")

out = st.session_state.get("lab")
if not out:
    note("info", "Choose a scenario and press <b>Run the experiment</b>. Nothing is trained until you do. The saved dashboard models are not used here: "
                 "unseen-family scenarios need a base model that never saw those families, so the lab fits its own on the seen traffic.")
else:
    sc, results = out["sc"], out["results"]
    summ = lab.summarise(sc, results)
    bs = sc.cfg.batch_size
    st.write("")
    section(f"2 · Results · {out['dataset']} · {out['model']} · " + ("unseen families" if out["kind"] == "unseen" else "covariate shift"), "chart")
    ix = summ.set_index("Policy")
    cols = []
    for pol, ic, ac in (("Static", "lock", "#ff4d6d"), ("Periodic", "clock", "#ffb547"), ("Drift-triggered", "radar", "#22e3ff"), ("Online HAT", "bolt", "#8b6bff")):
        if pol in ix.index:
            r = ix.loc[pol]
            rt = "online" if pol == "Online HAT" else f"{int(r['Retrains'])} retrain(s)"
            cols.append(dict(label=pol, value=f"{r['Post-onset acc (%)']:.2f}%", icon=ic, accent=ac,
                             series=lab.batch_accuracy(sc.y_stream, results[pol].yhat, bs).tolist(),
                             detail=f"post-onset accuracy<br>{rt}"))
    kpi_row(cols)

    with glass():
        md('<div class="panel-title">Accuracy per batch · drift starts after the dotted line</div>')
        acc = {p: lab.batch_accuracy(sc.y_stream, r.yhat, bs) for p, r in results.items()}
        rb = {p: r.retrain_batches for p, r in results.items() if p in ("Periodic", "Drift-triggered")}
        al = sorted({a["flow"] // bs + 1 for a in results["Drift-triggered"].alerts if a["triggered_retrain"]})
        charts.show(charts.policy_accuracy_chart(acc, sc.cfg.onset_batch, rb, al), key="lab_acc")
        st.caption("Diamonds mark retrainings; dashed amber lines mark ADWIN alerts that triggered a retrain (Drift-triggered).")

    left, right = st.columns([3, 2])
    with left:
        with glass():
            md('<div class="panel-title">Computed Table IV · all flows of the stream</div>')
            show = summ.copy()
            st.dataframe(show.style.format({"Acc (%)": "{:.2f}", "F1": "{:.3f}", "Rec (%)": "{:.2f}", "FPR (%)": "{:.2f}",
                                            "Post-onset acc (%)": "{:.2f}", "Retrain time (s)": "{:.2f}", "Delay (flows)": "{:.0f}", "Retrains": "{:.0f}"}, na_rep="–"),
                         hide_index=True, width="stretch")
            st.caption(f"Base model fitted in {out['base_fit']:.2f} s. Delay = flows from the onset to the first ADWIN alert that triggered a retrain. "
                       "‘–’ = not applicable (e.g. no alert fired).")
    with right:
        with glass():
            md('<div class="panel-title">ADWIN alert log · Drift-triggered</div>')
            al_rows = results["Drift-triggered"].alerts
            if not al_rows:
                st.caption("ADWIN raised no alert on this stream." + (" That is the expected outcome for a harmless covariate shift." if out["kind"] == "covariate" else ""))
            else:
                df = pd.DataFrame(al_rows)
                df["flow"] = df["flow"].astype(int)
                df["after onset"] = df["flow"] >= sc.onset_flow
                keep = [c for c in ("flow", "batch", "error_before", "error_after", "rose", "triggered_retrain", "after onset") if c in df]
                shown = df[keep].rename(columns={"error_before": "err before", "error_after": "err after", "triggered_retrain": "retrain", "after onset": "post-onset"})
                st.dataframe(shown, hide_index=True, width="stretch")
                false_alarms = int(((df["flow"] < sc.onset_flow) & df["triggered_retrain"]).sum()) if "triggered_retrain" in df else 0
                st.caption(f"{len(df)} alert(s); {false_alarms} retrain(s) triggered before the onset (false alarms).")

    with glass():
        md('<div class="panel-title">Per-category accuracy after the onset · Static vs Drift-triggered</div>')
        cs = lab.category_rates(sc.cat_stream, sc.y_stream, results["Static"].yhat, sc.onset_flow).set_index("category")
        cd = lab.category_rates(sc.cat_stream, sc.y_stream, results["Drift-triggered"].yhat, sc.onset_flow).set_index("category")
        cmp_ = pd.DataFrame({"Static": 100 * cs["rate"], "Drift-triggered": 100 * cd["rate"], "flows": cs["flows"]}).sort_values("flows", ascending=False)
        charts.show(charts.category_bars(cmp_.head(12), ["Static", "Drift-triggered"]), key="lab_cat")
        st.caption("Detection rate for attack categories, specificity for benign traffic. Held-out families are the ones to watch: "
                   + (", ".join(sc.cfg.unseen) if sc.cfg.unseen else "none (covariate scenario)") + ".")

    # ---- threshold study ----
    section("3 · Threshold re-tuning (paper §IV-F)", "target")
    pol_t = st.selectbox("Policy whose scores to re-threshold", [p for p in ("Static", "Drift-triggered", "Periodic") if p in results], key="lab_tpol")
    ts = lab.threshold_study(sc, results[pol_t].proba)
    if ts is None:
        st.caption("Not enough post-onset flows of both classes for an honest tune/evaluate split.")
    else:
        d, t = ts["default"], ts["tuned"]
        kpi_row([dict(label="Tuned threshold", value=f"{ts['threshold']:.2f}", icon="sliders", accent="#ffb547", detail=f"chosen on the first {ts['tune_flows']:,} post-onset flows"),
                 dict(label="FPR · default → tuned", value=f"{100*d['fpr']:.2f}% → {100*t['fpr']:.2f}%", icon="shield", accent="#ff4d6d", detail=f"evaluated on the later {ts['eval_flows']:,} flows"),
                 dict(label="F1 · default → tuned", value=f"{d['f1']:.3f} → {t['f1']:.3f}", icon="chart", accent="#22e3ff", detail="no retraining involved"),
                 dict(label="Recall · default → tuned", value=f"{100*d['recall']:.2f}% → {100*t['recall']:.2f}%", icon="target", accent="#2ee6a6", detail="same held-out evaluation flows")])
        with glass():
            md('<div class="panel-title">Evaluation stream · F1 and FPR vs threshold</div>')
            charts.show(charts.threshold_chart(ts["grid"], [m["f1"] for m in ts["eval_curve"]], [m["fpr"] for m in ts["eval_curve"]], ts["threshold"]), key="lab_thr")
            st.caption("The threshold is picked using labelled flows right after the onset and judged on flows it never saw - a cheap alternative to a full retrain when the ranking (AUC) is still good.")

    with st.expander("Compare with the numbers published in the paper", icon=":material/menu_book:"):
        st.caption("Published reference (Table IV of the paper, its own data split and seed) - not computed here. Differences are expected with other files, samples or seeds.")
        for key in ("S1", "S2", "S3"):
            st.markdown(f"**{SCENARIO_NAMES[key]}**")
            st.dataframe(TABLE_IV[TABLE_IV["Scenario"] == key].drop(columns="Scenario").style.format(na_rep="–"), hide_index=True, width="stretch")

    with st.expander("Caveats", icon=":material/warning:"):
        st.markdown(
            "- True labels are assumed to arrive immediately after each batch is scored; real deployments see delayed, partial labels, which lengthens detection delay.\n"
            "- The stream is **constructed** from recorded flows (held-out families / shifted inputs); it is not live traffic and nothing is blocked or monitored.\n"
            "- One seed and default settings per run: small differences between policies should not be over-read.\n"
            "- Drift here is abrupt; gradual drift is not covered.")
''',
    "paper": r'''
"""Paper Results - the published tables of the project paper, as a labelled reference (not computed by this app)."""
from __future__ import annotations

import streamlit as st

from components import cards, charts
from components.cards import glass, md, note, page_header, section
from utils import paper_data as pd_

page_header("book", "Research paper", "Paper Results", pd_.PAPER_TITLE + ".")
md(f'<span class="tag-sim">{cards.icon("book")} Published in the paper - transcribed, not computed by this app</span>')
st.write("")
note("info", "These tables reproduce what the paper reports so you can compare them with your own runs on the <b>Model Performance</b>, "
             "<b>Drift Lab</b> and <b>Explainability</b> pages, which compute their numbers from your files. "
             "Verify against the PDF before citing.")

t1, t2, t3, t4, t5, t6 = st.tabs(["Binary (Table II)", "Multi-class (Table III)", "Adaptation (Table IV)", "Explainers (Table V)", "Key findings", "Related work (Table I)"])
fmt = lambda d: d.style.format(precision=2, na_rep="–")  # noqa: E731

with t1:
    d = pd_.TABLE_II
    st.dataframe(fmt(d), hide_index=True, width="stretch")
    mets = ["Accuracy (%)", "F1 (%)", "ROC-AUC (%)"]
    for ds in d["Dataset"].unique():
        g = d[d["Dataset"] == ds].copy()
        g[mets] = g[mets] / 100  # chart helper expects fractions
        with glass():
            md(f'<div class="panel-title">{ds} · binary detection</div>')
            charts.show(charts.model_lines(g.rename(columns={"Model": "model"}), mets, {m: m for m in mets}), key=f"pp_b_{ds}")
    st.caption("CIC-IDS2017 uses a random split; UNSW-NB15 uses the official train/test split - so the two datasets are not directly comparable.")
with t2:
    st.dataframe(fmt(pd_.TABLE_III), hide_index=True, width="stretch")
    st.caption("Macro-F1 is far below accuracy when minority classes are rare - the aggregate number hides weak classes.")
with t3:
    for key, name in pd_.SCENARIO_NAMES.items():
        st.markdown(f"**{name}**")
        st.dataframe(fmt(pd_.TABLE_IV[pd_.TABLE_IV["Scenario"] == key].drop(columns="Scenario")), hide_index=True, width="stretch")
    st.caption("Prequential test-then-train, 40 batches of 1,000 flows after 20,000 training flows, drift from batch 16. Retrains/delay not applicable for Online HAT; ‘–’ means no alert.")
with t4:
    st.dataframe(fmt(pd_.TABLE_V), hide_index=True, width="stretch")
    st.caption("Top-10 feature Jaccard overlap between SHAP and LIME on the same alerts, and cost per alert. Reproduce it on the Explainability page (SHAP vs LIME tab).")
with t5:
    for title, body in pd_.FINDINGS:
        with glass():
            st.markdown(f"**{title}**")
            st.write(body)
with t6:
    st.dataframe(pd_.TABLE_I, hide_index=True, width="stretch")
''',
    "about": r'''
"""About Project - academic project information (team/university details come from project_info.json)."""
from __future__ import annotations

import streamlit as st

from components import cards
from components.cards import architecture_flow, chips, glass, md, note, page_header, section, timeline
from components.styles import icon
from utils import paper_data as pdata
from utils.config import PROJECT_INFO_PATH, PROJECT_TITLE, load_project_info

info = load_project_info()
page_header("book", "Academic project", "About the Project", PROJECT_TITLE + ".")

section("Abstract", "target")
with glass():
    st.write(pdata.ABSTRACT)
    md(f'<span style="color:var(--muted);font-size:.85rem">{", ".join(pdata.PAPER_AUTHORS)} · {pdata.PAPER_AFFILIATION}</span>')

section("Contributions", "bolt")
with glass():
    for f in pdata.CONTRIBUTIONS:
        st.markdown(f"- {f}")

c1, c2 = st.columns(2)
with c1:
    section("Datasets", "database")
    with glass():
        st.markdown("**CIC-IDS2017** - flow-based traffic with benign and multiple attack families (DoS, DDoS, brute force, web attacks, botnet, port scan, infiltration).")
        st.markdown("**UNSW-NB15** - mixed real/synthetic traffic with nine attack families (Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms).")
    section("Explainability & drift", "eye")
    with glass():
        chips(["SHAP", "LIME"], "c")
        st.write("")
        chips(["ADWIN (River)"], "g")
with c2:
    section("Machine-learning models", "brain")
    with glass():
        chips(["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "Adaptive Drift-Aware Model"])
        st.caption("The adaptive model is implemented on the Concept Drift page: drift-triggered retraining of the saved model's architecture "
                   "and an incremental River Hoeffding Adaptive Tree. Its results are produced live from your data.")
    section("Major features", "bolt")
    with glass():
        for f in ["Binary detection and multiclass attack classification", "Manual and batch (CSV) prediction with validation",
                  "Interactive dataset analysis", "Model comparison from real evaluation reports", "Global and local SHAP, local LIME",
                  "ADWIN drift detection with adaptive retraining and static-vs-adaptive comparison"]:
            st.markdown(f"- {f}")

section("Research gap", "search")
with glass():
    for h, d in pdata.RESEARCH_GAPS:
        st.markdown(f"**{h}.** {d}")
    note("info", "The Drift Lab builds a <b>constructed stream from recorded flows</b>; it demonstrates the mechanism but is not a claim about live-network performance.")

section("Limitations stated in the paper", "warning")
with glass():
    for f in pdata.LIMITATIONS:
        st.markdown(f"- {f}")

section("System architecture", "network")
with glass():
    architecture_flow([
        ("database", "Datasets", "CIC-IDS2017 · UNSW-NB15"), ("gear", "Preprocessing", "clean · validate"),
        ("brain", "Static models", "LR · DT · RF · XGB"), ("activity", "ADWIN", "error-stream monitor"),
        ("robot", "Adaptive model", "retrain / online"), ("eye", "XAI", "SHAP · LIME"), ("rocket", "Streamlit UI", "10 pages"),
    ])

section("Technologies used", "code")
with glass():
    chips(["Python", "Streamlit", "scikit-learn", "XGBoost", "River", "Pandas", "NumPy", "Imbalanced-learn", "SHAP", "LIME", "Plotly", "Matplotlib", "Seaborn", "Joblib"])

section("Project development timeline", "clock")
tl = info.get("timeline") or []
tl = [t for t in tl if isinstance(t, dict) and (t.get("title") or t.get("when"))]
with glass():
    if tl:
        timeline(tl)
    else:
        timeline([
            {"when": "Phase 1", "title": "Literature survey & problem definition", "desc": "Static IDS limitations, concept drift, explainability."},
            {"when": "Phase 2", "title": "Data preparation", "desc": "CIC-IDS2017 / UNSW-NB15 cleaning, feature selection, imbalance handling."},
            {"when": "Phase 3", "title": "Model training & evaluation", "desc": "LR, Decision Tree, Random Forest, XGBoost; binary and multiclass."},
            {"when": "Phase 4", "title": "Drift detection & adaptive learning", "desc": "ADWIN monitoring and adaptive retraining."},
            {"when": "Phase 5", "title": "Explainability & application", "desc": "SHAP/LIME integration and the Streamlit platform."},
        ])
        st.caption(f"Default phase outline without dates. Add your real timeline under `timeline` in `{PROJECT_INFO_PATH.name}` "
                   "(list of objects with `when`, `title`, `desc`).")

section("Future work", "rocket")
with glass():
    for f in pdata.FUTURE_WORK:
        st.markdown(f"- {f}")

section("Team information", "users")
with glass():
    rows = [("Developed by", info.get("developer")), ("USN", info.get("usn")), ("University", info.get("university")),
            ("Department", info.get("department")), ("Guide", info.get("guide"))]
    for k, v in rows:
        md(f'<div class="side-row" style="font-size:.95rem"><span style="color:#8fa3c0">{k}</span><span><b>{v or "—"}</b></span></div>')
    members = [m for m in (info.get("team_members") or []) if m]
    if members:
        st.markdown("**Team members**")
        for m in members:
            st.markdown(f"- {m if isinstance(m, str) else ' · '.join(str(x) for x in m.values())}")
    if not (info.get("university") and info.get("department")):
        st.caption(f"University, department, guide and team members are configurable: edit `{PROJECT_INFO_PATH.name}` in the project root. Nothing is invented here.")
''',
}

# --------------------------------------------------------------------------------------------- #
# Loader: registers the embedded modules and builds callable pages
# --------------------------------------------------------------------------------------------- #
_ORDER = list(_MODULE_SOURCES)


def _register_sources(name: str, filename: str, source: str) -> None:
    """Make tracebacks and inspect.getsource work for embedded code."""
    lines = source.splitlines(keepends=True)
    linecache.cache[filename] = (len(source), None, lines, filename)


def _load_modules() -> None:
    digest = hashlib.md5("".join(_MODULE_SOURCES.values()).encode()).hexdigest()
    if sys.modules.get("_nids_bundle_digest") == digest and all(n in sys.modules for n in _ORDER):
        return  # already loaded in this server process (Streamlit reruns the script on every interaction)
    for pkg in ("utils", "components"):
        m = types.ModuleType(pkg)
        m.__path__ = []  # mark as package so "from components import cards" works
        sys.modules[pkg] = m
    for name in _ORDER:
        filename = f"<nids-embedded:{name.replace('.', '/')}.py>"
        src = _MODULE_SOURCES[name]
        _register_sources(name, filename, src)
        mod = types.ModuleType(name)
        mod.__file__ = filename
        mod._BUNDLE_DIR = _BUNDLE_DIR
        sys.modules[name] = mod
        exec(compile(src, filename, "exec"), mod.__dict__)
        pkg, _, short = name.partition(".")
        setattr(sys.modules[pkg], short, mod)
    sys.modules["_nids_bundle_digest"] = digest


def _make_page(key: str):
    src = _PAGE_SOURCES[key]
    filename = f"<nids-embedded:pages/{key}.py>"
    _register_sources(key, filename, src)
    code = compile(src, filename, "exec")

    def _page() -> None:
        exec(code, {"__name__": f"nids_page_{key}", "__file__": filename})

    _page.__name__ = f"page_{key}"
    return _page


# --------------------------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------------------------- #
_load_modules()

from components.cards import reset_keys  # noqa: E402
from components.sidebar import render_sidebar  # noqa: E402
from components.styles import inject_css  # noqa: E402
from utils.config import APP_TITLE, NAV_PAGES  # noqa: E402

st.set_page_config(page_title=APP_TITLE, page_icon=":material/shield:", layout="wide", initial_sidebar_state="expanded")

_PAGE_META = [
    ("home", "Home", "home", "home"),
    ("overview", "Overview", "dashboard", "overview"),
    ("predict", "Predict", "radar", "predict"),
    ("insights", "Dataset Insights", "database", "dataset-insights"),
    ("performance", "Model Performance", "monitoring", "model-performance"),
    ("explain", "Explainability", "psychology", "explainability"),
    ("drift", "Concept Drift & Adaptive Learning", "trending_up", "concept-drift"),
    ("lab", "Drift Lab", "science", "drift-lab"),
    ("paper", "Paper Results", "menu_book", "paper-results"),
    ("about", "About Project", "info", "about"),
]

PAGES = [
    st.Page(_make_page(key), title=title, icon=f":material/{icon}:", url_path=url, default=(key == "home"))
    for key, title, icon, url in _PAGE_META
]
NAV_PAGES.update({key: page for (key, *_), page in zip(_PAGE_META, PAGES)})

inject_css()
reset_keys()
render_sidebar()
st.navigation(PAGES).run()
