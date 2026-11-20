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

All models (Random Forest, XGBoost, Decision Tree, Logistic Regression) are evaluated independently
with strict test-then-train ordering, disjoint train/stream splits, and zero target leakage.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from utils.prediction import is_benign
from utils.schema import assert_clean_features, filter_clean_features, is_prohibited_feature

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
MODEL_NAMES = ["Random Forest", "XGBoost", "Decision Tree", "Logistic Regression"]
POLICIES = ["Static", "Periodic", "Drift-triggered", "Online HAT"]


# ----------------------------------------------------------------------------------------------- #
# Models Registry & Factory
# ----------------------------------------------------------------------------------------------- #
def make_xgboost(seed: int = 42):
    """Instantiate XGBClassifier with explicit error handling if not installed."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost is not installed or could not be loaded in this Python environment. "
            "Please install xgboost via pip or select another base learner (Random Forest, Decision Tree, Logistic Regression)."
        ) from exc
    return XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.3,
        tree_method="hist",
        n_jobs=2,
        random_state=seed,
        eval_metric="logloss",
        verbosity=0,
    )


MODEL_REGISTRY = {
    "Random Forest": {
        "class_name": "RandomForestClassifier",
        "module": "sklearn.ensemble",
        "factory": lambda seed: Pipeline([
            ("scale", StandardScaler()),
            ("clf", __import__("sklearn.ensemble", fromlist=["RandomForestClassifier"]).RandomForestClassifier(
                n_estimators=100, n_jobs=2, random_state=seed
            )),
        ]),
    },
    "XGBoost": {
        "class_name": "XGBClassifier",
        "module": "xgboost",
        "factory": lambda seed: Pipeline([
            ("scale", StandardScaler()),
            ("clf", make_xgboost(seed)),
        ]),
    },
    "Decision Tree": {
        "class_name": "DecisionTreeClassifier",
        "module": "sklearn.tree",
        "factory": lambda seed: Pipeline([
            ("scale", StandardScaler()),
            ("clf", DecisionTreeClassifier(random_state=seed)),
        ]),
    },
    "Logistic Regression": {
        "class_name": "LogisticRegression",
        "module": "sklearn.linear_model",
        "factory": lambda seed: Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]),
    },
}


def make_model(name: str, seed: int = 42):
    """Create a fresh Pipeline instance for the requested model with standardized scaling."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Expected one of: {list(MODEL_REGISTRY.keys())}. "
            "Silent fallback to another model is prohibited."
        )
    return MODEL_REGISTRY[name]["factory"](seed)


def get_model_info(name: str, seed: int = 42) -> dict:
    """Return explicit metadata and hyperparameters for the requested model."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Expected one of: {list(MODEL_REGISTRY.keys())}")
    meta = MODEL_REGISTRY[name]
    pipe = meta["factory"](seed)
    clf = pipe.named_steps["clf"]
    params = clf.get_params()
    # JSON-serializable params
    clean_params = {}
    for k, v in params.items():
        if isinstance(v, (int, float, str, bool, list, dict)) or v is None:
            clean_params[k] = v
        else:
            clean_params[k] = str(v)
    return {
        "selected_model_name": name,
        "actual_estimator_class": meta["class_name"],
        "module": meta["module"],
        "hyperparameters": clean_params,
        "random_seed": seed,
    }


def _fit(model, X: pd.DataFrame, y: np.ndarray) -> tuple[object, list[str]]:
    """Fit a clone of the model, strictly asserting clean input features and recording convergence warnings."""
    assert_clean_features(X)
    m = clone(model)
    conv_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.fit(X, y.astype(int))
        for w in caught:
            msg = str(w.message)
            if "ConvergenceWarning" in w.category.__name__ or "converge" in msg.lower():
                conv_warnings.append(msg)
    return m, conv_warnings


def _proba(model, X: pd.DataFrame) -> np.ndarray:
    """Predict attack probability, strictly asserting clean features and mapping classes via model.classes_."""
    assert_clean_features(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probs = model.predict_proba(X)
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    classes = getattr(clf, "classes_", None)
    if classes is not None and len(classes) == 2:
        pos_idx = np.where((classes == 1) | (classes == True) | (classes == "1"))[0]
        if len(pos_idx) > 0:
            return probs[:, pos_idx[0]]
    if probs.ndim == 2 and probs.shape[1] >= 2:
        return probs[:, 1]
    return probs.ravel()


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
    split_audit: dict = field(default_factory=dict)
    stream_type: str = "constructed replay stream"

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
    """Turn a labelled pool into train + stream with strict disjointness and zero target leakage.

    Pre-onset stream flows and training set contain only *seen* categories.
    Held-out families appear exclusively after onset.
    """
    clean_feats = filter_clean_features(features)
    assert_clean_features(pool[clean_feats], clean_feats)

    rng = np.random.default_rng(cfg.seed)
    notes: list[str] = []
    cat = pool["_attack_type"].astype(str).to_numpy()
    attack = np.array([not is_benign(c) for c in cat], dtype=bool)
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
            notes.append("Official file split: training flows from *train* file, stream from *test* file.")
        else:
            notes.append("Split by file requested but train/test files not found; using random disjoint split.")

    same_pool = train_pool is stream_pool
    n_u = min(len(unseen_idx), int(round(cfg.unseen_share * n_post))) if cfg.kind == "unseen" else 0
    if cfg.kind == "unseen" and n_u == 0:
        raise ValueError("None of the selected unseen categories exist in the pool - pick other categories.")
    n_seen_post = n_post - n_u
    need_stream_seen = n_pre + n_seen_post

    # Verify capacity to prevent illegal leakage
    n_train = cfg.n_train
    if same_pool:
        if len(train_pool) < n_train + need_stream_seen:
            max_train = len(train_pool) - need_stream_seen
            if max_train >= 2000:
                n_train = max_train
                notes.append(f"Training set sized to {n_train:,} to guarantee strictly disjoint stream of {need_stream_seen:,} flows.")
            else:
                raise ValueError(
                    f"Insufficient records ({len(train_pool):,}) to construct disjoint training ({n_train:,}) and stream ({need_stream_seen:,}) sets. "
                    "Please reduce training size or batch count to prevent data leakage."
                )

    tr_idx = train_pool[:n_train]
    rest = stream_pool[n_train:] if same_pool else stream_pool
    if len(rest) < need_stream_seen:
        raise ValueError(
            f"Stream requires {need_stream_seen:,} flows but only {len(rest):,} available. Cannot construct disjoint sets."
        )

    pre_idx = rest[:n_pre]
    post_seen_idx = rest[n_pre:n_pre + n_seen_post]
    post_unseen_idx = unseen_idx[:n_u]
    post = np.concatenate([post_seen_idx, post_unseen_idx])
    rng.shuffle(post)
    stream_idx = np.concatenate([pre_idx, post])

    # Hard assertions on train/stream separation
    overlap = set(tr_idx).intersection(set(stream_idx))
    assert len(overlap) == 0, f"DATA LEAKAGE: {len(overlap)} rows overlap between training and evaluation stream!"

    # Held-out family strict presence/absence assertions
    if cfg.kind == "unseen":
        assert not np.isin(cat[tr_idx], cfg.unseen).any(), "Held-out category leaked into initial training set!"
        assert not np.isin(cat[stream_idx[:n_pre]], cfg.unseen).any(), "Held-out category leaked into pre-onset stream!"
        assert np.isin(cat[stream_idx[n_pre:]], cfg.unseen).sum() > 0, "Held-out category absent from post-onset stream!"

    X_train = pool.iloc[tr_idx][clean_feats].reset_index(drop=True)
    X_stream = pool.iloc[stream_idx][clean_feats].reset_index(drop=True).astype(float)
    y_train = attack[tr_idx]
    y_stream = attack[stream_idx]
    cat_stream = cat[stream_idx]

    # Compute row fingerprints (predictor columns only) to audit identical observations
    fps = pd.util.hash_pandas_object(pool[clean_feats], index=False).to_numpy()
    tr_fps = set(fps[tr_idx])
    stream_fps = set(fps[stream_idx])
    fingerprint_overlap = len(tr_fps.intersection(stream_fps))

    # Audit distributions
    train_dist = {str(k): int(v) for k, v in pd.Series(cat[tr_idx]).value_counts().items()}
    pre_dist = {str(k): int(v) for k, v in pd.Series(cat[stream_idx[:n_pre]]).value_counts().items()}
    post_dist = {str(k): int(v) for k, v in pd.Series(cat[stream_idx[n_pre:]]).value_counts().items()}

    split_audit = {
        "n_train": len(X_train),
        "n_stream": len(X_stream),
        "n_pre_onset": n_pre,
        "n_post_onset": n_post,
        "n_unseen_post_onset": int(n_u),
        "row_index_overlap": 0,
        "feature_fingerprint_overlap": fingerprint_overlap,
        "train_class_distribution": train_dist,
        "pre_onset_distribution": pre_dist,
        "post_onset_distribution": post_dist,
        "stream_type": "constructed replay stream",
    }

    if cfg.kind == "covariate":
        feats = [f for f in cfg.shift_features if f in clean_feats] or default_shift_features(X_train)
        sd = X_train[feats].std().replace(0, 1.0)
        X_stream.loc[X_stream.index >= n_pre, feats] = X_stream.loc[X_stream.index >= n_pre, feats] * cfg.shift_scale + sd * cfg.shift_sd
        notes.append(f"Covariate shift from flow {n_pre:,}: {', '.join(feats)} scaled ×{cfg.shift_scale:g} and shifted by {cfg.shift_sd:g} σ. Labels untouched.")

    if y_train.all() or not y_train.any():
        raise ValueError("The training set contains only one class (all benign or all attack) - choose different unseen categories.")

    return Scenario(cfg, clean_feats, X_train, y_train, X_stream, y_stream, cat_stream, n_pre, notes, n_u, split_audit)


# ----------------------------------------------------------------------------------------------- #
# Metrics
# ----------------------------------------------------------------------------------------------- #
def binary_metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    """Compute complete classification metrics. Returns N/A (None) on empty inputs, never 100%."""
    if len(y) == 0:
        return {
            "acc": None, "balanced_acc": None, "f1": None, "recall": None, "precision": None,
            "fpr": None, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "n": 0, "correct": 0, "errors": 0,
        }
    y = np.asarray(y, dtype=bool)
    yhat = np.asarray(yhat, dtype=bool)
    tp = int((y & yhat).sum())
    fp = int((~y & yhat).sum())
    fn = int((y & ~yhat).sum())
    tn = int((~y & ~yhat).sum())
    n = len(y)
    correct = tp + tn
    errors = fp + fn

    acc = correct / n
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    fpr = fp / (fp + tn) if (fp + tn) > 0 else None
    tnr = tn / (tn + fp) if (tn + fp) > 0 else None
    balanced_acc = ((recall or 0.0) + (tnr or 0.0)) / 2 if (recall is not None and tnr is not None) else acc
    f1 = (2 * precision * recall) / (precision + recall) if (precision and recall and (precision + recall) > 0) else 0.0

    return {
        "acc": float(acc),
        "balanced_acc": float(balanced_acc) if balanced_acc is not None else None,
        "f1": float(f1) if f1 is not None else None,
        "recall": float(recall) if recall is not None else None,
        "precision": float(precision) if precision is not None else None,
        "fpr": float(fpr) if fpr is not None else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n": n, "correct": correct, "errors": errors,
    }


def batch_accuracy(y: np.ndarray, yhat: np.ndarray, bs: int) -> np.ndarray:
    """Calculate per-batch accuracy in range [0, 1]."""
    n = len(y) // bs
    return np.array([(y[i * bs:(i + 1) * bs] == yhat[i * bs:(i + 1) * bs]).mean() for i in range(n)])


def category_rates(cat: np.ndarray, y: np.ndarray, yhat: np.ndarray, start: int) -> pd.DataFrame:
    """Post-onset detection rate per true category (specificity for benign traffic)."""
    rows = []
    for c in pd.unique(cat[start:]):
        m = cat[start:] == c
        yy, pp = y[start:][m], yhat[start:][m]
        flows = int(m.sum())
        rate = float((pp == yy).mean()) if flows > 0 else None
        rows.append({
            "category": c,
            "flows": flows,
            "rate": rate,
            "kind": "specificity" if not yy.any() else "detection rate",
        })
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
    convergence_warnings: list[str] = field(default_factory=list)

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
        parts_X.append(sc.X_stream.iloc[ridx])
        parts_y.append(sc.y_stream[ridx])
    k = min(anchor, len(sc.X_train))
    aidx = rng.choice(len(sc.X_train), size=k, replace=False)
    parts_X.append(sc.X_train.iloc[aidx])
    parts_y.append(sc.y_train[aidx])
    Xs = pd.concat(parts_X, ignore_index=True)
    ys = np.concatenate(parts_y)
    assert_clean_features(Xs, sc.features)
    return Xs, ys


def run_policy(
    sc: Scenario, f0, policy: str, delta: float = ADWIN_DELTA, seed: int = 42,
    recent: int = RECENT, replay: int = REPLAY, anchor: int = ANCHOR
) -> PolicyResult:
    """Prequential evaluation of one update policy (paper Algorithm 1 for 'Drift-triggered').

    Enforces strict test-then-train: batch b is scored BEFORE any retraining.
    Retraining only affects predictions for batches > b.
    Static policy NEVER updates.
    """
    from river import drift as rdrift

    bs, nb = sc.cfg.batch_size, sc.cfg.n_batches
    rng = np.random.default_rng(seed)
    # Clone estimator so policies do not share mutable state
    f = clone(f0)
    # Refit cloned initial model on initial training set
    f, initial_warns = _fit(f0, sc.X_train, sc.y_train)

    yhat = np.zeros(sc.n_flows, dtype=bool)
    proba = np.zeros(sc.n_flows, dtype=float)
    res = PolicyResult(policy, yhat, proba)
    res.convergence_warnings.extend(initial_warns)

    adwin = rdrift.ADWIN(delta=delta)
    cooldown = 0
    t_run = time.perf_counter()

    for b in range(nb):
        lo, hi = b * bs, (b + 1) * bs
        Xb, yb = sc.X_stream.iloc[lo:hi], sc.y_stream[lo:hi]

        # 1) TEST FIRST (strict test-then-train order)
        p = _proba(f, Xb)
        proba[lo:hi] = p
        yhat[lo:hi] = p >= 0.5

        # 2) Reveal true labels for drift detection / adaptation
        alert = False
        if policy == "Drift-triggered":
            err = (yhat[lo:hi] != yb).astype(int)
            for i, e in enumerate(err):
                prev = adwin.estimation
                adwin.update(int(e))
                if adwin.drift_detected:
                    rose = adwin.estimation > prev
                    counted = rose and cooldown == 0
                    res.alerts.append({
                        "flow": lo + i + 1, "batch": b + 1, "error_before": float(prev),
                        "error_after": float(adwin.estimation), "rose": bool(rose),
                        "in_cooldown": cooldown > 0, "triggered_retrain": bool(counted),
                    })
                    alert = alert or counted
            cooldown = max(0, cooldown - 1)

        retrain = alert or (policy == "Periodic" and (b + 1) % PERIODIC_EVERY == 0)
        # Static policy NEVER retrains
        if policy == "Static":
            retrain = False

        if retrain:
            # 3) ADAPT FOR FUTURE BATCHES ONLY
            t0 = time.perf_counter()
            Xs, ys = _retrain_set(sc, hi, rng, recent, replay, anchor)
            f, retrain_warns = _fit(f0, Xs, ys)
            res.convergence_warnings.extend(retrain_warns)
            res.retrain_seconds += time.perf_counter() - t0
            res.retrain_batches.append(b + 1)
            if policy == "Drift-triggered":
                adwin = rdrift.ADWIN(delta=delta)
                cooldown = 1

    res.runtime_seconds = time.perf_counter() - t_run
    res.final_model = f
    return res


def run_online_hat(sc: Scenario, seed: int = 42, progress: Callable[[float], None] | None = None) -> PolicyResult:
    """River Hoeffding Adaptive Tree pre-trained on D0, then prequential test-then-train per flow."""
    from river import compose, preprocessing, tree

    cols = list(sc.features)
    model = compose.Pipeline(
        preprocessing.StandardScaler(),
        tree.HoeffdingAdaptiveTreeClassifier(grace_period=100, max_depth=12, seed=seed)
    )
    t0 = time.perf_counter()
    Xt = sc.X_train.to_numpy(float)
    for i in range(len(Xt)):
        model.learn_one(dict(zip(cols, Xt[i])), bool(sc.y_train[i]))

    Xs = sc.X_stream.to_numpy(float)
    yhat = np.zeros(len(Xs), dtype=bool)
    for i in range(len(Xs)):
        x = dict(zip(cols, Xs[i]))
        # Test first, then train
        pred = model.predict_one(x)
        yhat[i] = bool(pred or False)
        model.learn_one(x, bool(sc.y_stream[i]))
        if progress and i % 2000 == 0:
            progress(i / len(Xs))

    return PolicyResult("Online HAT", yhat, None, [], 0.0, [], time.perf_counter() - t0, model)


def fit_initial(sc: Scenario, model_name: str, seed: int = 42):
    """Fit the clean initial base model on the initial training set D0."""
    t0 = time.perf_counter()
    f0, warns = _fit(make_model(model_name, seed=seed), sc.X_train, sc.y_train)
    return f0, time.perf_counter() - t0


def summarise(sc: Scenario, results: dict[str, PolicyResult], run_info: dict | None = None) -> pd.DataFrame:
    """Compute full paper Table IV metrics for all evaluated policies."""
    rows = []
    for name, r in results.items():
        m = binary_metrics(sc.y_stream, r.yhat)
        post = slice(sc.onset_flow, None)
        m_post = binary_metrics(sc.y_stream[post], r.yhat[post])

        # Held-out family recall
        held_out_rec = None
        if sc.cfg.kind == "unseen" and sc.cfg.unseen:
            unseen_mask = np.isin(sc.cat_stream[post], sc.cfg.unseen)
            if unseen_mask.any():
                held_out_rec = float((r.yhat[post][unseen_mask] == True).mean())

        delay = None
        if r.alerts:
            first = [a["flow"] for a in r.alerts if a["triggered_retrain"] and a["flow"] > sc.onset_flow]
            delay = (min(first) - sc.onset_flow) if first else None

        rows.append({
            "Policy": name,
            "Acc (%)": 100 * m["acc"] if m["acc"] is not None else None,
            "Balanced Acc (%)": 100 * m["balanced_acc"] if m["balanced_acc"] is not None else None,
            "F1": m["f1"],
            "Rec (%)": 100 * m["recall"] if m["recall"] is not None else None,
            "FPR (%)": 100 * m["fpr"] if m["fpr"] is not None else None,
            "Post-onset acc (%)": 100 * m_post["acc"] if m_post["acc"] is not None else None,
            "Held-out Rec (%)": 100 * held_out_rec if held_out_rec is not None else None,
            "Evaluated": m["n"],
            "Correct": m["correct"],
            "Errors": m["errors"],
            "Retrains": r.n_retrains,
            "Retrain time (s)": None if name == "Online HAT" else r.retrain_seconds,
            "Delay (flows)": delay if name == "Drift-triggered" else None,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- #
# Threshold study (paper Section IV-F)
# ----------------------------------------------------------------------------------------------- #
def threshold_study(sc: Scenario, proba: np.ndarray, tune_batches: int = 5) -> dict | None:
    """Re-tune the decision threshold on the first post-onset batches, evaluate on the rest."""
    if proba is None:
        return None
    start = sc.onset_flow
    cut = start + tune_batches * sc.cfg.batch_size
    if cut >= sc.n_flows - sc.cfg.batch_size:
        return None
    yt, pt = sc.y_stream[start:cut], proba[start:cut]
    ye, pe = sc.y_stream[cut:], proba[cut:]
    if not yt.any() or yt.all() or not ye.any() or ye.all():
        return None
    grid = np.linspace(0.02, 0.98, 97)
    f1s = [binary_metrics(yt, pt >= t)["f1"] or 0.0 for t in grid]
    t_star = float(grid[int(np.argmax(f1s))])
    fpr, tpr, _ = roc_curve(ye, pe)
    return {
        "threshold": t_star,
        "default": binary_metrics(ye, pe >= 0.5),
        "tuned": binary_metrics(ye, pe >= t_star),
        "grid": grid,
        "eval_curve": [binary_metrics(ye, pe >= t) for t in grid],
        "roc": (fpr, tpr),
        "tune_flows": int(cut - start),
        "eval_flows": int(len(ye)),
    }


# ----------------------------------------------------------------------------------------------- #
# Auditable Exports
# ----------------------------------------------------------------------------------------------- #
def export_drift_lab_reports(
    sc: Scenario, results: dict[str, PolicyResult], run_info: dict, out_dir: str = "reports/drift_lab"
) -> dict[str, str]:
    """Export all 9 required auditable artifacts to disk."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    bs = sc.cfg.batch_size
    exported_files = {}

    # 1. run_config.json
    run_cfg_path = out_path / "run_config.json"
    cfg_data = {
        "run_id": run_info.get("run_id", str(uuid.uuid4())[:8]),
        "timestamp": run_info.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
        "model": run_info.get("selected_model_name"),
        "estimator_class": run_info.get("actual_estimator_class"),
        "hyperparameters": run_info.get("hyperparameters"),
        "random_seed": run_info.get("random_seed"),
        "dataset": run_info.get("dataset"),
        "scenario_kind": sc.cfg.kind,
        "unseen_categories": sc.cfg.unseen,
        "shift_features": sc.cfg.shift_features,
        "n_train": sc.cfg.n_train,
        "n_batches": sc.cfg.n_batches,
        "batch_size": sc.cfg.batch_size,
        "onset_batch": sc.cfg.onset_batch,
        "adwin_delta": run_info.get("adwin_delta", ADWIN_DELTA),
        "stream_type": sc.stream_type,
    }
    run_cfg_path.write_text(json.dumps(cfg_data, indent=2), encoding="utf-8")
    exported_files["run_config"] = str(run_cfg_path)

    # 2. feature_audit.json
    feat_path = out_path / "feature_audit.json"
    feat_data = {
        "feature_count": len(sc.features),
        "features": sc.features,
        "prohibited_features_checked": True,
        "prohibited_features_found": [c for c in sc.features if is_prohibited_feature(c)],
        "target_leakage_prevented": True,
    }
    feat_path.write_text(json.dumps(feat_data, indent=2), encoding="utf-8")
    exported_files["feature_audit"] = str(feat_path)

    # 3. split_audit.json
    split_path = out_path / "split_audit.json"
    split_path.write_text(json.dumps(sc.split_audit, indent=2), encoding="utf-8")
    exported_files["split_audit"] = str(split_path)

    # 4. policy_summary.csv
    summ = summarise(sc, results, run_info)
    summ_path = out_path / "policy_summary.csv"
    summ.to_csv(summ_path, index=False)
    exported_files["policy_summary"] = str(summ_path)

    # 5. batch_metrics.csv
    batch_rows = []
    for pol_name, r in results.items():
        b_acc = batch_accuracy(sc.y_stream, r.yhat, bs)
        for b_idx, acc_val in enumerate(b_acc):
            lo, hi = b_idx * bs, (b_idx + 1) * bs
            y_b = sc.y_stream[lo:hi]
            yh_b = r.yhat[lo:hi]
            corr_b = int((y_b == yh_b).sum())
            batch_rows.append({
                "policy": pol_name,
                "batch": b_idx + 1,
                "evaluated": len(y_b),
                "correct": corr_b,
                "errors": len(y_b) - corr_b,
                "batch_accuracy": float(acc_val),
                "retrained": (b_idx + 1) in r.retrain_batches,
                "post_onset": (b_idx + 1) > sc.cfg.onset_batch,
            })
    b_df = pd.DataFrame(batch_rows)
    b_path = out_path / "batch_metrics.csv"
    b_df.to_csv(b_path, index=False)
    exported_files["batch_metrics"] = str(b_path)

    # 6. per_family_metrics.csv
    fam_rows = []
    for pol_name in ("Static", "Drift-triggered", "Periodic"):
        if pol_name in results:
            rates = category_rates(sc.cat_stream, sc.y_stream, results[pol_name].yhat, sc.onset_flow)
            for _, row in rates.iterrows():
                fam_rows.append({
                    "policy": pol_name,
                    "category": row["category"],
                    "kind": row["kind"],
                    "flows": row["flows"],
                    "score": row["rate"],
                })
    f_df = pd.DataFrame(fam_rows)
    f_path = out_path / "per_family_metrics.csv"
    f_df.to_csv(f_path, index=False)
    exported_files["per_family_metrics"] = str(f_path)

    # 7. retraining_events.csv
    retrain_rows = []
    for pol_name, r in results.items():
        for alert in r.alerts:
            retrain_rows.append({"policy": pol_name, **alert})
    r_df = pd.DataFrame(retrain_rows)
    r_path = out_path / "retraining_events.csv"
    r_df.to_csv(r_path, index=False)
    exported_files["retraining_events"] = str(r_path)

    # 8. predictions.csv (sample first 2,000 flows and onset window for auditable inspection)
    sample_indices = np.concatenate([
        np.arange(min(1000, sc.n_flows)),
        np.arange(max(0, sc.onset_flow - 200), min(sc.n_flows, sc.onset_flow + 800)),
    ])
    sample_indices = np.unique(sample_indices)
    pred_dict = {
        "flow_index": sample_indices,
        "batch_index": sample_indices // bs + 1,
        "ground_truth": sc.y_stream[sample_indices].astype(int),
        "attack_category": sc.cat_stream[sample_indices],
    }
    for pol_name, r in results.items():
        pred_dict[f"{pol_name}_pred"] = r.yhat[sample_indices].astype(int)
        if r.proba is not None:
            pred_dict[f"{pol_name}_proba"] = np.round(r.proba[sample_indices], 4)
    pred_df = pd.DataFrame(pred_dict)
    pred_path = out_path / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    exported_files["predictions"] = str(pred_path)

    # 9. model_comparison.csv row export
    comp_rows = []
    for pol_name, r in results.items():
        m = binary_metrics(sc.y_stream, r.yhat)
        post = slice(sc.onset_flow, None)
        held_out_rec = None
        if sc.cfg.kind == "unseen" and sc.cfg.unseen:
            unseen_mask = np.isin(sc.cat_stream[post], sc.cfg.unseen)
            if unseen_mask.any():
                held_out_rec = float((r.yhat[post][unseen_mask] == True).mean())
        comp_rows.append({
            "dataset": run_info.get("dataset"),
            "task": "binary",
            "scenario": sc.cfg.kind,
            "model": run_info.get("selected_model_name"),
            "estimator_class": run_info.get("actual_estimator_class"),
            "policy": pol_name,
            "seed": sc.cfg.seed,
            "n_train": len(sc.X_train),
            "n_evaluated": m["n"],
            "correct": m["correct"],
            "errors": m["errors"],
            "accuracy": m["acc"],
            "balanced_accuracy": m["balanced_acc"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "false_positive_rate": m["fpr"],
            "held_out_family_recall": held_out_rec,
            "retrain_count": r.n_retrains,
            "run_id": run_info.get("run_id"),
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_path = out_path / "model_comparison.csv"
    if comp_path.exists():
        try:
            prev_df = pd.read_csv(comp_path)
            comp_df = pd.concat([prev_df, comp_df], ignore_index=True)
        except Exception:
            pass
    comp_df.to_csv(comp_path, index=False)
    exported_files["model_comparison"] = str(comp_path)

    return exported_files
