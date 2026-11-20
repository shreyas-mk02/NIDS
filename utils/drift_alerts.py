"""Drift Intelligence System for Network Intrusion Detection.
Provides evidence-based feature-level data drift (PSI, KS-test, Wasserstein)
and prediction-level concept drift (ADWIN, attack-rate shift, class distribution,
confidence divergence), along with rich cybersecurity visual maps and modal alerts.
"""
from __future__ import annotations

import datetime
import io
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from river import drift

# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class FeatureDrift:
    name: str
    psi: float
    ks_stat: float
    ks_pvalue: float
    wasserstein: float
    severity: str  # "Stable" | "Low" | "Moderate" | "High"
    ref_mean: float
    curr_mean: float
    ref_std: float
    curr_std: float
    mean_shift: float
    importance: float = 0.0


@dataclass
class DriftAnalysisResult:
    status: str            # "STABLE" | "LOW_DRIFT" | "MODERATE_DRIFT" | "CRITICAL_DRIFT"
    severity_label: str    # Human readable label
    overall_score: float   # Normalized 0.0 - 1.0 composite drift score
    color: str             # Visual status color (#34d399, #fbbf24, #f97316, #f43f5e)
    n_features_evaluated: int
    n_drifted_features: int
    drifted_fraction: float
    top_drifted: list[FeatureDrift]
    feature_table: pd.DataFrame
    prediction_drift: dict[str, Any]
    confidence_drift: dict[str, Any]
    adwin_alert: bool
    adwin_alerts_count: int
    timestamp: str
    batch_size: int
    reference_name: str
    recommended_action: str
    interpretation: str
    feature_importances: dict[str, float] = field(default_factory=dict)
    has_reference: bool = True
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Reference Baseline Cache
# --------------------------------------------------------------------------- #
_CACHED_BASELINE: pd.DataFrame | None = None
_BASELINE_SOURCE_NAME: str = "CIC-IDS2017 Cleaned Baseline"


def get_reference_baseline(max_rows: int = 5000, seed: int = 42) -> tuple[pd.DataFrame | None, str]:
    """Retrieve or sample a stable reference dataset for baseline comparison."""
    global _CACHED_BASELINE, _BASELINE_SOURCE_NAME
    if _CACHED_BASELINE is not None and len(_CACHED_BASELINE) > 0:
        return _CACHED_BASELINE, _BASELINE_SOURCE_NAME

    candidates = [
        Path("data/cicids2017_cleaned.csv"),
        Path("data/unsw_nb15_cleaned_train.csv"),
    ]
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p, nrows=max_rows * 2, low_memory=False)
                if len(df) > max_rows:
                    df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
                _CACHED_BASELINE = df
                _BASELINE_SOURCE_NAME = f"{p.stem} (sample N={len(df):,})"
                return _CACHED_BASELINE, _BASELINE_SOURCE_NAME
            except Exception:
                continue

    return None, "Baseline unavailable"


# --------------------------------------------------------------------------- #
# Statistical Drift Computation
# --------------------------------------------------------------------------- #
def calculate_psi(ref: np.ndarray, curr: np.ndarray, bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) between reference and current samples."""
    ref_clean = ref[np.isfinite(ref)]
    curr_clean = curr[np.isfinite(curr)]

    if len(ref_clean) < 10 or len(curr_clean) < 5:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.percentile(ref_clean, quantiles * 100)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    # Expand outer boundaries to cover all current data
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    r_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    c_counts, _ = np.histogram(curr_clean, bins=bin_edges)

    # Laplace / epsilon smoothing to prevent div by zero
    eps = 1e-4
    r_pct = np.maximum(r_counts / max(len(ref_clean), 1), eps)
    c_pct = np.maximum(c_counts / max(len(curr_clean), 1), eps)
    r_pct /= r_pct.sum()
    c_pct /= c_pct.sum()

    psi = np.sum((c_pct - r_pct) * np.log(c_pct / r_pct))
    return float(max(0.0, psi))


def classify_psi_severity(psi: float) -> str:
    """Classify PSI into established risk tiers."""
    if psi < 0.10:
        return "Stable"
    elif psi < 0.20:
        return "Low"
    elif psi < 0.30:
        return "Moderate"
    else:
        return "High"


def evaluate_feature_drift(
    ref_vals: np.ndarray,
    curr_vals: np.ndarray,
    feat_name: str,
    importance: float = 0.0
) -> FeatureDrift:
    """Run full battery of statistical tests for a single continuous feature."""
    r = ref_vals[np.isfinite(ref_vals)]
    c = curr_vals[np.isfinite(curr_vals)]

    if len(r) == 0 or len(c) == 0:
        return FeatureDrift(
            name=feat_name, psi=0.0, ks_stat=0.0, ks_pvalue=1.0,
            wasserstein=0.0, severity="Stable", ref_mean=0.0, curr_mean=0.0,
            ref_std=0.0, curr_std=0.0, mean_shift=0.0, importance=importance
        )

    r_mean, r_std = float(np.mean(r)), float(np.std(r))
    c_mean, c_std = float(np.mean(c)), float(np.std(c))

    # Standardized Wasserstein distance
    scale = r_std if r_std > 1e-6 else 1.0
    r_scaled = (r - r_mean) / scale
    c_scaled = (c - r_mean) / scale
    try:
        wass = float(stats.wasserstein_distance(r_scaled, c_scaled))
    except Exception:
        wass = 0.0

    # KS-Test
    try:
        ks_res = stats.ks_2samp(r, c)
        ks_stat, ks_pval = float(ks_res.statistic), float(ks_res.pvalue)
    except Exception:
        ks_stat, ks_pval = 0.0, 1.0

    # PSI
    psi = calculate_psi(r, c)
    sev = classify_psi_severity(psi)

    shift = (c_mean - r_mean) / scale

    return FeatureDrift(
        name=feat_name,
        psi=psi,
        ks_stat=ks_stat,
        ks_pvalue=ks_pval,
        wasserstein=wass,
        severity=sev,
        ref_mean=r_mean,
        curr_mean=c_mean,
        ref_std=r_std,
        curr_std=c_std,
        mean_shift=shift,
        importance=importance,
    )


# --------------------------------------------------------------------------- #
# Main Analysis Pipeline
# --------------------------------------------------------------------------- #
def analyze_batch_drift(
    current_X: pd.DataFrame,
    prediction_table: pd.DataFrame,
    model: Any = None,
    feature_names: list[str] | None = None,
    reference_data: pd.DataFrame | None = None,
) -> DriftAnalysisResult:
    """Analyze the newly generated prediction batch against the historical baseline."""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_records = len(current_X)

    if feature_names is None:
        feature_names = [c for c in current_X.columns if pd.api.types.is_numeric_dtype(current_X[c])]

    ref_df, ref_name = (reference_data, "Custom Reference") if reference_data is not None else get_reference_baseline()
    has_ref = ref_df is not None and len(ref_df) > 0

    importances: dict[str, float] = {}
    if model is not None:
        raw_imp = getattr(model, "feature_importances_", None)
        if raw_imp is not None and len(raw_imp) == len(feature_names):
            importances = dict(zip(feature_names, [float(x) for x in raw_imp]))

    feature_drifts: list[FeatureDrift] = []
    if has_ref:
        for f in feature_names:
            if f in ref_df.columns and f in current_X.columns:
                r_arr = pd.to_numeric(ref_df[f], errors="coerce").dropna().to_numpy()
                c_arr = pd.to_numeric(current_X[f], errors="coerce").dropna().to_numpy()
                imp = importances.get(f, 0.0)
                fd = evaluate_feature_drift(r_arr, c_arr, f, importance=imp)
                feature_drifts.append(fd)

    # Sort descending by PSI
    feature_drifts.sort(key=lambda d: d.psi, reverse=True)
    n_drifted = sum(1 for d in feature_drifts if d.psi >= 0.10)
    frac_drifted = n_drifted / max(len(feature_drifts), 1)

    # Prediction level metrics
    pred_summary: dict[str, Any] = {}
    conf_summary: dict[str, Any] = {}
    adwin_alerts = 0

    if "is_attack" in prediction_table.columns or "final_verdict" in prediction_table.columns:
        if "is_attack" in prediction_table.columns:
            atk_flags = prediction_table["is_attack"].astype(bool).to_numpy()
        else:
            atk_flags = (prediction_table["final_verdict"] != "Normal").to_numpy()

        batch_atk_rate = float(np.mean(atk_flags))
        ref_atk_rate = 0.318  # Known CIC-IDS2017 attack proportion
        if has_ref and "Label" in ref_df.columns:
            ref_atk_rate = float((ref_df["Label"].astype(str).str.strip().str.upper() != "BENIGN").mean())

        atk_delta = abs(batch_atk_rate - ref_atk_rate)
        pred_summary["batch_attack_rate"] = batch_atk_rate
        pred_summary["reference_attack_rate"] = ref_atk_rate
        pred_summary["attack_rate_delta"] = atk_delta

        # Class distribution
        if "final_verdict" in prediction_table.columns:
            vc = prediction_table["final_verdict"].value_counts(normalize=True).to_dict()
            pred_summary["batch_categories"] = vc

        # Confidence analysis
        conf_col = next((c for c in ("attack_probability", "binary_confidence", "multiclass_confidence") if c in prediction_table.columns), None)
        if conf_col is not None:
            c_vals = pd.to_numeric(prediction_table[conf_col], errors="coerce").dropna().to_numpy()
            batch_conf = float(np.mean(c_vals)) if len(c_vals) > 0 else 0.95
            ref_conf = 0.985
            conf_summary["batch_confidence"] = batch_conf
            conf_summary["reference_confidence"] = ref_conf
            conf_summary["confidence_delta"] = float(ref_conf - batch_conf)

        # ADWIN change detection stream
        adwin = drift.ADWIN(delta=0.002)
        has_true_label = "original_label" in prediction_table.columns
        for idx in range(len(prediction_table)):
            if has_true_label:
                true_lbl = str(prediction_table["original_label"].iloc[idx]).strip().upper()
                is_true_atk = true_lbl not in ("BENIGN", "NORMAL", "0")
                is_pred_atk = bool(atk_flags[idx])
                err = int(is_true_atk != is_pred_atk)
                adwin.update(err)
            else:
                # Without ground truth, monitor prediction uncertainty (or high anomaly flag)
                conf = float(prediction_table["binary_confidence"].iloc[idx]) if "binary_confidence" in prediction_table.columns else 0.95
                uncertainty = 1.0 - conf
                adwin.update(uncertainty)

            if adwin.drift_detected:
                adwin_alerts += 1

    # Composite Drift Score (0.0 to 1.0)
    top5_psi = float(np.mean([d.psi for d in feature_drifts[:5]])) if feature_drifts else 0.0
    norm_top5 = min(1.0, top5_psi / 0.50)
    norm_frac = frac_drifted
    norm_pred = min(1.0, pred_summary.get("attack_rate_delta", 0.0) * 2.0)
    norm_adwin = 1.0 if adwin_alerts > 0 else 0.0

    composite_score = (
        0.40 * norm_top5 +
        0.25 * norm_frac +
        0.20 * norm_pred +
        0.15 * norm_adwin
    )
    composite_score = float(np.clip(composite_score, 0.0, 1.0))

    # Categorize overall status
    if not has_ref:
        status = "STABLE"
        sev_label = "Reference baseline unavailable"
        color = "#8fa3c0"
        rec = "Attach or place historical reference datasets in data/ to enable baseline drift comparison."
        interp = "Reference baseline unavailable; drift assessment limited to internal consistency."
    elif composite_score < 0.15:
        status = "STABLE"
        sev_label = "Stable (No Significant Drift)"
        color = "#34d399"  # Emerald
        rec = "Traffic profile aligns with historical baseline. Model predictions are statistically reliable. Continue routine monitoring."
        interp = f"Analyzed {len(feature_drifts)} flow features across {n_records:,} records. Feature distributions and prediction rates remain stable within baseline tolerances."
    elif composite_score < 0.30:
        status = "LOW_DRIFT"
        sev_label = "Low Drift Detected"
        color = "#fbbf24"  # Amber
        rec = "Minor distributional variance observed in peripheral traffic features. No immediate retraining needed; monitor subsequent batches."
        interp = f"Low drift detected: {n_drifted} feature(s) showed mild distributional divergence (PSI >= 0.10). Core attack patterns remain consistent."
    elif composite_score < 0.50:
        status = "MODERATE_DRIFT"
        sev_label = "Moderate Drift Warning"
        color = "#f97316"  # Orange
        rec = "Notable feature divergence and prediction shift detected. Evaluate top drifted features against recent attack signatures and prepare candidate retrain data."
        interp = f"Moderate drift detected: {n_drifted} feature(s) shifted significantly (top PSI: {top5_psi:.2f}). Attack rate shifted by {pred_summary.get('attack_rate_delta', 0.0):.1%}."
    else:
        status = "CRITICAL_DRIFT"
        sev_label = "Critical / High Drift Alert"
        color = "#f43f5e"  # Crimson
        rec = "Critical distributional shift confirmed across key flow features. Model predictions may suffer degraded fidelity. Recommend executing adaptive retraining on recent traffic."
        interp = f"High/Critical concept and data drift detected: {n_drifted} features drifted (composite score: {composite_score:.2f}). ADWIN triggered {adwin_alerts} alert(s)."

    # Build DataFrame
    tbl_rows = []
    for fd in feature_drifts:
        tbl_rows.append({
            "Feature": fd.name,
            "PSI Score": round(fd.psi, 4),
            "Severity": fd.severity,
            "KS Statistic": round(fd.ks_stat, 4),
            "p-value": f"{fd.ks_pvalue:.2e}" if fd.ks_pvalue < 0.001 else round(fd.ks_pvalue, 4),
            "Wasserstein": round(fd.wasserstein, 4),
            "Relative Mean Shift": round(fd.mean_shift, 4),
            "Reference Mean": round(fd.ref_mean, 2),
            "Batch Mean": round(fd.curr_mean, 2),
            "Model Importance": round(fd.importance, 4),
        })
    df_features = pd.DataFrame(tbl_rows)

    return DriftAnalysisResult(
        status=status,
        severity_label=sev_label,
        overall_score=composite_score,
        color=color,
        n_features_evaluated=len(feature_drifts),
        n_drifted_features=n_drifted,
        drifted_fraction=frac_drifted,
        top_drifted=feature_drifts[:8],
        feature_table=df_features,
        prediction_drift=pred_summary,
        confidence_drift=conf_summary,
        adwin_alert=(adwin_alerts > 0),
        adwin_alerts_count=adwin_alerts,
        timestamp=now_str,
        batch_size=n_records,
        reference_name=ref_name,
        recommended_action=rec,
        interpretation=interp,
        feature_importances=importances,
        has_reference=has_ref,
    )


# --------------------------------------------------------------------------- #
# Visualizations
# --------------------------------------------------------------------------- #
_PLOT_THEME = dict(
    paper_bgcolor="rgba(10, 17, 40, 0.6)",
    plot_bgcolor="rgba(5, 8, 22, 0.8)",
    font=dict(color="#e6edf7", family="Inter, sans-serif"),
    margin=dict(l=30, r=30, t=40, b=30),
)


def plot_drift_radar(res: DriftAnalysisResult) -> go.Figure:
    """Multi-axis radar chart showing normalized drift dimensions."""
    p = res.prediction_drift
    c = res.confidence_drift

    categories = [
        "Feature Drift (PSI)",
        "Drifted Features %",
        "Attack Rate Shift",
        "Confidence Drop",
        "KS Severity",
        "ADWIN Alert Intensity"
    ]
    top_psi = res.top_drifted[0].psi if res.top_drifted else 0.0
    c_delta = max(0.0, c.get("confidence_delta", 0.0))
    ks_max = max([d.ks_stat for d in res.top_drifted], default=0.0)

    values = [
        min(1.0, top_psi / 0.5),
        res.drifted_fraction,
        min(1.0, p.get("attack_rate_delta", 0.0) * 2.5),
        min(1.0, c_delta * 5.0),
        min(1.0, ks_max),
        1.0 if res.adwin_alert else 0.0,
    ]
    # Close polygon
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(34, 211, 238, 0.25)",
        line=dict(color="#22d3ee", width=2.5),
        name="Observed Drift",
        hovertemplate="%{theta}: <b>%{r:.2f}</b><extra></extra>",
    ))
    # Reference safe boundary (0.2)
    fig.add_trace(go.Scatterpolar(
        r=[0.2] * len(categories_closed),
        theta=categories_closed,
        line=dict(color="rgba(52, 211, 153, 0.6)", dash="dot", width=1.5),
        name="Tolerance Threshold",
        hoverinfo="skip"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.0], color="#8fa3c0", gridcolor="rgba(255, 255, 255, 0.08)"),
            angularaxis=dict(color="#e6edf7", gridcolor="rgba(255, 255, 255, 0.08)"),
            bgcolor="rgba(10, 17, 40, 0.5)"
        ),
        title=dict(text="Drift Dimension Radar Profile", font=dict(size=14, color="#e6edf7")),
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, font=dict(size=11)),
        height=340,
        **_PLOT_THEME,
    )
    return fig


def plot_drift_heatmap(res: DriftAnalysisResult, top_n: int = 12) -> go.Figure:
    """Interactive heatmap showing top drifted features across all statistical metrics."""
    top_feats = res.top_drifted[:top_n]
    if not top_feats:
        return go.Figure()

    feat_names = [f.name for f in reversed(top_feats)]
    metrics = ["PSI Index", "KS Statistic", "Wasserstein", "Relative Shift"]

    # Matrix normalized [0, 1] for heatmap coloring
    z_matrix = []
    text_matrix = []
    for f in reversed(top_feats):
        z_row = [
            min(1.0, f.psi / 0.4),
            min(1.0, f.ks_stat),
            min(1.0, f.wasserstein / 2.0),
            min(1.0, abs(f.mean_shift) / 2.0),
        ]
        t_row = [
            f"PSI: {f.psi:.3f} ({f.severity})",
            f"KS: {f.ks_stat:.3f} (p={f.ks_pvalue:.1e})",
            f"Wass: {f.wasserstein:.3f}",
            f"Shift: {f.mean_shift:+.2f}σ",
        ]
        z_matrix.append(z_row)
        text_matrix.append(t_row)

    colorscale = [
        [0.0, "#0a1128"],
        [0.2, "#134e4a"],
        [0.5, "#d97706"],
        [0.8, "#ea580c"],
        [1.0, "#f43f5e"]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=metrics,
        y=feat_names,
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=10, color="#ffffff"),
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(title=dict(text="Intensity", font=dict(color="#8fa3c0", size=10)), tickfont=dict(color="#8fa3c0", size=9)),
        hovertemplate="Feature: <b>%{y}</b><br>Metric: <b>%{x}</b><br>%{text}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text=f"Top {len(top_feats)} Drifted Flow Indicators Heatmap", font=dict(size=14, color="#e6edf7")),
        height=max(320, 25 * len(top_feats)),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        **_PLOT_THEME,
    )
    return fig


def plot_drift_galaxy(res: DriftAnalysisResult, current_X: pd.DataFrame, max_nodes: int = 22) -> go.Figure:
    """Traffic Drift Galaxy / Constellation Map (conceptual network graph)."""
    top_feats = res.top_drifted[:max_nodes]
    if not top_feats:
        return go.Figure()

    names = [f.name for f in top_feats]
    psi_dict = {f.name: f.psi for f in top_feats}
    sev_dict = {f.name: f.severity for f in top_feats}

    color_map = {
        "Stable": "#34d399",
        "Low": "#fbbf24",
        "Moderate": "#f97316",
        "High": "#f43f5e"
    }

    # Celestial layout coordinates
    n = len(names)
    np.random.seed(42)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Radius based on drift magnitude: higher drift placed further out from center
    radii = [0.35 + 0.65 * min(1.0, psi_dict[name] / 0.4) for name in names]
    x_coords = [radii[i] * np.cos(angles[i]) for i in range(n)]
    y_coords = [radii[i] * np.sin(angles[i]) for i in range(n)]
    sizes = [14 + 30 * min(1.0, psi_dict[name] / 0.5) for name in names]
    colors = [color_map.get(sev_dict[name], "#34d399") for name in names]

    fig = go.Figure()

    # Draw correlation constellation edges between top drifted features
    sub_df = current_X[[col for col in names if col in current_X.columns]]
    if len(sub_df) > 5 and len(sub_df.columns) > 1:
        corr = sub_df.corr().fillna(0).to_numpy()
        edge_x = []
        edge_y = []
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr[i, j]) > 0.65:
                    edge_x.extend([x_coords[i], x_coords[j], None])
                    edge_y.extend([y_coords[i], y_coords[j], None])
        if edge_x:
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode="lines",
                line=dict(color="rgba(34, 211, 238, 0.22)", width=1.2),
                hoverinfo="none",
                showlegend=False
            ))

    # Core galaxy hub
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="markers",
        marker=dict(size=18, color="#22d3ee", symbol="diamond", line=dict(color="#ffffff", width=2)),
        name="Baseline Core",
        hoverinfo="text",
        text=["Baseline Training Distribution Anchor"]
    ))

    # Feature Star Nodes
    hover_texts = [
        f"<b>{name}</b><br>Drift Tier: <b>{sev_dict[name]}</b><br>PSI: <b>{psi_dict[name]:.4f}</b>"
        for name in names
    ]
    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords,
        mode="markers+text",
        marker=dict(size=sizes, color=colors, line=dict(color="rgba(255,255,255,0.4)", width=1.5)),
        text=[n if len(n) <= 14 else n[:12] + "…" for n in names],
        textposition="top center",
        textfont=dict(size=9, color="#e6edf7"),
        hoverinfo="text",
        hovertext=hover_texts,
        name="Flow Features",
        showlegend=False
    ))

    fig.update_layout(
        title=dict(text="Traffic Drift Galaxy (Constellation View)", font=dict(size=14, color="#e6edf7")),
        xaxis=dict(visible=False, range=[-1.35, 1.35]),
        yaxis=dict(visible=False, range=[-1.35, 1.35]),
        height=360,
        **_PLOT_THEME,
    )
    return fig


def plot_drift_timeline(history: list[dict]) -> go.Figure:
    """Interactive line chart of drift scores across processed batches in this session."""
    fig = go.Figure()
    if not history:
        return fig

    batch_nums = [h.get("batch_id", i + 1) for i, h in enumerate(history)]
    scores = [h.get("overall_score", 0.0) for h in history]
    times = [h.get("timestamp", "") for h in history]
    statuses = [h.get("status", "STABLE") for h in history]

    # Threshold horizontal reference bands
    fig.add_hrect(y0=0.0, y1=0.15, fillcolor="rgba(52, 211, 153, 0.10)", line_width=0, annotation_text="Stable Zone", annotation_position="top left", annotation_font_size=9)
    fig.add_hrect(y0=0.15, y1=0.30, fillcolor="rgba(251, 191, 36, 0.10)", line_width=0, annotation_text="Low Drift", annotation_position="top left", annotation_font_size=9)
    fig.add_hrect(y0=0.30, y1=0.50, fillcolor="rgba(249, 115, 22, 0.10)", line_width=0, annotation_text="Moderate Drift", annotation_position="top left", annotation_font_size=9)
    fig.add_hrect(y0=0.50, y1=1.00, fillcolor="rgba(244, 63, 94, 0.12)", line_width=0, annotation_text="Critical Drift", annotation_position="top left", annotation_font_size=9)

    fig.add_trace(go.Scatter(
        x=batch_nums, y=scores,
        mode="lines+markers",
        line=dict(color="#22d3ee", width=2.5),
        marker=dict(size=9, color="#22d3ee", symbol="circle", line=dict(color="#ffffff", width=1.5)),
        name="Drift Score",
        customdata=list(zip(times, statuses)),
        hovertemplate="Batch: <b>#%{x}</b><br>Score: <b>%{y:.3f}</b><br>Status: <b>%{customdata[1]}</b><br>Time: %{customdata[0]}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="Session Drift Score Timeline", font=dict(size=14, color="#e6edf7")),
        xaxis=dict(title="Batch Generation Sequence", gridcolor="rgba(255,255,255,0.08)", dtick=1),
        yaxis=dict(title="Composite Drift Index", range=[0, 1.05], gridcolor="rgba(255,255,255,0.08)"),
        height=280,
        **_PLOT_THEME,
    )
    return fig


def plot_prediction_shift(res: DriftAnalysisResult) -> go.Figure:
    """Grouped bar chart comparing baseline vs current batch predicted traffic distributions."""
    fig = go.Figure()
    p = res.prediction_drift

    ref_atk = p.get("reference_attack_rate", 0.318)
    curr_atk = p.get("batch_attack_rate", 0.318)

    labels = ["Normal Traffic", "Malicious / Attack Traffic"]
    ref_vals = [1.0 - ref_atk, ref_atk]
    curr_vals = [1.0 - curr_atk, curr_atk]

    fig.add_trace(go.Bar(
        x=labels, y=ref_vals,
        name="Baseline Distribution",
        marker=dict(color="rgba(59, 130, 246, 0.75)"),
        text=[f"{v:.1%}" for v in ref_vals],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        x=labels, y=curr_vals,
        name="Current Generated Output",
        marker=dict(color="rgba(34, 211, 238, 0.85)"),
        text=[f"{v:.1%}" for v in curr_vals],
        textposition="auto"
    ))

    fig.update_layout(
        barmode="group",
        title=dict(text="Prediction Shift: Normal vs Attack Proportions", font=dict(size=14, color="#e6edf7")),
        yaxis=dict(title="Proportion of Records", tickformat=".0%", range=[0, 1.05], gridcolor="rgba(255,255,255,0.08)"),
        height=280,
        legend=dict(orientation="h", y=-0.2, font=dict(size=11)),
        **_PLOT_THEME,
    )
    return fig


def plot_feature_influence(res: DriftAnalysisResult) -> go.Figure:
    """Feature Influence Scatter Map: Drift Magnitude (PSI) vs Model Feature Importance."""
    fig = go.Figure()
    feats = res.feature_table
    if feats.empty:
        return fig

    # Quadrant threshold lines
    psi_thresh = 0.15
    imp_mean = float(feats["Model Importance"].mean()) if "Model Importance" in feats else 0.015

    # Highlight colors
    node_colors = []
    for _, row in feats.iterrows():
        p = float(row["PSI Score"])
        imp = float(row["Model Importance"])
        if p >= psi_thresh and imp >= imp_mean:
            node_colors.append("#f43f5e")  # High Drift + High Influence -> RED ALERT
        elif p >= psi_thresh:
            node_colors.append("#fbbf24")  # High Drift, Low Influence -> AMBER
        elif imp >= imp_mean:
            node_colors.append("#3b82f6")  # Low Drift, High Influence -> BLUE
        else:
            node_colors.append("#34d399")  # Low Drift, Low Influence -> GREEN

    fig.add_vline(x=psi_thresh, line=dict(color="rgba(255,255,255,0.25)", dash="dash"))
    fig.add_hline(y=imp_mean, line=dict(color="rgba(255,255,255,0.25)", dash="dash"))

    fig.add_trace(go.Scatter(
        x=feats["PSI Score"],
        y=feats["Model Importance"],
        mode="markers",
        marker=dict(size=10, color=node_colors, line=dict(color="#ffffff", width=1)),
        text=feats["Feature"],
        hovertemplate="Feature: <b>%{text}</b><br>PSI: <b>%{x:.4f}</b><br>Model Importance: <b>%{y:.4f}</b><extra></extra>",
        showlegend=False
    ))

    fig.update_layout(
        title=dict(text="Feature Influence Map (Drift Magnitude vs Model Weight)", font=dict(size=14, color="#e6edf7")),
        xaxis=dict(title="Drift Magnitude (PSI)", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="Model Importance Weight", gridcolor="rgba(255,255,255,0.08)"),
        height=300,
        annotations=[
            dict(x=max(feats["PSI Score"].max() * 0.85, psi_thresh + 0.05), y=max(feats["Model Importance"].max() * 0.85, imp_mean * 1.5),
                 text="🚨 Critical: High Drift + High Influence", showarrow=False, font=dict(color="#f43f5e", size=10)),
        ],
        **_PLOT_THEME,
    )
    return fig


# --------------------------------------------------------------------------- #
# Report Generation
# --------------------------------------------------------------------------- #
def generate_drift_report_csv(res: DriftAnalysisResult) -> str:
    """Generate standardized CSV drift audit report matching project requirements."""
    rows = []
    p = res.prediction_drift

    for _, f in res.feature_table.iterrows():
        rows.append({
            "timestamp": res.timestamp,
            "batch_size": res.batch_size,
            "overall_drift_status": res.status,
            "overall_drift_score": round(res.overall_score, 4),
            "feature_name": f["Feature"],
            "drift_metric": "PSI & KS-Test",
            "psi_value": f["PSI Score"],
            "ks_statistic": f["KS Statistic"],
            "ks_pvalue": f["p-value"],
            "wasserstein_distance": f["Wasserstein"],
            "severity_level": f["Severity"],
            "reference_mean": f["Reference Mean"],
            "current_mean": f["Batch Mean"],
            "relative_mean_shift": f["Relative Mean Shift"],
            "model_importance": f["Model Importance"],
            "batch_attack_rate": round(p.get("batch_attack_rate", 0.0), 4),
            "ref_attack_rate": round(p.get("reference_attack_rate", 0.0), 4),
            "attack_rate_delta": round(p.get("attack_rate_delta", 0.0), 4),
            "adwin_alert_triggered": res.adwin_alert,
            "adwin_alerts_count": res.adwin_alerts_count,
            "reference_baseline": res.reference_name,
            "recommended_action": res.recommended_action
        })
    df_out = pd.DataFrame(rows)
    return df_out.to_csv(index=False)
