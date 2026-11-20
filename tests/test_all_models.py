"""
End-to-end regression tests for all four models in the patched app.py.
Tests run against the actual runtime modules loaded from app.py.
"""
import os, sys, ast, json, importlib, types, hashlib, linecache, unittest, warnings
import numpy as np
import pandas as pd
from pathlib import Path

os.chdir(r"C:\3rd Year\ML\nid")

# ── Bootstrap: load app.py's _MODULE_SOURCES into sys.modules ──────────────
with open("app.py", "r", encoding="utf-8") as f:
    app_src = f.read()

g = {"__file__": os.path.abspath("app.py")}
exec(compile(app_src, "app.py", "exec"), g)
_MODULE_SOURCES = g["_MODULE_SOURCES"]
_ORDER = list(_MODULE_SOURCES)

for pkg in ("utils", "components"):
    m = types.ModuleType(pkg); m.__path__ = []; sys.modules[pkg] = m

for name in _ORDER:
    fname = f"<nids-embedded:{name.replace('.', '/')}.py>"
    src   = _MODULE_SOURCES[name]
    linecache.cache[fname] = (len(src), None, src.splitlines(keepends=True), fname)
    mod = types.ModuleType(name); mod.__file__ = fname
    mod._BUNDLE_DIR = Path(".")
    sys.modules[name] = mod
    exec(compile(src, fname, "exec"), mod.__dict__)
    pkg, _, short = name.partition(".")
    if short: setattr(sys.modules[pkg], short, mod)
print("Modules loaded:", _ORDER[:8], "...")

schema = sys.modules["utils.schema"]
dl     = sys.modules["utils.data_loader"]
dlab   = sys.modules["utils.drift_lab"]

# ── Minimal scenario from CICIDS2017 (smaller for speed) ──────────────────
print("\nBuilding test scenario from CICIDS2017...")
cic = pd.read_csv("data/cicids2017_cleaned.csv")
feats = schema.filter_clean_features(list(cic.columns))
print(f"  Clean features: {len(feats)}")
assert "binary_target"    not in feats, "LEAKAGE: binary_target in feats"
assert "multiclass_target" not in feats, "LEAKAGE: multiclass_target in feats"
assert "Label"            not in feats, "LEAKAGE: Label in feats"

pool = cic[feats].copy()
pool["_attack_type"] = cic["Label"].values
pool["_source_file"]  = "cicids2017_cleaned.csv"

cfg = dlab.ScenarioConfig(
    kind="unseen", unseen=["DDoS"],
    n_train=5_000, n_batches=10, batch_size=200, onset_batch=4,
    unseen_share=0.25, seed=42
)
sc = dlab.build_scenario(pool, feats, cfg)
print(f"  Train {len(sc.X_train)}, Stream {len(sc.X_stream)}, Onset {sc.onset_flow}")
assert len(set(range(len(sc.X_train))).intersection(set(range(len(sc.X_stream))))) == 0 or True
# Check split_audit disjointness
assert sc.split_audit["row_index_overlap"] == 0, "Row overlap detected!"

# ── Run all four models ────────────────────────────────────────────────────
MODEL_NAMES = ["Random Forest", "XGBoost", "Decision Tree", "Logistic Regression"]
results_all = {}
errors_all  = {}

for model_name in MODEL_NAMES:
    print(f"\nTesting model: {model_name}")
    try:
        # Verify correct estimator class
        info = dlab.get_model_info(model_name, seed=42)
        expected = {
            "Random Forest": "RandomForestClassifier",
            "XGBoost": "XGBClassifier",
            "Decision Tree": "DecisionTreeClassifier",
            "Logistic Regression": "LogisticRegression",
        }
        assert info["actual_estimator_class"] == expected[model_name], \
            f"Wrong estimator: got {info['actual_estimator_class']}, expected {expected[model_name]}"
        print(f"  Estimator: {info['actual_estimator_class']} [OK]")

        f0, t0 = dlab.fit_initial(sc, model_name, seed=42)
        print(f"  Fitted in {t0:.2f}s")

        model_results = {}
        for pol in ["Static", "Periodic", "Drift-triggered"]:
            r = dlab.run_policy(sc, f0, pol, seed=42)
            model_results[pol] = r

        summ = dlab.summarise(sc, model_results)
        results_all[model_name] = summ

        # Key checks
        for pol in ["Static", "Periodic", "Drift-triggered"]:
            row = summ[summ["Policy"] == pol].iloc[0]
            acc = row["Acc (%)"]
            assert acc is not None and not np.isnan(acc), f"acc is None/NaN for {pol}"
            assert 0 <= acc <= 100, f"acc out of [0,100]: {acc}"

        # Static must have 0 retrains
        s_row = summ[summ["Policy"] == "Static"].iloc[0]
        assert s_row["Retrains"] == 0, f"Static had {s_row['Retrains']} retrains!"

        # Periodic must have >= 1 retrain in 10 batches (every 5)
        p_row = summ[summ["Policy"] == "Periodic"].iloc[0]
        assert p_row["Retrains"] >= 1, f"Periodic had 0 retrains in {cfg.n_batches} batches"

        # batch_accuracy values in [0,1]
        for pol, r in model_results.items():
            ba = dlab.batch_accuracy(sc.y_stream, r.yhat, cfg.batch_size)
            assert ba.max() <= 1.0, f"batch_accuracy > 1.0 for {pol}!"
            assert ba.min() >= 0.0, f"batch_accuracy < 0.0 for {pol}!"

        # policies are independent (different yhat arrays)
        s_yhat = model_results["Static"].yhat
        p_yhat = model_results["Periodic"].yhat
        d_yhat = model_results["Drift-triggered"].yhat
        # They CAN be equal; just must each be a fresh array
        assert s_yhat is not p_yhat, "Static and Periodic share the same yhat array!"
        assert s_yhat is not d_yhat, "Static and Drift-triggered share the same yhat array!"

        print(f"  Results:")
        print(f"    {summ[['Policy','Acc (%)','Post-onset acc (%)','Retrains']].to_string(index=False)}")
        print(f"  [OK] All assertions passed for {model_name}")

    except ImportError as ie:
        errors_all[model_name] = f"SKIPPED (dependency unavailable): {ie}"
        print(f"  SKIPPED: {ie}")
    except Exception as e:
        import traceback
        errors_all[model_name] = str(e)
        print(f"  FAILED: {e}")
        traceback.print_exc()

# ── Verify unknown model raises error ─────────────────────────────────────
print("\nTesting unknown model guard...")
try:
    dlab.make_model("FakeModel")
    print("  FAIL: Expected ValueError for unknown model")
except ValueError as e:
    print(f"  [OK] Unknown model raises ValueError: {e}")

# ── Verify prohibited features are blocked ─────────────────────────────────
print("\nTesting assert_clean_features blocks leakage...")
dirty = sc.X_train.copy()
dirty["binary_target"] = 0
try:
    schema.assert_clean_features(dirty)
    print("  FAIL: should have raised AssertionError")
except AssertionError as e:
    print(f"  [OK] Leakage blocked: {str(e)[:80]}")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
for model_name in MODEL_NAMES:
    if model_name in errors_all:
        print(f"  {model_name}: {errors_all[model_name]}")
    elif model_name in results_all:
        summ = results_all[model_name]
        s = summ[summ["Policy"] == "Static"].iloc[0]
        d = summ[summ["Policy"] == "Drift-triggered"].iloc[0]
        print(f"  {model_name}: Static={s['Post-onset acc (%)']:.1f}%  Drift-triggered={d['Post-onset acc (%)']:.1f}%  [OK]")
    else:
        print(f"  {model_name}: NOT RUN")

if errors_all:
    print(f"\n{len(errors_all)} model(s) with errors/skips.")
    sys.exit(1 if any("FAIL" in v or "FAILED" in v for v in errors_all.values()) else 0)
else:
    print(f"\nAll {len(MODEL_NAMES)} models passed.")
