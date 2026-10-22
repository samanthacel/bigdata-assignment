import itertools
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from common import LAKE_DIR
from ml_common import (BIN_FEATURES, CAT_FEATURES, LABEL, LOAD_FEATURES, ML_DATASET, ML_DIR,
                       OOT_CUTOFF, PRED_DIR, RAW_FEATURES, SEED, best_f1_threshold, evaluate, save_json)

GRID = {
    "logistic_regression": {"C": [100.0, 1.0, 0.01]},
    "random_forest": {"max_depth": [8, 12]},
    "gbt": {"max_depth": [4, 6]},
}
log1p = FunctionTransformer(np.log1p, feature_names_out="one-to-one")


def make_model(name, params):
    if name == "logistic_regression":
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CAT_FEATURES),
            ("bin", "passthrough", BIN_FEATURES),
            ("load", Pipeline([("log", log1p), ("sc", StandardScaler())]), LOAD_FEATURES)])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED, **params)
    elif name == "random_forest":
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ("bin", "passthrough", BIN_FEATURES),
            ("load", log1p, LOAD_FEATURES)])
        clf = RandomForestClassifier(n_estimators=100, max_features="sqrt", min_samples_leaf=5,
                                     class_weight="balanced", n_jobs=-1, random_state=SEED, **params)
    else:
        pre = ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                   encoded_missing_value=-1), CAT_FEATURES),
            ("bin", "passthrough", BIN_FEATURES),
            ("load", log1p, LOAD_FEATURES)])
        clf = HistGradientBoostingClassifier(max_iter=60, learning_rate=0.1,
                                             categorical_features=list(range(len(CAT_FEATURES))),
                                             class_weight="balanced", early_stopping=False,
                                             random_state=SEED, **params)
    return Pipeline([("pre", pre), ("clf", clf)])


ABLATION = {
    "A. Klasifikasi + tim": (CAT_FEATURES[:5], []),
    "B. A + fitur waktu": (CAT_FEATURES, BIN_FEATURES),
    "C. B + beban kerja (lengkap)": (CAT_FEATURES, BIN_FEATURES + LOAD_FEATURES),
    "D. C tanpa batch_size": (CAT_FEATURES, BIN_FEATURES + [f for f in LOAD_FEATURES if f != "batch_size"]),
}


def hgb_subset(cats, nums, max_depth):
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-1), cats),
        ("num", "passthrough", nums)])
    clf = HistGradientBoostingClassifier(max_iter=60, learning_rate=0.1, max_depth=max_depth,
                                         categorical_features=list(range(len(cats))),
                                         class_weight="balanced", early_stopping=False, random_state=SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


def load():
    df = pd.read_parquet(LAKE_DIR / "gold" / ML_DATASET)
    for c in CAT_FEATURES:
        df[c] = df[c].astype(str)
    df[LABEL] = df[LABEL].astype(int)
    return df.sort_values("case_number").reset_index(drop=True)


def main():
    ML_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    (LAKE_DIR / "models").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    df = load()
    load_sec = time.perf_counter() - t0
    tr, va, te = (df[df.split == s] for s in ("train", "valid", "test"))
    Xtr, ytr = tr[RAW_FEATURES], tr[LABEL]
    print(f"[INFO] scikit-learn {sklearn.__version__}; train={len(tr):,} valid={len(va):,} "
          f"test={len(te):,}; load {load_sec:.1f}s; memori DataFrame "
          f"{df.memory_usage(deep=True).sum() / 2**20:.0f} MB")

    results, tuning = {}, []
    for name, grid in GRID.items():
        best = None
        keys = list(grid)
        for combo in itertools.product(*grid.values()):
            params = dict(zip(keys, combo))
            t0 = time.perf_counter()
            m = make_model(name, params).fit(Xtr, ytr)
            fit_sec = time.perf_counter() - t0
            pv = m.predict_proba(va[RAW_FEATURES])[:, 1]
            ev = evaluate(va[LABEL], pv, 0.5)
            tuning.append({"engine": "sklearn", "model": name, **params, "fit_sec": fit_sec,
                           "valid_pr_auc": ev["pr_auc"], "valid_roc_auc": ev["roc_auc"]})
            print(f"  [{name}] {params} fit={fit_sec:.1f}s valid PR-AUC={ev['pr_auc']:.4f} "
                  f"ROC-AUC={ev['roc_auc']:.4f}")
            if best is None or ev["pr_auc"] > best["valid_pr_auc"]:
                best = {"params": params, "model": m, "fit_sec": fit_sec, "pv": pv,
                        "valid_pr_auc": ev["pr_auc"]}

        m = best["model"]
        thr = best_f1_threshold(va[LABEL], best["pv"])
        t0 = time.perf_counter()
        pt = m.predict_proba(te[RAW_FEATURES])[:, 1]
        pred_sec = time.perf_counter() - t0
        pd.DataFrame({"case_number": te.case_number.values, LABEL: te[LABEL].values, "p": pt}) \
            .to_csv(PRED_DIR / f"sklearn_{name}.csv", index=False)
        metrics = evaluate(te[LABEL], pt, thr)
        results[name] = {"engine": "scikit-learn", "best_params": best["params"],
                         "estimator": type(m.named_steps["clf"]).__name__,
                         "fit_sec": best["fit_sec"], "predict_sec": pred_sec,
                         "valid_pr_auc": best["valid_pr_auc"], "test": metrics}
        joblib.dump(m, LAKE_DIR / "models" / f"sklearn_{name}.joblib")
        print(f"[TEST] {name}: ROC-AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f} "
              f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f} "
              f"Recall@5%={metrics['recall_at_5pct']:.3f}")

        if name == "gbt":   # permutation importance pada fitur mentah (data uji)
            pi = permutation_importance(m, te[RAW_FEATURES], te[LABEL], scoring="average_precision",
                                        n_repeats=3, random_state=SEED, n_jobs=1)
            results[name]["permutation_importance_pr_auc"] = dict(sorted(
                {f: float(v) for f, v in zip(RAW_FEATURES, pi.importances_mean)}.items(),
                key=lambda x: -x[1]))

    best_name = max(results, key=lambda k: results[k]["valid_pr_auc"])
    params = results[best_name]["best_params"]
    tro, teo = df[df.created_ts < OOT_CUTOFF], df[df.created_ts >= OOT_CUTOFF]
    m_oot = make_model(best_name, params).fit(tro[RAW_FEATURES], tro[LABEL])
    po = m_oot.predict_proba(teo[RAW_FEATURES])[:, 1]
    ev = evaluate(teo[LABEL], po, 0.5)
    oot = {"model": best_name, "params": params, "cutoff": OOT_CUTOFF, "train_rows": len(tro),
           "test_rows": len(teo), "roc_auc": ev["roc_auc"], "pr_auc": ev["pr_auc"],
           "base_rate": ev["base_rate"], "recall_at_5pct": ev["recall_at_5pct"],
           "lift_at_5pct": ev["lift_at_5pct"]}
    print(f"[OOT] {best_name}: ROC-AUC={oot['roc_auc']:.4f} PR-AUC={oot['pr_auc']:.4f}")

    # studi ablasi kelompok fitur (HistGB, hiperparameter terbaik)
    depth = results["gbt"]["best_params"]["max_depth"]
    ablation = []
    for label, (cats, nums) in ABLATION.items():
        m = hgb_subset(cats, nums, depth).fit(tr[cats + nums], ytr)
        ev = evaluate(te[LABEL], m.predict_proba(te[cats + nums])[:, 1], 0.5)
        ablation.append({"set": label, "n_features": len(cats) + len(nums), "roc_auc": ev["roc_auc"],
                         "pr_auc": ev["pr_auc"], "recall_at_5pct": ev["recall_at_5pct"]})
        print(f"[ABL] {label}: ROC-AUC={ev['roc_auc']:.4f} PR-AUC={ev['pr_auc']:.4f} "
              f"Recall@5%={ev['recall_at_5pct']:.3f}")

    save_json(ML_DIR / "metrics_sklearn.json", {"ablation": ablation,
        "engine": "scikit-learn", "sklearn_version": sklearn.__version__, "load_sec": load_sec,
        "models": results, "best_by_valid": best_name, "out_of_time": oot, "tuning": tuning})
    print(f"[OK] metrik -> {ML_DIR / 'metrics_sklearn.json'}")


if __name__ == "__main__":
    main()
