import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ml_common import LABEL, ML_DIR, PRED_DIR, SEED, save_json

B = int(os.environ.get("BOOT_B", "500"))
NAMES = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "gbt": "Gradient Boosting"}


def recall_at(y, p, frac=0.05):
    k = max(1, int(round(frac * len(p))))
    top = np.argsort(-p, kind="stable")[:k]
    return y[top].sum() / max(1, y.sum())


def main():
    rng = np.random.default_rng(SEED)
    preds = {}
    for eng in ("spark", "sklearn"):
        for k in NAMES:
            d = pd.read_csv(PRED_DIR / f"{eng}_{k}.csv").sort_values("case_number").reset_index(drop=True)
            preds[(eng, k)] = d
    n = len(preds[("spark", "gbt")])
    y = preds[("spark", "gbt")][LABEL].values.astype(int)
    for v in preds.values():
        assert (v[LABEL].values.astype(int) == y).all(), "baris uji tidak sejajar antar model"
    P = {key: v["p"].values for key, v in preds.items()}
    res = {key: {"pr": [], "roc": [], "rec5": []} for key in P}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0:
            continue
        for key, p in P.items():
            pb = p[idx]
            res[key]["pr"].append(average_precision_score(yb, pb))
            res[key]["roc"].append(roc_auc_score(yb, pb))
            res[key]["rec5"].append(recall_at(yb, pb))
    out = {"B": B, "n_test": n, "models": {}, "paired": {}}
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    for (eng, k), r in res.items():
        out["models"][f"{eng}_{k}"] = {"pr_auc": ci(r["pr"]), "roc_auc": ci(r["roc"]), "recall_at_5pct": ci(r["rec5"])}
    for k in NAMES:
        d = np.array(res[("sklearn", k)]["pr"]) - np.array(res[("spark", k)]["pr"])
        out["paired"][k] = {"mean_diff_pr_auc_sklearn_minus_spark": float(d.mean()), "ci": ci(d),
                            "share_sklearn_better": float((d > 0).mean())}
    save_json(ML_DIR / "bootstrap_ci.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
