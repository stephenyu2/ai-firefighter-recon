"""Label-leakage audit: retrain the linear probes with training labels shuffled. If the pipeline leaked
test labels anywhere, test ROC-AUC would stay high; if it is clean, it should collapse to ~0.5."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CACHE, load_index

df = load_index(); y = df.y.values
folds = pd.read_csv(CACHE / "index_folds.csv").set_index("uid").loc[df.uid, "fold_grouped"].values
feats = {n: np.load(CACHE / f"feat_{n}.npy") for n in ["dinov2b_rgb", "dinov2b_thermal", "clip-vit-large-patch14_rgb"]}
rng = np.random.default_rng(0)
out = []
for n, X in feats.items():
    real, shuf = np.zeros(len(y)), []
    for rep in range(5):
        s = np.zeros(len(y))
        for k in range(5):
            tr, te = folds != k, folds == k
            ytr = y[tr].copy()
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))
            if rep == 0:
                real[te] = m.fit(X[tr], ytr).predict_proba(X[te])[:, 1]
            s[te] = m.fit(X[tr], rng.permutation(ytr)).predict_proba(X[te])[:, 1]
        shuf.append(roc_auc_score(y, s))
    out.append(dict(features=n, real_auc=roc_auc_score(y, real), shuffled_auc_mean=np.mean(shuf), shuffled_auc_min=min(shuf), shuffled_auc_max=max(shuf)))
    print(f"{n:32s} real-label test AUC {roc_auc_score(y, real):.3f} | shuffled-label test AUC "
          f"mean {np.mean(shuf):.3f} (range {min(shuf):.3f}-{max(shuf):.3f})")

pd.DataFrame(out).to_csv("results/leak_check.csv", index=False)
