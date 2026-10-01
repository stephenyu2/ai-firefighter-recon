"""K-shot linear probes on cached frozen features. Seed 0 draws exactly the frames the K-shot LoRA runs use."""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CACHE, inner_split, load_index, pick_threshold_for_recall, purge_train, save_scores

KSHOT = int(sys.argv[1]) if len(sys.argv) > 1 else 16
PURGED = len(sys.argv) > 2 and sys.argv[2] == "purged"
df = load_index(); y = df.y.values
folds = pd.read_csv(CACHE / "index_folds.csv").set_index("uid").loc[df.uid, "fold_grouped"].values
feats = {"P1_dinov2b_probe_rgb": np.load(CACHE / "feat_dinov2b_rgb.npy"),
         "P2_dinov2b_probe_thermal": np.load(CACHE / "feat_dinov2b_thermal.npy"),
         "P4_clipL14_probe_rgb": np.load(CACHE / "feat_clip-vit-large-patch14_rgb.npy"),
         "P5_siglip2_probe_rgb": np.load(CACHE / "feat_siglip2-so400m-patch14-384_rgb.npy")}
for name, X in feats.items():
    for seed in [0, 1, 2]:
        rng = np.random.default_rng(seed)
        score, thr = np.zeros(len(y)), np.zeros(len(y))
        for k in range(folds.max() + 1):
            outer_tr = np.where(folds != k)[0]; te = np.where(folds == k)[0]
            if PURGED:
                outer_tr = purge_train(df, outer_tr, te)
            itr, iva = inner_split(df.iloc[outer_tr], seed=k)
            tr_idx, va_idx = outer_tr[itr], outer_tr[iva]
            if PURGED:
                tr_idx = purge_train(df, tr_idx, va_idx)
            tr_idx = np.concatenate([rng.choice(tr_idx[y[tr_idx] == c], KSHOT, replace=False) for c in (0, 1)])
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))
            m.fit(X[tr_idx], y[tr_idx])
            thr[te] = pick_threshold_for_recall(y[va_idx], m.predict_proba(X[va_idx])[:, 1], 0.98)
            score[te] = m.predict_proba(X[te])[:, 1]
        save_scores(f"{name}_{KSHOT}shot" + ("_PURGED" if PURGED else "") + ("" if seed == 0 else f"_seed{seed}"), df, score,
                    {"fold": folds, "thr_default": 0.5, "thr_tuned": thr})
    print("saved", name, KSHOT, "shot x3 seeds", flush=True)
