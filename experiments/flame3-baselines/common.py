"""Shared data loading, fold construction, thermal encoding, and metrics."""
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedGroupKFold

HERE = Path(__file__).parent
CACHE = HERE / "cache"
RESULTS = HERE / "results"
FIGS = HERE / "figures"
SEG_GAP_S = 10
WINDOW_S = 60
N_FOLDS = 5


def load_index():
    df = pd.read_csv(CACHE / "index.csv", parse_dates=["ts"])
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    gap = df.ts.diff().dt.total_seconds().fillna(1e9)
    df["segment"] = (gap > SEG_GAP_S).cumsum()
    seg_start = df.groupby("segment").ts.transform("min")
    df["window"] = ((df.ts - seg_start).dt.total_seconds() // WINDOW_S).astype(int)
    df["group"] = df.segment.astype(str) + "_" + df.window.astype(str)
    df["flight"] = (gap > 60).cumsum()
    mixed = df.groupby("flight").y.agg(lambda s: 0 < s.mean() < 1)
    df["mixed_flight"] = df.flight.map(mixed)
    df["uid"] = df.label.str.replace(" ", "") + "_" + df.stem.astype(str)
    return df


def make_folds(df, kind="grouped", seed=42):
    """Returns array fold[i] in 0..N_FOLDS-1. 'grouped' keeps each 60 s time window in one fold."""
    fold = np.full(len(df), -1)
    if kind == "grouped":
        splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        it = splitter.split(df, df.y, groups=df.group)
    elif kind == "random":
        it = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(df)
    else:
        raise ValueError(kind)
    for k, (_, te) in enumerate(it):
        fold[te] = k
    return fold


PURGE_S = 30


def purge_train(df, train_idx, test_idx, purge_s=PURGE_S):
    """Drop training frames captured within purge_s seconds of any test frame (purged k-fold for time series)."""
    t = df.ts.values.astype("datetime64[s]").astype(np.int64)
    tt = np.sort(t[test_idx])
    pos = np.searchsorted(tt, t[train_idx])
    left = np.abs(t[train_idx] - tt[np.clip(pos - 1, 0, len(tt) - 1)])
    right = np.abs(tt[np.clip(pos, 0, len(tt) - 1)] - t[train_idx])
    return train_idx[np.minimum(left, right) > purge_s]


def day_split(df):
    """Temporal holdout: train on Oct 25-26 (both no-fire flights), test on every Oct 27 frame (unseen flights)."""
    d = df.ts.dt.day.values
    return np.where(d <= 26)[0], np.where(d == 27)[0]


def inner_split(df_train, seed=0):
    """Split an outer-train frame into inner-train / thresh-val by time-window groups (~20% val)."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tr, va = next(splitter.split(df_train, df_train.y, groups=df_train.group))
    return tr, va


def load_rgb(row):
    return Image.open(row["rgb"]).convert("RGB")


def load_tiff(row):
    return tifffile.imread(row["tiff"]).astype(np.float32)


def thermal_3win(t):
    """Encode Celsius into 3 temperature windows as an 8-bit RGB image.
    R: ambient detail (-20..60 C), G: warm (0..200 C), B: fire intensity (0..600 C)."""
    def win(lo, hi):
        return (np.clip((t - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(np.stack([win(-20, 60), win(0, 200), win(0, 600)], axis=-1))


def load_thermal_img(row):
    return thermal_3win(load_tiff(row))


def binary_metrics(y, score, thr):
    y = np.asarray(y).astype(int)
    pred = (np.asarray(score) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    rec = tp / max(tp + fn, 1); prec = tp / max(tp + fp, 1); spec = tn / max(tn + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return dict(recall=rec, precision=prec, f1=f1, specificity=spec, bal_acc=(rec + spec) / 2,
                accuracy=(tp + tn) / len(y), tp=tp, fn=fn, fp=fp, tn=tn)


def ranking_metrics(y, score):
    y = np.asarray(y).astype(int); s = np.asarray(score, dtype=float)
    if len(np.unique(y)) < 2 or np.allclose(s, s[0]):
        return dict(roc_auc=float("nan") if len(np.unique(y)) < 2 else 0.5, pr_auc=float(np.mean(y)))
    return dict(roc_auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s))


def pick_threshold_for_recall(y, score, target=0.98):
    """Among thresholds with recall >= target on (y, score), maximize specificity; return the midpoint (in score
    space) of the tied interval so the chosen cut keeps a margin on both sides instead of hugging the training data."""
    y = np.asarray(y).astype(int); s = np.asarray(score, dtype=float)
    u = np.unique(s)
    cands = np.concatenate([[u[0] - 1e-6], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-6]])
    pos, neg = s[y == 1], s[y == 0]
    rec = np.array([(pos >= t).mean() if len(pos) else 1.0 for t in cands])
    spec = np.array([(neg < t).mean() if len(neg) else 1.0 for t in cands])
    ok = rec >= target
    if not ok.any():
        return float(cands[0])
    best = spec[ok].max()
    tied = cands[ok & (spec >= best - 1e-12)]
    return float((tied.min() + tied.max()) / 2)


def cluster_bootstrap_ci(df, score, thr, metric, n=1000, seed=0):
    """95% CI resampling 60 s time windows (frames inside a window are correlated)."""
    rng = np.random.default_rng(seed)
    groups = df.group.values; ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    y = df.y.values; s = np.asarray(score)
    thr = np.broadcast_to(np.asarray(thr, dtype=float), s.shape)
    vals = []
    for _ in range(n):
        pick = rng.choice(ug, size=len(ug), replace=True)
        ii = np.concatenate([idx_by_g[g] for g in pick])
        vals.append(binary_metrics(y[ii], s[ii], thr[ii])[metric])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def save_scores(name, df, score, extra=None):
    """Persist per-image out-of-fold scores so the report can recompute everything."""
    out = pd.DataFrame({"uid": df.uid.values, "y": df.y.values, "score": np.asarray(score, dtype=float)})
    if extra:
        for k, v in extra.items():
            out[k] = v
    RESULTS.mkdir(exist_ok=True)
    out.to_csv(RESULTS / f"scores_{name}.csv", index=False)
    return out
