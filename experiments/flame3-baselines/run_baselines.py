"""No-learning baselines: constant 'fire', thermal max-temperature rule, thermal-JPG darkness."""
import numpy as np
from PIL import Image

from common import load_index, pick_threshold_for_recall, save_scores

df = load_index()
folds = __import__("pandas").read_csv("cache/index_folds.csv").set_index("uid").loc[df.uid, "fold_grouped"].values


def tuned_thresholds(score, target=0.98):
    thr = np.zeros(len(df))
    for k in range(folds.max() + 1):
        tr = folds != k
        thr[folds == k] = pick_threshold_for_recall(df.y.values[tr], score[tr], target)
    return thr


save_scores("B0_constant_fire", df, np.ones(len(df)), {"fold": folds, "thr_default": 0.5, "thr_tuned": 0.5})
s = df.t_max.values
save_scores("B1_maxtemp_rule", df, s, {"fold": folds, "thr_default": 150.0, "thr_tuned": tuned_thresholds(s),
                                        "thr_alt80": 80.0})
dark = np.array([-np.asarray(Image.open(p).convert("L"), dtype=float).mean() for p in df.thermal_jpg])
save_scores("B2_thermaljpg_darkness", df, dark, {"fold": folds, "thr_default": np.median(dark),
                                                   "thr_tuned": tuned_thresholds(dark)})
print("baselines saved")
