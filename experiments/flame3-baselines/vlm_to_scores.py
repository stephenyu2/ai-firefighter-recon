"""Convert a raw VLM csv (uid, y, p_yes) into the standard scores_*.csv with fold-based thresholds."""
import sys

import numpy as np
import pandas as pd

from common import CACHE, load_index, pick_threshold_for_recall, save_scores

raw_path, name = sys.argv[1], sys.argv[2]
df = load_index()
raw = pd.read_csv(raw_path).drop_duplicates("uid").set_index("uid")
missing = set(df.uid) - set(raw.index)
if missing:
    sys.exit(f"{name}: {len(missing)} frames missing, not converting yet")
s = raw.loc[df.uid, "p_yes"].values
folds = pd.read_csv(CACHE / "index_folds.csv").set_index("uid").loc[df.uid, "fold_grouped"].values
thr = np.zeros(len(df))
for k in range(folds.max() + 1):
    tr = folds != k
    thr[folds == k] = pick_threshold_for_recall(df.y.values[tr], s[tr], 0.98)
save_scores(name, df, s, {"fold": folds, "thr_default": 0.5, "thr_tuned": thr,
                          "mass_yes_no": raw.loc[df.uid, "mass_yes_no"].values})
print(f"saved {name}: mean yes/no mass {raw.mass_yes_no.mean():.3f}")
