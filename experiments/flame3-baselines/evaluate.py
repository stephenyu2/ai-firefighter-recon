"""Aggregate every scores_*.csv into one comparison table (pooled out-of-fold predictions on all 738 images)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS, binary_metrics, cluster_bootstrap_ci, load_index, ranking_metrics

df = load_index().set_index("uid")
rows = []
for f in sorted(RESULTS.glob("scores_*.csv")):
    name = f.stem.replace("scores_", "")
    if "DAYHOLDOUT" in name:
        continue
    s = pd.read_csv(f).set_index("uid").loc[df.index]
    y = s.y.values; sc = s.score.values
    d = binary_metrics(y, sc, s.thr_default.values)
    t = binary_metrics(y, sc, s.thr_tuned.values) if "thr_tuned" in s else {}
    r = ranking_metrics(y, sc)
    mixed = df.mixed_flight.values
    dm = binary_metrics(y[mixed], sc[mixed], s.thr_default.values[mixed])
    small = (df.y.values == 1) & (df.frac_gt80.values < 0.0025)
    ds = binary_metrics(y[small], sc[small], s.thr_default.values[small])
    lo_r, hi_r = cluster_bootstrap_ci(df.reset_index(), sc, s.thr_default.values, "recall", n=400)
    lo_s, hi_s = cluster_bootstrap_ci(df.reset_index(), sc, s.thr_default.values, "specificity", n=400)
    rows.append(dict(model=name,
                     recall=d["recall"], precision=d["precision"], f1=d["f1"], specificity=d["specificity"],
                     rec_ci=f"{lo_r:.2f}-{hi_r:.2f}", spec_ci=f"{lo_s:.2f}-{hi_s:.2f}",
                     roc_auc=r["roc_auc"], pr_auc=r["pr_auc"],
                     tuned_recall=t.get("recall"), tuned_spec=t.get("specificity"), tuned_prec=t.get("precision"),
                     mixed_flight_fire_recall=dm["recall"], small_fire_recall=ds["recall"],
                     fp=d["fp"], fn=d["fn"]))
out = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(out.round(3).to_string(index=False))
out.to_csv(RESULTS / "summary.csv", index=False)
