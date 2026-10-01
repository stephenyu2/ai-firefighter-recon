"""Which fire frames do strong RGB models miss, and what do they look like?"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from common import FIGS, RESULTS, load_index, load_rgb, load_tiff

df = load_index().set_index("uid")
models = ["Z1_clipL14_zeroshot_rgb", "P1_dinov2b_probe_rgb", "P4_clipL14_probe_rgb", "P5_siglip2_probe_rgb"]
miss = pd.DataFrame(index=df.index)
for m in models:
    s = pd.read_csv(RESULTS / f"scores_{m}.csv").set_index("uid").loc[df.index]
    miss[m] = (s.y == 1) & (s.score < s.thr_default)
df["n_missed"] = miss.sum(1)
hard = df[df.n_missed >= 2].sort_values("n_missed", ascending=False)
print(f"fire frames missed by >=2 of {len(models)} strong RGB models: {len(hard)}")
print(hard[["n_missed", "flight", "mixed_flight", "t_max", "frac_gt80", "RelativeAltitude", "GimbalPitchDegree"]].round(4).to_string())
fire = df[df.y == 1]
print("\nall fire frames, medians: frac_gt80", round(fire.frac_gt80.median(), 4), "| t_max", round(fire.t_max.median(), 1))
n = min(len(hard), 6)
if n:
    fig, axes = plt.subplots(n, 2, figsize=(8, 2.9 * n), squeeze=False)
    for i, (uid, r) in enumerate(hard.head(n).iterrows()):
        rgb = np.asarray(load_rgb(r)); t = load_tiff(r)
        ov = rgb.astype(float) / 255; m = t >= 80
        ov[m] = 0.35 * ov[m] + 0.65 * np.array([1.0, 0.0, 1.0])
        axes[i, 0].imshow(rgb); axes[i, 0].set_title(f"missed by {r.n_missed}/{len(models)} RGB models | flight {r.flight}", fontsize=8)
        axes[i, 1].imshow(ov); axes[i, 1].set_title(f"thermal >=80 C overlay: {m.sum()} px, max {t.max():.0f} C", fontsize=8)
        for a in axes[i]:
            a.axis("off")
    fig.tight_layout(); fig.savefig(FIGS / "fig4_rgb_hard_fire_frames.png", dpi=110); plt.close(fig)
    print("figure written")
