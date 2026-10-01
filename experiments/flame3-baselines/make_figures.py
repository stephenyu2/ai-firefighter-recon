"""Data-inspection figures: max-temperature separability, timeline, and mask overlays."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from common import FIGS, load_index, load_rgb, load_tiff

FIGS.mkdir(exist_ok=True)
df = load_index()
fire, nofire = df[df.y == 1], df[df.y == 0]

fig, ax = plt.subplots(figsize=(9, 3.6))
bins = np.linspace(0, 620, 63)
ax.hist(nofire.t_max, bins=bins, color="#2b6cb0", alpha=0.85, label=f"No Fire (n={len(nofire)})")
ax.hist(fire.t_max, bins=bins, color="#dd6b20", alpha=0.75, label=f"Fire (n={len(fire)})")
ax.axvspan(nofire.t_max.max(), fire.t_max.min(), color="#38a169", alpha=0.15,
           label=f"empty gap {nofire.t_max.max():.0f}-{fire.t_max.min():.0f} C")
for t, ls in [(80, "--"), (150, ":"), (200, "-.")]:
    ax.axvline(t, color="k", ls=ls, lw=1)
    ax.text(t + 4, ax.get_ylim()[1] * 0.9, f"{t} C", fontsize=8)
ax.set_xlabel("per-image maximum temperature (C), radiometric TIFF")
ax.set_ylabel("images")
ax.set_title("FLAME 3 CV subset: max temperature alone separates the two labels")
ax.legend(fontsize=8, loc="upper center")
fig.tight_layout(); fig.savefig(FIGS / "fig1_maxtemp_separability.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 3.2))
day0 = df.ts.dt.normalize().min()
x = (df.ts - day0).dt.total_seconds() / 3600
ax.scatter(x[df.y == 1], df.flight[df.y == 1] + 0.12, s=6, c="#dd6b20", label="Fire")
ax.scatter(x[df.y == 0], df.flight[df.y == 0] - 0.12, s=6, c="#2b6cb0", label="No Fire")
ax.set_xlabel("hours since Oct 25 2022 00:00 (EXIF capture time)")
ax.set_ylabel("flight #")
ax.set_title("All 116 No Fire frames come from 2 of 14 flights (Oct 25 and Oct 26)")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(FIGS / "fig2_timeline.png", dpi=150); plt.close(fig)

rng = np.random.default_rng(3)
picks = [
    ("Fire, large hot area", fire.sort_values("frac_gt80").iloc[-5]),
    ("Fire, median hot area", fire.sort_values("frac_gt80").iloc[len(fire) // 2]),
    ("Fire, smallest max temp", fire.sort_values("t_max").iloc[0]),
    ("Fire, in the mixed flight", fire[fire.mixed_flight].sample(1, random_state=1).iloc[0]),
    ("No Fire, same flight", nofire.sample(1, random_state=2).iloc[0]),
]
fig, axes = plt.subplots(len(picks), 4, figsize=(13, 2.7 * len(picks)))
for r, (title, row) in enumerate(picks):
    rgb = np.asarray(load_rgb(row)); t = load_tiff(row)
    tj = np.asarray(Image.open(row["thermal_jpg"]).convert("RGB").resize((640, 512)))
    axes[r, 0].imshow(rgb); axes[r, 0].set_title(f"{title}\nmax {t.max():.0f} C, {100*(t>80).mean():.2f}% px >80 C", fontsize=8)
    axes[r, 1].imshow(tj); axes[r, 1].set_title("thermal JPG (inferno render)", fontsize=8)
    for c, thr in [(2, 80), (3, 200)]:
        ov = rgb.copy().astype(float) / 255
        m = t >= thr
        ov[m] = 0.35 * ov[m] + 0.65 * np.array([1.0, 0.0, 1.0])
        axes[r, c].imshow(ov); axes[r, c].set_title(f"mask T >= {thr} C on corrected RGB ({m.sum()} px)", fontsize=8)
    for a in axes[r]:
        a.axis("off")
fig.tight_layout(); fig.savefig(FIGS / "fig3_mask_overlays.png", dpi=110); plt.close(fig)
print("figures written")
print("fire frames with max<150C:", int((fire.t_max < 150).sum()), "| max<200C:", int((fire.t_max < 200).sum()))
print("mixed flights:", sorted(df[df.mixed_flight].flight.unique()), "frames", int(df.mixed_flight.sum()), "fire", int(df[df.mixed_flight].y.sum()))
