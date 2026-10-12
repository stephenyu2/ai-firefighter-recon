"""Find where the thermal camera's view sits inside the RGB frame (Autel EVO II Dual 640T, Hanna Hammock).

Matches the edge map of each raw thermal frame against the edge map of its RGB frame over a sweep of scales,
reports the best scale and center per frame, and saves an overlay of the median crop for a visual check.
Usage: python check_fov.py NADIR_ROOT OUT_PNG"""
import re, sys
from pathlib import Path
import cv2, numpy as np, tifffile

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
RW = 1600  # RGB working width


def edges(a):
    a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 2)
    g = cv2.magnitude(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))
    return (g - g.mean()) / (g.std() + 1e-6)


rows = []
pairs = []
for plot in ("plot 1", "plot 2"):
    th = sorted((ROOT / plot / "duringburn/raw_thermal_tiff").glob("*.TIFF"))
    for t in th[:: max(1, len(th) // 6)]:
        n = int(re.search(r"(\d+)", t.stem).group(1))
        rgb = ROOT / plot / "duringburn/raw_rgb_jpg" / f"MAX_{n:04d}.JPG"
        if rgb.exists():
            pairs.append((t, rgb))
for t, rgb in pairs:
    T = tifffile.imread(t).astype(np.float32) * 0.1 - 275
    T = np.clip(T, np.percentile(T, 1), np.percentile(T, 99))
    im = cv2.imread(str(rgb), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    im = cv2.resize(im, (RW, round(RW * im.shape[0] / im.shape[1])), interpolation=cv2.INTER_AREA)
    E = edges(im)
    best = (-1, None, None)
    for s in np.arange(0.30, 0.62, 0.01):
        w = int(RW * s); h = int(w * T.shape[0] / T.shape[1])
        tpl = edges(cv2.resize(T, (w, h), interpolation=cv2.INTER_AREA))
        r = cv2.matchTemplate(E, tpl, cv2.TM_CCOEFF_NORMED)
        _, v, _, loc = cv2.minMaxLoc(r)
        if v > best[0]:
            best = (v, s, (loc[0] + w / 2, loc[1] + h / 2, w, h))
    v, s, (cx, cy, w, h) = best
    rows.append((rgb.name, round(v, 3), round(s, 3), round(h / E.shape[0], 3), round(cx / E.shape[1], 3), round(cy / E.shape[0], 3)))
    print(*rows[-1], flush=True)
a = np.array([r[1:] for r in rows], dtype=float)
good = a[a[:, 0] >= np.median(a[:, 0])]
med = np.median(good, axis=0)
print("median of better half: score %.3f  width_frac %.3f  height_frac %.3f  center_x %.3f  center_y %.3f" % tuple(med))
# overlay: RGB with median box, plus the thermal frame resized into the box, for the best-scoring pair
t, rgb = pairs[int(np.argmax(a[:, 0]))]
im = cv2.imread(str(rgb), cv2.IMREAD_REDUCED_COLOR_4)
im = cv2.resize(im, (RW, round(RW * im.shape[0] / im.shape[1])), interpolation=cv2.INTER_AREA)
H, W = im.shape[:2]
w, h = int(W * med[1]), int(H * med[2]); x0, y0 = int(W * med[3] - w / 2), int(H * med[4] - h / 2)
T = tifffile.imread(t).astype(np.float32) * 0.1 - 275
Tn = cv2.applyColorMap(cv2.normalize(cv2.resize(T, (w, h)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_INFERNO)
blend = im.copy()
blend[y0:y0 + h, x0:x0 + w] = cv2.addWeighted(im[y0:y0 + h, x0:x0 + w], 0.5, Tn, 0.5, 0)
cv2.rectangle(blend, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 3)
crop = cv2.resize(im[y0:y0 + h, x0:x0 + w], (640, 512))
side = np.hstack([crop, cv2.resize(Tn, (640, 512))])
cv2.imwrite(str(OUT), np.vstack([cv2.resize(blend, (1280, round(1280 * H / W))), side]))
print("overlay", OUT, rgb.name)
