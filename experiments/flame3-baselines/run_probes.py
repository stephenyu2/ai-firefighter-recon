"""Linear probes: frozen pretrained features + logistic regression, trained per outer fold on FLAME 3.
Threshold for the recall-tuned operating point comes from inner cross-fitted scores (never the test fold)."""
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor, Dinov2Model

from common import CACHE, day_split, load_index, load_thermal_img, pick_threshold_for_recall, purge_train, save_scores

DEV = "mps"
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "grouped"
df = load_index()
fi = pd.read_csv(CACHE / "index_folds.csv").set_index("uid").loc[df.uid]
folds = fi["fold_random" if SPLIT == "random" else "fold_grouped"].values
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_tensor(ims, w, h):
    arr = np.stack([np.asarray(im.resize((w, h), Image.BICUBIC), dtype=np.float32) / 255 for im in ims])
    return (torch.from_numpy(arr).permute(0, 3, 1, 2) - MEAN) / STD


@torch.no_grad()
def dinov2_feats(kind, bs=16):
    path = CACHE / f"feat_dinov2b_{kind}.npy"
    if path.exists():
        return np.load(path)
    m = Dinov2Model.from_pretrained("facebook/dinov2-base").to(DEV).eval()
    out = []
    for i in range(0, len(df), bs):
        rows = df.iloc[i:i + bs]
        ims = [Image.open(p).convert("RGB") for p in rows.rgb] if kind == "rgb" else [load_thermal_img(r) for _, r in rows.iterrows()]
        h = m(pixel_values=to_tensor(ims, 448, 364).to(DEV)).last_hidden_state
        out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], -1).float().cpu().numpy())
    f = np.concatenate(out); np.save(path, f); return f


@torch.no_grad()
def cliplike_feats(name, size, bs=16):
    tag = name.split("/")[-1]
    path = CACHE / f"feat_{tag}_rgb.npy"
    if path.exists():
        return np.load(path)
    if "clip" in name:
        m, p = CLIPModel.from_pretrained(name).to(DEV).eval(), CLIPProcessor.from_pretrained(name)
    else:
        m, p = AutoModel.from_pretrained(name).to(DEV).eval(), AutoProcessor.from_pretrained(name)
    out = []
    for i in range(0, len(df), bs):
        ims = [Image.open(q).convert("RGB").resize((size, size), Image.BICUBIC) for q in df.rgb.iloc[i:i + bs]]
        px = p(images=ims, return_tensors="pt", do_resize=False, do_center_crop=False)["pixel_values"].to(DEV)
        f = m.get_image_features(pixel_values=px)
        f = f if torch.is_tensor(f) else f.pooler_output
        out.append(f.float().cpu().numpy())
    f = np.concatenate(out); np.save(path, f); return f


def probe(X, name):
    y = df.y.values; groups = df.group.values
    score = np.full(len(df), np.nan); thr = np.full(len(df), np.nan); fo = folds.copy()
    make = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))
    if SPLIT == "day":
        d_tr, d_te = day_split(df); plan = [(0, d_tr, d_te)]; fo = np.full(len(df), -1); fo[d_te] = 0
    else:
        plan = [(k, np.where(folds != k)[0], np.where(folds == k)[0]) for k in range(folds.max() + 1)]
    for k, tr, te in plan:
        if SPLIT == "purged":
            tr = purge_train(df, tr, te)
        inner = np.zeros(len(tr))
        for a, b in StratifiedGroupKFold(4, shuffle=True, random_state=k).split(tr, y[tr], groups[tr]):
            inner[b] = make().fit(X[tr][a], y[tr][a]).predict_proba(X[tr][b])[:, 1]
        thr[te] = pick_threshold_for_recall(y[tr], inner, 0.98)
        score[te] = make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    save_scores(name, df, score, {"fold": fo, "thr_default": 0.5, "thr_tuned": thr})
    print("saved", name, flush=True)


t0 = time.time()
d_rgb = dinov2_feats("rgb"); print(f"dinov2 rgb feats {time.time()-t0:.0f}s", flush=True)
d_th = dinov2_feats("thermal"); print(f"dinov2 thermal feats {time.time()-t0:.0f}s", flush=True)
c_rgb = cliplike_feats("openai/clip-vit-large-patch14", 224); print(f"clip feats {time.time()-t0:.0f}s", flush=True)
s_rgb = cliplike_feats("google/siglip2-so400m-patch14-384", 384); print(f"siglip feats {time.time()-t0:.0f}s", flush=True)
sfx = {"grouped": "", "random": "_RANDOMSPLIT", "purged": "_PURGED", "day": "_DAYHOLDOUT"}[SPLIT]
probe(d_rgb, f"P1_dinov2b_probe_rgb{sfx}")
probe(d_th, f"P2_dinov2b_probe_thermal{sfx}")
probe(np.concatenate([d_rgb, d_th], 1), f"P3_dinov2b_probe_fused{sfx}")
probe(c_rgb, f"P4_clipL14_probe_rgb{sfx}")
probe(s_rgb, f"P5_siglip2_probe_rgb{sfx}")
print(f"total {time.time()-t0:.0f}s")
