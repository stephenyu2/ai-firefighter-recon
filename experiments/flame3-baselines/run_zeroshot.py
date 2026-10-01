"""Off-the-shelf models with NO FLAME 3 training: CLIP / SigLIP 2 zero-shot, two HF fire classifiers."""
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import (AutoImageProcessor, AutoModel, AutoModelForImageClassification, AutoProcessor,
                          CLIPModel, CLIPProcessor)

from common import load_index, pick_threshold_for_recall, save_scores

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
df = load_index()
folds = pd.read_csv("cache/index_folds.csv").set_index("uid").loc[df.uid, "fold_grouped"].values

FIRE_RGB = ["an aerial drone photo of a wildfire with flames and smoke",
            "a drone photo of a forest fire burning",
            "aerial view of a prescribed burn with active fire and smoke",
            "a photo taken from above of fire flames burning vegetation"]
NOFIRE_RGB = ["an aerial drone photo of a forest with no fire",
              "a drone photo of trees and grassland with no smoke",
              "aerial view of a quiet meadow and pine forest",
              "a photo taken from above of a landscape with no fire"]
FIRE_TH = ["a thermal infrared image showing a hot fire", "a thermal camera image with bright hot spots from burning flames"]
NOFIRE_TH = ["a thermal infrared image of cool terrain with no fire", "a thermal camera image of a landscape with no hot spots"]


def tuned(score, target=0.98):
    thr = np.zeros(len(df))
    for k in range(folds.max() + 1):
        tr = folds != k
        thr[folds == k] = pick_threshold_for_recall(df.y.values[tr], score[tr], target)
    return thr


def images(kind, size):
    for p in (df.rgb if kind == "rgb" else df.thermal_jpg):
        yield Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC)


@torch.no_grad()
def zeroshot(model, proc, size, fire_p, nofire_p, kind, siglip=False, bs=16):
    tk = dict(padding="max_length", max_length=64) if siglip else dict(padding=True)
    t_in = proc(text=fire_p + nofire_p, return_tensors="pt", **tk).to(DEV)
    tf = model.get_text_features(**t_in)
    tf = tf if torch.is_tensor(tf) else tf.pooler_output
    tf = tf / tf.norm(dim=-1, keepdim=True)
    cls = torch.stack([tf[:len(fire_p)].mean(0), tf[len(fire_p):].mean(0)])
    cls = cls / cls.norm(dim=-1, keepdim=True)
    scale = model.logit_scale.exp()
    out, batch = [], []
    for im in list(images(kind, size)) + [None]:
        if im is not None:
            batch.append(im)
        if len(batch) == bs or (im is None and batch):
            px = proc(images=batch, return_tensors="pt", do_resize=False, do_center_crop=False)["pixel_values"].to(DEV)
            f = model.get_image_features(pixel_values=px)
            f = f if torch.is_tensor(f) else f.pooler_output
            f = f / f.norm(dim=-1, keepdim=True)
            logits = scale * f @ cls.T
            out.append(torch.softmax(logits, -1)[:, 0].float().cpu()); batch = []
    return torch.cat(out).numpy()


@torch.no_grad()
def hf_classifier(name, bs=32):
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModelForImageClassification.from_pretrained(name).to(DEV).eval()
    labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    normal = labels.get("normal", labels.get("none"))
    ims = [Image.open(p).convert("RGB") for p in df.rgb]
    probs = []
    for i in range(0, len(ims), bs):
        px = proc(images=ims[i:i + bs], return_tensors="pt")["pixel_values"].to(DEV)
        probs.append(torch.softmax(model(pixel_values=px).logits, -1).float().cpu())
    p = torch.cat(probs).numpy()
    return 1 - p[:, normal], p, model.config.id2label


t0 = time.time()
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEV).eval()
cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
for kind, fp, nfp in [("rgb", FIRE_RGB, NOFIRE_RGB), ("thermal", FIRE_TH, NOFIRE_TH)]:
    s = zeroshot(clip, cp, 224, fp, nfp, kind)
    save_scores(f"Z1_clipL14_zeroshot_{kind}", df, s, {"fold": folds, "thr_default": 0.5, "thr_tuned": tuned(s)})
    print(f"CLIP {kind} done {time.time()-t0:.0f}s", flush=True)
del clip

sg = AutoModel.from_pretrained("google/siglip2-so400m-patch14-384").to(DEV).eval()
sp = AutoProcessor.from_pretrained("google/siglip2-so400m-patch14-384")
for kind, fp, nfp in [("rgb", FIRE_RGB, NOFIRE_RGB), ("thermal", FIRE_TH, NOFIRE_TH)]:
    s = zeroshot(sg, sp, 384, fp, nfp, kind, siglip=True)
    save_scores(f"Z2_siglip2_zeroshot_{kind}", df, s, {"fold": folds, "thr_default": 0.5, "thr_tuned": tuned(s)})
    print(f"SigLIP2 {kind} done {time.time()-t0:.0f}s", flush=True)
del sg

for tag, name in [("Z3_firevit_edbianchi", "EdBianchi/vit-fire-detection"),
                  ("Z4_forestfire_siglip2", "prithivMLmods/Forest-Fire-Detection")]:
    s, p, id2l = hf_classifier(name)
    extra = {"fold": folds, "thr_default": 0.5, "thr_tuned": tuned(s)}
    for k, v in id2l.items():
        extra[f"p_{v}"] = p[:, int(k)]
    save_scores(f"{tag}_rgb", df, s, extra)
    print(f"{name} done {time.time()-t0:.0f}s", flush=True)
