"""FLAME 3 RGB fire/no-fire: LoRA and QLoRA variants under purged 5-fold CV + day holdout, then final checkpoints.

Outputs (under $OUTPUT_DIR, default ./ckpts):
  results/summary.csv               one row per variant x protocol
  results/scores_<variant>_<protocol>.csv
  final_<variant>/                  adapter + head + serving config (variants listed in FINAL)
"""
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import BitsAndBytesConfig, Dinov2Model

HERE = Path(__file__).parent
OUT = Path(os.environ.get("OUTPUT_DIR", "./ckpts"))
(OUT / "results").mkdir(parents=True, exist_ok=True)
DEV = os.environ.get("DEV", "cuda")
SMOKE = os.environ.get("SMOKE") == "1"
W, H = 308, 252
EPOCHS, BS = (1, 8) if SMOKE else (8, 16)
PURGE_S = 30
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

VARIANTS = {
    "dinov2b_lora": dict(model="facebook/dinov2-base", quant=False, aug="basic", lr=3e-4, dim=768),
    "dinov2b_lora_strongaug": dict(model="facebook/dinov2-base", quant=False, aug="strong", lr=3e-4, dim=768),
    "dinov2l_qlora": dict(model="facebook/dinov2-large", quant=True, aug="basic", lr=3e-4, dim=1024),
    "dinov2g_qlora": dict(model="facebook/dinov2-giant", quant=True, aug="basic", lr=2e-4, dim=1536),
}
FINAL = ["dinov2b_lora", "dinov2b_lora_strongaug", "dinov2g_qlora"]
FINAL_ONLY = os.environ.get("FINAL_ONLY") == "1"
ONLY = sys.argv[1:] or list(VARIANTS)

# Frames: FLAME 3 CV "Corrected FOV" RGB images, read from RECON_DATA_ROOT (default: the repo's data/ folder), or from
# FLAME3_RGB_DIR when it holds flat <uid>.jpg copies (handy for shipping the frames with a remote training job).
RGB_DIR = os.environ.get("FLAME3_RGB_DIR")
FLAME_ROOT = None if RGB_DIR else (Path(os.environ.get("RECON_DATA_ROOT") or Path(__file__).resolve().parents[2] / "data")
                                    / "flame3_cv/FLAME 3 CV Dataset (Sycan Marsh)")


def rgb_path(uid):
    if RGB_DIR:
        return Path(RGB_DIR) / f"{uid}.jpg"
    lab, k = uid.split("_")
    return FLAME_ROOT / ("Fire" if lab == "Fire" else "No Fire") / "RGB/Corrected FOV" / f"{int(k):05d}.JPG"

df = pd.read_csv(HERE / "index.csv", parse_dates=["ts"])
if SMOKE:
    df = pd.concat([df[df.y == 0].sample(40, random_state=0), df[df.y == 1].sample(80, random_state=0)]).sort_values("ts").reset_index(drop=True)
y_all = df.y.values.astype(np.float32)
TS = df.ts.values.astype("datetime64[s]").astype(np.int64)
SMALL = (df.y.values == 1) & (df.frac_gt80.values < 0.0025)
MIXED = df.mixed_flight.values.astype(bool)
X = torch.from_numpy(np.stack([np.asarray(Image.open(rgb_path(u)).convert("RGB").resize((W, H), Image.BICUBIC), dtype=np.uint8)
                               for u in df.uid])).permute(0, 3, 1, 2).contiguous()
Y = torch.tensor(y_all)
print("loaded", X.shape, flush=True)


def purge_train(train_idx, test_idx, s=PURGE_S):
    tt = np.sort(TS[test_idx]); t = TS[train_idx]
    pos = np.searchsorted(tt, t)
    d = np.minimum(np.abs(t - tt[np.clip(pos - 1, 0, len(tt) - 1)]), np.abs(tt[np.clip(pos, 0, len(tt) - 1)] - t))
    return train_idx[d > s]


def inner_split(idx, seed):
    sp = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    a, b = next(sp.split(idx, df.y.values[idx], df.group.values[idx]))
    return idx[a], idx[b]


def pick_threshold(y, s, target=0.98):
    u = np.unique(s); c = np.concatenate([[u[0] - 1e-6], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-6]])
    pos, neg = s[y == 1], s[y == 0]
    rec = np.array([(pos >= t).mean() if len(pos) else 1.0 for t in c])
    spec = np.array([(neg < t).mean() if len(neg) else 1.0 for t in c])
    ok = rec >= target
    if not ok.any():
        return float(c[0])
    tied = c[ok & (spec >= spec[ok].max() - 1e-12)]
    return float((tied.min() + tied.max()) / 2)


class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        kw = {}
        if cfg["quant"]:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        base = Dinov2Model.from_pretrained(cfg["model"], dtype=torch.bfloat16 if cfg["quant"] else torch.float32, **kw)
        self.enc = get_peft_model(base, LoraConfig(r=16, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none"))
        self.head = nn.Linear(2 * cfg["dim"], 1)

    def forward(self, x):
        h = self.enc(pixel_values=x).last_hidden_state.float()
        return self.head(torch.cat([h[:, 0], h[:, 1:].mean(1)], -1)).squeeze(-1)


def augment(xb, mode):
    xb = xb.float() / 255
    flip = torch.rand(len(xb)) < 0.5
    xb[flip] = xb[flip].flip(-1)
    if mode == "strong":
        v = torch.rand(len(xb)) < 0.5
        xb[v] = xb[v].flip(-2)
        b = 1 + (torch.rand(len(xb), 1, 1, 1) - 0.5) * 0.4
        c = 1 + (torch.rand(len(xb), 1, 1, 1) - 0.5) * 0.4
        m = xb.mean(dim=(1, 2, 3), keepdim=True)
        xb = ((xb - m) * c + m) * b
        sat = 1 + (torch.rand(len(xb), 1, 1, 1) - 0.5) * 0.6
        gray = xb.mean(1, keepdim=True)
        xb = gray + (xb - gray) * sat
        g = torch.rand(len(xb)) < 0.1
        xb[g] = xb[g].mean(1, keepdim=True).expand(-1, 3, -1, -1)
        xb = xb.clamp(0, 1)
    return (xb - MEAN) / STD


def norm(xb):
    return (xb.float() / 255 - MEAN) / STD


def train(cfg, idx, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net(cfg).to(DEV)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.01)
    steps = EPOCHS * int(np.ceil(len(idx) / BS))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.1)
    yt = Y[idx]; pw = torch.tensor(float((yt == 0).sum() / max((yt == 1).sum(), 1)), device=DEV)
    t0 = time.time()
    for _ in range(EPOCHS):
        net.train()
        perm = idx[np.random.permutation(len(idx))]
        for i in range(0, len(perm), BS):
            b = perm[i:i + BS]
            with (torch.autocast("cuda", dtype=torch.bfloat16) if DEV == "cuda" else contextlib.nullcontext()):
                logit = net(augment(X[b], cfg["aug"]).to(DEV))
            loss = F.binary_cross_entropy_with_logits(logit.float(), Y[b].to(DEV), pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net, time.time() - t0, sum(p.numel() for p in params)


@torch.no_grad()
def predict(net, idx):
    net.eval(); out = []
    for i in range(0, len(idx), 64):
        with (torch.autocast("cuda", dtype=torch.bfloat16) if DEV == "cuda" else contextlib.nullcontext()):
            out.append(torch.sigmoid(net(norm(X[idx[i:i + 64]]).to(DEV)).float()).cpu())
    return torch.cat(out).numpy()


def metrics(y, s, thr, mask=None):
    m = np.ones(len(y), bool) if mask is None else mask
    thr = np.broadcast_to(np.asarray(thr, dtype=float), s.shape)
    y, s, thr = y[m], s[m], thr[m]
    p = s >= thr
    tp, fn, fp, tn = int((p & (y == 1)).sum()), int((~p & (y == 1)).sum()), int((p & (y == 0)).sum()), int((~p & (y == 0)).sum())
    return dict(recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1), specificity=tn / max(tn + fp, 1), fp=fp, fn=fn)


rows = []
for name in ONLY:
  try:
      cfg = VARIANTS[name]
      for protocol in ([] if FINAL_ONLY else ["purged_cv", "day_holdout"]):
          y = df.y.values; score = np.full(len(df), np.nan); thr = np.full(len(df), np.nan)
          if protocol == "purged_cv":
              plan = [(k, np.where(df.fold.values != k)[0], np.where(df.fold.values == k)[0]) for k in (range(1) if SMOKE else range(5))]
          else:
              d = df.ts.dt.day.values
              plan = [(0, np.where(d <= 26)[0], np.where(d == 27)[0])]
          secs, ntrain = 0.0, 0
          for k, tr, te in plan:
              tr = purge_train(tr, te) if protocol == "purged_cv" else tr
              itr, iva = inner_split(tr, k)
              itr = purge_train(itr, iva)
              net, s_, ntrain = train(cfg, itr, seed=k)
              secs += s_
              thr[te] = pick_threshold(y[iva], predict(net, iva))
              score[te] = predict(net, te)
              del net; (torch.cuda.empty_cache() if DEV == "cuda" else None)
              print(f"{name} {protocol} fold {k}: {s_:.0f}s", flush=True)
          ok = ~np.isnan(score)
          r = dict(variant=name, protocol=protocol, n=int(ok.sum()), train_s=round(secs, 1), trainable_params=ntrain)
          r.update({f"default_{k}": v for k, v in metrics(y[ok], score[ok], 0.5).items()})
          r.update({f"tuned_{k}": v for k, v in metrics(y[ok], score[ok], thr[ok]).items()})
          r["mixed_recall"] = metrics(y, np.nan_to_num(score, nan=-1), 0.5, ok & MIXED & (y == 1))["recall"]
          r["small_recall"] = metrics(y, np.nan_to_num(score, nan=-1), 0.5, ok & SMALL)["recall"]
          r["roc_auc"] = roc_auc_score(y[ok], score[ok]) if len(np.unique(y[ok])) > 1 else float("nan")
          rows.append(r)
          pd.DataFrame(dict(uid=df.uid, y=y, score=score, thr_tuned=thr)).to_csv(OUT / f"results/scores_{name}_{protocol}.csv", index=False)
          pd.DataFrame(rows).to_csv(OUT / "results/summary.csv", index=False)
          print(json.dumps(r), flush=True)

      if name in FINAL:
          src = OUT / f"results/scores_{name}_purged_cv.csv"
          oof = pd.read_csv(src if src.exists() else HERE / f"prior_scores_{name}_purged_cv.csv")
          tuned = pick_threshold(oof.y.values, oof.score.values)
          net, s_, _ = train(cfg, np.arange(len(df)), seed=0)
          d = OUT / f"final_{name}"
          net.enc.save_pretrained(d / "adapter")
          torch.save(net.head.state_dict(), d / "head.pt")
          json.dump(dict(base_model=cfg["model"], width=W, height=H, mean=MEAN.flatten().tolist(), std=STD.flatten().tolist(),
                         threshold_default=0.5, threshold_recall98=tuned, trained_on="all 738 FLAME 3 CV-subset frames",
                         lora=dict(r=16, alpha=16, targets=["q_proj", "v_proj"]), feature="concat(CLS, mean patch)"),
                    open(d / "serving_config.json", "w"), indent=2)
          print(f"saved final_{name} ({s_:.0f}s), recall-98 threshold {tuned:.3f}", flush=True)
  except Exception as e:
    import traceback; traceback.print_exc()
    rows.append(dict(variant=name, protocol='FAILED', error=str(e)[:300]))
    pd.DataFrame(rows).to_csv(OUT / 'results/summary.csv', index=False)

print("ALL DONE")
