"""Fine-tune on FLAME 3 with outer 5-fold CV (time-window groups). Per fold: train on inner-train,
pick the recall-tuned threshold on inner-val, score the held-out outer fold.

Archs:
  dinov2_lora   DINOv2-B + LoRA (q,v) + linear head on [CLS, mean patch]
  firevit_lora  EdBianchi/vit-fire-detection (off-the-shelf fire classifier) + LoRA (q,v), its own 3-class head
  resnet50_full torchvision ResNet-50 ImageNet weights, every layer trained
Usage: python run_finetune.py ARCH MODALITY SPLIT [KSHOT] [EPOCHS]"""
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import Dinov2Model, ViTForImageClassification

from common import (CACHE, day_split, inner_split, load_index, load_thermal_img, pick_threshold_for_recall,
                    purge_train, save_scores)

ARCH, MOD, SPLIT = sys.argv[1], sys.argv[2], sys.argv[3]
KSHOT = int(sys.argv[4]) if len(sys.argv) > 4 else 0
EPOCHS = int(sys.argv[5]) if len(sys.argv) > 5 else 8
SEED = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6].isdigit() else 0
BENCH = "--bench" in sys.argv
DEV = "mps"
BS = 16
SIZE = {"dinov2_lora": (308, 252), "firevit_lora": (224, 224), "resnet50_full": (320, 256)}[ARCH]
torch.manual_seed(SEED); np.random.seed(SEED)

df = load_index()
fcol = {"grouped": "fold_grouped", "purged": "fold_grouped", "random": "fold_random", "day": "fold_grouped"}[SPLIT]
folds = pd.read_csv(CACHE / "index_folds.csv").set_index("uid").loc[df.uid, fcol].values
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
if ARCH == "firevit_lora":
    MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1); STD = MEAN.clone()


def load_all():
    path = CACHE / f"px_{MOD}_{SIZE[0]}x{SIZE[1]}.pt"
    if path.exists():
        return torch.load(path)
    ims = []
    for _, r in df.iterrows():
        im = Image.open(r.rgb).convert("RGB") if MOD == "rgb" else load_thermal_img(r)
        ims.append(np.asarray(im.resize(SIZE, Image.BICUBIC), dtype=np.uint8))
    x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).contiguous()
    torch.save(x, path); return x


X = load_all()
Y = torch.tensor(df.y.values, dtype=torch.float32)


def norm(xb):
    return ((xb.float() / 255) - MEAN) / STD


class DinoLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        base = Dinov2Model.from_pretrained("facebook/dinov2-base")
        cfg = LoraConfig(r=16, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none")
        self.enc = get_peft_model(base, cfg)
        self.head = nn.Linear(1536, 1)

    def forward(self, x):
        h = self.enc(pixel_values=x).last_hidden_state
        return self.head(torch.cat([h[:, 0], h[:, 1:].mean(1)], -1)).squeeze(-1)


class FireViTLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        base = ViTForImageClassification.from_pretrained("EdBianchi/vit-fire-detection")
        lab = {v.lower(): int(k) for k, v in base.config.id2label.items()}
        self.fire, self.normal = lab["fire"], lab["normal"]
        cfg = LoraConfig(r=16, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none",
                         modules_to_save=["classifier"])
        self.m = get_peft_model(base, cfg)

    def forward(self, x):
        return self.m(pixel_values=x).logits


def build():
    if ARCH == "dinov2_lora":
        return DinoLoRA()
    if ARCH == "firevit_lora":
        return FireViTLoRA()
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(2048, 1)
    return m


def loss_and_score(model, logits, yb, pos_w):
    if ARCH == "firevit_lora":
        tgt = torch.where(yb > 0.5, model.fire, model.normal).long()
        w = torch.ones(logits.shape[1], device=logits.device); w[model.fire] = 1 / pos_w
        return F.cross_entropy(logits, tgt, weight=w), 1 - torch.softmax(logits, -1)[:, model.normal]
    logits = logits.squeeze(-1) if logits.dim() > 1 else logits
    return (F.binary_cross_entropy_with_logits(logits, yb, pos_weight=torch.tensor(pos_w, device=logits.device)),
            torch.sigmoid(logits))


@torch.no_grad()
def predict(model, idx):
    model.eval(); out = []
    for i in range(0, len(idx), 32):
        xb = norm(X[idx[i:i + 32]]).to(DEV)
        logits = model(xb)
        if ARCH == "firevit_lora":
            s = 1 - torch.softmax(logits, -1)[:, model.normal]
        else:
            s = torch.sigmoid(logits.squeeze(-1) if logits.dim() > 1 else logits)
        out.append(s.float().cpu())
    return torch.cat(out).numpy()


def train(tr_idx):
    model = build().to(DEV)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    lr = 1e-4 if ARCH == "resnet50_full" else 3e-4
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps = EPOCHS * int(np.ceil(len(tr_idx) / BS))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    yt = Y[tr_idx]; pos_w = float((yt == 0).sum() / max((yt == 1).sum(), 1))
    step = 0; t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); perm = tr_idx[torch.randperm(len(tr_idx)).numpy()]
        for i in range(0, len(perm), BS):
            b = perm[i:i + BS]
            xb = X[b]
            flip = torch.rand(len(b)) < 0.5
            xb[flip] = xb[flip].flip(-1)
            logits = model(norm(xb).to(DEV))
            loss, _ = loss_and_score(model, logits, Y[b].to(DEV), pos_w)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
            if BENCH and step == 10:
                torch.mps.synchronize(); print(f"bench {ARCH} {SIZE}: {(time.time()-t0)/10:.2f}s/step, "
                                                f"{steps} steps/fold, trainable {n_train/1e6:.2f}M"); sys.exit()
    return model, n_train, time.time() - t0


score = np.full(len(df), np.nan); thr = np.full(len(df), np.nan); log = []
rng = np.random.default_rng(SEED)
if SPLIT == "day":
    d_tr, d_te = day_split(df)
    plan = [(0, d_tr, d_te)]
    folds = np.full(len(df), -1); folds[d_te] = 0
else:
    plan = [(k, np.where(folds != k)[0], np.where(folds == k)[0]) for k in range(folds.max() + 1)]
for k, outer_tr, te in plan:
    if SPLIT == "purged":
        outer_tr = purge_train(df, outer_tr, te)
    itr, iva = inner_split(df.iloc[outer_tr], seed=k)
    tr_idx, va_idx = outer_tr[itr], outer_tr[iva]
    if SPLIT in ("purged", "day"):
        tr_idx = purge_train(df, tr_idx, va_idx)
    if KSHOT:
        tr_idx = np.concatenate([rng.choice(tr_idx[df.y.values[tr_idx] == c], KSHOT, replace=False) for c in (0, 1)])
    model, n_train, secs = train(tr_idx)
    t_inf = time.time()
    thr[te] = pick_threshold_for_recall(df.y.values[va_idx], predict(model, va_idx), 0.98)
    score[te] = predict(model, te)
    log.append(dict(fold=k, n_train=len(tr_idx), n_val=len(va_idx), n_test=len(te), trainable_params=n_train,
                    train_s=secs, predict_s=time.time() - t_inf))
    print(f"fold {k}: train {len(tr_idx)} val {len(va_idx)} test {len(te)} | {secs:.0f}s | thr {thr[te][0]:.3f}", flush=True)
    del model; torch.mps.empty_cache()

tag = {"dinov2_lora": "L1_dinov2b_lora", "firevit_lora": "L2_firevit_lora", "resnet50_full": "F1_resnet50_fullft"}[ARCH]
name = (f"{tag}_{MOD}" + (f"_{KSHOT}shot" if KSHOT else "") + {"grouped": "", "random": "_RANDOMSPLIT", "purged": "_PURGED",
        "day": "_DAYHOLDOUT"}[SPLIT] + (f"_seed{SEED}" if SEED else ""))
save_scores(name, df, score, {"fold": folds, "thr_default": 0.5, "thr_tuned": thr})
pd.DataFrame(log).to_csv(f"results/trainlog_{name}.csv", index=False)
print("saved", name)
