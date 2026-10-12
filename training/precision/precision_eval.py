"""Paired precision study on FLAME 3: for each purged CV fold, train once in 16-bit (LoRA), merge, then score the held-out
fold in FP16, FP8, NVFP4 (weights + activations) and NVFP4 (weights only). Reports recall, false alarms, decision flips vs FP16,
and GPU time per frame (batch 32 and batch 1). Modes that fail on this GPU are recorded, not hidden."""
import contextlib, copy, json, os, sys, time, traceback
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import Dinov2Model

HERE = Path(__file__).parent
OUT = Path(os.environ.get("OUTPUT_DIR", "./out")); (OUT / "results").mkdir(parents=True, exist_ok=True)
DEV = os.environ.get("DEV", "cuda"); SMOKE = os.environ.get("SMOKE") == "1"
W, H = 308, 252; EPOCHS, BS = (1, 8) if SMOKE else (8, 16); PURGE_S = 30
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
VARIANTS = {"dinov2b_lora_strongaug": dict(model="facebook/dinov2-base", aug="strong", lr=3e-4, dim=768),
            "dinov2g_lora": dict(model="facebook/dinov2-giant", aug="basic", lr=2e-4, dim=1536)}
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

df = pd.read_csv(HERE / "index.csv" if (HERE / "index.csv").exists() else HERE.parent / "flame3-lora/index.csv", parse_dates=["ts"])
if SMOKE:
    df = pd.concat([df[df.y == 0].sample(40, random_state=0), df[df.y == 1].sample(80, random_state=0)]).sort_values("ts").reset_index(drop=True)
TS = df.ts.values.astype("datetime64[s]").astype(np.int64)
SMALL = (df.y.values == 1) & (df.frac_gt80.values < 0.0025); MIXED = df.mixed_flight.values.astype(bool)
X = torch.from_numpy(np.stack([np.asarray(Image.open(rgb_path(u)).convert("RGB").resize((W, H), Image.BICUBIC), dtype=np.uint8) for u in df.uid])).permute(0, 3, 1, 2).contiguous()
Y = torch.tensor(df.y.values.astype(np.float32))
env = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else DEV, "capability": torch.cuda.get_device_capability(0) if DEV == "cuda" else None}
try:
    import torchao; env["torchao"] = torchao.__version__
except Exception as e:
    env["torchao"] = f"missing: {e}"
print(json.dumps(env), flush=True); json.dump(env, open(OUT / "results/env.json", "w"))


def purge(tr, te):
    tt = np.sort(TS[te]); t = TS[tr]; pos = np.searchsorted(tt, t)
    d = np.minimum(np.abs(t - tt[np.clip(pos - 1, 0, len(tt) - 1)]), np.abs(tt[np.clip(pos, 0, len(tt) - 1)] - t))
    return tr[d > PURGE_S]


def augment(xb, mode):
    xb = xb.float() / 255; f = torch.rand(len(xb)) < 0.5; xb[f] = xb[f].flip(-1)
    if mode == "strong":
        v = torch.rand(len(xb)) < 0.5; xb[v] = xb[v].flip(-2)
        b = 1 + (torch.rand(len(xb), 1, 1, 1) - 0.5) * 0.4; c = 1 + (torch.rand(len(xb), 1, 1, 1) - 0.5) * 0.4
        m = xb.mean(dim=(1, 2, 3), keepdim=True); xb = ((xb - m) * c + m) * b
        s = 1 + (torch.rand(len(xb), 1, 1, 1) - 0.5) * 0.6; g = xb.mean(1, keepdim=True); xb = g + (xb - g) * s
        gr = torch.rand(len(xb)) < 0.1; xb[gr] = xb[gr].mean(1, keepdim=True).expand(-1, 3, -1, -1); xb = xb.clamp(0, 1)
    return (xb - MEAN) / STD


def train(cfg, idx, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    base = Dinov2Model.from_pretrained(cfg["model"], dtype=torch.float32)
    enc = get_peft_model(base, LoraConfig(r=16, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none")).to(DEV)
    head = nn.Linear(2 * cfg["dim"], 1).to(DEV)
    params = [p for p in enc.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.01)
    steps = EPOCHS * int(np.ceil(len(idx) / BS)); sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.1)
    pw = torch.tensor(float((Y[idx] == 0).sum() / max((Y[idx] == 1).sum(), 1)), device=DEV)
    ac = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if DEV == "cuda" else contextlib.nullcontext
    for _ in range(EPOCHS):
        enc.train(); perm = idx[np.random.permutation(len(idx))]
        for i in range(0, len(perm), BS):
            b = perm[i:i + BS]
            with ac():
                h = enc(pixel_values=augment(X[b], cfg["aug"]).to(DEV)).last_hidden_state.float()
            loss = F.binary_cross_entropy_with_logits(head(torch.cat([h[:, 0], h[:, 1:].mean(1)], -1)).squeeze(-1), Y[b].to(DEV), pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return enc.merge_and_unload().eval(), head.eval()


# fp16 and bf16 are plain casts. torchao's FP8 and NVFP4 paths require bf16 weights, so they quantize a bf16 copy.
DT = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.bfloat16, "nvfp4": torch.bfloat16, "nvfp4_weight_only": torch.bfloat16}


def quantized(enc32, mode):
    """Return a copy of the merged encoder in the given precision (Linear layers only; the head stays FP32)."""
    m = copy.deepcopy(enc32).to(dtype=DT[mode]) if DEV == "cuda" else copy.deepcopy(enc32)
    if mode in ("fp16", "bf16"):
        return m
    from torchao.quantization import quantize_
    if mode == "fp8":
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow
        quantize_(m, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))
    elif mode == "nvfp4":
        try:
            from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig as C
        except ImportError:
            from torchao.prototype.mx_formats.inference_workflow import NVFP4DynamicActivationNVFP4WeightConfig as C
        try:
            quantize_(m, C(use_dynamic_per_tensor_scale=True, use_triton_kernel=True))
            m(pixel_values=torch.zeros(2, 3, H, W, device=DEV, dtype=DT[mode]))
        except Exception as e:
            print(f"nvfp4 with the triton kernel failed ({type(e).__name__}: {str(e)[:200]}); retrying without it", flush=True)
            m = copy.deepcopy(enc32).to(dtype=DT[mode])
            quantize_(m, C(use_dynamic_per_tensor_scale=True, use_triton_kernel=False))
    elif mode == "nvfp4_weight_only":
        try:
            from torchao.prototype.mx_formats import NVFP4WeightOnlyConfig as C
        except ImportError:
            from torchao.prototype.mx_formats.inference_workflow import NVFP4WeightOnlyConfig as C
        quantize_(m, C())
    return m


@torch.no_grad()
def score(enc, head, idx, dt):
    out = []
    for i in range(0, len(idx), 32):
        x = ((X[idx[i:i + 32]].float() / 255 - MEAN) / STD).to(DEV, dtype=dt if DEV == "cuda" else torch.float32)
        h = enc(pixel_values=x).last_hidden_state.float()
        out.append(torch.sigmoid(head(torch.cat([h[:, 0], h[:, 1:].mean(1)], -1)).squeeze(-1)).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def timing(enc, bs, dt, reps=20):
    if DEV != "cuda":
        return None
    x = ((X[:bs].float() / 255 - MEAN) / STD).to(DEV, dtype=dt)
    for _ in range(3):
        enc(pixel_values=x)
    torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        enc(pixel_values=x)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / reps / bs


MODES = ["fp16", "bf16", "fp8", "nvfp4", "nvfp4_weight_only"]
rows, errors = [], {}
for name in ONLY:
    cfg = VARIANTS[name]; y = df.y.values
    scores = {m: np.full(len(df), np.nan) for m in MODES}; times = {m: [] for m in MODES}
    for k in (range(1) if SMOKE else range(5)):
        te = np.where(df.fold.values == k)[0]; tr = purge(np.where(df.fold.values != k)[0], te)
        t0 = time.time(); enc, head = train(cfg, tr, seed=k)
        enc = enc.float()
        print(f"{name} fold {k}: trained in {time.time() - t0:.0f}s on {len(tr)} frames", flush=True)
        for mode in MODES:
            if errors.get((name, mode)):
                continue
            try:
                q = quantized(enc, mode)
                scores[mode][te] = score(q, head, te, DT[mode])
                if k == 0:
                    times[mode] = [timing(q, 32, DT[mode]), timing(q, 1, DT[mode])]
                del q
            except Exception as e:
                errors[(name, mode)] = f"{type(e).__name__}: {str(e)[:300]}"
                print(f"{name} {mode} FAILED: {errors[(name, mode)]}", flush=True); traceback.print_exc()
        del enc, head
        if DEV == "cuda":
            torch.cuda.empty_cache()
    base = scores["fp16"]
    for mode in MODES:
        s = scores[mode]; ok = ~np.isnan(s)
        if not ok.any():
            rows.append(dict(variant=name, precision=mode, status=errors.get((name, mode), "no scores"))); continue
        p, pb = s >= 0.5, base >= 0.5
        rows.append(dict(variant=name, precision=mode, status="ok", n=int(ok.sum()),
                         recall=round(p[ok & (y == 1)].mean(), 4), false_alarms=int(p[ok & (y == 0)].sum()), missed=int((~p)[ok & (y == 1)].sum()),
                         mixed_recall=round(p[ok & MIXED & (y == 1)].mean(), 4) if (ok & MIXED & (y == 1)).any() else None,
                         small_recall=round(p[ok & SMALL].mean(), 4) if (ok & SMALL).any() else None,
                         roc_auc=round(roc_auc_score(y[ok], s[ok]), 4) if len(np.unique(y[ok])) > 1 else None,
                         flips_vs_fp16=int((p != pb)[ok].sum()), mean_abs_score_change=round(float(np.abs(s - base)[ok].mean()), 5),
                         gpu_ms_per_frame_b32=None if not times[mode] or times[mode][0] is None else round(times[mode][0], 3),
                         gpu_ms_per_frame_b1=None if not times[mode] or times[mode][1] is None else round(times[mode][1], 3)))
        pd.DataFrame(dict(uid=df.uid, y=y, score=s)).to_csv(OUT / f"results/scores_{name}_{mode}.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "results/precision_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
print("ALL DONE")
