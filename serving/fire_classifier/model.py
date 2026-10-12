"""Fire decision model: DINOv2-B + LoRA (FLAME 3 RGB), CLIP ViT-L/14 zero-shot, and Celsius-TIFF thermal tools.

Request:  {"frames": [{"id": "f1", "rgb_b64": "<jpeg/png base64>", "tiff_b64": "<optional float32 Celsius TIFF base64>"}],
           "models": ["dinov2_lora", "clip"], "mask_threshold_c": 80}
Response: {"frames": [{"id", "dinov2_lora": {...}, "clip": {...}, "thermal": {...} | null}], "model_info": {...}}
"""
import base64
import copy
import io
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image

FIRE_PROMPTS = ["an aerial drone photo of a wildfire with flames and smoke", "a drone photo of a forest fire burning",
                "aerial view of a prescribed burn with active fire and smoke", "a photo taken from above of fire flames burning vegetation"]
NOFIRE_PROMPTS = ["an aerial drone photo of a forest with no fire", "a drone photo of trees and grassland with no smoke",
                  "aerial view of a quiet meadow and pine forest", "a photo taken from above of a landscape with no fire"]


def _dev():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Model:
    def __init__(self, **kwargs):
        # detector/ (and optionally detector_giant/): serving_config.json, adapter/, head.pt from train_cv.py's final_<variant>/
        self._data_dir = Path(kwargs.get("data_dir") or os.environ.get("DETECTOR_DIR", Path(__file__).parent / "weights"))
        self.dev = _dev()

    def _load_detector(self, ck):
        from peft import PeftModel
        from transformers import Dinov2Model
        cfg = json.loads((ck / "serving_config.json").read_text())
        base = Dinov2Model.from_pretrained(cfg["base_model"], dtype=torch.float16 if self.dev == "cuda" else torch.float32)
        enc = PeftModel.from_pretrained(base, str(ck / "adapter")).merge_and_unload().to(self.dev).eval()
        head = torch.nn.Linear(2 * enc.config.hidden_size, 1)
        head.load_state_dict(torch.load(ck / "head.pt", map_location="cpu"))
        return dict(cfg=cfg, enc=enc, head=head.to(self.dev).eval(),
                    mean=torch.tensor(cfg["mean"]).view(1, 3, 1, 1), std=torch.tensor(cfg["std"]).view(1, 3, 1, 1),
                    thr=min(cfg["threshold_default"], cfg["threshold_recall98"]))

    def load(self):
        from transformers import CLIPModel, CLIPProcessor
        self.detectors = {}
        for key, sub in [("dinov2_lora", "detector"), ("dinov2g_qlora", "detector_giant")]:
            if (self._data_dir / sub / "serving_config.json").exists():
                self.detectors[key] = self._load_detector(self._data_dir / sub)
        self.cfg = self.detectors["dinov2_lora"]["cfg"]
        self.thr_op = self.detectors["dinov2_lora"]["thr"]
        # Optional compiled variants, e.g. QUANT_VARIANTS="dinov2g_qlora:fp16c,dinov2g_qlora:nvfp4". Each is served as
        # "<detector>_<precision>" with the parent's head and threshold. Off unless the env var is set.
        self._gpu_lock = threading.Lock()
        for spec in [v for v in os.environ.get("QUANT_VARIANTS", "").split(",") if v]:
            key, prec = spec.split(":")
            self.detectors[f"{key}_{prec}"] = self._variant(key, prec)
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.dev).eval()
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        with torch.no_grad():
            t = self.clip_proc(text=FIRE_PROMPTS + NOFIRE_PROMPTS, return_tensors="pt", padding=True).to(self.dev)
            tf = self.clip.get_text_features(**t)
            tf = tf if torch.is_tensor(tf) else tf.pooler_output
            tf = tf / tf.norm(dim=-1, keepdim=True)
            c = torch.stack([tf[:4].mean(0), tf[4:].mean(0)])
            self.clip_cls = c / c.norm(dim=-1, keepdim=True)

    PAD = 32  # compiled variants always run batches of 32 (short batches are zero-padded) so shapes never recompile

    def _variant(self, key, prec):
        """Compiled FP16 ("fp16c") or NVFP4 ("nvfp4", torchao, Linear layers only) copy of a loaded detector."""
        d = self.detectors[key]
        if prec == "fp16c":
            m, dt = d["enc"], torch.float16
        elif prec == "nvfp4":
            from torchao.quantization import quantize_
            try:
                from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig as C
            except ImportError:
                from torchao.prototype.mx_formats.inference_workflow import NVFP4DynamicActivationNVFP4WeightConfig as C
            m, dt = copy.deepcopy(d["enc"]).to(torch.bfloat16), torch.bfloat16  # torchao NVFP4 needs bf16 weights
            quantize_(m, C(use_dynamic_per_tensor_scale=True, use_triton_kernel=True))
        else:
            raise ValueError(f"unknown precision {prec}")
        fn = torch.compile(lambda x, m=m: m(pixel_values=x).last_hidden_state)
        x = torch.zeros(self.PAD, 3, d["cfg"]["height"], d["cfg"]["width"], device=self.dev, dtype=dt)
        t0 = time.time()
        with torch.no_grad():
            fn(x); fn(x)
        self._sync()
        return dict(d, enc=m, fn=fn, dtype=dt, precision=prec, compile_s=round(time.time() - t0, 1))

    @staticmethod
    def _img(b64):
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    def _sync(self):
        if self.dev == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def _dino(self, ims, key="dinov2_lora"):
        d = self.detectors[key]
        w, h = d["cfg"]["width"], d["cfg"]["height"]
        t0 = time.perf_counter()
        a = np.stack([np.asarray(im.resize((w, h), Image.BICUBIC), dtype=np.float32) / 255 for im in ims])
        x = ((torch.from_numpy(a).permute(0, 3, 1, 2) - d["mean"]) / d["std"]).to(self.dev, dtype=d.get("dtype") or next(d["enc"].parameters()).dtype)
        self._sync(); t1 = time.perf_counter()
        if "fn" in d:
            with self._gpu_lock:
                chunks = []
                for i in range(0, len(x), self.PAD):
                    xb = x[i:i + self.PAD]
                    pad = self.PAD - len(xb)
                    if pad:
                        xb = torch.cat([xb, xb.new_zeros(pad, *xb.shape[1:])])
                    chunks.append(d["fn"](xb)[:self.PAD - pad])
                hs = torch.cat(chunks).float()
        else:
            hs = d["enc"](pixel_values=x).last_hidden_state.float()
        p = torch.sigmoid(d["head"](torch.cat([hs[:, 0], hs[:, 1:].mean(1)], -1)).squeeze(-1))
        self._sync(); t2 = time.perf_counter()
        self._timing[key] = {"preprocess_ms": round(1000 * (t1 - t0), 2), "gpu_forward_ms": round(1000 * (t2 - t1), 2)}
        return p.float().cpu().numpy()

    @torch.no_grad()
    def _clip(self, ims):
        px = self.clip_proc(images=[im.resize((224, 224), Image.BICUBIC) for im in ims], return_tensors="pt",
                            do_resize=False, do_center_crop=False)["pixel_values"].to(self.dev)
        f = self.clip.get_image_features(pixel_values=px)
        f = f if torch.is_tensor(f) else f.pooler_output
        f = f / f.norm(dim=-1, keepdim=True)
        return torch.softmax(self.clip.logit_scale.exp() * f @ self.clip_cls.T, -1)[:, 0].float().cpu().numpy()

    @staticmethod
    def thermal(tiff_b64, thr_c=80.0):
        t = tifffile.imread(io.BytesIO(base64.b64decode(tiff_b64))).astype(np.float32)
        m = t >= thr_c
        h, w = t.shape
        out = dict(max_c=round(float(np.nanmax(t)), 1), p99_c=round(float(np.nanpercentile(t, 99)), 1),
                   hot_fraction=round(float(m.mean()), 5), hot_pixels=int(m.sum()), mask_threshold_c=thr_c,
                   fire_present=bool(m.any()), centroid_xy=None, bbox_xyxy=None, touches_edge=False)
        if m.any():
            ys, xs = np.nonzero(m)
            out["centroid_xy"] = [round(float(xs.mean() / w), 4), round(float(ys.mean() / h), 4)]
            out["bbox_xyxy"] = [round(float(xs.min() / w), 4), round(float(ys.min() / h), 4), round(float(xs.max() / w), 4), round(float(ys.max() / h), 4)]
            b = max(2, int(0.02 * min(h, w)))
            out["touches_edge"] = bool(m[:b].any() or m[-b:].any() or m[:, :b].any() or m[:, -b:].any())
        return out

    def predict(self, model_input):
        t0 = time.time()
        frames = model_input.get("frames", [])
        want = set(model_input.get("models", ["dinov2_lora", "clip"]))
        thr_c = float(model_input.get("mask_threshold_c", 80))
        self._timing = {}
        td = time.perf_counter()
        ims = [self._img(f["rgb_b64"]) for f in frames]
        self._timing["decode_ms"] = round(1000 * (time.perf_counter() - td), 2)
        pd_ = self._dino(ims) if ("dinov2_lora" in want and ims) else None
        pc = self._clip(ims) if ("clip" in want and ims) else None
        pg = self._dino(ims, "dinov2g_qlora") if ("dinov2g_qlora" in want and "dinov2g_qlora" in self.detectors and ims) else None
        extra = {k: self._dino(ims, k) for k in self.detectors if k not in ("dinov2_lora", "dinov2g_qlora") and k in want and ims}
        out = []
        for i, f in enumerate(frames):
            r = {"id": f.get("id", str(i))}
            if pd_ is not None:
                p = float(pd_[i])
                r["dinov2_lora"] = dict(p_fire=round(p, 4), fire_default=p >= self.cfg["threshold_default"],
                                        fire_recall_tuned=p >= self.thr_op)
            if pg is not None:
                g = float(pg[i])
                r["dinov2g_qlora"] = dict(p_fire=round(g, 4), fire_recall_tuned=g >= self.detectors["dinov2g_qlora"]["thr"])
            for k, pv in extra.items():
                v = float(pv[i])
                r[k] = dict(p_fire=round(v, 4), fire_recall_tuned=v >= self.detectors[k]["thr"])
            if pc is not None:
                r["clip"] = dict(p_fire=round(float(pc[i]), 4), fire_default=float(pc[i]) >= 0.5)
            r["thermal"] = self.thermal(f["tiff_b64"], thr_c) if f.get("tiff_b64") else None
            out.append(r)
        return {"frames": out, "model_info": {"detector": self.cfg.get("name", "DINOv2-B + LoRA (FLAME 3 RGB)"), "trained_on": self.cfg.get("trained_on"),
                                               "thresholds": {"default": self.cfg["threshold_default"], "operating": self.thr_op}, "detectors": list(self.detectors), "device": str(torch.cuda.get_device_name(0)) if self.dev == "cuda" else self.dev,
                                               "second_opinion": "CLIP ViT-L/14 zero-shot", "elapsed_s": round(time.time() - t0, 3),
                                               "timing": self._timing, "batch_size": len(frames),
                                               "variants": {k: {"precision": v["precision"], "compile_s": v["compile_s"]} for k, v in self.detectors.items() if "precision" in v}}}
