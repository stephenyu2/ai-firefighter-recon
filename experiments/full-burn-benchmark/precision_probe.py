"""Speed and score-drift probe for the served fire detectors at FP16, BF16, FP8 and NVFP4, eager and compiled.

For each detector (DINOv2-B LoRA strong-aug, DINOv2-g QLoRA), each precision and each execution mode:
  - GPU forward time per frame at batch 32 and batch 1 (CUDA events, after warm-up; compile time reported separately)
  - score drift against FP16 on the bundled real frames: max |score change| and decision flips at the operating threshold
Linear layers are quantized with torchao; the patch embedding, layer norms and the head stay in 16/32-bit.
Needs a Blackwell GPU for NVFP4. DETECTOR_DIR holds detector/ and detector_giant/ (each: serving_config.json, adapter/,
head.pt, as written by training/flame3-lora/train_cv.py); FRAMES_DIR holds the JPEG frames to score.
Usage: python precision_probe.py DETECTOR_DIR FRAMES_DIR [OUT_JSON]   (results/precision/probe_*.json came from this)
"""
import copy
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# torchao's FP8 and NVFP4 paths require bf16 weights, so they quantize a bf16 copy.
DT = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.bfloat16, "nvfp4": torch.bfloat16,
      "nvfp4_weight_only": torch.bfloat16}
MODES = ["fp16", "bf16", "fp8", "nvfp4", "nvfp4_weight_only"]


def env():
    info = dict(torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0),
                capability=list(torch.cuda.get_device_capability(0)))
    for mod in ("torchao", "mslk"):
        try:
            info[mod] = getattr(__import__(mod), "__version__", "installed")
        except Exception as e:
            info[mod] = f"missing: {e!r}"[:200]
    return info


def load_fp32(ck):
    from peft import PeftModel
    from transformers import Dinov2Model
    cfg = json.loads((ck / "serving_config.json").read_text())
    base = Dinov2Model.from_pretrained(cfg["base_model"], dtype=torch.float32)
    enc = PeftModel.from_pretrained(base, str(ck / "adapter")).merge_and_unload().eval()
    head = torch.nn.Linear(2 * enc.config.hidden_size, 1)
    head.load_state_dict(torch.load(ck / "head.pt", map_location="cpu"))
    return cfg, enc, head.cuda().eval()


def _nvfp4_cfg(name):
    try:
        import torchao.prototype.mx_formats as mx
        return getattr(mx, name)
    except (ImportError, AttributeError):
        from torchao.prototype.mx_formats import inference_workflow as iw
        return getattr(iw, name)


def quantize(enc32, mode, probe_x):
    m = copy.deepcopy(enc32).to("cuda", dtype=DT[mode]).eval()
    if mode in ("fp16", "bf16"):
        return m, ""
    from torchao.quantization import quantize_
    if mode == "fp8":
        # Per-row scaling runs on a CUTLASS kernel that fails to initialize on sm_120 (RTX PRO 6000); per-tensor
        # scaling goes through cuBLAS.
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow, PerTensor
        try:
            quantize_(m, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))
            with torch.no_grad():
                m(pixel_values=probe_x)
            return m, "per-row scales"
        except Exception as e:
            m = copy.deepcopy(enc32).to("cuda", dtype=DT[mode]).eval()
            quantize_(m, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))
            return m, f"per-tensor scales (per-row failed: {type(e).__name__}: {str(e)[:120]})"
    if mode == "nvfp4_weight_only":
        quantize_(m, _nvfp4_cfg("NVFP4WeightOnlyConfig")())
        return m, ""
    C = _nvfp4_cfg("NVFP4DynamicActivationNVFP4WeightConfig")
    try:
        quantize_(m, C(use_dynamic_per_tensor_scale=True, use_triton_kernel=True))
        with torch.no_grad():
            m(pixel_values=probe_x)
        return m, "triton activation-quant kernel"
    except Exception as e:
        m = copy.deepcopy(enc32).to("cuda", dtype=DT[mode]).eval()
        quantize_(m, C(use_dynamic_per_tensor_scale=True, use_triton_kernel=False))
        return m, f"no triton kernel ({type(e).__name__}: {str(e)[:160]})"


def features(enc, x):
    h = enc(pixel_values=x).last_hidden_state.float()
    return torch.cat([h[:, 0], h[:, 1:].mean(1)], -1)


@torch.no_grad()
def gpu_ms_per_frame(fn, x, reps):
    fn(x)  # warm-up (and compile, when compiled)
    fn(x)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn(x)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps / x.shape[0]


def run(data_dir, frames_dir, log=print, reps=30, modes=None):
    modes = modes or MODES
    rows = []
    frames = sorted(p for p in Path(frames_dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    for key, sub in [("dinov2_lora", "detector"), ("dinov2g_qlora", "detector_giant")]:
        cfg, enc32, head = load_fp32(Path(data_dir) / sub)
        thr = min(cfg["threshold_default"], cfg["threshold_recall98"])
        mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1); std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
        X = torch.from_numpy(np.stack([np.asarray(Image.open(f).convert("RGB").resize((cfg["width"], cfg["height"]), Image.BICUBIC),
                                                  dtype=np.float32) / 255 for f in frames])).permute(0, 3, 1, 2)
        X = ((X - mean) / std).cuda()
        ref = None
        for mode in (["fp16"] + [m for m in modes if m != "fp16"]):  # fp16 always runs first as the reference
            torch._dynamo.reset()
            try:
                m, note = quantize(enc32, mode, X[:2].to(DT[mode]))
                with torch.no_grad():
                    sc = torch.sigmoid(head(features(m, X.to(DT[mode]))).squeeze(-1)).cpu().numpy()
                ref = sc if mode == "fp16" else ref
                row = dict(model=key, precision=mode, note=note, n_frames=len(frames),
                           max_abs_score_change=round(float(np.abs(sc - ref).max()), 5),
                           flips_vs_fp16=int(((sc >= thr) != (ref >= thr)).sum()))
                for exec_mode in ("eager", "compiled"):
                    try:
                        fn = (lambda x, m=m: features(m, x)) if exec_mode == "eager" else torch.compile(lambda x, m=m: features(m, x))
                        t0 = time.time()
                        with torch.no_grad():
                            fn(X[:32].to(DT[mode])); torch.cuda.synchronize()
                        row[f"{exec_mode}_first_call_s"] = round(time.time() - t0, 1)
                        for bs in (32, 1):
                            row[f"{exec_mode}_ms_per_frame_b{bs}"] = round(gpu_ms_per_frame(fn, X[:bs].to(DT[mode]), reps), 3)
                    except Exception as e:
                        row[f"{exec_mode}_error"] = f"{type(e).__name__}: {str(e)[:300]}"
                        log(traceback.format_exc()[-1500:])
                del m
            except Exception as e:
                row = dict(model=key, precision=mode, error=f"{type(e).__name__}: {str(e)[:300]}")
                log(traceback.format_exc()[-1500:])
            rows.append(row)
            log(json.dumps(row))
            torch.cuda.empty_cache()
        del enc32, head
    return rows


if __name__ == "__main__":
    import sys
    rows = run(sys.argv[1], sys.argv[2])
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(json.dumps({"status": "done", "env": env(), "rows": rows}, indent=1))
