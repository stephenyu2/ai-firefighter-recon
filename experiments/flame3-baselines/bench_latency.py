"""Clean per-frame latency (batch size 1, end to end: read image, preprocess, forward, GPU sync).
Run with nothing else on the GPU. Usage: python bench_latency.py [vlm]  (vlm = include the big VLMs)"""
import os
import sys
import time

import numpy as np
import pandas as pd
import tifffile
import torch
from PIL import Image

from common import load_index, load_thermal_img

DEV = "mps"
N_WARM, N_TIME = 3, 25
df = load_index()
frames = pd.concat([df[df.y == 1].sample(13, random_state=1), df[df.y == 0].sample(12, random_state=1)])
frames = pd.concat([frames.head(N_WARM), frames])
OUT = "results/latency.csv"
rows = pd.read_csv(OUT).to_dict("records") if os.path.exists(OUT) else []
done = {r["model"] for r in rows}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def sync():
    if DEV == "mps":
        torch.mps.synchronize()


def record(name, load_s, fn, note=""):
    if name in done:
        return
    times = []
    for i, r in enumerate(frames.itertuples()):
        t = time.perf_counter()
        with torch.inference_mode():
            fn(r)
        sync()
        if i >= N_WARM:
            times.append(time.perf_counter() - t)
    mem = torch.mps.driver_allocated_memory() / 1e9 if DEV == "mps" else float("nan")
    rows.append(dict(model=name, load_s=round(load_s, 1), ms_median=1000 * np.median(times),
                     ms_p90=1000 * np.percentile(times, 90), gpu_mem_gb=round(mem, 2), note=note))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"{name:28s} load {load_s:6.1f}s  median {1000*np.median(times):8.1f} ms/frame  mem {mem:5.1f} GB", flush=True)


def px(img, w, h):
    a = np.asarray(img.resize((w, h), Image.BICUBIC), dtype=np.float32) / 255
    return ((torch.from_numpy(a).permute(2, 0, 1)[None] - MEAN) / STD).to(DEV)


def free():
    import gc
    gc.collect(); torch.mps.empty_cache()


# thermal rule (CPU only)
record("B1_maxtemp_rule", 0.0, lambda r: float(tifffile.imread(r.tiff).max()), "CPU, reads the Celsius TIFF")

from transformers import (AutoImageProcessor, AutoModel, AutoModelForImageClassification, AutoProcessor, CLIPModel,
                          CLIPProcessor, Dinov2Model, ViTForImageClassification)

if "Z1_clipL14_zeroshot_rgb" not in done:
    t = time.perf_counter(); m = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEV).eval()
    p = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14"); sync(); ls = time.perf_counter() - t
    record("Z1_clipL14_zeroshot_rgb", ls, lambda r: m.get_image_features(pixel_values=p(images=[Image.open(r.rgb).convert("RGB").resize((224, 224))], return_tensors="pt", do_resize=False, do_center_crop=False)["pixel_values"].to(DEV)), "text prompts embedded once up front")
    del m; free()
if "Z2_siglip2_zeroshot_rgb" not in done:
    t = time.perf_counter(); m = AutoModel.from_pretrained("google/siglip2-so400m-patch14-384").to(DEV).eval()
    p = AutoProcessor.from_pretrained("google/siglip2-so400m-patch14-384"); sync(); ls = time.perf_counter() - t
    record("Z2_siglip2_zeroshot_rgb", ls, lambda r: m.get_image_features(pixel_values=p(images=[Image.open(r.rgb).convert("RGB").resize((384, 384))], return_tensors="pt", do_resize=False, do_center_crop=False)["pixel_values"].to(DEV)))
    del m; free()
for key, name in [("Z3_firevit_edbianchi_rgb", "EdBianchi/vit-fire-detection"), ("Z4_forestfire_siglip2_rgb", "prithivMLmods/Forest-Fire-Detection")]:
    if key in done:
        continue
    t = time.perf_counter(); m = AutoModelForImageClassification.from_pretrained(name).to(DEV).eval()
    p = AutoImageProcessor.from_pretrained(name); sync(); ls = time.perf_counter() - t
    record(key, ls, lambda r, m=m, p=p: m(pixel_values=p(images=[Image.open(r.rgb).convert("RGB")], return_tensors="pt")["pixel_values"].to(DEV)).logits)
    del m; free()
if "P1_dinov2b_probe_rgb" not in done:
    t = time.perf_counter(); m = Dinov2Model.from_pretrained("facebook/dinov2-base").to(DEV).eval(); sync(); ls = time.perf_counter() - t
    record("P1_dinov2b_probe_rgb", ls, lambda r: m(pixel_values=px(Image.open(r.rgb).convert("RGB"), 448, 364)).last_hidden_state, "448x364 features; the logistic-regression step adds under 1 ms")
    record("P2_dinov2b_probe_thermal", 0.0, lambda r: m(pixel_values=px(load_thermal_img({"tiff": r.tiff}), 448, 364)).last_hidden_state, "reads TIFF + 3-window encoding")
    del m; free()
if "L1_dinov2b_lora_rgb" not in done:
    from peft import LoraConfig, get_peft_model
    t = time.perf_counter()
    m = get_peft_model(Dinov2Model.from_pretrained("facebook/dinov2-base"), LoraConfig(r=16, lora_alpha=16, target_modules=["q_proj", "v_proj"])).to(DEV).eval()
    head = torch.nn.Linear(1536, 1).to(DEV); sync(); ls = time.perf_counter() - t
    def f(r):
        h = m(pixel_values=px(Image.open(r.rgb).convert("RGB"), 308, 252)).last_hidden_state
        return head(torch.cat([h[:, 0], h[:, 1:].mean(1)], -1))
    record("L1_dinov2b_lora_rgb", ls, f, "308x252, adapters not merged")
    del m; free()
    t = time.perf_counter()
    m = get_peft_model(ViTForImageClassification.from_pretrained("EdBianchi/vit-fire-detection"), LoraConfig(r=16, lora_alpha=16, target_modules=["q_proj", "v_proj"], modules_to_save=["classifier"])).to(DEV).eval()
    sync(); ls = time.perf_counter() - t
    MEAN5 = torch.tensor([0.5] * 3).view(1, 3, 1, 1)
    record("L2_firevit_lora_rgb", ls, lambda r: m(pixel_values=((torch.from_numpy(np.asarray(Image.open(r.rgb).convert("RGB").resize((224, 224)), dtype=np.float32) / 255).permute(2, 0, 1)[None] - MEAN5) / MEAN5).to(DEV)).logits)
    del m; free()
if "F1_resnet50_fullft_rgb" not in done:
    import torchvision
    t = time.perf_counter(); m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).to(DEV).eval(); sync(); ls = time.perf_counter() - t
    record("F1_resnet50_fullft_rgb", ls, lambda r: m(px(Image.open(r.rgb).convert("RGB"), 320, 256)), "320x256")
    del m; free()

if len(sys.argv) > 1 and sys.argv[1] == "vlm":
    from transformers import AutoModelForImageTextToText
    PROMPT = ("Is there any fire (flames, or smoke from something burning) visible in this aerial drone image? "
              "Answer with only yes or no.")
    for key, mid in [("V3_qwen35_0p8b", "Qwen/Qwen3.5-0.8B"), ("V4_qwen35_2b", "Qwen/Qwen3.5-2B"), ("V5_qwen35_4b", "Qwen/Qwen3.5-4B"),
                     ("V6_qwen35_9b", "Qwen/Qwen3.5-9B"), ("V7_qwen3vl_8b", "Qwen/Qwen3-VL-8B-Instruct"), ("V8_gemma4_e4b", "google/gemma-4-E4B-it")]:
        if key in done:
            continue
        try:
            t = time.perf_counter(); pr = AutoProcessor.from_pretrained(mid)
            m = AutoModelForImageTextToText.from_pretrained(mid, dtype=torch.bfloat16).to(DEV).eval(); sync(); ls = time.perf_counter() - t
            def f(r, m=m, pr=pr):
                msgs = [{"role": "user", "content": [{"type": "image", "image": Image.open(r.rgb).convert("RGB")}, {"type": "text", "text": PROMPT}]}]
                x = pr.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", enable_thinking=False)
                return m(**{k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in x.items()}).logits[0, -1]
            record(key, ls, f, "bf16, one forward pass for the yes/no token")
        except Exception as e:
            print(key, "failed:", e)
        m = None; free()
