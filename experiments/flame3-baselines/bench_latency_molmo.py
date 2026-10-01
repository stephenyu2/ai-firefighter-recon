"""Clean per-frame latency for Molmo 2 (run in .venv-molmo). Appends to results/latency.csv."""
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

DEV = "mps"; N_WARM = 3
df = pd.read_csv("cache/index_folds.csv")
frames = pd.concat([df[df.y == 1].sample(13, random_state=1), df[df.y == 0].sample(12, random_state=1)])
frames = pd.concat([frames.head(N_WARM), frames])
PROMPT = ("Is there any fire (flames, or smoke from something burning) visible in this aerial drone image? "
          "Answer with only yes or no.")
OUT = "results/latency.csv"
rows = pd.read_csv(OUT).to_dict("records") if os.path.exists(OUT) else []
done = {r["model"] for r in rows}
for key, mid in [("V1_molmo2_4b", "allenai/Molmo2-4B"), ("V2_molmo2_8b", "allenai/Molmo2-8B")]:
    if key in done:
        continue
    t = time.perf_counter(); pr = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
    m = AutoModelForImageTextToText.from_pretrained(mid, trust_remote_code=True, dtype=torch.bfloat16).to(DEV).eval()
    torch.mps.synchronize(); ls = time.perf_counter() - t
    times = []
    for i, r in enumerate(frames.itertuples()):
        t = time.perf_counter()
        msgs = [{"role": "user", "content": [dict(type="image", image=Image.open(r.rgb).convert("RGB")), dict(type="text", text=PROMPT)]}]
        x = pr.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.inference_mode():
            m(**{k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in x.items()}).logits[0, -1]
        torch.mps.synchronize()
        if i >= N_WARM:
            times.append(time.perf_counter() - t)
    mem = torch.mps.driver_allocated_memory() / 1e9
    rows.append(dict(model=key, load_s=round(ls, 1), ms_median=1000 * np.median(times), ms_p90=1000 * np.percentile(times, 90),
                     gpu_mem_gb=round(mem, 2), note="bf16, one forward pass for the yes/no token"))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"{key} load {ls:.1f}s median {1000*np.median(times):.1f} ms mem {mem:.1f} GB", flush=True)
    del m; import gc; gc.collect(); torch.mps.empty_cache()
