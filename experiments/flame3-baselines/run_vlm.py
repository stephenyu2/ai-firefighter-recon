"""Zero-shot VLM scoring (transformers-native models): P(yes) to 'is there fire' per corrected-FOV RGB frame.
Usage: python run_vlm.py MODEL_ID TAG [LIMIT_PER_CLASS]"""
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MID, TAG = sys.argv[1], sys.argv[2]
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else None
PROMPT = ("Is there any fire (flames, or smoke from something burning) visible in this aerial drone image? "
          "Answer with only yes or no.")
DEV = "mps"
OUT = f"results/vlm_raw_{TAG}.csv" if not LIMIT else f"results/vlm_smoke_{TAG}.csv"

df = pd.read_csv("cache/index_folds.csv")
if LIMIT:
    df = pd.concat([df[df.y == 1].sample(LIMIT, random_state=0), df[df.y == 0].sample(LIMIT, random_state=0)])
proc = AutoProcessor.from_pretrained(MID)
model = AutoModelForImageTextToText.from_pretrained(MID, dtype=torch.bfloat16).to(DEV).eval()
tok = proc.tokenizer


def ids(words):
    out = set()
    for w in words:
        t = tok.encode(w, add_special_tokens=False)
        if len(t) == 1:
            out.add(t[0])
    return sorted(out)


YES, NO = ids(["yes", "Yes", " yes", " Yes", "YES"]), ids(["no", "No", " no", " No", "NO"])
rows, t0 = [], time.time()
for i, r in enumerate(df.itertuples()):
    img = Image.open(r.rgb).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                      return_tensors="pt", enable_thinking=False)
    inputs = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].float()
    lp = torch.log_softmax(logits, -1)
    ly, ln = torch.logsumexp(lp[YES], 0).item(), torch.logsumexp(lp[NO], 0).item()
    rows.append(dict(uid=r.uid, y=r.y, p_yes=float(np.exp(ly) / (np.exp(ly) + np.exp(ln))),
                     mass_yes_no=float(np.exp(ly) + np.exp(ln)), top_token=tok.decode([int(logits.argmax())]),
                     n_tokens=int(inputs["input_ids"].shape[1])))
    if LIMIT or (i + 1) % 50 == 0:
        print(f"{TAG} {i+1}/{len(df)} {time.time()-t0:.0f}s p_yes={rows[-1]['p_yes']:.3f} y={r.y} "
              f"top={rows[-1]['top_token']!r} mass={rows[-1]['mass_yes_no']:.2f} tokens={rows[-1]['n_tokens']}", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"{TAG} done {len(rows)} imgs in {time.time()-t0:.0f}s", flush=True)
