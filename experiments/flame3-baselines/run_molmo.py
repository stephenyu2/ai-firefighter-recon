"""Molmo2-4B zero-shot: P(yes) for 'is there fire' on each corrected-FOV RGB frame. Run in .venv-molmo."""
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None
import os
OUT = os.environ.get("MOLMO_OUT", "results/molmo_raw.csv")
PROMPT = ("Is there any fire (flames, or smoke from something burning) visible in this aerial drone image? "
          "Answer with only yes or no.")
DEV = "mps"
MID = os.environ.get("MOLMO_ID", "allenai/Molmo2-4B")

df = pd.read_csv("cache/index_folds.csv")
if LIMIT:
    df = pd.concat([df[df.y == 1].head(LIMIT), df[df.y == 0].head(LIMIT)])
proc = AutoProcessor.from_pretrained(MID, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(MID, trust_remote_code=True, dtype=torch.bfloat16).to(DEV).eval()
tok = proc.tokenizer


def ids(words):
    out = set()
    for w in words:
        t = tok.encode(w, add_special_tokens=False)
        if len(t) == 1:
            out.add(t[0])
    return sorted(out)


YES, NO = ids(["yes", "Yes", " yes", " Yes", "YES"]), ids(["no", "No", " no", " No", "NO"])
print("yes ids", YES, "no ids", NO, flush=True)
done = set()
try:
    prev = pd.read_csv(OUT); done = set(prev.uid)
except FileNotFoundError:
    prev = None
rows = [] if prev is None or LIMIT else prev.to_dict("records")
t0 = time.time()
for i, r in enumerate(df.itertuples()):
    if r.uid in done and not LIMIT:
        continue
    img = Image.open(r.rgb).convert("RGB")
    msgs = [{"role": "user", "content": [dict(type="image", image=img), dict(type="text", text=PROMPT)]}]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    inputs = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].float()
    lp = torch.log_softmax(logits, -1)
    ly, ln = torch.logsumexp(lp[YES], 0).item(), torch.logsumexp(lp[NO], 0).item()
    p_yes = float(np.exp(ly) / (np.exp(ly) + np.exp(ln)))
    top = tok.decode([int(logits.argmax())])
    rows.append(dict(uid=r.uid, y=r.y, p_yes=p_yes, mass_yes_no=float(np.exp(ly) + np.exp(ln)), top_token=top,
                     n_tokens=int(inputs["input_ids"].shape[1])))
    if LIMIT or (len(rows) % 25 == 0):
        pd.DataFrame(rows).to_csv(OUT if not LIMIT else "results/molmo_smoke.csv", index=False)
        el = time.time() - t0
        print(f"{len(rows)} imgs, {el:.0f}s, last p_yes={p_yes:.3f} y={r.y} top={top!r} tokens={rows[-1]['n_tokens']}", flush=True)
pd.DataFrame(rows).to_csv(OUT if not LIMIT else "results/molmo_smoke.csv", index=False)
print("done", len(rows), f"{time.time()-t0:.0f}s")
