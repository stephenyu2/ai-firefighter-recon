"""Generation speed per LLM route through the LiteLLM gateway (streaming): time to first token, decode tokens/s for one
request at a time, and total tokens/s with 8 requests at once. Fixed prompt, 384 output tokens, temperature 0.
Usage: source gateway_env.sh; python throughput.py ROUTE [ROUTE ...]   Appends to results/throughput.jsonl."""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

HERE = Path(__file__).parent
URL = os.environ["RECON_LLM_URL"] + "/chat/completions"
H = {"Authorization": f"Bearer {os.environ['RECON_GATEWAY_KEY']}"}
PROMPT = ("Write a detailed field note, about 350 words, describing how a prescribed burn moves through a longleaf pine "
          "stand over one afternoon: ignition pattern, fuels, smoke, wind shifts, and what the crew watches for.")


def one(route):
    body = {"model": route, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 384, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); first = None; n = None
    with requests.post(URL, headers=H, json=body, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            j = json.loads(line[6:])
            if j.get("usage"):
                n = j["usage"]["completion_tokens"]
            if first is None and any((c.get("delta") or {}).get("content") for c in j.get("choices", [])):
                first = time.perf_counter()
    end = time.perf_counter()
    return dict(ttft_s=first - t0, total_s=end - t0, tokens=n, decode_tok_s=(n - 1) / (end - first) if n and first else None)


for route in sys.argv[1:]:
    one(route)  # warm-up
    seq = [one(route) for _ in range(3)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(8) as pool:
        par = list(pool.map(lambda _: one(route), range(8)))
    wall = time.perf_counter() - t0
    med = lambda xs: sorted(xs)[len(xs) // 2]
    out = dict(route=route, ttft_s=round(med([s["ttft_s"] for s in seq]), 3), decode_tok_s_c1=round(med([s["decode_tok_s"] for s in seq]), 1),
               tokens=seq[0]["tokens"], aggregate_tok_s_c8=round(sum(p["tokens"] for p in par) / wall, 1),
               per_request_tok_s_c8=round(med([p["decode_tok_s"] for p in par]), 1))
    with open(HERE / "results/throughput.jsonl", "a") as fh:
        fh.write(json.dumps(out) + "\n")
    print(json.dumps(out), flush=True)
