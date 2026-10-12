"""Checks each LiteLLM model route for the four things the recon gateway needs: plain reply (no thinking text),
image + log-probs, a tool call, and JSON-schema output.
Usage: SMOKE_IMAGE=frame.jpg python check_routes.py ROUTE [ROUTE ...]   (any aerial JPEG works as the test image)"""
import base64, json, os, sys, time
from pathlib import Path

import requests

BASE = os.environ.get("RECON_LLM_URL", "http://127.0.0.1:4000/v1").rstrip("/") + "/chat/completions"
H = {"Authorization": f"Bearer {os.environ['LITELLM_MASTER_KEY']}"}
IMG = base64.b64encode(Path(os.environ["SMOKE_IMAGE"]).read_bytes()).decode()


def call(route, **body):
    t0 = time.perf_counter()
    r = requests.post(BASE, headers=H, json={"model": route, "temperature": 0, **body}, timeout=900)
    return r, round(time.perf_counter() - t0, 2)


for route in sys.argv[1:]:
    out = {"route": route}
    r, s = call(route, messages=[{"role": "user", "content": "Reply with one word: ready"}], max_tokens=16)
    m = r.json()["choices"][0]["message"] if r.ok else {}
    out["plain"] = (r.status_code, repr((m.get("content") or "")[:40]), "reasoning" if m.get("reasoning_content") or (m.get("provider_specific_fields") or {}).get("reasoning") else "no reasoning", s)
    r, s = call(route, max_tokens=1, logprobs=True, top_logprobs=5, messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{IMG}"}},
        {"type": "text", "text": "Are flames or glowing fire visible in this aerial image?\nA. yes\nB. no\nAnswer with only the letter."}]}])
    if r.ok:
        top = r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        out["image_logprobs"] = (r.status_code, [(t["token"], round(t["logprob"], 3)) for t in top[:3]], s)
    else:
        out["image_logprobs"] = (r.status_code, r.text[:200], s)
    tools = [{"type": "function", "function": {"name": "detect_fire", "description": "Run the fire detector on frames.",
                                               "parameters": {"type": "object", "properties": {"frame_ids": {"type": "array", "items": {"type": "string"}}}}}}]
    r, s = call(route, tools=tools, tool_choice="auto", max_tokens=256, messages=[
        {"role": "system", "content": "Use the tools. Always call detect_fire first."},
        {"role": "user", "content": "Sequence with frames f1, f2, f3. Analyze it."}])
    tc = r.json()["choices"][0]["message"].get("tool_calls") if r.ok else None
    out["tool_call"] = (r.status_code, [(c["function"]["name"], c["function"]["arguments"]) for c in (tc or [])], s)
    schema = {"type": "object", "properties": {"status": {"type": "string", "enum": ["fire", "no fire"]}, "why": {"type": "string"}},
              "required": ["status", "why"]}
    r, s = call(route, max_tokens=200, response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": schema}},
                messages=[{"role": "user", "content": "Thermal shows 400 C hot spots. Give the status as JSON."}])
    try:
        out["json_schema"] = (r.status_code, json.loads(r.json()["choices"][0]["message"]["content"]), s)
    except Exception as e:
        out["json_schema"] = (r.status_code, f"not JSON: {r.text[:200]}", s)
    print(json.dumps(out), flush=True)
