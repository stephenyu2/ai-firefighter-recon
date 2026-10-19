"""Decision-model-first cascade for fire status.

1. The gateway measures the sequence (detector, CLIP, thermal, spread) as usual.
2. A SystemOne decision model (d3 or d3-mini) reads every frame, the observer guidance, and the thermal facts only. The
   RGB detector's counts are left out: in the d3 trials they anchored the model toward "no fire" on small fires.
3. If the decision model's top status probability is at least ACCEPT_P and the status is not "unclear", its answer is
   the call.
4. Otherwise Qwen3.8-27B runs a deeper agent pass: guided image checks on every frame (not just first, middle, last),
   the decision model's probabilities attached as evidence, the planner, the writeup and the policy checks. A failed
   pass (HTTP error, unparsable writeup) is retried up to RETRIES times.
ACCEPT_P and RETRIES were fixed before any cascade run, not tuned on the test sequences."""
import os
import time

import requests

from . import agent, deciders as D, tools as T
from .kai_decider import QUESTIONS
from .prompts import WATCH_FOR
from .sequence import alias

ACCEPT_P = 0.80
RETRIES = 2
FRAMES_NOTE = "The images are the frames of one drone flight, in time order."


def thermal_facts(det):
    n_th, f_th = det.get("thermal_frames", 0), det.get("thermal_fire_frames", 0)
    if not n_th:
        return "No thermal camera data is available."
    hot = [f["thermal_hot_percent"] for f in det.get("frames", []) if f.get("thermal_hot_percent")]
    return (f"The thermal camera covers {n_th} of the frames; {f_th} of them show pixels at or above 80 C"
            + (f", covering up to {max(hot):.2g}% of the frame." if hot else "."))


def questions():
    guide = f"\n{FRAMES_NOTE}\n\nObserver guidance:\n{WATCH_FOR}"
    status = dict(QUESTIONS["status"], instructions=QUESTIONS["status"]["instructions"] + guide)
    visible = {"type": "noul", "instructions": "Is active fire (flame, glow or embers) visible in any of the images?" + guide}
    return {"status": status, "fire_visible": visible}


def ask(name, seq, det, headers):
    body = {"model": D.MODEL_NAMES[name],
            "state": f"{len(seq.frames)} aerial frames from one drone flight, in time order. {thermal_facts(det)}",
            "questions": questions(),
            "images": [D.media(name, f.rgb_b64(max_side=640)) for f in seq.frames]}
    r, t0 = None, time.perf_counter()
    for _ in range(30):  # cold starts and 408/429/5xx only
        try:
            r = requests.post(os.environ[D.URLS[name]], headers=headers, json=body, timeout=900)
            if r.status_code not in (408, 429) and r.status_code < 500:
                break
        except requests.RequestException:
            pass
        time.sleep(10)
    if r is None or r.status_code != 200:
        raise RuntimeError(f"{name}: {None if r is None else (r.status_code, r.text[:300])}")
    a = r.json()["answers"]
    probs = a["status"]["probabilities"]
    return {"decider": name, "status": D.STATUS_TO_BRIEF[a["status"]["choice"]],
            "status_probabilities": {k: round(v, 4) for k, v in probs.items()}, "top_p": round(max(probs.values()), 4),
            "p_fire_visible": round(a["fire_visible"]["noul"], 4), "seconds": round(time.perf_counter() - t0, 3),
            "server_ms": r.json().get("timing_ms")}


def run(seq, decider="d3", llm_model="qwen38-27b-bf16", run_tag=None, accept_p=ACCEPT_P):
    """One window through the cascade. Returns the decider's answer, the path ("decision" or "escalated"), the final
    status and, when escalated, the LLM's own status, the measured status and the full agent run under "run"."""
    t0 = time.time()
    aliased, _ = alias(seq)
    ep = T.Endpoints(llm_model=llm_model, run_tag=run_tag)
    det = T.detect_fire(ep, aliased, {})
    first = ask(decider, aliased, det, ep.h)
    out = {"decider": first, "accept_p": accept_p, "thermal_frames": det.get("thermal_frames", 0)}
    if first["top_p"] >= accept_p and first["status"] != "indeterminate":
        return {**out, "path": "decision", "status": first["status"], "seconds": round(time.time() - t0, 1)}
    extra = {"decision_model": {"name": decider, "status_probabilities": first["status_probabilities"],
                                "p_active_fire_visible": first["p_fire_visible"]}}
    attempts = []
    for attempt in range(1 + RETRIES):
        try:
            res = agent.run(seq, mode="agent", llm_model=llm_model, guidance="decide", run_tag=run_tag,
                            probe_all=True, extra_evidence=extra)
            break
        except Exception as e:  # noqa: BLE001 - recorded and retried
            attempts.append(f"{type(e).__name__}: {str(e)[:200]}")
            res = None
    if res is None:
        return {**out, "path": "escalated", "status": "indeterminate", "failed_attempts": attempts,
                "seconds": round(time.time() - t0, 1)}
    return {**out, "path": "escalated", "status": res["brief"]["fire_smoke"]["status"],
            "llm_status": res["llm_decision"]["status"], "measured_status": res["measured_decision"]["status"],
            "policy_corrections": res["policy_corrections"], "failed_attempts": attempts, "run": res,
            "seconds": round(time.time() - t0, 1)}
