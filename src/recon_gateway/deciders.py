"""Typed decision models as the deciders for fire status, confidence and whether direction is assessable.
Evidence they read: detector and CLIP frame counts and strongest detector score, thermal hot-pixel counts and the
largest hot area, and camera motion from spread tracking. They do not see the tracker's trend or direction, the image
probes, or the LLM's answers, so their direction_assessable answer does not use the tracker's direction result.

Kai 0.6B reads the evidence as sentences. Clef (27B) and Clef-Flash (9B) read the same sentences plus the first and
last frames. All three use the SystemOne request format and are called through the gateway (RECON_KAI_URL,
RECON_CLEF_URL, RECON_CLEF_FLASH_URL). They see measurements only (detector, CLIP, thermal, spread tracking), not the
LLM's own answers, so their calls do not depend on which LLM ran the brief."""
import os
import time

import requests

from .kai_decider import QUESTIONS, state_text

STATUS_TO_BRIEF = {"fire": "fire", "smoke_only": "smoke only", "none": "no fire or smoke", "unclear": "indeterminate"}
URLS = {"kai": "RECON_KAI_URL", "clef": "RECON_CLEF_URL", "clef_flash": "RECON_CLEF_FLASH_URL"}
SEES_IMAGES = {"kai": False, "clef": True, "clef_flash": True}
MODEL_NAMES = {"kai": "Decision-2.0-Kai-0.6B", "clef": "clef", "clef_flash": "clef-flash"}  # Kai's server accepts only its own name


def available():
    return [name for name, env in URLS.items() if os.environ.get(env)]


def decide(name, evidence, seq, headers, probes=None):
    """probes: optional fixed image-probe results (agent.PROBE_QUESTIONS on the first, middle and last frame) to verbalize
    for the decider, so it reads the same probe evidence the LLM writeup gets. Default: measurements only."""
    measured = {k: v for k, v in evidence.items() if k not in ("decisions", "probes", "planner_probes")}  # measurements only
    if probes:
        measured["decisions"] = probes  # kai_decider.state_text verbalizes question, answer and confidence per probe
    body = {"model": MODEL_NAMES[name], "state": state_text(measured), "questions": QUESTIONS}
    if SEES_IMAGES[name]:
        body["images"] = [f.rgb_b64(max_side=640) for f in (seq.frames[0], seq.frames[-1])]
    t0 = time.perf_counter()
    r = None
    for attempt in range(30):  # retry cold starts and 5xx only; a 4xx is a bad request and fails at once
        try:
            r = requests.post(os.environ[URLS[name]], headers=headers, json=body, timeout=600)
            if r.status_code not in (408, 429) and r.status_code < 500:
                break
        except requests.RequestException:
            pass
        time.sleep(10)
    if r is None or r.status_code != 200:
        raise RuntimeError(f"{name} decider failed: {None if r is None else (r.status_code, r.text[:300])}")
    a = r.json()["answers"]
    conf = a["confidence"]
    return {"decider": name, "sees_images": SEES_IMAGES[name], "status": STATUS_TO_BRIEF[a["status"]["choice"]],
            "status_probabilities": {k: round(v, 3) for k, v in a["status"]["probabilities"].items()},
            "direction_assessable_p": round(a["direction_assessable"]["noul"], 3),
            "confidence": (conf.get("legend") or {}).get(str(round(conf["score"]))) if "legend" in conf else None,
            "seconds": round(time.perf_counter() - t0, 3), "server_ms": r.json().get("timing_ms"), "state": body["state"]}
