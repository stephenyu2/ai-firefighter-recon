"""Sequence-level fire status from the d3 decision models with the images passed directly, through the gateway.

Every variant reads the same 20 test sequences (../gateway-model-trials/sequences.json) and, where it uses
measurements, the same measured evidence as the earlier decider runs, taken from the v2 reference brief run
qwen38-27b-bf16#r1 (like ../gateway-model-trials/deciders_with_probes.py):
  ends          measurements as sentences + first and last frame (exactly what Kai, Clef and Clef-Flash got)
  all8+guide    measurements as sentences + all 8 frames in time order + the observer guidance
  vision+guide  all 8 frames + the observer guidance, no measurements (pure vision)
Questions: the gateway's status choice (fire / smoke only / none / unclear), whether direction is assessable, a
confidence score, and a yes/no "active fire visible in any frame". Frame and sequence names never reach the model.
Usage: source ../gateway-model-trials/gateway_env.sh; python d3_sequences.py NAME [NAME ...]
       (NAME = d3, d3_rtx, d3_mini or d3_flash; see recon_gateway.deciders)
Appends one line per sequence, model and variant to results/sequence_variants.jsonl."""
import json, os, sys, time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
TRIALS = HERE.parent / "gateway-model-trials"
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(TRIALS))
from recon_gateway import deciders as D  # noqa: E402
from recon_gateway import tools as T  # noqa: E402
from recon_gateway.kai_decider import QUESTIONS, state_text  # noqa: E402
from recon_gateway.prompts import WATCH_FOR  # noqa: E402
from recon_gateway.sequence import alias  # noqa: E402
from run_briefs import SEQS, load  # noqa: E402
from runs_store import load_run  # noqa: E402

REF = "qwen38-27b-bf16#r1"
VARIANTS = ("ends", "all8+guide", "vision+guide")
FRAMES_NOTE = "The images are the frames of one drone flight, in time order."
FIRE_VISIBLE = {"type": "noul", "instructions": "Is active fire (flame, glow or embers) visible in any of the images?"}


def questions(guided):
    q = json.loads(json.dumps(QUESTIONS))
    q["fire_visible"] = dict(FIRE_VISIBLE)
    if guided:
        for k in ("status", "fire_visible"):
            q[k]["instructions"] = f"{q[k]['instructions']}\n{FRAMES_NOTE}\n\nObserver guidance:\n{WATCH_FOR}"
    return q


def request(name, variant, ev, seq):
    measured = {k: v for k, v in ev.items() if k not in ("decisions", "probes", "planner_probes")}
    frames = (seq.frames[0], seq.frames[-1]) if variant == "ends" else seq.frames
    state = (state_text(measured) if variant != "vision+guide"
             else f"{len(seq.frames)} aerial frames from one drone flight, in time order.")
    return {"model": D.MODEL_NAMES[name], "state": state, "questions": questions(variant != "ends"),
            "images": [D.media(name, f.rgb_b64(max_side=640)) for f in frames]}


def call(url, headers, body):
    r = None
    for _ in range(30):  # retry cold starts and 408/429/5xx; any other 4xx fails at once
        try:
            r = requests.post(url, headers=headers, json=body, timeout=900)
            if r.status_code not in (408, 429) and r.status_code < 500:
                break
        except requests.RequestException:
            pass
        time.sleep(10)
    if r is None or r.status_code != 200:
        raise RuntimeError(f"request failed: {None if r is None else (r.status_code, r.text[:300])}")
    return r.json()


def main(names):
    for name in names:
        url = os.environ[D.URLS[name]]
        ep = T.Endpoints(run_tag=f"v3/d3-sequences/{name}")
        for s in SEQS:
            ev = load_run("v2", REF, s["id"])["evidence"]
            seq, _ = alias(load(s))
            for variant in VARIANTS:
                body = request(name, variant, ev, seq)
                t0 = time.perf_counter()
                out = call(url, ep.h, body)
                a = out["answers"]
                row = dict(seq=s["id"], truth=s["truth"], thermal=s["thermal"], decider=name, variant=variant,
                           images=len(body["images"]), status=D.STATUS_TO_BRIEF[a["status"]["choice"]],
                           status_probabilities={k: round(v, 4) for k, v in a["status"]["probabilities"].items()},
                           p_fire_visible=round(a["fire_visible"]["noul"], 4), confidence=a.get("confidence"),
                           seconds=round(time.perf_counter() - t0, 3), server_ms=out.get("timing_ms"),
                           input_tokens=(out.get("usage") or {}).get("input_tokens"), reference=REF)
                with open(HERE / "results/sequence_variants.jsonl", "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(name, variant, s["id"], row["status"], row["p_fire_visible"], row["seconds"], flush=True)
    print("D3_SEQUENCES_DONE", names)


if __name__ == "__main__":
    main(sys.argv[1:])
