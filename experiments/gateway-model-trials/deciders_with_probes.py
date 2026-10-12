"""Fair decider comparison: Kai, Clef and Clef-Flash get the same fixed image probes the LLM writeup gets (first, middle
and last frame; smoke and flames visible), taken from one reference route so every decider reads identical evidence.
Usage: source gateway_env.sh; python deciders_with_probes.py [REFERENCE_LABEL]   (default qwen38-27b-bf16#r1, a v2 run)
Appends to results/deciders_probes_v2.jsonl."""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from recon_gateway import deciders as D, tools as T  # noqa: E402
from recon_gateway.sequence import alias  # noqa: E402
from run_briefs import SEQS, load  # noqa: E402
from runs_store import load_run  # noqa: E402

REF = sys.argv[1] if len(sys.argv) > 1 else "qwen38-27b-bf16#r1"
ep = T.Endpoints(run_tag=f"v2/deciders-with-probes")
for s in SEQS:
    run = load_run("v2", REF, s["id"])
    ev = run["evidence"]
    seq, _ = alias(load(s))
    for name in ("kai", "clef", "clef_flash"):
        r = D.decide(name, ev, seq, ep.h, probes=ev["probes"])
        row = dict(seq=s["id"], truth=s["truth"], thermal=s["thermal"], decider=name, status=r["status"],
                   p_fire=r["status_probabilities"].get("fire"), seconds=r["seconds"], reference=REF, state=r["state"])
        with open(HERE / "results/deciders_probes_v2.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(s["id"], name, r["status"], r["status_probabilities"].get("fire"), flush=True)
print("DECIDERS_WITH_PROBES_DONE")
