"""Runs the decision-model-first cascade (recon_gateway.cascade) on all 20 test sequences through the gateway.
Usage: source ../gateway-model-trials/gateway_env.sh; python cascade_eval.py DECIDER [DECIDER ...]   (d3, d3_mini, ...)
Appends one summary line per sequence to results/cascade.jsonl; escalated runs go to results/runs/v3/cascade-<decider>/
(git-ignored; `RESULTS_DIR=$PWD/results python ../gateway-model-trials/runs_store.py pack v3` packs them into
results/runs_v3.jsonl.gz)."""
import json, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parent / "gateway-model-trials"))
from recon_gateway import cascade  # noqa: E402
from run_briefs import SEQS, load  # noqa: E402

for name in sys.argv[1:]:
    out = HERE / "results" / "runs" / "v3" / f"cascade-{name}"
    out.mkdir(parents=True, exist_ok=True)
    for s in SEQS:
        t0 = time.time()
        try:
            r = cascade.run(load(s), decider=name, run_tag=f"v3/cascade-{name}/{s['id']}")
            run = r.pop("run", None)
            if run:
                (out / f"{s['id']}.json").write_text(json.dumps(run, indent=1))
            row = dict(seq=s["id"], truth=s["truth"], thermal=s["thermal"], ok=True, **r)
        except Exception as e:  # noqa: BLE001 - recorded, the next sequence still runs
            row = dict(seq=s["id"], ok=False, error=f"{type(e).__name__}: {str(e)[:300]}",
                       trace=traceback.format_exc()[-600:], seconds=round(time.time() - t0, 1))
        row["decider_name"] = name
        with open(HERE / "results" / "cascade.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(name, s["id"], row.get("path"), row.get("status"), row.get("seconds"), flush=True)
print("CASCADE_DONE", sys.argv[1:])
