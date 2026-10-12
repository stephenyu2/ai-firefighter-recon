"""Runs the recon gateway (agent mode) on every test sequence through the LiteLLM gateway, one LLM route per thread.
Every brief also gets the Kai, Clef and Clef-Flash decider calls, so "who decides fire status" can be compared on
identical evidence. Usage: source gateway_env.sh; python run_briefs.py ROUTE [ROUTE ...]
Writes results/runs/<route>/<sequence>.json and appends one summary line per brief to results/briefs.jsonl.
Briefs that already have a runs/ file are skipped, so rerunning after an interruption only does what is missing.
RECON_GUIDANCE=1 turns on the observer guidance and labels the route "<route>+guide".
REPEATS=N runs every brief N times (labels "<route>#r1".."#rN") to measure repeatability.
VERSION=v2 writes to results/runs/v2/ and results/briefs_v2.jsonl so a changed agent never mixes with earlier results.
Sequence folders in sequences.json are relative to RECON_DATA_ROOT (default: the repo's data/ folder).
Every request carries an x-recon-run tag (version/label) so the gateway log can be split by run."""
import json, os, sys, threading, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from recon_gateway import agent  # noqa: E402
from recon_gateway.sequence import from_flame_index, from_folder  # noqa: E402

SEQS = json.loads((HERE / "sequences.json").read_text())
INDEX = HERE.parents[1] / "training/flame3-lora/index.csv"
DATA = Path(os.environ.get("RECON_DATA_ROOT", HERE.parents[1] / "data"))
FLAME_ROOT = DATA / "flame3_cv"
RESULTS = HERE / "results"
DECIDERS = ["kai", "clef", "clef_flash"]
LOCK = threading.Lock()
G = os.environ.get("RECON_GUIDANCE", "")  # "1"/"all" = every placement; or a comma list of planner, decide, writeup
_bad = {p.strip() for p in G.split(",") if p.strip()} - set(agent.GUIDANCE_PLACES) - {"1", "all", "true", "0"}
if _bad:
    sys.exit(f"unknown RECON_GUIDANCE placement(s) {sorted(_bad)}; use all or {list(agent.GUIDANCE_PLACES)}")
FLAGS = agent.guidance_flags(G)
ON = [p for p in agent.GUIDANCE_PLACES if FLAGS[p]]
LABEL_SUFFIX = "" if not ON else ("+guide" if len(ON) == len(agent.GUIDANCE_PLACES) else "+g-" + "-".join(ON))
REPEATS = int(os.environ.get("REPEATS", "1"))
VERSION = os.environ.get("VERSION", "")
RUNS = RESULTS / "runs" / VERSION if VERSION else RESULTS / "runs"
LOG = RESULTS / (f"briefs_{VERSION}.jsonl" if VERSION else "briefs.jsonl")


def load(s):
    if s["kind"] == "flame":
        return from_flame_index(INDEX, FLAME_ROOT, group=s["group"], max_frames=8)
    return from_folder(DATA / s["rgb_dir"], DATA / s["tiff_dir"] if s["tiff_dir"] else None, seq_id=s["id"], interval_s=s["interval_s"])


def worker(route):
    for rep in range(1, REPEATS + 1):
        brief_pass(route, rep)


def brief_pass(route, rep):
    label = route + LABEL_SUFFIX + (f"#r{rep}" if REPEATS > 1 else "")
    out = RUNS / label
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{VERSION or 'v1'}/{label}"
    for s in SEQS:
        if (out / f"{s['id']}.json").exists():
            continue
        t0 = time.time()
        try:
            res = agent.run(load(s), mode="agent", llm_model=route, deciders=DECIDERS, run_tag=f"{tag}/{s['id']}",
                            guidance=",".join(ON))
            (out / f"{s['id']}.json").write_text(json.dumps(res, indent=1))
            pl = res.get("planner") or {}
            calls = [c for t in pl.get("turns", []) for c in t["tool_calls"]]
            wr = next((t for t in res["trace"] if t["tool"] == "llm_writeup"), {})
            row = dict(route=label, model_route=route, guided=bool(ON), guidance=FLAGS, repeat=rep, version=res.get("version", VERSION or "v1"),
                       seq=s["id"], site=s["site"], thermal=s["thermal"], truth=s["truth"], ok=True,
                       llm=res["llm_decision"], measured=res["measured_decision"]["status"], measured_conf=res["measured_decision"]["confidence"],
                       final=res["brief"]["fire_smoke"]["status"],
                       deciders={k: {"status": v["status"], "p_fire": v["status_probabilities"].get("fire"), "seconds": v["seconds"],
                                     "server_ms": v.get("server_ms")} for k, v in res["deciders"].items()},
                       n_policy_fixes=len(res["policy_corrections"]), policy_fixes=res["policy_corrections"],
                       status_overridden=any(f.startswith("fire status") for f in res["policy_corrections"]),
                       planner_rounds=pl.get("rounds"), planner_finished=pl.get("finished"), planner_exhausted=pl.get("exhausted"),
                       planner_finish_reasons=[t["finish_reason"] for t in pl.get("turns", [])],
                       planner_decides=[c["args"]["frame_id"] for c in calls if c["name"] == "decide" and not c["error"]],
                       planner_errors=[c["error"] for c in calls if c["error"]],
                       writeup_finish_reason=wr.get("finish_reason"), writeup_s=wr.get("seconds"),
                       total_s=round(time.time() - t0, 1))
        except Exception as e:
            row = dict(route=label, seq=s["id"], ok=False, error=f"{type(e).__name__}: {str(e)[:300]}", trace=traceback.format_exc()[-800:],
                       total_s=round(time.time() - t0, 1))
        with LOCK:
            with open(LOG, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(label, s["id"], "ok" if row["ok"] else row["error"], row.get("llm", {}).get("status") if row["ok"] else "",
                  row["total_s"], flush=True)


if __name__ == "__main__":
    threads = [threading.Thread(target=worker, args=(r,)) for r in sys.argv[1:]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("ALL_BRIEFS_DONE", sys.argv[1:])
