"""Every number in report/decision_cascade_trials.html, recomputed from saved results only (no network).

Reads results/ here plus the earlier methods' saved results in ../gateway-model-trials/results, prints the tables and
writes results/summary.json."""
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
TRIALS = HERE.parent / "gateway-model-trials"
SEQS = {s["id"]: s for s in json.loads((TRIALS / "sequences.json").read_text())}
SMALL = {k for k in SEQS if "smallfire_rgb" in k}


def jl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()] if Path(path).exists() else []


def right(status, truth):
    return status != "indeterminate" and (status == "fire") == (truth == "fire")


def score(calls):
    """{sequence: status} -> right of 20, small RGB-only fires caught (of 7), false alarms."""
    return dict(right=sum(right(st, SEQS[q]["truth"]) for q, st in calls.items()),
                small=sum(right(st, "fire") for q, st in calls.items() if q in SMALL),
                false_alarms=sum(st == "fire" for q, st in calls.items() if SEQS[q]["truth"] == "no_fire"))


def attempts(row):
    x = row.get("failed_attempts")
    return len(x) if isinstance(x, list) else 0


def raised(row):
    """Cascade call when the measurement rule may raise an escalated window to fire."""
    return "fire" if row.get("path") == "escalated" and row.get("measured_status") == "fire" else row["status"]


out = {}

# single frames: the d3 family from here, Clef, Clef-Flash and Qwen3.8 BF16 from the gateway trials
frames = {r["name"]: r for r in jl(TRIALS / "results/frames_v2b.jsonl")}
frames.update({r["name"]: r for r in jl(RES / "frames_v2b.jsonl")})
out["single_frame"] = {}
for base in ("d3", "d3_rtx", "d3_flash", "d3_mini", "clef", "clef_flash", "qwen38-27b-bf16"):
    row = {}
    for suf, key in (("", "seed0"), ("+guide", "seed0_guided"), ("@seed1", "seed1"), ("+guide@seed1", "seed1_guided")):
        f = frames.get(base + suf)
        if f:
            row[key] = dict(recall=f["recall"], false_alarms=f["false_alarms"], negatives=f["negatives"], auc=f["auc"],
                            median_s=f["median_latency_s"], small_fires=f["by_group"].get("Florida fire, small"))
    rep = RES / f"frames_v2b_{base}#rep2.csv"
    if rep.exists():
        a, b = pd.read_csv(RES / f"frames_v2b_{base}.csv"), pd.read_csv(rep)
        m = a.merge(b, on="uid", suffixes=("_1", "_2"))
        row["repeat_answers_changed"] = f"{int((m.fire_1 != m.fire_2).sum())} of {len(m)}"
    out["single_frame"][base] = row

# fire status on the 20 sequences
seq = {}
v = pd.DataFrame(jl(RES / "sequence_variants.jsonl")).drop_duplicates(["seq", "decider", "variant"], keep="last")
for (dec, var), g in v.groupby(["decider", "variant"]):
    seq[f"{dec} {var}"] = dict(**score(dict(zip(g.seq, g.status))), median_s=round(g.seconds.median(), 2))
S = json.loads((TRIALS / "results/summary_v2.json").read_text())
for k in ("kai", "clef", "clef_flash"):
    seq[f"{k} ends (gateway trials)"] = dict(**score({r["seq"]: r[k] for r in S["per_sequence"]}),
                                            median_s=S["deciders"][k]["seconds_median"])
v2 = [r for r in jl(TRIALS / "results/briefs_v2.jsonl") if r.get("ok")]
bf16 = {r["seq"]: r["llm"]["status"] for r in v2 if r["route"] == "qwen38-27b-bf16#r1"}
seq["qwen38-27b-bf16 writeup call (gateway trials)"] = score(bf16)
rule = {r["seq"]: {"fire": "fire", "no fire signal": "no fire or smoke"}.get(r["measured"], "indeterminate")
        for r in v2 if r["route"] == "qwen38-27b-bf16+guide#r1"}
seq["measurement rule, guided bf16 checks (gateway trials)"] = score(rule)
v3 = [r for r in jl(RES / "briefs_v3.jsonl") if r.get("ok")]
seq["qwen38-27b-bf16 writeup call (rerun)"] = score({r["seq"]: r["llm"]["status"] for r in v3})
out["rerun_reproduces_writeup_calls"] = f"{sum(bf16[r['seq']] == r['llm']['status'] for r in v3)} of {len(v3)}"
for k in ("d3", "d3_mini"):
    calls = {r["seq"]: r["deciders"][k]["status"] for r in v3 if k in r["deciders"]}
    seq[f"{k} in-brief decider"] = dict(**score(calls), median_s=round(median(r["deciders"][k]["seconds"] for r in v3), 2))
c = pd.DataFrame([r for r in jl(RES / "cascade.jsonl") if r.get("ok")]).drop_duplicates(["seq", "decider_name"], keep="last")
out["cascade"] = {}
for name, g in c.groupby("decider_name"):
    rows = g.to_dict("records")
    dec = [r for r in rows if r["path"] == "decision"]
    esc = [r for r in rows if r["path"] == "escalated"]
    seq[f"cascade {name} (llm final call)"] = score({r["seq"]: r["status"] for r in rows})
    seq[f"cascade {name} (rule may raise fire)"] = score({r["seq"]: raised(r) for r in rows})
    out["cascade"][name] = dict(escalated=sorted(r["seq"] for r in esc),
                                decided_right=f"{sum(right(r['status'], r['truth']) for r in dec)} of {len(dec)}",
                                confident_misses=[(r["seq"], r["decider"]["top_p"], r["decider"]["p_fire_visible"])
                                                  for r in dec if not right(r["status"], r["truth"])],
                                decided_median_s=round(median(r["seconds"] for r in dec), 2),
                                escalated_median_s=round(median(r["seconds"] for r in esc), 2) if esc else None,
                                failed_attempts=sum(attempts(r) for r in rows))
out["sequences"] = seq

# the full Florida burn
fb = [r for r in jl(RES / "fullburn.jsonl") if r.get("ok")]
burn = dict(windows=len(fb), images=sum(r["n"] for r in fb), errors=len(jl(RES / "fullburn.jsonl")) - len(fb),
            truth_to_call=Counter(f"{r['truth']} -> {raised(r)}" for r in fb),
            escalated=sum(r["path"] == "escalated" for r in fb),
            escalated_why=Counter(f"{r['truth']}, d3 said {r['decider']['status']}" for r in fb if r["path"] == "escalated"),
            small_fire_windows=f"{sum(raised(r) == 'fire' for r in fb if r['min_hot'] is not None and r['min_hot'] < 0.005)} of "
                               f"{sum(1 for r in fb if r['min_hot'] is not None and r['min_hot'] < 0.005)} called fire",
            total_min=round(sum(r["seconds"] for r in fb) / 60, 1),
            s_per_image=round(sum(r["seconds"] for r in fb) / sum(r["n"] for r in fb), 2),
            s_per_window=round(sum(r["seconds"] for r in fb) / len(fb), 1),
            failed_attempts=sum(attempts(r) for r in fb))
groups = defaultdict(list)
for r in fb:
    groups[r["plot"]].append(r)
    groups[f"{r['plot']} {r['phase']}"].append(r)
    groups[r["path"]].append(r)
burn["by_part"] = {k: dict(windows=len(g), images=sum(r["n"] for r in g), escalated=sum(r["path"] == "escalated" for r in g),
                           calls=Counter(raised(r) for r in g), minutes=round(sum(r["seconds"] for r in g) / 60, 1),
                           s_per_image=round(sum(r["seconds"] for r in g) / sum(r["n"] for r in g), 2),
                           median_window_s=round(median(r["seconds"] for r in g), 1),
                           median_d3_call_s=round(median(r["decider"]["seconds"] for r in g), 2))
                   for k, g in sorted(groups.items())}
out["full_burn"] = burn

(RES / "summary.json").write_text(json.dumps(out, indent=1, default=str))
print("== single frame (seed0 recall | AUC | small fires | guided recall | median s; seed1 recall)")
for k, v in out["single_frame"].items():
    a, g, b = v.get("seed0", {}), v.get("seed0_guided", {}), v.get("seed1", {})
    print(f"  {k:16s} {a.get('recall')} | {a.get('auc')} | {a.get('small_fires')} | {g.get('recall')} | {a.get('median_s')}; "
          f"{b.get('recall')}  repeat {v.get('repeat_answers_changed', '-')}")
print("== fire status on 20 sequences (right | small of 7 | false alarms | median s)")
for k, v in sorted(seq.items(), key=lambda kv: -kv[1]["right"]):
    print(f"  {k:52s} {v['right']:2d} | {v['small']} | {v['false_alarms']} | {v.get('median_s', '')}")
print("== cascade", json.dumps(out["cascade"], indent=1))
print("== rerun reproduces the writeup call on", out["rerun_reproduces_writeup_calls"])
print("== full burn", json.dumps({k: v for k, v in burn.items() if k != "by_part"}, default=str))
for k, v in burn["by_part"].items():
    print(f"  {k:20s} {json.dumps(v, default=str)}")
