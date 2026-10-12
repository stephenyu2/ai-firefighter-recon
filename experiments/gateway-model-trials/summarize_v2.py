"""Tables for the v2 run -> results/summary_v2.json. Reads only results/, so it runs offline.

Briefs (briefs_v2.jsonl + saved runs, see runs_store.py): 3 repeats per route and sequence; arms are the unguided route,
"+guide" (planner, decide and writeup guidance all on) and any "+g-<places>" arm. Grading is fire vs not fire on the
LLM's own call before the policy step; "indeterminate" is never correct. The final status is recomputed from each saved
raw brief with the current policy (brief.enforce_policy), so every arm is judged by the same rule.
Per-frame (frames_v2b.jsonl if present, else frames_v2.jsonl), deciders with probes (deciders_probes_v2.jsonl),
throughput (throughput.jsonl)."""
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE.parents[1] / "src"))
from recon_gateway.brief import enforce_policy  # noqa: E402
from runs_store import load_run  # noqa: E402

ROUTES = ["qwen38-27b-bf16", "qwen38-27b-fp8", "qwen38-27b-int4", "qwen38-27b-nvfp4", "qwen3vl-4b"]
DECIDERS = ["kai", "clef", "clef_flash"]
SEQS = {s["id"]: s for s in json.loads((HERE / "sequences.json").read_text())}


def grade(status, truth):
    if status == "indeterminate":
        return "indeterminate"
    return "correct" if (status == "fire") == (truth == "fire") else "wrong"


def load(name):
    p = RESULTS / name
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


rows = load("briefs_v2.jsonl")
ok_keys = {(r["route"], r["seq"]) for r in rows if r.get("ok")}
failed = [r for r in rows if not r.get("ok") and (r["route"], r["seq"]) not in ok_keys]
b = pd.DataFrame([r for r in rows if r.get("ok")]).drop_duplicates(["route", "seq"], keep="last")
out = {"n_briefs": len(b), "n_failed_unrecovered": len(failed),
       "failed": [(r["route"], r["seq"], (r.get("error") or "")[:160]) for r in failed],
       "sequences": {"total": len(SEQS), "rgb_only": sum(1 for s in SEQS.values() if not s["thermal"]),
                     "thermal": sum(1 for s in SEQS.values() if s["thermal"])},
       "routes": {}, "deciders": {}, "deciders_with_probes": {}, "guidance": {}, "per_sequence": []}

if len(b):
    b["base"] = b.model_route
    b["arm"] = b.route.str.replace(r"#r\d+$", "", regex=True)
    b["raw"] = b.llm.map(lambda x: x["status"])
    b["g_raw"] = [grade(s, t) for s, t in zip(b.raw, b.truth)]
    finals, overridden, probe_dups = [], [], []
    for r in b.itertuples():
        run = load_run("v2", r.route, r.seq)
        brief, fixes = enforce_policy(copy.deepcopy(run["raw_brief"]), run["evidence"])
        finals.append(brief["fire_smoke"]["status"])
        overridden.append(any(f.startswith("fire status") for f in fixes))
        fixed = run["evidence"]["probes"]
        probe_dups.append(sum(1 for t in (run.get("planner") or {}).get("turns", []) for c in t["tool_calls"]
                              if c["name"] == "decide" and not c["error"] and set(c["args"]["questions"]) <= set(fixed.get(c["args"]["frame_id"], {}))))
    b["final_v21"], b["overridden_v21"], b["probe_dups"] = finals, overridden, probe_dups
    b["g_final"] = [grade(s, t) for s, t in zip(b.final_v21, b.truth)]
    b["g_measured"] = [grade("fire" if s == "fire" else ("indeterminate" if s == "indeterminate" else "no fire"), t)
                       for s, t in zip(b.measured, b.truth)]
    for arm, d in b.groupby("arm"):
        reps = [g for _, g in d.groupby("seq") if len(g) > 1]
        same_raw = sum(1 for g in reps if g.raw.nunique() == 1)
        same_final = sum(1 for g in reps if g.final_v21.nunique() == 1)
        same_plan = sum(1 for g in reps if len({tuple(sorted(x)) for x in g.planner_decides}) == 1)
        th, rgb = d[d.thermal], d[~d.thermal]
        rgb_fire = rgb[rgb.truth == "fire"]
        ov = d.overridden_v21.astype(bool)
        out["routes"][arm] = dict(
            base=d.base.iloc[0], guided=bool(d.guided.iloc[0]), briefs=len(d), sequences=d.seq.nunique(), repeats=int(d.repeat.max()),
            raw_correct=int((d.g_raw == "correct").sum()), raw_wrong=int((d.g_raw == "wrong").sum()), raw_indeterminate=int((d.g_raw == "indeterminate").sum()),
            raw_correct_thermal=f"{int((th.g_raw == 'correct').sum())}/{len(th)}", raw_correct_rgb=f"{int((rgb.g_raw == 'correct').sum())}/{len(rgb)}",
            raw_correct_rgb_fire=f"{int((rgb_fire.g_raw == 'correct').sum())}/{len(rgb_fire)}",
            rgb_false_alarms=int(((rgb.truth == "no_fire") & (rgb.raw == "fire")).sum()),
            fl_g_raw=dict(Counter(d[d.seq == "FL-G_plot2_smallfire_rgb"].raw)),
            final_correct=int((d.g_final == "correct").sum()), measured_correct=int((d.g_measured == "correct").sum()),
            status_overrides=int(ov.sum()),
            overrides_helped=int((ov & (d.g_raw != "correct") & (d.g_final == "correct")).sum()),
            overrides_hurt=int((ov & (d.g_raw == "correct") & (d.g_final != "correct")).sum()),
            repeat_same_raw=f"{same_raw}/{len(reps)}", repeat_same_final=f"{same_final}/{len(reps)}", repeat_same_plan=f"{same_plan}/{len(reps)}",
            planner_rounds_mean=round(d.planner_rounds.mean(), 2), planner_unfinished=int((~d.planner_finished.astype(bool)).sum()),
            planner_tool_errors=int(d.planner_errors.map(len).sum()), planner_decides_mean=round(d.planner_decides.map(len).mean(), 2),
            planner_repeat_probes=int(d.probe_dups.sum()),
            planner_length_stops=int(d.planner_finish_reasons.map(lambda x: x.count("length")).sum()),
            writeup_length_stops=int((d.writeup_finish_reason == "length").sum()),
            writeup_s_median=round(d.writeup_s.median(), 1), brief_s_median=round(d.total_s.median(), 1))
    for name in DECIDERS:
        d = b.assign(call=b.apply(lambda r: (r.deciders.get(name) or {}).get("status"), axis=1))
        per_seq = d.groupby("seq").call.agg(lambda s: sorted(set(s)))
        first = d.drop_duplicates("seq")
        g = [grade(c, t) for c, t in zip(first.call, first.truth)]
        out["deciders"][name] = dict(correct=g.count("correct"), wrong=g.count("wrong"), indeterminate=g.count("indeterminate"), n=len(g),
                                     correct_rgb=sum(1 for x, th in zip(g, first.thermal) if not th and x == "correct"),
                                     rgb_n=int((~first.thermal).sum()),
                                     same_call_every_brief=bool(all(len(v) == 1 for v in per_seq)),
                                     seconds_median=round(float(d.apply(lambda r: (r.deciders.get(name) or {}).get("seconds"), axis=1).median()), 3))
    truth = {k: v["truth"] for k, v in SEQS.items()}
    for arm in sorted(set(b.arm)):
        base = b[b.arm == arm].base.iloc[0]
        if arm == base:
            continue
        a, g = b[b.arm == base], b[b.arm == arm]
        maj = lambda d: d.groupby("seq").g_raw.agg(lambda s: Counter(s).most_common(1)[0][0])
        ma, mg = maj(a), maj(g)
        both = ma.index.intersection(mg.index)
        trans = Counter((ma[q], mg[q]) for q in both)
        fa_a = {q for q in both if truth[q] == "no_fire" and ma[q] == "wrong"}
        fa_g = {q for q in both if truth[q] == "no_fire" and mg[q] == "wrong"}
        out["guidance"][arm] = dict(base=base, sequences=len(both), transitions={f"{x}->{y}": n for (x, y), n in trans.items()},
                                    raw_correct_without=int((a[a.seq.isin(both)].g_raw == "correct").sum()),
                                    raw_correct_with=int((g[g.seq.isin(both)].g_raw == "correct").sum()),
                                    briefs_each=int(len(g[g.seq.isin(both)])), new_false_alarm_sequences=sorted(fa_g - fa_a))
    for seq in SEQS:
        d = b[b.seq == seq]
        if not len(d):
            continue
        r0 = d.iloc[0]
        row = dict(seq=seq, site=r0.site, thermal=bool(r0.thermal), truth=r0.truth,
                   **{k: (r0.deciders.get(k) or {}).get("status") for k in DECIDERS})
        for arm, g in d.groupby("arm"):
            row[arm] = "/".join(g.sort_values("repeat").raw.map(lambda s: {"fire": "F", "no fire or smoke": "N", "smoke only": "S", "indeterminate": "?"}[s]))
        out["per_sequence"].append(row)

dp = pd.DataFrame(load("deciders_probes_v2.jsonl"))
if len(dp):
    dp = dp.drop_duplicates(["seq", "decider"], keep="last")
    for name, d in dp.groupby("decider"):
        g = [grade(c, t) for c, t in zip(d.status, d.truth)]
        out["deciders_with_probes"][name] = dict(correct=g.count("correct"), wrong=g.count("wrong"), indeterminate=g.count("indeterminate"), n=len(g),
                                                 correct_rgb=sum(1 for x, th in zip(g, d.thermal) if not th and x == "correct"),
                                                 rgb_n=int((~d.thermal.astype(bool)).sum()), reference=d.reference.iloc[0])

frames_src = "frames_v2b.jsonl" if (RESULTS / "frames_v2b.jsonl").exists() else "frames_v2.jsonl"
fr = {}
for r in load(frames_src):
    fr[r["name"]] = r  # last line per name wins
out["frames_source"] = frames_src
out["frames"] = fr
noise = {}
for name, r in fr.items():
    if "#rep" in name:
        base = name.split("#rep")[0]
        a, c = RESULTS / f"frames_v2b_{base}.csv", RESULTS / f"frames_v2b_{name}.csv"
        if a.exists() and c.exists():
            x, y = pd.read_csv(a), pd.read_csv(c)
            m = x.merge(y, on="uid", suffixes=("_1", "_2"))
            noise[base] = dict(frames=len(m), answers_changed=int((m.fire_1 != m.fire_2).sum()),
                               recall_1=round(m[m.y_1 == 1].fire_1.mean(), 4), recall_2=round(m[m.y_1 == 1].fire_2.mean(), 4),
                               max_abs_log_odds_change=round(float((m.log_odds_1 - m.log_odds_2).abs().max()), 4))
out["frames_noise"] = noise
out["throughput"] = {r["route"]: r for r in load("throughput.jsonl")}
(RESULTS / "summary_v2.json").write_text(json.dumps(out, indent=1, default=str))
brief_view = {k: v for k, v in out.items() if k not in ("per_sequence", "frames")}
print(json.dumps(brief_view, indent=1, default=str)[:15000])
if out["per_sequence"]:
    print(pd.DataFrame(out["per_sequence"]).to_string(index=False))
