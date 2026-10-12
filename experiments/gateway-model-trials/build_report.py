"""Builds report/gateway_model_trials.html from summary_v2.json (run summarize_v2.py first), frames_v2.jsonl,
throughput.jsonl and report_text.json (the narrative). Shares the stylesheet of ../report_template.html.
Usage: python build_report.py"""
import html
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
S = json.loads((RESULTS / "summary_v2.json").read_text())
TXT = json.loads((HERE / "report_text.json").read_text())
STYLE = re.search(r"<style>.*?</style>", (HERE.parent / "full-burn-benchmark/report_template.html").read_text(), re.S).group(0)
SEQS = json.loads((HERE / "sequences.json").read_text())

ROUTE_NAME = {"qwen38-27b-bf16": "Qwen3.8-27B BF16", "qwen38-27b-fp8": "Qwen3.8-27B FP8", "qwen38-27b-int4": "Qwen3.8-27B INT4 (weights only)",
              "qwen38-27b-nvfp4": "Qwen3.8-27B NVFP4", "qwen3vl-4b": "Qwen3-VL-4B (previous)", "clef": "Clef 27B", "clef_flash": "Clef-Flash 9B",
              "kai": "Kai 0.6B"}
ROUTES = ["qwen38-27b-bf16", "qwen38-27b-fp8", "qwen38-27b-int4", "qwen38-27b-nvfp4", "qwen3vl-4b"]
FRAME_MODELS = ROUTES + ["clef", "clef_flash"]
GROUPS = ["Florida fire, large", "Florida fire, small", "Sycan fire", "Florida pre-burn", "Florida burn, no hot pixel", "Sycan no fire"]


def esc(x):
    return html.escape(str(x))


def table(head, body, cls="num"):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in body)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def tk(t):
    return f'<div class="takeaway"><span class="tk-label">In short</span><p>{t}</p></div>'


def result(t):
    return f'<p class="result"><strong>Result:</strong> {t}</p>'


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def frames_table(seed, guided):
    suffix = ("+guide" if guided else "") + (f"@seed{seed}" if seed else "")
    rows = []
    for m in FRAME_MODELS:
        r = S["frames"].get(m + suffix)
        if not r:
            continue
        g = r["by_group"]
        rows.append([esc(ROUTE_NAME[m]), f3(r["recall"]), f'{r["false_alarms"]} of {r["negatives"]}', f3(r["auc"])]
                    + [f3(g.get(x)) for x in GROUPS[:3]] + [str(r.get("unreliable", 0))])
    return table(["Model", "Recall, all fire frames", "False alarms", "ROC AUC", "Recall: Florida large fires", "Recall: Florida small fires",
                  "Recall: Sycan fires", "Unreliable answers (no clear yes or no)"], rows)


def guidance_table():
    rows = []
    for m in FRAME_MODELS:
        cells = [esc(ROUTE_NAME[m])]
        for seed in (0, 1):
            sfx = f"@seed{seed}" if seed else ""
            a, b = S["frames"].get(m + sfx), S["frames"].get(m + "+guide" + sfx)
            if not (a and b):
                cells += ["n/a"] * 3
                continue
            d = b["recall"] - a["recall"]
            cells += [f'{a["recall"]:.3f} to {b["recall"]:.3f}', f'<span class="{"ok" if d > 0.02 else "bad" if d < -0.02 else ""}">{d:+.3f}</span>',
                      f'{a["false_alarms"]} to {b["false_alarms"]}']
        rows.append(cells)
    return table(["Model", "Recall, first sample", "Change", "False alarms", "Recall, fresh sample", "Change", "False alarms"], rows)


def _seq_counts(arm, pick):
    """Sequences (of the per-sequence table) where the arm's majority call over its 3 repeats is correct."""
    n = c = 0
    for row in S["per_sequence"]:
        if not pick(row) or arm not in row:
            continue
        calls = row[arm].split("/")
        maj = max(set(calls), key=calls.count)
        n += 1
        c += (maj == "F") == (row["truth"] == "fire") and maj != "?"
    return c, n


SMALL_RGB = lambda r: (not r["thermal"]) and "smallfire" in r["seq"]
BRIEFS = [json.loads(l) for l in (RESULTS / "briefs_v2.jsonl").read_text().splitlines() if l.strip()]
SEQ_BY_ID = {s["id"]: s for s in SEQS}
DPROBES = [json.loads(l) for l in (RESULTS / "deciders_probes_v2.jsonl").read_text().splitlines() if l.strip()]


def measured_seqs(arm, pick=lambda s: True):
    seqs = {}
    for r in BRIEFS:
        if r.get("ok") and r["route"].split("#")[0] == arm and pick(SEQ_BY_ID[r["seq"]]):
            seqs.setdefault(r["seq"], []).append((r["measured"] == "fire") == (r["truth"] == "fire") and r["measured"] != "indeterminate")
    return sum(1 for v in seqs.values() if sum(v) * 2 > len(v)), len(seqs)


def measured_small(arm):
    seqs = {}
    for r in BRIEFS:
        if r.get("ok") and r["route"].split("#")[0] == arm and not r["thermal"] and "smallfire" in r["seq"]:
            seqs.setdefault(r["seq"], []).append(r["measured"] == "fire")
    return sum(1 for v in seqs.values() if sum(v) * 2 > len(v)), len(seqs)


def probes_small(decider):
    rows = {r["seq"]: r for r in DPROBES if r["decider"] == decider}
    return sum(1 for q, r in rows.items() if "smallfire" in q and not r["thermal"] and r["status"] == "fire")
NOFIRE_RGB = lambda r: (not r["thermal"]) and r["truth"] == "no_fire"


def briefs_table():
    rows = []
    for arm in ROUTES + [r + "+guide" for r in ROUTES]:
        v = S["routes"].get(arm)
        if not v:
            continue
        name = ROUTE_NAME[arm.replace("+guide", "")] + (" + guidance" if arm.endswith("+guide") else "")
        sm, smn = _seq_counts(arm, SMALL_RGB)
        nf, nfn = _seq_counts(arm, NOFIRE_RGB)
        rows.append([esc(name), f'{v["raw_correct"]} of {v["briefs"]}', v["raw_correct_thermal"], f"{sm} of {smn}", f"{nf} of {nfn}",
                     str(v["raw_indeterminate"]), v["repeat_same_raw"], v["repeat_same_plan"]])
    return table(["LLM", "Own call correct (of 60 briefs)", "Thermal briefs correct", "RGB-only small-fire sequences correct",
                  "RGB-only no-fire sequences correct", "Indeterminate briefs", "Same call in all 3 repeats (sequences)",
                  "Same extra frames chosen in all 3 repeats (sequences)"], rows)


def planner_table():
    rows = []
    for arm in ROUTES + [r + "+guide" for r in ROUTES]:
        v = S["routes"].get(arm)
        if not v:
            continue
        name = ROUTE_NAME[arm.replace("+guide", "")] + (" + guidance" if arm.endswith("+guide") else "")
        rows.append([esc(name), f'{v["planner_rounds_mean"]:.2f}', f'{v["planner_decides_mean"]:.2f}', str(v["planner_repeat_probes"]),
                     str(v["planner_unfinished"]), str(v["planner_tool_errors"]), str(v["planner_length_stops"] + v["writeup_length_stops"]),
                     f'{v["writeup_s_median"]:.1f}', f'{v["brief_s_median"]:.1f}'])
    return table(["LLM", "Planner turns (mean per brief)", "Extra image checks (mean per brief)", "Extra checks repeating a gateway check (total)",
                  "Plans not finished (of 60)", "Rejected tool calls (total)", "Cut off by token limit (total)", "Writeup s (median)",
                  "Whole brief s (median)"], rows)


def deciders_table():
    rows = []
    for arm in ROUTES:
        c, n = _seq_counts(arm, lambda r: True)
        sm, smn = _seq_counts(arm, SMALL_RGB)
        rows.append([esc(ROUTE_NAME[arm] + ": its own call in the brief"), f"{c} of {n}", f"{sm} of {smn}", "frames, all measurements and image checks"])
    for arm, label in (("qwen38-27b-bf16+guide", "guided BF16 image checks"), ("qwen38-27b-bf16", "unguided BF16 image checks")):
        if any(r["route"].split("#")[0] == arm for r in BRIEFS):
            c, n = measured_seqs(arm)
            ms, msn = measured_small(arm)
            rows.append([esc(f"Measurement rule with {label}"), f"{c} of {n}", f"{ms} of {msn}", "thermal, detector, three image checks; code, not a model"])
    for d in ("kai", "clef", "clef_flash"):
        v, w = S["deciders"].get(d), S["deciders_with_probes"].get(d)
        reads = "text only" if d == "kai" else "text plus first and last frame"
        if v:
            small = sum(1 for r in S["per_sequence"] if (not r["thermal"]) and "smallfire" in r["seq"] and r[d] == "fire")
            rows.append([esc(ROUTE_NAME[d] + ", measurements only"), f'{v["correct"]} of {v["n"]}', f"{small} of 7", f'{reads}; {v["seconds_median"]:.2f} s per call'])
        if w:
            rows.append([esc(ROUTE_NAME[d] + ", plus unguided BF16 image checks"), f'{w["correct"]} of {w["n"]}', f"{probes_small(d)} of 7", reads])
    return table(["Who decides fire status", "Sequences correct (of 20)", "RGB-only small fires (of 7)", "Reads"], rows)


def speed_table():
    rows = []
    for r in ROUTES:
        t = S["throughput"].get(r)
        if t:
            up = lambda x: str(int(x + 0.5))
            rows.append([esc(ROUTE_NAME[r]), f'{t["ttft_s"]:.2f}', up(t["decode_tok_s_c1"]), up(t["aggregate_tok_s_c8"])])
    return table(["Model", "Time to first token (s)", "Decode, one request (tokens/s)", "Total throughput, 8 requests at once (tokens/s)"], rows)


def sequences_table():
    notes = {"SY-2_5": "Sycan fire, weak signal (hot in 6 of 8 thermal frames)", "SY-15_0": "Sycan fire, hot in all 8 frames",
             "SY-13_0": "Sycan fire, hot in all 8 frames", "SY-2_7": "Sycan, no pixel at 80 C", "FL-G_plot2_smallfire_rgb": "same 8 frames as FL-B, with thermal withheld (hardest case)"}
    sid = lambda i: i if i.startswith("SY-") else i.split("_")[0]
    rows = [[esc(sid(s["id"])), esc(s["site"]), "yes" if s["thermal"] else "no", esc(s["truth"].replace("_", " ")), esc(notes.get(s["id"], s["note"]))] for s in SEQS]
    return table(["Sequence", "Site", "Thermal", "Truth", "What it is"], rows, cls="")


def stats():
    return '<div class="stats">' + "".join(f'<div class="stat"><b>{esc(v)}</b><span>{esc(l)}</span></div>' for v, l in TXT["stats"]) + "</div>"


page = f"""<title>Gateway Model Trials</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@500;600;700&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
{STYLE}
<style>
.summary {{ display: grid; gap: 14px; padding-bottom: 8px; }}
.summary h2 {{ margin: 8px 0 0; }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }}
.stat {{ background: var(--surface); border: 1px solid var(--rule); padding: 14px 16px; display: grid; gap: 4px; align-content: start; min-width: 0; }}
.stat b {{ font: 700 28px/1.05 var(--display); color: var(--ink); font-variant-numeric: tabular-nums; }}
.stat span {{ font-size: 13px; line-height: 1.4; color: var(--muted); }}
.recs {{ margin: 0; padding-left: 20px; }}
.recs li {{ margin-bottom: 6px; }}
.takeaway {{ background: var(--surface); border: 1px solid var(--rule); padding: 12px 16px; margin: 6px 0 10px; display: grid; gap: 2px; max-width: 80ch; }}
.takeaway p {{ margin: 0; }}
.tk-label {{ font: 600 12px/1.3 var(--display); letter-spacing: .1em; text-transform: uppercase; color: var(--accent); }}
.result {{ margin: -4px 0 16px; font-size: 15px; max-width: 80ch; }}
.result strong {{ color: var(--accent); font-weight: 600; }}
pre.prompt {{ white-space: pre-wrap; font: 13px/1.5 var(--mono); background: var(--surface); border: 1px solid var(--rule); padding: 14px 16px; max-width: 90ch; overflow-x: auto; }}
</style>
<main>
<header>
  <div class="eyebrow">CIS 5980 · Wildfire reconnaissance · Gateway model trials</div>
  <h1>Gateway Model Trials</h1>
  <div class="meta">{TXT["subtitle"]}</div>
</header>

<section class="summary">
  <h2>Summary</h2>
  <div class="prose"><p>{TXT["summary"]}</p></div>
  {stats()}
  <div class="prose"><p style="margin-bottom:6px"><strong>What we recommend</strong></p><ul class="recs">{"".join(f"<li>{r}</li>" for r in TXT["recommendations"])}</ul></div>
</section>

<h2>What we tested</h2>
{tk(TXT["tk_setup"])}
<div class="prose">{TXT["setup"]}</div>
<h3>Test sequences</h3>
{sequences_table()}

<h2>How the gateway calls the models</h2>
{tk(TXT["tk_gateway"])}
<div class="prose">{TXT["gateway"]}</div>

<h2>Can each model see fire in one frame?</h2>
{tk(TXT["tk_frames"])}
<div class="prose">{TXT["frames_intro"]}</div>
<h3>First sample: 240 frames</h3>
{frames_table(0, False)}
{result(TXT["r_frames0"])}
<h3>Fresh sample: 221 frames, none reused</h3>
{frames_table(1, False)}
{result(TXT["r_frames1"])}

<h2>Does the observer guidance help?</h2>
{tk(TXT["tk_guidance"])}
<div class="prose">{TXT["guidance_intro"]}</div>
{guidance_table()}
{result(TXT["r_guidance"])}

<h2>Full briefs: the LLM's own call</h2>
{tk(TXT["tk_briefs"])}
<div class="prose">{TXT["briefs_intro"]}</div>
{briefs_table()}
{result(TXT["r_briefs"])}
<h3>Is the tool calling clean and repeatable?</h3>
{planner_table()}
{result(TXT["r_planner"])}

<h2>Who should decide fire status?</h2>
{tk(TXT["tk_deciders"])}
<div class="prose">{TXT["deciders_intro"]}</div>
{deciders_table()}
{result(TXT["r_deciders"])}

<h2>Speed on one GPU</h2>
{tk(TXT["tk_speed"])}
{speed_table()}
{result(TXT["r_speed"])}

<h2>Method notes</h2>
<ul class="prose">{"".join(f"<li>{x}</li>" for x in TXT["method_notes"])}</ul>

<h2>Appendix: the observer guidance prompt</h2>
<div class="prose"><p>{TXT["prompt_intro"]}</p></div>
<pre class="prompt">{esc(__import__("sys").path.insert(0, str(HERE.parents[1] / "src")) or __import__("recon_gateway.prompts", fromlist=["WATCH_FOR"]).WATCH_FOR)}</pre>
</main>
"""
assert "\u2014" not in page, "em dash in report"
(HERE / "report").mkdir(exist_ok=True)
(HERE / "report" / "gateway_model_trials.html").write_text(page)
print("wrote", HERE / "report" / "gateway_model_trials.html", len(page))
