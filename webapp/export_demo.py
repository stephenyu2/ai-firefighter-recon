"""
Export real pipeline output for the replay console.

Unlike make_mock_data.py (synthetic), this reads the committed result files from
the real runs and writes the per-sequence JSON the page replays. Nothing here
runs a model; it replays what the gateway and the d3 cascade already produced.

Sources (all committed in the repo):
  experiments/decision-cascade/results/cascade.jsonl
      the d3-first cascade on the 20 test sequences: per sequence, what d3
      answered, its probabilities, whether the cascade accepted that call or
      escalated to Qwen3.8-27B, the final status and the timing.
  experiments/gateway-model-trials/results/runs_v2.jsonl.gz
      the full Qwen3.8-27B BF16 pipeline run on the same sequences: per-frame
      detector / CLIP / thermal measurements, the hot-area trend, and the real
      structured brief (fire_smoke, extent, trend, confidence, limitations,
      observations, inferences, summary).

We join the two by sequence id: the cascade record is the routing story, the
run record is the measurements and the brief. Spread DIRECTION is intentionally
dropped (on this data it comes out indeterminate; see the console README).

Outputs (same layout as make_mock_data.py):
  data/index.json, data/sequences/<id>.json, data.js
"""
import gzip
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = HERE / "data"
SEQ_DIR = DATA / "sequences"

CASCADE = REPO / "experiments/decision-cascade/results/cascade.jsonl"
RUNS = REPO / "experiments/gateway-model-trials/results/runs_v2.jsonl.gz"
RUN_LABEL = "qwen38-27b-bf16#r1"  # the full-pipeline Qwen3.8-27B BF16 run, repeat 1

# The sequences shown, in order. Picked to cover both cascade paths, thermal and
# RGB-only, fire and no-fire, and one honest miss. display_name is what the
# picker shows; the rest is read from the result files.
SELECTED = [
    ("FL-A_plot1_burn_thermal",          "Active burn"),
    ("SY-2_5",                           "Sycan Marsh burn"),
    ("FL-G_plot2_smallfire_rgb",         "Small fire"),
    ("FL-E_plot2_takeoff_smoke_thermal", "Takeoff, smoke no fire"),
    ("FL-H_plot2_smallfire_rgb",         "Small fire missed"),
]

# location_in_frame -> a coarse normalized centroid (x right, y down), so the
# viewer can place the hot blob plausibly. The pipeline does not store a
# per-frame centroid, so this is a display approximation from the LLM's
# location word, not a measurement.
LOC = {
    "top-left": (0.24, 0.24), "upper-left": (0.24, 0.24),
    "top-right": (0.76, 0.24), "upper-right": (0.76, 0.24),
    "bottom-left": (0.24, 0.76), "lower-left": (0.24, 0.76),
    "bottom-right": (0.76, 0.76), "lower-right": (0.76, 0.76),
    "top": (0.5, 0.22), "upper": (0.5, 0.22),
    "bottom": (0.5, 0.78), "lower": (0.5, 0.78),
    "left": (0.2, 0.5), "right": (0.8, 0.5),
    "center": (0.5, 0.5), "centre": (0.5, 0.5), "middle": (0.5, 0.5),
}


def centroid_from_location(loc):
    if not loc:
        return (0.5, 0.5)
    s = loc.lower()
    for key in ("top-left", "upper-left", "top-right", "upper-right",
                "bottom-left", "lower-left", "bottom-right", "lower-right"):
        if key in s:
            return LOC[key]
    for key in ("upper", "lower", "top", "bottom", "left", "right", "center", "centre", "middle"):
        if key in s:
            return LOC[key]
    return (0.5, 0.5)


def radius_frac_from_hot(hot_percent):
    """A circle whose area equals the hot-area share, as a radius in frame units."""
    if hot_percent is None:
        return None
    return round(math.sqrt((hot_percent / 100.0) / math.pi), 4)


def clock(t_sec, start="13:02:00"):
    h, m, s = (int(x) for x in start.split(":"))
    total = h * 3600 + m * 60 + round(t_sec)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def trend_from_run(run):
    """Thermal trend from the spread tracker when supported, else indeterminate.
    The schema words are increasing / decreasing / stable / indeterminate."""
    sp = run["evidence"].get("spread") or {}
    if sp.get("thermal_frames") and sp.get("trend_supported"):
        return sp.get("trend", "indeterminate")
    return "indeterminate"


def case_of(seq, truth, thermal):
    if "preburn" in seq:
        return "no_fire"
    if "takeoff" in seq or "smoke" in seq:
        return "smoke_obscured"
    if "smallfire" in seq:
        return "small_edge_fire"
    if truth == "fire":
        return "clear_growing_fire"
    return "no_fire"


def final_correct(final_status, truth):
    if truth == "fire":
        return final_status == "fire"
    return final_status in ("no fire or smoke", "smoke only")


def load_cascade():
    out = {}
    for line in CASCADE.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("decider_name") == "d3":
            out[r["seq"]] = r
    return out


def load_runs():
    out = {}
    with gzip.open(RUNS, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("label") == RUN_LABEL:
                out[r["seq"]] = r["run"]
    return out


def build_sequence(seq, display_name, cas, run):
    meta = run["meta"]
    det = run["evidence"]["detection"]
    frames_in = det["frames"]
    n = len(frames_in)
    span = meta.get("span_s") or 0
    interval = (span / (n - 1)) if n > 1 else 0
    thermal = bool(det.get("thermal_frames", 0))
    truth = cas["truth"]
    cen = centroid_from_location(run["brief"].get("location_in_frame"))

    frames = []
    for i, f in enumerate(frames_in):
        rf = radius_frac_from_hot(f.get("thermal_hot_percent"))
        has_hot = rf is not None and rf > 0
        frames.append({
            "index": i,
            "id": f["id"],
            "t_sec": round(i * interval, 1),
            "clock": clock(i * interval),
            "detector_p_fire": round(f.get("detector_p_fire") or 0.0, 4),
            "clip_p_fire": round(f.get("clip_p_fire") or 0.0, 4),
            "thermal_hot_percent": f.get("thermal_hot_percent"),
            "thermal_max_c": f.get("thermal_max_c"),
            "fire_touches_edge": f.get("fire_touches_edge"),
            "fire": bool(f.get("detector_fire")),
            "radius_frac": rf,
            "centroid": [cen[0], cen[1]] if has_hot else None,
            "overlay_png": None,
        })

    d = cas["decider"]
    cascade = {
        "decider": "d3",
        "decider_label": "Decision 3.0 (d3, 26B)",
        "accept_p": cas.get("accept_p"),
        "decider_status": d["status"],
        "decider_top_p": round(d["top_p"], 4),
        "status_probabilities": {k: round(v, 4) for k, v in d["status_probabilities"].items()},
        "p_fire_visible": round(d.get("p_fire_visible", 0.0), 4),
        "decider_seconds": round(d.get("seconds", 0.0), 2),
        "path": cas["path"],  # "decision" (d3 resolved) or "escalated" (to the LLM)
        "escalated": cas["path"] == "escalated",
        "llm_label": "Qwen3.8-27B",
        "final_status": cas["status"],
        "total_seconds": round(cas.get("seconds", 0.0), 2),
        "correct": final_correct(cas["status"], truth),
    }

    br = run["brief"]
    fs = br["fire_smoke"]
    brief = {
        "presence": fs["status"],
        "presence_confidence": fs["confidence"],
        "presence_evidence": fs["evidence"],
        "extent": br["extent"]["description"],
        "hot_area_percent": br["extent"].get("hot_area_percent_of_frame"),
        "location_in_frame": br.get("location_in_frame"),
        "trend": br["trend"]["status"],
        "trend_basis": br["trend"]["basis"],
        "overall_confidence": br["overall_confidence"],
        "limitations": br.get("limitations", []),
        "observations": br.get("observations", []),
        "inferences": br.get("inferences", []),
        "summary": br.get("summary", ""),
    }

    return {
        "sequence_id": seq,
        "display_name": display_name,
        "case": case_of(seq, truth, thermal),
        "truth": truth,
        "source": {
            "dataset": "FLAME3",
            "site": "Sycan Marsh, OR" if seq.startswith("SY") else "Florida burn",
            "flight": seq,
            "has_thermal": thermal,
            "n_frames": n,
            "span_s": span,
            "frame_interval_sec": round(interval, 1),
            "llm_model": meta.get("llm_model"),
        },
        "frames": frames,
        "thermal_reference": {
            "available": thermal,
            "fire_present": truth == "fire",
            "trend": trend_from_run(run),
        },
        "cascade": cascade,
        "brief": brief,
    }


def summary(seq):
    c = seq["cascade"]
    return {
        "id": seq["sequence_id"],
        "display_name": seq["display_name"],
        "case": seq["case"],
        "truth": seq["truth"],
        "has_thermal": seq["source"]["has_thermal"],
        "path": c["path"],
        "final_status": c["final_status"],
        "correct": c["correct"],
        "n_frames": len(seq["frames"]),
    }


def main():
    cas = load_cascade()
    runs = load_runs()
    SEQ_DIR.mkdir(parents=True, exist_ok=True)

    seqs = []
    for seq, name in SELECTED:
        if seq not in cas or seq not in runs:
            print(f"skip {seq}: cascade={seq in cas} run={seq in runs}")
            continue
        seqs.append(build_sequence(seq, name, cas[seq], runs[seq]))

    index = {"sequences": [summary(s) for s in seqs], "source": "real pipeline output"}
    (DATA / "index.json").write_text(json.dumps(index, indent=2) + "\n")

    bundle = {"index": index, "sequences": {}}
    for s in seqs:
        (SEQ_DIR / f"{s['sequence_id']}.json").write_text(json.dumps(s, indent=2) + "\n")
        bundle["sequences"][s["sequence_id"]] = s

    js = "// Generated by export_demo.py from committed real run results. Do not edit by hand.\n"
    js += "window.DEMO = " + json.dumps(bundle, indent=2) + ";\n"
    (HERE / "data.js").write_text(js)

    print(f"Wrote {len(seqs)} real sequences:")
    for s in seqs:
        c = s["cascade"]
        print(f"  {s['sequence_id']:<34} {c['path']:<10} d3={c['decider_status']}({c['decider_top_p']:.2f}) "
              f"final={c['final_status']} {'OK' if c['correct'] else 'MISS'}")


if __name__ == "__main__":
    main()
