"""
Generate mock recon sequences for the replay console.

The console is decoupled from the pipeline: it only reads per-sequence JSON in
the shape produced here. While the backend (detector + thermal + the brief
models) is being wired up, this script stands in for export_demo.py and writes
example files in the exact contract the page expects. When the real pipeline is
ready it writes the same shape and the page needs no changes.

Outputs:
    data/index.json              one row per sequence, for the picker
    data/sequences/<id>.json     one file per sequence (the contract)
    data.js                      all of the above on window.DEMO, so the page
                                 runs by just opening index.html (no server)

Per-sequence contract (see data/sequences/*.json for filled examples):

    sequence_id      stable id, also the file name
    display_name     shown in the picker
    case             clear_growing_fire | small_edge_fire | smoke_obscured | no_fire
    source           dataset, flight, has_thermal, frame_interval_sec,
                     pose_stable, gsd_m_per_px
                       gsd_m_per_px: ground sample distance. null when altitude
                         and FOV are not logged, which means radius stays in
                         percent of frame and meters are not reported.

    frames           ordered list of per-frame MEASUREMENTS (shared by all
                     models). Each frame:
                       index, t_sec, clock, detection_score (0..1 RGB Tier 1),
                       fire (bool), radius_frac (0..1 of frame),
                       radius_m (number or null), centroid ([x,y] norm or null),
                       overlay_png (path or null)

    thermal_reference  the Tier 2 thermal-derived reference conclusion:
                       { available, fire_present, trend }
                       Every model is compared against this.

    models           map of model_id -> that model's CONCLUSION about the
                     sequence:
                       label, family,
                       brief { presence, extent, trend, confidence,
                               limitations },
                       agreement { presence, trend }  (vs thermal)
    default_model    which model to show first

Spread direction is intentionally not reported. On this data the frame interval
(~1 min) is too long and the per-frame centroid shift stays inside the camera
stability margin, so direction comes out indeterminate across the board. It is
left out of the brief rather than shown as a permanent "indeterminate".
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SEQ_DIR = DATA / "sequences"


def clock(t_sec, start="13:02:00"):
    h, m, s = (int(x) for x in start.split(":"))
    total = h * 3600 + m * 60 + s + t_sec
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def lerp(a, b, k):
    return a + (b - a) * k


def path(p0, p1, n):
    """Interpolate p0 to p1 so the fire blob sits in a plausible spot each frame.
    Centroids position the current hot region; they are not used to claim a
    spread direction."""
    return [(round(lerp(p0[0], p1[0], i / (n - 1)), 3),
             round(lerp(p0[1], p1[1], i / (n - 1)), 3)) for i in range(n)]


def frames(scores, radii, centroids, gsd=None, interval=60):
    out = []
    for i in range(len(scores)):
        c = centroids[i]
        out.append(
            {
                "index": i,
                "t_sec": i * interval,
                "clock": clock(i * interval),
                "detection_score": round(scores[i], 3),
                "fire": scores[i] >= 0.5,
                "radius_frac": round(radii[i], 4),
                "radius_m": None if gsd is None or radii[i] == 0 else round(radii[i] * 1280 * gsd, 1),
                "centroid": None if c is None else [round(c[0], 3), round(c[1], 3)],
                "overlay_png": None,
            }
        )
    return out


def model(label, family, brief, agree_presence, agree_trend):
    return {
        "label": label,
        "family": family,
        "brief": brief,
        "agreement": {
            "presence": agree_presence,
            "trend": agree_trend,
        },
    }


def brief(presence, extent, trend, confidence, limitations):
    return {
        "presence": presence,
        "extent": extent,
        "trend": trend,
        "confidence": confidence,
        "limitations": limitations,
    }


def build():
    seqs = []

    # 1. Clear growing fire. Thermal: present, growing.
    n = 10
    seqs.append({
        "sequence_id": "clear_growing_fire",
        "display_name": "Clear growing fire",
        "case": "clear_growing_fire",
        "source": {"dataset": "FLAME3", "flight": "Sycan Marsh, Oct 27 PM", "has_thermal": True,
                   "frame_interval_sec": 60, "pose_stable": True, "gsd_m_per_px": None},
        "frames": frames(
            scores=[0.94, 0.95, 0.96, 0.97, 0.97, 0.98, 0.98, 0.99, 0.99, 0.99],
            radii=[0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.19, 0.22, 0.25, 0.28],
            centroids=path((0.56, 0.63), (0.38, 0.40), n)),
        "thermal_reference": {"available": True, "fire_present": True, "trend": "growing"},
        "default_model": "qwen38-27b",
        "models": {
            "qwen38-27b": model("Qwen3.8 (27B)", "Reasoning + tools",
                brief("Fire confirmed, open flame with active spread",
                      "Radius grew from about 4% to 28% of the frame over 9 minutes",
                      "growing", "high",
                      ["Size is percent of frame; no ground sample distance logged, so no acreage"]),
                True, True),
            "qwen3vl-4b": model("Qwen3-VL (4B)", "VLM",
                brief("Fire confirmed with active spread",
                      "Expanding, roughly a quarter of the frame by the end",
                      "growing", "medium",
                      ["Size estimate runs low against the thermal area"]),
                True, True),
            "mask-tracker": model("Mask tracker", "Deterministic",
                brief("Hot region present and expanding",
                      "Mask radius grew about 7x from the first to the last frame",
                      "growing", "high",
                      ["Geometry only, no scene understanding; reads the thermal mask area"]),
                True, True),
        },
    })

    # 2. Small fire at the frame edge. Thermal: present, stable.
    n = 10
    edge = path((0.9, 0.47), (0.9, 0.49), n)
    seqs.append({
        "sequence_id": "small_edge_fire",
        "display_name": "Small fire at frame edge",
        "case": "small_edge_fire",
        "source": {"dataset": "FLAME3", "flight": "Sycan Marsh, Oct 27 AM", "has_thermal": True,
                   "frame_interval_sec": 60, "pose_stable": True, "gsd_m_per_px": None},
        "frames": frames(
            scores=[0.54, 0.58, 0.52, 0.61, 0.57, 0.63, 0.55, 0.60, 0.58, 0.62],
            radii=[0.030, 0.032, 0.030, 0.035, 0.033, 0.036, 0.034, 0.037, 0.035, 0.038],
            centroids=edge),
        "thermal_reference": {"available": True, "fire_present": True, "trend": "stable"},
        "default_model": "qwen38-27b",
        "models": {
            "qwen38-27b": model("Qwen3.8 (27B)", "Reasoning + tools",
                brief("Small hotspot confirmed near the right edge",
                      "Under 4% of the frame throughout",
                      "stable", "medium",
                      ["Hotspot sits at the frame edge and may be partially out of view",
                       "Detection score stays near the threshold, so a missed frame is plausible"]),
                True, True),
            "qwen3vl-4b": model("Qwen3-VL (4B)", "VLM",
                brief("Growing fire on the right side",
                      "Small but increasing",
                      "growing", "low",
                      ["Read the frame-to-frame score noise as real growth"]),
                True, False),
            "mask-tracker": model("Mask tracker", "Deterministic",
                brief("Small hot region, roughly constant",
                      "Mask radius flat within noise",
                      "stable", "medium",
                      ["Mask area barely changes, so the trend reads as stable"]),
                True, True),
        },
    })

    # 3. Smoke-obscured fire. Thermal sees through smoke: present, growing.
    #    RGB models should abstain; a weak model hallucinates.
    n = 10
    seqs.append({
        "sequence_id": "smoke_obscured",
        "display_name": "Smoke-obscured fire",
        "case": "smoke_obscured",
        "source": {"dataset": "FLAME3", "flight": "Sycan Marsh, Oct 27 PM", "has_thermal": True,
                   "frame_interval_sec": 60, "pose_stable": True, "gsd_m_per_px": None},
        "frames": frames(
            scores=[0.82, 0.86, 0.71, 0.63, 0.49, 0.55, 0.46, 0.52, 0.44, 0.50],
            radii=[0.06, 0.07, 0.08, 0.075, 0.09, 0.085, 0.095, 0.09, 0.10, 0.095],
            centroids=path((0.48, 0.55), (0.60, 0.62), n)),
        "thermal_reference": {"available": True, "fire_present": True, "trend": "growing"},
        "default_model": "qwen38-27b",
        "models": {
            "qwen38-27b": model("Qwen3.8 (27B)", "Reasoning + tools",
                brief("Fire present but obscured by dense smoke in RGB",
                      "Not reliably measurable from RGB after frame 4",
                      "indeterminate", "low",
                      ["Dense smoke obscures the fire in RGB from frame 4 onward",
                       "Deferring trend to the thermal reference"]),
                True, False),
            "qwen3vl-4b": model("Qwen3-VL (4B)", "VLM",
                brief("Fire appears to be dying down",
                      "Shrinking behind the smoke",
                      "receding", "low",
                      ["Read the smoke plume as a shrinking fire; contradicted by thermal"]),
                True, False),
            "mask-tracker": model("Mask tracker", "Deterministic",
                brief("Hot region present and expanding",
                      "Mask radius increasing through the sequence",
                      "growing", "high",
                      ["Reads the thermal mask, which sees through smoke; RGB-only models cannot"]),
                True, True),
        },
    })

    # 4. No fire. Thermal: no fire. All models agree.
    n = 8
    no_fire_brief = brief("No fire or smoke detected", "Not applicable",
                          "not applicable", "high",
                          ["Clear scene, no thermal signature above the 80 C threshold"])
    seqs.append({
        "sequence_id": "no_fire",
        "display_name": "No fire",
        "case": "no_fire",
        "source": {"dataset": "FLAME3", "flight": "Sycan Marsh, Oct 27 AM", "has_thermal": True,
                   "frame_interval_sec": 60, "pose_stable": True, "gsd_m_per_px": None},
        "frames": frames(
            scores=[0.05, 0.04, 0.07, 0.03, 0.06, 0.05, 0.08, 0.04],
            radii=[0.0] * n, centroids=[None] * n),
        "thermal_reference": {"available": True, "fire_present": False, "trend": "not applicable"},
        "default_model": "qwen38-27b",
        "models": {
            "qwen38-27b": model("Qwen3.8 (27B)", "Reasoning + tools", no_fire_brief, True, True),
            "qwen3vl-4b": model("Qwen3-VL (4B)", "VLM", no_fire_brief, True, True),
            "mask-tracker": model("Mask tracker", "Deterministic",
                brief("No hot region above threshold", "Not applicable",
                      "not applicable", "high",
                      ["No thermal mask to track"]), True, True),
        },
    })

    # Per-model predicted final size (radius as fraction of frame), the model's
    # own guess at how big the fire got. null where a model cannot estimate it
    # (for example the 27B abstaining through smoke). The thermal truth final
    # radius is in parentheses for reference.
    size_est = {
        "clear_growing_fire": {"qwen38-27b": 0.26, "qwen3vl-4b": 0.18, "mask-tracker": 0.28},   # truth 0.28
        "small_edge_fire":    {"qwen38-27b": 0.04, "qwen3vl-4b": 0.07, "mask-tracker": 0.038},  # truth 0.038
        "smoke_obscured":     {"qwen38-27b": None, "qwen3vl-4b": 0.05, "mask-tracker": 0.095},  # truth 0.095
        "no_fire":            {"qwen38-27b": None, "qwen3vl-4b": None, "mask-tracker": None},
    }
    for s in seqs:
        for mid, m in s["models"].items():
            m["size_estimate_frac"] = size_est.get(s["sequence_id"], {}).get(mid)

    # The thermal truth lives in frames[] (per-frame radius and centroid) and in
    # thermal_reference (trend). The old mask-tracker "model" was just that same
    # thermal reference under another name, so it is NOT a model and is not
    # listed as one.
    for s in seqs:
        s["models"].pop("mask-tracker", None)

    return seqs


def summary(seq):
    return {
        "id": seq["sequence_id"],
        "display_name": seq["display_name"],
        "case": seq["case"],
        "models": [{"id": mid, "label": m["label"], "family": m["family"]}
                   for mid, m in seq["models"].items()],
        "n_frames": len(seq["frames"]),
    }


def main():
    SEQ_DIR.mkdir(parents=True, exist_ok=True)
    seqs = build()

    index = {"sequences": [summary(s) for s in seqs]}
    (DATA / "index.json").write_text(json.dumps(index, indent=2) + "\n")

    bundle = {"index": index, "sequences": {}}
    for seq in seqs:
        (SEQ_DIR / f"{seq['sequence_id']}.json").write_text(json.dumps(seq, indent=2) + "\n")
        bundle["sequences"][seq["sequence_id"]] = seq

    js = "// Generated by make_mock_data.py. Do not edit by hand.\n"
    js += "window.DEMO = " + json.dumps(bundle, indent=2) + ";\n"
    (HERE / "data.js").write_text(js)

    print(f"Wrote {len(seqs)} sequences, models each: " +
          ", ".join(str(len(s['models'])) for s in seqs))


if __name__ == "__main__":
    main()
