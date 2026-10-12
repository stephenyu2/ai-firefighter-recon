"""Gateway orchestration (v2): measure -> planner probes frames with tools -> evidence -> LLM writeup -> policy checks.

The gateway runs every measurement itself: detector, CLIP and thermal tools on all frames, spread tracking, and a fixed
set of image probes (first, middle and last frame). The measured evidence therefore never depends on what a model chose
to do. The planner LLM then picks up to DECIDE_BUDGET more frames to probe with `decide` and calls `finish`.
Tool calls are forced (tool_choice "required"), arguments are checked against the sequence's real frame ids, bad calls
get an error message back instead of a silent fallback, and each planner turn's finish_reason, tool calls and token use
are recorded. Frame and sequence ids shown to models are neutral (S, f01..): dataset ids can encode the answer.
Every LLM call uses temperature 0 and a fixed seed (tools.SEED)."""
import copy
import json
import os
import time

from . import tools as T
from .brief import BRIEF_SCHEMA, enforce_policy, evidence_status, smoke_signal
from .prompts import WATCH_FOR
from .sequence import alias

AGENT_VERSION = "v2"
DECIDE_BUDGET = 3
MAX_ROUNDS = 5
PROBE_QUESTIONS = ["smoke_visible", "flames_visible"]
GUIDANCE_PLACES = ("planner", "decide", "writeup")

SYSTEM = """You coordinate a wildfire reconnaissance analysis for an air attack supervisor. The gateway has already run
the fire detector, a general image model, the thermal tools and spread tracking on every frame, and has asked whether
smoke or flames are visible on the first, middle and last frames. Those results are in the user message.
Your job: choose which other frames need a closer look and ask about them with `decide` (one frame per call, at most
3 frames in total). Good choices are frames where the detector, the image model and thermal disagree, frames where smoke
may hide fire, and frames near the edge of a fire. Call `finish` when you have enough evidence, or right away if nothing
needs a closer look. Do not write analysis text; the brief is written in a later step."""

WRITEUP = """Write the reconnaissance brief as JSON that matches the schema. Rules:
- Use only the evidence below and what you can see in the attached first and last frames.
- Image probes are one vision model's answers about single frames, given as option probabilities. A high-probability
  "yes" to flames is evidence of fire. Weigh probes together with the detector and thermal; do not ignore any source.
- Separate observations (what the tools or images show) from inferences (what you conclude).
- If the evidence does not support a direction, trend, extent or location, say "indeterminate" and why.
- No acres, GPS, speeds, or instructions to crews. Describe location relative to the frame.
- extent.description: one sentence on how much of the frame the fire or smoke covers, in words. Do not quote hot-area
  numbers; the gateway adds the measured value.
- Name real limitations (smoke, edge of frame, short sequence, camera motion). Say RGB-only only if no thermal frames were analyzed.
Evidence (JSON; units are listed under "units"):
"""


def guidance_flags(guidance=None):
    """True/'1'/'all' -> every placement; a comma list ('decide,writeup') -> those; None reads RECON_GUIDANCE."""
    if guidance is None:
        guidance = os.environ.get("RECON_GUIDANCE", "")
    if guidance is True or str(guidance).lower() in ("1", "all", "true"):
        return {p: True for p in GUIDANCE_PLACES}
    chosen = {p.strip() for p in str(guidance or "").split(",") if p.strip()}
    return {p: p in chosen for p in GUIDANCE_PLACES}


def tool_specs(frame_ids):
    return [
        {"type": "function", "function": {
            "name": "decide",
            "description": "Ask structured multiple-choice questions about one frame. Returns option probabilities taken from the vision model's answer-letter probabilities.",
            "parameters": {"type": "object", "properties": {
                "frame_id": {"type": "string", "enum": frame_ids, "description": "One frame id from this sequence."},
                "questions": {"type": "array", "items": {"type": "string", "enum": list(T.QUESTION_BANK)}, "minItems": 1, "uniqueItems": True}},
                "required": ["frame_id", "questions"], "additionalProperties": False}}},
        {"type": "function", "function": {
            "name": "finish", "description": "Call when the evidence is sufficient. Takes no arguments.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    ]


def _slim(probe):
    """What a model sees of a probe result: no raw log-probs."""
    return {q: {k: v[k] for k in ("answer", "probabilities", "coverage")} for q, v in probe.items()}


def _measurement_view(state):
    det, sp = state["detection"], state["spread"]
    keep = ("trend", "trend_supported", "trend_reason", "direction", "direction_supported", "direction_reason",
            "hot_percent_start", "hot_percent_end", "camera_shift_fraction")
    return {"units": det["units"], "frames": det["frames"],
            "counts": {k: det[k] for k in ("detector_fire_frames", "clip_fire_frames", "thermal_frames", "thermal_fire_frames")},
            "disagreements": det["disagreements"],
            "spread": {k: sp[k] for k in keep if k in sp} if sp else {"note": "not run: fewer than 3 thermal frames"},
            "image_probes": {"fixed": {f: _slim(p) for f, p in state["probes"].items()},
                             "planner": {f: _slim(p) for f, p in state["planner_probes"].items()}}}


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _validate(name, args, ids, state):
    if name == "finish":
        return None  # finish carries no information; stray arguments are recorded in the trace, not rejected
    if name != "decide":
        return f"unknown tool '{name}'; the tools are decide and finish"
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    extra = set(args) - {"frame_id", "questions"}
    if extra:
        return f"unknown argument(s) {sorted(extra)}; decide takes frame_id and questions"
    if args.get("frame_id") not in ids:
        return f"frame_id must be one of {ids}"
    qs = args.get("questions")
    if (not isinstance(qs, list) or not qs or not all(isinstance(q, str) for q in qs) or len(set(qs)) != len(qs)
            or any(q not in T.QUESTION_BANK for q in qs)):
        return f"questions must be a non-empty list of distinct values from {list(T.QUESTION_BANK)}"
    if args["frame_id"] in state["planner_probes"]:
        return f"frame {args['frame_id']} was already probed; pick another frame or call finish"
    if len(state["planner_probes"]) >= DECIDE_BUDGET:
        return f"decide limit reached ({DECIDE_BUDGET} frames); call finish"
    return None


def _plan(ep, seq, state, system_extra, decide_system):
    ids = [f.id for f in seq.frames]
    n_th = sum(1 for f in seq.frames if f.tiff_path)
    msgs = [{"role": "system", "content": SYSTEM + system_extra},
            {"role": "user", "content": f"Sequence of {len(ids)} frames over {seq.span_s} s. Frame ids in time order: {ids}. "
                                        f"Thermal frames: {n_th}. Measurements so far:\n{_canon(_measurement_view(state))}"}]
    specs = tool_specs(ids)
    turns, finished = [], False
    for rnd in range(1, MAX_ROUNDS + 1):
        t0 = time.time()
        r = ep.chat(msgs, tools=specs, tool_choice="required", max_tokens=256)
        ch = r["choices"][0]; m = ch["message"]; calls = m.get("tool_calls") or []
        turn = {"round": rnd, "finish_reason": ch.get("finish_reason"), "seconds": round(time.time() - t0, 2),
                "usage": r.get("usage"), "tool_calls": []}
        msgs.append({"role": "assistant", "content": m.get("content"), **({"tool_calls": calls} if calls else {})})
        for c in calls:
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
                err = _validate(name, args, ids, state)
            except json.JSONDecodeError as e:
                args, err = c["function"].get("arguments"), f"arguments are not valid JSON ({e.msg})"
            except TypeError as e:
                err = f"arguments have the wrong types ({e})"
            entry = {"name": name, "args": args, "error": err}
            if name == "finish" and args not in ({}, None):
                entry["note"] = "finish called with stray arguments (ignored)"
            if err:
                result = {"error": err}
            elif name == "finish":
                result, finished = {"ok": True}, True
            else:
                t1 = time.time()
                probe = T.decide(ep, next(f for f in seq.frames if f.id == args["frame_id"]), args["questions"], system=decide_system)
                state["planner_probes"][args["frame_id"]] = probe
                result = _slim(probe)
                entry["seconds"] = round(time.time() - t1, 2)
            turn["tool_calls"].append(entry)
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": _canon(result)})
        turns.append(turn)
        if finished or not calls or ch.get("finish_reason") == "length":
            break
    return {"rounds": len(turns), "finished": finished, "exhausted": not finished and len(turns) == MAX_ROUNDS,
            "turns": turns, "transcript": msgs}


def run(seq, mode="agent", llm_model=None, deciders=None, guidance=None, run_tag=None):
    """mode "agent": the planner picks extra probes; "pipeline": fixed probes only.
    deciders: names from recon_gateway.deciders (kai, clef, clef_flash), each making its own fire-status call on the
    measured evidence. Returns the final brief plus the raw brief, the LLM's own call, the measured call and a full trace."""
    src_id = seq.id
    seq, id_map = alias(seq)
    ep = T.Endpoints(llm_model=llm_model, run_tag=run_tag)
    flags = guidance_flags(guidance)
    state = {"cache": {}, "trace": [], "probes": {}, "planner_probes": {}}
    step = lambda tool, t0, **kw: state["trace"].append({"step": len(state["trace"]) + 1, "tool": tool, "seconds": round(time.time() - t0, 3), **kw})

    t0 = time.time(); state["detection"] = T.detect_fire(ep, seq, state["cache"]); step("detect_fire", t0, by="gateway")
    t0 = time.time()
    state["spread"] = T.track_spread(seq, state["cache"])
    step("track_spread", t0, by="gateway")
    n = len(seq.frames)
    t0 = time.time()
    for i in sorted({0, n // 2, n - 1}):
        state["probes"][seq.frames[i].id] = T.decide(ep, seq.frames[i], PROBE_QUESTIONS, system=WATCH_FOR if flags["decide"] else None)
    step("fixed_probes", t0, by="gateway", frames=list(state["probes"]))

    planner = None
    if mode == "agent":
        t0 = time.time()
        planner = _plan(ep, seq, state, "\n\n" + WATCH_FOR if flags["planner"] else "", WATCH_FOR if flags["decide"] else None)
        step("planner", t0, rounds=planner["rounds"], finished=planner["finished"])

    ev = {"detection": state["detection"], "spread": state["spread"], "probes": state["probes"], "planner_probes": state["planner_probes"]}
    det = ev["detection"]
    header = (f"THERMAL DATA ANALYZED: {'yes, ' + str(det['thermal_frames']) + ' frames' if det['thermal_frames'] else 'no'}. "
              "Thermal pixels above 80 C mean fire even when RGB shows only smoke.\n")
    content = [T.image_part(f.rgb_b64(max_side=640)) for f in (seq.frames[0], seq.frames[-1])]
    content.append({"type": "text", "text": ("Observer guidance:\n" + WATCH_FOR + "\n\n" if flags["writeup"] else "")
                    + WRITEUP + header + _canon(_measurement_view(state))})
    t0 = time.time()
    r = ep.chat([{"role": "user", "content": content}], max_tokens=1500,
                response_format={"type": "json_schema", "json_schema": {"name": "recon_brief", "schema": BRIEF_SCHEMA, "strict": True}})
    ch = r["choices"][0]
    step("llm_writeup", t0, finish_reason=ch.get("finish_reason"), usage=r.get("usage"))
    try:
        raw_brief = json.loads(ch["message"]["content"])
    except (TypeError, ValueError) as e:
        raise RuntimeError(f"writeup not parsable (finish_reason={ch.get('finish_reason')}, "
                           f"completion_tokens={(r.get('usage') or {}).get('completion_tokens')}): {e}") from e

    llm_decision = {"status": raw_brief["fire_smoke"]["status"], "confidence": raw_brief["fire_smoke"]["confidence"],
                    "direction": raw_brief["spread_direction"]["direction"], "trend": raw_brief["trend"]["status"],
                    "overall_confidence": raw_brief["overall_confidence"], "seconds": state["trace"][-1]["seconds"]}
    measured = dict(zip(("status", "confidence", "why"), evidence_status(det, ev["probes"])))
    measured["smoke_frames"] = smoke_signal(ev["probes"])
    brief, fixes = enforce_policy(copy.deepcopy(raw_brief), ev)

    decider_calls = {}
    if deciders:
        from . import deciders as D
        for name in deciders:
            t0 = time.time()
            decider_calls[name] = D.decide(name, ev, seq, ep.h)
            step(f"decider_{name}", t0)
    return {"version": AGENT_VERSION, "brief": brief, "raw_brief": raw_brief, "policy_corrections": fixes, "evidence": ev,
            "trace": state["trace"], "planner": planner, "llm_decision": llm_decision, "measured_decision": measured,
            "deciders": decider_calls,
            "meta": {"sequence_id": src_id, "model_sequence_id": seq.id, "n_frames": n, "span_s": seq.span_s, "mode": mode, "llm_model": ep.llm_model,
                     "guidance": flags, "seed": T.SEED, "run_tag": run_tag, "id_map": id_map}}
