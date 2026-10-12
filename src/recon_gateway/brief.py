"""Brief schema, policy checks (prefer 'indeterminate' over unsupported claims), and Markdown rendering."""
import json
import re

TREND = ["increasing", "decreasing", "stable", "indeterminate"]
DIRECTION = ["toward top of frame", "toward bottom of frame", "toward left of frame", "toward right of frame",
             "toward top-left of frame", "toward top-right of frame", "toward bottom-left of frame",
             "toward bottom-right of frame", "no clear movement", "indeterminate"]
PRESENCE = ["fire", "smoke only", "no fire or smoke", "indeterminate"]
CONF = ["high", "medium", "low"]

BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "fire_smoke": {"type": "object", "properties": {
            "status": {"type": "string", "enum": PRESENCE}, "confidence": {"type": "string", "enum": CONF},
            "evidence": {"type": "string"}}, "required": ["status", "confidence", "evidence"], "additionalProperties": False},
        "extent": {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"],
                   "additionalProperties": False},
        "location_in_frame": {"type": "string"},
        "spread_direction": {"type": "object", "properties": {
            "direction": {"type": "string", "enum": DIRECTION}, "basis": {"type": "string"}}, "required": ["direction", "basis"],
            "additionalProperties": False},
        "trend": {"type": "object", "properties": {
            "status": {"type": "string", "enum": TREND}, "basis": {"type": "string"}}, "required": ["status", "basis"],
            "additionalProperties": False},
        "overall_confidence": {"type": "string", "enum": CONF},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "observations": {"type": "array", "items": {"type": "string"}},
        "inferences": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["fire_smoke", "extent", "location_in_frame", "spread_direction", "trend", "overall_confidence",
                 "limitations", "observations", "inferences", "summary"],
    "additionalProperties": False,
}

BANNED = re.compile(r"\b(acres?|hectares?|gps|latitude|longitude|mph|km/h|miles? per hour|dispatch|send crews|evacuat\w*)\b", re.I)
PROBE_YES = 0.8  # an image probe counts as "yes" when its yes-probability is at least this and its answer is reliable
CONF_RANK = {"low": 0, "medium": 1, "high": 2}


def _probe_count(probes, key):
    """How many probed frames answered yes to `key` with probability >= PROBE_YES (unreliable answers are ignored)."""
    n = 0
    for frame in (probes or {}).values():
        q = frame.get(key) or {}
        if q.get("answer") is not None and (q.get("probabilities") or {}).get("yes", 0.0) >= PROBE_YES:
            n += 1
    return n


def evidence_status(det, probes=None):
    """Fire status from measurements, as (status, confidence, why). status is "fire", "no fire signal" or "indeterminate".

    Thermal decides when present. Without thermal, the detector decides, and the gateway's fixed image probes (flames
    visible, asked on the same frames for every brief) can raise an RGB-only sequence with no detector hits to "fire" at
    low confidence. Smoke is reported separately (smoke_signal) and never decides the status."""
    n_th, f_th = det.get("thermal_frames", 0), det.get("thermal_fire_frames", 0)
    n = len(det.get("frames", [])) or 1
    f_det = det.get("detector_fire_frames", 0)
    flames = _probe_count(probes, "flames_visible")
    n_probe = len(probes or {})
    if n_th:
        if f_th / n_th >= 0.5:
            return "fire", "high" if f_det / n >= 0.5 else "medium", f"thermal shows fire above 80 C in {f_th} of {n_th} frames; detector flags {f_det} of {n}"
        if f_th:
            return "fire", "medium", f"thermal shows fire above 80 C in {f_th} of {n_th} frames (part of the sequence); detector flags {f_det} of {n}"
        if f_det == 0:
            return "no fire signal", "high", f"no pixels above 80 C in {n_th} thermal frames and no detector hits"
        return "indeterminate", "low", f"detector flags {f_det} of {n} frames but thermal shows nothing above 80 C"
    if f_det / n >= 0.5:
        return "fire", "medium", f"detector flags {f_det} of {n} frames (RGB only, no thermal)"
    if flames >= 2:
        return "fire", "low", f"detector flags {f_det} of {n} RGB frames, but image probes see flames on {flames} of {n_probe} probed frames (RGB only, no thermal)"
    if f_det:
        return "indeterminate", "low", f"detector flags {f_det} of {n} frames (RGB only, no thermal)"
    return "no fire signal", "medium", f"no detector hits in {n} RGB frames and image probes see flames on {flames} of {n_probe} probed frames (no thermal)"


def smoke_signal(probes):
    return _probe_count(probes, "smoke_visible")


def rule_is_authoritative(det):
    """The rule may overrule the LLM only on strong measurements: thermal data, or the detector flagging most RGB frames.
    On RGB-only sequences with weak evidence (few or no detector hits) the rule is advice, recorded as a note."""
    n = len(det.get("frames", [])) or 1
    return bool(det.get("thermal_frames", 0)) or det.get("detector_fire_frames", 0) / n >= 0.5


def enforce_policy(brief, evidence):
    """Apply the Milestone 2 failure policy using the tool evidence. Returns (brief, list of corrections).
    The status rule only overrides the LLM when they conflict: "fire" vs not fire, or a status the rule cannot support.
    "smoke only" and "no fire or smoke" are both accepted when the rule finds no fire signal."""
    fixes = []
    det0 = evidence.get("detection") or {}
    probes = evidence.get("probes")
    if det0:
        st, conf, why = evidence_status(det0, probes)
        llm = brief["fire_smoke"]["status"]
        new = None
        if st == "fire" and llm != "fire":
            new = "fire"
        elif st == "no fire signal" and llm == "fire":
            new = "smoke only" if smoke_signal(probes) >= 2 else "no fire or smoke"
        elif st == "indeterminate" and llm in ("fire", "no fire or smoke", "smoke only"):
            new = "indeterminate"
        if new and rule_is_authoritative(det0):
            fixes.append(f"fire status '{llm}' contradicted the measurements; set to '{new}' ({why})")
            brief["fire_smoke"] = {"status": new, "confidence": conf, "evidence": why}
            if CONF_RANK[brief["overall_confidence"]] > CONF_RANK[conf]:
                brief["overall_confidence"] = conf
            brief["summary"] = f"Fire status set by the measurement rule to '{new}' ({why}). " + brief["summary"]
            brief["limitations"].append(f"The writeup's own fire status was '{llm}'; the measurement rule overrode it.")
        elif new:
            fixes.append(f"note: measurement rule suggests '{new}' at {conf} confidence ({why}); not authoritative on RGB-only evidence, LLM status kept")
            brief["limitations"].append(f"The measurement rule suggests '{new}' at {conf} confidence ({why}).")
        elif CONF_RANK[brief["fire_smoke"]["confidence"]] > CONF_RANK[conf]:
            brief["fire_smoke"]["confidence"] = conf  # agreement: never more confident than the measurements
        hot = [f["thermal_hot_percent"] for f in det0.get("frames", []) if f.get("thermal_hot_percent") is not None]
        brief["extent"]["hot_area_percent_of_frame"] = round(sum(hot) / len(hot), 4) if hot else None  # computed, never asked of the model
    for key in ("observations", "inferences", "limitations"):
        seen, uniq = set(), []
        for x in brief.get(key, []):
            if x.strip().lower() not in seen:
                seen.add(x.strip().lower()); uniq.append(x)
        brief[key] = uniq
    spread = evidence.get("spread") or {}
    if not spread.get("direction_supported"):
        if brief["spread_direction"]["direction"] not in ("indeterminate",):
            fixes.append(f"spread direction '{brief['spread_direction']['direction']}' not supported by the tracked evidence; set to indeterminate")
            brief["spread_direction"] = {"direction": "indeterminate", "basis": spread.get("direction_reason", "no supporting evidence")}
    if not spread.get("trend_supported"):
        if brief["trend"]["status"] != "indeterminate":
            fixes.append(f"trend '{brief['trend']['status']}' not supported by the tracked evidence; set to indeterminate")
            brief["trend"] = {"status": "indeterminate", "basis": spread.get("trend_reason", "no supporting evidence")}
    det = evidence.get("detection") or {}
    text = json.dumps(brief)
    if BANNED.search(text):
        fixes.append("removed out-of-scope units or operational directives (acres, GPS, speeds, dispatch)")
        for key in ("summary",):
            brief[key] = BANNED.sub("[out of scope]", brief[key])
        brief["inferences"] = [BANNED.sub("[out of scope]", s) for s in brief["inferences"]]
    no_incident = brief["fire_smoke"]["status"] == "no fire or smoke"
    if no_incident:
        for k in ("spread_direction", "trend"):
            fld = "direction" if k == "spread_direction" else "status"
            if brief[k][fld] == "indeterminate":
                brief[k]["basis"] = "not applicable: no fire or smoke detected"
    if det.get("thermal_frames", 0) > 0:
        kept = [x for x in brief["limitations"] if not re.search(r"rgb[- ]only|no thermal|lack of thermal", x, re.I)]
        if len(kept) < len(brief["limitations"]):
            fixes.append("removed a limitation claiming RGB-only data; thermal frames were analyzed")
            brief["limitations"] = kept
    if brief["overall_confidence"] == "high" and not no_incident and (brief["spread_direction"]["direction"] == "indeterminate"
                                                                    or brief["fire_smoke"]["status"] == "indeterminate"):
        brief["overall_confidence"] = "medium"
        fixes.append("lowered overall confidence to medium because a core field is indeterminate")
    return brief, fixes


def to_markdown(brief, meta):
    fs, sp, tr = brief["fire_smoke"], brief["spread_direction"], brief["trend"]
    area = brief["extent"].get("hot_area_percent_of_frame")
    lines = [f"# Reconnaissance brief: {meta.get('sequence_id', 'sequence')}", "",
             f"{brief['summary']}", "",
             "| Field | Assessment | Basis |", "|---|---|---|",
             f"| Fire / smoke | {fs['status']} ({fs['confidence']} confidence) | {fs['evidence']} |",
             f"| Extent | {brief['extent']['description']}" + (f" ({area:.2f}% of frame above 80 C)" if area is not None else "") + " | |",
             f"| Location in frame | {brief['location_in_frame']} | |",
             f"| Spread direction | {sp['direction']} | {sp['basis']} |",
             f"| Trend | {tr['status']} | {tr['basis']} |",
             f"| Overall confidence | {brief['overall_confidence']} | |", ""]
    for title, key in [("Observations", "observations"), ("Inferences", "inferences"), ("Limitations", "limitations")]:
        if brief.get(key):
            lines += [f"**{title}**", ""] + [f"- {s}" for s in brief[key]] + [""]
    lines += [f"*Frames analyzed: {meta.get('n_frames')}, span {meta.get('span_s')} s. Decision support only; a person makes every call.*"]
    return "\n".join(lines) + "\n"
