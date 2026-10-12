"""Optional Jev-like decision layer: Decision-2.0-Kai-0.6B (text-only, Apache-2.0) over the gateway's evidence, phrased in words.
Kai reads numbers poorly (FLAME 3 ROC-AUC 0.76 with raw probabilities) but well when the evidence is verbalized (AUC 1.0 on a
200-frame check), so the state is written as sentences. Runs locally on CPU or Apple GPU; no deployment needed."""
KAI_REPO = "vllm-sr/Decision-2.0-Kai-0.6B"
KAI_REV = "cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764"
_model = None


def _load():
    global _model
    if _model is None:
        from transformers import AutoModel
        _model = AutoModel.from_pretrained(KAI_REPO, revision=KAI_REV, trust_remote_code=True)
    return _model


def _words(p):
    return ("very confident there is fire" if p > 0.9 else "leaning toward fire" if p > 0.5
            else "leaning toward no fire" if p > 0.1 else "very confident there is no fire")


def state_text(evidence):
    det = evidence.get("detection") or {}
    frames = det.get("frames", [])
    n = len(frames) or 1
    s = [f"A wildfire drone sequence of {n} frames was analyzed."]
    if frames:
        mx = max(f["detector_p_fire"] for f in frames)
        s.append(f"The trained fire detector flags fire in {det.get('detector_fire_frames', 0)} of {n} frames and is {_words(mx)} on its strongest frame.")
        s.append(f"A general image model flags fire in {det.get('clip_fire_frames', 0)} of {n} frames.")
    if det.get("thermal_frames"):
        hot = [f["thermal_hot_percent"] for f in frames if f.get("thermal_hot_percent")]
        s.append(f"The thermal camera shows hot spots above 80 C in {det.get('thermal_fire_frames', 0)} of {det['thermal_frames']} frames"
                 + (f", covering up to {max(hot):.2g}% of the frame." if hot else "."))  # .2f printed small fires as 0.00%
    else:
        s.append("No thermal camera data is available.")
    for fid, qs in (evidence.get("decisions") or {}).items():
        for q in qs.values():
            if q.get("answer"):
                s.append(f"A vision model answered '{q['question']}' with '{q['answer']}' at {100 * (q.get('confidence') or 0):.0f}% probability.")
    sp = evidence.get("spread") or {}
    if sp.get("camera_shift_fraction") is not None:
        s.append(f"The camera moved about {100 * sp['camera_shift_fraction']:.0f}% of the frame during the sequence.")
    return " ".join(s)


QUESTIONS = {
    "status": {"type": "choice", "instructions": "What does the evidence show about this sequence?",
               "criteria": {"fire": "Active fire is present", "smoke_only": "Smoke is present but no fire",
                            "none": "No fire or smoke", "unclear": "The evidence conflicts or is too weak to tell"}},
    "direction_assessable": {"type": "noul", "instructions": "Is the evidence strong enough to state which way the fire is spreading?"},
    "confidence": {"type": "score", "instructions": "How strongly does the evidence support the overall assessment?", "criteria": ["Low", "Medium", "High"]},
}


def decide(evidence):
    st = state_text(evidence)
    r = _load().system_one(state=st, questions=QUESTIONS)["answers"]
    return {"state": st, "status": r["status"]["choice"], "status_probabilities": {k: round(v, 3) for k, v in r["status"]["probabilities"].items()},
            "direction_assessable_p": round(r["direction_assessable"]["noul"], 3),
            "confidence": r["confidence"]["legend"][str(round(r["confidence"]["score"]))] if "legend" in r["confidence"] else None}
