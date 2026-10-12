"""Tools the gateway agent can call, plus thin HTTP clients for the classifier and the LLM."""
import base64
import io
import json
import math
import os
import time

import numpy as np
import requests
from PIL import Image

SEED = int(os.environ.get("RECON_SEED", "1234"))  # fixed sampling seed for every LLM call, so reruns can be compared


class Endpoints:
    """HTTP clients for the classifier and the LLM. With RECON_GATEWAY_KEY set, every call goes through the LiteLLM
    gateway with that key (the gateway holds the upstream keys). Otherwise calls go straight to the endpoints, with
    RECON_API_KEY sent as "<RECON_AUTH_SCHEME> <key>" (scheme defaults to Bearer; no key means no auth header).
    run_tag (or RECON_RUN_TAG) is sent as the x-recon-run header so gateway logs can be split by run."""

    def __init__(self, llm_model=None, run_tag=None):
        self.gateway_key = os.environ.get("RECON_GATEWAY_KEY", "")
        self.key = self.gateway_key or os.environ.get("RECON_API_KEY", "")
        self.classifier_url = os.environ.get("RECON_CLASSIFIER_URL", "")
        self.llm_url = os.environ.get("RECON_LLM_URL", "").rstrip("/")
        self.llm_model = llm_model or os.environ.get("RECON_LLM_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
        if not (self.classifier_url and self.llm_url):
            raise SystemExit("Set RECON_CLASSIFIER_URL and RECON_LLM_URL, plus RECON_GATEWAY_KEY or RECON_API_KEY (see gateway/README.md).")
        scheme = "Bearer" if self.gateway_key else os.environ.get("RECON_AUTH_SCHEME", "Bearer")
        self.h = {"Content-Type": "application/json"}
        if self.key:
            self.h["Authorization"] = f"{scheme} {self.key}"
        tag = run_tag or os.environ.get("RECON_RUN_TAG")
        if tag:
            self.h["x-recon-run"] = tag

    def classify(self, frames, timeout=300):
        r = None
        for attempt in range(4):
            try:
                r = requests.post(self.classifier_url, headers=self.h, json={"frames": frames}, timeout=timeout)
                if r.status_code not in (408, 429) and r.status_code < 500:
                    break
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 3:
                    raise
            time.sleep(10 * (attempt + 1))
        r.raise_for_status()
        return r.json()

    def chat(self, messages, timeout=300, **kw):
        """One chat completion. temperature 0 and a fixed seed unless the caller overrides them. Retries 408, 429, 5xx
        and timeouts with the identical body; any other 4xx fails at once."""
        body = {"model": self.llm_model, "messages": messages, "temperature": 0, "seed": SEED, **kw}
        r = None
        for attempt in range(4):  # retries cold starts and transient errors (408/429/5xx, timeouts) with the identical body
            try:
                r = requests.post(f"{self.llm_url}/chat/completions", headers=self.h, json=body, timeout=timeout)
                if r.status_code not in (408, 429) and r.status_code < 500:
                    break
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 3:
                    raise
            time.sleep(10 * (attempt + 1))
        r.raise_for_status()
        return r.json()


def image_part(b64, mime="image/jpeg"):
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


# ---------------- tool implementations ----------------

UNITS = {"detector_p_fire": "probability 0 to 1 from the fine-tuned RGB fire detector",
         "clip_p_fire": "probability 0 to 1 from a general image model (CLIP)",
         "thermal_hot_percent": "percent of the frame (0 to 100) at or above 80 C; 0.5 means half of one percent",
         "thermal_hot_pixels": "count of pixels at or above 80 C",
         "thermal_max_c": "hottest pixel, degrees C"}


def detect_fire(ep, seq, cache):
    """Detector, CLIP and thermal tools on every frame of the sequence (the gateway always runs this itself)."""
    todo = [f for f in seq.frames if f.id not in cache]
    for i in range(0, len(todo), 8):
        batch = todo[i:i + 8]
        res = ep.classify([{"id": f.id, "rgb_b64": f.rgb_b64(), **({"tiff_b64": f.tiff_b64()} if f.tiff_path else {})} for f in batch])
        for r in res["frames"]:
            cache[r["id"]] = r
    rows = [cache[f.id] for f in seq.frames if f.id in cache]
    det = [r["dinov2_lora"]["p_fire"] for r in rows]
    clip = [r["clip"]["p_fire"] for r in rows]
    therm = [r["thermal"] for r in rows if r.get("thermal")]
    summary = {
        "units": UNITS,
        "frames": [{"id": r["id"], "detector_p_fire": r["dinov2_lora"]["p_fire"], "detector_fire": r["dinov2_lora"]["fire_recall_tuned"],
                    "clip_p_fire": r["clip"]["p_fire"],
                    "thermal_hot_percent": round(100 * r["thermal"]["hot_fraction"], 4) if r.get("thermal") else None,
                    "thermal_hot_pixels": r["thermal"]["hot_pixels"] if r.get("thermal") else None,
                    "thermal_max_c": r["thermal"]["max_c"] if r.get("thermal") else None,
                    "fire_touches_edge": r["thermal"]["touches_edge"] if r.get("thermal") else None} for r in rows],
        "detector_fire_frames": int(sum(r["dinov2_lora"]["fire_recall_tuned"] for r in rows)),
        "clip_fire_frames": int(sum(r["clip"]["fire_default"] for r in rows)),
        "thermal_frames": len(therm),
        "thermal_fire_frames": int(sum(t["fire_present"] for t in therm)),
        "disagreements": [r["id"] for r in rows if r["dinov2_lora"]["fire_recall_tuned"] != r["clip"]["fire_default"]
                          or (r.get("thermal") and r["thermal"]["fire_present"] != r["dinov2_lora"]["fire_recall_tuned"])],
        "max_detector_p": round(max(det), 4) if det else None, "max_clip_p": round(max(clip), 4) if clip else None,
    }
    summary["any_fire"] = bool(summary["detector_fire_frames"] or summary["thermal_fire_frames"])
    return summary


def _gray_small(frame, side=160):
    im = Image.open(frame.rgb_path).convert("L")
    im = im.resize((side, int(side * im.height / im.width)))
    a = np.asarray(im, dtype=np.float32)
    return (a - a.mean()) / (a.std() + 1e-6)


def camera_shift(f0, f1):
    """Global translation between two frames by phase correlation, as a fraction of frame size."""
    a, b = _gray_small(f0), _gray_small(f1)
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    F0, F1 = np.fft.fft2(a * win), np.fft.fft2(b * win)
    R = F0 * np.conj(F1); R /= np.abs(R) + 1e-9
    r = np.abs(np.fft.ifft2(R))
    dy, dx = np.unravel_index(np.argmax(r), r.shape)
    dy = dy - a.shape[0] if dy > a.shape[0] // 2 else dy
    dx = dx - a.shape[1] if dx > a.shape[1] // 2 else dx
    peak = float(r.max() / (r.mean() + 1e-9))
    return {"dx": round(-dx / a.shape[1], 4), "dy": round(-dy / a.shape[0], 4), "peak_ratio": round(peak, 1)}


def _direction(dx, dy):
    ang = math.degrees(math.atan2(-dy, dx)) % 360
    names = ["toward right of frame", "toward top-right of frame", "toward top of frame", "toward top-left of frame",
             "toward left of frame", "toward bottom-left of frame", "toward bottom of frame", "toward bottom-right of frame"]
    return names[int(((ang + 22.5) % 360) // 45)]


def track_spread(seq, cache):
    """Trend from the 80 C hot-area change; direction from hot-area centroid displacement, corrected for camera motion."""
    th = [(f, cache.get(f.id, {}).get("thermal")) for f in seq.frames]
    th = [(f, t) for f, t in th if t]
    out = {"thermal_frames": len(th), "trend": "indeterminate", "trend_supported": False, "direction": "indeterminate",
           "direction_supported": False}
    if len(th) < 3:
        out["trend_reason"] = out["direction_reason"] = "fewer than 3 frames with thermal data"
        return out
    k = max(1, len(th) // 3)
    a0 = float(np.mean([t["hot_fraction"] for _, t in th[:k]])); a1 = float(np.mean([t["hot_fraction"] for _, t in th[-k:]]))
    shift = camera_shift(th[0][0], th[-1][0])
    cam = math.hypot(shift["dx"], shift["dy"])
    out.update(hot_percent_start=round(100 * a0, 4), hot_percent_end=round(100 * a1, 4), hot_percent_units=UNITS["thermal_hot_percent"],
               camera_shift=shift, camera_shift_fraction=round(cam, 3), span_s=round(th[-1][0].t - th[0][0].t, 1))
    moving = cam > 0.08 or shift["peak_ratio"] < 8
    if a0 == 0 and a1 == 0:
        out.update(trend="indeterminate", trend_reason="no pixels above 80 C in the sequence")
    elif moving:
        out.update(trend_reason=f"camera moved about {100 * cam:.0f}% of the frame between first and last frame, so hot-area change mixes framing with fire change")
    else:
        ratio = (a1 + 1e-6) / (a0 + 1e-6)
        trend = "increasing" if ratio > 1.25 else "decreasing" if ratio < 0.8 else "stable"
        out.update(trend=trend, trend_supported=True, trend_reason=f"hot area went from {100 * a0:.3g}% to {100 * a1:.3g}% of the frame (ratio {ratio:.2f}) with a steady camera")
    c = [(f, t["centroid_xy"]) for f, t in th if t.get("centroid_xy")]
    if len(c) < 3:
        out["direction_reason"] = "hot area visible in fewer than 3 frames"
    elif moving:
        out["direction_reason"] = out.get("trend_reason") if out["trend"] == "indeterminate" else f"camera moved about {100 * cam:.0f}% of the frame"
    else:
        dx = c[-1][1][0] - c[0][1][0] - shift["dx"]; dy = c[-1][1][1] - c[0][1][1] - shift["dy"]
        mag = math.hypot(dx, dy)
        out.update(centroid_shift={"dx": round(dx, 4), "dy": round(dy, 4)})
        if mag < 0.05:
            out.update(direction="no clear movement", direction_supported=True, direction_reason=f"hot-area center moved only {100 * mag:.1f}% of the frame")
        else:
            out.update(direction=_direction(dx, dy), direction_supported=True,
                       direction_reason=f"hot-area center moved {100 * mag:.1f}% of the frame after removing camera motion")
    return out


QUESTION_BANK = {
    "smoke_visible": ("Is smoke visible in this aerial image?", ["yes", "no"]),
    "flames_visible": ("Are flames or glowing fire visible in this aerial image?", ["yes", "no"]),
    "view_obscured": ("How much of the ground is hidden by smoke or haze?", ["none", "some", "most or all"]),
    "fire_region": ("Where in the image is the fire or smoke concentrated?", ["left", "center", "right", "top", "bottom", "spread across the frame", "none visible"]),
}
LETTERS = "ABCDEFG"
MIN_COVERAGE = 0.5  # below this much probability on the option letters, the answer is reported as unreliable


def score_options(top_logprobs, choices):
    """Option probabilities from the first generated token's top log-probs.

    A token counts for option k only if, stripped of spaces and a trailing '.', ')' or ':', it is exactly that option's
    letter. Probabilities are returned in the question's own option order (not sorted), unrounded. coverage is the total
    probability on option letters; the answer is None when coverage is below MIN_COVERAGE. Ties go to the earlier option."""
    want = {LETTERS[i]: c for i, c in enumerate(choices)}
    mass = {c: 0.0 for c in choices}
    for t in top_logprobs:
        tok = t["token"].strip().rstrip(".):")
        if tok in want:
            mass[want[tok]] += math.exp(t["logprob"])
    z = sum(mass.values())
    probs = {c: (m / z if z > 0 else 0.0) for c, m in mass.items()}
    ranked = sorted(choices, key=lambda c: (-probs[c], choices.index(c)))
    answer = ranked[0] if z >= MIN_COVERAGE else None
    margin = probs[ranked[0]] - probs[ranked[1]] if len(ranked) > 1 else 1.0
    return {"probabilities": {c: round(p, 6) for c, p in probs.items()}, "answer": answer,
            "confidence": round(probs[ranked[0]], 6) if answer else None, "margin": round(margin, 6), "coverage": round(z, 6)}


def decide(ep, frame, question_keys, system=None):
    """Jev-style structured decision: per-option probabilities from next-token log-probs, no free-text generation.
    system: optional system message (prompts.WATCH_FOR observer guidance). Raw top log-probs are kept for audit."""
    out = {}
    b64 = frame.rgb_b64(max_side=640)
    for key in question_keys:
        if key not in QUESTION_BANK:
            continue
        q, choices = QUESTION_BANK[key]
        opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
        msg = ([{"role": "system", "content": system}] if system else []) + \
              [{"role": "user", "content": [image_part(b64), {"type": "text", "text": f"{q}\n{opts}\nAnswer with only the letter."}]}]
        r = ep.chat(msg, max_tokens=1, logprobs=True, top_logprobs=20)
        top = r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        out[key] = {"question": q, **score_options(top, choices),
                    "raw_top_logprobs": [[t["token"], round(t["logprob"], 4)] for t in top]}
    return out
