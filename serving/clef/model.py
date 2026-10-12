"""Clef decision model (Jev/SystemOne API) using the model's own release code, with images.
CLEF_PATH is a local copy of the model repo (Clef 27B or Clef-Flash 9B); its joint_schema_model.py does the work.

Request: {"state": ..., "questions": {...}, "images": ["<base64 JPEG/PNG or data URL>", ...]}
Response: SystemOne body ({"model", "answers", "usage"}) plus "timing_ms" (time inside the model call).
"""
import base64
import io
import os
import sys
import threading
import time

from PIL import Image

PATH = os.environ.get("CLEF_PATH", "/models/clef")


def _image(s):
    return Image.open(io.BytesIO(base64.b64decode(s.split(",", 1)[-1]))).convert("RGB")


class Model:
    def __init__(self, **kwargs):
        self._lock = threading.Lock()

    def load(self):
        sys.path.insert(0, PATH)
        from joint_schema_model import load_release_model, systemone
        self._systemone = systemone
        self._model, self._processor = load_release_model(PATH, device="cuda")
        self.predict({"state": "The parcel arrived damaged.", "questions": {"damaged": {"type": "noul", "instructions": "Is the parcel damaged?"}}})

    def predict(self, request):
        req = dict(request)
        req.setdefault("model", "clef")
        if req.get("images"):
            req["images"] = [_image(s) for s in req["images"]]
        t0 = time.perf_counter()
        with self._lock:
            out = self._systemone(self._model, self._processor, req)
        out["timing_ms"] = round(1000 * (time.perf_counter() - t0), 1)
        return out
