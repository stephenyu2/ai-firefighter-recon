"""d3 decision model (vLLM Semantic Router Decision 3.0) through the package's own runtime, with image input.

Request:  {"state": ..., "questions": {...}, "images": ["<base64 PNG/JPEG/WebP or data URL>", ...], "model": ignored}
Response: System One body ({"model", "answers", "usage"}) plus "timing_ms" (time inside the model call).
Images go through the runtime's strict loader, the same checks as the package's own HTTP server: PNG, JPEG or WebP
only, at most 8 MB and 16 MP each, integrity-checked. At most D3_MAX_IMAGES images (default 16); videos are not
accepted. A request that fails these checks gets 422 with {"detail": ...}.
D3_PATH is the model package (weights plus d3_runtime.py and friends, as published on the Hub).
"""
import os
import sys
import threading
import time

from fastapi.responses import JSONResponse

PATH = os.environ.get("D3_PATH", "/models/d3")
MAX_IMAGES = int(os.environ.get("D3_MAX_IMAGES", "16"))
FIELDS = {"state", "questions", "images", "model"}


def _refuse(detail):
    return JSONResponse(status_code=422, content={"detail": detail})


class Model:
    def __init__(self, **kwargs):
        self._lock = threading.Lock()

    def load(self):
        sys.path.insert(0, PATH)
        from d3_runtime import D3
        # D3_BATCH_SIZE: questions per forward pass (default 8); a smaller value lowers warm-up memory on 24 GB cards
        self._model = D3.from_pretrained(PATH, device="cuda:0", local_files_only=True,
                                         batch_size=int(os.environ.get("D3_BATCH_SIZE", "8")))
        self._model.warmup()  # compiles the linear-attention kernels for every batch size before serving

    def predict(self, request):
        if not isinstance(request, dict) or not isinstance(request.get("questions"), dict):
            return _refuse("the body must be a JSON object with a questions object")
        extra = set(request) - FIELDS
        if extra:
            return _refuse(f"unsupported fields {sorted(extra)}; this deployment takes state, questions and images")
        images = request.get("images") or []
        if not isinstance(images, list) or len(images) > MAX_IMAGES or not all(isinstance(s, str) for s in images):
            return _refuse(f"images must be a list of at most {MAX_IMAGES} base64 strings")
        try:
            decoded = self._model.load_images(
                [s if s.startswith("data:") else f"data:image/jpeg;base64,{s}" for s in images], strict=True)
        except ValueError as exc:
            return _refuse(str(exc))
        call = {"state": request.get("state"), "questions": request["questions"]}
        if decoded:
            call["images"] = decoded
        t0 = time.perf_counter()
        with self._lock:  # the runtime runs one request at a time
            out = self._model.system_one(**call)
        out["timing_ms"] = round(1000 * (time.perf_counter() - t0), 1)
        return out
