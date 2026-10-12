"""Kai decision model (Decision-2.0-Kai-0.6B, text only) behind the same SystemOne request format as Clef.

Request:  {"state": "<evidence in sentences>", "questions": {...}}   ("model" is accepted and ignored)
Response: SystemOne body ({"answers": ...}) plus "timing_ms" (time inside the model call).
Loads the pinned release with transformers (trust_remote_code), the same way recon_gateway.kai_decider does.
"""
import threading
import time

KAI_REPO = "vllm-sr/Decision-2.0-Kai-0.6B"
KAI_REV = "cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764"  # a later README-only commit breaks the file-hash check


class Model:
    def __init__(self, **kwargs):
        self._lock = threading.Lock()

    def load(self):
        from transformers import AutoModel
        self._model = AutoModel.from_pretrained(KAI_REPO, revision=KAI_REV, trust_remote_code=True)

    def predict(self, request):
        t0 = time.perf_counter()
        with self._lock:
            out = dict(self._model.system_one(state=request["state"], questions=request["questions"]))
        out["timing_ms"] = round(1000 * (time.perf_counter() - t0), 1)
        return out
