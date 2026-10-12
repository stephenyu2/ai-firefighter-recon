"""Runs the precision probe once on the deployment's GPU, in a background thread started at load.

Request: {} -> {"status": "running" | "done" | "failed", "env": {...}, "rows": [...], "log": [...]}
"""
import os
import sys
import threading
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


class Model:
    def __init__(self, **kwargs):
        self._data_dir = Path(kwargs.get("data_dir", Path(__file__).parent.parent / "data"))
        self.state = {"status": "starting", "env": {}, "rows": [], "log": []}

    def load(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            import probe
            self.state["env"] = probe.env()
            self.state["status"] = "running"
            modes = [m for m in os.environ.get("PROBE_MODES", "").split(",") if m] or None
            self.state["rows"] = probe.run(self._data_dir, self._data_dir / "frames", log=self.state["log"].append, modes=modes)
            self.state["status"] = "done"
        except Exception:
            self.state["log"].append(traceback.format_exc()[-3000:])
            self.state["status"] = "failed"

    def predict(self, request):
        return self.state
