"""Writes one JSON line per gateway request to logs/requests.jsonl: route, run tag, latency, time to first token
(streamed calls only), tokens, finish_reason and tool-call names. Pass-through failures are logged too.
Pass-through routes are named by the last segment of their path in the proxy config (LITELLM_CONFIG, default
config.yaml next to this file), matched on the target URL."""
import json
import os
from datetime import datetime
from pathlib import Path

import yaml
from litellm.integrations.custom_logger import CustomLogger

HERE = Path(__file__).parent
LOG = HERE / "logs" / "requests.jsonl"


def _passthrough_names():
    cfg = Path(os.environ.get("LITELLM_CONFIG", HERE / "config.yaml"))
    try:
        eps = (yaml.safe_load(cfg.read_text()).get("general_settings") or {}).get("pass_through_endpoints") or []
    except (OSError, AttributeError, yaml.YAMLError):
        return {}
    names = {str(e["target"]).rstrip("/"): e["path"].rstrip("/").rsplit("/", 1)[-1] for e in eps if e.get("target") and e.get("path")}
    return dict(sorted(names.items(), key=lambda kv: -len(kv[0])))  # longest target first, so prefixes never shadow


PASSTHROUGH = _passthrough_names()


def _secs(a, b):
    try:
        return round((b - a).total_seconds(), 4)
    except Exception:
        return None


def _headers(kwargs):
    lp = kwargs.get("litellm_params") or {}
    h = (lp.get("proxy_server_request") or {}).get("headers") or (lp.get("metadata") or {}).get("headers") or {}
    try:
        return {str(k).lower(): v for k, v in dict(h).items()}
    except Exception:
        return {}


def _write(row):
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


class RequestLog(CustomLogger):
    def _row(self, kwargs, response_obj, start_time, end_time, ok):
        lp = kwargs.get("litellm_params") or {}
        meta = lp.get("metadata") or {}
        target = str(lp.get("api_base") or kwargs.get("api_base") or (kwargs.get("passthrough_logging_payload") or {}).get("url") or "")
        passthrough = next((v for k, v in PASSTHROUGH.items() if target.startswith(k)), None)
        usage = getattr(response_obj, "usage", None) if ok else None
        choice = (getattr(response_obj, "choices", None) or [None])[0] if ok else None
        msg = getattr(choice, "message", None)
        calls = getattr(msg, "tool_calls", None) or []
        return {"ok": ok, "route": meta.get("model_group") or (f"pass-through/{passthrough}" if passthrough else kwargs.get("model")),
                "run_tag": _headers(kwargs).get("x-recon-run"), "target": target[:120],
                "start": start_time.isoformat() if start_time else None, "latency_s": _secs(start_time, end_time),
                "ttft_s": _secs(start_time, kwargs.get("completion_start_time")) if kwargs.get("stream") is True else None,
                "prompt_tokens": getattr(usage, "prompt_tokens", None), "completion_tokens": getattr(usage, "completion_tokens", None),
                "finish_reason": getattr(choice, "finish_reason", None),
                "tool_calls": [getattr(getattr(c, "function", None), "name", None) for c in calls],
                "error": None if ok else str(kwargs.get("exception"))[:300]}

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        _write(self._row(kwargs, response_obj, start_time, end_time, True))

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        _write(self._row(kwargs, response_obj, start_time, end_time, False))

    async def async_post_call_failure_hook(self, request_data, original_exception, user_api_key_dict, traceback_str=None):
        """Pass-through failures do not reach async_log_failure_event; chat failures already do."""
        if (request_data or {}).get("call_type") != "pass_through_endpoint":
            return
        target = str(((request_data or {}).get("passthrough_logging_payload") or {}).get("url") or "")
        _write({"ok": False, "route": "pass-through/" + next((v for k, v in PASSTHROUGH.items() if target.startswith(k)), "unknown"),
                "run_tag": _headers(request_data).get("x-recon-run"), "target": target[:120], "start": datetime.now().isoformat(),
                "error": f"{type(original_exception).__name__}: {str(original_exception)[:300]}"})


request_log = RequestLog()
