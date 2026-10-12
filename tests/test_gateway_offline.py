"""
Offline tests for the recon gateway (src/recon_gateway): no network, no GPU, no dataset. They cover the parts that
decide results without a model in the loop: answer-letter scoring, the measurement rule and policy checks, tool-call
validation, id aliasing, the brief schema, client auth and retries, and the committed experiment results.
"""
import csv
import json
import math
import re
import sys
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recon_gateway import agent, tools as T  # noqa: E402
from recon_gateway.brief import BRIEF_SCHEMA, enforce_policy, evidence_status, to_markdown  # noqa: E402
from recon_gateway.prompts import WATCH_FOR  # noqa: E402
from recon_gateway.sequence import Frame, Sequence, alias  # noqa: E402

TRIALS = ROOT / "experiments" / "gateway-model-trials"


def lp(token, p):
    return {"token": token, "logprob": math.log(p)}


# ---------------- answer-letter scoring ----------------

def test_whitespace_token_is_not_an_answer():
    r = T.score_options([lp(" ", 0.6), lp("B", 0.3), lp("A", 0.1)], ["yes", "no"])
    assert r["coverage"] == pytest.approx(0.4)
    assert r["answer"] is None  # below MIN_COVERAGE, so unreliable
    assert r["probabilities"]["no"] == pytest.approx(0.75)


def test_letter_variants_count_and_order_is_kept():
    r = T.score_options([lp("A", 0.5), lp(" A", 0.1), lp("A.", 0.1), lp("B", 0.2), lp("Answer", 0.05)], ["yes", "no"])
    assert list(r["probabilities"]) == ["yes", "no"]
    assert r["answer"] == "yes"
    assert r["coverage"] == pytest.approx(0.9)
    assert r["confidence"] == pytest.approx(0.7 / 0.9, abs=1e-6)


def test_tie_goes_to_earlier_option():
    r = T.score_options([lp("A", 0.4), lp("B", 0.4)], ["yes", "no"])
    assert r["answer"] == "yes" and r["margin"] == 0


def test_decide_scores_from_logprobs_and_sends_guidance():
    class FakeEP:
        def __init__(self):
            self.calls = []

        def chat(self, messages, **kw):
            self.calls.append((messages, kw))
            return {"choices": [{"logprobs": {"content": [{"top_logprobs": [lp("A", 0.9), lp("B", 0.1)]}]}}]}

    class FakeFrame:
        def rgb_b64(self, max_side=None):
            return "aGVsbG8="

    ep = FakeEP()
    out = T.decide(ep, FakeFrame(), ["flames_visible", "not_a_question"], system="watch for fire")
    assert list(out) == ["flames_visible"]
    assert out["flames_visible"]["answer"] == "yes"
    assert out["flames_visible"]["raw_top_logprobs"][0][0] == "A"
    messages, kw = ep.calls[0]
    assert messages[0] == {"role": "system", "content": "watch for fire"}
    assert kw["max_tokens"] == 1 and kw["logprobs"] is True and kw["top_logprobs"] == 20


# ---------------- measurement rule ----------------

def det(n=8, thermal=0, thermal_fire=0, detector=0, hot=None):
    frames = [{"id": f"f{i + 1:02d}", "thermal_hot_percent": (hot[i] if hot else None)} for i in range(n)]
    return {"frames": frames, "thermal_frames": thermal, "thermal_fire_frames": thermal_fire, "detector_fire_frames": detector}


def probes(*yes_probs):
    return {f"f{i + 1:02d}": {"flames_visible": {"answer": "yes" if p >= 0.5 else "no", "probabilities": {"yes": p, "no": 1 - p}}}
            for i, p in enumerate(yes_probs)}


def test_thermal_decides_when_present():
    assert evidence_status(det(thermal=8, thermal_fire=8, detector=8))[:2] == ("fire", "high")
    assert evidence_status(det(thermal=8, thermal_fire=0, detector=0))[:2] == ("no fire signal", "high")
    assert evidence_status(det(thermal=8, thermal_fire=0, detector=3))[:2] == ("indeterminate", "low")


def test_rgb_only_needs_two_confident_flame_probes():
    assert evidence_status(det(), probes(0.9, 0.85, 0.1))[:2] == ("fire", "low")
    assert evidence_status(det(), probes(0.9, 0.1, 0.1))[0] == "no fire signal"
    unreliable = probes(0.95, 0.95)
    for q in unreliable.values():
        q["flames_visible"]["answer"] = None
    assert evidence_status(det(), unreliable)[0] == "no fire signal"


# ---------------- policy ----------------

def brief(status="no fire or smoke", direction="toward top of frame", trend="increasing", summary="Summary."):
    return {"fire_smoke": {"status": status, "confidence": "high", "evidence": "x"}, "extent": {"description": "small"},
            "location_in_frame": "center", "spread_direction": {"direction": direction, "basis": "x"},
            "trend": {"status": trend, "basis": "x"}, "overall_confidence": "high", "limitations": [],
            "observations": [], "inferences": [], "summary": summary}


def test_thermal_rule_overrides_llm():
    ev = {"detection": det(thermal=8, thermal_fire=8, detector=0, hot=[1.0, 3.0] * 4), "spread": {}, "probes": {}}
    b, fixes = enforce_policy(brief(), ev)
    assert b["fire_smoke"]["status"] == "fire"
    assert any(f.startswith("fire status") for f in fixes)
    assert b["extent"]["hot_area_percent_of_frame"] == pytest.approx(2.0)  # computed, never asked of the model


def test_weak_rgb_rule_is_a_note_not_an_override():
    ev = {"detection": det(), "spread": {}, "probes": probes(0.9, 0.9, 0.1)}
    b, fixes = enforce_policy(brief(), ev)
    assert b["fire_smoke"]["status"] == "no fire or smoke"
    assert any(f.startswith("note:") for f in fixes)
    assert any("measurement rule suggests 'fire'" in x for x in b["limitations"])


def test_unsupported_direction_trend_and_banned_terms():
    ev = {"detection": det(thermal=8, thermal_fire=8, detector=8), "spread": {"direction_supported": False, "trend_supported": False},
          "probes": {}}
    b, fixes = enforce_policy(brief(status="fire", summary="About 5 acres burning; send crews."), ev)
    assert b["spread_direction"]["direction"] == "indeterminate"
    assert b["trend"]["status"] == "indeterminate"
    assert "acres" not in b["summary"] and "[out of scope]" in b["summary"]
    assert b["overall_confidence"] == "medium"


def test_markdown_has_core_rows():
    b, _ = enforce_policy(brief(status="fire"), {"detection": det(thermal=8, thermal_fire=8, detector=8, hot=[0.5] * 8), "spread": {}, "probes": {}})
    md = to_markdown(b, {"sequence_id": "demo", "n_frames": 8, "span_s": 35.0})
    assert "| Fire / smoke | fire" in md and "% of frame above 80 C" in md and "a person makes every call" in md


# ---------------- schema, prompts, tool calls ----------------

def strict_objects(schema, path="brief"):
    if schema.get("type") == "object":
        yield path, schema
        for k, v in schema.get("properties", {}).items():
            yield from strict_objects(v, f"{path}.{k}")
    if schema.get("type") == "array":
        yield from strict_objects(schema["items"], f"{path}[]")


def test_brief_schema_is_strict():
    for path, obj in strict_objects(BRIEF_SCHEMA):
        assert obj.get("additionalProperties") is False, path
        assert set(obj.get("required", [])) == set(obj["properties"]), path


def test_observer_guidance_has_no_burn_specific_values():
    assert not re.search(r"\d", WATCH_FOR)
    assert not re.search(r"florida|oregon|sycan|hanna|marsh|flame 3", WATCH_FOR, re.I)


def test_tool_specs_pin_frame_ids():
    spec = agent.tool_specs(["f01", "f02"])[0]["function"]["parameters"]
    assert spec["properties"]["frame_id"]["enum"] == ["f01", "f02"]
    assert spec["additionalProperties"] is False


@pytest.mark.parametrize("name,args,expect", [
    ("decide", {"frame_id": "f09", "questions": ["smoke_visible"]}, "frame_id must be one of"),
    ("decide", {"frame_id": "f01", "questions": ["smoke_visible", "smoke_visible"]}, "distinct"),
    ("decide", {"frame_id": "f01", "questions": ["is_it_fire"]}, "distinct values"),
    ("decide", {"frame_id": "f01", "questions": ["smoke_visible"], "why": "x"}, "unknown argument"),
    ("detect", {}, "unknown tool"),
    ("decide", {"frame_id": "f01", "questions": ["smoke_visible"]}, None),
    ("finish", {"stray": 1}, None),
])
def test_validate_tool_calls(name, args, expect):
    err = agent._validate(name, args, ["f01", "f02"], {"planner_probes": {}})
    assert (err is None) if expect is None else (expect in err)


def test_validate_budget_and_repeats():
    full = {"planner_probes": {"f01": {}, "f02": {}, "f03": {}}}
    assert "limit reached" in agent._validate("decide", {"frame_id": "f04", "questions": ["smoke_visible"]}, ["f01", "f02", "f03", "f04"], full)
    assert "already probed" in agent._validate("decide", {"frame_id": "f01", "questions": ["smoke_visible"]}, ["f01", "f04"], full)


def test_guidance_flags(monkeypatch):
    monkeypatch.delenv("RECON_GUIDANCE", raising=False)
    assert not any(agent.guidance_flags().values())
    assert all(agent.guidance_flags("all").values())
    assert agent.guidance_flags("decide,writeup") == {"planner": False, "decide": True, "writeup": True}


def test_alias_hides_dataset_ids_and_sorts_by_time():
    seq = Sequence(id="FL-D_plot1_preburn_rgb", frames=[Frame(id="plot1_preburn_0041", rgb_path=Path("b.jpg"), t=5.0),
                                                       Frame(id="plot1_preburn_0040", rgb_path=Path("a.jpg"), t=0.0)])
    a, id_map = alias(seq)
    assert a.id == "S" and [f.id for f in a.frames] == ["f01", "f02"]
    assert id_map == {"f01": "plot1_preburn_0040", "f02": "plot1_preburn_0041"}


# ---------------- client auth and retries ----------------

@pytest.fixture
def urls(monkeypatch):
    monkeypatch.setenv("RECON_CLASSIFIER_URL", "http://classifier.test/predict")
    monkeypatch.setenv("RECON_LLM_URL", "http://llm.test/v1/")
    for k in ("RECON_GATEWAY_KEY", "RECON_API_KEY", "RECON_AUTH_SCHEME", "RECON_RUN_TAG"):
        monkeypatch.delenv(k, raising=False)


def test_endpoint_auth(urls, monkeypatch):
    assert "Authorization" not in T.Endpoints().h
    monkeypatch.setenv("RECON_API_KEY", "k1")
    monkeypatch.setenv("RECON_AUTH_SCHEME", "Token")
    assert T.Endpoints().h["Authorization"] == "Token k1"
    monkeypatch.setenv("RECON_GATEWAY_KEY", "gw")
    ep = T.Endpoints(run_tag="v2/test")
    assert ep.h["Authorization"] == "Bearer gw" and ep.h["x-recon-run"] == "v2/test"
    assert ep.llm_url == "http://llm.test/v1"


def test_endpoint_needs_urls(monkeypatch):
    monkeypatch.delenv("RECON_LLM_URL", raising=False)
    with pytest.raises(SystemExit):
        T.Endpoints()


class Resp:
    def __init__(self, code):
        self.status_code = code

    def json(self):
        return {"ok": self.status_code}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


def test_chat_retries_429_then_succeeds(urls, monkeypatch):
    codes, bodies = iter([429, 503, 200]), []
    monkeypatch.setattr(T.requests, "post", lambda url, headers, json, timeout: bodies.append(json) or Resp(next(codes)))
    monkeypatch.setattr(T.time, "sleep", lambda s: None)
    assert T.Endpoints().chat([{"role": "user", "content": "hi"}]) == {"ok": 200}
    assert len(bodies) == 3 and all(b == bodies[0] for b in bodies)  # identical body on every retry
    assert bodies[0]["temperature"] == 0 and bodies[0]["seed"] == T.SEED


def test_chat_fails_fast_on_bad_request(urls, monkeypatch):
    calls = []
    monkeypatch.setattr(T.requests, "post", lambda *a, **k: calls.append(1) or Resp(400))
    monkeypatch.setattr(T.time, "sleep", lambda s: None)
    with pytest.raises(requests.HTTPError):
        T.Endpoints().chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 1


# ---------------- committed results ----------------

def test_sequences_use_relative_paths():
    seqs = json.loads((TRIALS / "sequences.json").read_text())
    assert len(seqs) == 20
    for s in seqs:
        for k in ("rgb_dir", "tiff_dir"):
            assert not str(s.get(k) or "").startswith("/"), s["id"]


def test_frame_results_use_relative_paths():
    files = sorted((TRIALS / "results").glob("frames_v2b_*.csv"))
    assert files
    for f in files:
        with f.open() as fh:
            assert all(not row["path"].startswith("/") for row in csv.DictReader(fh)), f.name


def test_packed_runs_load():
    sys.path.insert(0, str(TRIALS))
    from runs_store import load_run
    run = load_run("v2", "qwen38-27b-bf16#r1", "FL-A_plot1_burn_thermal")
    assert run["raw_brief"]["fire_smoke"]["status"] in ("fire", "smoke only", "no fire or smoke", "indeterminate")
    assert "transcript" not in (run.get("planner") or {})
