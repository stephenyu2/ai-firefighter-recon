"""
Offline tests for the decision-model cascade (src/recon_gateway/cascade.py) and the decision-cascade experiment: no
network, no GPU, no dataset images. The decision model and the LLM pass are stubbed; the committed results are checked
against the numbers in the report.
"""
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "decision-cascade"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP))

from recon_gateway import agent, cascade, deciders as D  # noqa: E402
from recon_gateway.kai_decider import QUESTIONS  # noqa: E402
from recon_gateway.sequence import Frame, Sequence  # noqa: E402

SEQS = {s["id"]: s for s in json.loads((ROOT / "experiments/gateway-model-trials/sequences.json").read_text())}


def jl(name):
    return [json.loads(l) for l in (EXP / "results" / name).read_text().splitlines() if l.strip()]


# ---------------- request format ----------------

def test_d3_models_get_data_urls_and_clef_does_not():
    assert D.media("d3", "abc") == "data:image/jpeg;base64,abc"
    assert D.media("d3_mini", "abc") == "data:image/jpeg;base64,abc"
    assert D.media("clef", "abc") == "abc"


def test_every_d3_name_is_wired():
    for name in D.D3_FAMILY:
        assert name in D.URLS and D.SEES_IMAGES[name]
        assert D.MODEL_NAMES[name].startswith("vllm-sr/d3")


def test_cascade_questions_keep_the_options_and_add_guidance():
    q = cascade.questions()
    assert set(q) == {"status", "fire_visible"}
    assert q["status"]["criteria"] == QUESTIONS["status"]["criteria"]
    assert "Observer guidance" in q["status"]["instructions"] and q["fire_visible"]["type"] == "noul"


def test_thermal_facts_leave_out_the_rgb_detector():
    assert cascade.thermal_facts({"thermal_frames": 0, "detector_fire_frames": 0}) == "No thermal camera data is available."
    text = cascade.thermal_facts({"thermal_frames": 8, "thermal_fire_frames": 3, "detector_fire_frames": 0,
                                  "frames": [{"thermal_hot_percent": 0.012}, {"thermal_hot_percent": None}]})
    assert "3 of them" in text and "0.012%" in text and "detector" not in text


def test_extra_evidence_reaches_the_measurement_view():
    state = {"detection": {"units": {}, "frames": [], "detector_fire_frames": 0, "clip_fire_frames": 0, "thermal_frames": 0,
                           "thermal_fire_frames": 0, "disagreements": []},
             "spread": None, "probes": {}, "planner_probes": {}}
    assert "other_evidence" not in agent._measurement_view(state)
    state["extra"] = {"decision_model": {"name": "d3"}}
    assert agent._measurement_view(state)["other_evidence"] == {"decision_model": {"name": "d3"}}


# ---------------- cascade paths (stubbed) ----------------

class Resp:
    def __init__(self, body):
        self.status_code, self._body, self.text = 200, body, json.dumps(body)

    def json(self):
        return self._body


def answers(probs, visible):
    return {"answers": {"status": {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs},
                        "fire_visible": {"type": "noul", "noul": visible}}}


@pytest.fixture
def window(tmp_path, monkeypatch):
    frames = []
    for i in range(8):
        p = tmp_path / f"frame{i}.jpg"
        Image.new("RGB", (64, 48), (40, 90, 40)).save(p)
        frames.append(Frame(id=f"plot2_duringburn_{900 + i:04d}", rgb_path=p, t=5.0 * i))
    for k, v in {"RECON_CLASSIFIER_URL": "http://classifier.test", "RECON_LLM_URL": "http://llm.test/v1",
                 "RECON_D3_URL": "http://d3.test/predict"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(cascade.T, "detect_fire", lambda ep, seq, cache: {"thermal_frames": 0, "frames": []})
    monkeypatch.setattr(cascade.time, "sleep", lambda s: None)
    return Sequence(id="FL-X_plot2_smallfire_rgb", frames=frames)


def stub_d3(monkeypatch, body, sent):
    def post(url, headers, json, timeout):
        sent.append(json)
        return Resp(body)
    monkeypatch.setattr(cascade.requests, "post", post)


def test_confident_answer_is_the_call(window, monkeypatch):
    sent = []
    stub_d3(monkeypatch, answers({"fire": 0.92, "smoke_only": 0.03, "none": 0.03, "unclear": 0.02}, 0.9), sent)
    monkeypatch.setattr(cascade.agent, "run", lambda *a, **k: pytest.fail("a confident answer must not escalate"))
    r = cascade.run(window, decider="d3")
    assert r["path"] == "decision" and r["status"] == "fire"
    body = sent[0]
    assert len(body["images"]) == 8 and all(i.startswith("data:image/jpeg;base64,") for i in body["images"])
    assert "plot2" not in json.dumps({k: body[k] for k in ("state", "questions")})  # dataset names never reach the model


def test_unsure_answer_escalates_with_evidence_and_retries(window, monkeypatch):
    stub_d3(monkeypatch, answers({"fire": 0.6, "smoke_only": 0.1, "none": 0.25, "unclear": 0.05}, 0.65), [])
    calls = []

    def fake_run(seq, **kw):
        calls.append(kw)
        if len(calls) == 1:
            raise RuntimeError("writeup not parsable")
        return {"brief": {"fire_smoke": {"status": "fire"}}, "llm_decision": {"status": "indeterminate"},
                "measured_decision": {"status": "fire"}, "policy_corrections": []}
    monkeypatch.setattr(cascade.agent, "run", fake_run)
    r = cascade.run(window, decider="d3")
    assert r["path"] == "escalated" and r["status"] == "fire" and r["measured_status"] == "fire"
    assert len(r["failed_attempts"]) == 1 and len(calls) == 2
    kw = calls[-1]
    assert kw["probe_all"] is True and kw["guidance"] == "decide" and kw["llm_model"] == "qwen38-27b-bf16"
    assert kw["extra_evidence"]["decision_model"]["name"] == "d3"


def test_unclear_escalates_even_when_confident(window, monkeypatch):
    stub_d3(monkeypatch, answers({"fire": 0.02, "smoke_only": 0.02, "none": 0.01, "unclear": 0.95}, 0.1), [])
    monkeypatch.setattr(cascade.agent, "run", lambda seq, **kw: {
        "brief": {"fire_smoke": {"status": "no fire or smoke"}}, "llm_decision": {"status": "no fire or smoke"},
        "measured_decision": {"status": "no fire signal"}, "policy_corrections": []})
    assert cascade.run(window, decider="d3")["path"] == "escalated"


def test_every_retry_failing_gives_indeterminate(window, monkeypatch):
    stub_d3(monkeypatch, answers({"fire": 0.5, "smoke_only": 0.1, "none": 0.3, "unclear": 0.1}, 0.4), [])

    def boom(seq, **kw):
        raise RuntimeError("upstream 500")
    monkeypatch.setattr(cascade.agent, "run", boom)
    r = cascade.run(window, decider="d3")
    assert r["status"] == "indeterminate" and len(r["failed_attempts"]) == 1 + cascade.RETRIES


# ---------------- committed results ----------------

def right(status, truth):
    return status != "indeterminate" and (status == "fire") == (truth == "fire")


def raised(row):
    return "fire" if row["path"] == "escalated" and row.get("measured_status") == "fire" else row["status"]


def test_fullburn_windows_match_the_report():
    import fullburn_cascade as F
    wins = list(F.windows())
    truths = [t for _, _, _, t, _ in wins]
    assert len(wins) == 170 and sum(len(w) for *_, w in wins) == 1348
    assert truths.count("fire") == 101 and truths.count("no_fire") == 22 and truths.count("unscored") == 47


def test_committed_cascade_results_match_the_report():
    rows = [r for r in jl("cascade.jsonl") if r["ok"] and r["decider_name"] == "d3"]
    assert len(rows) == 20
    assert sum(right(raised(r), SEQS[r["seq"]]["truth"]) for r in rows) == 17
    assert not any(raised(r) == "fire" for r in rows if SEQS[r["seq"]]["truth"] == "no_fire")
    burn = [r for r in jl("fullburn.jsonl") if r["ok"]]
    assert all(raised(r) == "fire" for r in burn if r["truth"] == "fire")
    assert all(raised(r) != "fire" for r in burn if r["truth"] == "no_fire")
    assert sum(r["path"] == "escalated" for r in burn) == 38


def test_committed_frame_results_use_relative_paths():
    import csv
    files = sorted((EXP / "results").glob("frames_v2b_*.csv"))
    assert len(files) == 20
    for f in files:
        with f.open() as fh:
            assert all(not row["path"].startswith("/") for row in csv.DictReader(fh)), f.name
