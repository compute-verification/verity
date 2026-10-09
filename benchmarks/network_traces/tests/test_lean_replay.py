"""`lean_replay.py` on a tiny synthetic trace: the Python audit against itself agrees on every link-window, raw,
committed and certified, matches the verdicts recorded for it, and rejects every injected control identically; a broken
verifier is caught, as failed controls or as differing verdicts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from verity.core.protocols.certifier import audit, calibration

HERE = Path(__file__).resolve().parents[1]
T = 60
PYTHON = "verity.core.protocols.certifier.audit:Audit"


def _load(name: str):
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(f"network_traces_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lr = _load("lean_replay")


def trace() -> dict:
    frames, arrival_ms = [], []
    for w in range(3):
        for j, (offset, latency) in enumerate(((0, 1), (10, 3), (20, 2))):
            anchor = w * T + offset
            frames += [{"session": f"s{w}.{j}", "index": i, "anchor": anchor, "ready": anchor + latency + i}
                       for i in range(4)]
            arrival_ms.append(anchor * 100 + 30.0)
    frames.sort(key=lambda f: (f["ready"], f["anchor"], f["session"], f["index"]))
    return {"bucket_ms": 100, "window_buckets": T, "frame_bytes": 4096, "queue_capacity": 1 << 16,
            "configurations": [{"name": "synthetic", "labels": {"run": "r0", "windows": [0, 3]},
                                "egress": [{"link": "responses", "frames": frames}],
                                "ingress": [{"link": "requests", "arrival_ms": arrival_ms}]}]}


class AcceptsDuplicates(audit.Audit):
    """A broken verifier: it keeps one record per link-window, so a duplicate passes."""

    def check_link(self, link, records, *rest, **kw):
        return super().check_link(link, list({(r.link, r.window): r for r in records}.values()), *rest, **kw)

    def check_ingress_link(self, link, records, *rest, **kw):
        return super().check_ingress_link(link, list({(r.link, r.window): r for r in records}.values()), *rest, **kw)


class RejectsWindowZero(audit.Audit):
    """A broken verifier: egress window 0 is always an `advice` violation."""

    def check_link(self, *args, **kw):
        out = super().check_link(*args, **kw)
        out[:1] = [audit.Verdict(v.link, v.window, "advice") for v in out[:1]]
        return out


def job(tmp_path: Path, other, recorded=None) -> dict:
    doc = trace()
    (tmp_path / "trace.json").write_text(json.dumps(doc))
    report = calibration.calibrate(doc)
    return lr.trace_job((tmp_path / "trace.json", report, "hindsight", False, recorded, True, other))


def test_the_python_audit_against_itself_agrees_and_every_control_is_rejected(tmp_path: Path) -> None:
    first = job(tmp_path, PYTHON)
    assert "error" not in first, first.get("error")
    assert first["link_windows"] == first["same"] == 6 and not first["diffs"]
    names = [c["control"] for c in first["controls"]]
    assert names == ["drop", "duplicate", "raw-slot", "status", "committed-digest-byte", "certificate-other-window",
                     "sync-outside-set", "sync-not-units", "ingress-drop", "ingress-duplicate", "ingress-status"]
    assert all(c["passed"] and c["expected_kind"] for c in first["controls"]), first["controls"]
    # the recorded verdicts: equal ones match, and one recorded rejection the replay doesn't give is a diff
    rows = [{"config": "synthetic", "run": "r0", "direction": d, "link": link, "window": w, "raw": None,
             "committed": None} for d, link in (("egress", "responses"), ("ingress", "requests")) for w in range(3)]
    rows[1]["committed"] = "mismatch"
    again = job(tmp_path, PYTHON, rows)
    assert again["recorded_compared"] == 6 and again["recorded_same_python"] == again["recorded_same_lean"] == 5
    assert [(d["window"], d["recorded"]) for d in again["diffs"]] == [(1, {"raw": None, "committed": "mismatch"})]
    s = lr.summarize([first, again])
    assert s["runs"] == 1 and s["runs_same_verdict"] == 0 and s["K"] == {"natural": 0, "injected": 22, "failed": 0}


def test_a_broken_verifier_is_caught(tmp_path: Path) -> None:
    r = job(tmp_path, AcceptsDuplicates)
    assert not r["diffs"]
    assert {c["control"] for c in r["controls"] if not c["passed"]} == {"duplicate", "ingress-duplicate"}
    assert lr.summarize([r])["injected_failed"] == {"duplicate": 1, "ingress-duplicate": 1}
    r = job(tmp_path, RejectsWindowZero)
    assert [(d["direction"], d["window"], d["python"]["raw"], d["lean"]["raw"]) for d in r["diffs"]] == \
        [("egress", 0, None, "advice")]
    assert r["violations"]["lean"]["certified"] == r["violations"]["lean"]["link_windows"] == {"advice": 1}
    s = lr.summarize([r])
    assert s["runs_same_verdict"] == 0 and s["natural_controls"] == {"python": {}, "lean": {"advice": 1}}
