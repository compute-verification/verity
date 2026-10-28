"""Adversarial cases, turned into regressions of what the end-to-end report (`e2e.py`) marks OK on the advice
profile. There the I/O tie is the only check of the served content (N2 gives no verdict, and the rehearsal's own check
reads the certified I/O, not the committed tokens), so the tie is a required gate (`chain.tie_gate`):

1. Positive control: the tie ran and refused the honest run, so the report is NOT OK.
2. An honest tie that raises gives the honest run no verdict, which fails the report; token's refusal by the tie, which
   did run on it, stays token's verdict rather than falling back to pending.
3. One ``--ids`` map is handed to every run, so it must cover each run's certified requests (`chain.uncovered`): a map
   that is nobody's ids gives the honest run no verdict, and the tie doesn't run on it. So does one that names a
   request with fewer ids than it served (`test_e2e_tie_coverage.py`).

Beside them: without `io_tie` in the tree or without the ids the report is NOT OK and says why, never listing the tie as
not run; and a part the tie must establish beside its verdict (`chain.TIE_REQUIRED`, the hook for the compact
finish) fails the report the same way.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("network_e2e", HERE / "e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e2e = _load()
chain = e2e.chain
NG = chain.NG

OWN = {"late": {"audit": "status:coverage-never-established"}, "extra": {"io": "longer than max_tokens"},
       "drop": {"io": "short of max_tokens without EOS"}}
TOKEN_BY = "the I/O tie (IoTieWindow.content at link-windows [1], the link-windows its changed ids fall in)"


def _io(users=("q0", "q1")):
    return [{"user": users[0], "prompt": [1, 2, 3, 4], "max_tokens": 2, "output": [5, 6]},
            {"user": users[1], "prompt": [1, 2, 3], "max_tokens": 4, "output": [9]}]


def _args(src, out, **kw):
    return SimpleNamespace(**{"sources": [str(src)], "upstream": None, "vllm": None, "out": str(out), "work": None,
                             "bin": None, "lean": None, "delta": 2.0 ** -20, "no_prove": True, "proofs": None,
                             "advice": None, "serving_config": None, "prove_path": None, "ids": None,
                             "no_negatives": False, **kw})


def _varied(d, virtual, verdicts, found=None):
    """A varied rehearsal's runs on the advice profile, each with the certified I/O its check writes (``io.json``):
    every strategy sends the same users."""
    found = OWN | (found or {})
    for s, v in verdicts.items():
        (d / s).mkdir(parents=True)
        (d / s / "certified.jsonl").write_text("")
        (d / s / "server.json").write_text(json.dumps({"requests": 2}))
        shutil.copy(virtual / "honest" / "calibration.json", d / s / "calibration.json")
        (d / s / "plan.json").write_text(json.dumps({"windows": 3, "occupancy": {"kind": "advice"}, "live_model": "m",
                                                     "varied": {"requests": 2, "max_tokens": 4}}))
        f = found.get(s, {}) if v == "refused" else {}
        (d / s / "check.json").write_text(json.dumps({
            "verdict": v, "network_verdict": {"accepted": None}, "audit_accepted": "audit" not in f,
            "salt_accepted": True, "certified_accepted": True,
            "rows": [{"window": 0, "audit": {"egress": {"raw": f.get("audit"), "committed": f.get("audit")}}}],
            "io": {"accepted": "io" not in f, "requests": 2, "tokens": 3, "finish": {"eos": 1, "length": 1},
                   "violations": [[1, f["io"]]] if "io" in f else []}}))
        (d / s / "io.json").write_text(json.dumps({"windows": 3, "requests": _io()}))
    return d


def _ties(**edits):
    """`negatives.tie_run`'s row on each run: honest accepted, token refused by content at link-window 1, drop by content
    for the output it left out, late and extra accepted."""
    rows = {"honest": {}, "late": {}, "extra": {}, "token": {"failing": {1: ["content"]}},
            "drop": {"failing": {-1: ["content"]}, "unplaced": 1}}
    out = {}
    for name, x in rows.items():
        x = x | edits.get(name, {})
        codes = sorted({f"IoTieWindow.{p}" for parts in x.get("failing", {}).values() for p in parts})
        out[name] = {"verdict": "refused" if codes else "accepted", "codes": codes, "failing": {}, "unplaced": 0,
                     "held": {"egress": 3, "ingress": 3}, "missing": {}, "outputs": 3, "placed": 3, "finish": True} | x
    return out


@pytest.fixture(scope="module")
def virtual(tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs")
    e2e.n2_zk.virtual(runs, requests=8, tokens=4, strategies=("honest", "late", "extra"))
    return runs


@pytest.fixture
def runs(virtual, tmp_path):
    return _varied(tmp_path / "varied", virtual, {"honest": "accepted", "late": "refused", "extra": "refused",
                                                  "drop": "refused", "token": "accepted"})


@pytest.fixture
def tied(monkeypatch):
    """`io_tie` in the tree, `tie_run` stubbed to ``_ties``' rows and token's changed ids in link-window 1."""
    monkeypatch.setattr(NG, "io_tie_module", lambda: object())
    monkeypatch.setattr(chain, "changed_windows", lambda run: [1] if run.name == "token" else None)

    def stub(rows):
        monkeypatch.setattr(NG, "tie_run", lambda run, ids, aborted=(): rows(run, ids))
    return stub


def _ids(tmp_path, pairs):
    p = tmp_path / "ids.jsonl"
    p.write_text("".join(json.dumps({"user": u, "output_ids": x}) + "\n" for u, x in pairs))
    return p


def test_the_tie_refusing_the_honest_run_fails_the_report(runs, tmp_path, tied):
    """Positive control: the tie ran and refused the honest run, so the report is NOT OK."""
    tied(lambda run, ids: _ties(honest={"failing": {2: ["port"]}})[run.name])
    ids = _ids(tmp_path, (("q0", [5, 6]), ("q1", [9])))
    r = e2e.run(_args(runs, tmp_path / "out", ids=str(ids)), [], clock=lambda: 0.0)
    assert r["ok"] is False and r["why_not"] == ["the I/O tie didn't accept the honest run"]
    assert r["negatives"]["token"]["refused"] is True


def test_an_honest_tie_that_raises_fails_the_report_and_token_keeps_its_refusal(runs, tmp_path, tied):
    """`io_tie` present and --ids given, but honest's tie raises: the honest run has no verdict, so the report is NOT OK
    and says why. Token's tie ran and refused it at the link-window it changed: that stays its verdict, not pending."""
    ties = _ties()

    def tie_run(run, ids):
        if run.name == "honest":
            raise ValueError("honest's sampled ids are not in the file")
        return ties[run.name]

    tied(tie_run)
    ids = _ids(tmp_path, (("q0", [5, 6]), ("q1", [9])))
    r = e2e.run(_args(runs, tmp_path / "out", ids=str(ids)), [], clock=lambda: 0.0)
    assert r["ok"] is False and r["why_not"] == [
        "the I/O tie gave no verdict on the honest run: ValueError(\"honest's sampled ids are not in the file\")"]
    tie = r["chain"]["tie"]
    assert tie["ran"] is True and tie["runs"]["honest"]["verdict"] == NG.NOT_APPLICABLE
    assert tie["runs"]["token"]["verdict"] == "refused" and tie["runs"]["token"]["codes"] == ["IoTieWindow.content"]
    assert r["negatives"]["token"]["refused"] is True and r["negatives"]["token"]["by"] == [TOKEN_BY]
    assert r["pending"] == {} and e2e.TIE_STAGE in r["ran"] and e2e.TIE_STAGE not in r["not_run"]
    md = (tmp_path / "out" / "report.md").read_text()
    assert "Not OK: the I/O tie gave no verdict on the honest run" in md and f"- **token**: refused by {TOKEN_BY}" in md


def test_ids_that_dont_cover_the_honest_runs_certified_requests_fail_the_report(runs, tmp_path, tied):
    """--ids hands one map to every run, so it must cover each run's certified requests (its io.json's, an id for each
    served output): a map that is nobody's ids, misses one request or names one with no ids gives the run no verdict,
    and the tie never runs on it."""
    seen = []
    tied(lambda run, ids: seen.append(run.name) or _ties()[run.name])
    r = e2e.run(_args(runs, tmp_path / "out", ids=str(_ids(tmp_path, (("nobody", [999]),)))), [], clock=lambda: 0.0)
    assert r["ok"] is False and r["why_not"] == [
        "the I/O tie gave no verdict on the honest run: no sampled ids for 2 of its 2 certified requests",
        "token not refused"]
    assert seen == [] and r["chain"]["tie"]["runs"]["honest"]["ok"] is None
    assert "the I/O tie, which alone refuses it, gave it no verdict (no sampled ids for 2 of its 2 certified requests)" \
        in (tmp_path / "out" / "report.md").read_text()
    r = e2e.run(_args(runs, tmp_path / "out2", ids=str(_ids(tmp_path, (("q0", [5, 6]),)))), [], clock=lambda: 0.0)
    assert r["why_not"][0] == "the I/O tie gave no verdict on the honest run: no sampled ids for 1 of its 2 certified " \
                              "requests" and seen == []
    r = e2e.run(_args(runs, tmp_path / "out5", ids=str(_ids(tmp_path, (("q0", []), ("q1", [9]))))), [],
                clock=lambda: 0.0)
    assert r["ok"] is False and r["why_not"] == [
        "the I/O tie gave no verdict on the honest run: fewer sampled ids than served outputs for 1 of its 2 certified "
        "requests (0 sampled, 2 served)", "token not refused"] and seen == []
    (runs / "honest" / "io.json").unlink()
    r = e2e.run(_args(runs, tmp_path / "out3", ids=str(_ids(tmp_path, (("q0", [5, 6]), ("q1", [9]))))), [],
                clock=lambda: 0.0)
    assert r["why_not"] == ["the I/O tie gave no verdict on the honest run: no certified I/O (io.json) to hold the "
                            "sampled ids to"] and "honest" not in seen and r["negatives"]["token"]["refused"] is True


def test_without_io_tie_or_the_ids_the_report_isnt_ok_and_says_why(runs, tmp_path, monkeypatch):
    """The tie that didn't run fails the report with its reason, and token, which only it refuses, isn't refused; the
    tie is listed neither as run nor as not run."""
    monkeypatch.setattr(NG, "io_tie_module", lambda: None)
    ids = _ids(tmp_path, (("q0", [5, 6]), ("q1", [9])))
    r = e2e.run(_args(runs, tmp_path / "out", ids=str(ids)), [], clock=lambda: 0.0)
    assert r["ok"] is False and r["why_not"] == [
        f"the I/O tie didn't run, and the advice profile requires it: {NG.TIE_MISSING['io-tie']}", "token not refused"]
    assert e2e.TIE_STAGE not in r["ran"] + r["not_run"] and r["pending"] == {}
    md = (tmp_path / "out" / "report.md").read_text()
    assert "I/O tie" not in md.splitlines()[0] and "## I/O tie\n\nDidn't run: " in md
    assert f"- **token**: NOT refused by nothing (not proved); the I/O tie, which alone refuses it, didn't run " \
           f"({NG.TIE_MISSING['io-tie']})" in md
    monkeypatch.setattr(NG, "io_tie_module", lambda: object())
    r = e2e.run(_args(runs, tmp_path / "out2"), [], clock=lambda: 0.0)
    assert r["why_not"] == ["the I/O tie didn't run, and the advice profile requires it: no sampled ids (neither "
                            "--endpoint-run nor --ids given)", "token not refused"]


def test_a_part_the_tie_must_establish_beside_its_verdict_fails_the_report(runs, tmp_path, tied, monkeypatch):
    """`chain.TIE_REQUIRED`, the hook for the compact finish: a key of the tie's row that must be true on the
    honest run, else the report is NOT OK with its reason, as when the tie refuses the honest run."""
    monkeypatch.setattr(chain, "TIE_REQUIRED", {"finish": "the compact finish didn't hold on the honest run"})
    ids = _ids(tmp_path, (("q0", [5, 6]), ("q1", [9])))
    tied(lambda run, ids: _ties(honest={"finish": False})[run.name])
    r = e2e.run(_args(runs, tmp_path / "out", ids=str(ids)), [], clock=lambda: 0.0)
    assert r["ok"] is False and r["why_not"] == ["the compact finish didn't hold on the honest run"]
    assert r["chain"]["tie"]["runs"]["honest"]["finish"] is False
    tied(lambda run, ids: _ties(honest={"finish": True})[run.name])
    r = e2e.run(_args(runs, tmp_path / "out2", ids=str(ids)), [], clock=lambda: 0.0)
    assert r["ok"] is True, r["why_not"]
