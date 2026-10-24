"""REDTEAM phase 12, target 2 (#3748): is the e2e report OK only if the I/O tie accepts the honest run?

On the advice profile the I/O tie is the only check of the served content (N2 gives no verdict, and the rehearsal's own
check reads the certified I/O, not the committed tokens). `e2e.report` computes ``tie_ok = not ties or
(ties[HONEST] or {})["ok"] is True`` where ``ties = tie["runs"] if tie["ran"] else {}``, and ``tie["ran"]`` is true only
when the honest run has a verdict (`chain.tie`). Three cases:

- **The tie is a real gate only when the honest run has a verdict** (positive control): when the tie ran and refused the
  honest run, the report is NOT OK (``why_not`` says so).
- **A missing honest verdict silently disables the gate.** With IO-TIE in the tree and ``--ids`` given, if honest's tie
  raises (its sampled ids or records aren't there) the whole tie is marked not-run — even though token's tie ran and
  refused it. token's refusal is discarded (token falls to ``pending``, not a refused negative), ``tie_ok`` is True, and
  the report is OK. The honest run's content is never tied, yet the report reads OK. This is the lead's "no io_tie in the
  tree, missing ids or records" edge, and it is sharper than the PR's own out4 (no IO-TIE at all): here the tie ran and
  refused token, and the report still discards that and reads OK because honest produced no verdict.
- **honest's acceptance is over a caller-supplied ids map not bound to the run.** With ``--ids`` one map is handed to
  every run; nothing ties it to the honest run's committed Commit port. A map that is nobody's ids still "accepts" honest
  and the report is OK. "honest's tie ran on another run's ids" is the norm on this path; the acceptance rests on the
  stand-in map, not a binding (the stated stand-in, IoTieWindow over vllm's Commit, isn't here).

One-line fix: on the advice profile make the tie a required gate — a report is OK only when the tie ran and accepted the
honest run, so a missing honest verdict (or a tie that didn't run) is a failure in ``why_not``, not a silent downgrade of
token to pending. Owner: NET-E2E.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
T = 4


def _load():
    spec = importlib.util.spec_from_file_location("network_e2e", HERE / "e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e2e = _load()
chain = e2e.chain
I1 = chain.I1
NG = chain.NG

OWN = {"late": {"audit": "status:coverage-never-established"}, "extra": {"io": "longer than max_tokens"},
       "drop": {"io": "short of max_tokens without EOS"}}


def _io(users=("q0", "q1")):
    return [{"user": users[0], "prompt": [1, 2, 3, 4], "max_tokens": 2, "output": [5, 6]},
            {"user": users[1], "prompt": [1, 2, 3], "max_tokens": 4, "output": [9]}]


def _args(src, out, **kw):
    return SimpleNamespace(**{"sources": [str(src)], "upstream": None, "vllm": None, "out": str(out), "work": None,
                             "bin": None, "lean": None, "delta": 2.0 ** -20, "no_prove": True, "proofs": None,
                             "advice": None, "serving_config": None, "prove_path": None, "ids": None,
                             "no_negatives": False, **kw})


def _varied(d, virtual, verdicts, found=None):
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
    (d / "honest" / "io.json").write_text(json.dumps({"windows": 3, "requests": _io()}))
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
                     "held": {"egress": 3, "ingress": 3}, "missing": {}, "outputs": 3, "placed": 3} | x
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


def _ids(tmp_path, pairs):
    p = tmp_path / "ids.jsonl"
    p.write_text("".join(json.dumps({"user": u, "output_ids": x}) + "\n" for u, x in pairs))
    return p


def test_the_tie_is_a_gate_only_when_the_honest_run_has_a_verdict(runs, tmp_path, monkeypatch):
    """Positive control: the tie ran and refused the honest run, so the report is NOT OK."""
    monkeypatch.setattr(NG, "io_tie_module", lambda: object())
    monkeypatch.setattr(NG, "tie_run", lambda run, ids: _ties(honest={"failing": {2: ["port"]}})[run.name])
    monkeypatch.setattr(chain, "changed_windows", lambda run: [1] if run.name == "token" else None)
    ids = _ids(tmp_path, (("q0", [5]), ("q1", [5])))
    r = e2e.run(_args(runs, tmp_path / "out", ids=str(ids)), [], clock=lambda: 0.0)
    assert r["ok"] is False and r["why_not"] == ["the I/O tie didn't accept the honest run"]


def test_a_missing_honest_verdict_disables_the_gate_and_discards_tokens_refusal(runs, tmp_path, monkeypatch):
    """IO-TIE present and --ids given, but honest's tie raises (its sampled ids/records aren't there). token's tie ran
    and refused, yet the whole tie is marked not-run: token's refusal is discarded (token falls to pending), tie_ok is
    True, and the report is OK — the honest run's content is never tied."""
    ties = _ties()

    def tie_run(run, ids):
        if run.name == "honest":
            raise ValueError("honest's sampled ids are not in the file")
        return ties[run.name]

    monkeypatch.setattr(NG, "io_tie_module", lambda: object())
    monkeypatch.setattr(NG, "tie_run", tie_run)
    monkeypatch.setattr(chain, "changed_windows", lambda run: [1] if run.name == "token" else None)
    ids = _ids(tmp_path, (("q0", [5]), ("q1", [5])))
    r = e2e.run(_args(runs, tmp_path / "out", ids=str(ids)), [], clock=lambda: 0.0)

    assert r["ok"] is True and r["why_not"] == []
    assert r["chain"]["tie"]["ran"] is False and "honest" in r["chain"]["tie"]["why"]
    # token's tie actually refused it, but that verdict is discarded with the whole tie:
    assert r["chain"]["tie"]["runs"]["token"]["verdict"] == "refused"
    assert r["chain"]["tie"]["runs"]["token"]["codes"] == ["IoTieWindow.content"]
    assert "token" not in r["negatives"] and "token" in (r.get("pending") or {})


def test_honest_accepts_over_a_caller_supplied_ids_map_not_bound_to_the_run(runs, tmp_path, monkeypatch):
    """--ids hands one map to every run (honest included); nothing ties it to the honest run's committed Commit port. A
    map that is nobody's ids still 'accepts' honest and the report is OK: honest's acceptance rests on the stand-in map,
    not a binding ('honest's tie ran on another run's ids' is the norm on this path)."""
    seen = []
    monkeypatch.setattr(NG, "io_tie_module", lambda: object())
    monkeypatch.setattr(NG, "tie_run", lambda run, ids: seen.append((run.name, dict(ids))) or _ties()[run.name])
    monkeypatch.setattr(chain, "changed_windows", lambda run: [1] if run.name == "token" else None)
    ids = _ids(tmp_path, (("nobody", [999]),))
    r = e2e.run(_args(runs, tmp_path / "out", ids=str(ids)), [], clock=lambda: 0.0)
    assert r["ok"] is True and r["chain"]["tie"]["runs"]["honest"]["ok"] is True
    # the identical, run-unrelated map reached every run, honest among them
    assert {n for n, _ in seen} == {"honest", "late", "extra", "drop", "token"}
    assert all(m == {"nobody": [999]} for _, m in seen)
