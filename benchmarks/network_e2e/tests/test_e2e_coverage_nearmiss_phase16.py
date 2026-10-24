"""REDTEAM phase 16, target 2 against #3841 (cursor/net-e2e-commit-tie-061a @ac3067386b): the tie gate's coverage
(`chain.uncovered`) holds each certified request's sampled ids to its served outputs (my phase-14 fix), with the
gating in `chain.tie`/`chain.tie_gate`. The near misses the lead named: ids >= served but the wrong ids, duplicate
ids, a request whose served count is 0, and an extra request not in the certified set.

`uncovered` is a COUNT gate: it guarantees every served output has an id row for the content tie to open, and leaves
the id VALUES to that tie (`negatives.tie_run`'s `reference_parts`, which needs the richer io_tie that lives on the
IO-TIE lane, exercised in target 3). So the sharp line to hold here is the boundary:

- a count that covers but with wrong or duplicated values PASSES `uncovered` by design, so the tie *runs* on the honest
  run (unlike a short map, where the tie never runs) and the report's OK then follows the content tie's verdict;
- an extra request not in the certified set is inert (`uncovered` reads only the certified requests, and `tie_run`
  reads only the certified users);
- a request that served nothing needs an empty entry, not an absent one.

The content tie is stubbed exactly as the PR's own `test_e2e_tie_phase12.py` stubs it (the richer io_tie isn't in
#3841's base); every `uncovered` assertion is against the real landed function.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

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

#: two certified requests, q0 served two outputs and q1 one (the io.json the rehearsal writes on the advice profile)
IO = [{"user": "q0", "prompt": [1, 2, 3, 4], "max_tokens": 2, "output": [5, 6]},
      {"user": "q1", "prompt": [1, 2, 3], "max_tokens": 4, "output": [9]}]


def _run(tmp_path, requests=IO, name="honest"):
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "io.json").write_text(json.dumps({"windows": 3, "requests": requests}))
    return d


def _ids_file(tmp_path, pairs, name="ids.jsonl"):
    p = tmp_path / name
    p.write_text("".join(json.dumps({"user": u, "output_ids": x}) + "\n" for u, x in pairs))
    return p


def _row(verdict="accepted", **kw):
    """A `tie_run` row shaped like `negatives.tie_run`'s, standing in for the content tie (the real one needs the
    richer io_tie absent from #3841's base, as the PR's phase-12 test notes)."""
    return {"verdict": verdict, "codes": [], "failing": {}, "unplaced": 0, "held": {"egress": 3, "ingress": 3},
            "missing": {}, "outputs": 3, "placed": 3, **kw}


def _tie(monkeypatch, run, ids_path, stub):
    """Drive the real `chain.tie` (and so the real `uncovered`/`tie_ok`) with IO-TIE present and the content tie
    stubbed; return the honest row and the list of runs the stub was actually called on."""
    seen: list[str] = []
    monkeypatch.setattr(NG, "io_tie_module", lambda: object())

    def tie_run(d, ids):
        seen.append(d.name)
        return stub
    monkeypatch.setattr(NG, "tie_run", tie_run)
    result = chain.tie({"honest": run}, None, ids_path)
    return result["runs"]["honest"], seen


def test_wrong_ids_with_the_right_count_pass_the_count_gate_and_reach_the_content_tie(tmp_path, monkeypatch):
    """ids >= served but the wrong values: `uncovered` passes (it counts, it doesn't read values), so the content tie
    runs on the honest run — unlike a short map, where it never runs — and the report's OK follows the tie's verdict.
    The count gate's job (every served output has a row to open) holds; the values are the tie's to refuse."""
    run = _run(tmp_path)
    assert chain.uncovered(run, {"q0": [7, 8], "q1": [3]}) is None          # 2>=2, 1>=1; every value wrong
    # the content tie (stub) refuses the wrong ids as it would in reality; the tie DID run on honest (coverage passed)
    row, seen = _tie(monkeypatch, run, _ids_file(tmp_path, (("q0", [7, 8]), ("q1", [3]))),
                     _row("refused", codes=["IoTieWindow.content"], failing={1: ["content"]}))
    assert seen == ["honest"] and row["ok"] is False and row["verdict"] == "refused"
    assert chain.tie_gate({"ran": True, "runs": {"honest": row}}) == ["the I/O tie didn't accept the honest run"]
    # a short map, by contrast, never reaches the tie (the phase-14 fix): the tie isn't called at all
    _, seen_short = _tie(monkeypatch, run, _ids_file(tmp_path, (("q0", [5]), ("q1", [9])), "short.jsonl"), _row())
    assert seen_short == []


def test_duplicate_ids_with_the_right_count_pass_the_count_gate(tmp_path, monkeypatch):
    """Duplicated ids keep the count up (2 for two served outputs) so `uncovered` passes; the duplicate is a wrong
    value at the second output, which is the content tie's to refuse, and the tie runs on the honest run."""
    run = _run(tmp_path)
    assert chain.uncovered(run, {"q0": [5, 5], "q1": [9]}) is None
    row, seen = _tie(monkeypatch, run, _ids_file(tmp_path, (("q0", [5, 5]), ("q1", [9]))),
                     _row("refused", codes=["IoTieWindow.content"], failing={0: ["content"]}))
    assert seen == ["honest"] and row["ok"] is False


def test_a_request_that_served_nothing_needs_an_empty_entry_not_an_absent_one(tmp_path, monkeypatch):
    """A certified request whose served count is 0 (its `output` is []): it needs 0 ids, so an empty entry covers it,
    but leaving it out of the map is still a gap (`uncovered` counts every certified request, served or not). An id
    handed for a 0-output request over-covers it — the count gate passes (1 >= 0), leaving the spurious id for the
    content tie to refuse as an output no cell carries."""
    io0 = IO + [{"user": "q2", "prompt": [1], "max_tokens": 1, "output": []}]
    run = _run(tmp_path, io0)
    assert chain.uncovered(run, {"q0": [5, 6], "q1": [9], "q2": []}) is None            # empty entry covers it
    assert chain.uncovered(run, {"q0": [5, 6], "q1": [9]}) == "no sampled ids for 1 of its 3 certified requests"
    assert chain.uncovered(run, {"q0": [5, 6], "q1": [9], "q2": [7]}) is None           # 1 >= 0: count gate passes
    # the empty entry reaches the tie; the tie runs on honest
    _, seen = _tie(monkeypatch, run, _ids_file(tmp_path, (("q0", [5, 6]), ("q1", [9]), ("q2", []))), _row())
    assert seen == ["honest"]


def test_an_extra_request_not_in_the_certified_set_is_inert(tmp_path, monkeypatch):
    """ids naming a user that is not a certified request: `uncovered` reads only the certified requests, so the extra
    entry neither covers nor breaks anything — the honest map plus an extra key covers exactly as the honest map does,
    and the tie still runs (`tie_run` reads only the certified users, so the extra ids never enter the tie)."""
    run = _run(tmp_path)
    assert chain.uncovered(run, {"q0": [5, 6], "q1": [9], "qX": [99]}) is None
    # and an extra key can't paper over a real gap: a short real request is still refused despite the extra
    assert chain.uncovered(run, {"q0": [5], "q1": [9], "qX": [99, 1, 2]}) == \
        "fewer sampled ids than served outputs for 1 of its 2 certified requests (1 sampled, 2 served)"
    _, seen = _tie(monkeypatch, run, _ids_file(tmp_path, (("q0", [5, 6]), ("q1", [9]), ("qX", [99]))), _row())
    assert seen == ["honest"]
