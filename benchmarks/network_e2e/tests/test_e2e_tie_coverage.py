"""An adversarial case against the I/O tie as a required gate on the advice profile (`chain.tie_gate`,
`chain.uncovered`), turned into regressions.

The finding: the gate's coverage check tested only that each certified request's user was a key of the ids map, not
that its ids covered the request's served outputs. `negatives.tie_run` gives each sampled id one row of the Commit's
port and has no backstop for a served output with none (its unplaced count is committed outputs no cell carries, 0
when nothing is committed), so a map naming a request with an empty or short list left those outputs untied, and on
the advice profile, where the tie is the only check of the served content, the report read OK.

The fix: `uncovered` holds each certified request's sampled ids to its served outputs (``io.json``'s
``output``): fewer ids than outputs, an empty list for a request that served any among them, is a gap, so the run has
no verdict and the report isn't OK (`test_e2e_tie_report.py` holds that end to end).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("network_e2e", HERE / "e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e2e = _load()
chain = e2e.chain
NG = chain.NG


def _run(tmp_path, requests):
    """A run dir with the io.json the rehearsal's check writes on the advice profile: its certified requests' users and
    served outputs."""
    d = tmp_path / "honest"
    d.mkdir(parents=True)
    (d / "io.json").write_text(json.dumps({"windows": 3, "requests": requests}))
    return d


def _ids_file(tmp_path, pairs):
    p = tmp_path / "ids.jsonl"
    p.write_text("".join(json.dumps({"user": u, "output_ids": x}) + "\n" for u, x in pairs))
    return p


IO = [{"user": "q0", "prompt": [1, 2, 3, 4], "max_tokens": 2, "output": [5, 6]},
      {"user": "q1", "prompt": [1, 2, 3], "max_tokens": 1, "output": [9]}]


def test_an_empty_id_list_for_a_certified_request_is_a_gap(tmp_path):
    """A map that names every certified request but gives one of them no ids at all is refused, whichever request it
    is; the honest map, an id for each served output, passes."""
    run = _run(tmp_path, IO)
    assert chain.uncovered(run, {"q0": [5, 6], "q1": [9]}) is None
    assert chain.uncovered(run, {"q0": [], "q1": [9]}) == \
        "fewer sampled ids than served outputs for 1 of its 2 certified requests (0 sampled, 2 served)"
    assert chain.uncovered(run, {"q0": [5, 6], "q1": []}) == \
        "fewer sampled ids than served outputs for 1 of its 2 certified requests (0 sampled, 1 served)"


def test_a_list_shorter_than_the_served_outputs_is_a_gap(tmp_path):
    """One id for a request that served two leaves its second output untied: refused. More ids than served outputs
    pass here; `tie_run` refuses the committed outputs no cell carries."""
    run = _run(tmp_path, IO)
    assert chain.uncovered(run, {"q0": [5], "q1": [9]}) == \
        "fewer sampled ids than served outputs for 1 of its 2 certified requests (1 sampled, 2 served)"
    assert chain.uncovered(run, {"q0": [5, 6, 7], "q1": [9]}) is None


def test_a_missing_key_and_a_short_list_are_both_gaps(tmp_path):
    """The missing key (the phase-12 regression's case) and the empty list a prover would hand instead are refused
    alike, each counted."""
    run = _run(tmp_path, IO)
    assert chain.uncovered(run, {"q1": [9]}) == "no sampled ids for 1 of its 2 certified requests"
    assert chain.uncovered(run, {}) == "no sampled ids for 2 of its 2 certified requests"
    assert chain.uncovered(run, {"q0": [], "q1": []}) == \
        "fewer sampled ids than served outputs for 2 of its 2 certified requests (0 sampled, 3 served)"
    assert chain.uncovered(run, {"q0": [5]}) == ("no sampled ids for 1 and fewer sampled ids than served outputs for 1 "
                                                 "of its 2 certified requests (1 sampled, 2 served)")


def test_a_request_without_its_served_outputs_is_refused(tmp_path):
    """io.json without a request's ``output`` gives nothing to hold its ids to: refused, never counted as covered."""
    run = _run(tmp_path, [IO[0], {"user": "q1", "prompt": [1, 2, 3], "max_tokens": 1}])
    assert chain.uncovered(run, {"q0": [5, 6], "q1": [9]}) == \
        "a certified request in its io.json without its served outputs to hold the sampled ids to"


def test_the_empty_list_from_an_ids_file_is_refused(tmp_path):
    """The attack's `--ids` file: an entry with an empty ``output_ids`` reads as an empty list under the request's
    user (`negatives.read_ids`), which `uncovered` now refuses."""
    got = NG.read_ids(_ids_file(tmp_path, (("q0", []), ("q1", [9]))))
    assert got == {"q0": [], "q1": [9]}
    assert chain.uncovered(_run(tmp_path, IO), got) == \
        "fewer sampled ids than served outputs for 1 of its 2 certified requests (0 sampled, 2 served)"
