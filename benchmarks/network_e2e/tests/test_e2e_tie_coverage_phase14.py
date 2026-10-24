"""REDTEAM phase 14, target 2 against NET-E2E's #3788, the I/O tie as a required gate on the advice profile
(`chain.tie_gate`, `chain.uncovered`). The skipped and raised cases are closed (the phase-12 regressions in
`test_e2e_tie_phase12.py`): a tie that didn't run or gave the honest run no verdict fails the report. This file attacks
the third: an `--ids` map that doesn't cover every certified request.

Finding: `chain.uncovered` — the gate's coverage check — tests only that each certified request's user is a **key** of
the ids map, not that its id list actually covers the request's served outputs. Its own docstring says it returns None
"when every one has its sampled ids", and the tie it guards "gives a request with no ids no rows of the Commit's port,
so a tie over ids that miss the run's requests ties none of their outputs" — exactly the case it is meant to prevent.
But a request mapped to an **empty** list, or a list **shorter** than its served outputs, passes `uncovered`:
`tie_run` then makes no port row for that request's uncovered outputs (`for k, x in enumerate(ids.get(user, ()))`), and
its accounting has no backstop for served cells with no committed id (`unplaced = max(0, committed - served)` is 0 when
`committed == 0`). So on the advice profile, where the tie is the only check of the served content, a prover can hand an
ids map that names every certified request yet leaves one request's served content un-tied, and the coverage gate passes.

The full `io_tie.reference_parts` isn't landed in this tree (it's IO-TIE's, and the phase-12 tie tests stub `tie_run`),
so this is demonstrated against the actual gate guard (`chain.uncovered`, `chain.read_ids` via NG), which are runnable;
the tie_run lines that leave the outputs un-tied are cited above.
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
    served outputs (`uncovered` reads the users)."""
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


def test_uncovered_admits_an_empty_id_list_for_a_certified_request(tmp_path):
    """The gate's coverage check passes a map that names every certified request but gives one of them no ids at all —
    the case `tie_run` ties nothing for. The honest-shaped map passes too, so the gate can't tell them apart."""
    run = _run(tmp_path, IO)
    assert chain.uncovered(run, {"q0": [5, 6], "q1": [9]}) is None, "the honest ids cover both requests"
    assert chain.uncovered(run, {"q0": [], "q1": [9]}) is None, \
        "FINDING: q0 is a key but has no ids, yet coverage passes"
    assert chain.uncovered(run, {"q0": [5, 6], "q1": []}) is None, "either request can be left with no ids"


def test_uncovered_admits_a_list_shorter_than_the_served_outputs(tmp_path):
    """A list shorter than a request's served outputs also passes: `uncovered` never compares the id count to the
    output count, so the tail outputs (`out_ids.append(xs[k] if k < len(xs) else 0)`) are left un-tied."""
    run = _run(tmp_path, IO)
    assert chain.uncovered(run, {"q0": [5], "q1": [9]}) is None, \
        "FINDING: q0 served two outputs but one id passes coverage"


def test_the_missing_key_case_is_the_only_one_the_gate_catches(tmp_path):
    """Contrast: only a request whose user is absent from the map is caught — the phase-12 regression's case. An empty
    or short list for a present key is not, so a prover drops coverage by naming the request with no ids rather than
    omitting it."""
    run = _run(tmp_path, IO)
    assert chain.uncovered(run, {"q1": [9]}) == "no sampled ids for 1 of its 2 certified requests"
    assert chain.uncovered(run, {}) == "no sampled ids for 2 of its 2 certified requests"
    # the empty-list map a prover would hand instead passes the same check the missing key fails
    assert chain.uncovered(run, {"q0": [], "q1": []}) is None


def test_read_ids_builds_the_empty_list_a_prover_hands(tmp_path):
    """The attack is reachable from the `--ids` file: an entry with an empty `output_ids` yields an empty list under the
    request's user, which `uncovered` then admits."""
    got = NG.read_ids(_ids_file(tmp_path, (("q0", []), ("q1", [9]))))
    assert got == {"q0": [], "q1": [9]}
    run = _run(tmp_path, IO)
    assert chain.uncovered(run, got) is None, "the gate admits the prover's under-covering ids file"
