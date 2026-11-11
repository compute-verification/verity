"""`stop_audit.py`'s verifier's half on a stop class's hand-off, sampled on a test's own challenger
(`fake_challenger`): the stop stream built again from ``stage1.json`` alone, the verifier's own ``--registered`` file,
the audit record's entries and what each refuses. No proof runs here: a session's verdicts are stand-ins of the form
`a0.lean_verify` reports, and ``stop_audit_small.sh`` is the run that proves. The class is the two-stage driver's
tests' (`test_two_stage_driver._stop_class`)."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

from verity.circuits.codec import canonical_json
from verity.commitments.windows import rows as CS
from verity.protocols.spot.verifier import service as S
from verity_vllm.pipeline import boundary_rows as BR, boundary_stop as BS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
T = importlib.import_module("test_two_stage_driver")
SA = importlib.import_module("stop_audit")
SMALL = importlib.import_module("stop_audit_small")

DESIGN = f"rate:{T.Q}/{T.N}"
SESSION = "5e" * 64
NULL = {SA.BOUNDARY_AUDIT, SA.REQUEST_ROW, SA.STOP_RULE}


def _stage1(tmp_path: Path, monkeypatch, seed: str) -> dict:
    """A stop class's hand-off as `two_stage.py first --stop` writes it, on one session of a test's challenger."""
    pubs = [json.loads(json.dumps(T._stop_class().public()))]
    monkeypatch.setattr(T.AC, "start", T._start(T.OneSession(seed)))
    (got,) = T.TS._first_all(T._args(tmp_path), pubs, T.PAIR, T.RULE, None, T.STOP)
    assert got["stop"]["order"] == {"held": True, "reason": None}
    return json.loads(json.dumps(got))


def _proved(r, file: dict, **lean: object) -> dict:
    """A session's records as `stop_audit.session` returns them, its verdicts stand-ins: Lean's accepts, names the
    verifier's file and reports its ports' values and roots."""
    units = list(r.sample.units)
    verdict = {"accepted": True, "of_record": True, "digest": "dd" * 64,
               "registered": {p: {"value": e["value"], "root": e["root"]} for p, e in file["ports"].items()},
               "registered_sha512": hashlib.sha512(canonical_json(file)).hexdigest(), **lean}
    return {"sessions": [SESSION], "served": units, "proved_units": units, "instances": len(units),
            "verdicts": {SA.LEAN: verdict, SA.M0: {"accepted": True}}, "statement": {SA.MEMBER: {"statement": "stub"}}}


def _audit(stage1: dict, outcome: dict | None = None, **lean: object) -> tuple[dict, dict]:
    """One audit of the hand-off's stop sample on a stream built again: `decide`'s outcome and its summary."""
    r = SA.resume(stage1, T.PAIR, T.STOP, DESIGN)
    file = SA.reads(r)
    got = SA.decide(r, stage1, file, _proved(r, file, **lean), outcome)
    return got, SA.summary_of(r, T.STOP, DESIGN, got)


def test_the_verifiers_half_is_built_from_the_hand_off_and_the_verifiers_own_copies(tmp_path, monkeypatch):
    """`resume` registers the stop record again on the boundary record its own boundary stream registered, and samples
    the stop blocks the first stage sampled. `reads` is the verifier's ``--registered`` file: the two parameters at the
    boundary record's roots and the output row at the stop record's, unit i at row i. Neither opens the class's
    directory nor calls the prover's half. Refused: another stop rule, another sampling design, a sample that is not
    the hand-off's, a hand-off with no stop sample, and a stop record on a boundary record the boundary stream never
    logged (case 2)."""
    stage1 = _stage1(tmp_path, monkeypatch, "resume")

    def prover(*_a: object, **_k: object):
        raise AssertionError("the verifier's half called the prover's")
    for owner, name in ((BR, "reopen"), (SA, "opened"), (SA, "staged")):
        monkeypatch.setattr(owner, name, prover)
    r = SA.resume(stage1, T.PAIR, T.STOP, DESIGN)
    s = stage1["stop"]
    assert r.registration.digest == s["registration"] and list(r.sample.units) == s["units"]
    assert r.first.registration.digest == stage1["registration"] == s["record"]["previous"]
    file = SA.reads(r)
    boundary = {e["port"]: e["root"] for e in stage1["record"]["roots"]}
    (stop,) = s["record"]["roots"]
    assert set(file["ports"]) == {*BS.STOP_READS, SA.OUT}
    assert {p: file["ports"][p]["root"] for p in BS.STOP_READS} == {p: boundary[v] for p, v in BS.STOP_READS.items()}
    out = file["ports"][SA.OUT]
    assert (out["value"], out["root"], out["leaves"]) == (CS.own_value(BR.STOP), stop["root"], T.N)
    assert all(e["reads"] == list(range(T.N)) for e in file["ports"].values())

    with pytest.raises(ValueError, match="not the verifier's own"):
        SA.resume(stage1, T.PAIR, BS.Stop((7,), 64, 4), DESIGN)
    with pytest.raises(ValueError, match="Spec isn't the one"):
        SA.resume(stage1, T.PAIR, T.STOP, f"rate:1/{T.N}")
    other = [u for u in range(T.N) if u not in s["units"]][:len(s["units"])]
    with pytest.raises(ValueError, match="samples other stop blocks"):
        SA.resume({**stage1, "stop": {**s, "units": other}}, T.PAIR, T.STOP, DESIGN)
    with pytest.raises(ValueError, match="registered or sampled no stop block"):
        SA.resume({**stage1, "stop": {"error": "refused"}}, T.PAIR, T.STOP, DESIGN)
    with pytest.raises(SA.Unlogged, match="no record of the class's boundary stream's log"):
        SA.resume(SMALL.unlogged(stage1, T.PAIR), T.PAIR, T.STOP, DESIGN)


def test_an_accepted_record_names_what_it_did_not_check_and_is_not_complete(tmp_path, monkeypatch):
    """On Lean's accepting verdict of record the stop stream's audit record accepts with the order recomputed from the
    challenger's signed log (case 1), and three entries without a verdict: the boundary stream's audit, the request
    row and the stop rule, each with its premise. So the record is not complete, and the summary says no length is
    held and lists the premises, soundness's two sentences among them."""
    stage1 = _stage1(tmp_path, monkeypatch, "accept")
    got, summary = _audit(stage1)
    audit = got["audit"]
    assert got["accepted"] is True and got["symbol"] == S.ACCEPT and audit["accepted"] is True
    assert all(c["ok"] for c in audit["checks"].values())
    verdicts = audit["verdicts"]
    assert {k for k, v in verdicts.items() if v["accepted"] is None} == NULL
    assert all(verdicts[k]["accepted"] is True for k in (SA.LEAN, SA.M0, SA.READS, SA.ORDER))
    assert [k for k, v in verdicts.items() if v.get("of_record")] == [SA.LEAN]
    order = verdicts[SA.ORDER]
    assert order["case"] == 1 and order["premise"] == SA.ORDER_PREMISE and order["tool"] == SA.TOOL
    assert order["boundary"] == {"registration": stage1["registration"], "received": 0, "audit": None}
    assert order["stop"] == {"registration": stage1["stop"]["registration"], "previous": stage1["registration"]}
    assert verdicts[SA.BOUNDARY_AUDIT]["premise"] == SA.BOUNDARY_PREMISE == (
        "boundary stream's audit: its verdict carried from that stream's audit record by the audit driver, not checked "
        "by the Lean verifier")
    assert verdicts[SA.STOP_RULE]["premise"] == SA.STOP_RULE_PREMISE == (
        "stop rule: the driver's parameter, not held to the pinned checkpoint")
    assert verdicts[SA.REQUEST_ROW]["premise"] == SA.REQUEST_PREMISE
    assert audit["complete"] is False
    assert (summary["accepted"], summary["complete"], summary["length_held"]) == (True, False, False)
    assert summary["not_run"] == sorted(NULL)
    assert {SA.BOUNDARY_PREMISE, SA.STOP_RULE_PREMISE, SA.REQUEST_PREMISE, SA.ORDER_PREMISE} == set(summary["premises"])


def test_a_record_is_refused_by_each_entry_that_ran_and_did_not_accept(tmp_path, monkeypatch):
    """One hand-off, audited on a stream built again each time. Refused: a session Lean's verdict refuses (a length
    that is not the rule's); a boundary stream's outcome that did not accept, or that is another registration's
    (case 3); a verdict that names another ``--registered`` file than the verifier's, or reports another root for a
    port; and an order the challenger's signed log does not show, here a log with no coin after the stop registration.
    With an accepting boundary outcome the entry accepts and names the audit record, and the record is still not
    complete."""
    stage1 = _stage1(tmp_path, monkeypatch, "refusals")
    _, summary = _audit(stage1, accepted=False, why="the proof does not verify")
    assert summary["accepted"] is False and summary["verdicts"][SA.LEAN] is False and summary["verdicts"][SA.ORDER] is True

    refused = SMALL.unaccepted(stage1)
    got, summary = _audit(stage1, refused)
    entry = got["audit"]["verdicts"][SA.BOUNDARY_AUDIT]
    assert summary["accepted"] is False and summary["failed"] == [] and entry["accepted"] is False
    assert entry["audit"] == SA.sha512(refused["audit"]) and "did not accept" in summary["why"][SA.BOUNDARY_AUDIT]
    assert got["audit"]["verdicts"][SA.ORDER]["boundary"]["audit"] == entry["audit"]
    _, summary = _audit(stage1, {**refused, "registration": "00" * 64, "symbol": S.ACCEPT})
    assert summary["accepted"] is False and "another registration's" in summary["why"][SA.BOUNDARY_AUDIT]

    accepted = {**refused, "symbol": S.ACCEPT, "audit": {**refused["audit"], "accepted": True}}
    got, summary = _audit(stage1, accepted)
    assert summary["accepted"] is True and got["audit"]["verdicts"][SA.BOUNDARY_AUDIT]["accepted"] is True
    assert summary["complete"] is False and summary["not_run"] == sorted(NULL - {SA.BOUNDARY_AUDIT})

    _, summary = _audit(stage1, registered_sha512="00" * 64)
    assert summary["accepted"] is False and summary["verdicts"][SA.READS] is False
    r = SA.resume(stage1, T.PAIR, T.STOP, DESIGN)
    file = SA.reads(r)
    wrong = {**file["ports"], SA.OUT: {**file["ports"][SA.OUT], "root": "00" * 64}}
    _, summary = _audit(stage1, registered={p: {"value": e["value"], "root": e["root"]} for p, e in wrong.items()})
    assert summary["accepted"] is False and summary["verdicts"][SA.READS] is False

    s = stage1["stop"]
    log = [x for x in s["evidence"]["session"]["log"] if s["registration"] not in x]
    unseen = {**stage1, "stop": {**s, "evidence": {**s["evidence"], "session": {**s["evidence"]["session"], "log": log}}}}
    got, summary = _audit(unseen)
    assert summary["accepted"] is False
    assert got["audit"]["verdicts"][SA.ORDER]["accepted"] is False and got["audit"]["verdicts"][SA.ORDER]["case"] == 2


def test_the_boundary_audit_entry_reads_the_boundary_streams_outcome_record():
    """`boundary_entry` alone: null without a record; an outcome record on the class's boundary registration gives
    its symbol's verdict and the audit record's SHA-512; anything else is refused by name."""
    boundary = "ab" * 64
    record = {"format": S.OUTCOME, "registration": boundary, "symbol": S.ACCEPT, "audit": {"accepted": True}}
    assert SA.boundary_entry(boundary, None) == {
        "accepted": None, "of_record": False, "tool": SA.TOOL, "premise": SA.BOUNDARY_PREMISE, "audit": None,
        "registration": boundary}
    got = SA.boundary_entry(boundary, record)
    assert got["accepted"] is True and got["audit"] == SA.sha512(record["audit"]) and got["of_record"] is False
    for wrong, why in (({**record, "symbol": S.REJECT}, "did not accept"),
                       ({**record, "audit": {"accepted": False}}, "did not accept"),
                       ({**record, "audit": None}, "did not accept"),
                       ({**record, "registration": "cd" * 64}, "another registration's"),
                       ({**record, "format": "verity/one-stage/audit/v0"}, "not a ")):
        got = SA.boundary_entry(boundary, wrong)
        assert got["accepted"] is False and why in got["why"], wrong


def test_the_rehearsals_wrong_class_commits_stop_words_that_are_not_the_rules():
    """`stop_audit_small`'s wrong class: its ``commit.json`` differs from the class's own in `b.stop`'s root alone, the
    root of the tree over every stop word with bit 0 flipped, which is what the negative's prover opens
    (`stop_audit.opened`); and the honest prover's rows open the class's own roots."""
    c = SMALL.committed(check=T.PAIR, outputs=4, spans=T.N)
    pub, wrong = json.loads(json.dumps(c.public())), SMALL.wrong_length(c)
    assert {k for k in pub["values"] if pub["values"][k] != wrong["values"][k]} == {BR.STOP}
    assert {k: v for k, v in pub.items() if k != "values"} == {k: v for k, v in wrong.items() if k != "values"}
    honest, forged = SA.opened(c), SA.opened(c, range(T.N))
    assert honest["roots"] == {v: c.roots[v].hex() for v in (*BS.STOP_READS.values(), BR.STOP)}
    assert forged["roots"][BR.STOP] == wrong["values"][BR.STOP]["root"] != honest["roots"][BR.STOP]
    assert (honest["rows"][SA.OUT][:, 0] ^ forged["rows"][SA.OUT][:, 0]).tolist() == [1] * T.N
    assert len(honest["paths"][SA.OUT]) == len(honest["salts"][SA.OUT]) == T.N
