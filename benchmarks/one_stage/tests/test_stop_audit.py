"""`stop_audit.py`'s verifier's half on a stop class's hand-off, sampled on a test's own challenger
(`fake_challenger`): the stop stream built again from the hand-off alone, the verifier's own ``--registered`` file,
the audit record's entries and what each refuses, and Lean's verdicts on the read across the class's two streams
(`Record.across`) held to the Python's on fourteen written cases. No input is random. No proof runs here: a session's
verdicts are stand-ins of the form `a0.lean_verify` reports, and ``stop_audit_small.sh`` is the run that proves. The
class is the two-stage driver's tests' (`test_two_stage_driver._stop_class`)."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

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


def _resume(handoff: dict, stop: BS.Stop = T.STOP, design: str = DESIGN):
    """`stop_audit.resume` under the proof deadline the test's first stage ran under."""
    return SA.resume(handoff, T.PAIR, stop, design, proof_deadline=T.RULE.proof_deadline)


def _handoff(tmp_path: Path, monkeypatch, seed: str) -> dict:
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


def _audit(handoff: dict, outcome: dict | None = None, **lean: object) -> tuple[dict, dict]:
    """One audit of the hand-off's stop sample on a stream built again: `decide`'s outcome and its summary."""
    r = _resume(handoff)
    file = SA.reads(r)
    got = SA.decide(r, handoff, file, _proved(r, file, **lean), outcome)
    return got, SA.summary_of(r, T.STOP, DESIGN, got)


def test_the_verifiers_half_is_built_from_the_hand_off_and_the_verifiers_own_copies(tmp_path, monkeypatch):
    """`resume` registers the stop record again on the boundary record its own boundary stream registered, and samples
    the stop blocks the first stage sampled. `reads` is the verifier's ``--registered`` file: the two parameters at the
    boundary record's roots and the output row at the stop record's, unit i at row i. Neither opens the class's
    directory nor calls the prover's half. Refused: another stop rule, another sampling design, a sample that is not
    the hand-off's, a hand-off with no stop sample, and a stop record on a boundary record the boundary stream never
    logged (case 2)."""
    handoff = _handoff(tmp_path, monkeypatch, "resume")

    def prover(*_a: object, **_k: object):
        raise AssertionError("the verifier's half called the prover's")
    for owner, name in ((BR, "reopen"), (SA, "opened"), (SA, "staged")):
        monkeypatch.setattr(owner, name, prover)
    r = _resume(handoff)
    s = handoff["stop"]
    assert r.registration.digest == s["registration"] and list(r.sample.units) == s["units"]
    assert r.first.registration.digest == handoff["registration"] == s["record"]["previous"]
    file = SA.reads(r)
    boundary = {e["port"]: e["root"] for e in handoff["record"]["roots"]}
    (stop,) = s["record"]["roots"]
    assert set(file["ports"]) == {*BS.STOP_READS, SA.OUT}
    assert {p: file["ports"][p]["root"] for p in BS.STOP_READS} == {p: boundary[v] for p, v in BS.STOP_READS.items()}
    out = file["ports"][SA.OUT]
    assert (out["value"], out["root"], out["leaves"]) == (CS.own_value(BR.STOP), stop["root"], T.N)
    assert all(e["reads"] == list(range(T.N)) for e in file["ports"].values())

    with pytest.raises(ValueError, match="not the verifier's own"):
        _resume(handoff, BS.Stop((7,), 64, 4))
    with pytest.raises(ValueError, match="Spec isn't the one"):
        _resume(handoff, design=f"rate:1/{T.N}")
    other = [u for u in range(T.N) if u not in s["units"]][:len(s["units"])]
    with pytest.raises(ValueError, match="samples other stop blocks"):
        _resume({**handoff, "stop": {**s, "units": other}})
    with pytest.raises(ValueError, match="registered or sampled no stop block"):
        _resume({**handoff, "stop": {"error": "refused"}})
    with pytest.raises(SA.Unlogged, match="no record of the class's boundary stream's log"):
        _resume(SMALL.unlogged(handoff, T.PAIR))


def test_an_accepted_record_names_what_it_did_not_check_and_is_not_complete(tmp_path, monkeypatch):
    """On Lean's accepting verdict of record the stop stream's audit record accepts with Lean's verdict on the order,
    read from the challenger's signed log, and three entries without a verdict: the boundary stream's audit, the
    request row and the stop rule, each with its premise. The boundary stream's audit record is missing, which is case
    3. The order's entry carries that ``case``, which reads the boundary stream's audit record as handed, so it names
    the boundary audit's premise too, and no other entry names one. So the record is not complete, and the summary
    says no length is held and lists the three premises, the boundary audit's and the stop rule's held here word for
    word."""
    handoff = _handoff(tmp_path, monkeypatch, "accept")
    got, summary = _audit(handoff)
    audit = got["audit"]
    assert got["accepted"] is True and got["symbol"] == S.ACCEPT and audit["accepted"] is True
    assert all(c["ok"] for c in audit["checks"].values())
    verdicts = audit["verdicts"]
    assert {k for k, v in verdicts.items() if v["accepted"] is None} == NULL
    assert all(verdicts[k]["accepted"] is True for k in (SA.LEAN, SA.M0, SA.READS, SA.ORDER))
    assert [k for k, v in verdicts.items() if v.get("of_record")] == [SA.LEAN]
    order = verdicts[SA.ORDER]
    assert order["case"] == summary["case"] == 3
    assert order["tool"] == SA.ACROSS == f"{SA.V.EXE} across"
    assert order["boundary"] == {"registration": handoff["registration"], "received": 0, "audit": None}
    assert order["stop"] == {"registration": handoff["stop"]["registration"], "previous": handoff["registration"]}
    assert {k for k, v in verdicts.items() if "premise" in v} == NULL | {SA.ORDER}
    assert order["premise"] == verdicts[SA.BOUNDARY_AUDIT]["premise"] == SA.BOUNDARY_PREMISE == (
        "boundary stream's audit: read from the audit record the audit driver hands; Lean checks its format and "
        "registration and reads its `accepted` and `complete`; nothing holds it to the record the boundary stream's "
        "verifier wrote")
    assert verdicts[SA.STOP_RULE]["premise"] == SA.STOP_RULE_PREMISE == (
        "stop rule: the driver's parameter, not held to the pinned checkpoint")
    assert verdicts[SA.REQUEST_ROW]["premise"] == SA.REQUEST_PREMISE
    assert audit["complete"] is False
    assert (summary["accepted"], summary["complete"], summary["length_held"]) == (True, False, False)
    assert summary["not_run"] == sorted(NULL)
    assert {SA.BOUNDARY_PREMISE, SA.STOP_RULE_PREMISE, SA.REQUEST_PREMISE} == set(summary["premises"])


def test_a_record_is_refused_by_each_entry_that_ran_and_did_not_accept(tmp_path, monkeypatch):
    """One hand-off, audited on a stream built again each time. Refused: a session Lean's verdict refuses (a length
    that is not the rule's); a boundary stream's outcome that did not accept, that is another registration's, or
    whose audit record Lean's verdict refuses (case 3); a verdict that names another ``--registered`` file than the
    verifier's, or reports another root for a port; and an order the challenger's signed log does not show, here a
    log with no coin after the stop registration (case 2). With a boundary outcome that accepted and is complete the
    entry accepts and names the audit record (case 1), and the record is still not complete; the record handed is
    one made by hand, which no stream wrote, and the two entries that read it name that premise: case 1 on a handed
    record is on no entry of record, and the summary holds no length on it. With one that accepted
    with an entry that did not run the entry has no verdict and gives its reason (case 4): the record accepts, and
    the boundary-audit entry is among those that keep it from being complete."""
    handoff = _handoff(tmp_path, monkeypatch, "refusals")
    _, summary = _audit(handoff, accepted=False, why="the proof does not verify")
    assert summary["accepted"] is False and summary["verdicts"][SA.LEAN] is False and summary["verdicts"][SA.ORDER] is True
    assert summary["failed"] == ["registered", "roots"]

    refused = SMALL.unaccepted(handoff)
    got, summary = _audit(handoff, refused)
    entry = got["audit"]["verdicts"][SA.BOUNDARY_AUDIT]
    assert summary["accepted"] is False and summary["failed"] == [] and entry["accepted"] is False
    assert entry["audit"] == SA.sha512(refused["audit"]) and "did not accept" in summary["why"][SA.BOUNDARY_AUDIT]
    assert entry["lean"]["accepted"] is False and "did not accept" in entry["lean"]["why"] and summary["case"] == 3
    assert got["audit"]["verdicts"][SA.ORDER]["boundary"]["audit"] == entry["audit"]
    _, summary = _audit(handoff, {**refused, "registration": "00" * 64, "symbol": S.ACCEPT})
    assert summary["accepted"] is False and "another registration's" in summary["why"][SA.BOUNDARY_AUDIT]

    accepted = {**refused, "symbol": S.ACCEPT, "audit": {**refused["audit"], "accepted": True}}
    got, summary = _audit(handoff, accepted)
    entry = got["audit"]["verdicts"][SA.BOUNDARY_AUDIT]
    assert summary["accepted"] is True and entry["accepted"] is True and summary["case"] == 1
    assert entry["lean"] == {"tool": SA.ACROSS, "accepted": True, "complete": True} and entry["tool"] == SA.TOOL
    assert entry["complete"] is True and "why" not in entry
    assert entry["premise"] == got["audit"]["verdicts"][SA.ORDER]["premise"] == SA.BOUNDARY_PREMISE
    assert SA.BOUNDARY_PREMISE in summary["premises"]
    assert [k for k, v in got["audit"]["verdicts"].items() if v.get("of_record")] == [SA.LEAN]
    assert summary["complete"] is False and summary["not_run"] == sorted(NULL - {SA.BOUNDARY_AUDIT})
    assert summary["length_held"] is False
    got, summary = _audit(handoff, SMALL.incomplete(handoff))
    entry = got["audit"]["verdicts"][SA.BOUNDARY_AUDIT]
    assert got["accepted"] is True and got["symbol"] == S.ACCEPT and got["audit"]["complete"] is False
    assert (entry["accepted"], entry["complete"], entry["why"]) == (None, False, SA.NOT_COMPLETE)
    assert (entry["lean"]["accepted"], entry["lean"]["complete"]) == (True, False) and "not complete" in entry["lean"]["why"]
    assert summary["case"] == got["audit"]["verdicts"][SA.ORDER]["case"] == 4
    assert (summary["accepted"], summary["complete"], summary["length_held"]) == (True, False, False)
    assert summary["not_run"] == sorted(NULL) and summary["why"] == {SA.BOUNDARY_AUDIT: SA.NOT_COMPLETE}
    _, summary = _audit(handoff, {**accepted, "audit": {**accepted["audit"], "registration_digest": "00" * 64}})
    assert summary["accepted"] is False and summary["case"] == 3
    assert "Lean's verdict on the boundary stream's audit record" in summary["why"][SA.BOUNDARY_AUDIT]

    _, summary = _audit(handoff, registered_sha512="00" * 64)
    assert summary["accepted"] is False and summary["verdicts"][SA.READS] is False
    r = _resume(handoff)
    file = SA.reads(r)
    wrong = {**file["ports"], SA.OUT: {**file["ports"][SA.OUT], "root": "00" * 64}}
    _, summary = _audit(handoff, registered={p: {"value": e["value"], "root": e["root"]} for p, e in wrong.items()})
    assert summary["accepted"] is False and summary["verdicts"][SA.READS] is False and summary["failed"] == ["roots"]

    s = handoff["stop"]
    log = [x for x in s["evidence"]["session"]["log"] if s["registration"] not in x]
    unseen = {**handoff, "stop": {**s, "evidence": {**s["evidence"], "session": {**s["evidence"]["session"], "log": log}}}}
    got, summary = _audit(unseen)
    assert summary["accepted"] is False
    assert got["audit"]["verdicts"][SA.ORDER]["accepted"] is False and got["audit"]["verdicts"][SA.ORDER]["case"] == 2
    assert summary["case"] == 2 and "the later sample's coins" in summary["why"][SA.ORDER]


def _generic(why: str | None) -> str | None:
    """`two_stage.stop_order`'s reason in `Record.across`'s names: the stop stream is the later one, and the class's
    boundary stream the earlier."""
    for mine, leans in (("the stop sample", "the later sample"), ("the class's boundary registration", "the earlier registration"),
                        ("the boundary registration's", "the earlier registration's"),
                        ("the stop registration", "the later registration")):
        why = why if why is None else why.replace(mine, leans)
    return why


def _released(seed: str, *commits: tuple[str, int]) -> tuple[T.OneSession, dict[str, bytes], dict[str, str]]:
    """One session of the test's challenger that received ``commits``' digests in that order, each in a family of its
    own and followed by its count of coins, released and read whole. -> the challenger, the bytes read after each
    digest and the family they were read from."""
    one, read, family = T.OneSession(seed), {}, {}
    for digest, coins in commits:
        one.commit(bytes.fromhex(digest))
        read[digest], family[digest] = one(64 * coins), one.label
    return one, read, family


def test_lean_and_the_python_agree_on_each_case_of_a_read_across_streams(tmp_path, monkeypatch):
    """`Record.across` (the Lean verifier's piece, as `stop_audit.across` calls it) against the Python on one
    hand-off, verdict by verdict: ``logged`` against the boundary stream's log's digests (`resume`'s test), ``order``
    against `two_stage.stop_order`, reason for reason, ``audit``, its ``accepted`` and its ``complete``, against
    `boundary_audit` on the outcome record, and ``case`` against the four. Fourteen written cases: no input is random,
    and none is one the Lean refuses, so the two sides are held to each other on these and no further. Case 1: the
    hand-off with a boundary outcome that accepted and is complete, on the first stage's session and on one made by
    hand, the two registrations' coins each in a family of its own. Case 2: a stop record on a
    boundary record never logged; an empty log; and, on sessions of the test's challenger that received the two
    registrations by hand, the stop registration's coins released first, no receipt of the boundary registration, a
    coin opened and not read, no evidence, no family named, and the boundary stream's family named for the stop
    sample's coins, which are held to the family named and not to the one their evidence names. Case 3: the boundary
    stream's audit record missing, refused, and another registration's. Case 4: that audit record accepted with an
    entry that did not run."""
    handoff = _handoff(tmp_path, monkeypatch, "across")
    r, s = _resume(handoff), handoff["stop"]
    earlier, later = handoff["registration"], s["registration"]
    refused = SMALL.unaccepted(handoff)
    accepted = {**refused, "symbol": S.ACCEPT, "audit": {**refused["audit"], "accepted": True}}
    others = {**accepted, "registration": "00" * 64, "audit": {**accepted["audit"], "registration_digest": "00" * 64}}

    def handed(seed: str, *commits: tuple[str, int], read: int | None = None) -> dict:
        one, got, family = _released(seed, *commits)
        return {"evidence": one.evidence(later), "read": got[later][:read].hex(), "family": family[later]}
    assert s["family"] == T.AC.family(1) != handoff["family"] == T.AC.family(0)
    cases = {
        "accepted": (1, {}, None, accepted, None),
        "by hand": (1, handed("by-hand", (earlier, 1), (later, 2)), None, accepted, None),
        "unlogged": (2, {"record": SMALL.unlogged(handoff, T.PAIR)["stop"]["record"]}, None, None, "the later sample's coins"),
        "an empty log": (2, {}, [], accepted, None),
        "the stop registration's coins first": (2, handed("first", (later, 1), (earlier, 1)), None, accepted,
                                                "the earlier registration's coins come first"),
        "no receipt": (2, handed("no-receipt", (later, 1)), None, accepted, "no coin released after the earlier registration"),
        "not read whole": (2, handed("short", (earlier, 1), (later, 2), read=64), None, accepted, "a coin opened and not read"),
        "no evidence": (2, {"evidence": None}, None, accepted, "published no signed log"),
        "no family": (2, {"family": None}, None, accepted, "read no coins of a challenger's family"),
        "another family": (2, {"family": handoff["family"]}, None, accepted,
                           f"coins are not the challenger's {handoff['family']} family's"),
        "no audit record": (3, {}, None, None, None),
        "a refused audit record": (3, {}, None, refused, None),
        "another registration's audit record": (3, {}, None, others, None),
        "an audit record that accepted and is not complete": (4, {}, None, SMALL.incomplete(handoff), None),
    }
    assert len(cases) == 14 and sorted(c for c, *_ in cases.values()) == [1] * 2 + [2] * 8 + [3] * 3 + [4]
    for name, (case, stop, log, outcome, why) in cases.items():
        mine = {**handoff, "stop": {**s, **stop}}
        log = list(r.first.stream.log) if log is None else log
        previous = mine["stop"]["record"]["previous"]
        logged = previous in [SA.R.digest(rec) for rec in log]
        order = T.TS.stop_order(mine["stop"]["evidence"], r.layout.run, mine["stop"]["family"], previous,
                                SA.R.digest(mine["stop"]["record"]), bytes.fromhex(mine["stop"]["read"]), T.AC.lms_verify)
        audit = SA.boundary_audit(previous, outcome)
        lean = SA.across(mine, NS(layout=r.layout, stream=NS(log=log)), outcome)
        assert "refused" not in lean, (name, lean)
        assert lean["logged"]["accepted"] is logged and lean["received"] == (0 if logged else None), (name, lean)
        assert lean["order"] == ({"accepted": True} if order is None else {"accepted": False, "why": _generic(order)}), (name, lean, order)
        assert (lean["audit"]["accepted"], lean["audit"]["complete"]) == (audit["accepted"], audit["complete"]), (name, lean, audit)
        assert ("why" in lean["audit"]) == (audit["accepted"] is False or (audit["accepted"] is True and audit["complete"] is not True)), (name, lean)
        assert lean["case"] == case == (2 if not logged or order is not None else 3 if audit["accepted"] is not True
                                        else 4 if audit["complete"] is not True else 1), (name, lean)
        assert (lean["earlier"], lean["later"]) == (previous, SA.R.digest(mine["stop"]["record"])), name
        assert why is None or why in lean["order"]["why"], (name, lean)
        assert (order is None) == (why is None), (name, order)


def test_the_boundary_audit_entry_reads_the_boundary_streams_outcome_record():
    """`boundary_entry` alone: null without a record; an outcome record on the class's boundary registration whose
    audit record accepted and is complete gives its symbol's verdict and the audit record's SHA-512, and needs Lean's
    verdict on the audit record, its ``accepted`` and its ``complete``, when one is given; one whose audit record
    accepted and is not complete, by the driver's reading or by Lean's, has no verdict and says why; anything else is
    refused by name."""
    boundary = "ab" * 64
    record = {"format": S.OUTCOME, "registration": boundary, "symbol": S.ACCEPT, "audit": {"accepted": True, "complete": True}}
    assert SA.boundary_entry(boundary, None) == {
        "accepted": None, "complete": None, "of_record": False, "tool": SA.TOOL, "premise": SA.BOUNDARY_PREMISE,
        "audit": None, "registration": boundary}
    got = SA.boundary_entry(boundary, record)
    assert got["accepted"] is True and got["complete"] is True and "why" not in got
    assert got["audit"] == SA.sha512(record["audit"]) and got["of_record"] is False
    whole = {"accepted": True, "complete": True}
    assert SA.boundary_entry(boundary, record, whole) == {**got, "lean": {"tool": SA.ACROSS, **whole}}
    got = SA.boundary_entry(boundary, record, {"accepted": False, "complete": None, "why": "another registration's"})
    assert got["accepted"] is False and got["why"].endswith("audit record: another registration's")
    partial = {**record, "audit": {"accepted": True, "complete": False}}
    unsaid = {**record, "audit": {"accepted": True}}
    for outcome, lean in ((partial, None), (unsaid, None), (partial, {"accepted": True, "complete": False}),
                          (record, {"accepted": True, "complete": False}), (record, {"accepted": True})):
        got = SA.boundary_entry(boundary, outcome, lean)
        assert (got["accepted"], got["complete"], got["why"]) == (None, False, SA.NOT_COMPLETE), (outcome, lean)
    for wrong, why in (({**record, "symbol": S.REJECT}, "did not accept"),
                       ({**record, "audit": {"accepted": False, "complete": True}}, "did not accept"),
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
