"""`stop_audit.py`'s verifier's half on a stop class's hand-off, sampled on a test's own challenger
(`fake_challenger`): the stop stream built again from the hand-off alone, the verifier's own ``--registered`` file,
the audit record's entries and what each refuses, the one word a summary gives for a refusal (`reason`) and
what the negative control changes, and Lean's verdicts on the read across the class's two streams
(`Record.across`) held to the Python's on eighteen written cases, and on a list of reads with the last stream's own
audit record (`Record.acrossAll`), on the stop stream alone and on a chain of three records. No input is random. No
proof runs here: a session's verdicts are stand-ins of the form `a0.lean_verify` reports, and ``stop_audit_small.sh``
is the run that proves. The class is the two-stage driver's tests' (`test_two_stage_driver._stop_class`)."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from verity.circuits.codec import canonical_json
from verity.commitments.periods import rows as CS
from verity.protocols.proofs.spot.verifier import service as S
from verity_vllm.pipeline import boundary_rows as BR, boundary_stop as BS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "tests" / "adapters" / "vllm" / "tests" / "pipeline"))
T = importlib.import_module("test_two_stage_driver")
SA = importlib.import_module("stop_audit")
SMALL = importlib.import_module("stop_audit_small")

DESIGN = f"rate:{T.Q}/{T.N}"
SESSION = "5e" * 64
NULL = {SA.BOUNDARY_AUDIT, SA.REQUEST_ROW, SA.STOP_RULE}


def _resume(handoff: dict, stop: BS.Stop = T.STOP, design: str = DESIGN, **held: object):
    """`stop_audit.resume` under the proof deadline the test's first stage ran under, the verifier holding the test's
    challenger's key and its copy of main's history unless ``held`` gives its own."""
    return SA.resume(handoff, T.PAIR, stop, design, proof_deadline=T.RULE.proof_deadline,
                     **{"key": T.FC.pinned_key(), "main": T.FC.MAIN, **held})


def _handoff(tmp_path: Path, monkeypatch, seed: str, pub: dict | None = None) -> dict:
    """A stop class's hand-off as `two_stage.py first --stop` writes it, on one session of a test's challenger.
    ``pub``: the class's commitment as it is registered, where it is not `test_two_stage_driver._stop_class`'s own."""
    pubs = [pub or json.loads(json.dumps(T._stop_class().public()))]
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


def _audit(handoff: dict, outcome: dict | None = None, held: dict | None = None, **lean: object) -> tuple[dict, dict]:
    """One audit of the hand-off's stop sample on a stream built again (``held``: the key or the copy of main's history
    its verifier holds, where not the test's challenger's): `decide`'s outcome and its summary."""
    r = _resume(handoff, **(held or {}))
    file = SA.reads(r)
    got = SA.decide(r, handoff, file, _proved(r, file, **lean), outcome)
    return got, SA.summary_of(r, T.STOP, DESIGN, got)


def test_the_verifiers_half_is_built_from_the_hand_off_and_the_verifiers_own_copies(tmp_path, monkeypatch):
    """`resume` registers the stop record again on the boundary record its own boundary stream registered, and samples
    the stop blocks the first stage sampled. `reads` is the verifier's ``--registered`` file: the two parameters at the
    boundary record's roots and the output row at the stop record's, unit i at row i. Neither opens the class's
    directory nor calls the prover's half. Refused: another stop rule, another sampling design, a sample that is not
    the hand-off's, a hand-off with no stop sample, and a stop record on a boundary record the boundary stream never
    logged (case 2). The stop sample's family and the session's root are `resume`'s to name, from the class's layout
    and its record's window: a hand-off that names the boundary stream's family for the stop sample, or the stop
    sample's for the boundary stream's, or another session root, is refused, and `link` hands Lean the layout's
    whatever the hand-off names."""
    handoff = _handoff(tmp_path, monkeypatch, "resume")

    def prover(*_a: object, **_k: object):
        raise AssertionError("the verifier's half called the prover's")
    for owner, name in ((BR, "reopen"), (SA, "opened"), (SA, "staged")):
        monkeypatch.setattr(owner, name, prover)
    r = _resume(handoff)
    s = handoff["stop"]
    assert r.docket.digest == s["registration"] and list(r.sample.units) == s["units"]
    assert r.first.docket.digest == handoff["registration"] == s["record"]["previous"]
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

    spans, blocks = T.TS.family_of(r.layout), T.TS.family_of(r.layout, blocks=True)
    assert (handoff["family"], s["family"], handoff["window_root"]) == (spans, blocks, r.layout.run)
    swapped = {**handoff, "stop": {**s, "family": spans}}
    for named in (swapped, {**handoff, "family": blocks}):
        with pytest.raises(ValueError, match="does not choose among the families the session reserved"):
            _resume(named)
    elsewhere = {**handoff, "window_root": "ab" * 64}
    with pytest.raises(ValueError, match="the hand-off names the session root abababab"):
        _resume(elsewhere)
    for named in (swapped, elsewhere):
        read = SA.link(named, r.first)
        assert (read["window"], read["family"]) == (r.layout.run, blocks)


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


def test_the_stop_streams_answer_carries_its_audit_records_complete(tmp_path, monkeypatch):
    """`answer_of` is one-stage's answer of the stop stream's outcome. The record that accepts with three entries null
    gives an accept of the stop docket over the stop sample whose public ``complete`` is false, which a reader
    that wants a complete audit refuses (`audit.answer_complete`). The same record saying complete gives one it takes:
    the field is the record's own. A session Lean's verdict refuses gives a reject of that docket with no public
    outputs, naming the failed checks and the two verdicts that did not accept: Lean's, and the registered reads'
    entry, which is Lean's verdict held to the verifier's file."""
    first = _handoff(tmp_path, monkeypatch, "answer")
    r = _resume(first)
    got, _ = _audit(first)
    a = SA.answer_of(r, got)
    assert (a["verdict"], a["reason"], a["protocol"]) == ("accept", None, SA.R.PROTOCOL)
    assert a["commitment"] == first["stop"]["registration"]
    units, others = sorted(first["stop"]["units"]), set(a["public"]) - {"bound", "complete", "per_template"}
    assert len(others) == 2 and "proved" in others and all(a["public"][k] == units for k in others)
    assert a["public"]["complete"] is got["audit"]["complete"] is False and not SA.A.answer_complete(a)
    whole = SA.answer_of(r, {**got, "audit": {**got["audit"], "complete": True}})
    assert whole["public"] == {**a["public"], "complete": True} and SA.A.answer_complete(whole)
    refused, _ = _audit(first, accepted=False, why="the proof does not verify")
    b = SA.answer_of(r, refused)
    assert (b["verdict"], b["public"], b["profile"], b["commitment"]) == ("reject", None, None, a["commitment"])
    assert b["reason"] == f"checks: roots; verdicts: {SA.LEAN}, {SA.READS}" and not SA.A.answer_complete(b)


def test_an_answer_names_what_refused_it_or_that_no_record_was_written():
    """`answer_of` on outcomes written by hand, of a stream that holds only its δ and its docket's digest: a
    record that did not accept for a verdict alone names the verdict; an outcome the stream refused, which has no
    audit record, says so; an outcome that did not accept beside a record that did is a reject."""
    r = NS(stream=NS(delta=0.5), docket=NS(digest="ab" * 64))
    entry = {"accepted": False, "checks": {"proved": {"ok": True}},
             "verdicts": {SA.LEAN: {"accepted": True}, SA.BOUNDARY_AUDIT: {"accepted": False}, SA.STOP_RULE: {"accepted": None}}}
    reject = {"commitment": "ab" * 64, "profile": None, "protocol": SA.R.PROTOCOL, "public": None, "verdict": "reject"}
    assert SA.answer_of(r, {"accepted": False, "audit": entry}) == {**reject, "reason": f"verdicts: {SA.BOUNDARY_AUDIT}"}
    none = {**reject, "reason": "the stop stream refused the outcome"}
    assert SA.answer_of(r, {"accepted": False, "refused": {"codes": ["sessions"]}}) == none
    assert SA.answer_of(r, {"accepted": False, "audit": {"accepted": True}}) == none
    assert not any(SA.A.answer_complete(SA.answer_of(r, g)) for g in ({"accepted": False, "audit": entry}, {"accepted": False}))


def test_a_record_is_refused_by_each_entry_that_ran_and_did_not_accept(tmp_path, monkeypatch):
    """One hand-off, audited on a stream built again each time. Refused: a session Lean's verdict refuses (a length
    that is not the rule's); a boundary stream's outcome that did not accept, that is another docket's, or
    whose audit record Lean's verdict refuses (case 3); a verdict that names another ``--registered`` file than the
    verifier's, or reports another root for a port; and an order the challenger's signed log does not show, here a
    log with no coin after the stop docket (case 2), and the hand-off's own evidence before a verifier that holds
    another challenger's key, no key, or another copy of main's history (case 2, and the stream's own run refuses the
    coins). With a boundary outcome that accepted and is complete the
    entry accepts and names the audit record (case 1), and the record is still not complete; the record handed is
    one made by hand, which no stream wrote, and the two entries that read it name that premise: case 1 on a handed
    record is on no entry of record, and the summary holds no length on it. With one that accepted
    with an entry that did not run the entry has no verdict and gives its reason (case 4): the record accepts, and
    the boundary-audit entry is among those that keep it from being complete."""
    handoff = _handoff(tmp_path, monkeypatch, "refusals")
    _, summary = _audit(handoff, accepted=False, why="the proof does not verify")
    assert summary["accepted"] is False and summary["verdicts"][SA.LEAN] is False and summary["verdicts"][SA.ORDER] is True
    assert summary["failed"] == ["roots"]

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

    ours = T.FC.pinned_key()
    for held in ({"key": replace(ours, public=bytes(x ^ 1 for x in ours.public))}, {"key": None}, {"main": T.FC.MAIN[1:]}):
        got, summary = _audit(handoff, held=held)
        assert summary["accepted"] is False and "run" in summary["failed"], (held, summary["failed"])
        assert got["audit"]["verdicts"][SA.ORDER]["accepted"] is False and summary["case"] == 2, held


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
    hand, the two dockets' coins each in a family of its own. Case 2: a stop record on a
    boundary record never logged; an empty log; and, on sessions of the test's challenger that received the two
    dockets by hand, the stop docket's coins released first, no receipt of the boundary docket, a
    coin opened and not read, no evidence, no family named, and the boundary stream's family named for the stop
    sample's coins, which are held to the family named and not to the one their evidence names; and the hand-off's
    own evidence read by a verifier that holds another challenger's key, no key, or another copy of main's history,
    and evidence that names a key, the verifier's own. The family is the one `link` names, the stop blocks' place's;
    a case that names its own gives it to Lean and to the Python by hand, since `link` takes none from a hand-off.
    Case 3: the boundary
    stream's audit record missing, refused, and another docket's. Case 4: that audit record accepted with an
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
    spans, blocks = (T.AC.family(rank) for rank in T._stop_places()[:2])
    assert s["family"] == blocks != handoff["family"] == spans
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
        "another key": (2, {}, None, accepted, "the later sample's coins: "),
        "no key": (2, {}, None, accepted, "holds no challenger key"),
        "another main": (2, {}, None, accepted, T.CC.COINS_MAIN),
        "evidence naming a key": (2, {"evidence": {**s["evidence"], "key": T.FC.pinned_key().to_json()}}, None, accepted,
                                  T.CC.DRAW_FORM),
        "no audit record": (3, {}, None, None, None),
        "a refused audit record": (3, {}, None, refused, None),
        "another registration's audit record": (3, {}, None, others, None),
        "an audit record that accepted and is not complete": (4, {}, None, SMALL.incomplete(handoff), None),
    }
    ours = T.FC.pinned_key()
    holds = {"another key": (replace(ours, public=bytes(x ^ 1 for x in ours.public)), T.FC.MAIN), "no key": (None, T.FC.MAIN),
             "another main": (ours, T.FC.MAIN[1:])}
    assert len(cases) == 18 and sorted(c for c, *_ in cases.values()) == [1] * 2 + [2] * 12 + [3] * 3 + [4]
    for name, (case, stop, log, outcome, why) in cases.items():
        mine = {**handoff, "stop": {**s, **stop}}
        log = list(r.first.stream.log) if log is None else log
        key, main = holds.get(name, (ours, T.FC.MAIN))
        previous = mine["stop"]["record"]["previous"]
        logged = previous in [SA.R.digest(rec) for rec in log]
        read = SA.link(mine, NS(layout=r.layout, stream=NS(log=log, key=key, main=tuple(main))), outcome)
        assert (read["window"], read["family"]) == (r.layout.run, blocks), name
        family = stop.get("family", blocks)
        order = T.TS.stop_order(key, main, mine["stop"]["evidence"], r.layout.run, family, previous,
                                SA.R.digest(mine["stop"]["record"]), bytes.fromhex(mine["stop"]["read"]), T.AC.lms_verify)
        audit = SA.boundary_audit(previous, outcome)
        lean = SA.V.lean("across", {**read, "family": family})
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


def _python(read: dict, outcome: dict | None) -> tuple[bool, str | None, bool | None, bool | None]:
    """The Python's verdicts on one read across streams (`stop_audit.link`'s shape), ``outcome`` the earlier stream's
    outcome record: whether the earlier stream's log holds the later record's ``previous``, `two_stage.stop_order`'s
    reason on the family the read names, under the test's challenger's key and its copy of main's history, in
    `Record.across`'s names, and `boundary_audit`'s ``accepted`` and ``complete``."""
    previous = read["later"]["previous"]
    order = T.TS.stop_order(T.FC.pinned_key(), T.FC.MAIN, read["coins"], read["window"], read["family"], previous,
                            SA.R.digest(read["later"]), bytes.fromhex(read["read"]), T.AC.lms_verify)
    audit = SA.boundary_audit(previous, outcome)
    return previous in [SA.R.digest(rec) for rec in read["log"]], _generic(order), audit["accepted"], audit["complete"]


def _reaches(boundary: str, pairs: list[tuple[str, str]]) -> list[bool]:
    """Whether each read's earlier record is ``boundary`` or the later record of a read that reaches it."""
    got = {boundary}
    for _ in pairs:
        got |= {later for earlier, later in pairs if earlier in got}
    return [earlier in got for earlier, _ in pairs]


def _ends(pairs: list[tuple[str, str]]) -> list[str]:
    """The later records no read of the list reads, each once, in the list's order."""
    read, out = {earlier for earlier, _ in pairs}, []
    for _, later in pairs:
        if later not in read and later not in out:
            out.append(later)
    return out


def test_lean_and_the_python_agree_on_a_list_of_reads_across_streams(tmp_path, monkeypatch):
    """`Record.acrossAll` (the Lean verifier's piece ``across_all``, as `stop_audit.across_all` calls it) against the
    Python, on the stop stream alone and on a chain of three records. The stop stream alone is the class as it is
    audited today: one read, the stop record's on the boundary stream, with that stream's audit record, accepted and
    complete, and as the last stream's own the audit record `decide` wrote for the stop stream: it accepted, and two
    of its entries, the request row and the stop rule, did not run. The chain adds a third docket chained after
    the stop record, the three received in that order by one session of the test's challenger: its second read is
    the third record's on the stop stream, whose log holds the stop record and whose audit record is `decide`'s, and
    the last stream is then the third record's, its audit record written here, since no stream after the stop
    stream writes one yet. Each read's verdicts are `Record.across`'s on it alone and the Python's (`_python`),
    ``reaches`` is `_reaches`'s, ``last`` is the list's one end (`_ends`), ``audit`` is `boundary_audit`'s reading
    of the last stream's outcome record on that end, and the list's case is 2 when a read does not reach the boundary
    record or its log or order fails, else 3 when an audit record, a read's or the last stream's, is missing or no
    accept, else 4 when one accepted and is not complete, else 1.

    The stop stream alone. Case 4: as it is, since the stop stream's own audit record is not complete. Case 1: that
    audit record as it would be had every entry run. Case 3: no audit record of the stop stream, one that did not
    accept, and the boundary stream's handed in its place.

    The chain. Case 4: as it is, in either order, since the stop stream's audit record is not complete; and that one
    complete with the third record's not. Case 1: every audit record complete, in either order. Case 2: the second
    read alone, another boundary record, a read with no coin evidence, a read on an empty log, the third record's
    coins released before the stop record's, and a second read that rests on a record the list does not hold, which
    leaves the list two ends and no one last record. Case 3: the first read's audit record missing, whether or not
    the second's is complete; no audit record of the third record; and the stop stream's handed in its place.
    Refused: an empty list, a boundary that is not hex, a read whose later docket names no previous record, by
    its index, and a list without its ``last``."""
    handoff = _handoff(tmp_path, monkeypatch, "across-all")
    r, s = _resume(handoff), handoff["stop"]
    boundary, stop = handoff["registration"], s["registration"]
    refused = SMALL.unaccepted(handoff)
    accepted = {**refused, "symbol": S.ACCEPT, "audit": {**refused["audit"], "accepted": True}}
    file = SA.reads(r)
    assert SA.decide(r, handoff, file, _proved(r, file), accepted)["accepted"] is True
    stopped = r.stream.records[-1]
    assert stopped["registration"] == stop == stopped["audit"]["registration_digest"] and stopped["symbol"] == S.ACCEPT
    assert stopped["audit"]["accepted"] is True and stopped["audit"]["complete"] is False
    assert {k for k, v in stopped["audit"]["verdicts"].items() if v["accepted"] is None} == {SA.REQUEST_ROW, SA.STOP_RULE}
    whole = {**stopped, "audit": {**stopped["audit"], "complete": True}}
    third = {**s["record"], "previous": stop}
    tie = SA.R.digest(third)
    window = r.layout.run
    # what every read here is read under, as `link` writes them: the test's challenger's key and its copy of main's history
    held = {"key": T.FC.pinned_key().to_json(), "main": list(T.FC.MAIN)}

    def chain(seed: str, *commits: tuple[str, int]) -> tuple[dict, dict]:
        one, got, family = _released(seed, *commits)
        return ({"window": window, "log": [dict(x) for x in r.first.stream.log], "audit": accepted["audit"],
                 "later": dict(s["record"]), "coins": one.evidence(stop), "family": family[stop], "read": got[stop].hex(),
                 **held},
                {"window": window, "log": [dict(x) for x in r.stream.log], "audit": stopped["audit"], "later": third,
                 "coins": one.evidence(tie), "family": family[tie], "read": got[tie].hex(), **held})
    first, second = chain("chain", (boundary, 1), (stop, 2), (tie, 2))
    assert set(first) == set(second) == set(SA.link(handoff, r.first, accepted))
    assert [SA.R.digest(x) for x in second["log"]] == [boundary, stop]
    _, late = chain("third-first", (boundary, 1), (tie, 1), (stop, 2))
    a, b, c = (first, accepted), (second, stopped), ({**second, "audit": whole["audit"]}, whole)
    unaudited = ({**first, "audit": None}, None)
    # the third record's stream's outcome record: no stream after the stop stream writes one yet
    tied = {**whole, "registration": tie, "audit": {**whole["audit"], "registration_digest": tie}}
    untied = {**tied, "audit": {**tied["audit"], "complete": False}}
    refused_stop = {**stopped, "symbol": S.REJECT, "audit": {**stopped["audit"], "accepted": False}}
    stray = {**second, "later": {**third, "previous": "5a" * 64}}
    cases = {
        "the stop stream alone": (4, boundary, [a], stopped),
        "the stop stream alone, its audit record complete": (1, boundary, [a], whole),
        "the stop stream alone, no audit record of its own": (3, boundary, [a], None),
        "the stop stream alone, its audit record no accept": (3, boundary, [a], refused_stop),
        "the stop stream alone, the boundary stream's audit record in its place": (3, boundary, [a], accepted),
        "the chain": (4, boundary, [a, b], tied),
        "the chain, its second read first": (4, boundary, [b, a], tied),
        "the chain, the stop stream's audit record complete": (1, boundary, [a, c], tied),
        "the chain, the stop stream's audit record complete, its second read first": (1, boundary, [c, a], tied),
        "the chain, the third record's audit record not complete": (4, boundary, [a, c], untied),
        "the second read alone": (2, boundary, [b], tied),
        "another boundary record": (2, "5a" * 64, [a, c], tied),
        "a read with no coin evidence": (2, boundary, [a, ({**second, "coins": None}, stopped)], tied),
        "a read on an empty log": (2, boundary, [({**first, "log": []}, accepted), c], tied),
        "the third record's coins first": (2, boundary, [a, (late, stopped)], tied),
        "a second read on a record the list does not hold": (2, boundary, [a, (stray, stopped)], whole),
        "no audit record of the boundary stream": (3, boundary, [unaudited, c], tied),
        "no audit record of the boundary stream, and the stop stream's not complete": (3, boundary, [unaudited, b], tied),
        "the chain, no audit record of the third record": (3, boundary, [a, c], None),
        "the chain, the stop stream's audit record in the third record's place": (3, boundary, [a, c], whole),
    }
    assert sorted(c for c, *_ in cases.values()) == [1] * 3 + [2] * 6 + [3] * 7 + [4] * 4
    for name, (case, anchor, reads, last) in cases.items():
        lean = SA.across_all(anchor, [x for x, _ in reads], None if last is None else last["audit"])
        assert "refused" not in lean and lean["boundary"] == anchor and len(lean["links"]) == len(reads), (name, lean)
        mine = [_python(x, outcome) for x, outcome in reads]
        pairs = [(x["later"]["previous"], SA.R.digest(x["later"])) for x, _ in reads]
        reaches, ends = _reaches(anchor, pairs), _ends(pairs)
        final = SA.boundary_audit(ends[0], last) if len(ends) == 1 else {"accepted": False, "complete": None}
        assert lean["last"] == (ends[0] if len(ends) == 1 else None), (name, lean)
        assert (lean["audit"]["accepted"], lean["audit"]["complete"]) == (final["accepted"], final["complete"]), (name, lean)
        assert ("why" in lean["audit"]) == (final["accepted"] is False or (final["accepted"] is True and final["complete"] is not True)), (name, lean)
        for (x, _), got, (logged, order, audit, complete), reach in zip(reads, lean["links"], mine, reaches):
            assert got["logged"]["accepted"] is logged, (name, got)
            assert (got["audit"]["accepted"], got["audit"]["complete"]) == (audit, complete), (name, got)
            assert got["order"] == ({"accepted": True} if order is None else {"accepted": False, "why": order}), (name, got)
            assert got["reaches"] is reach and (got["earlier"], got["later"]) == (x["later"]["previous"], SA.R.digest(x["later"])), (name, got)
            if name == "the chain":
                assert {k: v for k, v in got.items() if k != "reaches"} == SA.V.lean("across", x), (name, got)
        want = (2 if any(not reach or not logged or order is not None for (logged, order, _, _), reach in zip(mine, reaches))
                else 3 if final["accepted"] is not True or any(audit is not True for _, _, audit, _ in mine)
                else 4 if final["complete"] is not True or any(complete is not True for _, _, _, complete in mine) else 1)
        assert lean["case"] == case == want, (name, lean)
    alone = SA.across_all(boundary, [first], stopped["audit"])
    assert (alone["last"], alone["links"][0]["case"], alone["case"]) == (stop, 1, 4)
    assert alone["audit"] == {"accepted": True, "complete": False, "why": alone["audit"]["why"]} and "not complete" in alone["audit"]["why"]
    got = SA.across_all(boundary, [first, second], tied["audit"])
    assert [x["received"] for x in got["links"]] == [0, 1] and [x["case"] for x in got["links"]] == [1, 4]
    assert "not complete" in got["links"][1]["audit"]["why"] and "why" not in got["links"][0]["audit"]
    assert got["last"] == tie and got["audit"] == {"accepted": True, "complete": True}
    forked = SA.across_all(boundary, [first, stray], whole["audit"])
    assert forked["last"] is None and forked["audit"]["accepted"] is False and "2 records" in forked["audit"]["why"]

    assert "no read" in SA.across_all(boundary, [], None)["refused"]
    assert "boundary" in SA.across_all("not hex", [first], None)["refused"]
    unnamed = {**second, "later": {k: v for k, v in third.items() if k != "previous"}}
    assert "link 1: the later registration names no previous record" in SA.across_all(boundary, [first, unnamed], None)["refused"]
    assert "link 0: exactly" in SA.across_all(boundary, [{**first, "more": 1}], None)["refused"]
    assert "exactly {boundary, links, last}" in SA.V.lean("across_all", {"boundary": boundary, "links": [first]})["refused"]


def test_the_boundary_audit_entry_reads_the_boundary_streams_outcome_record():
    """`boundary_entry` alone: null without a record; an outcome record on the class's boundary docket whose
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


#: Lean's verdict of record as `a0.lean_verify` reports it of a proof its inner check's batched constraint refused
UNSATISFIED = {"accepted": False, "of_record": True, "digest": "dd" * 64, "why": f"{SA.UNSATISFIED} (⟨c, Y⟩ ≠ t + β τ)"}
#: Lean's verdicts of record that are no refusal of the stop relation, each as `a0.lean_verify` reports it
OTHERS = {
    "an accept": {**UNSATISFIED, "accepted": True, "why": None},
    "a refusal at setup": {**UNSATISFIED, "why": "setup: the registered file reads a port under another root"},
    "a file of the session that was not there": {**UNSATISFIED, "why": f"{SA.a0.NO_SESSION_FILES} no such file"},
    "another of the inner check's refusals": {**UNSATISFIED, "why": "zk inner: τ does not open its commitment"},
    "a refusal at setup that quotes the constraint": {**UNSATISFIED, "why": f"setup: {SA.UNSATISFIED}"},
    "no verdict line of Lean's": {**UNSATISFIED, "digest": None, "why": "no verdict (rc 1): "},
    "no reason": {**UNSATISFIED, "why": None},
    "an entry that is not of record": {**UNSATISFIED, "of_record": False},
}


def test_the_reason_is_a_function_of_leans_verdict_of_record_alone():
    """`reason` is the stop word's only for Lean's own verdict of record that refused the proof at its inner check's
    batched constraint. Every other verdict of Lean's gets none: an accept, a refusal at setup, for a file of the
    session that was not there, at another check, a stand-in for a run that printed no verdict, and an entry that is
    not of record. It reads Lean's entry and no other: the same word whatever the other entries hold, and none from
    a record without the entry or with that verdict under another entry's name."""
    assert SA.STOP_WORD == "stop_word" and SA.reason({SA.LEAN: UNSATISFIED}) == SA.STOP_WORD
    for name, verdict in OTHERS.items():
        assert SA.reason({SA.LEAN: verdict}) is None, name
    refused = {"accepted": False, "of_record": False, "why": "did not accept"}
    assert SA.reason({SA.LEAN: UNSATISFIED, SA.M0: refused, SA.BOUNDARY_AUDIT: refused, SA.ORDER: refused}) == SA.STOP_WORD
    assert SA.reason({SA.LEAN: OTHERS["an accept"], SA.M0: UNSATISFIED, SA.BOUNDARY_AUDIT: refused}) is None
    assert SA.reason({}) is None and SA.reason({SA.M0: UNSATISFIED}) is None


def test_a_summary_states_the_stop_word_only_where_lean_refused_the_stop_relation(tmp_path, monkeypatch):
    """One hand-off, audited on a stream built again each time. The summary's ``reason`` is what `reason` gives of the
    audit record's entry of record: the stop word where Lean's verdict refused the proof at the batched constraint,
    and still that where the boundary stream's outcome did not accept either. No reason: an accepted record; a record
    Lean's verdict refused at setup, or for a file of the session that was not there; and a record only another entry
    refused, the boundary stream's outcome or the ``--registered`` file's name. A command that stopped before a
    verdict writes a summary whose reason is null."""
    hand = _handoff(tmp_path, monkeypatch, "reason")
    told = {k: UNSATISFIED[k] for k in ("accepted", "why")}
    got, summary = _audit(hand, **told)
    entry = got["audit"]["verdicts"][SA.LEAN]
    assert (entry["of_record"], entry["digest"], entry["why"]) == (True, UNSATISFIED["digest"], UNSATISFIED["why"])
    assert summary["reason"] == SA.reason({SA.LEAN: entry}) == SA.STOP_WORD
    assert summary["accepted"] is False and summary["failed"] == ["roots"]
    _, summary = _audit(hand, SMALL.unaccepted(hand), **told)
    assert summary["reason"] == SA.STOP_WORD and summary["verdicts"][SA.BOUNDARY_AUDIT] is False

    _, summary = _audit(hand)
    assert summary["accepted"] is True and summary["reason"] is None
    for name in ("a refusal at setup", "a file of the session that was not there"):
        _, summary = _audit(hand, accepted=False, why=OTHERS[name]["why"])
        assert summary["accepted"] is False and summary["verdicts"][SA.LEAN] is False and summary["reason"] is None, name
    for outcome, lean in ((SMALL.unaccepted(hand), {}), (None, {"registered_sha512": "00" * 64})):
        _, summary = _audit(hand, outcome, **lean)
        assert summary["accepted"] is False and summary["verdicts"][SA.LEAN] is True and summary["reason"] is None

    unsampled = tmp_path / "unsampled.json"
    unsampled.write_text(json.dumps({**hand, "stop": {"error": "refused"}}))
    out = tmp_path / "stopped"
    monkeypatch.setattr(T.TS, "template_fn", lambda _id: T.PAIR)
    k = T.FC.pinned_key()
    (key := tmp_path / "challenger-key.json").write_text(json.dumps(
        {"schema": "challenger-key/v1", "W": k.W, "H": k.H, "pubkey": k.public.hex(),
         "pubkey_sha512": hashlib.sha512(k.public).hexdigest()}))
    (main := tmp_path / "main.txt").write_text("".join(f"{x}\n" for x in T.FC.MAIN))
    assert SA.main(["--handoff", str(unsampled), "--boundary", str(tmp_path), "--stop", json.dumps(T.STOP.to_json()),
                    "--design", DESIGN, "--main", str(main), "--challenger-key", str(key), "--bin", "none",
                    "--lean", "none", "--out", str(out)]) == 1
    stopped = json.loads((out / "summary.json").read_text())
    assert stopped["accepted"] is False and stopped["reason"] is None and "verdicts" not in stopped
    assert "registered or sampled no stop block: refused" in stopped["error"], stopped


def test_the_control_is_the_accepted_class_with_its_stop_word_changed_and_nothing_else(tmp_path, monkeypatch):
    """The negative control changes the stop word alone, from the accepted class. Its prover (`opened` with spans to
    forge) opens the two parameters' rows, salts, paths and roots as the class's own, and the stop row with bit 0 of
    its first word flipped at the forged spans, every other bit and its salts the class's. The commitment a first
    stage registers for it (`stop_audit_small.wrong_length`) gives the accepted class's boundary record, and a stop
    record that differs from the accepted class's in `b.stop`'s root alone, the forged tree's. So `opens`, which the
    command holds before any session, finds every port under the root its record registers for the class and for the
    control, the stop row's alone off where one's prover meets the other's record, and names a parameter whose tree a
    prover changed."""
    c = T._stop_class()
    honest, forged = SA.opened(c), SA.opened(c, range(T.N))
    for p in BS.STOP_READS:
        assert (honest["rows"][p] == forged["rows"][p]).all() and honest["salts"][p] == forged["salts"][p], p
        assert honest["paths"][p] == forged["paths"][p], p
    assert {v for v in honest["roots"] if honest["roots"][v] != forged["roots"][v]} == {BR.STOP}
    changed = honest["rows"][SA.OUT] ^ forged["rows"][SA.OUT]
    assert changed[:, 0].tolist() == [1] * T.N and not changed[:, 1:].any()
    assert honest["salts"][SA.OUT] == forged["salts"][SA.OUT]
    one = honest["rows"][SA.OUT] ^ SA.opened(c, [1])["rows"][SA.OUT]
    assert one[:, 0].tolist() == [0, 1] + [0] * (T.N - 2) and not one[:, 1:].any()

    pub = json.loads(json.dumps(c.public()))
    for name in ("class", "control"):
        (tmp_path / name).mkdir()
    mine = _handoff(tmp_path / "class", monkeypatch, "class", pub)
    its = _handoff(tmp_path / "control", monkeypatch, "control", SMALL.wrong_length(c))
    assert mine["record"] == its["record"] and mine["registration"] == its["registration"]
    stop, other = mine["stop"]["record"], its["stop"]["record"]
    assert {k for k in stop if stop[k] != other[k]} == {"roots"}
    assert stop["previous"] == other["previous"] == mine["registration"]
    (x,), (y,) = stop["roots"], other["roots"]
    assert {k for k in x if x[k] != y[k]} == {"root"} and x["port"] == y["port"] == BR.STOP
    assert (x["root"], y["root"]) == (honest["roots"][BR.STOP], forged["roots"][BR.STOP])

    held, wrong = SA.reads(_resume(mine)), SA.reads(_resume(its))
    assert SA.opens(honest["roots"], held) == [] == SA.opens(forged["roots"], wrong)
    assert SA.opens(forged["roots"], held) == [SA.OUT] == SA.opens(honest["roots"], wrong)
    for p, v in BS.STOP_READS.items():
        assert SA.opens({**forged["roots"], v: "00" * 64}, wrong) == [p]


def test_the_rehearsal_expects_the_stop_word_of_its_control_and_of_no_other_run():
    """`stop_audit_small.expect` on five summaries: ``length``, the control's, holds only with the reason
    ``stop_word``, and ``no_other_reason`` only where each of the four other runs' summaries states a null reason."""
    runs = {name: {"reason": None} for name in SMALL.RUNS}
    runs["length"] = {"accepted": False, "verdicts": {SA.LEAN: False, SA.ORDER: True}, "length_held": False,
                      "reason": SA.STOP_WORD}
    got = SMALL.expect(runs)
    assert got["length"] is True and got["no_other_reason"] is True
    assert SMALL.expect({**runs, "length": {**runs["length"], "reason": None}})["length"] is False
    for name in set(SMALL.RUNS) - {"length"}:
        assert SMALL.expect({**runs, name: {"reason": SA.STOP_WORD}})["no_other_reason"] is False, name
        assert SMALL.expect({**runs, name: {}})["no_other_reason"] is False, name
