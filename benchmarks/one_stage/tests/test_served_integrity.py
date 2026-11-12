"""Sampled proofs' answer on a served window from what the records audit wrote (`served_integrity.answer`):

* an accepted audit record is sampled proofs' accept: the registration's digest, the drawn and proved units with the bound
  on wrong units at δ, and the profile as `Core.Integrity` with δ an exact ratio, all read back by the core answer's
  decoder; the report's ε is that bound over the window's units;
* a refusal of the records, a check before the sample that failed, one after it that left no audit record (a drawn
  member whose --registered file the audit couldn't complete, by name, or the tie failing once the sessions were done,
  ``record``), an audit record that doesn't accept and advice fixed after the sample each reject by name, with no public
  outputs or profile;
* no answer accepts beside an audit that exited nonzero: an accepting audit record under a summary that doesn't say the
  audit passed (``ok``, its exit) is a reject saying so, on the audit and on its public copy;
* an audit that left nothing behind is an error, not a verdict, and so is one that opened the advice (``--opening``, a
  development check), accepted or not;
* an audit record decided at δ = 0.01 (a float in the record) is the exact ratio 1/100 in the profile;
* the public copy of an audit (`public_copy`, what a verifier on another machine receives) is what `answer` and the
  onsite record read of it alone, of the public kinds (`kind_scan.V3C_DOCS`), and answers as the audit does.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.sampled_proofs.one_stage import registration as R

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import served_integrity as I  # noqa: E402
from served_verdicts import check_answer, epsilon  # noqa: E402

REG = {"format": R.FORMAT, "protocol": R.PROTOCOL, "program": "aa" * 64, "partition": "bb" * 64, "population": 1024,
       "law": {"law": "subset", "k": 2}}
DRAW = {"law": "subset", "k": 2, "population": 1024, "units": [700, 5]}


def write(out: Path, summary: dict, record: dict | None = None) -> Path:
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(summary))
    if record is not None:
        (out / "window").mkdir()
        (out / "window" / "audit.json").write_text(json.dumps(record))
    return out


def record(accepted: bool, **checks) -> dict:
    return {"accepted": accepted, "registration": REG, "draw": DRAW, "proved_units": [700, 5],
            "checks": {"draw": {"ok": True}, **{k: {"ok": v} for k, v in checks.items()}},
            "wrong_units": {"delta": 0.5, "of": 1024, "bound": 3} if accepted else None}


#: a drawn member, and the audit's checks after the draw as a5 leaves them: its --registered file completed before any
#: session (`a5.complete_drawn`), with the ports it named, and the tie held once the sessions were done
#: (`a5.tie_after_sessions`); or the file refused for a port with no committed tree, before any session
MEMBER = "ToyGemm"
TIED = {"ok": True, "codes": [], "detail": {}, "record": "auditor"}
COMPLETED = {"wiring": {"ok": True, "codes": [], "detail": {}},
             "registered_complete": {"ok": True, "codes": [], "detail": {},
                                     "members": {MEMBER: {"window_ports": 2, "parameter_ports": 1}}},
             "record": TIED}
UNCOMPLETED = {"wiring": {"ok": True, "codes": [], "detail": {}},
               "registered_complete": {"ok": False, "codes": [MEMBER], "members": {},
                                       "detail": {MEMBER: "port o0 has no committed tree"}}}
#: the member's Lean verdict of record's reads beside its completed file's (`tables.reported_reads`)
READS = {"ok": True, "chunks": {MEMBER: {"ok": True, "reads": {"o0": "ab" * 64}, "reported": {"o0": "ab" * 64}}}}
#: the audit's registered reads (`records_audit.audit`): the pins' roots, the window's and the members with a file,
#: beside the entry they opened the window's own scan state at (``own``)
STATE = "verity/certifier/scan-state/v1"
OWN = {STATE: {"value": STATE, "root": "ab" * 64, "domain": "cd" * 64, "leaves": 64, "words": 8}}
REGISTERED = {"pins": {STATE: "ab" * 64, "weights/w@8": "ef" * 64}, "roots": {STATE: "ab" * 64}, "members": 1,
              "own": OWN}
AFTER_DRAW = {"checks_before_draw": {"registration": True}, "checks": {"registration": {"ok": True}, **UNCOMPLETED}}
#: an audit whose tie failed once its sessions were done (`a5.refuse_after_sessions`), which wrote no audit record
UNTIED = {"checks_before_draw": {"registration": True},
          "checks": {"registration": {"ok": True}, **COMPLETED,
                     "record": {**TIED, "ok": False, "codes": ["record"],
                                "detail": {"record": "the challenger's session gave no evidence"}}}}


#: the summary of an audit that passed (`records_audit.audit`'s ``ok``, what its exit says) and fixed its inputs before
#: the draw
PASSED = {"ok": True, "checks_before_draw": {"registration": True}, "fixed_before_draw": {"ok": True}}


def test_an_accepted_audit_is_an_accept(tmp_path):
    out = write(tmp_path / "a", PASSED, record(True))
    a = I.answer(out, REG)
    assert check_answer(a) == a
    assert a["verdict"] == "accept" and a["protocol"] == "verity/sampled-proofs/one-stage/v1"
    assert a["commitment"] == R.digest(REG)
    assert a["public"] == {"bound": 3, "drawn": [5, 700], "per_template": None, "proved": [5, 700]}
    p = CC.integrity(a["profile"])
    assert str(p["delta"]) == "1/2" and p["drawn"] == [("unit", 2)]
    assert p["levels"] == [{"name": "unit", "query": "bb" * 64, "size": 1024, "draw": {"subset": 2}}]
    assert epsilon(a) == "3/1024"


@pytest.mark.parametrize("summary,rec,why", [
    ({"refused": {"records": "ValueError: the window has 2 row tables"}}, None, "refused: "),
    ({"checks_before_draw": {"registration": False, "m0": True}}, None, "before the draw: registration"),
    ({"checks_before_draw": {"registration": True}}, record(False, proved=False), "audit record: proved"),
    ({"checks_before_draw": {"registration": True}, "audit": {"accepted": False, "outcome": "coins refused"}}, None,
     "no audit record: "),
    ({"fixed_before_draw": {"ok": False, "codes": ["after-the-draw"]}}, record(True), "not fixed before the draw: after"),
    (AFTER_DRAW, None, f"after the draw: registered_complete ({MEMBER})"),
    (UNTIED, None, "after the draw: record (record)"),
])
def test_refusals_reject_by_name(tmp_path, summary, rec, why):
    a = I.answer(write(tmp_path / "a", summary, rec), REG)
    assert check_answer(a) == a
    assert a["verdict"] == "reject" and a["reason"].startswith(why)
    assert a["commitment"] == R.digest(REG)


@pytest.mark.parametrize("ok", [False, None, "absent", 1, "true"])
def test_no_answer_accepts_beside_an_audit_that_exited_nonzero(tmp_path, ok):
    """An audit that exits nonzero after it wrote an accepting audit record (a negative that wasn't refused, its coins'
    session's error: `records_audit.audit` writes ``ok`` false and exits 1) leaves no accepted answer: every check
    `answer` names passed and the record accepts, and the answer is a reject saying the audit exited nonzero, with no
    public outputs or profile, on the audit and on its public copy alike. So is a summary that doesn't say it passed
    (no ``ok``, or one that isn't true); the same audit with ``ok`` true is the accept."""
    summary = {k: v for k, v in {**PASSED, "ok": ok, "negatives": {"coins-no-evidence": False}}.items() if v != "absent"}
    out = write(tmp_path / "a", summary, record(True))
    I.public_copy(out, tmp_path / "copy")
    for d in (out, tmp_path / "copy"):
        a = I.answer(d, REG)
        assert check_answer(a) == a
        assert a == {"commitment": R.digest(REG), "profile": None, "protocol": I.PROTOCOL, "public": None,
                     "reason": I.EXITED_NONZERO, "verdict": "reject"}
    assert "exited nonzero" in I.EXITED_NONZERO
    assert I.answer(write(tmp_path / "b", {**summary, "ok": True}, record(True)), REG)["verdict"] == "accept"


def test_a_refused_audit_record_names_leans_refusals(tmp_path):
    why = {f"lean-verify/m{k}": f"registered reads: port o0: table row 0 does not open out's registered root at "
                                f"position {k}" for k in range(5)}
    summary = {"checks_before_draw": {"registration": True}, "audit": {"why": {**why, "draw": None}}}
    a = I.answer(write(tmp_path / "a", summary, record(False, proved=False)), REG)
    assert a["verdict"] == "reject" and a["reason"].startswith("audit record: proved; lean-verify/m0: registered reads")
    assert "position 2" in a["reason"] and "position 3" not in a["reason"] and a["reason"].endswith("and 2 more")


def test_a_tampered_wired_window_names_the_drawn_units_its_sessions_refused_before_leans_missing_files(tmp_path):
    """A drawn member whose live session the verifier of record rejects leaves no proof files, so Lean's replay finds
    none: the reason names the sampled proof blocks not proved, by template, and the verifier of record's rejection first."""
    mm, quant = "ScaledMmS8_v1{K=128,N=128}", "QuantRowS8_v1{N=128,RELU=true}"
    rec = record(False, proved=False)
    rec["checks"]["proved"]["detail"] = {"units": {"drawn": list(range(12)), "statement": list(range(8, 12))}}
    rec["draw"] = {**DRAW, "strata": [{"template": mm, "units": [[0, 8]]}, {"template": quant, "units": [[8, 12]]}]}
    why = {**{f"lean-verify/m{k}": "session files: no such file or directory" for k in range(4)},
           **{f"m0-verifier/m{k}": "R7: no proof for circuit/rep0 (lone rep)" for k in range(4)}}
    a = I.answer(write(tmp_path / "a", {"checks_before_draw": {"registration": True}, "audit": {"why": why}}, rec), REG)
    assert a["verdict"] == "reject"
    assert a["reason"].startswith(f"audit record: proved (drawn units not proved: {mm} 0-7); m0-verifier/m0: R7: ")
    assert "session files" not in a["reason"] and quant not in a["reason"] and a["reason"].endswith("and 5 more")


def test_an_audit_that_left_nothing_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="no summary"):
        I.answer(tmp_path, REG)
    with pytest.raises(RuntimeError, match="no audit record and no refusal"):
        I.answer(write(tmp_path / "a", {"checks_before_draw": {"registration": True}}), REG)


def receipt(out: Path) -> Path:
    """``out`` with a5's receipt, as an attempt leaves it once its checks before the draw are made, before it draws."""
    (out / "window").mkdir(parents=True, exist_ok=True)
    (out / "window" / "receipt.json").write_text(json.dumps({"registration": R.digest(REG), "verifier": "a5"}))
    return out


def test_an_attempt_that_reached_its_draw_and_died_is_a_reject_never_an_error_or_an_accept(tmp_path):
    """An audit's directory that holds a5's receipt is one an attempt reached its draw under. One that then died
    leaves no summary, or a summary with neither an audit record nor a refusal; each was an error, so nothing said
    the directory is spent. Each is a reject that says so, on the registration, with no public outputs or profile.
    So is an accepting audit record whose audit died before its summary, with a reason of its own: the record alone,
    with or without the receipt, says an attempt drew; a record cut short or one that rejects reads as no summary.
    The same directories without receipt or record are no audit's, and raise."""
    passed = {"checks_before_draw": {"registration": True}}
    died = {"no summary": (receipt(tmp_path / "a"), I.NO_SUMMARY),
            "no record": (receipt(write(tmp_path / "b", passed)), I.NO_RECORD),
            "a record, no summary": (receipt(tmp_path / "c"), I.ACCEPTING_NO_SUMMARY),
            "a record alone": (tmp_path / "d", I.ACCEPTING_NO_SUMMARY),
            "a record cut short": (tmp_path / "e", I.NO_SUMMARY),
            "a rejecting record, no summary": (tmp_path / "f", I.NO_SUMMARY)}
    for d, body in ((tmp_path / "c", json.dumps(record(True))), (tmp_path / "d", json.dumps(record(True))),
                    (tmp_path / "e", json.dumps(record(True))[:40]), (tmp_path / "f", json.dumps(record(False)))):
        (d / "window").mkdir(parents=True, exist_ok=True)
        (d / "window" / "audit.json").write_text(body)
    for out, reason in died.values():
        a = I.answer(out, REG)
        assert check_answer(a) == a
        assert a == {"commitment": R.digest(REG), "profile": None, "protocol": I.PROTOCOL, "public": None,
                     "reason": reason, "verdict": "reject"}
    assert I.attempt(tmp_path / "a") == tmp_path / "a" / "window" / "receipt.json"
    assert I.attempt(tmp_path / "d") == tmp_path / "d" / "window" / "audit.json" and I.attempt(tmp_path / "none") is None
    assert I.ATTEMPT == ("window/receipt.json", "window/audit.json")
    assert len({I.NO_SUMMARY, I.ACCEPTING_NO_SUMMARY, I.NO_RECORD}) == 3
    assert "written before the draw" in I.NO_SUMMARY and "written before the draw" in I.NO_RECORD


def test_only_the_reject_of_an_audit_that_did_not_finish_says_so(tmp_path):
    """A reader tells an audit that did not finish from a refused proof by the reason: each reject of an attempt that
    died starts with `UNFINISHED`, and no other reject does: a refusal before or after the draw, a rejecting audit
    record, or an audit that finished and exited nonzero beside its accepting record."""
    assert I.UNFINISHED == "the audit did not finish"
    assert all(r.startswith(I.UNFINISHED + ": ") for r in (I.NO_SUMMARY, I.ACCEPTING_NO_SUMMARY, I.NO_RECORD))
    finished = [write(tmp_path / "a", {"refused": {"records": "ValueError: no such value"}}),
                write(tmp_path / "b", {"checks_before_draw": {"registration": False}}),
                write(tmp_path / "c", {"checks_before_draw": {"registration": True},
                                       "checks": {"registration": {"ok": True}, "wiring": {"ok": False, "codes": ["wiring"]}}}),
                write(tmp_path / "d", PASSED, record(False)),
                write(tmp_path / "e", PASSED | {"ok": False}, record(True))]
    for out in finished:
        a = I.answer(receipt(out), REG)
        assert a["verdict"] == "reject" and I.UNFINISHED not in a["reason"]
    assert I.answer(receipt(finished[-1]), REG)["reason"] == I.EXITED_NONZERO


@pytest.mark.parametrize("summary, reason", [
    ({"refused": {"records": "ValueError: no such value"}}, 'refused: {"records": "ValueError: no such value"}'),
    ({"checks_before_draw": {"registration": False}}, "before the draw: registration"),
    ({"checks_before_draw": {"registration": True},
      "checks": {"registration": {"ok": True}, "wiring": {"ok": False, "codes": ["wiring"]}}}, "after the draw: wiring (wiring)"),
])
def test_an_attempts_own_refusal_stays_its_answer_beside_its_receipt(tmp_path, summary, reason):
    """A refusal the audit wrote is the answer, with the receipt or without it: the receipt adds a reject only where
    the audit left none of its own."""
    assert I.answer(receipt(write(tmp_path / "a", summary)), REG)["reason"] == reason
    assert I.answer(write(tmp_path / "b", summary), REG)["reason"] == reason


def test_an_audit_that_opened_the_advice_is_no_answer(tmp_path):
    """The records audit's compile check ran (``--opening``): accepted or refused, `answer` makes no verdict of it."""
    before = PASSED
    for name, compiled in (("ok", {"ok": True}), ("refused", {"ok": False})):
        out = write(tmp_path / name, before | {"compile": compiled}, record(True))
        assert I.opened(json.loads((out / "summary.json").read_text()))
        with pytest.raises(RuntimeError, match=r"opened the advice \(records_audit.py audit --opening\)"):
            I.answer(out, REG)
    assert not I.opened(before | {"compile": None}) and I.answer(write(tmp_path / "c", before | {"compile": None},
                                                                       record(True)), REG)["verdict"] == "accept"


def test_an_audit_at_one_percent_is_the_exact_ratio(tmp_path):
    rec = {**record(True), "wrong_units": {"delta": 0.01, "of": 1024, "bound": 69}}
    a = I.answer(write(tmp_path / "a", PASSED, rec), REG)
    assert a["profile"]["delta"] == "1/100" and str(CC.integrity(a["profile"])["delta"]) == "1/100"


def test_an_audit_at_2_to_the_minus_64_keeps_it_exact_where_a_nearby_ratio_would_be_0(tmp_path):
    rec = {**record(True), "wrong_units": {"delta": 2.0 ** -64, "of": 1024, "bound": 1023}}
    a = I.answer(write(tmp_path / "a", PASSED, rec), REG)
    assert a["profile"]["delta"] == f"1/{2 ** 64}" and CC.integrity(a["profile"])["delta"] == Fraction(1, 2 ** 64)
    assert Fraction(2.0 ** -64).limit_denominator(10 ** 12) == 0


@pytest.mark.parametrize("given, want", [
    (0.5, Fraction(1, 2)), (0.01, Fraction(1, 100)), (1e-12, Fraction(1, 10 ** 12)), (2.0 ** -64, Fraction(1, 2 ** 64)),
    (5.421010862427522e-20, Fraction(1, 2 ** 64)), (f"1/{2 ** 64}", Fraction(1, 2 ** 64)), ("0.01", Fraction(1, 100)),
    (Fraction(1, 2 ** 128), Fraction(1, 2 ** 128)),
])
def test_exact_delta_is_the_ratio_given_never_a_nearby_one(given, want):
    assert I.exact_delta(given) == want


#: what a records audit writes beside what `answer` reads, each a count or name of the window's own
UNREAD_SUMMARY = {"window": 7, "requests": 3, "population": 37, "compile": None, "audit_log": "/srv/serve/w7/audit.log",
                  "fixed_before_draw": {"detail": {"window": 7}}}
UNREAD_RECORD = {"format": "f", "registration_digest": "cc" * 64, "receipt": {"window": 7}, "statement": {"requests": 3},
                 "verdicts": {"m0": {"window": 7, "accepted": True}}, "complete": True, "profile": {"window": 7},
                 "integrity_profile": {"evidence": {"window": 7}}, "law": {"law": "subset"}}


def _planted(summary: dict, rec: dict | None) -> tuple[dict, dict | None]:
    s = {**UNREAD_SUMMARY, **summary, "fixed_before_draw": {**UNREAD_SUMMARY["fixed_before_draw"],
                                                            **summary.get("fixed_before_draw", {})}}
    if rec is None:
        return s, None
    checks = {k: {**v, "detail": v.get("detail") or {"window": 7}} for k, v in rec["checks"].items()}
    return s, {**UNREAD_RECORD, **rec, "checks": checks}


def _kind_scan():
    """``kind_scan`` by its file: network_traces' dir on the path would shadow this suite's modules of its names."""
    if "kind_scan" not in sys.modules:
        spec = importlib.util.spec_from_file_location("kind_scan", HERE.parents[1] / "network_traces" / "kind_scan.py")
        sys.modules["kind_scan"] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules["kind_scan"])
    return sys.modules["kind_scan"]


@pytest.mark.parametrize("summary,rec", [
    ({"ok": True, "checks_before_draw": {"registration": True}, "fixed_before_draw": {"ok": True}}, "accepted"),
    ({"ok": False, "checks_before_draw": {"registration": True}, "fixed_before_draw": {"ok": True}}, "accepted"),
    ({"refused": {"records": "ValueError: the window has 2 row tables"}}, None),
    ({"checks_before_draw": {"registration": False, "m0": True}}, None),
    ({"checks_before_draw": {"registration": True}, "audit": {"accepted": False, "outcome": "coins refused"}}, None),
    ({"fixed_before_draw": {"ok": False, "codes": ["after-the-draw"]}}, "accepted"),
    ({"checks_before_draw": {"registration": True},
      "audit": {"why": {f"lean-verify/m{k}": "session files: no such file" for k in range(5)}}}, "unproved"),
    (AFTER_DRAW, None),
    (UNTIED, None),
    ({"ok": True, "checks_before_draw": {"registration": True}, "fixed_before_draw": {"ok": True},
      "registered": REGISTERED}, "completed"),
])
def test_the_public_copy_is_what_answer_reads_and_answers_as_the_audit_does(tmp_path, summary, rec):
    """`public_copy` of an audit that also wrote what `answer` doesn't read (the window's number, its requests and
    population, a check's detail, the verdicts and statement): the copy holds none of it, answers as the audit does
    (an accept, each refusal, a drawn member's file the audit couldn't complete, a tie that failed after the sessions,
    and Lean's refusals with the drawn units not proved), and every field of it is one of the public kinds as
    `kind_scan.py --received` declares V3c's audit (`V3C_DOCS`). Of the checks after the draw and the registered reads it
    keeps what the onsite record credits members by: each check's outcome with the drawn members completed, by name, and
    each member's reads' outcome, not the ports, roots and values they compared. Of the summary's registered reads it
    keeps the entries they opened the window's own values at (``registered.own``), which the onsite record holds the
    scan's state to, not the other pins' roots or the members' count."""
    if rec == "unproved":
        rec = {**record(False, proved=False), "draw": {**DRAW, "strata": [{"template": "T", "units": [[0, 8]]}]}}
        rec["checks"]["proved"]["detail"] = {"units": {"drawn": [5, 700], "statement": [700]}}
    if rec == "completed":
        rec = {**record(True), "checks_after_draw": COMPLETED, "registered_reads": READS}
    s, r = _planted(summary, record(True) if rec == "accepted" else rec)
    out = write(tmp_path / "audit", s, r)
    I.public_copy(out, tmp_path / "copy")
    assert I.answer(tmp_path / "copy", REG) == I.answer(out, REG)
    if rec in ("accepted", "completed"):
        assert (I.answer(out, REG)["verdict"] == "accept") == (summary.get("ok") is True and summary["fixed_before_draw"]["ok"])
    for f in ("summary.json", "window/audit.json"):
        text = (tmp_path / "copy" / f).read_text() if (tmp_path / "copy" / f).exists() else ""
        assert '"window"' not in text and '"requests"' not in text and '"population": 37' not in text, (f, text)
    assert (tmp_path / "copy" / "window").exists() == (rec is not None)
    copied = json.loads((tmp_path / "copy" / "summary.json").read_text())
    assert copied.get("registered") == ({"own": OWN} if "registered" in summary else None)
    if "checks" in summary and "record" in summary["checks"]:
        assert copied["checks"]["registered_complete"] == {"ok": True, "codes": [], "members": [MEMBER]}
        assert copied["checks"]["record"] == {"ok": False, "codes": ["record"]}
    elif "checks" in summary:
        assert copied["checks"]["registered_complete"] == {"ok": False, "codes": [MEMBER], "members": []}
    if isinstance(rec, dict) and "checks_after_draw" in rec:
        copied = json.loads((tmp_path / "copy" / "window" / "audit.json").read_text())
        assert copied["checks_after_draw"]["registered_complete"] == {"ok": True, "codes": [], "members": [MEMBER]}
        assert copied["registered_reads"] == {"ok": True, "chunks": {MEMBER: {"ok": True}}}
    KS = _kind_scan()
    found, kinds = KS.scan(tmp_path / "copy", KS.V3C_DOCS)
    assert found == [] and sum(kinds.values()) and KS.scan(out, KS.V3C_DOCS)[0], found


def test_the_public_copy_of_no_audit_is_none_and_its_cli_writes_it(tmp_path, capsys):
    """An audit that left no summary leaves no copy, which `answer` reads as the same error, and one whose summary isn't
    a JSON object raises rather than leave none; the CLI writes the copy."""
    I.public_copy(tmp_path / "none", tmp_path / "copy")
    assert list((tmp_path / "copy").iterdir()) == []
    with pytest.raises(RuntimeError, match="no summary"):
        I.answer(tmp_path / "copy", REG)
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "summary.json").write_text("[]")
    with pytest.raises(ValueError, match="not a JSON object"):
        I.public_copy(tmp_path / "bad", tmp_path / "bad-copy")
    out = write(tmp_path / "a", {"checks_before_draw": {"registration": True}}, record(True))
    assert I.main(["public-copy", str(out), str(tmp_path / "cli")]) == 0
    assert I.answer(tmp_path / "cli", REG) == I.answer(out, REG)
