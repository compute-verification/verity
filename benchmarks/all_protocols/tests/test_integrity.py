"""The integrity leg's answer from what the records audit wrote, and the leg end to end:

* an accepted audit record is sampled proofs' accept: the registration's digest, the drawn and proved units with the bound
  on wrong units at δ, and the profile as `Core.Integrity` with δ an exact ratio, all read back by the core answer's
  decoder; the report's ε is that bound over the window's units;
* a refusal of the records, a check before the draw that failed, an audit record that doesn't accept and advice fixed
  after the draw each reject by name, with no public outputs or profile;
* an audit that left nothing behind is the leg's error, not a verdict;
* an audit record decided at δ = 0.01 (a float in the record) is the exact ratio 1/100 in the profile;
* the audit's case for onsite is handed on as ``sampled-proofs.case.json``, with whether `OneStage.check`'s Python on it
  gave the leg's verdict and bound;
* the records kept in ``$OUT`` hold no prover's file;
* two audits at once hold different blocks of ports;
* opt-in (``VERITY_ALL_PROTOCOLS_TOOLS=tools.json``, the built executables `run.sh` writes; minutes): two windows of the
  loop through the runner at once, the honest one accepted by the Lean verifier of record with its case for onsite and the
  tampered one rejected, which rejects the composition and the monitoring period.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from verity.ml.pouw import core_check as CC
from verity.core.protocols.sampled_proofs.one_stage import registration as R
from verity_all_protocols import runner as RN, table as T
from verity_all_protocols.table import check_answer

I = T.load("sampled-proofs")

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


def test_an_accepted_audit_is_an_accept(tmp_path):
    out = write(tmp_path / "a", {"checks_before_draw": {"registration": True}, "fixed_before_draw": {"ok": True}},
                record(True))
    a = I.answer(out, REG)
    assert check_answer(a) == a
    assert a["verdict"] == "accept" and a["protocol"] == "verity/sampled-proofs/one-stage/v1"
    assert a["commitment"] == R.digest(REG)
    assert a["public"] == {"bound": 3, "drawn": [5, 700], "per_template": None, "proved": [5, 700]}
    p = CC.integrity(a["profile"])
    assert str(p["delta"]) == "1/2" and p["drawn"] == [("unit", 2)]
    assert p["levels"] == [{"name": "unit", "query": "bb" * 64, "size": 1024, "draw": {"subset": 2}}]
    assert RN.epsilon(a) == "3/1024"


@pytest.mark.parametrize("summary,rec,why", [
    ({"refused": {"records": "ValueError: the window has 2 row tables"}}, None, "refused: "),
    ({"checks_before_draw": {"registration": False, "m0": True}}, None, "before the draw: registration"),
    ({"checks_before_draw": {"registration": True}}, record(False, proved=False), "audit record: proved"),
    ({"checks_before_draw": {"registration": True}, "audit": {"accepted": False, "outcome": "coins refused"}}, None,
     "no audit record: "),
    ({"fixed_before_draw": {"ok": False, "codes": ["after-the-draw"]}}, record(True), "not fixed before the draw: after"),
])
def test_refusals_reject_by_name(tmp_path, summary, rec, why):
    a = I.answer(write(tmp_path / "a", summary, rec), REG)
    assert check_answer(a) == a
    assert a["verdict"] == "reject" and a["reason"].startswith(why)
    assert a["commitment"] == R.digest(REG)


def test_an_audit_that_left_nothing_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="no summary"):
        I.answer(tmp_path, REG)
    with pytest.raises(RuntimeError, match="no audit record and no refusal"):
        I.answer(write(tmp_path / "a", {"checks_before_draw": {"registration": True}}), REG)


def test_an_audit_at_one_percent_is_the_exact_ratio(tmp_path):
    rec = {**record(True), "wrong_units": {"delta": 0.01, "of": 1024, "bound": 69}}
    a = I.answer(write(tmp_path / "a", {"checks_before_draw": {"registration": True}}, rec), REG)
    assert a["profile"]["delta"] == "1/100" and str(CC.integrity(a["profile"])["delta"]) == "1/100"


def test_the_case_is_handed_on_with_its_check(tmp_path):
    work = tmp_path / "work"
    (work / "window").mkdir(parents=True)
    case = {"commitment": "aa", "evidence": "bb", "params": {"delta": "1/100"}}
    (work / "window" / "case.json").write_text(json.dumps(case))
    (work / "window" / "case-python.json").write_text(json.dumps({"verdict": "accept", "reason": None,
                                                                  "public": {"bound": 3}}))
    out = tmp_path / "out"
    out.mkdir()
    a = {"verdict": "accept", "public": {"bound": 3}}
    assert I.case_check(work, a, out)["agree"] is True
    assert json.loads((out / "sampled-proofs.case.json").read_text()) == case
    assert I.case_check(work, {"verdict": "accept", "public": {"bound": 4}}, out)["agree"] is False
    assert I.case_check(work, {"verdict": "reject", "public": None}, out)["agree"] is False
    assert I.case_check(tmp_path / "none", a, out) is None


def test_two_audits_at_once_hold_different_ports():
    with I.ports() as a, I.ports() as b:
        assert abs(a - b) >= I.SPAN
    assert not I._held


def test_kept_records_hold_no_prover_file(tmp_path):
    src = tmp_path / "work"
    for f in ("summary.json", "window/audit.json", "window/m0/circuit.txt", "window/m0/prover/inst-4.bin",
              "window/m0/m0-inst-1.bin", "window/m3-private.bin", "window/m0/archive/x.json", "window/m0/pub-4.bin"):
        (src / f).parent.mkdir(parents=True, exist_ok=True)
        (src / f).write_text("x")
    I.keep(src, tmp_path / "kept")
    assert sorted(str(f.relative_to(tmp_path / "kept")) for f in (tmp_path / "kept").rglob("*") if f.is_file()) == [
        "summary.json", "window/audit.json", "window/m0/pub-4.bin"]


TOOLS = os.environ.get("VERITY_ALL_PROTOCOLS_TOOLS")


@pytest.mark.slow
@pytest.mark.skipif(not TOOLS, reason="opt-in: VERITY_ALL_PROTOCOLS_TOOLS=tools.json (run.sh's built executables); "
                    "two windows proved under --zk, minutes")
def test_two_windows_through_the_verifier_of_record(tmp_path):
    rc = RN.main(["--out", str(tmp_path / "out"), "--tools", TOOLS, "--windows", "2", "--tamper", "1",
                  "--law", "subset:8", "--requests", "3", "--steps", "1"])
    reports = sorted((json.loads(x) for x in (tmp_path / "out" / "reports.jsonl").read_text().splitlines()),
                     key=lambda r: r["window"])
    assert rc == 0
    honest, tampered = (r["protocols"]["integrity"] for r in reports)
    check = json.loads((tmp_path / "out" / "w0" / "integrity" / "case-check.json").read_text())
    assert check["agree"] and (tmp_path / "out" / "w0" / "integrity" / "sampled-proofs.case.json").exists()
    assert honest["verdict"] == "accept"
    assert honest["answer"]["profile"]["levels"][0]["size"] == reports[0]["loop"]["population"]
    assert tampered["verdict"] == "reject"
    assert [r["composed"]["verdict"] for r in reports] == ["pending-ports", "reject"]
    m = json.loads((tmp_path / "out" / "monitoring-period.json").read_text())
    assert m["integrity"]["observed"]["verdict"] == "reject" and m["composed"]["verdict"] == "reject"
