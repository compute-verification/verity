"""The recursion bench's headline (`rec_profile.headline`, `rec_table`): one whole recursive run, and whether it is an
accepted one, from its steps' records: the lean step's composition of V*'s parts (its record, else its step log), Lean's
verdict on the recursive statement, a forced retry, a release without its report. The records are made here
(`test_recursion_e2e.job`)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import rec_logs as RL
import rec_profile as RP
import rec_rows as RR
import rec_table as RT
from test_recursion_e2e import job, row


def test_headline_is_a_whole_run_only_when_every_step_ran(tmp_path: Path) -> None:
    o = job(tmp_path)
    assert RP.read(o)["headline"]["complete"] is False                     # no vcap step: no V* verdict
    (o / "steps" / "vcap" / "r1" / "out").mkdir(parents=True)
    (o / "steps" / "vcap" / "r1" / "rc").write_text("0\n")
    doc = RP.read(o)
    assert doc["headline"]["complete"] is True and doc["headline"]["ok"] is False      # every step ran; alg-p0 is refused
    assert "Verdict: one whole recursive run, **not accepted**" in RT.markdown(doc)
    h = RP.headline([], {"not_ok": []}, {s: ["0"] for s in RP.E2E_STEPS})
    assert h["complete"] is True and h["parts_ok"] is True and h["ok"] is False      # its parts passed; nothing composed
    h = RP.headline([RR.compose_row([{"composed": True, "accepted": True, "why": ""}])], {"not_ok": []},
                    {s: ["0"] for s in RP.E2E_STEPS})
    assert h["complete"] is True and h["parts_ok"] is True and h["ok"] is True
    h = RP.headline([], {"not_ok": []}, {s: ["0"] for s in RP.E2E_STEPS if s != "lean"})
    assert h["complete"] is False and h["parts_ok"] is False and h["ok"] is False
    h = RP.headline([], {"not_ok": []}, {s: ["0"] for s in RP.E2E_STEPS} | {"lean": ["0", "1"]})
    assert h["complete"] is True and h["parts_ok"] is False and h["ok"] is False
    shutil.rmtree(o / "steps" / "oprove")
    assert "Verdict: **not a whole recursive run**" in RT.markdown(RP.read(o))


def whole(tmp: Path, *left: dict | str | None, modes: str = "dev") -> Path:
    """A job's out/ of one whole run whose parts all passed (`job` with vcap and without its refused alg-p0), under oprove
    modes ``modes``. Each of the lean step's two invocations left what ``left`` says of its composition (the last entry
    for the rest): the step's record (a dict), its log's line alone as before that record (a str), or neither (None)."""
    o = job(tmp)
    (o / "params.txt").write_text((o / "params.txt").read_text().strip() + f" modes={modes}\n")
    (o / "steps" / "vcap" / "r1" / "out").mkdir(parents=True)
    (o / "steps" / "vcap" / "r1" / "rc").write_text("0\n")
    shutil.rmtree(o / "steps" / "oprove" / "r1" / "out" / "oprove" / "alg-p0-dev")
    for i, r in enumerate(sorted((o / "steps" / "lean").glob("r*"))):
        (r / "out" / "lean" / "alg-p0.log").write_text('VERDICT {"accepted": true, "why": ""}\n')
        c = left[min(i, len(left) - 1)]
        if isinstance(c, dict):
            (r / "out" / "lean" / RL.COMPOSITION).write_text(json.dumps(c))
        elif c is not None:
            (r / "step.log").write_text(f"lean L0: exit 0, accepted true, why ''\n{c}\nlean: 2/2 accepted\n")
    return o


ACCEPTED = {"composed": True, "accepted": True, "why": ""}


NOT_ACCEPTED_RUN = "its outer proofs accepted and its recursive statement {}, so not an accepted recursive run (oprove modes {}"


def test_a_composed_and_accepted_run_is_an_accepted_recursive_run(tmp_path: Path) -> None:
    doc = RP.read(whole(tmp_path, ACCEPTED))
    c = row(doc, "compose")
    assert c["step"] == "lean" and c["ok"] is True and c["note"] == "" and c["extra"] == {"composed": True, "accepted": True}
    h = doc["headline"]
    assert (h["complete"], h["parts_ok"], h["composed"], h["not_composed"], h["ok"]) == (True, True, True, None, True)
    assert h["refused"] is None and "Verdict: one whole recursive run, accepted." in RT.markdown(doc)


def test_a_composed_and_refused_run_is_not_accepted(tmp_path: Path) -> None:
    """Lean composed V*'s parts and refused: not accepted whatever the steps' exits (a forced retry's lean step exits 0)."""
    why = "part 1: 2 sessions opened in the run record: a part is never retried"
    doc = RP.read(whole(tmp_path, ACCEPTED, {"composed": True, "accepted": False, "why": why}))
    c = row(doc, "compose")
    assert c["ok"] is False and c["note"] == "refused: " + why and c["extra"] == {"composed": True, "accepted": False}
    h = doc["headline"]
    assert doc["steps"]["lean"] == ["0", "0"] and "compose" in doc["summary"]["not_ok"]
    assert (h["complete"], h["parts_ok"], h["composed"], h["ok"], h["refused"]) == (True, False, True, False, why)
    assert f"Verdict: one whole recursive run, **not accepted**: Lean refused its recursive statement ({why})." in RT.markdown(doc)


RETRIED = "a forced retry of L0 (RETRY=L0): part 0: 2 sessions opened in the run record: a part is never retried"


@pytest.mark.parametrize("left", [
    {"composed": True, "accepted": False, "why": RETRIED},
    ('lean compose (RETRY=L0, part 0: refused as retried, a negative control): REC-COMPOSE {"accepted": false, "why": '
     '"part 0: 2 sessions opened in the run record: a part is never retried", "parts": 3}'),
], ids=["the step's record", "a step log from before the record"])
def test_a_forced_retry_is_not_accepted_and_the_table_names_the_retry(tmp_path: Path, left) -> None:
    """85's RETRY proves a part again on purpose and Lean refuses the composition for it: the lean step exits 0 and the
    run is not an accepted recursive run, with the retry as the reason in the headline and in the table's line."""
    doc = RP.read(whole(tmp_path, left))
    h = doc["headline"]
    assert doc["steps"]["lean"] == ["0", "0"] and (h["composed"], h["ok"], h["refused"]) == (True, False, RETRIED)
    assert f"Verdict: one whole recursive run, **not accepted**: Lean refused its recursive statement ({RETRIED})." in RT.markdown(doc)


@pytest.mark.parametrize("why", ["1 of V*'s 3 parts run", "no interactive-run record: its parts are not shown to be one run"])
def test_a_run_not_composed_is_not_an_accepted_recursive_run(tmp_path: Path, why: str) -> None:
    """The lean step exited 0 on its parts and composed nothing: the row has no verdict, the headline is not ``ok``, and
    the table says the outer proofs were accepted and the recursive statement was not composed, with the modes and why."""
    doc = RP.read(whole(tmp_path, {"composed": False, "accepted": None, "why": why}))
    c = row(doc, "compose")
    assert c["ok"] is None and c["note"] == "not composed: " + why and c["extra"] == {"composed": False, "accepted": None}
    h = doc["headline"]
    assert all(rc == "0" for v in doc["steps"].values() for rc in v) and doc["summary"]["not_ok"] == []
    assert (h["complete"], h["parts_ok"], h["composed"], h["not_composed"], h["ok"]) == (True, True, False, c["note"], False)
    md = RT.markdown(doc)
    assert "Verdict: one whole run, " + NOT_ACCEPTED_RUN.format("**not composed**", "dev") + f": {why})." in md
    assert "recursive run, accepted" not in md


def test_one_invocation_not_composed_is_no_accepted_run(tmp_path: Path) -> None:
    doc = RP.read(whole(tmp_path, ACCEPTED, {"composed": False, "accepted": None, "why": "1 of V*'s 3 parts run"}))
    assert row(doc, "compose")["ok"] is None and doc["headline"]["composed"] is False and doc["headline"]["ok"] is False


def test_a_lean_step_that_left_no_composition_is_no_accepted_run(tmp_path: Path) -> None:
    """No record and no line of the composition (the step was not asked to compose): not recorded as composed."""
    doc = RP.read(whole(tmp_path, ACCEPTED, None, modes="host"))
    c = row(doc, "compose")
    assert c["ok"] is None and c["extra"] == {"composed": None, "accepted": None}
    assert c["note"] == "no record of the composition (rec-compose.json)"
    h = doc["headline"]
    assert (h["parts_ok"], h["composed"], h["not_composed"], h["ok"]) == (True, None, c["note"], False)
    assert "Verdict: one whole run, " + NOT_ACCEPTED_RUN.format("**not recorded as composed**", "host") in RT.markdown(doc)


def test_a_record_lean_gave_no_answer_in_is_no_accepted_run(tmp_path: Path) -> None:
    doc = RP.read(whole(tmp_path, {"composed": True, "accepted": None, "why": "exit 2: 'usage'"}))
    c = row(doc, "compose")
    assert c["ok"] is None and c["note"] == "no answer: exit 2: 'usage'" and doc["headline"]["ok"] is False
    (tmp_path / "out" / "steps" / "lean" / "r1" / "out" / "lean" / RL.COMPOSITION).write_text("{not json")
    assert row(RP.read(tmp_path / "out"), "compose")["extra"]["composed"] is None


@pytest.mark.parametrize("line, composed, accepted", [
    ('lean compose: REC-COMPOSE {"accepted": true, "why": "", "parts": 3}', True, True),
    ('lean compose: REC-COMPOSE {"accepted": false, "why": "part 1: x", "parts": 3}  <- FAILS', True, False),
    (('lean compose (RETRY=L0, part 0: refused as retried, a negative control): REC-COMPOSE {"accepted": false, "why": '
      '"part 0: 2 sessions opened in the run record: a part is never retried", "parts": 3}'), True, False),
    ("lean compose: 1 of V*'s 3 parts run, so the recursive statement is not composed", False, None),
    (("lean compose: no interactive-run record (rec-run.json), so the recursive statement is not composed: its parts are "
      "not shown to be one run (C10 OneRun)"), False, None),
    ("lean compose: exit 2: 'usage'  <- FAILS", True, None),
    ('lean compose forged-top-L0: REC-COMPOSE {"accepted": false, "why": "x", "parts": 3}', None, None),
])
def test_a_run_from_before_the_record_is_read_from_its_step_log(tmp_path: Path, line: str, composed, accepted) -> None:
    """A lean step from before its record left only its log's line on the honest parts, read as the record is; a
    forgery's line is no composition of the honest parts."""
    doc = RP.read(whole(tmp_path, line))
    c = row(doc, "compose")
    assert c["extra"] == {"composed": composed, "accepted": accepted} and c["ok"] is accepted
    assert doc["headline"]["ok"] is (accepted is True)


def test_the_record_is_read_ahead_of_the_log(tmp_path: Path) -> None:
    o = whole(tmp_path, {"composed": False, "accepted": None, "why": "1 of V*'s 3 parts run"})
    for r in (o / "steps" / "lean").glob("r*"):
        (r / "step.log").write_text('lean compose: REC-COMPOSE {"accepted": true, "why": "", "parts": 3}\n')
    assert RP.read(o)["headline"]["composed"] is False


def test_a_result_from_before_the_composition_is_not_printed_as_accepted(tmp_path: Path) -> None:
    """A stored result whose headline has ``ok`` and no ``composed`` records no composition, and neither does a merge
    of rows without the composition's."""
    assert RT.verdict({"complete": True, "ok": True}, "host,dev") == (
        "one whole run, " + NOT_ACCEPTED_RUN.format("**not recorded as composed**", "host,dev") + ")")
    assert RT.verdict({"complete": True, "ok": False}, "dev") == "one whole recursive run, **not accepted** (see ok and the steps' exits)"
    doc = RP.read(whole(tmp_path, ACCEPTED)) | {"ref": "rA"}
    assert RP.merge([doc])["headline"]["ok"] is True
    doc["rows"] = [r for r in doc["rows"] if r["stage"] != "compose"]
    h = RP.merge([doc])["headline"]
    assert (h["parts_ok"], h["composed"], h["ok"]) == (True, None, False)


def test_release_1_without_a_report_is_not_its_parts_passing(tmp_path: Path) -> None:
    o = whole(tmp_path, ACCEPTED)
    (o / "params.txt").write_text((o / "params.txt").read_text() + "release=1\n")
    h = RP.read(o)["headline"]
    assert (h["composed"], h["parts_ok"], h["ok"]) == (True, False, False)
