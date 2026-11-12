"""response_tie_verdict: a response tie's verdict becomes one record of the run that produced it, naming the run, the
verifier's module and commit, the verdict with both of its bits, and no proof; a refused tie is written like an
accepted one, each tie a line and only once, with the registration it was decided under kept beside it, and a record
that could name no run or no commit is refused."""
from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import pytest
from verity.protocols.certifier.verifier import response_tie_service as TS

HERE = Path(__file__).resolve().parents[1]
RUN, COMMIT = "r-test", "0123456789abcdef0123456789abcdef01234567"


def _load():
    spec = importlib.util.spec_from_file_location("network_traces_response_tie_verdict", HERE / "response_tie_verdict.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RV = _load()
ACCEPTED = TS.TieVerdict("responses", 0, 2, "aa" * 64, "bb" * 64, True, True)
PARTIAL = TS.TieVerdict("responses", 0, 0, "aa" * 64, "bb" * 64, True, False)
REFUSED = TS.TieVerdict("responses", 1, 2, "cc" * 64, None, False, False, ("R7-certified",))
UNREAD = TS.TieVerdict("responses", 2, 2, None, None, False, False, ("head",))
#: stand-ins for two execution phases' registrations: the writer keeps a document as handed and reads nothing of it
PHASE0 = {"link": "responses", "window": 0, "tables": [{"name": "end.2", "rows": 2}], "certificate": {"sig": "00"}}
PHASE1 = {"link": "responses", "window": 1, "tables": [{"name": "end.2", "rows": 1}], "certificate": {"sig": "01"}}


def test_a_record_names_its_run_the_verifier_and_its_commit_and_no_proof():
    """The record holds exactly the run, the verifier (the module that decides a tie, importable under that name,
    and the commit), the verdict as the verifier writes it, accepted and complete apart under the names a reader
    binds it to a head by, and `proved_by` null."""
    got = RV.record(PARTIAL, run=RUN, commit=COMMIT)
    assert got == {"run": RUN, "verifier": {"module": "verity.protocols.certifier.verifier.response_tie_service",
                                           "commit": COMMIT}, "verdict": PARTIAL.to_json(), "proved_by": None}
    assert tuple(got) == RV.KEYS and importlib.import_module(got["verifier"]["module"]) is TS
    assert {k: got["verdict"][k] for k in ("format", "link", "window", "class", "head", "accepted", "complete")} == {
        "format": "verity-network-certifier/response-tie-verdict/v1", "link": "responses", "window": 0, "class": 0,
        "head": "aa" * 64, "accepted": True, "complete": False}
    assert (RV.RECORD, RV.FILE, RV.REGISTRATIONS) == (
        "exfiltration-glm53-stop-verdicts", "response-tie-verdicts.jsonl", "registrations.jsonl")


def test_the_run_and_the_commit_are_the_callers_or_the_environments_and_never_missing():
    """Without `run=` and `commit=` the record takes the environment's, as `research run` sets them; what the
    caller gives wins; and with neither, or a commit that is no full hash, the record is refused, naming the
    variable."""
    env = {RV.RUN: "r-env", RV.COMMIT: "f" * 40}
    assert RV.record(ACCEPTED, environ=env)["run"] == "r-env"
    assert RV.record(ACCEPTED, environ=env)["verifier"]["commit"] == "f" * 40
    assert RV.record(ACCEPTED, run=RUN, commit=COMMIT, environ=env)["run"] == RUN
    assert RV.record(ACCEPTED, run=RUN, commit=COMMIT, environ=env)["verifier"]["commit"] == COMMIT
    with pytest.raises(ValueError, match=RV.RUN):
        RV.record(ACCEPTED, commit=COMMIT, environ={})
    for commit in (None, "abc1234", "F" * 40, "g" * 40):
        with pytest.raises(ValueError, match=RV.COMMIT):
            RV.record(ACCEPTED, run=RUN, commit=commit, environ={})
    for no_verdict in (ACCEPTED.to_json(), None, True):
        with pytest.raises(ValueError, match="a response tie's verdict"):
            RV.record(no_verdict, run=RUN, commit=COMMIT)


def test_each_tie_is_one_line_with_its_registration_kept_once(tmp_path):
    """`write` appends a line a tie to the run directory's file, making the directory, and writes a refused tie
    with its codes like an accepted one. The registration a verdict was decided under is kept beside it as handed,
    once for the classes that read the same one, and a document that didn't read (its verdict names no head) is
    not kept. `read` and `registrations` return them in order; a directory with neither file has none."""
    out = tmp_path / "out"
    assert RV.read(out) == [] and RV.registrations(out) == [] and not out.exists()
    handed = ((ACCEPTED, PHASE0), (REFUSED, PHASE1), (PARTIAL, dict(reversed(PHASE0.items()))), (UNREAD, {"not": "one"}))
    for verdict, registration in handed:
        assert RV.write(out, verdict, registration, run=RUN, commit=COMMIT) == out / RV.FILE
    lines = (out / RV.FILE).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == RV.read(out) == [
        RV.record(v, run=RUN, commit=COMMIT) for v, _ in handed]
    assert RV.registrations(out) == [PHASE0, PHASE1]
    refused = RV.read(out)[1]
    assert refused["verdict"]["codes"] == ["R7-certified"] and refused["verdict"]["accepted"] is False
    assert refused["verdict"]["registration"] is None and refused["proved_by"] is None
    (out / RV.FILE).write_text(json.dumps({"run": RUN, "verdict": "accept"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="a record holds"):
        RV.read(out)


def test_a_tie_has_one_verdict_and_a_refused_write_leaves_nothing(tmp_path):
    """A second verdict of the same class of the same execution phase of a link is refused, whatever it says:
    nothing retries a tie, so a refused one is never followed by an accepted one. A record that names no run, and
    a verdict that names a head with no registration, are refused too, and none of the three writes a byte."""
    out = tmp_path / "out"
    RV.write(out, REFUSED, PHASE1, run=RUN, commit=COMMIT)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    again = TS.TieVerdict("responses", 1, 2, "cc" * 64, "dd" * 64, True, True)
    with pytest.raises(ValueError, match="a tie has one verdict"):
        RV.write(out, again, PHASE1, run=RUN, commit=COMMIT)
    with pytest.raises(ValueError, match=RV.RUN):
        RV.write(out, ACCEPTED, PHASE0, commit=COMMIT, environ={})
    for no_registration in (None, [PHASE0], "registration"):
        with pytest.raises(ValueError, match="kept with the registration"):
            RV.write(out, ACCEPTED, no_registration, run=RUN, commit=COMMIT)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before and set(before) == {RV.FILE, RV.REGISTRATIONS}
    RV.write(out, TS.TieVerdict("responses", 1, 0, "cc" * 64, "dd" * 64, True, True), PHASE1, run=RUN, commit=COMMIT)
    assert len(RV.read(out)) == 2 and RV.registrations(out) == [PHASE1]
