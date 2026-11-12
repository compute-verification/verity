"""response_tie_verdict: a response tie's verdict becomes one record of the run that produced it, naming the run, the
verifier's module and commit, the verdict with both of its bits, and no proof; a refused tie is written like an
accepted one, each tie a line, and a record that could name no run or no commit is refused."""
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
PARTIAL = TS.TieVerdict("responses", 0, 2, "aa" * 64, "bb" * 64, True, False)
REFUSED = TS.TieVerdict("responses", 1, 2, "cc" * 64, None, False, False, ("R7-certified",))


def test_a_record_names_its_run_the_verifier_and_its_commit_and_no_proof():
    """The record holds exactly the run, the verifier (the module that decides a tie, importable under that name,
    and the commit), the verdict as the verifier writes it, accepted and complete apart, and `proved_by` null."""
    got = RV.record(PARTIAL, run=RUN, commit=COMMIT)
    assert got == {"run": RUN, "verifier": {"module": "verity.protocols.certifier.verifier.response_tie_service",
                                           "commit": COMMIT}, "verdict": PARTIAL.to_json(), "proved_by": None}
    assert tuple(got) == RV.KEYS and importlib.import_module(got["verifier"]["module"]) is TS
    assert (got["verdict"]["accepted"], got["verdict"]["complete"]) == (True, False)
    assert RV.RECORD == "exfiltration-glm53-stop-verdicts"


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


def test_each_tie_is_one_line_refused_or_accepted(tmp_path):
    """`write` appends a line a tie to the run directory's file, making the directory, and writes a refused tie
    with its codes like an accepted one; `read` returns them in order and refuses a line that is no record."""
    out = tmp_path / "out"
    for verdict in (ACCEPTED, REFUSED, PARTIAL):
        assert RV.write(out, verdict, run=RUN, commit=COMMIT) == out / RV.FILE
    lines = (out / RV.FILE).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == RV.read(out) == [
        RV.record(v, run=RUN, commit=COMMIT) for v in (ACCEPTED, REFUSED, PARTIAL)]
    refused = RV.read(out)[1]
    assert refused["verdict"]["codes"] == ["R7-certified"] and refused["verdict"]["accepted"] is False
    assert refused["verdict"]["registration"] is None and refused["proved_by"] is None
    with pytest.raises(ValueError, match=RV.RUN):
        RV.write(out, ACCEPTED, commit=COMMIT, environ={})
    assert len(RV.read(out)) == 3
    (out / RV.FILE).write_text(json.dumps({"run": RUN, "verdict": "accept"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="a record holds"):
        RV.read(out)
