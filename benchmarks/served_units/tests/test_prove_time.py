"""prove_time.py's summary on prove_time.sh's records: a statement's time enters the type's fit only when its prover exited 0,
every session it filed was accepted, and it filed the timed sessions the run asked for."""
from __future__ import annotations

import json

import pytest

from conftest import load

STAGE = {"tag": "census-t", "k_log": 22, "unit_log": 21, "vus_per_block": 1, "ands": 1000}


@pytest.fixture
def pt(monkeypatch):
    return load(monkeypatch, "prove_time.py")


def statement(out, name, n, sessions, rc="0"):
    """prove_time.sh's records of one statement: its stage.json, its prover's stamped LIVE lines (``sessions``: (warm,
    accepted, prove_total_s) each) and its exit status (none written where ``rc`` is None)."""
    (out / "stage" / name).mkdir(parents=True)
    (out / "stage" / name / "stage.json").write_text(json.dumps({**STAGE, "n": n}))
    d = out / "gpu" / name
    d.mkdir(parents=True)
    (d / "prove.tsv").write_text("".join(f"{100 + i}.0\tLIVE {json.dumps({'warm': w, 'accepted': a, 'prove_total_s': t})}\n"
                                         for i, (w, a, t) in enumerate(sessions)))
    if rc is not None:
        (d / "exit").write_text(rc + "\n")


@pytest.mark.parametrize("sessions, rc, why", [
    ([(True, True, 9.0), (False, True, 2.0)], "0", None),
    ([(True, True, 9.0), (False, True, 2.0)], None, None),
    ([(True, True, 9.0), (False, True, 2.0)], "137", "the prover exited 137"),
    ([(True, True, 9.0), (False, False, 2.0)], "0", "1 of its 2 sessions not accepted"),
    ([(True, False, 9.0), (False, True, 2.0)], "0", "1 of its 2 sessions not accepted"),
    ([(True, True, 9.0)], "1", "the prover exited 1"),
    ([(True, True, 9.0)], "0", "0 of 1 timed sessions filed"),
])
def test_a_statement_gives_a_time_only_on_its_proved_sessions(pt, tmp_path, sessions, rc, why):
    (tmp_path / "config.txt").write_text("device=gpu units=u ns=1 statements= warm=1 runs=1\n")
    statement(tmp_path, "u-n4", 4, [(True, True, 8.0), (False, True, 4.0)])
    statement(tmp_path, "u-n16", 16, sessions, rc)
    res = pt.main(tmp_path, None)
    rec = res["statements"]["gpu/u-n16"]
    assert rec["untimed"] == why
    assert rec["prove_total_s"] == (None if why else 2.0)
    t = res["types"]["gpu/census-t"]
    assert sorted(t["amortized"]) == ([4] if why else [4, 16])
    assert ("per_block_s" in t) == (why is None)


def test_no_session_filed_is_an_error_not_a_time(pt, tmp_path):
    (tmp_path / "config.txt").write_text("device=gpu units=u ns=1 statements= warm=1 runs=1\n")
    statement(tmp_path, "u-n4", 4, [], "1")
    rec = pt.main(tmp_path, None)["statements"]["gpu/u-n4"]
    assert rec["prove_total_s"] is None and rec["untimed"] is None and rec["error"] == ""
