"""S1's run of record (``runner --of-record``): it sets ``--tie --pouw h1 --registered --law work:64 --jobs 1
--session-jobs 1`` (and ``CERTIFIER_TAMPER=byte`` under ``--tie-tamper``), each of them reaching the loop, and refuses a
command that contradicts any of them, or gives ``--wired``, with the reason; it names the wiring gap and its companion
run. Every run records its mode and exact flags in ``mode.json``, each window's ``period.json`` and report and its
``ALLP`` line."""
from __future__ import annotations

import json

import pytest

from verity_all_protocols import runner as RN


def refused(tmp_path, capsys, *argv: str) -> str:
    with pytest.raises(SystemExit) as x:
        RN.main(["--out", str(tmp_path / "out"), "--tools", str(tmp_path / "missing.json"), "--of-record", *argv])
    assert x.value.code == 2
    return capsys.readouterr().err


def test_a_run_of_record_applies_its_flags_and_records_them(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("CERTIFIER_TAMPER", raising=False)
    shapes, loops = [], []

    def serve(argv):
        shapes.append(list(argv))
        return {"program": "ab" * 64}

    def loop(index, run, a, scratch, out):
        loops.append({k: getattr(a, k) for k in ("tie", "pouw", "registered", "law", "jobs", "session_jobs", "relay")})
        d = scratch / f"w{index}"
        d.mkdir(parents=True)
        (d / "period.json").write_text(json.dumps({"window": index, "circuit": "cd" * 64, "run_root": "ef" * 64,
                                                   "law": a.law, "cap": 8, "delta": a.delta, "prover_env": {},
                                                   "mode": a.mode}))
        return d, {"population": 4}

    monkeypatch.setattr(RN, "serve", serve)
    monkeypatch.setattr(RN.T, "relay", lambda *args: {"h": 1})
    monkeypatch.setattr(RN, "loop_window", loop)
    monkeypatch.setattr(RN, "pouw_audit", lambda d, out: {"rc": 0, "line": None})
    decide = RN.decide
    monkeypatch.setattr(RN, "decide", lambda d, node, out, exe=None: decide(d, node, out, legs={}.get))
    (tmp_path / "tools.json").write_text("{}")
    argv = ["--out", str(tmp_path / "out"), "--tools", str(tmp_path / "tools.json"), "--of-record", "--windows", "2",
            "--tie-tamper", "0"]
    assert RN.main(argv) == 0
    commit, shape = shapes[0], shapes[1]
    assert commit[0] == "--commit-weights" and "--wired" not in commit
    assert shape[shape.index("--law") + 1] == "work:64" and "--tie" in shape and shape[shape.index("--pouw") + 1] == "h1"
    assert "--registered" in shape and "--wired" not in shape
    assert loops == [{"tie": True, "pouw": "h1", "registered": True, "law": "work:64", "jobs": 1, "session_jobs": 1,
                      "relay": {"h": 1}}] * 2
    mode = json.loads((tmp_path / "out" / "mode.json").read_text())
    assert mode["of_record"] is True and mode["argv"] == argv
    assert {k: mode["flags"][k] for k in ("tie", "pouw", "registered", "wired", "law", "jobs", "session_jobs",
                                          "tie_tamper", "certifier_tamper")} \
        == {"tie": True, "pouw": "h1", "registered": True, "wired": False, "law": "work:64", "jobs": 1,
            "session_jobs": 1, "tie_tamper": [0], "certifier_tamper": "byte"}
    assert mode["gap"] == RN.WIRING_GAP and "companion run (wired, registered, no tie)" in mode["gap"]
    assert "--out-words" in mode["gap"] and mode["companion"] == RN.COMPANION
    for w in (0, 1):
        assert json.loads((tmp_path / "out" / f"w{w}" / "report.json").read_text())["mode"] == mode
    lines = [json.loads(x.split(" ", 1)[1]) for x in capsys.readouterr().out.splitlines() if x.startswith("ALLP ")]
    assert [x["window"]["of_record"] for x in lines] == [True, True]
    assert [x["window"]["gap"] for x in lines] == [RN.WIRING_GAP] * 2


def test_the_wiring_companion_run_serves_wired_and_registered_and_is_not_of_record(tmp_path, monkeypatch, capsys):
    shapes, loops = [], []

    def serve(argv):
        shapes.append(list(argv))
        return {"program": "ab" * 64}

    def loop(index, run, a, scratch, out):
        loops.append(RN.shape_of(index, run, a))
        d = scratch / f"w{index}"
        d.mkdir(parents=True)
        (d / "period.json").write_text(json.dumps({"window": index, "circuit": "cd" * 64, "run_root": "ef" * 64,
                                                   "law": a.law, "cap": 8, "delta": a.delta, "prover_env": {},
                                                   "mode": a.mode}))
        return d, {"population": 4}

    monkeypatch.setattr(RN, "serve", serve)
    monkeypatch.setattr(RN, "loop_window", loop)
    monkeypatch.setattr(RN, "pouw_audit", lambda d, out: {"rc": 0, "line": None})
    decide = RN.decide
    monkeypatch.setattr(RN, "decide", lambda d, node, out, exe=None: decide(d, node, out, legs={}.get))
    (tmp_path / "tools.json").write_text("{}")
    argv = ["--out", str(tmp_path / "out"), "--tools", str(tmp_path / "tools.json"), *RN.COMPANION.split()]
    assert RN.main(argv) == 0
    assert shapes[0][0] == "--commit-weights" and "--wired" in shapes[0]
    assert all("--wired" in s and "--registered" in s and "--tie" not in s for s in loops) and len(loops) == 2
    mode = json.loads((tmp_path / "out" / "mode.json").read_text())
    assert mode["of_record"] is False and "gap" not in mode
    assert {k: mode["flags"][k] for k in ("wired", "registered", "tie", "pouw", "law", "tamper")} \
        == {"wired": True, "registered": True, "tie": False, "pouw": "h1", "law": "work:14", "tamper": [0]}
    assert mode["flags"]["cap"] == 2 and all(s[s.index("--cap") + 1] == "2" for s in loops)


def test_a_wired_cap_must_divide_the_request_count(tmp_path, capsys):
    with pytest.raises(SystemExit) as x:
        RN.main(["--out", str(tmp_path), "--tools", "/dev/null", "--check", "--wired", "--cap", "8"])
    assert x.value.code == 2 and "the cap divides the request count, 2: give --cap 2" in capsys.readouterr().err


def test_a_run_of_record_refuses_the_wired_circuit_and_names_its_companion(tmp_path, capsys):
    err = refused(tmp_path, capsys, "--wired")
    assert "S1's run of record is unwired" in err and "--out-words" in err and "--registered --wired" in err


def test_the_tie_and_the_wired_circuit_are_refused_together(tmp_path, capsys):
    with pytest.raises(SystemExit) as x:
        RN.main(["--out", str(tmp_path), "--tools", "/dev/null", "--check", "--tie", "--wired"])
    assert x.value.code == 2 and "per-coordinate circuit's output units" in capsys.readouterr().err


def test_a_run_of_record_is_refused_when_a_flag_of_its_mode_is_unmet(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(RN, "of_record", lambda a: None)
    err = refused(tmp_path, capsys, "--check")
    assert "S1's run of record runs with" in err and "registered" in err and "tie" in err


def test_a_development_run_says_it_is_not_of_record(tmp_path, capsys):
    assert RN.main(["--out", str(tmp_path), "--tools", "/dev/null", "--check", "--windows", "2"]) == 0
    mode = json.loads(capsys.readouterr().out)
    assert mode["of_record"] is False and mode["flags"]["tie"] is False and mode["flags"]["pouw"] is None


def test_check_prints_a_run_of_records_mode_without_reading_its_tools(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("CERTIFIER_TAMPER", raising=False)
    assert RN.main(["--out", str(tmp_path), "--tools", "/dev/null", "--of-record", "--check"]) == 0
    mode = json.loads(capsys.readouterr().out)
    assert mode["of_record"] is True and mode["flags"]["law"] == "work:64" and mode["flags"]["certifier_tamper"] is None
    assert mode["flags"]["registered"] is True and mode["flags"]["wired"] is False and mode["gap"] == RN.WIRING_GAP


@pytest.mark.parametrize("law", ["subset:64", "work:4"])
def test_a_run_of_record_refuses_another_law(law, tmp_path, capsys):
    err = refused(tmp_path, capsys, "--law", law)
    assert "S1's run of record draws under work:64" in err and f"not --law {law}" in err


def test_a_run_of_record_refuses_windows_audited_at_once(tmp_path, capsys):
    assert "one firewalled session at a time on the run" in refused(tmp_path, capsys, "--jobs", "2")


def test_a_run_of_record_refuses_sessions_at_once(tmp_path, capsys):
    assert "an audit runs its sessions one at a time" in refused(tmp_path, capsys, "--session-jobs", "2")


def test_a_run_of_record_refuses_a_tie_tamper_on_another_relay(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CERTIFIER_TAMPER", "extra")
    err = refused(tmp_path, capsys, "--tie-tamper", "0")
    assert "--tie-tamper ties a window to the CERTIFIER_TAMPER=byte relay" in err and "CERTIFIER_TAMPER=extra" in err
