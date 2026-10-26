"""S1's run of record (``runner --of-record``), the joint run: it sets ``--wired --tie --pouw h1 --registered --jobs 1
--session-jobs 1`` and a work law drawing every unit (``work:14`` on the toy, `runner.wired_units`), and
``CERTIFIER_TAMPER=byte`` under ``--tie-tamper``, each of them reaching the loop, and refuses a command that contradicts
any of them with the reason. ``--tie`` and ``--wired`` combine, drawing every unit by default. Every run records its mode
and exact flags in ``mode.json``, each window's ``period.json`` and report and its ``ALLP`` line."""
from __future__ import annotations

import json

import pytest

from verity_all_protocols import runner as RN

SHAPE = {"program": "ab" * 64, "value": "out", "words": 1, "rows": 512, "word_bits": 16,
         "row_schema": "hm96-sha512/row/v2"}


def refused(tmp_path, capsys, *argv: str) -> str:
    with pytest.raises(SystemExit) as x:
        RN.main(["--out", str(tmp_path / "out"), "--tools", str(tmp_path / "missing.json"), "--of-record", *argv])
    assert x.value.code == 2
    return capsys.readouterr().err


def run_of_record(tmp_path, monkeypatch, argv: list[str]) -> tuple[list, list, list]:
    """``runner`` on ``argv`` with the loop, the relay and the legs stubbed: the serve calls, the loops' flags and the
    relay's arguments."""
    shapes, loops, relays = [], [], []

    def serve(argv):
        shapes.append(list(argv))
        return {"program": SHAPE["program"], "out_shape": SHAPE}

    def loop(index, run, a, scratch, out):
        loops.append({**{k: getattr(a, k) for k in ("tie", "pouw", "registered", "law", "jobs", "session_jobs",
                                                     "relay", "wired", "cap")}, "shape": RN.shape_of(index, run, a)})
        d = scratch / f"w{index}"
        d.mkdir(parents=True)
        (d / "period.json").write_text(json.dumps({"window": index, "circuit": "cd" * 64, "run_root": "ef" * 64,
                                                   "law": a.law, "cap": 8, "delta": a.delta, "prover_env": {},
                                                   "mode": a.mode}))
        return d, {"population": 4}

    monkeypatch.setattr(RN, "serve", serve)
    monkeypatch.setattr(RN.T, "relay", lambda *args: relays.append(args) or {"h": 1})
    monkeypatch.setattr(RN, "loop_window", loop)
    monkeypatch.setattr(RN, "pouw_audit", lambda d, out: {"rc": 0, "line": None})
    decide = RN.decide
    monkeypatch.setattr(RN, "decide", lambda d, node, out, exe=None: decide(d, node, out, legs={}.get))
    (tmp_path / "tools.json").write_text("{}")
    assert RN.main(["--out", str(tmp_path / "out"), "--tools", str(tmp_path / "tools.json"), *argv]) == 0
    return shapes, loops, relays


def test_a_run_of_record_is_the_joint_run_and_records_its_flags(tmp_path, monkeypatch, capsys):
    """``--of-record`` alone is the joint run: the wired circuit with registered weights, the output tied at the
    circuit's out shape, every unit drawn (``work:14`` on the toy), and no gap in its mode."""
    monkeypatch.delenv("CERTIFIER_TAMPER", raising=False)
    argv = ["--of-record", "--windows", "2", "--tie-tamper", "0"]
    shapes, loops, relays = run_of_record(tmp_path, monkeypatch, argv)
    commit, shape = shapes[0], shapes[1]
    assert commit[0] == "--commit-weights" and "--wired" in commit
    assert shape[shape.index("--law") + 1] == "work:14" and "--tie" in shape and shape[shape.index("--pouw") + 1] == "h1"
    assert "--registered" in shape and "--wired" in shape and relays[0][-2] == SHAPE and relays[0][-3] is True
    assert all({"--wired", "--registered", "--tie"} <= set(x["shape"]) for x in loops) and len(loops) == 2
    assert [{k: v for k, v in x.items() if k != "shape"} for x in loops] == [
        {"tie": True, "pouw": "h1", "registered": True, "law": "work:14", "jobs": 1, "session_jobs": 1,
         "relay": {"h": 1}, "wired": True, "cap": 2}] * 2
    mode = json.loads((tmp_path / "out" / "mode.json").read_text())
    assert mode["of_record"] is True and mode["argv"][4:] == argv
    assert {k: mode["flags"][k] for k in ("tie", "pouw", "registered", "wired", "law", "jobs", "session_jobs",
                                          "tie_tamper", "certifier_tamper")} \
        == {"tie": True, "pouw": "h1", "registered": True, "wired": True, "law": "work:14", "jobs": 1,
            "session_jobs": 1, "tie_tamper": [0], "certifier_tamper": "byte"}
    assert "gap" not in mode and "companion" not in mode
    for w in (0, 1):
        assert json.loads((tmp_path / "out" / f"w{w}" / "report.json").read_text())["mode"] == mode
    lines = [json.loads(x.split(" ", 1)[1]) for x in capsys.readouterr().out.splitlines() if x.startswith("ALLP ")]
    assert [x["window"]["of_record"] for x in lines] == [True, True]
    assert all("gap" not in x["window"] for x in lines)


def test_giving_wired_to_a_run_of_record_is_the_same_run(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("CERTIFIER_TAMPER", raising=False)
    for argv in (["--of-record"], ["--of-record", "--wired"]):
        assert RN.main(["--out", str(tmp_path), "--tools", "/dev/null", "--check", *argv]) == 0
        flags = json.loads(capsys.readouterr().out)["flags"]
        assert (flags["wired"], flags["tie"], flags["law"], flags["cap"]) == (True, True, "work:14", 2)


def test_a_wired_cap_must_divide_the_request_count(tmp_path, capsys):
    with pytest.raises(SystemExit) as x:
        RN.main(["--out", str(tmp_path), "--tools", "/dev/null", "--check", "--wired", "--cap", "8"])
    assert x.value.code == 2 and "the cap divides the request count, 2: give --cap 2" in capsys.readouterr().err


@pytest.mark.parametrize("shape, law", [([], "work:14"), (["--requests", "3,7,9", "--steps", "3"], "work:33")])
def test_the_tie_and_the_wired_circuit_combine_and_draw_every_unit(shape, law, tmp_path, capsys):
    assert RN.wired_units(2, 2) == 14 and RN.wired_units(3, 3) == 33
    assert RN.main(["--out", str(tmp_path), "--tools", "/dev/null", "--check", "--tie", "--wired", *shape]) == 0
    mode = json.loads(capsys.readouterr().out)
    assert mode["flags"]["tie"] is True and mode["flags"]["wired"] is True and mode["flags"]["law"] == law


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
    assert mode["of_record"] is True and mode["flags"]["law"] == "work:14" and mode["flags"]["certifier_tamper"] is None
    assert mode["flags"]["registered"] is True and mode["flags"]["wired"] is True and mode["flags"]["tie"] is True
    assert "gap" not in mode


@pytest.mark.parametrize("law", ["subset:64", "work:64", "work:4"])
def test_a_run_of_record_refuses_another_law(law, tmp_path, capsys):
    err = refused(tmp_path, capsys, "--law", law)
    assert "S1's run of record draws under work:14" in err and f"not --law {law}" in err


def test_a_run_of_record_refuses_windows_audited_at_once(tmp_path, capsys):
    assert "one firewalled session at a time on the run" in refused(tmp_path, capsys, "--jobs", "2")


def test_a_run_of_record_refuses_sessions_at_once(tmp_path, capsys):
    assert "an audit runs its sessions one at a time" in refused(tmp_path, capsys, "--session-jobs", "2")


def test_a_run_of_record_refuses_a_tie_tamper_on_another_relay(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CERTIFIER_TAMPER", "extra")
    err = refused(tmp_path, capsys, "--tie-tamper", "0")
    assert "--tie-tamper ties a window to the CERTIFIER_TAMPER=byte relay" in err and "CERTIFIER_TAMPER=extra" in err
