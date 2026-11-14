"""pouw_piece.py: PoUW's pieces for `composite.py`, ``verity/pouw/v1`` and ``verity/pouw/period/v1`` (onsite's and
remote's), each a case in its protocol's formats and its answer on it. An honest window (`pouw_check_cases`) and an
honest period (`pouw_period_cases`) written from their inputs accept, with the digest the product binds; a served H1
window (``fixtures/served-h1-window``, window 0's line of a served H200 run's windows.jsonl, a development window in no
period) is MISSING from both, with why, and writes no case, so `composite.py` leaves it out and names it missing; the
directory is private as `composite.py` reads it; and ``--exe`` is the answer of record. The period's is the Lean's
alone: with no ``pouw-period`` its piece is missing, never the Python reference's. Its cases here are answered by a
stand-in ``pouw-period`` (`period_exe`), as credit's tests stand in for ``flock-verify``."""

from __future__ import annotations

import importlib.util
import json
import random
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verity/protocols/accounting/pouw/verifier/lean/check/tests"))
import composite as C
import pouw_check_cases
import pouw_period_cases
import pouw_piece as P
from verity.protocols.accounting.pouw.verifier import core_check as CC
from verity.protocols.accounting.pouw.verifier import credit as Cr
from verity.protocols.accounting.pouw.verifier import period_check as PC
from verity.protocols.accounting.pouw.verifier import verifier as V
from verity.protocols.proofs.spot.verifier.one_stage import proofs as PF

SERVED = Path(__file__).resolve().parent / "fixtures" / "served-h1-window"
STATEMENTS = Path(__file__).resolve().parents[3] / "verity/protocols/accounting/pouw/verifier/tests/test_pouw_credit_statements.py"


@pytest.fixture
def statements(tmp_path, monkeypatch):
    """credit's statements kit (`test_pouw_credit_statements`): its periods of proof files and its stand-in
    flock-verify, which accepts a session only when it proves the files credit hands it; credit's Python asks, which
    the difftests hold to the Lean's."""
    spec = importlib.util.spec_from_file_location("pouw_credit_statements", STATEMENTS)
    kit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kit)
    exe = tmp_path / "flock-verify"
    exe.write_text(kit.FAKE)
    exe.chmod(0o755)
    monkeypatch.setattr(PF, "executable", lambda: exe)
    monkeypatch.setattr(V, "RECORD_VERDICT", "python")
    return kit


@pytest.fixture
def period_exe(tmp_path) -> Path:
    """A stand-in ``pouw-period``: ``check params.json commitment.bin evidence.json``, answered as the Lean answers it
    (`period_check.case`, which ``pouw-period``'s difftest holds to it), its verdict its exit."""
    exe = tmp_path / "pouw-period"
    exe.write_text(f"""#!{sys.executable}
import json, sys
from pathlib import Path
from verity.protocols.accounting.pouw.verifier import period_check as PC
_, params, commitment, evidence = sys.argv[1:]
a = PC.case({{"params": json.loads(Path(params).read_text()), "commitment": Path(commitment).read_bytes().hex(),
             "evidence": Path(evidence).read_bytes().hex()}})
print(json.dumps(a))
sys.exit({{"accept": 0, "reject": 1, "undecided": 2}}[a["verdict"]])
""")
    exe.chmod(0o755)
    return exe


def _json(p: Path, x) -> Path:
    p.write_text(json.dumps(x))
    return p


def _lines(p: Path, xs: list[str]) -> Path:
    p.write_text("".join(x + "\n" for x in xs))
    return p


def test_an_honest_window_is_a_v1_piece_that_accepts(tmp_path):
    case = pouw_check_cases.instance(random.Random(0), "honest")
    reg, ans = (json.loads(bytes.fromhex(case[k])) for k in ("commitment", "evidence"))
    out = tmp_path / "pouw"
    assert P.main(["v1", "--out", str(out), "--params", str(_json(tmp_path / "p.json", case["params"])),
                   "--registration", str(_json(tmp_path / "r.json", reg)),
                   "--answer", str(_json(tmp_path / "a.json", ans))]) == 0
    got = json.loads((out / "answer.json").read_text())
    assert got["verdict"] == "accept" and got["protocol"] == P.V1
    assert got["commitment"] == CC.digest(CC.commitment(bytes.fromhex(case["commitment"]))).hex()
    piece = C.piece(out)
    assert piece["id"] == P.V1 and piece["commitment"].hex() == case["commitment"] and piece["blobs"] == []
    assert stat.S_IMODE(out.stat().st_mode) == 0o700
    assert {f.name: stat.S_IMODE(f.stat().st_mode) for f in out.iterdir()} == dict.fromkeys(P.CASE_FILES, 0o600)


def _honest_period_args(tmp_path: Path) -> tuple[dict, list[str]]:
    case = pouw_period_cases.instance(random.Random(0), "honest")
    p = case["params"]
    ledger = json.loads(bytes.fromhex(case["commitment"]))
    ev = json.loads(bytes.fromhex(case["evidence"]))
    return case, ["--key", str(_json(tmp_path / "k.json", p["key"])),
                  "--main", str(_lines(tmp_path / "main.txt", p["main"])), "--name", p["name"],
                  "--ledger", str(_lines(tmp_path / "ledger.jsonl", ledger)),
                  "--pins", str(_lines(tmp_path / "pins.jsonl", ev["pins"])),
                  "--windows", str(_json(tmp_path / "w.json", ev["windows"]))]


def test_an_honest_period_is_a_period_piece_that_accepts(tmp_path, period_exe, capsys):
    case, args = _honest_period_args(tmp_path)
    p = case["params"]
    out = tmp_path / "period"
    assert P.main(["period", "--out", str(out), "--exe", str(period_exe), *args]) == 0
    assert capsys.readouterr().out.splitlines() == [f"PIECE {P.PERIOD} accept by=lean"]
    got = json.loads((out / "answer.json").read_text())
    assert got["verdict"] == "accept" and got["protocol"] == P.PERIOD
    assert not {"registration_head", "head", "place", "part"} & set(got["public"])
    assert got["commitment"] == PC.digest(PC.commitment(bytes.fromhex(case["commitment"]))).hex()
    assert json.loads((out / "case.json").read_text())["evidence"] == case["evidence"]
    assert json.loads((out / "case.json").read_text())["params"] == p and p["sessions"] != []


def test_a_period_with_no_lean_executable_is_missing_not_the_python_references(tmp_path, capsys):
    """No ``pouw-period`` (``--exe``; `served_pouw.sh` passes none where it doesn't build): the piece is MISSING, exit
    2, holding only ``missing.json``, even over a case the Python reference accepts, and a stale case is removed."""
    case, args = _honest_period_args(tmp_path)
    assert PC.case({k: case[k] for k in ("params", "commitment", "evidence")})["verdict"] == "accept"
    out = tmp_path / "period"
    out.mkdir(mode=0o700)
    (out / "answer.json").write_text("{}\n")
    assert P.main(["period", "--out", str(out), *args]) == 2
    assert capsys.readouterr().out.splitlines() == [f"MISSING {P.PERIOD}: {P.NO_EXE}"]
    assert [f.name for f in out.iterdir()] == ["missing.json"]
    assert json.loads((out / "missing.json").read_text()) == {"protocol": P.PERIOD, "why": P.NO_EXE}
    assert C.case_dirs([out]) == []


def test_a_period_piece_takes_no_device(tmp_path):
    """The period names no device, so the piece takes no registration head or place."""
    with pytest.raises(SystemExit):
        P.main(["period", "--out", str(tmp_path / "p"), "--head", "ab" * 64, "--place", "gpu0"])


def _period_args(tmp_path: Path, d: dict, windows: list[dict]) -> list[str]:
    return ["--key", str(_json(tmp_path / "k.json", d["key"])), "--main", str(_lines(tmp_path / "main.txt", d["main"])),
            "--name", d["name"],
            "--ledger", str(_lines(tmp_path / "ledger.jsonl", d["ledger"])),
            "--pins", str(_lines(tmp_path / "pins.jsonl", d["pins"])),
            "--windows", str(_json(tmp_path / "w.json", windows))]


def test_a_period_of_statements_is_put_in_the_lean_form_by_the_verifier_of_record(tmp_path, statements, period_exe,
                                                                                   capsys):
    """The kit's windows are over PouwH1Tile_v1, whose per-unit noise the period's guarantee doesn't cover, so the
    case credit accepts is refused at its form, as a served window over h1zy's default tile is."""
    d, windows = statements._period(0, tmp_path / "proofs")
    out, (credit, why) = P.proved(d["key"], d["ledger"], d["pins"], d["main"], d["name"], windows)
    assert why is None and credit["windows"] == [statements._credited(w) for w in windows]
    assert [sorted(w) for w in out] == [list(Cr.PROVED_KEYS)] * len(windows)
    assert all(p["verdict"] == "accept" for w in out for p in w["proofs"] + w["live"]["proofs"])
    assert Cr._period_outcomes(d["key"], d["ledger"], d["pins"], d["main"], d["name"], out, CC.lms_verify) == (
        credit, None)
    piece = tmp_path / "period"
    assert PC.form(d["name"], out).startswith("window 0 of period") and "PouwH1Tile_v1" in PC.form(d["name"], out)
    assert P.main(["period", "--out", str(piece), "--exe", str(period_exe), *_period_args(tmp_path, d, windows)]) == 1
    said = capsys.readouterr().out.splitlines()
    assert said == [f"REFERENCE {P.PERIOD} accept", f"PIECE {P.PERIOD} reject form by=lean"]
    assert json.loads(bytes.fromhex(json.loads((piece / "case.json").read_text())["evidence"]))["windows"] == out


def test_a_rejected_live_session_is_an_outcome_the_lean_form_rejects(tmp_path, statements, period_exe, capsys):
    d, windows = statements._period(1, tmp_path / "proofs")
    p = Path(windows[0]["live"]["statements"][0]["sessions"][0]) / "session.json"
    rec = json.loads(p.read_text())
    rec["proof"] = ("0" if rec["proof"][0] != "0" else "1") + rec["proof"][1:]
    p.write_text(json.dumps(rec))
    out, (credit, why) = P.proved(d["key"], d["ledger"], d["pins"], d["main"], d["name"], windows)
    assert credit is None and why[0] == "window"
    assert [x["verdict"] for x in out[0]["live"]["proofs"]][:1] == ["reject"]
    assert Cr._period_outcomes(d["key"], d["ledger"], d["pins"], d["main"], d["name"], out, CC.lms_verify)[1][0] == (
        "window")
    assert P.main(["period", "--out", str(tmp_path / "period"), "--exe", str(period_exe),
                   *_period_args(tmp_path, d, windows)]) == 1
    assert capsys.readouterr().out.splitlines() == [f"REFERENCE {P.PERIOD} reject window",
                                                    f"PIECE {P.PERIOD} reject form by=lean"]


def test_statements_naming_the_same_proof_files_twice_are_refused(tmp_path, statements):
    d, windows = statements._period(0, tmp_path / "proofs")
    w = {**windows[0], "statements": windows[0]["statements"] + windows[0]["statements"][:1]}
    with pytest.raises(ValueError, match="name the same proof files"):
        P.proved(d["key"], d["ledger"], d["pins"], d["main"], d["name"], [w, *windows[1:]])


def test_a_deployment_key_file_is_the_key_the_period_pins():
    doc = {"H": 20, "W": 1, "pubkey": "ab" * 8, "host": "vy-challenger-1", "pubkey_sha512": "cd" * 64}
    assert P.challenger_key(doc) == {"H": 20, "W": 1, "public": "ab" * 8}


def test_a_served_h1_window_is_missing_from_both_with_why(tmp_path, capsys):
    assert P.main(["v1", "--out", str(tmp_path / "v1"), "--serve-run", str(SERVED), "--window", "0"]) == 2
    assert P.main(["period", "--out", str(tmp_path / "period"), "--serve-run", str(SERVED), "--window", "0"]) == 2
    said = capsys.readouterr().out
    assert f"MISSING {P.V1}: window 0's PoUW is H1's per-call credit (pouw:eps1-sm120-h1fx1)" in said
    assert f"MISSING {P.PERIOD}: window 0's session s20261027T201610Z-c43d is in no period" in said
    for d, protocol in ((tmp_path / "v1", P.V1), (tmp_path / "period", P.PERIOD)):
        assert [f.name for f in d.iterdir()] == ["missing.json"]
        assert json.loads((d / "missing.json").read_text())["protocol"] == protocol
        assert C.case_dirs([d]) == []
    assert C.main(["remote", "--cases", str(tmp_path / "period"), "--private-out", str(tmp_path / "priv"),
                   "--public-out", str(tmp_path / "pub")]) == 0
    assert f"NOT_ESTABLISHED missing verity/inference-only/v0, {P.PERIOD}, verity/sampled-proofs/v1\n" in \
        (tmp_path / "pub" / "verdicts.txt").read_text()


def test_a_rerun_that_goes_missing_leaves_no_stale_case(tmp_path):
    case = pouw_check_cases.instance(random.Random(0), "honest")
    reg, ans = (json.loads(bytes.fromhex(case[k])) for k in ("commitment", "evidence"))
    out = tmp_path / "pouw"
    P.main(["v1", "--out", str(out), "--params", str(_json(tmp_path / "p.json", case["params"])),
            "--registration", str(_json(tmp_path / "r.json", reg)), "--answer", str(_json(tmp_path / "a.json", ans))])
    assert P.main(["v1", "--out", str(out), "--serve-run", str(SERVED), "--window", "0"]) == 2
    assert [f.name for f in out.iterdir()] == ["missing.json"]


def test_a_served_period_is_its_windows_session_then_its_samples(tmp_path):
    """The period's close lists the served window's session and then its sample's (``--cap 2``), and credit reads the
    sample from the sample's session: a record naming no sample session, or a close listing other sessions, is
    missing with why."""
    sid, dsid, serve, pouw = "s-window", "s-sample", tmp_path / "serve", tmp_path / "pouw"
    (serve / "h1/w0").mkdir(parents=True)
    _lines(serve / "windows.jsonl", [json.dumps({"window": 0, "pouw_h1": {"session": {"session": sid, "period": "p"}}})])
    _json(serve / "period.json", {"name": "p", "close": {}})
    for s in (sid, dsid):
        (pouw / "session" / s / "out").mkdir(parents=True)
    (pouw / "session" / sid / "out" / "key.json").write_text("{}")
    _lines(pouw / "pins.jsonl", [])
    args = (serve, 0, pouw, "00" * 64, tmp_path / "main.txt", None, None)

    def say(record: dict, listed: list[str]) -> str:
        _json(serve / "h1/w0/pouw-h1-calls.json", {"commitment": record})
        _lines(pouw / "ledger.jsonl", [json.dumps({"kind": "period-close", "body": {
            "period": "p", "windows": [{"session": s} for s in listed]}})])
        with pytest.raises(P.Missing) as e:
            P.period_inputs(*args)
        return str(e.value)

    assert "names no sample session" in say({"session": {"session": sid}}, [sid])
    both = {"session": {"session": sid}, "draw_session": {"session": dsid}}
    assert f"lists the sessions ['{sid}'], and this piece is built for a period" in say(both, [sid])
    assert "no declared window Program" in say(both, [sid, dsid])


def test_a_window_the_run_lacks_is_missing(tmp_path):
    assert "has no window 3" in P.v1_missing(SERVED, 3)
    assert "has no windows.jsonl" in P.period_missing(tmp_path, 0)


def test_the_exe_is_the_answer_of_record(tmp_path):
    exe = tmp_path / "pouw-check"
    line = {"commitment": None, "profile": None, "protocol": P.V1, "public": None, "reason": "stub", "verdict": "reject"}
    exe.write_text(f"#!/bin/sh\necho \"$@\" > {tmp_path}/argv\necho '{json.dumps(line)}'\nexit 1\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    case = pouw_check_cases.instance(random.Random(0), "honest")
    reg, ans = (json.loads(bytes.fromhex(case[k])) for k in ("commitment", "evidence"))
    out = tmp_path / "pouw"
    assert P.main(["v1", "--out", str(out), "--exe", str(exe),
                   "--params", str(_json(tmp_path / "p.json", case["params"])),
                   "--registration", str(_json(tmp_path / "r.json", reg)),
                   "--answer", str(_json(tmp_path / "a.json", ans))]) == 1
    assert json.loads((out / "answer.json").read_text())["reason"] == "stub"
    assert (tmp_path / "argv").read_text().split() == ["check", *(str(out / f) for f in
                                                                  ("params.json", "commitment.bin", "evidence.json"))]
