"""pouw_piece.py: PoUW's pieces for `composite.py`, remote's ``verity/pouw/v1`` and onsite's ``verity/pouw/period/v1``,
each a case in its protocol's formats and its answer on it. An honest window (`pouw_check_cases`) and an honest period
(`pouw_period_cases`) written from their inputs accept, with the digest the product binds; a served H1 window
(``fixtures/served-h1-window``, window 0's line of a served H200 run's windows.jsonl, a development window in no
period) is MISSING from both, with why, and writes no case, so `composite.py` leaves it out and names it missing; the
directory is private as `composite.py` reads it; and ``--exe`` is the answer of record."""

from __future__ import annotations

import json
import random
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verity/core/protocols/pouw/lean/check/tests"))
import composite as C
import pouw_check_cases
import pouw_period_cases
import pouw_piece as P
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import period_check as PC

SERVED = Path(__file__).resolve().parent / "fixtures" / "served-h1-window"


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


def test_an_honest_period_is_a_period_piece_that_accepts(tmp_path):
    case = pouw_period_cases.instance(random.Random(0), "honest")
    p = case["params"]
    ledger = json.loads(bytes.fromhex(case["commitment"]))
    ev = json.loads(bytes.fromhex(case["evidence"]))
    out = tmp_path / "period"
    assert P.main(["period", "--out", str(out), "--key", str(_json(tmp_path / "k.json", p["key"])),
                   "--main", str(_lines(tmp_path / "main.txt", p["main"])), "--name", p["name"],
                   "--ledger", str(_lines(tmp_path / "ledger.jsonl", ledger)),
                   "--pins", str(_lines(tmp_path / "pins.jsonl", ev["pins"])),
                   "--windows", str(_json(tmp_path / "w.json", ev["windows"]))]) == 0
    got = json.loads((out / "answer.json").read_text())
    assert got["verdict"] == "accept" and got["protocol"] == P.PERIOD
    assert got["commitment"] == PC.digest(PC.commitment(bytes.fromhex(case["commitment"]))).hex()
    assert json.loads((out / "case.json").read_text())["evidence"] == case["evidence"]


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
    assert C.main(["remote", "--cases", str(tmp_path / "v1"), "--private-out", str(tmp_path / "priv"),
                   "--public-out", str(tmp_path / "pub")]) == 0
    assert f"NOT_ESTABLISHED missing verity/inference-only/v0, {P.V1}, verity/sampled-proofs/one-stage/v2\n" in \
        (tmp_path / "pub" / "verdicts.txt").read_text()


def test_a_rerun_that_goes_missing_leaves_no_stale_case(tmp_path):
    case = pouw_check_cases.instance(random.Random(0), "honest")
    reg, ans = (json.loads(bytes.fromhex(case[k])) for k in ("commitment", "evidence"))
    out = tmp_path / "pouw"
    P.main(["v1", "--out", str(out), "--params", str(_json(tmp_path / "p.json", case["params"])),
            "--registration", str(_json(tmp_path / "r.json", reg)), "--answer", str(_json(tmp_path / "a.json", ans))])
    assert P.main(["v1", "--out", str(out), "--serve-run", str(SERVED), "--window", "0"]) == 2
    assert [f.name for f in out.iterdir()] == ["missing.json"]


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
