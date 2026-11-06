"""The bandwidth audit's piece (`served_bandwidth.sh`, BANDWIDTH_CMD) as `served_verdicts.py pieces` runs it, on the toy's
P2 store on the CPU (P2's reference decode) against `p2_party --listen` on loopback in the auditor gate's party shape:
the honest session's answer is a core accept that `piece_row` reads, with its credited reads, eps, δ and γ, and the
scope beside the served window; a prover that serves later passes from its cache of the first's, reading nothing of
the store, is rejected (`receipt`); and a session that left no record, asked about a window that wasn't served, is a
reject that says both. ``collect``, the composed run's piece, reads the answer the PoUS piece's session wrote, which
`pieces` has run first; it names the audit missing where there is none, and refuses another window's or protocol's."""
from __future__ import annotations

import json
import os
import re
import sys
from fractions import Fraction
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks" / "one_stage"))
sys.path.insert(0, str(ROOT / "benchmarks" / "pous"))
sys.path.insert(0, str(ROOT / "examples" / "h200" / "interfaces"))
import served_bandwidth as SB  # noqa: E402
import served_verdicts as S  # noqa: E402
from p2_v1 import bandwidth as B  # noqa: E402
from verity_e2e_demo import block_record as BR  # noqa: E402

from verity.core.protocols.remote_bandwidth import p2, p2_party  # noqa: E402

CMD = 'bash benchmarks/one_stage/served_bandwidth.sh "$1" "$2" "$3"'


def _serve_dir(path: Path) -> Path:
    """A serve dir whose window 0 is sealed, its Commit passed over 3 steps, with one checked request."""
    path.mkdir()
    (path / "seals.jsonl").write_text(json.dumps({"window": 0, "requests": ["u0"]}) + "\n")
    (path / "windows.jsonl").write_text(json.dumps({"window": 0, "passed": True, "commit": {"steps": 3}}) + "\n")
    (path / "requests.jsonl").write_text(json.dumps({"user": "u0", "windows": [0], "check": {"passed": True}}) + "\n")
    return path


def _env(tmp: Path, cheat: str = "") -> dict[str, str]:
    toy = tmp / "toy"
    party = (f"cd {ROOT} && exec {sys.executable} -S -E -m verity.core.protocols.remote_bandwidth.p2_party "
             f"--device {tmp / 'party'} --listen 127.0.0.1:0 --registration {toy / 'registration.json'} "
             "--windows 1 --passes 2 --samples 64 --window-ns 1000000 --deadline-ns 1000000")
    return dict(os.environ) | {"BANDWIDTH_CMD": CMD, "PY": sys.executable, "BW_STORE": str(toy / "store"),
            "BW_LAYOUT": str(toy / "layout.json"), "BW_REGISTRATION": str(toy / "registration.json"),
            "BW_NODE": "p2-toy-node", "BW_DEVICE": "cpu", "BW_REF_DECODE": "1", "BW_LEASE_MIN": "0", "BW_WORKERS": "1",
            "BW_WINDOWS": "1", "BW_PASSES": "2", "BW_SAMPLES": "64", "BW_WINDOW_S": "0.001", "BW_DEADLINE_S": "0.001",
            "BW_PARTY": party, "BW_PARTY_HOST": "127.0.0.1", "BW_R_BYTES_PER_S": "1000000000", "BW_CHEAT": cheat}


def test_the_honest_session_is_a_core_accept_and_the_cache_cheat_a_reject(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    reg = B.toy(tmp_path / "toy")
    serve = _serve_dir(tmp_path / "serve")

    got = S.pieces(serve, 0, tmp_path / "honest", env=_env(tmp_path))
    assert got["bandwidth"]["verdict"] == "accept", (tmp_path / "honest" / "bandwidth" / "piece.log").read_text()
    row = S.piece_row("bandwidth", tmp_path / "honest")
    a = row["answer"]
    assert row["verdict"] == "accept" and a["commitment"] == reg.digest.hex() and a["reason"] is None
    q = row["quantity"]
    assert q["q_bytes"] == (a["profile"]["receipts"] - a["profile"]["wrong"]) * p2.UNIT_BYTES > 0
    assert a["profile"]["receipts"] == 2 * reg.blocks and row["delta"] == str(Fraction(1, 2**40))
    assert Fraction(q["eps"]) == Fraction(1, 2**40) + Fraction(sum(a["profile"]["draws"]), 2**64)
    assert 0 < Fraction(q["gamma"]) == 1 - Fraction(q["q_bytes"] * 10**9, 10**9 * q["span_ns"]) < 1
    assert a["profile"]["scope"].startswith("beside the served window") and a["profile"]["gamma_terms"]
    assert BR._quantity("bandwidth", row)[0] == {"symbol": "Q_cert", "value": q["q_bytes"], "unit": "bytes"}
    served = json.loads((tmp_path / "honest" / "bandwidth" / "served.json").read_text())
    assert served["served"] == {"window": 0, "serve_dir": str(serve), "steps": [0, 3], "requests": 1}
    kept = [json.loads(x) for x in (tmp_path / "party" / "answers.jsonl").read_text().splitlines()]
    assert p2_party.core_answer(kept[-1], 10**9)["public"] == {k: q[k] for k in ("eps", "gamma", "q_bytes",
                                                                                  "r_bytes_per_s", "span_ns")}
    assert not (tmp_path / "honest" / "bandwidth" / "scratch").exists()

    got = S.pieces(serve, 0, tmp_path / "cheat", env=_env(tmp_path, "cache"))
    row = S.piece_row("bandwidth", tmp_path / "cheat")
    assert got["bandwidth"]["verdict"] == row["verdict"] == "reject"
    assert "receipt at window 0, draw [1, " in row["reason"] and "beside the served window" in row["reason"]
    assert row["quantity"] is None and row["answer"]["public"] is None


COLLECT = 'python benchmarks/one_stage/served_bandwidth.py collect --serve-dir "$1" --window "$2" --out "$3"'
HELD = {"commitment": "ab" * 64, "protocol": p2_party.PROTOCOL, "reason": None, "verdict": "accept",
        "public": {"eps": "1/1099511627776", "gamma": "1/2", "q_bytes": 8224, "r_bytes_per_s": 1000, "span_ns": 16448},
        "profile": {"delta": "1/1099511627776", "draws": [0], "receipts": 1, "wrong": 0, "units": 1,
                    "scope": "beside the served window"}}


def _held(tmp: Path, window: int, answer: dict = HELD) -> Path:
    """What the PoUS piece's bandwidth session leaves in the bandwidth piece's directory (`held_bandwidth.session`)."""
    d = tmp / "held"
    d.mkdir(exist_ok=True)
    (d / "answer.json").write_text(json.dumps(answer))
    (d / "bandwidth.json").write_text(json.dumps({"window": window, "verdict": answer["verdict"]}))
    return d


def test_the_pous_piece_runs_first_and_the_bandwidth_piece_collects_what_its_session_wrote(tmp_path, monkeypatch):
    """`pieces` starts PoUS's piece before the bandwidth audit's, so the answer PoUS's piece wrote to the bandwidth
    piece's directory is there when ``collect`` reads it, and the record reads it as the bandwidth audit's."""
    monkeypatch.chdir(ROOT)
    serve, held = _serve_dir(tmp_path / "serve"), _held(tmp_path, 0)
    env = dict(os.environ) | {"POUS_CMD": f'mkdir -p "$3/../bandwidth" && cp {held}/*.json "$3/../bandwidth/"',
                              "BANDWIDTH_CMD": COLLECT.replace("python", sys.executable, 1)}
    got = S.pieces(serve, 0, tmp_path / "pieces", env=env)
    assert got["bandwidth"] == {"verdict": "accept", "rc": 0}
    log = (tmp_path / "pieces" / "bandwidth" / "piece.log").read_text()
    assert log.startswith("BANDWIDTH ") and json.loads(log.split(" ", 1)[1])["public"] == HELD["public"]
    row = S.piece_row("bandwidth", tmp_path / "pieces")
    assert row["verdict"] == "accept" and row["quantity"]["q_bytes"] == 8224
    assert BR._quantity("bandwidth", row)[0] == {"symbol": "Q_cert", "value": 8224, "unit": "bytes"}


def test_collect_names_the_audit_missing_where_the_pous_piece_wrote_nothing(tmp_path, capsys):
    out = tmp_path / "pieces" / "bandwidth"
    assert SB.main(["collect", "--serve-dir", str(tmp_path), "--window", "3", "--out", str(out)]) == 2
    assert capsys.readouterr().out.startswith("MISSING the bandwidth audit: the PoUS piece wrote no bandwidth answer")
    assert not (out / "answer.json").exists()
    row = S.piece_row("bandwidth", tmp_path / "pieces")
    assert row["verdict"] == "missing" and row["owner"] == S.PIECES["bandwidth"][2]


def test_collect_replaces_another_windows_answer_and_errs_on_another_protocols(tmp_path, capsys):
    out = _held(tmp_path, 2)
    assert SB.main(["collect", "--serve-dir", str(tmp_path), "--window", "3", "--out", str(out)]) == 1
    a = json.loads((out / "answer.json").read_text())
    assert a["verdict"] == "undecided" and "is window 2's, not window 3's" in a["reason"] and S.check_answer(a)
    out = _held(tmp_path, 3, HELD | {"protocol": "verity/pous/p2/v4/run"})
    assert SB.main(["collect", "--serve-dir", str(tmp_path), "--window", "3", "--out", str(out)]) == 1
    assert "not the remote bandwidth audit's" in capsys.readouterr().out.splitlines()[-1]


def test_a_session_with_no_record_is_a_reject_that_says_its_window_was_not_served(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    out = tmp_path / "pieces" / "bandwidth"
    assert SB.main(["answer", "--session", str(out / "session" / "session.json"), "--serve-dir", str(tmp_path / "none"),
                    "--window", "0", "--out", str(out)]) == 1
    row = S.piece_row("bandwidth", tmp_path / "pieces")
    assert row["verdict"] == "reject" and row["quantity"] is None and "no session record" in row["reason"]
    assert "with no served window: window 0 of " in row["reason"] and "beside the served window" not in row["reason"]
    assert json.loads((out / "served.json").read_text())["scope"].startswith("with no served window")


def test_the_served_pieces_request_is_the_auditors_session() -> None:
    """`served_bandwidth.sh` requests, by default, the session the auditor's party holds every session to
    (`p2_party.SESSION`, the held session's in the PoUS piece), as `bandwidth.py session` turns its flags into the
    request: one off it is refused before any coin."""
    text = (ROOT / "benchmarks" / "one_stage" / "served_bandwidth.sh").read_text()
    got = dict(re.findall(r"\$\{BW_(WINDOWS|PASSES|SAMPLES|WINDOW_S|DEADLINE_S):-([0-9.]+)\}", text))
    assert {"windows": int(got["WINDOWS"]), "passes": int(got["PASSES"]), "samples": int(got["SAMPLES"]),
            "window_ns": int(float(got["WINDOW_S"]) * 1e9),
            "deadline_ns": int(float(got["DEADLINE_S"]) * 1e9)} == p2_party.SESSION
