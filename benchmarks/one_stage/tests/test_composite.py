"""composite.py: onsite's and remote's case from their subprotocols' cases and answers, in the formats the composition's
Lean reads (params and evidence canonical, the subprotocols in the order of their ids, the commitment `bind` of their
digests then the registration's), and the product's check run on it."""

from __future__ import annotations

import hashlib
import json
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import composite as C

V2 = "verity/sampled-proofs/one-stage/v2"
POUW = "verity/pouw/v1"


def put(d: Path, id: str, params: dict, commitment: bytes, evidence: bytes, digest: bytes | None,
        blobs: list[str] | None = None, verdict: str = "undecided", reason: str | None = None) -> Path:
    d.mkdir(parents=True)
    case = {"commitment": commitment.hex(), "evidence": evidence.hex(), "params": params}
    (d / "case.json").write_text(json.dumps(case | ({"blobs": blobs} if blobs else {})))
    (d / "answer.json").write_text(json.dumps({"commitment": digest.hex() if digest else None, "protocol": id,
                                               "reason": reason, "verdict": verdict}))
    return d


def onsite_pieces(tmp_path: Path) -> list[str]:
    """A piece for each of onsite's subprotocols, as ``--piece`` arguments."""
    args = []
    for n, id in enumerate(C.COMPONENTS["onsite"]):
        args += ["--piece", str(put(tmp_path / f"p{n}", id, {}, bytes([n]), b"{}", bytes([n]) * 64))]
    return args


def test_remote_case_orders_by_id_and_binds_the_digests(tmp_path):
    v2 = put(tmp_path / "v2", V2, {"delta": "1/100"}, b"\x01" * 3, b'{"e":1}', b"\xaa" * 64, ["/c.txt", "/p.bin"])
    pw = put(tmp_path / "pouw", POUW, {"k": 2}, b"\x02", b"{}", b"\xbb" * 64)
    assert C.main(["remote", "--piece", str(v2), "--piece", str(pw), "--out", str(tmp_path / "out")]) == 0
    out = tmp_path / "out"
    assert (out / "params.json").read_bytes() == \
        b'{"components":[{"id":"verity/pouw/v1","params":{"k":2}},{"id":"verity/sampled-proofs/one-stage/v2",' \
        b'"params":{"delta":"1/100"}}]}'
    body = ('{"components":[{"commitment":"' + "bb" * 64 + '","id":"verity/pouw/v1"},{"commitment":"' + "aa" * 64 +
            '","id":"verity/sampled-proofs/one-stage/v2"}]}')
    assert (out / "commitment.bin").read_bytes() == hashlib.sha512(b"verity/remote/v0\0" + body.encode()).digest()
    e = json.loads((out / "evidence.json").read_bytes())
    assert [x["id"] for x in e["components"]] == [POUW, V2]
    assert e["components"][1] == {"commitment": "010101", "evidence": b'{"e":1}'.hex(), "id": V2,
                                  "params": {"delta": "1/100"}}
    assert (out / "blobs.txt").read_text() == "/c.txt\n/p.bin\n"


def test_onsite_appends_the_registrations_digest_and_case(tmp_path):
    nci = put(tmp_path / "nci", C.REGISTRATION, {"n": 1}, b"\x03", b"{}", b"\xcc" * 64, verdict="accept")
    assert C.main(["onsite", *onsite_pieces(tmp_path), "--registration", str(nci), "--out", str(tmp_path / "out")]) == 0
    out = tmp_path / "out"
    c = (out / "commitment.bin").read_bytes()
    assert len(c) == 128 and c[64:] == b"\xcc" * 64
    assert json.loads((out / "params.json").read_bytes())["registration"] == {"id": C.REGISTRATION, "params": {"n": 1}}
    assert json.loads((out / "evidence.json").read_bytes())["registration"] == \
        {"commitment": "03", "evidence": b"{}".hex(), "id": C.REGISTRATION, "params": {"n": 1}}


def test_a_case_whose_commitment_did_not_decode_stands_in_its_own_commitment(tmp_path):
    pw = put(tmp_path / "pouw", POUW, {}, b"\x05\x06", b"{}", None)
    assert C.piece(pw)["digest"] == b"\x05\x06"


def lean(tmp_path: Path, line: dict, rc: int) -> str:
    """A stand-in for the product's Lean executable: prints ``line`` and its arguments, and exits ``rc``."""
    exe = tmp_path / "lean-check"
    exe.write_text(f"#!{sys.executable}\nimport json, sys\n"
                   f"print(json.dumps({line!r} | {{'args': sys.argv[1:]}}))\nsys.exit({rc})\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return str(exe)


def answer(out: Path) -> dict:
    return json.loads((out / "answer.json").read_text())


MISSING_POUW = {"protocol": "verity/remote/v0", "verdict": "reject", "reason": "params: verity/pouw/v1 is missing"}


def test_a_usage_error_exits_64_not_lean_s_undecided(tmp_path):
    v2 = put(tmp_path / "v2", V2, {}, b"\x01", b"{}", b"\xaa" * 64)
    with pytest.raises(SystemExit) as e:
        C.main(["remote", "--piece", str(v2)])
    assert e.value.code == C.USAGE


def test_a_missing_piece_is_left_out_and_lean_s_verdict_stands(tmp_path, capsys):
    v2 = put(tmp_path / "v2", V2, {}, b"\x01", b"{}", b"\xaa" * 64, verdict="accept")
    out = tmp_path / "out"
    assert C.main(["remote", "--piece", str(v2), "--out", str(out), "--exe", lean(tmp_path, MISSING_POUW, 1)]) == 1
    assert [c["id"] for c in json.loads((out / "params.json").read_bytes())["components"]] == [V2]
    assert answer(out)["reason"] == "params: verity/pouw/v1 is missing"
    lines = capsys.readouterr().out.splitlines()
    assert "NOT_ESTABLISHED missing verity/pouw/v1" in lines
    assert any(ln.startswith("PIECE ") and '"accept"' in ln for ln in lines)
    assert any(ln.startswith("VERDICT ") and "is missing" in ln for ln in lines)


def test_the_exit_code_is_lean_s_whatever_the_pieces_say(tmp_path):
    v2 = put(tmp_path / "v2", V2, {}, b"\x01", b"{}", b"\xaa" * 64, verdict="reject", reason="bad")
    pw = put(tmp_path / "pouw", POUW, {}, b"\x02", b"{}", b"\xbb" * 64, verdict="reject", reason="bad")
    for rc in (0, 1, 2):
        out = tmp_path / f"out{rc}"
        assert C.main(["remote", "--piece", str(v2), "--piece", str(pw), "--out", str(out),
                       "--exe", lean(tmp_path, {"verdict": "x"}, rc)]) == rc


def test_a_piece_without_its_answer_is_missing(tmp_path, capsys):
    v2 = put(tmp_path / "v2", V2, {}, b"\x01", b"{}", b"\xaa" * 64)
    pw = put(tmp_path / "pouw", POUW, {}, b"\x02", b"{}", b"\xbb" * 64)
    (pw / "answer.json").unlink()
    assert C.main(["remote", "--piece", str(v2), "--piece", str(pw), "--out", str(tmp_path / "out")]) == 0
    assert "NOT_ESTABLISHED missing verity/pouw/v1" in capsys.readouterr().out.splitlines()
    assert [c["id"] for c in json.loads((tmp_path / "out" / "params.json").read_bytes())["components"]] == [V2]


def test_onsite_without_its_registration_leaves_it_out_for_lean_to_refuse(tmp_path, capsys):
    assert C.main(["onsite", *onsite_pieces(tmp_path), "--out", str(tmp_path / "a")]) == 0
    assert "registration" not in json.loads((tmp_path / "a" / "params.json").read_bytes())
    assert len((tmp_path / "a" / "commitment.bin").read_bytes()) == 64
    assert "NOT_ESTABLISHED missing verity/nci/v1" in capsys.readouterr().out.splitlines()


def test_the_check_runs_the_products_executable_on_the_files_and_blobs(tmp_path):
    v2 = put(tmp_path / "v2", V2, {}, b"\x01", b"{}", b"\xaa" * 64, ["/blob"])
    pw = put(tmp_path / "pouw", POUW, {}, b"\x02", b"{}", b"\xbb" * 64)
    exe = tmp_path / "remote"
    exe.write_text(f"#!{sys.executable}\nimport json, sys\n"
                   "print(json.dumps({'args': sys.argv[1:], 'verdict': 'undecided'}))\nsys.exit(2)\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    out = tmp_path / "out"
    assert C.main(["remote", "--piece", str(v2), "--piece", str(pw), "--out", str(out), "--exe", str(exe)]) == 2
    got = json.loads((out / "answer.json").read_text())
    assert got["args"] == ["check", str(out / "params.json"), str(out / "commitment.bin"), str(out / "evidence.json"),
                           "/blob"]
