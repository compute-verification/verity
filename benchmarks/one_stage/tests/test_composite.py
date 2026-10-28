"""composite.py: onsite's and remote's case from their subprotocols' private cases and answers, in the formats the
composition's Lean reads (params and evidence canonical, the subprotocols in the order of their ids, the commitment `bind`
of their digests then the registration's), the product's check run on it, and the output split so the public part holds
no case's evidence or params."""

from __future__ import annotations

import hashlib
import json
import os
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
    d.mkdir(parents=True, mode=0o700)
    case = {"commitment": commitment.hex(), "evidence": evidence.hex(), "params": params}
    (d / "case.json").write_text(json.dumps(case | ({"blobs": blobs} if blobs else {})))
    (d / "answer.json").write_text(json.dumps({"commitment": digest.hex() if digest else None, "protocol": id,
                                               "reason": reason, "verdict": verdict}))
    return d


def outs(tmp_path: Path, name: str = "") -> list[str]:
    return ["--private-out", str(tmp_path / f"private{name}"), "--public-out", str(tmp_path / f"public{name}")]


def onsite_cases(tmp_path: Path) -> list[str]:
    """A case for each of onsite's subprotocols, as ``--cases`` arguments."""
    args = []
    for n, id in enumerate(C.COMPONENTS["onsite"]):
        args += ["--cases", str(put(tmp_path / f"p{n}", id, {}, bytes([n]), b"{}", bytes([n]) * 64))]
    return args


def lean(tmp_path: Path, line: dict, rc: int) -> str:
    """A stand-in for the product's Lean executable: prints ``line`` and its arguments, and exits ``rc``."""
    exe = tmp_path / "lean-check"
    exe.write_text(f"#!{sys.executable}\nimport json, sys\n"
                   f"print(json.dumps({line!r} | {{'args': sys.argv[1:]}}))\nsys.exit({rc})\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return str(exe)


def test_remote_case_orders_by_id_and_binds_the_digests(tmp_path):
    v2 = put(tmp_path / "v2", V2, {"delta": "1/100"}, b"\x01" * 3, b'{"e":1}', b"\xaa" * 64, ["/c.txt", "/p.bin"])
    pw = put(tmp_path / "pouw", POUW, {"k": 2}, b"\x02", b"{}", b"\xbb" * 64)
    assert C.main(["remote", "--cases", str(v2), "--cases", str(pw), *outs(tmp_path)]) == 0
    out = tmp_path / "private"
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


def test_a_cases_dir_of_case_dirs_reads_each(tmp_path):
    put(tmp_path / "cases" / "v2", V2, {}, b"\x01", b"{}", b"\xaa" * 64)
    put(tmp_path / "cases" / "pouw", POUW, {}, b"\x02", b"{}", b"\xbb" * 64)
    assert C.main(["remote", "--cases", str(tmp_path / "cases"), *outs(tmp_path)]) == 0
    assert [c["id"] for c in json.loads((tmp_path / "private" / "params.json").read_bytes())["components"]] == [POUW, V2]


def test_onsite_takes_nci_s_case_as_its_registration(tmp_path):
    nci = put(tmp_path / "nci", C.REGISTRATION, {"n": 1}, b"\x03", b"{}", b"\xcc" * 64, verdict="accept")
    assert C.main(["onsite", *onsite_cases(tmp_path), "--cases", str(nci), *outs(tmp_path)]) == 0
    out = tmp_path / "private"
    c = (out / "commitment.bin").read_bytes()
    assert len(c) == 128 and c[64:] == b"\xcc" * 64
    assert json.loads((out / "params.json").read_bytes())["registration"] == {"id": C.REGISTRATION, "params": {"n": 1}}
    assert json.loads((out / "evidence.json").read_bytes())["registration"] == \
        {"commitment": "03", "evidence": b"{}".hex(), "id": C.REGISTRATION, "params": {"n": 1}}


def test_a_case_whose_commitment_did_not_decode_stands_in_its_own_commitment(tmp_path):
    pw = put(tmp_path / "pouw", POUW, {}, b"\x05\x06", b"{}", None)
    assert C.piece(pw)["digest"] == b"\x05\x06"


def test_the_public_output_holds_no_evidence_and_no_params_bodies(tmp_path):
    secret = os.urandom(300)
    params = {"seed": os.urandom(48).hex()}
    v2 = put(tmp_path / "v2", V2, params, b"\x01" * 64, secret, b"\xaa" * 64, verdict="accept")
    pw = put(tmp_path / "pouw", POUW, {}, b"\x02", b'{"opening":"' + secret.hex().encode() + b'"}', b"\xbb" * 64)
    assert C.main(["remote", "--cases", str(v2), "--cases", str(pw), *outs(tmp_path),
                   "--exe", lean(tmp_path, {"verdict": "accept"}, 0)]) == 0
    public = b"".join(f.read_bytes() for f in sorted((tmp_path / "public").iterdir()))
    for e in (secret, secret.hex().encode(), params["seed"].encode()):
        assert not any(e[i:i + 16] in public for i in range(len(e) - 15))
    record = json.loads((tmp_path / "public" / "composite.json").read_bytes())
    assert [(c["id"], c["params_sha512"]) for c in record["cases"]] == [
        (POUW, hashlib.sha512(b"{}").hexdigest()), (V2, hashlib.sha512(C.canonical(params)).hexdigest())]
    assert record["cases"][1]["commitment"] == "01" * 64
    priv = tmp_path / "private"
    assert stat.S_IMODE(priv.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(f.stat().st_mode) == 0o600 for f in priv.rglob("*") if f.is_file())
    assert secret.hex() in (priv / "evidence.json").read_text()


def test_a_piece_answer_quoting_evidence_is_refused_before_anything_public_is_written(tmp_path):
    secret = os.urandom(100)
    v2 = put(tmp_path / "v2", V2, {}, b"\x01", secret, b"\xaa" * 64, reason=f"unit opened {secret.hex()}")
    assert C.main(["remote", "--cases", str(v2), *outs(tmp_path)]) == C.LEAK
    assert not (tmp_path / "public").exists()


def test_the_private_out_may_not_lie_in_the_public_out(tmp_path):
    v2 = put(tmp_path / "v2", V2, {}, b"\x01", b"{}", b"\xaa" * 64)
    args = ["remote", "--cases", str(v2), "--public-out", str(tmp_path / "pub")]
    assert C.main([*args, "--private-out", str(tmp_path / "pub" / "x")]) == C.USAGE
    assert C.main([*args, "--private-out", str(tmp_path / "pub")]) == C.USAGE
    with pytest.raises(SystemExit) as e:
        C.main(["remote", "--cases", str(v2)])
    assert e.value.code == C.USAGE


def test_a_missing_case_is_left_out_and_lean_s_verdict_stands(tmp_path, capsys):
    v2 = put(tmp_path / "v2", V2, {}, b"\x01", b"{}", b"\xaa" * 64, verdict="accept")
    exe = lean(tmp_path, {"protocol": "verity/remote/v0", "verdict": "reject",
                          "reason": "params: verity/pouw/v1 is missing"}, 1)
    assert C.main(["remote", "--cases", str(v2), *outs(tmp_path), "--exe", exe]) == 1
    assert [c["id"] for c in json.loads((tmp_path / "private" / "params.json").read_bytes())["components"]] == [V2]
    lines = (tmp_path / "public" / "verdicts.txt").read_text().splitlines()
    assert lines[0].startswith("PIECE ") and '"accept"' in lines[0]
    assert lines[1] == "NOT_ESTABLISHED missing verity/pouw/v1"
    assert lines[2].startswith("VERDICT ") and "is missing" in lines[2]
    assert capsys.readouterr().out.splitlines()[1:4] == lines
    record = json.loads((tmp_path / "public" / "composite.json").read_bytes())
    assert record["missing"] == [POUW] and "is missing" in record["verdict"]


def test_the_exit_code_is_lean_s_whatever_the_pieces_say(tmp_path):
    v2 = put(tmp_path / "v2", V2, {}, b"\x01", b"{}", b"\xaa" * 64, verdict="reject", reason="bad")
    pw = put(tmp_path / "pouw", POUW, {}, b"\x02", b"{}", b"\xbb" * 64, verdict="reject", reason="bad")
    for rc in (0, 1, 2):
        assert C.main(["remote", "--cases", str(v2), "--cases", str(pw), *outs(tmp_path, str(rc)),
                       "--exe", lean(tmp_path, {"verdict": "x"}, rc)]) == rc


def test_a_case_without_its_answer_is_missing(tmp_path):
    v2 = put(tmp_path / "v2", V2, {}, b"\x01", b"{}", b"\xaa" * 64)
    pw = put(tmp_path / "pouw", POUW, {}, b"\x02", b"{}", b"\xbb" * 64)
    (pw / "answer.json").unlink()
    assert C.main(["remote", "--cases", str(v2), "--cases", str(pw), *outs(tmp_path)]) == 0
    assert "NOT_ESTABLISHED missing verity/pouw/v1" in (tmp_path / "public" / "verdicts.txt").read_text()


def test_onsite_without_its_registration_leaves_it_out_for_lean_to_refuse(tmp_path):
    assert C.main(["onsite", *onsite_cases(tmp_path), *outs(tmp_path)]) == 0
    assert "registration" not in json.loads((tmp_path / "private" / "params.json").read_bytes())
    assert len((tmp_path / "private" / "commitment.bin").read_bytes()) == 64
    assert "NOT_ESTABLISHED missing verity/nci/v1" in (tmp_path / "public" / "verdicts.txt").read_text()


def test_the_check_runs_the_products_executable_on_the_private_files_and_blobs(tmp_path):
    v2 = put(tmp_path / "v2", V2, {}, b"\x01", b"{}", b"\xaa" * 64, ["/blob"])
    pw = put(tmp_path / "pouw", POUW, {}, b"\x02", b"{}", b"\xbb" * 64)
    assert C.main(["remote", "--cases", str(v2), "--cases", str(pw), *outs(tmp_path),
                   "--exe", lean(tmp_path, {"verdict": "undecided"}, 2)]) == 2
    priv = tmp_path / "private"
    got = json.loads((priv / "answer.json").read_text())
    assert got["args"] == ["check", str(priv / "params.json"), str(priv / "commitment.bin"),
                           str(priv / "evidence.json"), "/blob"]
