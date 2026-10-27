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
        blobs: list[str] | None = None) -> Path:
    d.mkdir(parents=True)
    case = {"commitment": commitment.hex(), "evidence": evidence.hex(), "params": params}
    (d / "case.json").write_text(json.dumps(case | ({"blobs": blobs} if blobs else {})))
    (d / "answer.json").write_text(json.dumps({"commitment": digest.hex() if digest else None, "protocol": id,
                                               "verdict": "undecided"}))
    return d


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
    v2 = put(tmp_path / "v2", V2, {}, b"\x01", b"{}", b"\xaa" * 64)
    nci = put(tmp_path / "nci", C.REGISTRATION, {"n": 1}, b"\x03", b"{}", b"\xcc" * 64)
    C.main(["onsite", "--piece", str(v2), "--registration", str(nci), "--out", str(tmp_path / "out")])
    out = tmp_path / "out"
    c = (out / "commitment.bin").read_bytes()
    assert len(c) == 128 and c[64:] == b"\xcc" * 64
    assert json.loads((out / "params.json").read_bytes())["registration"] == {"id": C.REGISTRATION, "params": {"n": 1}}
    assert json.loads((out / "evidence.json").read_bytes())["registration"] == \
        {"commitment": "03", "evidence": b"{}".hex(), "id": C.REGISTRATION, "params": {"n": 1}}


def test_a_case_whose_commitment_did_not_decode_stands_in_its_own_commitment(tmp_path):
    pw = put(tmp_path / "pouw", POUW, {}, b"\x05\x06", b"{}", None)
    assert C.piece(pw)["digest"] == b"\x05\x06"


def test_onsite_needs_the_registration_and_remote_takes_none(tmp_path):
    v2 = put(tmp_path / "v2", V2, {}, b"\x01", b"{}", b"\xaa" * 64)
    with pytest.raises(SystemExit):
        C.main(["onsite", "--piece", str(v2), "--out", str(tmp_path / "a")])
    with pytest.raises(SystemExit):
        C.main(["remote", "--piece", str(v2), "--registration", str(v2), "--out", str(tmp_path / "b")])


def test_the_check_runs_the_products_executable_on_the_files_and_blobs(tmp_path):
    v2 = put(tmp_path / "v2", V2, {}, b"\x01", b"{}", b"\xaa" * 64, ["/blob"])
    exe = tmp_path / "remote"
    exe.write_text(f"#!{sys.executable}\nimport json, sys\n"
                   "print(json.dumps({'args': sys.argv[1:], 'verdict': 'undecided'}))\nsys.exit(2)\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    out = tmp_path / "out"
    assert C.main(["remote", "--piece", str(v2), "--out", str(out), "--exe", str(exe)]) == 2
    got = json.loads((out / "answer.json").read_text())
    assert got["args"] == ["check", str(out / "params.json"), str(out / "commitment.bin"), str(out / "evidence.json"),
                           "/blob"]
