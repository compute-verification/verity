"""composite.py: onsite's and remote's case from their subprotocols' private cases and answers, in the formats the
composition's Lean reads (params and evidence canonical, the subprotocols in the order of their ids, the commitment `bind`
of their digests), the product's check run on it, and the output split so the public part holds no case's evidence or
params."""

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
POLICY = "verity/inference-only/v0"


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


def onsite_cases(tmp_path: Path, without: str | None = None) -> list[str]:
    """A case for each of onsite's subprotocols but ``without``, as ``--cases`` arguments."""
    args = []
    for n, id in enumerate(C.COMPONENTS["onsite"]):
        if id != without:
            args += ["--cases", str(put(tmp_path / f"p{n}", id, {}, bytes([n]), b"{}", bytes([n]) * 64))]
    return args


def policy(tmp_path: Path) -> Path:
    """The inference-only policy's case, accepted."""
    return put(tmp_path / "policy", POLICY, {"g": 1}, b"\x03", b"{}", b"\xcc" * 64, verdict="accept")


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
    assert C.main(["remote", "--cases", str(v2), "--cases", str(pw), "--cases", str(policy(tmp_path)),
                   *outs(tmp_path)]) == 0
    out = tmp_path / "private"
    assert (out / "params.json").read_bytes() == \
        b'{"components":[{"id":"verity/inference-only/v0","params":{"g":1}},{"id":"verity/pouw/v1","params":{"k":2}},' \
        b'{"id":"verity/sampled-proofs/one-stage/v2","params":{"delta":"1/100"}}]}'
    body = ('{"components":[{"commitment":"' + "cc" * 64 + '","id":"verity/inference-only/v0"},{"commitment":"' +
            "bb" * 64 + '","id":"verity/pouw/v1"},{"commitment":"' + "aa" * 64 +
            '","id":"verity/sampled-proofs/one-stage/v2"}]}')
    assert (out / "commitment.bin").read_bytes() == hashlib.sha512(b"verity/remote/v0\0" + body.encode()).digest()
    e = json.loads((out / "evidence.json").read_bytes())
    assert [x["id"] for x in e["components"]] == [POLICY, POUW, V2]
    assert e["components"][2] == {"commitment": "010101", "evidence": b'{"e":1}'.hex(), "id": V2,
                                  "params": {"delta": "1/100"}}
    assert (out / "blobs.txt").read_text() == "/c.txt\n/p.bin\n"


def test_a_cases_dir_of_case_dirs_reads_each(tmp_path):
    put(tmp_path / "cases" / "v2", V2, {}, b"\x01", b"{}", b"\xaa" * 64)
    put(tmp_path / "cases" / "pouw", POUW, {}, b"\x02", b"{}", b"\xbb" * 64)
    assert C.main(["remote", "--cases", str(tmp_path / "cases"), *outs(tmp_path)]) == 0
    assert [c["id"] for c in json.loads((tmp_path / "private" / "params.json").read_bytes())["components"]] == [POUW, V2]


def test_onsite_takes_the_policy_as_one_of_its_subprotocols(tmp_path):
    assert C.main(["onsite", *onsite_cases(tmp_path), *outs(tmp_path)]) == 0
    out = tmp_path / "private"
    assert len((out / "commitment.bin").read_bytes()) == 64
    params = json.loads((out / "params.json").read_bytes())
    assert list(params) == ["components"]
    assert [c["id"] for c in params["components"]] == list(C.COMPONENTS["onsite"])
    assert POLICY in C.COMPONENTS["onsite"] and POLICY in C.COMPONENTS["remote"]
    assert "NOT_ESTABLISHED" not in (tmp_path / "public" / "verdicts.txt").read_text()


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


def test_a_v2_answer_whose_query_is_a_digest_its_evidence_carries_stays_private(tmp_path):
    """v2's answer names its partition's digest as its profile's query, and v2's evidence carries that digest too; the
    public output holds the outcome only, so it passes the guard, and the answers are in the private directory whole."""
    partition = os.urandom(64).hex()
    evidence = json.dumps({"partition": partition, "sessions": ["s0"]}).encode()
    v2 = put(tmp_path / "v2", V2, {"partition": {"digest": partition}}, b"\x01" * 64, evidence, b"\xaa" * 64)
    profile = {"levels": [{"query": partition, "programs": [os.urandom(64).hex()], "units": 3}]}
    answer = {"commitment": "aa" * 64, "profile": profile, "protocol": V2, "public": None, "reason": None,
              "verdict": "accept"}
    (v2 / "answer.json").write_text(json.dumps(answer))
    pw = put(tmp_path / "pouw", POUW, {}, b"\x02", b"{}", b"\xbb" * 64, verdict="accept")
    exe = lean(tmp_path, {"profile": profile, "protocol": "verity/remote/v0", "verdict": "accept"}, 0)
    assert C.main(["remote", "--cases", str(v2), "--cases", str(pw), "--cases", str(policy(tmp_path)), *outs(tmp_path),
                   "--exe", exe]) == 0
    public = b"".join(f.read_bytes() for f in sorted((tmp_path / "public").iterdir()))
    for body in (partition.encode(), b"\"profile\"", b"\"query\"", b"\"programs\""):
        assert body not in public
    assert (tmp_path / "public" / "verdicts.txt").read_text().splitlines() == [
        f"PIECE {POLICY} accept", f"PIECE {POUW} accept", f"PIECE {V2} accept", "VERDICT verity/remote/v0 accept",
        f"EXCEPTION {C.EXCEPTIONS[POLICY]}"]
    assert partition in (tmp_path / "private" / "pieces" / "0.answer.json").read_text()
    assert partition in (tmp_path / "private" / "answer.json").read_text()


def test_a_refusal_is_public_by_its_name_only(tmp_path):
    secret = os.urandom(100)
    v2 = put(tmp_path / "v2", V2, {}, b"\x01", secret, b"\xaa" * 64, verdict="reject",
             reason=f"proofs: unit opened {secret.hex()}")
    pw = put(tmp_path / "pouw", POUW, {}, b"\x02", secret, b"\xbb" * 64, verdict="reject",
             reason=f"unit {secret.hex()[:32]} opened")
    assert C.main(["remote", "--cases", str(v2), "--cases", str(pw), "--cases", str(policy(tmp_path)),
                   *outs(tmp_path)]) == 0
    assert (tmp_path / "public" / "verdicts.txt").read_text().splitlines() == [
        f"PIECE {POLICY} accept", f"PIECE {POUW} reject refused", f"PIECE {V2} reject proofs",
        f"EXCEPTION {C.EXCEPTIONS[POLICY]}"]
    assert secret.hex()[:16].encode() not in b"".join(f.read_bytes() for f in (tmp_path / "public").iterdir())


def test_the_guard_refuses_public_bytes_that_share_a_run_with_evidence(tmp_path):
    secret = os.urandom(100)
    p = C.piece(put(tmp_path / "v2", V2, {}, b"\x01", secret, b"\xaa" * 64))
    with pytest.raises(C.Leak):
        C.check_public(b"x" + secret[10:80] + b"y", [p])
    with pytest.raises(C.Leak):
        C.check_public(secret.hex()[20:150].encode(), [p])
    C.check_public(b"aa" * 64 + secret[1:60], [p])


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
    assert C.main(["remote", "--cases", str(v2), "--cases", str(policy(tmp_path)), *outs(tmp_path), "--exe", exe]) == 1
    assert [c["id"] for c in json.loads((tmp_path / "private" / "params.json").read_bytes())["components"]] == \
        [POLICY, V2]
    lines = (tmp_path / "public" / "verdicts.txt").read_text().splitlines()
    assert lines == [f"PIECE {POLICY} accept", f"PIECE {V2} accept", "NOT_ESTABLISHED missing verity/pouw/v1",
                     "VERDICT verity/remote/v0 reject params", f"EXCEPTION {C.EXCEPTIONS[POLICY]}"]
    assert capsys.readouterr().out.splitlines()[1:6] == lines
    record = json.loads((tmp_path / "public" / "composite.json").read_bytes())
    assert record["missing"] == [POUW]
    assert record["verdict"] == {"id": "verity/remote/v0", "verdict": "reject", "refusal": "params"}
    assert record["exceptions"] == [C.EXCEPTIONS[POLICY]]
    assert "is missing" in (tmp_path / "private" / "answer.json").read_text()


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
    assert C.main(["remote", "--cases", str(v2), "--cases", str(pw), "--cases", str(policy(tmp_path)),
                   *outs(tmp_path)]) == 0
    assert "NOT_ESTABLISHED missing verity/pouw/v1\n" in (tmp_path / "public" / "verdicts.txt").read_text()


def test_onsite_without_the_policy_leaves_it_out_for_lean_to_refuse(tmp_path):
    assert C.main(["onsite", *onsite_cases(tmp_path, without=POLICY), *outs(tmp_path)]) == 0
    params = json.loads((tmp_path / "private" / "params.json").read_bytes())
    assert POLICY not in [c["id"] for c in params["components"]]
    assert "NOT_ESTABLISHED missing verity/inference-only/v0\n" in (tmp_path / "public" / "verdicts.txt").read_text()


def test_the_check_runs_the_products_executable_on_the_private_files_and_blobs(tmp_path):
    v2 = put(tmp_path / "v2", V2, {}, b"\x01", b"{}", b"\xaa" * 64, ["/blob"])
    pw = put(tmp_path / "pouw", POUW, {}, b"\x02", b"{}", b"\xbb" * 64)
    assert C.main(["remote", "--cases", str(v2), "--cases", str(pw), *outs(tmp_path),
                   "--exe", lean(tmp_path, {"verdict": "undecided"}, 2)]) == 2
    priv = tmp_path / "private"
    got = json.loads((priv / "answer.json").read_text())
    assert got["args"] == ["check", str(priv / "params.json"), str(priv / "commitment.bin"),
                           str(priv / "evidence.json"), "/blob"]
