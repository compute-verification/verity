"""The batch rule of PROTOCOL.md 8c "Batch verification" (lane b-batch-bound) in the Python verifier: a batch of
sub-batch proofs is accepted iff every proof verifies, the union of the verifier's OWN per-proof bounds over the N
presented is <= 2^-target, and every proof's n_proofs is N.  ``protocol.batch_bound`` is the rule (arithmetic; the
reason strings are shared with archive/ligero-verify); ``serialize.verify_batch_dir`` is the file-level batch verifier,
exercised on copies of the fetched B = 1 fixture (archive/ligero-verify/fixtures, l = 256, ZK interactive, sized
for n_proofs = 1 at 2^-128.16) exactly as tests/fixture.rs exercises the Rust one.  Needs torch (the Python verifier)
-- skipped on a laptop without it; the fixture verifications run on the CPU (a few seconds each)."""
from __future__ import annotations

import math
import struct
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("blake3")

from archive.direct.ligero import protocol, serialize  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[3] / "ligero-verify" / "fixtures"
N_PROOFS_OFF = 42       # ligero-statement/v2: magic 8 | layout 4 | l n D t 16 | target 8 | zk 1 | t_pad 4 | mode 1 | n_proofs


def test_batch_bound_is_the_union_bound_with_the_shared_reason_strings():
    # four equal bounds: the batch is 2 bits weaker; n_proofs must be the count presented; the target is a parameter
    b = [-130.2, -130.2, -130.2, -130.2]
    r = protocol.batch_bound(b, [4, 4, 4, 4], 128)
    assert r["accepted"] and abs(r["batch_log2"] - (-128.2)) < 1e-9
    assert r["reason"] == "accept: all 4 sub-batches accepted; union bound 2^-128.20 <= 2^-128"
    r = protocol.batch_bound(b, [4, 4, 4, 4], 128.5)
    assert not r["accepted"]
    assert r["reason"] == "batch: union bound 2^-128.20 over 4 sub-batches exceeds the target 2^-128.50"
    # the finding: N proofs each sized for one (2^-132.9 each) pass on their own; 25 of them are 2^-128.26, and the
    # claim n_proofs = 1 is rejected before the bound is even consulted
    r = protocol.batch_bound([-132.9] * 25, [1] * 25, 128, names=[f"sub_{i:02d}.proof" for i in range(25)])
    assert not r["accepted"]
    assert r["reason"] == "batch: n_proofs mismatch (sub_00.proof: n_proofs = 1, 25 sub-batches presented)"
    assert abs(r["batch_log2"] - (-132.9 + math.log2(25))) < 1e-9
    r = protocol.batch_bound([-132.9] * 25, [25] * 25, 128)
    assert r["accepted"] and r["reason"].startswith("accept: all 25 sub-batches accepted; union bound 2^-128.26")
    assert not protocol.batch_bound([-132.9] * 25, [25] * 25, 128.3)["accepted"]
    # sized for a larger batch is another batch's proof; heterogeneous bounds sum as they should
    r = protocol.batch_bound([-129.0, -129.0, -140.0], [3, 3, 4], 128)
    assert r["reason"] == "batch: n_proofs mismatch (sub_02: n_proofs = 4, 3 sub-batches presented)"
    r = protocol.batch_bound([-129.0, -129.0, -140.0], [3, 3, 3], 128)
    assert not r["accepted"], "2 x 2^-129 + 2^-140 is just over 2^-128"
    r = protocol.batch_bound([-129.5, -129.5, -140.0], [3, 3, 3], 128)
    assert r["accepted"] and -128.51 < r["batch_log2"] < -128.49
    with pytest.raises(ValueError):
        protocol.batch_bound([], [], 128)


def _copies(tmp_path: Path, name: str, n: int, claimed: int) -> Path:
    assert (FIXTURES / "b1.proof").is_file(), "b1.proof is a fetched fixture: `uv run research data fetch-fixtures archive/ligero-verify`"
    st = bytearray((FIXTURES / "b1.stmt").read_bytes())
    assert struct.unpack_from("<I", st, N_PROOFS_OFF)[0] == 1, "the fixture was proved at n_proofs = 1"
    struct.pack_into("<I", st, N_PROOFS_OFF, claimed)
    pf = (FIXTURES / "b1.proof").read_bytes()
    d = tmp_path / name
    d.mkdir()
    for i in range(n):
        (d / f"sub_{i:02d}.stmt").write_bytes(bytes(st))
        (d / f"sub_{i:02d}.proof").write_bytes(pf)
        (d / f"sub_{i:02d}.coins").write_bytes(pf[108:236])          # the transcript's own step-0 openings
    return d


@pytest.mark.slow
def test_python_file_batch_verifier_enforces_the_rule(tmp_path):
    # the statement's n_proofs is not in the statement digest: relabelling it leaves the proof valid, so copies of the
    # fixture are a batch of independently accepted sub-batches whose accounting is whatever the rule says
    d4 = _copies(tmp_path, "claim4", 4, 4)
    res = serialize.verify_batch_dir(d4, "cpu", 128.0)
    assert res["n_accepted"] == 4 and all(it["accepted"] for it in res["items"]), res
    assert not res["accepted"], "four 2^-128.16 proofs are a 2^-126.16 batch"
    assert res["reason"].startswith("batch: union bound 2^-126.") and res["reason"].endswith("over 4 sub-batches exceeds the target 2^-128")
    per = res["per_proof_log2"]
    assert len(per) == 4 and 128.1 < -per[0] < 128.2
    assert abs(res["batch_log2"] - (per[0] + 2.0)) < 1e-9
    assert all(it["own_coins"] for it in res["items"])
    # at a target the batch meets it is accepted, with the bits
    ok = serialize.verify_batch_dir(d4, "cpu", 126.0)
    assert ok["accepted"] and ok["reason"].startswith("accept: all 4 sub-batches accepted; union bound 2^-126.")
    assert ok["batch_bits"] == -ok["batch_log2"] and abs(ok["batch_bits"] - res["batch_bits"]) < 1e-12
    # (c) four proofs each claiming n_proofs = 1 presented together: the Rust verifier's exact string
    d1 = _copies(tmp_path, "claim1", 4, 1)
    res = serialize.verify_batch_dir(d1, "cpu", 126.0)
    assert res["n_accepted"] == 4 and not res["accepted"]
    assert res["reason"] == "batch: n_proofs mismatch (sub_00.proof: n_proofs = 1, 4 sub-batches presented)"
    # a rejected sub-batch takes precedence over the accounting
    p = d1 / "sub_01.proof"
    pf = bytearray(p.read_bytes())
    pf[-100] ^= 1
    p.write_bytes(bytes(pf))
    res = serialize.verify_batch_dir(d1, "cpu", 126.0)
    assert not res["accepted"] and res["reason"].startswith("sub_01.proof: ") and res["n_rejected"] == 1
