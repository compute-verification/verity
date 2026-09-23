"""The Ajtai / Ring-SIS leaf (``leaf/ajtai.py``) behind ``hashchain.compose(rel, "ajtai-n<N>")``: the matrix derivation
(ajtai-design's regression vector, the pinned ``B`` digests the Rust verifier re-derives), the native digest against a
scalar ring reference, the registry family, the compose census and the ``n >= steps`` refusal -- torch-free -- and, with
torch, ``chain_witness`` == ``native``, the ``fp8-ada+ajtai-n64`` chain proved and verified on the CPU, the v5 statement
round trip with the wider digest table, and the leaf negatives (digest of a different row, one bit flipped, non-binary
s).  The generic seams (tree, statement, hashed negatives) are the conformance suite's: run it with
``VERITY_LEAF_CONFORMANCE=ajtai-n64,ajtai-n128``."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from .. import hashchain
from ..compile import P
from ..relations import relation
from . import registry
from .ajtai import (B_DIGEST_SHA256, B_TAG, AjtaiLeaf, derive_matrix, derive_ring_element, digest_rows, matrix_sha256,
                    resolve, shift_x)


def _ref_row(i: int, m: int) -> list[int]:
    """ajtai-design's `derive_ring_element`, scalar: SHAKE-256(B_TAG || LE32(i)), LE u32 words, & 0x7fffffff, reject >= p."""
    out, off, buf = [], 0, hashlib.shake_256(B_TAG + i.to_bytes(4, "little")).digest(8 * m)
    while len(out) < m:
        v = int.from_bytes(buf[off:off + 4], "little") & 0x7FFFFFFF
        off += 4
        if v < P:
            out.append(v)
    return out


def test_matrix_derivation_matches_the_scalar_reference_and_is_prefix_consistent() -> None:
    B = derive_matrix(64, 40)
    assert B.shape == (64, 40) and B.dtype == np.int64 and (B >= 0).all() and (B < P).all()
    for i in (0, 3, 39):
        assert B[:, i].tolist() == _ref_row(i, 64)                      # column i = the coefficients of b_i
    assert np.array_equal(derive_matrix(16, 8), B[:16, :8])            # prefix-consistent in n and in m
    assert np.array_equal(derive_ring_element(5, 20), derive_ring_element(5, 3000)[:20])
    assert len(set(map(tuple, B.T))) == 40                             # distinct ring elements
    # the design lane's regression vector (ajtai_params_test): b_0 for n = 64 straight from the XOF, no rejection missed
    xof = hashlib.shake_256(B_TAG + (0).to_bytes(4, "little")).digest(4096)
    ref = [v for v in (int.from_bytes(xof[k:k + 4], "little") & 0x7FFFFFFF for k in range(0, 4096, 4)) if v < P][:64]
    assert derive_ring_element(0, 64).tolist() == ref


def test_pinned_B_digests_are_the_derivation(n_values=(64, 128)) -> None:
    """``B_DIGEST_SHA256`` (what ``params_sha256`` absorbs and Rust's ``leaf.rs`` re-derives) is SHA-256 of B as n x 256
    row-major LE u32; a derivation change must change these constants (and every relation string's params digest)."""
    for n in n_values:
        B = derive_matrix(n, 256)
        assert matrix_sha256(B) == B_DIGEST_SHA256[n]
        assert hashlib.sha256(B.astype("<u4").tobytes()).hexdigest() == B_DIGEST_SHA256[n]
    assert np.array_equal(derive_matrix(64, 256), derive_matrix(128, 256)[:64])      # one family: n64 is a prefix of n128


def _ref_digest(leaf: AjtaiLeaf, row: list[int], word_bits: int) -> list[int]:
    """The ring definition, scalar: h = sum_j X^(steps-1-j) B s_j in F_p[X]/(X^n+1)."""
    n, m = leaf.n, leaf.m_col
    k = m // word_bits
    steps = len(row) // k
    acc = [0] * n
    for j in range(steps):
        bits = [(row[j * k + w] >> b) & 1 for w in range(k) for b in range(word_bits)]
        shifted = [(-acc[n - 1]) % P] + acc[:n - 1]
        acc = [(shifted[x] + sum(int(leaf.B[x, i]) * bits[i] for i in range(m))) % P for x in range(n)]
    return acc


@pytest.mark.parametrize("n", [16, 64])
def test_native_digest_equals_the_scalar_ring_reference(n: int) -> None:
    leaf = AjtaiLeaf(n, m_col=256)
    rng = np.random.default_rng(n)
    rows = rng.integers(0, 256, size=(3, 32 * 20), dtype=np.int64)           # 20 columns of 32 x 8-bit words
    d = leaf.native(rows, word_bits=8, role=0)
    assert d.shape == (3, n) and d.dtype == np.int64
    for r in range(3):
        assert d[r].tolist() == _ref_digest(leaf, rows[r].tolist(), 8)
    # states are consistent with the transition and end at the digest
    acc_in, acc_out = digest_rows(rows, 8, leaf.B, 256)
    assert np.array_equal(acc_out[:, -1], d) and not acc_in[:, 0].any()
    assert np.array_equal(acc_in[:, 1:], acc_out[:, :-1])
    # column order matters (the ring shift): swapping two columns changes the digest
    sw = rows.copy(); sw[:, :32], sw[:, 32:64] = rows[:, 32:64], rows[:, :32]
    assert not np.array_equal(leaf.native(sw, word_bits=8, role=0), d)
    # role is not absorbed (module docstring: separate tree domains)
    assert np.array_equal(leaf.native(rows, word_bits=8, role=2), d)
    # leaf bytes: n x LE u32, from an array, a list or a tuple
    lb = leaf.leaf_bytes(d[0])
    assert len(lb) == 4 * n and np.frombuffer(lb, dtype="<u4").astype(np.int64).tolist() == d[0].tolist()
    assert leaf.leaf_bytes(d[0].tolist()) == lb
    with pytest.raises(ValueError):
        leaf.leaf_bytes(np.append(d[0], 1))
    with pytest.raises(ValueError):
        leaf.leaf_bytes(np.where(np.arange(n) == 0, P, d[0]))
    # the chunked path (1024 rows per chunk) is one function of the row
    big = rng.integers(0, 256, size=(2050, 32 * 4), dtype=np.int64)
    dd = leaf.native(big, word_bits=8, role=0)
    assert np.array_equal(dd[[0, 1023, 1024, 2049]], digest_rows(big[[0, 1023, 1024, 2049]], 8, leaf.B, 256)[1][:, -1])


def test_shift_x_is_multiplication_by_x() -> None:
    v = np.arange(1, 9, dtype=np.int64)
    s = shift_x(v)
    assert s.tolist() == [(P - 8) % P, 1, 2, 3, 4, 5, 6, 7]


def test_registry_family_and_relation_strings() -> None:
    a = registry.get("ajtai")
    assert isinstance(a, AjtaiLeaf) and a.n == 64 and a.tag == "ajtai-n64" and a.name == "ajtai"
    assert registry.suffix_of(a) == "ajtai-n64" and registry.relation_name("fp8-ada", a) == "fp8-ada+ajtai-n64"
    assert registry.get("ajtai-n128").n == 128 and registry.get("ajtai-n128").carry_elems == 128
    assert registry.split_relation_name("fp8-ada+ajtai-n64") == ("fp8-ada", "ajtai-n64")
    assert "ajtai" in registry.families() and "ajtai" not in registry.names()
    for bad in ("ajtai-n96", "ajtai-n4", "ajtai-n1024", "ajtai-x", "ajtai-n"):        # reducible / out of range / malformed
        assert resolve(bad) is None
        with pytest.raises(KeyError):
            registry.get(bad)
    with pytest.raises(ValueError):
        AjtaiLeaf(96)
    with pytest.raises(NotImplementedError):
        AjtaiLeaf(64, salt_bits=8)
    assert a.schema == "ajtai-babybear-n64-b1-r0/row/v1" and a.hash_name == "ajtai-babybear-n64-b1-r0/v1" and a.salt_bits == 0
    assert len(a.params_sha256) == 64 and a.params_sha256 != AjtaiLeaf(128).params_sha256
    assert "binding-only" in a.privacy_note and "Ring-SIS" in a.assumption
    # the params digest absorbs the derived key: another B is another scheme
    assert B_DIGEST_SHA256[64] in __import__("backends.direct.ligero.leaf.ajtai", fromlist=["params_descriptor"]).params_descriptor(
        64, 256, 0, B_DIGEST_SHA256[64]).decode()


def test_compose_census_and_pins() -> None:
    hr = hashchain.compose(relation("fp8-ada"), "ajtai-n64")
    c = hr.census
    assert (c["rows_base"], c["rows_private_operands"], c["rows_hash"], c["rows_digest_pins"]) == (3769, 704, 4 * 64, 2 * 64 + 1)
    assert c["rows_total"] == 4858 and c["linked_rows"] == 3 + 2 * 64 and c["digest_elems"] == c["carry_elems"] == 64
    assert c["leaf"] == "ajtai" and c["ajtai_n"] == 64 and c["ajtai_linear_rows_per_unit"] == 128
    assert sum(1 for L in hr.sys.linear if L.name.startswith("hash.ajtai.") and ".out[" in L.name) == 2 * 64
    assert all(len(L.lin) == 2 + 256 for L in hr.sys.linear if L.name.startswith("hash.ajtai."))     # out, acc_in[x-1], 256 bits
    assert hr.n_digest == 128 and hr.name == "fp8-ada+ajtai-n64"
    assert [n for _, n in hr.sys.pins] == [hashchain.HASH_PIN_END] + [hashchain.hash_pin(x) for x in range(128)]
    assert len(hr.sys.chain["c"]) == len(hr.sys.chain["y"]) == 131
    names = {r.name for r in hr.sys.rows}
    assert {f"hash.a.cap[{x}]" for x in range(64)} <= names and {f"hash.ajtai.r2.out[{x}]" for x in range(64)} <= names
    # the ring degree must cover the VU's columns (48 for fp8-ada, 96 for bf16-hopper)
    with pytest.raises(ValueError, match="ring degree"):
        hashchain.compose(relation("fp8-ada"), "ajtai-n32")
    with pytest.raises(ValueError, match="ring degree"):
        hashchain.compose(relation("bf16-hopper"), "ajtai-n64")
    h = hashchain.compose(relation("bf16-hopper"), "ajtai-n128")
    assert h.census["linked_rows"] == 3 + 2 * 128 and h.name == "bf16-hopper+ajtai-n128"
    # Poseidon2 unchanged
    h0 = hashchain.compose(relation("fp8-ada"))
    assert h0.n_digest == 16 and h0.name == "fp8-ada" and h0.leaf.name == "poseidon2"


# --- torch ---------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def runner():
    pytest.importorskip("torch", exc_type=ImportError)
    pytest.importorskip("blake3", exc_type=ImportError)
    from ..relchain import HashedRelationRunner, instances
    rel = relation("fp8-ada")
    R = HashedRelationRunner(rel, "cpu", -128.0, zk=False, mode="fiat-shamir", leaf="ajtai-n64")
    data = instances(rel, 4)
    R.commit_vus(data)
    return R, data


def test_chain_witness_equals_native_on_the_device_path() -> None:
    torch = pytest.importorskip("torch", exc_type=ImportError)
    leaf = AjtaiLeaf(64)
    rng = np.random.default_rng(1)
    n_vus, steps, k, l = 3, 48, 32, 256
    rows = rng.integers(0, 256, size=(n_vus, steps * k), dtype=np.int64)
    words = torch.zeros((l, k), dtype=torch.int64)
    words[:n_vus * steps] = torch.as_tensor(rows.reshape(n_vus * steps, k))
    carry, dg, extras = leaf.chain_witness(words, word_bits=8, role=1, row_words=steps * k, n_vus=n_vus, steps=steps, l=l)
    assert carry.shape == (64, l) and dg.shape == (n_vus, 64) and set(extras) == {f"ajtai.r1.out[{x}]" for x in range(64)}
    assert np.array_equal(dg.numpy(), leaf.native(rows, word_bits=8, role=1))
    starts = torch.arange(n_vus) * steps
    assert not carry[:, starts].any() and not carry[:, n_vus * steps:].any()          # 0 entering a chain start / pad
    ends = starts + steps - 1
    outs = torch.stack([extras[f"ajtai.r1.out[{x}]"] for x in range(64)])
    assert torch.equal(outs[:, ends], dg.T) and torch.equal(outs[:, :n_vus * steps - 1][:, torch.arange(n_vus * steps - 1) % steps != steps - 1],
                                                            carry[:, 1:n_vus * steps][:, torch.arange(1, n_vus * steps) % steps != 0])
    # torch native == numpy native
    assert np.array_equal(leaf.native(torch.as_tensor(rows), word_bits=8, role=0).numpy(), leaf.native(rows, word_bits=8, role=0))
    assert torch.equal(shift_x(torch.arange(1, 9)), torch.as_tensor(shift_x(np.arange(1, 9))))


def test_prove_verify_and_committer_digest(runner) -> None:
    R, data = runner
    assert R.relation == "fp8-ada+ajtai-n64" and R.leaf_name == "ajtai-n64"
    proof, pubs, lay = R.prove_vus(data[:2], check=True, vu_ids=[0, 1])
    table, idx, ypub, auth, sibs = pubs
    assert table.shape == (lay.l, 1 + 128)
    ends = np.arange(2) * lay.steps + lay.steps - 1
    assert np.array_equal(table[ends, 1:65], R.digests["a"][:2]) and np.array_equal(table[ends, 65:], R.digests["b"][:2])
    assert R.verify_vus(proof, pubs, lay) == (True, "")
    # the committer's leaf on the words the prover used, independently
    a0 = np.asarray(data[0][0])[None]
    assert R.leaf.native(a0, word_bits=8, role=1)[0].tolist() == table[ends[0], 1:65].tolist()
    assert R.trees["a"].ref.count == 4


def test_statement_v5_round_trip_and_file_verifier(runner, tmp_path) -> None:
    from ..relchain import HashedRelationRunner
    from ..serialize import STMT_MAGIC_V5, _statement_of, proof_bytes, read_statement, statement_bytes, verify_files
    R, data = runner
    proof, pubs, lay = R.prove_vus([data[1], data[3]], check=False, vu_ids=[1, 3])
    st = _statement_of(R, pubs, lay, R.cfg(lay.l), 1)
    raw = statement_bytes(st)
    assert raw[:8] == STMT_MAGIC_V5
    st2 = read_statement(raw)
    assert st2.v5 and st2.relation == "fp8-ada+ajtai-n64" and st2.digests().shape == (2, 128)
    assert np.array_equal(st2.a, st.a) and st2.hash_auth == st.hash_auth and st2.hash_sibs == st.hash_sibs
    assert b"ajtai-babybear-n64-b1-r0/row/v1" in raw and bytes.fromhex(R.leaf.params_sha256) in raw
    (tmp_path / "s.stmt").write_bytes(raw)
    (tmp_path / "s.proof").write_bytes(proof_bytes(proof))
    ok, why, _ = verify_files(tmp_path / "s.stmt", tmp_path / "s.proof", "cpu", runner=R)
    assert ok, why
    # the file verifier resolves the runner from the relation string alone
    ok, why, _ = verify_files(tmp_path / "s.stmt", tmp_path / "s.proof", "cpu")
    assert ok, why
    # a Poseidon2 runner refuses the Ajtai statement
    P2 = HashedRelationRunner(relation("fp8-ada"), "cpu", -128.0, zk=False, mode="fiat-shamir")
    ok, why, _ = verify_files(tmp_path / "s.stmt", tmp_path / "s.proof", "cpu", runner=P2)
    assert not ok
    # the statement's digest width follows the relation string: another n does not read
    bad = raw.replace(b"fp8-ada+ajtai-n64", b"fp8-ada+ajtai-n32")
    with pytest.raises((ValueError, KeyError)):
        read_statement(bad)


def test_negatives_rejected(runner) -> None:
    from ..relchain import hashed_negatives, leaf_negatives
    R, data = runner
    negs = hashed_negatives(R, data, row_negatives=3) + leaf_negatives(R, data)
    accepted = [name for name, (ok, _) in negs if ok]
    assert accepted == [], accepted
    names = " | ".join(n for n, _ in negs)
    for must in ("digest of a different row", "one bit of the hashed x row flipped", "non-binary operand bit", "same word value",
                 "accumulator output row tampered", "leaf carry link broken", "nonzero leaf carry"):
        assert must in names, must
