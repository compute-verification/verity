"""The Ajtai / Ring-SIS leaf (``leaf/ajtai.py``) behind ``hashchain.compose(rel, leaf)``: the matrix derivation, the
native digest against a scalar reference, native == committer == in-circuit digest, the ``fp8-ada+ajtai-n64`` chain proved
and verified on the CPU, the v5 statement round trip with the wider digest table, and the leaf negatives (digest of a
different row, one bit flipped, non-binary s).  CPU only, two VUs."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("blake3")

from .. import hashauth, hashchain  # noqa: E402
from ..compile import P  # noqa: E402
from ..relations import relation  # noqa: E402
from ..relchain import HashedRelationRunner, hashed_negatives, instances, leaf_negatives  # noqa: E402
from ..serialize import STMT_MAGIC_V5, _statement_of, read_statement, statement_bytes, verify_files  # noqa: E402
from .ajtai import B_TAG, AjtaiLeaf, derive_matrix, derive_ring_element, digest_rows, shift_x  # noqa: E402
from .registry import join_relation, leaf_for, split_relation  # noqa: E402


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
    assert d.shape == (3, n)
    for r in range(3):
        assert d[r].tolist() == _ref_digest(leaf, rows[r].tolist(), 8)
    # torch path == numpy path
    dt = leaf.native(torch.as_tensor(rows), word_bits=8, role=0)
    assert np.array_equal(dt.numpy(), d)
    # states are consistent with the transition and end at the digest
    acc_in, acc_out = digest_rows(rows, 8, leaf.B, 256)
    assert np.array_equal(acc_out[:, -1], d) and not acc_in[:, 0].any()
    assert np.array_equal(acc_in[:, 1:], acc_out[:, :-1])
    # column order matters (the ring shift): swapping two columns changes the digest
    sw = rows.copy(); sw[:, :32], sw[:, 32:64] = rows[:, 32:64], rows[:, :32]
    assert not np.array_equal(leaf.native(sw, word_bits=8, role=0), d)
    # leaf bytes: n x LE u32
    lb = leaf.leaf_bytes(d[0])
    assert len(lb) == 4 * n and np.frombuffer(lb, dtype="<u4").astype(np.int64).tolist() == d[0].tolist()


def test_shift_x_is_multiplication_by_x() -> None:
    v = np.arange(1, 9, dtype=np.int64)
    s = shift_x(v)
    assert s.tolist() == [(P - 8) % P, 1, 2, 3, 4, 5, 6, 7]
    assert torch.equal(shift_x(torch.as_tensor(v)), torch.as_tensor(s))


def test_registry_and_relation_strings() -> None:
    assert leaf_for(None) is None and leaf_for("hash") is None and leaf_for("poseidon2") is None
    a = leaf_for("ajtai")
    assert isinstance(a, AjtaiLeaf) and a.n == 64 and a.tag == "ajtai-n64"
    assert leaf_for("ajtai-n128").n == 128
    with pytest.raises(ValueError):
        leaf_for("ajtai-n96")                                          # reducible ring degree (design §8)
    with pytest.raises(ValueError):
        leaf_for("sha3")
    assert split_relation("fp8-ada+ajtai-n64") == ("fp8-ada", "ajtai-n64") and split_relation("fp8-ada") == ("fp8-ada", None)
    assert join_relation("fp8-ada", a) == "fp8-ada+ajtai-n64" and join_relation("fp8-ada", None) == "fp8-ada"
    assert a.schema == "ajtai-babybear-n64-b1-r0/row/v1" and a.salt_bits == 0 and len(a.params_sha256) == 64
    assert a.params_sha256 != AjtaiLeaf(128).params_sha256


def test_compose_census_and_pins() -> None:
    hr = hashchain.compose(relation("fp8-ada"), AjtaiLeaf(64))
    c = hr.census
    assert (c["rows_base"], c["rows_private_operands"], c["rows_hash"], c["rows_digest_pins"]) == (3769, 704, 4 * 64, 2 * 64 + 1)
    assert c["linked_rows"] == 3 + 2 * 64 and c["digest_elems"] == 64
    assert sum(1 for L in hr.sys.linear if L.name.startswith("hash.") and ".acc_out[" in L.name) == 2 * 64
    assert hr.n_digest == 128 and hr.relation_name == "fp8-ada+ajtai-n64" and hr.lanes == 64
    assert [n for _, n in hr.sys.pins] == [hashchain.HASH_PIN_END] + [hashchain.hash_pin(x) for x in range(128)]
    assert len(hr.sys.chain["c"]) == len(hr.sys.chain["y"]) == 131
    # Poseidon2 unchanged
    h0 = hashchain.compose(relation("fp8-ada"))
    assert h0.n_digest == 16 and h0.relation_name == "fp8-ada" and h0.leaf is None
    # the ring degree must cover the VU's columns
    with pytest.raises(ValueError, match="ring degree"):
        hashchain.compose(relation("fp8-ada"), AjtaiLeaf(32))
    with pytest.raises(ValueError, match="ring degree"):
        hashchain.compose(relation("bf16-hopper"), AjtaiLeaf(64))


@pytest.fixture(scope="module")
def runner():
    rel = relation("fp8-ada")
    R = HashedRelationRunner(rel, "cpu", -128.0, zk=False, mode="fiat-shamir", leaf=leaf_for("ajtai-n64"))
    data = instances(rel, 4)
    R.commit_vus(data)
    return R, data


def test_prove_verify_and_committer_digest(runner) -> None:
    R, data = runner
    assert R.relation == "fp8-ada+ajtai-n64" and R.sponges is None
    proof, pubs, lay = R.prove_vus(data[:2], check=True, vu_ids=[0, 1])
    table, idx, ypub, auth, sibs = pubs
    assert table.shape == (lay.l, 1 + 128)
    ends = np.arange(2) * lay.steps + lay.steps - 1
    assert np.array_equal(table[ends, 1:65], R.digests["a"][:2]) and np.array_equal(table[ends, 65:], R.digests["b"][:2])
    assert R.verify_vus(proof, pubs, lay) == (True, "")
    # the committer's leaf on the words the prover used, independently
    leaf = R.leaf
    a0 = np.asarray(data[0][0])[None]
    assert leaf.native(a0, word_bits=8, role=0)[0].tolist() == table[ends[0], 1:65].tolist()
    assert R.trees["a"].ref.count == 4
    lp = hashauth.leaf_params(leaf)
    assert lp.schema == leaf.schema and lp.lanes == 64 and lp.leaf_bytes(table[ends[0], 1:65]) == leaf.leaf_bytes(R.digests["a"][0])


def test_statement_v5_round_trip_and_file_verifier(runner, tmp_path) -> None:
    R, data = runner
    proof, pubs, lay = R.prove_vus([data[1], data[3]], check=False, vu_ids=[1, 3])
    st = _statement_of(R, pubs, lay, R.cfg(lay.l), 1)
    raw = statement_bytes(st)
    assert raw[:8] == STMT_MAGIC_V5
    st2 = read_statement(raw)
    assert st2.v5 and st2.relation == "fp8-ada+ajtai-n64" and st2.digests().shape == (2, 128)
    assert np.array_equal(st2.a, st.a) and st2.hash_auth == st.hash_auth and st2.hash_sibs == st.hash_sibs
    assert b"ajtai-babybear-n64-b1-r0/row/v1" in raw and bytes.fromhex(R.leaf.params_sha256) in raw
    from ..serialize import proof_bytes
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
    assert not ok and "relation" in why
    # the statement's digest width follows the relation string: a different n is unreadable
    bad = raw.replace(b"fp8-ada+ajtai-n64", b"fp8-ada+ajtai-n32")
    with pytest.raises(ValueError):
        read_statement(bad)


def test_negatives_rejected(runner) -> None:
    R, data = runner
    negs = hashed_negatives(R, data, row_negatives=3) + leaf_negatives(R, data)
    accepted = [name for name, (ok, _) in negs if ok]
    assert accepted == [], accepted
    names = " | ".join(n for n, _ in negs)
    for must in ("digest of a different row", "one bit of the hashed x row flipped", "non-binary operand bit", "same word value",
                 "accumulator output row tampered", "ajtai state link broken", "computed on the flipped row"):
        assert must in names, must
