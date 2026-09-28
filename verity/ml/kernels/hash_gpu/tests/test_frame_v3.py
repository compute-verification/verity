"""The GPU frame-v3 trees and row digests (``frame_v3.py``) byte for byte against :mod:`verity.ml.commitments`, hashlib and
the ``blake3`` package (GPU parts skip on a laptop)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # kernels

from hash_gpu import frame_v3 as fv3

try:
    import cupy as cp

    GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # noqa: BLE001
    cp, GPU = None, False

try:
    import blake3

    HAVE_BLAKE3 = True
except ImportError:
    HAVE_BLAKE3 = False

gpu = pytest.mark.skipif(not GPU, reason="needs a CUDA device and cupy")


def _domain(count: int, owner: int = -1, binding: bytes = b"\x11" * 32):
    from verity.ml.commitments import CommitmentDomain, RangeIndexedDomain
    return CommitmentDomain(binding, owner, RangeIndexedDomain(0, count))


def _reference_levels(domain, schema: str, values: list[bytes]) -> list[list[bytes]]:
    from verity.ml.commitments.merkle import _PAD, _hash, _uint, merkle_depth
    level = [domain.leaf(r, r, schema, v) for r, v in enumerate(values)]
    level += [_hash(_PAD, domain.domain_id, _uint(r)) for r in range(len(values), 1 << merkle_depth(len(values)))]
    out = [level]
    lv = 0
    while len(level) > 1:
        level = [domain.node(lv, i // 2, level[i], level[i + 1]) for i in range(0, len(level), 2)]
        out.append(level)
        lv += 1
    return out


def test_midstate_is_sha256_prefix_state():
    for n in (0, 17, 63, 64, 65, 81, 128, 200):
        data = bytes(range(256))[:n] if n <= 256 else b""
        mid, done, tail = fv3.midstate(data)
        assert done == n - n % 64 and tail == data[done:]
        if done:
            # the midstate of whole blocks, finished with the standard padding, is SHA-256 of those blocks
            st = tuple(int(x) for x in mid)
            pad = b"\x80" + b"\0" * 55 + (done * 8).to_bytes(8, "big")
            assert b"".join(x.to_bytes(4, "big") for x in fv3._compress(st, pad)) == hashlib.sha256(data[:done]).digest()


@gpu
def test_device_heads_are_the_host_midstates():
    datas = [bytes((7 * i + n) % 256 for i in range(n)) for n in (0, 1, 63, 64, 65, 97, 128, 200, 300)]
    H = fv3.heads(datas, b"xyz")
    assert H.extra_bytes() == b"xyz"
    for j, d in enumerate(datas):
        mid, done, tail = fv3.midstate(d)
        m, dn, t = H.host(j)
        assert np.array_equal(m, mid) and dn == done and t == tail
        h = H.head(j)
        assert int(h.done) == done and int(h.tail_len) == len(tail)


def test_head_is_the_domain_framing():
    from verity.ml.commitments.merkle import _FRAME, _NODE
    d = _domain(5)
    assert fv3.FRAME == _FRAME
    msg = fv3.head(_NODE, d.domain_id, fv3.uint(3)) + fv3.part(fv3.uint(7)) + fv3.part(b"L" * 32) + fv3.part(b"R" * 32)
    assert hashlib.sha256(msg).digest() == d.node(3, 7, b"L" * 32, b"R" * 32)


@gpu
@pytest.mark.parametrize("count", [1, 2, 3, 5, 64, 255, 256, 257, 1000, 4096])
def test_u16_word_tree_is_the_reference(count):
    rng = np.random.default_rng(count)
    words = rng.integers(0, 1 << 16, count, dtype=np.uint16)
    d = _domain(count)
    ref = _reference_levels(d, "u16", [int(w).to_bytes(2, "big") for w in words])
    t = fv3.build_tree(d.domain_id, count, "u16", cp.asarray(words), vmode=fv3.VALUE_U16)
    got = t.levels
    assert len(got) == len(ref)
    for lv, (g, r) in enumerate(zip(got, ref)):
        assert g.tobytes() == b"".join(r), f"level {lv}"
    assert t.root == ref[-1][0]


@gpu
@pytest.mark.parametrize("count", [1, 7, 300])
def test_u32_and_byte_value_trees_are_the_reference(count):
    rng = np.random.default_rng(100 + count)
    words = rng.integers(0, 1 << 32, count, dtype=np.uint32)
    d = _domain(count, owner=-2, binding=bytes(range(32)))
    ref = _reference_levels(d, "u32", [int(w).to_bytes(4, "big") for w in words])
    assert fv3.build_tree(d.domain_id, count, "u32", cp.asarray(words), vmode=fv3.VALUE_U32).root == ref[-1][0]
    vals = rng.integers(0, 256, (count, 32), dtype=np.uint8)
    schema = "blake3-keyed/row/v2"
    ref = _reference_levels(d, schema, [v.tobytes() for v in vals])
    t = fv3.build_tree(d.domain_id, count, schema, cp.asarray(vals), vlen=32)
    assert t.levels[0].tobytes() == b"".join(ref[0]) and t.root == ref[-1][0]


@gpu
def test_large_ranks_and_lazy_host_copy():
    count = (1 << 17) + 3                                     # ranks of 3 bytes, a tree past the eager-copy size
    rng = np.random.default_rng(7)
    words = rng.integers(0, 1 << 16, count, dtype=np.uint16)
    d = _domain(count)
    t = fv3.build_tree(d.domain_id, count, "u16", cp.asarray(words), vmode=fv3.VALUE_U16)
    assert t._host is None
    lv0 = t.levels[0]
    for r in (0, 255, 256, 65535, 65536, count - 1):
        assert lv0[r].tobytes() == d.leaf(r, r, "u16", int(words[r]).to_bytes(2, "big"))
    from verity.ml.commitments.merkle import _PAD, _hash, _uint
    assert lv0[count].tobytes() == _hash(_PAD, d.domain_id, _uint(count))
    lv5 = t.levels[5]
    i = 1234
    assert lv5[i].tobytes() == d.node(4, i, t.levels[4][2 * i].tobytes(), t.levels[4][2 * i + 1].tobytes())


@gpu
@pytest.mark.parametrize("prefix_len", [0, 5, 60, 64, 81, 130])
@pytest.mark.parametrize("nbytes", [1, 37, 64, 1536, 3072])
def test_sha256_rows_is_hashlib(prefix_len, nbytes):
    rng = np.random.default_rng(prefix_len * 7919 + nbytes)
    rows = rng.integers(0, 256, (33, nbytes), dtype=np.uint8)
    prefix = bytes(rng.integers(0, 256, prefix_len, dtype=np.uint8))
    got = cp.asnumpy(fv3.sha256_rows(cp.asarray(rows), prefix))
    for r in range(rows.shape[0]):
        assert got[r].tobytes() == hashlib.sha256(prefix + rows[r].tobytes()).digest()


@gpu
@pytest.mark.skipif(not HAVE_BLAKE3, reason="needs the blake3 package")
@pytest.mark.parametrize("nbytes", [0, 1, 2, 63, 64, 65, 1000, 1024, 1025, 1536, 2048, 3072, 4096, 5120, 8192 + 64, 9000])
def test_keyed_blake3_rows_is_the_package(nbytes):
    rng = np.random.default_rng(nbytes)
    rows = rng.integers(0, 256, (41, nbytes), dtype=np.uint8)
    key = b"verity/blake3-leaf/v1/x".ljust(32, b"\0")
    whole = nbytes and nbytes % 64 == 0
    out = fv3.blake3_keyed_rows(cp.asarray(rows), key, block_out=bool(whole))
    cvs, roots = out[0], cp.asnumpy(out[1]).astype("<u4")
    assert cvs.shape == (41, max(1, -(-nbytes // 1024)), 8)
    if whole:
        assert out[2].shape == (41, nbytes // 64, 8)
    for r in range(rows.shape[0]):
        assert roots[r].tobytes() == blake3.blake3(rows[r].tobytes(), key=key).digest()


def _core_vectors(name: str) -> dict:
    import json

    import verity.ml.commitments as vc
    return json.loads((Path(vc.__file__).parent / name / "vectors.json").read_text())


@gpu
def test_frame_v3_row_leaf_vectors():
    for v in _core_vectors("frame_v3")["row_leaves"]:
        row = cp.asarray(np.frombuffer(bytes.fromhex(v["row_bytes"]), dtype=np.uint8)[None, :])
        if v["hash"] == "sha256":
            got = cp.asnumpy(fv3.sha256_rows(row, bytes.fromhex(v["prefix"])))[0].tobytes()
        else:
            got = cp.asnumpy(fv3.blake3_keyed_rows(row, bytes.fromhex(v["key"]))[1]).astype("<u4")[0].tobytes()
        assert got.hex() == v["digest"], v


@gpu
def test_frame_v3_blake3_vectors():
    for v in _core_vectors("frame_v3")["blake3"]:
        msg = bytes(i % 251 for i in range(v["len"]))
        row = cp.asarray(np.frombuffer(msg, dtype=np.uint8).reshape(1, -1))
        got = cp.asnumpy(fv3.blake3_keyed_rows(row, bytes.fromhex(v["key"]))[1]).astype("<u4")[0].tobytes()
        assert got.hex() == v["keyed_hash"], v["len"]


@gpu
def test_frame_v3_row_tree_and_word_leaf_vectors():
    d = _core_vectors("frame_v3")
    for v in d["row_trees"]:
        dom = bytes.fromhex(v["domain"]["domain_id"])
        rows = np.stack([np.frombuffer(bytes.fromhex(r), dtype=np.uint8) for r in v["rows"]])
        if v["hash"] == "sha256":
            from verity.ml.commitments.rowleaf import sha256_row_prefix
            dg = fv3.sha256_rows(cp.asarray(rows), sha256_row_prefix(v["role"], v["word_bits"], v["n_words"]))
        else:
            from verity.ml.commitments.rowleaf import blake3_row_key
            dg = fv3.blake3_keyed_rows(cp.asarray(rows), blake3_row_key(v["role"]))[1].view(cp.uint8).reshape(-1, 32)
        assert [bytes(x).hex() for x in cp.asnumpy(dg)] == v["digests"]
        t = fv3.build_tree(dom, len(v["rows"]), v["schema"], dg, vlen=32)
        assert [x.tobytes().hex() for x in t.levels[0][:len(v["rows"])]] == v["leaves"]
        assert t.root.hex() == v["root"]
    for v in d["word_leaves"]:
        out = cp.empty((1, 32), dtype=cp.uint8)
        val = bytes.fromhex(v["value"])
        fv3.leaves_into(out, bytes.fromhex(v["domain"]["domain_id"]), v["schema"],
                        cp.asarray(np.frombuffer(val, dtype=np.uint8).reshape(1, -1)), vlen=len(val), start=v["position"])
        assert cp.asnumpy(out)[0].tobytes().hex() == v["leaf"]


@gpu
@pytest.mark.parametrize("n", [1, 2, 3, 5, 7, 8, 9, 1000, 4097])
def test_vllm_tree_is_the_reference(n):
    from verity.ml.commitments import leaves as L
    from verity.ml.commitments import vllm_v1
    rng = np.random.default_rng(n)
    vals = rng.integers(0, 256, (n, 24), dtype=np.uint8)
    lay = vllm_v1.pos_leaf_layout(24)
    lv = cp.asnumpy(fv3.sha256_rows(cp.asarray(vals), lay.prefix))
    ref_leaves = [vllm_v1.pos_leaf(v.tobytes()) for v in vals]
    assert [x.tobytes() for x in lv] == ref_leaves
    t = fv3.vllm_tree(cp.asarray(lv))
    ref = L.levels(ref_leaves)
    assert [[x.tobytes() for x in g] for g in t.levels] == ref
    assert t.root == vllm_v1.fold(ref_leaves)


@gpu
def test_vllm_pos_leaf_vectors():
    from verity.ml.commitments import vllm_v1
    for v in _core_vectors("vllm_v1")["pos_leaf"]:
        val = bytes.fromhex(v["value"])
        row = cp.asarray(np.frombuffer(val, dtype=np.uint8).reshape(1, -1))
        assert cp.asnumpy(fv3.sha256_rows(row, vllm_v1.pos_leaf_layout(len(val)).prefix))[0].tobytes().hex() == v["leaf"]
