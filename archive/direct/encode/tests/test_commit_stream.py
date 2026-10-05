"""Streaming commitment: bit-exact against the unfused path (encode with v2, hash with hash_gpu) and the CPU reference."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from encode import babybear as bb

cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:                          # noqa: BLE001 - pragma: no cover - laptop
    pytest.skip("no CUDA device", allow_module_level=True)

from hash_gpu import merkle

from encode.commit_stream import StreamingCommitter, reference_leaves
from encode.encode_ntt_simt import RSEncoderSIMT


def _unfused(x, hash):
    code = RSEncoderSIMT()(cp.asarray(x))
    t = merkle.commit(code, hash)
    return cp.asnumpy(t.levels[0]), cp.asnumpy(t.root)


@pytest.mark.parametrize("hash", ["sha256", "blake3"])
@pytest.mark.parametrize("rows,slice_rows,overlap,graph,keep", [
    (26, 256, False, False, False),          # < one slice, tail rows
    (512 + 16 * 3 + 5, 256, True, True, False),
    (1024 + 300, 512, True, True, True),
    (2048, 512, False, True, False),
])
def test_matches_unfused(hash, rows, slice_rows, overlap, graph, keep):
    x = bb.fill_rows(rows, bb.K_CODE, seed=rows)
    ref_leaves, ref_root = _unfused(x, hash)
    sc = StreamingCommitter(hash=hash, slice_rows=slice_rows, overlap=overlap, graph=graph, keep_codeword=keep)
    leaves, root = sc.commit(cp.asarray(x))
    cp.cuda.Device().synchronize()
    assert np.array_equal(cp.asnumpy(leaves), ref_leaves)
    assert np.array_equal(cp.asnumpy(root), ref_root)
    if keep:
        code = bb.encode_rows(x, bb.N_CODE, layout="coset")
        assert np.array_equal(cp.asnumpy(sc.store[:rows]).astype(np.uint64), code[:, bb.K_CODE:])
    # the graph is replayed on the same buffers: same answer
    _leaves2, root2 = sc.commit(cp.asarray(x))
    cp.cuda.Device().synchronize()
    assert np.array_equal(cp.asnumpy(root2), ref_root)


@pytest.mark.parametrize("hash", ["sha256", "blake3"])
def test_matches_cpu(hash):
    if hash == "blake3":
        pytest.importorskip("blake3")
    x = bb.fill_rows(37, bb.K_CODE, seed=7)
    ref = reference_leaves(x, hash)
    sc = StreamingCommitter(hash=hash, slice_rows=256 if hash == "blake3" else 16, overlap=True, graph=False)
    leaves, root = sc.commit(cp.asarray(x))
    cp.cuda.Device().synchronize()
    assert np.array_equal(cp.asnumpy(leaves), ref)
    # root = the hash_gpu tree over the leaves
    t = merkle.Tree(hash, 37, bb.N_CODE, [cp.asarray(ref)])
    lvl = t.levels[0]
    while lvl.shape[0] > 1:
        lvl = merkle.node_level(lvl, hash)
    assert np.array_equal(cp.asnumpy(root), cp.asnumpy(lvl[0]))
    if hash == "sha256":       # one leaf by hand
        col = np.ascontiguousarray(bb.encode_rows(x, bb.N_CODE)[:, 5000].astype("<u4")).tobytes()
        assert hashlib.sha256(col).digest() == ref[5000].astype(">u4").tobytes()
