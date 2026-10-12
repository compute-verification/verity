"""``commit_cost.py --impl gpu``: the same four schemes committed on the GPU (``verity/ml/kernels/hash_gpu/frame_v3.py``),
each root checked against the plain-Python reference's before its time counts.

Per rep, synchronised around each phase: ``h2d_s`` the leaf values to the device (pageable host memory), ``leaf_s``
the leaf digests (row digest + frame-v3 leaf, word leaf, or ``pos_leaf``), ``tree_s`` the tree (pads and nodes,
or node / lift) and the root binding.  ``commit_s = leaf_s + tree_s`` is the committer on device-resident values
(what serving pays when the values are already on its GPU); ``commit_h2d_s`` adds the copy.  Medians over the reps,
after one untimed warm-up (kernel compilation).  The opening check reads the device tree's host copy.
"""

from __future__ import annotations

import hashlib
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_SHARED = Path(__file__).resolve().parents[2] / "verity" / "ml" / "kernels"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))


def values_array(n: int, nbytes: int) -> np.ndarray:
    """``commit_cost.values(n, nbytes)`` as one (n, nbytes) uint8 array (the counter in the first 8 bytes, little-endian)."""
    block = np.frombuffer(hashlib.sha256(b"commit-cost").digest() * (nbytes // 32 + 1), dtype=np.uint8)
    out = np.empty((n, nbytes), dtype=np.uint8)
    head = min(8, nbytes)
    out[:, :head] = np.arange(n, dtype="<u8").view(np.uint8).reshape(n, 8)[:, :head]
    if nbytes > 8:
        out[:, 8:] = block[:nbytes - 8][None, :]
    return out


def _reference(impl, scheme: str, vals: np.ndarray) -> tuple[bytes, list[bytes]]:
    """The root and leaves of the plain reference (``CommitmentScheme``); BLAKE3 row digests from the ``blake3``
    package when importable (equal to the core's by the frame-v3 vectors), else the core's plain-Python BLAKE3."""
    n = vals.shape[0]
    rows = [vals[i].tobytes() for i in range(n)]
    if scheme == "frame-v3-blake3-row":
        lay = impl.leaf_layout("v", 0)
        try:
            import blake3
            digests = [blake3.blake3(r, key=lay.key).digest() for r in rows]
        except ImportError:
            digests = [lay.digest(r) for r in rows]
    else:
        digests = [impl.leaf_layout("v", i).digest(r) for i, r in enumerate(rows)]
    leaves = [impl.tree_leaf("v", i, d) for i, d in enumerate(digests)]
    return impl.root("v", leaves), leaves


def measure(scheme: str, n: int, nbytes: int, reps: int, setup) -> dict[str, Any]:
    import cupy as cp
    from hash_gpu import frame_v3 as fv3
    from verity.ml.commitments import vllm_v1
    from verity.ml.commitments.rowleaf import ROLE_X, blake3_row_key, sha256_row_prefix

    impl = setup(scheme, n, nbytes)
    host = values_array(n, nbytes)
    ref_root, ref_leaves = _reference(impl, scheme, host)
    sync = cp.cuda.Device().synchronize

    def commit(vals):
        if scheme == "vllm-v1":
            leaves = fv3.sha256_rows(vals, vllm_v1.pos_leaf_layout(nbytes).prefix)
            sync()
            t1 = time.perf_counter()
            tree = fv3.vllm_tree(leaves)
            root = impl.ports["v"].bind(tree.root)
            return t1, tree, root
        dom = impl.ports["v"].domain
        if scheme == "frame-v3":
            leaves = cp.empty((n, 32), dtype=cp.uint8)
            fv3.leaves_into(leaves, dom.domain_id, impl.ports["v"].leaf.schema, vals, vlen=nbytes)
        else:
            if scheme == "frame-v3-sha256-row":
                dg = fv3.sha256_rows(vals, sha256_row_prefix(ROLE_X, 16, nbytes // 2))
            else:
                dg = fv3.blake3_keyed_rows(vals, blake3_row_key(ROLE_X))[1].view(cp.uint8).reshape(n, 32)
            leaves = cp.empty((n, 32), dtype=cp.uint8)
            fv3.leaves_into(leaves, dom.domain_id, impl.ports["v"].leaf.schema, dg, vlen=32)
        sync()
        t1 = time.perf_counter()
        tree = fv3.frame_tree(dom.domain_id, leaves)
        return t1, tree, tree.root

    h2d_s, leaf_s, tree_s = [], [], []
    root, tree = b"", None
    for rep in range(1 + reps):
        sync()
        t0 = time.perf_counter()
        vals = cp.asarray(host)
        sync()
        ta = time.perf_counter()
        t1, tree, root = commit(vals)
        sync()
        t2 = time.perf_counter()
        if root != ref_root:
            raise AssertionError(f"{scheme}: GPU root {root.hex()} is not the reference root {ref_root.hex()}")
        if rep:
            h2d_s.append(ta - t0)
            leaf_s.append(t1 - ta)
            tree_s.append(t2 - t1)
        del vals
    lv0 = tree.levels[0]
    if [lv0[i].tobytes() for i in range(n)] != ref_leaves:
        raise AssertionError(f"{scheme}: GPU leaves are not the reference leaves")
    index = n // 2
    path = impl.path("v", ref_leaves, index)
    ours = tuple((None if s is None else tree.levels[lv][s].tobytes()) for lv, s in enumerate(_sibling_indices(scheme, n, index)))
    verified = ours == tuple(path) and impl.verify_path("v", root, n, index, ref_leaves[index], ours)
    leaf, tr, h2d = statistics.median(leaf_s), statistics.median(tree_s), statistics.median(h2d_s)
    return {"scheme": scheme, "hashes": list(impl.hashes), "leaves": n, "leaf_bytes": nbytes, "reps": reps, "leaf_s": leaf,
            "tree_s": tr, "commit_s": leaf + tr, "h2d_s": h2d, "commit_h2d_s": h2d + leaf + tr, "root": root.hex(),
            "reference_root": ref_root.hex(), "opening_verified": verified, "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}


def _sibling_indices(scheme: str, n: int, index: int) -> list[int | None]:
    """Per level, the sibling's index in the level (``None``: the node is lifted, vllm-v1's odd trailing child)."""
    out, width, cur = [], n, index
    if scheme.startswith("frame-v3"):
        width = 1 << max(0, (n - 1).bit_length())
    while width > 1:
        sib = cur ^ 1
        out.append(sib if sib < width else None)
        cur >>= 1
        width = (width + 1) // 2
    return out
