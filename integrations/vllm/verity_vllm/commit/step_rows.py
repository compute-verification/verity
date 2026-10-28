"""The Commit's ``rows-sha512`` scheme: COMMIT-GPU's construction (``benchmarks/pouw/commit_gpu``) over a step's
stream, on the windowed worker (``NativeCollectCommitter``, whose unset scheme it is) and its offline CPU core
(``NativeHostCommitter.commit_block_offline``, by name).  Every Commit commits under it by default: the single-GPU Commit
(``pipeline.commit``'s ``--commit-scheme``, on the windowed ``native_collect_v2b``) and every rank of a TP Commit (``pipeline.tp.commit``,
under the SHA-512 TP run root ``pipeline.tp.run_root``); ``vllm-v1`` is by name (any other collector, an old record's configuration),
and a single-GPU Commit takes it only to commit again the recorded vllm-v1 Commit ``--of-record`` names (`pipeline.commit.scheme_refusal`).
Every rule is core's (``hm96.SHA512``, ``rowleaf``, ``merkle``, ``vllm_v1``); this module only names the step's pieces, and ``native_rows.cu`` hashes the same bytes (``tests/commit/test_step_rows.py`` pins this module to core's
conformance vectors, ``tests/commit/test_rows_cuda.py`` the kernels to this module).

Each ``ROW_BYTES``-byte chunk of a step's zero-padded stream is a row of ``8 ROW_BYTES`` bits, committed as a hiding leaf at its
chunk index ``pos`` (rank = position):

  x       ``SHA-512(P(n_bits) || row)``                                          sha512/row/v2
  y       ``SHA-512(Ks || u64be(pos) || u64be(r))`` for r = 0, 1, 2, concatenated  192 bytes
  b || c  ``hm96.SHA512.commit_string(default key, x, y)``
  h       ``hm96.SHA512.tree_leaf(default key, b || c)``
  leaf    the ``frame-v3-sha512`` row leaf of ``h``, schema ``hm96-sha512/row/v2``

``Ks`` is 64 fresh bytes from the OS generator per step, kept by the committer (a replay of the run takes the same keys) and never
logged; a leaf's salt ``y`` is derived from it by SHA-512 alone, as COMMIT-GPU derives it.  The step's ``N`` leaves are one tree under
``CommitmentDomain(binding, -1, RangeIndexedDomain(0, N), "sha512")`` with ``binding = identity_digest_sha512(DOMAIN_TAG, {program,
ctx, geo, row_bits})`` over the step's vllm-v1 identity digests, so the domain is known when the step's length is (a hit plan's, or
the step's end); the tree is core's ``tree_levels`` (padded to a power of two with frame pads).  The step root is ``vllm_v1.step_root``
over the tree root and the run root ``vllm_v1.run_root``, both under SHA-512: 64-byte roots.  An opening carries the row, its salt
and the frame path, from which ``h`` is recomputed and opened under the tree root by core's ``verify_opening``.  A range opening of
leaves ``[lo, hi)`` carries their rows and salts and, per level, the run's boundary siblings (:func:`range_shape`); :func:`fold_range`
folds them as core's ``verify_multiproof`` folds those leaves' multiproof, which is the path's siblings in order.

The registered weights are committed the same way (``NativeHostCommitter.register_weights``, which `pipeline.cpu_replay` opens
against): registered tensor ``t``'s bytes are its rows (the last zero-padded), salted from ``Ks_t = SHA-512(WEIGHTS_TAG || Kw ||
u64be t)`` under the registration's fresh key ``Kw``, in a tree under ``CommitmentDomain(binding_t, -1, RangeIndexedDomain(0, N_t),
"sha512")`` with ``binding_t = identity_digest_sha512(WEIGHTS_TAG, {geo, tensor: t, entry, row_bits})`` over its names entry
``[name, dtype, shape, bytes]``; the weights root is ``vllm_v1.weights_root`` over the tensor roots under SHA-512 (:func:`weights_root`).
The tree (:func:`weights_tree`) keeps ``Kw`` beside the roots, so a replay of the registration reproduces them.  ``Kw`` stays in the
private bundle: a published slim keep carries ``Ks_t`` of the tensors its replay opened instead (`pipeline.cpu_replay.slim_tree`).
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

from verity.ml.commitments import hm96
from verity.ml.commitments import vllm_v1 as V
from verity.ml.commitments.hm96 import batch as hm96_batch
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, merkle_depth, tree_levels
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2, row_leaf, sha512_row_v2_digest, sha512_row_v2_prefix

__all__ = ["DOMAIN_TAG", "INST", "KEY", "KEY_BYTES", "NAME", "OWNER", "ROOT_BYTES", "ROW_BYTES", "SALT_BYTES", "SCHEMA", "WEIGHTS_TAG", "binding",
           "domain", "fold_range", "hiding_leaf", "keyed_tensor_root", "leaf", "levels", "path", "range_shape", "row_prefix", "run_root", "salt",
           "step_root", "t_nibbles", "t_table", "tensor_hs", "tensor_key", "tensor_root", "weights_domain", "weights_root", "weights_tree"]

NAME = "rows-sha512"
INST = hm96.SHA512
KEY = INST.default_key
SCHEMA = SCHEMA_HM96_SHA512_ROW_V2
DOMAIN_TAG = "verity-vllm/step-rows/v1"
WEIGHTS_TAG = "verity-vllm/weights-rows/v1"
OWNER = -1
ROW_BYTES = 4096
SALT_BYTES = INST.salt_bytes
KEY_BYTES = 64
ROOT_BYTES = 64


def binding(program: bytes, ctx: bytes, geo: bytes, row_bits: int = 8 * ROW_BYTES) -> bytes:
    return identity_digest_sha512(DOMAIN_TAG, {"program": program.hex(), "ctx": ctx.hex(), "geo": geo.hex(), "row_bits": int(row_bits)})


def domain(program: bytes, ctx: bytes, geo: bytes, n: int, row_bits: int = 8 * ROW_BYTES) -> CommitmentDomain:
    """The step's tree domain: N leaves at positions 0 .. N-1."""
    return CommitmentDomain(binding(program, ctx, geo, row_bits), OWNER, RangeIndexedDomain(0, int(n)), hash="sha512")


def row_prefix(n_bits: int) -> bytes:
    return sha512_row_v2_prefix(n_bits)


def salt(ks: bytes, pos: int) -> bytes:
    """Leaf ``pos``'s 192-byte salt under the step's salt key."""
    if len(ks) != KEY_BYTES:
        raise ValueError(f"a salt key is {KEY_BYTES} bytes")
    return b"".join(hashlib.sha512(ks + int(pos).to_bytes(8, "big") + r.to_bytes(8, "big")).digest() for r in range(3))


def hiding_leaf(row: bytes, y: bytes, n_bits: int, key: bytes = KEY) -> tuple[bytes, bytes]:
    """``(b || c, h)`` for a row of ``n_bits`` bits and its salt."""
    bc = INST.commit_string(key, sha512_row_v2_digest(row, n_bits), y)
    return bc, INST.tree_leaf(key, bc)


def leaf(dom: CommitmentDomain, pos: int, h: bytes) -> bytes:
    return row_leaf(dom, pos, h, SCHEMA)


def levels(dom: CommitmentDomain, rows: Sequence[bytes], ks: bytes, n_bits: int, key: bytes = KEY) -> tuple[tuple[bytes, ...], ...]:
    """Every level of the step's tree (leaves first, padded to a power of two), from its rows."""
    return tree_levels(dom, [leaf(dom, i, hiding_leaf(r, salt(ks, i), n_bits, key)[1]) for i, r in enumerate(rows)])


def path(lv: Sequence[Sequence[bytes]], pos: int) -> tuple[bytes, ...]:
    """Leaf ``pos``'s siblings, leaves first (``MerkleTree.open``'s path)."""
    return tuple(lv[d][(pos >> d) ^ 1] for d in range(len(lv) - 1))


def range_shape(n: int, lo: int, hi: int) -> tuple[tuple[bool, bool], ...]:
    """Per level, leaves first, whether the run ``[lo, hi)`` of a tree of ``n`` leaves needs its left and its right sibling (the tree
    is complete, so every node has one)."""
    out = []
    for _ in range(merkle_depth(n)):
        out.append((lo % 2 == 1, hi % 2 == 1))
        lo, hi = lo // 2, (hi + 1) // 2
    return tuple(out)


def fold_range(dom: CommitmentDomain, lo: int, hs: Sequence[bytes], sibs: Sequence[tuple[bytes | None, bytes | None]]) -> bytes | None:
    """The tree root that leaves ``lo .. lo + len(hs) - 1`` with hiding digests ``hs`` and their boundary siblings ``sibs`` (per level
    ``(left | None, right | None)``, :func:`range_shape`'s) stand for; None for a run or a path of the wrong shape."""
    hi = lo + len(hs)
    if not hs or not 0 <= lo < hi <= dom.count:
        return None
    shape = range_shape(dom.count, lo, hi)
    if len(sibs) != len(shape) or any((left is None) == need_l or (right is None) == need_r
                                      for (left, right), (need_l, need_r) in zip(sibs, shape)):
        return None
    if any(s is not None and (type(s) is not bytes or len(s) != dom.digest_bytes) for pair in sibs for s in pair):
        return None
    nodes = [leaf(dom, lo + k, h) for k, h in enumerate(hs)]
    for d, (left, right) in enumerate(sibs):
        if left is not None:
            nodes.insert(0, left)
            lo -= 1
        if right is not None:
            nodes.append(right)
        lo >>= 1
        nodes = [dom.node(d, lo + k, nodes[2 * k], nodes[2 * k + 1]) for k in range(len(nodes) // 2)]
    return nodes[0]


def step_root(program: bytes, ctx: bytes, geo: bytes, layout: bytes, n: int, tree_root: bytes) -> bytes:
    return V.step_root(program, ctx, geo, layout, n, tree_root, hash="sha512")


def run_root(program: bytes, geo: bytes, step_roots: Sequence[bytes]) -> bytes:
    return V.run_root(program, geo, step_roots, hash="sha512")


def weights_domain(geo: bytes, tensor: int, entry: Sequence, row_bits: int = 8 * ROW_BYTES) -> CommitmentDomain:
    """Registered tensor ``tensor``'s tree domain, its names entry ``[name, dtype, shape, bytes]``: one leaf a row of its bytes."""
    name, dtype, shape, nb = entry
    b = identity_digest_sha512(WEIGHTS_TAG, {"geo": geo.hex(), "tensor": int(tensor), "name": str(name), "dtype": str(dtype),
                                             "shape": [int(s) for s in shape], "bytes": int(nb), "row_bits": int(row_bits)})
    return CommitmentDomain(b, OWNER, RangeIndexedDomain(0, -(-int(nb) // (row_bits // 8))), hash="sha512")


def tensor_key(wk: bytes, tensor: int) -> bytes:
    """Registered tensor ``tensor``'s salt key under the registration's key ``wk``."""
    if len(wk) != KEY_BYTES:
        raise ValueError(f"a weights key is {KEY_BYTES} bytes")
    return hashlib.sha512(WEIGHTS_TAG.encode() + wk + int(tensor).to_bytes(8, "big")).digest()


def tensor_hs(data: bytes | memoryview, ks: bytes, key: bytes = KEY) -> list[bytes]:
    """The hiding digests ``h`` of a tensor's rows, its bytes in ``ROW_BYTES`` rows with the last zero-padded, under salt key ``ks``."""
    mv = memoryview(data).cast("B")
    n = -(-len(mv) // ROW_BYTES)
    pre = hashlib.sha512(row_prefix(8 * ROW_BYTES))
    xs = bytearray()
    for i in range(n):
        h = pre.copy()
        row = mv[i * ROW_BYTES:(i + 1) * ROW_BYTES]
        h.update(row)
        if len(row) < ROW_BYTES:
            h.update(bytes(ROW_BYTES - len(row)))
        xs += h.digest()
    return hm96_batch.tree_leaves(INST, key, bytes(xs), b"".join(salt(ks, i) for i in range(n)))


def tensor_root(geo: bytes, tensor: int, entry: Sequence, data: bytes | memoryview, wk: bytes) -> bytes:
    """Registered tensor ``tensor``'s root: its rows' tree under :func:`weights_domain`, salted from ``tensor_key(wk, tensor)``."""
    return keyed_tensor_root(geo, tensor, entry, data, tensor_key(wk, tensor))


def keyed_tensor_root(geo: bytes, tensor: int, entry: Sequence, data: bytes | memoryview, ks: bytes) -> bytes:
    """:func:`tensor_root` under the tensor's own salt key ``ks``, for a reader that holds it and not ``Kw``."""
    dom = weights_domain(geo, tensor, entry)
    hs = tensor_hs(data, ks)
    if len(hs) != dom.count:
        raise ValueError(f"tensor {tensor}: {len(hs)} rows, its entry {list(entry)} is {dom.count}")
    return tree_levels(dom, [leaf(dom, i, h) for i, h in enumerate(hs)])[-1][0]


def weights_root(geo: bytes, names_digest: bytes, tensor_roots: Sequence[bytes]) -> bytes:
    return V.weights_root(geo, names_digest, tensor_roots, hash="sha512")


def weights_tree(geo: bytes, names: Sequence[Sequence], names_digest: bytes, tensor_roots: Sequence[bytes], wk: bytes) -> dict:
    """The registration's tree as the replay bundle carries it (`replay_bundle.weights_extra`), ``Kw`` with it."""
    return {"scheme": NAME, "names": [list(e) for e in names], "tensor_roots": [r.hex() for r in tensor_roots],
            "root": weights_root(geo, names_digest, tensor_roots).hex(), "chunk": ROW_BYTES, "geo_digest": geo.hex(), "key": wk.hex()}


def t_table(key: bytes = KEY) -> bytes:
    """``T_v`` = the XOR of ``(K >> s)`` over the set bits ``s`` of ``v``, ``K`` the key as a little-endian integer: 256 rows of 256
    bytes.  ``M_key`` is Hankel, so ``byte_tables(key)[p][v]`` is bytes ``[p, p + 64)`` of ``T_v`` (COMMIT-GPU's ``reference.t_table``);
    refused unless it is."""
    k = int.from_bytes(INST.check_key(key), "little")
    rows = [0] * 256
    for v in range(1, 256):
        rows[v] = rows[v & (v - 1)] ^ (k >> ((v & -v).bit_length() - 1))
    t = b"".join(r.to_bytes(256, "little") for r in rows)
    tables = INST.byte_tables(key)
    for p in range(INST.salt_bytes):
        for v in range(256):
            if tables[p][v] != int.from_bytes(t[256 * v + p:256 * v + p + 64], "little"):
                raise AssertionError(f"T table differs from byte_tables at p {p}, v {v}")
    return t


def t_nibbles(key: bytes = KEY) -> bytes:
    """The kernel's 8 KB mask table: ``T`` is linear in ``v``, so ``T_v = T_(v & 15) ^ T_(v & 0xF0)``; rows ``T_u`` and ``T_16u``
    (u < 16) stored chunk-major, 16-byte chunk ``c`` of row ``u`` at ``256 c + 16 u``, the high nibble's rows 4 KB on."""
    t = t_table(key)
    out = bytearray(8192)
    for hi in (0, 1):
        for c in range(16):
            for u in range(16):
                src = 256 * (16 * u if hi else u) + 16 * c
                out[4096 * hi + 256 * c + 16 * u:4096 * hi + 256 * c + 16 * u + 16] = t[src:src + 16]
    return bytes(out)
