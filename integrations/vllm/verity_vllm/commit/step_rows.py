"""The Commit's ``rows-sha512`` scheme (VLLM-17 step 1): COMMIT-GPU's construction (``benchmarks/pouw/commit_gpu``) over a step's
stream, opt-in on the windowed worker (``NativeCollectCommitter(commit_scheme="rows-sha512")``); ``vllm-v1`` stays the record.
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
and the frame path; :func:`fold_path` recomputes ``h`` and folds the path as core's ``verify_opening`` does.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

from verity.ml.commitments import hm96
from verity.ml.commitments import vllm_v1 as V
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, merkle_depth, tree_levels
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2, row_leaf, sha512_row_v2_digest, sha512_row_v2_prefix

__all__ = ["DOMAIN_TAG", "INST", "KEY", "KEY_BYTES", "NAME", "OWNER", "ROOT_BYTES", "ROW_BYTES", "SALT_BYTES", "SCHEMA", "binding", "domain",
           "fold_path", "hiding_leaf", "leaf", "levels", "path", "row_prefix", "run_root", "salt", "step_root", "t_nibbles", "t_table"]

NAME = "rows-sha512"
INST = hm96.SHA512
KEY = INST.default_key
SCHEMA = SCHEMA_HM96_SHA512_ROW_V2
DOMAIN_TAG = "verity-vllm/step-rows/v1"
OWNER = -1
ROW_BYTES = 256
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


def fold_path(dom: CommitmentDomain, pos: int, h: bytes, sibs: Sequence[bytes]) -> bytes | None:
    """The tree root that leaf ``pos`` with hiding digest ``h`` and path ``sibs`` stands for; None for a path of the wrong shape."""
    if not 0 <= pos < dom.count or len(sibs) != merkle_depth(dom.count) or any(len(s) != dom.digest_bytes for s in sibs):
        return None
    digest, cursor = leaf(dom, pos, h), pos
    for d, sib in enumerate(sibs):
        digest = dom.node(d, cursor >> 1, *((digest, sib) if cursor % 2 == 0 else (sib, digest)))
        cursor >>= 1
    return digest


def step_root(program: bytes, ctx: bytes, geo: bytes, layout: bytes, n: int, tree_root: bytes) -> bytes:
    return V.step_root(program, ctx, geo, layout, n, tree_root, hash="sha512")


def run_root(program: bytes, geo: bytes, step_roots: Sequence[bytes]) -> bytes:
    return V.run_root(program, geo, step_roots, hash="sha512")


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
