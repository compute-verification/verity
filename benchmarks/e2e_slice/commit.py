"""The slice's operand commitments: hm96-sha512/row/v2 rows in frame-v3-sha512 trees, by circuit.py's own leaf rule
(`verity_flock.circuit.statement`: tree leaf hm96.SHA512.tree_leaf(default key, b || c), then rowleaf.row_leaf under a
CommitmentDomain), so a spot check's row in its statement's public file is a leaf of a tree committed before the coin.

A leaf is row i, step j of an operand X (rows x k int8): the 32 values X[i, 32 j .. 32 j + 31] as bf16 words, twice (64 words,
1024 bits: the spot check's xa or wb row), at position i (k / 32) + j. Its salt is the committer's secret: SHAKE-256 of the
salt key, the tree's label and the chunk (CHUNK leaves), 192 bytes a leaf. Built in CHUNK-leaf pieces by a process pool;
the tree's levels are kept for the paths. `verify_path` is the challenger's side.
"""
from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import time

import numpy as np

from verity.ml.commitments import hm96
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2, sha512_row_v2_prefix

INST = hm96.SHA512
KEY = INST.default_key
SALT = INST.salt_bytes
ROW_BITS = 1024
CHUNK = 1 << 14
BINDING_TAG = "verity/e2e-slice/operand/v1"
OWNER = -1
_TAB = None


def tables() -> np.ndarray:
    """M_key's byte tables as (192, 256, 8) u64: entry (p, v) is the XOR of the columns 8p + s over v's set bits, little-endian."""
    global _TAB
    if _TAB is None:
        t = INST.byte_tables(KEY)
        _TAB = np.frombuffer(b"".join(x.to_bytes(64, "little") for row in t for x in row), "<u8").reshape(SALT, 256, 8).copy()
    return _TAB


_H_ROW = hashlib.sha512(sha512_row_v2_prefix(ROW_BITS))
_H_SALT = hashlib.sha512(INST.salt_prefix)
_H_LEAF = hashlib.sha512(INST.leaf_prefix(KEY))


def bf16_words(v: np.ndarray) -> np.ndarray:
    """int8 (or small int) values as bf16 bit patterns, u16 (exact: |v| < 256)."""
    return (np.asarray(v, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def rows_of(X: np.ndarray) -> np.ndarray:
    """Every leaf's row of X (rows x k): (rows k / 32, 64) u16, the 32 values' bf16 words twice."""
    w = bf16_words(X.reshape(-1, 32))
    return np.concatenate([w, w], axis=1)


def commit_strings(rows: np.ndarray, salts: np.ndarray) -> np.ndarray:
    """``b || c`` (n, 128) of hm96-sha512/row/v2 rows (n, 64 u16) under their salts (n, 192): hm96.SHA512.commit_string."""
    rb = np.ascontiguousarray(rows.astype("<u2")).view(np.uint8).reshape(len(rows), -1)
    x = np.empty((len(rows), 64), np.uint8)
    c = np.empty((len(rows), 64), np.uint8)
    for i in range(len(rows)):
        h = _H_ROW.copy()
        h.update(rb[i].tobytes())
        x[i] = np.frombuffer(h.digest(), np.uint8)
        h = _H_SALT.copy()
        h.update(salts[i].tobytes())
        c[i] = np.frombuffer(h.digest(), np.uint8)
    tab, acc = tables(), np.zeros((len(rows), 8), np.uint64)
    for p in range(SALT):
        acc ^= tab[p][salts[:, p]]
    b = x ^ acc.astype("<u8").view(np.uint8).reshape(len(rows), 64)
    return np.concatenate([b, c], axis=1)


def label_a(session: str, rnd: int, call: int) -> dict:
    return {"session": session, "round": rnd, "call": call, "operand": "A"}


def label_b(session: str) -> dict:
    return {"session": session, "operand": "B", "layer": "llama-3.1-8b/o_proj"}


def domain(label: dict, count: int) -> CommitmentDomain:
    return CommitmentDomain(identity_digest_sha512(BINDING_TAG, label), OWNER, RangeIndexedDomain(0, count), hash="sha512")


def tree_leaf(dom: CommitmentDomain, pos: int, bc: bytes) -> bytes:
    h = _H_LEAF.copy()
    h.update(bc)
    return dom.leaf(pos, pos, V2, h.digest())


def chunk_salts(salt_key: bytes, label: dict, chunk: int) -> np.ndarray:
    tag = b"verity/e2e-slice/salt/v1\0" + salt_key + identity_digest_sha512(BINDING_TAG, label) + chunk.to_bytes(8, "big")
    return np.frombuffer(hashlib.shake_256(tag).digest(SALT * CHUNK), np.uint8).reshape(CHUNK, SALT)


def leaf_salt(salt_key: bytes, label: dict, pos: int) -> bytes:
    return chunk_salts(salt_key, label, pos // CHUNK)[pos % CHUNK].tobytes()


def _chunk(args):
    t = time.process_time()
    path, shape, label, count, salt_key, ci = args
    X = np.memmap(path, np.int8, "r", shape=shape)
    per_row = shape[1] // 32
    r0, r1 = ci * CHUNK // per_row, (ci + 1) * CHUNK // per_row
    bc = commit_strings(rows_of(np.asarray(X[r0:r1])), chunk_salts(salt_key, label, ci))
    dom = domain(label, count)
    level = [tree_leaf(dom, ci * CHUNK + i, bc[i].tobytes()) for i in range(CHUNK)]
    levels, depth = [level], 0
    while len(level) > 1:
        base = (ci * CHUNK >> depth) // 2
        level = [dom.node(depth, base + i // 2, level[i], level[i + 1]) for i in range(0, len(level), 2)]
        levels.append(level)
        depth += 1
    return bc.tobytes(), [b"".join(lv) for lv in levels], time.process_time() - t


class Tree:
    """One operand's committed tree: its root, every leaf's b || c, the levels (for paths) and the workers' CPU seconds."""

    def __init__(self, label: dict, count: int, bc: bytes, levels: list[bytes], cpu_s: float = 0.0):
        self.label, self.count, self.bc, self.levels, self.cpu_s = label, count, bc, levels, cpu_s
        self.root = levels[-1]

    def path(self, pos: int) -> list[bytes]:
        out = []
        for d in range(len(self.levels) - 1):
            s = (pos >> d) ^ 1
            out.append(self.levels[d][64 * s:64 * s + 64])
        return out

    def leaf_bc(self, pos: int) -> bytes:
        return self.bc[128 * pos:128 * pos + 128]


def commit_operand(pool, path: str, shape: tuple[int, int], label: dict, salt_key: bytes) -> Tree:
    """X's tree (X the int8 file at ``path``, rows x k), its leaves built CHUNK at a time by ``pool``."""
    count = shape[0] * shape[1] // 32
    if count % CHUNK or count & (count - 1):
        raise ValueError(f"{count} leaves: a power of two, whole chunks")
    parts = pool.map(_chunk, [(path, shape, label, count, salt_key, ci) for ci in range(count // CHUNK)])
    bc = b"".join(p[0] for p in parts)
    levels = [b"".join(p[1][d] for p in parts) for d in range(len(parts[0][1]))]
    dom, depth = domain(label, count), len(levels) - 1
    level = [levels[-1][64 * i:64 * i + 64] for i in range(len(parts))]
    while len(level) > 1:
        level = [dom.node(depth, i // 2, level[i], level[i + 1]) for i in range(0, len(level), 2)]
        levels.append(b"".join(level))
        depth += 1
    return Tree(label, count, bc, levels, sum(p[2] for p in parts))


def verify_path(label: dict, count: int, root: bytes, pos: int, bc: bytes, path: list[bytes]) -> bool:
    dom = domain(label, count)
    if len(path) != (count - 1).bit_length():
        return False
    cur = tree_leaf(dom, pos, bc)
    for d, sib in enumerate(path):
        idx = pos >> d
        cur = dom.node(d, idx // 2, cur, sib) if idx % 2 == 0 else dom.node(d, idx // 2, sib, cur)
    return cur == root


RECEIPT_SCHEMA = "verity/e2e-slice/duo-receipts/v1"


def receipt_root(R: np.ndarray, session: str, rnd: int, call: int, salt_key: bytes) -> bytes:
    """A call's duo receipts (tiles x NCT x 8 u32) under one frame-v3-sha512 tree, a leaf a tile: its 64-byte salt (SHAKE-256
    of the salt key and the tree's label) then the tile's 16 KB of receipts."""
    from verity.ml.commitments.merkle import tree_levels

    label = {"session": session, "round": rnd, "call": call, "operand": "receipts"}
    tiles, raw = R.shape[0], np.ascontiguousarray(R).tobytes()
    per = len(raw) // tiles
    salts = hashlib.shake_256(b"verity/e2e-slice/receipt-salt/v1\0" + salt_key + identity_digest_sha512(BINDING_TAG, label)).digest(64 * tiles)
    dom = domain(label, tiles)
    return tree_levels(dom, [dom.leaf(t, t, RECEIPT_SCHEMA, salts[64 * t:64 * t + 64] + raw[per * t:per * t + per])
                             for t in range(tiles)])[-1][0]


def pool(workers: int | None = None):
    tables()
    return mp.get_context("fork").Pool(workers or max(1, (os.cpu_count() or 2) // 2))
