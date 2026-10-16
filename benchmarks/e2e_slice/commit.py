"""The slice's commitments: hm96-sha512/row/v2 rows in frame-v3-sha512 trees, by circuit.py's own leaf rule
(`verity_flock.circuit.statement`: tree leaf hm96.SHA512.tree_leaf(default key, b || c), then rowleaf.row_leaf under a
CommitmentDomain), so a spot check's row in its statement's public file is a leaf of a tree committed before it is read.

An operand leaf is decision 46's tile leaf of an operand X (rows x k int8): the RL x 32 block of rows RL i' .. RL i' + RL - 1
and values 32 j .. 32 j + 31, row-major (16384 bits, the spot check's a or b, `verity.core.protocols.pouw.circuit.segment`),
at position j (rows / RL) + i' (benchmarks/commit_gpu's tile mode). A receipt leaf is RW consecutive words of a call's
checkpointed receipts, laid out (tile, thread, window) (the spot check's rc), so thread t of tile T's window w is word
(T NCT + t) NW + w. A leaf's salt is the committer's secret, commit_gpu's PRF: SHA-512(Ks || BE64(pos) || BE64(r)) for
r = 0, 1, 2, Ks = SHA-512 of the salt key and the tree's label. Built in CHUNK-leaf pieces by a process pool; the tree's
levels are kept for the paths. `verify_path` is the challenger's side.
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
NCT = 512
RL, RW = 64, 512
CHUNK = 1 << 11
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


_H_ROWS: dict[int, "hashlib._Hash"] = {}
_H_SALT = hashlib.sha512(INST.salt_prefix)
_H_LEAF = hashlib.sha512(INST.leaf_prefix(KEY))


def tile_pos(rows: int, i: int, j: int) -> int:
    """The tile leaf holding row i at segment j of an operand of ``rows`` rows."""
    return j * (rows // RL) + i // RL


def tile_rows(X: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Tile leaves [lo, hi)'s rows of X (rows x k int8): (hi - lo, RL 16) u16, each block's bytes row-major."""
    mr = X.shape[0] // RL
    out = np.empty((hi - lo, RL * 32), np.int8)
    for j in range(lo // mr, (hi - 1) // mr + 1):
        a, b = max(lo, j * mr), min(hi, (j + 1) * mr)
        blk = np.asarray(X[(a - j * mr) * RL:(b - j * mr) * RL, 32 * j:32 * j + 32])
        out[a - lo:b - lo] = blk.reshape(b - a, RL * 32)
    return out.view("<u2")


def receipt_rows(R: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Receipt leaves [lo, hi)'s rows of a call's receipts R (any shape, (tile, thread, window) order) u32: (hi - lo, 2 RW)
    u16."""
    return np.ascontiguousarray(np.asarray(R, "<u4").reshape(-1)[RW * lo:RW * hi]).reshape(hi - lo, RW).view("<u2")


def _h_row(bits: int):
    if bits not in _H_ROWS:
        _H_ROWS[bits] = hashlib.sha512(sha512_row_v2_prefix(bits))
    return _H_ROWS[bits]


def commit_strings(rows: np.ndarray, salts: np.ndarray, bits: int | None = None) -> np.ndarray:
    """``b || c`` (n, 128) of hm96-sha512/row/v2 rows (n, words u16; ``bits`` bits, 16 words by default) under their salts
    (n, 192): hm96.SHA512.commit_string."""
    bits = 16 * rows.shape[1] if bits is None else bits
    rb = np.ascontiguousarray(rows.astype("<u2")).view(np.uint8).reshape(len(rows), -1)[:, :(bits + 7) // 8]
    h_row = _h_row(bits)
    x = np.empty((len(rows), 64), np.uint8)
    c = np.empty((len(rows), 64), np.uint8)
    for i in range(len(rows)):
        h = h_row.copy()
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


def label_a(session: str, rnd: int, call: int, stage: int = 0) -> dict:
    return {"session": session, "round": rnd, "call": call, "operand": "A", **({"stage": stage} if stage else {})}


def label_b(session: str, stage: int = 0) -> dict:
    return {"session": session, "operand": "B", "layer": "llama-3.1-8b/o_proj" if stage == 0 else f"stage-{stage}"}


def label_r(session: str, rnd: int, call: int, stage: int = 0, kind: str = "duo/S1") -> dict:
    return {"session": session, "round": rnd, "call": call, "operand": "receipts", "kind": kind, "stage": stage}


def domain(label: dict, count: int) -> CommitmentDomain:
    return CommitmentDomain(identity_digest_sha512(BINDING_TAG, label), OWNER, RangeIndexedDomain(0, count), hash="sha512")


def tree_leaf(dom: CommitmentDomain, pos: int, bc: bytes) -> bytes:
    h = _H_LEAF.copy()
    h.update(bc)
    return dom.leaf(pos, pos, V2, h.digest())


def tree_key(salt_key: bytes, label: dict) -> bytes:
    """The tree's Ks (commit_gpu's 64-byte salt key)."""
    return hashlib.sha512(b"verity/e2e-slice/salt/v2\0" + salt_key + identity_digest_sha512(BINDING_TAG, label)).digest()


def chunk_salts(salt_key: bytes, label: dict, lo: int, hi: int) -> np.ndarray:
    ks = tree_key(salt_key, label)
    return np.frombuffer(b"".join(hashlib.sha512(ks + p.to_bytes(8, "big") + r.to_bytes(8, "big")).digest()
                                  for p in range(lo, hi) for r in range(3)), np.uint8).reshape(hi - lo, SALT)


def leaf_salt(salt_key: bytes, label: dict, pos: int) -> bytes:
    return chunk_salts(salt_key, label, pos, pos + 1)[0].tobytes()


def leaf_row(src: tuple, pos: int) -> np.ndarray:
    """Leaf ``pos``'s row (`_rows`)."""
    return _rows(src, pos, pos + 1)[0]


def _rows(src: tuple, lo: int, hi: int) -> np.ndarray:
    """Leaves [lo, hi)'s rows: of an operand file (``("operand", path, (rows, k))``) or a receipts file
    (``("receipts", path, shape)``)."""
    kind, path, shape = src
    if kind == "operand":
        return tile_rows(np.memmap(path, np.int8, "r", shape=shape), lo, hi)
    return receipt_rows(np.memmap(path, np.uint32, "r", shape=shape), lo, hi)


def _chunk(args):
    t = time.process_time()
    src, label, count, salt_key, ci = args
    nl = min(CHUNK, count)
    bc = commit_strings(_rows(src, ci * nl, (ci + 1) * nl), chunk_salts(salt_key, label, ci * nl, (ci + 1) * nl))
    dom = domain(label, count)
    level = [tree_leaf(dom, ci * nl + i, bc[i].tobytes()) for i in range(nl)]
    levels, depth = [level], 0
    while len(level) > 1:
        base = (ci * nl >> depth) // 2
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
    """X's tree (X the int8 file at ``path``, rows x k), its tile leaves built CHUNK at a time by ``pool``."""
    return _commit(pool, ("operand", path, shape), shape[0] // RL * (shape[1] // 32), label, salt_key)


def commit_receipts(pool, path: str, shape: tuple, label: dict, salt_key: bytes) -> Tree:
    """A call's receipts' tree (the u32 file at ``path``, (tiles, NCT, NW)), RW words a leaf."""
    return _commit(pool, ("receipts", path, shape), int(np.prod(shape)) // RW, label, salt_key)


def _commit(pool, src: tuple, count: int, label: dict, salt_key: bytes) -> Tree:
    if count & (count - 1) or (count > CHUNK and count % CHUNK):
        raise ValueError(f"{count} leaves: a power of two, whole chunks")
    parts = pool.map(_chunk, [(src, label, count, salt_key, ci) for ci in range(max(1, count // CHUNK))])
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


def pool(workers: int | None = None):
    tables()
    return mp.get_context("fork").Pool(workers or max(1, (os.cpu_count() or 2) // 2))
