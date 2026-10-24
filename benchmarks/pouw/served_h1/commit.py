"""served_h1's tables: hm96-sha512 hiding leaves (`verity.ml.commitments.hm96.SHA512`, its pinned key) under
hm96-sha512/row-seg/v1, in a frame-v3-sha512 tree per (step, linear) call and kind:

    a-table   slot row i: its K s8 bytes, then its f32 scale LE (`s8_rows`), committed as AROW's row table
              (`s8_table`, `verity.ml.commitments.row_table`); an empty slot is a zero row with scale 0 (a served row's
              scale is at least the quantizer's floor, so never 0)
    b-table   weight row j of a linear: its K s8 bytes, then its f32 weight scale LE, the same format (`s8_table`)
    weights   the linears' B-table roots in linear order (`step_root`): the weights' root, registered before every salt
    a-step    a step's A-table roots in linear order (`step_root`): the step's A root, which AROW's step root replaces
              once AROW computes it at serve time (the same tree over the same tables)
    receipts  unit u: its receipt's 2 (KS + m) u32 words LE (`receipt_rows`)

Stdlib, numpy and `verity.ml.commitments` only, so pool workers start light.

A tree's domain is `domain(fields, count, kind)`: identity_digest_sha512("verity/pouw/served-h1/<kind>/v0", fields), owner -1,
ranks 0 .. count - 1.  A leaf's 192-byte hm96 salt is ``salt(key, label, i)``: SHA-512(SALT_TAG ‖ key ‖ u32be len(label) ‖
label ‖ u64be i ‖ u8 c) for c = 0, 1, 2, from the window's secret 64-byte key (the prover's), so the prover's data is the
key; the reference hm96 salt is 192 fresh OS bytes a leaf, which this replaces by a SHA-512 PRF of one fresh key.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from verity.ml.commitments import hm96
from verity.ml.commitments import row_table as RT
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, merkle_depth, tree_levels
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG, row_leaf, sha512_row_seg_digest

INST = hm96.SHA512
KEY = INST.default_key
SALT_TAG = b"verity/pouw/served-h1/salt/v0\0"
SCHEMA = SCHEMA_HM96_SHA512_ROW_SEG


def domain(fields: dict, count: int, kind: str = "receipts") -> CommitmentDomain:
    return CommitmentDomain(identity_digest_sha512(f"verity/pouw/served-h1/{kind}/v0", fields), -1,
                            RangeIndexedDomain(0, count), hash="sha512")


def salt(key: bytes, label: str, i: int) -> bytes:
    head = SALT_TAG + key + len(label.encode()).to_bytes(4, "big") + label.encode() + i.to_bytes(8, "big")
    return b"".join(hashlib.sha512(head + bytes([c])).digest() for c in range(3))


def s8_rows(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """(n, K + 4) uint8: each row's s8 bytes, then its f32 scale little-endian."""
    return np.concatenate([q.astype(np.int8).view(np.uint8), scales.astype("<f4").reshape(-1, 1).view(np.uint8)], axis=1)


def receipt_rows(receipts: np.ndarray) -> np.ndarray:
    return receipts.astype("<u4").view(np.uint8).reshape(receipts.shape[0], -1)


def leaf(dom: CommitmentDomain, key: bytes, label: str, i: int, row: bytes) -> tuple[bytes, bytes]:
    """(frame leaf, b ‖ c) of row i."""
    bc = INST.commit_string(KEY, sha512_row_seg_digest(row, 8 * len(row)), salt(key, label, i))
    return row_leaf(dom, i, INST.tree_leaf(KEY, bc), SCHEMA), bc


def s8_table(dom: CommitmentDomain, key: bytes, label: str, q: np.ndarray, scales: np.ndarray) -> RT.RowTable:
    """A table of s8 rows in AROW's row-table format (`row_table.s8_row`, `row_table.commit`), a call's A table or a linear's
    B table: row i its s8 codes, then its f32 scale, under ``salt(key, label, i)``; its root is the tree `leaf` and `levels`
    build over `s8_rows`."""
    bits = np.ascontiguousarray(scales, dtype="<f4").view("<u4")
    rows = [RT.s8_row(bytes(np.ascontiguousarray(q[i], dtype=np.int8).view(np.uint8)), int(bits[i])) for i in range(len(q))]
    return RT.commit(dom, rows, [salt(key, label, i) for i in range(len(rows))])


def step_root(dom: CommitmentDomain, roots: Sequence[bytes]) -> bytes:
    """The tree over a step's table roots in linear order, each root a frame leaf."""
    return levels(dom, roots)[-1][0]


def leaves(args: tuple) -> bytes:
    """A pool task: (the domain's binding, count, key, label, start, rows) -> the rows' frame leaves, concatenated."""
    binding, count, key, label, start, rows = args
    dom = CommitmentDomain(binding, -1, RangeIndexedDomain(0, count), hash="sha512")
    return b"".join(leaf(dom, key, label, start + i, bytes(r))[0] for i, r in enumerate(rows))


def levels(dom: CommitmentDomain, frame_leaves: Sequence[bytes]) -> tuple[tuple[bytes, ...], ...]:
    return tree_levels(dom, list(frame_leaves))


def path(lv, index: int) -> list[bytes]:
    out = []
    for level in lv[:-1]:
        out.append(level[index ^ 1])
        index >>= 1
    return out


def verify(dom: CommitmentDomain, root: bytes, index: int, frame_leaf: bytes, sibs: list[bytes]) -> bool:
    if not 0 <= index < dom.count or len(sibs) != merkle_depth(dom.count):
        return False
    d = frame_leaf
    for lvl, s in enumerate(sibs):
        d = dom.node(lvl, index >> 1, *((d, s) if index % 2 == 0 else (s, d)))
        index >>= 1
    return d == root


def split(frame_leaves: bytes) -> list[bytes]:
    return [frame_leaves[i:i + 64] for i in range(0, len(frame_leaves), 64)]
