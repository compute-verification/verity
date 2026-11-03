"""A registered value committed from an array of its leaves, as `one_stage.registered.Registered.commit` commits rows.

`Registered.commit` takes rows of Python ints, which can't hold a model's weights (9.45·10¹⁰ leaves a GLM-5.3 rank). `commit`
takes the parameter's leaves as one unsigned array, in the parameter's leaf order, and makes the same rows, leaves, root and
entry: row `r` is leaves `[r * words, (r + 1) * words)`, each leaf `word_bits / 16` u16 words low first, zero-padded to
whole SHA-512 blocks (`registered.row_words`), hashed as `registered.row_digest` hashes it, committed under the row's salt
(`hm96.SHA512.commit_string`) and framed as `registered.leaf` frames it, under `registered.domain`. The salts are the
prover's: sampled from the operating system, one per row, unless given, and returned beside the entry, which is all a
registration names. Only `hm96-sha512/row/v1` rows, the format of a value no bit-row port reads.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from verity.core.protocols.sampled_proofs.one_stage import registered as G
from verity.ml.commitments import hm96
from verity.ml.commitments.merkle import tree_levels
from verity.ml.commitments.rowleaf import ROLE_X, sha512_row_layout

#: the widest row `row_width` picks, in leaves
MAX_WORDS = 4096
#: rows hashed per numpy block, which bounds the copy a block makes
BLOCK_ROWS = 4096
_INST = hm96.SHA512


def row_width(n: int, cap: int = MAX_WORDS) -> int:
    """The widest row of at most `cap` leaves that divides `n` leaves into whole rows (`registered.shape` takes no ragged
    row)."""
    if n < 1:
        raise ValueError(f"a registered value is at least one leaf, not {n}")
    return next(d for d in range(min(n, cap), 0, -1) if n % d == 0)


@dataclass(frozen=True)
class Committed:
    """The registration's entry (`Registered.entry`'s keys) and the prover's salts, row by row."""

    entry: dict[str, Any]
    salts: bytes

    @property
    def root(self) -> str:
        return self.entry["root"]


def commit(program: str, value: str, leaves: np.ndarray, word_bits: int, words: int | None = None,
           salts: bytes | None = None) -> Committed:
    """`value`'s leaves under `program`'s digest, in rows of `words` leaves (default `row_width`), each `word_bits` bits (16
    or 32: the parameter's leaf width rounded up to whole u16 words, `registered.width`). `salts`: `rows * salt_bytes`
    bytes, row order, never reused."""
    a = np.ascontiguousarray(leaves).reshape(-1)
    G.width_entry(word_bits)
    if a.dtype.kind != "u" or a.dtype.itemsize * 8 > 32:
        raise ValueError(f"{value}: leaves are an unsigned array of at most 32 bits, not {a.dtype}")
    if a.size and int(a.max()) >> word_bits:
        raise ValueError(f"{value}: a leaf does not fit {word_bits} bits")
    words = row_width(a.size) if words is None else int(words)
    if words < 1 or a.size % words:
        raise ValueError(f"{value}: {a.size} leaves are not rows of {words}")
    rows, k, n = a.size // words, word_bits // 16, _INST.salt_bytes
    if salts is None:
        salts = os.urandom(rows * n)
    if len(salts) != rows * n:
        raise ValueError(f"{value}: one {n}-byte salt per row: {rows * n} bytes, not {len(salts)}")
    if len({salts[i:i + n] for i in range(0, len(salts), n)}) != rows:
        raise ValueError(f"{value}: a salt is never reused: equal salts on two rows reveal the XOR of their digests")
    pad = -(words * k) % G.BLOCK
    layout = sha512_row_layout(ROLE_X, 16, words * k + pad)
    dom = G.domain(program, value, words, rows, word_bits)
    out: list[bytes] = []
    m = a.reshape(rows, words)
    for r0 in range(0, rows, BLOCK_ROWS):
        u16 = np.ascontiguousarray(m[r0:r0 + BLOCK_ROWS].astype("<u4" if k == 2 else "<u2")).view("<u2")
        if pad:
            u16 = np.concatenate([u16, np.zeros((u16.shape[0], pad), "<u2")], axis=1)
        for i, row in enumerate(u16, r0):
            bc = _INST.commit_string(_INST.default_key, layout.digest(row.tobytes()), salts[i * n:(i + 1) * n])
            out.append(G.leaf(dom, i, bc))
    root = tree_levels(dom, out)[-1][0]
    return Committed({"value": value, "root": root.hex(), "leaves": rows, "words": words, **G.width_entry(word_bits),
                      "domain": dom.domain_id.hex()}, salts)
