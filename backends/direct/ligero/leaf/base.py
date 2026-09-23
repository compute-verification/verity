"""The ``LeafScheme`` contract (coordinator brief 2026-09-23T16:30Z §2).

Every scheme must give the committer and the verifier the same function (``native`` / ``leaf_bytes``) that the in-circuit
``gadget`` evaluates on the chain's private operand bits, column by column, carrying a state across the VU's columns
and pinning the digest at the last one.  ``poseidon2`` (today's ``+hash``) stays implemented in ``hashchain.py`` until
lane leaf-iface moves it; this module only names the interface the other schemes code against.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class LeafScheme(Protocol):
    name: str            #: "poseidon2" | "blake3" | "ajtai"  (relation suffix "<rel>+<tag>")
    tag: str             #: the parametrised name carried by statements / relation strings, e.g. "ajtai-n256"
    assumption: str      #: one line for Table 1
    digest_elems: int    #: BabyBear elements per row digest
    schema: str          #: the leaf schema string of the SHA-256 tree framing ("<tag>/row/v2h")
    params_sha256: str   #: hex digest of the scheme's parameters (what the statement's auth block carries)

    def native(self, rows: np.ndarray, *, word_bits: int, role: int) -> np.ndarray:
        """(R, digest_elems) int64 < p of the rows (R, n_words).  Deterministic, unsalted."""

    def leaf_bytes(self, digest_row) -> bytes:
        """Canonical framing of one digest for the tree leaf value."""

    def gadget(self, ctx, words: Sequence, *, word_bits: int, role: int, carry_in: Sequence | None,
               first: bool, last: bool) -> tuple[list, list]:
        """Per chain column: absorb the column's words; returns (digest_exprs_if_last, carry_out)."""

    def witness(self):
        """Optional device witness hook for the gadget's hint rows."""
