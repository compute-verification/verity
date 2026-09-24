"""The ``LeafScheme`` contract: how an operand-row commitment leaf (Poseidon2 · BLAKE3 · Ajtai · ...) plugs into the
self-hashing relation of ``hashchain.compose`` and the committed-object side of ``hashauth.py`` (campaign brief §2).

A leaf scheme is ONE function of a tensor row (``row_words`` words of ``word_bits`` bits) to ``digest_elems`` BabyBear
elements, evaluated three ways that must agree element for element:

* **native** -- the committer: :meth:`LeafScheme.native` on many rows at once (numpy; the tree builder of
  ``hashauth.build_row_tree`` and the negative batteries);
* **chain witness** -- the prover: :meth:`LeafScheme.chain_witness` on the (l, k) operand words of a sub-batch, giving the
  carried state at every column (the hint rows ``<nm>.cap[x]`` the chain links) and the digest of every VU (torch, host or
  device);
* **gadget** -- the relation: :meth:`LeafScheme.gadget` on the affine word expressions of ONE column, emitting rows and
  constraints into a :class:`~backends.direct.ligero.compile.LigeroCtx`, returning the digest expressions (pinned against
  the statement at a VU's last column) and the carried state for the next column.

The chain compiler compiles ONE column system that serves every position of every VU (``chain.py``): the carried state is
``carry_elems`` hint rows per operand that the chain test forces to (a) 0 at a chain start and (b) equal to the previous
column's ``carry_out`` elsewhere.  So the scheme must arrange that "carried value 0" means "initial state" (Poseidon2 stores
``capacity - IV``; a chaining-value scheme stores ``CV - IV``; a linear scheme's partial digest starts at 0 anyway), and
the digest expressions are tied to the public digest pins by ``is_end * (digest - pin) = 0`` in ``compose`` -- the
``first`` / ``last`` flags of :meth:`gadget` are therefore always ``(False, True)`` under this compiler and exist for a
straight-line (non-chain) compiler.

Rows the gadget needs beyond the carried state are either derived rows the fused witness program computes
(``ctx.product``, ``ctx.lookup``, ``ctx.inverse_nonzero``: program-evaluable) or extra hint rows (``ctx.hint(name)``, e.g.
the bit decompositions / carries of an ARX round) whose per-column values the scheme returns as the optional third item
of :meth:`chain_witness` (``{name: (l,) int64 tensor}`` keyed by the name given to ``ctx.hint``; ``hashchain.hash_hints``
places them).  A scheme that wants a faster device generator for its derived rows returns a :class:`WitnessHook` from
:meth:`witness` (``witness_device.py`` runs it after the fused program on the same stream; ``leaf/poseidon2.py`` writes
the 1704 S-box rows that way).

Serialization (statement v5, ``serialize.py``; the Rust verifier's ``auth.rs`` mirrors the table): the scheme is named
by the relation suffix ``<rel>+<name>`` (omitted for ``poseidon2``, the scheme statement v5 was defined with, so today's
files and pins are byte-identical), and the hash-auth block carries :attr:`schema` and :attr:`params_sha256` which the
reader checks against the named scheme.  Fingerprint: ``authentication=included-hash``, ``hash=<hash_name>``.

Hiding: every scheme here is deterministic and unsalted in v1 -- the digest of a low-entropy row is preimage-searchable;
the ZK rows hide the states inside the proof, not the digest itself.  Say so in the lane note (the brief does).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = ["LeafScheme", "WitnessHook", "check_scheme"]


class WitnessHook(Protocol):
    """A device generator of derived rows (``witness_device.py`` ``sys.witness_hooks``): ``rows`` are the row indices it
    fills (the fused program skips them), ``launch`` writes them into the (m, l) u32 witness on the current stream."""

    rows: frozenset

    def launch(self, cp, W_u32, l: int) -> None: ...


@runtime_checkable
class LeafScheme(Protocol):
    #: registry name: "poseidon2" | "blake3" | "ajtai" | ...; relation suffix "<rel>+<name>" ("+hash" is an alias of "+poseidon2")
    name: str
    #: one line for Table 1, e.g. "Poseidon2-BabyBear collision resistance (algebraic hash)"
    assumption: str
    #: BabyBear field elements per row digest (Poseidon2: 8)
    digest_elems: int
    #: carried elements per operand chain = linked hint rows per operand per column (Poseidon2: 8 capacity lanes)
    carry_elems: int
    #: the leaf schema string of the SHA-256 leaf framing (``CommitmentDomain.leaf(rank, rank, schema, value)``), the tree
    #: bindings and the v5 hash-auth block, e.g. "poseidon2-babybear-w24/row/v2h"
    schema: str
    #: 64 hex chars: SHA-256 of the scheme's constants / parameters (the 32 B field of the v5 block)
    params_sha256: str
    #: the fingerprint's ``hash`` value, e.g. "poseidon2-babybear-w24"
    hash_name: str

    def native(self, rows: np.ndarray, *, word_bits: int, role: int) -> np.ndarray:
        """(R, row_words) int64 words -> (R, digest_elems) int64 < p.  Deterministic, unsalted in v1.  Pure numpy: this is
        what the committer runs and what the verifier checks against the SHA-256 tree leaves."""
        ...

    def leaf_bytes(self, digest_row) -> bytes:
        """Canonical framing of one digest for ``CommitmentDomain.leaf`` (Poseidon2: 8 big-endian u32 = 32 bytes -- the
        leaf/v2h framing statement v5 was defined with; a new scheme picks its own and the Rust table mirrors it)."""
        ...

    def gadget(self, ctx, words: Sequence[Any], *, word_bits: int, role: int, row_words: int, carry_in: Sequence[Any],
               first: bool, last: bool) -> tuple[list, list]:
        """In-circuit, one chain column: absorb this column's operand ``words`` (affine ``Expr`` over the committed bit
        rows, ``len(words) = k``) into the state carried in ``carry_in`` (``carry_elems`` hint rows).  Returns
        ``(digest_exprs, carry_out)``: ``digest_elems`` expressions the compiler pins against the statement's public digest
        at a VU's last column, and ``carry_elems`` expressions the chain links to the next column's ``carry_in``.  May
        record row indices for :meth:`witness`."""
        ...

    def witness(self):
        """The optional device :class:`WitnessHook` for the gadget's derived rows, after every :meth:`gadget` call of the
        compose (one scheme instance per compose).  ``None``: the fused program computes them."""
        ...

    def chain_witness(self, words, *, word_bits: int, role: int, row_words: int, n_vus: int, steps: int, l: int):
        """        ``words`` (l, k) int64 torch tensor (pad columns 0; VU v occupies columns [v steps, (v + 1) steps)) ->
        ``(carry (carry_elems, l) int64 torch, digests (n_vus, digest_elems) int64 torch[, extra: dict])``: the carried
        state ENTERING every column (0 at a chain start) and every VU's digest -- the values of the hint rows the gadget's
        ``carry_in`` names and of the public digest pins -- and optionally the values of the gadget's other hint rows by
        name.  Must equal :meth:`native` on the VU's row (the conformance suite checks)."""
        ...

    def census(self) -> dict:
        """Scheme-specific counts for the census / fingerprint (may be empty)."""
        ...


def check_scheme(s) -> None:
    """Static sanity of a scheme object (the registry calls it): the attributes exist and have the right shape."""
    for attr in ("name", "assumption", "schema", "params_sha256", "hash_name"):
        v = getattr(s, attr, None)
        if not isinstance(v, str) or not v:
            raise TypeError(f"leaf scheme {s!r}: {attr} must be a non-empty str")
    for attr in ("digest_elems", "carry_elems"):
        v = getattr(s, attr, None)
        if not isinstance(v, int) or v <= 0:
            raise TypeError(f"leaf scheme {s.name}: {attr} must be a positive int")
    if len(s.params_sha256) != 64 or set(s.params_sha256) - set("0123456789abcdef"):
        raise TypeError(f"leaf scheme {s.name}: params_sha256 must be 64 lowercase hex chars")
    if "+" in s.name or "/" in s.name or not s.name.islower():
        raise TypeError(f"leaf scheme {s.name!r}: the name is a lowercase identifier without '+' or '/'")
    for meth in ("native", "leaf_bytes", "gadget", "witness", "chain_witness", "census"):
        if not callable(getattr(s, meth, None)):
            raise TypeError(f"leaf scheme {s.name}: missing {meth}()")
