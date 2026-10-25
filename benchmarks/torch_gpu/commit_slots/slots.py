"""The serve-time commit call's CPU reference: a serving step's unit outputs as ``hm96-sha512/row/v2`` rows under hiding
leaves in ``frame-v3-sha512`` trees, one subtree per sequence slot (README.md is the call's specification).

A step's tree is three levels of trees, each a ``frame-v3-sha512`` tree padded to a power of two:

- **unit tree** (slot s, unit u): over the unit's rows, rank r its row r, leaf schema ``hm96-sha512/row/v2``, domain
  :func:`unit_domain` (u);
- **slot tree** (slot s): over the slot's unit roots, index u, leaf schema ``frame-v3-sha512/root/v1`` (``row_table.root_leaf``),
  domain :func:`slot_domain`;
- **step tree**: over the slots' roots, index s, the same leaf schema, domain :func:`step_domain` (the step): its root is the
  step's root, the one 64 bytes the step records before its live coin.

A row's leaf, row g of slot s (rows numbered in the layout's order within a slot) and rank r in its unit::

  x    = SHA-512(P(8 len(row)) || row)                       rowleaf.sha512_row_v2_digest
  y    = ChaCha20 blocks 3g, 3g+1, 3g+2, key the step's salt key, nonce s      C-Flock's hm96_salt
  b||c = hm96.SHA512.commit_string(KEY, x, y)                public, 128 bytes
  leaf = unit_domain(u).leaf(r, r, "hm96-sha512/row/v2", hm96.SHA512.tree_leaf(KEY, b||c))

The layout (:class:`Layout`) is the step's public structure: groups (what one ``add`` commits, e.g. a layer), each of units,
each of rows, each row the concatenation of byte ranges (:class:`Part`) of the slot's row of the group's tensors. Every slot
has the same layout. A unit and a slot tree's domain don't name the slot or the step: the step tree's leaf index fixes the
slot, and the step domain the step. :func:`commit` is the reference commitment; :func:`open_unit` opens one unit of one slot
from that slot's rows alone (a replay of its request), and :func:`verify_unit` checks an opening against the step's root.
"""
from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from verity.ml.commitments import hm96
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, MerkleTree
from verity.ml.commitments.row_table import ROOTS_SCHEMA, RootOpening, RootTree, verify_roots
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2, sha512_row_v2_digest
from verity.ml.crypto.chacha20 import keystream

__all__ = ["Group", "Layout", "Part", "Row", "StepCommit", "UnitOpening", "commit", "commit_slot", "fresh_salt_key",
           "h200_layout", "layout_digest", "open_unit", "random_outputs", "row_salts", "slot_domain", "slot_rows",
           "step_domain", "unit_domain", "verify_unit"]

LAYOUT_TAG = "verity/commit-slots/layout/v1"
UNIT_TAG = "verity/commit-slots/unit/v1"
SLOT_TAG = "verity/commit-slots/slot/v1"
STEP_TAG = "verity/commit-slots/step/v1"
ROW_SCHEMA = SCHEMA_HM96_SHA512_ROW_V2
INST = hm96.SHA512
KEY = INST.default_key
SALT_BYTES = INST.salt_bytes
STRING_BYTES = INST.proven_bytes
DIGEST_BYTES = INST.digest_bytes
#: a unit's outputs fit its size class, at most 262,144 bits (AGENTS.md, "Everything zero-knowledge")
UNIT_MAX_BITS = 262_144


# --- the layout --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Part:
    """``nbytes`` bytes at ``offset`` of a slot's row of the group's tensor ``source``."""

    source: int
    offset: int
    nbytes: int


@dataclass(frozen=True, slots=True)
class Row:
    """One committed row: its parts' bytes, concatenated."""

    parts: tuple[Part, ...]

    @property
    def nbytes(self) -> int:
        return sum(p.nbytes for p in self.parts)

    @property
    def bits(self) -> int:
        return 8 * self.nbytes


@dataclass(frozen=True, slots=True)
class Group:
    """What one ``add`` commits: ``sources[i]`` is the bytes a slot has in the group's tensor i (a tensor is (slots, ...),
    each slot's bytes contiguous), and ``units`` the group's units, each its rows."""

    sources: tuple[int, ...]
    units: tuple[tuple[Row, ...], ...]

    def __post_init__(self) -> None:
        for unit in self.units:
            if not unit:
                raise ValueError("a unit has at least one row")
            if sum(r.bits for r in unit) > UNIT_MAX_BITS:
                raise ValueError(f"a unit's outputs are at most {UNIT_MAX_BITS} bits")
            for row in unit:
                if not row.parts or row.nbytes == 0:
                    raise ValueError("a row has at least one byte")
                for p in row.parts:
                    if not 0 <= p.source < len(self.sources) or p.offset < 0 or p.nbytes < 1 \
                            or p.offset + p.nbytes > self.sources[p.source]:
                        raise ValueError(f"part {p} is outside the group's tensors {self.sources}")


@dataclass(frozen=True)
class Layout:
    """A step's public structure: its groups in order, the same for every slot; ``name`` names it."""

    name: str
    groups: tuple[Group, ...]

    @cached_property
    def units(self) -> tuple[tuple[Row, ...], ...]:
        """Every unit, in order: unit u's rows."""
        return tuple(u for g in self.groups for u in g.units)

    @cached_property
    def unit_base(self) -> tuple[int, ...]:
        """Group i's first unit."""
        out, n = [], 0
        for g in self.groups:
            out.append(n)
            n += len(g.units)
        return tuple(out)

    @cached_property
    def row_base(self) -> tuple[int, ...]:
        """Unit u's first row g in its slot."""
        out, n = [], 0
        for u in self.units:
            out.append(n)
            n += len(u)
        return tuple(out)

    @property
    def unit_count(self) -> int:
        return len(self.units)

    @property
    def row_count(self) -> int:
        return sum(len(u) for u in self.units)

    @property
    def slot_bytes(self) -> int:
        """The bytes a slot commits."""
        return sum(r.nbytes for u in self.units for r in u)

    def manifest(self) -> dict[str, Any]:
        """What the layout digest binds: the name and every unit's row lengths in bits (where a row's bytes come from is the
        prover's plumbing, not the commitment's)."""
        return {"name": self.name, "units": [[r.bits for r in u] for u in self.units]}


def layout_digest(layout: Layout) -> bytes:
    return identity_digest_sha512(LAYOUT_TAG, layout.manifest())


# --- domains and salts -------------------------------------------------------------------------------------------------

def _context(context: bytes) -> str:
    if type(context) is not bytes or len(context) != DIGEST_BYTES:
        raise ValueError(f"a context is {DIGEST_BYTES} bytes")
    return context.hex()


def _domain(tag: str, manifest: dict[str, Any], leaves: int) -> CommitmentDomain:
    return CommitmentDomain(identity_digest_sha512(tag, manifest), -1, RangeIndexedDomain(0, leaves), hash="sha512")


def unit_domain(context: bytes, layout: Layout, unit: int) -> CommitmentDomain:
    """Unit ``unit``'s tree over its rows, the same in every slot and step of a run."""
    rows = len(layout.units[unit])
    return _domain(UNIT_TAG, {"context": _context(context), "layout": layout_digest(layout).hex(), "unit": unit, "rows": rows,
                              "schema": ROW_SCHEMA}, rows)


def slot_domain(context: bytes, layout: Layout) -> CommitmentDomain:
    """A slot's tree over its units' roots, the same in every slot and step of a run."""
    return _domain(SLOT_TAG, {"context": _context(context), "layout": layout_digest(layout).hex(), "units": layout.unit_count,
                              "schema": ROOTS_SCHEMA}, layout.unit_count)


def step_domain(context: bytes, layout: Layout, step: int, slots: int) -> CommitmentDomain:
    """Step ``step``'s tree over its ``slots`` slots' roots."""
    if type(step) is not int or step < 0 or type(slots) is not int or slots < 1:
        raise ValueError("a step is a nonnegative integer and has at least one slot")
    return _domain(STEP_TAG, {"context": _context(context), "layout": layout_digest(layout).hex(), "step": step,
                              "slots": slots, "schema": ROOTS_SCHEMA}, slots)


def fresh_salt_key() -> bytes:
    """A step's salt key from the operating system's generator: the prover's alone, kept with the step's record."""
    return os.urandom(32)


def row_salts(salt_key: bytes, slot: int, first: int, n: int) -> bytes:
    """The salts of rows ``first .. first + n - 1`` of ``slot``, 192 bytes each: ChaCha20 blocks 3g .. 3g + 2 under the step's
    salt key with nonce ``slot`` (C-Flock's ``hm96_salt``, ``verity.core.protocols.pouw.audit.hm96_salt``)."""
    return keystream(salt_key, slot, 3 * first, 3 * n)


# --- committing --------------------------------------------------------------------------------------------------------

def slot_rows(layout: Layout, outputs: Sequence[Sequence[Any]], slot: int) -> list[bytes]:
    """Slot ``slot``'s rows in order, from ``outputs[i][j]``, group i's tensor j as (slots, bytes) uint8 (or anything
    ``np.asarray`` reads as (slots, ...) and views as bytes)."""
    if len(outputs) != len(layout.groups):
        raise ValueError(f"{len(outputs)} groups of outputs for {len(layout.groups)} groups")
    rows: list[bytes] = []
    for g, tensors in zip(layout.groups, outputs, strict=True):
        if len(tensors) != len(g.sources):
            raise ValueError(f"a group of {len(g.sources)} tensors got {len(tensors)}")
        mine = []
        for t, n in zip(tensors, g.sources, strict=True):
            a = np.ascontiguousarray(np.asarray(t)[slot]).view(np.uint8).reshape(-1)
            if a.size != n:
                raise ValueError(f"a slot's row of a tensor is {n} bytes, not {a.size}")
            mine.append(a)
        rows += [b"".join(mine[p.source][p.offset:p.offset + p.nbytes].tobytes() for p in r.parts) for u in g.units for r in u]
    return rows


def _strings(rows: Sequence[bytes], salts: bytes) -> list[bytes]:
    return [INST.commit_string(KEY, sha512_row_v2_digest(row, 8 * len(row)), salts[SALT_BYTES * i:SALT_BYTES * (i + 1)])
            for i, row in enumerate(rows)]


def _unit_tree(domain: CommitmentDomain, strings: Sequence[bytes]) -> MerkleTree:
    return MerkleTree(domain, {r: INST.tree_leaf(KEY, s) for r, s in enumerate(strings)}, lambda _r: ROW_SCHEMA)


@dataclass(frozen=True)
class SlotCommit:
    """One slot's committed rows: every row's ``b || c`` in order, its unit trees and its slot tree (the prover's)."""

    strings: tuple[bytes, ...]
    units: tuple[MerkleTree, ...]
    tree: RootTree

    @property
    def root(self) -> bytes:
        return self.tree.root


def commit_slot(context: bytes, layout: Layout, salt_key: bytes, slot: int, rows: Sequence[bytes]) -> SlotCommit:
    """Slot ``slot``'s subtree from its rows alone (:func:`slot_rows`): what a replay of its request rebuilds."""
    flat = [r for u in layout.units for r in u]
    if len(rows) != len(flat) or any(len(r) != row.nbytes for r, row in zip(rows, flat, strict=True)):
        raise ValueError("a slot's rows are the layout's, in order")
    strings = _strings(rows, row_salts(salt_key, slot, 0, len(rows)))
    trees = tuple(_unit_tree(unit_domain(context, layout, u), strings[b:b + len(rows_u)])
                  for u, (b, rows_u) in enumerate(zip(layout.row_base, layout.units, strict=True)))
    return SlotCommit(tuple(strings), trees, RootTree(slot_domain(context, layout), [t.commitment.root for t in trees]))


@dataclass(frozen=True)
class StepCommit:
    """A step's commitment: its root and every slot's root, in slot order (what the prover keeps, with the salt key)."""

    root: bytes
    slot_roots: tuple[bytes, ...]


def step_tree(context: bytes, layout: Layout, step: int, slot_roots: Sequence[bytes]) -> RootTree:
    return RootTree(step_domain(context, layout, step, len(slot_roots)), slot_roots)


def commit(context: bytes, layout: Layout, step: int, salt_key: bytes, outputs: Sequence[Sequence[Any]], slots: int) -> StepCommit:
    """The reference commitment of a step of ``slots`` slots: ``outputs`` as :func:`slot_rows` reads them."""
    roots = [commit_slot(context, layout, salt_key, s, slot_rows(layout, outputs, s)).root for s in range(slots)]
    return StepCommit(step_tree(context, layout, step, roots).root, tuple(roots))


# --- opening a unit ----------------------------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class UnitOpening:
    """Unit ``unit`` of slot ``slot``: its rows' ``b || c`` strings (rank order), its root opened in the slot's tree
    (``unit_path``, one index) and the slot's root opened in the step's tree (``slot_path``, one index)."""

    slot: int
    unit: int
    strings: tuple[bytes, ...]
    unit_path: RootOpening
    slot_path: RootOpening

    def to_json(self) -> dict[str, Any]:
        return {"slot": self.slot, "unit": self.unit, "strings": [s.hex() for s in self.strings],
                "unit_path": self.unit_path.to_json(), "slot_path": self.slot_path.to_json()}

    @classmethod
    def of_json(cls, doc: Mapping[str, Any]) -> UnitOpening:
        return cls(int(doc["slot"]), int(doc["unit"]), tuple(bytes.fromhex(s) for s in doc["strings"]),
                   RootOpening.of_json(doc["unit_path"]), RootOpening.of_json(doc["slot_path"]))


def open_unit(context: bytes, layout: Layout, step: int, salt_key: bytes, slot_roots: Sequence[bytes], slot: int, unit: int,
              rows: Sequence[bytes]) -> UnitOpening:
    """The opening of ``unit`` of ``slot`` from the slot's rows (a replay of its request) and the step's slot roots."""
    sc = commit_slot(context, layout, salt_key, slot, rows)
    if sc.root != slot_roots[slot]:
        raise ValueError(f"slot {slot}'s replayed rows are not the ones committed")
    b = layout.row_base[unit]
    return UnitOpening(slot, unit, sc.strings[b:b + len(layout.units[unit])], sc.tree.open([unit]),
                       step_tree(context, layout, step, slot_roots).open([slot]))


def verify_unit(step_root: bytes, context: bytes, layout: Layout, step: int, slots: int, opening: UnitOpening,
                rows: Mapping[int, tuple[bytes, bytes]] | None = None) -> bool:
    """Whether ``opening`` opens unit ``opening.unit`` of slot ``opening.slot`` of the step whose root is ``step_root``; with
    ``rows``, ``rows[r] = (row, salt)`` are exactly its rows' bytes and salts."""
    o = opening
    if not (0 <= o.slot < slots and 0 <= o.unit < layout.unit_count) or o.slot_path.indices != (o.slot,) \
            or o.unit_path.indices != (o.unit,) or len(o.strings) != len(layout.units[o.unit]):
        return False
    if any(type(s) is not bytes or len(s) != STRING_BYTES for s in o.strings):
        return False
    if rows is not None:
        if sorted(rows) != list(range(len(o.strings))):
            return False
        for r, (row, salt) in rows.items():
            if len(row) != layout.units[o.unit][r].nbytes or len(salt) != SALT_BYTES \
                    or INST.commit_string(KEY, sha512_row_v2_digest(row, 8 * len(row)), salt) != o.strings[r]:
                return False
    unit_root = _unit_tree(unit_domain(context, layout, o.unit), o.strings).commitment.root
    if unit_root != o.unit_path.roots[0]:
        return False
    sd = slot_domain(context, layout)
    if not verify_roots(o.slot_path.roots[0], sd.domain_id, sd.count, o.unit_path):
        return False
    td = step_domain(context, layout, step, slots)
    return verify_roots(step_root, td.domain_id, td.count, o.slot_path)


# --- layouts and test data ---------------------------------------------------------------------------------------------

def _cut(source: int, offset: int, nbytes: int, cap: int) -> list[tuple[Row, ...]]:
    """``nbytes`` at ``offset`` of ``source`` as one-row units of at most ``cap`` bytes."""
    return [(Row((Part(source, offset + a, min(cap, nbytes - a)),)),) for a in range(0, nbytes, cap)]


def h200_layout(layers: int = 52, d: int = 16384, heads: int = 256, head_dim: int = 64, kv_heads: int = 1, mlp: int = 65536,
                ctx_rows: int = 16) -> Layout:
    """The frozen H200 shape's step layout, every op's per-token output committed (compute-accounting's "every op" count),
    one group a layer, each op's output cut into one-row units of at most ``UNIT_MAX_BITS`` (32 KiB):

    attention RMSNorm (bf16 d), its s8 quantizer (d codes, then the f32 scale), QKV (bf16 heads x head_dim + 2 kv_heads x
    head_dim), the scores (f32 heads x ``ctx_rows``) and softmax (bf16 heads x ``ctx_rows``), PV (bf16 d), o_proj's quantizer
    and output, the residual add, the MLP's RMSNorm and quantizer, up (bf16 mlp), ReLU (bf16 mlp), down's quantizer (mlp codes
    and the scale), down (bf16 d) and the residual add. ``ctx_rows`` stands in for the scores' t + 1 (16, the mean at a
    context of 32). The group's tensors are the ops' outputs in that order, a quantizer's codes and scale two tensors."""
    cap = UNIT_MAX_BITS // 8
    qkv = 2 * (heads * head_dim + 2 * kv_heads * head_dim)
    ops = [("bf16", 2 * d), ("quant", d), ("bf16", qkv), ("f32", 4 * heads * ctx_rows), ("bf16", 2 * heads * ctx_rows),
           ("bf16", 2 * d), ("quant", d), ("bf16", 2 * d), ("bf16", 2 * d), ("bf16", 2 * d), ("quant", d), ("bf16", 2 * mlp),
           ("bf16", 2 * mlp), ("quant", mlp), ("bf16", 2 * d), ("bf16", 2 * d)]
    sources: list[int] = []
    units: list[tuple[Row, ...]] = []
    for kind, n in ops:
        src = len(sources)
        if kind == "quant":
            sources += [n, 4]
            if n + 4 <= cap:
                units.append((Row((Part(src, 0, n), Part(src + 1, 0, 4))),))
            else:
                units += _cut(src, 0, n, cap) + [(Row((Part(src + 1, 0, 4),)),)]
        else:
            sources.append(n)
            units += _cut(src, 0, n, cap)
    group = Group(tuple(sources), tuple(units))
    return Layout(f"h200-mqa-d{d}-h{heads}x{head_dim}-kv{kv_heads}-f{mlp}-c{ctx_rows}-every-op/v1", (group,) * layers)


def random_outputs(layout: Layout, slots: int, rng: np.random.Generator, groups: Iterable[int] | None = None) -> list[list[np.ndarray]]:
    """Uniform bytes for every tensor of the chosen groups (all by default), (slots, bytes) uint8 each."""
    chosen = range(len(layout.groups)) if groups is None else groups
    return [[rng.integers(0, 256, (slots, n), dtype=np.uint8) for n in layout.groups[i].sources] for i in chosen]
