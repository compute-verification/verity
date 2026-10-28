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
  y    = ChaCha20 chunks 3g, 3g+1, 3g+2, key the step's salt key, nonce s      C-Flock's hm96_salt
  b||c = hm96.SHA512.commit_string(KEY, x, y)                public, 128 bytes
  leaf = unit_domain(u).leaf(r, r, "hm96-sha512/row/v2", hm96.SHA512.tree_leaf(KEY, b||c))

The layout (:class:`Layout`) is the step's public structure: groups (what one ``add`` commits, e.g. a layer), each of units,
each of rows, each row the concatenation of byte ranges (:class:`Part`) of the slot's row of the group's tensors. Every slot
has the same layout. A unit and a slot tree's domain don't name the slot or the step: the step tree's leaf index fixes the
slot, and the step domain the step. :func:`commit` is the reference commitment; :func:`open_unit` opens one unit of one slot
from that slot's rows alone (a replay of its request), and :func:`verify_unit` checks an opening against the step's root.

Parts, rows and groups, with their checks, are ``verity.ml.commitments.layout``'s, which also reads a layout file
(:func:`load`) in the form :func:`_layout_json` writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from verity.ml.commitments import hm96
from verity.ml.commitments import layout as L
from verity.ml.commitments.identity import identity_digest_sha512, tagged_sha512, tagged_sha512_prefix
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.layout import UNIT_MAX_BITS, Group, Part, Row
from verity.ml.commitments.merkle import CommitmentDomain, MerkleTree, domain_message
from verity.ml.commitments.row_table import ROOTS_SCHEMA, RootOpening, RootTree, verify_roots
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2, sha512_row_v2_digest
from verity.ml.crypto.chacha20 import keystream

__all__ = ["UNIT_MAX_BITS", "Group", "Layout", "Part", "Row", "StepCommit", "UnitOpening", "commit", "commit_slot",
           "edge_layout", "fixture", "fresh_salt_key", "h200_layout", "layout_digest", "load", "open_unit", "random_outputs",
           "row_salts", "slot_domain", "slot_rows", "step_domain", "step_tree", "unit_domain", "unit_domain_ids",
           "verify_unit"]

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


# --- the layout --------------------------------------------------------------------------------------------------------

class Layout(L.Layout):
    """A step's public structure, ``commitments.layout``'s (its groups in order, the same for every slot, a repeated group one
    object; ``name`` names it), with the indices and the digest the slot's trees take."""

    @cached_property
    def units(self) -> tuple[tuple[Row, ...], ...]:
        """Every unit, in order: unit u's rows."""
        return tuple(u for g in self.groups for u in g.units)

    @cached_property
    def group_rows(self) -> tuple[int, ...]:
        """Group i's rows."""
        rows: dict[int, int] = {}
        for g in self.groups:
            if id(g) not in rows:
                rows[id(g)] = sum(len(u) for u in g.units)
        return tuple(rows[id(g)] for g in self.groups)

    @cached_property
    def group_row_base(self) -> tuple[int, ...]:
        """Group i's first row g in its slot."""
        return tuple(int(n) for n in np.cumsum((0, *self.group_rows[:-1]), dtype=np.int64)) if self.groups else ()

    @cached_property
    def row_base(self) -> tuple[int, ...]:
        """Unit u's first row g in its slot."""
        n = np.fromiter((len(u) for u in self.units), dtype=np.int64, count=len(self.units))
        return tuple(np.concatenate(([0], np.cumsum(n)[:-1])).tolist()) if len(n) else ()

    @property
    def unit_count(self) -> int:
        return sum(len(g.units) for g in self.groups)

    @property
    def row_count(self) -> int:
        return sum(self.group_rows)

    @property
    def slot_bytes(self) -> int:
        """The bytes a slot commits."""
        return sum(r.nbytes for u in self.units for r in u)

    def manifest(self) -> dict[str, Any]:
        """What the layout digest binds: the name and every unit's row lengths in bits (where a row's bytes come from is the
        prover's plumbing, not the commitment's)."""
        return {"name": self.name, "units": [[r.bits for r in u] for u in self.units]}

    @cached_property
    def digest(self) -> bytes:
        """``identity_digest_sha512(LAYOUT_TAG, manifest())``, computed once: the manifest's canonical JSON written a group at
        a time, a group the layout repeats (its layers) written once."""
        text: dict[int, str] = {}
        for g in self.groups:
            if id(g) not in text:
                text[id(g)] = ",".join("[" + ",".join(str(r.bits) for r in u) + "]" for u in g.units)
        units = ",".join(t for t in (text[id(g)] for g in self.groups) if t)
        return tagged_sha512(LAYOUT_TAG, ('{"name":' + json.dumps(self.name, ensure_ascii=False) + ',"units":[' + units + "]}")
                             .encode("utf-8"))


def layout_digest(layout: Layout) -> bytes:
    return layout.digest


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


def unit_domain_ids(context: bytes, layout: Layout) -> bytes:
    """Every unit's ``unit_domain(context, layout, u).domain_id``, 64 bytes each in unit order, in one pass: the same bytes
    hashed, with each unit's manifest written straight into its canonical JSON and the domain message framed around its
    binding once per row count."""
    head = ('{"context":' + json.dumps(_context(context)) + ',"layout":' + json.dumps(layout_digest(layout).hex())
            + ',"rows":').encode()
    mid = (',"schema":' + json.dumps(ROW_SCHEMA, ensure_ascii=False) + ',"unit":').encode()
    tag = tagged_sha512_prefix(UNIT_TAG, 0)[:-8]
    frames: dict[int, tuple[bytes, bytes]] = {}
    out = bytearray(DIGEST_BYTES * layout.unit_count)
    sha512 = hashlib.sha512
    u = 0
    for g in layout.groups:
        for unit in g.units:
            n = len(unit)
            if n not in frames:
                frames[n] = _domain_frame(n)
            pre, post = frames[n]
            payload = b"%b%d%b%d}" % (head, n, mid, u)
            binding = sha512(tag + len(payload).to_bytes(8, "big") + payload).digest()
            out[DIGEST_BYTES * u:DIGEST_BYTES * (u + 1)] = sha512(pre + binding + post).digest()
            u += 1
    return bytes(out)


def _domain_frame(rows: int) -> tuple[bytes, bytes]:
    """A unit domain of ``rows`` rows: its domain message's bytes before and after its 64-byte binding."""
    positions = RangeIndexedDomain(0, rows).identity_sha512()
    lo, hi = (domain_message(bytes([b]) * DIGEST_BYTES, -1, positions, rows, "sha512") for b in (0, 255))
    k = next(i for i, (a, b) in enumerate(zip(lo, hi, strict=True)) if a != b)
    if lo[:k] != hi[:k] or lo[k + DIGEST_BYTES:] != hi[k + DIGEST_BYTES:]:
        raise AssertionError("a domain message holds its binding in one place")
    return lo[:k], lo[k + DIGEST_BYTES:]


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
    """The salts of rows ``first .. first + n - 1`` of ``slot``, 192 bytes each: ChaCha20 chunks 3g .. 3g + 2 under the step's
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


def _rows(source: int, *sizes: int) -> tuple[Row, ...]:
    return tuple(Row((Part(source, sum(sizes[:i]), n),)) for i, n in enumerate(sizes))


def edge_layout() -> Layout:
    """A small layout over SHA-512's chunk edges: rows of 1 to 240 bytes, a proof block of 4 rows (no pad), one of 3 (a pad), one of 1,
    a row of two parts (64 s8 codes, then their f32 scale) and one of 3 bytes; 5 proof blocks, so the slot tree pads 3."""
    g0 = Group((961,), (_rows(0, 1, 111, 112, 113), tuple(Row((Part(0, 337 + a, n),)) for a, n in ((0, 127), (127, 128), (255, 129))),
                        (Row((Part(0, 721, 240),)),)))
    g1 = Group((64, 4), ((Row((Part(0, 0, 64), Part(1, 0, 4))),), (Row((Part(1, 1, 3),)),)))
    return Layout("commit-slots/edge/v1", (g0, g1))


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "commit-slots-v1.json"


def _layout_json(layout: Layout) -> dict[str, Any]:
    return {"name": layout.name, "groups": [{"sources": list(g.sources), "units": [[[[p.source, p.offset, p.nbytes] for p in r.parts]
                                                                                   for r in u] for u in g.units]}
                                            for g in layout.groups]}


def load(path: str | os.PathLike[str]) -> Layout:
    """The layout in the file ``path`` (:func:`_layout_json`'s form), read and checked by ``commitments.layout.load``: equal
    groups are one object, so a plan lays each out once."""
    lay = L.load(path)
    return Layout(lay.name, lay.groups)


def fixture() -> dict[str, Any]:
    """The interface's fixture: :func:`edge_layout` at 3 slots under a public test context and salt key, every value a consumer
    checks against (domain ids, each row's digest, salt and string in slot 0, every slot's unit roots and root, the step's root),
    one unit's opening, and the frozen H200 layout's public numbers."""
    layout, slots, step = edge_layout(), 3, 7
    context = identity_digest_sha512("verity/commit-slots/fixture/context/v1", {"fixture": 1})
    salt_key = bytes(range(32))
    outputs = random_outputs(layout, slots, np.random.default_rng(20261026))
    per_slot = []
    for s in range(slots):
        rows = slot_rows(layout, outputs, s)
        sc = commit_slot(context, layout, salt_key, s, rows)
        doc: dict[str, Any] = {"strings": [x.hex() for x in sc.strings], "unit_roots": [t.commitment.root.hex() for t in sc.units],
                               "root": sc.root.hex()}
        if s == 0:
            salts = row_salts(salt_key, s, 0, len(rows))
            doc["rows"] = [{"bytes": r.hex(), "bits": 8 * len(r), "x": sha512_row_v2_digest(r, 8 * len(r)).hex(),
                            "salt": salts[SALT_BYTES * i:SALT_BYTES * (i + 1)].hex()} for i, r in enumerate(rows)]
        per_slot.append(doc)
    roots = [bytes.fromhex(d["root"]) for d in per_slot]
    step_root = step_tree(context, layout, step, roots).root
    slot, unit = 1, 1
    rows1 = slot_rows(layout, outputs, slot)
    opening = open_unit(context, layout, step, salt_key, roots, slot, unit, rows1)
    b, n = layout.row_base[unit], len(layout.units[unit])
    salts1 = row_salts(salt_key, slot, b, n)
    h200 = h200_layout()
    return {
        "schema": "verity/commit-slots/fixture/v1",
        "about": "A step's unit outputs as hm96-sha512/row/v2 rows under hiding leaves in frame-v3-sha512 trees, "
                 "step tree over slot trees over unit trees over rows (benchmarks/torch_gpu/commit_slots/README.md). The context "
                 "and salt key are public test values.",
        "tags": {"layout": LAYOUT_TAG, "unit": UNIT_TAG, "slot": SLOT_TAG, "step": STEP_TAG, "row_schema": ROW_SCHEMA,
                 "roots_schema": ROOTS_SCHEMA, "hm96": INST.name},
        "context": context.hex(), "step": step, "slots": slots, "salt_key": salt_key.hex(),
        "layout": _layout_json(layout), "layout_manifest": layout.manifest(), "layout_digest": layout_digest(layout).hex(),
        "outputs": [[a.tobytes().hex() for a in tensors] for tensors in outputs],
        "domains": {"units": [unit_domain(context, layout, u).domain_id.hex() for u in range(layout.unit_count)],
                    "slot": slot_domain(context, layout).domain_id.hex(),
                    "step": step_domain(context, layout, step, slots).domain_id.hex()},
        "per_slot": per_slot,
        "root": step_root.hex(),
        "opening": {**opening.to_json(), "rows": [[r.hex(), salts1[SALT_BYTES * i:SALT_BYTES * (i + 1)].hex()]
                                                  for i, r in enumerate(rows1[b:b + n])]},
        "h200": {"name": h200.name, "layers": len(h200.groups), "units_per_layer": len(h200.groups[0].units),
                 "unit_bits": [[r.bits for r in u] for u in h200.groups[0].units], "slot_bytes": h200.slot_bytes,
                 "step_bytes_at_20480_slots": 20480 * h200.slot_bytes, "layout_digest": layout_digest(h200).hex()},
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fixture", help="write the interface's fixture")
    f.add_argument("--out", type=Path, default=FIXTURE)
    a = ap.parse_args(argv)
    if a.cmd == "fixture":
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(fixture(), indent=1) + "\n")
        print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
