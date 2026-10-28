"""The drawn-unit replay bundle: everything the prover behind the firewall reads to prove the drawn units
of one window, one JSON document (``verity/drawn-unit-bundle/v2``) and one binary payload of the private bytes.

The bundle is the prover's: its payload holds values and salts, so it is written mode 0600, never published and never put in
the evidence store (a test fixture's salts are a test key's). The JSON names every private byte by its ``[offset, length]``
in the payload and holds only public structure, hiding commitments and paths beside it.

**The document.**

* ``window``: the window's run root, registration digest, circuit record SHA-512, partition and program digests, scheme,
  population and leaf layer (``registration.json`` / ``window.json``), and ``roots``: every port root a unit's opening
  reaches, ``{port: {root, leaves, domain, schema}}``, each the registration's root under the window-domain rule
  (`verity.ml.commitments.window`).
* ``target``: the card's binding of the templates' dot, ``{binding, card, dot}`` (`BINDINGS`: ``GEMM_SM90_S8`` for the
  H200's INT8 and ``GEMM_HOPPER`` for its BF16, ``GEMM_SM120_S8`` for the RTX PRO 6000; ``BOOLEAN_…`` the same cards' dots
  on bits, which the Boolean-profile templates bind). Every unit's template binds that dot, as circuits resolves the binding.
* ``draw``: the drawn units (global indices), the law, and ``source``: ``live`` (the proof system's draw of record, with
  ``receipt``, the challenger's live-coin receipt, which a live draw must carry) or ``fixture`` (chosen by hand). The law is
  the window's registered one, fixed before any coin, and a live draw's ``record`` (the verifier's draw object, when it
  carries one) is a draw under it over the registered population, of exactly the bundle's units (`check` against the
  registration).
* ``weights``: the registered values the units read (core's ``one_stage.registered``, as `e2e_h200.served_weights`
  registers the seeded weights: ``registered.domain(program, value, words, leaves, word_bits)``, rows in
  ``hm96-sha512/row/v2``): ``{program, values: [{value, root, leaves, words, word_bits, domain, row_schema}]}``.
* ``commitment``: the serving steps' commitment (kernels' commit call, `commit_slots`: step tree over
  slot trees over unit trees over rows, only each unit's outputs committed, one row ``hm96-sha512/row/v2`` under the step's
  salt key, C-Flock's ``hm96_salt`` with the slot as nonce). ``{format, reference, context, layout, layout_digest, slots,
  blocks: [{block, root, domain, steps: [{step, root, domain}]}]}``: the run's 64-byte context, the step layout (groups of
  units of rows of byte ranges, `layout_json`), and per server-time block every step's root (what went to the challenger
  before the step's live coin) under this bundle's block tree over them (`block_domain`).
* ``members``: per member with a drawn unit, what C-Flock's ``--zk`` session reads today (`benchmarks/one_stage/a5.py`): its
  key, template, base and count, the drawn units' local indices and the session's subset draw over them, its header's
  ``frame_v3`` entry, its public body's SHA-512 and row schemas. The prover's input file is
  ``flock_inputs.flock_input_files(window, k, circuit, out, write=("inst",))`` under the circuit the verifier composed for
  the template (``compose_circuit.py``, bound to ``circuit.partition`` and ``circuit.program``).
* ``units``: per drawn unit ``{unit, coords, template, member, rows}``. ``coords`` is ``{block, step, slot, layer, op,
  unit}`` (``unit``: the coordinate within its op at that slot and step). ``template`` is ``{id, definition, binding,
  rows, outs}``: the descriptor id, the Definition's name, its binding statics as JSON (`interface._bindings`) and its bit
  rows. ``rows`` is each port's row in the member's port order, inputs then outputs: ``{port, kind: in|out, schema, words,
  row: [offset, length], salt: [offset, length], string, window: {port, leaf, path}, slot | null, registered | null}``.
  ``string`` is the row's public ``b || c`` (hex) and every row opens against its window port's registered root.
  Each row opens against exactly one commitment:

  - ``slot``: the step commitment. ``{block, step, slot, unit, rank, path, slot_path, step_path, producer}``: ``path`` the
    row's frame-v3 siblings in its unit's tree (leaf level first, hex), ``slot_path`` the unit root's opening in the slot
    tree and ``step_path`` the slot root's in the step tree (`row_table.RootOpening` JSON). An output row opens in the
    unit's own commitment unit (``producer`` null); an input row opens where its producer committed it as an output, and
    ``producer`` names that unit's coordinates ``{block, step, slot, layer, op, unit}`` (``unit`` its index in the layout).
    Every opening is in the unit's own slot and step: one request's replay opens a drawn unit.
  - ``registered``: a registered read (the weights), committed under the registered row's salt, opening against the
    value's root: ``{value, position, path}``. It is in no step commitment.
  - ``reads``: a row made of pieces of committed rows (`Read`; `WindowLocate` takes the H200 window writer's), each from
    its unit's own slot and block at its own step or an earlier one (an attention block at step s reads the K and V rows
    of steps 0 .. s): ``{kind: step, at, words, start, row, salt, string, slot}``, its committed row's bytes and salt in
    the payload, ``slot`` its opening as above (``producer`` optional) against its own step's root; a position word
  ``{kind: position, at, offset}``, the unit's step plus ``offset``; or a piece of a registered row (`RegisteredRead`: a
  norm's gain, one registered word a row) ``{kind: registered, at, words, start, value, position, row, salt, string,
  path}``, its row's bytes and salt in the payload and ``path`` its opening against the value's root. Words no piece
  covers are zero, and a row of no pieces is all zero (a V row past n, which the Definition masks).

  With ``registration``, `check` holds every step root the bundle opens against to the one the registration's
  ``window.commitment`` names (`steps_provenance`, every step's root registered before the coin) and fails closed on a
  step it doesn't name.

  An output row (``kind: out``) is the unit's committed output: a value, in the payload only.

**The payload** is every drawn unit's rows in bundle order, each row's words (u16 little-endian) then its 192-byte salt,
then each of its step reads' committed row and salt.

`check` is the consumer's check of a bundle from its bytes alone: the payload's digest, every string recomputed from its row
and salt, every path to its root, every root's domain recomputed from public data (the window's from the registration's
fields, a registered value's from its entry, the commitment's from its context, layout and coordinates), the draw's receipt
when live, the target's dot in every template, and each unit's Definition replayed on its input rows against its output rows.
`build` writes a bundle from a window ``Window.write`` wrote and the prover's step records (`Commitment`).
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import multiprocessing
import os
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from verity.core.protocols.sampled_proofs.one_stage import draw as D, registered as REG
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.commitments import hm96
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, MerkleTree, merkle_depth
from verity.ml.commitments.row_table import ROOTS_SCHEMA, RootOpening, RootTree, verify_roots
from verity.ml.commitments.window import window_domain
from verity.ml.windows import interface as IF, rows as CS

COMMIT_SLOTS = Path(__file__).resolve().parents[4] / "benchmarks" / "torch_gpu" / "commit_slots"
if str(COMMIT_SLOTS) not in sys.path:
    sys.path.insert(0, str(COMMIT_SLOTS))
import slots as S  # noqa: E402

FORMAT = "verity/drawn-unit-bundle/v2"
COMMITMENT = "verity/commit-slots/v1"
REFERENCE = {"path": "benchmarks/torch_gpu/commit_slots", "commit": "05038d5ab4d481218df7443d3ff9fe661a9f5256", "pr": 4469}
BLOCK_TAG = "verity/drawn-unit-bundle/block/v1"
PAYLOAD_LAYOUT = ("each drawn unit in bundle order, each row in its member's port order: the row's u16 words little-endian, "
                  "then its 192-byte salt, then each of its step reads' committed row bytes and 192-byte salt")
INST = hm96.SHA512
SALT = INST.salt_bytes
COORDS = ("block", "step", "slot", "layer", "op", "unit")
#: a card's binding of the templates' dot, by name: (card, the binding's statics as circuits resolves them)
BINDINGS = {"GEMM_SM120_S8": ("sm_120", G.GEMM_SM120_S8), "GEMM_SM90_S8": ("sm_90", G.GEMM_SM90_S8),
            "GEMM_HOPPER": ("sm_90", G.GEMM_HOPPER),
            "BOOLEAN_SM120_S8": ("sm_120", BG.GEMM_SM120_S8), "BOOLEAN_SM90_S8": ("sm_90", BG.GEMM_SM90_S8),
            "BOOLEAN_HOPPER": ("sm_90", BG.GEMM_HOPPER)}

@dataclass(frozen=True)
class Read:
    """Words of a port's row read from one committed row of the step commitment: the row is rank ``rank`` of layout unit
    ``unit`` in slot ``slot`` of step ``step`` (the reading unit's own slot and block, at its own step or an earlier one: a
    K or V row the attention reads), its bytes ``start .. start + nbytes`` (``nbytes`` None: to the row's end) are the
    port row's words from ``at``, and ``producer`` (``{layer, op, unit}``, optional) is the unit that committed it as an
    output. An output row made of reads reads its own unit's rows at its own step (no producer)."""

    step: int
    slot: int
    unit: int
    rank: int
    at: int = 0
    start: int = 0
    nbytes: int | None = None
    producer: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Position:
    """A run input of kind position, one word of the port row at ``at``: the reading unit's step plus ``offset`` (a key
    count n = position + 1 is offset 1, a position p offset 0; in a lockstep run a slot's position is the step)."""

    offset: int
    at: int = 0

    #: its read entry's kind in a bundle row: a record of the bundle, not a circuit port (tests/test_seed_ports.py
    #: scans dict literals of that kind as ports)
    KIND = "position"


@dataclass(frozen=True)
class RegisteredRead:
    """Words of a port's row read from row ``position`` of registered value ``value``: its bytes ``start .. start + nbytes``
    (``nbytes`` None: to the row's end) are the port row's words from ``at``. A port of many registered rows (a norm's gain,
    one word a row) reads each so; a port that is one whole registered row is a registered read (`Reads`)."""

    value: str
    position: int
    at: int = 0
    start: int = 0
    nbytes: int | None = None


Coords = Callable[[Any, int], dict[str, Any]]
Reads = Callable[[Mapping[str, Any], str, str], "tuple[str, int] | None"]
#: a unit's port's row in the step commitment: (layout unit, rank, the producer's coordinates or None for the unit's own) for
#: one whole row at the unit's own step and slot, or the `Read`s, `Position`s and `RegisteredRead`s it is made of (words no
#: piece covers are zero: absent K and V rows, and none at all for a V row past n); None for a registered read
Locate = Callable[[Mapping[str, Any], str, str],
                  "tuple[int, int, dict[str, Any] | None] | Sequence[Read | Position | RegisteredRead] | None"]


@dataclass(frozen=True)
class StepRecord:
    """What the prover keeps of one step's commitment: its salt key and its slots' rows, in layout order. ``rows``: every
    slot's (a sequence), or only the slots a bundle opens (``{slot: rows}``) with every slot's root in ``slot_roots``."""

    salt_key: bytes
    rows: Sequence[Sequence[bytes]] | Mapping[int, Sequence[bytes]]
    slot_roots: tuple[bytes, ...] = ()


@dataclass(frozen=True)
class Commitment:
    """The run's step commitments, the prover's side: its context, step layout, slots, and each ``(block, step)``'s record."""

    context: bytes
    layout: S.CommitShape
    slots: int
    steps: Mapping[tuple[int, int], StepRecord]


def target(binding: str) -> dict[str, Any]:
    """The ``target`` of a bundle whose templates bind the dot of ``binding`` (`BINDINGS`)."""
    if binding not in BINDINGS:
        raise ValueError(f"{binding}: not one of {sorted(BINDINGS)}")
    card, statics = BINDINGS[binding]
    return {"binding": binding, "card": card, "dot": statics["DOT"].id}


def block_domain(context: bytes, layout: S.CommitShape, block: int, steps: Sequence[int]) -> CommitmentDomain:
    """This bundle's tree over one server-time block's step roots, in ascending step order."""
    return CommitmentDomain(identity_digest_sha512(BLOCK_TAG, {"context": context.hex(), "layout": S.layout_digest(layout).hex(),
                                                               "block": block, "steps": list(steps), "schema": ROOTS_SCHEMA}),
                            -1, RangeIndexedDomain(0, len(steps)), hash="sha512")


def layout_json(layout: S.CommitShape) -> dict[str, Any]:
    """The layout as JSON, a group the layout repeats (the same object: its layers) written once and then as the index of
    its first place, so an H200 step's six million units are three groups' worth."""
    first: dict[int, int] = {}
    groups: list[Any] = []
    for i, g in enumerate(layout.groups):
        if id(g) in first:
            groups.append(first[id(g)])
            continue
        first[id(g)] = i
        groups.append({"sources": list(g.sources),
                       "units": [[[[p.source, p.offset, p.nbytes] for p in r.parts] for r in u] for u in g.units]})
    return {"name": layout.name, "groups": groups}


def layout_of_json(doc: Mapping[str, Any]) -> S.CommitShape:
    groups: list[S.Group] = []
    for g in doc["groups"]:
        if isinstance(g, int) and not isinstance(g, bool):
            if not 0 <= g < len(groups) or isinstance(doc["groups"][g], int):
                raise ValueError(f"group {len(groups)} repeats {g}, not an earlier group written out")
            groups.append(groups[g])
            continue
        groups.append(S.Group(tuple(int(n) for n in g["sources"]),
                              tuple(tuple(S.Row(tuple(S.Part(*map(int, p)) for p in r)) for r in u) for u in g["units"])))
    return S.CommitShape(str(doc["name"]), tuple(groups))


def _leaf_value(bc: bytes) -> bytes:
    return INST.tree_leaf(INST.default_key, bc)


def _path(tree: MerkleTree, rank: int) -> list[str]:
    return [p.hex() for p in tree.open(rank).path]


def _fold(domain: CommitmentDomain, rank: int, schema: str, value: bytes, path: Sequence[str]) -> bytes | None:
    """The root ``value`` at ``rank`` reaches through ``path`` under ``domain``, or None for a path of the wrong depth."""
    if not 0 <= rank < domain.count or len(path) != merkle_depth(domain.count):
        return None
    digest, i = domain.leaf(rank, rank, schema, value), rank
    for level, sib in enumerate(path):
        s = bytes.fromhex(sib)
        digest = domain.node(level, i >> 1, *((digest, s) if i % 2 == 0 else (s, digest)))
        i >>= 1
    return digest


def string_of(schema: str, row: bytes, salt: bytes) -> bytes:
    """A row's ``b || c`` from its u16 words' bytes and its salt, under its row schema (`rows._inner`)."""
    return INST.commit_string(INST.default_key, CS._inner(schema, len(row) // 2)(row), salt)


def _template(tid: str) -> dict[str, Any]:
    m = IF._ID.fullmatch(tid)
    if m is None:
        raise ValueError(f"{tid}: not a descriptor id")
    lay = IF.layout(IF.resolve(tid))
    return {"id": tid, "definition": m[1], "binding": IF._bindings(m[2]) if m[2] else {},
            "rows": [list(r) for r in lay.rows], "outs": [list(o) for o in lay.outs]}


def _dot_of(template: Mapping[str, Any]) -> str | None:
    dot = template["binding"].get("DOT")
    return dot.get("fn") if isinstance(dot, Mapping) else dot


def _draw(draw: Mapping[str, Any] | None, units: list[int]) -> dict[str, Any]:
    d = {"source": "fixture", **dict(draw or {}), "units": units}
    if d["source"] not in ("live", "fixture"):
        raise ValueError(f"draw source {d['source']!r}: live or fixture")
    if d["source"] == "live" and not d.get("receipt"):
        raise ValueError("a live draw carries the challenger's live-coin receipt (draw.receipt)")
    return d


class _Window:
    """A written window's files, read once: members, labels, bodies."""

    def __init__(self, d: Path) -> None:
        self.dir = d
        self.win = json.loads((d / "window.json").read_text())
        self.rec = json.loads((d / "registration.json").read_text())
        self.index = json.loads((d / "index.json").read_text())
        self.members = self.win["members"]
        for m in self.members:
            if m.get("shared_rows") or m.get("row_links") or m.get("wired"):
                raise ValueError(f"{m['name']}: a shared-row, row-table or wired member; the bundle reads members of whole rows")
        self._spans = _run_starts(self.index)
        self._lo = [a for a, _, _, _ in self._spans]
        self._pub: dict[int, bytes] = {}
        self._priv: dict[int, bytes] = {}

    def member_of(self, u: int) -> int:
        for k, m in enumerate(self.members):
            if m["base"] <= u < m["base"] + m["n"]:
                return k
        raise ValueError(f"unit {u} is in no member of the window's {self.win['population']}")

    def label(self, u: int) -> tuple[Any, int]:
        j = bisect.bisect_right(self._lo, u) - 1
        if j < 0 or not u < self._spans[j][1]:
            raise ValueError(f"unit {u} has no label in index.json")
        return self._spans[j][2], u - self._spans[j][3]

    def public(self, k: int) -> bytes:
        if k not in self._pub:
            self._pub[k] = (self.dir / f"m{k}-public.bin").read_bytes()
        return self._pub[k]

    def private(self, k: int) -> bytes:
        if k not in self._priv:
            self._priv[k] = (self.dir / f"m{k}-private.bin").read_bytes()
        return self._priv[k]

    def rows_of(self, k: int, i: int) -> list[tuple[str, str, int, bytes, bytes, bytes]]:
        """Member ``k``'s instance ``i``: each port's ``(port, schema, words, row, salt, string)``."""
        m = self.members[k]
        ports, n = m["ports"], m["n"]
        total = sum(w for _, w in ports)
        priv, pub = self.private(k), self.public(k)
        rows_at, salts_at = 2 * total * i, 2 * total * n + SALT * len(ports) * i
        out, wo = [], 0
        for j, (p, w) in enumerate(ports):
            row = priv[rows_at + 2 * wo:rows_at + 2 * (wo + w)]
            salt = priv[salts_at + SALT * j:salts_at + SALT * (j + 1)]
            bc = pub[CS.BC_BYTES * (len(ports) * i + j):CS.BC_BYTES * (len(ports) * i + j + 1)]
            out.append((p, m["frame_v3"]["schemas"][p], w, row, salt, bc))
            wo += w
        return out

    def strings(self, k: int, p: str) -> list[bytes]:
        m = self.members[k]
        j = [q for q, _ in m["ports"]].index(p)
        P, pub = len(m["ports"]), self.public(k)
        return [pub[CS.BC_BYTES * (P * i + j):CS.BC_BYTES * (P * i + j + 1)] for i in range(m["n"])]


def _run_starts(index: Mapping[str, Any]) -> list[tuple[int, int, Any, int]]:
    """Every labelled span of ``index.json`` as ``(lo, hi, label, start)``, ``start`` where its recorded run begins: members
    cut a run, so a span may start inside one, and a run's spans are the ones with its label (a label names one run)."""
    spans = sorted((a, b, x) for m in index["members"] for a, b, x in m["labels"])
    first: dict[str, int] = {}
    for a, _, x in spans:
        first.setdefault(json.dumps(x, sort_keys=True), a)
    return [(a, b, x, first[json.dumps(x, sort_keys=True)]) for a, b, x in spans]


@dataclass(frozen=True)
class _Slot:
    """One slot's subtree as `_Steps.slot` rebuilds it: every row's string, the kept units' trees, the slot tree."""

    strings: Sequence[bytes]
    units: Mapping[int, MerkleTree]
    tree: RootTree

    @property
    def root(self) -> bytes:
        return self.tree.root


class _Steps:
    """The step commitments a bundle opens, rebuilt from the prover's records: a slot's subtree from its rows (one at a time:
    at the H200's six million units a slot, a bundle can't hold many), step trees over the slot roots, block trees."""

    def __init__(self, com: Commitment) -> None:
        self.com = com
        self._roots: dict[tuple[int, int], tuple[bytes, ...]] = {}
        self._steps: dict[tuple[int, int], RootTree] = {}

    def record(self, b: int, s: int) -> StepRecord:
        rec = self.com.steps.get((b, s))
        if rec is None:
            raise ValueError(f"block {b} step {s}: no step record")
        return rec

    def rows(self, b: int, s: int, j: int) -> Sequence[bytes]:
        rows = self.record(b, s).rows
        if isinstance(rows, Mapping):
            if j not in rows:
                raise ValueError(f"block {b} step {s}: no rows kept of slot {j}")
        elif len(rows) != self.com.slots:
            raise ValueError(f"block {b} step {s}: {len(rows)} slots' rows for {self.com.slots} slots")
        return rows[j]

    def slot(self, b: int, s: int, j: int, units: Iterable[int] = ()) -> _Slot:
        """Slot ``j``'s subtree (`commit_slot`'s), keeping only ``units``' trees: at the H200's six million units a slot,
        every unit's tree is gigabytes."""
        rec, com = self.record(b, s), self.com
        rows, keep = self.rows(b, s, j), set(units)
        if len(rows) != com.layout.row_count:
            raise ValueError(f"block {b} step {s} slot {j}: {len(rows)} rows for the layout's {com.layout.row_count}")
        strings = S._strings(rows, S.row_salts(rec.salt_key, j, 0, len(rows)))
        roots, trees = [], {}
        for u, (base, rows_u) in enumerate(zip(com.layout.row_base, com.layout.units, strict=True)):
            t = S._unit_tree(S.unit_domain(com.context, com.layout, u), strings[base:base + len(rows_u)])
            roots.append(t.commitment.root)
            if u in keep:
                trees[u] = t
        sc = _Slot(strings, trees, RootTree(S.slot_domain(com.context, com.layout), roots))
        if rec.slot_roots and sc.root != rec.slot_roots[j]:
            raise ValueError(f"block {b} step {s} slot {j}: its rows are not its committed slot root")
        return sc

    def roots(self, b: int, s: int) -> tuple[bytes, ...]:
        if (b, s) not in self._roots:
            rec = self.record(b, s)
            roots = tuple(rec.slot_roots) or tuple(self.slot(b, s, j).root for j in range(self.com.slots))
            if len(roots) != self.com.slots:
                raise ValueError(f"block {b} step {s}: {len(roots)} slot roots for {self.com.slots} slots")
            self._roots[b, s] = roots
        return self._roots[b, s]

    def step(self, b: int, s: int) -> RootTree:
        if (b, s) not in self._steps:
            self._steps[b, s] = S.step_tree(self.com.context, self.com.layout, s, self.roots(b, s))
        return self._steps[b, s]

    def blocks(self) -> list[dict[str, Any]]:
        out = []
        for b in sorted({b for b, _ in self.com.steps}):
            steps = sorted(s for bb, s in self.com.steps if bb == b)
            roots = [self.step(b, s).root for s in steps]
            dom = block_domain(self.com.context, self.com.layout, b, steps)
            out.append({"block": b, "root": RootTree(dom, roots).root.hex(), "domain": dom.domain_id.hex(),
                        "steps": [{"step": s, "root": r.hex(), "domain": self.step(b, s).domain.domain_id.hex()}
                                  for s, r in zip(steps, roots, strict=True)]})
        return out


def steps_provenance(commitment: Commitment) -> dict[str, Any]:
    """What a window's registration names of the run's step commitments, under its ``window`` provenance (``Window.write``'s
    ``provenance``): ``{"commitment": {format, context, layout_digest, slots, blocks}}``, every block's root and every step's
    root and domain, as a bundle's ``commitment`` has them. So the registration's digest fixes every step root a drawn
    unit can read (attention's K and V rows at every earlier step) before the coin, and `check` holds a bundle's openings
    to them."""
    com = commitment
    return {"commitment": {"format": COMMITMENT, "context": com.context.hex(), "layout_digest": S.layout_digest(com.layout).hex(),
                           "slots": com.slots, "blocks": _Steps(com).blocks()}}


def _pieces(at: Any) -> list[Read | Position | RegisteredRead] | None:
    """A `Locate` answer as its pieces (none: an all-zero row), or None for the one whole row of the legacy 3-tuple."""
    if isinstance(at, tuple) and len(at) == 3 and all(isinstance(x, int) for x in at[:2]):
        return None
    pieces = list(at)
    if not all(isinstance(x, (Read, Position, RegisteredRead)) for x in pieces):
        raise ValueError(f"{at!r}: a whole row (unit, rank, producer), or Reads, Positions and RegisteredReads")
    return pieces


class WindowLocate:
    """`build`'s ``locate`` and ``reads`` from the H200 window writer's ``Window.locate(step, slot, family, index, port)``
    (circuits' interface, ``demos/h200/e2e_h200/window.py``), whose pieces are the port's bytes in
    order: ``Segment(step, slot, row, offset, nbytes)`` (bytes of a committed row, ``row`` its index in the slot: a `Read`),
    ``Absent(nbytes)`` (zero bytes, a key at or past n), ``Run(name)`` (``n``, the position + 1, one word: a `Position`) and
    ``Registered(value, row, offset, nbytes)`` (bytes of a registered weight). A port of one Registered piece is a registered
    read: its position is the piece's byte offset in the value, ``row * stride(value) + offset``, over the registered row's
    bytes, which the piece must be exactly (``stride`` defaults to one registered row: ``row`` is the registered row, as
    circuits' `Locator.locate` names the weights registration's rows). ``Run("token")`` is ingress, which no commitment holds: refused. Coordinates
    are ``{step, slot, op: the family, unit: the block's index in it}``."""

    def __init__(self, window: Any, layout: S.CommitShape, weights: Mapping[str, Any] | None = None,
                 stride: Callable[[str], int] | None = None) -> None:
        self.window, self.layout, self.weights, self.stride = window, layout, dict(weights or {}), stride
        self._base = list(layout.row_base[:layout.unit_count])
        self._got: dict[tuple[Any, ...], Any] = {}

    def _answer(self, c: Mapping[str, Any], port: str) -> Any:
        key = (c["block"], c["step"], c["slot"], c["op"], c["unit"], port)
        if key not in self._got:
            self._got[key] = self._convert(self.window.locate(c["step"], c["slot"], c["op"], c["unit"], port), port)
        return self._got[key]

    def _convert(self, pieces: Sequence[Any], port: str) -> Any:
        kinds = [type(p).__name__ for p in pieces]
        if "Registered" in kinds:
            if kinds != ["Registered"]:
                raise ValueError(f"{port}: a registered read is the port's one piece")
            p = pieces[0]
            e = self.weights[p.value].entry()
            nbytes = e["words"] * e.get("word_bits", REG.WORD_BITS) // 8
            flat = p.row * (self.stride(p.value) if self.stride else nbytes) + p.offset
            if p.nbytes != nbytes or flat % nbytes:
                raise ValueError(f"{port}: {p} is not one whole {nbytes}-byte row of {p.value}")
            return ("registered", p.value, flat // nbytes)
        out: list[Read | Position] = []
        at = 0
        for p, k in zip(pieces, kinds, strict=True):
            if at % 2:
                raise ValueError(f"{port}: piece {p} starts inside a word")
            if k == "Segment":
                j = bisect.bisect_right(self._base, p.row) - 1
                if j < 0 or not p.row - self._base[j] < len(self.layout.units[j]):
                    raise ValueError(f"{port}: {p} names no row of the layout")
                out.append(Read(p.step, p.slot, j, p.row - self._base[j], at // 2, p.offset, p.nbytes))
                at += p.nbytes
            elif k == "Absent":
                at += p.nbytes
            elif k == "Run" and p.name == "n":
                out.append(Position(1, at // 2))
                at += 2
            else:
                raise ValueError(f"{port}: {p} is in no commitment (a run's ingress is not opened)")
        return out or ("absent",)

    def locate(self, c: Mapping[str, Any], template: str, port: str) -> Any:
        a = self._answer(c, port)
        if a[:1] == ("registered",):
            return None
        if a == ("absent",):
            raise ValueError(f"{port}: no piece in any commitment")
        return a

    def reads(self, c: Mapping[str, Any], template: str, port: str) -> tuple[str, int] | None:
        a = self._answer(c, port)
        return (a[1], a[2]) if a[:1] == ("registered",) else None


#: an opening to fill once its slot's subtree is rebuilt: ``(unit, rank, the string the row must have or None, the opening
#: dict, what refuses it)``, by ``(block, step, slot)``
Wants = dict[tuple[int, int, int], list[tuple[int, int, "bytes | None", dict[str, Any], str]]]


def _opener(want: Wants) -> Callable[..., dict[str, Any]]:
    def opening(b: int, s: int, slot: int, lu: int, rank: int, producer: Mapping[str, Any] | None, bc: bytes | None,
                why: str) -> dict[str, Any]:
        o = {"block": b, "step": s, "slot": slot, "unit": lu, "rank": rank, "path": None, "slot_path": None, "step_path": None,
             "producer": None if producer is None else {"block": b, "step": s, "slot": slot, **producer}}
        want.setdefault((b, s, slot), []).append((lu, rank, bc, o, why))
        return o
    return opening


def _slot_paths(steps: _Steps, want: Wants, key: tuple[int, int, int]) -> list[dict[str, Any]]:
    """The paths of the openings wanted in one slot of one step, from its rebuilt subtree."""
    b, s, slot = key
    items, layout = want[key], steps.com.layout
    sc, step_path = steps.slot(b, s, slot, {lu for lu, *_ in items}), steps.step(b, s).open([slot]).to_json()
    out = []
    for lu, rank, bc, _o, why in items:
        if bc is not None and sc.strings[layout.row_base[lu] + rank] != bc:
            raise ValueError(f"{why} (other bytes or salts)")
        out.append({"path": _path(sc.units[lu], rank), "slot_path": sc.tree.open([lu]).to_json(), "step_path": step_path})
    return out


def _task(steps: _Steps, want: Wants, key: tuple[int, int, int]) -> tuple[Any, ...]:
    """What a worker needs to rebuild one (step, slot): that slot's rows alone, under its step's slot roots."""
    b, s, slot = key
    com = steps.com
    rec = StepRecord(steps.record(b, s).salt_key, {slot: steps.rows(b, s, slot)}, steps.roots(b, s))
    return com.context, com.layout, com.slots, key, rec, [(lu, rank, bc, None, why) for lu, rank, bc, _o, why in want[key]]


def _worker_paths(task: tuple[Any, ...]) -> list[dict[str, Any]]:
    context, layout, slots, key, rec, items = task
    return _slot_paths(_Steps(Commitment(context, layout, slots, {key[:2]: rec})), {key: items}, key)


def _fill(want: Wants, steps: _Steps, workers: int = 1) -> None:
    """Every wanted opening's paths, a slot's subtree at a time; with ``workers`` > 1 in spawned processes, one (step, slot)
    each, sent its slot's rows alone (never forked: the prover's process holds threads and a GPU)."""
    keys = sorted(want)
    if workers > 1 and len(keys) > 1:
        with multiprocessing.get_context("spawn").Pool(min(workers, len(keys)), maxtasksperchild=1) as pool:
            got = pool.map(_worker_paths, (_task(steps, want, k) for k in keys), chunksize=1)
    else:
        got = [_slot_paths(steps, want, k) for k in keys]
    for k, paths in zip(keys, got, strict=True):
        for (_lu, _rank, _bc, o, _why), p in zip(want[k], paths, strict=True):
            o.update(p)


def _read_entries(who: str, c: Mapping[str, Any], kind: str, row: bytes, pieces: Sequence[Read | Position | RegisteredRead],
                  steps: _Steps, payload: bytearray, opening: Callable[..., dict[str, Any]],
                  weights: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """A port row's ``reads``: each step read's committed row and salt appended to ``payload`` and its opening wanted, each
    registered piece's row and salt appended with its path (``weights``: the values, `build`'s); refused when the row is
    not the pieces' bytes and zero elsewhere, or a read is outside the unit's slot, after its step, or (for an output)
    anywhere but its own step, or an output reads a registered row."""
    layout, words = steps.com.layout, len(row) // 2
    out: list[dict[str, Any]] = []
    covered = np.zeros(words, dtype=bool)
    for q in pieces:
        if isinstance(q, Position):
            if kind == "out" or q.offset not in (0, 1) or not 0 <= q.at < words or covered[q.at] \
                    or int.from_bytes(row[2 * q.at:2 * q.at + 2], "little") != c["step"] + q.offset:
                raise ValueError(f"{who}: word {q.at} is not its step + {q.offset}, or a position in an output")
            covered[q.at] = True
            out.append({"kind": Position.KIND, "at": q.at, "offset": q.offset})
            continue
        if isinstance(q, RegisteredRead):
            reg = (weights or {}).get(q.value)
            if kind == "out" or reg is None:
                raise ValueError(f"{who}: {q} reads a registered value the bundle has no rows of, or is in an output")
            crow, csalt = bytes(reg.row_bytes(q.position)), bytes(reg.salt(q.position))
            cs = reg.string(q.position) if hasattr(reg, "string") else reg.bcs[q.position]
            if string_of(reg.schema, crow, csalt) != cs:
                raise ValueError(f"{who}: {q.value} row {q.position}'s bytes and salt are not its registered string")
            n = len(crow) - q.start if q.nbytes is None else q.nbytes
            nw = n // 2
            if q.start % 2 or n % 2 or nw < 1 or not (0 <= q.start and q.start + n <= len(crow) and 0 <= q.at
                                                      and q.at + nw <= words) or covered[q.at:q.at + nw].any():
                raise ValueError(f"{who}: read {q} is outside its row or the port's, or overlaps another")
            if row[2 * q.at:2 * (q.at + nw)] != crow[q.start:q.start + n]:
                raise ValueError(f"{who}: words {q.at}.. are not {q.value} row {q.position}'s")
            covered[q.at:q.at + nw] = True
            out.append({"kind": "registered", "at": q.at, "words": nw, "start": q.start, "value": q.value,
                        "position": q.position, "row": [len(payload), len(crow)], "salt": [len(payload) + len(crow), SALT],
                        "string": cs.hex(), "path": reg.path(q.position).hex()})
            payload += crow + csalt
            continue
        if not 0 <= q.step <= c["step"] or q.slot != c["slot"] or (kind == "out" and (q.step != c["step"] or q.producer)):
            raise ValueError(f"{who}: a read is at the unit's own slot, at its step or an earlier one (an output's at its "
                             f"own step, no producer)")
        if not (0 <= q.unit < layout.unit_count and 0 <= q.rank < len(layout.units[q.unit])):
            raise ValueError(f"{who}: read {q} names no row of the layout")
        g = layout.row_base[q.unit] + q.rank
        crow = bytes(steps.rows(c["block"], q.step, q.slot)[g])
        n = len(crow) - q.start if q.nbytes is None else q.nbytes
        nw = n // 2
        if q.start % 2 or n % 2 or nw < 1 or not (0 <= q.start and q.start + n <= len(crow) and 0 <= q.at
                                                  and q.at + nw <= words) or covered[q.at:q.at + nw].any():
            raise ValueError(f"{who}: read {q} is outside its row or the port's, or overlaps another")
        if row[2 * q.at:2 * (q.at + nw)] != crow[q.start:q.start + n]:
            raise ValueError(f"{who}: words {q.at}.. are not step {q.step}'s row {q.rank} of unit {q.unit}")
        covered[q.at:q.at + nw] = True
        csalt = S.row_salts(steps.record(c["block"], q.step).salt_key, q.slot, g, 1)
        cs = string_of(S.ROW_SCHEMA, crow, csalt)
        out.append({"kind": "step", "at": q.at, "words": nw, "start": q.start,
                    "row": [len(payload), len(crow)], "salt": [len(payload) + len(crow), SALT], "string": cs.hex(),
                    "slot": opening(c["block"], q.step, q.slot, q.unit, q.rank, q.producer, cs,
                                    f"{who}: step {q.step}'s row {q.rank} of unit {q.unit}")})
        payload += crow + csalt
    if np.frombuffer(row, dtype="<u2")[~covered].any():
        raise ValueError(f"{who}: a word no read covers is not zero")
    return out


def commitment_doc(commitment: Commitment, steps: _Steps | None = None) -> dict[str, Any]:
    """A bundle's ``commitment`` section: every block's and step's root of the prover's records."""
    com = commitment
    return {"format": COMMITMENT, "reference": REFERENCE, "context": com.context.hex(), "layout": layout_json(com.layout),
            "layout_digest": S.layout_digest(com.layout).hex(), "slots": com.slots,
            "blocks": (steps or _Steps(com)).blocks()}


def open_reads(commitment: Commitment, rows: Sequence[tuple[Mapping[str, Any], bytes, Sequence[Read | Position]]], *,
               kind: str = "in", workers: int = 1) -> tuple[list[list[dict[str, Any]]], bytes]:
    """Port rows' reads opened outside a window, as `build` opens them (offsets into the one returned payload), each
    ``(coords, row, pieces)``: what a writer or a run checks before its window exists, with `check_reads`. Each (step,
    slot) read is rebuilt once, in ``workers`` processes."""
    payload, want = bytearray(), {}
    steps, opening = _Steps(commitment), _opener(want)
    reads = [_read_entries(f"row {i}", c, kind, row, pieces, steps, payload, opening) for i, (c, row, pieces) in enumerate(rows)]
    _fill(want, steps, workers)
    return reads, bytes(payload)


def check_reads(commitment: Mapping[str, Any], rows: Sequence[tuple[Mapping[str, Any], bytes, Sequence[Mapping[str, Any]]]],
                payload: bytes, *, kind: str = "in", registration: Mapping[str, Any] | None = None) -> dict[str, str]:
    """`check`'s verdict on port rows' reads (each ``(coords, row, reads)``, `open_reads`'s) against a ``commitment``
    section (`commitment_doc`), each step root held to the ``registration``'s when given: ``{key: why}``, empty when they
    hold."""
    bad: dict[str, str] = {}
    com = _check_commitment({"commitment": commitment}, bad, registration)
    for i, (coords, row, reads) in enumerate(rows):
        r = {"kind": kind, "words": len(row) // 2, "reads": list(reads)}
        if (why := _check_reads(r, {"coords": coords}, row, payload, com)) is not None:
            bad[f"row {i}"] = why
    return bad


def build(window_dir: str | Path, drawn: Sequence[int], *, commitment: Commitment, locate: Locate, binding: str,
          coords: Coords, reads: Reads | None = None,
          weights: Mapping[str, Any] | None = None, draw: Mapping[str, Any] | None = None, workers: int = 1
          ) -> tuple[dict[str, Any], bytes]:
    """The bundle of ``drawn`` (global unit indices) from the window in ``window_dir`` (what ``Window.write`` wrote, prover's
    files included): ``(document, payload)``.

    ``commitment``: the run's step commitments, the prover's records (each step's every slot root, and the rows of the slots
    the drawn units read). ``locate(coords, template, port)``: where a unit's port's row is in the step commitment (`Locate`):
    ``(layout unit, rank, producer)`` for one whole row at the unit's own step and slot (``producer`` the producing unit's
    coordinates for an input, None for the unit's own output), or the `Read`s (rows at the unit's step or an earlier one, its
    own slot) and `Position`s the row is made of, or None for a registered read. ``binding``: the card's dot binding
    (`BINDINGS`), which every template must bind. ``coords(label, offset)``: a unit's coordinates from its label and its
    offset in its labelled run. ``reads(coords, template, port)``: the ``(value, position)`` a unit's port reads of a
    registered value; ``weights``: those values, the prover's side (``program``, ``schema``, ``entry()``, ``path(position)``
    and the row's string, ``string(position)`` or ``bcs[position]``: `registered.Registered`). ``draw``: the draw's ``law``,
    ``source`` and ``receipt`` (default ``{"source": "fixture"}``; a live draw without its receipt is refused). ``workers``:
    processes rebuilding the opened (step, slot) subtrees, one each.

    Refused (ValueError) when a row is neither in the step commitment nor a registered read, when a window string is not
    the step commitment's or the registered value's (a row committed under other salts has no path to that root), when a
    row made of pieces is not their bytes (and zero elsewhere), when a kept slot's rows are not its committed slot root, or
    when a template binds another dot than ``binding``'s."""
    w = _Window(Path(window_dir))
    run, part = w.win["run"], w.win["partition"]
    reads = reads or (lambda c, t, p: None)
    weights = dict(weights or {})
    drawn = [int(u) for u in drawn]
    if len(set(drawn)) != len(drawn):
        raise ValueError("a unit is drawn once")
    tgt = target(binding)
    doc_draw = _draw(draw, drawn)
    com, steps, layout = commitment, _Steps(commitment), commitment.layout

    roots = {r["port"]: r for r in w.rec["roots"]}
    port_trees: dict[tuple[int, str], MerkleTree] = {}
    payload = bytearray()
    units, members, used_roots, used_values = [], {}, {}, set()
    want: Wants = {}
    opening = _opener(want)

    for u in drawn:
        k = w.member_of(u)
        m, i = w.members[k], u - w.members[k]["base"]
        c = coords(*w.label(u))
        tpl = _template(m["template"])
        if _dot_of(tpl) not in (None, tgt["dot"]):
            raise ValueError(f"unit {u}: its template binds {_dot_of(tpl)}, not {binding}'s {tgt['dot']}")
        rows = []
        for j, (p, schema, words, row, salt, bc) in enumerate(w.rows_of(k, i)):
            if string_of(schema, row, salt) != bc:
                raise ValueError(f"unit {u} {p}: its row and salt are not its committed string")
            name = f"{m['name']}/{p}"
            if (k, p) not in port_trees:
                ss = w.strings(k, p)
                dom = CommitmentDomain(bytes.fromhex(m["frame_v3"]["bindings"][p]), CS.OWNER, RangeIndexedDomain(0, len(ss)),
                                       hash="sha512")
                port_trees[k, p] = MerkleTree(dom, {r: _leaf_value(s) for r, s in enumerate(ss)}, lambda _r, schema=schema: schema)
                if port_trees[k, p].commitment.root.hex() != roots[name]["root"]:
                    raise ValueError(f"{name}: its strings are not the registered root")
            used_roots[name] = {"root": roots[name]["root"], "leaves": roots[name]["leaves"], "domain": roots[name]["domain"],
                                "schema": schema}
            entry: dict[str, Any] = {"port": p, "kind": "in" if j < m["input_ports"] else "out", "schema": schema, "words": words,
                                     "row": [len(payload), len(row)], "salt": [len(payload) + len(row), SALT], "string": bc.hex(),
                                     "window": {"port": name, "leaf": i, "path": _path(port_trees[k, p], i)},
                                     "slot": None, "registered": None}
            payload += row + salt
            at, hit = locate(c, m["template"], p), reads(c, m["template"], p)
            if (at is None) == (hit is None):
                raise ValueError(f"unit {u} {p}: in the step commitment or a registered read, exactly one")
            pieces = None if at is None else _pieces(at)
            if hit is not None:
                value, pos = hit
                reg = weights[value]
                if (reg.string(pos) if hasattr(reg, "string") else reg.bcs[pos]) != bc:
                    raise ValueError(f"unit {u} {p}: committed under its own salt, not {value} row {pos}'s registered one")
                entry["registered"] = {"value": value, "position": pos, "path": reg.path(pos).hex()}
                used_values.add(value)
            elif pieces is None:
                lu, rank, producer = at
                if (producer is None) != (entry["kind"] == "out"):
                    raise ValueError(f"unit {u} {p}: an output opens in its own unit, an input where its producer committed it")
                if len(row) != layout.units[lu][rank].nbytes:
                    raise ValueError(f"unit {u} {p}: not the step commitment's row {rank} of unit {lu} (other bytes or salts)")
                entry["slot"] = opening(c["block"], c["step"], c["slot"], lu, rank, producer, bc,
                                        f"unit {u} {p}: not the step commitment's row {rank} of unit {lu} (other bytes or salts)")
            else:
                entry["reads"] = _read_entries(f"unit {u} {p}", c, entry["kind"], row, pieces, steps, payload, opening,
                                               weights)
                used_values |= {q["value"] for q in entry["reads"] if q["kind"] == "registered"}
            rows.append(entry)
        units.append({"unit": u, "coords": c, "template": tpl, "member": {"k": k, "local": i}, "rows": rows})
        mm = members.setdefault(k, {"k": k, "name": m["name"], "template": m["template"], "base": m["base"], "n": m["n"],
                                    "local": [], "frame_v3": m["frame_v3"], "public_sha512": m["public_sha512"],
                                    "row_schemas": m.get("row_schemas")})
        mm["local"].append(i)
    for mm in members.values():
        mm["local"].sort()
        mm["draw"] = {"law": "subset", "population": mm["n"], "k": len(mm["local"]), "units": mm["local"]}

    _fill(want, steps, workers)

    progs = {weights[v].program for v in used_values}
    if len(progs) > 1:
        raise ValueError(f"the drawn units read registered values of {len(progs)} programs; a bundle names one")
    pay = bytes(payload)
    doc = {"format": FORMAT,
           "payload": {"bytes": len(pay), "sha512": hashlib.sha512(pay).hexdigest(), "layout": PAYLOAD_LAYOUT},
           "window": {"run": run, "registration": w.win["registration_digest"], "circuit": w.rec["window"].get("circuit"),
                      "partition": w.win["partition_digest"], "program": part["program"], "scheme": w.win["scheme"],
                      "population": w.win["population"], "leaf_layer": w.win["leaf_layer"], "roots": used_roots},
           "target": tgt,
           "draw": doc_draw,
           "weights": {"program": next(iter(progs)) if progs else None,
                       "values": [{**weights[v].entry(), "row_schema": weights[v].schema} for v in sorted(used_values)]},
           "commitment": commitment_doc(com, steps),
           "members": [members[k] for k in sorted(members)],
           "units": units}
    return doc, pay


def _elements(row: bytes, width: int, shape: tuple[int, ...]) -> np.ndarray:
    a = np.frombuffer(row, dtype=f"<u{width // 8}").astype(f"u{width // 8}")
    return a.reshape((1, *shape)) if shape else a.reshape(1)


def replay(unit: Mapping[str, Any], payload: bytes) -> bool:
    """Whether the unit's committed output rows are its Definition's outputs on its committed input rows."""
    fn = IF.resolve(unit["template"]["id"])
    params, ret = fn.signature()
    lay = IF.layout(fn)
    rows = {r["port"]: payload[r["row"][0]:r["row"][0] + r["row"][1]] for r in unit["rows"]}
    if any(ty.width_set() == {1} for _, ty in params):
        return _replay_bits(fn, params, ret, lay, rows)
    ins = []
    for (name, ty), width in zip(params, lay.widths):
        n = len(ty.leaf_widths())
        ins.append(_elements(rows[name], width, () if n == 1 and not getattr(ty, "shape", None) else (n,)))
    want = np.asarray(evaluate_batch(fn, *ins)).reshape(1, -1).astype(np.uint64)
    got = []
    for (o, words), f, width in zip(lay.outs, IF._fields(ret), lay.widths[len(params):]):
        leaves = len(f.leaf_widths())
        got.append(np.frombuffer(rows[o][:leaves * width // 8], dtype=f"<u{width // 8}").astype(np.uint64))
    got_a = np.concatenate(got).reshape(1, -1)
    return got_a.shape == want.shape and bool((got_a == want).all())


def _bits_of(row: bytes, n: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(row, np.uint8), bitorder="little")[:n].reshape(1, n)


def _replay_bits(fn: Any, params: Sequence[Any], ret: Any, lay: IF.RowLayout, rows: Mapping[str, bytes]) -> bool:
    """`replay` of a Boolean-profile template: each row is its port's bits in order, packed little-endian into its u16
    words (`IF.layout`), and an output field of fewer than 16 bits is its bits low in one zero-padded word, which must be
    zero."""
    ins = [_bits_of(rows[name], sum(ty.leaf_widths())) for name, ty in params]
    out = evaluate_batch(fn, *ins)
    want = np.concatenate([np.asarray(o).reshape(1, -1) for o in (out if isinstance(out, (tuple, list)) else (out,))], axis=1)
    got = []
    for (o, words), f in zip(lay.outs, IF._fields(ret)):
        n = sum(f.leaf_widths())
        row = np.unpackbits(np.frombuffer(rows[o], np.uint8), bitorder="little")
        if row.size != 16 * words or row[n:].any():
            return False
        got.append(row[:n].reshape(1, n))
    got_a = np.concatenate(got, axis=1)
    return got_a.shape == want.shape and bool((got_a == want.astype(np.uint8)).all())


def _check_commitment(doc: Mapping[str, Any], bad: dict[str, str],
                      registration: Mapping[str, Any] | None = None) -> tuple[Any, ...] | None:
    """The commitment section's layout, context and block trees recomputed: ``(context, layout, slots, step roots by (block,
    step))``, or None (with ``bad`` filled) when its layout or context can't be read. With ``registration``, every step's
    root is held to the one its ``window.commitment`` names (`steps_provenance`), and a step it doesn't name is refused."""
    com = doc["commitment"]
    if com.get("format") != COMMITMENT:
        bad["commitment"] = f"not {COMMITMENT}"
        return None
    try:
        layout, context = layout_of_json(com["layout"]), bytes.fromhex(com["context"])
        S.layout_digest(layout)
        S.slot_domain(context, layout)
    except (ValueError, KeyError, TypeError) as e:
        bad["commitment"] = f"its layout or context is not one: {e}"
        return None
    if S.layout_digest(layout).hex() != com["layout_digest"]:
        bad["commitment"] = "its layout digest is not its layout's"
    registered: dict[tuple[int, int], dict[str, Any]] | None = None
    if registration is not None:
        rc = (registration.get("window") or {}).get("commitment") or {}
        registered = {}
        if {k: rc.get(k) for k in ("format", "context", "layout_digest", "slots")} != \
                {"format": COMMITMENT, **{k: com[k] for k in ("context", "layout_digest", "slots")}}:
            bad["commitment:registration"] = "the registration names no step commitment, or another context, layout or slots"
        else:
            registered = {(b["block"], s["step"]): s for b in rc.get("blocks") or () for s in b["steps"]}
    step_roots = {}
    for b in com["blocks"]:
        steps = [s["step"] for s in b["steps"]]
        dom = block_domain(context, layout, b["block"], steps)
        roots = [bytes.fromhex(s["root"]) for s in b["steps"]]
        if dom.domain_id.hex() != b["domain"] or RootTree(dom, roots).root.hex() != b["root"]:
            bad[f"block:{b['block']}"] = "its root is not the block tree's over its steps' roots"
        for s, r in zip(b["steps"], roots, strict=True):
            key = f"step:{b['block']}/{s['step']}"
            if S.step_domain(context, layout, s["step"], com["slots"]).domain_id.hex() != s["domain"]:
                bad[key] = "its domain is not the commit call's"
            elif registered is not None and {k: (registered.get((b["block"], s["step"])) or {}).get(k) for k in ("root", "domain")} \
                    != {"root": s["root"], "domain": s["domain"]}:
                bad[key] = "not the step root the registration names"
                continue
            step_roots[b["block"], s["step"]] = r
    return context, layout, com["slots"], step_roots


def _check_opening(s: Mapping[str, Any], nbytes: int, bc: bytes, com: tuple[Any, ...]) -> str | None:
    """Why a committed row's opening (``slot``: unit, rank and paths) does not reach its step's root, or None."""
    context, layout, slots, step_roots = com
    lu, rank = s["unit"], s["rank"]
    if not (0 <= lu < layout.unit_count and 0 <= rank < len(layout.units[lu])) or nbytes != layout.units[lu][rank].nbytes:
        return "not a row of the layout"
    step_root = step_roots.get((s["block"], s["step"]))
    if step_root is None or not 0 <= s["slot"] < slots:
        return "its step is not in the commitment, or not the registration's"
    try:
        sp, tp = RootOpening.of_json(s["slot_path"]), RootOpening.of_json(s["step_path"])
    except (ValueError, KeyError, TypeError) as e:
        return f"its openings are not ones: {e}"
    unit_root = _fold(S.unit_domain(context, layout, lu), rank, S.ROW_SCHEMA, _leaf_value(bc), s["path"])
    sd, td = S.slot_domain(context, layout), S.step_domain(context, layout, s["step"], slots)
    if (unit_root is None or sp.indices != (lu,) or sp.roots[0] != unit_root or tp.indices != (s["slot"],)
            or not verify_roots(tp.roots[0], sd.domain_id, sd.count, sp) or not verify_roots(step_root, td.domain_id, td.count, tp)):
        return "its paths do not reach its unit, slot and step roots"
    return None


def _producer_at(s: Mapping[str, Any]) -> bool:
    return ({x: s["producer"].get(x) for x in ("block", "step", "slot", "unit")} == {x: s[x] for x in ("block", "step", "slot", "unit")}
            and {"layer", "op"} <= set(s["producer"]))


def _check_slot(r: Mapping[str, Any], unit: Mapping[str, Any], row: bytes, bc: bytes, com: tuple[Any, ...]) -> str | None:
    """Why a row's step-commitment opening fails, or None."""
    s, c = r["slot"], unit["coords"]
    if (s["block"], s["step"], s["slot"]) != (c["block"], c["step"], c["slot"]):
        return "it opens outside its unit's slot and step"
    if (s["producer"] is None) != (r["kind"] == "out"):
        return "an output opens in its own unit, an input names its producer"
    if s["producer"] is not None and not _producer_at(s):
        return "its producer's coordinates are not where it opens"
    return _check_opening(s, len(row), bc, com)


def _check_reads(r: Mapping[str, Any], unit: Mapping[str, Any], row: bytes, payload: bytes, com: tuple[Any, ...] | None,
                 registered: Callable[[Mapping[str, Any], bytes, bytes], bool] | None = None) -> str | None:
    """Why a row made of reads (`Read`, `Position`, `RegisteredRead`) fails, or None: each step read is its committed row's
    bytes and salt under its string, opens at the unit's own slot and block at its step or an earlier one (an output's at
    its own step), where its producer (when named) committed it, against that step's root; each registered piece (an
    input's) is its registered row's bytes and salt under its string, whose path reaches its value's root
    (``registered(read, row, salt)``); its bytes are the row's words from ``at``; a position (an input's) is the unit's
    step plus its offset; the reads don't overlap; every other word is zero (every word, for a row of no reads)."""
    c, words, out = unit["coords"], r["words"], r["kind"] == "out"
    if r["kind"] not in ("in", "out") or not isinstance(r["reads"], list):
        return "a row made of reads"
    covered = np.zeros(words, dtype=bool)
    for j, q in enumerate(r["reads"]):
        at = q.get("at")
        if not isinstance(at, int) or isinstance(at, bool) or not 0 <= at < words:
            return f"read {j}: its place is outside the row"
        if q.get("kind") == Position.KIND:
            if out or q.get("offset") not in (0, 1) or covered[at] \
                    or int.from_bytes(row[2 * at:2 * at + 2], "little") != c["step"] + q["offset"]:
                return f"read {j}: the word is not its step plus its offset, or another read's"
            covered[at] = True
            continue
        if q.get("kind") == "registered":
            nw, start = q.get("words"), q.get("start")
            crow = payload[q["row"][0]:q["row"][0] + q["row"][1]]
            csalt = payload[q["salt"][0]:q["salt"][0] + q["salt"][1]]
            if out or registered is None or len(crow) != q["row"][1] or len(csalt) != SALT or not registered(q, crow, csalt):
                return f"read {j}: its registered row and salt are not its string, its path doesn't reach its value's root, " \
                       f"or it is in an output"
            if not (isinstance(nw, int) and isinstance(start, int) and start % 2 == 0 and 0 <= start and nw >= 1
                    and start + 2 * nw <= len(crow) and at + nw <= words) or covered[at:at + nw].any():
                return f"read {j}: its bytes are outside its row or the port's, or another read's"
            if row[2 * at:2 * (at + nw)] != crow[start:start + 2 * nw]:
                return f"read {j}: the row's words are not its bytes"
            covered[at:at + nw] = True
            continue
        if q.get("kind") != "step":
            return f"read {j}: a step read, a registered piece or a position"
        s, nw, start = q["slot"], q["words"], q["start"]
        crow = payload[q["row"][0]:q["row"][0] + q["row"][1]]
        csalt = payload[q["salt"][0]:q["salt"][0] + q["salt"][1]]
        bc = bytes.fromhex(q["string"])
        if len(crow) != q["row"][1] or len(csalt) != SALT or string_of(S.ROW_SCHEMA, crow, csalt) != bc:
            return f"read {j}: its row and salt are not its string"
        if (s["block"], s["slot"]) != (c["block"], c["slot"]) or not 0 <= s["step"] <= c["step"] \
                or (out and (s["step"] != c["step"] or s["producer"] is not None)):
            return f"read {j}: it opens outside its unit's slot, or after its step (an output's at its own step)"
        if s["producer"] is not None and not _producer_at(s):
            return f"read {j}: its producer's coordinates are not where it opens"
        if not (start % 2 == 0 and 0 <= start and nw >= 1 and start + 2 * nw <= len(crow) and at + nw <= words) \
                or covered[at:at + nw].any():
            return f"read {j}: its bytes are outside its row or the port's, or another read's"
        if row[2 * at:2 * (at + nw)] != crow[start:start + 2 * nw]:
            return f"read {j}: the row's words are not its bytes"
        covered[at:at + nw] = True
        if com is None:
            return "the commitment it opens against can't be read"
        if (why := _check_opening(s, len(crow), bc, com)) is not None:
            return f"read {j}: {why}"
    if np.frombuffer(row, dtype="<u2")[~covered].any():
        return "a word no read covers is not zero"
    return None


def check(doc: Mapping[str, Any], payload: bytes, *, registration: Mapping[str, Any] | None = None,
          replay_units: bool = True) -> dict[str, str]:
    """The consumer's check of a bundle (module docstring): ``{key: why}``, empty when it holds. ``registration``: the
    window's ``registration.json``, whose roots and law the bundle's must be, and whose ``window.commitment``
    (`steps_provenance`) names every step root the bundle opens against: a step it doesn't name fails closed."""
    bad: dict[str, str] = {}
    if doc.get("format") != FORMAT:
        return {"format": f"not {FORMAT}"}
    if len(payload) != doc["payload"]["bytes"] or hashlib.sha512(payload).hexdigest() != doc["payload"]["sha512"]:
        return {"payload": "its bytes are not the ones the bundle names"}
    d = doc["draw"]
    if d.get("source") not in ("live", "fixture") or (d["source"] == "live" and not d.get("receipt")):
        bad["draw"] = "a live draw carries the challenger's live-coin receipt, and a draw is live or fixture"
    try:
        tgt = target(doc["target"]["binding"])
    except ValueError as e:
        tgt = None
        bad["target"] = str(e)
    if tgt is not None and tgt != doc["target"]:
        bad["target"] = f"not {doc['target']['binding']}'s card and dot as circuits resolves it ({tgt})"
    win = doc["window"]
    for port, r in win["roots"].items():
        if window_domain(win["program"], win["partition"], port, r["schema"], r["leaves"], win["run"]) != r["domain"]:
            bad[f"root:{port}"] = "its domain is not the window-domain rule's over the registration's fields"
    if registration is not None:
        law = registration["law"]
        if d.get("law") != law or (law.get("law") == "subset" and len(d["units"]) != law.get("k")):
            bad["draw:law"] = f"the draw is not one under the registered law {law}"
        if "record" in d and not (D.check_draw(d["record"], registration, law).ok
                                  and d["record"].get("units") == sorted(d["units"])):
            bad["draw:record"] = "the draw's record is not a draw of its units under the registered law and population"
        reg_roots = {r["port"]: r for r in registration["roots"]}
        for port, r in win["roots"].items():
            if (reg_roots.get(port) or {}).get("root") != r["root"]:
                bad[f"root:{port}"] = "not the registration's root"
    values = {v["value"]: v for v in doc["weights"]["values"]}
    for name, v in values.items():
        dom = REG.domain(doc["weights"]["program"], name, v["words"], v["leaves"], v.get("word_bits", REG.WORD_BITS))
        if dom.domain_id.hex() != v["domain"]:
            bad[f"value:{name}"] = "its domain is not the registered-value rule's over its entry"
    com = _check_commitment(doc, bad, registration)
    members = {m["k"]: m for m in doc["members"]}

    def registered_piece(q: Mapping[str, Any], crow: bytes, csalt: bytes) -> bool:
        v = values.get(q.get("value"))
        if v is None or not isinstance(q.get("position"), int):
            return False
        schema = v.get("row_schema", REG.SCHEMA)
        bc = bytes.fromhex(q["string"])
        d = REG.domain(doc["weights"]["program"], v["value"], v["words"], v["leaves"], v.get("word_bits", REG.WORD_BITS))
        return string_of(schema, crow, csalt) == bc and REG.verify_read(d, bytes.fromhex(v["root"]), q["position"], bc,
                                                                         bytes.fromhex(q["path"]), schema)
    for unit in doc["units"]:
        u = unit["unit"]
        mm = members.get(unit["member"]["k"])
        if mm is None or mm["base"] + unit["member"]["local"] != u or unit["member"]["local"] not in mm["local"]:
            bad[f"unit:{u}"] = "its member does not hold it at its local index"
            continue
        if tgt is not None and _dot_of(unit["template"]) not in (None, tgt["dot"]):
            bad[f"unit:{u}"] = f"its template binds another dot than the target's {tgt['dot']}"
            continue
        outs = {r["slot"]["unit"] for r in unit["rows"] if r["kind"] == "out" and r["slot"] is not None}
        if len(outs) > 1:
            bad[f"unit:{u}"] = "its outputs open in more than one commitment unit"
        for r in unit["rows"]:
            key = f"unit:{u}/{r['port']}"
            row = payload[r["row"][0]:r["row"][0] + r["row"][1]]
            salt = payload[r["salt"][0]:r["salt"][0] + r["salt"][1]]
            bc = bytes.fromhex(r["string"])
            if len(row) != 2 * r["words"] or string_of(r["schema"], row, salt) != bc:
                bad[key] = "its row and salt are not its string"
                continue
            root = win["roots"].get(r["window"]["port"])
            if root is None:
                bad[key] = "its window port names no root"
                continue
            dom = CommitmentDomain(_binding(win, r["window"]["port"], r["schema"], root["leaves"]), CS.OWNER,
                                   RangeIndexedDomain(0, root["leaves"]), hash="sha512")
            if _fold(dom, r["window"]["leaf"], r["schema"], _leaf_value(bc), r["window"]["path"]) != bytes.fromhex(root["root"]):
                bad[key] = "its window path does not reach its port's registered root"
            if sum(r.get(x) is not None for x in ("slot", "registered", "reads")) != 1:
                bad[key] = "a row opens against the step commitment, a registered value or its reads, exactly one"
            elif r.get("reads") is not None:
                if (why := _check_reads(r, unit, row, payload, com, registered_piece)) is not None:
                    bad[key] = why
            elif r["registered"] is not None:
                v = values.get(r["registered"]["value"])
                d = v and REG.domain(doc["weights"]["program"], v["value"], v["words"], v["leaves"], v.get("word_bits", REG.WORD_BITS))
                if v is None or not REG.verify_read(d, bytes.fromhex(v["root"]), r["registered"]["position"], bc,
                                                    bytes.fromhex(r["registered"]["path"]), v.get("row_schema", REG.SCHEMA)):
                    bad[key] = "its registered path does not reach the value's root"
            elif com is None:
                bad[key] = "the commitment it opens against can't be read"
            elif r["schema"] != S.ROW_SCHEMA:
                bad[key] = f"the step commitment's rows are {S.ROW_SCHEMA}"
            elif (why := _check_slot(r, unit, row, bc, com)) is not None:
                bad[key] = why
        if replay_units and f"unit:{u}" not in bad and not any(k.startswith(f"unit:{u}/") for k in bad):
            try:
                if not replay(unit, payload):
                    bad[f"unit:{u}"] = "its committed outputs are not its Definition's on its committed inputs"
            except ValueError as e:
                bad[f"unit:{u}"] = f"not replayed: {e}"
    return bad


#: a step's commit-slots roots registered before a coin draws its proof blocks (M3): the draw's population is
#: every (slot, layout unit) of the step, and each drawn block is proved from a one-member window of its own. ``steps``
#: (optional, `STEPS_KEY`) names the root of every step of the block a drawn block can read (attention's K and V rows at
#: steps 0 .. the drawn step), ``[{step, root}]``; a registration without it holds every read to the drawn step
STEP_FORMAT = "verity/commit-slots-step-registration/v1"
STEP_KEYS = {"format", "run", "program", "block", "step", "step_root", "slot_roots", "slots", "layout_digest", "context",
             "units_per_slot", "population", "law", "weights"}
STEPS_KEY = "steps"


def step_digest(reg: Mapping[str, Any]) -> str:
    """The step registration's digest, which the challenger's receipt names and the gate binds before the coin."""
    return identity_digest_sha512(STEP_FORMAT, dict(reg)).hex()


def step_registration(com: Commitment, weights: Mapping[str, REG.Registered], law: Mapping[str, Any], *, run: str,
                      program: str, block: int, step: int, read_steps: bool = False) -> dict[str, Any]:
    """The registration of step ``(block, step)`` of ``com`` (the prover's records) under ``law``, naming the registered
    values ``weights`` its blocks read; ``read_steps``: and the root of every step of ``block`` up to ``step`` (`STEPS_KEY`),
    which ``com`` must hold."""
    steps = _Steps(com)
    reg = {"format": STEP_FORMAT, "run": run, "program": program, "block": block, "step": step,
           "step_root": steps.step(block, step).root.hex(), "slot_roots": [r.hex() for r in steps.roots(block, step)],
           "slots": com.slots, "layout_digest": S.layout_digest(com.layout).hex(), "context": com.context.hex(),
           "units_per_slot": com.layout.unit_count, "population": com.slots * com.layout.unit_count, "law": dict(law),
           "weights": [{"value": v, "root": w.root.hex()} for v, w in sorted(weights.items())]}
    if read_steps:
        reg[STEPS_KEY] = [{"step": s, "root": steps.step(block, s).root.hex()} for s in range(step + 1)]
    return reg


def block_of(reg: Mapping[str, Any], index: int) -> tuple[int, int]:
    """A drawn proof block's ``(slot, layout unit)``: index ``slot * units_per_slot + unit``."""
    return divmod(int(index), int(reg["units_per_slot"]))


def check_step(doc: Mapping[str, Any], payload: bytes, reg: Mapping[str, Any], index: int, *,
               registration: Mapping[str, Any] | None = None) -> dict[str, str]:
    """`check` of a drawn block's bundle, then the bundle held to the step registration ``reg`` at drawn ``index``:
    ``{key: why}``, empty when it holds. Its commitment is the registered step (root, slot roots, layout, context), it is
    one proof block whose outputs open in the layout unit ``index`` names, every row opens against the registered step
    or a registered value whose root ``reg`` names, and each step row's slot root is the registered one. A row made of
    reads (`Read`) holds each read to the registered step's slot root, or, at an earlier step, to that step's root as
    ``reg``'s `STEPS_KEY` names it: a read at a step ``reg`` doesn't name is refused. The paths themselves are `check`'s,
    which runs first, so a row whose paths don't reach the registered step root is refused."""
    bad = check(doc, payload, registration=registration)
    if set(reg) - {STEPS_KEY} != STEP_KEYS or reg["format"] != STEP_FORMAT:
        return {**bad, "step": f"not a {STEP_FORMAT} with exactly {sorted(STEP_KEYS)} and optionally {STEPS_KEY}"}
    if (reg["population"] != reg["slots"] * reg["units_per_slot"] or len(reg["slot_roots"]) != reg["slots"]
            or not 0 <= int(index) < reg["population"]):
        return {**bad, "step": "its population is not slots x units_per_slot over its slot roots, or the index is outside it"}
    named = {s["step"]: s["root"] for s in reg.get(STEPS_KEY) or ()}
    if STEPS_KEY in reg and (len(named) != len(reg[STEPS_KEY]) or named.get(reg["step"]) != reg["step_root"]
                             or any(not 0 <= s <= reg["step"] for s in named)):
        return {**bad, "step": f"its {STEPS_KEY} are not distinct steps up to its own, with its own step's root"}
    com = doc["commitment"]
    try:
        unit_count = layout_of_json(com["layout"]).unit_count
    except (ValueError, KeyError, TypeError) as e:
        return {**bad, "step:layout": f"not read: {e}"}
    if (com["layout_digest"], com["context"], com["slots"], unit_count) != (reg["layout_digest"], reg["context"],
                                                                             reg["slots"], reg["units_per_slot"]):
        bad["step:commitment"] = "its layout, context, slots or units a slot are not the registered step's"
    roots = {(b["block"], s["step"]): s["root"] for b in com["blocks"] for s in b["steps"]}
    if roots.get((reg["block"], reg["step"])) != reg["step_root"]:
        bad["step:root"] = "its commitment does not hold the registered step root"
    if len(doc["units"]) != 1:
        return {**bad, "step:units": f"{len(doc['units'])} units, not the one drawn proof block"}
    unit, (slot, lu) = doc["units"][0], block_of(reg, index)
    c = unit["coords"]
    outs = {s["unit"] for r in unit["rows"] if r["kind"] == "out"
            for s in ([r["slot"]] if r["slot"] is not None else [q["slot"] for q in r.get("reads") or () if q["kind"] == "step"])}
    if (c["block"], c["step"], c["slot"]) != (reg["block"], reg["step"], slot) or outs != {lu}:
        bad["step:index"] = f"its block is not drawn index {index} (slot {slot}, layout unit {lu})"
    values = {v["value"]: v["root"] for v in doc["weights"]["values"]}
    weights = {w["value"]: w["root"] for w in reg["weights"]}

    def registered_step(s: Mapping[str, Any]) -> bool:
        if s["block"] != reg["block"] or not 0 <= s["slot"] < reg["slots"]:
            return False
        if s["step"] == reg["step"]:
            return s["step_path"]["roots"] == [reg["slot_roots"][s["slot"]]]
        return s["step"] in named and roots.get((reg["block"], s["step"])) == named[s["step"]]

    for r in unit["rows"]:
        key = f"step:{r['port']}"
        if r["slot"] is not None:
            s = r["slot"]
            if (s["block"], s["step"]) != (reg["block"], reg["step"]) or not registered_step(s):
                bad[key] = "it does not open under the registered step's slot root"
        elif r.get("reads") is not None:
            if not all(registered_step(q["slot"]) for q in r["reads"] if q["kind"] == "step"):
                bad[key] = "a read does not open under the registered step's slot root or a step root the registration names"
            elif not all(q["value"] in weights and values.get(q["value"]) == weights[q["value"]]
                         for q in r["reads"] if q["kind"] == "registered"):
                bad[key] = "a registered piece's value's root is not one the step registration names"
        elif r["registered"] is not None:
            name = r["registered"]["value"]
            if name not in weights or values.get(name) != weights[name]:
                bad[key] = "its registered value's root is not one the step registration names"
        else:
            bad[key] = "it opens against neither the registered step nor a registered value"
    return bad


def _binding(win: Mapping[str, Any], port: str, schema: str, leaves: int) -> bytes:
    return identity_digest_sha512(CS.DOMAIN_TAG, {"program": win["program"], "partition": win["partition"], "port": port,
                                                  "schema": schema, "leaves": int(leaves), "run": win["run"]})


def write(doc: Mapping[str, Any], payload: bytes, out_dir: str | Path, name: str = "bundle") -> tuple[Path, Path]:
    """``<name>.json`` and ``<name>.bin`` (mode 0600: values and salts) in ``out_dir``."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    pj, pb = d / f"{name}.json", d / f"{name}.bin"
    pj.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    fd = os.open(pb, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(payload)
    return pj, pb


def read(path: str | Path) -> tuple[dict[str, Any], bytes]:
    """A bundle written by `write`: its document and its payload (``<name>.bin`` beside ``<name>.json``)."""
    p = Path(path)
    return json.loads(p.read_text()), p.with_suffix(".bin").read_bytes()


def digests(doc: Mapping[str, Any], payload: bytes) -> dict[str, str]:
    """The SHA-512s of a bundle's document as `write` writes it and of its payload."""
    return {"json": hashlib.sha512((json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()).hexdigest(),
            "bin": hashlib.sha512(payload).hexdigest()}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="The drawn-unit replay bundle: check one.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("bundle")
    c.add_argument("--registration")
    a = ap.parse_args(argv)
    doc, pay = read(a.bundle)
    reg = json.loads(Path(a.registration).read_text()) if a.registration else None
    bad = check(doc, pay, registration=reg)
    print(json.dumps({"units": len(doc["units"]), "ok": not bad, "bad": bad}, sort_keys=True))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
