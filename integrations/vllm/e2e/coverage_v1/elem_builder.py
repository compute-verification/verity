"""Fallback obligation builder for the elementwise kinds the approved guest already evaluates (A5/6, R18 §4).

Kinds (one gate of ``veritor.elem-bf16@1``, ``zk/sp1/common/src/gateset.rs:412``, ``elem.rs``; the same digest as
``zk/bench/elem_gates.ELEM_BF16_DIGEST``), whose *whole* boundary is committed in a record's RU tree:

* ``SiluMulBf16_v1``  -- ``out[i] = silu_mul_bf16(g[i], u[i])``: ``g``/``u`` are words ``i`` and ``I+i`` of the gate_up
  GEMM output row (``.../op=layers.<L>.mlp.gate_up_proj.gemm/port=0``, ``[2I] Value<16>``), ``out`` word ``i`` of the
  act_mul row (``.../op=layers.<L>.mlp.act_fn.act_mul/port=0``, ``[I] Value<16>``) -- the down-proj GEMM's input.
* ``Bf16AddF2fp_v1`` -- ``out[i] = bf16_add_f2fp(a[i], b[i])``: the fused add+norm's residual add over two committed
  bf16 rows into the committed residual-out row.

The builder emits a **raw** ``veritor.protocol.proofs.Statement`` in format 3 (packed ``verity-vllm/leaf/v1`` rows, the
framing the record commits) -- not a ``typed-obligation/v0`` object, because ``veritor.interface.typed_obligation``'s
``SUPPORTED_FAMILIES`` / ``family_gate_set`` (A2's file) know only the tc families; extending them is an interface
request (checkpoint 1905Z, Q-A56-4).  Statement -> witness -> transparent verdict is exercised here; an SP1 proof of the
statement under the approved guest ``9fd00e60...`` is the same relation (the guest resolves ``veritor.elem-bf16@1`` by
id and digest, ``kernel/mod.rs:168-176``) but was not run in this lane (no GPU/pod) -- see the report.

Verifier side: :func:`expectations_of` gives A2-shaped ``ExpectedObligation`` rows (the VU's committed positions, from
the record description the caller located -- never from provenance) and :func:`obligations_of_raw_statement` reads a
raw statement back into :class:`~e2e.coverage_v1.coverage.ObligationView` rows using the committed id lists (rank ->
canonical id) so :func:`e2e.coverage_v1.check_coverage` accounts for them like typed-object obligations.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from veritor.interface.lowering import _lower_subcircuit
from veritor.interface.programs import Definition, make_subcircuit
from veritor.interface.typed_obligation import IN
from veritor.protocol.proofs.packed import VLLM_LEAF_FRAMING, encode_packed_witness, packed_schema
from veritor.protocol.proofs.statement import CommitmentRef, KindProgram, Obligation, PositionRef, Statement, make_statement

from .coverage import CoverageError, ObligationView, Position

__all__ = [
    "ELEM_FAMILY",
    "ELEM_WIDTH",
    "KINDS",
    "KindSpec",
    "Word",
    "ElementwiseUnit",
    "CommittedTree",
    "RawExport",
    "elem_gate_set",
    "kind_program",
    "build_statement",
    "expectations_of",
    "obligations_of_raw_statement",
    "verify_locally",
]

ELEM_FAMILY = "veritor.elem-bf16@1"
ELEM_WIDTH = 32  # the statement width the family resolves at (gateset.rs `resolve(id, width)`; bench uses 32)


@dataclass(frozen=True)
class KindSpec:
    kind: str
    family: str
    op: str
    in_widths: tuple[int, ...]
    out_width: int


KINDS: dict[str, KindSpec] = {
    "SiluMulBf16_v1": KindSpec("SiluMulBf16_v1", ELEM_FAMILY, "silu_mul_bf16", (16, 16), 16),
    "Bf16AddF2fp_v1": KindSpec("Bf16AddF2fp_v1", ELEM_FAMILY, "bf16_add_f2fp", (16, 16), 16),
}


@dataclass(frozen=True)
class Word:
    """One word of a committed packed row: the tree, the leaf's canonical id, its word count and the word's index."""

    tree: str
    canonical_id: str
    count: int
    index: int

    @property
    def position(self) -> Position:
        return (self.tree, self.canonical_id, self.index)


@dataclass(frozen=True)
class ElementwiseUnit:
    vu_id: str
    kind: str
    inputs: tuple[Word, ...]
    output: Word


@dataclass(frozen=True)
class CommittedTree:
    """A committed tree as the record's ``commitments.json`` states it (root, leaf count, ``ids_digest`` = domain)."""

    tree: str
    root: bytes
    leaf_count: int
    ids_digest: bytes
    owner: int  # -2 weights, -1 boundary, RU ordinal otherwise (typed_obligation_export._owner_of)


@dataclass(frozen=True)
class RawExport:
    statement: Statement
    programs: dict[str, tuple[str, str]]  # KindProgram.kind hex -> (kind name, subcircuit digest)
    units: tuple[ElementwiseUnit, ...]
    views: tuple[ObligationView, ...]
    positions_by_unit: dict[str, tuple[Position, ...]] = field(default_factory=dict)


def elem_gate_set():
    """The Python mirror of ``veritor.elem-bf16@1`` (``zk/bench/elem_gates.py``; digest pinned there and in
    ``gateset.rs``).  The bench module is imported by path: it is the only Python construction of the family."""

    repo = Path(__file__).resolve().parents[3]
    bench = repo / "zk" / "bench"
    if str(bench) not in sys.path:
        sys.path.insert(0, str(bench))
    from elem_gates import ELEM_BF16_DIGEST, make_elem_bf16_gate_set  # type: ignore[import-not-found]

    gs = make_elem_bf16_gate_set()
    if str(gs.digest) != ELEM_BF16_DIGEST:
        raise CoverageError(f"elem-bf16 gate set digest {gs.digest} is not the pinned {ELEM_BF16_DIGEST}")
    return gs


def kind_program(spec: KindSpec, gate_set) -> tuple[KindProgram, str]:
    """The one-gate program of ``spec`` lowered under ``gate_set``; returns ``(program, subcircuit digest)``."""

    gates = [(spec.op, [(IN, k) for k in range(len(spec.in_widths))])]
    sub = make_subcircuit(list(spec.in_widths), gates, [0], Definition(spec.kind, ()))
    program, _ = _lower_subcircuit(sub, gate_set)
    return program, sub.digest


def _check_unit(u: ElementwiseUnit) -> KindSpec:
    spec = KINDS.get(u.kind)
    if spec is None:
        raise CoverageError(f"{u.vu_id}: no elementwise builder for kind {u.kind!r}")
    if len(u.inputs) != len(spec.in_widths):
        raise CoverageError(f"{u.vu_id}: {u.kind} takes {len(spec.in_widths)} inputs, {len(u.inputs)} given")
    for w in (*u.inputs, u.output):
        if not 0 <= w.index < w.count:
            raise CoverageError(f"{u.vu_id}: word {w.index} outside leaf {w.canonical_id!r} of {w.count} words")
    return spec


def build_statement(
    units: Sequence[ElementwiseUnit],
    trees: Mapping[str, CommittedTree],
    rank_of: Callable[[str, str], int | None],
    *,
    session: bytes,
    compiled: bytes,
    replay_unit_of: Callable[[str], int] | None = None,
    gate_set=None,
) -> RawExport:
    """One format-3 statement proving every unit's equation over the record's committed rows.

    ``trees`` are the committed trees the units read (root/count/ids_digest as committed *before* the challenge);
    ``rank_of(tree, canonical_id)`` the committed id-list oracle (A2 ``rank_oracle``).  Positions are the unit's
    inputs in body order then its output; each is ``PositionRef(commitment, rank, rank, packed/v1 schema)``, the
    spelling ``veritor.interface.lowering`` gives a word of a uniformly packed ``verity-vllm/leaf/v1`` leaf.  Refuses a
    word whose tree or leaf is not committed.  Obligation ``unit`` ordinals follow the given order."""

    if not units:
        raise CoverageError("no units: an empty statement is not a batch")
    gs = gate_set if gate_set is not None else elem_gate_set()
    programs: dict[str, KindProgram] = {}
    program_meta: dict[str, tuple[str, str]] = {}
    obligations: list[Obligation] = []
    views: list[ObligationView] = []
    positions_by_unit: dict[str, tuple[Position, ...]] = {}
    seen: set[str] = set()
    for ordinal, u in enumerate(units):
        if u.vu_id in seen:
            raise CoverageError(f"VU {u.vu_id} given twice")
        seen.add(u.vu_id)
        spec = _check_unit(u)
        if spec.kind not in programs:
            program, sub_digest = kind_program(spec, gs)
            programs[spec.kind] = program
            program_meta[program.kind.hex()] = (spec.kind, sub_digest)
        program = programs[spec.kind]
        words = (*u.inputs, u.output)
        tree_ids = sorted({w.tree for w in words})
        refs: list[CommitmentRef] = []
        for t in tree_ids:
            ct = trees.get(t)
            if ct is None:
                raise CoverageError(f"{u.vu_id}: tree {t!r} is not a committed tree of the record")
            refs.append(CommitmentRef(ct.owner, ct.ids_digest, ct.root, ct.leaf_count))
        index_of_tree = {t: i for i, t in enumerate(tree_ids)}
        prefs: list[PositionRef] = []
        for w, width in zip(words, (*spec.in_widths, spec.out_width), strict=True):
            rank = rank_of(w.tree, w.canonical_id)
            if rank is None:
                raise CoverageError(f"{u.vu_id}: leaf {w.canonical_id!r} is not committed in {w.tree!r}")
            prefs.append(PositionRef(index_of_tree[w.tree], rank, rank, packed_schema(VLLM_LEAF_FRAMING, width, True, w.count, w.index)))
        n_in = len(u.inputs)
        ru = replay_unit_of(u.vu_id) if replay_unit_of is not None else max(0, max(ct.owner for ct in (trees[t] for t in tree_ids)))
        obligations.append(Obligation(session, compiled, ordinal, ru, program.kind, tuple(refs), tuple(prefs), tuple(range(n_in)), (n_in,)))
        _, sub_digest = program_meta[program.kind.hex()]
        views.append(ObligationView(spec.family, sub_digest, tuple(w.position for w in u.inputs), (u.output.position,)))
        positions_by_unit[u.vu_id] = tuple(w.position for w in words)
    statement = make_statement(gs.id, bytes.fromhex(str(gs.digest)), ELEM_WIDTH, list(programs.values()), obligations)
    return RawExport(statement, program_meta, tuple(units), tuple(views), positions_by_unit)


def expectations_of(units: Iterable[ElementwiseUnit], gate_set=None) -> dict[str, Any]:
    """A2 ``ExpectedObligation`` per VU (``veritor.interface.expected``): ordered incoming words then the output
    word as ``ExpectedValue(tree, leaf, (index,))``, the kind's subcircuit digest, no pins.  The caller supplies the
    units from the record description (its locator); this only shapes them."""

    from veritor.interface.expected import ExpectedObligation, ExpectedValue

    gs = gate_set if gate_set is not None else elem_gate_set()
    digests: dict[str, str] = {}
    out: dict[str, Any] = {}
    for u in units:
        spec = _check_unit(u)
        if spec.kind not in digests:
            digests[spec.kind] = kind_program(spec, gs)[1]
        out[u.vu_id] = ExpectedObligation(
            tuple(ExpectedValue(w.tree, w.canonical_id, (w.index,)) for w in u.inputs),
            (ExpectedValue(u.output.tree, u.output.canonical_id, (u.output.index,)),),
            digests[spec.kind], (None,), {}, f"{spec.kind} {u.vu_id}",
        )
    return out


def obligations_of_raw_statement(
    statement: Statement,
    tree_of_root: Mapping[bytes, str],
    id_at_rank: Callable[[str, int], str | None],
    programs: Mapping[str, tuple[str, str]],
) -> tuple[ObligationView, ...]:
    """Read a raw format-3 statement back into :class:`ObligationView` rows, using only the verifier's data:
    ``tree_of_root`` maps a committed root to its tree id, ``id_at_rank(tree, rank)`` the committed id list, and
    ``programs`` the family's ``KindProgram.kind`` hex -> ``(kind, subcircuit digest)`` table (the fixed circuit).
    Refuses an unknown root, rank or program, or a non-packed position."""

    from veritor.protocol.proofs.packed import parse_packed_schema

    views: list[ObligationView] = []
    family = statement.gate_set_id
    for ob in statement.obligations:
        ob = ob.expand()
        meta = programs.get(ob.kind.hex())
        if meta is None:
            raise CoverageError(f"obligation {ob.unit}: program {ob.kind.hex()[:12]} is not a kind of the fixed circuit")
        _, sub_digest = meta
        positions: list[Position] = []
        for p in ob.positions:
            c = ob.commitments[p.commitment]
            tree = tree_of_root.get(c.root)
            if tree is None:
                raise CoverageError(f"obligation {ob.unit}: root {c.root.hex()[:12]} is not a committed tree")
            cid = id_at_rank(tree, p.rank)
            if cid is None or p.position != p.rank:
                raise CoverageError(f"obligation {ob.unit}: rank {p.rank} of {tree!r} is not a committed leaf")
            packed = parse_packed_schema(p.schema)
            if packed is None:
                raise CoverageError(f"obligation {ob.unit}: position schema {p.schema!r} is not a packed word")
            positions.append((tree, cid, packed.index))
        views.append(ObligationView(family, sub_digest, tuple(positions[k] for k in ob.inputs), tuple(positions[m] for m in ob.outputs)))
    return tuple(views)


def verify_locally(export: RawExport, openings: Iterable[Any], gate_set=None) -> bool:
    """The transparent backend's verdict on ``export.statement`` with a format-3 witness assembled from the record's
    ``verity_vllm.commit.Opening`` rows (``typed_obligation_export.packed_witness_of``).  ``False`` on rejection."""

    from veritor.protocol.messages import Reject
    from veritor.protocol.proofs import TransparentBackend
    from verity_vllm.typed_obligation_export import ExportError, packed_witness_of

    gs = gate_set if gate_set is not None else elem_gate_set()
    try:
        witness = packed_witness_of(openings, export.statement)  # ExportError: an opened rank/root no opening covers
        return bool(TransparentBackend(gs).verify(export.statement, encode_packed_witness(witness)))
    except (Reject, ExportError):
        return False
