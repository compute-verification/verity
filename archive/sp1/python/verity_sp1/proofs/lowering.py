"""The lowering boundary: ``typed-obligation/v0`` -> the legacy SP1 statement, and back.

Everything above this module is Verity-level (IR §F): subcircuits of one-output
gates, boundaries of value references, obligations.  Everything below it is the
legacy backend's representation of a lowered local relation -- ``KindProgram``
(ports, ``port k`` / ``local i`` arguments, opened offsets), ``Obligation``
slots, ``PositionRef`` pins -- which this module *derives* with :func:`lower`
and never exposes upward.  In the theory side's words: ``KindProgram`` is a
backend representation of the selected subcircuit relation; L7 is the theorem
``Eval_backend(lower(S)) = Eval_Verity(S)`` that justifies this boundary.

* :func:`lower` maps an accepted object to a :class:`Lowering`: the canonical
  :class:`~veritor.protocol.proofs.Statement` (wire format 2 through
  :func:`statement_bytes`) and a JSON description of the backend data it
  produced (``lowering_json``), so a conformance vector can check that another
  implementation recomputes the same lowering from the Verity-level object.
  The lowering is a *function* of the Verity-level data: incoming boundary
  value ``i`` -> port ``i``; gate ``j`` -> local ``j``; an ``Input`` gate -> the
  family's ``in`` source gate, opened, its position pinned to the prescribed
  value (``PositionRef.expected``: the backend's form of the sampled input
  equation ``claimed = x[k]``, IR §E); ``out[m]`` -> opened offset; positions
  are ``boundary.in ++ boundary.out``.
* :func:`from_statement` lifts a statement of a supported family to a
  Verity-level object: a port or an unpinned source is an incoming boundary
  value; a pinned position of a source or a port is an ``Input`` gate; opened
  gates are the outgoing boundary.  Lowering the lift gives a statement with
  the same relation on the same boundary values (the legacy statement's free
  choices -- position order, source-vs-port, compiled kind digests -- are
  canonicalised, so the bytes are not in general the original's).
* :func:`commitment_position` is the checked ``ValueRef -> CommitmentPosition``
  mapping (IR §H.4): a reference's ``(leaf, offset)`` resolved through the
  leaf's layout and its root's framing to the backend's ``PositionRef`` --
  ``u<w>`` for a one-word leaf under ``veritor/protocol/merkle/frame/v3``,
  ``packed/v1:verity-vllm/leaf/v1:u<w>le:<count>:<index>`` for a word of a
  row leaf committed under the serving side's framing (the guest authenticates
  the whole leaf and reads the word at the statement-fixed index; witness
  format 3, ``veritor.protocol.proofs.packed``).
* :func:`authenticity` says whether every boundary word's position is one the
  guest can authenticate: the two shapes above.  A leaf packing a whole row
  under frame v3, or a tuple/record layout under the serving side's framing,
  lowers fine -- the relation is checkable on the decoded words -- but the
  guest cannot open one word of it; those remain the recorded gaps.
* :func:`check_relations` is ``Eval_Verity``: the relation check on the words
  at the boundary, including the ``Input`` equations.  It agrees with the
  backend's ``check_relations`` (``TransparentBackend._check_relations``,
  ``archive/sp1/common/src/kernel/relations.rs``) plus the backend's pin check.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from verity.ml.commitments.identity import canonical_json_bytes, tagged_sha256

from .gateset import GateSet
from .statement import CommitmentRef, GateOp, KindProgram, PositionRange, PositionRef, Statement
from .statement import Obligation as WireObligation
from .statement import PositionEntry
from .wire import encode_statement, encode_statement_v3

from .schema import VLLM_LEAF_FRAMING, packed_schema, parse_packed_schema

from .typed_obligation import (
    GATE,
    IN,
    INPUT,
    MERKLE_FRAMING,
    VLLM_FRAMING,
    Array,
    Binding,
    Bits,
    Boundary,
    Code,
    Family,
    Gate,
    InterfaceReject,
    Leaf,
    Obligation,
    Reference,
    Root,
    Named,
    Selector,
    Subcircuit,
    Type,
    TypedObligationSet,
    component,
    family_gate_set,
    scalar_width,
    subcircuit_digest,
    validate,
)

__all__ = [
    "Authenticity",
    "CommitmentPosition",
    "LEGACY_PROGRAM_TAG",
    "Lowering",
    "authenticity",
    "check_relations",
    "commitment_position",
    "from_statement",
    "lower",
    "lower_subcircuit",
    "lower_v3",
    "lowering_json",
    "lowering_v3_json",
    "merge_ranges",
    "statement_bytes",
    "statement_bytes_v3",
    "to_statement",
    "to_statement_v3",
]

#: The legacy backend's identity of a lowered body ("kind"): a tagged digest of
#: the lowered program.  Behind the lowering boundary; never a field of the object.
LEGACY_PROGRAM_TAG = "veritor/typed-obligation/v0/program"

_BACKEND_SOURCE = "in"


@dataclass(frozen=True, slots=True)
class Lowering:
    """What :func:`lower` produces: the legacy statement and its description."""

    statement: Statement
    description: dict[str, Any]


# --- the authentication-location layer: ValueRef -> CommitmentPosition ------------------


@dataclass(frozen=True, slots=True)
class CommitmentPosition:
    """Where one ``Value<w>`` is authenticated (IR §H.4): ``(commitment_domain, leaf_id,
    word_index)`` in the backend's spelling -- the root (``commitment`` indexes the
    obligation's roots), the leaf ``rank`` and domain ``position``, and the leaf
    ``schema`` that says how the word is read out of the leaf.

    ``schema`` is ``u<w>`` for a one-word leaf under ``veritor/protocol/merkle/frame/v3``
    (the guest hashes the word itself as the leaf) and
    ``packed/v1:verity-vllm/leaf/v1:u<w>le:<count>:<index>`` for a word packed into a
    leaf committed under the serving side's framing (the guest authenticates the whole
    leaf and reads word ``index``; :mod:`veritor.protocol.proofs.packed`).  ``word_index``
    is ``None`` for the former and the row-major index of the reference's offset for
    the latter.  This layer is *lowering* data: never a field of the Verity-level object.
    """

    commitment: int
    rank: int
    position: int
    schema: str
    width: int
    word_index: int | None


def _uniform_packing(t: Type) -> tuple[int, tuple[int, ...]] | None:
    """``(w, extents)`` if ``t`` is ``Array<n1, ... Array<nk, Value<w>>>`` (through aliases): one
    word type, row-major; ``None`` for a tuple or record layout (heterogeneous packing)."""

    extents: list[int] = []
    while True:
        while isinstance(t, Named):
            t = t.of
        if isinstance(t, Bits):
            return t.width, tuple(extents)
        if isinstance(t, Array):
            extents.append(t.length)
            t = t.of
            continue
        return None


def _flat_index(extents: Sequence[int], offset: Sequence[Selector]) -> int:
    """The row-major index of ``offset`` (already validated against the layout) in ``extents``."""

    index = 0
    for extent, step in zip(extents, offset, strict=True):
        assert isinstance(step, int)
        index = index * extent + step
    return index


def commitment_position(root: Root, leaf: Leaf, ref: Reference, commitment: int) -> CommitmentPosition:
    """The checked ``ValueRef -> CommitmentPosition`` mapping for one reference.

    The reference's offset is resolved through the leaf's layout to one ``Value<w>``
    (``validate`` has already required that).  Under ``verity-vllm/leaf/v1`` a leaf
    whose layout is a uniform packing of ``Value<w>`` words (``Array<..., Value<w>>``,
    W3c's row leaves) gets the packed schema with the word's row-major index -- the
    serving side commits each word as ``ceil(w/8)`` raw little-endian bytes, so the
    byte order is ``le``.  Every other leaf keeps the one-word schema ``u<w>``, which
    only a one-word leaf under frame v3 can authenticate (:func:`authenticity` says
    which).  A statement's positions thus fix the word: a prover cannot pick which
    word of a leaf stands for a value.
    """

    width = scalar_width(component(leaf.layout, ref.offset))
    assert width is not None  # validated
    if root.framing == VLLM_FRAMING:
        packing = _uniform_packing(leaf.layout)
        if packing is not None:
            word_width, extents = packing
            assert word_width == width
            count = 1
            for extent in extents:
                count *= extent
            index = _flat_index(extents, ref.offset)
            return CommitmentPosition(
                commitment,
                leaf.leaf,
                leaf.position,
                packed_schema(VLLM_LEAF_FRAMING, width, True, count, index),
                width,
                index,
            )
    return CommitmentPosition(commitment, leaf.leaf, leaf.position, f"u{width}", width, None)


def _lower_subcircuit(s: Subcircuit, gate_set: GateSet) -> tuple[KindProgram, dict[str, Any]]:
    gates = []
    for g in s.gates:
        if g.is_input:
            gates.append(GateOp(_BACKEND_SOURCE, ()))
        else:
            gates.append(
                GateOp(g.op, tuple(("port" if space == IN else "local", index) for space, index in g.args))
            )
    body = {
        "ports": [{"ordinal": i, "width": w} for i, w in enumerate(s.incoming)],
        "gates": [{"op": g.op, "args": [list(a) for a in g.args]} for g in gates],
        "outputs": list(s.outgoing),
    }
    kind = str(tagged_sha256(LEGACY_PROGRAM_TAG, canonical_json_bytes(body)))
    program = KindProgram(
        bytes.fromhex(kind), len(gates), tuple(range(len(s.incoming))), tuple(gates), s.outgoing
    )
    description = {"subcircuit": s.digest, "kind": kind, **body}
    return program, description


def lower(obj: TypedObligationSet, *, obligation_format: str | None = None) -> Lowering:
    """Lower an accepted object (validates first) to the legacy statement.  ``obligation_format`` is :func:`validate`'s [E9, D28];
    an aliased incoming word lowers to two positions naming one commitment slot (the positional rule below is unchanged)."""

    gate_set = validate(obj, obligation_format=obligation_format)
    roots = {r.id: r for r in obj.roots}
    leaves = {v.id: v for v in obj.leaves}
    references = {r.ref: r for r in obj.references}
    lowered = {s.digest: _lower_subcircuit(s, gate_set) for s in obj.subcircuits}
    # the statement orders bodies by their legacy digest, the object by the subcircuit
    # digest; subcircuits that differ only in their Input gates' prescribed values
    # lower to one legacy body (the values become pins on the obligations)
    kinds = tuple(sorted({program.kind: program for program, _ in lowered.values()}.values(), key=lambda p: p.kind))
    subcircuits = {s.digest: s for s in obj.subcircuits}
    obligations = []
    descriptions = []
    for o in obj.obligations:
        s = subcircuits[o.subcircuit]
        program, _ = lowered[o.subcircuit]
        commitments = tuple(
            CommitmentRef(
                roots[rid].owner, bytes.fromhex(roots[rid].domain), bytes.fromhex(roots[rid].root), roots[rid].count
            )
            for rid in o.roots
        )
        # the sampled Input equations: the pin on the position of each Input gate's value
        pins: dict[str, str] = {}
        for m, ref_id in enumerate(o.boundary.outgoing):
            gate = s.gates[s.outgoing[m]]
            if gate.is_input:
                assert gate.value is not None
                pins[ref_id] = gate.value
        positions = []
        ordered = (*o.boundary.incoming, *o.boundary.outgoing)
        for ref_id in ordered:
            ref = references[ref_id]
            leaf = leaves[ref.leaf]
            where = commitment_position(roots[leaf.root], leaf, ref, o.roots.index(leaf.root))
            expected = pins.get(ref_id)
            positions.append(
                PositionRef(
                    where.commitment,
                    where.rank,
                    where.position,
                    where.schema,
                    None if expected is None else bytes.fromhex(expected),
                )
            )
        n_in = len(o.boundary.incoming)
        inputs = tuple(range(n_in))
        outputs = tuple(n_in + m for m in range(len(o.boundary.outgoing)))
        obligations.append(
            WireObligation(
                bytes.fromhex(o.binding.session),
                bytes.fromhex(o.binding.compiled),
                o.binding.unit,
                o.binding.replay_unit,
                program.kind,
                commitments,
                tuple(positions),
                inputs,
                outputs,
            )
        )
        descriptions.append(
            {
                "kind": program.kind.hex(),
                "positions": list(ordered),
                "inputs": list(inputs),
                "outputs": list(outputs),
                "expected": {ref_id: value for ref_id, value in pins.items()},
            }
        )
    statement = Statement(
        obj.family.id, bytes.fromhex(obj.family.digest), obj.family.width, kinds, tuple(obligations)
    )
    return Lowering(
        statement,
        {
            "note": "derived by veritor.interface.lowering.lower(); legacy backend data, not part of the Verity ontology",
            "programs": sorted((description for _, description in lowered.values()), key=lambda d: (d["kind"], d["subcircuit"])),
            "obligations": descriptions,
        },
    )


def to_statement(obj: TypedObligationSet, *, obligation_format: str | None = None) -> Statement:
    """The canonical statement of an accepted object."""

    return lower(obj, obligation_format=obligation_format).statement


def statement_bytes(obj: TypedObligationSet, *, obligation_format: str | None = None) -> bytes:
    """The wire-format-2 bytes of :func:`to_statement`."""

    return encode_statement(to_statement(obj, obligation_format=obligation_format))


def lowering_json(obj: TypedObligationSet) -> dict[str, Any]:
    """The JSON description of :func:`lower`'s backend data, for conformance records."""

    return lower(obj).description


# --- statement format 3: ranges of packed words -----------------------------------------


def merge_ranges(positions: Sequence[PositionRef]) -> tuple[PositionEntry, ...]:
    """Greedily replace maximal runs of consecutive packed positions by :class:`PositionRange` entries.

    This is the format-3 lowering of a position list, and the only rule by which
    the lowering ever emits a range: a run of ``>= 2`` consecutive entries
    that are unpinned, packed (``packed/v1``), name one leaf (same
    ``commitment``, ``rank``, ``position`` and leaf descriptor) and whose word
    indices advance by one constant positive stride becomes one range; every
    other position stays a :class:`PositionRef`.  Pinned positions (sampled
    ``Input`` gates) are never merged.  By construction
    ``expand(merge_ranges(ps)) == ps`` -- the same words in the same order --
    so slots, aliases and domain qualification are untouched.  Deterministic:
    two implementations of the lowering emit the same entries.
    """

    entries: list[PositionEntry] = []
    i = 0
    n = len(positions)
    while i < n:
        head = positions[i]
        parsed = None if head.expected is not None else parse_packed_schema(head.schema)
        if parsed is None:
            entries.append(head)
            i += 1
            continue
        descriptor = parsed.descriptor
        key = (head.commitment, head.rank, head.position, descriptor)
        # the run's members and their indices
        indices = [parsed.index]
        j = i + 1
        stride: int | None = None
        while j < n:
            candidate = positions[j]
            if candidate.expected is not None:
                break
            packed = parse_packed_schema(candidate.schema)
            if packed is None or (candidate.commitment, candidate.rank, candidate.position, packed.descriptor) != key:
                break
            step = packed.index - indices[-1]
            if step < 1 or (stride is not None and step != stride):
                break
            stride = step
            indices.append(packed.index)
            j += 1
        if len(indices) >= 2:
            assert stride is not None
            entries.append(
                PositionRange(
                    head.commitment, head.rank, head.position, descriptor.text(), indices[0], len(indices), stride
                )
            )
        else:
            entries.append(head)
        i = j if len(indices) >= 2 else i + 1
    return tuple(entries)


def lower_v3(obj: TypedObligationSet, *, obligation_format: str | None = None) -> Lowering:
    """Lower an accepted object to the format-3 statement: :func:`lower` with its position
    lists passed through :func:`merge_ranges`.

    The typed object is untouched (``h_O`` is the same digest as for :func:`lower`); only
    the backend's spelling of the positions changes, so ``lower_v3(obj).statement.expand()
    == lower(obj).statement`` and the two statements have the same relation, the same
    authenticated words and the same verdict on the same witness.  The description lists,
    per obligation, the entries as the per-word ``lowering`` record would see them: a
    single is ``{"ref", "schema"}``; a range ``{"refs": [first, last], "descriptor",
    "start", "count", "stride", "first_schema", "last_schema"}`` where the two schemas
    are the per-word schemas of the run's first and last member (what the expansion
    must reproduce), and ``slots`` is the expanded position count.
    """

    base = lower(obj, obligation_format=obligation_format)
    obligations = []
    descriptions = []
    for wire, description in zip(base.statement.obligations, base.description["obligations"], strict=True):
        entries = merge_ranges(wire.positions)
        refs = description["positions"]
        listed = []
        offset = 0
        for entry in entries:
            if isinstance(entry, PositionRange):
                first, last = wire.positions[offset], wire.positions[offset + entry.count - 1]
                listed.append(
                    {
                        "refs": [refs[offset], refs[offset + entry.count - 1]],
                        "commitment": entry.commitment,
                        "rank": entry.rank,
                        "position": entry.position,
                        "descriptor": entry.descriptor,
                        "start": entry.start,
                        "count": entry.count,
                        "stride": entry.stride,
                        "first_schema": first.schema,
                        "last_schema": last.schema,
                    }
                )
                offset += entry.count
            else:
                listed.append({"ref": refs[offset], "schema": entry.schema})
                offset += 1
        assert offset == len(wire.positions)
        obligations.append(
            WireObligation(
                wire.session,
                wire.compiled,
                wire.unit,
                wire.replay_unit,
                wire.kind,
                wire.commitments,
                entries,
                wire.inputs,
                wire.outputs,
            )
        )
        descriptions.append(
            {
                "kind": description["kind"],
                "entries": listed,
                "ranges": sum(isinstance(e, PositionRange) for e in entries),
                "slots": len(wire.positions),
                "inputs": description["inputs"],
                "outputs": description["outputs"],
                "expected": description["expected"],
            }
        )
    statement = Statement(
        base.statement.gate_set_id,
        base.statement.gate_set_digest,
        base.statement.width,
        base.statement.kinds,
        tuple(obligations),
    )
    assert statement.expand() == base.statement
    return Lowering(
        statement,
        {
            "note": (
                "derived by veritor.interface.lowering.lower_v3(): the format-2 lowering with maximal runs of "
                "consecutive packed words of one leaf merged into ranges (merge_ranges); expands to `lowering`"
            ),
            "programs": base.description["programs"],
            "obligations": descriptions,
        },
    )


def to_statement_v3(obj: TypedObligationSet, *, obligation_format: str | None = None) -> Statement:
    """The canonical format-3 statement of an accepted object (ranges where the rows are uniform)."""

    return lower_v3(obj, obligation_format=obligation_format).statement


def statement_bytes_v3(obj: TypedObligationSet, *, obligation_format: str | None = None) -> bytes:
    """The wire-format-3 bytes of :func:`to_statement_v3`."""

    return encode_statement_v3(to_statement_v3(obj, obligation_format=obligation_format))


def lowering_v3_json(obj: TypedObligationSet) -> dict[str, Any]:
    """The JSON description of :func:`lower_v3`'s entries, for the ``lowered_v3`` conformance column."""

    return lower_v3(obj).description


@dataclass(frozen=True, slots=True)
class Authenticity:
    """Whether the lowered statement's openings can authenticate in the guest."""

    authenticatable: bool
    reasons: tuple[str, ...]


def authenticity(obj: TypedObligationSet) -> Authenticity:
    """Whether every boundary word's :func:`commitment_position` is one the guest can authenticate.

    Two shapes are: a one-word leaf under ``veritor/protocol/merkle/frame/v3`` (schema
    ``u<w>``, witness formats 1 and 2) and a word of a uniformly packed leaf under
    ``verity-vllm/leaf/v1`` (schema ``packed/v1``, witness format 3).  A packed leaf
    under frame v3 -- the guest's leaf hash covers one word -- or a tuple/record
    layout under the serving side's framing is not.
    """

    roots = {r.id: r for r in obj.roots}
    leaves = {v.id: v for v in obj.leaves}
    references = {r.ref: r for r in obj.references}
    reasons: list[str] = []
    for o in obj.obligations:
        for ref_id in (*o.boundary.incoming, *o.boundary.outgoing):
            leaf = leaves[references[ref_id].leaf]
            framing = roots[leaf.root].framing
            if framing == MERKLE_FRAMING:
                if scalar_width(leaf.layout) is None:
                    reasons.append(
                        f"leaf {leaf.id!r} packs several words under {MERKLE_FRAMING!r}; "
                        "the guest opens one-word leaves only under that framing"
                    )
            elif framing == VLLM_FRAMING:
                if _uniform_packing(leaf.layout) is None:
                    reasons.append(
                        f"leaf {leaf.id!r} under {VLLM_FRAMING!r} is not a uniform packing of one "
                        "Value<w> (packed/v1 authenticates Array<..., Value<w>> leaves only)"
                    )
            else:
                reasons.append(f"root {leaf.root!r} is framed {framing!r}, which no witness format authenticates")
    unique = tuple(dict.fromkeys(reasons))
    return Authenticity(not unique, unique)


def from_statement(
    statement: Statement,
    *,
    root_ids: Sequence[str] | None = None,
    value_id: str | None = None,
) -> TypedObligationSet:
    """Lift a legacy statement of a supported family to a Verity-level object.

    Roots are named ``r<n>`` in order of first appearance (or ``root_ids``);
    every opened position becomes one one-word leaf and one reference, both
    named ``<root id>/<position>`` (the position *is* Track B's identity of
    the value); leaf layouts come from the leaf schemas ``u<w>``; incoming
    widths from the consuming gates' declared argument widths.  Per
    obligation: a port or an unpinned source gate is an incoming boundary
    value; a pinned source or port is an ``Input`` gate with the pin as its
    prescribed value -- in the *first* obligation that pins the position; a
    later obligation pinning the same position (the legacy statement's
    redundant re-check in a consumer) reads it as incoming, since the gate
    belongs to one subcircuit; opened non-source gates are outgoing.  Bodies
    that agree after this are one subcircuit.  A pin on a produced
    (non-source) value has no Verity-level form and is refused
    (``E_NOT_LOWERABLE``).
    """

    gate_set = family_gate_set(statement.gate_set_id, statement.width)
    if gate_set is None:
        raise InterfaceReject(Code.E_UNSUPPORTED_FAMILY, f"unknown family {statement.gate_set_id!r}")
    roots: dict[tuple[int, bytes, bytes, int], Root] = {}
    for o in statement.obligations:
        for c in o.commitments:
            key = (c.owner, c.domain_id, c.root, c.count)
            if key not in roots:
                rid = root_ids[len(roots)] if root_ids is not None else f"r{len(roots)}"
                roots[key] = Root(rid, c.owner, c.domain_id.hex(), c.root.hex(), c.count)
    leaves: dict[tuple[str, int, int], Leaf] = {}
    references: dict[tuple[str, int, int], Reference] = {}
    for o in statement.obligations:
        for ref in o.positions:
            c = o.commitments[ref.commitment]
            rid = roots[(c.owner, c.domain_id, c.root, c.count)].id
            width = _schema_width(ref.schema)
            key = (rid, ref.rank, ref.position)
            vid = f"{rid}/{ref.position}" if value_id is None else f"{value_id}/{rid}/{ref.position}"
            existing = leaves.get(key)
            if existing is None:
                leaves[key] = Leaf(vid, rid, ref.rank, ref.position, Bits(width))
                references[key] = Reference(vid, vid, ())
            elif scalar_width(existing.layout) != width:
                raise InterfaceReject(Code.E_IDENTITY_CONFLICT, f"leaf {key} opened under two schemas")
    # A pinned position is the backend's form of a sampled Input gate.  The legacy
    # statement may pin one position in several obligations (a dedicated
    # single-source obligation *and* every consumer that reads it); the gate
    # belongs to one subcircuit, so the first obligation that pins it gets the
    # Input gate and every other one reads the value as incoming.  Two pins with
    # different values are two prescriptions of one input: E_IDENTITY_CONFLICT.
    pin_owner: dict[tuple[str, int, int], tuple[int, str]] = {}
    for index, o in enumerate(statement.obligations):
        for ref in o.positions:
            if ref.expected is None:
                continue
            c = o.commitments[ref.commitment]
            key = (roots[(c.owner, c.domain_id, c.root, c.count)].id, ref.rank, ref.position)
            owner = pin_owner.setdefault(key, (index, ref.expected.hex()))
            if owner[1] != ref.expected.hex():
                raise InterfaceReject(
                    Code.E_IDENTITY_CONFLICT, f"position {key} pinned to two public words"
                )
    programs = {p.kind: p for p in statement.kinds}
    subcircuits: dict[str, Subcircuit] = {}
    obligations = []
    for index, o in enumerate(statement.obligations):
        rids = tuple(roots[(c.owner, c.domain_id, c.root, c.count)].id for c in o.commitments)
        keys = [(rids[ref.commitment], ref.rank, ref.position) for ref in o.positions]
        slot_ref = [references[key].ref for key in keys]
        slot_pin = [
            ref.expected.hex() if ref.expected is not None and pin_owner[key][0] == index else None
            for key, ref in zip(keys, o.positions, strict=True)
        ]
        program = programs[o.kind]
        s, incoming_refs, outgoing_refs = _lift_body(program, gate_set, o.inputs, o.outputs, slot_ref, slot_pin)
        subcircuits.setdefault(s.digest, s)
        obligations.append(
            Obligation(
                Binding(o.session.hex(), o.compiled.hex(), o.unit, o.replay_unit),
                s.digest,
                rids,
                Boundary(tuple(incoming_refs), tuple(outgoing_refs)),
            )
        )
    return TypedObligationSet(
        Family(statement.gate_set_id, gate_set.digest, statement.width),
        (),
        tuple(sorted(roots.values(), key=lambda r: r.id)),
        tuple(sorted(leaves.values(), key=lambda v: v.id)),
        tuple(sorted(references.values(), key=lambda r: r.ref)),
        tuple(sorted(subcircuits.values(), key=lambda s: s.digest)),
        tuple(obligations),
    )


def _schema_width(schema: str) -> int:
    if schema.startswith("packed/v1:"):
        # a packed position is lowered from an object (commitment_position), never lifted:
        # the lift's leaves are one-word by construction
        raise InterfaceReject(Code.E_NOT_LOWERABLE, f"packed position {schema!r} has no lift to a one-word leaf")
    if not schema.startswith("u") or not schema[1:].isdigit() or not 1 <= int(schema[1:]) <= 64:
        raise InterfaceReject(Code.E_FORMAT, f"leaf schema {schema!r} is not u<width>")
    return int(schema[1:])


def _lift_body(
    p: KindProgram,
    gate_set: GateSet,
    inputs: Sequence[int],
    outputs: Sequence[int],
    slot_ref: Sequence[str],
    slot_pin: Sequence[str | None],
) -> tuple[Subcircuit, list[str], list[str]]:
    """One obligation's legacy body and bindings -> (subcircuit, boundary.in refs, boundary.out refs)."""

    source_width = gate_set["in"].width
    # a port's width is what its consumers declare; an unread port gets the source width
    port_width: dict[int, int] = {}
    for gate in p.gates:
        try:
            semantics = gate_set[gate.op]
        except Exception:
            continue
        for i, (space, index) in enumerate(gate.args):
            if space == "port" and i < len(semantics.arg_widths):
                port_width.setdefault(index, semantics.arg_widths[i])
    opened = dict(zip(p.outputs, outputs, strict=True))  # offset -> slot

    gates: list[Gate] = []
    incoming: list[int] = []
    incoming_refs: list[str] = []
    outgoing: list[int] = []
    outgoing_refs: list[str] = []
    port_arg: dict[int, tuple[str, int]] = {}
    # pinned ports become Input gates at the front of the body
    for k, slot in enumerate(inputs):
        width = port_width.get(k, source_width)
        pin = slot_pin[slot]
        if pin is not None:
            port_arg[k] = (GATE, len(gates))
            outgoing.append(len(gates))
            outgoing_refs.append(slot_ref[slot])
            gates.append(Gate(INPUT, (), width, pin))
        else:
            port_arg[k] = (IN, len(incoming))
            incoming.append(width)
            incoming_refs.append(slot_ref[slot])
    local_arg: dict[int, tuple[str, int]] = {}
    for offset, gate in enumerate(p.gates):
        slot = opened.get(offset)
        is_source = gate_set[gate.op].source is not None
        if is_source:
            assert slot is not None
            pin = slot_pin[slot]
            if pin is None:
                local_arg[offset] = (IN, len(incoming))
                incoming.append(gate_set[gate.op].width)
                incoming_refs.append(slot_ref[slot])
                continue
            local_arg[offset] = (GATE, len(gates))
            outgoing.append(len(gates))
            outgoing_refs.append(slot_ref[slot])
            gates.append(Gate(INPUT, (), gate_set[gate.op].width, pin))
            continue
        if slot is not None and slot_pin[slot] is not None:
            raise InterfaceReject(
                Code.E_NOT_LOWERABLE, f"gate {offset} ({gate.op}) is produced and pinned: no Verity-level form"
            )
        args = tuple(port_arg[index] if space == "port" else local_arg[index] for space, index in gate.args)
        local_arg[offset] = (GATE, len(gates))
        if slot is not None:
            outgoing.append(len(gates))
            outgoing_refs.append(slot_ref[slot])
        gates.append(Gate(gate.op, args))
    order = sorted(range(len(outgoing)), key=lambda m: outgoing[m])
    outgoing = [outgoing[m] for m in order]
    outgoing_refs = [outgoing_refs[m] for m in order]
    body = Subcircuit("00" * 32, tuple(incoming), tuple(gates), tuple(outgoing))
    body = Subcircuit(subcircuit_digest(body), body.incoming, body.gates, body.outgoing)
    return body, incoming_refs, outgoing_refs


def check_relations(
    obj: TypedObligationSet, values: Sequence[Sequence[int]], gate_set: GateSet | None = None
) -> bool:
    """``Eval_Verity``: every obligation's outgoing values agree with the body on its incoming values.

    ``values[i]`` are the decoded words of obligation ``i``'s boundary, incoming
    then outgoing.  Returns ``False`` where the reference checker rejects: a
    gate whose evaluation fails (an argument outside its width, a non-finite
    tensor-core operand), an outgoing value that differs from the
    recomputation, or a sampled ``Input`` gate whose claimed word is not its
    prescribed value.  Well-formedness is the validator's.
    """

    if gate_set is None:
        gate_set = validate(obj)
    subcircuits = {s.digest: s for s in obj.subcircuits}
    if len(values) != len(obj.obligations):
        return False
    for o, row in zip(obj.obligations, values, strict=True):
        s = subcircuits[o.subcircuit]
        n_in = len(s.incoming)
        if len(row) != n_in + len(s.outgoing):
            return False
        claimed = {j: row[n_in + m] for m, j in enumerate(s.outgoing)}
        produced: list[int] = []
        for j, gate in enumerate(s.gates):
            if gate.is_input:
                assert gate.value is not None
                word = claimed[j]
                if word != int(gate.value, 16):
                    return False
                produced.append(word)
                continue
            semantics = gate_set[gate.op]
            args = tuple(produced[index] if space == GATE else row[index] for space, index in gate.args)
            try:
                if j not in claimed:
                    produced.append(semantics.evaluate(args))
                    continue
                satisfied = semantics.check(args, claimed[j])
            except Exception:
                return False
            if not satisfied:
                return False
            produced.append(claimed[j])
    return True


def lower_subcircuit(sub: Any) -> dict:
    """Lowering adapter: a supported IR subcircuit (``verity.ml.circuits.layout.GateSelection``) -> the legacy backend's local
    straight-line program (the `typed-obligation/v0` lowering section). Semantics: gates read `port k` (k-th
    distinct boundary-in value, ascending producing-gate index) or `local i` (i-th gate of the subcircuit in
    canonical order); `outputs` are the local indices of the subcircuit's `out` boundary. An `Input` gate inside the subcircuit lowers to a `port` whose expected
    value is the prescribed input (the sampled `Input[k]` obligation: authenticated supplied value == x[k])."""
    gates = list(sub)
    index_of = {g.index: i for i, g in enumerate(gates)}
    ports = list(sub.in_)
    port_of = {g: k for k, g in enumerate(ports)}
    lowered_gates = []
    input_ports = []
    for i, g in enumerate(gates):
        if getattr(g.prim, "family", None) == "Input":
            # prescribed-input equation: the backend authenticates a supplied value and pins it to x[k]
            input_ports.append({"local": i, "gate": g.index, "expected": "prescribed x[k]", "width": g.prim.ret.w})
            lowered_gates.append({"kind": g.prim.id, "operands": [], "width": g.prim.ret.w, "input_equation": True})
            continue
        ops = []
        for o in g.operands:
            if o in index_of:
                ops.append({"local": index_of[o]})
            else:
                ops.append({"port": port_of[o]})
        lowered_gates.append({"kind": g.prim.id, "operands": ops, "width": g.prim.ret.w})
    return {
        "schema": "typed-obligation/v0/lowering",
        "KindProgram": {
            "ports": [{"port": k, "gate": g, "width": None} for k, g in enumerate(ports)],
            "gates": lowered_gates,
            "outputs": [index_of[o] for o in sub.out],
            "input_equations": input_ports,
        },
    }
