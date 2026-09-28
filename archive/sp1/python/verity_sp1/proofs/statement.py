"""The public statements of the proof layer: obligations, kind programs, batches.

An :class:`Obligation` is what the verifier demands for one sampled
proof unit: *this* copy of *this* kind, under *these* commitment
roots, has every input and output value it touches authenticated at *these*
``(rank, position, schema)`` coordinates, and the kind's relation holds over
them: the copy's gates, recomputed from the opened inputs, produce the opened
outputs.  It names no value: the values are the witness.  A
:class:`KindProgram` is the relation of one kind in coordinates relative to
the copy (its ports, its own gate offsets and which of those are opened), so
one program serves every copy of the kind.  A :class:`Statement` is a batch:
one gate set, the programs of every kind that occurs, and the obligations, all
sorted so the encoding of a set is unique.

Everything here is plain data with strict validation; the canonical bytes are
in :mod:`veritor.protocol.proofs.wire`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from verity.errors import ProtocolError
from verity.ml.commitments.multiproof import multiproof_from_paths

PORT: Literal["port"] = "port"
LOCAL: Literal["local"] = "local"

type ArgSpace = Literal["port", "local"]
type Arg = tuple[ArgSpace, int]
"""A gate argument: ``("port", k)`` reads the copy's ``k``-th port as listed in
:attr:`KindProgram.ports`; ``("local", j)`` reads the copy's own gate at offset ``j``."""

_U32 = 1 << 32
_U64 = 1 << 64


def _check_index(value: object, name: str, bound: int = _U32) -> int:
    if type(value) is not int or not 0 <= value < bound:
        raise ProtocolError(f"{name} must be an integer in [0, {bound})")
    return value


def _check_digest(value: object, name: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise ProtocolError(f"{name} must be 32 bytes")
    return value


@dataclass(frozen=True, slots=True)
class GateOp:
    """One gate of a kind: its op and where it reads."""

    op: str
    args: tuple[Arg, ...]

    def __post_init__(self) -> None:
        if type(self.op) is not str or not self.op:
            raise ProtocolError("gate op must be a nonempty string")
        if type(self.args) is not tuple:
            raise ProtocolError("gate args must be a tuple")
        for arg in self.args:
            if (
                type(arg) is not tuple
                or len(arg) != 2
                or arg[0] not in (PORT, LOCAL)
                or type(arg[1]) is not int
                or not 0 <= arg[1] < _U32
            ):
                raise ProtocolError("gate args must be ('port' | 'local', index) pairs")


@dataclass(frozen=True, slots=True)
class KindProgram:
    """The relation of one kind, in coordinates relative to a copy.

    ``ports`` lists (ascending) the port ordinals of the definition that its
    gates read, transitively; a ``("port", k)`` argument means the port
    ``ports[k]``.  ``gates[j]`` is the copy's gate at offset ``j``, and every
    ``("local", i)`` argument of it has ``i < j``: the kind is a program that
    reads only what it has already produced.  ``outputs`` lists (ascending)
    the gate offsets whose values a copy opens: the kind's declared outputs
    (``Out``, what other units read and the check compares against the
    recomputation) and its source gates (``in`` and ``weight`` gates have no
    relation; their opened value is their value, pinned by the boundary or by
    ``kappa_W``).  Every other gate is internal: recomputed, never committed.
    """

    kind: bytes
    size: int
    ports: tuple[int, ...]
    gates: tuple[GateOp, ...]
    outputs: tuple[int, ...]

    def __post_init__(self) -> None:
        _check_digest(self.kind, "kind digest")
        _check_index(self.size, "kind size")
        if type(self.ports) is not tuple or any(
            type(item) is not int or not 0 <= item < _U32 for item in self.ports
        ):
            raise ProtocolError("kind ports must be a tuple of ordinals")
        if tuple(sorted(set(self.ports))) != self.ports:
            raise ProtocolError("kind ports must be strictly increasing")
        if type(self.gates) is not tuple or len(self.gates) != self.size:
            raise ProtocolError(
                f"kind declares {self.size} gates but lists {len(self.gates)}"
            )
        if type(self.outputs) is not tuple or any(
            type(item) is not int or not 0 <= item < self.size for item in self.outputs
        ):
            raise ProtocolError("kind outputs must be a tuple of gate offsets")
        if tuple(sorted(set(self.outputs))) != self.outputs:
            raise ProtocolError("kind outputs must be strictly increasing")
        for offset, gate in enumerate(self.gates):
            if not isinstance(gate, GateOp):
                raise ProtocolError("kind gates must be GateOp values")
            for space, value in gate.args:
                if space == PORT and value >= len(self.ports):
                    raise ProtocolError(
                        f"gate {offset} reads port index {value} of {len(self.ports)}"
                    )
                if space == LOCAL and value >= offset:
                    raise ProtocolError(
                        f"gate {offset} reads local offset {value}, not earlier"
                    )


@dataclass(frozen=True, slots=True)
class CommitmentRef:
    """One commitment an obligation opens against: its owner, domain and root."""

    owner: int
    domain_id: bytes
    root: bytes
    count: int

    def __post_init__(self) -> None:
        if type(self.owner) is not int or not -2 <= self.owner < _U64 - 2:
            raise ProtocolError(
                "commitment owner must be -2, -1 or a replay unit index"
            )
        _check_digest(self.domain_id, "domain id")
        _check_digest(self.root, "commitment root")
        _check_index(self.count, "commitment count", _U64)


@dataclass(frozen=True, slots=True)
class PositionRef:
    """One authenticated coordinate: which commitment, at which rank and position.

    ``expected`` pins the value the position must hold (an ``in`` gate holds
    the public input of its rank); it is part of the public statement.
    """

    commitment: int
    rank: int
    position: int
    schema: str
    expected: bytes | None = None

    def __post_init__(self) -> None:
        _check_index(self.commitment, "position commitment")
        _check_index(self.rank, "rank", _U64)
        _check_index(self.position, "position", _U64)
        if type(self.schema) is not str or not self.schema:
            raise ProtocolError("leaf schema must be a nonempty string")
        if self.expected is not None and type(self.expected) is not bytes:
            raise ProtocolError("expected value must be bytes or None")


@dataclass(frozen=True, slots=True)
class PositionRange:
    """``count`` consecutive positions reading words ``start, start+stride, ...`` of one packed leaf.

    Statement format 3 (``veritor/proofs/statement/v3``) admits a range where
    format 2 has one :class:`PositionRef` per word.  Its meaning is *exactly*
    :meth:`expand`: the ``count`` per-word positions ``PositionRef(commitment,
    rank, position, f"{descriptor}:{start + i*stride}")`` for ``i`` in
    ``0..count``, unpinned, in that order -- so operand order, aliases (the
    same word named by two entries) and domain qualification are those of the
    expansion.  Rules, checked here and in the guest with checked arithmetic:
    ``descriptor`` is a well-formed ``packed/v1`` leaf descriptor (no index),
    ``count >= 1``, ``stride >= 1``, every field below ``2^64``, and the last
    index ``start + (count-1)*stride`` is below the leaf's word count.

    A range carries no pin: a sampled ``Input`` whose word is public is a
    single :class:`PositionRef` with ``expected`` (the lowering never merges a
    pinned position into a range), so "expected on a range" is not a thing the
    format can say rather than a case to define.
    """

    commitment: int
    rank: int
    position: int
    descriptor: str
    start: int
    count: int
    stride: int

    def __post_init__(self) -> None:
        from .schema import parse_leaf_descriptor

        _check_index(self.commitment, "range commitment")
        _check_index(self.rank, "rank", _U64)
        _check_index(self.position, "position", _U64)
        if type(self.descriptor) is not str or parse_leaf_descriptor(self.descriptor) is None:
            raise ProtocolError(f"range descriptor {self.descriptor!r} is not a packed/v1 leaf descriptor")
        _check_index(self.start, "range start", _U64)
        _check_index(self.count, "range count", _U64)
        _check_index(self.stride, "range stride", _U64)
        if self.count < 1 or self.stride < 1:
            raise ProtocolError("a range has at least one word and a stride of at least one")
        last = self.start + (self.count - 1) * self.stride
        if last >= _U64:
            raise ProtocolError("range arithmetic overflows")
        words = self.leaf_words
        if last >= words:
            raise ProtocolError(f"range reaches word {last} of a leaf of {words} words")

    @property
    def leaf_words(self) -> int:
        from .schema import parse_leaf_descriptor

        descriptor = parse_leaf_descriptor(self.descriptor)
        assert descriptor is not None
        return descriptor.count

    def indices(self) -> range:
        """The word indices this range stands for, in order."""

        return range(self.start, self.start + self.count * self.stride, self.stride)

    def expand(self) -> tuple[PositionRef, ...]:
        """The per-word positions this range abbreviates (the definition of its meaning)."""

        return tuple(
            PositionRef(self.commitment, self.rank, self.position, f"{self.descriptor}:{index}")
            for index in self.indices()
        )


type PositionEntry = PositionRef | PositionRange
"""One entry of a format-3 obligation's position list: a word, or a range of words."""


@dataclass(frozen=True, slots=True)
class Obligation:
    """The public statement for one sampled proof unit.

    ``session`` is the header digest of the run (it binds the session id, the
    compiled circuit, the policy, the public I/O and ``kappa_W``);
    ``compiled`` repeats ``H(C, I)`` explicitly.  ``unit`` is the proof unit's index and
    ``replay_unit`` the RU it lies in.  ``positions`` are every coordinate the
    relation touches -- the copy's inputs and its opened outputs, nothing
    else -- in ascending address order; ``inputs[k]`` is the slot of the
    ``k``-th read port of the kind and ``outputs[m]`` the slot of the copy's
    gate at the kind's ``m``-th opened offset (:attr:`KindProgram.outputs`).

    In statement format 3 an entry of ``positions`` may be a
    :class:`PositionRange`; slots then index the *expanded* position list
    (:meth:`slot_count`), and :meth:`expand` gives the format-2 obligation the
    entries stand for.  Format-2 encoding and every witness-side table
    (``opened_trees``, ``packed_table``) take expanded obligations only.
    """

    session: bytes
    compiled: bytes
    unit: int
    replay_unit: int
    kind: bytes
    commitments: tuple[CommitmentRef, ...]
    positions: tuple[PositionEntry, ...]
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]

    def __post_init__(self) -> None:
        _check_digest(self.session, "session digest")
        _check_digest(self.compiled, "compiled digest")
        _check_index(self.unit, "unit", _U64)
        _check_index(self.replay_unit, "replay unit", _U64)
        _check_digest(self.kind, "kind digest")
        if type(self.commitments) is not tuple or not all(
            isinstance(item, CommitmentRef) for item in self.commitments
        ):
            raise ProtocolError("obligation commitments must be CommitmentRef values")
        if type(self.positions) is not tuple or not all(
            isinstance(item, (PositionRef, PositionRange)) for item in self.positions
        ):
            raise ProtocolError("obligation positions must be PositionRef or PositionRange values")
        slots_total = self.slot_count
        if len(self.commitments) >= _U32 or len(self.positions) >= _U32 or slots_total >= _U32:
            raise ProtocolError("obligation is too large")
        for item in self.positions:
            if item.commitment >= len(self.commitments):
                raise ProtocolError("position names a commitment the obligation lacks")
        for name, slots in (("inputs", self.inputs), ("outputs", self.outputs)):
            if type(slots) is not tuple or any(
                type(slot) is not int or not 0 <= slot < slots_total for slot in slots
            ):
                raise ProtocolError(f"obligation {name} must index its positions")
        if len(set(self.outputs)) != len(self.outputs):
            raise ProtocolError("obligation outputs must be distinct positions")

    @property
    def key(self) -> tuple[bytes, bytes, int]:
        """The sort key of the canonical batch order."""

        return (self.session, self.compiled, self.unit)

    @property
    def slot_count(self) -> int:
        """The number of per-word positions: ``len(positions)`` once every range is expanded."""

        return sum(item.count if isinstance(item, PositionRange) else 1 for item in self.positions)

    @property
    def has_ranges(self) -> bool:
        return any(isinstance(item, PositionRange) for item in self.positions)

    def expand(self) -> Obligation:
        """This obligation with every :class:`PositionRange` replaced by its per-word positions."""

        if not self.has_ranges:
            return self
        expanded: list[PositionRef] = []
        for item in self.positions:
            if isinstance(item, PositionRange):
                expanded.extend(item.expand())
            else:
                expanded.append(item)
        return Obligation(
            self.session,
            self.compiled,
            self.unit,
            self.replay_unit,
            self.kind,
            self.commitments,
            tuple(expanded),
            self.inputs,
            self.outputs,
        )

    def check_program(self, program: KindProgram) -> None:
        """Raise unless this obligation is shaped for ``program``."""

        if program.kind != self.kind:
            raise ProtocolError("obligation and program name different kinds")
        if len(self.inputs) != len(program.ports):
            raise ProtocolError(
                f"obligation binds {len(self.inputs)} inputs but its kind reads "
                f"{len(program.ports)} ports"
            )
        if len(self.outputs) != len(program.outputs):
            raise ProtocolError(
                f"obligation binds {len(self.outputs)} outputs but its kind opens "
                f"{len(program.outputs)}"
            )


@dataclass(frozen=True, slots=True)
class Statement:
    """One batch: a gate set, the programs of its kinds and its obligations.

    Kinds are sorted by digest and obligations by :attr:`Obligation.key`, so
    a batch is a set and its encoding is unique.  Use :func:`make_statement`
    to build one from unordered inputs.
    """

    gate_set_id: str
    gate_set_digest: bytes
    width: int
    kinds: tuple[KindProgram, ...]
    obligations: tuple[Obligation, ...]

    def __post_init__(self) -> None:
        if type(self.gate_set_id) is not str or not self.gate_set_id:
            raise ProtocolError("gate set id must be a nonempty string")
        _check_digest(self.gate_set_digest, "gate set digest")
        _check_index(self.width, "width")
        if type(self.kinds) is not tuple or not all(
            isinstance(item, KindProgram) for item in self.kinds
        ):
            raise ProtocolError("statement kinds must be KindProgram values")
        digests = [item.kind for item in self.kinds]
        if sorted(set(digests)) != digests:
            raise ProtocolError("statement kinds must be strictly increasing by digest")
        if type(self.obligations) is not tuple or not all(
            isinstance(item, Obligation) for item in self.obligations
        ):
            raise ProtocolError("statement obligations must be Obligation values")
        keys = [item.key for item in self.obligations]
        if sorted(set(keys)) != keys:
            raise ProtocolError("statement obligations must be distinct and sorted")
        programs = {item.kind: item for item in self.kinds}
        for obligation in self.obligations:
            try:
                program = programs[obligation.kind]
            except KeyError:
                raise ProtocolError(
                    "obligation names a kind the statement lacks"
                ) from None
            obligation.check_program(program)

    def program(self, kind: bytes) -> KindProgram:
        for item in self.kinds:
            if item.kind == kind:
                return item
        raise KeyError(kind.hex())

    @property
    def has_ranges(self) -> bool:
        """Whether any obligation lists a :class:`PositionRange` (a format-3-only statement)."""

        return any(obligation.has_ranges for obligation in self.obligations)

    def expand(self) -> Statement:
        """The format-2 statement this one stands for: every range replaced by its words.

        The identity on a statement without ranges (the same object is
        returned).  Sort order and kinds are unchanged, so ``expand`` commutes
        with ``make_statement``.
        """

        if not self.has_ranges:
            return self
        return Statement(
            self.gate_set_id,
            self.gate_set_digest,
            self.width,
            self.kinds,
            tuple(obligation.expand() for obligation in self.obligations),
        )


def expand_statement(statement: Statement) -> Statement:
    """:meth:`Statement.expand` as a function: the one definition of what a range means."""

    return statement.expand()


def require_expanded(statement: Statement, what: str) -> None:
    """``ProtocolError`` if ``statement`` still holds ranges: ``what`` takes per-word positions only."""

    if statement.has_ranges:
        raise ProtocolError(f"{what} takes an expanded statement; expand the format-3 ranges first")


def make_statement(
    gate_set_id: str,
    gate_set_digest: bytes,
    width: int,
    kinds: Iterable[KindProgram],
    obligations: Iterable[Obligation],
) -> Statement:
    """Build the canonical statement of an unordered batch.

    Duplicate kinds must be identical programs; duplicate obligations are an
    error (a batch is a set of demands).
    """

    programs: dict[bytes, KindProgram] = {}
    for program in kinds:
        previous = programs.setdefault(program.kind, program)
        if previous != program:
            raise ProtocolError(
                f"kind {program.kind.hex()[:12]} has two different programs"
            )
    ordered = sorted(obligations, key=lambda item: item.key)
    return Statement(
        gate_set_id,
        gate_set_digest,
        width,
        tuple(programs[kind] for kind in sorted(programs)),
        tuple(ordered),
    )


@dataclass(frozen=True, slots=True)
class Witness:
    """The secret side of a statement: per obligation, per position, the opened value and path."""

    obligations: tuple[tuple[tuple[bytes, tuple[bytes, ...]], ...], ...]

    def __post_init__(self) -> None:
        if type(self.obligations) is not tuple:
            raise ProtocolError("witness must be a tuple of per-obligation openings")
        for openings in self.obligations:
            if type(openings) is not tuple:
                raise ProtocolError("witness openings must be tuples")
            for item in openings:
                if (
                    type(item) is not tuple
                    or len(item) != 2
                    or type(item[0]) is not bytes
                    or type(item[1]) is not tuple
                    or any(
                        type(digest) is not bytes or len(digest) != 32
                        for digest in item[1]
                    )
                ):
                    raise ProtocolError(
                        "a witness opening is (value, path of 32-byte digests)"
                    )

    def for_statement(self, statement: Statement) -> None:
        """Raise unless the witness has the statement's shape."""

        require_expanded(statement, "a per-position witness")
        if len(self.obligations) != len(statement.obligations):
            raise ProtocolError(
                f"witness covers {len(self.obligations)} obligations, the statement "
                f"{len(statement.obligations)}"
            )
        for obligation, openings in zip(
            statement.obligations, self.obligations, strict=True
        ):
            if len(openings) != len(obligation.positions):
                raise ProtocolError("witness opens a different number of positions")


def select(
    statement: Statement, witness: Witness, obligations: Sequence[Obligation]
) -> Witness:
    """The sub-witness for ``obligations`` (a subset of the statement's), in canonical order."""

    lookup = {
        item.key: openings
        for item, openings in zip(
            statement.obligations, witness.obligations, strict=True
        )
    }
    return Witness(
        tuple(lookup[item.key] for item in sorted(obligations, key=lambda o: o.key))
    )


# -- the multiproof view of a statement --------------------------------------------


@dataclass(frozen=True, slots=True)
class OpenedTree:
    """One commitment of a statement and every leaf the batch opens in it.

    ``ranks`` are ascending and distinct; ``positions[k]`` and ``schemas[k]``
    are the leaf coordinates of ``ranks[k]`` (a rank has one position and one
    schema in a domain, whichever obligations open it).
    """

    domain_id: bytes
    root: bytes
    count: int
    ranks: tuple[int, ...]
    positions: tuple[int, ...]
    schemas: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TreeTable:
    """The trees of a statement in canonical order and where every position lands.

    Trees are numbered by first appearance of their ``(domain_id, root,
    count)`` while scanning the obligations in statement order and each
    obligation's ``commitments`` in order.  ``slots[i][j]`` is ``(tree,
    leaf)`` for position ``j`` of obligation ``i``: its value is the
    ``leaf``-th opened value of ``trees[tree]``.
    """

    trees: tuple[OpenedTree, ...]
    slots: tuple[tuple[tuple[int, int], ...], ...]


def opened_trees(statement: Statement) -> TreeTable:
    """Group the positions of ``statement`` by the tree they open; both parties derive it."""

    require_expanded(statement, "opened_trees")
    order: dict[tuple[bytes, bytes, int], int] = {}
    refs: list[CommitmentRef] = []
    opened: list[dict[int, tuple[int, str]]] = []
    coordinates: list[list[tuple[int, int]]] = []
    for obligation in statement.obligations:
        trees_of: list[int] = []
        for ref in obligation.commitments:
            key = (ref.domain_id, ref.root, ref.count)
            index = order.setdefault(key, len(refs))
            if index == len(refs):
                refs.append(ref)
                opened.append({})
            trees_of.append(index)
        coordinates.append([])
        for position in obligation.positions:
            tree = trees_of[position.commitment]
            leaf = (position.position, position.schema)
            if opened[tree].setdefault(position.rank, leaf) != leaf:
                raise ProtocolError(
                    f"the statement opens rank {position.rank} of one tree at two coordinates"
                )
            coordinates[-1].append((tree, position.rank))
    trees = tuple(
        OpenedTree(
            ref.domain_id,
            ref.root,
            ref.count,
            tuple(sorted(leaves)),
            tuple(leaves[rank][0] for rank in sorted(leaves)),
            tuple(leaves[rank][1] for rank in sorted(leaves)),
        )
        for ref, leaves in zip(refs, opened, strict=True)
    )
    leaf_index = [{rank: k for k, rank in enumerate(tree.ranks)} for tree in trees]
    return TreeTable(
        trees,
        tuple(
            tuple((tree, leaf_index[tree][rank]) for tree, rank in row)
            for row in coordinates
        ),
    )


@dataclass(frozen=True, slots=True)
class MultiproofWitness:
    """The secret side of a statement, one multiproof per tree (witness format 2).

    ``trees[t]`` is ``(values, siblings)`` for the ``t``-th tree of
    :func:`opened_trees`: the opened values in ascending rank order, one per
    distinct opened rank, and the multiproof of :mod:`veritor.protocol.multiproof`
    (empty for a tree the batch opens nothing in).
    """

    trees: tuple[tuple[tuple[bytes, ...], tuple[bytes, ...]], ...]

    def __post_init__(self) -> None:
        if type(self.trees) is not tuple:
            raise ProtocolError("a multiproof witness is a tuple of per-tree entries")
        for item in self.trees:
            if (
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not tuple
                or type(item[1]) is not tuple
                or any(type(value) is not bytes for value in item[0])
                or any(
                    type(digest) is not bytes or len(digest) != 32 for digest in item[1]
                )
            ):
                raise ProtocolError(
                    "a multiproof witness entry is (values, siblings of 32-byte digests)"
                )

    def for_statement(self, statement: Statement) -> TreeTable:
        """Raise unless the witness has the statement's shape; return the statement's trees."""

        table = opened_trees(statement)
        if len(self.trees) != len(table.trees):
            raise ProtocolError(
                f"witness covers {len(self.trees)} trees, the statement {len(table.trees)}"
            )
        for tree, (values, _siblings) in zip(table.trees, self.trees, strict=True):
            if len(values) != len(tree.ranks):
                raise ProtocolError("witness opens a different number of leaves")
        return table


def multiproof_witness(statement: Statement, witness: Witness) -> MultiproofWitness:
    """Fold a per-opening witness into the multiproof format of ``statement``.

    Positions that open the same leaf must carry the same value, and the
    paths of one tree must agree on every shared node: a witness some tree
    produced does; any other is rejected here rather than encoded.
    """

    witness.for_statement(statement)
    table = opened_trees(statement)
    values: list[dict[int, bytes]] = [{} for _ in table.trees]
    paths: list[dict[int, tuple[bytes, ...]]] = [{} for _ in table.trees]
    for openings, row in zip(witness.obligations, table.slots, strict=True):
        for (value, path), (tree, leaf) in zip(openings, row, strict=True):
            rank = table.trees[tree].ranks[leaf]
            if (
                values[tree].setdefault(rank, value) != value
                or paths[tree].setdefault(rank, path) != path
            ):
                raise ProtocolError(
                    f"the witness opens rank {rank} of one tree with two openings"
                )
    return MultiproofWitness(
        tuple(
            (
                tuple(values[index][rank] for rank in tree.ranks),
                tuple(multiproof_from_paths(tree.count, paths[index]))
                if tree.ranks
                else (),
            )
            for index, tree in enumerate(table.trees)
        )
    )
