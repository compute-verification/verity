"""Canonical bytes for statements and witnesses.

The batch statement is what a proof is *about*: a guest program commits
``sha256(encode_statement(statement))`` as a public value, so both parties
must produce the same bytes from the same demands.  The encoding is a strict,
length-prefixed binary framing (big-endian integers; ``u32`` length prefixes
for byte strings and lists; raw 32-byte digests) that a zkVM guest parses in a
few cycles per field, and it is *canonical*: :func:`decode_statement` accepts
exactly the outputs of :func:`encode_statement` (every index range-checked,
kinds and obligations sorted, no trailing bytes) and re-encoding a decoded
statement reproduces the input bytes.

The Rust mirror is ``backends/sp1/common/src/codec.rs``; the layout is::

    Statement   = MAGIC str gate_set_id digest gate_set_digest u32 width
                  list<KindProgram> kinds  list<Obligation> obligations
    KindProgram = digest kind u32 size list<u32> ports list<GateOp> gates list<u32> outputs
    GateOp      = str op list<Arg> args ;  Arg = u8 space (0 port, 1 local) u32 value
    Obligation  = digest session digest compiled u64 unit u64 replay_unit digest kind
                  list<CommitmentRef> list<PositionRef> list<u32> inputs list<u32> outputs
    CommitmentRef = u64 owner+2 digest domain_id digest root u64 count
    PositionRef   = u32 commitment u64 rank u64 position str schema
                    u8 has_expected [bytes expected]
    Witness     = MAGIC list< list< bytes value list<digest> path > >

Statement format 3 (:data:`STATEMENT_V3_MAGIC`) is format 2 with a tag byte in
front of every position entry, so that an entry may be a **range** of words of
one packed leaf (:class:`~veritor.protocol.proofs.statement.PositionRange`)::

    StatementV3   = MAGIC3 ... list<ObligationV3> obligations          (else as Statement)
    ObligationV3  = ... list<PositionEntry> ...                       (else as Obligation)
    PositionEntry = u8 0 PositionRef
                  | u8 1 u32 commitment u64 rank u64 position str descriptor
                         u64 start u64 count u64 stride

A range means exactly its expansion into ``count`` format-2 positions with
schemas ``<descriptor>:<start + i*stride>`` (``Statement.expand``); a format-3
statement's slots index that expansion.  Format 2 is unchanged and a format-2
statement never holds a range (:func:`encode_statement` refuses one); the two
formats have different bytes and therefore different digests for the same
relation.

A witness has two formats, told apart by their magic; the statement and its
digest are the same under both.  Format 1 (:data:`WITNESS_MAGIC`) carries one
opening -- value and full authentication path -- per obligation position.
Format 2 (:data:`MULTIPROOF_WITNESS_MAGIC`) carries, per tree of the
statement in the order of :func:`~veritor.protocol.proofs.statement.opened_trees`,
the opened values once (ascending rank) and one Merkle multiproof
(:mod:`veritor.protocol.multiproof`) for all of them::

    MultiproofWitness = MAGIC list< list<bytes> values  list<digest> siblings >

Format 3 (:data:`PACKED_WITNESS_MAGIC`, :mod:`veritor.protocol.proofs.packed`)
carries, per tree in the same order, whole *packed* leaves under the serving
side's framing with one authentication path each; every position's schema is
``packed/v1:...`` and names the word of its leaf::

    PackedWitness = MAGIC list< list< str id str dtype str shape bytes value list<PathStep> > >
    PathStep      = u8 kind (0 = lift, 1 = sibling) [digest]
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from verity.errors import ProtocolError

from .statement import (
    LOCAL,
    PORT,
    Arg,
    ArgSpace,
    CommitmentRef,
    GateOp,
    KindProgram,
    MultiproofWitness,
    Obligation,
    PositionEntry,
    PositionRange,
    PositionRef,
    Statement,
    Witness,
    multiproof_witness,
    opened_trees,
)

STATEMENT_MAGIC = b"veritor/proofs/statement/v2\0"
"""v2: obligations open a unit's inputs and outputs and the checker recomputes its gates."""
STATEMENT_V3_MAGIC = b"veritor/proofs/statement/v3\0"
"""v3: v2 whose position entries are tagged and may be ranges of words of one packed leaf."""
_ENTRY_SINGLE = 0
_ENTRY_RANGE = 1
WITNESS_MAGIC = b"veritor/proofs/witness/v2\0"
"""Witness format 1: one opening (value, full path) per obligation position."""
MULTIPROOF_WITNESS_MAGIC = b"veritor/proofs/witness/multiproof/v1\0"
"""Witness format 2: per tree, the opened values once and one multiproof."""
PACKED_WITNESS_MAGIC = b"veritor/proofs/witness/packed/v1\0"
"""Witness format 3: per tree, whole packed leaves under the serving side's framing, one path each.

Not one of :data:`WITNESS_FORMATS`: a format-3 witness is not an encoding of a
:class:`Witness` (it carries leaves the statement's positions read words of)
and is produced by the serving side's exporter, never by ``prove``."""
WITNESS_FORMATS = (1, 2)
DEFAULT_WITNESS_FORMAT = 1

_SPACE_CODE: dict[ArgSpace, int] = {PORT: 0, LOCAL: 1}
_SPACE_NAME: dict[int, ArgSpace] = {0: PORT, 1: LOCAL}


class _Writer:
    __slots__ = ("_parts",)

    def __init__(self) -> None:
        self._parts: list[bytes] = []

    def raw(self, data: bytes) -> None:
        self._parts.append(data)

    def u8(self, value: int) -> None:
        self._parts.append(value.to_bytes(1, "big"))

    def u32(self, value: int) -> None:
        self._parts.append(value.to_bytes(4, "big"))

    def u64(self, value: int) -> None:
        self._parts.append(value.to_bytes(8, "big"))

    def digest(self, value: bytes) -> None:
        if len(value) != 32:
            raise ProtocolError("digests are 32 bytes")
        self._parts.append(value)

    def blob(self, value: bytes) -> None:
        self.u32(len(value))
        self._parts.append(value)

    def string(self, value: str) -> None:
        self.blob(value.encode("utf-8"))

    def getvalue(self) -> bytes:
        return b"".join(self._parts)


class _Reader:
    __slots__ = ("_data", "_offset")

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0

    def take(self, count: int, what: str) -> bytes:
        end = self._offset + count
        if end > len(self._data):
            raise ProtocolError(f"proof input truncated while reading {what}")
        chunk = self._data[self._offset : end]
        self._offset = end
        return chunk

    def magic(self, expected: bytes) -> None:
        if self.take(len(expected), "magic") != expected:
            raise ProtocolError("proof input has the wrong magic")

    def u8(self, what: str) -> int:
        return self.take(1, what)[0]

    def u32(self, what: str) -> int:
        return int.from_bytes(self.take(4, what), "big")

    def u64(self, what: str) -> int:
        return int.from_bytes(self.take(8, what), "big")

    def digest(self, what: str) -> bytes:
        return self.take(32, what)

    def blob(self, what: str) -> bytes:
        return self.take(self.u32(what), what)

    def string(self, what: str) -> str:
        try:
            return self.blob(what).decode("utf-8")
        except UnicodeDecodeError:
            raise ProtocolError(f"{what} is not UTF-8") from None

    def count(self, what: str) -> int:
        count = self.u32(what)
        if count > len(self._data) - self._offset:
            raise ProtocolError(f"{what} count {count} exceeds the remaining bytes")
        return count

    def finish(self, what: str) -> None:
        if self._offset != len(self._data):
            raise ProtocolError(
                f"{len(self._data) - self._offset} trailing bytes after the {what}"
            )


# -- statements ---------------------------------------------------------------


def _write_program(writer: _Writer, program: KindProgram) -> None:
    writer.digest(program.kind)
    writer.u32(program.size)
    writer.u32(len(program.ports))
    for ordinal in program.ports:
        writer.u32(ordinal)
    writer.u32(len(program.gates))
    for gate in program.gates:
        writer.string(gate.op)
        writer.u32(len(gate.args))
        for space, value in gate.args:
            writer.u8(_SPACE_CODE[space])
            writer.u32(value)
    writer.u32(len(program.outputs))
    for offset in program.outputs:
        writer.u32(offset)


def _write_position(writer: _Writer, position: PositionRef) -> None:
    writer.u32(position.commitment)
    writer.u64(position.rank)
    writer.u64(position.position)
    writer.string(position.schema)
    if position.expected is None:
        writer.u8(0)
    else:
        writer.u8(1)
        writer.blob(position.expected)


def _write_obligation(writer: _Writer, obligation: Obligation, *, ranged: bool = False) -> None:
    writer.digest(obligation.session)
    writer.digest(obligation.compiled)
    writer.u64(obligation.unit)
    writer.u64(obligation.replay_unit)
    writer.digest(obligation.kind)
    writer.u32(len(obligation.commitments))
    for commitment in obligation.commitments:
        writer.u64(commitment.owner + 2)
        writer.digest(commitment.domain_id)
        writer.digest(commitment.root)
        writer.u64(commitment.count)
    writer.u32(len(obligation.positions))
    for position in obligation.positions:
        if isinstance(position, PositionRange):
            if not ranged:
                raise ProtocolError(
                    "a format-2 statement has no ranges: encode it as format 3 (encode_statement_v3) "
                    "or expand it first (Statement.expand)"
                )
            writer.u8(_ENTRY_RANGE)
            writer.u32(position.commitment)
            writer.u64(position.rank)
            writer.u64(position.position)
            writer.string(position.descriptor)
            writer.u64(position.start)
            writer.u64(position.count)
            writer.u64(position.stride)
        else:
            if ranged:
                writer.u8(_ENTRY_SINGLE)
            _write_position(writer, position)
    for slots in (obligation.inputs, obligation.outputs):
        writer.u32(len(slots))
        for slot in slots:
            writer.u32(slot)


def encode_obligation(obligation: Obligation) -> bytes:
    """The canonical bytes of one obligation (a component of the statement)."""

    writer = _Writer()
    _write_obligation(writer, obligation)
    return writer.getvalue()


def _encode_statement(statement: Statement, magic: bytes, *, ranged: bool) -> bytes:
    writer = _Writer()
    writer.raw(magic)
    writer.string(statement.gate_set_id)
    writer.digest(statement.gate_set_digest)
    writer.u32(statement.width)
    writer.u32(len(statement.kinds))
    for program in statement.kinds:
        _write_program(writer, program)
    writer.u32(len(statement.obligations))
    for obligation in statement.obligations:
        _write_obligation(writer, obligation, ranged=ranged)
    return writer.getvalue()


def encode_statement(statement: Statement) -> bytes:
    """The canonical format-2 bytes of a batch statement (``ProtocolError`` if it holds ranges)."""

    if not isinstance(statement, Statement):
        raise ProtocolError("encode_statement requires a Statement")
    return _encode_statement(statement, STATEMENT_MAGIC, ranged=False)


def encode_statement_v3(statement: Statement) -> bytes:
    """The canonical format-3 bytes of a batch statement: tagged entries, ranges allowed.

    Every statement has format-3 bytes, ranges or not; a statement without
    ranges has both formats and they differ (magic and tags), so their digests
    differ too.
    """

    if not isinstance(statement, Statement):
        raise ProtocolError("encode_statement_v3 requires a Statement")
    return _encode_statement(statement, STATEMENT_V3_MAGIC, ranged=True)


def statement_digest(statement: Statement | bytes) -> bytes:
    """``sha256`` of the canonical statement bytes: the public value a proof commits."""

    data = statement if isinstance(statement, bytes) else encode_statement(statement)
    return hashlib.sha256(data).digest()


def _read_program(reader: _Reader) -> KindProgram:
    kind = reader.digest("kind digest")
    size = reader.u32("kind size")
    ports = tuple(reader.u32("port ordinal") for _ in range(reader.count("ports")))
    gate_count = reader.count("gates")
    gates: list[GateOp] = []
    for _ in range(gate_count):
        op = reader.string("gate op")
        args: list[Arg] = []
        for _ in range(reader.count("gate args")):
            space = reader.u8("arg space")
            value = reader.u32("arg value")
            if space not in _SPACE_NAME:
                raise ProtocolError(f"unknown argument space {space}")
            args.append((_SPACE_NAME[space], value))
        gates.append(GateOp(op, tuple(args)))
    outputs = tuple(reader.u32("output offset") for _ in range(reader.count("outputs")))
    return KindProgram(kind, size, ports, tuple(gates), outputs)


def _read_position(reader: _Reader) -> PositionRef:
    commitment = reader.u32("position commitment")
    rank = reader.u64("rank")
    position = reader.u64("position")
    schema = reader.string("schema")
    flag = reader.u8("expected flag")
    if flag == 0:
        expected = None
    elif flag == 1:
        expected = reader.blob("expected value")
    else:
        raise ProtocolError(f"bad expected flag {flag}")
    return PositionRef(commitment, rank, position, schema, expected)


def _read_obligation(reader: _Reader, *, ranged: bool = False) -> Obligation:
    session = reader.digest("session")
    compiled = reader.digest("compiled digest")
    unit = reader.u64("unit")
    replay_unit = reader.u64("replay unit")
    kind = reader.digest("obligation kind")
    commitments: list[CommitmentRef] = []
    for _ in range(reader.count("commitments")):
        owner_plus_two = reader.u64("owner")
        commitments.append(
            CommitmentRef(
                owner_plus_two - 2,
                reader.digest("domain id"),
                reader.digest("root"),
                reader.u64("commitment count"),
            )
        )
    positions: list[PositionEntry] = []
    for _ in range(reader.count("positions")):
        tag = reader.u8("position entry tag") if ranged else _ENTRY_SINGLE
        if tag == _ENTRY_SINGLE:
            positions.append(_read_position(reader))
        elif tag == _ENTRY_RANGE:
            positions.append(
                PositionRange(
                    reader.u32("range commitment"),
                    reader.u64("range rank"),
                    reader.u64("range position"),
                    reader.string("range descriptor"),
                    reader.u64("range start"),
                    reader.u64("range count"),
                    reader.u64("range stride"),
                )
            )
        else:
            raise ProtocolError(f"bad position entry tag {tag}")
    inputs = tuple(reader.u32("inputs") for _ in range(reader.count("inputs")))
    outputs = tuple(reader.u32("outputs") for _ in range(reader.count("outputs")))
    return Obligation(
        session,
        compiled,
        unit,
        replay_unit,
        kind,
        tuple(commitments),
        tuple(positions),
        inputs,
        outputs,
    )


def _decode_statement(data: bytes, magic: bytes, *, ranged: bool) -> Statement:
    reader = _Reader(data)
    reader.magic(magic)
    gate_set_id = reader.string("gate set id")
    gate_set_digest = reader.digest("gate set digest")
    width = reader.u32("width")
    kinds = tuple(_read_program(reader) for _ in range(reader.count("kinds")))
    obligations = tuple(
        _read_obligation(reader, ranged=ranged) for _ in range(reader.count("obligations"))
    )
    reader.finish("statement")
    statement = Statement(gate_set_id, gate_set_digest, width, kinds, obligations)
    if _encode_statement(statement, magic, ranged=ranged) != data:
        raise ProtocolError("statement bytes are not canonical")
    return statement


def decode_statement(data: bytes) -> Statement:
    """Parse canonical format-2 statement bytes strictly; ``encode_statement`` inverts it."""

    if type(data) is not bytes:
        raise ProtocolError("decode_statement requires bytes")
    return _decode_statement(data, STATEMENT_MAGIC, ranged=False)


def decode_statement_v3(data: bytes) -> Statement:
    """Parse canonical format-3 statement bytes strictly; ``encode_statement_v3`` inverts it.

    Every range is validated as it is read (descriptor grammar, ``count``,
    ``stride``, overflow, bounds against the leaf's word count); the statement
    comes back *unexpanded* -- ``Statement.expand`` gives its format-2 meaning.
    """

    if type(data) is not bytes:
        raise ProtocolError("decode_statement_v3 requires bytes")
    return _decode_statement(data, STATEMENT_V3_MAGIC, ranged=True)


def statement_format_of(data: bytes) -> int:
    """``2`` or ``3`` by the magic of ``data``; ``ProtocolError`` for anything else."""

    if type(data) is bytes:
        if data.startswith(STATEMENT_MAGIC):
            return 2
        if data.startswith(STATEMENT_V3_MAGIC):
            return 3
    raise ProtocolError("not a statement of a known format")


def decode_any_statement(data: bytes) -> Statement:
    """Decode format-2 or format-3 statement bytes by their magic (format 3 stays unexpanded)."""

    return decode_statement(data) if statement_format_of(data) == 2 else decode_statement_v3(data)


# -- witnesses ----------------------------------------------------------------


def encode_witness(witness: Witness) -> bytes:
    """The canonical bytes of a witness (the guest's private input)."""

    if not isinstance(witness, Witness):
        raise ProtocolError("encode_witness requires a Witness")
    writer = _Writer()
    writer.raw(WITNESS_MAGIC)
    writer.u32(len(witness.obligations))
    for openings in witness.obligations:
        writer.u32(len(openings))
        for value, path in openings:
            writer.blob(value)
            writer.u32(len(path))
            for digest in path:
                writer.digest(digest)
    return writer.getvalue()


def decode_witness(data: bytes) -> Witness:
    """Parse canonical witness bytes strictly."""

    if type(data) is not bytes:
        raise ProtocolError("decode_witness requires bytes")
    reader = _Reader(data)
    reader.magic(WITNESS_MAGIC)
    obligations: list[tuple[tuple[bytes, tuple[bytes, ...]], ...]] = []
    for _ in range(reader.count("witness obligations")):
        openings: list[tuple[bytes, tuple[bytes, ...]]] = []
        for _ in range(reader.count("openings")):
            value = reader.blob("opening value")
            path = tuple(
                reader.digest("path digest") for _ in range(reader.count("path"))
            )
            openings.append((value, path))
        obligations.append(tuple(openings))
    reader.finish("witness")
    return Witness(tuple(obligations))


def encode_multiproof_witness(witness: MultiproofWitness) -> bytes:
    """The canonical bytes of a format-2 witness."""

    if not isinstance(witness, MultiproofWitness):
        raise ProtocolError("encode_multiproof_witness requires a MultiproofWitness")
    writer = _Writer()
    writer.raw(MULTIPROOF_WITNESS_MAGIC)
    writer.u32(len(witness.trees))
    for values, siblings in witness.trees:
        writer.u32(len(values))
        for value in values:
            writer.blob(value)
        writer.u32(len(siblings))
        for digest in siblings:
            writer.digest(digest)
    return writer.getvalue()


def decode_multiproof_witness(data: bytes) -> MultiproofWitness:
    """Parse canonical format-2 witness bytes strictly."""

    if type(data) is not bytes:
        raise ProtocolError("decode_multiproof_witness requires bytes")
    reader = _Reader(data)
    reader.magic(MULTIPROOF_WITNESS_MAGIC)
    trees: list[tuple[tuple[bytes, ...], tuple[bytes, ...]]] = []
    for _ in range(reader.count("witness trees")):
        values = tuple(
            reader.blob("opened value") for _ in range(reader.count("opened values"))
        )
        siblings = tuple(
            reader.digest("sibling digest") for _ in range(reader.count("siblings"))
        )
        trees.append((values, siblings))
    reader.finish("witness")
    return MultiproofWitness(tuple(trees))


PLAN_MAGIC = b"veritor/proofs/plan/tree-table/v1\0"
"""An untrusted execution plan: the statement's tree table, spelled out for the guest to validate."""


def encode_plan(statement: Statement) -> bytes:
    """The honest plan for ``statement``: its canonical tree table in the guest's flat layout.

    Per tree (in :func:`~veritor.protocol.proofs.statement.opened_trees` order)
    the number of distinct opened leaves; then, for every obligation in order
    and every commitment it lists, the tree index; then, for every obligation
    in order and every position, the index of its leaf within its tree.  The
    guest (``check::validate_plan``) accepts exactly this table for the
    statement and rejects every other plan, so the encoding is advice, not
    trust: it lets the guest check the table in one linear pass instead of
    sorting and searching for it.
    """

    table = opened_trees(statement)
    order = {
        (tree.domain_id, tree.root, tree.count): index
        for index, tree in enumerate(table.trees)
    }
    writer = _Writer()
    writer.raw(PLAN_MAGIC)
    writer.u32(len(table.trees))
    for tree in table.trees:
        writer.u32(len(tree.ranks))
    tree_of = [
        order[(ref.domain_id, ref.root, ref.count)]
        for obligation in statement.obligations
        for ref in obligation.commitments
    ]
    writer.u32(len(tree_of))
    for index in tree_of:
        writer.u32(index)
    leaf_of = [leaf for row in table.slots for _tree, leaf in row]
    writer.u32(len(leaf_of))
    for index in leaf_of:
        writer.u32(index)
    return writer.getvalue()


def witness_format_of(data: bytes) -> int:
    """Which witness format ``data`` is in, by its magic (1, 2 or 3)."""

    if type(data) is bytes:
        if data.startswith(WITNESS_MAGIC):
            return 1
        if data.startswith(MULTIPROOF_WITNESS_MAGIC):
            return 2
        if data.startswith(PACKED_WITNESS_MAGIC):
            return 3
    raise ProtocolError("proof input has the wrong magic")


def encode_witness_as(
    statement: Statement, witness: Witness, witness_format: int
) -> bytes:
    """The canonical bytes of ``witness`` for ``statement`` in the given format."""

    if witness_format == 1:
        return encode_witness(witness)
    if witness_format == 2:
        return encode_multiproof_witness(multiproof_witness(statement, witness))
    raise ProtocolError(f"unknown witness format {witness_format}")


def encode_obligations(obligations: Sequence[Obligation]) -> bytes:
    """Concatenated canonical obligations, for logging and vectors (not a statement)."""

    return b"".join(encode_obligation(item) for item in obligations)
