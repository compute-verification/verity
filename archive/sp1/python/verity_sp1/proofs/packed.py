"""Packed-word authentication: witness format 3 and the ``packed/v1`` leaf schema.

The theory side's decision of 2026-09-07 (IR §H.4; GUIDANCE §15.5): a logical
value ``gate -> Value<w>`` may live physically inside a committed leaf that
packs a whole row -- ``weight-row leaf 81, word 237``.  Gate identities do not
acquire components; the *authentication-location layer* is extended instead::

    ValueRef           = gate_id                       (the interface's references[].ref)
    CommitmentPosition = (commitment_domain, leaf_id, word_index)

and the guest authenticates the whole packed leaf against the committed root,
then reads the word at the statement-fixed index as the position's supplied
bytes.  Range openings are a later optimisation.

In the legacy statement (``veritor.protocol.proofs``) a ``CommitmentPosition``
is ``PositionRef = (commitment, rank, position, schema)``: the commitment names
the tree (domain, root, count), ``rank`` the leaf, and the **schema** -- until
now ``u<w>``, "one big-endian word under ``veritor/protocol/merkle/frame/v3``"
-- now also admits the packed form

    packed/v1:<framing>:u<w><le|be>:<count>:<index>

meaning: the leaf at ``rank`` is hashed under ``<framing>`` (the only one
implemented is the serving side's ``verity-vllm/leaf/v1``), its value bytes
are exactly ``count`` words of ``w`` bits each, ``ceil(w/8)`` bytes per word,
in the named byte order, and this position is word ``index`` (``0 <= index <
count``).  ``(framing, u<w><order>, count)`` is the **leaf descriptor**: every
position that names one rank must agree on it (and on ``position``); the
index is the position's own.  The statement bytes therefore fix the word: a
prover cannot choose which word of the leaf stands for the value.

The leaf scheme -- ``H``, the leaf/node/lift hashes, the sibling-or-lift path
shape -- is :mod:`verity.ml.commitments.leaves` (the Verity protocol's commitment
to runtime values); this module only reads words out of authenticated leaves.
An authentication path is one step per level: a sibling digest, or ``None``
where the node was lifted; the sibling/lift pattern is a function of
``(leaf_count, index)`` (``path_shape``) and a path of another shape is
rejected before any hash is compared.  The leaf's
``canonical_id``, ``dtype`` and ``shape_text`` are the prover's to supply: the
hash chain binds them to the committed root, and the statement fixes only
what the proof is about -- the tree, the rank, the word layout and the index.

Witness format 3 (``PACKED_WITNESS_MAGIC``)::

    PackedWitness = MAGIC list<PackedTree>       per tree, in the statement's tree order
    PackedTree    = list<PackedLeaf>             one per distinct opened rank, ascending
    PackedLeaf    = str canonical_id  str dtype  str shape_text  bytes value  list<PathStep> path
    PathStep      = u8 kind (0 = lift, 1 = sibling)  [digest sibling]

The tree order is the format-2 one (first appearance of ``(domain_id, root,
count)`` over the obligations and their commitments); a leaf is hashed once
however many positions read words of it.  :func:`check_packed_openings` is the
reference opening pass for this format, mirrored by ``check_openings_packed``
in ``archive/sp1/common/src/kernel/openings.rs``.
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.ml.commitments.leaves import (
    H,
    fold_path,
    ids_digest,
    leaf_hash,
    levels,
    lift_hash,
    node_hash,
    open_path,
    path_shape,
    tree_depth,
)
from verity.errors import ProtocolError

from verity_vllm.check.codes import Reject, VerificationCode

from .schema import (
    PACKED_SCHEMA_PREFIX,
    VLLM_LEAF_FRAMING,
    LeafDescriptor,
    PackedSchema,
    packed_schema,
    parse_packed_schema,
)
from .statement import CommitmentRef, Statement, require_expanded
from .wire import PACKED_WITNESS_MAGIC, _Reader, _Writer

__all__ = [
    "PACKED_SCHEMA_PREFIX",
    "VLLM_LEAF_FRAMING",
    "LeafDescriptor",
    "PackedLeaf",
    "PackedSchema",
    "PackedTable",
    "PackedTree",
    "PackedWitness",
    "check_packed_openings",
    "decode_packed_witness",
    "encode_packed_witness",
    "extract_word",
    "packed_schema",
    "packed_table",
    "parse_packed_schema",
    "vllm_fold_path",
    "vllm_hash",
    "vllm_ids_digest",
    "vllm_leaf_hash",
    "vllm_levels",
    "vllm_lift_hash",
    "vllm_node_hash",
    "vllm_open",
    "vllm_path_shape",
    "vllm_tree_depth",
]

def extract_word(schema: PackedSchema, value: bytes) -> int:
    """Word ``schema.index`` of a leaf's ``value`` bytes, or ``ProtocolError``.

    The leaf must be exactly ``count * ceil(w/8)`` bytes; the word is the
    ``ceil(w/8)``-byte slice at ``index * ceil(w/8)``, read in the schema's
    byte order, and must be below ``2^w``.
    """

    size = schema.word_bytes
    if len(value) != schema.count * size:
        raise ProtocolError(
            f"packed leaf holds {len(value)} bytes, the schema says {schema.count} words of {size}"
        )
    start = schema.index * size
    word = int.from_bytes(value[start : start + size], "little" if schema.little_endian else "big")
    if word >= 1 << schema.width:
        raise ProtocolError(f"word {schema.index} is not a {schema.width}-bit value")
    return word


# --- the leaf scheme (owned by verity.ml.commitments.leaves; the names below are kept for the callers) --------

vllm_hash = H
vllm_leaf_hash = leaf_hash
vllm_node_hash = node_hash
vllm_lift_hash = lift_hash
vllm_path_shape = path_shape
vllm_tree_depth = tree_depth
vllm_fold_path = fold_path
vllm_levels = levels
vllm_open = open_path
vllm_ids_digest = ids_digest


# --- the witness ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PackedLeaf:
    """One opened packed leaf: the parts of its leaf hash and its path (``None`` = a lift)."""

    canonical_id: str
    dtype: str
    shape_text: str
    value: bytes
    path: tuple[bytes | None, ...]

    def __post_init__(self) -> None:
        if type(self.value) is not bytes or type(self.path) is not tuple:
            raise ProtocolError("a packed leaf is (id, dtype, shape, bytes, path)")
        for step in self.path:
            if step is not None and (type(step) is not bytes or len(step) != 32):
                raise ProtocolError("a path step is a 32-byte sibling or None")

    def leaf_hash(self) -> bytes:
        return vllm_leaf_hash(self.canonical_id, self.dtype, self.shape_text, self.value)


@dataclass(frozen=True, slots=True)
class PackedWitness:
    """Format 3: per tree in the statement's tree order, its opened packed leaves by ascending rank."""

    trees: tuple[tuple[PackedLeaf, ...], ...]

    def __post_init__(self) -> None:
        if type(self.trees) is not tuple or any(
            type(tree) is not tuple or any(not isinstance(leaf, PackedLeaf) for leaf in tree)
            for tree in self.trees
        ):
            raise ProtocolError("a packed witness is a tuple of tuples of PackedLeaf")


def encode_packed_witness(witness: PackedWitness) -> bytes:
    """The canonical bytes of a format-3 witness."""

    if not isinstance(witness, PackedWitness):
        raise ProtocolError("encode_packed_witness requires a PackedWitness")
    writer = _Writer()
    writer.raw(PACKED_WITNESS_MAGIC)
    writer.u32(len(witness.trees))
    for tree in witness.trees:
        writer.u32(len(tree))
        for leaf in tree:
            writer.string(leaf.canonical_id)
            writer.string(leaf.dtype)
            writer.string(leaf.shape_text)
            writer.blob(leaf.value)
            writer.u32(len(leaf.path))
            for step in leaf.path:
                if step is None:
                    writer.u8(0)
                else:
                    writer.u8(1)
                    writer.digest(step)
    return writer.getvalue()


def decode_packed_witness(data: bytes) -> PackedWitness:
    """Parse canonical format-3 witness bytes strictly."""

    if type(data) is not bytes:
        raise ProtocolError("decode_packed_witness requires bytes")
    reader = _Reader(data)
    reader.magic(PACKED_WITNESS_MAGIC)
    trees: list[tuple[PackedLeaf, ...]] = []
    for _ in range(reader.count("witness trees")):
        leaves: list[PackedLeaf] = []
        for _ in range(reader.count("packed leaves")):
            canonical_id = reader.string("leaf id")
            dtype = reader.string("leaf dtype")
            shape_text = reader.string("leaf shape")
            value = reader.blob("leaf value")
            path: list[bytes | None] = []
            for _ in range(reader.count("path")):
                kind = reader.u8("path step kind")
                if kind == 0:
                    path.append(None)
                elif kind == 1:
                    path.append(reader.digest("sibling digest"))
                else:
                    raise ProtocolError(f"bad path step kind {kind}")
            leaves.append(PackedLeaf(canonical_id, dtype, shape_text, value, tuple(path)))
        trees.append(tuple(leaves))
    reader.finish("witness")
    return PackedWitness(tuple(trees))


# --- the table --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PackedTree:
    """One tree of a statement whose positions are all packed: its distinct opened ranks
    (ascending) with the ``position`` and leaf descriptor every position naming each agreed on."""

    commitment: CommitmentRef
    ranks: tuple[int, ...]
    positions: tuple[int, ...]
    descriptors: tuple[LeafDescriptor, ...]


@dataclass(frozen=True, slots=True)
class PackedTable:
    """The packed trees of a statement in tree order and, per obligation position, its ``(tree, leaf)``."""

    trees: tuple[PackedTree, ...]
    slots: tuple[tuple[tuple[int, int], ...], ...]
    schemas: tuple[tuple[PackedSchema, ...], ...]


def packed_table(statement: Statement) -> PackedTable:
    """Group the positions of ``statement`` by tree and by rank; every position must be packed.

    Trees are numbered by first appearance of ``(domain_id, root, count)`` as
    in :func:`~veritor.protocol.proofs.statement.opened_trees`.  Two positions
    naming one rank of one tree must agree on ``position`` and on the leaf
    descriptor; they may differ in word index.  A position whose schema is not
    a packed schema is ``ProtocolError`` (this witness format does not carry
    frame-v3 openings).
    """

    require_expanded(statement, "packed_table")
    order: dict[tuple[bytes, bytes, int], int] = {}
    refs: list[CommitmentRef] = []
    opened: list[dict[int, tuple[int, LeafDescriptor]]] = []
    coordinates: list[list[tuple[int, int]]] = []
    schemas: list[list[PackedSchema]] = []
    for index, obligation in enumerate(statement.obligations):
        trees_of: list[int] = []
        for ref in obligation.commitments:
            key = (ref.domain_id, ref.root, ref.count)
            tree = order.setdefault(key, len(refs))
            if tree == len(refs):
                refs.append(ref)
                opened.append({})
            trees_of.append(tree)
        coordinates.append([])
        schemas.append([])
        for slot, position in enumerate(obligation.positions):
            schema = parse_packed_schema(position.schema)
            if schema is None:
                raise ProtocolError(
                    f"obligation {index} position {slot}: schema {position.schema!r} is not a packed schema"
                )
            tree = trees_of[position.commitment]
            leaf = (position.position, schema.descriptor)
            if opened[tree].setdefault(position.rank, leaf) != leaf:
                raise ProtocolError(
                    f"the statement opens rank {position.rank} of one tree at two coordinates"
                )
            coordinates[-1].append((tree, position.rank))
            schemas[-1].append(schema)
    trees = tuple(
        PackedTree(
            ref,
            tuple(sorted(leaves)),
            tuple(leaves[rank][0] for rank in sorted(leaves)),
            tuple(leaves[rank][1] for rank in sorted(leaves)),
        )
        for ref, leaves in zip(refs, opened, strict=True)
    )
    leaf_index = [{rank: k for k, rank in enumerate(tree.ranks)} for tree in trees]
    return PackedTable(
        trees,
        tuple(tuple((tree, leaf_index[tree][rank]) for tree, rank in row) for row in coordinates),
        tuple(tuple(row) for row in schemas),
    )


# --- the reference opening pass ---------------------------------------------------------


def check_packed_openings(statement: Statement, witness: PackedWitness) -> list[list[int]]:
    """Authenticate every packed leaf once and hand every position its word.

    Returns the decoded words per obligation, indexed like ``positions``;
    raises :class:`Reject` where the guest rejects: a witness of the wrong
    shape, a rank outside its tree, a leaf of the wrong byte length for its
    descriptor, a path of the wrong shape or one that does not reach the
    root, a pinned position whose word is not the public input.  The pin of
    a packed position compares the extracted word's canonical big-endian
    encoding (``ceil(w/8)`` bytes) with ``PositionRef.expected``.
    """

    try:
        table = packed_table(statement)
    except ProtocolError as error:
        raise Reject(VerificationCode.MALFORMED_TRANSCRIPT, str(error)) from error
    if len(witness.trees) != len(table.trees):
        raise Reject(
            VerificationCode.MALFORMED_TRANSCRIPT,
            f"{len(witness.trees)} trees in the witness, {len(table.trees)} in the statement",
        )
    for t, (tree, leaves) in enumerate(zip(table.trees, witness.trees, strict=True)):
        if len(leaves) != len(tree.ranks):
            raise Reject(
                VerificationCode.MALFORMED_TRANSCRIPT,
                f"tree {t} has {len(leaves)} leaves for {len(tree.ranks)} opened ranks",
            )
        commitment = tree.commitment
        for rank, descriptor, leaf in zip(tree.ranks, tree.descriptors, leaves, strict=True):
            failed = Reject(
                VerificationCode.INVALID_OPENING,
                f"packed leaf {rank} of owner {commitment.owner} does not authenticate",
            )
            if rank >= commitment.count:
                raise failed
            if descriptor.framing != VLLM_LEAF_FRAMING:
                raise Reject(VerificationCode.INVALID_OPENING, f"unknown packed framing {descriptor.framing!r}")
            if len(leaf.value) != descriptor.leaf_bytes:
                raise Reject(
                    VerificationCode.INVALID_OPENING,
                    f"packed leaf {rank} holds {len(leaf.value)} bytes, its descriptor says {descriptor.leaf_bytes}",
                )
            shape = vllm_path_shape(commitment.count, rank)
            if len(leaf.path) != len(shape) or any(
                (step is None) == has_sibling for step, has_sibling in zip(leaf.path, shape, strict=True)
            ):
                raise failed
            if vllm_fold_path(leaf.leaf_hash(), rank, leaf.path) != commitment.root:
                raise failed
    decoded: list[list[int]] = []
    for obligation, row, row_schemas in zip(statement.obligations, table.slots, table.schemas, strict=True):
        values: list[int] = []
        for ref, (t, k), schema in zip(obligation.positions, row, row_schemas, strict=True):
            try:
                word = extract_word(schema, witness.trees[t][k].value)
            except ProtocolError as error:
                raise Reject(VerificationCode.INVALID_VALUE, str(error)) from error
            if ref.expected is not None and ref.expected != word.to_bytes(schema.word_bytes, "big"):
                raise Reject(
                    VerificationCode.PUBLIC_IO_MISMATCH,
                    f"word {schema.index} of packed leaf {ref.rank} differs from the public input",
                )
            values.append(word)
        decoded.append(values)
    return decoded
