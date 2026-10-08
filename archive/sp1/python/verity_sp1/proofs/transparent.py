"""The transparent checker: the proof *is* the openings (promoted from veritor ``protocol/proofs/transparent.py``).

This is the protocol as it stood before proofs were pluggable, expressed
through :class:`ProofBackend`: the prover hands over the witness (values and
Merkle paths), and the verifier authenticates every opening under the roots
the statement names, compares pinned public inputs, decodes each value with
the gate set's canonical codec and recomputes every sampled unit's gates from
its opened inputs with the gate set's pinned semantics, comparing the opened
outputs.  It gives the verifier every value it checks, so it is not
zero-knowledge; it is the reference every other backend must agree with.
"""

from __future__ import annotations

from verity.ml.commitments.limits import VerificationLimits

# The framing primitives are private to merkle.py on purpose: nothing but the
# commitment code and this reference checker may spell out a leaf or node hash.
from verity.ml.commitments.merkle import _hash as framed_hash
from verity.ml.commitments.merkle import _uint, merkle_depth
from verity.ml.commitments.multiproof import verify_multiproof
from verity.errors import ProtocolError

from verity_vllm.check.codes import Reject, VerificationCode
from .gateset import GateSet

#: The owner index of a per-model weight commitment (``CommitmentDomain.owner``); the boundary is ``-1``.
WEIGHT_OWNER = -2

from .statement import (
    LOCAL,
    Obligation,
    OpenedTree,
    PositionRef,
    Statement,
    TreeTable,
    Witness,
)
from .packed import check_packed_openings, decode_packed_witness
from .wire import (
    DEFAULT_WITNESS_FORMAT,
    WITNESS_FORMATS,
    decode_multiproof_witness,
    decode_witness,
    encode_witness_as,
    witness_format_of,
)

TRANSPARENT_BACKEND = "transparent"


def fold_path(
    domain_id: bytes, rank: int, leaf: bytes, path: tuple[bytes, ...]
) -> bytes:
    """Fold an authentication path to its root with the ``merkle.py`` node framing."""

    digest = leaf
    cursor = rank
    for level, sibling in enumerate(path):
        pair = (digest, sibling) if cursor % 2 == 0 else (sibling, digest)
        digest = framed_hash(
            b"node", domain_id, _uint(level), _uint(cursor >> 1), *pair
        )
        cursor >>= 1
    return digest


def leaf_hash(
    domain_id: bytes, rank: int, position: int, schema: str, value: bytes
) -> bytes:
    """``CommitmentDomain.leaf`` from the domain id alone."""

    return framed_hash(
        b"leaf", domain_id, _uint(rank), _uint(position), schema.encode("utf-8"), value
    )


class TransparentBackend:
    """Openings as the proof; Merkle paths and relations checked in Python.

    ``gate_set`` supplies the pinned semantics and must be the statement's.
    ``witness_format`` is what ``prove`` emits (``1``: a path per opening;
    ``2``: one multiproof per tree, see :mod:`.wire`); ``verify`` accepts both.
    """

    backend_id = TRANSPARENT_BACKEND

    def __init__(
        self,
        gate_set: GateSet,
        limits: VerificationLimits | None = None,
        witness_format: int = DEFAULT_WITNESS_FORMAT,
    ) -> None:
        if not isinstance(gate_set, GateSet):
            raise ProtocolError("TransparentBackend requires a GateSet")
        if witness_format not in WITNESS_FORMATS:
            raise ProtocolError(f"unknown witness format {witness_format}")
        self._gate_set = gate_set
        self._limits = VerificationLimits() if limits is None else limits
        self.witness_format = witness_format

    def prove(self, statement: Statement, witness: Witness) -> bytes:
        witness.for_statement(statement)
        return encode_witness_as(statement, witness, self.witness_format)

    def verify(self, statement: Statement, proof: bytes) -> bool:
        self.check(statement, proof)
        return True

    def _where(self, obligation: Obligation, ref: PositionRef) -> str:
        owner = obligation.commitments[ref.commitment].owner
        if owner == WEIGHT_OWNER:
            return f"weight rank {ref.rank}"
        return f"address {ref.position}"

    def check(self, statement: Statement, proof: bytes) -> None:
        """Verify, raising :class:`Reject` with the precise code on failure.

        A format-3 statement (one with :class:`~.statement.PositionRange`
        entries) is checked as its expansion: the reference semantics of a
        range *is* the per-word statement it abbreviates, so the verdict here
        is by construction the verdict on ``statement.expand()``.
        """

        statement = statement.expand()
        if statement.gate_set_digest != bytes.fromhex(self._gate_set.digest):
            raise Reject(
                VerificationCode.EXPECTATION_MISMATCH,
                "the statement names another gate set than the transparent checker's",
            )
        self._check_programs(statement)
        try:
            witness_format = witness_format_of(proof)
        except ProtocolError as error:
            raise Reject(VerificationCode.MALFORMED_TRANSCRIPT, str(error)) from error
        if witness_format == 2:
            self._check_multiproof(statement, proof)
            return
        if witness_format == 3:
            self._check_packed(statement, proof)
            return
        try:
            witness = decode_witness(proof)
            witness.for_statement(statement)
        except ProtocolError as error:
            raise Reject(VerificationCode.MALFORMED_TRANSCRIPT, str(error)) from error
        for obligation, openings in zip(
            statement.obligations, witness.obligations, strict=True
        ):
            values = self._open_all(obligation, openings)
            self._check_relations(statement, obligation, values)

    def _check_multiproof(self, statement: Statement, proof: bytes) -> None:
        """Format 2: reconstruct every tree's root once, then check the relations.

        Same outcome as the per-opening pass: a position's value is
        authenticated iff the leaf it names is in a tree whose root the
        multiproof reconstructs, and every position of the statement is such
        a leaf.
        """

        self._limits.enforce("max_proof_bytes", len(proof))
        try:
            witness = decode_multiproof_witness(proof)
            table: TreeTable = witness.for_statement(statement)
        except ProtocolError as error:
            raise Reject(VerificationCode.MALFORMED_TRANSCRIPT, str(error)) from error
        for tree, (values, siblings) in zip(table.trees, witness.trees, strict=True):
            if not tree.ranks:
                if siblings:
                    raise Reject(
                        VerificationCode.MALFORMED_TRANSCRIPT,
                        "a tree with no openings has no multiproof",
                    )
                continue
            leaves = {
                rank: leaf_hash(tree.domain_id, rank, position, schema, value)
                for rank, position, schema, value in zip(
                    tree.ranks, tree.positions, tree.schemas, values, strict=True
                )
            }
            if not verify_multiproof(
                tree.root, tree.count, tree.domain_id, leaves, siblings
            ):
                raise Reject(
                    VerificationCode.INVALID_OPENING,
                    f"multiproof of {self._tree_name(statement, tree)} fails",
                )
        for obligation, row in zip(statement.obligations, table.slots, strict=True):
            decoded: list[int] = []
            for ref, (index, leaf) in zip(obligation.positions, row, strict=True):
                value = witness.trees[index][0][leaf]
                where = self._where(obligation, ref)
                if ref.expected is not None and ref.expected != value:
                    raise Reject(
                        VerificationCode.PUBLIC_IO_MISMATCH,
                        f"input at {where} differs from the public input",
                    )
                try:
                    decoded.append(self._decode(ref.schema, value))
                except ProtocolError as error:
                    raise Reject(
                        VerificationCode.INVALID_VALUE,
                        f"value at {where} is not canonical: {error}",
                    ) from error
            self._check_relations(statement, obligation, decoded)

    def _check_packed(self, statement: Statement, proof: bytes) -> None:
        """Format 3: authenticate every packed leaf once under the serving side's framing,
        extract each position's word at its statement-fixed index, then check the relations
        (:func:`veritor.protocol.proofs.packed.check_packed_openings`)."""

        self._limits.enforce("max_proof_bytes", len(proof))
        try:
            witness = decode_packed_witness(proof)
        except ProtocolError as error:
            raise Reject(VerificationCode.MALFORMED_TRANSCRIPT, str(error)) from error
        decoded = check_packed_openings(statement, witness)
        for obligation, values in zip(statement.obligations, decoded, strict=True):
            self._check_relations(statement, obligation, values)

    @staticmethod
    def _tree_name(statement: Statement, tree: OpenedTree) -> str:
        for obligation in statement.obligations:
            for ref in obligation.commitments:
                if (ref.domain_id, ref.root, ref.count) == (
                    tree.domain_id,
                    tree.root,
                    tree.count,
                ):
                    return f"owner {ref.owner} ({len(tree.ranks)} leaves opened)"
        return "an unknown tree"

    def _open_all(
        self,
        obligation: Obligation,
        openings: tuple[tuple[bytes, tuple[bytes, ...]], ...],
    ) -> list[int]:
        values: list[int] = []
        for ref, (value, path) in zip(obligation.positions, openings, strict=True):
            commitment = obligation.commitments[ref.commitment]
            where = self._where(obligation, ref)
            failed = Reject(
                VerificationCode.INVALID_OPENING,
                f"opening of {where} failed under owner {commitment.owner}",
            )
            if ref.rank >= commitment.count or len(path) != merkle_depth(
                commitment.count
            ):
                raise failed
            self._limits.enforce("max_proof_bytes", len(value) + 32 * len(path))
            leaf = leaf_hash(
                commitment.domain_id, ref.rank, ref.position, ref.schema, value
            )
            if fold_path(commitment.domain_id, ref.rank, leaf, path) != commitment.root:
                raise failed
            if ref.expected is not None and ref.expected != value:
                raise Reject(
                    VerificationCode.PUBLIC_IO_MISMATCH,
                    f"input at {where} differs from the public input",
                )
            try:
                values.append(self._decode(ref.schema, value))
            except ProtocolError as error:
                raise Reject(
                    VerificationCode.INVALID_VALUE,
                    f"value at {where} is not canonical: {error}",
                ) from error
        return values

    def _decode(self, schema: str, value: bytes) -> int:
        if not schema.startswith("u"):
            raise ProtocolError(f"unknown leaf schema {schema!r}")
        try:
            width = int(schema[1:])
        except ValueError:
            raise ProtocolError(f"unknown leaf schema {schema!r}") from None
        if len(value) != (width + 7) // 8:
            raise ProtocolError(
                f"encoded value must be exactly {(width + 7) // 8} bytes"
            )
        decoded = int.from_bytes(value, "big")
        if decoded >= 1 << width:
            raise ProtocolError(f"encoded value is not a {width}-bit value")
        return decoded

    def _check_programs(self, statement: Statement) -> None:
        """Every program of the statement is well-formed for the gate set.

        Each op is one of the gate set's, has as many arguments as its arity,
        and each source gate is one of the program's outputs -- the Lean
        ``Program.WellFormed`` (``lean/VeritorSpec/Program.lean``) beyond what
        :class:`KindProgram` checks structurally.  It holds for *every*
        program, used by an obligation or not, as ``checkReference``
        (``Batch.lean``) requires; until the ``typed-obligation/v0``
        conformance work it was enforced only per obligation, so a statement
        carrying an unused program with a wrong-arity gate was accepted here
        and rejected by the Lean reference.  No honest statement has a
        malformed program.  Mirrors ``check_programs`` in
        ``archive/sp1/common/src/kernel/relations.rs``.
        """

        for program in statement.kinds:
            opened = set(program.outputs)
            for offset, gate in enumerate(program.gates):
                try:
                    semantics = self._gate_set[gate.op]
                except Exception as error:
                    raise Reject(
                        VerificationCode.INVALID_COMPILED_RESULT,
                        f"unknown gate {gate.op}",
                    ) from error
                if len(gate.args) != semantics.arity:
                    raise Reject(
                        VerificationCode.INVALID_COMPILED_RESULT,
                        f"gate {offset} of kind {program.kind.hex()[:12]} has the wrong arity for {gate.op}",
                    )
                if semantics.source is not None and offset not in opened:
                    raise Reject(
                        VerificationCode.INVALID_COMPILED_RESULT,
                        f"source gate {offset} of kind {program.kind.hex()[:12]} is not opened",
                    )

    def _check_relations(
        self, statement: Statement, obligation: Obligation, values: list[int]
    ) -> None:
        """Recompute the copy's gates from its opened inputs; every opened gate must agree.

        A source gate has no relation: its opened value *is* its value (the
        boundary pins an ``in`` gate to the public input, ``kappa_W`` a
        weight).  Every other gate is evaluated from what it reads -- an
        opened input, or an earlier gate of the copy -- and, when the kind
        opens it, checked against the opened value; an internal gate is never
        committed, so the recomputation is what stands for it.
        """

        program = statement.program(obligation.kind)
        opened = dict(zip(program.outputs, obligation.outputs, strict=True))
        local: list[int] = []
        for offset, gate in enumerate(program.gates):
            try:
                semantics = self._gate_set[gate.op]
            except Exception as error:
                raise Reject(
                    VerificationCode.INVALID_COMPILED_RESULT, f"unknown gate {gate.op}"
                ) from error
            slot = opened.get(offset)
            if semantics.source is not None:
                if slot is None:
                    raise Reject(
                        VerificationCode.INVALID_COMPILED_RESULT,
                        f"source gate {offset} of kind {obligation.kind.hex()[:12]} is not opened",
                    )
                local.append(values[slot])
                continue
            args = tuple(
                local[value] if space == LOCAL else values[obligation.inputs[value]]
                for space, value in gate.args
            )
            if slot is None:
                try:
                    local.append(semantics.evaluate(args))
                except Exception as error:
                    raise Reject(
                        VerificationCode.TRUSTED_SERVICE_FAILURE,
                        f"gate {gate.op} raised at offset {offset} of unit {obligation.unit}: {error}",
                    ) from error
                continue
            where = self._where(obligation, obligation.positions[slot])
            try:
                satisfied = semantics.check(args, values[slot])
            except Exception as error:
                raise Reject(
                    VerificationCode.TRUSTED_SERVICE_FAILURE,
                    f"gate {gate.op} raised at {where}: {error}",
                ) from error
            if not satisfied:
                raise Reject(
                    VerificationCode.RELATION_REJECTED,
                    f"gate at {where} violates {gate.op}",
                )
            local.append(values[slot])
