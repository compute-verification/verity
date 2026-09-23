"""The relation-only diagnostic's compact input, built from a ``typed-obligation/v0`` object and its witness.

A host built with ``--features unsound-relation-only`` embeds a guest that runs nothing but the gate chain of one
``GemmCoordinate<K>`` obligation in ``veritor.tc-ampere-bf16@1`` -- no statement, no openings, no authentication
(``backends/sp1/common/src/relation_only.rs``).  It reads the **compact input**

    K: u32 LE || x[0..K]: u16 LE || W[0..K]: u16 LE

through the host's ``--witness`` argument (the statement is ignored) and commits

    sha256(compact input) || sha256(output word: u16 LE) || 0xEE        (65 bytes)

:func:`compact_input` derives that input from the object and the witness the sound path would be given: the
witness is authenticated here, in Python, against the object's roots (:func:`~verity.verification.packed.check_packed_openings`),
the one ``GemmCoordinate`` obligation's boundary words are read off it -- the x row, the W row, the committed
output -- and the Python reference (:func:`~verity.verification.lowering.check_relations`) is required to agree
that the committed output is the relation's value.  What the guest then proves is the arithmetic on the real
words; what it does not prove is anything about where they came from.  Such a run is labelled
``proof_class = ARITHMETIC_DIAGNOSTIC``, ``authentication = excluded``, ``statement_machinery = excluded``, and
is never a result.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from verity.errors import ProtocolError
from verity.verification.lowering import check_relations, to_statement
from verity.verification.packed import check_packed_openings, decode_packed_witness
from verity.verification.typed_obligation import TypedObligationSet, validate
from verity.verification.wire import witness_format_of

__all__ = ["FAMILY", "FEATURE", "STEP", "VERDICT_DIAGNOSTIC_ACCEPT", "CompactInput", "compact_input", "encode_compact", "public_values_for"]

#: The cargo feature of the relation-only host/guest, as ``info`` lists it under ``guest_features``.
FEATURE = "unsound-relation-only"
#: The one family whose gate chain the guest evaluates (``@2`` differs in the NaN word of ``f32_to_bf16``).
FAMILY = "veritor.tc-ampere-bf16@1"
#: Operand words per tensor-core step gate.
STEP = 16
#: The verdict byte a diagnostic guest commits for "accepted" (``VERDICT_ACCEPT`` in ``common/src/lib.rs``).
VERDICT_DIAGNOSTIC_ACCEPT = 0xEE


def encode_compact(x: Sequence[int], w: Sequence[int]) -> bytes:
    """``K || x || W`` as the guest reads it; ``ProtocolError`` for rows that are not a GemmCoordinate's."""

    k = len(x)
    if k == 0 or k % STEP or len(w) != k:
        raise ProtocolError(f"a compact input takes two rows of K words, K a positive multiple of {STEP}; got {k} and {len(w)}")
    words = [*x, *w]
    if any(not (0 <= word < 1 << 16) for word in words):
        raise ProtocolError("every word of a compact input is a 16-bit BF16 word")
    return k.to_bytes(4, "little") + b"".join(word.to_bytes(2, "little") for word in words)


def public_values_for(compact: bytes, output_word: int) -> bytes:
    """The 65 bytes the relation-only guest commits for ``compact`` when the chain gives ``output_word``."""

    return hashlib.sha256(compact).digest() + hashlib.sha256(output_word.to_bytes(2, "little")).digest() + bytes([VERDICT_DIAGNOSTIC_ACCEPT])


@dataclass(frozen=True, slots=True)
class CompactInput:
    """One GemmCoordinate obligation's rows as the relation-only guest takes them, and what it must commit."""

    K: int
    x: tuple[int, ...]
    w: tuple[int, ...]
    #: The output word committed in the witness (authenticated here); equals the reference evaluation.
    output_word: int
    data: bytes

    @property
    def public_values(self) -> bytes:
        return public_values_for(self.data, self.output_word)


def _gemm_chain_is_canonical(sub: Any, k: int) -> bool:
    """The body is ``tc_dot16_0`` over block 0, ``tc_dot16`` chained over the rest, ``f32_to_bf16``, with
    ``x = in[0..K]`` and ``W = in[K..2K]`` -- the layout the compact input assumes."""

    steps = k // STEP
    if tuple(sub.incoming) != (16,) * (2 * k) or tuple(sub.outgoing) != (steps,) or len(sub.gates) != steps + 1:
        return False
    for i, gate in enumerate(sub.gates[:steps]):
        block = [("in", j) for j in range(i * STEP, (i + 1) * STEP)] + [("in", k + j) for j in range(i * STEP, (i + 1) * STEP)]
        want = (("tc_dot16_0", block) if i == 0 else ("tc_dot16", [("gate", i - 1), *block]))
        if gate.op != want[0] or [(str(s), int(j)) for s, j in gate.args] != want[1]:
            return False
    last = sub.gates[steps]
    return last.op == "f32_to_bf16" and [(str(s), int(j)) for s, j in last.args] == [("gate", steps - 1)]


def compact_input(obj: TypedObligationSet | Mapping[str, Any], witness: bytes) -> CompactInput:
    """The compact input of ``obj``'s one ``GemmCoordinate<K>`` obligation, its words read off ``witness``.

    ``obj`` must validate in :data:`FAMILY` and carry exactly one ``GemmCoordinate`` obligation whose body is the
    canonical chain; ``witness`` must be a format-3 (packed) witness that authenticates under the object's roots
    (a :class:`~verity.verification.codes.Reject` otherwise, as the reference checker raises it).  The committed
    output word must be what the Python reference computes from the rows.  Anything else is ``ProtocolError``.
    """

    typed = obj if isinstance(obj, TypedObligationSet) else TypedObligationSet.from_json(obj)
    gate_set = validate(typed)
    if typed.family.id != FAMILY:
        raise ProtocolError(f"the relation-only guest evaluates {FAMILY}; the object is in {typed.family.id}")
    subcircuits = {s.digest: s for s in typed.subcircuits}
    gemm = [
        (index, o, subcircuits[o.subcircuit])
        for index, o in enumerate(typed.obligations)
        if subcircuits[o.subcircuit].definition is not None and subcircuits[o.subcircuit].definition.name == "GemmCoordinate"
    ]
    if len(gemm) != 1:
        raise ProtocolError(f"the relation-only guest takes one GemmCoordinate obligation; the object has {len(gemm)}")
    index, obligation, sub = gemm[0]
    k = dict(sub.definition.static).get("K")
    if not isinstance(k, int) or k <= 0 or k % STEP:
        raise ProtocolError(f"GemmCoordinate K must be a positive multiple of {STEP}, got {k!r}")
    if not _gemm_chain_is_canonical(sub, k):
        raise ProtocolError("the GemmCoordinate body is not the canonical tc_dot16_0 / tc_dot16 / f32_to_bf16 chain over x || W")
    if witness_format_of(witness) != 3:
        raise ProtocolError("the compact input is read off a format-3 (packed) witness; other formats are not supported here")
    statement = to_statement(typed).expand()
    values = check_packed_openings(statement, decode_packed_witness(witness))
    row = values[index]
    if len(row) != 2 * k + 1:
        raise ProtocolError(f"obligation {index} opens {len(row)} words, expected 2K + 1 = {2 * k + 1}")
    if not check_relations(typed, values, gate_set):
        raise ProtocolError("the Python reference rejects the witness's relation: the committed output is not the chain's value")
    x, w, out = tuple(row[:k]), tuple(row[k : 2 * k]), int(row[2 * k])
    return CompactInput(K=k, x=x, w=w, output_word=out, data=encode_compact(x, w))
