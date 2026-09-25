"""Binding the typed object to the backend statement a proof commits (lane IB).

The proof backend proves a *lowered* statement ``L`` and exposes its digest
``h_L = sha256(statement_bytes)`` as a public value.  The typed object ``O``
(``typed-obligation/v0``) that ``L`` was lowered from has its own canonical
digest ``h_O`` (:func:`~veritor.interface.typed_obligation.object_digest`).
The lowering erases fields of ``O`` -- reference naming, targets, the declared
interface and its projection, ... (``ir-lowering.md`` §9.4, ``lower_eq_iff``)
-- so two different objects can lower to one ``L``: hashing only ``L`` into the
proof cannot tell them apart, and hashing only ``O`` would let a prover pair
``H(O_1)`` with an unrelated ``L_2`` (IR §I.3, GUIDANCE §17.3).  The fence is
therefore

.. code-block:: text

    typed obligation O  --validate-->  h_O  --deterministic lower-->  L, h_L
    public values = h_O || h_L || verdict                    (65 bytes)
    verifier: h_L == digest(lower(O))  for the O with digest h_O

:func:`check_binding` is that verifier-side check, fail-closed.  It
establishes **identity**: the statement the guest verified is the lowering of
*this* object and of no object with another digest.  It establishes nothing
else -- in particular

* not **plan validity**: whether the object's obligations are a well-formed,
  complete plan is the validator's (RA's) question and is answered by
  :func:`~veritor.interface.typed_obligation.validate`, which this check runs
  first and reports as ``E_OBJECT_INVALID``; a digest is never a substitute
  for it (GUIDANCE §17.6);
* not **canonical derivation**: that ``O`` is the object the protocol expected
  for a given challenge (the deterministic derivation from the bound program,
  plan and challenge, GUIDANCE §17.4) is decided above this seam, by whoever
  knows the challenge.

The theorems this check relies on are L7b's (``lean/VeritorSpec/IRErasure.lean``):
``lower_deterministic`` (``lower`` is a function of the object, so ``h_L`` is
determined by ``O``), ``lower_eq_iff`` (two well-formed objects with one
lowering agree on exactly the read fields), ``lowerQuot_injective``.  Nothing is
re-proved here; the check is the digest comparison the theorems make
meaningful.  The guest itself carries ``h_O`` through *unchecked*
(``zk/sp1/common/src/lib.rs::public_values_with_object``); this module, its Rust
twin ``zk/sp1/common/src/binding.rs`` and the ``binding`` column of the
conformance vectors are where the relationship is checked.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .wire import statement_digest

from .lowering import statement_bytes, statement_bytes_v3
from .typed_obligation import Code, InterfaceReject, TypedObligationSet, object_digest, validate

__all__ = [
    "BOUND_PUBLIC_VALUES_LEN",
    "STATEMENT_FORMATS",
    "UNBOUND_PUBLIC_VALUES_LEN",
    "VERDICT_ACCEPT",
    "BindingCode",
    "BindingResult",
    "bound_public_values",
    "check_binding",
]

#: ``h_O || h_L || verdict``: the public values of a proof that binds its typed object.
BOUND_PUBLIC_VALUES_LEN = 65
#: ``h_L || verdict``: the public values of a proof that binds no typed object.
UNBOUND_PUBLIC_VALUES_LEN = 33
#: The verdict byte of an accepted batch in a sound guest build (``VERDICT_ACCEPT`` in
#: ``zk/sp1/common/src/lib.rs``; a diagnostic build commits ``0xEE`` / ``0xEF``, which
#: no verifier accepts as a verdict).
VERDICT_ACCEPT = 1


class BindingCode(StrEnum):
    """Why the binding check fails, in the order the check tries them."""

    #: The typed object does not validate: the validator's code is reported alongside.
    #: An invalid object binds nothing, whatever the public values say.
    E_OBJECT_INVALID = "E_OBJECT_INVALID"
    #: The public values are 33 bytes: the proof commits ``h_L || verdict`` only and
    #: binds no typed object.  Such a proof says nothing about *which* object lowered
    #: to the statement.
    E_UNBOUND_OBJECT = "E_UNBOUND_OBJECT"
    #: The public values are neither 33 nor 65 bytes.
    E_PUBLIC_VALUES_LENGTH = "E_PUBLIC_VALUES_LENGTH"
    #: The first 32 bytes are not this object's digest ``h_O``.
    E_OBJECT_DIGEST_MISMATCH = "E_OBJECT_DIGEST_MISMATCH"
    #: The next 32 bytes are not ``sha256(statement_bytes(lower(O)))``: the statement
    #: the guest verified is not the lowering of this object (the ``H(O_1)`` with
    #: ``L_2`` pairing IR §I.3 names).
    E_LOWERING_MISMATCH = "E_LOWERING_MISMATCH"


@dataclass(frozen=True, slots=True)
class BindingResult:
    """The outcome of :func:`check_binding`.

    ``ok`` is ``True`` iff the public values bind exactly this object to exactly its
    lowering; then ``verdict`` is the committed verdict byte (as committed: ``1``
    accept, ``0`` reject; ``0xEE`` / ``0xEF`` are a diagnostic build's and are the
    backend's ``decode_verdict``'s business).  Otherwise ``code`` says why, ``detail``
    says it in words and ``verdict`` is ``None``.  ``object_code`` is the validator's
    code when ``code`` is ``E_OBJECT_INVALID``.  ``object_digest`` and
    ``statement_digest`` are ``h_O`` and ``h_L`` of the object as computed here (hex),
    absent when the object did not validate.
    """

    ok: bool
    code: BindingCode | None = None
    detail: str = ""
    verdict: int | None = None
    object_code: Code | None = None
    object_digest: str | None = None
    statement_digest: str | None = None


#: The statement formats a lowering can be recorded in (``veritor/proofs/statement/v2``
#: and ``v3``); ``h_L`` is the digest of the bytes in the *recorded* format.
STATEMENT_FORMATS = (2, 3)


def _lowering_bytes(typed: TypedObligationSet, statement_format: int, obligation_format: str | None = None) -> bytes:
    """``lower(O)`` encoded in ``statement_format`` (2: format-2 bytes; 3: format-3 bytes)."""

    if statement_format == 2:
        return statement_bytes(typed, obligation_format=obligation_format)
    if statement_format == 3:
        return statement_bytes_v3(typed, obligation_format=obligation_format)
    raise ValueError(f"statement_format must be one of {STATEMENT_FORMATS}, got {statement_format!r}")


def _digests(obj: TypedObligationSet | Mapping[str, Any], statement_format: int = 2, obligation_format: str | None = None) -> tuple[TypedObligationSet, str, str]:
    """Validate ``obj`` and compute ``(object, h_O, h_L)``; raises :class:`InterfaceReject`.

    ``h_L`` is the digest of the lowering in ``statement_format`` (default 2, the
    historical record format); a format-3 record binds the format-3 bytes.
    """

    typed = obj if isinstance(obj, TypedObligationSet) else TypedObligationSet.from_json(obj)
    validate(typed, obligation_format=obligation_format)
    # h_O over the form given (a JSON object's canonical bytes are the record's `canonical`)
    return typed, object_digest(obj), statement_digest(_lowering_bytes(typed, statement_format, obligation_format)).hex()


def bound_public_values(
    obj: TypedObligationSet | Mapping[str, Any], verdict: int = VERDICT_ACCEPT, *, statement_format: int = 2, obligation_format: str | None = None
) -> bytes:
    """The 65 public bytes ``h_O || h_L || verdict`` a correct prover produces for ``obj``.

    ``obj`` must validate (:class:`InterfaceReject` otherwise: an invalid object has
    no lowering to bind).  ``verdict`` defaults to the accept byte; a driver that
    expects the guest to reject passes ``0``.  ``statement_format`` names the format
    the statement was given to the guest in (default 2).  This is what
    :func:`check_binding` accepts for ``obj`` (in that format) and nothing else.
    """

    _, h_o, h_l = _digests(obj, statement_format, obligation_format)
    if not 0 <= verdict <= 0xFF:
        raise ValueError(f"verdict byte out of range: {verdict}")
    return bytes.fromhex(h_o) + bytes.fromhex(h_l) + bytes([verdict])


def check_binding(
    obj: TypedObligationSet | Mapping[str, Any], public_values: bytes, *, statement_format: int = 2, obligation_format: str | None = None
) -> BindingResult:
    """Check that ``public_values`` bind ``obj`` to its own lowering; fail-closed.

    ``statement_format`` (2 or 3, default 2) is the format the record says the
    statement was given to the guest in: ``h_L`` is recomputed from the object in
    THAT format, never read from the record.  A format-3 record checked as format 2
    (or the reverse) fails ``E_LOWERING_MISMATCH`` -- the two formats' digests differ
    by construction.

    In order: ``obj`` is validated (``E_OBJECT_INVALID`` with the validator's code --
    the object digest never stands in for validity); ``h_O`` and ``h_L`` are computed;
    ``public_values`` must be exactly 65 bytes ``h_O' || h_L' || verdict``
    (``E_UNBOUND_OBJECT`` for the 33-byte layout that binds no object,
    ``E_PUBLIC_VALUES_LENGTH`` for anything else); ``h_O' == h_O``
    (``E_OBJECT_DIGEST_MISMATCH``); ``h_L' == h_L`` (``E_LOWERING_MISMATCH``: the
    checked lowering relationship -- the statement the guest verified is the
    lowering of *this* object).  Only when both hold is the verdict byte returned.

    What passing establishes: the proof's statement is ``lower(O)`` for this ``O``
    and, by ``lower_deterministic``, for no object with another lowering; by
    ``lower_eq_iff`` any other well-formed object with the same ``h_L`` differs from
    ``O`` only in erased fields, and its ``h_O`` differs -- which is exactly what the
    first 32 bytes distinguish.  What it does not establish: that ``O``'s plan is
    valid beyond what ``validate`` checks (RA's validator), or that ``O`` is the
    canonical derivation for the challenge at hand (GUIDANCE §17.4).
    """

    try:
        _, h_o, h_l = _digests(obj, statement_format, obligation_format)  # [E9, D28] the record's obligation format, by name
    except InterfaceReject as error:
        return BindingResult(
            False,
            BindingCode.E_OBJECT_INVALID,
            f"the typed object does not validate ({error.code}: {error.detail}); nothing to bind",
            object_code=error.code,
        )
    n = len(public_values)
    if n == UNBOUND_PUBLIC_VALUES_LEN:
        return BindingResult(
            False,
            BindingCode.E_UNBOUND_OBJECT,
            f"{n} public bytes: the proof commits h_L || verdict and binds no typed object",
            object_digest=h_o,
            statement_digest=h_l,
        )
    if n != BOUND_PUBLIC_VALUES_LEN:
        return BindingResult(
            False,
            BindingCode.E_PUBLIC_VALUES_LENGTH,
            f"{n} public bytes, expected {BOUND_PUBLIC_VALUES_LEN} (h_O || h_L || verdict)",
            object_digest=h_o,
            statement_digest=h_l,
        )
    committed_o, committed_l, verdict = public_values[:32].hex(), public_values[32:64].hex(), public_values[64]
    if committed_o != h_o:
        return BindingResult(
            False,
            BindingCode.E_OBJECT_DIGEST_MISMATCH,
            f"public h_O {committed_o[:16]}... is not this object's {h_o[:16]}...",
            object_digest=h_o,
            statement_digest=h_l,
        )
    if committed_l != h_l:
        return BindingResult(
            False,
            BindingCode.E_LOWERING_MISMATCH,
            f"public h_L {committed_l[:16]}... is not the digest of this object's lowering {h_l[:16]}...",
            object_digest=h_o,
            statement_digest=h_l,
        )
    return BindingResult(True, verdict=verdict, object_digest=h_o, statement_digest=h_l)
