"""Total semantics of one BF16 tensor-core step: ``veritor.tensor-core.ampere_bf16_m16n8k16@3``.

:mod:`verity.ml.circuits.silicon.silicon` models the *finite* domain of an Ampere
``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`` step (Hawkeye's
parameters: two groups of eight exact products, a 25-bit truncating adder,
exponent floor ``-132``) and raises on any NaN or infinity.  This module
extends that model to a **total function on every encoding** of the gate's
signature -- every 16-bit word as a BF16 operand, every 32-bit word as the
accumulator -- as a mathematical definition, not a lookup of measurements:

* a *value* is a NaN, a signed infinity, or a finite :class:`~verity.ml.circuits.silicon.silicon.Term`;
* the **product** of two BF16 words is NaN if either is a NaN (quiet or
  signalling, any payload, either sign); otherwise, if either is an infinity,
  NaN if the other is a zero (``±0``; a subnormal is *not* a zero) and else an
  infinity with the XOR of the signs; otherwise the exact finite product of
  :func:`~verity.ml.circuits.silicon.silicon.bf16_product`;
* the **accumulator** word decodes likewise (NaN, ``±inf``, or :func:`~verity.ml.circuits.silicon.silicon.fp32_term`);
* one **group step** takes the incoming value and the group's products: NaN if
  any of them is NaN; NaN if infinities of both signs are present; that
  infinity if infinities of one sign are present (whatever the finite terms);
  otherwise the finite :func:`~verity.ml.circuits.silicon.silicon.group_sum` of the finite
  terms, which *saturates* to an infinity of its sign when it leaves the FP32
  range (exponent above ``127``).  A saturated infinity is an ordinary
  infinity for the following group (``+inf`` from an overflowing group 1 and
  a ``-inf`` operand in group 2 give NaN);
* the **result word** is ``0x7FFFFFFF`` for NaN (one canonical word: positive
  sign, every mantissa bit set, whatever the payloads and signs of the input
  NaNs), ``0x7F800000`` / ``0xFF800000`` for the infinities, and
  :func:`~verity.ml.circuits.silicon.silicon.pack_fp32` for a finite value -- whose zero is
  always ``+0`` (``-0`` accumulators and ``-0`` products never produce ``-0``).

On finite operands this is exactly :func:`verity.ml.circuits.silicon.silicon.tc_dot`
(``lean/VeritorSpec/TCTotal.lean`` proves the Lean transcription of the two
agree, ``tcDot_total_eq_of_finite``).  On the special encodings it is a
*specification*; the evidence that the A100 (sm_80) and the RTX 4090 (sm_89)
implement it is the campaign recorded under ``docs/data/tc-total-2026-09-07/``
(``gpu/tc-total/campaign.py``): 147,258 structured tiles x 128 outputs x 3
repetitions per device, every output reproduced, no nondeterminism, no
position or k-index dependence, and no difference between the two devices.

Also here: :func:`cvt_rn_bf16_f32`, the FP32 -> BF16 conversion
(``cvt.rn.bf16.f32``, SASS ``F2FP.BF16.PACK_AB``) as measured on both devices:
round to nearest even on finite words, ``±inf`` for ``±inf`` and for finite
words that round beyond BF16's range, and the canonical ``0x7FFF`` for every
NaN.  Note that :func:`verity.ml.circuits.silicon.silicon.f32_to_bf16` maps NaN to ``0x7FC0``
instead; the difference is only on NaN inputs, which the pinned gate set's
``check`` rejects.

**Hopper** (``veritor.tensor-core.hopper_bf16_wgmma_k16@1``, :data:`HOPPER_BF16_WGMMA_K16`):
the same total semantics over a different finite pipeline -- *one* group of
sixteen products plus the accumulator, a 26-bit adder, exponent floor
``-133`` -- as one accumulator coordinate of ``wgmma.mma_async.sync.aligned
.m64n128k16.f32.bf16.bf16`` (SASS ``HGMMA.64x128x16.F32.BF16``) computes it
on the H100.  The pipeline parameters were *recovered* on the device (lane HP,
``gpu/tc-hopper/campaign.py``: neutral-subgroup, internal-precision and
parameter-fit families; ``16:26:-133`` is the unique zero-mismatch candidate
over groups x width x floor) and the special-value rules R1-R8 were re-tested
with L6's case families; the record is ``docs/data/tc-hopper-2026-09-07/``.
The instruction's ``scale-d = 0`` form (accumulator ignored, CUTLASS
``ScaleOut::Zero``) was measured equal to ``acc = +0``.  On finite operands
the function is :func:`verity.ml.circuits.silicon.silicon.tc_dot` on
:data:`~verity.ml.circuits.silicon.silicon.HOPPER_BF16_M16N8K16` (identical parameters,
Hawkeye's H100 numbers, now re-measured).

Promoted verbatim from ``veritor`` ``src/veritor/core/silicon_total.py`` lines
1-258 (commit eb865ce29); the gate-set builders were left behind.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .errors import InvalidArtifact
from .models import HOPPER_BF16_WGMMA_K16
from .silicon import (
    AMPERE_BF16_M16N8K16,
    FP32_MAX_EXPONENT,
    Pipeline,
    Term,
    bf16_product,
    fp32_term,
    group_sum,
    pack_fp32,
)

__all__ = [
    "AMPERE_BF16_DOT16_TOTAL",
    "HOPPER_BF16_WGMMA_K16",
    "HOPPER_WGMMA_FAMILY_VERSION",
    "BF16_CANONICAL_NAN",
    "FP32_CANONICAL_NAN",
    "Inf",
    "NaN",
    "Value",
    "bf16_class",
    "bf16_product_total",
    "cvt_rn_bf16_f32",
    "fp32_class",
    "fp32_value",
    "group_step_total",
    "pack_value",
    "tc_dot_total",
]

FP32_CANONICAL_NAN = 0x7FFFFFFF
BF16_CANONICAL_NAN = 0x7FFF
FP32_INF = 0x7F800000


@dataclass(frozen=True, slots=True)
class NaN:
    """The NaN value (payload-free: the step produces one canonical word)."""


@dataclass(frozen=True, slots=True)
class Inf:
    """A signed infinity."""

    negative: bool


Value = NaN | Inf | Term
NAN = NaN()


def bf16_class(word: int) -> str:
    """``"nan"``, ``"inf"``, ``"zero"`` or ``"finite"`` (nonzero finite, subnormals included)."""

    if not 0 <= word < 1 << 16:
        raise InvalidArtifact("BF16 operand outside 16 bits")
    exponent, mantissa = (word >> 7) & 0xFF, word & 0x7F
    if exponent == 0xFF:
        return "nan" if mantissa else "inf"
    return "zero" if exponent == 0 and mantissa == 0 else "finite"


def fp32_class(word: int) -> str:
    if not 0 <= word < 1 << 32:
        raise InvalidArtifact("FP32 word outside 32 bits")
    exponent, mantissa = (word >> 23) & 0xFF, word & 0x7F_FFFF
    if exponent == 0xFF:
        return "nan" if mantissa else "inf"
    return "zero" if exponent == 0 and mantissa == 0 else "finite"


def bf16_product_total(a: int, b: int) -> Value:
    """The product of two BF16 words as a :data:`Value` (see the module docstring)."""

    ca, cb = bf16_class(a), bf16_class(b)
    if ca == "nan" or cb == "nan":
        return NAN
    if ca == "inf" or cb == "inf":
        if ca == "zero" or cb == "zero":
            return NAN
        return Inf(((a ^ b) >> 15) & 1 == 1)
    return bf16_product(a, b)


def fp32_value(bits: int) -> Value:
    """An FP32 accumulator word as a :data:`Value`."""

    c = fp32_class(bits)
    if c == "nan":
        return NAN
    if c == "inf":
        return Inf(bits >> 31 == 1)
    return fp32_term(bits)


def group_step_total(
    acc: Value, products: Sequence[Value], *, width: int, zero_exponent: int
) -> Value:
    """One group: the incoming value and the group's products to the outgoing value."""

    terms = [acc, *products]
    if any(isinstance(t, NaN) for t in terms):
        return NAN
    signs = {t.negative for t in terms if isinstance(t, Inf)}
    if len(signs) == 2:
        return NAN
    if signs:
        return Inf(signs.pop())
    total = group_sum(
        [t for t in terms if isinstance(t, Term)],
        width=width,
        zero_exponent=zero_exponent,
    )
    if total.magnitude and total.exponent > FP32_MAX_EXPONENT:
        return Inf(total.negative)
    return total


def pack_value(value: Value) -> int:
    if isinstance(value, NaN):
        return FP32_CANONICAL_NAN
    if isinstance(value, Inf):
        return (int(value.negative) << 31) | FP32_INF
    return pack_fp32(value)


def tc_dot_total(
    pipeline: Pipeline, acc_bits: int, a: Sequence[int], b: Sequence[int]
) -> int:
    """``D = acc + sum_k a[k] * b[k]`` on every encoding, as ``pipeline``'s silicon computes it."""

    if pipeline.dtype != "bf16":
        raise InvalidArtifact("the total semantics are defined for BF16 pipelines")
    if len(a) != pipeline.k or len(b) != pipeline.k:
        raise InvalidArtifact(f"{pipeline.name} takes {pipeline.k} operand pairs")
    acc = fp32_value(acc_bits)
    start = 0
    for size in pipeline.groups:
        products = [bf16_product_total(a[i], b[i]) for i in range(start, start + size)]
        acc = group_step_total(
            acc, products, width=pipeline.width, zero_exponent=pipeline.zero_exponent
        )
        start += size
    return pack_value(acc)


def cvt_rn_bf16_f32(bits: int) -> int:
    """``cvt.rn.bf16.f32``: RNE on finite words (overflow to ``±inf``), ``±inf`` kept, every NaN to ``0x7FFF``."""

    if not 0 <= bits < 1 << 32:
        raise InvalidArtifact("FP32 word outside 32 bits")
    if fp32_class(bits) == "nan":
        return BF16_CANONICAL_NAN
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


# --- the registered family -------------------------------------------------------

#: The registered total function: Hawkeye's Ampere finite model plus the special-value rules
#: above, measured on the A100 (sm_80) and the RTX 4090 (sm_89) -- both conform, so a single
#: version serves both architectures (no ``AdaBF16Dot16`` variant is needed).
AMPERE_BF16_DOT16_TOTAL = AMPERE_BF16_M16N8K16

TOTAL_FAMILY_VERSION = "3"

#: :data:`verity.ml.circuits.silicon.models.HOPPER_BF16_WGMMA_K16` (the model of ``sm90.wgmma.m64n8k16.bf16``) is re-exported
#: here, where lane HP's campaign first registered it: all sixteen products and the accumulator in one grouped
#: sum, 26-bit adder, exponent floor ``-133``, the parameters of :data:`~verity.ml.circuits.silicon.silicon.HOPPER_BF16_M16N8K16`.
HOPPER_WGMMA_FAMILY_VERSION = "1"
