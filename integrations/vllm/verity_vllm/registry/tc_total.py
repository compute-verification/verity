"""The TOTAL tensor-core k16 step, pure Python (3.11) -- `veritor.core.silicon_total.tc_dot_total` ported gate for gate.

[R17 dense, rel M-0227 / rev M-0287 / revb M-0290] The registered scalar tensor-core primitives (`prims.AmpereBF16TcDot16`,
`hopper.HopperBF16WgmmaDot16`) are the Definitions of record; the Commit replays their rows through the vectorised twins in
`derived_rows`, and the twins are held word-equal to the scalar prims by tests.  Until R17 the scalar prims evaluated through
`src/veritor` (`veritor.core.silicon` / `silicon_total`), whose `type X = ...` statements are Python 3.12 syntax: on every pod
(the Commit venv `/workspace/venv-cu129` is 3.11) the reference raised `SyntaxError`, so "twin == definition" was an assertion
carried over from the 3.12 laptop, never checkable where the rows run.  This module states the relation in Python the pods run;
`veritor.core.silicon_total` stays the REFERENCE the port is held against (verity.ir/tests/test_derived_rows.py, on every
encoding, skipped BY NAME where veritor does not import).

The relation (`silicon_total`, TOTAL_FAMILY_VERSION 3 for Ampere/Ada -- measured on the A100 and the RTX 4090 -- and the
re-measured Hopper wgmma k16 pipeline): `D = acc + sum_k a[k] * b[k]` over the pipeline's groups of products; per group
(`group_step_total`): any NaN among the incoming value and the group's products -> canonical NaN 0x7FFFFFFF; infinities of both
signs -> NaN, of one sign -> that infinity (finite terms ignored); else `silicon.group_sum` over the finite terms (every term
rescaled from the 24-bit FP32 grid to the adder `width`, aligned to the largest exponent (floor `zero_exponent`) by a truncating
right shift, summed exactly, normalised to `width` bits by truncation, denormalised below the FP32 normal range, truncated back
to 24 bits) and a result exponent above 127 -> the infinity of the total's sign (STICKY: a later group cannot bring it back).
Products (`bf16_product_total`): NaN operand -> NaN; inf * zero -> NaN (a subnormal is finite, not zero); inf * finite -> the
infinity of the product's sign; else the exact product on the 25-bit grid (8x8-bit significands << 9, subnormals unnormalised at
exponent -126).  The accumulator word (`fp32_value`): NaN -> NaN, inf -> that infinity, subnormal -> its unnormalised
significand at exponent -126.  Packing (`pack_value` / `pack_fp32`): zero is +0; a magnitude without bit 23 at exponent -126 is
a subnormal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "AMPERE_BF16_M16N8K16",
    "FP32_CANONICAL_NAN",
    "FP32_INF",
    "HOPPER_BF16_WGMMA_K16",
    "HOPPER_WGMMA_FAMILY_VERSION",
    "TOTAL_FAMILY_VERSION",
    "TcPipeline",
    "tc_dot_total",
]

#: the measured total families this module states (`silicon_total.TOTAL_FAMILY_VERSION` / `HOPPER_WGMMA_FAMILY_VERSION`, same strings);
#: [revb M-0461] a record's evaluator provenance names them (`sampled_replay.EVALUATORS`, `TcPipeline.family_version`), so two
#: Definitions of the same name with different non-finite content are told apart on the record itself, not only by versions.json.
TOTAL_FAMILY_VERSION = "3"
HOPPER_WGMMA_FAMILY_VERSION = "1"
FP32_CANONICAL_NAN = 0x7FFFFFFF
FP32_INF = 0x7F800000
_FP32_MIN_NORMAL_EXPONENT = -126
_FP32_MAX_EXPONENT = 127
_FP32_SIGNIFICAND_WIDTH = 24
_PRODUCT_SHIFT = 9  # bf16 8x8-bit significand products carried on the 25-bit grid (silicon.PRODUCT_WIDTH)

# value tags (the `Value = NaN | Inf | Term` of silicon_total, as tuples)
_ZERO, _TERM, _INF, _NAN = 0, 1, 2, 3


@dataclass(frozen=True)
class TcPipeline:
    """The silicon parameters of one k-step (`veritor.core.silicon.Pipeline`'s numerical fields)."""

    name: str
    groups: tuple[int, ...]   # products per grouped accumulation, in issue order; the accumulator joins the first group
    width: int                # internal significand width of the adder
    zero_exponent: int        # exponent floor (the maximum exponent used when every term is zero)
    family_version: str       # the measured total family the pipeline's rules were taken from (silicon_total's version string)

    @property
    def k(self) -> int:
        return sum(self.groups)


#: `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` (sm_80 / sm_89): two groups of eight, 25-bit adder, floor -132
#: (`silicon.AMPERE_BF16_M16N8K16` = `silicon_total.AMPERE_BF16_DOT16_TOTAL`, Hawkeye's parameters; A100 + RTX 4090 conform).
AMPERE_BF16_M16N8K16 = TcPipeline("ampere_bf16_m16n8k16", (8, 8), 25, -132, TOTAL_FAMILY_VERSION)

#: `wgmma.mma_async.sync.aligned.m64n{N}k16.f32.bf16.bf16` (sm_90): one group of all sixteen products + the accumulator, 26-bit
#: adder, floor -133 (`silicon_total.HOPPER_BF16_WGMMA_K16`; lane HP campaign, H100, 2026-09-07: 37,510 tiles x 8,192 outputs x 3).
HOPPER_BF16_WGMMA_K16 = TcPipeline("hopper_bf16_wgmma_k16", (16,), 26, -133, HOPPER_WGMMA_FAMILY_VERSION)


def _fp32_value(bits: int) -> tuple:
    if not 0 <= bits < 1 << 32:
        raise ValueError("FP32 word outside 32 bits")
    exponent_field, mantissa = (bits >> 23) & 0xFF, bits & 0x7F_FFFF
    negative = bits >> 31 == 1
    if exponent_field == 0xFF:
        return (_NAN,) if mantissa else (_INF, negative)
    if exponent_field == 0:
        return (_TERM, negative, _FP32_MIN_NORMAL_EXPONENT, mantissa) if mantissa else (_ZERO,)
    return (_TERM, negative, exponent_field - 127, (1 << 23) | mantissa)


def _bf16_product_total(x: int, y: int) -> tuple:
    if not (0 <= x < 1 << 16 and 0 <= y < 1 << 16):
        raise ValueError("BF16 operand outside 16 bits")
    ex, ey = (x >> 7) & 0xFF, (y >> 7) & 0xFF
    xn, yn = ex == 0xFF, ey == 0xFF
    if xn or yn:
        # a NaN operand, or inf * zero (a subnormal is finite, not zero) -> NaN; else the infinity of the product's sign
        if (xn and x & 0x7F) or (yn and y & 0x7F) or (x & 0x7FFF) == 0 or (y & 0x7FFF) == 0:
            return (_NAN,)
        return (_INF, ((x ^ y) >> 15) & 1 == 1)
    sx = (x & 0x7F) | (0x80 if ex else 0)
    sy = (y & 0x7F) | (0x80 if ey else 0)
    magnitude = (sx * sy) << _PRODUCT_SHIFT
    if magnitude == 0:
        return (_ZERO,)
    return (_TERM, ((x ^ y) >> 15) & 1 == 1, (ex or 1) + (ey or 1) - 254, magnitude)


def _group_step_total(acc: tuple, products: list, width: int, zero_exponent: int) -> tuple:
    terms = [acc, *products]
    if any(t[0] == _NAN for t in terms):
        return (_NAN,)
    signs = {t[1] for t in terms if t[0] == _INF}
    if len(signs) == 2:
        return (_NAN,)
    if signs:
        return (_INF, signs.pop())
    # silicon.group_sum over the finite nonzero terms
    finite = [t for t in terms if t[0] == _TERM]
    rescale = width - _FP32_SIGNIFICAND_WIDTH
    max_exponent = max([zero_exponent, *(t[2] for t in finite)])
    total = 0
    for _, negative, exponent, magnitude in finite:
        scaled = magnitude << rescale if rescale >= 0 else magnitude >> -rescale
        aligned = scaled >> (max_exponent - exponent)
        total += -aligned if negative else aligned
    if total == 0:
        return (_ZERO,)
    negative = total < 0
    magnitude = -total if negative else total
    bit_length = magnitude.bit_length()
    exponent = max_exponent + bit_length - width
    if bit_length > width:
        magnitude >>= bit_length - width
    else:
        magnitude <<= width - bit_length
    if exponent < _FP32_MIN_NORMAL_EXPONENT:
        magnitude >>= _FP32_MIN_NORMAL_EXPONENT - exponent
        exponent = _FP32_MIN_NORMAL_EXPONENT
    magnitude = magnitude >> rescale if rescale >= 0 else magnitude << -rescale
    if magnitude == 0:
        return (_ZERO,)
    if exponent > _FP32_MAX_EXPONENT:  # group_step_total's overflow rule: sticky infinity
        return (_INF, negative)
    return (_TERM, negative, exponent, magnitude)


def _pack_value(value: tuple) -> int:
    tag = value[0]
    if tag == _NAN:
        return FP32_CANONICAL_NAN
    if tag == _INF:
        return (int(value[1]) << 31) | FP32_INF
    if tag == _ZERO:
        return 0
    _, negative, exponent, magnitude = value
    if exponent > _FP32_MAX_EXPONENT:
        return (int(negative) << 31) | FP32_INF
    exponent_field = exponent + 127
    if magnitude & (1 << 23) == 0:
        exponent_field -= 1
    return (int(negative) << 31) | (exponent_field << 23) | (magnitude & 0x7F_FFFF)


def tc_dot_total(pipeline: TcPipeline, acc_bits: int, a: Sequence[int], b: Sequence[int]) -> int:
    """`D = acc + sum_k a[k] * b[k]` on EVERY encoding, as `pipeline`'s silicon computes one accumulator coordinate
    (`veritor.core.silicon_total.tc_dot_total`, gate for gate)."""
    if len(a) != pipeline.k or len(b) != pipeline.k:
        raise ValueError(f"{pipeline.name} takes {pipeline.k} operand pairs (got {len(a)}, {len(b)})")
    acc = _fp32_value(acc_bits)
    start = 0
    for size in pipeline.groups:
        products = [_bf16_product_total(a[i], b[i]) for i in range(start, start + size)]
        acc = _group_step_total(acc, products, pipeline.width, pipeline.zero_exponent)
        start += size
    return _pack_value(acc)
