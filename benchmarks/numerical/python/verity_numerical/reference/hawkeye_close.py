"""The source-close evaluator: Hawkeye's Ampere BF16 simulator as defined integer operations.

A line-by-line port of the pinned simulator (``verity/ml/circuits/silicon/hawkeye/PIN.json``: https://github.com/badasherez/gpu-simulator at
``30703fcb309c943a6df5eee0277cb81815deb8f4``) restricted to the routines the transition unit uses:

* ``utils/utils.cpp::multiply_bfloat16_to_gfloat``          -> :func:`multiply_bfloat16_to_gfloat`
* ``src/Ampere_simulator.cpp::Ampere_simulator::group_sum`` -> :func:`group_sum`
* ``src/gfloat.cpp::Gfloat::operator float``                -> :func:`gfloat_to_float_bits`
* ``utils/utils.h`` constants                               -> :data:`E_ZERO`, :data:`W_WORK`, :data:`W_FP32`, :data:`N_GROUP`

The port keeps the C++ names, statement order and branch structure so that it can be read against the source.  Everything is a
Python ``int``; every shift is of a non-negative value by a non-negative amount.  Where the C++ relies on fixed-width behaviour
(``int32_t`` significand, ``int16_t`` exponent, the ``shift >= 32`` guard) the comment at that line states the bound under which
the unbounded integer computes the same value.  Nothing here is generalised: the group structure (acc + 8 products, twice) and the
constants are Ampere's.

What is *not* in the source and is added here, each stated as such:

* :func:`fp32_to_gfloat`, the **arbitrary-accumulator adapter** (log entry D7).  The pinned matrix wrapper
  (``Ampere_simulator::matrix_multiply_template``) only ever starts a coordinate from ``Gfloat(false, E_ZERO, 0)``; the transition
  unit of the target takes any finite FP32 word ``c``.  The adapter decodes ``c`` into the Gfloat that the simulator's own
  ``group_sum`` would have produced for that value (it is the left inverse of ``Gfloat::operator float`` on every ``group_sum``
  output; the zero's internal sign, which ``operator float`` discards, is the only information lost, and a zero term's sign
  never influences ``group_sum``).  See :func:`tc_dot16`.
* :func:`wrapper_element`, the pinned wrapper's per-coordinate loop verbatim (zero start, groups of ``N_GROUP``, and the
  wrapper's **finalisation**: one more ``group_sum`` over the single remaining Gfloat, then ``operator float``).  The finalisation
  is a fixed point of ``group_sum`` -- :func:`tc_dot16` omits it, and ``tests/reference`` checks the omission (``group_sum([g]) == g``
  for every reachable ``g``).
* :func:`normalise_closed_form` and :func:`group_sum_closed_form`, the closed-form normalisation of log entry D3
  (``e' = max(E + l - 25, -126)``, ``M' = floor(v * 2^(E - e' - 1))``) as a separate function, to be compared against the
  successive shifts of :func:`group_sum`.
* :func:`is_finite_bf16` / :func:`is_finite_fp32`.  The simulator has no NaN/infinity branch anywhere: an exponent field ``0xFF``
  is treated as the exponent ``128`` with an implicit leading one, and ``Gfloat::operator float`` masks the exponent field to
  8 bits instead of saturating.  :func:`tc_dot16` rejects non-finite words up front (the target's domain); the faithful routines
  below do not.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

__all__ = [
    "E_ZERO",
    "FP32_MIN_NONZERO_EXPONENT",
    "Gfloat",
    "N_GROUP",
    "NonFinite",
    "W_FP32",
    "W_WORK",
    "fp32_to_gfloat",
    "gfloat_to_float_bits",
    "group_sum",
    "group_sum_closed_form",
    "is_finite_bf16",
    "is_finite_fp32",
    "multiply_bfloat16_to_gfloat",
    "normalise_closed_form",
    "tc_dot16",
    "wrapper_element",
]

# utils/utils.h
E_ZERO = -132  # AMPERE_ZERO_EXPONENT: the exponent of every zero Gfloat and the floor of the group's maximum exponent
W_WORK = 25  # AMPERE_SIGNIFICAND_WIDTH: the adder's significand width
N_GROUP = 9  # AMPERE_ACCUMULATOR_GROUP_SIZE: the accumulator plus eight products
W_FP32 = 24  # FP32_SIGNIFICAND_WIDTH
FP32_MIN_NONZERO_EXPONENT = -126


class Gfloat(NamedTuple):
    """``class Gfloat`` (include/gfloat.h): ``sign_``, ``exponent_`` (int16_t), ``significand_`` (int32_t)."""

    sign: bool
    exponent: int
    significand: int


class NonFinite(ValueError):
    """A NaN or infinity word: outside the target's domain and outside what the simulator models."""


def is_finite_bf16(word: int) -> bool:
    return ((word >> 7) & 0xFF) != 0xFF


def is_finite_fp32(bits: int) -> bool:
    return ((bits >> 23) & 0xFF) != 0xFF


# --- utils/utils.cpp ---------------------------------------------------------------------------------------------------------


def multiply_bfloat16_to_gfloat(a: int, b: int, zero_exponent: int = E_ZERO) -> Gfloat:
    """``Gfloat multiply_bfloat16_to_gfloat(uint16_t a, uint16_t b, int16_t zero_exponent)``.

    No NaN/infinity branch, as in the source: an exponent field ``0xFF`` contributes ``255`` to the exponent sum and an implicit
    leading one to the significand.
    """

    sign_a = (a >> 15) & 1
    exp_bits_a = (a >> 7) & 0xFF
    sig_bits_a = a & 0x7F

    sign_b = (b >> 15) & 1
    exp_bits_b = (b >> 7) & 0xFF
    sig_bits_b = b & 0x7F

    significand_a = (sig_bits_a | 0x80) if exp_bits_a != 0 else sig_bits_a
    significand_b = (sig_bits_b | 0x80) if exp_bits_b != 0 else sig_bits_b
    if exp_bits_a == 0:
        exp_bits_a = 1
    if exp_bits_b == 0:
        exp_bits_b = 1
    result_significand = (significand_a * significand_b) << 9  # < 2^16 << 9 = 2^25: fits the int32_t

    result_exponent = exp_bits_a + exp_bits_b - 254  # in [-252, 256]: fits the int16_t

    result_sign = bool(sign_a ^ sign_b)

    if result_significand == 0:
        return Gfloat(result_sign, zero_exponent, 0)

    return Gfloat(result_sign, result_exponent, result_significand)


# --- src/Ampere_simulator.cpp ------------------------------------------------------------------------------------------------


def group_sum(array: Sequence[Gfloat]) -> Gfloat:
    """``Gfloat Ampere_simulator::group_sum(const std::vector<Gfloat>& array)``: the successive-shift form.

    Fixed-width facts the port relies on (all hold for at most 32 terms whose significands are below ``2^25``, i.e. for every
    input the transition unit forms): each aligned term is below ``2^26`` and the signed sum below ``2^31`` in magnitude, so the
    C++ ``int32_t`` never overflows and its right shifts act on non-negative values; the ``shift >= 32`` guard keeps the C++
    shift defined, and a Python shift by 32 or more of a value below ``2^26`` is ``0`` -- the same contribution; the
    denormalising shift ``FP32_MIN_NONZERO_EXPONENT - exp`` is at most ``30`` because ``max_exponent >= E_ZERO`` and
    ``width >= 1``.
    """

    ampere_to_fp32_shift = W_WORK - W_FP32

    # Step 1: Find the maximal exponent of the Gfloats
    max_exponent = E_ZERO
    for gfloat in array:
        if gfloat.exponent > max_exponent:  # a zero Gfloat carries E_ZERO, so it never raises the maximum
            max_exponent = gfloat.exponent

    significand = 0
    for gfloat in array:
        shift = max_exponent - gfloat.exponent
        if shift >= 32:
            continue
        if gfloat.sign:
            significand -= (gfloat.significand << ampere_to_fp32_shift) >> shift
        else:
            significand += (gfloat.significand << ampere_to_fp32_shift) >> shift
    sign = significand < 0
    significand = abs(significand)
    width = 0 if significand == 0 else significand.bit_length()  # 32 - __builtin_clz(significand)
    if width == 0:
        return Gfloat(sign, E_ZERO, 0)
    exp = max_exponent + width - W_WORK
    if width > W_WORK:
        significand = significand >> (width - W_WORK)
    else:
        significand = significand << (W_WORK - width)
    if exp < FP32_MIN_NONZERO_EXPONENT:
        significand = significand >> (FP32_MIN_NONZERO_EXPONENT - exp)
        exp = FP32_MIN_NONZERO_EXPONENT
    significand = significand >> ampere_to_fp32_shift
    if significand == 0:
        return Gfloat(sign, E_ZERO, 0)  # a magnitude truncated to zero keeps its computed sign internally
    return Gfloat(sign, exp, significand)


def wrapper_element(a_row: Sequence[int], b_col: Sequence[int]) -> int:
    """One output coordinate of ``Ampere_simulator::matrix_multiply_template`` (the ``compute_range`` lambda), verbatim.

    The accumulator vector starts as ``[Gfloat(false, E_ZERO, 0)]``; each product is appended and a full vector of ``N_GROUP``
    is collapsed by ``group_sum`` into its single first element; after the last product the wrapper applies ``group_sum`` once
    more to whatever remains (for ``k = 16`` that is the single Gfloat the second group produced) and casts it to ``float``.
    That last ``group_sum`` is the *finalisation*; :func:`tc_dot16` omits it.
    """

    if len(a_row) != len(b_col):
        raise ValueError("Matrix dimensions don't match for multiplication")
    accumulator = [Gfloat(False, E_ZERO, 0)]
    for l in range(len(a_row)):
        product = multiply_bfloat16_to_gfloat(a_row[l], b_col[l], E_ZERO)
        accumulator.append(product)
        if len(accumulator) == N_GROUP:
            total = group_sum(accumulator)
            accumulator = [total]
    final_result = group_sum(accumulator)
    return gfloat_to_float_bits(final_result)


# --- src/gfloat.cpp ----------------------------------------------------------------------------------------------------------


def gfloat_to_float_bits(gfloat: Gfloat) -> int:
    """``Gfloat::operator float() const``, returning the 32 bits it reinterprets.

    The exponent field is ``(exponent_ + 127 - [bit 23 clear]) & 0xFF`` -- masked, not saturated: an exponent above ``127``
    wraps into the infinity/NaN field (``128``) or into a small finite field (``129`` and above).  A zero significand returns
    ``+0.0f`` whatever the internal sign.
    """

    if gfloat.significand == 0:
        return 0

    float_bits = 0

    if gfloat.sign:
        float_bits |= 0x80000000

    exponent_bits = gfloat.exponent + 127

    if (gfloat.significand & 0x800000) == 0:
        exponent_bits -= 1

    float_bits |= (exponent_bits & 0xFF) << 23  # Python's & on a negative int is two's complement, as the C++ int's is

    float_bits |= gfloat.significand & 0x7FFFFF

    return float_bits


# --- not in the source: the arbitrary-accumulator adapter (D7) -----------------------------------------------------------------


def fp32_to_gfloat(bits: int) -> Gfloat:
    """An FP32 word as the Gfloat ``group_sum`` would have produced for it (the left inverse of :func:`gfloat_to_float_bits`).

    Zero (either sign) is the simulator's zero ``Gfloat(sign, E_ZERO, 0)``: it must carry ``E_ZERO`` because ``group_sum``
    takes the exponent of *every* term, zero or not, into its maximum.  A subnormal keeps its stored significand at exponent
    ``-126`` (as ``group_sum`` leaves a denormalised result and as ``multiply_bfloat16_to_gfloat`` treats a subnormal operand).
    A normal word gets its implicit leading one.  Non-finite words raise: the simulator does not model them.
    """

    exponent_field = (bits >> 23) & 0xFF
    if exponent_field == 0xFF:
        raise NonFinite(f"non-finite FP32 accumulator {bits:#010x}")
    sign = bool((bits >> 31) & 1)
    mantissa = bits & 0x7FFFFF
    if exponent_field == 0:
        if mantissa == 0:
            return Gfloat(sign, E_ZERO, 0)
        return Gfloat(sign, FP32_MIN_NONZERO_EXPONENT, mantissa)
    return Gfloat(sign, exponent_field - 127, (1 << 23) | mantissa)


def tc_dot16(acc_bits: int, a: Sequence[int], b: Sequence[int]) -> int:
    """The transition unit ``tc_dot16(c, a[0:16], b[0:16])`` through the simulator's routines with an arbitrary finite ``c``.

    ``group_sum([c, p0..p7])`` then ``group_sum([., p8..p15])`` then ``operator float`` -- the pinned wrapper's element loop with
    the zero start replaced by :func:`fp32_to_gfloat` and without the wrapper's finalisation (see :func:`wrapper_element`).
    Non-finite operands or accumulators raise :class:`NonFinite`; the result word is whatever ``operator float`` yields, so a
    result whose exponent exceeds ``127`` is returned masked, as the simulator would (the hardware saturates; see the log).
    """

    if len(a) != 16 or len(b) != 16:
        raise ValueError("tc_dot16 takes 16 operand pairs")
    for word in (*a, *b):
        if not is_finite_bf16(word):
            raise NonFinite(f"non-finite BF16 operand {word:#06x}")
    accumulator = fp32_to_gfloat(acc_bits)
    for start in (0, 8):
        group = [accumulator]
        for l in range(start, start + 8):
            group.append(multiply_bfloat16_to_gfloat(a[l], b[l], E_ZERO))
        assert len(group) == N_GROUP
        accumulator = group_sum(group)
    return gfloat_to_float_bits(accumulator)


# --- not in the source: the closed-form normalisation (D3) ---------------------------------------------------------------------


def normalise_closed_form(sign: bool, max_exponent: int, magnitude: int) -> Gfloat:
    """The normalisation stage of ``group_sum`` in closed form: ``e' = max(E + l - 25, -126)``, ``M' = floor(v * 2^(E - e' - 1))``.

    ``E`` is the group's maximum exponent, ``v = |S|`` the aligned integer sum on the 25-bit grid, ``l`` its bit length.
    A left shift (``E - e' - 1 >= 0``) is exact; a right shift is the floor.  The zero rules are ``group_sum``'s.
    """

    if magnitude == 0:
        return Gfloat(False, E_ZERO, 0)
    l = magnitude.bit_length()
    e = max(max_exponent + l - W_WORK, FP32_MIN_NONZERO_EXPONENT)
    shift = max_exponent - e - 1
    m = magnitude << shift if shift >= 0 else magnitude >> -shift
    if m == 0:
        return Gfloat(sign, E_ZERO, 0)
    return Gfloat(sign, e, m)


def group_sum_closed_form(array: Sequence[Gfloat]) -> Gfloat:
    """``group_sum`` with its alignment and summation unchanged and its normalisation replaced by :func:`normalise_closed_form`."""

    ampere_to_fp32_shift = W_WORK - W_FP32
    max_exponent = E_ZERO
    for gfloat in array:
        if gfloat.exponent > max_exponent:
            max_exponent = gfloat.exponent
    significand = 0
    for gfloat in array:
        shift = max_exponent - gfloat.exponent
        if shift >= 32:
            continue
        aligned = (gfloat.significand << ampere_to_fp32_shift) >> shift
        significand += -aligned if gfloat.sign else aligned
    return normalise_closed_form(significand < 0, max_exponent, abs(significand))
