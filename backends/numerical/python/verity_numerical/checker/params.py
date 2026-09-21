"""Widths of the transition unit as data, so that every gadget, contract and search is parametric.

``REAL`` is the frozen target (:data:`verity.ml.tc.silicon.AMPERE_BF16_M16N8K16` seen through the checker's
decomposition); ``TOY`` and ``TINY`` are reduced widths for exhaustive search.  Every derived quantity below is a
*bound* used by a contract; the bounds are conservative (a table may have a few rows nobody reaches) and are the
same formulas at every width.
"""
from __future__ import annotations

from dataclasses import dataclass


class Reject(Exception):
    """The checker refuses (x, y, h): the message names the first violated predicate."""


@dataclass(frozen=True)
class Params:
    sig: int = 8               # operand significand bits, hidden bit included (BF16: 8)
    exp: int = 8               # exponent field bits, shared by operand and accumulator formats (8)
    acc_sig: int = 24          # accumulator significand bits, hidden bit included (FP32: 24)
    width: int = 25            # adder / normalisation width W (Ampere: 25)
    floor: int = -132          # exponent floor: the group maximum when every term is zero (Ampere: -132)
    groups: tuple[int, ...] = (8, 8)

    def __post_init__(self) -> None:
        if self.sig < 2 or self.exp < 2 or self.acc_sig < 2 or not self.groups:
            raise ValueError("degenerate widths")
        if self.width < self.acc_sig or self.width < 2 * self.sig - 1:
            raise ValueError("the adder must be at least as wide as the accumulator and the product")
        if self.floor > self.e_min:
            raise ValueError("the floor must not exceed the smallest normal exponent")

    # --- formats ---------------------------------------------------------------------------------------------------
    @property
    def bias(self) -> int:
        return 2 ** (self.exp - 1) - 1

    @property
    def e_min(self) -> int:               # smallest normal exponent; also the exponent of every subnormal
        return 1 - self.bias

    @property
    def e_max(self) -> int:               # largest finite exponent; e' above it saturates
        return self.bias

    @property
    def t_finite(self) -> int:            # exponent fields 0 .. t_finite-1 are finite; t_finite (all ones) is not
        return 2**self.exp - 1

    @property
    def frac(self) -> int:
        return self.sig - 1

    @property
    def acc_frac(self) -> int:
        return self.acc_sig - 1

    @property
    def word_bits(self) -> int:           # operand word: sign, exponent, fraction
        return 1 + self.exp + self.frac

    @property
    def acc_word_bits(self) -> int:       # accumulator word
        return 1 + self.exp + self.acc_frac

    # --- the adder grid ----------------------------------------------------------------------------------------------
    @property
    def prod_shift(self) -> int:          # a product u (2*sig bits) enters the adder as u * 2^prod_shift
        return self.width - 2 * self.sig + 1

    @property
    def prod_bits(self) -> int:           # u * 2^prod_shift < 2^prod_bits (= W + 1)
        return 2 * self.sig + self.prod_shift

    @property
    def acc_shift(self) -> int:           # a state magnitude M (acc_sig bits) enters as M * 2^acc_shift
        return self.width - self.acc_sig

    @property
    def acc_bits(self) -> int:            # M * 2^acc_shift < 2^acc_bits (= W)
        return self.acc_sig + self.acc_shift

    # --- exponent ranges ---------------------------------------------------------------------------------------------
    @property
    def ep_lo(self) -> int:               # smallest nonzero product exponent (two subnormals)
        return 2 * self.e_min

    @property
    def ep_hi(self) -> int:
        return 2 * self.e_max

    @property
    def exp_lo(self) -> int:              # smallest masked term exponent anywhere
        return min(self.floor, self.ep_lo)

    @property
    def n(self) -> int:
        return max(self.groups)

    @property
    def s_max(self) -> int:               # |S| <= s_max: one state term plus n products, unshifted
        return (2**self.acc_bits - 2**self.acc_shift) + self.n * (2**self.prod_bits - 2**self.prod_shift)

    @property
    def s_bits(self) -> int:              # |S| < 2^s_bits (30 on Ampere): the premise of the exact sum
        return self.s_max.bit_length()

    @property
    def l_hi(self) -> int:                # leading-bit position l in [0, l_hi]
        return self.s_bits

    @property
    def growth(self) -> int:              # a group can raise the exponent by at most this much: l_hi - W
        return self.l_hi - self.width

    def e_hi(self, g: int) -> int:
        """Largest group maximum E of group ``g`` (0-based): products reach ep_hi; the state grows by ``growth`` per group."""
        return max(self.e_max, self.ep_hi) + g * self.growth

    @property
    def exp_hi(self) -> int:              # largest masked term exponent anywhere (a state after the last group)
        return self.e_hi(len(self.groups) - 1) + self.growth

    @property
    def shift_rows(self) -> int:          # alignment shift d in [0, shift_rows): 512 on Ampere
        return self.e_hi(len(self.groups) - 1) - self.exp_lo + 1

    @property
    def cmp_rows(self) -> int:
        """Every comparison the checker proves by a flag has |difference| < cmp_rows (512 on Ampere)."""
        last = self.e_hi(len(self.groups) - 1)
        spans = (
            self.e_min - self.floor + self.width,                   # clamp: e_raw below e_min
            last + self.l_hi - self.width - self.e_min,             # clamp: e_raw above e_min
            self.exp_hi - self.e_max,                               # overflow: e' above e_max
            self.e_max - self.e_min,                                # overflow: e' below e_max
        )
        return max(self.shift_rows, max(spans) + 1)

    # --- normalisation -----------------------------------------------------------------------------------------------
    @property
    def k_lo(self) -> int:                # M' = floor(v * 2^k), k = E - e' + acc_sig - W; smallest k (clamped, E = floor)
        return min(self.acc_sig - self.l_hi, self.floor - self.e_min + self.acc_sig - self.width)

    @property
    def k_hi(self) -> int:                # largest k: l = 0, unclamped
        return self.acc_sig

    @property
    def norm_rem_bits(self) -> int:       # the normalisation remainder r_n < 2^(-k) <= 2^norm_rem_bits
        return -self.k_lo


REAL = Params()
#: Reduced widths for the exhaustive searches: 3-bit significands, two groups of two.
TOY = Params(sig=3, exp=3, acc_sig=5, width=6, floor=-8, groups=(2, 2))
#: Smaller still, for the gadgets whose joint hint space is otherwise too large to enumerate in seconds.
TINY = Params(sig=2, exp=3, acc_sig=4, width=5, floor=-4, groups=(2, 2))
