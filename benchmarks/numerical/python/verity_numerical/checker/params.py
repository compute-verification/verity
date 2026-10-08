"""Widths of the transition unit as data, so that every gadget, contract and search is parametric.

``REAL`` is the frozen target (:data:`verity.ml.circuits.silicon.silicon.AMPERE_BF16_M16N8K16` seen through the checker's
decomposition); ``TOY`` and ``TINY`` are reduced widths for exhaustive search.  Every derived quantity below is a
*bound* used by a contract; the bounds are conservative (a table may have a few rows nobody reaches) and are the
same formulas at every width.

Two formats meet in a step: the *operand* format (``sig``, ``exp``, ``op_nan_rule``) and the *accumulator* format
(``acc_sig``, ``acc_exp``); BF16 -> FP32 share the exponent width, FP8 -> FP32 do not.  The unqualified names
(``bias``, ``e_min``, ``e_max``, ``t_finite``) are the operand's, the ``acc_`` names the accumulator's; at ``REAL``
they coincide, so nothing written for BF16 changes meaning.

An adder narrower than the accumulator (``width < acc_sig``: Hopper/Ada FP8, 14 < 24) is the *lossy* case: the
incoming accumulator's significand is truncated to ``width`` bits before the grouped add (:class:`RescaleState`),
the normalised magnitude has ``width`` bits and is scaled back up exactly (``norm_up``).  At ``REAL`` (25 >= 24)
both are the identity and the witness layout is the historical one, byte for byte.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover
    from verity.ml.circuits.silicon.models import Model


class Reject(Exception):
    """The checker refuses (x, y, h): the message names the first violated predicate."""


@dataclass(frozen=True)
class Params:
    sig: int = 8               # operand significand bits, hidden bit included (BF16: 8; E4M3: 4; E5M2: 3)
    exp: int = 8               # operand exponent field bits (BF16: 8; E4M3: 4; E5M2: 5)
    acc_sig: int = 24          # accumulator significand bits, hidden bit included (FP32: 24)
    width: int = 25            # adder / normalisation width W (Ampere: 25; Hopper/Ada FP8: 14)
    floor: int = -132          # exponent floor: the group maximum when every term is zero (Ampere: -132)
    groups: tuple[int, ...] = (8, 8)
    acc_exp: int = 0           # accumulator exponent field bits; 0 means "the operand's" (BF16 -> FP32: 8 = 8)
    op_nan_rule: Literal["ieee", "e4m3"] = "ieee"   # which operand words are not finite (see ``t_max_finite``)

    def __post_init__(self) -> None:
        if self.acc_exp == 0:
            object.__setattr__(self, "acc_exp", self.exp)
        if self.sig < 2 or self.exp < 2 or self.acc_sig < 2 or self.acc_exp < 2 or not self.groups:
            raise ValueError("degenerate widths")
        if self.width < 2 * self.sig - 1:
            raise ValueError("the adder must be at least as wide as the product")
        if self.op_nan_rule not in ("ieee", "e4m3"):
            raise ValueError(f"unknown operand NaN rule {self.op_nan_rule!r}")
        if self.floor > self.acc_e_min:
            raise ValueError("the floor must not exceed the accumulator's smallest normal exponent")

    @classmethod
    def from_model(cls, model: Model) -> Params:
        """The checker's widths of a :class:`verity.ml.circuits.silicon.models.GroupSum` (FP32 accumulator).

        ``Params.from_model(AMPERE_BF16_M16N8K16) == REAL``; ``HOPPER_E4M3_K32`` / ``ADA_E4M3_M16N8K32`` give the
        lossy FP8 parameters.  Block-scaled models have no grouped-sum decomposition and are refused."""
        from verity.ml.circuits.silicon.models import GroupSum
        from verity.ml.circuits.silicon.term import FORMATS

        if not isinstance(model, GroupSum):
            raise ValueError(f"{getattr(model, 'name', model)!r} is not a GroupSum: the checker has no gadgets for it")
        fmt = FORMATS[model.dtype]
        if fmt.nan_rule == "none":
            raise ValueError(f"{model.name}: operand format {fmt.name} has no exponent field the decode gadget can use")
        return cls(sig=fmt.sig_bits, exp=fmt.exp_bits, acc_sig=24, width=model.width, floor=model.zero_exponent,
                   groups=tuple(model.groups), acc_exp=8, op_nan_rule=fmt.nan_rule)

    # --- operand format ----------------------------------------------------------------------------------------------
    @property
    def bias(self) -> int:
        return 2 ** (self.exp - 1) - 1

    @property
    def e_min(self) -> int:               # smallest normal exponent; also the exponent of every subnormal
        return 1 - self.bias

    @property
    def t_finite(self) -> int:            # exponent fields 0 .. t_finite-1 are finite for "ieee"; t_finite is all ones
        return 2**self.exp - 1

    @property
    def t_max_finite(self) -> int:        # the largest exponent field a finite operand can carry
        return self.t_finite if self.op_nan_rule == "e4m3" else self.t_finite - 1

    @property
    def e_max(self) -> int:               # largest finite operand exponent (BF16: 127; E4M3: 8, the 448 word)
        return self.t_max_finite - self.bias

    @property
    def frac(self) -> int:
        return self.sig - 1

    @property
    def word_bits(self) -> int:           # operand word: sign, exponent, fraction
        return 1 + self.exp + self.frac

    # --- accumulator format ------------------------------------------------------------------------------------------
    @property
    def acc_bias(self) -> int:
        return 2 ** (self.acc_exp - 1) - 1

    @property
    def acc_e_min(self) -> int:           # smallest normal accumulator exponent (FP32: -126); the subnormal clamp
        return 1 - self.acc_bias

    @property
    def acc_e_max(self) -> int:           # largest finite accumulator exponent; e' above it saturates (FP32: 127)
        return self.acc_bias

    @property
    def acc_t_finite(self) -> int:        # the all-ones accumulator exponent field: infinities and NaN
        return 2**self.acc_exp - 1

    @property
    def acc_frac(self) -> int:
        return self.acc_sig - 1

    @property
    def acc_word_bits(self) -> int:       # accumulator word
        return 1 + self.acc_exp + self.acc_frac

    # --- the adder grid ----------------------------------------------------------------------------------------------
    @property
    def prod_shift(self) -> int:          # a product u (2*sig bits) enters the adder as u * 2^prod_shift
        return self.width - 2 * self.sig + 1

    @property
    def prod_bits(self) -> int:           # u * 2^prod_shift < 2^prod_bits (= W + 1)
        return 2 * self.sig + self.prod_shift

    @property
    def lossy(self) -> bool:              # the adder is narrower than the accumulator: the state is truncated first
        return self.width < self.acc_sig

    @property
    def acc_drop(self) -> int:            # accumulator significand bits dropped by the rescale (FP8: 10; BF16: 0)
        return max(0, self.acc_sig - self.width)

    @property
    def acc_shift(self) -> int:           # a state magnitude M (acc_sig bits) enters as M * 2^acc_shift (< 0 when lossy)
        return self.width - self.acc_sig

    @property
    def state_shift(self) -> int:         # the left shift AlignState applies to the (rescaled) state: max(acc_shift, 0)
        return max(0, self.acc_shift)

    @property
    def state_bits(self) -> int:          # bits of the state magnitude AlignState receives: acc_sig, or W when lossy
        return min(self.acc_sig, self.width)

    @property
    def acc_bits(self) -> int:            # the state term < 2^acc_bits (= W) after rescale and shift
        return self.state_bits + self.state_shift

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
        return (2**self.acc_bits - 2**self.state_shift) + self.n * (2**self.prod_bits - 2**self.prod_shift)

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
        return max(self.acc_e_max, self.ep_hi) + g * self.growth

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
            self.acc_e_min - self.floor + self.width,               # clamp: e_raw below e_min
            last + self.l_hi - self.width - self.acc_e_min,         # clamp: e_raw above e_min
            self.exp_hi - self.acc_e_max,                           # overflow: e' above e_max
            self.acc_e_max - self.acc_e_min,                        # overflow: e' below e_max
        )
        return max(self.shift_rows, max(spans) + 1)

    # --- normalisation -----------------------------------------------------------------------------------------------
    # The W-bit normalisation, the subnormal shift and the rescale to acc_sig bits compose into one floor division
    # M_n = floor(v 2^k) only while the rescale is a right shift (W >= acc_sig): k = E - e' + acc_sig - W.  When the
    # rescale is a left shift (lossy) it must follow the truncations: M_n = floor(v 2^(E - e')) has W bits and
    # M' = M_n 2^(acc_sig - W).  ``norm_k_offset`` / ``norm_up`` / ``norm_bits`` express both cases; at W >= acc_sig
    # they are the historical formulas.
    @property
    def norm_k_offset(self) -> int:       # k = E - e' + norm_k_offset (BF16: -1; FP8: 0)
        return min(0, self.acc_sig - self.width)

    @property
    def norm_up(self) -> int:             # M' = M_n * 2^norm_up (BF16: 0; FP8: 10)
        return max(0, self.acc_sig - self.width)

    @property
    def norm_bits(self) -> int:           # M_n < 2^norm_bits (BF16: 24; FP8: 14)
        return min(self.acc_sig, self.width)

    def norm_k(self, E: int, e1: int) -> int:
        return E - e1 + self.norm_k_offset

    @property
    def k_lo(self) -> int:                # smallest k: clamped at the floor, or l = l_hi unclamped
        return min(self.width - self.l_hi, self.floor - self.acc_e_min) + self.norm_k_offset

    @property
    def k_hi(self) -> int:                # largest k: l = 0, unclamped
        return self.width + self.norm_k_offset

    @property
    def norm_rem_bits(self) -> int:       # the normalisation remainder r_n < 2^(-k) <= 2^norm_rem_bits
        return -self.k_lo


REAL = Params()
#: Reduced widths for the exhaustive searches: 3-bit significands, two groups of two.
TOY = Params(sig=3, exp=3, acc_sig=5, width=6, floor=-8, groups=(2, 2))
#: Smaller still, for the gadgets whose joint hint space is otherwise too large to enumerate in seconds.
TINY = Params(sig=2, exp=3, acc_sig=4, width=5, floor=-4, groups=(2, 2))
#: A lossy toy (adder narrower than the accumulator, operand exponent narrower than the accumulator's, E4M3 NaN
#: rule): the reduced-width stand-in for the FP8 pipelines, for the Python differential and the mutation tests.
TOY_LOSSY = Params(sig=3, exp=3, acc_sig=7, width=5, floor=-12, groups=(2, 2), acc_exp=4, op_nan_rule="e4m3")
