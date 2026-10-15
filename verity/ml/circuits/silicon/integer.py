"""Integer tensor-core steps: ``mma.sync.aligned.m16n8k32.row.col{.satfinite}.s32.s8.s8.s32`` (SASS IMMA.16832).

The K = 32 products of 8-bit integer operands are exact, and they are summed exactly with the incoming s32 accumulator:
no rounding happens anywhere.  What leaves the s32 range is the instruction's choice.  With PTX's ``.satfinite`` the
result D is clamped to [-2^31, 2^31 - 1]; without it D wraps modulo 2^32.  Accumulator words are 32-bit two's-complement
encodings, carried as unsigned ints like the FP32 words of the other models; operand words are 8-bit codes (two's
complement for s8).

What the model rests on (``pinned`` in ``instructions``): one ``mma.sync`` per tile on an RTX PRO 6000, run
r20261015-084000-06d8 (``benchmarks/fp8_instantiation/imma_probe.py``), in eight forms: s8/u8 x s8/u8, with and without
``.satfinite``, each lowered to one ``IMMA.16832.<A>.<B>[.SAT]``.  Every form reproduces ``IntegerMma`` at its operand
formats bit for bit on 10,243,968 words, with code -128, random and +-2^31-edge incoming accumulators, and 0.29M-1.1M
words per form whose exact sum leaves s32.  ``.satfinite`` clamps once, at D: tiles whose products run past the range
and come back within the instruction exclude a clamp after every product or every 2, 4, 8 or 16 products, in either k
order (in each form with a signed operand; u8 x u8 sums only grow, so there every clamp point gives the same word).
Without it, D wraps modulo 2^32.  An earlier library-kernel probe (run r20261015-071801-6d3c, CUTLASS ``i16832gemm_s8``
through ``torch._int_mm``) agrees: random int8 products exact to k = 133,120, and the s32 edge saturating.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .errors import InvalidArtifact
from .term import FP32_SIGNIFICAND_WIDTH, Term

__all__ = [
    "BLACKWELL_SM120_S8_M16N8K32",
    "BLACKWELL_SM120_S8_M16N8K32_WRAP",
    "IntegerMma",
    "S32_MAX",
    "S32_MIN",
    "decode_int8",
    "s32_value",
    "s32_word",
]

S32_MIN, S32_MAX = -(1 << 31), (1 << 31) - 1


def decode_int8(fmt: str, code: int) -> int:
    if not 0 <= code < 256:
        raise InvalidArtifact(f"{fmt} code {code} is not a byte")
    if fmt == "s8":
        return code - 256 if code >= 128 else code
    if fmt == "u8":
        return code
    raise InvalidArtifact(f"no integer format {fmt!r}")


def s32_value(word: int) -> int:
    if not 0 <= word < 1 << 32:
        raise InvalidArtifact(f"s32 word {word} is not 32 bits")
    return word - (1 << 32) if word >= 1 << 31 else word


def s32_word(value: int) -> int:
    if not S32_MIN <= value <= S32_MAX:
        raise InvalidArtifact(f"{value} is outside s32")
    return value & 0xFFFFFFFF


@dataclass(frozen=True, slots=True)
class IntegerMma:
    """One integer ``mma`` accumulator coordinate: ``D = C + sum_k a[k] * b[k]`` exactly, then clamped (``satfinite``)
    or wrapped modulo 2^32.  ``product`` is the exact integer product as a :class:`Term` (exponent 23: weight 1)."""

    name: str
    arch: str
    dtype: Literal["s8", "u8"]
    b_dtype: Literal["s8", "u8"]
    operand_bits: int
    k_per_instruction: int
    satfinite: bool
    validated: str

    @property
    def k(self) -> int:
        return self.k_per_instruction

    @property
    def words_per_step(self) -> int:
        return self.k_per_instruction

    def product(self, a: int, b: int) -> Term:
        p = decode_int8(self.dtype, a) * decode_int8(self.b_dtype, b)
        return Term(p < 0, FP32_SIGNIFICAND_WIDTH - 1, abs(p))

    def step(self, c_word: int, a_words: Sequence[int], b_words: Sequence[int]) -> int:
        if len(a_words) != self.k or len(b_words) != self.k:
            raise InvalidArtifact(f"{self.name} takes {self.k} operand pairs")
        d = s32_value(c_word) + sum(decode_int8(self.dtype, a) * decode_int8(self.b_dtype, b)
                                    for a, b in zip(a_words, b_words))
        if self.satfinite:
            return s32_word(min(max(d, S32_MIN), S32_MAX))
        return d & 0xFFFFFFFF


_PRO6000_IMMA_PROBE = (
    "run r20261015-084000-06d8 (RTX PRO 6000 Blackwell Server Edition, cc 12.0, driver 580.173.02, CUDA 13.0 V13.0.88, "
    "sm_120a, GPU-415b238c): one mma.sync m16n8k32 per tile, 80,031 tiles (seed 20261015), 10,243,968 words per form, "
    "0 mismatches in each of s8.s8, s8.u8, u8.s8, u8.u8 with and without .satfinite; SASS one IMMA.16832.<A>.<B>[.SAT] "
    "per kernel"
)

_PRO6000_IMMA = (
    "the satfinite form on the device: " + _PRO6000_IMMA_PROBE + "; the clamp tiles admit only a clamp at D; trust "
    "dossier art:86f39a17b5a0c8a8c92e5e25dcdfbee95d6f65408130bc1db6c54d440dfd8474 (PINNED).  Earlier, run "
    "r20261015-071801-6d3c (torch._int_mm -> CUTLASS cutlass_80_tensorop_i16832gemm_s8_64x64_64x6_tn_align16, zero "
    "incoming accumulators): random int8 at k = 4096, 14,336, 65,536 and 133,120 exact; all-127 against all +-127 at "
    "k = 133,120 gives +-2,147,092,480; at k = 133,152 (true sum 2,147,608,608) it gives 2,147,483,647"
)

BLACKWELL_SM120_S8_M16N8K32 = IntegerMma(
    name="blackwell_sm120_s8_m16n8k32",
    arch="sm_120",
    dtype="s8",
    b_dtype="s8",
    operand_bits=8,
    k_per_instruction=32,
    satfinite=True,
    validated=_PRO6000_IMMA,
)

#: The same instruction without ``.satfinite``: D wraps modulo 2^32.  It has no registry entry of its own.
BLACKWELL_SM120_S8_M16N8K32_WRAP = IntegerMma(
    name="blackwell_sm120_s8_m16n8k32_wrap",
    arch="sm_120",
    dtype="s8",
    b_dtype="s8",
    operand_bits=8,
    k_per_instruction=32,
    satfinite=False,
    validated="the form without .satfinite on the device: " + _PRO6000_IMMA_PROBE + "; out-of-range sums wrap modulo 2^32",
)
