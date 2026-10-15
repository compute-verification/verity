"""Integer tensor-core steps: ``mma.sync.aligned.m16n8k32.row.col{.satfinite}.s32.s8.s8.s32`` (SASS IMMA.16832).

The K = 32 products of 8-bit integer operands are exact, and they are summed exactly with the incoming s32 accumulator:
no rounding happens anywhere.  What leaves the s32 range is the instruction's choice.  With PTX's ``.satfinite`` the
result D is clamped to [-2^31, 2^31 - 1]; without it D wraps modulo 2^32.  Accumulator words are 32-bit two's-complement
encodings, carried as unsigned ints like the FP32 words of the other models; operand words are 8-bit codes (two's
complement for s8).

What the model rests on (``hypothesis`` in ``instructions``): the PTX ISA's integer ``mma`` semantics, and one
library-kernel probe on an RTX PRO 6000 (run r20261015-071801-6d3c: CUTLASS ``i16832gemm_s8`` through ``torch._int_mm``,
which saturates).  Random int8 products are exact to k = 133,120; at the s32 edge, all-127 rows give +-2,147,092,480
exactly; and the first sum past it (k = 133,152) is 2^31 - 1.  Not probed: one instruction per tile, nonzero incoming
accumulators, code -128, u8 operands, the form without ``.satfinite``, and where ``.satfinite`` clamps when a sum leaves
the range and comes back within one instruction (at D, as modelled here, or along the k order).
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


_PRO6000_IMMA = (
    "HYPOTHESIS: the PTX ISA's integer mma semantics; one library-kernel probe, run r20261015-071801-6d3c (RTX PRO 6000 "
    "Blackwell Server Edition, cc 12.0, driver 580.173.02, torch 2.13.0+cu129, torch._int_mm -> CUTLASS "
    "cutlass_80_tensorop_i16832gemm_s8_64x64_64x6_tn_align16, zero incoming accumulators): random int8 in [-127, 127], "
    "256 x 256 outputs at k = 4096, 14,336, 65,536 and 133,120 equal to the exact product; all-127 against all +-127 at "
    "k = 133,120 gives +-2,147,092,480 exactly; at k = 133,152 (true sum 2,147,608,608) it gives 2,147,483,647"
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

#: The same instruction without ``.satfinite``: the PTX ISA's wrap modulo 2^32.  Nothing has run it.
BLACKWELL_SM120_S8_M16N8K32_WRAP = IntegerMma(
    name="blackwell_sm120_s8_m16n8k32_wrap",
    arch="sm_120",
    dtype="s8",
    b_dtype="s8",
    operand_bits=8,
    k_per_instruction=32,
    satfinite=False,
    validated="HYPOTHESIS: the PTX ISA's integer mma without .satfinite (D wraps modulo 2^32); unprobed",
)
