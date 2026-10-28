"""Integer tensor-core steps: ``mma.sync.aligned.m16n8k32.row.col{.satfinite}.s32.s8.s8.s32`` (SASS IMMA.16832).

The K = 32 products of 8-bit integer operands are exact, and they are summed exactly with the incoming s32 accumulator:
no rounding happens anywhere.  What leaves the s32 range is the instruction's choice.  With PTX's ``.satfinite`` the
result D is clamped to [-2^31, 2^31 - 1]; without it D wraps modulo 2^32.  Accumulator words are 32-bit two's-complement
encodings, carried as unsigned ints like the FP32 words of the other models; operand words are 8-bit codes (two's
complement for s8).

What the model rests on (``pinned`` in ``instructions``): one device, an RTX PRO 6000; no other sm_120 part has been
probed.  An IMMA probe ran one ``mma.sync`` per tile in eight forms: s8/u8 x
s8/u8, with and without ``.satfinite``, each lowered to one ``IMMA.16832.<A>.<B>[.SAT]``.  Every form reproduces
``IntegerMma`` at its operand formats bit for bit on 10,243,968 words, with code -128, random and +-2^31-edge incoming
accumulators, and 0.29M-1.1M words per form whose exact sum leaves s32.  ``.satfinite`` clamps once, at D: tiles whose products run past the range
and come back within the instruction exclude a clamp after every product or every 2, 4, 8 or 16 products, in either k
order (in each form with a signed operand; u8 x u8 sums only grow, so there every clamp point gives the same word).
Without it, D wraps modulo 2^32.  An earlier library-kernel probe (CUTLASS ``i16832gemm_s8``
through ``torch._int_mm``) agrees: random int8 products exact to k = 133,120, and the s32 edge saturating.

The datacenter cards compute the same function (`benchmarks/pouw/dc/s8_mma.py`, the same families at each instruction's tile).
An H200 (sm_90a) ran ``wgmma.mma_async`` m64n8k32 (SASS ``IGMMA.64x8x32.<A>.<B>[.SAT]``) and
``mma.sync`` m16n8k32; a B200 (sm_100a) ran ``tcgen05.mma.cta_group::1.kind::i8`` M128 N32 K32
(SASS ``UTCIMMA``, the formats and the saturation in the instruction descriptor, its bit 3 ``.satfinite``) and ``mma.sync``
m16n8k32.  Each of the eight forms of each instruction reproduces this module's step at its formats on every word, with the
clamp at D.  `HOPPER_SM90_S8_K32` and `BLACKWELL_SM100_S8_K32` are that step under each architecture's name.

`step_s8_bits` is the s8 x s8 step as a GF(2) circuit (`verity.ml.circuits.boolean.forms`), the builder of
`verity.ml.circuits.ops.boolean.gemm.BlackwellS8ImmaDot32_v2`.

The store ids each model's ``validated`` string rests on are :data:`EVIDENCE_IDS` (``integer_evidence.json``), keyed by
model name and then by role: a name in backticks in a ``validated`` string (``run``, ``device``) is a role of that model's
entry.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from verity.ml.circuits.boolean.forms import NOT, ONE, compress_columns

from .errors import InvalidArtifact
from .term import FP32_SIGNIFICAND_WIDTH, Term

__all__ = [
    "BLACKWELL_SM120_S8_M16N8K32",
    "BLACKWELL_SM120_S8_M16N8K32_WRAP",
    "BLACKWELL_SM120_U8S8_M16N8K32",
    "BLACKWELL_SM100_S8_K32",
    "EVIDENCE_IDS",
    "HOPPER_SM90_S8_K32",
    "IntegerMma",
    "S32_MAX",
    "S32_MIN",
    "decode_int8",
    "exact_k_limit",
    "s32_value",
    "s32_word",
    "step_s8_bits",
]

S32_MIN, S32_MAX = -(1 << 31), (1 << 31) - 1

#: Model name -> role -> the store id (run, ``art:``) or GPU UUID its ``validated`` string names by that role.
EVIDENCE_IDS: dict[str, dict] = json.loads(Path(__file__).with_name("integer_evidence.json").read_text())


def decode_int8(fmt: str, code: int) -> int:
    if not 0 <= code < 256:
        raise InvalidArtifact(f"{fmt} code {code} is not a byte")
    if fmt == "s8":
        return code - 256 if code >= 128 else code
    if fmt == "u8":
        return code
    raise InvalidArtifact(f"no integer format {fmt!r}")


def exact_k_limit(dtype: str, b_dtype: str) -> int:
    """The least K at which a sum of K products of ``dtype`` x ``b_dtype`` codes can leave s32.

    With ``lo <= 0 <= hi`` the extreme products, every partial sum of at most K products from a zero accumulator, in any k
    order and any grouping into steps, lies in ``[K * lo, K * hi]``. Below this K no step clamps or wraps, so the s32 word
    is the exact sum whatever order the k steps run in; at it, K extreme products leave s32. For s8 x s8, ``hi`` =
    (-128)^2 = 2^14 binds (``lo`` = -128 * 127 would allow K up to 132,104): the limit is 2^17 exactly, where K codes -128
    sum to 2^31 and ``.satfinite`` clamps it to 2^31 - 1."""
    ends = [[decode_int8(f, c) for c in (0, 127, 128, 255)] for f in (dtype, b_dtype)]
    products = [x * y for x in ends[0] for y in ends[1]]
    lo, hi = min(products), max(products)
    limit = S32_MAX // hi + 1
    return min(limit, -S32_MIN // -lo + 1) if lo < 0 else limit


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
    "the `run` (RTX PRO 6000 Blackwell Server Edition, cc 12.0, driver 580.173.02, CUDA 13.0 V13.0.88, "
    "sm_120a, GPU `device`): one mma.sync m16n8k32 per tile, 80,031 tiles (seed 20261015), 10,243,968 words per form, "
    "0 mismatches in each of s8.s8, s8.u8, u8.s8, u8.u8 with and without .satfinite; SASS one IMMA.16832.<A>.<B>[.SAT] "
    "per kernel"
)

_PRO6000_IMMA = (
    "the satfinite form on the device: " + _PRO6000_IMMA_PROBE + "; the clamp tiles admit only a clamp at D; trust "
    "dossier `dossier` (PINNED).  Earlier, run `int_mm` (torch._int_mm -> CUTLASS "
    "cutlass_80_tensorop_i16832gemm_s8_64x64_64x6_tn_align16, zero incoming accumulators): random int8 at k = 4096, 14,336, 65,536 and 133,120 exact; all-127 against all +-127 at "
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

#: ``mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32`` (SASS ``IMMA.16832.U8.S8``): A u8, B s8, no ``.satfinite``, so D wraps
#: modulo 2^32.  The edited GEMM's dithered side runs this form; `ops.prims.BlackwellU8S8ImmaDot32_v1` registers it.
BLACKWELL_SM120_U8S8_M16N8K32 = IntegerMma(
    name="blackwell_sm120_u8s8_m16n8k32",
    arch="sm_120",
    dtype="u8",
    b_dtype="s8",
    operand_bits=8,
    k_per_instruction=32,
    satfinite=False,
    validated="the u8.s8 form without .satfinite on the device: " + _PRO6000_IMMA_PROBE + "; out-of-range sums wrap modulo 2^32",
)

_DC_FORMS = (
    "eight forms (s8.s8, s8.u8, u8.s8, u8.u8, each with and without .satfinite), each reproducing IntegerMma at its formats "
    "on every word, the clamp tiles admitting only a clamp at D; CUDA 13.0 V13.0.88, driver 580.173.02, seed 20261015"
)

#: ``wgmma.mma_async.sync.aligned.m64nNk32.s32.s8.s8.satfinite`` and ``mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32``
#: on sm_90a: the sm_120 step's function (``BLACKWELL_SM120_S8_M16N8K32``) under Hopper's name.
HOPPER_SM90_S8_K32 = IntegerMma(
    name="hopper_sm90_s8_k32",
    arch="sm_90",
    dtype="s8",
    b_dtype="s8",
    operand_bits=8,
    k_per_instruction=32,
    satfinite=True,
    validated=(
        "the `run` (NVIDIA H200, cc 9.0, sm_90a, GPU `device`): wgmma m64n8k32 (SASS IGMMA.64x8x32.<A>.<B>[.SAT], "
        "one per tile kernel), 20,028 tiles, 10,254,336 words per form, and mma.sync m16n8k32 (IMMA.16832.<A>.<B>[.SAT]), "
        "80,031 tiles, 10,243,968 words per form; " + _DC_FORMS
    ),
)

#: ``tcgen05.mma.cta_group::1.kind::i8`` (idesc D s32, A and B s8, saturate) and ``mma.sync.aligned.m16n8k32.row.col.satfinite.
#: s32.s8.s8.s32`` on sm_100a: the sm_120 step's function under datacenter Blackwell's name.
BLACKWELL_SM100_S8_K32 = IntegerMma(
    name="blackwell_sm100_s8_k32",
    arch="sm_100",
    dtype="s8",
    b_dtype="s8",
    operand_bits=8,
    k_per_instruction=32,
    satfinite=True,
    validated=(
        "the `run` (NVIDIA B200, cc 10.0, sm_100a, GPU `device`): tcgen05.mma kind::i8 M128 N32 K32 (SASS UTCIMMA, "
        "one per tile kernel; idesc bit 3 selects .satfinite), 2,523 tiles, 10,334,208 words per form, and mma.sync m16n8k32 "
        "(IMMA.16832.<A>.<B>[.SAT]), 80,031 tiles, 10,243,968 words per form; " + _DC_FORMS
    ),
)

#: Baugh-Wooley's constant for one s8 x s8 product: the two negated rows' -(2^14 - 2^7) each
_BW_CONSTANT = -(1 << 15) + (1 << 8)


def step_s8_bits(C, model: IntegerMma, acc, a, b):
    """``model``'s step on bits (an s8 x s8 model): ``acc`` 32 bits, ``a`` and ``b`` k operands of 8 bits each, LSB first.

    Each product is a Baugh-Wooley array: the 49 unsigned partial products, the 14 that carry one sign bit negated, the
    sign bits' product, and a constant. The k arrays, the accumulator sign-extended to 33 bits and the k constants are
    reduced once, modulo 2^33, which holds every sum (``|acc| <= 2^31`` and ``|products| <= k * 2^14``). With ``.satfinite``
    the sum is clamped at D: it leaves s32 exactly when its bits 31 and 32 differ, and bit 32 is then its sign.

    The result is committed (``C.commit``, a row of a lowering and no gate): uncommitted, each accumulator bit is an XOR form
    over every earlier step's, and a K/k-step chain's lowering grows as K^2 (C-Flock's unit at K = 256: 20.9M nonzeros, against
    2.0M committed)."""
    if (model.dtype, model.b_dtype) != ("s8", "s8"):
        raise InvalidArtifact(f"{model.name}: step_s8_bits takes s8 x s8 models")
    if len(acc) != 32 or len(a) != model.k or len(b) != model.k:
        raise InvalidArtifact(f"{model.name} takes a 32-bit accumulator and {model.k} operand pairs of 8 bits")
    W = 33
    cols: list[list] = [[] for _ in range(W)]
    for x, y in zip(a, b):
        for i in range(8):
            for j in range(8):
                p = C.AND(x[i], y[j])
                cols[i + j].append(NOT(p) if (i == 7) != (j == 7) else p)
    for i in range(32):
        cols[i].append(acc[i])
    cols[32].append(acc[31])
    const = (model.k * _BW_CONSTANT) % (1 << W)
    for i in range(W):
        if const >> i & 1:
            cols[i].append(ONE)
    s = compress_columns(C, cols)[:W]
    if not model.satfinite:
        return C.commit(s[:32])
    ovf = s[31] ^ s[32]
    return C.commit([s[i] ^ C.AND(ovf, NOT(s[32]) ^ s[i]) for i in range(31)] + [s[32]])
