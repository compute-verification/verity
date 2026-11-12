"""[GLM-5.3] The dense prefill's read-back of an `fp8_ds_mla` cache row on the H200: vLLM d9105ea80's
`cp_gather_and_upconvert_fp8_kv_cache` (csrc/libtorch_stable/cache_kernels.cu:1624-1704), whose kernel
`cp_gather_and_upconvert_fp8_kv_cache_page` (:1387-1425) writes each context row of a chunk into the bf16 workspace by
`gather_and_upconvert_fp8_token` (:1360-1385).  The model calls it from `_compute_prefill_context`
(vllm/model_executor/layers/attention/mla_attention.py:2833-2841) when the cache is `fp8_ds_mla`, on a step whose prefill
attends densely (`use_dense_mha`: the step's largest prefill length at most `index_topk`, sparse_mla_attention.py:307-308) and
has cached context (a prefix-cache hit or an earlier chunk).  Line numbers are at that commit; "the SASS" is the kernel's sm_90
SASS in the pin's `vllm/_C_stable_libtorch.abi3.so` (cache_kernels.cu gets no `--use_fast_math`, CMakeLists.txt).

The row (DN NoPE values, DR RoPE values; GLM-5.3: DN = 512, DR = 64, 656 bytes, `FP8_DS_MLA_GLM53`), little-endian:

    bytes [0, DN)                       the NoPE values, one E4M3 byte each
    bytes [DN, DN + 4 DN/QT)            the DN/QT f32 scales, scale j for NoPE values [QT j, QT j + QT)
    bytes [DN + 4 DN/QT, ROW_BYTES)     the DR RoPE values, one bf16 word each

QT = 128 is the kernel's own grouping, not a static: lane l of a warp converts the 8-byte chunks c = l and c = 32 + l with
`scales[c >> 4]` (:1371-1373).  The output is the row's DN + DR bf16 words, NoPE first.

A NoPE value is `scaled_vec_conversion<__nv_bfloat16, uint8_t>` (quantization/w8a8/fp8/nvidia/quant_utils.cuh:112-124):
`__nv_cvt_fp8_to_halfraw`, `half_to_float` (an `asm volatile` `cvt.f32.f16`, attention/dtype_float16.cuh:87-95), `tmp * scale`,
`__float2bfloat16`.  The SASS has, for each value, exactly

    F2FP.F16.E4M3.UNPACK_B   h, b          E4M3 -> f16 (exact on every non-NaN code)
    HADD2.F32                x, -RZ, h.H0_H0   f16 -> f32 (-0 + h: exact, either zero keeps its sign)
    FMUL                     y, s, x        RN, subnormals kept (no .FTZ)
    F2FP.BF16.F32.PACK_AB    w, y1, y0      RN to bf16, every NaN 0x7FFF

and no FFMA: the multiply and the conversion do not contract.  The RoPE words are copied as 32 4-byte words (LDG then STG,
:1381-1384).  So a NoPE value is `F2fpBf16(F32Mul_v2(E4m3ToF32(b), s))`: the E4M3 -> f16 -> f32 chain is `E4m3ToF32` on every
non-NaN code, and a NaN code (0x7F, 0xFF), like a NaN or infinite scale, is a NaN product, which `F2fpBf16` writes 0x7FFF
whatever its payload.

    Fp8DsMlaUpconvertRow{DN,DR}   (nope: DN e4m3, scale: DN/128 f32, rope: DR bf16) -> DN + DR bf16

`upconvert_rows` is the same function over numpy rows, registered as the Definition's numpy kernel; `split_rows` cuts rows of
bytes into its three inputs, and `reference_word` is one NoPE value in exact rational arithmetic, rounded once to f32 and once
to bf16.  `HAND_VECTORS` are single NoPE values worked by hand, `EDGE_SCALES` scale words at the edges.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from verity.circuits.defs import bind, composite
from verity.circuits.evaluation import register_kernel
from verity.circuits.refs import Coll, concat
from verity.circuits.types import Array
from verity.definitions.kernels import tc as K
from verity.definitions.ops.fp32 import F32MulV2
from verity.definitions.ops.prims import E4m3ToF32, F2fpBf16
from verity.definitions.registry.fp8 import E4M3
from verity.definitions.registry.prims import BF16, F32

_CONF = ("model-untested: read from vLLM d9105ea80's cp_gather_and_upconvert_fp8_kv_cache (the module docstring's lines) and its "
         "sm_90 SASS; tests/properties/fp8_read_back_capture_gpu.py is its hardware check")

#: NoPE values a scale covers: the kernel's 8-value chunks, 16 to a scale (`scales[chunk >> 4]`)
QT = 128
#: GLM-5.3's `fp8_ds_mla` row: kv_lora_rank 512 NoPE values, qk_rope_head_dim 64 RoPE values
FP8_DS_MLA_GLM53 = {"DN": 512, "DR": 64}


def row_bytes(DN: int, DR: int) -> int:
    """A row's bytes: DN E4M3 bytes, DN/QT f32 scales, DR bf16 words (656 at GLM-5.3's statics)."""
    return DN + 4 * (DN // QT) + 2 * DR


def _refuse(S: dict) -> str | None:
    DN, DR = S.get("DN"), S.get("DR")
    if DN is not None and (DN < QT or DN % QT):
        return f"DN={DN} must be a positive multiple of {QT} (one f32 scale per {QT} NoPE values)"
    if DR is not None and DR < 1:
        return f"DR={DR} must be at least 1"
    return None


def _row_type(S):
    return ((("nope", Array(S.DN, E4M3)), ("scale", Array(S.DN // QT, F32)), ("rope", Array(S.DR, BF16))),
            Array(S.DN + S.DR, BF16))


@composite("Fp8DsMlaUpconvertRow", 1, ["DN", "DR"], _row_type, conformance=_CONF, refuse=_refuse)
def Fp8DsMlaUpconvertRow(B, S, nope, scale, rope):
    """One `fp8_ds_mla` cache row read back to bf16 (cache_kernels.cu:1360-1385): NoPE value d is
    `F2fpBf16(F32Mul_v2(E4m3ToF32(nope[d]), scale[d / 128]))`, then the RoPE words as stored."""
    x = B.batch(E4m3ToF32, nope, axes=(0,))
    y = [B.batch(F32MulV2, x[QT * j:QT * (j + 1)], scale[j], axes=(0, None)) for j in range(S.DN // QT)]
    w = B.batch(F2fpBf16, Coll(Array(S.DN, F32), concat([p.refs for p in y])), axes=(0,))
    return Coll(Array(S.DN + S.DR, BF16), concat([w.refs, rope.refs]))


FP8_DS_MLA_UPCONVERT_COMPOSITES = [Fp8DsMlaUpconvertRow]


# ---- the same function over numpy rows -------------------------------------------------------------------------------------

def _e4m3_f32() -> np.ndarray:
    from verity.definitions.silicon.cast import e4m3_to_f32_word

    return np.array([e4m3_to_f32_word(b) for b in range(256)], dtype=np.uint32)


_E4M3_F32 = _e4m3_f32()


def upconvert_rows(nope, scale, rope, **_statics):
    """`Fp8DsMlaUpconvertRow_v1` on N rows at once: nope (N, DN) uint8, scale (N, DN/128) uint32, rope (N, DR) uint16 ->
    (N, DN + DR) uint16.  numpy's f32 multiply is IEEE RN with subnormals (no FTZ on the host), the product's NaN payload
    is dropped by `F2fpBf16`'s every-NaN-0x7FFF."""
    nope, scale, rope = np.asarray(nope, dtype=np.uint8), np.asarray(scale, dtype=np.uint32), np.asarray(rope, dtype=np.uint16)
    x = _E4M3_F32[nope].view(np.float32)
    s = np.repeat(scale, QT, axis=1).view(np.float32)
    with np.errstate(all="ignore"):
        y = (x * s).astype(np.float32).view(np.uint32)
    w = K.f2fp_bf16_batch(y)[0].astype(np.uint16)
    return np.concatenate([w, rope], axis=1), None


register_kernel(Fp8DsMlaUpconvertRow, upconvert_rows, domain="every row")


def split_rows(rows, DN: int, DR: int):
    """Rows of `row_bytes(DN, DR)` bytes (N, ROW_BYTES) uint8 -> the Definition's inputs (nope, scale, rope) as numpy words."""
    rows = np.ascontiguousarray(np.asarray(rows, dtype=np.uint8))
    if rows.ndim != 2 or rows.shape[1] != row_bytes(DN, DR):
        raise ValueError(f"split_rows: rows of {row_bytes(DN, DR)} bytes at DN={DN}, DR={DR}, got shape {rows.shape}")
    g = DN // QT
    nope = rows[:, :DN]
    scale = rows[:, DN:DN + 4 * g].copy().view("<u4").astype(np.uint32)
    rope = rows[:, DN + 4 * g:].copy().view("<u2").astype(np.uint16)
    return nope, scale, rope


# ---- one NoPE value in exact arithmetic ---------------------------------------------------------------------------------------

def _round(x: Fraction, p: int, emin: int) -> Fraction:
    """x rounded to nearest, ties to even, at p significant bits with exponents floored at emin (subnormals kept)."""
    if x == 0:
        return x
    a = abs(x)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    ulp = Fraction(2) ** (max(e, emin) - (p - 1))
    q, r = divmod(a / ulp, 1)
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and q % 2):
        q += 1
    return (q * ulp) if x > 0 else -(q * ulp)


def _e4m3_value(b: int) -> Fraction:
    e, m = (b >> 3) & 0xF, b & 0x7
    v = Fraction(m, 8) * Fraction(2) ** -6 if e == 0 else (1 + Fraction(m, 8)) * Fraction(2) ** (e - 7)
    return -v if b & 0x80 else v


def _f32_value(u: int) -> Fraction | None:
    """The f32 word u as a rational, ±inf as None (a NaN is the caller's)."""
    e, m = (u >> 23) & 0xFF, u & 0x7FFFFF
    if e == 0xFF:
        return None
    v = Fraction(m, 1 << 23) * Fraction(2) ** -126 if e == 0 else (1 + Fraction(m, 1 << 23)) * Fraction(2) ** (e - 127)
    return -v if u >> 31 else v


def _bf16_word(v: Fraction, neg: bool) -> int:
    """A finite bf16 value (or 0 with its sign) as its word."""
    s = 0x8000 if neg else 0
    a = abs(v)
    if a == 0:
        return s
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    if e < -126:
        return s | int(a / Fraction(2) ** -133)
    return s | ((e + 127) << 7) | int((a / Fraction(2) ** e - 1) * 128)


def reference_word(b: int, scale: int) -> int:
    """One NoPE value's bf16 word from first principles: the E4M3 byte b's value times the f32 word scale's value, exact, then
    rounded once to f32 (FMUL: RN, subnormals, past the largest finite f32 ±inf) and once to bf16 (F2FP: RN, subnormals,
    past the largest finite bf16 ±inf); a NaN byte, a NaN scale and 0 times ±inf are 0x7FFF."""
    b, scale = b & 0xFF, scale & 0xFFFFFFFF
    neg = bool((b >> 7) ^ (scale >> 31))
    if b & 0x7F == 0x7F or ((scale >> 23) & 0xFF == 0xFF and scale & 0x7FFFFF):
        return 0x7FFF
    x, s = _e4m3_value(b), _f32_value(scale)
    if s is None:
        return 0x7FFF if x == 0 else (0xFF80 if neg else 0x7F80)
    y = _round(x * s, 24, -126)
    if abs(y) >= Fraction(2) ** 128:
        return 0xFF80 if neg else 0x7F80
    z = _round(y, 8, -126)
    if abs(z) >= Fraction(2) ** 128:
        return 0xFF80 if neg else 0x7F80
    return _bf16_word(z, neg)


# ---- vectors ---------------------------------------------------------------------------------------------------------------

_ONE, _NEG_ONE, _INF, _NAN = 0x3F800000, 0xBF800000, 0x7F800000, 0x7FC00000

#: (E4M3 byte, f32 scale word, the bf16 word of their NoPE value, what it shows), each worked by hand
HAND_VECTORS = [
    (0x7E, _ONE, 0x43E0, "largest code: 448 = 1.75 * 2^8"),
    (0xFE, _ONE, 0xC3E0, "largest negative code: -448"),
    (0x01, _ONE, 0x3B00, "smallest code: 2^-9, an E4M3 subnormal, a bf16 normal"),
    (0x81, _ONE, 0xBB00, "smallest negative code: -2^-9"),
    (0x07, _ONE, 0x3C60, "largest E4M3 subnormal: 7/8 * 2^-6"),
    (0x08, _ONE, 0x3C80, "smallest E4M3 normal: 2^-6"),
    (0x00, _ONE, 0x0000, "+0"),
    (0x80, _ONE, 0x8000, "-0 keeps its sign through cvt.f32.f16 (HADD2.F32 -RZ + h)"),
    (0x00, _NEG_ONE, 0x8000, "+0 * -1 = -0"),
    (0x80, _NEG_ONE, 0x0000, "-0 * -1 = +0"),
    (0x7F, _ONE, 0x7FFF, "NaN code"),
    (0xFF, _ONE, 0x7FFF, "negative NaN code: F2FP writes 0x7FFF whatever the sign and payload"),
    (0x7F, 0x00000000, 0x7FFF, "NaN code times 0"),
    (0x3E, 0x3F808008, 0x3FE1, "scale 1 + 2^-8 + 2^-20, not a power of two: 1.75 * s = 1.7568359... exact in f32, RN to 1.7578125"),
    (0x38, _NEG_ONE, 0xBF80, "negative scale: 1 * -1"),
    (0x38, 0x3F808000, 0x3F80, "tie to bf16: 1 + 2^-8 is halfway, to even (down)"),
    (0x38, 0x3F818000, 0x3F82, "tie to bf16: 1 + 2^-7 + 2^-8 is halfway, to even (up)"),
    (0x3C, 0x3FACAAAA, 0x4002, "double rounding: 1.5 * s = 2.0234375 - 2^-23 is an f32 tie, to 2.0234375, a bf16 tie, to "
                               "even 2.03125; rounding the exact product once would give 0x4001"),
    (0x08, 0x00800000, 0x0002, "subnormal product: 2^-6 * 2^-126 = 2^-132, kept (FMUL has no .FTZ)"),
    (0x01, 0x01000000, 0x0000, "subnormal product on a tie: 2^-9 * 2^-125 = 2^-134, half the least bf16 subnormal, to even 0"),
    (0x03, 0x01000000, 0x0002, "subnormal product on a tie: 1.5 * 2^-133, to even 2 * 2^-133"),
    (0x7E, 0x00000200, 0x0004, "subnormal scale 2^-140: 448 * 2^-140 = 3.5 * 2^-133, to even 4 * 2^-133"),
    (0x7E, 0x00400000, 0x0460, "subnormal scale 2^-127, normal product 1.75 * 2^-119 (DAZ would give 0)"),
    (0x7E, 0x7B800000, 0x7F80, "f32 overflow: 448 * 2^120 = 1.75 * 2^128 -> +inf"),
    (0x38, 0x7F7FFFFF, 0x7F80, "finite f32, bf16 overflow: the largest f32 rounds to +inf in bf16"),
    (0x00, _INF, 0x7FFF, "0 * inf is NaN"),
    (0x38, _NAN, 0x7FFF, "NaN scale"),
    (0xB8, _INF, 0xFF80, "-1 * inf = -inf"),
]

#: f32 scale words at the edges: ±1, the vectors' odd scales, subnormals, the largest finite, ±0, inf, NaN
EDGE_SCALES = [_ONE, _NEG_ONE, 0x3F808008, 0x3FACAAAA, 0x3B800000, 0x00800000, 0x00000200, 0x00400000, 0x7B800000, 0x7F7FFFFF,
               0x00000000, 0x80000000, _INF, _NAN, 0x3E2AAAAB, 0x4C000000]


def glm53_row():
    """`Fp8DsMlaUpconvertRow_v1` at GLM-5.3's statics."""
    return bind(Fp8DsMlaUpconvertRow, **FP8_DS_MLA_GLM53)
