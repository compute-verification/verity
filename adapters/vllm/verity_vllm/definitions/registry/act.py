"""verity_vllm.definitions.registry.act — vLLM d9105ea80 `csrc/libtorch_stable/activation_kernels.cu` activations as served Definitions:

  _C.silu_and_mul   act_and_mul_kernel<bf16, silu_kernel>   ->  SiluMul_v2{I} over SiluMulBf16_v2
  aten.gelu (erf)   torch's GeluCUDAKernelImpl              ->  GeluErf_v1{N} over GeluErfBf16_v1

PROMOTED from the vLLM adapter's quarantine: each Definition below is the quarantined body VERBATIM — same name, same
version, same gates — so its specialization ids are the ones the quarantine admitted. Bound as the `silu_mul` kind
of the `b1-eager-v3` vocabulary (`frontend/rules/vocab.py`), asked for by `SiluAndMulRule`; `SiluMul_v1` (b1) stays registered,
the kind of the older vocabularies and of the bodies that call it directly.

The Boolean form of `SiluMul_v2{I}` is here, so that whatever binds it registers its builder: `SiluMulExpf` (`SiluMul_v4{I}`)
over `SiluMulBf16Expf` (`SiluMulBf16_v4`), computed as `boolean_silu`'s `SiluMul_v3` computes `SiluMul_v1` (vLLM `_C.silu_and_mul` on the
measured sm_120 edge words: ``f / (1 + expf(-f))`` with expf's overflow): `silu_bf16_expf` sends NaN and -inf to 0x7FFF, keeps
-0, and reads `expf_band_table`, where the gates whose ``expf(-g)`` overflows (g <= -89) are -0; the up word passes ``+ 0.0f``
(-0 to +0), and every NaN result is 0x7FFF.  No sub-Call is a word Definition.

`SiluMulBf16_v2` is one output word of `act_and_mul_kernel<bf16, silu_kernel, true>`, the vec and scalar paths alike:

    silu_kernel(x):  (T)(f / (1.0f + expf(-f)))          f = (float)x; cvt.rn.bf16.f32 (NaN -> 0x7FFF)
    out = (T)(ACT(gate) * ((float)up + beta))           beta = 0.0f: up = -0 enters as +0, identity otherwise

`SiluMulBf16_v1` (prims) is this function except on the words measured on sm_120 (the
`silu-edge-cases` labels):
  * NaN results are 0x7FFF, not 0x7FC0;
  * up = -0 is +0 after `+ beta`, so the product's zero takes the gate's sign (`[g=0x3F80 | u=0x8000]` -> 0x0000);
  * gate -inf is `-inf / (1 + inf)` = NaN, not the limit -0;
  * gates whose `expf(-g)` overflows (bf16 g <= -89, past ln(FLT_MAX) = 88.72) are `g / inf` = -0, not a tiny g * e^g;
  * gate -0 is -0 (`_v1` decodes it to an unsigned zero).
Every other word is `_v1`'s: `silu_f32` is the same RN32-of-the-exact-real stand-in for expf + div.rn.

`GeluErfBf16_v1` is one output word of `aten.gelu(x, approximate='none')` on bf16 (torch 2.13.0+cu129
`aten/src/ATen/native/cuda/ActivationGeluKernel.cu`, `GeluCUDAKernelImpl`, `opmath_t = float`):

    return static_cast<opmath_t>(x) * opmath_t(0.5) * (opmath_t(1) + ::erf(static_cast<opmath_t>(x) * kAlpha));   // kAlpha = M_SQRT1_2

i.e. `bf16( (f * 0.5f) * (1.0f + erff(f * 0x3F3504F3)) )`, `f = (float)x`, stored through `c10::BFloat16(float)` (`cvt.rn.bf16.f32`,
NaN -> 0x7FFF).  `erff` is CUDA libdevice's `__nv_erff` (a minimax polynomial pair with FMAs, not correctly rounded), so the
word is NOT `RN(gelu(x))` for every x; rather than transcribe libdevice, the function is pinned by its **exhaustive table**: the
domain is the 65,536 bf16 words, and `gelu_erf_table.py` holds the kernel's output word for every one of them, measured on the
pod (RTX 4090, sm_89, torch 2.13.0+cu129) by a probe that ran `aten.gelu` on `arange(65536).view(bfloat16)` and
stored the words zlib + base64.  The primitive is the table lookup.  Precedent: `prims.MufuEx2Ftz` /
`MufuRcpFtz` (measured hardware tables).  `GeluErf_v1{N}` is `out[j] = GeluErfBf16(x[j])` over a row, the GPT-NeoX MLP
activation (`nn.GELU()`, vLLM `get_act_fn("gelu")`).  Both were promoted from quarantine with their bodies verbatim, so
their ids keep the quarantine's `ln/` prefix; the Match fold binds them (`observe/fold/profiles/gen_ln_patterns.py`), and `tests/engine/test_gen_ln.py`
checks the table's sanity facts: gelu(+0) = +0, gelu(-0) = -0, gelu(x) = x for x >= 2^4, gelu(x) = -0 for x <= -2^4, NaN words
-> 0x7FFF, +inf -> +inf, -inf -> NaN.
"""
from __future__ import annotations

import math
from functools import lru_cache

from verity.circuits.boolean.forms import NOT, ONE, OR, ZERO, add, and_reduce, const_bits, or_reduce, sub
from verity.circuits.boolean.trace import definition
from verity.circuits.defs import CompositeDefinition, bind, composite, primitive
from verity.circuits.silicon import fp
from verity.circuits.types import Array

from verity_vllm.definitions.evidence import EVIDENCE
from verity_vllm.definitions.registry import boolean_silu as BSL
from verity_vllm.definitions.registry import prims as P
from verity_vllm.definitions.registry.boolean_activation import band
from verity_vllm.definitions.registry.gelu_erf_table import TABLE_WORDS

#: `SEMANTIC_ID`s of the promoted Definitions (their quarantine identities)
SEMANTIC_IDS = {"SiluMulBf16": "act/SiluMulBf16@v2", "SiluMul": "act/SiluMul@v2", "GeluErfBf16": "ln/GeluErfBf16@v1",
                "GeluErf": "ln/GeluErf@v1"}

BF16 = P.BF16
F32_NEG_ZERO = 0x80000000
ZERO_BITS = 0x00000000
#: ln of the smallest real that RN32 sends to +inf, (2 - 2^-24) * 2^127: no f32 word lies within float64 error of it
LN_F32_OVERFLOW = math.log((2.0 - 2.0 ** -24) * 2.0 ** 127)

CONFORMANCE = ("exact-model-tested: SiluMulBf16_v1 on finite ordinary words; the edge words (NaN, gate -inf, gates past expf "
               f"overflow, gate -0, up -0) as measured ({EVIDENCE['silu_edge_cases']['result']}, silu-edge-cases)")


def silu_f32_bits(g: int) -> int:
    """`f / (1.0f + expf(-f))` on an f32 word, total."""
    c, v = P._decode(g)
    if c == 2 or c == -1:
        return P.F32_NAN
    if c == 1:
        return 0x7F800000
    if g == F32_NEG_ZERO:
        return F32_NEG_ZERO
    if -float(v) >= LN_F32_OVERFLOW:
        return F32_NEG_ZERO
    return P._silu_f32_bits(g)


@primitive("SiluMulBf16", 2, [("g", BF16), ("u", BF16)], BF16, conformance=CONFORMANCE)
def SiluMulBf16(g: int, u: int) -> int:
    """`bf16(f32(bf16(silu_f32(g))) * (f32(u) + 0.0f))`, both conversions `cvt.rn.bf16.f32` (NaN -> 0x7FFF)."""
    s = P.f32_to_bf16_rn_bits(silu_f32_bits(P.bf16_to_f32_bits(g)), P.BF16_NAN_F2FP)
    uf = P.f32_add_bits(P.bf16_to_f32_bits(u), ZERO_BITS)
    return P.f32_to_bf16_rn_bits(P.f32_mul_bits(P.bf16_to_f32_bits(s), uf), P.BF16_NAN_F2FP)


@composite("SiluMul", 2, ["I"], lambda S: ((("gu", Array(2 * S.I, BF16)),), Array(S.I, BF16)),
           conformance="exact-model-tested: SiluMulBf16_v2 per element")
def SiluMul(B, S, gu):
    """`out[i] = SiluMulBf16_v2(gu[i], gu[I + i])`."""
    return B.batch(SiluMulBf16, gu[0:S.I], gu[S.I:2 * S.I], axes=(0, 0))


def gelu_erf_bf16_bits(x: int) -> int:
    """The kernel's output word for the bf16 input word `x` (table lookup)."""
    i = 2 * (x & 0xFFFF)
    return TABLE_WORDS[i] | (TABLE_WORDS[i + 1] << 8)


@primitive("GeluErfBf16", 1, [("x", BF16)], BF16,
           conformance="quarantine (op-level evidence only): exhaustive measured table (65,536/65,536 bf16 words, torch 2.13.0+cu129 aten.gelu on "
                       "sm_89); difftest vs torch.nn.functional.gelu on the pod",
           tables=(("gelu_erf_bf16", "registered"),))
def GeluErfBf16(x: int) -> int:
    """`bf16((f * 0.5f) * (1.0f + erff(f * M_SQRT1_2)))`, `f = (float)x` — as the pinned kernel computes it (measured table)."""
    return gelu_erf_bf16_bits(x)


@composite("GeluErf", 1, ["N"], lambda S: ((("x", Array(S.N, BF16)),), Array(S.N, BF16)),
           conformance="quarantine (op-level evidence only): difftest vs torch.nn.functional.gelu (bf16, sm_89) on the pod")
def GeluErf(B, S, x):
    """`out[j] = GeluErfBf16(x[j])` over one row of N bf16 words."""
    return B.batch(GeluErfBf16, x, axes=(0,))


@lru_cache(maxsize=None)
def expf_band_table() -> tuple[int, ...]:
    """`band_table` of `SiluMulBf16_v2`: on the band only the gates whose ``expf(-g)`` overflows differ, each 0 (``g / inf``)."""
    out = [0] * (1 << 12)
    for sign in (0, 1):
        for e in range(BSL.BAND_LO, BSL.BAND_HI[sign] + 1):
            for m in range(128):
                out[m | (e & 0xF) << 7 | sign << 11] = SiluMulBf16.evaluate(sign << 15 | e << 7 | m, BSL.ONE_BF16) & 0x7FFF
    return tuple(out)


def silu_bf16_expf(C, g):
    """``bf16_rn(silu_f32(f32(g)))``, the silu half of `SiluMulBf16_v2`: `silu_bf16` with -inf and NaN to 0x7FFF, -0 kept, and
    `expf_band_table` on the band."""
    m, e, s = list(g[:7]), list(g[7:15]), g[15]
    nan = C.AND(and_reduce(C, e), OR(C, or_reduce(C, m), s))               # NaN, and -inf / (1 + inf)
    _, below = sub(C, e, const_bits(BSL.BAND_LO, 8), 8)
    _, under = sub(C, e, const_bits(BSL.BAND_HI[0] + 1, 8), 8)
    keep = C.AND(C.AND(NOT(s), NOT(under)), NOT(nan))
    e01 = NOT(or_reduce(C, e[1:]))
    low = C.AND(below, e01)
    high = below ^ low
    half_low = add(C, m[1:] + [e[0]], [C.AND(m[0], m[1])], width=8)
    half_high = m + sub(C, e, [ONE], 8)[0]
    on_band = band(C, g, expf_band_table(), [(sign, e0) for sign in (0, 1) for e0 in range(BSL.BAND_LO, BSL.BAND_HI[sign] + 1, 2)])
    mag = [x ^ C.AND(low, y) ^ C.AND(high, z) ^ C.AND(keep, w)
           for x, y, z, w in zip(on_band, half_low + [ZERO] * 7, half_high, g[:15])]
    mag = [x ^ nan if P.BF16_NAN_F2FP >> j & 1 else x for j, x in enumerate(mag)]
    return mag + [C.AND(s, NOT(nan))]


def silu_mul_expf(C, g, u):
    """`SiluMulBf16_v2`: ``bf16_rn(f32(silu_bf16_expf(g)) * (f32(u) + 0.0f))``, every NaN 0x7FFF; ``+ 0.0f`` sends -0 to +0."""
    s = silu_bf16_expf(C, g)
    up = list(u[:15]) + [C.AND(u[15], or_reduce(C, u[:15]))]
    return fp.f32_to_bf16(C, fp.f32_mul(C, fp.bf16_to_f32(s), fp.bf16_to_f32(up)), P.BF16_NAN_F2FP)




SiluMulBf16Expf = definition("SiluMulBf16", 4, silu_mul_expf, (("g", BSL.BF16), ("u", BSL.BF16)), BSL.BF16, word=SiluMulBf16,
                             doc="vLLM _C.silu_and_mul's element on bits: SiluMulBf16_v2, "
                                 "bf16(f32(bf16(silu_f32(g))) * (f32(u) + 0.0f)), NaN 0x7FFF")


def _silu_mul_expf_rows(B, S, gu):
    """vLLM `_C.silu_and_mul` on bits: `out[i] = SiluMulBf16_v4(gu[i], gu[I + i])`."""
    return B.batch(SiluMulBf16Expf, gu[0:S.I], gu[S.I:2 * S.I], axes=(0, 0))


SiluMulExpf = CompositeDefinition("SiluMul", 4, ("I",), lambda S: ((("gu", Array(2 * S.I, BSL.BF16)),), Array(S.I, BSL.BF16)),
                                  _silu_mul_expf_rows, doc=_silu_mul_expf_rows.__doc__, conformance="exact-model-tested")
SiluMulExpf.word = lambda b: bind(SiluMul, I=b["I"])  # type: ignore[attr-defined]
