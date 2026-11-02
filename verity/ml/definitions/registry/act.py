"""verity.ml.definitions.registry.act — vLLM d9105ea80 `csrc/libtorch_stable/activation_kernels.cu` activations as served Definitions:

  _C.silu_and_mul   act_and_mul_kernel<bf16, silu_kernel>   ->  SiluMul_v2{I} over SiluMulBf16_v2
  aten.gelu (erf)   torch's GeluCUDAKernelImpl              ->  GeluErf_v1{N} over GeluErfBf16_v1

PROMOTED from `verity_vllm.program.quarantine.act`: each Definition below is the quarantine module's
body VERBATIM — same name, same version, same gates — so its specialization ids are the quarantine's. The quarantine modules
re-export these objects (their difftest adapter, `ALL_DEFINITIONS` and `SEMANTIC_ID`s are unchanged). Bound as the `silu_mul` kind
of the `b1-eager-v3` vocabulary (`frontend/rules/vocab.py`), asked for by `SiluAndMulRule`; `SiluMul_v1` (b1) stays registered,
the kind of the older vocabularies and of the bodies that call it directly. The Boolean form is `boolean_silu.SiluMulExpf`
(`SiluMul_v4{I}`, word view `SiluMul_v2{I}`).

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
pod (RTX 4090, sm_89, torch 2.13.0+cu129) by `verity_vllm/ops/gelu_erf_table_probe.py` (`aten.gelu` on
`arange(65536).view(bfloat16)`, then zlib + base64).  The primitive is the table lookup.  Precedent: `prims.MufuEx2Ftz` /
`MufuRcpFtz` (measured hardware tables).  `GeluErf_v1{N}` is `out[j] = GeluErfBf16(x[j])` over a row, the GPT-NeoX MLP
activation (`nn.GELU()`, vLLM `get_act_fn("gelu")`).  Both were promoted from quarantine with their bodies verbatim, so
their ids keep the quarantine's `ln/` prefix; the Match fold binds them (`engine/profiles/gen_ln_patterns.py`), and `tests/engine/test_gen_ln.py`
checks the table's sanity facts: gelu(+0) = +0, gelu(-0) = -0, gelu(x) = x for x >= 2^4, gelu(x) = -0 for x <= -2^4, NaN words
-> 0x7FFF, +inf -> +inf, -inf -> NaN.
"""
from __future__ import annotations

import math

from verity.ml.circuits.defs import composite, primitive
from verity.ml.circuits.types import Array

from verity.ml.definitions.evidence import EVIDENCE
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.gelu_erf_table import TABLE_WORDS

#: `SEMANTIC_ID`s of the promoted Definitions (their quarantine identities; the quarantine `act` modules keep the same strings)
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
