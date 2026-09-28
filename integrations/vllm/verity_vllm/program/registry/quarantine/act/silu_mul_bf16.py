"""`SiluMulBf16_v2` — one output word of vLLM `_C.silu_and_mul` (d9105ea80 `csrc/libtorch_stable/activation_kernels.cu`,
`act_and_mul_kernel<bf16, silu_kernel, true>`, the vec and scalar paths alike):

    silu_kernel(x):  (T)(f / (1.0f + expf(-f)))          f = (float)x; cvt.rn.bf16.f32 (NaN -> 0x7FFF)
    out = (T)(ACT(gate) * ((float)up + beta))           beta = 0.0f: up = -0 enters as +0, identity otherwise

`SiluMulBf16_v1` (prims) is this function except on the words the red-team measured on sm_120 (labels on
r20260930-092038-7dbc, `silu-edge-cases`):
  * NaN results are 0x7FFF, not 0x7FC0;
  * up = -0 is +0 after `+ beta`, so the product's zero takes the gate's sign (`[g=0x3F80 | u=0x8000]` -> 0x0000);
  * gate -inf is `-inf / (1 + inf)` = NaN, not the limit -0;
  * gates whose `expf(-g)` overflows (bf16 g <= -89, past ln(FLT_MAX) = 88.72) are `g / inf` = -0, not a tiny g * e^g;
  * gate -0 is -0 (`_v1` decodes it to an unsigned zero).
Every other word is `_v1`'s: `silu_f32` is the same RN32-of-the-exact-real stand-in for expf + div.rn.
"""
from __future__ import annotations

import math

from verity.ir.defs import primitive

from verity_vllm.program.registry import prims as P

SEMANTIC_ID = "act/SiluMulBf16@v2"

BF16 = P.BF16
F32_NEG_ZERO = 0x80000000
ZERO_BITS = 0x00000000
#: ln of the smallest real that RN32 sends to +inf, (2 - 2^-24) * 2^127: no f32 word lies within float64 error of it
LN_F32_OVERFLOW = math.log((2.0 - 2.0 ** -24) * 2.0 ** 127)


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


@primitive("SiluMulBf16", 2, [("g", BF16), ("u", BF16)], BF16,
           conformance="quarantine (lane vllm-coverage-defs): sm_120 red-team edge words (r20260930-092038-7dbc); "
                       "SiluMulBf16_v1 on every other word")
def SiluMulBf16(g: int, u: int) -> int:
    """`bf16(f32(bf16(silu_f32(g))) * (f32(u) + 0.0f))`, both conversions `cvt.rn.bf16.f32` (NaN -> 0x7FFF)."""
    s = P.f32_to_bf16_rn_bits(silu_f32_bits(P.bf16_to_f32_bits(g)), P.BF16_NAN_F2FP)
    uf = P.f32_add_bits(P.bf16_to_f32_bits(u), ZERO_BITS)
    return P.f32_to_bf16_rn_bits(P.f32_mul_bits(P.bf16_to_f32_bits(s), uf), P.BF16_NAN_F2FP)
