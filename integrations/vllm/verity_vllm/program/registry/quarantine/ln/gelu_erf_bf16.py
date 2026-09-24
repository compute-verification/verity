"""`GeluErfBf16` — one output word of `aten.gelu(x, approximate='none')` on bf16 (torch 2.13.0+cu129
`aten/src/ATen/native/cuda/ActivationGeluKernel.cu`, `GeluCUDAKernelImpl`, `opmath_t = float`):

    return static_cast<opmath_t>(x) * opmath_t(0.5) * (opmath_t(1) + ::erf(static_cast<opmath_t>(x) * kAlpha));   // kAlpha = M_SQRT1_2

i.e. `bf16( (f * 0.5f) * (1.0f + erff(f * 0x3F3504F3)) )`, `f = (float)x`, stored through `c10::BFloat16(float)` (`cvt.rn.bf16.f32`,
NaN -> 0x7FFF).  `erff` is CUDA libdevice's `__nv_erff` (a minimax polynomial pair with FMAs, not correctly rounded), so the
word is NOT `RN(gelu(x))` for every x; rather than transcribe libdevice, the function is pinned by its **exhaustive table**: the
domain is the 65,536 bf16 words, and `gelu_erf_table.py` holds the kernel's output word for every one of them, measured on the
pod (RTX 4090, sm_89, torch 2.13.0+cu129) by `verity_vllm/ops/gelu_erf_table_probe.py` (`aten.gelu` on
`arange(65536).view(bfloat16)`, then zlib + base64).  The primitive is the table lookup; the docstring above is what the table is
a table *of*.  Precedent: `prims.MufuEx2Ftz` / `MufuRcpFtz` (measured hardware tables).

Sanity facts the table satisfies (checked by the lane's test): gelu(+0) = +0, gelu(-0) = -0, gelu(x) = x for x >= 2^4 (erff = 1),
gelu(x) = -0 for x <= -2^4 (erff = -1, `(-x * 0.5f) * 0 = -0`), NaN words -> 0x7FFF, +inf -> +inf, -inf -> NaN (`-inf * 0`)."""
from __future__ import annotations

from verity.ir.defs import primitive

from verity_vllm.program.registry.quarantine.ln._common import BF16
from verity_vllm.program.registry.quarantine.ln.gelu_erf_table import TABLE_WORDS

SEMANTIC_ID = "ln/GeluErfBf16@v1"


def gelu_erf_bf16_bits(x: int) -> int:
    """The kernel's output word for the bf16 input word `x` (table lookup)."""
    i = 2 * (x & 0xFFFF)
    return TABLE_WORDS[i] | (TABLE_WORDS[i + 1] << 8)


@primitive("GeluErfBf16", 1, [("x", BF16)], BF16,
           conformance="quarantine (lane cov-ln): exhaustive measured table (65,536/65,536 bf16 words, torch 2.13.0+cu129 aten.gelu on "
                       "sm_89); difftest vs torch.nn.functional.gelu on the pod")
def GeluErfBf16(x: int) -> int:
    """`bf16((f * 0.5f) * (1.0f + erff(f * M_SQRT1_2)))`, `f = (float)x` — as the pinned kernel computes it (measured table)."""
    return gelu_erf_bf16_bits(x)
