"""The three Triton helpers `benchmarks/hash/dither/dither_a.py` inlines into its own kernels: the round-to-nearest-even
cast (`_rni_s32`), the split-K memory barrier (`_membar_gl`) and the epilogue, `int8_exact.dequantize` (`_epilogue`).
Each is the text of `verity_vllm.engine.int8_fused`'s helper of the same name, which is private there
(`tests/test_helpers.py` holds the two to each other).  Everything else the benches here and dither's arms run is
`int8_fused` itself: its kernels, its `Cfg`, `pick` and `Workspace`, and the tiles of `engine.profiles.int8_gemm_configs`.
"""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _rni_s32(v):
    return tl.inline_asm_elementwise("cvt.rni.s32.f32 $0, $1;", "=r,r", [v], dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _membar_gl(v):
    return tl.inline_asm_elementwise("membar.gl;\n\tmov.b32 $0, $1;", "=r,r", [v], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _epilogue(acc, sa_ptr, sb_ptr, c_ptr, offs_m, offs_n, mask_m, mask_n, stride_cm):
    sa = tl.load(sa_ptr + offs_m, mask=mask_m, other=0.0)
    sb = tl.load(sb_ptr + offs_n, mask=mask_n, other=0.0)
    y = acc.to(tl.float32) * sa[:, None]
    y = y * sb[None, :]
    tl.store(c_ptr + stride_cm * offs_m[:, None] + offs_n[None, :], y.to(tl.bfloat16),
             mask=mask_m[:, None] & mask_n[None, :])
