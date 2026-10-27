"""H1's served activation quantizer (co32c-u16) is bit-exact against steps.act_quant.

The GPU gate (benchmarks/pouw/edited_gemm/crun.sh) ties the served kernels crow_s + cquant_h1 to cgemm_ref.py's numpy
recomputation; this ties that same numpy formula to steps.act_quant, so the served X equals steps' H1 arm. act_quant's
H1 call is the int8 linear's dither at act_top 127, one bit (dither_shift(1) = 0, t = 126): s = amax / 126, then
X = floor((x / s).clamp(-126, 126)) + b. The served path upcasts bf16 activations to fp32 (the kernel reads uint16 and
shifts), so the check feeds fp32 and compares the pre-dither integer and the scale, both of which must match bit for bit.
Dividing matters: x / s and x * (1 / s) disagree in the last bit for some entries, so the formula below mirrors cgemm_ref's
H1 branch (one divide per entry, as crow_s stores amax / 126 and cquant_h1 divides), not a reciprocal multiply.
"""
import numpy as np
import pytest
import torch

from steps import act_quant, dither_shift


def h1_act_quant(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """crow_s + cquant_h1 + cgemm_ref.py's H1 branch, in fp32."""
    s = (np.maximum(np.abs(x).max(axis=1), np.float32(1e-30)) / np.float32(126.0)).astype(np.float32)
    return np.clip(np.floor(x / s[:, None]), -126, 126).astype(np.int64), s


@pytest.mark.parametrize("rows,cols,scale,seed", [(53, 517, 7.0, 0), (1, 128, 1.0, 1), (128, 4096, 20.0, 2),
                                                  (7, 257, 0.01, 3)])
def test_h1_matches_steps_act_quant(rows, cols, scale, seed):
    assert dither_shift(1) == 0
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(rows, cols, generator=g) * scale).to(torch.bfloat16).float()
    bits = torch.zeros_like(x)  # the dither bit b is added separately (SHA coins on the device); compare the floor map
    q_steps, s_steps = act_quant(x, "dither", bits, top=127, dither_bits=1)
    q_ref, s_ref = h1_act_quant(x.numpy().astype(np.float32))
    assert np.array_equal(s_steps.numpy().astype(np.float32), s_ref), "per-token scale amax/126 differs"
    assert np.array_equal(q_steps.numpy().astype(np.int64), q_ref), "floor(clamp(x/s)) differs"
    assert q_ref.max() <= 126 and q_ref.min() >= -126  # within int8's H1 band before the dither bit
