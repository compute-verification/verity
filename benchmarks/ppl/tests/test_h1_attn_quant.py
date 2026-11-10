"""H1's attention quantizers (co32c-u16) are bit-exact against steps.int8_attention's dither arm.

The GPU gates (benchmarks/pouw/edited_gemm/fattn.cu, apv.cu) tie the device's QK^T keys and queries and P.V's V to the
CPU references (fattn_ref.py, apv_ref.py); this test ties the numpy floor-quant formula those mirror to steps.floor_quant,
the function int8_attention calls, so the served X equals steps' H1 attention arm. The H1 arm floors Q and K per
(head, token) at top 111 and V per (KV head, channel) at top 127, each with one bit (dither_shift(1) = 0, added on the
device as SHA coins, so this compares the floor map and the scale, both of which must match bit for bit). The scale is the
smallest to 2^-20 that keeps x / s in the band, s = max(max(x) / (top - m), max(-x) / (top + 1 - m)) (1 + 2^-20); it is
asymmetric (max(x), not |x|), unlike the linears' act_quant. Dividing matters: x / s and x * (1 / s) disagree in the last
bit for some entries, so the formula below divides once per entry, as the kernels do (a per-row amax pass, then a divide).
"""
import numpy as np
import pytest
import torch

from steps import dither_shift, floor_quant

QK_TOP = 111
V_TOP = 127


def h1_floor_quant(x: np.ndarray, dims, top: int) -> tuple[np.ndarray, np.ndarray]:
    """fattn.cu's hquant_h1 / hwedit_h1 (Q, K) and apv.cu's V quantizer, in fp32: floor(x / s) with s per `dims`."""
    m = dither_shift(1)  # 0 for a one-bit dither
    amax = x.max(axis=dims, keepdims=True)
    namax = (-x).max(axis=dims, keepdims=True)
    s = np.maximum(amax / np.float32(top - m), namax / np.float32(top + 1 - m))
    s = (np.maximum(s, np.float32(1e-30)) * np.float32(1 + 2 ** -20)).astype(np.float32)
    return (np.floor(x / s) - m).astype(np.int64), s.astype(np.float32)


@pytest.mark.parametrize("heads,s,scale,seed", [(4, 127, 7.0, 0), (1, 128, 1.0, 1), (8, 512, 20.0, 2), (2, 63, 0.01, 3)])
def test_qk_matches_floor_quant(heads, s, scale, seed):
    """Q and K: per (head, token) floor over the head dimension (128) at top 111, the kernels' hquant_h1/hwedit_h1."""
    assert dither_shift(1) == 0
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(heads, s, 128, generator=g) * scale).to(torch.bfloat16).float()
    q_steps, s_steps = floor_quant(x, -1, QK_TOP)
    q_ref, s_ref = h1_floor_quant(x.numpy().astype(np.float32), -1, QK_TOP)
    assert np.array_equal(s_steps.numpy().astype(np.float32), s_ref), "per-(head, token) scale differs"
    assert np.array_equal(q_steps.numpy().astype(np.int64), q_ref), "floor(x / s) differs"
    assert q_ref.max() <= QK_TOP - 1 and q_ref.min() >= -QK_TOP - 1  # within [-112, 110] before the dither bit


@pytest.mark.parametrize("kvh,s,scale,seed", [(2, 127, 7.0, 0), (1, 128, 1.0, 1), (8, 512, 20.0, 2), (2, 65, 0.01, 3)])
def test_v_channel_matches_floor_quant(kvh, s, scale, seed):
    """V, v_scale = "channel": per (KV head, channel) floor over the tokens (dims = 0) at top 127, apv.cu's V quantizer."""
    g = torch.Generator().manual_seed(seed)
    v = (torch.randn(s, kvh, 128, generator=g) * scale).to(torch.bfloat16).float()
    q_steps, s_steps = floor_quant(v, 0, V_TOP)
    q_ref, s_ref = h1_floor_quant(v.numpy().astype(np.float32), 0, V_TOP)
    assert np.array_equal(s_steps.numpy().astype(np.float32), s_ref), "per-(KV head, channel) scale differs"
    assert np.array_equal(q_steps.numpy().astype(np.int64), q_ref), "floor(v / s) differs"
    assert q_ref.max() <= V_TOP and q_ref.min() >= -V_TOP - 1  # within [-128, 126] before the dither bit
