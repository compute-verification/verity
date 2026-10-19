"""`engine.int8_ops`: each int8 linear is two named ops (its quantizer and its product) and each request's int8 attention two (its KV
quantization and its rows), with the values of the plain torch they wrap."""
from __future__ import annotations

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from verity_vllm.engine import int8_attention as A
from verity_vllm.engine import int8_exact as E
from verity_vllm.engine import int8_ops as OPS

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _Ops(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.seen: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.seen.append(str(func))
        return func(*args, **(kwargs or {}))


def _linear_args(m=5, k=64, n=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    return ((torch.randn(m, k, generator=g) * 3).bfloat16(), torch.randint(-127, 128, (n, k), generator=g, dtype=torch.int8),
            (torch.rand(n, generator=g) * 0.01 + 1e-4).float())


def test_the_linear_op_is_the_plain_linear_and_its_values():
    x, w, sw = _linear_args()
    xq, sa, y = OPS.linear(x, w, sw)
    wq, ws = E.quantize_rows(x)
    assert torch.equal(xq, wq) and torch.equal(sa, ws)
    assert torch.equal(y.view(torch.int16), E.plain_linear(x, w, sw).view(torch.int16))
    torch.library.opcheck(OPS.row_quant, (x,))
    torch.library.opcheck(OPS.scaled_mm, (xq, sa, w, sw))


def test_the_attention_ops_are_the_window():
    g = torch.Generator().manual_seed(3)
    s, h, kvh, d, bs = 21, 4, 2, 32, 8
    q, k, v = ((torch.randn(s, n, d, generator=g) * 2).bfloat16() for n in (h, kvh, kvh))
    blocks = torch.tensor([2, 0, 3], dtype=torch.int32)
    kc, vc = torch.zeros(4, bs, kvh, d, dtype=torch.bfloat16), torch.zeros(4, bs, kvh, d, dtype=torch.bfloat16)
    for t in range(s):
        kc[blocks[t // bs], t % bs], vc[blocks[t // bs], t % bs] = k[t], v[t]
    ks = torch.tensor([0.03, 0.05])
    kv = OPS.kv_quant(kc, vc, blocks, s, ks)
    assert all(torch.equal(a, b) for a, b in zip(kv, A.kv_quant(k, v, ks)))
    assert [t.dtype for t in kv] == [torch.int8, torch.int8, torch.float32, torch.float32]
    assert [tuple(t.shape) for t in kv] == [(kvh, s, d), (kvh, s, d), (kvh, d), (kvh, d)]
    for n in (s, 1):
        got = OPS.window_rows(q[s - n:], *kv, ks, 32 ** -0.5)
        assert torch.equal(got.view(torch.int16), A.window_attention(q[s - n:], k, v, ks, 32 ** -0.5).view(torch.int16))
    torch.library.opcheck(OPS.kv_quant, (kc, vc, blocks, s, ks))
    torch.library.opcheck(OPS.window_rows, (q[s - 1:], *kv, ks, 32 ** -0.5))


def test_a_capture_sees_one_op_per_call():
    x, w, sw = _linear_args(m=3)
    with _Ops() as mode:
        OPS.linear(x, w, sw)
    assert mode.seen == ["verity_int8.row_quant.default", "verity_int8.scaled_mm.default"]


@cuda
@pytest.mark.parametrize("m,k,n", [(1, 4096, 6144), (16, 4096, 4096), (17, 14336, 1024), (256, 4096, 1024)])
def test_the_cuda_kernels_are_the_torch_bodies(m, k, n):
    """On CUDA the two ops run `engine.int8_fused`'s kernels (Llama-3.1-8B's K, M on both sides of int8_exact's padding): xq,
    s_a and y bit for bit the torch bodies' on the CPU, and a capture still sees the two ops, one per call."""
    x, w, sw = _linear_args(m, k, n, seed=m)
    on_gpu = x.cuda(), w.cuda(), sw.cuda()
    with _Ops() as mode:
        got = OPS.linear(*on_gpu)
    assert mode.seen == ["verity_int8.row_quant.default", "verity_int8.scaled_mm.default"]
    xq, sa, y = (t.cpu() for t in got)
    wq, ws = E.quantize_rows(x)
    assert torch.equal(xq, wq) and torch.equal(sa, ws)
    assert torch.equal(y.view(torch.int16), E.plain_linear(x, w, sw).view(torch.int16))
    torch.library.opcheck(OPS.row_quant, (x.cuda(),))
    torch.library.opcheck(OPS.scaled_mm, (xq.cuda(), sa.cuda(), w.cuda(), sw.cuda()))
