"""`pouw_h1_hooks.warm_zy`: one call compiles each kernel a committing run's z/y calls of 1..T rows would compile, so
serving compiles none, and a layer whose calls serve no z/y records is left alone."""
from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
PHK = pytest.importorskip("verity_vllm.protocols.pouw.h1.hooks")

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _Arm:
    zy_rows = None


class _Plain:
    """An arm with no z/y records."""


def _layer(n: int, k: int, arm: type = _Arm, device: str = "cpu") -> types.SimpleNamespace:
    g = torch.Generator().manual_seed(n * k)
    w = torch.randint(-127, 128, (n, k), dtype=torch.int8, generator=g).to(device)
    return types.SimpleNamespace(weight=w, weight_scale=(torch.rand(n, 1, generator=g) * 0.01 + 1e-4).to(device),
                                 quant_method=arm())


def test_each_shape_s_config_and_m_specialization_is_called_once():
    """Rows 1..64 at two configs (M <= 32, M > 32): M = 1, 1 < M with 16 not dividing it and 16 | M at the first, the last
    two at the second, once for each weight shape however many layers share it; an arm without ``zy_rows`` and an N that
    isn't whole tile columns are never called."""
    calls = []
    layers = [_layer(64, 32), _layer(64, 32), _layer(96, 64), _layer(64, 32, _Plain), _layer(48, 32)]

    def call(x, w, s_w, rows):
        calls.append((tuple(x.shape), str(x.dtype), int(x.abs().sum()), w is layers[0].weight or w is layers[2].weight,
                      rows))

    got = PHK.warm_zy(layers, 64, pick=lambda m, n, k: (32,) if m <= 32 else (64,), call=call)
    firsts = (1, 2, 16, 33, 48)
    assert got == len(calls) == 10
    assert calls == [((m, k), "torch.bfloat16", 0, True, 64) for k in (32, 64) for m in firsts]


def test_nothing_is_called_off_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert PHK.warm_zy([_layer(64, 32)], 64) == 0


def _compiled() -> int:
    from verity_vllm.engine import int8_fused as F

    return sum(len(c[0]) for fn in (F.exactlin_scaled_mm_zy, F.exactlin_scaled_mm_zy_tma) for c in fn.device_caches.values())


@cuda
@pytest.mark.slow
def test_cuda_a_call_of_any_rows_after_the_warm_up_compiles_nothing():
    """Llama-3.2-1B's int8 linear shapes at T 128: after `warm_zy`, `int8_ops.linear_zy` at every M from 1 to T on each
    shape adds no compiled kernel."""
    from verity_vllm.engine import int8_ops

    rows = 128
    layers = [_layer(n, k, device="cuda") for n, k in ((3072, 2048), (2048, 2048), (16384, 2048), (2048, 8192))]
    assert PHK.warm_zy(layers, rows) > 0
    before = _compiled()
    for layer in layers:
        for m in range(1, rows + 1):
            x = torch.randn(m, layer.weight.shape[1], device="cuda").bfloat16()
            int8_ops.linear_zy(x, layer.weight, layer.weight_scale, rows)
    torch.cuda.synchronize()
    assert _compiled() == before
