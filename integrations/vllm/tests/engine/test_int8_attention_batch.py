"""The batched int8 attention (`int8_ops.paged_attention` and `window_attention`: one ``kv_quant_batch`` /
``kv_quant_window_batch`` and one ``window_rows_batch`` for the whole step) against the per-request reference
(`int8_attention.reference_paged`: `kv_quant` and `window_rows` one request and one KV head at a time), word for word on the
CPU, over random batches that mix decode rows, chunked-prefill chunks and whole prompts over varied windows and GQA groups.
`tests/engine/int8_batch_gpu.py` holds the CUDA body (`int8_attention_triton`) to the same reference on the GPU."""
from __future__ import annotations

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tests.engine.int8_batch_gpu import batch, words
from verity_vllm.engine import int8_attention as A
from verity_vllm.engine import int8_ops


class _Ops(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.seen: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.seen.append(str(func))
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("h, g, d, n_req, s_hi, bs, spread", [(8, 2, 64, 6, 120, 16, 1.0), (4, 1, 32, 9, 70, 8, 300.0),
                                                              (12, 4, 16, 5, 40, 16, 1e-3), (16, 2, 32, 12, 33, 4, 1.0),
                                                              (6, 6, 16, 3, 1, 16, 1.0), (8, 4, 64, 1, 150, 16, 0.5)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_random_batch_is_the_reference_word_for_word(h, g, d, n_req, s_hi, bs, spread, seed):
    query, kc, vc, qsl, lens, table, ks, scale = batch(torch.Generator().manual_seed(seed * 97 + h + n_req), h, g, d, n_req, s_hi, bs, spread)
    want, got = torch.full_like(query, float("nan")), torch.full_like(query, float("nan"))
    A.reference_paged(query, kc, vc, qsl, lens, table, ks, scale, want)
    int8_ops.paged_attention(query, kc, vc, qsl, lens, table, ks, scale, got)
    assert int((words(got) != words(want)).sum()) == 0
    keys = torch.cat([A.window_of(kc, vc, table[r], s)[0] for r, s in enumerate(lens.tolist())])
    values = torch.cat([A.window_of(kc, vc, table[r], s)[1] for r, s in enumerate(lens.tolist())])
    offsets = [0, *lens.cumsum(0).tolist()]
    flat = torch.full_like(query, float("nan"))
    int8_ops.window_attention(query, keys, values, offsets, qsl, lens, ks, scale, flat)
    assert torch.equal(words(flat), words(want))


def test_each_requests_slice_of_the_batch_kv_quantization_is_its_kv_quant():
    query, kc, vc, qsl, lens, table, ks, _ = batch(torch.Generator().manual_seed(4), 8, 2, 32, 7, 50, 8, 1.0)
    ls = lens.tolist()
    kq, vq, sv, v0 = int8_ops.kv_quant_batch(kc, vc, table, ls, ks)
    assert kq.shape == vq.shape == (7, 2, max(ls), 32) and sv.shape == v0.shape == (7, 2, 32)
    for r, s in enumerate(ls):
        want = A.kv_quant(*A.window_of(kc, vc, table[r], s), ks)
        for x, y in zip((kq[r, :, :s], vq[r, :, :s], sv[r], v0[r]), want):
            assert torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))
        assert not kq[r, :, s:].any() and not vq[r, :, s:].any()


def test_a_step_is_two_ops_whatever_its_requests():
    """One kv_quant_batch and one window_rows_batch a layer: no op runs once per request or KV head."""
    for n_req in (1, 9):
        query, kc, vc, qsl, lens, table, ks, scale = batch(torch.Generator().manual_seed(n_req), 8, 2, 32, n_req, 40, 8, 1.0)
        out = torch.empty_like(query)
        with _Ops() as mode:
            int8_ops.paged_attention(query, kc, vc, qsl, lens, table, ks, scale, out, host=(qsl.tolist(), lens.tolist()))
        named = [f for f in mode.seen if f.startswith("verity_int8.")]
        assert named == ["verity_int8.kv_quant_batch.default", "verity_int8.window_rows_batch.default"], named


def test_a_request_with_no_rows_has_no_window():
    """vLLM may hand a request no rows in a step; it is skipped (its output rows untouched, its window quantized by none)."""
    query, kc, vc, qsl, lens, table, ks, scale = batch(torch.Generator().manual_seed(8), 8, 2, 32, 4, 30, 8, 1.0)
    rows = (qsl[1:] - qsl[:-1]).tolist()
    keep = [0, 2, 3]
    q2 = torch.cat([query[qsl[r]:qsl[r + 1]] for r in keep])
    qsl2 = torch.tensor([0, rows[0], rows[0], rows[0] + rows[2], rows[0] + rows[2] + rows[3]], dtype=torch.int32)
    want, got = torch.full_like(q2, float("nan")), torch.full_like(q2, float("nan"))
    A.reference_paged(q2, kc, vc, qsl2, lens, table, ks, scale, want)
    seen = []
    real = int8_ops.kv_quant_batch
    int8_ops.kv_quant_batch = lambda *a: seen.append(list(a[3])) or real(*a)
    try:
        int8_ops.paged_attention(q2, kc, vc, qsl2, lens, table, ks, scale, got)
    finally:
        int8_ops.kv_quant_batch = real
    assert torch.equal(words(got), words(want))
    assert seen == [[s if r != 1 else 0 for r, s in enumerate(lens.tolist())]]


def test_the_batch_ops_check_as_custom_ops():
    query, kc, vc, qsl, lens, table, ks, scale = batch(torch.Generator().manual_seed(2), 4, 2, 16, 3, 20, 8, 1.0)
    ls, q = lens.tolist(), qsl.tolist()
    kv = int8_ops.kv_quant_batch(kc, vc, table, ls, ks)
    torch.library.opcheck(int8_ops.kv_quant_batch, (kc, vc, table, ls, ks))
    torch.library.opcheck(int8_ops.window_rows_batch, (query, *kv, ks, q, ls, scale))
    keys = torch.cat([A.window_of(kc, vc, table[r], s)[0] for r, s in enumerate(ls)])
    values = torch.cat([A.window_of(kc, vc, table[r], s)[1] for r, s in enumerate(ls)])
    starts = [0, *lens.cumsum(0).tolist()][:-1]
    torch.library.opcheck(int8_ops.kv_quant_window_batch, (keys, values, starts, ls, ks))
