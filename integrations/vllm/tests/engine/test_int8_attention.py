"""`engine.int8_attention` against its oracle, `benchmarks/ppl`'s own `model.Layer.edited_attention` under the plain int8
attention step (`tests/engine/ppl_oracle.int8_attention`, steps.py's `int8_attention` with its defaults), bit for bit on the
CPU: whole windows (a prefill), last rows (a decode step), the paged caches vLLM hands the backend, and batches."""
from __future__ import annotations

import pytest
import torch

from verity_vllm.engine import int8_attention as A

from tests.engine import ppl_oracle as O


def _words(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _window(s: int, h: int, g: int, d: int, seed: int, scale: float = 1.0):
    gen = torch.Generator().manual_seed(seed)
    return tuple((torch.randn(s, n, d, generator=gen) * scale).bfloat16() for n in (h, g, g))


def _same(a: torch.Tensor, b: torch.Tensor) -> None:
    assert a.shape == b.shape and a.dtype == b.dtype == torch.bfloat16
    bad = int((_words(a) != _words(b)).sum())
    assert bad == 0, f"{bad} of {a.numel()} words differ"


@pytest.mark.parametrize("s, h, g, d", [(1, 4, 4, 8), (5, 8, 2, 64), (16, 8, 2, 64), (17, 4, 1, 32), (37, 8, 2, 64), (70, 32, 8, 128)])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 300.0])
def test_a_window_is_the_bench_attention(s, h, g, d, scale):
    q, k, v = _window(s, h, g, d, seed=s * 7 + h, scale=scale)
    _same(A.window_attention(q, k, v, d ** -0.5), O.int8_attention(q, k, v))


def test_each_decode_row_is_the_last_row_of_its_window():
    q, k, v = _window(40, 8, 2, 64, seed=3)
    for t in (1, 2, 9, 16, 17, 33, 40):
        _same(A.window_attention(q[t - 1:t], k[:t], v[:t], 64 ** -0.5), O.int8_attention(q[:t], k[:t], v[:t])[-1:])


def test_zero_and_floored_rows_and_channels():
    q, k, v = _window(24, 4, 2, 32, seed=5)
    q[3] = 0                                      # a zero query token: its scores are 0, P uniform over its keys
    q[7, 1] = 1e-31                               # one head's row under the 1e-30 floor
    k[0] = 0                                      # a zero key
    k[11, 0] = torch.tensor(2.0 ** -133).bfloat16()
    v[:, 1, 5] = 0                                # a zero value channel
    v[:, 0, 9] = 1e-31
    _same(A.window_attention(q, k, v, 32 ** -0.5), O.int8_attention(q, k, v))


def test_the_score_scale_is_the_product_of_the_two_scales_first():
    q, k, v = _window(64, 8, 2, 64, seed=11)
    qi, sq = A.quant(q[:, :4].transpose(0, 1).float(), -1)
    ki, sk = A.quant(k[:, 0].float(), -1)
    exact = A.int_mm(qi.reshape(-1, 64), ki).view(4, 64, 64).float()
    first, other = exact * (sq * sk.T[None]), (exact * sq) * sk.T[None]
    assert not torch.equal(first, other)          # the order is observable on this window, and the step takes the product first
    _same(A.window_attention(q, k, v, 64 ** -0.5), O.int8_attention(q, k, v))


def test_the_products_are_exact_past_float32():
    s, d = 1100, 128
    q = torch.full((s, 2, d), 3.0).bfloat16()     # Q' = K' = P' = V' = 127 in size: S = -d 127^2, the last row's P'V' 127^2 * 1100 > 2^24
    k = torch.full((s, 1, d), -2.0).bfloat16()
    v = torch.full((s, 1, d), 5.0).bfloat16()
    _same(A.window_attention(q, k, v, d ** -0.5), O.int8_attention(q, k, v))


@pytest.mark.parametrize("m, k, n", [(1, 1, 128), (4, 8, 128), (20, 24, 128), (33, 120, 128), (17, 129, 5)])
def test_the_products_hand_cublas_k_a_multiple_of_16(monkeypatch, m, k, n):
    gen = torch.Generator().manual_seed(m * k + n)
    a, b = (torch.randint(-127, 128, shape, generator=gen).float() for shape in ((m, k), (n, k)))
    shapes, real = [], A.E.int_mm
    monkeypatch.setattr(A.E, "int_mm", lambda x, w: shapes.append((x.shape, w.shape)) or real(x, w))
    assert torch.equal(A.int_mm(a, b).long(), a.long() @ b.long().T)
    ((xs, ws),) = shapes
    assert xs[1] == ws[1] and xs[1] % 16 == 0 and xs[1] >= k and ws[0] % 8 == 0 and ws[0] >= n


def test_the_softmax_is_the_callers():
    q, k, v = _window(9, 4, 2, 16, seed=2)
    seen = []

    def softmax(x, dim):
        seen.append(x.shape)
        return torch.softmax(x, dim)
    _same(A.window_attention(q, k, v, 0.25, softmax), A.window_attention(q, k, v, 0.25))
    assert seen == [(2, 9, 9)] * 2


def _paged(windows, block_size: int, seed: int):
    """Paged caches holding each window's k, v under a shuffled block table (vLLM's FlashAttention layout)."""
    g, d = windows[0][1].shape[1:]
    per = [-(-w[1].shape[0] // block_size) for w in windows]
    order = torch.randperm(sum(per) + 3, generator=torch.Generator().manual_seed(seed))
    kc = torch.zeros(sum(per) + 3, block_size, g, d, dtype=torch.bfloat16)
    vc = torch.zeros_like(kc)
    table = torch.full((len(windows), max(per)), -1, dtype=torch.int32)
    at = 0
    for r, (_, k, v) in enumerate(windows):
        for b in range(per[r]):
            blk = int(order[at])
            at += 1
            table[r, b] = blk
            rows = slice(b * block_size, min((b + 1) * block_size, k.shape[0]))
            kc[blk, :rows.stop - rows.start], vc[blk, :rows.stop - rows.start] = k[rows], v[rows]
    return kc, vc, table


def test_a_paged_batch_is_each_request_alone():
    lens, rows = [5, 40, 23, 1, 17], [5, 40, 1, 1, 1]          # two prefills, three decode steps
    windows = [_window(s, 8, 2, 32, seed=100 + i) for i, s in enumerate(lens)]
    kc, vc, table = _paged(windows, 16, seed=1)
    query = torch.cat([q[s - n:] for (q, _, _), s, n in zip(windows, lens, rows)])
    qsl = torch.tensor([0, *torch.tensor(rows).cumsum(0).tolist()], dtype=torch.int32)
    out = torch.full_like(query, float("nan"))
    A.paged_attention(query, kc, vc, qsl, torch.tensor(lens, dtype=torch.int32), table, 32 ** -0.5, out)
    for r, ((q, k, v), s, n) in enumerate(zip(windows, lens, rows)):
        got = out[qsl[r]:qsl[r + 1]]
        _same(got, O.int8_attention(q, k, v)[s - n:])
        alone = torch.empty_like(got)
        A.paged_attention(q[s - n:], kc, vc, torch.tensor([0, n], dtype=torch.int32), torch.tensor([s], dtype=torch.int32),
                          table[r:r + 1], 32 ** -0.5, alone)
        _same(got, alone)


def test_refusals():
    q, k, v = _window(8, 6, 4, 16, seed=1)
    with pytest.raises(ValueError, match="query heads"):
        A.window_attention(q, k, v, 0.25)                          # 6 heads over 4 KV heads
    q, k, v = _window(8, 4, 2, 16, seed=1)
    with pytest.raises(ValueError, match="rows of a"):
        A.window_attention(q, k[:4], v[:4], 0.25)                  # more rows than keys
    kc, vc, table = _paged([(q, k, v)], 16, seed=0)
    with pytest.raises(ValueError, match="chunked prefill"):
        A.paged_attention(q[:3], kc, vc, torch.tensor([0, 3], dtype=torch.int32), torch.tensor([8], dtype=torch.int32), table,
                          0.25, torch.empty_like(q[:3]))
