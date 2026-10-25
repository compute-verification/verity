"""`engine.int8_attention` against compute accounting's references, bit for bit on the CPU: `fattn_ref.check` itself judges the
LUT softmax of query rows (S, m, P, e(0), s_q, z_q), and `tests/engine/fattn_oracle.attention` (fattn_ref.py's softmax and
apv_ref.py's served f, no dither) the whole output: windows (a prefill), last rows (a decode step), the paged caches vLLM hands the
backend, and batches."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.engine import fattn_oracle as O
from verity_vllm.engine import int8_attention as A
from verity_vllm.engine import int8_ops


def _words(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _window(s: int, h: int, g: int, d: int, seed: int, scale: float = 1.0):
    gen = torch.Generator().manual_seed(seed)
    return tuple((torch.randn(s, n, d, generator=gen) * scale).bfloat16() for n in (h, g, g))


def _k_scales(k: torch.Tensor, spread: float = 1.0) -> torch.Tensor:
    """Per-KV-head scales as calibration would set them (amax / 127), times `spread` (< 1: some keys clamp)."""
    amax = k.float().abs().amax(dim=(0, 2)) * spread
    return (amax.clamp(min=1e-30) / torch.tensor(127.0)).float()


def _same(a: torch.Tensor, b: torch.Tensor) -> None:
    assert a.shape == b.shape and a.dtype == b.dtype == torch.bfloat16
    bad = int((_words(a) != _words(b)).sum())
    assert bad == 0, f"{bad} of {a.numel()} words differ"


def test_the_lut_is_fattn_refs():
    assert A.LUT.tolist() == O.LUT.tolist() and A.LUT[0] == 127 and A.LUT.dtype == np.int64


@pytest.mark.parametrize("s, h, g, d", [(1, 4, 4, 8), (5, 8, 2, 64), (16, 8, 2, 64), (17, 4, 1, 32), (37, 8, 2, 64),
                                        pytest.param(70, 32, 8, 128, marks=pytest.mark.slow)])
@pytest.mark.parametrize("scale, spread", [(1e-3, 1.0), (1.0, 1.0), (300.0, 1.0), (1.0, 0.3)])
def test_a_window_is_the_reference(s, h, g, d, scale, spread):
    q, k, v = _window(s, h, g, d, seed=s * 7 + h, scale=scale)
    ks = _k_scales(k, spread)
    _same(A.window_attention(q, k, v, ks, d ** -0.5), O.attention(q, k, v, ks, d ** -0.5))


def test_each_decode_row_is_the_last_row_of_its_window():
    q, k, v = _window(40, 8, 2, 64, seed=3)
    ks = _k_scales(k)
    for t in (1, 2, 9, 16, 17, 33, 40):
        _same(A.window_attention(q[t - 1:t], k[:t], v[:t], ks, 64 ** -0.5), O.attention(q[:t], k[:t], v[:t], ks, 64 ** -0.5)[-1:])


@pytest.mark.parametrize("s, pos, seed", [(64, 0, 1), (64, 63, 2), (200, 130, 3), (300, 299, 4)])
def test_fattn_ref_judges_the_softmax(tmp_path, s, pos, seed):
    q, k, _ = _window(s, 1, 1, 128, seed=seed)
    ks = _k_scales(k)
    qi, sq = A.quant(q[:, 0].float(), -1)
    ki = A.quant_static(k[:, 0].float(), ks[0])
    S = A.int_mm(qi, ki)
    a = A.fixed_point_a(sq, ks[0], 128 ** -0.5)
    causal = torch.arange(s)[None, :] <= torch.arange(s)[:, None]
    e, m = A.lut_e(S, a, causal)
    p, r, z = A.lut_softmax(S, a, causal)
    served = {"P": p[pos].numpy(), "m": int(m[pos, 0]), "e0": int(e[pos, 0]), "sp": r[pos].numpy(), "sz": z[pos].numpy()}
    assert int(a[pos, 0]) == O.fixed_point_a(sq[pos, 0].numpy(), ks[0].numpy(), 128 ** -0.5)
    got = O.check(O.gate_dir(tmp_path / "gate", ki.numpy(), qi[pos].numpy(), pos, int(a[pos, 0]), served, S[pos].numpy()))
    assert got["ok"], got
    (col,) = got["cols"]
    assert col["L"] == int(e[pos].sum()) and col["nonzero"] >= 1


def test_a_wrong_softmax_is_refused_by_fattn_ref(tmp_path):
    q, k, _ = _window(64, 1, 1, 128, seed=9)
    ks = _k_scales(k)
    qi, sq = A.quant(q[:, 0].float(), -1)
    ki = A.quant_static(k[:, 0].float(), ks[0])
    S = A.int_mm(qi, ki)
    a = A.fixed_point_a(sq, ks[0], 128 ** -0.5)
    causal = torch.arange(64)[None, :] <= torch.arange(64)[:, None]
    e, m = A.lut_e(S, a, causal)
    p, r, z = A.lut_softmax(S, a, causal)
    bad = p[50].clone()
    bad[7] += 1 if bad[7] < 127 else -1
    served = {"P": bad.numpy(), "m": int(m[50, 0]), "e0": int(e[50, 0]), "sp": r[50].numpy(), "sz": z[50].numpy()}
    got = O.check(O.gate_dir(tmp_path / "gate", ki.numpy(), qi[50].numpy(), 50, int(a[50, 0]), served, S[50].numpy()))
    assert not got["ok"] and got["cols"][0]["P"] is False


def test_zero_floored_and_clamped_values():
    q, k, v = _window(24, 4, 2, 32, seed=5)
    q[3] = 0                                      # a zero query token: S = 0, every e 127
    q[7, 1] = 1e-31                               # one head's row under the 1e-30 floor
    k[0] = 0                                      # a zero key
    k[11, 0] = torch.tensor(2.0 ** -133).bfloat16()
    k[5, 1] = 1e4                                 # far past the K scale: clamps to +-127
    v[:, 1, 5] = 0                                # a zero value channel
    v[:, 0, 9] = 1e-31
    ks = _k_scales(k[:5])
    _same(A.window_attention(q, k, v, ks, 32 ** -0.5), O.attention(q, k, v, ks, 32 ** -0.5))


def test_a_large_a_saturates_like_the_reference():
    q, k, v = _window(30, 2, 1, 16, seed=8, scale=50.0)
    ks = torch.tensor([100.0])                   # sigma far past 1 / (16 log2 e): A past 2^32, saturated at 2^32 - 1
    qi, sq = A.quant(q[:, :2].transpose(0, 1).float(), -1)
    a = A.fixed_point_a(sq, ks[0], 1.0)
    assert (a == A.A_CAP).all()
    _same(A.window_attention(q, k, v, ks, 1.0), O.attention(q, k, v, ks, 1.0))


def test_a_row_at_position_0_is_its_value():
    q, k, v = _window(1, 2, 1, 16, seed=4)
    vi, sv = A.quant(v[:, 0].float(), 0)
    _same(A.window_attention(q, k, v, _k_scales(k), 0.25)[0, 0], (vi[0] * sv[0]).to(torch.bfloat16))


def test_the_products_are_exact_past_float32():
    s, d = 1100, 128
    q = torch.full((s, 2, d), 3.0).bfloat16()     # Q' = K' = 127 everywhere: S equal, every e 127, P V' = 127^2 (pos) > 2^24
    k = torch.full((s, 1, d), -2.0).bfloat16()
    v = torch.full((s, 1, d), 5.0).bfloat16()
    ks = _k_scales(k)
    _same(A.window_attention(q, k, v, ks, d ** -0.5), O.attention(q, k, v, ks, d ** -0.5))


@pytest.mark.parametrize("m, k, n", [(1, 1, 128), (4, 8, 128), (20, 24, 128), (33, 120, 128), (17, 129, 5), (4, 64, 33)])
def test_the_products_hand_cublas_k_and_n_multiples_of_16(monkeypatch, m, k, n):
    gen = torch.Generator().manual_seed(m * k + n)
    a, b = (torch.randint(-127, 128, shape, generator=gen).float() for shape in ((m, k), (n, k)))
    shapes, real = [], A.E.int_mm
    monkeypatch.setattr(A.E, "int_mm", lambda x, w: shapes.append((x.shape, w.shape)) or real(x, w))
    assert torch.equal(A.int_mm(a, b).long(), a.long() @ b.long().T)
    ((xs, ws),) = shapes
    assert xs[1] == ws[1] and xs[1] % 16 == 0 and xs[1] >= k and ws[0] % 16 == 0 and ws[0] >= n


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
    ks = torch.tensor([0.02, 0.031])
    kc, vc, table = _paged(windows, 16, seed=1)
    query = torch.cat([q[s - n:] for (q, _, _), s, n in zip(windows, lens, rows)])
    qsl = torch.tensor([0, *torch.tensor(rows).cumsum(0).tolist()], dtype=torch.int32)
    out = torch.full_like(query, float("nan"))
    int8_ops.paged_attention(query, kc, vc, qsl, torch.tensor(lens, dtype=torch.int32), table, ks, 32 ** -0.5, out)
    for r, ((q, k, v), s, n) in enumerate(zip(windows, lens, rows)):
        got = out[qsl[r]:qsl[r + 1]]
        _same(got, O.attention(q[s - n:], k, v, ks, 32 ** -0.5))
        alone = torch.empty_like(got)
        int8_ops.paged_attention(q[s - n:], kc, vc, torch.tensor([0, n], dtype=torch.int32), torch.tensor([s], dtype=torch.int32),
                          table[r:r + 1], ks, 32 ** -0.5, alone)
        _same(got, alone)


def _both(query, windows, qsl, lens, ks, scale):
    """`paged_attention` and `window_attention` (the credited read's form) over the same requests: their two outputs."""
    kc, vc, table = _paged(windows, 16, seed=len(windows))
    paged, flat = torch.full_like(query, float("nan")), torch.full_like(query, float("nan"))
    seq_lens, qsl = torch.tensor(lens, dtype=torch.int32), torch.tensor(qsl, dtype=torch.int32)
    int8_ops.paged_attention(query, kc, vc, qsl, seq_lens, table, ks, scale, paged)
    offsets = [0]
    for s in lens:
        offsets.append(offsets[-1] + s)
    keys = torch.cat([k[:s] for (_, k, _), s in zip(windows, lens)])
    values = torch.cat([v[:s] for (_, _, v), s in zip(windows, lens)])
    int8_ops.window_attention(query, keys, values, offsets, qsl, seq_lens, ks, scale, flat)
    return paged, flat


@pytest.mark.parametrize("s, m", [(1, 1), (40, 1), (9, 9), (40, 40), (40, 7), (64, 33), (70, 64), (130, 64), (130, 2)])
def test_a_chunk_is_the_whole_prompt_of_its_window_row_for_row(s, m):
    """A request's m rows (1 <= m <= s, a chunked prefill's chunk between a decode step's 1 and a whole prompt's s) are positions
    s - m .. s - 1 over its s-token window: the same rows of that window's whole prompt at once, bit for bit, by both forms."""
    q, k, v = _window(s, 8, 2, 64, seed=s * 31 + m)
    ks = _k_scales(k)
    whole = A.window_attention(q, k, v, ks, 64 ** -0.5)
    for got in _both(q[s - m:], [(q, k, v)], [0, m], [s], ks, 64 ** -0.5):
        _same(got, whole[s - m:])
        _same(got, O.attention(q[s - m:], k, v, ks, 64 ** -0.5))


def test_a_prompt_served_in_chunks_beside_other_requests():
    """A 130-token prompt prefilled 64 rows a step (max_num_batched_tokens = 64's chunks: 64, 64, 2) in a batch with a decode step
    and a whole short prompt: each step's chunk is its window's whole prompt's rows, its last row the whole prompt's.  An earlier
    chunk reads its own window's V scales, so the first chunk's rows are not the 130-token prompt's."""
    scale = 32 ** -0.5
    q, k, v = _window(130, 8, 2, 32, seed=11)
    other = [_window(23, 8, 2, 32, seed=12), _window(5, 8, 2, 32, seed=13)]
    ks = _k_scales(torch.cat([k, other[0][1], other[1][1]]))
    whole = A.window_attention(q, k, v, ks, scale)
    for lo, hi in ((0, 64), (64, 128), (128, 130)):
        query = torch.cat([other[0][0][22:], q[lo:hi], other[1][0]])
        qsl, lens = [0, 1, 1 + hi - lo, 6 + hi - lo], [23, hi, 5]
        for got in _both(query, [other[0], (q, k, v), other[1]], qsl, lens, ks, scale):
            _same(got[1:1 + hi - lo], A.window_attention(q[:hi], k[:hi], v[:hi], ks, scale)[lo:])
            _same(got[:1], A.window_attention(*other[0], ks, scale)[22:])
            _same(got[1 + hi - lo:], A.window_attention(*other[1], ks, scale))
            if hi == 130:
                _same(got[hi - lo:1 + hi - lo], whole[-1:])
            if lo == 0:
                assert not torch.equal(_words(got[1:1 + hi - lo]), _words(whole[lo:hi]))


def test_refusals():
    q, k, v = _window(8, 6, 4, 16, seed=1)
    with pytest.raises(ValueError, match="query heads"):
        A.window_attention(q, k, v, torch.ones(4), 0.25)                # 6 heads over 4 KV heads
    q, k, v = _window(8, 4, 2, 16, seed=1)
    with pytest.raises(ValueError, match="K scales"):
        A.window_attention(q, k, v, torch.ones(3), 0.25)                # a scale per KV head
    with pytest.raises(ValueError, match="8 rows of a 4-key window"):
        A.window_attention(q, k[:4], v[:4], torch.ones(2), 0.25)        # more rows than keys
    with pytest.raises(ValueError, match="0 rows of a 8-key window"):
        A.window_attention(q[:0], k, v, torch.ones(2), 0.25)            # no rows
    q2, k2, v2 = _window(4, 2, 1, 264, seed=1)
    with pytest.raises(ValueError, match="head dim"):
        A.window_attention(q2, k2, v2, torch.ones(1), 0.25)


@pytest.mark.parametrize("qsl, s, m", [([0, 9], 8, 9), ([0, 1], 0, 1), ([3, 2], 8, -1)])
def test_a_request_whose_rows_are_not_its_windows_last_rows_is_refused(qsl, s, m):
    """More rows than its window has keys (a window of none bringing a row among them), or a negative count, by either form."""
    q, k, v = _window(9, 4, 2, 16, seed=1)
    kc, vc, table = _paged([(q, k, v)], 16, seed=0)
    args = (torch.tensor(qsl, dtype=torch.int32), torch.tensor([s], dtype=torch.int32))
    why = f"request 0 brings {m} rows of a {s}-token window; its rows are the window's last 1 .. {s}"
    with pytest.raises(ValueError, match=why):
        int8_ops.paged_attention(q, kc, vc, *args, table, torch.ones(2), 0.25, torch.empty_like(q))
    with pytest.raises(ValueError, match=why):
        int8_ops.window_attention(q, k[:s], v[:s], [0, s], *args, torch.ones(2), 0.25, torch.empty_like(q))


def test_the_cache_is_pages_slots_heads_k_then_v():
    """vLLM hands the backend a layer's cache as the logical [pages, KV heads, slots, 2D]; the int8 backend's (`LBNHC`) is laid out
    [pages][slots][KV heads][K‖V], 4 KiB a token-layer at the 8B; head-major pages, a split K and V, or a padded page are refused."""
    pages, g, n, d = 3, 8, 16, 128
    cache = torch.empty(pages, n, g, 2 * d, dtype=torch.bfloat16).transpose(1, 2)
    assert A.page_layout_refusal(cache, d) is None and g * 2 * d * cache.element_size() == 4096
    kc, vc = cache.transpose(1, 2).split(d, dim=-1)
    assert kc.stride() == vc.stride() == (n * g * 2 * d, g * 2 * d, 2 * d, 1) and vc.data_ptr() - kc.data_ptr() == d * 2
    assert "strides" in A.page_layout_refusal(torch.empty(pages, g, n, 2 * d, dtype=torch.bfloat16), d)
    assert "strides" in A.page_layout_refusal(torch.empty(pages, n + 1, g, 2 * d, dtype=torch.bfloat16)[:, :n].transpose(1, 2), d)
    assert "shape" in A.page_layout_refusal(torch.empty(pages, g, n, d, dtype=torch.bfloat16), d)
