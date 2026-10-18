"""`registry.int8_exact`'s Definitions against the torch the int8 profile serves (`engine.int8_exact`, `engine.int8_attention`),
word for word on the CPU, and its primitives at their edges."""
from __future__ import annotations

import math
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity_vllm.engine import int8_attention as IA
from verity_vllm.engine import int8_exact as E
from verity_vllm.program.registry import int8_exact as X
from verity_vllm.program.registry.prims import bits_of, f32_of


def _bf16(t: torch.Tensor) -> list[int]:
    return [int(w) & 0xFFFF for w in t.contiguous().view(torch.int16).reshape(-1).tolist()]


def _f32(t: torch.Tensor) -> list[int]:
    return [int(w) & 0xFFFFFFFF for w in t.contiguous().view(torch.int32).reshape(-1).tolist()]


def _s8(t: torch.Tensor) -> list[int]:
    return [int(w) & 0xFF for w in t.to(torch.int8).reshape(-1).tolist()]


def _f64(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def _rows(g: torch.Generator, *shape: int, spread: float = 4.0) -> torch.Tensor:
    mag = torch.exp(torch.randn(*shape, generator=g) * spread / 4)
    return (torch.randn(*shape, generator=g) * mag).to(torch.bfloat16)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_int8_linear_is_the_served_linear(seed):
    g = torch.Generator().manual_seed(seed)
    K, N, M = 64, 8, 3
    x = _rows(g, M, K)
    x[0, :] = 0                       # the floored scale
    w = torch.randint(-127, 128, (N, K), generator=g, dtype=torch.int8)
    sw = (torch.rand(N, generator=g) * 0.01 + 1e-4).float()
    xq, sa = E.quantize_rows(x)
    y = E.plain_linear(x, w, sw)
    defn = bind(X.Int8Linear, K=K, N=N)
    for i in range(M):
        got = evaluate(defn, _bf16(x[i]), _s8(w), _f32(sw))
        assert list(got[:K]) == _s8(xq[i]) and got[K] == _f32(sa[i:i + 1])[0] and list(got[K + 1:]) == _bf16(y[i])


@pytest.mark.parametrize("n,s,h,g_heads,spread,ks_scale", [(5, 5, 2, 1, 4.0, 1.0), (1, 40, 4, 2, 4.0, 1.0),
                                                           (3, 33, 2, 2, 8.0, 0.25), (2, 6, 2, 1, 1.0, 40.0)])
def test_int8_lut_attention_is_the_served_attention(n, s, h, g_heads, spread, ks_scale):
    """`Int8KvQuant` per KV head is `kv_quant`'s values, and `Int8LutAttentionRow` per (row, head) over its first pos + 1 rows is
    `window_attention`: prefill (n = s) and decode rows, keys past one k step (s > 32), K clamped by a small scale (ks_scale < 1)
    and A saturated by a large one."""
    g = torch.Generator().manual_seed(n * 1000 + s)
    d, scale = 32, 32 ** -0.5
    q, k, v = _rows(g, n, h, d, spread=spread), _rows(g, s, g_heads, d, spread=spread), _rows(g, s, g_heads, d, spread=spread)
    ks = (k.float().abs().amax(dim=(0, 2)) / 127 * ks_scale).float()
    kq, vq, sv, v0 = IA.kv_quant(k, v, ks)
    want = IA.window_attention(q, k, v, ks, scale)
    kvd = bind(X.Int8KvQuant, S=s, D=d)
    for gi in range(g_heads):
        got = evaluate(kvd, _bf16(k[:, gi]), _bf16(v[:, gi]), _f32(ks[gi:gi + 1])[0])
        sd = s * d
        assert list(got[:sd]) == _s8(kq[gi]) and list(got[sd:2 * sd]) == _s8(vq[gi])
        assert list(got[2 * sd:2 * sd + d]) == _f32(sv[gi]) and list(got[2 * sd + d:]) == _f32(v0[gi])
    rep = h // g_heads
    for i in range(n):
        p = s - n + i + 1
        row = bind(X.Int8LutAttentionRow, D=d, S=p, SCALE=scale)
        for hd in range(h):
            gi = hd // rep
            got = evaluate(row, _bf16(q[i, hd]), _s8(kq[gi, :p]), _f32(ks[gi:gi + 1])[0], _s8(vq[gi, :p]), _f32(sv[gi]), _f32(v0[gi]))
            assert list(got) == _bf16(want[i, hd]), (i, hd)


@pytest.mark.parametrize("s", [1, 2, 33])
def test_attention_rows_cut_under_q_word(s):
    """A prefill's first row (S = 1, its softmax over one key) cuts under Q_word like the longer ones: no gate recomputed."""
    from verity_vllm.query import word as W

    assert W.unit_rule(bind(X.Int8LutAttentionRow, D=128, S=s, SCALE=128 ** -0.5))["violations"] == []


@pytest.mark.parametrize("sq,sk,scale", [(0.01, 0.02, 128 ** -0.5), (1e-30 / 127, 1e-30 / 127, 0.125), (3.0, 100.0, 1.0)])
def test_attn_scale_is_fa_a(sq, sk, scale):
    tq, tk = torch.tensor([sq], dtype=torch.float32), torch.tensor(sk, dtype=torch.float32)
    want = int(IA.fixed_point_a(tq, tk, scale)[0])
    assert evaluate(bind(X.Int8AttnScale, SCALE=scale), _f32(tq)[0], _f32(tk.reshape(1))[0]) == (want,)


def test_lut_softmax_is_the_served_softmax():
    g = torch.Generator().manual_seed(7)
    S = torch.randint(-20000, 20000, (1, 37), generator=g)
    a = torch.tensor([[3 << 28]], dtype=torch.int64)
    p, r, z = IA.lut_softmax(S, a, torch.ones(1, 37, dtype=torch.bool))
    got = evaluate(bind(X.Int8LutSoftmax, S=37), [int(v) & 0xFFFFFFFF for v in S[0].tolist()], int(a))
    assert list(got[:37]) == _s8(p[0]) and got[37] == _f32(r)[0] and got[38] == _f32(z)[0]


def test_primitives_at_their_edges():
    assert X.LUT == tuple(int(v) for v in IA.LUT)
    for v, want in [(0.5, 0.0), (1.5, 2.0), (-2.5, -2.0), (126.5, 126.0), (-0.3, -0.0)]:
        assert X.F32Rint.evaluate(bits_of(v)) == bits_of(want)
    nan, inf = 0x7FC00000, 0x7F800000
    assert X.F32Rint.evaluate(nan) == nan and X.F32Rint.evaluate(inf) == inf
    assert [X.F32ToS8Rzi.evaluate(bits_of(v)) for v in (-127.0, 127.0, -0.0, 300.0)] == [0x81, 0x7F, 0, 0x7F]
    assert X.F32ToS8Rzi.evaluate(nan) == 0 and X.F32ToS8Rzi.evaluate(inf | 0x80000000) == 0x80
    assert [float(f32_of(X.S8ToF32.evaluate(c))) for c in (0x81, 0x80, 0x7F)] == [-127.0, -128.0, 127.0]
    assert X.F64ToU32RniSat.evaluate(_f64(2.0 ** 50)) == 0xFFFFFFFF and X.F64ToU32RniSat.evaluate(_f64(2.5)) == 2
    assert X.F64ToU32RniSat.evaluate(_f64(-1.0)) == 0 and X.F64ToU32RniSat.evaluate(_f64(math.nan)) == 0
    assert X.LutIndex.evaluate(5, 0xFFFFFFFF) == 4 and X.LutIndex.evaluate(300, 0xFFFFFFFF) == 255
    assert X.LutIndex.evaluate(0xFFFFFFFF, 0xFFFFFFFF) == 0 and X.LutIndex.evaluate(3, (1 << 31) + 1) == 1
    assert X.Exp2Lut127.id == "Exp2Lut127_v1" and [p.w for _, p in X.Exp2Lut127.params] == [8] and X.Exp2Lut127.ret.w == 32
    assert [X.Exp2Lut127.evaluate(t) for t in range(256)] == list(X.LUT) and X.Exp2Lut127.evaluate(0) == 127 and min(X.LUT) == 0
    assert X.F64Mul.evaluate(_f64(0.1), _f64(3.0)) == _f64(0.1 * 3.0)
    tiny, big = np.float32(1e-30) / np.float32(127), np.float32(3.4e38)
    for a, b in [(np.float32(0.1), np.float32(0.3)), (tiny, tiny), (big, big), (np.float32(1e-45), np.float32(-2.0))]:
        assert X.F32MulWide.evaluate(bits_of(a), bits_of(b)) == _f64(float(a) * float(b))


def test_matmuls_are_the_plain_s8_arm():
    from verity.ml.circuits.ops import s8 as S8

    for K, N, A in [(4096, 6144, (-127, 127)), (14336, 4096, (-127, 127)), (X.padded(4096), 128, (0, 127))]:
        st = X.plain_arm(K, N, A)
        assert S8.no_wrap(SimpleNamespace(**st)) and st["NOISE"] == "none" and st["COMMIT"] == "final" and not st["TAIL"]
    assert X.padded(1) == 32 and X.padded(32) == 32 and X.padded(33) == 64
