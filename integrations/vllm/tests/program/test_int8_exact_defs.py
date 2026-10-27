"""`registry.int8_exact`'s Definitions against the torch the int8 profile serves (`engine.int8_exact`, `engine.int8_attention`),
word for word on the CPU, and its primitives at their edges."""
from __future__ import annotations

import json
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
from verity.ml.definitions.registry import int8_exact as X
from verity.ml.definitions.registry.prims import bits_of, f32_of


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


@pytest.mark.parametrize("K", [1, 7, 64, 4096])
def test_the_replay_row_quantizer_is_the_definition_and_the_served_quantizer(K):
    """`kernels.int8_rows` (the Commit's sampled replay of `Int8RowQuant_v1`) word for word against the Definition and
    `engine.int8_exact.quantize_rows` on rows at four spreads, a zero row (the floored scale), a row of subnormals, -0 and +-1,
    and one whose max is bf16's largest finite."""
    from verity.ml.definitions.numerics import int8_rows as Q
    from verity.ml.definitions.numerics import rows

    g = torch.Generator().manual_seed(K)
    x = torch.cat([_rows(g, 4, K, spread=sp) for sp in (0.5, 4.0, 12.0, 30.0)])
    edge = torch.tensor([w - (w >> 15 << 16) for w in (0x0001, 0x8001, 0x8000, 0x3F80, 0xBF80, 0x007F)] * (-(-K // 6)),
                        dtype=torch.int16)[:K]
    big = _rows(g, 1, K).view(torch.int16).clone()
    big[0, K // 2] = 0x7F7F
    x = torch.cat([x, torch.zeros(1, K, dtype=torch.bfloat16), edge.view(torch.bfloat16)[None], big.view(torch.bfloat16)])
    codes, s = Q.int8_row_quant_rows(x.view(torch.int16).numpy().view(np.uint16))
    xq, sa = E.quantize_rows(x)
    assert codes.tolist() == [_s8(r) for r in xq] and s.tolist() == _f32(sa)
    defn = bind(X.Int8RowQuant, K=K)
    for i in (0, 5, 15, 16, 17, 18) if K == 4096 else range(x.shape[0]):
        assert list(evaluate(defn, _bf16(x[i]))) == codes[i].tolist() + [int(s[i])], i
    assert rows.ROWS[Q.FAMILY].returns == ("0", "1") and Q.FAMILY in rows.EVALUATORS


def _u8(t: torch.Tensor) -> np.ndarray:
    return np.asarray(_s8(t), dtype=np.uint8)


def _w32(t: torch.Tensor) -> np.ndarray:
    return np.asarray(_f32(t), dtype=np.uint32)


def _w16(t: torch.Tensor) -> np.ndarray:
    return np.asarray(_bf16(t), dtype=np.uint16)


@pytest.mark.parametrize("K,N,seed", [(64, 8, 0), (32, 3, 1), (4096, 40, 2)])
def test_the_replay_linear_is_the_definition_and_the_served_linear(K, N, seed):
    """`kernels.int8_rows.scaled_mm` (the replay of `ScaledMmS8_v1`) word for word against `engine.int8_exact.plain_linear` on
    quantized rows (a zero row among them) and against the Definition, there and on codes the quantizer never makes (-128),
    with DOT spelt as the spec text, a bare id and the bound primitive; another DOT is declined by name."""
    from verity.ml.circuits.ops import gemm
    from verity.ml.definitions.numerics import int8_rows as Q
    from verity.ml.definitions.numerics import rows
    from verity.ml.definitions.numerics.kernel_registry import Declined

    g = torch.Generator().manual_seed(seed)
    x = _rows(g, 3, K)
    x[0, :] = 0
    w = torch.randint(-127, 128, (N, K), generator=g, dtype=torch.int8)
    sw = (torch.rand(N, generator=g) * 0.01 + 1e-4).float()
    xq, sa = E.quantize_rows(x)
    y = E.plain_linear(x, w, sw)
    defn = bind(gemm.ScaledMmS8, K=K, N=N, **gemm.GEMM_SM120_S8)
    spellings = ['{"fn":"BlackwellS8ImmaDot32_v1"}', "BlackwellS8ImmaDot32_v1", gemm.GEMM_SM120_S8["DOT"]]
    wild = torch.randint(-128, 128, (K,), generator=g, dtype=torch.int8)
    for i, dot in zip(range(3), spellings):
        st = {"K": K, "N": N, "DOT": dot}
        got = Q.scaled_mm(st, [_u8(xq[i]), _w32(sa[i:i + 1]), _u8(w), _w32(sw)])["0"]
        assert got.tolist() == _bf16(y[i]), i
        if K <= 64:
            assert list(evaluate(defn, _s8(xq[i]), _f32(sa[i:i + 1])[0], _s8(w), _f32(sw))) == got.tolist()
            wg = Q.scaled_mm(st, [_u8(wild), _w32(sa[i:i + 1]), _u8(w), _w32(sw)])["0"]
            assert list(evaluate(defn, _s8(wild), _f32(sa[i:i + 1])[0], _s8(w), _f32(sw))) == wg.tolist()
    with pytest.raises(Declined, match="BlackwellU8S8ImmaDot32_v1"):
        Q.scaled_mm({"K": K, "N": N, "DOT": "BlackwellU8S8ImmaDot32_v1"}, [_u8(xq[0]), _w32(sa[:1]), _u8(w), _w32(sw)])
    assert rows.ROWS[Q.SCALED_MM].returns == ("0",) and Q.SCALED_MM in rows.EVALUATORS


@pytest.mark.parametrize("Y,sx,sw", [(2925, 0x3D3C0A61, 0x3B6EC803), (1868, 0x3D1C2BBD, 0x3C23183A), (130, 0x3D249B95, 0x3B464D31),
                                     (3501, 0x3CF55E64, 0x3C1A1B2E)])
def test_the_replay_linear_scales_by_the_token_first(Y, sx, sw):
    """Rows whose bf16 word tells the epilogue's order apart (searched: about 4 in 10^6 random ones do): the replay's word is the
    Definition's, (Y sx) sw, and not (Y sw) sx."""
    from verity.ml.circuits.ops import gemm
    from verity.ml.definitions.numerics import derived_rows as D
    from verity.ml.definitions.numerics import int8_rows as Q

    x, w = [1] * 32, [127] * (Y // 127) + [Y % 127] + [0] * (31 - Y // 127)
    got = Q.scaled_mm({"K": 32, "N": 1, "DOT": "BlackwellS8ImmaDot32_v1"}, [np.asarray(x, np.uint8), np.asarray([sx], np.uint32),
                                                                             np.asarray(w, np.uint8), np.asarray([sw], np.uint32)])["0"]
    y = np.asarray([Y], np.float32).view(np.uint32)
    other = D.f2fp_bf16(D.mul_rn(D.mul_rn(y, np.asarray([sw], np.uint32)), np.asarray([sx], np.uint32)))
    assert got.tolist() != other.tolist()
    assert list(evaluate(bind(gemm.ScaledMmS8, K=32, N=1, **gemm.GEMM_SM120_S8), x, sx, w, [sw])) == got.tolist()


def test_the_replay_s8_chain_clamps_each_step():
    """`s8_chain` is the `BlackwellS8ImmaDot32_v1` chain step for step where a step clamps: 4,097 steps of 32 x (-128)^2 past
    2^31 - 1, then 64 of -(128 x 127) back down, one long row against its negation."""
    from verity.ml.circuits.ops import prims as CP
    from verity.ml.definitions.numerics import int8_rows as Q

    K = (4097 + 64) * 32
    a = np.full(K, -128, dtype=np.int64)
    b = np.concatenate([np.full(4097 * 32, -128), np.full(64 * 32, 127)]).astype(np.int64)
    got = Q.s8_chain(a, np.stack([b, -b.clip(-127)]))
    for row, want in zip((b, -b.clip(-127)), got):
        acc = 0
        for s in range(K // 32):
            acc = CP.BlackwellS8ImmaDot32.evaluate(acc, *(a[32 * s:32 * s + 32] & 0xFF).tolist(), *(row[32 * s:32 * s + 32] & 0xFF).tolist())
        assert (acc - (1 << 32) if acc >> 31 else acc) == int(want)
    assert int(got[0]) == Q.S32_MAX - 64 * 32 * 128 * 127


@pytest.mark.parametrize("n,s,h,g_heads,spread,ks_scale", [(5, 5, 2, 1, 4.0, 1.0), (1, 40, 4, 2, 4.0, 1.0),
                                                           (3, 33, 2, 2, 8.0, 0.25), (2, 6, 2, 1, 1.0, 40.0)])
def test_the_replay_attention_is_the_definition_and_the_served_attention(n, s, h, g_heads, spread, ks_scale):
    """`kernels.int8_rows.kv_quant` and `lut_attention_row` (the replay of `Int8KvQuant_v1` and `Int8LutAttentionRow_v1`) word for
    word against `engine.int8_attention.kv_quant` / `window_attention` on every KV head and (row, head), and against the Definitions
    on one KV head and the last row's heads: prefill, chunk (1 < n < s) and decode rows, a window past one k step, K clamped by a
    small scale and A saturated by a large one, SCALE spelt as a float, the spec text's f64 literal and a decimal string."""
    from verity.ml.definitions.numerics import int8_rows as Q
    from verity.ml.definitions.numerics import rows

    g = torch.Generator().manual_seed(n * 1000 + s)
    d, scale = 32, 32 ** -0.5
    q, k, v = _rows(g, n, h, d, spread=spread), _rows(g, s, g_heads, d, spread=spread), _rows(g, s, g_heads, d, spread=spread)
    ks = (k.float().abs().amax(dim=(0, 2)) / 127 * ks_scale).float()
    kq, vq, sv, v0 = IA.kv_quant(k, v, ks)
    want = IA.window_attention(q, k, v, ks, scale)
    for gi in range(g_heads):
        got = Q.kv_quant({"S": s, "D": d}, [_w16(k[:, gi]), _w16(v[:, gi]), _w32(ks[gi:gi + 1])])
        assert [got[m].tolist() for m in "0123"] == [_s8(kq[gi]), _s8(vq[gi]), _f32(sv[gi]), _f32(v0[gi])], gi
    flat = Q.kv_quant({"S": s, "D": d}, [_w16(k[:, 0]), _w16(v[:, 0]), _w32(ks[:1])])
    assert list(evaluate(bind(X.Int8KvQuant, S=s, D=d), _bf16(k[:, 0]), _bf16(v[:, 0]), _f32(ks[:1])[0])) == \
        np.concatenate([flat[m].astype(np.uint64) for m in "0123"]).tolist()
    rep = h // g_heads
    spellings = (scale, json.dumps({"f64": scale.hex()}), repr(scale))
    for i in range(n):
        p = s - n + i + 1
        for hd in range(h):
            gi = hd // rep
            ops = [_w16(q[i, hd]), _u8(kq[gi, :p]), _w32(ks[gi:gi + 1]), _u8(vq[gi, :p]), _w32(sv[gi]), _w32(v0[gi])]
            got = Q.lut_attention_row({"D": d, "S": p, "SCALE": spellings[(i + hd) % 3]}, ops)["0"]
            assert got.tolist() == _bf16(want[i, hd]), (i, hd)
            if i == n - 1:
                row = bind(X.Int8LutAttentionRow, D=d, S=p, SCALE=scale)
                assert list(evaluate(row, *(o.tolist() if o.size > 1 else int(o[0]) for o in ops))) == got.tolist(), (i, hd)
    assert {f: rows.ROWS[f].returns for f in (Q.KV_QUANT, Q.ATTENTION_ROW)} == {Q.KV_QUANT: ("0", "1", "2", "3"), Q.ATTENTION_ROW: ("0",)}
    assert {Q.KV_QUANT, Q.ATTENTION_ROW} <= set(rows.EVALUATORS)


@pytest.mark.parametrize("n,s,h,g_heads,spread,ks_scale", [(5, 5, 2, 1, 4.0, 1.0), (1, 40, 4, 2, 4.0, 1.0),
                                                           (3, 33, 2, 2, 8.0, 0.25), (2, 6, 2, 1, 1.0, 40.0)])
def test_int8_lut_attention_is_the_served_attention(n, s, h, g_heads, spread, ks_scale):
    """`Int8KvQuant` per KV head is `kv_quant`'s values, and `Int8LutAttentionRow` per (row, head) over its first pos + 1 rows is
    `window_attention`: prefill (n = s), chunk (1 < n < s) and decode rows, keys past one k step (s > 32), K clamped by a small
    scale (ks_scale < 1) and A saturated by a large one."""
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


def test_attention_row_binds_up_to_the_pv_window():
    """P V' steps over the keys zero-padded to 32: any S >= 1 binds while padded(S) <= 132,096 (LutPvWindow's 32·S, past the
    generic s8 cut of 2^17); past it, and a head dim off the 32-wide step, `bind` refuses by name."""
    from verity.ml.circuits.ops.gemm import S8_K_LIMIT

    sc = 128 ** -0.5
    for s in (1, 31, 33, S8_K_LIMIT - 31, S8_K_LIMIT, X.PV_KEY_LIMIT):
        assert X.row_refusal({"D": 128, "S": s}) is None
    for s, why in [(X.PV_KEY_LIMIT + 1, "past its window of 132096"), (0, "at least its own key")]:
        with pytest.raises(ValueError, match=why):
            bind(X.Int8LutAttentionRow, D=128, S=s, SCALE=sc)
    with pytest.raises(ValueError, match="multiple of 32"):
        bind(X.Int8LutAttentionRow, D=80, S=4, SCALE=sc)


@pytest.mark.parametrize("s", [1, 2, 33, 257])
def test_attention_rows_cut_under_q_word(s):
    """A prefill's first row (S = 1, its softmax over one key) cuts under Q_word like the longer ones: no gate recomputed."""
    from verity.ml.definitions import word as W

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


def test_the_registered_table_forms_are_v1_reading_exp2_lut127():
    """`Exp2Lut127_v2` reads LUT[t] from the registered `exp2_lut127` (`TableRead8x2_v1` at the byte zero-extended, the 16-bit index
    the Lean evaluator decodes for every read of at most 16 index bits): every byte gives v1's word.
    `Int8LutSoftmax_v2` and `Int8LutAttentionRow_v2` compute v1's words, and a Program of them holds no table."""
    from verity.ml.circuits import operations as O
    from verity.ml.circuits.program import Program
    from verity.ml.definitions.registry import tables as T

    table = X.exp2_lut127_words()
    assert len(table) == X.EXP2_LUT127_TABLE.leaves == 512 and T.table_read(8, 2).params[0][1].w == 16
    assert [evaluate(X.Exp2Lut127V2, t, table) for t in range(256)] == [(X.Exp2Lut127.evaluate(t),) for t in range(256)]
    g = torch.Generator().manual_seed(8)
    sc = [int(v) & 0xFFFFFFFF for v in torch.randint(-20000, 20000, (37,), generator=g).tolist()]
    assert evaluate(bind(X.Int8LutSoftmaxV2, S=37), sc, 3 << 28, table) == evaluate(bind(X.Int8LutSoftmax, S=37), sc, 3 << 28)
    d, s = 32, 33
    k, v, q = _rows(g, s, d), _rows(g, s, d), _rows(g, d)
    ks = (k.float().abs().amax() / 127).reshape(1)
    kq, vq, sv, v0 = IA.kv_quant(k[:, None], v[:, None], ks)
    args = (_bf16(q), _s8(kq[0]), _f32(ks)[0], _s8(vq[0]), _f32(sv[0]), _f32(v0[0]))
    v1, v2 = bind(X.Int8LutAttentionRow, D=d, S=s, SCALE=d ** -0.5), bind(X.Int8LutAttentionRowV2, D=d, S=s, SCALE=d ** -0.5)
    assert v2.id.startswith("Int8LutAttentionRow_v2{") and evaluate(v2, *args, table) == evaluate(v1, *args)
    held = O.program_data(Program(X.Int8LutSoftmaxV2.bind(S=2), registered=("exp2_lut127",)).root)
    assert X.Exp2Lut127.id not in held and all(t != "exp2_lut127" for v in held.values() for _, t in v)
    with pytest.raises(ValueError, match="past its window"):
        bind(X.Int8LutAttentionRowV2, D=d, S=X.PV_KEY_LIMIT + 1, SCALE=d ** -0.5)


def test_the_int8_attention_rule_declares_the_table_only_over_an_int8_layer(monkeypatch):
    """A `registered_values` derivation registers `exp2_lut127` when an attention call's layer is served by the int8 impl, and not
    for an FA2 layer under the same op."""
    from verity_vllm.program.frontend.rules.vllm_bindings import int8 as R

    rule, op = R.Int8UnifiedAttentionRule(), object()
    monkeypatch.setattr(rule, "targets", lambda: {op})
    ep = SimpleNamespace(graph=SimpleNamespace(nodes=[SimpleNamespace(op="call_function", target=op, args=(None,) * 4 + ("l0",), kwargs={})]))
    monkeypatch.setattr(rule, "impl_refusal", lambda ln: None)
    assert rule.registered_tables(ep, "profile") == ("exp2_lut127",)
    monkeypatch.setattr(rule, "impl_refusal", lambda ln: f"{ln} is FA2's")
    assert rule.registered_tables(ep, "profile") == ()


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
