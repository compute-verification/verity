"""`engine.int8_wgmma`, `int8_fused.product`'s and `product_zy`'s sm_90a plan: its configs and plan rows on the CPU, its
PTX's rounding where nvcc is, and on an sm_90 GPU with nvcc every variant bit for bit against `exactlin_scaled_mm_tma` and
an exact int32 product with exactlin's epilogue, at shapes that are no multiple of any tile, at -128 and 127, into a
poisoned y, and replayed from a CUDA graph; and every zy variant's y and z/y records byte for byte against
`exactlin_scaled_mm_zy_tma`'s and `int8_exact.zy_records` of that product, over slots past M and row tiles past M."""
from __future__ import annotations

import subprocess

import pytest
import torch

from verity_vllm.engine import int8_exact as E
from verity_vllm.engine import int8_wgmma as W
from verity_vllm.engine.profiles import int8_gemm_configs as C


def _sm90_nvcc() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (9, 0) and W.nvcc() is not None


sm90 = pytest.mark.skipif(not _sm90_nvcc(), reason="an sm_90 GPU and nvcc required")


def test_configs_parse_and_print():
    assert W.WgCfg.parse("ppm2_128x128") == ("ppm2_128x128", 8, 0)
    assert W.WgCfg.parse("co_128x256:g16:c66") == ("co_128x256", 16, 66)
    for text in ("com2_128x256:g8", "pp_128x128:g4:c64"):
        assert str(W.WgCfg.parse(text)) == text
    for bad in ("co_128x128", "pp_128x128:g0", "pp_128x128:c-1"):
        with pytest.raises(ValueError):
            W.WgCfg.parse(bad)


def test_every_plan_row_is_a_config_and_rows_rise():
    for sm, shapes in C.WGMMA.items():
        assert sm in C.CONFIGS, sm
        for (n, k), rows in shapes.items():
            assert n % 2 == 0 and k % 16 == 0, (sm, n, k)
            assert [m for m, _ in rows] == sorted({m for m, _ in rows}), (sm, n, k)
            for _, text in rows:
                W.WgCfg.parse(text)


def test_the_row_is_the_last_whose_m_min_is_reached(monkeypatch):
    monkeypatch.setattr(W, "WGMMA", {"sm_90": {(256, 64): [(1024, "pp_128x128:g4"), (8192, "com2_128x256:g8")]}})
    assert W.row(1023, 256, 64, "sm_90") is None
    assert W.row(1024, 256, 64, "sm_90") == ("pp_128x128", 4, 0)
    assert W.row(8191, 256, 64, "sm_90") == ("pp_128x128", 4, 0)
    assert W.row(1 << 20, 256, 64, "sm_90") == ("com2_128x256", 8, 0)
    assert W.row(8192, 256, 128, "sm_90") is None
    assert W.row(8192, 256, 64, "sm_120") is None


@pytest.mark.skipif(W.nvcc() is None, reason="nvcc required")
def test_the_ptx_rounds_as_exactlin(tmp_path):
    """The PTX nvcc makes of int8_wgmma.cu with the library's float flags: no FMA contraction and no FTZ, and the epilogue's
    cvt.rn.f32.s32, two mul.rn.f32 and cvt.rn.bf16x2.f32 over wgmma's s8 product (as `int8_fused.ptx_check` holds the
    Triton kernels)."""
    assert "-fmad=false" in W.NVCC_FLAGS and not {"--use_fast_math", "-ftz=true"} & set(W.NVCC_FLAGS)
    out = tmp_path / "int8_wgmma.ptx"
    subprocess.run([W.nvcc(), "-ptx", "-O3", "-std=c++17", f"-arch={C.WGMMA_ARCH}", "-fmad=false", "-o", str(out),
                    str(W.HERE / "int8_wgmma.cu")], check=True)
    ptx = out.read_text()
    assert ptx.count(".ftz") == 0
    assert sum(ptx.count(f) for f in ("fma.rn.f32", "mad.f32", "mad.rn.f32", "fma.rn.ftz.f32")) == 0
    assert sum(ptx.count(f) for f in ("div.rn.f32", "div.approx", "div.full")) == 0
    assert ptx.count("cvt.rn.f32.s32") >= 1 and ptx.count("mul.rn.f32") >= 2 and ptx.count("cvt.rn.bf16x2.f32") >= 1
    assert ptx.count(".s32.s8.s8") >= 1


def _operands(m, n, k, seed, extremes=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    xq = torch.randint(-128, 128, (m, k), dtype=torch.int8, device="cuda", generator=g)
    w = torch.randint(-128, 128, (n, k), dtype=torch.int8, device="cuda", generator=g)
    if extremes:
        for t, rows in ((xq, m), (w, n)):
            r = torch.arange(rows, device="cuda")
            t[r % 5 == 0] = -128
            t[r % 5 == 2] = 127
    sa = torch.exp(torch.empty(m, device="cuda").uniform_(-9, 2, generator=g)).float()
    sw = torch.exp(torch.empty(n, device="cuda").uniform_(-9, 2, generator=g)).float()
    return xq, sa, w, sw


def _reference(xq, sa, w, sw) -> torch.Tensor:
    z = (xq.double() @ w.double().T).to(torch.int32)
    return ((z.float() * sa[:, None]) * sw[None, :]).to(torch.bfloat16).view(torch.int16)


def _poisoned(m, n) -> torch.Tensor:
    return torch.full((m, n), 0xA5A5 - 0x10000, dtype=torch.int16, device="cuda").view(torch.bfloat16)


@sm90
@pytest.mark.parametrize("variant", list(W.VARIANTS))
@pytest.mark.parametrize("m,n,k,extremes", [(1, 2, 16, False), (129, 136, 160, False), (300, 520, 2064, True),
                                            (1000, 3000, 2048, False), (513, 264, 8192, True)])
def test_every_variant_is_exactlin_bit_for_bit(variant, m, n, k, extremes):
    from verity_vllm.engine import int8_fused as X

    xq, sa, w, sw = _operands(m, n, k, seed=m + n + k, extremes=extremes)
    ref = _reference(xq, sa, w, sw)
    yt = _poisoned(m, n)
    X.scaled_mm(xq, sa, w, sw, yt, X.Cfg.parse("128x128x128:w4:s3:k1:t1"))
    assert torch.equal(yt.view(torch.int16), ref)
    for cfg in (f"{variant}:g8", f"{variant}:g1:c8"):
        y = _poisoned(m, n)
        W.scaled_mm(xq, sa, w, sw, y, W.WgCfg.parse(cfg))
        assert torch.equal(y.view(torch.int16), ref), cfg


@sm90
def test_a_graph_replays_it_on_new_operands():
    m, n, k = 384, 512, 1024
    xq, sa, w, sw = _operands(m, n, k, seed=1)
    y = _poisoned(m, n)
    cfg = W.WgCfg.parse("com2_128x256:g8")
    W.scaled_mm(xq, sa, w, sw, y, cfg)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        W.scaled_mm(xq, sa, w, sw, y, cfg)
    x2, s2, w2, sw2 = _operands(m, n, k, seed=2)
    for dst, src in ((xq, x2), (sa, s2), (w, w2), (sw, sw2)):
        dst.copy_(src)
    y.view(torch.int16).fill_(0x5A5A)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(y.view(torch.int16), _reference(x2, s2, w2, sw2))


@sm90
def test_product_takes_the_row_and_keeps_the_values(monkeypatch):
    from verity_vllm.engine import int8_fused as X

    m, n, k = 2048, 768, 2048
    xq, sa, w, sw = _operands(m, n, k, seed=3)
    want = X.product(xq, sa, w, sw).view(torch.int16)
    calls = []
    real = W.scaled_mm
    monkeypatch.setattr(W, "scaled_mm", lambda *a: calls.append(a[-1]) or real(*a))
    monkeypatch.setattr(W, "WGMMA", {"sm_90": {(n, k): [(1024, "co_128x256:g8")]}})
    got = X.product(xq, sa, w, sw).view(torch.int16)
    assert calls == [("co_128x256", 8, 0)]
    assert torch.equal(got, want) and torch.equal(got, _reference(xq, sa, w, sw))


def _zy_reference(xq, sa, w, sw, rows) -> tuple[torch.Tensor, torch.Tensor]:
    """(y bits [M, N], records) of the exact product over `rows` slots, zero codes under scale 0 past M."""
    m = xq.shape[0]
    z = torch.zeros(rows, w.shape[0], dtype=torch.int32, device="cuda")
    z[:m] = (xq.double() @ w.double().T).to(torch.int32)
    scales = torch.zeros(rows, dtype=torch.float32, device="cuda")
    scales[:m] = sa
    y = E.dequantize(z, scales, sw)
    return y[:m].view(torch.int16), E.zy_records(z, y)


def _poisoned_zy(m, n, rows) -> torch.Tensor:
    return torch.full((E.zy_units(m, n, rows), E.ZY_BYTES), 0xA5, dtype=torch.uint8, device="cuda")


@sm90
@pytest.mark.parametrize("variant", list(W.VARIANTS))
@pytest.mark.parametrize("m,n,k,rows,extremes", [(1, 32, 16, 32, False), (129, 160, 160, 320, False),
                                                 (300, 512, 2064, 512, True), (1000, 3008, 2048, 1024, False),
                                                 (513, 288, 8192, 704, True)])
def test_every_zy_variant_writes_exactlins_records_byte_for_byte(variant, m, n, k, rows, extremes):
    """Slots past M in a multiplying tile and in row tiles (or pairs) that multiply nothing, negative s_w (whose empty
    slots' y is -0), raster groups of 8 and of 1 on 8 CTAs, twice into the same poisoned buffers."""
    from verity_vllm.engine import int8_fused as X

    xq, sa, w, sw = _operands(m, n, k, seed=m + n + k + rows, extremes=extremes)
    if extremes:
        sw[::3] = -sw[::3]
    ref_y, ref_zy = _zy_reference(xq, sa, w, sw, rows)
    yt, zt = _poisoned(m, n), _poisoned_zy(m, n, rows)
    X.scaled_mm_zy(xq, sa, w, sw, yt, zt, rows, X.Cfg.parse("128x128x128:w4:s3:k1:t1"))
    assert torch.equal(yt.view(torch.int16), ref_y) and torch.equal(zt, ref_zy)
    for cfg in (f"{variant}:g8", f"{variant}:g1:c8"):
        y, zy = _poisoned(m, n), _poisoned_zy(m, n, rows)
        for _ in range(2):
            W.scaled_mm_zy(xq, sa, w, sw, y, zy, rows, W.WgCfg.parse(cfg))
            assert torch.equal(y.view(torch.int16), ref_y), cfg
            assert torch.equal(zy, ref_zy), (cfg, int((zy != ref_zy).any(dim=1).sum()))


@sm90
def test_a_graph_replays_the_zy_kernel_on_new_operands():
    m, n, k, rows = 384, 512, 1024, 448
    xq, sa, w, sw = _operands(m, n, k, seed=4)
    y, zy = _poisoned(m, n), _poisoned_zy(m, n, rows)
    cfg = W.WgCfg.parse("ppm2_128x128:g8")
    W.scaled_mm_zy(xq, sa, w, sw, y, zy, rows, cfg)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        W.scaled_mm_zy(xq, sa, w, sw, y, zy, rows, cfg)
    x2, s2, w2, sw2 = _operands(m, n, k, seed=5)
    for dst, src in ((xq, x2), (sa, s2), (w, w2), (sw, sw2)):
        dst.copy_(src)
    y.view(torch.int16).fill_(0x5A5A)
    zy.fill_(0x5A)
    g.replay()
    torch.cuda.synchronize()
    ref_y, ref_zy = _zy_reference(x2, s2, w2, sw2, rows)
    assert torch.equal(y.view(torch.int16), ref_y) and torch.equal(zy, ref_zy)


@sm90
def test_product_zy_takes_the_row_and_keeps_the_values(monkeypatch):
    from verity_vllm.engine import int8_fused as X

    m, n, k, rows = 2048, 768, 2048, 2112
    xq, sa, w, sw = _operands(m, n, k, seed=6)
    want_y, want_zy = X.product_zy(xq, sa, w, sw, rows)
    calls = []
    real = W.scaled_mm_zy
    monkeypatch.setattr(W, "scaled_mm_zy", lambda *a: calls.append(a[-1]) or real(*a))
    monkeypatch.setattr(W, "WGMMA", {"sm_90": {(n, k): [(1024, "co_128x256:g8")]}})
    got_y, got_zy = X.product_zy(xq, sa, w, sw, rows)
    assert calls == [("co_128x256", 8, 0)]
    ref_y, ref_zy = _zy_reference(xq, sa, w, sw, rows)
    assert torch.equal(got_y.view(torch.int16), want_y.view(torch.int16)) and torch.equal(got_zy, want_zy)
    assert torch.equal(got_y.view(torch.int16), ref_y) and torch.equal(got_zy, ref_zy)
