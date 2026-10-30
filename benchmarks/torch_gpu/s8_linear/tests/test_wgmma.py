"""wgmma.py's host side and sass_gate.py, no GPU: configs, each tile's shared memory against the H200's, the CTAs a launch
takes, `Params` against s8_wgmma.cu's layout, and the SASS gate on small listings."""
from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import sass_gate as SG  # noqa: E402
import wgmma as W  # noqa: E402

#: the H100's and H200's opt-in shared memory a CTA (227 KiB)
SM90_SMEM = 232448


def test_a_config_round_trips_and_refuses_an_unbuilt_tile():
    for text in ("128x256:c0:g8", "64x64:c32:g1", "128x128:c66:g16", "128x256:c0:g8:m2", "64x64:c32:g1:m2", "64x64:c0:g8:k0",
                 "128x64:c0:g1:k4"):
        assert str(W.Cfg.parse(text)) == text
    assert W.Cfg.parse("64x128") == (64, 128, 0, 8, 1, 1)
    assert str(W.Cfg.parse("64x128:m1:k1")) == "64x128:c0:g8"
    for bad in ((256, 128), (128, 256, 0, 0), (128, 256, 0, 8, 3), (128, 256, 1, 8, 2), (128, 256, 0, 8, 2, 4),
                (128, 256, 0, 8, 1, -1)):
        with pytest.raises(ValueError):
            W.Cfg(*bad)


def test_split_k_fills_the_sms_and_keeps_two_blocks_a_split():
    assert W.auto_splits(2, 128) == 64 and W.auto_splits(1, 128) == 64 and W.auto_splits(40, 128) == 3
    assert W.auto_splits(200, 128) == 1 and W.auto_splits(1, 3) == 1 and W.auto_splits(1, 1) == 1


def test_split_k_runs_only_where_the_tiles_leave_sms_idle():
    assert W.splits_for(132, 128, 3) == 1 and W.splits_for(40960, 128, 0) == 1
    assert W.splits_for(64, 128, 3) == 3 and W.splits_for(64, 128, 0) == 2 and W.splits_for(8, 4, 16) == 4


def test_every_tile_fits_the_cards_shared_memory_and_its_swizzle():
    for tm, bn in W.TILES:
        assert 2 <= W.stages(tm, bn) <= W.MAX_STAGES and W.smem(tm, bn) <= SM90_SMEM
        assert (tm * W.BK) % 1024 == 0 and ((tm + bn) * W.BK) % 1024 == 0


def test_params_is_the_kernels_layout():
    src = (HERE / "s8_wgmma.cu").read_text()
    sa, m, ws, size = map(int, re.search(r"offsetof\(Params, sa\) == (\d+) && offsetof\(Params, M\) == (\d+) && "
                                         r"offsetof\(Params, ws\) == (\d+) && sizeof\(Params\) == (\d+)", src).groups())
    assert (W.Params.sa.offset, W.Params.M.offset, W.Params.ws.offset, W.C.sizeof(W.Params)) == (sa, m, ws, size)
    assert W.Params.clk.offset == int(re.search(r"offsetof\(Params, clk\) == (\d+)", src).group(1))
    assert re.findall(r"^S8MM\((\d+), (\d+)\)", src, re.M) == [(str(tm), str(bn)) for tm, bn in W.TILES]
    assert int(re.search(r"constexpr int BK = (\d+)", src).group(1)) == W.BK
    assert int(re.search(r"SMEM_BUDGET = (\d+) \* 1024", src).group(1)) * 1024 == W.SMEM_BUDGET
    assert int(re.search(r"MAX_STAGES = (\d+)", src).group(1)) == W.MAX_STAGES


def test_the_default_config():
    assert W.default_cfg(1, 16384, 16384) == W.Cfg(64, 128)
    assert W.default_cfg(64, 128, 16384) == W.Cfg(64, 128)
    assert W.default_cfg(8192, 65536, 16384) == W.Cfg(128, 256)
    assert W.default_cfg(256, 64, 16384) == W.Cfg(128, 64)


def test_the_tuned_table_holds_built_tiles_in_ascending_m():
    from configs import CONFIGS
    for rows in CONFIGS["wgmma:sm_90"].values():
        assert [m for m, _ in rows] == sorted({m for m, _ in rows})
        for _, text in rows:
            assert str(W.Cfg.parse(text)) == text


def test_the_grid_keeps_the_rounds_and_holds_at_most_its_ctas():
    lib = types.SimpleNamespace(sms=132)
    grid = lambda tiles, ctas: W.Lib.grid(lib, tiles, ctas)  # noqa: E731
    assert grid(128, 0) == 128 and grid(264, 0) == 132 and grid(200, 0) == 100 and grid(1, 0) == 1
    assert grid(40960, 32) == 32 and grid(40960, 500) == 132
    pairs = lambda units, ctas: W.Lib.grid(lib, units, ctas, 66)  # noqa: E731
    assert pairs(64, 0) == 128 and pairs(200, 0) == 100 and pairs(200, 32) == 32 and pairs(1, 0) == 2
    assert pairs(40960, 500) == 132 and pairs(40960, 0) == 132


LISTING = """
        Function : s8mm_64x64
        /*0010*/                   UTMALDG.2D [UR8], [UR4] ;
{igmma}
        /*0030*/                   F2I.FTZ.U32.TRUNC.NTZ R9, R8 ;
        /*0040*/                   I2FP.F32.S32 R21, R25 ;
        /*0050*/                   FMUL R60, R21, R12 ;
        /*0060*/                   F2FP.BF16.F32.PACK_AB R61, R60, R61 ;
{extra}
"""
PTXAS = "ptxas info    : Compiling entry function 's8mm_64x64' for 'sm_90a'\nptxas info    : Used 66 registers, 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"


def _gate(igmma: str = "IGMMA.64x64x32.S8.S8 R24, gdesc[UR4], R24 ;", extra: str = "", name: str = "s8mm_64x64") -> list[str]:
    listing = LISTING.format(igmma="\n".join([igmma] * 4), extra=extra).replace("s8mm_64x64", name)
    funcs, problems = SG.gate(PTXAS.replace("s8mm_64x64", name), listing)
    return [p for p in problems if "missing from the cubin" not in p]


def test_the_sass_gate_passes_the_s8_atom_and_integer_divisions_f2i():
    assert _gate() == [] and _gate(name="s8mm2_64x64") == []
    assert _gate(name="s8mm2_64x64", igmma="IGMMA.64x128x32.S8.S8 R24, gdesc[UR4], R24 ;")


def test_the_sass_gate_wants_every_kernel_and_its_pair():
    _, problems = SG.gate(PTXAS, LISTING.format(igmma="IGMMA.64x64x32.S8.S8 R24, gdesc[UR4], R24 ;", extra=""))
    missing = next(p for p in problems if "missing from the cubin" in p)
    assert "'s8mm2_128x256'" in missing and "'s8mm_128x256'" in missing and "'s8mm_64x64'" not in missing


@pytest.mark.parametrize("bad", ["FFMA R1, R2, R3, R4 ;", "FMUL.FTZ R1, R2, R3 ;", "LDL R1, [R2] ;",
                                 "HGMMA.64x64x16.F32.BF16 R24, gdesc[UR4], R24 ;"])
def test_the_sass_gate_refuses(bad):
    assert _gate(extra=bad)


def test_the_sass_gate_refuses_an_unsigned_or_other_shaped_atom():
    assert _gate(igmma="IGMMA.64x64x32.U8.S8 R24, gdesc[UR4], R24 ;")
    assert _gate(igmma="IGMMA.64x128x32.S8.S8 R24, gdesc[UR4], R24 ;")


def test_the_kernel_order_is_a_permutation_within_each_32_bytes():
    assert sorted(W.KERNEL_ORDER) == list(range(32))
    assert W.KERNEL_ORDER[:8] == (0, 1, 2, 3, 8, 9, 10, 11) and W.KERNEL_ORDER[16:20] == (4, 5, 6, 7)


def _fragments(o: np.ndarray) -> np.ndarray:
    """A 64 x BN tile of s32 accumulators as the reduction's A [64, 4 BN] s8 in the kernel's order: thread (warp w, lane)
    holds accumulator 4 j + i at row 16 w + lane / 4 + 8 (i / 2), column 8 j + 2 (lane % 4) + i % 2, and passes
    (4 j, 4 j + 2, 4 j + 1, 4 j + 3) as the fragment's registers: rows g, g + 8, g, g + 8 at bytes 4 t, 4 t, 16 + 4 t, 16 + 4 t."""
    bn = o.shape[1]
    a = np.zeros((64, 4 * bn), np.int8)
    for w in range(4):
        for lane in range(32):
            g, t = lane // 4, lane % 4
            for j in range(bn // 8):
                acc = [o[16 * w + g + 8 * (i // 2), 8 * j + 2 * t + i % 2] for i in range(4)]
                for reg, row, k0 in zip((acc[0], acc[2], acc[1], acc[3]), (g, g + 8, g, g + 8), (4 * t, 4 * t, 16 + 4 * t, 16 + 4 * t)):
                    a[16 * w + row, 32 * j + k0:32 * j + k0 + 4] = np.array([reg], "<i4").view(np.int8)
    return a


@pytest.mark.parametrize("bn,r", [(128, 8), (256, 16)])
def test_the_registers_times_r_in_kernel_order_are_os_bytes_times_r(bn, r):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(bn + r)
    o = rng.integers(-2**31, 2**31, (64, bn), dtype=np.int64).astype(np.int32)
    o[0], o[1, :8] = -2**31, 2**31 - 1
    rmat = rng.integers(-128, 128, (r, 4 * bn), dtype=np.int64).astype(np.int8)
    rk = W.kernel_order(torch.from_numpy(rmat)).numpy()
    want = o.view(np.int8).astype(np.int64) @ rmat.astype(np.int64).T
    assert (_fragments(o).astype(np.int64) @ rk.astype(np.int64).T == want).all()
    assert not (o.view(np.int8).astype(np.int64) @ rk.astype(np.int64).T == want).all()


def test_rparams_and_the_reduction_kernels_are_the_kernels():
    src = (HERE / "s8_wgmma.cu").read_text()
    r, z, size = map(int, re.search(r"offsetof\(RParams, r\) == (\d+) && offsetof\(RParams, z\) == (\d+) && "
                                    r"sizeof\(RParams\) == (\d+)", src).groups())
    assert (W.RParams.r.offset, W.RParams.z.offset, W.C.sizeof(W.RParams)) == (r, z, size)
    built = {f"{prefix}_128x{bn}_r{rr}" for bn, rr in re.findall(r"^S8MR\((\d+), (\d+)\)", src, re.M) for prefix in W.FLUSHES.values()}
    built |= {f"{name}_128x{bn}_r{rr}" for bn, rr, name in re.findall(r"^S8MR1\((\d+), (\d+), \d, (\w+)\)", src, re.M)}
    cfg = {bn: W.Cfg.parse(f"128x{bn}") for _, bn in W.FUSED_TILES}
    assert built == {f"{prefix}_128x{bn}_r{rr}" for _, bn in W.FUSED_TILES for rr in W.RS for flush, prefix in W.FLUSHES.items()
                     if W.has_reduction(cfg[bn], rr, flush)}
    assert SG.UNBUILT == {f"{W.FLUSHES[flush]}_{tm}x{bn}_r{rr}" for tm, bn, rr, flush in W.UNBUILT}
    assert all(tm == 128 for tm, _ in W.FUSED_TILES)


def test_a_tiles_reduction_fits_the_ring_beside_a_stage_of_the_next():
    for tm, bn in W.FUSED_TILES:
        for r in W.RS:
            q, rst = W.red_geometry(tm, bn, r)
            assert q * r * 128 <= (tm + bn) * W.BK and q * rst >= bn // 32 and rst < W.stages(tm, bn)
            assert (1 if rst >= tm // 64 else tm // 64) * 64 * r * 4 <= (tm + bn) * W.BK
    assert W.red_geometry(128, 256, 128) == (3, 3) and W.red_geometry(128, 256, 32) == (8, 1)
    assert W.red_geometry(128, 128, 128) == (2, 2) and W.red_geometry(128, 128, 64) == (4, 1)


def test_the_sm_clock_is_each_ctas_cycles_over_its_nanoseconds_and_skips_ctas_that_wrote_none():
    import torch
    clk = torch.zeros(4 * 4, dtype=torch.int64)
    assert W.sm_mhz(clk) is None
    clk.view(-1, 4)[0] = torch.tensor([10, 5_000, 10 + 1_830_000, 5_000 + 1_000_000])
    clk.view(-1, 4)[1] = torch.tensor([7, 9_000, 7 + 1_755_000, 9_000 + 1_000_000])
    clk.view(-1, 4)[2] = torch.tensor([7, 9_000, 7 + 1_800_000, 9_000 + 1_000_000])
    assert W.sm_mhz(clk) == pytest.approx(1800.0)


def test_the_sass_gate_passes_a_reduction_kernels_two_atoms_and_refuses_a_third():
    name = "s8mr_128x256_r16"
    main = "IGMMA.64x256x32.S8.S8 R24, gdesc[UR4], R24 ;"
    red = "\n".join(["IGMMA.64x16x32.S8.S8 R200, R24, gdesc[UR8], R200 ;"] * 32)
    assert _gate(igmma=main, extra=red, name=name) == []
    assert _gate(igmma=main, extra=red.replace("64x16x32", "64x32x32"), name=name)
    assert _gate(igmma=main, extra=red.replace("S8.S8 R200", "U8.S8 R200"), name=name)
    listing = LISTING.format(igmma="\n".join([main] * 4), extra=red).replace("s8mm_64x64", name)
    funcs, _ = SG.gate(PTXAS.replace("s8mm_64x64", name), listing)
    assert funcs[name]["max_reg"] == 201 and funcs[name]["IGMMA"] == 36
