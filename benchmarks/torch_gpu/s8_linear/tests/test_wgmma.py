"""wgmma.py's host side and sass_gate.py, no GPU: configs, each tile's shared memory against the H200's, the CTAs a launch
takes, `Params` against s8_wgmma.cu's layout, and the SASS gate on small listings."""
from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import sass_gate as SG  # noqa: E402
import wgmma as W  # noqa: E402

#: the H100's and H200's opt-in shared memory a block (227 KiB)
SM90_SMEM = 232448


def test_a_config_round_trips_and_refuses_an_unbuilt_tile():
    for text in ("128x256:c0:g8", "64x64:c32:g1", "128x128:c66:g16"):
        assert str(W.Cfg.parse(text)) == text
    assert W.Cfg.parse("64x128") == (64, 128, 0, 8)
    with pytest.raises(ValueError):
        W.Cfg(256, 128)
    with pytest.raises(ValueError):
        W.Cfg(128, 256, 0, 0)


def test_every_tile_fits_the_cards_shared_memory_and_its_swizzle():
    for tm, bn in W.TILES:
        assert 2 <= W.stages(tm, bn) <= W.MAX_STAGES and W.smem(tm, bn) <= SM90_SMEM
        assert (tm * W.BK) % 1024 == 0 and ((tm + bn) * W.BK) % 1024 == 0


def test_params_is_the_kernels_layout():
    src = (HERE / "s8_wgmma.cu").read_text()
    sa, m, size = map(int, re.search(r"offsetof\(Params, sa\) == (\d+) && offsetof\(Params, M\) == (\d+) && sizeof\(Params\) == (\d+)",
                                     src).groups())
    assert (W.Params.sa.offset, W.Params.M.offset, W.C.sizeof(W.Params)) == (sa, m, size)
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


def _gate(igmma: str = "IGMMA.64x64x32.S8.S8 R24, gdesc[UR4], R24 ;", extra: str = "") -> list[str]:
    funcs, problems = SG.gate(PTXAS, LISTING.format(igmma="\n".join([igmma] * 4), extra=extra))
    return [p for p in problems if "missing from the cubin" not in p]


def test_the_sass_gate_passes_the_s8_atom_and_integer_divisions_f2i():
    assert _gate() == []


@pytest.mark.parametrize("bad", ["FFMA R1, R2, R3, R4 ;", "FMUL.FTZ R1, R2, R3 ;", "LDL R1, [R2] ;",
                                 "HGMMA.64x64x16.F32.BF16 R24, gdesc[UR4], R24 ;"])
def test_the_sass_gate_refuses(bad):
    assert _gate(extra=bad)


def test_the_sass_gate_refuses_an_unsigned_or_other_shaped_atom():
    assert _gate(igmma="IGMMA.64x64x32.U8.S8 R24, gdesc[UR4], R24 ;")
    assert _gate(igmma="IGMMA.64x128x32.S8.S8 R24, gdesc[UR4], R24 ;")
