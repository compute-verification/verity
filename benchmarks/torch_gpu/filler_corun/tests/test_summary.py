"""summary.py's rates on synthetic stamps: the clock from clock64 over %globaltimer, MACs an SM-clock, θ, the hash against
the bar, placement, coverage and the gap to the target."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import summary as S  # noqa: E402

MACS_PER_TILE = 128 * 256 * 8192


def gemm_stamps(tiles: int, mhz: int, ns: int, sms: int = S.SMS) -> list[S.Stamp]:
    return [S.Stamp(t0=1000, t1=1000 + ns, c0=0, c1=ns * mhz // 1000, sm=i, n=tiles) for i in range(sms)]


def test_the_bar_an_sm_clock():
    assert S.BAR_PER_CLOCK == pytest.approx(0.01803, abs=5e-6)
    assert S.THETA_MACS_PER_S == pytest.approx(4096 * 132 * 1830e6, rel=1e-4)


def test_gemm_at_the_peak():
    mhz, ns = 1830, 2_000_000_000
    cycles = ns * mhz // 1000
    tiles = cycles * 4096 // MACS_PER_TILE
    r = S.gemm_rate(gemm_stamps(tiles, mhz, ns), MACS_PER_TILE)
    assert r["mhz"] == pytest.approx(1830)
    assert r["peak_share"] == pytest.approx(1.0, abs=1e-4)
    assert r["theta"] == pytest.approx(1.0, abs=1e-3)


def test_gemm_slower_clock_keeps_its_share_but_not_theta():
    mhz, ns = 1650, 2_000_000_000
    tiles = (ns * mhz // 1000) * 4096 // MACS_PER_TILE
    r = S.gemm_rate(gemm_stamps(tiles, mhz, ns), MACS_PER_TILE)
    assert r["peak_share"] == pytest.approx(1.0, abs=1e-4)
    assert r["theta"] == pytest.approx(1650 / 1830, abs=1e-3)


def test_hash_counts_each_sm_once_whatever_its_ctas():
    cycles = 1_000_000
    one = [S.Stamp(0, 1000, 0, cycles, sm=i, n=0, work=18_030) for i in range(S.SMS)]
    two = [S.Stamp(0, 1000, 0, cycles, sm=i // 2, n=0, work=9_015) for i in range(2 * S.SMS)]
    for st in (one, two):
        r = S.hash_rate(st)
        assert r["per_clock"] == pytest.approx(0.01803)
        assert r["bar"] == pytest.approx(0.01803 / S.BAR_PER_CLOCK)


def test_placement_names_an_empty_sm():
    st = [S.Stamp(0, 1, 0, 1, sm=i, n=0) for i in range(S.SMS - 1)] + [S.Stamp(0, 1, 0, 1, sm=0, n=0)]
    assert S.placement(st) == (0, 2)
    assert S.placement([S.Stamp(0, 1, 0, 1, sm=i, n=0) for i in range(S.SMS)]) == (1, 1)


def test_covers_allows_the_hash_its_last_message():
    g = [S.Stamp(100, 10_000_000, 0, 1, sm=0, n=0)]
    assert S.covers(g, [S.Stamp(200, 10_000_500, 0, 1, sm=0, n=0)])
    assert not S.covers(g, [S.Stamp(50, 9_000_000, 0, 1, sm=0, n=0)])
    assert not S.covers(g, [S.Stamp(200, 12_000_000, 0, 1, sm=0, n=0)])


def test_gemm_rate_in_reads_only_stamps_inside_the_window():
    # every CTA logs every 4 tiles, a tile each 1 us; the window holds stamps 10..20 us
    logs = [[1000 * 4 * j for j in range(1, 40)] for _ in range(S.SMS)]
    r = S.gemm_rate_in(logs, 4, MACS_PER_TILE, 40_000, 80_000, 1830)
    assert r["macs_per_s"] == pytest.approx(S.SMS * MACS_PER_TILE / 1e-6)
    assert r["macs_per_clock"] == pytest.approx(MACS_PER_TILE / 1e-6 / 1830e6)


def test_gap():
    assert S.gap(0.99, 0.43)["met"]
    g = S.gap(0.95, 0.40)
    assert not g["met"]
    assert g["peak_short"] == pytest.approx(0.035)
    assert g["bar_short"] == pytest.approx(0.014)
    assert S.pj_per_mac(700.0, 700 * 10**12) == pytest.approx(1.0)
