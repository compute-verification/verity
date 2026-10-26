"""theory/public_best.py (D5, CMP-125): every scheme it prices computes v1's words exactly, the honest kernel prices at
1, and the SHA-512 schedules reproduce RT-UNITS' counts."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "theory"))

import public_best as PB  # noqa: E402


@pytest.fixture(scope="module")
def cat():
    return PB.catalogue()


def test_catalogue_keeps_the_integral_schemes_in_every_orientation(cat):
    assert {k: len(v[0]) for k, v in cat.items()} == {"2x3x3": 15, "2x3x4": 20, "2x4x4": 26, "2x5x5": 40, "3x3x3": 23}
    assert all(len(v) == 6 and all(PB.is_valid(s) for s in v) for v in cat.values())
    assert PB.lrp_scheme("4x4x4") is None and PB.lrp_scheme("2x4x5") is None


@pytest.mark.parametrize("name,depth,data,uX", [("strassen", 2, "zero", 64), ("winograd", 1, "real", 80),
                                                 ("3x3x3", 1, "zero", 80), ("2x3x3", 1, "real", 64)])
def test_words_are_exact(cat, name, depth, data, uX):
    scheme = {"strassen": PB.STRASSEN, "winograd": PB.WINOGRAD}.get(name) or cat[name][0]
    ok, cnt = PB.check_words(scheme, depth, data, uX=uX)
    assert ok and cnt.sp > 0


def test_assembly_counts_iadd3_with_algo_costs_fork_floor():
    assert PB._iadd3(PB.STRASSEN) == 6 and PB.assembly(PB.STRASSEN, 1) == 1.0
    assert PB.assembly(PB.STRASSEN, 2) == pytest.approx(1 + 7 / 4)


def test_v1_permutation_breaks_the_packing_pattern():
    without, with_perm = PB.check_packing_invariance()
    assert without > 0.9 and with_perm < 0.3


@pytest.mark.parametrize("L,C", [(512, 128), (2048, 512)])
def test_without_v_the_complement_split_and_the_packed_correction_are_exact(L, C):
    r = PB.check_no_v(L=L, m=8, n=32, C=C)
    assert r["m7_exact"] and r["y_2to4"] and r["g_planes_int8"] and r["packed_exact"] and r["e2_int8"]
    assert (r["unpacked_chunks"] > 0.5) == (C == 512)


def test_v_blocks_m7_on_sm120():
    eX, eW, _ = PB.operand_cases()["A=B=0, no V (U ±80; is V needed?)"]
    p = PB.price(PB.SM120, PB.leaves_of(None, 0, eX, eW), 640, 16512, PB.TOKENS, beside=False)
    assert p.c == pytest.approx(0.75, abs=0.01) and p.routes == {"2:4 bin A": pytest.approx(1.0)}
    eX, eW, _ = PB.operand_cases()["A=B=0, v1 (V ±64, U ±80)"]
    v1 = PB.price(PB.SM120, PB.leaves_of(None, 0, eX, eW), 640, 16512, PB.TOKENS, beside=False)
    assert v1.c == pytest.approx(1.0)


@pytest.mark.parametrize("card", PB.CARDS, ids=lambda c: c.name)
def test_the_honest_kernel_prices_at_one(card):
    eX, eW, _ = PB.operand_cases()["A=B=0, v1 (V ±64, U ±80)"]
    p = PB.price(card, PB.leaves_of(None, 0, eX, eW), 1280, 16384, PB.TOKENS, beside=False)
    assert p.c == pytest.approx(1.0) and "tensor" in p.binding


def test_strassen_depth_one_is_tensor_bound_on_sm120_at_zero_data():
    eX, eW, _ = PB.operand_cases()["A=B=0, v1 (V ±64, U ±80)"]
    p = PB.price_scheme(PB.SM120, PB.STRASSEN, 1, eX, eW, 640, 16512, PB.TOKENS, beside=False)
    assert p.c == pytest.approx(7 / 8) and "tensor" in p.binding


def test_sha_schedules_match_rt_units():
    t0, r0 = PB.sha_clocks(PB.H200, "default build")
    t1, _ = PB.sha_clocks(PB.H200, "A-wide + R-hi")
    assert r0 == "alu" and t0 / t1 == pytest.approx(1.38, abs=0.01)
    measured = 132 * 1980e6 / 4.79e9
    assert t0 == pytest.approx(measured, rel=0.02)
    t2, r2 = PB.sha_clocks(PB.SM120, "R-shf")
    assert r2 == "rf" and PB.sha_clocks(PB.SM120, "default build")[0] / t2 == pytest.approx(1.23, abs=0.01)
