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


def test_v2_tile_route_holds_at_two_tiles_and_breaks_four_on_sm120():
    for card in (PB.H200, PB.SM120):
        bit = PB.BIT_COST[card.family][0][1]
        assert PB.tile_route(card, 2, 512, 16384, 16512, bit, beside=True)["c"] > 1
    assert PB.tile_route(PB.SM120, 4, 512, 16384, 16512, 0.383, beside=True)["c"] < 0.9


def test_fixed_tensor_load_counts_in_the_joint_facets():
    _, bit, depth = PB.BIT_COST["sm_90"][0]
    p = PB.tile_job(PB.H200, 2, 1024, 16384, 16512, bit, depth, 80, 0.3)["route"]
    assert max(p.loads.values()) <= p.c * (1 + 1e-6)


def test_named_public_programs_for_worksound_inst():
    out = PB.pub_programs()
    assert out["pub_mm.H200.v2"]["job"].c == pytest.approx(1.0)
    rtx = out["pub_mm.RTXPRO6000.v2"]["standalone"]
    assert rtx.c == pytest.approx(0.863, abs=0.002) and set(rtx.mix) == {"naive", "2x2 orbit d1"}
    assert out["pub_sha.H200", "A-wide + R-hi"]["rate"] == pytest.approx(6.69e9, rel=0.01)
    assert out["pub_sha.RTXPRO6000", "R-hi, IMAD.HI at 32 (unmeasured)"]["rate"] == pytest.approx(10.22e9, rel=0.01)


def test_v2_tile_route_on_h200_wins_through_the_joint_facet_when_peels_pass_three_eighths():
    _, bit, depth = PB.BIT_COST["sm_90"][0]
    big = PB.tile_job_at(PB.H200, 2, 1024, 4096, 6144, bit, depth, 80, 1.15)
    small = PB.tile_job_at(PB.H200, 2, 384, 4096, 6144, bit, depth, 80, 1.10)
    assert "joint0" in big["honest"].binding and big["c"] < 0.9
    assert small["c_route"] >= 1 and small["c"] == pytest.approx(1.0)


def test_v2_tile_route_in_the_job_beats_four_tiles_on_sm120_and_mixed_with_dense_tiles_reaches_further():
    _, bit, depth = PB.BIT_COST["sm_120"][0]
    four = PB.tile_job_at(PB.SM120, 4, 256, 16384, 16512, bit, depth, 80, 1.0)
    past = PB.tile_job_at(PB.SM120, 4, 256, 16384, 16512, bit, depth, 80, 1.1)
    three = PB.tile_job_at(PB.SM120, 3, 256, 16384, 16512, bit, depth, 80, 1.0)
    assert four["alu"] == pytest.approx(1.0, abs=1e-3) and four["c"] < four["c_route"] < 0.98
    assert four["c_opt"] < 0.98
    assert past["c_route"] >= 1 and past["c_opt"] < 1 and 0 < past["route"].mix["tiles"] < 1
    assert three["c_route"] >= 1 and three["route"].mix.get("tiles", 0.0) < 1e-6
    assert three["c_opt"] == pytest.approx(1.0) and three["eps_alg"] > 0.005


def test_the_hull_of_methods_beats_each_method_where_they_bind_different_resources():
    """ALGO-COST A19: a method a tile, so the public best is the methods' convex hull. On the H200 with tensor routes
    only, the plain product binds the tensor pipe and depth-1 Strassen the shared-memory port; the mix beats both."""
    eX, eW, _ = PB.operand_cases()["A=B=0, v1 (V ±64, U ±80)"]
    chosen = {"naive": (None, 0), "strassen d1": (PB.STRASSEN, 1)}
    singles = [PB.price_scheme(PB.H200, s, d, eX, eW, 1280, 16512, PB.TOKENS, beside=False).c
               for s, d in chosen.values()]
    h = PB.price_hull(PB.H200, chosen, eX, eW, 1280, 16512, PB.TOKENS, beside=False)
    assert min(singles) == pytest.approx(1.0) and h.c < 0.97
    assert set(h.mix) == set(chosen) and sum(h.mix.values()) == pytest.approx(1.0)
    one = PB.price_methods(PB.H200, [PB.scheme_method("naive", None, 0, eX, eW)], 1280, 16512, PB.TOKENS, beside=False)
    assert one.c == pytest.approx(1.0)


def test_sha_schedules_match_rt_units():
    t0, r0 = PB.sha_clocks(PB.H200, "default build")
    t1, _ = PB.sha_clocks(PB.H200, "A-wide + R-hi")
    assert r0 == "alu" and t0 / t1 == pytest.approx(1.38, abs=0.01)
    measured = 132 * 1980e6 / 4.79e9
    assert t0 == pytest.approx(measured, rel=0.02)
    t2, r2 = PB.sha_clocks(PB.SM120, "R-shf")
    assert r2 == "rf" and PB.sha_clocks(PB.SM120, "default build")[0] / t2 == pytest.approx(1.23, abs=0.01)
