"""The cheater's squarings under randomized gates (`cheater.py`) compute what they claim at the units the verdict reads,
their seams hold the reference's failure bound, and `verdict.py` reproduces the census's f."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import cheater as C  # noqa: E402
import verdict as V  # noqa: E402

#: the base constructions' units (property/v1, fixed count), as `verdict.py` reads them
BASE_UNITS = {"anyfn_seam": 160, "anyfn_fixup_b8": 173, "restricted_det_b8": 259, "restricted_sr_b7": 301,
              "restricted_noise1_b6": 357}


@pytest.mark.parametrize("name", C.BASE)
def test_base_constructions_are_right_proven_and_within_the_reference_seam(name):
    r = C.measure(name, trials=16)
    assert (r["right"], r["in_window"], r["declared_hold"]) == (16, 16, 16)
    assert r["proven"] and r["operand_bits_128"]
    assert r["units"] == BASE_UNITS[name]
    assert r["seam_fail_log2"] is None or r["seam_fail_log2"] <= C.reference_fail_log2()


def test_the_reference_bound_covers_seam_squares_measured_rate():
    """`squaring.py` measured `seam_square` wrong on 2^−12.4 of inputs; the bound, at the top carry's extreme, is no
    smaller, and is the one every window is held to."""
    assert -12.4 <= C.reference_fail_log2() < -6


@pytest.mark.parametrize("name", ["restricted_sr_b7", "restricted_noise1_b6"])
def test_a_randomized_laws_centered_window_is_exact(name):
    bld, _, _ = C.build(C.spec(name))
    assert bld.seam_fail_log2 == float("-inf")


def test_a_window_under_the_bound_is_wrong_on_inputs():
    """The bound is what makes the seam right: `restricted_det_b8` with a 2-digit window is flagged and is wrong."""
    sp = C.spec("restricted_det_b8")
    bld = C.RBuilder(C.DET)
    out = sp.build(bld, [bld.inp(i, *sp.digits) for i in range(C.n_digits(8))], 2)
    assert bld.seam_fail_log2 > C.reference_fail_log2()
    rng = random.Random(1)
    wrong = 0
    for _ in range(16):
        x = rng.randrange(C.P)
        vals, _, _ = bld.evaluate(dict(enumerate(C.to_digits(x, 8, False))))
        wrong += C.value([vals[g] for g in out], 8) % C.P != x * x % C.P
    assert wrong > 0


def test_sr_digits_need_radix_128():
    """An SR carry leaves a digit in `[−R, R)`: at `b = 8` that is outside s8, so the next squaring's dots aren't Proven."""
    assert C.digit_range(C.SR, 8) == (-256, 255)
    bld, _, _ = C.build(C.spec("restricted_sr_b8"))
    assert C.Q.proven(bld.gates)


@pytest.mark.parametrize("mech", [C.DET, C.SR, C.Mech("noise", 1), C.Mech("noise", 2)], ids=lambda m: m.name)
def test_exact_floor_is_exact_under_every_law(mech):
    bld = C.RBuilder(mech)
    x = bld.inp(0, -5000, 5000)
    q = C.exact_floor(bld, x, 6, 17)
    rng = random.Random(2)
    for _ in range(300):
        v = rng.randint(-5000, 5000)
        bld.rng = random.Random(rng.getrandbits(64))
        vals, _, _ = bld.evaluate({0: v})
        assert vals[q] == (17 + v) >> 6


def test_randomized_gates_refuse_any_function_gates():
    bld = C.RBuilder(C.SR)
    x = bld.inp(0, 0, 255)
    with pytest.raises(AssertionError):
        bld.anyfn(lambda v: v[0] % 7, [x], 0, 6)


@pytest.mark.parametrize("name", ["restricted_acc_det_b8_g4_r4", "restricted_acc_sr_b7_g4_r4", "anyfn_fixup_b8_g4_r4"])
def test_accumulator_noise_constructions_clean_every_draw(name):
    """Right on fresh noise each draw, at 2^4 times the honest square's n² products (each repeated 2^rep times)."""
    r = C.measure(name, trials=4)
    assert (r["right"], r["proven"], r["operand_bits_128"]) == (4, True, True)
    assert r["macs"] == C.n_digits(r["b"]) ** 2 * 16


def test_accumulator_noise_needs_its_guard_bits():
    with pytest.raises(AssertionError, match="guard 4 < noise 6"):
        C.build(C.spec("restricted_acc_sr_b7_g6_r4"))


def test_a_three_digit_fold_is_refused():
    assert len(C.fold_split(4)) == 2
    with pytest.raises(AssertionError, match="3 radix"):
        C.fold_split(3)


def test_a_rotated_dot_is_exact_with_an_anchor_and_small_digits():
    """`hadamard_anchor`: exact on fresh signs and at the ℓ1 bound; one digit more a block (past ℓ1 127 at the bound), or
    no anchor (the amax then rides on the signs), and the codes aren't the integers."""
    rng = random.Random(3)
    assert all(C.hadamard_anchor(rng, 4, 3, extreme=e) for e in (True, False, False))
    assert not C.hadamard_anchor(rng, 4, 3, extreme=True, q=C.anchor_q(4) + 1)
    assert not all(C.hadamard_anchor(rng, 4, 3, anchor=False) for _ in range(4))


def test_the_anchor_squaring_is_right_at_40x_the_macs():
    """`hadamard_anchor_b4`, the cheater's best under the MAC bound at Llama-3.1-8B: its squaring right, Proven, within
    OperandBits 128 and its seam bound, its longest column's dot exact through the rotation, at 40x `seam_square`'s
    MACs instead of the 128x of one nonzero a block."""
    r = C.measure("hadamard_anchor_b4", trials=4)
    assert (r["right"], r["proven"], r["operand_bits_128"]) == (4, True, True)
    assert r["rotation_right"] == r["rotation_trials"] == 2
    assert r["seam_fail_log2"] <= C.reference_fail_log2()
    assert (r["units"], r["macs"]) == (518, 655232)
    assert r["macs"] / 16384 == pytest.approx(40, rel=0.01)


def test_prf_floor():
    assert C.prf_units()["compression_floor"] == 80 * 16 - 48


# share.json `models.*.levers["as built"].f` at E conjectured and R_in = 1e5 (the census, art:dc25549e6f14…)
CENSUS_F = {("Llama-3.1-8B (8 engines)", "10 min"): 3.1348, ("Llama-3.1-8B (8 engines)", "1 h"): 18.8088,
            ("Llama-3.1-70B TP8", "10 min"): 0.755, ("Llama-3.1-70B TP8", "1 h"): 4.5297}


@pytest.mark.parametrize("model,life", list(CENSUS_F), ids=lambda x: str(x).split(" ")[0])
def test_verdict_reproduces_the_census_f(model, life):
    assert V.f(model, V.LIFETIMES[life], V.E_WORD["conjectured"]) == pytest.approx(CENSUS_F[(model, life)], rel=2e-4)


def test_the_mac_bound_only_binds_with_more_macs():
    m, L, E = "Llama-3.1-8B (8 engines)", 600, V.E_WORD["conjectured"]
    assert V.f(m, L, E, 16384) == V.f(m, L, E)
    assert V.f(m, L, E, 16384 * 128) < V.f(m, L, E) / 5
