"""The constructions: the shard port against `benchmarks/nci/slowdown.py`'s `search`, the tile against a brute-force
search, and measured executions extended monotonically. Every allowance must be achieved, never an estimate."""

import importlib.util
import itertools
import math
from dataclasses import replace
from pathlib import Path

import pytest
from rho import attacks, classes
from rho.attacks import ATTACKS, Measured, Point
from rho.classes import FlopBudget, Outside
from rho.params import Registry
from rho.profile import Profile
from rho.scenario import build

KIMI = Profile(
    name="full", kappa=0.602, state_bits=1.04e12 * 16, kappa_forward=0.294, token_state_bits=8e10,
    flops_per_token=2.07e11, cut_bits=2 * 28672, token_bits=20, decode_flops=0, state_splits=True,
    shared_state_bits=math.inf, tile_weight_bits=8, beta_x_marginal=2.6, psum_bits=16, tile_inputs=7168,
    tile_outputs=4096, layer_bits=1.04e12 * 2.5 / 61, layers=61, beta_x=0.87, beta_g=8, residual_width=7168,
    per_param=2, step_tokens=9.44e7)
DECLARED = FlopBudget(6.4e16, 2e7)
WINDOW = FlopBudget(5.5e20, 7.3e11)
#: a Kimi K2 layer's forward products a token
LAYER_PRODUCTS = 0.294 * 2.07e11 / (2 * 61)


def _nci_slowdown():
    path = Path(__file__).resolve().parents[2] / "nci" / "slowdown.py"
    spec = importlib.util.spec_from_file_location("nci_slowdown", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CASES = [
    (8e10, True, 2.07e11, 2 * 28672, 20, 0, 2e7, 6.4e16),
    (8e10, False, 2.07e11, 2 * 28672, 20, 0, 2e7, 6.4e16),
    (1e7, True, 2.07e11, 2 * 28672, 20, 0, 2e7, 6.4e16),
    (1e7, False, 1e9, 1e3, 0, 1e15, 2e7, 6.4e16),
    (8e10, True, 2.07e11, 2 * 28672, 20, 0, 7.3e11, 5.5e20),
    (3e9, True, 6e9, 4096, 17, 1e14, 1e9, 1e18),
    (5e7, False, 1e10, 1e4, 20, 0, 2e7, 6.4e16),
]


@pytest.mark.parametrize("case", CASES)
def test_the_shard_search_is_nci_search(case):
    nci = _nci_slowdown()
    gamma = case[-1]
    work = attacks.search(*case)
    try:
        expected = nci.search(*case)["rho"]
    except ValueError:
        assert work == 0
        return
    assert math.isclose(max(1.0, gamma / work), expected, rel_tol=1e-12)


def test_the_shard_allowance_is_the_counted_share_of_its_reference_work():
    case = (8e10, True, 2.07e11, 2 * 28672, 20, 0, 2e7, 6.4e16)
    a = ATTACKS["shard"].allowance(KIMI)(DECLARED)
    assert math.isclose(a, 0.602 / 2 * attacks.search(*case))
    assert math.isclose(0.602 * 6.4e16 / (2 * a), _nci_slowdown().search(*case)["rho"], rel_tol=1e-12)


def test_a_shared_state_cap_only_removes_slices():
    window = (8e10, True, 2.07e11, 2 * 28672, 20, 0, 7.3e11, 5.5e20)
    free = attacks.search(*window)
    capped = attacks.search(*window, shared=1.6e9)
    assert 0 < capped < free
    assert attacks.search(*window, shared=0) == 0
    unsplit = (1e7, False, 1e10, 1e4, 20, 0, 2e7, 6.4e16)
    assert attacks.search(*unsplit, shared=1.0) == attacks.search(*unsplit) > 0


def _brute(k, p, beta, psi, rows, cols):
    best = 0
    for r, w in itertools.product(range(1, int(rows) + 1), range(1, int(cols) + 1)):
        q = math.floor((k - r * w * p - 0) / (r * beta + w * psi))
        if q >= 1 and r * w * p + r * q * beta + w * q * psi <= k:
            best = max(best, r * w * q)
    return best


@pytest.mark.parametrize("k,rows,cols", [(3000, 40, 40), (8000, 12, 60), (20000, 80, 5)])
def test_the_tile_is_achieved_and_near_the_best_tile(k, rows, cols):
    got = attacks.tile(k, 2, 1.5, 3, rows, cols, grid=2000)
    best = _brute(k, 2, 1.5, 3, rows, cols)
    assert 0.9 * best <= got <= best


def test_the_tile_reaches_the_am_gm_optimum_when_unconstrained():
    k, p, beta, psi = 2e7, 8, 2.6, 16
    optimum = (k / 3) ** 1.5 / math.sqrt(p * beta * psi)
    got = attacks.tile(k, p, beta, psi, 1e9, 1e9)
    assert 0.98 * optimum <= got <= optimum
    assert attacks.tile(10, p, beta, psi, 1e9, 1e9) == 0


def test_the_tile_allowance_converts_its_forward_products():
    a = ATTACKS["tile"].allowance(KIMI)(DECLARED)
    assert math.isclose(a, attacks.tile(2e7, 8, 2.6, 16, 7168, 4096) * 0.602 / 0.294)
    with pytest.raises(Outside, match="psum_bits"):
        ATTACKS["tile"].allowance(replace(KIMI, psum_bits=None))


def test_held_layers_takes_the_best_whole_number_of_layers():
    assert attacks.held_layers(10, 3, 1, 1, 1, 1, math.inf, 5) == 4
    assert attacks.held_layers(10, 3, 1, 1, 1, 1, 1, 5) == 2
    assert attacks.held_layers(10, 3, 1, 1, 1, 1, math.inf, 1) == 3.5
    assert attacks.held_layers(2, 3, 1, 1, 1, 1, math.inf, 5) == 0


def test_the_held_layers_allowance_is_its_own_products():
    a = ATTACKS["held-layers"].allowance(KIMI)(WINDOW)
    assert a == attacks.held_layers(7.3e11, KIMI.layer_bits, LAYER_PRODUCTS, 0.87 * 7168, 8 * 7168, 2, 9.44e7, 61)
    assert ATTACKS["held-layers"].allowance(KIMI)(DECLARED) == 0
    with pytest.raises(Outside, match="beta_g"):
        ATTACKS["held-layers"].allowance(replace(KIMI, beta_g=None))


def test_the_pair_bound_answers_the_held_layers_attack():
    """Forward only, the attack's products are (weight, entry) pairs of the layers it holds, each layer's update at
    `b` bits a weight, so `PairBound` caps them at every `k`."""
    layer_weights = 1.04e12 / 61
    for k in (1e9, 5e10, 3.2e11, 7.3e11, 4e12, 1e14):
        got = attacks.held_layers(k, 2.5 * layer_weights, LAYER_PRODUCTS, 0.87 * 7168, 0, 1, math.inf, 61)
        assert got <= classes.pair(k, 2.5, 0.87, 7168)


def test_held_layers_breaks_the_activation_rate_at_the_serving_block():
    """At the reference serving block (`k_S` = 3.2e11 bits) a block holding two whole layers finishes 1.93 times the
    forward products E1's Proposition 3 allows it on its activation rate."""
    s = build(Registry.load(), "reference-onsite-h200")
    k = s.regime.budget(s.context).inputs
    p = s.families[0].profile(s.model, s.run)
    products = p.kappa_forward * p.flops_per_token / (2 * p.layers)
    forward = attacks.held_layers(k, p.layer_bits, products, p.beta_x * p.residual_width, 0, 1, p.step_tokens,
                                  p.layers)
    assert forward / classes.fanout_priced(k, p.fanout, p.beta_x) == pytest.approx(1.93, rel=0.01)


def test_a_measured_execution_extends_monotonically_and_never_beyond_what_it_ran():
    m = Measured("m", ("full",), (Point(2e7, 1.34e17, 6.8e5), Point(8e6, 5.36e16, 1.25e6)), ("art:x",))
    a = m.allowance(KIMI)
    assert a(FlopBudget(1.34e17, 2e7)) == pytest.approx(0.602 * 1.34e17 / (2 * 6.8e5))
    assert a(FlopBudget(6.4e16, 2e7)) == pytest.approx(0.602 * 5.36e16 / (2 * 1.25e6))
    assert a(FlopBudget(1e17, 7e6)) == 0
    assert a(FlopBudget(1.34e17 * 3, 2e7 * 3)) == pytest.approx(3 * a(FlopBudget(1.34e17, 2e7)))
    budgets = [FlopBudget(c, k) for c in (1e16, 6e16, 2e17, 1e19) for k in (1e6, 9e6, 3e7, 1e9)]
    for x, y in itertools.product(budgets, budgets):
        if x.compute <= y.compute and x.inputs <= y.inputs:
            assert a(x) <= a(y)
    for b in budgets:
        assert a(b) <= 0.602 * b.compute / 2
    with pytest.raises(Outside, match="frozen-prefix"):
        ATTACKS["measured-frozen-prefix"].allowance(KIMI)


def test_a_measured_state_that_fits_runs_unthrottled():
    p = replace(KIMI, name="rl-elicitation")
    a = ATTACKS["measured-rl-elicitation"].allowance(p)
    assert a(FlopBudget(6.4e16, 1e5)) == pytest.approx(0.602 * 6.4e16 / 2)
    assert a(FlopBudget(6.4e16, 7e4)) == 0


def test_the_measured_full_training_points_are_the_c1_rerun_high_ends():
    pts = ATTACKS["measured-full"].points
    assert [(p.bits, p.rho) for p in pts] == [(2e7, 6.8e5), (8e6, 1.25e6)]
    assert all(math.isclose(p.flops / p.bits, attacks.PHI) for p in pts)
    a = ATTACKS["measured-full"].allowance(KIMI)(DECLARED)
    assert math.isclose(0.602 * DECLARED.compute / (2 * a), 6.4e16 / 5.36e16 * 1.25e6)
