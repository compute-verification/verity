"""The constructions: the shard port against `benchmarks/nci/bracket.py`'s `search`, the tile against a brute-force
search, held layers against E1's Proposition 3 on its activation rate, and measured executions extended monotonically.
Every allowance must be achieved, never an estimate."""

import importlib.util
import itertools
import math
from dataclasses import replace
from pathlib import Path

import pytest
from slowdown import attacks, classes
from slowdown.attacks import ATTACKS, Measured, Point
from slowdown.classes import FlopBudget, Outside
from slowdown.params import Registry
from slowdown.profile import Profile
from slowdown.scenario import build

KIMI = Profile(
    name="full", kappa=0.602, state_bits=1.04e12 * 16, kappa_forward=0.294, token_state_bits=8e10,
    flops_per_token=2.07e11, cut_bits=2 * 28672, token_bits=20, decode_flops=0, state_splits=True,
    shared_state_bits=math.inf, tile_weight_bits=8, beta_x_marginal=2.6, psum_bits=16, tile_inputs=7168,
    tile_outputs=4096)
DECLARED = FlopBudget(6.4e16, 2e7)


def _nci_search():
    path = Path(__file__).resolve().parents[2] / "nci" / "bracket.py"
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
    nci = _nci_search()
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
    assert math.isclose(0.602 * 6.4e16 / (2 * a), _nci_search().search(*case)["rho"], rel_tol=1e-12)


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
    assert attacks.held_layers(10, 3, 1, 2, math.inf, 5) == 4
    assert attacks.held_layers(10, 3, 1, 2, 1, 5) == 2
    assert attacks.held_layers(10, 3, 1, 2, math.inf, 1) == 3.5
    assert attacks.held_layers(2, 3, 1, 2, math.inf, 5) == 0


def _held(name):
    """The scenario's server block, its first family's profile, and the forward products a block holding whole layers
    finishes over E1's Proposition 3's limit on its activation rate, by the block's fresh input bits."""
    s = build(Registry.load(), name)
    p = s.families[0].profile(s.model, s.run)
    products = p.kappa_forward * p.flops_per_token / (2 * p.layers)

    def over(k):
        forward = attacks.held_layers(k, p.layer_bits, products, p.beta_x * p.map_width, p.step_tokens, p.layers)
        return forward / classes.fanout_priced(k, p.fanout, p.beta_x)
    return s, p, products, over


def test_held_layers_breaks_the_activation_rate_at_the_server_block():
    """At the reference server block a block holding two whole layers finishes 1.93 times the forward products E1's
    Proposition 3 allows it on its activation rate, within the block's compute, and the calculator applies the rate
    there: the block is past the premise's credible size and short of the update's last layer."""
    s, p, products, over = _held("reference-onsite-h200")
    block = s.regime.budget(s.context)
    k = block.inputs
    assert over(k) == pytest.approx(1.93, rel=0.01)
    two = 2 * products * (k - 2 * p.layer_bits) / (p.beta_x * p.map_width)
    assert over(k) * classes.fanout_priced(k, p.fanout, p.beta_x) == pytest.approx(two)
    assert 2 * two / p.kappa_forward < s.context.units.flops(block.gates)
    assert k > s.premises["draft:E1-activation-rate"]["credible_bits"]
    assert classes.expensive_share(k, p.layer_bits, p.layers) > 0.9


def test_held_layers_breaks_the_activation_rate_from_one_layer_and_its_tokens_entries():
    """The construction finishes nothing below one layer's update, and passes the limit once the block also holds
    enough tokens' entries: from `layer_bits / (1 − aOut d / P)`, 2.56 layers' updates at the reference model, and at
    every larger block the calculator applies the rate at."""
    _, p, products, over = _held("reference-onsite-h200")
    assert over(p.layer_bits) == 0
    least = p.layer_bits / (1 - p.fanout * p.map_width / products)
    assert least / p.layer_bits == pytest.approx(2.56, rel=0.01)
    assert over(least * 0.999) < 1 < over(least * 1.001)
    last = (p.layers - 1) * p.layer_bits
    assert classes.expensive_share(last, p.layer_bits, p.layers) == 0
    assert all(over(least * (last / least) ** (i / 500)) > 1 for i in range(1, 501))


def test_held_layers_finishes_nothing_where_no_layer_fits():
    """`onsite-w63`'s server block is past the activation rate's credible size and smaller than one layer's update:
    the construction finishes nothing there, and the rate is unbroken at that size."""
    s, p, _, over = _held("onsite-w63")
    k = s.regime.budget(s.context).inputs
    assert s.premises["draft:E1-activation-rate"]["credible_bits"] < k < p.layer_bits
    assert over(k) == 0


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
    assert [(p.bits, p.slowdown) for p in pts] == [(2e7, 6.8e5), (8e6, 1.25e6)]
    assert all(math.isclose(p.flops / p.bits, attacks.PHI) for p in pts)
    a = ATTACKS["measured-full"].allowance(KIMI)(DECLARED)
    assert math.isclose(0.602 * DECLARED.compute / (2 * a), 6.4e16 / 5.36e16 * 1.25e6)
