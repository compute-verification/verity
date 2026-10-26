import math
import numbers
from dataclasses import dataclass, replace

import pytest
from rho import instances, lean, pofc
from rho.classes import Outside
from rho.params import Registry
from rho.profile import Profile
from rho.scenario import build, evaluate
from rho.theorem import Budget

REG = Registry.load()
RULES = pofc.rules(REG)
N_MAX = (1e4, 1e5, 1e6)

#: Daniel's Notion page "Proof of forward computation": one layer's boundary positions, and the toll in % at n_max 1e4,
#: 1e5 and 1e6 when the gradient must reach the top L layers.
PER_LAYER = {"llama-3.1-405b": 800e3, "kimi-k3": 25e6}
PAGE = {
    "llama-3.1-405b": {1: (8_000, 800, 80), 4: (32_000, 3_200, 320), 16: (130_000, 13_000, 1_300),
                       64: (510_000, 51_000, 5_100)},
    "kimi-k3": {1: (250_000, 25_000, 2_500), 4: (1_000_000, 100_000, 10_000), 16: (4_000_000, 400_000, 40_000),
                64: (16_000_000, 1_600_000, 160_000)},
}


def sig(x: float, n: int) -> float:
    return float(f"{x:.{n}g}")


@pytest.mark.parametrize("name", PAGE)
def test_one_layer_costs_the_pages_positions(name):
    """784,738 positions a Llama-405B layer (the page's ~800k) and 25.4M a Kimi K3 layer (~25M)."""
    m = pofc.stack(REG, name)
    assert RULES.weight_bits == 16 and RULES.act_bits == 4 and RULES.s(m) == m.d_model * 4
    assert math.isclose(pofc.positions(m, RULES.weight_bits, RULES.act_bits), PER_LAYER[name], rel_tol=0.025)


@pytest.mark.parametrize("name", PAGE)
def test_the_toll_tables_are_the_pages_to_every_cell(name):
    """The page multiplies its per-layer positions by the depth over `n_max`, to two figures; the exact toll is within
    that rounding, and Kimi K3's top 64 is its whole model."""
    m = pofc.stack(REG, name)
    s = RULES.s(m)
    exact = pofc.tolls(m, PAGE[name], N_MAX, RULES)
    for depth, row in PAGE[name].items():
        for n, cell in zip(N_MAX, row, strict=True):
            assert sig(100 * pofc.toll(depth * PER_LAYER[name] * s, n, s), 2) == cell
            assert math.isclose(100 * exact[depth, n], cell, rel_tol=0.05)
    if name == "kimi-k3":
        assert all(sig(100 * exact[64, n], 2) == cell for n, cell in zip(N_MAX, PAGE[name][64], strict=True))
        assert math.isclose(exact[64, 1e4], pofc.toll(m.n_total * RULES.weight_bits, 1e4, s))


def test_the_toll_is_the_weights_over_an_episodes_boundary():
    assert pofc.toll(8e5, 1e4, 4) == 20.0
    m = pofc.Stack("tiny", 1e6, 4, 1000)
    assert pofc.depth_toll(m, 9, 1e3, 16, 4) == pofc.depth_toll(m, 4, 1e3, 16, 4) == 1e6 * 16 / (1e3 * 4000)


def reference():
    s = build(REG, "reference-onsite-h200")
    return s, s.regime.partition(s.context)


def test_the_wrapper_gives_episodes_the_forward_cut_and_leaves_nci_blocks_without_one():
    s, inner = reference()
    (r0,), (r,) = inner.regions, pofc.wrap(s.regime, RULES, s.model).partition(s.context).regions
    cuts = {c.name: c for c in r.cuts}
    assert set(cuts) == {"declared", "physical", "declared+pofc"}
    assert r.cuts[:2] == r0.cuts and all(math.isinf(b.budget.cut) for c in r0.cuts for b in c.blocks)
    (ep,), (decl,) = cuts["declared+pofc"].blocks, cuts["declared"].blocks
    assert ep.kind == "episode" and ep.budget.cut == RULES.n_max * s.model.d_model * RULES.act_bits
    assert ep.budget.inputs == RULES.x and cuts["declared+pofc"].free == cuts["declared"].free
    assert pofc.RULES in cuts["declared+pofc"].cites and set(cuts["declared"].cites) <= set(cuts["declared+pofc"].cites)
    # at the reference n_max an episode is the 1M-token session the NCI policy declares, with its cut
    assert math.isclose(ep.budget.compute, decl.budget.compute) and math.isclose(ep.budget.inputs, decl.budget.inputs)
    assert math.isclose(ep.count, decl.count)
    layout = {b.kind: b.budget for b in r.layout.blocks}
    assert set(layout) == {"episode", "uncertified"} and math.isinf(layout["uncertified"].cut)


def test_a_shorter_episode_recuts_the_declared_compute_without_changing_it():
    s, inner = reference()
    r = pofc.wrap(s.regime, replace(RULES, n_max=1e4), s.model).partition(s.context).regions[0]
    (decl,), (ep,) = inner.regions[0].cuts[0].blocks, next(c for c in r.cuts if c.name == "declared+pofc").blocks
    assert math.isclose(ep.count * ep.budget.compute, decl.count * decl.budget.compute)
    assert math.isclose(ep.count, 100 * decl.count) and ep.budget.inputs == 1e4 * RULES.input_bits


def test_the_physical_instantiation_gives_the_window_the_links_bits():
    s, _ = reference()
    bw = pofc.link(REG, s.context.location.name)
    r = pofc.wrap(s.regime, RULES, s.model, bw).partition(s.context).regions[0]
    cuts = {c.name: c for c in r.cuts}
    (phys,) = cuts["physical+pofc"].blocks
    assert math.isclose(phys.budget.cut, bw * s.regime.wipe_s) and pofc.PHYSICAL in cuts["physical+pofc"].cites
    assert all(math.isinf(b.budget.cut) for b in cuts["physical"].blocks)
    assert math.isinf({b.kind: b.budget for b in r.layout.blocks}["uncertified"].cut)
    # the window's link carries many episodes' cuts: a matched link is three orders slower than NVLink's forward half
    assert 1e3 < bw / pofc.matched_link(RULES, s.model, s.context) < 1e4


def test_the_useful_share_is_the_pages_when_a_token_costs_a_position():
    t = pofc.Trained("x", kappa=0.5, depth=2, layer_params=1e3, weight_bits=16, held=0, spare=0.0, position_bits=8,
                     token_flops=10)
    n = 1e4
    budget = Budget(compute=t.token_flops * n, inputs=0.0, cut=n * t.position_bits)
    assert math.isclose(t.toll(budget), 0.4)
    assert math.isclose(t.within(budget), (1 - 0.4) * t.kappa * budget.compute / 2)
    assert t.within(replace(budget, cut=t.behind())) == 0.0
    assert t.behind(bits=4) == t.behind() / 4
    assert replace(t, spare=t.behind()).behind() == 0.0 and replace(t, held=2).behind() == 0.0


def profile(**kw):
    return Profile("full", kappa=0.6, state_bits=1.0, **kw)


def test_backprop_caps_only_under_a_forward_cut_below_a_toll_of_one():
    s, _ = reference()
    a = pofc.Backprop(s.model, RULES).allowance(profile())
    ep = pofc.episode(RULES, s.model)
    assert math.isinf(a(ep)) and math.isinf(a(Budget(ep.compute, ep.inputs)))
    wide = pofc.episode(replace(RULES, n_max=1e9), s.model)
    t = pofc.trained(profile(), s.model, RULES)
    assert 0 < t.toll(wide) < 1 and math.isclose(a(wide), t.within(wide)) and a(wide) < 0.6 * wide.compute / 2
    signal = pofc.Backprop(s.model, RULES, signal=True).allowance(profile())
    assert math.isclose(signal(ep), t.tokens(ep.inputs)) and signal(ep) < 0.6 * ep.compute / 2


@dataclass(frozen=True)
class Deep(Profile):
    depth: int | None = None


def test_the_profiles_depth_is_read_when_the_argument_gives_none():
    s, _ = reference()
    assert pofc.trained(Deep("x", 0.6, 1.0, depth=3), s.model, RULES).depth == 3
    assert pofc.trained(Deep("x", 0.6, 1.0, depth=3), s.model, RULES, depth=5).depth == 5
    assert pofc.trained(profile(), s.model, RULES).depth == s.model.layers


def test_the_constructions_open_where_the_page_says():
    s, _ = reference()
    m, p = s.model, profile()
    with pytest.raises(Outside):
        pofc.LastLayers(m, RULES, 1).allowance(p)
    wide_p = replace(RULES, per_operator=2.0)
    assert pofc.trained(p, m, wide_p).held == 1
    ep = pofc.episode(wide_p, m)
    assert pofc.LastLayers(m, wide_p, 1).allowance(p)(ep) == 0.6 * ep.compute / 2
    assert pofc.LastLayers(m, wide_p, 1).allowance(p)(Budget(ep.compute, ep.inputs)) == 0.0
    for rules in (RULES, replace(RULES, commit=3.0)):
        with pytest.raises(Outside):
            pofc.Palindrome(m, rules).allowance(p)
    assert pofc.Palindrome(m, replace(RULES, commit=3.0, per_operator=2.0)).allowance(p)(ep) == 0.6 * ep.compute / 2
    big = pofc.episode(replace(RULES, n_max=1e8), m)
    t = pofc.trained(p, m, RULES, 1)
    full, served = (pofc.WeightsAsActivations(m, RULES, 1, b).allowance(p)(big) for b in (None, RULES.act_bits))
    assert math.isclose(full, t.within(big)) and math.isclose(served, t.within(big, RULES.act_bits)) and served > full
    assert pofc.WeightsAsActivations(m, RULES).allowance(p)(pofc.episode(RULES, m)) == 0.0
    assert pofc.junk_share(RULES, m, s.context) == 1.0
    assert pofc.junk_share(replace(RULES, episodes_per_day=1.0), m, s.context) < 1e-3


def test_every_pofc_number_says_where_it_comes_from_and_how_sure_it_is():
    params = [p for p in REG.params() if p.key.startswith("pofc.") and isinstance(p.value, numbers.Number)]
    assert params and all(p.source and p.status for p in params)
    assert {"n_max", "act_bits", "weight_bits", "input_bits", "output_bits", "commit", "per_operator",
            "episodes_per_day"} <= set(REG.table("pofc.rules.reference"))
    assert REG.param("pofc.rules.reference.n_max").low == 1e4 and REG.param("pofc.rules.reference.n_max").high == 1e6


def test_every_pofc_argument_is_a_draft_citing_the_page():
    s, _ = reference()
    args = (*pofc.bounds(s.model, RULES), *pofc.attacks(s.model, RULES, s.context))
    assert len({a.name for a in args}) == len(args)
    assert all(lean.weakest(a.cites) == "draft" for a in args)
    assert all(a.side == "bound" for a in pofc.bounds(s.model, RULES))
    assert all(a.side == "attack" for a in pofc.attacks(s.model, RULES, s.context))


def test_pofc_never_lowers_the_guarantee_and_its_class_binds_only_below_a_toll_of_one():
    points = pofc.versus(REG, n_maxes=(1e4, 1e6, 1e8), depths=(None, 1))
    assert points
    for p in points:
        assert p.lo >= p.unwrapped * (1 - 1e-9)
        assert (p.toll >= 1) == math.isclose(p.pofc, 1.0) and p.signal >= 1.0
    full = [p for p in points if p.family == "full"]
    assert [p.toll for p in full] == sorted((p.toll for p in full), reverse=True) and min(p.toll for p in full) > 1


def test_a_scenario_under_pofc_evaluates_and_its_bracket_is_ordered_unless_a_premise_is_named_false():
    s, _ = reference()
    for n in N_MAX:
        w = pofc.apply(s, replace(RULES, n_max=n))
        assert w.regime.name == f"{s.regime.name}+pofc" and len(w.bounds) == len(s.bounds) + 2
        for f in evaluate(w).families:
            assert f.lo <= f.hi * (1 + 1e-9) or f.contradictions


def test_a_stack_builds_from_its_table_and_a_model_reads_as_one():
    m = instances.model(REG, "deepseek-v4-pro")
    assert pofc.layer_params(m) == m.n_total / m.layers
    assert pofc.stack(REG, "kimi-k3").layers == 61
