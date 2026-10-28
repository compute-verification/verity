"""Each class's count against the Kimi K2 numbers its Lean docstring (or E1's draft) states, and the class layer's rules:
counted sets, premise shares, `Outside`, monotonicity in `k`, and every cited premise listed in `data/premises.toml`.

Lean's overheads are in its cover model's units, `α₀ s / a` with `s = 4.52·10¹⁶` gates and `α₀` forward products per
gate (0.147, or 0.301 with Dgrad); E1's are `c / (κ a)` at matched compute `c = φ k` with `κ = 6` FLOPs per product.
"""

import importlib.util
import itertools
import math
from dataclasses import replace
from pathlib import Path

import pytest
from rho import classes, hardware, lean
from rho.classes import (
    CLASSES,
    STANDINGS,
    FlopBudget,
    Outside,
    counted,
    premise_status,
    premises,
)
from rho.params import Registry
from rho.profile import Profile
from rho.theorem import Budget

K = 2e7
S = 4.52e16
FWD, DGRAD = 0.147, 0.301
PHI = 6.7e9

#: Kimi K2 full pretraining with every field the classes read, at the values their docstrings use
KIMI = Profile(
    name="full", kappa=0.602, state_bits=1.04e12 * 16, b=2.5, b_w=2.5, c_w=2.5, c_a=4, b_a=4, f=4, width=390,
    beta_x=0.87, legs=512, fanout=37248, per_param=2, vocab=163840, bcol=1, thread_d=10, thread_e=37248, thread_c=2,
    kappa_forward=0.294, kappa_wgrad=0.294, layers=61, layer_weights=4e8, layer_bits=1.04e12 * 2.5 / 61,
    thread_k=2e7, entry_width=512, price=2.8, price_max=8, gamma=1, beta_v=0.63, map_width=7168, map_width_min=7168,
    h_eff=17, b_ps=1, step_tokens=6.7e7)
DECLARED = FlopBudget(6.4e16, K)
WINDOW = FlopBudget(5.5e20, 7.3e11)


def gates(alpha0, a):
    return alpha0 * S / a


def e1(a, k=K):
    return PHI * k / (6 * a)


def test_fanout_matches_its_docstrings():
    assert math.isclose(gates(FWD, classes.fanout(K, 2.5, 512, 37248, 1, 163840, 1)), 8900, rel_tol=0.01)
    a = classes.fanout(K, 2.5, 512, 37248, 2, 163840, 1)
    assert math.isclose(a / K - 1, 41026, rel_tol=1e-4)
    assert math.isclose(gates(DGRAD, a), 1.66e4, rel_tol=0.01)
    assert math.isclose(gates(FWD, classes.fanout_read(K, 2.5, 4, 512, 37248, 1, 163840, 1)), 3.6e4, rel_tol=0.01)
    assert math.isclose(gates(DGRAD, classes.fanout_read(K, 2.5, 4, 512, 37248, 2, 163840, 1)), 7.3e4, rel_tol=0.01)
    assert math.isclose(gates(FWD, K * 37248), 8900, rel_tol=0.01)


def test_partial_sums_match_their_docstrings():
    assert math.isclose(gates(FWD, classes.partial_sum(K, 2.5, 4, 390, 37248)), 6.1e4, rel_tol=0.02)
    assert math.isclose(gates(FWD, classes.partial_sum_keyed(K, 2.5, 512, 37248, 163840, 1, 512)), 1.6e4, rel_tol=0.02)
    assert math.isclose(gates(FWD, classes.partial_sum_keyed(K, 2.5, 512, 37248, 163840, 2.5, 512)), 1.9e4,
                        rel_tol=0.02)
    r = max(classes.ps_rate(2.5, 1, 390, 37248, 2 * K, i / 1000) for i in range(1001))
    assert math.isclose(r, 9.6e-5, rel_tol=0.02)
    assert classes.partial_sum_keyed(K, 2.5, 100, 37248, 163840, 1, 512) == math.inf


def test_the_covering_family_matches_its_docstrings():
    assert math.isclose(gates(FWD, classes.chained(K, 2.5, 4)), 166, rel_tol=0.01)
    assert math.isclose(gates(FWD, classes.covering_additive(K, 2.5, 4)), 660, rel_tol=0.01)
    assert classes.covering(K, 2.5, 4) == (K / 2.5) ** 2 + K
    assert classes.covering(K, 4, 4) == classes.chained(K, 4, 4)
    mb = 8e6
    assert math.isclose(gates(FWD, mb * mb), 104, rel_tol=0.01)
    assert math.isclose(gates(FWD, classes.covering_priced(mb, 16, 6.7, 16)), 104 * 6.7 * 16, rel_tol=0.01)
    assert math.isclose(gates(FWD, classes.covering_priced(mb, 6.7, 6.7, 16)), 104 * 6.7 ** 2, rel_tol=0.01)
    assert math.isclose(classes.covering_priced(K, 4, 2.5, 2.5), classes.chained(K, 2.5, 4))


def test_the_cover_theorem_is_the_covering_count_at_c_a_one_and_aligned_at_f():
    assert classes.cover(K, 2.5) == classes.chained(K, 2.5, 1)
    assert classes.cover_aligned(K, 2.5, 1) == classes.cover(K, 2.5)
    assert classes.cover_aligned(K, 2.5, 4) == classes.chained(K, 2.5, 4)


def _nci_search():
    path = Path(__file__).resolve().parents[2] / "nci" / "slowdown.py"
    spec = importlib.util.spec_from_file_location("nci_slowdown", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_threaded_agrees_with_nci_thread_lower_at_kimi_k2():
    row = {"kappa": 0.6, "b_w": 2.5, "b_a": 4, "d": 10, "e": 37248, "c": 2, "n_updated": 1.04e12, "U_bits": 2.6e12,
           "U_token_bits": 8e10, "decode_flops": 0, "w_cut": 28672, "crossings": 2, "flops_per_token": 2.07e11}
    gamma = 6.4e16
    nci = _nci_search().thread_lower(row, K, gamma, 20)
    ours = 0.6 * gamma / (2 * classes.threaded(K, 4, 2.5, 10, 37248, 2))
    assert math.isclose(nci, ours, rel_tol=1e-12)
    assert math.isclose(ours, 9600, rel_tol=0.01)
    assert classes.threaded_checked(2 * K, 4, 2.5, 10, 37248, 2, K) == math.inf


def test_e1_drafts_match_its_numbers():
    assert math.isclose(e1(classes.fanout_priced(K, 37248, 0.39)), 1.2e4, rel_tol=0.03)
    assert math.isclose(e1(classes.fanout_priced(K, 37248, 0.87)), 2.6e4, rel_tol=0.03)
    assert math.isclose(classes.fanout_split(K, 37248, 0.87, 1, 3, 3), 3 * classes.fanout_priced(K, 37248, 0.87) + K)
    for k, b_ps, overhead in ((2e7, 1, 950), (8e6, 1, 2.4e3), (2e7, 16, 1.5e4), (8e6, 16, 3.8e4)):
        assert math.isclose(e1(k * k / (b_ps * 17), k), overhead, rel_tol=0.03)
        assert classes.accumulator(k, 17, b_ps, 6.7e7) > k * k / (b_ps * 17)
    assert classes.accumulator(17 * 6.7e7, 17, 1, 6.7e7) == math.inf
    for beta_v, overhead in ((0.18, 2.8e4), (0.63, 9.8e4)):
        assert math.isclose(e1(classes.output_premise(K, 7168, 7168, 2.5, 0.87, beta_v)), overhead, rel_tol=0.03)
    h = (17 + 2700) / 7168, (17 + 6200) / 7168
    assert math.isclose(classes.output_cap(7168, 0.06, h[0], 0.63) / 8e6, 0.9, rel_tol=0.05)
    assert math.isclose(classes.output_cap(7168, 0.19, h[1], 0.18) / 8e6, 23, rel_tol=0.05)
    assert classes.output_premise(1e9, 7168, 7168, 0.06, h[0], 0.63) == math.inf


def test_premise_shares():
    assert classes.upstream_share(K, 2.5, 61, 4e8) == 60 / 61
    assert classes.upstream_share(1e10, 2.5, 61, 4e8) == 5 / 61
    assert classes.upstream_share(WINDOW.inputs, 2.5, 61, 4e8) == 0
    assert classes.chained_share(K, 2.5, 61, 4e8) == 60 / 61
    assert classes.chained_share(1e10, 2.5, 61, 4e8) == 50 / 61
    assert classes.chained_share(WINDOW.inputs, 2.5, 61, 4e8) == 0
    assert classes.chained_share_one(1e10, 4, 2.5, 61, 4e8) == classes.chained_share(1e10, 2.5, 61, 4e8)
    bits = KIMI.layer_bits
    assert classes.expensive_share(K, bits, 61) == 60 / 61
    assert classes.expensive_share(17.5 * bits, bits, 61) == 43 / 61
    assert classes.expensive_share(61 * bits, bits, 61) == 0


def test_a_count_on_another_set_is_converted_to_the_profiles_counted_products_and_divided_by_its_share():
    for name in ("covering-two-costs", "cover-chained"):
        a = CLASSES[name].allowance(KIMI)(DECLARED)
        assert math.isclose(a, classes.chained(K, 2.5, 4) * 0.602 / 0.294 * 61 / 60)
    assert math.isclose(CLASSES["cover"].allowance(KIMI)(DECLARED), classes.cover(K, 2.5) * 61 / 60)
    assert math.isclose(CLASSES["fanout-priced"].allowance(KIMI)(DECLARED),
                        classes.fanout_priced(K, 37248, 0.87) * 0.602 / 0.294 * 61 / 60)
    assert math.isclose(CLASSES["accumulator"].allowance(KIMI)(DECLARED),
                        classes.accumulator(K, 17, 1, 6.7e7) * 0.602 / 0.294)
    assert counted(KIMI, "keyed") == 1.0
    assert math.isclose(counted(replace(KIMI, per_param=1), "keyed"), 0.602 / 0.294)
    assert math.isclose(CLASSES["fanout"].allowance(replace(KIMI, per_param=1))(DECLARED),
                        classes.fanout(K, 2.5, 512, 37248, 1, 163840, 1) * 0.602 / 0.294)
    for arg in CLASSES.values():
        assert arg.counts in ("kappa", "keyed") or arg.counts in Profile.__dataclass_fields__


def test_at_the_serving_block_only_the_linear_counts_cap():
    finite = {n for n, arg in CLASSES.items() if arg.allowance(KIMI)(WINDOW) < KIMI.kappa * WINDOW.compute / 2}
    assert finite == {"fanout-priced", "fanout-split"}


def test_a_class_whose_field_is_unknown_is_outside():
    for name, fieldname in (("cover", "layers"), ("covering", "kappa_forward"), ("threaded", "thread_k"),
                            ("fanout-priced", "layer_bits"), ("accumulator", "kappa_wgrad")):
        with pytest.raises(Outside, match=fieldname):
            CLASSES[name].allowance(replace(KIMI, **{fieldname: None}))


@pytest.mark.parametrize("name", list(CLASSES))
def test_every_class_allows_no_less_with_more_bits(name):
    a = CLASSES[name].allowance(KIMI)
    ks = [10 ** (4 + i / 4) for i in range(36)]
    counts = [a(FlopBudget(1e30, k)) for k in ks]
    assert all(x <= y for x, y in itertools.pairwise(counts)), counts


def test_every_premise_a_class_cites_is_listed_with_a_known_standing():
    table = premises()
    for arg in CLASSES.values():
        for c in arg.cites[1:]:
            if lean.status(c) in ("assumption", "draft"):
                assert c in table, f"{arg.name} cites {c}, which data/premises.toml doesn't list"
    for row in table.values():
        assert row["standing"] in STANDINGS and row["beyond"] in STANDINGS
        assert STANDINGS.index(row["beyond"]) >= STANDINGS.index(row["standing"])


def test_premise_status_is_the_weakest_and_never_strengthens_with_k():
    assert premise_status(CLASSES["fanout"].cites, K) == "supported"
    assert premise_status(CLASSES["fanout-read"].cites, K) == "contested"
    assert premise_status(CLASSES["partial-sum"].cites, WINDOW.inputs) == "refuted"
    assert premise_status(("Nci.Guarantees.CoveringBound",), K) == "supported"
    for arg in CLASSES.values():
        s = [STANDINGS.index(premise_status(arg.cites, 10 ** e)) for e in range(4, 13)]
        assert s == sorted(s), arg.name


def test_every_class_cites_its_statement_first():
    for arg in CLASSES.values():
        assert lean.status(arg.cites[0]) in ("proved", "draft"), arg.name


def test_a_class_caps_a_block_of_gates_through_the_one_conversion():
    """`q_T(c, k)` on gates is the FLOP-model statement at `c / G` FLOPs, and a gate holds `w = κ / (2G)`: the capacity
    `w c` is `κ/2` a FLOP."""
    u = hardware.units(Registry.load())
    a = CLASSES["fanout-priced"].allowance(KIMI)
    c = u.gates(WINDOW.compute)
    assert classes.per_block(a, u)(Budget(c, WINDOW.inputs)) == a(WINDOW)
    assert classes.weight_per_gate(KIMI, u) * c == pytest.approx(KIMI.kappa * WINDOW.compute / 2)
