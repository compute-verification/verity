import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import slowdown as S  # noqa: E402
import tiers as T  # noqa: E402


def test_each_tier_gives_the_adversary_no_more_than_the_last():
    effs = [T.evaluate(T.DEFAULTS, T.TIERS[k])["efficiency"] for k in ("T1", "T2", "T3", "T4")]
    assert all(b <= a * (1 + 1e-12) for a, b in zip(effs, effs[1:]))


def test_every_parameter_has_a_slider_and_its_default_is_in_range():
    assert {s["key"] for s in T.SPEC} == set(T.DEFAULTS)
    for s in T.SPEC:
        v = T.DEFAULTS[s["key"]]
        assert s["min"] <= v <= s["max"] or (v == 0 and s.get("zero")), s["key"]


def test_without_memory_mechanisms_the_residual_trains_at_full_efficiency():
    p = T.DEFAULTS | {"lag_s": 0.0}
    r = T.evaluate(p, T.TIERS["T1"])
    assert r["largest"] == "residual" and not r["memory_caps_residual"]
    assert r["residual"] == pytest.approx(T.residual_share(p), rel=1e-3)


def test_unmonitored_compute_trains_at_full_efficiency():
    r = T.evaluate(T.DEFAULTS | {"residual": 0.0, "host_share": 0.0, "clock_share": 0.0, "unmonitored": 0.3}, T.TIERS["T5"])
    assert r["unmonitored"] == pytest.approx(0.3, rel=1e-6)


def test_the_residual_voids_the_partial_sum_law():
    p = T.DEFAULTS | {"residual": 0.01, "host_share": 0.05, "clock_share": 0.0}
    r = T.evaluate(p, T.TIERS["T4"])
    assert r["declared"] == pytest.approx(2 / p["kappa"] / r["rho_T"])


def test_without_dark_compute_the_declared_side_is_the_chain_bound():
    """At zero residual and zero unmonitored compute nothing absorbs partial sums, so efficiency is the inverse of the
    calculator's chain bound for pretrain-full."""
    p = T.DEFAULTS | {"residual": 0.0, "host_share": 0.0, "clock_share": 0.0, "unmonitored": 0.0, "lag_s": 0.0}
    r = T.evaluate(p, T.TIERS["T4"])
    alpha = p["K"] / p["Gamma"]
    chain = p["kappa_chain"] * S.read_factor(p["K"], p["b_w"], p["b_a"], p["d_chain"]) / (2 * alpha)
    assert r["largest"] == "declared (partial-sum law)"
    assert 1 / r["efficiency"] == pytest.approx(chain, rel=1e-3)


def test_the_adversary_relays_partial_sums_only_when_it_pays():
    """With width-1 blocks the partial-sum law never reaches the thread bound, and relaying the whole residual into the
    declared blocks' sums would buy far less than the residual trains directly: the adversary relays only the few
    partial sums that pay, and the residual still trains at full efficiency."""
    p = T.DEFAULTS | {"d": 1.0, "e": 1e6, "lag_s": 0.0}
    r = T.evaluate(p, T.TIERS["T4"])
    assert r["largest"] == "residual" and 0 < r["declared"] < r["residual"]
    assert r["residual"] == pytest.approx(T.residual_share(p), rel=1e-6)
    assert r["crossing_bits"] / (2 * p["b_p"]) < 1e-6 * T.residual_share(p) * p["nodes"] * p["node_flops"] * T.YEAR


def test_absorbed_partial_sums_weaken_the_law_as_ps_prime_says():
    """PS' on Kimi K2's forward chain: G_D b_a / R at zeta = b_a Z / R of 0, 1 and 10 (Tier 1's table)."""
    rates = T.chain_rates(T.DEFAULTS, 2000)
    got = [T.chain_products(rates, 1.0, z) * T.DEFAULTS["b_a"] for z in (0.0, 1.0, 10.0)]
    assert got == pytest.approx([1714, 2482, 6483], rel=1e-3)


def test_the_wipe_floor_is_the_closed_form():
    """With every gate windowed and the wipe period below R / r, efficiency is 4 R r / (kappa b_a b_w d Phi)."""
    p = T.DEFAULTS | {"residual": 1.0, "unfilled_gb": 1.0, "pous_rounds": 1e6, "pous_floor": 0.0, "sigma": 0.0, "wipe_s": 1.0}
    r = T.evaluate(p, T.TIERS["T3"])
    rate = p["tau"] + p["K"] / p["Gamma"] * p["node_flops"]
    closed = 4 * r["wipe_state_bits"] * rate / (p["kappa"] * p["b_a"] * p["b_w"] * p["d"] * p["node_flops"])
    assert r["memory_caps_residual"]
    assert r["efficiency"] == pytest.approx(closed, rel=1e-3)


def test_memory_caps_the_residual_below_the_state_threshold():
    threshold = T.state_threshold(T.DEFAULTS, T.TIERS["T3"]) / T.GB
    tight = T.DEFAULTS | {"pous_rounds": 1e6, "pous_floor": 0.0}
    below = T.evaluate(tight | {"unfilled_gb": threshold / 2}, T.TIERS["T3"])
    above = T.evaluate(tight | {"unfilled_gb": threshold * 2}, T.TIERS["T3"])
    assert below["memory_caps_residual"] and not above["memory_caps_residual"]


def test_proof_of_space_rounds_set_the_slack():
    """rho^k <= delta: today's 1/19 certificate takes 86 rounds at delta = 1/100, and about 12,000 rounds leave under 1 GB
    of a 2.64 TB fill (memory-accounting, 19 Oct)."""
    no_floor = T.DEFAULTS | {"pous_floor": 0.0, "pous_rounds": 86.0}
    assert T.slack(no_floor) == pytest.approx(1 / 19, rel=0.02)
    mem_gb = T.DEFAULTS["hbm_gb"] + T.DEFAULTS["host_gb"]
    assert T.slack(no_floor | {"pous_rounds": 12000}) * mem_gb < 1.03
    assert T.slack(T.DEFAULTS | {"pous_rounds": 1e9, "pous_floor": 0.015}) == pytest.approx(0.015, rel=1e-3)


def test_unbounded_ingress_makes_wipes_cap_nothing():
    """Until the count check is in the ZK flow, declared ingress is bounded only by line rate (network-accounting, 19 Oct),
    and then even a small wipe residual caps nothing."""
    tight = T.DEFAULTS | {"pous_rounds": 1e6, "pous_floor": 0.0, "unfilled_gb": 0.01}
    assert T.evaluate(tight, T.TIERS["T3"])["memory_caps_residual"]
    assert not T.evaluate(tight | {"intake_x": 1.3e6}, T.TIERS["T3"])["memory_caps_residual"]


def test_disks_left_in_place_count_as_state():
    """An unwiped disk is state a wipe leaves (dark-compute red team, 19 Oct: 8 TB of NVMe undoes the memory cap)."""
    p = T.DEFAULTS | {"residual": 0.5}
    assert T.evaluate(p, T.TIERS["T3"])["memory_caps_residual"]
    assert T.evaluate(p | {"disk_gb": 8000.0}, T.TIERS["T3"])["efficiency"] > T.evaluate(p, T.TIERS["T3"])["efficiency"]


def test_helpers_within_the_read_deadline_refill_what_a_wipe_leaves():
    quiet = T.evaluate(T.DEFAULTS | {"sigma": 1e3}, T.TIERS["T3"])["wipe_state_bits"]
    loud = T.evaluate(T.DEFAULTS | {"sigma": 1e6}, T.TIERS["T3"])["wipe_state_bits"]
    mem = (T.DEFAULTS["hbm_gb"] + T.DEFAULTS["host_gb"]) * T.GB
    assert loud - quiet == pytest.approx((1e6 * 1e-3 - 25) / 524288 * mem)


def test_the_cheating_pool_does_not_grow_with_the_fleet():
    small = T.evaluate(T.DEFAULTS | {"nodes": 1e3}, T.TIERS["T5"])
    large = T.evaluate(T.DEFAULTS, T.TIERS["T5"])
    assert small["cheat_steps"] == large["cheat_steps"] > 0
    assert small["cheat"] == pytest.approx(100 * large["cheat"])
    q = T.DEFAULTS["share"] / (T.DEFAULTS["overhead"] + T.DEFAULTS["replay"])
    assert large["cheat_steps"] == pytest.approx(T.DEFAULTS["luck"] * math.log(1 / T.DEFAULTS["p_evade"]) / q)


def test_luck_multiplies_skipped_work_but_not_injected_bits():
    """A block that injects a chosen output fails whenever it is drawn, so ln(1/delta) (1/q + floor_steps) blocks inject
    step_bits each, whatever the luck factor."""
    p = T.DEFAULTS | {"floor_steps": 1e9}
    lucky, unlucky = T.evaluate(p, T.TIERS["T5"]), T.evaluate(p | {"luck": 1.0}, T.TIERS["T5"])
    assert lucky["cheat_steps"] == pytest.approx(8 * unlucky["cheat_steps"])
    assert lucky["cheat"] == unlucky["cheat"] > 0
    q = p["share"] / (p["overhead"] + p["replay"])
    assert lucky["cheat_bits"] == pytest.approx(math.log(1 / p["p_evade"]) * (1 / q + 1e9) * 16)


def _block_bound(p: dict, S: float) -> float:
    top = p["block_mults"] * p["Gamma"]
    de = S * max(p["e"], 2 * S / (4 * p["d"] * p["b_w"])) / p["b_a"] if S <= p["thread_bits"] else math.inf
    return min(top, S * S / (2 * p["b_w"] * p["b_a"]), de)


@pytest.mark.parametrize("over", [{}, {"thread_bits": 2.94e9}, {"thread_bits": math.inf}, {"K": 1e5},
                                  {"block_mults": 0.3, "thread_bits": 1e11}, {"K": 5e8}, {"d": 1.0, "e": 1e6}])
def test_the_chord_is_the_steepest_secant_of_one_block_bound(over):
    """Over a fine grid of reads, and out to a finite reach, no secant from K beats the chord, and one comes within
    0.1% of it."""
    p = T.DEFAULTS | over
    K, tb = p["K"], p["thread_bits"]
    grid = [K * 10 ** (i / 4000) for i in range(1, 24001)] + [tb * (1 + 1e-12)]
    for bits in (math.inf, 1e5, abs(tb - K), 3e8, 1e12):
        reads = [S for S in grid if S <= K + bits] + ([K + bits] if bits < math.inf else [])
        secants = [(_block_bound(p, S) - _block_bound(p, K)) / (S - K) for S in reads]
        got = T.chord(p, bits)
        assert max(secants) <= got * (1 + 1e-9)
        assert max(secants) >= got * (1 - 1e-3)


def test_an_injected_bit_buys_a_full_block_at_s_1():
    """At the defaults (d, e)-threading stops at thread_bits, below S_1 = sqrt(2 b_w b_a Gamma), where (1, 0)-threading
    first lets one block hold all its gates; with (d, e) to any reads and kappa Gamma / 2 a block, the chord is the
    secant to the (d, e) bound's full reads."""
    p = T.DEFAULTS
    s_1 = math.sqrt(2 * p["b_w"] * p["b_a"] * p["Gamma"])
    assert T.chord(p) == pytest.approx((p["Gamma"] - p["K"] * p["e"] / p["b_a"]) / (s_1 - p["K"]))
    old = p | {"thread_bits": math.inf, "block_mults": p["kappa"] / 2}
    S_full = math.sqrt(p["kappa"] * p["Gamma"] * p["d"] * p["b_w"] * p["b_a"])
    assert T.chord(old) == pytest.approx((p["kappa"] * p["Gamma"] / 2 - T.region_cap(p, p["K"])) / (S_full - p["K"]))
    assert T.chord(p | {"thread_bits": 2.94e9}) < T.chord(p) / 2


def test_the_amplifier_multiplies_every_term():
    """A recipe g times better in counted multiplications per unit of loss multiplies efficiency by g (paper red team)."""
    for tier in ("T1", "T5"):
        base = T.evaluate(T.DEFAULTS, T.TIERS[tier])
        amped = T.evaluate(T.DEFAULTS | {"g": 10.0}, T.TIERS[tier])
        for term in ("efficiency", "unmonitored", "residual", "declared", "cheat"):
            assert amped[term] == pytest.approx(10 * base[term], rel=1e-9)


def test_the_verdict_lag_adds_a_lag_of_skipped_work_per_identity():
    """Until the halt a developer can skip all declared work, once per identity (paper red team: 1 hour is 1.1e-4)."""
    p = T.DEFAULTS | {"host_share": 0.0}
    no_lag = T.evaluate(p | {"lag_s": 0.0}, T.TIERS["T1"])
    lag = T.evaluate(p | {"identities": 3.0}, T.TIERS["T1"])
    extra = 3 * p["lag_s"] / (p["years"] * T.YEAR)
    assert lag["residual"] - no_lag["residual"] == pytest.approx(extra, rel=1e-6)


def test_the_dampener_is_a_hardware_multiple_against_the_unconstrained_run():
    """Dark fragments of monitored capacity are counted at peak and unmonitored compute trains like the unconstrained
    developer, so D = 1 / (s + rest / mfu) (audience red team: against a real run at MFU 0.4, D at the defaults is 6.6)."""
    p = T.DEFAULTS | {"unmonitored": 0.01}
    r = T.evaluate(p, T.TIERS["T4"])
    rest = r["residual"] + r["declared"] + r["cheat"]
    assert r["dampener"] == pytest.approx(1 / (r["unmonitored"] + rest / p["mfu"]), rel=1e-9)
    assert T.evaluate(T.DEFAULTS | {"residual": 1.0}, T.TIERS["T1"])["dampener"] == 1.0


def test_the_recipe_multiplier_scales_only_the_constrained_terms():
    """m scales nu, chi and the memory cap but not the residual or unmonitored compute (C2 dossier)."""
    p = T.DEFAULTS | {"unmonitored": 0.01, "residual": 0.01, "host_share": 0.05, "clock_share": 0.0}
    base, amped = T.evaluate(p, T.TIERS["T5"]), T.evaluate(p | {"m": 3.0}, T.TIERS["T5"])
    assert amped["declared"] == pytest.approx(3 * base["declared"], rel=1e-6)
    assert amped["cheat"] == pytest.approx(3 * base["cheat"], rel=1e-6)
    assert amped["residual"] == pytest.approx(base["residual"], rel=1e-6)
    assert amped["unmonitored"] == pytest.approx(base["unmonitored"], rel=1e-6)
    capped = p | {"residual": 0.9}
    assert T.evaluate(capped | {"m": 3.0}, T.TIERS["T3"])["residual"] == pytest.approx(
        3 * T.evaluate(capped, T.TIERS["T3"])["residual"], rel=0.02)
