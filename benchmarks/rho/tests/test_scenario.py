import math
from dataclasses import replace

import pytest
from rho import analysis, classes, lean
from rho.attacks import ATTACKS
from rho.classes import CLASSES
from rho.envelopes import KINDS, PARTS, ROUTES
from rho.params import Registry
from rho.scenario import build, evaluate, evaluate_named, reads
from rho.theorem import multiple_bound

REG = Registry.load()


@pytest.mark.parametrize("name", REG.names("scenario"))
def test_every_scenario_evaluates_below_its_best_attack_unless_a_premise_is_named_false(name):
    """The lower bound is at most the best known attack, or a bound above the attack is reported as a contradiction.
    It never stands on a refuted premise, and is at least every bound alone whose premises aren't refuted where it
    binds."""
    r = evaluate_named(REG, name)
    assert r.families
    for f in r.families:
        assert f.lo <= f.attack * (1 + 1e-9) or f.contradictions
        assert f.combined.premise != "refuted"
        assert all(e.rho <= f.lo * (1 + 1e-9) for e in f.bounds if e.overhead and e.premise != "refuted")


def test_a_bound_refuted_at_a_block_does_not_bound_it_and_is_still_reported_alone():
    """partial-sum's `ActivationReads` is refuted past 4·10⁹ bits, so on its own it binds NVL72's serving cut (`k_S`
    far past that) at a standing of refuted, its padded size ratio 76 times the lower bound's (both below 1, so both
    slowdowns are 1). Every bound at once takes no cap from it at that block, its standing doesn't reach the lower
    bound's, and the lower bound is still at least every bound alone whose premises aren't refuted."""
    f = evaluate_named(REG, "reference-onsite-nvl72-elementary").families[0]
    ps = next(e for e in f.bounds if e.argument == "partial-sum")
    assert ps.premise == "refuted" and {s.cut for s in ps.overhead.serving.values()} == {"serving"}
    assert f.combined.padded < ps.padded / 10 and f.combined.premise != "refuted"
    assert all(e.rho <= f.lo * (1 + 1e-9) for e in f.bounds if e.overhead and e.premise != "refuted")


def test_a_bound_whose_premise_is_refuted_everywhere_never_sets_the_lower_bound():
    """Refuting E1's activation rate at every block size takes fanout-priced and fanout-split (which cite it) out of
    the lower bound: onsite H200's is then exactly the lower bound without them, and both are still reported alone,
    tagged refuted, at the overhead they had."""
    name, premise = "reference-onsite-h200", "premise.e1-activation-rate"
    base = evaluate_named(REG, name).families[0]
    refuted = REG.override({f"{premise}.standing": "refuted", f"{premise}.beyond": "refuted"})
    f = evaluate_named(refuted, name).families[0]
    cite = REG[f"{premise}.lean"]
    citing = {a.name for a in CLASSES.values() if cite in a.cites}
    assert citing == {"fanout-priced", "fanout-split"}
    s = build(REG, name)
    without = evaluate(replace(s, bounds=tuple(a for a in s.bounds if a.name not in citing))).families[0]
    assert set(f.combined.excluded) == citing and replace(f.combined, excluded=()) == without.combined
    assert f.lo < base.lo / 10 and f.combined.premise != "refuted"
    alone = {e.argument: e for e in f.bounds}
    assert all(alone[a].premise == "refuted" and alone[a].rho == next(e.rho for e in base.bounds if e.argument == a)
               for a in citing)
    assert max(e.rho for e in f.bounds if e.overhead) == pytest.approx(base.lo)


def test_s_times_and_gates_per_flop_come_from_one_source():
    """The bound holds a gate at `u/w = ω_max 2G / (κ s_×)` of the reference's weight, so the FLOP model's `G` beside
    a Boolean design's measured `s_×` (681 gates a bf16 product) scales every weight counted at `u` (the dark
    leaves, most of the rate of onsite H200's lower bound, `Overhead.rates`) by `681 w`, about 0.82, with nothing to say
    so. Building refuses the mix;
    with both from that design it builds, and the lower bound moves by that much on the share counted at `u`."""
    design = "boolean:AmpereBF16TcDot16_v3"
    name = "reference-onsite-h200"
    mixed = REG.override({"structure.reference.s_times": 681, "structure.reference.basis": design})
    with pytest.raises(ValueError, match="one source"):
        build(mixed, name)
    with pytest.raises(ValueError, match="one source"):
        build(REG.override({"units.basis": design}), name)
    base, one = evaluate_named(REG, name), evaluate_named(mixed.override({"units.basis": design}), name)
    w = classes.weight_per_gate(base.families[0].profile, base.scenario.context.units)
    assert base.scenario.structure.basis == base.scenario.context.units.basis == "flop-model"
    at_u = sum(t.share for t in analysis.attribution(base) if t.name == "dark" or t.name.split(".")[0] in ROUTES)
    assert 0.5 < at_u <= 1
    assert one.lo / base.lo == pytest.approx(1 / (at_u / (681 * w) + 1 - at_u), rel=5e-3)
    assert build(REG.override({"units.basis": design}), "reference-onsite-h200-elementary").structure.s_times == 1


def test_unmonitored_is_the_algorithm_alone():
    """The bound counts every gate at capacity, so its padded size ratio is `μ`; a construction trains at its MFU on a
    circuit padded to capacity, so the best attack's is `μ / MFU`. The overhead is against the reference at that MFU, so
    no protocol gives `max(1, μ)` for the attack, 1 for the reference's own family, and the reference's time."""
    r = evaluate_named(REG, "unmonitored-h200-kimi")
    f0 = r.scenario.context.fleet
    mfu = f0.eps_dark * f0.dark_mfu + (1 - f0.eps_dark) * f0.mfu_train
    assert r.scenario.mfu == pytest.approx(mfu)
    for f in r.families:
        mu = f.profile.efficiency
        assert math.isclose(f.combined.padded, mu) and math.isclose(f.best_attack.padded, mu / mfu)
        assert f.lo == pytest.approx(multiple_bound(mfu, f0.eps_dark / mu, (1 - f0.eps_dark) / mu))
        assert f.attack == pytest.approx(max(1.0, mu))
    s = r.scenario
    assert r.lo == r.attack == 1.0 and math.isclose(s.seconds(r.attack), s.reference / (mfu * s.total_phi))


def test_onsite_adds_a_cut_so_its_lower_bound_is_never_below_remotes():
    assert evaluate_named(REG, "onsite-h200-kimi").lo >= evaluate_named(REG, "remote-h200-kimi").lo * (1 - 1e-9)


def test_the_counts_match_the_lean_docstrings_at_kimi_k2():
    k = 2e7
    # FanOutIntensity: 7.45e11 products per block at c = 1; FanOutReadIntensity: 1.86e11 at f = 4
    assert math.isclose(classes.fanout(k, 2.5, 512, 37248, 1, 163840, 1), 7.45e11, rel_tol=2e-3)
    assert math.isclose(classes.fanout_read(k, 2.5, 4, 512, 37248, 1, 163840, 1), 1.86e11, rel_tol=2e-3)
    assert classes.lookups(k, 1, 163840) == 122
    assert classes.fanout(k, 2.5, 100, 37248, 1, 163840, 1) == math.inf
    assert classes.cover(k, 2.5) == k * k / 2.5 + k


def test_every_lean_citation_is_in_a_lock():
    cites = [c for a in (*CLASSES.values(), *ATTACKS.values()) for c in a.cites]
    cites += [c for kind in (*KINDS.values(), *PARTS.values())
              for c in getattr(kind, "CITES", ()) + getattr(kind, "PHYSICAL", ())]
    absent = sorted({c for c in cites if lean.status(c) == "absent"})
    assert not absent, f"no lock lists {absent}"


@pytest.mark.parametrize("name", REG.names("scenario"))
def test_admitting_weaker_statuses_never_lowers_the_guarantee(name):
    r = evaluate_named(REG, name)
    rhos = [analysis.upto(s)(r) for s in ("proved", "conditional", "draft", "absent")]
    assert rhos == sorted(rhos) and math.isclose(rhos[-1], r.lo)
    assert all(0 <= f.combined.overhead.monitored <= 1 for f in r.families)


def test_a_scenario_reads_its_own_tables_and_not_the_others():
    keys = reads(REG, "reference-onsite-h200")
    assert "fleet.world-h200.eps_dark" in keys and "regime.onsite.k_wipe" in keys and "units.gates_per_flop" in keys
    assert not any(k.startswith(("fleet.world-nvl72.", "model.kimi-k2.")) for k in keys)


@pytest.mark.parametrize("name", ["reference-remote-h200", "reference-onsite-h200", "reference-onsite-nvl72",
                                  "reference-onsite-h200-per-device"])
def test_rho_and_the_time_to_the_target_do_not_depend_on_gates_per_flop(name):
    """Every FLOP-model input becomes gates through the one `units.gates_per_flop`, and the bound compares gates with
    gates (`s_×` at the reference's gates per counted product moves with it), so the conversion cancels: at its low
    and high ends the lower bound, the best attack and their times are today's."""
    def numbers(g: float) -> tuple[float, ...]:
        r = evaluate_named(REG.override({"units.gates_per_flop": g}), name)
        return r.lo, r.attack, r.scenario.seconds(r.lo), r.scenario.seconds(r.attack)

    p = REG.param("units.gates_per_flop")
    base = numbers(p.value)
    for g in (p.low, p.high):
        assert numbers(g) == pytest.approx(base, rel=1e-9)
    assert build(REG.override({"units.gates_per_flop": p.high}), name).context.c == pytest.approx(
        build(REG, name).context.c * p.high / p.value)


def test_s_times_defaults_to_the_reference_average_under_the_straddling_premise():
    """`structure.reference` is `s_×` at the reference's gates per counted product (`2G/κ`, so `u = w` and the overhead
    is what it was), and every bound that counts uncertified or dark gates there cites `draft:no-straddle`. The
    elementary cap (one operation a gate) needs no such premise; it leaves the attacks as they are, takes remote's lower
    bound down by `2G/κ` (inference is 0.0% of it), and is what `upto("conditional")` falls back to."""
    r, e = evaluate_named(REG, "reference-remote-h200"), evaluate_named(REG, "reference-remote-h200-elementary")
    st, f = r.scenario.structure, r.families[0]
    w = classes.weight_per_gate(f.profile, r.scenario.context.units)
    assert st.s_times == classes.REFERENCE and st.weight(w) == w and st.gates(w) == pytest.approx(2 * 250 / 0.602)
    assert "draft:no-straddle" in f.combined.cites and f.combined.status == "draft"
    assert not any("draft:no-straddle" in a.cites for a in f.attacks)
    assert e.scenario.structure.cites() == () and "draft:no-straddle" not in e.families[0].combined.cites
    assert e.attack == pytest.approx(r.attack, rel=1e-12)
    assert e.families[0].combined.padded * st.gates(w) == pytest.approx(f.combined.padded, rel=1e-3)
    assert analysis.upto("conditional")(r) == pytest.approx(analysis.upto("conditional")(e), rel=1e-12)


def test_no_protocol_gives_1_and_the_overhead_is_against_the_reference_at_its_mfu():
    """The slowdown is a hardware multiple over the reference training algorithm at `MFU₀` (0.4), at least 1, so the
    adversary no protocol encumbers is at 1, and remote under PoUW as deployed on the H200 about 1.04; remote with the
    floor credit at the card table's clock is above it, and onsite far above. Under reference efficiency free compute
    counts at `MFU₀`, so each is its padded size ratio (3.04 remote); without it the monitored terms shrink by 2.5 and
    the dark leaves don't (1.22 remote, onsite still above 10)."""
    u = evaluate_named(REG, "unmonitored-h200-kimi")
    assert u.lo == u.attack == 1.0
    remote, onsite = evaluate_named(REG, "reference-remote-h200"), evaluate_named(REG, "reference-onsite-h200")
    deployed = evaluate_named(REG, "reference-remote-h200-pouw")
    assert remote.scenario.mfu == onsite.scenario.mfu == pytest.approx(0.4) and deployed.lo == pytest.approx(1.04, abs=5e-3)
    assert remote.lo == pytest.approx(3.04, abs=5e-3) == remote.families[0].combined.padded
    assert onsite.lo == pytest.approx(onsite.families[0].combined.padded) and onsite.lo > 10
    peak = REG.override({"task.reference_efficiency": False})
    remote, onsite = evaluate_named(peak, "reference-remote-h200"), evaluate_named(peak, "reference-onsite-h200")
    assert remote.lo == pytest.approx(1.22, abs=5e-3) and remote.families[0].combined.padded == pytest.approx(3.04, abs=0.01)
    assert onsite.lo > 10 and onsite.lo < onsite.families[0].combined.padded


def test_the_time_to_the_target_is_the_overhead_times_the_references_time():
    r = evaluate_named(REG, "reference-onsite-h200")
    s = r.scenario
    f, loc = s.context.fleet, s.context.location
    total = f.count * loc.phi / (1 - f.eps_dark)
    assert s.total_phi == pytest.approx(total)
    assert s.seconds(r.lo) == pytest.approx(r.lo * s.context.units.gates(s.run.target(s.model)) / (s.mfu * total))
