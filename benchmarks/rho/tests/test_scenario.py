import math

import pytest
from rho import analysis, classes, lean
from rho.attacks import ATTACKS
from rho.classes import CLASSES
from rho.envelopes import KINDS, PARTS
from rho.params import Registry
from rho.scenario import build, evaluate_named, reads

REG = Registry.load()


@pytest.mark.parametrize("name", REG.names("scenario"))
def test_every_scenario_evaluates_and_its_bracket_is_ordered_unless_a_premise_is_named_false(name):
    r = evaluate_named(REG, name)
    assert r.families
    for f in r.families:
        assert f.lo <= f.hi * (1 + 1e-9) or f.contradictions


def test_unmonitored_is_the_algorithm_alone():
    """The bound counts every gate at capacity, so `lo = μ`; a construction trains at its MFU on a circuit padded to
    capacity, so `hi = μ / MFU` and its time is the algorithm's at that MFU."""
    r = evaluate_named(REG, "unmonitored-h200-kimi")
    f0 = r.scenario.context.fleet
    mfu = f0.eps_dark * f0.dark_mfu + (1 - f0.eps_dark) * f0.mfu_train
    assert all(math.isclose(f.lo, f.profile.efficiency) and math.isclose(f.hi, f.profile.efficiency / mfu)
               for f in r.families)
    s = r.scenario
    assert math.isclose(s.seconds(r.hi), min(f.profile.efficiency for f in r.families) * s.reference
                        / (mfu * s.total_phi))


def test_onsite_adds_a_cut_so_it_never_slows_less_than_remote():
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
    assert all(0 <= f.combined.slowdown.monitored <= 1 for f in r.families)


def test_a_scenario_reads_its_own_tables_and_not_the_others():
    keys = reads(REG, "reference-onsite-h200")
    assert "fleet.world-h200.eps_dark" in keys and "regime.onsite.k_wipe" in keys and "units.gates_per_flop" in keys
    assert not any(k.startswith(("fleet.world-nvl72.", "model.kimi-k2.")) for k in keys)


@pytest.mark.parametrize("name", ["reference-remote-h200", "reference-onsite-h200", "reference-onsite-nvl72",
                                  "reference-onsite-h200-per-device"])
def test_rho_and_the_time_to_the_target_do_not_depend_on_gates_per_flop(name):
    """Every FLOP-model input becomes gates through the one `units.gates_per_flop`, and the bound compares gates with
    gates, so the conversion cancels: at its low and high ends ρ's bracket and its times are today's."""
    def numbers(g: float) -> tuple[float, ...]:
        r = evaluate_named(REG.override({"units.gates_per_flop": g}), name)
        return r.lo, r.hi, r.scenario.seconds(r.lo), r.scenario.seconds(r.hi)

    p = REG.param("units.gates_per_flop")
    base = numbers(p.value)
    for g in (p.low, p.high):
        assert numbers(g) == pytest.approx(base, rel=1e-9)
    assert build(REG.override({"units.gates_per_flop": p.high}), name).context.c == pytest.approx(
        build(REG, name).context.c * p.high / p.value)


def test_the_time_to_the_target_is_rho_reference_gates_over_the_inventorys_capacity():
    r = evaluate_named(REG, "reference-onsite-h200")
    s = r.scenario
    f, loc = s.context.fleet, s.context.location
    total = f.count * loc.phi / (1 - f.eps_dark)
    assert s.total_phi == pytest.approx(total)
    assert s.seconds(r.lo) == pytest.approx(r.lo * s.context.units.gates(s.run.target(s.model)) / total)
