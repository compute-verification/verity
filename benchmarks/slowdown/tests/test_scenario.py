import itertools
import math
import sys
from dataclasses import replace

import pytest
from slowdown import analysis, classes, lean
from slowdown.attacks import ATTACKS
from slowdown.classes import CLASSES
from slowdown.envelopes import KINDS, PARTS, ROUTES
from slowdown.params import Registry
from slowdown.scenario import build, evaluate, evaluate_named, reads
from slowdown.theorem import multiple_bound

REG = Registry.load()
#: every bound whose premises aren't refuted at a block admitted, a `draft:` statement among its own citations or not
DRAFTS_TOO = REG.override({"bounds.lock_only": False})
#: the scenarios a draft bound sets when admitted, each at its own fleet and period: the lower bound and the bound that
#: sets it with the draft bounds admitted, then with only the bounds a lock states
DRAFT_SET = {
    "reference-onsite-h200": (886.210, "fanout-priced", 3.04022, "min-cut"),
    "reference-onsite-h200-per-device": (606.259, "fanout-priced", 3.04022, "min-cut"),
    "reference-onsite-h200-elementary": (1.20381, "fanout-priced", 1, "cover"),
    "onsite-w63": (982.293, "fanout-priced", 3.04022, "min-cut"),
    "onsite-h200-kimi": (892.212, "fanout-priced", 3.04022, "min-cut"),
    "onsite-h100-kimi": (892.212, "fanout-priced", 3.04022, "min-cut"),
    "onsite-b200-kimi": (865.501, "fanout-priced", 3.04022, "min-cut"),
    "onsite-h200-kimi-5gb": (728.309, "fanout-priced", 3.04022, "min-cut"),
    "onsite-h100-kimi-5gb": (728.309, "fanout-priced", 3.04022, "min-cut"),
    "onsite-h200-rack4-kimi": (737.832, "fanout-priced", 3.04022, "min-cut"),
    "onsite-h200-frontier": (798.713, "fanout-priced", 3.03946, "threaded"),
    "onsite-h200-llama405b": (479.492, "fanout-priced", 2.34097, "min-cut"),
    "onsite-no-pouw-h200-kimi": (892.212, "fanout-priced", 1, "cover"),
    "onsite-h200-kimi-rl": (3.03935, "fanout-priced", 3.03628, "min-cut"),
    "onsite-h200-kimi-sft": (3.03944, "output-premise", 2.95655, "threaded"),
    "onsite-h200-kimi-rlvr": (1.11, "covering", 1, "covering"),
    "remote-h200-frontier": (3.04005, "output-premise", 3.03946, "threaded"),
    "remote-h200-kimi-rl": (3.03935, "fanout-priced", 3.03628, "min-cut"),
}


@pytest.mark.parametrize("name", REG.names("scenario"))
def test_every_scenario_evaluates_below_its_best_attack_unless_a_premise_is_named_false(name):
    """The lower bound is at most the best known attack, or a bound above the attack is reported as a contradiction.
    It never stands on a refuted premise, and is at least every bound alone that a lock states and whose premises
    aren't refuted where it binds; with the draft bounds admitted, at least every one of those too."""
    locked, every = evaluate_named(REG, name), evaluate_named(DRAFTS_TOO, name)
    for r in (locked, every):
        assert r.families
        for f in r.families:
            assert f.lo <= f.attack * (1 + 1e-9) or f.contradictions
            assert f.combined.premise != "refuted"
            assert all(e.slowdown <= f.lo * (1 + 1e-9) for e in f.bounds
                       if e.parts and e.premise != "refuted" and e.argument not in f.combined.drafts)
    assert locked.lo <= every.lo * (1 + 1e-9) and not any(f.combined.drafts for f in every.families)


def test_only_a_bound_a_lock_states_sets_the_lower_bound_and_a_draft_one_is_still_reported_alone(monkeypatch):
    """A class one of whose own citations is a `draft:` statement bounds no block, whatever its size: onsite H200's
    lower bound is exactly the one without the four such classes, set by min-cut at the declared block, so it is
    remote's, 2.99, 3.03 and 3.04 a week, a month and a year over one datacenter and over the world. Each of the four
    is still reported alone, labeled draft, at the slowdown it had. A cut's premises and the structure's are not a
    bound's own: the lower bound still cites `draft:no-straddle`. With `bounds.lock_only` off the four are admitted
    and the lower bound is what fanout-priced gave, 886 at every period and fleet. Asked for alone
    (`periods.py --bounds`), a draft bound sets no row, and the header says so with the setting that admits it."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    import periods
    import run

    name, remote = "reference-onsite-h200", "reference-remote-h200"
    drafts = {a.name for a in CLASSES.values() if any(lean.status(c) == "draft" for c in a.cites)}
    assert drafts == {"fanout-priced", "fanout-split", "accumulator", "output-premise"}
    f, was = (evaluate_named(reg, name).families[0] for reg in (REG, DRAFTS_TOO))
    s = build(REG, name)
    assert s.lock_only and not build(DRAFTS_TOO, name).lock_only
    without = evaluate(replace(s, bounds=tuple(a for a in s.bounds if a.name not in drafts))).families[0]
    assert set(f.combined.drafts) == drafts and replace(f.combined, drafts=()) == without.combined
    assert f.bounds == was.bounds and {e.status for e in f.bounds if e.argument in drafts} == {"draft"}
    assert run.setter(f).argument == "min-cut" and run.setter(was).argument == "fanout-priced"
    assert "draft:no-straddle" in f.combined.cites and not any(c.startswith("draft:E1") for c in f.combined.cites)
    assert was.combined.drafts == () and any(c.startswith("draft:E1") for c in was.combined.cites)
    record = {"week": 2.99, "month": 3.03, "year": 3.04}
    for (fleet, _), period in itertools.product(periods.FLEETS, periods.PERIODS):
        at = periods.choices(name, fleet, period) | periods.choices(remote, fleet, period)
        lo = evaluate_named(REG.override(at), name).lo
        assert lo == pytest.approx(record[period], abs=5e-3)
        assert lo == pytest.approx(evaluate_named(REG.override(at), remote).lo, rel=1e-9)
        assert evaluate_named(DRAFTS_TOO.override(at), name).lo == pytest.approx(886, abs=0.5)
    year = (("world-h200", "the world"),), ("year",)
    for reg, slowdown, setter, said in ((REG, 1, None, True), (DRAFTS_TOO, 886, "fanout-priced", False)):
        alone = periods.only(reg, [name], ["fanout-priced"])
        rows = periods.evaluate(alone, [name], *year)
        assert rows[0]["slowdown"] == pytest.approx(slowdown, abs=0.5) and rows[0]["parts"].get("setter") == setter
        header = periods.alone_md(alone, rows, ["fanout-priced"])[0]
        assert ("own citations of `fanout-priced`, which so bounds no block" in header) is said


def test_the_draft_bounds_set_these_scenarios_and_no_other(monkeypatch):
    """Leaving the draft bounds out lowers the scenarios `DRAFT_SET` names and moves no other: the onsite scenarios a
    draft bound set at the server block, and five it set at the declared block, two of them remote. With
    `bounds.lock_only` off each has the lower bound the draft bounds gave it."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    import run

    def at(reg, name):
        r = evaluate_named(reg, name)
        sets = run.setter(run.picked(r))
        return r.lo, sets.argument if sets else None

    for name in REG.names("scenario"):
        (was, set_was), (lo, sets) = at(DRAFTS_TOO, name), at(REG, name)
        if name in DRAFT_SET:
            assert lo < was and (pytest.approx(was, rel=1e-5), set_was, pytest.approx(lo, rel=1e-5), sets) == DRAFT_SET[name]
        else:
            assert lo == pytest.approx(was, rel=1e-12) and sets == set_was
    remote = {n for n in DRAFT_SET if not REG[f"scenario.{n}.regime"].startswith("onsite")}
    assert remote == {"remote-h200-frontier", "remote-h200-kimi-rl"}


def test_a_bound_refuted_at_a_block_does_not_bound_it_and_is_still_reported_alone():
    """With the draft bounds admitted and E1's activation rate refuted past its credible 1.1·10⁹ bits, fanout-priced on
    its own binds onsite H200's serving cut (`k_S` past that) at a standing of refuted, its padded size ratio about 290
    times the lower bound's. Every bound at once takes no cap from it at that block, its standing doesn't reach the
    lower bound's, and the lower bound is still at least every bound alone whose premises aren't refuted."""
    premise = "premise.e1-activation-rate"
    f = evaluate_named(DRAFTS_TOO.override({f"{premise}.beyond": "refuted"}), "reference-onsite-h200").families[0]
    fp = next(e for e in f.bounds if e.argument == "fanout-priced")
    assert fp.premise == "refuted" and {s.cut for s in fp.parts.serving.values()} == {"serving"}
    assert f.combined.padded < fp.padded / 10 and f.combined.premise != "refuted"
    assert all(e.slowdown <= f.lo * (1 + 1e-9) for e in f.bounds if e.parts and e.premise != "refuted")


def test_a_bound_whose_premise_is_refuted_everywhere_never_sets_the_lower_bound():
    """With the draft bounds admitted, refuting E1's activation rate at every block size takes fanout-priced and
    fanout-split (which cite it) out of the lower bound: onsite H200's is then exactly the lower bound without them,
    and both are still reported alone, tagged refuted, at the overhead they had."""
    name, premise = "reference-onsite-h200", "premise.e1-activation-rate"
    base = evaluate_named(DRAFTS_TOO, name).families[0]
    refuted = DRAFTS_TOO.override({f"{premise}.standing": "refuted", f"{premise}.beyond": "refuted"})
    f = evaluate_named(refuted, name).families[0]
    cite = REG[f"{premise}.lean"]
    citing = {a.name for a in CLASSES.values() if cite in a.cites}
    assert citing == {"fanout-priced", "fanout-split"}
    s = build(DRAFTS_TOO, name)
    without = evaluate(replace(s, bounds=tuple(a for a in s.bounds if a.name not in citing))).families[0]
    assert set(f.combined.excluded) == citing and replace(f.combined, excluded=()) == without.combined
    assert f.lo < base.lo / 10 and f.combined.premise != "refuted"
    alone = {e.argument: e for e in f.bounds}
    assert all(alone[a].premise == "refuted" and alone[a].slowdown == next(e.slowdown for e in base.bounds if e.argument == a)
               for a in citing)
    assert max(e.slowdown for e in f.bounds if e.parts) == pytest.approx(base.lo)


def test_s_times_and_gates_per_flop_come_from_one_source():
    """The bound holds a gate at `u/w = ω_max 2G / (κ s_×)` of the reference's weight, so the FLOP model's `G` beside
    a Boolean design's measured `s_×` (681 gates a bf16 product) scales every weight counted at `u` (the unregistered
    leaves and the uncertified routes, nearly all the rate of onsite H200's lower bound, `Slowdown.rates`) by `681 w`,
    about 0.82, with nothing to say so. Building refuses the mix;
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
    at_u = sum(t.share for t in analysis.attribution(base) if t.name == "unregistered" or t.name.split(".")[0] in ROUTES)
    assert 0.5 < at_u <= 1
    assert one.lo / base.lo == pytest.approx(1 / (at_u / (681 * w) + 1 - at_u), rel=5e-3)
    assert build(REG.override({"units.basis": design}), "reference-onsite-h200-elementary").structure.s_times == 1


def test_unmonitored_is_the_algorithm_alone():
    """The bound counts every gate at capacity, so its padded size ratio is `μ`; a construction trains at its MFU on a
    circuit padded to capacity, so the best attack's is `μ / MFU`. The overhead is against the reference at that MFU, so
    no protocol gives `max(1, μ)` for the attack, 1 for the reference's own family, and the reference's time."""
    r = evaluate_named(REG, "unmonitored-h200-kimi")
    f0 = r.scenario.context.fleet
    mfu = f0.eps_unreg * f0.unreg_mfu + (1 - f0.eps_unreg) * f0.mfu_train
    assert r.scenario.mfu == pytest.approx(mfu)
    for f in r.families:
        mu = f.profile.efficiency
        assert math.isclose(f.combined.padded, mu) and math.isclose(f.best_attack.padded, mu / mfu)
        assert f.lo == pytest.approx(multiple_bound(mfu, f0.eps_unreg / mu, (1 - f0.eps_unreg) / mu))
        assert f.attack == pytest.approx(max(1.0, mu))
    s = r.scenario
    assert r.lo == r.attack == 1.0 and math.isclose(s.seconds(r.attack), s.reference / (mfu * s.total_phi))


def test_onsite_adds_a_cut_so_its_lower_bound_is_never_below_remotes():
    assert evaluate_named(REG, "onsite-h200-kimi").lo >= evaluate_named(REG, "remote-h200-kimi").lo * (1 - 1e-9)


def test_every_lean_citation_is_in_a_lock():
    cites = [c for a in (*CLASSES.values(), *ATTACKS.values()) for c in a.cites]
    cites += [c for kind in (*KINDS.values(), *PARTS.values())
              for c in getattr(kind, "CITES", ()) + getattr(kind, "PHYSICAL", ())]
    absent = sorted({c for c in cites if lean.status(c) == "absent"})
    assert not absent, f"no lock lists {absent}"


@pytest.mark.parametrize("name", REG.names("scenario"))
def test_admitting_weaker_statuses_never_lowers_the_guarantee(name):
    r = evaluate_named(REG, name)
    slowdowns = [analysis.upto(s)(r) for s in ("proved", "conditional", "draft", "absent")]
    assert slowdowns == sorted(slowdowns) and math.isclose(slowdowns[-1], r.lo)
    assert all(0 <= f.combined.parts.monitored <= 1 for f in r.families)


def test_a_scenario_reads_its_own_tables_and_not_the_others():
    keys = reads(REG, "reference-onsite-h200")
    assert "fleet.world-h200.eps_unreg" in keys and "regime.onsite.k_wipe" in keys and "units.gates_per_flop" in keys
    assert not any(k.startswith(("fleet.world-nvl72.", "model.kimi-k2.")) for k in keys)


@pytest.mark.parametrize("name", ["reference-remote-h200", "reference-onsite-h200", "reference-onsite-nvl72",
                                  "reference-onsite-h200-per-device"])
def test_the_slowdown_and_the_time_to_the_target_do_not_depend_on_gates_per_flop(name):
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
    is what it was), and every bound that counts uncertified or unregistered gates there cites `draft:no-straddle`. The
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
    adversary no protocol encumbers is at 1, and remote under PoUW as deployed on the H200 about 1.05 (1.04 with the
    H200's tensor at 1,980 MHz); remote with the floor credit at the card table's clock is above it, and onsite is
    remote's, the wipe's serving cut adding nothing under the bounds a lock states. Under reference efficiency free
    compute counts at `MFU₀`, so each is its padded size ratio (3.04); without it the monitored terms shrink by 2.5
    and the unregistered leaves don't (1.22). With the draft bounds admitted onsite is far above, and still above 10
    without reference efficiency."""
    u = evaluate_named(REG, "unmonitored-h200-kimi")
    assert u.lo == u.attack == 1.0
    remote, onsite = evaluate_named(REG, "reference-remote-h200"), evaluate_named(REG, "reference-onsite-h200")
    deployed = evaluate_named(REG, "reference-remote-h200-pouw")
    assert remote.scenario.mfu == onsite.scenario.mfu == pytest.approx(0.4) and deployed.lo == pytest.approx(1.048, abs=5e-3)
    assert remote.lo == pytest.approx(3.04, abs=5e-3) == remote.families[0].combined.padded
    assert onsite.lo == pytest.approx(onsite.families[0].combined.padded) == pytest.approx(remote.lo, rel=1e-9)
    drafts = evaluate_named(DRAFTS_TOO, "reference-onsite-h200")
    assert drafts.lo == pytest.approx(drafts.families[0].combined.padded) and drafts.lo > 10
    without = {"task.reference_efficiency": False}
    peak = REG.override(without)
    remote, onsite = evaluate_named(peak, "reference-remote-h200"), evaluate_named(peak, "reference-onsite-h200")
    assert remote.lo == pytest.approx(1.22, abs=5e-3) and remote.families[0].combined.padded == pytest.approx(3.04, abs=0.01)
    assert onsite.lo == pytest.approx(remote.lo, rel=1e-9) and onsite.lo < onsite.families[0].combined.padded
    drafts = evaluate_named(DRAFTS_TOO.override(without), "reference-onsite-h200")
    assert drafts.lo > 10 and drafts.lo < drafts.families[0].combined.padded


def test_the_time_to_the_target_is_the_overhead_times_the_references_time():
    r = evaluate_named(REG, "reference-onsite-h200")
    s = r.scenario
    f, loc = s.context.fleet, s.context.location
    total = f.count * loc.phi / (1 - f.eps_unreg)
    assert s.total_phi == pytest.approx(total)
    assert s.seconds(r.lo) == pytest.approx(r.lo * s.context.units.gates(s.run.target(s.model)) / (s.mfu * total))
