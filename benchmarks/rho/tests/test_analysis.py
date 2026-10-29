import json
import math
import sys

import pytest
from rho import analysis, figures
from rho.analysis import LO, crossing
from rho.params import Param, Registry
from rho.scenario import build

REG = Registry.load()
NAMES = REG.names("scenario")


def synthetic(monkeypatch, overhead):
    """A registry of four ranged parameters `a`, `b`, `c`, `d` (each 1.0 today), whose overhead is `overhead(values)`."""
    reg = Registry({k: Param(k, 1.0, low=lo, high=hi) for k, lo, hi in
                    (("a", 0.1, 10.0), ("b", 0.5, 2.0), ("c", 0.0, 2.0), ("d", 0.5, 2.0))})

    def at(registry, scenario, values=None, metric=LO):
        return overhead({"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0} | dict(values or {}))

    monkeypatch.setattr(analysis, "at", at)
    monkeypatch.setattr(analysis, "reads", lambda registry, scenario: frozenset(p.key for p in registry.params()))
    return reg


def monotone(v):
    """100 today; `a` swings 1 decade, `b` 1.2, `c` 0.48, and `d` not at all."""
    return 100 * v["a"] ** -0.5 * v["b"] ** 2 * (1 + v["c"]) / 2


def test_crossing_bisects_a_monotone_function_geometrically_or_linearly():
    assert crossing(lambda x: x * x, 2.0, 1e-3, 1e3, rtol=1e-12) == pytest.approx(math.sqrt(2), rel=1e-10)
    assert crossing(lambda x: 1 - x, 0.25, -1.0, 1.0, rtol=1e-12) == pytest.approx(0.75, rel=1e-10)
    assert crossing(lambda x: 1 / x, 0.5, 4.0, 1.0, rtol=1e-12) == pytest.approx(2.0, rel=1e-10)
    assert crossing(lambda x: math.inf if x < 3 else 1.0, 2.0, 1.0, 10.0) == pytest.approx(3.0, rel=1e-3)
    assert crossing(lambda x: x, 10.0, 1.0, 5.0) is None
    assert crossing(lambda x: x, 1.0, 1.0, 5.0) == 1.0


def test_threshold_finds_where_the_synthetic_metric_crosses_the_target(monkeypatch):
    reg = synthetic(monkeypatch, monotone)
    assert analysis.threshold(reg, "s", "a", 60.0, 1.0, 10.0) == pytest.approx((100 / 60) ** 2, rel=1e-3)
    assert analysis.threshold(reg, "s", "b", 60.0, 0.5, 2.0, given={"a": 4.0}) == pytest.approx(math.sqrt(1.2),
                                                                                                 rel=1e-3)
    assert analysis.threshold(reg, "s", "d", 60.0, 0.5, 2.0) is None


def test_tornado_orders_by_decades_of_swing_and_drops_what_does_not_move(monkeypatch):
    reg = synthetic(monkeypatch, monotone)
    swings = analysis.tornado(reg, "s")
    assert [s.key for s in swings] == ["b", "a", "c"]
    assert [s.size for s in swings] == pytest.approx([math.log10(16), 1.0, math.log10(3)])
    assert [s.worse for s in swings] == [(0.5, 25.0), (10.0, pytest.approx(100 / math.sqrt(10))), (0.0, 50.0)]


def test_dangers_and_rescues_find_each_crossing_on_the_side_that_reaches_it(monkeypatch):
    reg = synthetic(monkeypatch, monotone)
    found = {d.key: d for d in analysis.dangers(reg, "s", 60.0)}
    assert set(found) == {"a", "b", "c"}
    assert found["a"].crossing == pytest.approx((100 / 60) ** 2, rel=1e-3)
    assert found["b"].crossing == pytest.approx(math.sqrt(0.6), rel=1e-3)
    assert found["c"].crossing == pytest.approx(0.2, rel=1e-3)
    lifted = {d.key: d.crossing for d in analysis.rescues(reg, "s", 200.0)}
    assert lifted == {"a": pytest.approx(0.25, rel=1e-3), "b": pytest.approx(math.sqrt(2), rel=1e-3)}
    assert analysis.dangers(reg, "s", 1000.0) == [] and analysis.rescues(reg, "s", 50.0) == []


def test_the_corner_takes_every_parameter_to_its_worse_end(monkeypatch):
    reg = synthetic(monkeypatch, monotone)
    c = analysis.corner(reg, "s")
    assert c.values == {"b": 0.5, "a": 10.0, "c": 0.0}
    assert c.rho == c.joint == pytest.approx(100 / math.sqrt(10) * 0.25 * 0.5)
    assert [k for k, _ in c.path] == ["b", "a", "c"]


def test_the_corner_never_rises_above_the_base_when_an_interaction_makes_the_joint_corner_higher(monkeypatch):
    def interacting(v):
        return 1000.0 if v["a"] > 5 and v["b"] < 0.6 else monotone(v)

    reg = synthetic(monkeypatch, interacting)
    c = analysis.corner(reg, "s")
    assert c.joint == 1000.0 and c.base == 100.0
    assert c.rho == pytest.approx(12.5) and c.values == {"b": 0.5, "c": 0.0}


def test_frontier_traces_the_target_curve_in_a_plane(monkeypatch):
    reg = synthetic(monkeypatch, monotone)
    front = analysis.frontier(reg, "s", "a", [0.5, 1.0, 4.0], "b", 0.5, 2.0, 100.0)
    assert [y for _, y in front] == [pytest.approx(a ** 0.25, rel=1e-3) for a in (0.5, 1.0, 4.0)]
    assert analysis.frontier(reg, "s", "a", [1.0], "d", 0.5, 2.0, 150.0) == [(1.0, None)]


def test_margins_find_the_crossings_on_either_side_within_the_window(monkeypatch):
    reg = synthetic(monkeypatch, monotone)
    m = {x.key: x for x in analysis.margins(reg, "s", [60.0, 200.0])}
    assert m["a"].crossings[60.0] == (None, pytest.approx((100 / 60) ** 2, rel=1e-2))
    assert m["a"].crossings[200.0] == (pytest.approx(0.25, rel=1e-2), None)
    assert m["a"].window == pytest.approx((0.001, 1000.0))
    assert m["c"].window[0] == 0.0
    assert analysis.window(Param("share", 0.5, low=0.01, high=0.6)) == pytest.approx((0.01 / 60, 1.0))


@pytest.mark.parametrize("name", NAMES)
def test_every_scenario_s_tornado_is_sorted_and_its_corner_is_never_above_the_base(name):
    swings = analysis.tornado(REG, name)
    assert [s.size for s in swings] == sorted((s.size for s in swings), reverse=True)
    for s in swings[:3]:
        assert s.at_low == analysis.at(REG, name, {s.key: s.low})
    c = analysis.corner(REG, name, swings=swings)
    assert c.rho <= c.base * (1 + 1e-12)
    assert c.rho <= min((s.worse[1] for s in swings), default=c.base) * (1 + 1e-12)


@pytest.mark.parametrize("name", NAMES)
def test_threshold_on_a_scenario_lands_where_the_scenario_crosses(name):
    for s in analysis.tornado(REG, name)[:2]:
        if not (math.isfinite(s.at_low) and math.isfinite(s.at_high)):
            continue
        target = math.sqrt(s.at_low * s.at_high)
        x = analysis.threshold(REG, name, s.key, target, s.low, s.high, rtol=1e-6)
        assert x is not None
        if analysis.at(REG, name, {s.key: x}) != pytest.approx(target, rel=1e-3):
            # the overhead jumps over the target at x (a premise share reaching its end, as `classes.expensive_share` at
            # the last layer): the crossing is the jump
            d = 1e-5 * (x if s.low > 0 else max(abs(s.low), abs(s.high)))
            near = [analysis.at(REG, name, {s.key: v}) for v in (x - d, x + d)]
            assert min(near) <= target <= max(near), (s.key, x, near)


@pytest.mark.parametrize("name", NAMES)
def test_lo_grows_with_the_statuses_admitted_and_admitting_all_is_lo(name):
    los = [analysis.at(REG, name, None, analysis.upto(st)) for st in analysis.STATUSES]
    assert los == sorted(los)
    assert los[-1] == pytest.approx(analysis.at(REG, name))
    with pytest.raises(ValueError, match="not one of"):
        analysis.upto("measured")


@pytest.mark.parametrize("name", NAMES)
def test_attribution_splits_the_adversary_s_rate_and_each_ceiling_caps_rho(name):
    r = analysis.result(REG, name)
    terms = analysis.attribution(r)
    assert sum(t.share for t in terms) == pytest.approx(1.0)
    assert terms == sorted(terms, key=lambda t: -t.part)
    fam = min(r.families, key=lambda f: f.lo)
    assert sum(t.part for t in terms) == pytest.approx(fam.profile.efficiency / r.lo)
    assert all(t.ceiling >= r.lo * (1 - 1e-9) for t in terms)
    assert any(t.name == "dark" for t in terms) == (r.scenario.context.fleet.eps_dark > 0)


def test_results_are_cached_per_registry_and_override():
    name = NAMES[0]
    assert analysis.result(REG, name) is analysis.result(REG, name)
    assert analysis.result(REG, name, {}) is analysis.result(REG, name)
    other = REG.override({})
    assert analysis.result(other, name) is not analysis.result(REG, name)


def test_run_writes_markdown_json_and_figures_into_out(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "path", list(sys.path))
    import run

    drawn = []
    monkeypatch.setattr(figures, "render", lambda reg, out, names, targets: drawn.append((out, targets)) or {})
    monkeypatch.setenv("OUT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["run.py", "--target", "10", "--scenario", NAMES[0]])
    run.main()
    text = capsys.readouterr().out
    assert text.startswith("# The overhead") and "## Where the overhead falls below target" in text
    assert "**Corner:**" in text
    assert (tmp_path / "rho.md").read_text() == text
    record = json.loads((tmp_path / "rho.json").read_text(), parse_constant=lambda c: pytest.fail(f"JSON has {c}"))
    assert set(record["scenarios"]) == {NAMES[0]}
    assert {"corner", "overhead_by_status", "attribution", "dangers", "rescues"} <= set(record["scenarios"][NAMES[0]])
    assert drawn == [(tmp_path / "figures", [10.0])]


def test_run_prints_one_overhead_with_its_comparator_and_the_attack_in_a_table_of_its_own(tmp_path, monkeypatch,
capsys):
    """One number a scenario, the lower bound, flagged vacuous against `μ / MFU`, with the inference overhead beside
    it (not modeled); the best known attack only in the attacks table, never as an interval's end."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    import run

    monkeypatch.setattr(figures, "render", lambda reg, out, names, targets: {})
    monkeypatch.setenv("OUT", str(tmp_path))
    # nothing credited (residual 1) keeps remote below the comparator whatever the other inputs are
    monkeypatch.setattr(sys, "argv", ["run.py", "--target", "10", "--scenario", "reference-remote-h200",
                                      "--set", "regime.remote.residual=1"])
    run.main()
    text = capsys.readouterr().out
    summary, rest = text.split("## Attacks", 1)
    attacks = rest.split("\n## ", 1)[0]
    row = next(line for line in summary.splitlines() if line.startswith("| reference-remote-h200 |"))
    assert "(vacuous)** |" in row and "| 2.5 | not modeled |" in row and "[" not in row
    record = json.loads((tmp_path / "rho.json").read_text())["scenarios"]["reference-remote-h200"]
    attack = record["attack"]
    assert f"| reference-remote-h200 | {attack['argument']} | {attack['family']} | {attack['status']} | " in attacks
    assert run.fmt(attack["rho"]) not in row.split("|")[2]
    assert "With no protocol the adversary sits at **overhead 2.5**" in text and "**vacuous**" in text
    assert record["vacuous"] and record["unmonitored"] == pytest.approx(2.5)
    assert record["overhead"] < record["unmonitored"] <= attack["rho"]
    assert record["inference_overhead"]["value"] is None and record["inference_overhead"]["source"].startswith(
        "not modeled")
    assert "hi" not in record and "lo" not in record


def test_run_prints_the_certified_ingress_a_server_declares(tmp_path, monkeypatch, capsys):
    """The per-server bound at W63 (`onsite-w63`): 91.1 kbit/s a server over a 32-bucket link-window, intake 0.0282 at
    the reference policy, labelled the honest-δ form with a faulty ingress unit's 356,592 bits excluded (and 0 when
    every ingress unit is sampled), line rate beside it; the reference itself at `intake · α Φ`, 3.23 Mbit/s."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    import run

    monkeypatch.setattr(figures, "render", lambda reg, out, names, targets: {})
    monkeypatch.setenv("OUT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["run.py", "--target", "10", "--scenario", "onsite-w63",
                                      "--scenario", "reference-remote-h200"])
    run.main()
    text = capsys.readouterr().out
    assert ("**Certified ingress:** 9.11e+04 bit/s into a location, intake 0.0282 × α Φ: 1 × each server's declared "
            "inputs, r_in = 9.11e+04 bit/s (r_out = 2.74e+03 bit/s out) over a 32-bucket link-window, at the warden "
            "form's public parameters, the honest-δ form of draft:per-server-ingress's bound (a faulty ingress unit's "
            "3.57e+05 bits are excluded, and 0 when every ingress unit is drawn)") in text
    assert "line rate, 4.8e+12 bit/s, is its upper end until Lean proves that bound" in text
    assert text.count("Certified") == 1
    record = json.loads((tmp_path / "rho.json").read_text())["scenarios"]
    assert record["reference-remote-h200"]["certified"] is None
    certified = record["onsite-w63"]["certified"]
    assert certified["r_in"] == pytest.approx(91_075) and certified["window_buckets"] == 32
    assert certified["faulty_unit_bits"] == 356_592
    reference = run.certified(build(REG, "reference-onsite-h200"))
    assert not reference["declared"] and reference["intake"] == 1
    assert run.certified_md(reference)[0] == ("**Certified ingress:** 3.23e+06 bit/s into a location, intake 1 × α Φ: "
                                              "`intake · α Φ`, a server's ingress only its declared inputs "
                                              "(draft:declared-ingress); plus 1.86e+03 bit/s of timing and side channels.")


def test_periods_writes_each_fleet_and_period_as_the_scenario_with_those_choices(tmp_path, monkeypatch, capsys):
    """Remote: the serving block is one server over the period, one a server; each row's overhead is the scenario's
    with `fleet` and `horizon` set."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    import periods

    name = "reference-remote-h200"
    monkeypatch.setattr(sys, "argv", ["periods.py", "--scenario", name, "--out", str(tmp_path)])
    periods.main()
    text = capsys.readouterr().out
    assert all(f"\n## {s}\n" in text for s in ("The lower bound", "Its parts", "The premises, ranked"))
    assert (tmp_path / "periods.md").read_text() == text
    rows = json.loads((tmp_path / "periods.json").read_text(), parse_constant=lambda c: pytest.fail(f"JSON has {c}"))
    rows = rows["rows"]
    assert [(x["fleet"], x["period"]) for x in rows] == [(f, p) for f, _ in periods.FLEETS for p in periods.PERIODS]
    for x in rows:
        at = REG.override(periods.choices(name, x["fleet"], x["period"]))
        assert x["overhead"] == analysis.result(at, name).lo
        (block,) = x["scaled"]["blocks"]
        assert block["count"] == REG[f"fleet.{x['fleet']}.count"]
        assert block["seconds"] == pytest.approx(x["period_s"]) == pytest.approx(REG[f"horizon.{x['period']}.seconds"])
        assert x["range"][0] <= x["overhead"] <= x["range"][1] and x["corner"] <= x["overhead"]
        assert [r["decades"] for r in x["ranking"]] == sorted((r["decades"] for r in x["ranking"]), reverse=True)
        assert any(r["what"] == "draft:no-straddle" for r in x["ranking"])
