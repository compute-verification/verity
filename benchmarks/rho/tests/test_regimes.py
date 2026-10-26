import importlib.util
import math
from pathlib import Path

import pytest
from rho import hardware, lean, regimes
from rho.params import Param, Registry
from rho.regimes import Context, Onsite, Remote
from rho.scenario import evaluate_named
from rho.theorem import Budget, slowdown

ROOT = Path(__file__).resolve().parents[3]
REG = Registry.load()

_spec = importlib.util.spec_from_file_location("tiers", ROOT / "benchmarks" / "nci" / "tiers.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

#: our regime field: tiers' DEFAULTS key
TIERS_KEYS = {"residual": "residual", "clock_share": "clock_share", "eps_w": "eps_w", "eps_agg": "eps_agg",
              "identities": "identities", "lag_s": "lag_s", "proof_share": "share", "overhead": "overhead",
              "replay": "replay", "step_flop": "step_flop", "p_evade": "p_evade", "luck": "luck",
              "step_bits": "step_bits", "floor_steps": "floor_steps", "chord_kappa": "kappa"}


def context(fleet: str, registry: Registry = REG, seconds: float | None = None) -> Context:
    f = hardware.fleet(registry, fleet)
    h = hardware.horizon(registry, "year") if seconds is None else hardware.Horizon("given", seconds)
    return Context(f, hardware.location(registry, f.location), regimes.policy(registry, "kimi-k2-1m-raw"), h)


def tiers_context(registry: Registry = REG) -> Context:
    return context("tiers-reference", registry, T.YEAR)


def tiers_params(ctx: Context) -> dict:
    """tiers' DEFAULTS with the host share its server's CPUs give (0.11796, which tiers rounds to 0.118)."""
    return T.DEFAULTS | {"host_share": ctx.location.host_ops / ctx.location.phi}


def test_the_remote_regimes_numbers_are_tiers_defaults():
    r = regimes.regime(REG, "remote-sampled")
    for ours, theirs in TIERS_KEYS.items():
        assert getattr(r, ours) == pytest.approx(T.DEFAULTS[theirs], rel=1e-12), ours
    ctx = tiers_context()
    assert ctx.location.host_ops / ctx.location.phi == pytest.approx(T.DEFAULTS["host_share"], rel=1e-3)
    assert (ctx.location.phi, ctx.fleet.count, ctx.fleet.mfu_train) == (
        T.DEFAULTS["node_flops"], T.DEFAULTS["nodes"], T.DEFAULTS["mfu"])
    assert (ctx.policy.gamma, ctx.policy.k) == (T.DEFAULTS["Gamma"], T.DEFAULTS["K"])


def test_the_remote_uncertified_share_is_tiers_residual_term():
    ctx = tiers_context()
    p = tiers_params(ctx)
    remote = regimes.regime(REG, "remote")
    assert remote.audited(ctx.location) == pytest.approx(T.audited_residual(p), rel=1e-12)
    assert remote.uncertified(ctx) == pytest.approx(T.evaluate(p, T.TIERS["T1"])["residual"], rel=1e-9)


def test_with_sampled_proofs_it_adds_tiers_escape_and_cheating_shares():
    ctx = tiers_context()
    p = tiers_params(ctx)
    sampled = regimes.regime(REG, "remote-sampled")
    t5 = T.evaluate(p, T.TIERS["T5"])
    free = sampled.free(ctx)
    assert sampled.injected() == pytest.approx(t5["cheat_bits"], rel=1e-12)
    assert sampled.chord == pytest.approx(T.chord(p, t5["cheat_bits"]), rel=1e-7)
    assert math.fsum(v for k, v in free.items() if k != "cheat") == pytest.approx(t5["residual"], rel=1e-9)
    assert free["cheat"] == pytest.approx(t5["cheat"], rel=1e-7)


def test_at_todays_defaults_the_remote_bound_is_one_as_tiers_dampener_is():
    assert evaluate_named(REG, "remote-tiers-kimi").lo == pytest.approx(1.0)
    assert all(T.evaluate(tiers_params(tiers_context()), m)["dampener"] == 1.0 for m in T.TIERS.values())


def test_with_nothing_else_bounding_rho_is_lean_remote_dampener():
    # DampenerRemote with no declared products (ν_T = 0) and χ = 0: ρ = max(1, 1/(s + (1 − s) share / MFU₀))
    reg = REG.override({"regime.remote.residual": 0.01, "regime.remote.clock_share": 0.0, "regime.remote.host": 0.0})
    ctx = tiers_context(reg)
    remote = regimes.regime(reg, "remote")
    part = remote.partition(ctx)
    lean_rho = max(1.0, 1 / (part.dark + (1 - part.dark) * remote.uncertified(ctx) / ctx.fleet.mfu_train))
    assert slowdown(part, lambda b: 0.0, 0.6).rho == pytest.approx(lean_rho, rel=1e-12)
    assert lean_rho > 10


def test_the_declared_blocks_are_the_certified_work_that_ran():
    ctx = context("reference-h200")
    remote = regimes.regime(REG, "remote")
    loc, s = ctx.location, ctx.horizon.seconds
    n = (1 - remote.audited(loc)) * (1 - remote.eps_w) * loc.phi * s / ctx.policy.gamma
    assert remote.blocks(ctx, s).count == pytest.approx(n)


def test_the_layout_gets_only_what_an_abiding_strategy_gets():
    ctx = context("reference-h200")
    sampled = regimes.regime(REG, "remote-sampled")
    layout = sampled.partition(ctx).regions[0].layout
    assert set(layout.free) == set(Remote.ABIDING)
    assert set(sampled.free(ctx)) - set(layout.free) == {"slack", "lag", "escape", "cheat"}
    work = ctx.location.phi * ctx.horizon.seconds * ctx.fleet.mfu_train
    assert layout.free["residual"] == pytest.approx(sampled.residual * work)


def test_a_regime_inherits_its_base_and_overrides_what_differs():
    onsite, five = regimes.regime(REG, "onsite"), regimes.regime(REG, "onsite-5gb")
    assert isinstance(five, Onsite) and five.residual_bits == 4e10 and five.wipe_s == onsite.wipe_s
    assert regimes.regime(REG.override({"regime.remote.residual": 0.1}), "onsite-5gb").residual == 0.1


def test_a_regime_table_with_a_stray_key_or_a_base_loop_is_refused():
    params = {p.key: p for p in REG.params()}
    stray = Registry(params | {"regime.onsite-5gb.residul": Param("regime.onsite-5gb.residul", 0.1)})
    with pytest.raises(ValueError, match="not fields of Onsite"):
        regimes.regime(stray, "onsite-5gb")
    loop = Registry(params | {"regime.remote.base": Param("regime.remote.base", "onsite")})
    with pytest.raises(ValueError, match="loop"):
        regimes.regime(loop, "onsite")
    with pytest.raises(ValueError, match="form"):
        regimes.regime(REG.override({"regime.onsite-links.form": "widths"}), "onsite-links")
    with pytest.raises(ValueError, match="sampled proofs need"):
        Remote("r", 0.3, 0.01, 1e-4, 1, 3600, sampled=True)


def test_the_certified_budget_is_the_wipe_residual_and_what_enters_over_a_wipe():
    ctx = context("reference-h200")
    r = regimes.regime(REG, "onsite")
    loc, pol, w = ctx.location, ctx.policy, r.wipe_s
    state = loc.devices * r.residual_bits + loc.hosts * r.host_residual_bits
    rate = r.intake * pol.k / pol.gamma * loc.phi + loc.hosts * (r.timing + r.side)
    assert r.budget(ctx) == Budget(loc.phi * w, state + rate * w)
    # an H200 server today: 8 x 10 GB, 36.5 GB of host, and α Φ = 4.9e6 bit/s for a day
    assert r.budget(ctx).inputs == pytest.approx(1.36e12, rel=0.01)


def test_the_links_form_is_what_windows_proves():
    ctx = context("reference-h200")
    r = regimes.regime(REG, "onsite-links")
    loc = ctx.location
    assert r.budget(ctx).inputs == pytest.approx(loc.memory + loc.host_memory + loc.links * r.wipe_s)
    assert "Accounting.Guarantees.Actual.Windows" in r.cites()
    assert lean.weakest(r.cites()) == "conditional"
    assert lean.weakest(regimes.regime(REG, "onsite").cites()) == "draft"


def test_four_servers_as_one_location_have_four_times_a_servers_budget():
    r = regimes.regime(REG, "onsite")
    one, four = r.budget(context("reference-h200")), r.budget(context("reference-h200-rack4"))
    assert four.compute == pytest.approx(4 * one.compute) and four.inputs == pytest.approx(4 * one.inputs)


def test_without_pouw_only_the_physical_cut_bounds():
    ctx = context("reference-h200")
    region = regimes.regime(REG, "onsite-no-pouw").partition(ctx).regions[0]
    declared, physical = region.cuts
    assert declared.blocks[0].count == 0 and math.fsum(declared.free.values()) >= region.compute
    assert physical.name == "physical"


def test_an_nvl72_rack_takes_kimi_k2s_state_in_one_day_where_an_hgx_server_does_not():
    rack, server = evaluate_named(REG, "onsite-gb200-kimi"), evaluate_named(REG, "onsite-h200-kimi")
    rack_in, server_in = (r.scenario.regime.budget(r.scenario.context).inputs for r in (rack, server))
    assert rack_in > rack.families[0].profile.state_bits > server_in
    assert rack.hi < server.hi


@pytest.mark.parametrize("name", REG.names("regime"))
def test_every_regime_builds_and_cites_only_locked_names_or_drafts(name):
    r = regimes.regime(REG, name)
    cites = getattr(r, "CITES", ()) + (r.cites() if isinstance(r, Onsite) else ())
    assert all(lean.status(c) != "absent" for c in cites), cites
