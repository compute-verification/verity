import importlib.util
import math
from pathlib import Path

import pytest
from rho import envelopes, hardware, lean
from rho.envelopes import Context, Detection, Onsite, Remote
from rho.params import Param, Registry
from rho.scenario import build, evaluate_named
from rho.theorem import Budget, slowdown

ROOT = Path(__file__).resolve().parents[3]
REG = Registry.load()

_spec = importlib.util.spec_from_file_location("tiers", ROOT / "benchmarks" / "nci" / "tiers.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

#: our envelope field: tiers' DEFAULTS key
TIERS_KEYS = {"residual": "residual", "clock_share": "clock_share", "eps_w": "eps_w", "eps_agg": "eps_agg",
              "identities": "identities", "lag_s": "lag_s", "proof_share": "share", "overhead": "overhead",
              "replay": "replay", "step_flop": "step_flop", "p_evade": "p_evade", "luck": "luck",
              "step_bits": "step_bits", "floor_steps": "floor_steps", "chord_kappa": "kappa"}


def context(fleet: str, registry: Registry = REG, seconds: float | None = None) -> Context:
    f = hardware.fleet(registry, fleet)
    h = hardware.horizon(registry, "year") if seconds is None else hardware.Horizon("given", seconds)
    return Context(f, hardware.location(registry, f.location), envelopes.policy(registry, "kimi-k2-1m-raw"), h,
                   hardware.units(registry))


def tiers_context(registry: Registry = REG) -> Context:
    return context("tiers-reference", registry, T.YEAR)


def tiers_params(ctx: Context) -> dict:
    """tiers' DEFAULTS with the host share its server's CPUs give (0.11796, which tiers rounds to 0.118)."""
    return T.DEFAULTS | {"host_share": ctx.location.host_phi / ctx.location.phi}


def field(regime, name: str):
    """`name` from whichever of the regime's envelopes holds it."""
    return next(getattr(e, name) for e in regime.envelopes().values() if hasattr(e, name))


def test_the_remote_envelopes_numbers_are_tiers_defaults():
    r = envelopes.regime(REG, "remote-sampled")
    for ours, theirs in TIERS_KEYS.items():
        assert field(r, ours) == pytest.approx(T.DEFAULTS[theirs], rel=1e-12), ours
    ctx = tiers_context()
    assert ctx.location.host_phi / ctx.location.phi == pytest.approx(T.DEFAULTS["host_share"], rel=1e-3)
    assert (ctx.units.flops(ctx.location.phi), ctx.fleet.count, ctx.fleet.mfu_train) == pytest.approx(
        (T.DEFAULTS["node_flops"], T.DEFAULTS["nodes"], T.DEFAULTS["mfu"]))
    assert (ctx.policy.gamma, ctx.policy.k) == (T.DEFAULTS["Gamma"], T.DEFAULTS["K"])


def test_the_remote_uncertified_share_is_tiers_residual_term():
    ctx = tiers_context()
    p = tiers_params(ctx)
    remote = envelopes.regime(REG, "remote")
    assert remote.pouw.audited(ctx.location) == pytest.approx(T.audited_residual(p), rel=1e-12)
    assert remote.uncertified(ctx) == pytest.approx(T.evaluate(p, T.TIERS["T1"])["residual"], rel=1e-9)


def test_with_sampled_proofs_it_adds_tiers_escape_and_cheating_shares():
    ctx = tiers_context()
    p = tiers_params(ctx)
    sampled = envelopes.regime(REG, "remote-sampled")
    t5 = T.evaluate(p, T.TIERS["T5"])
    slack = sampled.slack(ctx)
    assert sampled.detection.injected() == pytest.approx(t5["cheat_bits"], rel=1e-12)
    assert sampled.detection.chord == pytest.approx(T.chord(p, t5["cheat_bits"]), rel=1e-7)
    assert math.fsum(v for k, v in slack.items() if k != "exfil.faulty") == pytest.approx(t5["residual"], rel=1e-9)
    assert slack["exfil.faulty"] == pytest.approx(t5["cheat"], rel=1e-7)


def test_with_nothing_else_bounding_rho_is_lean_remote_dampener_at_full_capacity():
    """DampenerRemote with no declared products (ν_T = 0) and χ = 0 is `max(1, 1/(s + (1 − s) u / MFU₀))`, against an
    unencumbered trainer at `MFU₀`. The bound counts every gate at capacity against the reference at capacity, so its ρ
    is the dampener at `MFU₀ = 1`; tiers' dampener at its own MFU is 1 at today's defaults."""
    reg = REG.override({"regime.remote.residual": 0.01, "regime.remote.clock_share": 0.0, "regime.remote.host": 0.0})
    ctx = tiers_context(reg)
    remote = envelopes.regime(reg, "remote")
    part = remote.partition(ctx)
    s, u = part.eps_dark, remote.uncertified(ctx)
    assert slowdown(part, lambda b: 0.0, 0.6).rho == pytest.approx(max(1.0, 1 / (s + (1 - s) * u)), rel=1e-12)
    assert slowdown(part, lambda b: 0.0, 0.6).rho > 10
    ctx = tiers_context()
    today = envelopes.regime(REG, "remote")
    assert all(T.evaluate(tiers_params(ctx), m)["dampener"] == 1.0 for m in T.TIERS.values())
    s, u = ctx.fleet.eps_dark, today.uncertified(ctx)
    # the inference blocks hold a little weight too, under their caps
    lo = evaluate_named(REG, "remote-tiers-kimi").lo
    assert lo <= 1 / (s + (1 - s) * u) and lo == pytest.approx(1 / (s + (1 - s) * u), rel=1e-4)


def test_the_inference_blocks_are_the_certified_work_that_ran():
    ctx = context("reference-h200")
    remote = envelopes.regime(REG, "remote")
    loc, s = ctx.location, ctx.horizon.seconds
    n = (1 - remote.pouw.audited(loc)) * (1 - remote.pouw.eps_w) * loc.phi * s / ctx.c
    assert remote.blocks(ctx, loc.phi * s).count == pytest.approx(n)
    assert remote.blocks(ctx, loc.phi * s).budget == Budget(ctx.c, ctx.policy.k)


def test_the_layout_gets_only_what_an_abiding_strategy_gets():
    ctx = context("reference-h200")
    sampled = envelopes.regime(REG, "remote-sampled")
    layout = sampled.partition(ctx).serving[0].layout
    assert set(layout.uncertified) == set(Remote.ABIDING)
    assert set(sampled.slack(ctx)) - set(layout.uncertified) == {"skip.certs", "break.lag", "break.escape",
                                                                 "exfil.faulty"}
    used = ctx.location.phi * ctx.horizon.seconds * ctx.fleet.mfu_train
    assert layout.uncertified["gap.floor"] == pytest.approx(sampled.pouw.residual * used)


def test_a_regime_inherits_its_base_and_overrides_what_differs():
    onsite, five = envelopes.regime(REG, "onsite"), envelopes.regime(REG, "onsite-5gb")
    assert isinstance(five, Onsite) and five.wipe.per == "device" and five.wipe.residual_bits == 4e10
    assert five.wipe.wipe_s == onsite.wipe.wipe_s and onsite.wipe.per == "server" and onsite.wipe.k_wipe == 4e10
    assert envelopes.regime(REG.override({"regime.remote.residual": 0.1}), "onsite-5gb").pouw.residual == 0.1


def test_a_regime_table_with_a_stray_key_or_a_base_loop_is_refused():
    params = {p.key: p for p in REG.params()}
    stray = Registry(params | {"regime.onsite-5gb.residul": Param("regime.onsite-5gb.residul", 0.1)})
    with pytest.raises(ValueError, match="not fields of Onsite"):
        envelopes.regime(stray, "onsite-5gb")
    loop = Registry(params | {"regime.remote.base": Param("regime.remote.base", "onsite")})
    with pytest.raises(ValueError, match="loop"):
        envelopes.regime(loop, "onsite")
    with pytest.raises(ValueError, match="form"):
        envelopes.regime(REG.override({"regime.onsite-links.form": "widths"}), "onsite-links")
    with pytest.raises(ValueError, match="wipe per"):
        envelopes.regime(REG.override({"regime.onsite.per": "rack"}), "onsite")
    with pytest.raises(ValueError, match="sampled proofs need"):
        Detection(1, 3600, sampled=True)


def test_the_serving_block_is_a_server_between_two_wipes_and_k_wipe_per_server():
    """`(c_S, k_S) = (⌊Φ_s Δ_wipe⌋, k_wipe + (r_in + r_side) Δ_wipe)`, `k_wipe` of the server's whole memory (Daniel, 26 Oct
    12:16 PM PDT)."""
    ctx = context("reference-h200")
    r = envelopes.regime(REG, "onsite")
    loc, pol, w = ctx.location, ctx.policy, r.wipe.wipe_s
    rate = r.certifier.intake * pol.k / ctx.c * loc.phi + loc.hosts * (r.certifier.timing + r.certifier.side)
    assert r.budget(ctx) == Budget(float(math.floor(loc.phi * w)), loc.hosts * r.wipe.k_wipe + rate * w)
    # an H200 server: 5 GB, and α Φ = 4.9e6 bit/s for a day
    assert r.budget(ctx).inputs == pytest.approx(4.67e11, rel=0.01)
    # per device: 8 x 10 GB, 36.5 GB of host
    per_device = envelopes.regime(REG, "onsite-per-device")
    state = loc.devices * per_device.wipe.residual_bits + loc.hosts * per_device.wipe.host_residual_bits
    assert per_device.budget(ctx).inputs == pytest.approx(state + rate * w) == pytest.approx(1.36e12, rel=0.01)


def test_todays_wipe_leaves_about_12_gb_a_server_and_its_boot_disk():
    """`regime.onsite-today`: `k_wipe` of about 12 GB a server plus each server's (or compute tray's) boot disk, which
    the wipe doesn't reach; without the boot disk an H200 serving block's input is within 2x of the 5 GB reference's."""
    onsite, today = envelopes.regime(REG, "onsite"), envelopes.regime(REG, "onsite-today")
    for fleet in ("reference-h200", "reference-gb200"):
        loc = context(fleet).location
        assert today.wipe.state(loc) == pytest.approx(loc.hosts * (9.6e10 + 1.536e13))
        assert onsite.wipe.state(loc) == loc.hosts * 4e10 and onsite.wipe.boot_disk_bits == 0
    ctx = context("reference-h200")
    bare = envelopes.regime(REG.override({"regime.onsite-today.boot_disk_bits": 0}), "onsite-today")
    assert onsite.budget(ctx).inputs < bare.budget(ctx).inputs < 2 * onsite.budget(ctx).inputs
    assert today.budget(ctx).inputs > 30 * onsite.budget(ctx).inputs


def test_the_links_form_is_what_windows_proves():
    ctx = context("reference-h200")
    r = envelopes.regime(REG, "onsite-links")
    loc = ctx.location
    assert r.budget(ctx).inputs == pytest.approx(loc.hosts * loc.server_memory + loc.links * r.wipe.wipe_s)
    assert "Accounting.Guarantees.Actual.Windows" in r.cites()
    assert lean.weakest(r.cites()) == "conditional"
    assert lean.weakest(envelopes.regime(REG, "onsite").cites()) == "draft"


def test_four_servers_as_one_location_have_four_times_a_servers_budget():
    r = envelopes.regime(REG, "onsite")
    one, four = r.budget(context("reference-h200")), r.budget(context("reference-h200-rack4"))
    assert four.gates == pytest.approx(4 * one.gates) and four.inputs == pytest.approx(4 * one.inputs)


def test_without_pouw_only_the_serving_cut_bounds():
    ctx = context("reference-h200")
    block = envelopes.regime(REG, "onsite-no-pouw").partition(ctx).serving[0]
    inference, serving = block.cuts
    assert inference.blocks[0].count == 0 and math.fsum(inference.uncertified.values()) >= block.gates
    assert serving.name == "serving"


def test_at_5_gb_a_server_no_location_takes_kimi_k2s_state_in_a_day_and_per_device_an_nvl72_rack_does():
    def inputs(registry: Registry, name: str):
        r = evaluate_named(registry, name)
        return r, r.scenario.regime.budget(r.scenario.context).inputs

    per_device = REG.override({f"scenario.{n}.regime": "onsite-per-device" for n in ("onsite-gb200-kimi",
                                                                                     "onsite-h200-kimi")})
    (rack, rack_in), (_, server_in) = inputs(REG, "onsite-gb200-kimi"), inputs(REG, "onsite-h200-kimi")
    (rack_pd, rack_pd_in), (server_pd, server_pd_in) = (inputs(per_device, "onsite-gb200-kimi"),
                                                        inputs(per_device, "onsite-h200-kimi"))
    state = rack.families[0].profile.state_bits
    assert rack_pd_in > state > rack_in > server_pd_in > server_in
    assert rack_pd.hi < server_pd.hi


@pytest.mark.parametrize("name", REG.names("regime"))
def test_every_regime_builds_and_cites_only_locked_names_or_drafts(name):
    r = envelopes.regime(REG, name)
    cites = getattr(r, "CITES", ()) + (r.cites() if isinstance(r, Onsite) else ())
    assert all(lean.status(c) != "absent" for c in cites), cites


def test_every_uncertified_term_is_keyed_by_the_route_it_escapes_by():
    """The working draft's §8: each gate of the inference cut that no inference block constrains names its route."""
    ctx = context("reference-h200")
    for name in ("remote", "remote-sampled"):
        slack = envelopes.regime(REG, name).slack(ctx)
        assert slack and all(k.split(".")[0] in envelopes.ROUTES for k in slack)


def test_pouws_delta_splits_into_coins_and_cryptography():
    """`p = δ_s + δ_bind + δ_ks` per window, by Markov at `ε'`: the coins' share and the cryptography's add to the
    theorem's `δ = p/ε'` (2.7e-8 at `p = 3 · 2⁻⁴⁰`, `ε' = 10⁻⁴`), and sampled proofs add their escape bound's `δ`."""
    ctx = context("reference-h200")
    remote, sampled = envelopes.regime(REG, "remote"), envelopes.regime(REG, "remote-sampled")
    d = remote.delta(ctx)
    assert d.coin == pytest.approx(2**-40 / remote.pouw.eps_agg, rel=1e-2)
    assert d.crypto == pytest.approx(2 * d.coin, rel=1e-2)
    assert d.coin + d.crypto == pytest.approx(3 * 2**-40 / 1e-4, rel=1e-2)
    assert sampled.delta(ctx).coin == pytest.approx(d.coin + sampled.detection.p_evade)
    assert (envelopes.regime(REG, "unmonitored").delta(ctx).coin, envelopes.regime(REG, "unmonitored").delta(ctx).crypto) \
        == (0.0, 0.0)


@pytest.mark.parametrize(("scenario", "per"), [("reference-onsite-h200", "hosts"),
                                               ("reference-onsite-nvl72-per-device", "devices")])
def test_delta_sums_every_subprotocol_and_the_wipe_per_server_and_per_wipe(scenario, per):
    """`Σ_i Σ_S δ_i(S)`: PoUW's `p/ε'` over all windows, and the wipe's `2⁻¹²⁸ + Adv_PRF` once per game, a game per
    server (or, per device, per device) per wipe over the horizon; the certifier adds nothing (it is the auditor's)."""
    s = build(REG, scenario)
    ctx, r = s.context, s.regime
    d = r.delta(ctx)
    assert set(d.by) == {"pouw", "detection", "wipe", "certifier"}
    games = ctx.fleet.count * getattr(ctx.location, per) * ctx.horizon.seconds / r.wipe.wipe_s
    assert d.by["wipe"] == pytest.approx((games * 2**-128, games * r.wipe.prf_advantage))
    assert d.by["pouw"] == pytest.approx(r.pouw.delta(ctx)) and d.by["certifier"] == (0.0, 0.0)
    assert d.coin == pytest.approx(math.fsum(c for c, _ in d.by.values()))
    assert d.crypto == pytest.approx(math.fsum(x for _, x in d.by.values()))
    assert d.coin == pytest.approx(r.pouw.window_delta_coin / r.pouw.eps_agg + games * 2**-128)
    assert d.json()["by"]["wipe"]["coin"] == d.by["wipe"][0]
