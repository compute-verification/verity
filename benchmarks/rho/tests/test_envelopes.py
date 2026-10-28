import dataclasses
import importlib.util
import math
from pathlib import Path

import pytest
from rho import envelopes, hardware, lean
from rho.envelopes import Certifier, Context, Detection, Onsite, Remote
from rho.params import Param, Registry
from rho.scenario import build, evaluate_named
from rho.theorem import Budget, multiple_bound, overhead

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
    """tiers' DEFAULTS with the host share remote's PoUW envelope counts: 0, hosts carry no compute."""
    return T.DEFAULTS | {"host_share": envelopes.regime(REG, "remote").pouw.host * ctx.location.host_phi / ctx.location.phi}


def field(regime, name: str):
    """`name` from whichever of the regime's envelopes holds it."""
    return next(getattr(e, name) for e in regime.envelopes().values() if hasattr(e, name))


def test_the_remote_envelopes_numbers_are_tiers_defaults():
    r = envelopes.regime(REG, "remote-sampled")
    for ours, theirs in TIERS_KEYS.items():
        assert field(r, ours) == pytest.approx(T.DEFAULTS[theirs], rel=1e-12), ours
    ctx = tiers_context()
    assert envelopes.regime(REG, "remote").pouw.host == T.DEFAULTS["host_share"] == 0.0
    assert ctx.location.host_phi / ctx.location.phi == pytest.approx(0.12608, rel=1e-3)
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


def test_with_nothing_else_bounding_the_padded_size_ratio_is_lean_dampener_remote_at_full_capacity():
    """Lean's `DampenerRemote` with no declared products (ν_T = 0) and χ = 0 is `max(1, 1/(s + (1 − s) u / MFU₀))`,
    against an unencumbered trainer at `MFU₀`. The bound counts every gate at capacity against the reference at
    capacity, so its padded size ratio is `DampenerRemote`'s `ρ` at `MFU₀ = 1` (where at least 1), and the overhead is
    that `ρ` at the reference's MFU."""
    reg = REG.override({"regime.remote.residual": 0.01, "regime.remote.clock_share": 0.0, "regime.remote.host": 0.0})
    ctx = tiers_context(reg)
    remote = envelopes.regime(reg, "remote")
    part = remote.partition(ctx)
    s, u = part.eps_dark, remote.uncertified(ctx)
    o = overhead(part, lambda b: 0.0, 0.6)
    assert o.padded == pytest.approx(multiple_bound(1.0, s, (1 - s) * u), rel=1e-12) and o.padded > 10
    assert o.rho == o.multiple(1.0) == pytest.approx(o.padded, rel=1e-12)
    assert overhead(part, lambda b: 0.0, 0.6, mfu=0.4).rho == o.multiple(0.4) == pytest.approx(
        multiple_bound(0.4, s, (1 - s) * u), rel=1e-12)
    ctx = tiers_context()
    s, u = ctx.fleet.eps_dark, envelopes.regime(REG, "remote").uncertified(ctx)
    # the inference blocks hold a little weight too, under their caps
    pad = evaluate_named(REG, "remote-tiers-kimi").families[0].combined.padded
    assert pad <= 1 / (s + (1 - s) * u) and pad == pytest.approx(1 / (s + (1 - s) * u), rel=1e-4)


#: `Nci.Dampener.remoteDampener` as Security/Definitions/Nci/Remote.lean defines it, and the conclusion of
#: `Nci.Guarantees.DampenerRemote` (Security/Definitions/Nci/Guarantees/Remote.lean) that names it `ρ`
REMOTE_DAMPENER = "noncomputable def remoteDampener (mfu s x : ℝ) : ℝ := max 1 (1 / (s + x / mfu))"
REMOTE_CONCLUSION = "remoteDampener mfu (unmonitoredShare C U)"


def test_the_remote_overhead_is_lean_remote_dampener_at_the_references_mfu():
    """Without reference efficiency the remote slowdown is `DampenerRemote`'s `ρ`, `remoteDampener MFU₀ s x ≤
    hardwareMultiple MFU₀`, at the reference training algorithm's MFU (0.4): free compute counts at peak. Under it free
    compute counts at `MFU₀`, so `ρ = max(1, 1/(s + x))`, the padded size ratio. Hosts carry no compute, so tiers'
    server and the H200 give the same `x = 0.33`, and `ρ = 3.04` (1.22 without reference efficiency), as
    `benchmarks/nci`'s compute dampener reads it. Without both assumptions tiers' server (host CPUs 12.6% of the GPUs)
    has `x = 0.45`, which outruns that developer, so `ρ = 1`; the H200 (2.9%) has 1.12."""
    lean_dir = ROOT / "Security" / "Definitions" / "Nci"
    assert REMOTE_DAMPENER in (lean_dir / "Remote.lean").read_text()
    assert REMOTE_CONCLUSION in (lean_dir / "Guarantees" / "Remote.lean").read_text()
    for mfu, s, x in ((0.4, 0.001, 0.45), (1.0, 0.0, 0.2), (0.4, 0.0, 0.0), (0.5, 0.2, 0.01)):
        rate = s + x / mfu
        assert multiple_bound(mfu, s, x, reference_efficiency=False) == (max(1.0, 1 / rate) if rate > 0 else math.inf)
        assert multiple_bound(mfu, s, x) == multiple_bound(1.0, s, x, reference_efficiency=False)

    ctx = tiers_context()
    t1 = T.evaluate(tiers_params(ctx), T.TIERS["T1"])
    x = t1["residual"] + t1["declared"] + t1["cheat"]
    assert x / T.DEFAULTS["mfu"] < 1
    peak = REG.override({"task.reference_efficiency": False})
    # tiers counts no dark share (s = 0) where rho's fleet has 0.1%
    for reg, ref, want in ((REG, True, 3.040), (peak, False, 1.218)):
        lo = evaluate_named(reg, "remote-tiers-kimi").lo
        assert lo == pytest.approx(multiple_bound(T.DEFAULTS["mfu"], t1["unmonitored"], x, ref), rel=5e-3)
        assert lo == pytest.approx(want, abs=1e-3)
    hosts = peak.override({"regime.remote.host": 1.0})
    assert evaluate_named(hosts, "remote-tiers-kimi").lo == 1.0
    assert evaluate_named(hosts, "remote-h200-kimi").lo == pytest.approx(1.120, abs=1e-3)

    r = evaluate_named(REG, "remote-h200-kimi")
    ctx, remote = r.scenario.context, r.scenario.regime
    s, u = ctx.fleet.eps_dark, remote.uncertified(ctx)
    assert r.scenario.mfu == ctx.fleet.mfu_train == 0.4
    assert r.lo == pytest.approx(multiple_bound(0.4, s, (1 - s) * u), rel=1e-4) == pytest.approx(3.040, abs=1e-3)
    assert r.families[0].combined.padded == pytest.approx(r.lo, rel=1e-12)
    assert evaluate_named(peak, "remote-h200-kimi").lo == pytest.approx(
        multiple_bound(0.4, s, (1 - s) * u, reference_efficiency=False), rel=1e-4) == pytest.approx(1.218, abs=1e-3)


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
    assert layout.uncertified["undecl.gpu"] == pytest.approx(sampled.pouw.residual * used)


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
    """`(c_S, k_S) = (⌊Φ_s Δ_wipe⌋, k_wipe + (r_in + r_side) Δ_wipe)`, `k_wipe` of the server's whole memory (ruled)."""
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


def test_the_declared_io_bound_at_w63_is_the_per_server_bounds_per_window_honest_delta_form():
    """The per-server bound at W63 (`regime.onsite-w63`: B 512, r 4, 4 lanes, C 47, IDB 17, D 27, KO 64, τ_b 100 ms,
    32-bucket link-windows), its `requestBits`, `bucketBits`, `windowRequests` and `windowBits`
    (`Properties.NetworkCertifier.PerServerIngress`, `draft:per-server-ingress`). A request's head leaves 988 bits free
    (the ordinal, content-length, `max_tokens` and `seed` at 32 each, `temperature` and `top_p` at 24, `model` and
    `user` at 406), and its prompt is K0 + (C − 1) K = 12 + 46 · 45 = 2,082 ids: 36,382 bits in. A bucket is
    4 ⌈log₂ 6⌉ = 12 bits, and a request 1,095 out (1,088 id bits and the count). A link-window holds `4 ⌈32 / 28⌉ = 8`
    requests, so a server takes at most 291,440 bits in a 3.2 s window, 91.1 kbit/s in and 2.7 kbit/s out, against the
    warden's line rate of 655 kbit/s each way. That is never less than the rate over many windows, 52.1 kbit/s, which it
    equals only at T = D + 1. A faulty ingress unit, which this honest-δ form excludes, would add 356,592 bits
    (8 · 512 · 16 + 2 · 4 · 36,382), 122% of a window. `regime.onsite` doesn't take it."""
    assert envelopes.REQUEST_HEAD == {"connection ordinal": 32, "content-length": 32, "max_tokens": 32, "seed": 32,
                                      "temperature": 24, "top_p": 24, "model": 406, "user": 406}
    c = envelopes.regime(REG, "onsite-w63").certifier
    assert (c.first_cell_ids, c.ids_per_cell, c.request_ids, c.request_in) == (12, 45, 2_082, 988 + 2_082 * 17)
    assert (c.request_in, c.bucket_in, c.request_out, c.hold, c.window_buckets) == (36_382, 12, 1_095, 27, 32)
    assert c.window_bits(32) == (291_440, 8_760) == (8 * 36_382 + 32 * 12, 8 * 1_095)
    assert (c.r_in, c.r_out) == (pytest.approx(91_075), pytest.approx(8_760 / 3.2))
    assert (round(c.r_in / 1e3, 1), round(c.r_out / 1e3, 1), c.warden_rate) == (91.1, 2.7, 655_360)
    assert c.faulty_unit_bits == 356_592 and round(c.faulty_unit_bits / c.window_bits(32)[0], 2) == 1.22
    # a slot is held D + 1 = 28 buckets from its admission, so the ceiling steps past 28
    assert c.window_bits(28) == (4 * 36_382 + 28 * 12, 4 * 1_095)
    assert c.window_bits(29) == (8 * 36_382 + 29 * 12, 8 * 1_095)
    long_run = 4 * 36_382 / 2.8 + 120
    at = {t: dataclasses.replace(c, window_buckets=t).r_in for t in (28, 32, 600)}
    assert at[28] == pytest.approx(long_run) and at[32] > at[600] > long_run
    assert round(long_run / 1e3, 1) == 52.1
    assert [envelopes.clog2(x) for x in (1, 2, 5, 48, 65)] == [0, 1, 3, 6, 7]
    assert not envelopes.regime(REG, "onsite").certifier.declared
    for refused in (lambda: Certifier(declared=True),
                    lambda: envelopes.regime(REG.override({"regime.onsite.declared": True}), "onsite")):
        with pytest.raises(ValueError, match=r"declared I/O bound needs \['cell_bytes'"):
            refused()
    with pytest.raises(ValueError, match=r"declared I/O bound needs \['window_buckets'\]"):
        dataclasses.replace(c, window_buckets=math.nan)


def test_a_servers_certified_ingress_is_its_declared_inputs_and_line_rate_the_upper_end():
    """`regime.onsite`'s certified ingress is `intake · α Φ`, cited to draft:declared-ingress (a server's ingress only
    its declared inputs, at most α Φ at full use). With `declared` (`regime.onsite-w63`) it is `hosts · r_in`, cited to
    draft:per-server-ingress too: as an intake `r_in / (α Φ_s)`, 0.0184 at Kimi K2's policy on an H200 server.
    `declared_x` scales it up to at most the location's links (its high end is line rate, the upper end until Lean
    proves the per-server bound)."""
    line = REG.override({"regime.onsite-w63.declared_x": REG.param("regime.onsite-w63.declared_x").high})
    on, off, at_line = (envelopes.regime(REG, "onsite-w63"), envelopes.regime(REG, "onsite"),
                        envelopes.regime(line, "onsite-w63"))
    c = on.certifier
    for fleet in ("reference-h200", "reference-gb200"):
        ctx = context(fleet)
        loc, alpha_phi = ctx.location, ctx.policy.k / ctx.c * ctx.location.phi
        assert c.prompts(ctx) == pytest.approx(loc.hosts * c.r_in)
        assert c.intake_of(ctx) == pytest.approx(loc.hosts * c.r_in / alpha_phi) and c.intake_of(ctx) < 0.02
        assert on.rate_in(ctx) == pytest.approx(loc.hosts * (c.r_in + c.timing + c.side))
        assert at_line.rate_in(ctx) == pytest.approx(loc.links + loc.hosts * (c.timing + c.side))
        assert off.rate_in(ctx) == pytest.approx(alpha_phi + loc.hosts * (c.timing + c.side))
        assert off.certifier.intake_of(ctx) == 1
    assert c.intake_of(context("reference-h200")) == pytest.approx(0.01841, rel=1e-3)
    assert {"draft:declared-ingress", "draft:per-server-ingress"} <= set(on.cites())
    assert "draft:declared-ingress" in off.cites() and "draft:per-server-ingress" not in off.cites()
    assert lean.weakest(on.cites()) == lean.weakest(off.cites()) == "draft"


def test_onsite_w63_beside_the_reference():
    """W63's cadence beside the reference H200 server (`reference-onsite-h200`, ingress α Φ = 3.23 Mbit/s): 91.1 kbit/s
    a 32-bucket link-window, 0.028 α Φ at v4-pro-session, so a day's input drops from 3.19e11 to 4.80e10 bits and the
    lower bound's padded size ratio, which a dark share of 1% caps near 100, rises from 98.7 to 99.8, with the best
    attack's at 249 and 250; at line rate both fall to remote's (2.99, at the card table's clock). The cadence feeds the
    serving this package credits only if outputs outnumber prompts more than 20:1, so it isn't the default."""
    reg = REG.override({"fleet.world-h200.eps_dark": 0.01})
    ref, w63 = evaluate_named(reg, "reference-onsite-h200"), evaluate_named(reg, "onsite-w63")
    inputs = [r.scenario.regime.budget(r.scenario.context).inputs for r in (ref, w63)]
    assert inputs == [pytest.approx(3.193e11, rel=1e-3), pytest.approx(4.803e10, rel=1e-3)]
    assert w63.scenario.regime.certifier.intake_of(w63.scenario.context) == pytest.approx(0.02819, rel=1e-3)
    padded = [(round(r.families[0].combined.padded, 1), round(r.families[0].best_attack.padded)) for r in (ref, w63)]
    assert padded == [(98.7, 249), (99.8, 250)]
    line = reg.override({"regime.onsite.intake": REG.param("regime.onsite.intake").high,
                         "regime.onsite-w63.declared_x": REG.param("regime.onsite-w63.declared_x").high})
    at_line = [evaluate_named(line, n) for n in ("reference-onsite-h200", "onsite-w63")]
    remote = evaluate_named(reg, "reference-remote-h200")
    assert all(r.lo == pytest.approx(remote.lo, rel=1e-9) for r in at_line)
    assert round(remote.families[0].combined.padded, 2) == 2.99


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


def test_pouw_as_deployed_on_the_h200_certifies_so_little_that_rho_stays_near_one():
    """`regime.remote-h200-pouw`: per-call J10-R credits a step's 5.72e14 served INT8 MACs against the H200's 1.0705e15
    MACs/s (the ACCEL card table: 4,096 a SM-clock, 132 SMs, at the clock's 1,980 MHz ceiling) over its 12.60 s (served
    3.78, its commit 0.42, PoUW's units 5.59 and their commit 2.81). J10-R credits words, and C counts the card at its
    ceiling, so the clock term is 0. Hosts carry no compute, so `r̂` is the residual, 0.958. onsite-today's twin carries
    the same PoUW envelope, and both differ from remote's in three keys."""
    remote, today = envelopes.regime(REG, "remote-h200-pouw"), envelopes.regime(REG, "onsite-today-h200-pouw")
    assert today.pouw == remote.pouw and today.wipe == envelopes.regime(REG, "onsite-today").wipe
    macs = 4096 * (52 * (2 * 16384**2 + 2 * 16384 * 64 + 2 * 16384 * 65536) + 16384 * 256)
    assert remote.pouw.residual == pytest.approx(1 - macs / (4096 * 132 * 1.98e9 * (3.78 + 0.42 + 5.59 + 2.81)), abs=5e-4)
    assert remote.pouw.clock_share == 0
    p = remote.pouw
    assert dataclasses.replace(envelopes.regime(REG, "remote").pouw, residual=p.residual, clock_share=p.clock_share,
                               host=p.host) == p
    loc = context("reference-h200").location
    assert p.host == 0 and p.audited(loc) == p.residual == pytest.approx(0.958, abs=5e-4)
    assert evaluate_named(REG, "reference-remote-h200-pouw").lo < 1.05


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
    assert rack_pd.attack < server_pd.attack


@pytest.mark.parametrize("name", REG.names("policy"))
def test_every_policy_states_its_inference_overhead_or_that_it_is_not_modeled(name):
    v = envelopes.policy(REG, name).inference_overhead
    p = REG.param(f"policy.{name}.inference_overhead")
    assert (math.isnan(v) and p.source.startswith("not modeled")) or (v >= 1 and p.status)


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
    """`Σ_a δ_a`: PoUW's `p/ε'` over all windows, and the wipe's `2⁻¹²⁸ + Adv_PRF` once per game, a game per
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
