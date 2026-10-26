"""rho against the training lane's reference-DC model at matched parameters.

That model (`benchmarks/nci/reference_dc.py` on `cursor/nci-dc-model-4207`, which does not land; its rendered output
is art:dee8d74e220884143c40b6d7d28acdf13c358f1de84c5f0df2ea7bf525bf4e83) gives D, the fleet's compute over the training
it can do on monitored compute, so D = 1/g (`Slowdown.g`: monitored training as a share of its unencumbered rate). Its
numbers are the constants in `MINE`, run at rho's model facts (aOut = 65,536, layer 0's inputs cheap); rho's are
computed. What is matched, and what each side does differently:

* **Counted sets (κ).** Mine credits a counted forward product with κ_F = 6 FLOP of reference training (6 N_active a
  token) and a counted multiplication with κ_F/c = 3 FLOP for thread (c = 2 reuses). rho's fanout-priced counts forward
  products against `kappa_forward` = κ · forward/counted (0.298 at κ = 0.602: a forward product is 6.71 FLOP of the
  recipe, attention included); thread and cover count κ's own set. Matched: κ_forward = 1/3 for fan, κ = 2/3 for thread,
  κ = 1/3 for cover.
* **Compute.** Mine runs at σ = 1 throughout. rho counts free capacity and declared blocks at peak (as Lean does) and the
  region's training at `mfu_train`, so `g`'s free share is share/MFU₀. Matched: `mfu_train` = 1. At rho's defaults the
  remote audit's r̂ = 0.323 + 0.064 + 0.029 (GPU floor, clock, host CPUs at peak) is 0.416 of peak, more than the 0.4
  the location trains at, so remote g = 1: no slowdown, against my 131 at r̂ = 0.
* **The declared block.** Mine is c = k φ_F with φ_F = 2 × 49e9 / 17 FLOP per novel bit (17-bit tokens). rho's
  Γ = 9.8e16 is a 1M-token session. Matched to mine.
* **U.** Mine is r̂ + ε_W(1 − r̂) + p_fail/δ + JΔ/H, with r̂ = 0 and p_fail/δ = 3·2⁻⁴⁰/1e-6. Matched: `residual`,
  `clock_share` and `host` 0, `eps_agg` p_fail/δ. The lag is over a 365-day year here and a Julian one in mine. Mine also
  counts what sampled proofs let through at δ (escapes and injected outputs: 7.6e-9 of U for fan, 1.6e-6 for thread,
  more for cover); the reference regimes are unsampled (`χ = θ = 0`). These make the remote gaps, all under 0.05%.
* **aOut.** rho's is wq_b's 128 × 512 = 65,536 outputs; mine was the MoE input's 43,392. At 65,536 mine moves from
  131.14 to 130.53 remotely (the inference blocks' own term grows 1.51×) and from 5,354 to 3,545 onsite on H200 (×0.662).
* **The cheap-input share.** Both take the products whose inputs a block can compute from token ids out of fan's count,
  dividing it by 1 − x, at the same per-layer budget (one layer's update at b_w = 2.5 bits a parameter). rho counts whole
  layers, x = (⌊K / layer_bits⌋ + 1) / L: layer 0's inputs are always cheap, and n layers' updates make n + 1 cheap.
  Mine is continuous, x = max(x_lead, K / (2.5 n_total)), run here at x_lead = 1/61 so the inference block's x is rho's.
  At the H200 window rho's x is 9/61 = 0.1475 against my 0.1393, so its 1/g is 0.95% below mine.
* **The whole update.** Once a window's K holds every layer's update at 2.5 bits a parameter both drop the physical
  bound: rho when x reaches 1, at K ≥ (L − 1) layer_bits = 3.93e12, mine at K ≥ 2.5 n_total = 4e12. NVL72's window
  (8.3e12) is past both, so onsite NVL72 is the remote cut in both.
* **I_cert.** rho's certified ingress is `intake` α Φ with α = k/Γ, the declared intake at full use; mine is the
  location's share of the committed requests' input tokens at σ = 1, Φ/φ_F. They agree at matched Γ. At rho's own policy
  α Φ is 20/17 of mine, since the session's 2e7 bits over 1M tokens are 20 bits a token. rho also adds what a wipe leaves
  on each host (`host_residual_bits`, 2.92e11) and the certifier's timing channel (860 bit/s a host), matched to 0, and
  counts side channels per host (18 on an NVL72 rack), mine per location: matched to 1 kb/s a location. Mine's R is
  Daniel's 5 GB a GPU; rho's value is today's 10 GB.
* **β_x.** Mine is 0.66, the cheapest boundary (d = 7168, FP8); rho's 0.87 is the exact-block measurement on
  Qwen2.5-1.5B. Matched to 0.66.
* **Thread.** rho's family has d = 10 and e = aOut; mine has d = 1 (what Lean certifies for a run that trains its
  embedding). Matched d = 1. At k = 2e7 and d = 1 the thread is quadratic, Λ = c k/(4 d b_w) = 4e6 > e, so e doesn't bind.
* **Cover.** rho divides the cover count by Upstream's share ⌊(L − 1)/m⌋/L = 60/61 at the inference block (the first
  layer's products have no certificate); mine multiplies it by u products per counted unit, a flagged normalization
  (u = 16, D = 7.15). Matched at u = 61/60.
"""

import math

import pytest
from rho.params import Registry
from rho.scenario import Result, build, evaluate, evaluate_named

#: FLOP an inference block does per novel bit: 2 N_active per 17-bit token
PHI_F = 2 * 49e9 / 17
#: mine's certificate slack at δ = 1e-6: a window's certificate fails with chance 3 × 2^-40, over δ (Markov)
SLACK = 3 * 2.0**-40 / 1e-6
#: bits a parameter at which a block holds the whole update (C1's floor b_w)
WHOLE_UPDATE = 2.5

#: the reference-DC model (art:dee8d74e…) at a year, δ = 1e-6, σ = 1, r̂ = 0, R = 5 GB a GPU, β_x = 0.66, and rho's
#: model facts: fan's aOut 65,536 and x_lead 1/61, thread's e 65,536, cover's u 61/60
MINE = {
    "remote fan": 130.530,
    "remote thread": 123.840,
    "remote cover": 62.633,
    "onsite fan h200": 3545.25,
    # the H200 window: compute σ Φ × 1 day, inputs R + I_cert + side, and fan's x there
    "C h200": 1.36788e21,
    "K h200": 5.57373e11,
    "x h200": 0.139343,
    # every premise fails past the whole update, so onsite falls back to the remote cut
    "onsite fan nvl72": 130.530,
}


def forward_share() -> float:
    """rho's forward products per counted product, `kappa_forward / kappa`."""
    s = build(Registry.load(), "reference-onsite-h200")
    p = s.families[0].profile(s.model, s.run)
    return p.kappa_forward / p.kappa


def matched(fleet: str, onsite: bool = False, **family: float) -> Registry:
    reg = Registry.load()
    values = {f"fleet.{fleet}.mfu_train": 1, "regime.remote.residual": 0, "regime.remote.clock_share": 0,
              "regime.remote.host": 0, "regime.remote.eps_agg": SLACK,
              "policy.v4-pro-session.gamma": reg["policy.v4-pro-session.k"] * PHI_F,
              "family.full.kappa": 1 / 3 / forward_share(), "family.full.beta_x": 0.66}
    values |= {f"family.full.{k}": v for k, v in family.items()}
    if onsite:
        hosts = reg[f"location.{reg[f'fleet.{fleet}.location']}.hosts"]
        values |= {"regime.onsite.residual_bits": 4e10, "regime.onsite.host_residual_bits": 0,
                   "regime.onsite.timing": 0, "regime.onsite.side": reg["regime.onsite.side"] / hosts}
    return reg.override(values)


def inverse_g(r: Result, argument: str) -> float:
    (f,) = r.families
    (e,) = [e for e in f.bounds if e.argument == argument]
    return 1 / e.slowdown.g


def cheap(k: float, layer_bits: float, layers: int) -> float:
    """rho's cheap-input share at `k`, whole layers (`classes.expensive_share`'s complement)."""
    return min(1.0, (math.floor(k / layer_bits) + 1) / layers)


@pytest.mark.parametrize("fleet", ["h200", "nvl72"])
@pytest.mark.parametrize(("argument", "family", "mine"), [
    ("fanout-priced", {}, MINE["remote fan"]),
    ("threaded", {"kappa": 2 / 3, "thread_d": 1}, MINE["remote thread"]),
    ("cover", {"kappa": 1 / 3}, MINE["remote cover"]),
])
def test_remote_at_no_residual_matches_mine(fleet, argument, family, mine):
    """U-bound: 1/g ≈ 1/(ε_W + ε' + JΔ/H + 1/ρ_inference), the same on both fleets. Within 0.1%; what is left (at most
    4e-4, cover) is the escapes and injected outputs mine counts and the year's length."""
    r = evaluate_named(matched(f"world-{fleet}", **family), f"reference-remote-{fleet}")
    assert math.isclose(inverse_g(r, argument), mine, rel_tol=1e-3)


def test_onsite_h200_fanout_priced_is_mine_but_for_whole_layers():
    """The window binds: 1/g = κ_forward C (1 − x) β_x / (2 aOut K), mine's form. Its compute and inputs (R = 5 GB a GPU,
    I_cert at σ = 1, 1 kb/s of side channels) are mine within 0.1%, and with rho's whole-layer x in place of my
    continuous one (0.1475 against 0.1393, 0.95%) so is 1/g."""
    s = build(matched("world-h200", onsite=True), "reference-onsite-h200")
    budget = s.regime.budget(s.context)
    assert math.isclose(budget.compute, MINE["C h200"], rel_tol=1e-3)
    assert math.isclose(budget.inputs, MINE["K h200"], rel_tol=1e-3)
    p = s.families[0].profile(s.model, s.run)
    x = cheap(budget.inputs, p.layer_bits, p.layers)
    assert math.isclose(x, (math.floor(MINE["x h200"] * p.layers) + 1) / p.layers)
    rho = inverse_g(evaluate(s), "fanout-priced")
    assert math.isclose(rho * (1 - MINE["x h200"]) / (1 - x), MINE["onsite fan h200"], rel_tol=1e-3)


def test_onsite_h200_threaded_is_the_remote_cut():
    """Thread is quadratic at the window (and unchecked past `thread_k`), so the physical cut doesn't bind, in rho as in
    mine."""
    r = evaluate_named(matched("world-h200", onsite=True, kappa=2 / 3, thread_d=1), "reference-onsite-h200")
    assert math.isclose(inverse_g(r, "threaded"), MINE["remote thread"], rel_tol=1e-3)


def test_at_its_defaults_rho_differs_from_mine_by_the_named_factors():
    """rho's own onsite H200 fan-out-priced 1/g (615) is mine (3,545) times 3 κ_forward (0.894), its window's training
    work over mine (mfu_train 0.4, and Φ's rounding), β_x/0.66 (1.32), mine's K over its own (0.460: 10 GB a GPU
    against 5, the host's 2.92e11, the timing channel, ingress at 20-bit tokens) and (1 − x)/(1 − x_mine) (0.800, x at
    19 of 61 layers): nothing else, within 0.1%."""
    reg = Registry.load()
    s = build(reg, "reference-onsite-h200")
    budget = s.regime.budget(s.context)
    p = s.families[0].profile(s.model, s.run)
    factors = (3 * p.kappa_forward * budget.compute * s.context.fleet.mfu_train / MINE["C h200"] * p.beta_x / 0.66
               * MINE["K h200"] / budget.inputs * (1 - cheap(budget.inputs, p.layer_bits, p.layers))
               / (1 - MINE["x h200"]))
    assert math.isclose(inverse_g(evaluate(s), "fanout-priced"), MINE["onsite fan h200"] * factors, rel_tol=1e-3)


def test_the_nvl72_window_holds_the_whole_update_in_both():
    """Past every layer's update at 2.5 bits a parameter neither prices the window: rho's x reaches 1 at (L − 1)
    layer_bits, mine's premise fails at 2.5 n_total, and NVL72's K is past both. So onsite is the remote cut, within
    0.1%."""
    reg = matched("world-nvl72", onsite=True)
    s = build(reg, "reference-onsite-nvl72")
    k = s.regime.budget(s.context).inputs
    p = s.families[0].profile(s.model, s.run)
    assert k >= reg["model.deepseek-v4-pro.n_total"] * WHOLE_UPDATE and cheap(k, p.layer_bits, p.layers) == 1
    assert math.isclose(inverse_g(evaluate(s), "fanout-priced"), MINE["onsite fan nvl72"], rel_tol=1e-3)


@pytest.mark.parametrize("fleet", ["h200", "nvl72"])
def test_ingress_is_mine_at_matched_tokens(fleet):
    """rho's certified ingress, `intake` α Φ, is I_cert = Φ/φ_F at matched Γ; at its own policy it is 20/17 of it (the
    session's 2e7 bits are 20 bits a token, mine 17)."""
    quiet = {"regime.onsite.timing": 0, "regime.onsite.side": 0}
    for reg, tokens in ((matched(f"world-{fleet}", onsite=True), 1.0), (Registry.load(), 20 / 17)):
        s = build(reg.override(quiet), f"reference-onsite-{fleet}")
        assert math.isclose(s.regime.rate_in(s.context), s.context.location.phi / PHI_F * tokens, rel_tol=1e-9)
