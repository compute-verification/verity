"""rho against the training lane's reference-DC model at matched parameters.

That model (`benchmarks/nci/reference_dc.py` on `cursor/nci-dc-model-4207`, which does not land; its rendered output
is art:dee8d74e220884143c40b6d7d28acdf13c358f1de84c5f0df2ea7bf525bf4e83) gives D, the fleet's compute over the training
it can do on monitored compute, so D = 1/g (`Slowdown.g`: monitored training as a share of its unencumbered rate). Its
numbers are the constants in `MINE`; rho's are computed. Where the two differ, and why:

* **Units (κ).** Mine credits each counted forward product with κ_F = 6 FLOP of reference training (6 per active
  parameter per token), and a counted multiplication with κ_F/c = 3 FLOP for thread (c = 2 reuses); rho credits a
  counted product with 2/κ FLOP. So κ = 1/3 matches fan and cover, κ = 2/3 matches thread. rho's family value
  κ = 0.602 (FanOutIntensity's α0 = 0.301 a FLOP, with the activation gradient's products counted, c = 2) makes rho's
  1/g 1.81× mine at its defaults.
* **Compute.** Mine runs every GPU at σ = 1 for serving and training alike; rho has `mfu_train` (training work) and
  `mfu_serve` (how many declared blocks the certified time holds), both 0.4. Matched at 1.
* **The declared block.** Mine is c = k φ_F with φ_F = 2 × 49e9 / 17 FLOP per novel bit (17-bit tokens): 1.153e17 FLOP
  at k = 2e7. rho's Γ = 9.8e16 is a 1M-token session (Γ/k = 4.9e9 FLOP a bit, not φ_F). Matched to mine.
* **U's steady share.** Mine is r̂ + ε_W(1 − r̂) + p_fail/δ, with r̂ = 0 (the fleet sized to its committed work at full
  use) and p_fail/δ = 2.73e-12/1e-6; rho's `residual` 0.505 (today, nothing credited) and `eps_agg` 1e-4 are matched to
  0 and 2.73e-6. At its defaults rho's remote 1/g is therefore 1.96, the residual's, against my 131. The verdict lag
  JΔ/H is the same term, over a 365-day year here and a Julian one in mine (9e-8 apart). Mine also counts what sampled
  proofs let through at δ (escapes, and injected outputs: 5e-9 of U for fan, 1.6e-6 for thread, 3.5e-5 for cover at
  u = 16), which rho's `cheat` = 0 leaves out: it is most of the remote gaps, all under 0.1%.
* **x_lead.** Mine takes the token-reachable leading-layer products out of fan's count, dividing its allowance by 1 − x
  with x = max(0.008, K / (n_total × 2.5)): the share of parameters whose update at C1's floor fits in K. rho's
  fanout-priced has no such share. At the inference block x = 0.8% hardly moves the U-bound remote D; at the H200
  server-window x = 0.139, so rho's onsite 1/g is mine / (1 − x), 16% higher.
* **The whole update.** Mine refuses a block whose K holds the whole update at 2.5 bits a parameter (4e12 bits); rho's
  fanout-priced has no cutoff, and its state-fits attack takes Adam's state, 16 bits a parameter (2.56e13). So on the
  NVL72 server-window (K = 8.3e12) mine has no onsite bound and falls back to the remote D, while rho's fanout-priced
  prices it at about 9,500 with no attack against it. At `state_bits_per_param` = 2.5, the low end of its range, rho's
  attack is my cutoff.
* **I_cert.** Both are the location's share of the committed requests' input tokens at 17 bits, Φ σ / φ_F. Mine is at
  σ = 1 (2.75e6 bit/s on H200); rho's `ingress` 1.1e6 is at mfu_serve = 0.4, the fleet's count being sized at it, and is
  a number: it doesn't follow an override of mfu_serve or the count. Matched to Φ / φ_F. Mine's R is Daniel's 5 GB per
  GPU, rho's low end; rho's value is today's 10 GB.
* **β_x.** Mine is 0.66, the cheapest boundary (d = 7168, FP8); rho's 0.87 is the exact-block measurement on
  Qwen2.5-1.5B. Matched to 0.66.
* **Thread** uses Kimi K2's d = 10 and e = 37,248 in rho's family; mine uses d = 1 (what Lean certifies for a run that
  trains its embedding) and DeepSeek's e = 43,392. Matched to mine.
* **Cover.** Mine counts a cover unit as a Dot16 step, u = 16 scalar products (D = 7.15, a flagged normalization);
  rho has no u, so it matches mine at u = 1 (D = 63.2).
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
THREAD = {"kappa": 2 / 3, "thread_d": 1, "thread_e": 43392}

#: the reference-DC model (art:dee8d74e…) at its defaults: a year, δ = 1e-6, σ = 1, r̂ = 0, R = 5 GB per GPU, β_x = 0.66
MINE = {
    "remote fan": 131.143,
    "remote thread": 123.840,
    "remote cover u=1": 63.178,
    "onsite fan h200": 5354.48,
    # the H200 server-window: compute σ Φ × 1 day, inputs R + I_cert + side, and fan's x_lead there
    "C h200": 1.36788e21,
    "K h200": 5.57373e11,
    "x_lead h200": 0.139343,
    # above the whole update every premise fails, so onsite falls back to the remote cut
    "onsite fan nvl72": 131.143,
}


def matched(fleet: str, regime: str, onsite: bool = False, **family: float) -> Registry:
    reg = Registry.load()
    values = {f"fleet.{fleet}.mfu_train": 1, f"fleet.{fleet}.mfu_serve": 1, f"regime.{regime}.residual": 0,
              f"regime.{regime}.eps_agg": SLACK, "policy.v4-pro-session.gamma": reg["policy.v4-pro-session.k"] * PHI_F,
              "family.full.kappa": 1 / 3, "family.full.beta_x": 0.66}
    values |= {f"family.full.{k}": v for k, v in family.items()}
    if onsite:
        phi = reg[f"location.{reg[f'fleet.{fleet}.location']}.phi"]
        values |= {f"regime.{regime}.residual_bits": 4e10, f"regime.{regime}.ingress": phi / PHI_F}
    return reg.override(values)


def inverse_g(r: Result, argument: str, of: str = "bounds") -> float:
    (f,) = r.families
    (e,) = [e for e in getattr(f, of) if e.argument == argument]
    return 1 / e.slowdown.g


@pytest.mark.parametrize("fleet", ["h200", "nvl72"])
@pytest.mark.parametrize(("argument", "family", "mine"), [
    ("fanout-priced", {}, MINE["remote fan"]),
    ("threaded", THREAD, MINE["remote thread"]),
    ("cover", {}, MINE["remote cover u=1"]),
])
def test_remote_at_no_residual_matches_mine(fleet, argument, family, mine):
    """U-bound: 1/g ≈ 1/(ε_W + ε' + JΔ/H + 1/ρ_inference), the same on both fleets. Within 0.1%; what is left is the
    escapes and injected outputs mine counts, the year's length and fan's x_lead at the inference block."""
    r = evaluate_named(matched(f"world-{fleet}", "reference-remote", **family), f"reference-remote-{fleet}")
    assert math.isclose(inverse_g(r, argument), mine, rel_tol=1e-3)


def test_onsite_h200_fanout_priced_is_mine_without_x_lead():
    """The server-window binds: 1/g = (κ C/2) / (fanout K/β_x), mine's ρ_sw times 1/(1 − x_lead), within 0.1%. The
    window's compute and inputs (R = 5 GB per GPU, I_cert at σ = 1, 1 kb/s of side channels) are mine within 0.1%."""
    s = build(matched("world-h200", "reference-onsite-h200", onsite=True), "reference-onsite-h200")
    budget = s.regime.budget(s.context)
    assert math.isclose(budget.compute, MINE["C h200"], rel_tol=1e-3)
    assert math.isclose(budget.inputs, MINE["K h200"], rel_tol=1e-3)
    rho = inverse_g(evaluate(s), "fanout-priced")
    assert math.isclose(rho * (1 - MINE["x_lead h200"]), MINE["onsite fan h200"], rel_tol=1e-3)


def test_onsite_h200_threaded_is_the_remote_cut():
    """Thread is quadratic at the server-window, so the physical cut doesn't bind, in rho as in mine."""
    r = evaluate_named(matched("world-h200", "reference-onsite-h200", onsite=True, **THREAD), "reference-onsite-h200")
    assert math.isclose(inverse_g(r, "threaded"), MINE["remote thread"], rel_tol=1e-3)


def test_at_its_defaults_rho_differs_from_mine_by_the_named_factors():
    """rho's own onsite H200 fan-out-priced 1/g is mine times κ/(1/3) (1.81), its window compute over mine (mfu_train
    0.4, and Φ's rounding), β_x/0.66 (1.32), mine's K over its own (0.76: R = 10 GB against 5 GB, I_cert at mfu_serve
    0.4) and 1/(1 − x_lead) (1.16): nothing else, within 0.1%."""
    reg = Registry.load()
    s = build(reg, "reference-onsite-h200")
    budget = s.regime.budget(s.context)
    factors = (3 * reg["family.full.kappa"] * budget.compute / MINE["C h200"] * reg["family.full.beta_x"] / 0.66
               * MINE["K h200"] / budget.inputs / (1 - MINE["x_lead h200"]))
    assert math.isclose(inverse_g(evaluate(s), "fanout-priced"), MINE["onsite fan h200"] * factors, rel_tol=1e-3)


def test_the_nvl72_server_window_holds_the_whole_update():
    """Mine has no onsite bound on NVL72 and falls back to the remote D; rho's fanout-priced still prices the window.
    rho's state-fits attack at 2.5 bits a parameter is my cutoff: it gives mine within 1% (it leaves out the declared
    blocks' own training, 1/ρ_inference, 0.9% of g) and so contradicts fanout-priced there."""
    reg = matched("world-nvl72", "reference-onsite-nvl72", onsite=True)
    s = build(reg, "reference-onsite-nvl72")
    assert s.regime.budget(s.context).inputs >= reg["model.deepseek-v4-pro.n_total"] * WHOLE_UPDATE
    assert inverse_g(evaluate(s), "fanout-priced") > 10 * MINE["onsite fan nvl72"]
    (f,) = evaluate_named(reg.override({"family.full.state_bits_per_param": WHOLE_UPDATE}),
                          "reference-onsite-nvl72").families
    (attack,) = [a for a in f.attacks if a.argument == "state-fits"]
    assert math.isclose(1 / attack.slowdown.g, MINE["onsite fan nvl72"], rel_tol=0.01)
    assert "fanout-priced" in f.contradictions


@pytest.mark.parametrize("fleet", ["h200", "nvl72"])
def test_ingress_is_the_served_share_of_input_tokens(fleet):
    """`ingress` is I_cert = Φ mfu_serve / φ_F (the fleet's count is sized at mfu_serve), within 0.2%."""
    reg = Registry.load()
    phi = reg[f"location.{reg[f'fleet.world-{fleet}.location']}.phi"]
    mine = phi * reg[f"fleet.world-{fleet}.mfu_serve"] / PHI_F
    assert math.isclose(reg[f"regime.reference-onsite-{fleet}.ingress"], mine, rel_tol=2e-3)
