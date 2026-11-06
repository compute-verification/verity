"""training_slowdown.formalism and training_slowdown.policy: Fit and Slowdown as the formalism v2 states them (a frozen tail always
charged), the attack side beside them, the stubs, assumptions and upper bounds labelled, the grid."""
import dataclasses
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from training_slowdown import VERSION as F_VERSION
from training_slowdown import formalism as F
from training_slowdown import frontier, policy
from training_slowdown.gates import per_mac

SHAPE = {"d_model": 512, "n_layer": 6, "vocab": 50304, "ctx": 1024, "parameters": 45_029_888, "tokens": 154_206_208,
         "forward_macs_per_token": 51_342_080}
FIXTURES = Path(__file__).parents[1] / "capture" / "tests" / "fixtures"


def measured(u1=1e8, sigma=0.8, delta=None, dgrad=None, decoder=0.0, tail=0.0, affected=0.5) -> F.Measured:
    """One part of 1e12 gates and 1e3 token blocks, σ of the gates in them and `affected` of those affected; no decoder
    (d = 0) unless given."""
    u, u_in = F.tail(u1, tail)
    part = F.Part(gates=1e12, tokens=1e3, token_gates=sigma * 1e12, affected_gates=affected * sigma * 1e12,
                  delta_grad_bits=dgrad)
    return F.Measured((part,), tau_bits=32, a_bits=8192, u=u, u_in=u_in,
                      delta_bits=delta if delta is not None else u1, decoder_gates=decoder, tail_share=tail)


def test_even_update_slowdown_is_the_corollary():
    """Evenly spread with u(1) >> k: x0 <= k/u(1), s* <= k^2/(4 a u(1)), so rho ≳ 4 σ s a u(1) / k^2."""
    m, c, k = measured(u1=1e9), 1e13, 1e5
    r = F.at(m, c, k)
    s = c / 0.8e9
    assert (r["sigma"], r["s"]) == pytest.approx((0.8, s))
    assert r["x0"] == pytest.approx(k / 1e9)
    assert r["s_star"] == pytest.approx(k * k / (4 * 1e9) / 8192)
    assert r["rho"] == pytest.approx(4 * 0.8 * s * 8192 * 1e9 / k ** 2, rel=1e-3) and not r["fit"]


def test_a_frozen_tail_is_always_charged():
    """Frozen layers above changed ones with u(1) > k: u_in jumps at once, so x0 = 0 and the tail's work stays charged;
    s* <= t k / a, so rho ≈ σ s a / (t k), below the evenly spread update's (the tail leaves a block more room)."""
    t, k, u1, c = 0.05, 1e5, 1e9, 1e13
    r = F.at(measured(u1=u1, tail=t), c, k)
    assert r["x0"] == 0 and r["s_star"] == pytest.approx(t * k / 8192)
    assert r["rho"] == pytest.approx(0.8 * (c / 0.8e9) * 8192 / (t * k), rel=1e-6)
    assert F.at(measured(u1=u1), c, k)["rho"] > r["rho"]


def test_the_tail_profiles():
    """u: 0 up to t, then the changed update evenly over the rest; u_in: the full update at once; at t = 0, both even."""
    t, u1, k = 0.2, 1e9, 1e5
    u, u_in = F.tail(u1, t)
    assert F.u_at(u, 0.1) == 0 and F.u_at(u, 0.6) == pytest.approx(0.5 * u1) and F.u_at(u, 1.0) == u1
    assert F.u_at(u_in, 1e-9) == u1 and F.bottom_share(u_in, k) == 0 and F.bottom_share(u_in, u1) == 1
    assert F.room(u, k) == pytest.approx(t * k)
    assert F.tail(u1, 0.0) == (F.even(u1), F.even(u1))
    assert F.room(u_in, k) == 0 and F.room(F.with_tail(u_in, 0.0), 2 * u1) == pytest.approx(u1)


def test_varying_token_blocks_take_the_largest_per_token_affected_count():
    """Two parts whose token blocks differ: the token form holds with α_max in place of a common count,
    ρ ≥ (1 − x0) c Σ|T ∩ A| / (α_max |C| s*) = σ (1 − x0) (s/s*) (ᾱ/α_max)."""
    a = F.Part(gates=1e12, tokens=1e3, token_gates=0.9e12, affected_gates=0.8e12, delta_grad_bits=None)
    b = F.Part(gates=1e12, tokens=1e3, token_gates=0.5e12, affected_gates=0.2e12, delta_grad_bits=None)
    u = F.even(1e9)
    m = F.Measured((a, b), 32, 8192, u, u, 1e9)
    c, k = 1e13, 1e5
    r = F.at(m, c, k)
    s_star = F.room(u, k) / 8192
    alpha_max = 0.8e12 / 1e3
    assert r["alpha_ratio"] == pytest.approx((1e12 / 2e3) / alpha_max)
    assert r["slowdown"] == pytest.approx((1 - k / 1e9) * c * 1e12 / (alpha_max * 2e12 * s_star))
    assert r["slowdown"] == pytest.approx(m.sigma * (1 - k / 1e9) * (c / m.w) / s_star * r["alpha_ratio"])
    same = F.Measured((a, a), 32, 8192, u, u, 1e9)
    assert F.at(same, c, k)["alpha_ratio"] == 1


def test_fit_is_three_kinds_of_block_and_gives_c_over_c_minus_d():
    """Training fits when compute (Δ + sτ), reduction (2Δ_grad) and update application (Δ + Δ_grad) each fit in k;
    ρ is then c/(c − d), 1 with no decoder (d = 0); an unmeasured decoder fits no block. There Δ ≤ k bounds u and u_in
    (`capped`), so x0's bound is 1 and Slowdown's bound 0."""
    c, k = 1e13, 1e6
    s_tau = c / 0.8e9 * 32
    fits = F.at(measured(u1=1e5, dgrad=1e5, decoder=1e12), c, k)
    assert fits["fit"] and fits["rho"] == pytest.approx(c / (c - 1e12))
    assert F.at(measured(u1=1e5, dgrad=1e5), c, k)["rho"] == 1.0
    unmeasured = F.at(measured(u1=1e5, dgrad=1e5, decoder=None), c, k)
    assert not unmeasured["fit"] and not unmeasured["delta_counted"] and unmeasured["rho_fit"] == math.inf
    kinds = {"compute": measured(u1=k - s_tau + 1, dgrad=1),
             "reduction": measured(u1=1, dgrad=k / 2 + 1),
             "update": measured(u1=5.5e5, dgrad=4.6e5)}
    for name, m in kinds.items():
        got = F.at(m, c, k)["parts"][0]["fit"]
        assert not got[name] and all(v for n, v in got.items() if n != name), name
    assert not F.at(measured(u1=1e5), c, k)["fit"]  # Δ_grad unmeasured: training Fit unchecked
    for tail in (0.0, 0.5):
        wide = F.at(measured(u1=1e8, delta=1e3, dgrad=1e3, tail=tail), c, k)
        assert wide["fit"] and wide["rho"] == 1.0 and wide["x0"] == 1.0 and wide["slowdown"] == 0.0


def test_u_and_u_in_are_at_most_delta_where_delta_counts():
    """A profile above Δ is cut at Δ (a block reads the update code instead), at the crossing point of each segment; a
    decoder as wide as c leaves it as given."""
    u = ((0.0, 0.0), (0.5, 10.0), (0.75, 40.0), (1.0, 40.0))
    cut = F.capped(u, 20.0)
    assert [x for x, _ in cut] == pytest.approx([0.0, 0.5, 0.5 + 0.25 / 3, 0.75, 1.0])
    assert [b for _, b in cut] == [0.0, 10.0, 20.0, 20.0, 20.0]
    assert F.capped(u, 50.0) == u and F.capped(u, 10.0)[1:] == ((0.5, 10.0), (0.75, 10.0), (1.0, 10.0))
    part = F.Part(gates=1e12, tokens=1e3, token_gates=0.8e12, affected_gates=0.4e12)
    c, k = 1e13, 30.0
    r = F.at(F.Measured((part,), 32, 8192, u, u, 20.0, decoder_gates=0.0), c, k)
    assert r["x0"] == 1.0 and r["s_star"] == pytest.approx((k - 20.0) / 8192)
    wide = F.at(F.Measured((part,), 32, 8192, u, u, 20.0, decoder_gates=c), c, k)
    assert wide["x0"] == pytest.approx(0.5 + 0.25 * 2 / 3)


def test_fit_k_is_the_least_k_at_which_fit_holds():
    """Running θ': Δ + (c/w_run)τ; training: the most of its three kinds (None without Δ_grad), a part with no token
    blocks (the setup) needing Δ alone; none when the decoder reaches c or is unmeasured."""
    m, c, w_run = measured(u1=1e6, delta=1e3, dgrad=1e4), 1e13, 2e8
    s_tau = c / 0.8e9 * 32
    run, train = F.fit_k(m, c, w_run)
    assert (run, train) == pytest.approx((1e3 + c / w_run * 32, max(1e3 + s_tau, 2e4, 1e3 + 1e4)))
    assert F.at(m, c, train)["fit"] and not F.at(m, c, train * (1 - 1e-9))["fit"]
    setup = F.Part(gates=1e9, tokens=0, token_gates=0.0, affected_gates=0.0)
    assert F.fit_k(F.Measured(m.parts + (setup,), 32, 8192, m.u, m.u_in, 1e3, 0.0), c, w_run) == pytest.approx(
        (run, train))
    assert F.fit_k(measured(u1=1e6), c, w_run)[1] is None
    assert F.fit_k(measured(u1=1e6, decoder=c), c, w_run) == (None, None)
    assert F.fit_k(measured(u1=1e6, delta=1e3, dgrad=1e4, decoder=None), c, w_run) == (None, None)


def test_update_application_reads_delta_t_where_the_run_has_latents():
    """A straight-through run's update application reads its fp32 latents Δ_t, not its code Δ (coordinator 00:47Z):
    Δ_t + Δ_grad ≤ k, while compute still reads Δ; a run with no Δ_t reads Δ there."""
    c, k = 1e13, 1e6
    code = measured(u1=1e5, dgrad=1e5)
    latent = dataclasses.replace(code, delta_t_bits=k)
    assert F.at(code, c, k)["fit"] and code.delta_t == code.delta_bits
    assert F.at(latent, c, k)["parts"][0]["fit"] == {"compute": True, "reduction": True, "update": False}
    assert F.fit_k(latent, c, 2e8)[1] == pytest.approx(k + 1e5)
    assert F.from_json(latent.to_json()).delta_t_bits == k


def test_the_attack_side_is_reported_beside_the_bounds():
    """Best-known values from exhibited blocks: x0 1 where the compute block fits; s* from the frozen tail's block,
    t·min(⌊k/(a + τ)⌋, ⌊c/(t α_max)⌋), at most A1's bound; none without a tail."""
    c, k = 1e13, 1e5
    tail = F.at(measured(u1=1e9, tail=0.05), c, k)
    assert tail["x0_known"] == 0 and tail["s_star_known"] == pytest.approx(0.05 * math.floor(k / (8192 + 32)))
    assert tail["s_star_known"] <= tail["s_star"]
    narrow = F.at(measured(u1=1e9, tail=0.05), 1e8, k)
    assert narrow["s_star_known"] == pytest.approx(0.05 * math.floor(1e8 / (0.05 * 0.4e9)))
    assert F.at(measured(u1=1e9), c, k)["s_star_known"] == 0
    fits = F.at(measured(u1=1e3, tail=0.05), c, 1e6)
    assert (fits["x0_known"], fits["s_star_known"]) == (1, 0)


def charged(m: F.Measured, h: float, positions: int = 1024, kv: float = 2 * 512 * 16) -> F.Measured:
    return replace(m, context={"kv_bits": kv, "positions": positions, "kv_layer": 0, "free_share": h})


def test_the_context_charge_is_its_formula_on_a_toy_profile():
    """u(x) = 8000x, k = 5000, a = 1000, L = 4: x(k − u(x)) peaks at x* = 5/16 with u(x*) = 2500, so S1(p) is v2's
    s* = 0.78125 while a(p + 1) ≤ 2500 and then the limit's x(k − u(x)), 750 and 500 over a; S0 at h = 0.2 is
    0.2 (5000 − 1600)/1000. u_in jumps, so x0 = 0, and ρ is σ s times the positions' mean of
    h/max(S0, S1(p)) + (1 − h)/S1(p)."""
    part = F.Part(gates=1e12, tokens=1e3, token_gates=0.8e12, affected_gates=0.4e12)
    m = F.Measured((part,), tau_bits=32, a_bits=1000, u=F.even(8000), u_in=((0, 0), (0, 8000), (1, 8000)),
                   delta_bits=8000)
    s1, s0 = [0.78125, 0.78125, 0.75, 0.5], 0.2 * 3400 / 1000
    assert F.context_rooms(m.u, 5000, 1000, 4) == pytest.approx(s1)
    assert F.room(m.u, 5000, 0.2) / 1000 == pytest.approx(s0)
    mean = sum(0.2 / max(s0, s) + 0.8 / s for s in s1) / 4
    r = F.at(charged(m, 0.2, positions=4, kv=2000), 1e12, 5000)
    assert r["x0"] == 0 and r["s_star_a1"] == pytest.approx(0.78125) and r["s_star"] == pytest.approx(1 / mean)
    assert r["slowdown"] == pytest.approx(0.8 * 1250 * mean) and r["rho"] == r["slowdown"]
    v2 = F.at(m, 1e12, 5000)["slowdown"]
    assert v2 == pytest.approx(0.8 * 1250 / 0.78125) and v2 < r["slowdown"]
    with pytest.raises(ValueError, match="best span"):
        F.at(charged(m, 0.2, positions=4, kv=999), 1e12, 5000)


def test_a_narrow_cut_whose_best_block_holds_the_sequence_keeps_v2s_rho():
    """a = 32 and a frozen tail: v2's best block holds k/a = 2500 tokens, more than L = 1024, so every position's
    S1(p) is s* and the charge changes nothing."""
    m = replace(measured(u1=1e9, tail=0.3), a_bits=32)
    v2, r = F.at(m, 1e13, 8e4), F.at(charged(m, 0.6), 1e13, 8e4)
    assert r["s_star"] == pytest.approx(v2["s_star"]) == r["s_star_a1"] and r["rho"] == pytest.approx(v2["rho"])


def test_a_dense_cut_is_unbounded_where_k_holds_few_tokens():
    """a = d_model·16 at k = 80,000 bits: a block doing a token's charged work must hold every earlier token, so none
    serves a position past k/a − 1 ≈ 8.8, and the charged share 1 − h left there makes ρ infinite (s* 0). At k = 10^8
    the block holds the sequence and ρ is v2's. With no affected K/V (h = 1) nothing is charged."""
    m = measured(u1=1e9, tail=0.3)
    r = F.at(charged(m, 0.6), 1e13, 8e4)
    assert r["s_star"] == 0 and r["rho"] == math.inf and r["s_star_a1"] == pytest.approx(0.3 * 8e4 / 8192)
    assert F.at(charged(m, 0.6), 1e13, 1e8)["rho"] == pytest.approx(F.at(m, 1e13, 1e8)["rho"])
    assert F.at(charged(m, 1.0), 1e13, 8e4)["rho"] == pytest.approx(F.at(m, 1e13, 8e4)["rho"])
    assert F.at(charged(replace(m, u_in=F.even(1e3)), 0.6), 1e13, 8e4)["slowdown"] == 0


def test_the_context_quantities_from_the_changed_tensors():
    """ℓ_kv: the embedding 0, a layer's wk (its own K/V) that layer, its wq the next; the top layer's MLP and the final
    norm none (h 1). h's upper bound: three times the context-free forward over that plus twice the charged forward,
    less ℓ_kv's Q, K and V projections where its input depends on θ alone."""
    layer, head, d = (SHAPE["parameters"] - 50304 * 512 - 512) / 6, 50304 * 512, 512
    emb = F.context(["embed", "blocks.3.wq.weight"], SHAPE)
    assert emb == {"kv_bits": 2 * 512 * 16, "positions": 1024, "kv_layer": 0,
                   "free_share": pytest.approx(3 * head / (3 * head + 2 * 6 * layer))}
    wq = F.context(["blocks.2.wq.weight", "blocks.4.wk.weight"], SHAPE)
    assert wq["kv_layer"] == 3 and wq["free_share"] == pytest.approx(
        3 * (layer + head) / (3 * (layer + head) + 2 * 3 * layer))
    wk = F.context(["blocks.2.wk.weight"], SHAPE)
    assert wk["kv_layer"] == 2 and wk["free_share"] == pytest.approx(
        3 * head / (3 * head + 2 * (4 * layer - 3 * d * d)))
    for top in (["blocks.5.w_down.weight", "norm"], []):
        assert F.context(top, SHAPE)["kv_layer"] is None and F.context(top, SHAPE)["free_share"] == 1
    with pytest.raises(ValueError, match="lm_head"):
        F.context(["lm_head.weight"], SHAPE)
    m = F.complete({"context": emb}, {"output": {"e_bytes": 10}, "macs_log": {"by": {}, "ops": {}}}, SHAPE)
    assert m.context == emb and m.labels["context"] == F.CONTEXT and F.from_json(m.to_json()).context == emb


def test_fit_reads_delta_and_slowdown_reads_u1_as_separate_fields():
    """A total update (Fit's Δ) larger than the per-token update (Slowdown's u(1)), as for an update each token's path
    reads only part of: the compute block fails on Δ while the Slowdown bound reads u(1). With a small Δ the compute
    block fits and Δ bounds u too. The record says which value each came from."""
    row = {"output": {"e_bytes": 10**9, "rebuild_macs": 0}, "macs_log": {"by": {}, "ops": {}}}
    part = {"gates": 1e12, "tokens": 1e3, "token_gates": 1e12, "affected_gates": 5e11, "steps": 1,
            "delta_grad_bits": None}
    f = {"parts": [part], "u1_bits": 1e9, "labels": {"u1_bits": "upper bound (witness): a test"}}
    m = F.complete(f, row, SHAPE)
    assert (m.delta_bits, m.u1_bits, m.u, m.u_in, m.tail_share) == (8e9, 1e9, F.even(1e9), F.even(1e9), 0.0)
    assert m.labels["u1_bits"] == "upper bound (witness): a test" and m.labels["u"] == F.STUBS["u"]
    assert m.labels["u"].startswith("assumption (A1)") and m.labels["u_in"].startswith("assumption (A1')")
    k = 1e6
    r = F.at(m, 1e13, k)
    assert not r["parts"][0]["fit"]["compute"] and r["x0"] == pytest.approx(k / 1e9)
    assert r["rho"] == pytest.approx(4 * 1e4 * 512 * 16 * 1e9 / k ** 2, rel=2e-3)
    small = F.at(F.complete(f, row | {"output": {"e_bytes": 10, "format": "full"}}, SHAPE), 1e13, k)
    assert small["parts"][0]["fit"]["compute"] and not small["fit"] and small["x0"] == 1.0 and small["rho"] == 1.0
    with pytest.raises(ValueError):
        F.complete({"parts": [part], "u": [[0, 0], [1, 5.0]], "u1_bits": 6.0}, row, SHAPE)


def test_a_block_s_tail_share_gives_the_tail_profiles_labelled_as_assumptions():
    row = {"output": {"e_bytes": 10}, "macs_log": {"by": {}, "ops": {}}}
    part = {"gates": 1e12, "tokens": 1e3, "token_gates": 1e12, "affected_gates": 5e11}
    m = F.complete({"parts": [part], "u1_bits": 1e9, "tail_share": 0.25}, row, SHAPE)
    assert m.tail_share == 0.25 and (m.u, m.u_in) == F.tail(1e9, 0.25)
    assert m.labels["u"].startswith("assumption (A1)") and m.labels["u_in"].startswith("assumption (A1')")
    assert F.from_json(m.to_json()) == m
    with pytest.raises(ValueError):
        F.complete({"parts": [part], "u1_bits": 1e9, "tail_share": 1.0}, row, SHAPE)


def test_a_te_core_4_block_s_dep_gates_are_its_affected_gates():
    row = {"output": {"e_bytes": 10}, "macs_log": {"by": {}, "ops": {}}}
    m = F.complete({"parts": [{"gates": 1e12, "tokens": 1e3, "dep_gates": 5e11}]}, row, SHAPE)
    assert m.parts[0].affected_gates == 5e11 and m.parts[0].token_gates == 1e12
    assert m.labels["token_gates"] == F.STUBS["token_gates"]


def test_a_decoder_as_wide_as_c_counts_no_update():
    r = F.at(measured(u1=1e6, dgrad=1e6, decoder=1e13), 1e13, 1e9)
    assert r["delta_counted"] is False and not r["parts"][0]["fit"]["compute"] and not r["fit"]


@pytest.mark.parametrize("output, d, source", [
    ({"decoder_gates": 5e9, "rebuild_macs": 10}, 5e9, "notes"),
    ({"rebuild_macs": 4 * 10 ** 6, "format": "seedlog"}, 4e6 * 12637, "witness"),
    ({"rebuild_macs": 0, "format": "lora"}, 0.0, "none"),
    ({"format": "full"}, 0.0, "none"),
    ({"format": "lora"}, None, "unmeasured"),
])
def test_the_decoder_is_the_notes_else_the_witness_rebuild_and_0_only_without_one(output, d, source):
    """seed-mezo's 04:15Z finding: d is the program's notes.decoder gates for the checkpoint's E, else
    prepare.rebuild_macs(θ, E) at F32Fma_v4's 12,637 gates a multiply-add, 0 only where E stores θ' directly; with none,
    Fit is refused. The block's own d stands over all of them."""
    row = {"output": {"e_bytes": 10} | output, "macs_log": {"by": {}, "ops": {}}}
    part = {"gates": 1e12, "tokens": 1e3, "token_gates": 1e12, "affected_gates": 5e11, "delta_grad_bits": 10.0}
    m = F.complete({"parts": [part], "u1_bits": 80.0}, row, SHAPE)
    assert m.decoder_gates == d and m.labels["decoder_gates"] == F.DECODER[source]
    assert F.at(m, 1e13, 1e6)["fit"] is (d is not None)
    own = F.complete({"parts": [part], "u1_bits": 80.0, "decoder_gates": 7.0}, row, SHAPE)
    assert own.decoder_gates == 7.0 and own.labels["decoder_gates"] == "the record's block: its decoder_gates"
    if d:
        assert F.at(m, 1e13, 1e6)["rho"] == pytest.approx(1e13 / (1e13 - d))
        assert not F.at(m, 0.5 * d, 1e6)["fit"]


def traced(trace: dict, e_bytes=10) -> dict:
    """A traced row: one part with affected gates, and the trace's quantities (`F.of_trace`'s `trace`)."""
    f = {"tail_share": 0.0, "u_in": [[0, 0], [1, 80.0]], "u1_bits": 80.0, "tau_ids": 1.0, "tau_other_bits": 0.0,
         "affected_share": 0.5, "data_dependent": [], "carried_scalars": []} | trace
    block = F.of_trace(f) | {"parts": [{"gates": 1e12, "tokens": 1e3, "token_gates": 1e12, "affected_gates": 5e11,
                                        "delta_grad_bits": 10.0}]}
    return {"id": "r", "start": "s", "output": {"e_bytes": e_bytes, "format": "full"}, "formalism": block,
            "checkpoint_step": 3}


CLIP = {"kind": "a comparison of a read-back float", "where": "train.py:399", "steps": [0, 4], "count": 5}
EMA = {"kind": "a Python number read back in several earlier steps", "where": "train.py:395", "steps": [2, 4],
       "count": 3, "earlier_steps": 4}


@pytest.mark.parametrize("trace, e_bytes, reason", [
    ({"data_dependent": [CLIP]}, 10, "data-dependent: not measured (a comparison of a read-back float at train.py:399, "
                                     "steps 0 to 4)"),
    ({"carried_scalars": [EMA]}, 10, "carried Python number: Delta_grad grows with the step"),
    ({"affected_share": 0.0, "u1_bits": 0.0, "u_in": [[0, 0], [1, 0.0]]}, 10,
     "update-free trace: E is 10 bytes, but the traced step shows no update (its affected share is 0 and u(1) is 0)"),
    ({"u1_bits": 0.0, "u_in": [[0, 0], [1, 0.0]]}, None, "update-free trace: E is of unknown size"),
    ({"data_dependent": None}, 10, "read-backs unchecked: traced before training_slowdown 4.2 (no data_dependent)"),
])
def test_a_trace_the_capture_cannot_vouch_for_is_refused_never_scored(trace, e_bytes, reason):
    """seed-mezo's 04:15Z LOOPHOLE: a clip on a read-back estimate left the trace with no update (affected share 0,
    u(1) 0), which scored ρ = 1 whatever the update. A data-dependent capture, a Python number folded over steps, a
    nonempty E with no update in its trace and a trace without the read-back lists are each refused with the reason,
    and the frontier leaves the row out with it."""
    row = traced({k: v for k, v in trace.items() if v is not None}, e_bytes)
    if trace.get("data_dependent", []) is None:
        del row["formalism"]["trace"]["data_dependent"]
    assert F.refusal(row).startswith(reason)
    with pytest.raises(F.Refused, match="^" + reason.split(" (")[0]):
        F.measure(row, SHAPE)
    scored, skipped = frontier.formalism_rows([row], {"s": SHAPE})
    assert scored == [] and skipped["r"] == frontier.REFUSED + F.refusal(row)


def test_an_empty_e_with_no_update_is_scored_and_a_nonempty_one_with_an_update_free_prefix_is_not():
    """E = 0 bytes (the start itself) may show no update; a nonempty E whose prefix reads θ' nowhere (a checkpoint
    taken before its first update reads one) is refused."""
    none = {"affected_share": 0.0, "u1_bits": 0.0, "u_in": [[0, 0], [1, 0.0]]}
    empty = traced(none, e_bytes=0)
    empty["formalism"]["parts"][0]["affected_gates"] = 0.0
    assert F.refusal(empty) is None and F.measure(empty, SHAPE).delta_bits == 0
    early = traced({})
    early["formalism"]["parts"][0]["affected_gates"] = 0.0
    assert F.refusal(early) is None
    with pytest.raises(F.Refused, match="update-free prefix: E is 10 bytes, but no step through checkpoint 3 reads"):
        F.measure(early, SHAPE)
    assert F.measure(traced({}), SHAPE).decoder_gates == 0.0


def test_a_v23_row_is_measured_from_its_old_fields_with_every_stub_named():
    fwd = 3 * SHAPE["forward_macs_per_token"]
    row = {"start": "s50", "output": {"e_bytes": 1000}, "annotation": {"steps": 3},
           "macs_log": {"by": {"fwd/theta": fwd, "fwd/updated": 10, "bwd/grad": 20, "other/param": 7},
                        "ops": {"fwd theta mm [[1]]": [1, fwd], "fwd updated mm [[1]]": [1, 10],
                                "bwd grad mm [[1]]": [1, 20], "other param _fused_adamw_ [[1]]": [1, 7]}}}
    m = F.measure(row, SHAPE)
    assert m.tokens == pytest.approx(3 + 10 / SHAPE["forward_macs_per_token"])
    assert m.parts[0].affected_gates == pytest.approx(10 * per_mac("gemm/bf16"))
    assert m.parts[0].token_gates == pytest.approx((fwd + 30) * per_mac("gemm/bf16")) and m.sigma < 1
    assert m.delta_bits == 8000 and m.u == m.u_in == ((0.0, 0.0), (1.0, 8000))
    assert m.tau_bits == 16 and m.a_bits == 512 * 16 and m.u1_bits == 8000
    assert set(m.labels) == {"gates", "tokens", "token_gates", "affected_gates", "tau_bits", "a_bits", "delta_bits",
                             "decoder_gates", "u", "u_in", "u1_bits", "delta_grad_bits", "ops"}
    assert all(m.labels[k].startswith("stub") for k in ("tokens", "affected_gates", "tau_bits"))
    assert m.labels["ops"].startswith("counted") and m.ops[0]["products"] == []
    assert m.decoder_gates is None and m.labels["decoder_gates"] == F.DECODER["unmeasured"]
    assert all(m.labels[k].startswith("upper bound") for k in ("a_bits", "u1_bits", "delta_bits"))
    assert all(m.labels[k].startswith("assumption") for k in ("u", "u_in"))
    with pytest.raises(ValueError):
        F.measure(row | {"output": {}}, SHAPE)


def test_a_record_with_te_core_4_cuts_reads_them_as_delta_grad():
    record = {"steps": [{"step": 0, "tokens": 8, "struct": "a", "by": {"fwd/theta": 80}},
                        {"step": 1, "tokens": 8, "struct": "a", "by": {"fwd/theta": 80}},
                        {"step": 2, "tokens": 8, "struct": "b", "by": {"fwd/theta": 40}}],
              "structures": {"a": {"updated": [[2, 0, 16, 8, 16]], "log": [["fwd", "theta", "mm", "[[8]]", 1, 80],
                                                                         ["other", "param", "mul", "[[8]]", 1, 4]],
                                   "cut_bits": 512},
                             "b": {"updated": [], "log": [["fwd", "theta", "mm", "[[8]]", 1, 40]], "cut_bits": 0}},
              "formalism": {"a_bits": 99, "labels": {"a_bits": "measured: the traced cut"}}}
    f = F.of_record(record)
    a, b = f["parts"]
    assert (a["steps"], a["tokens"], a["delta_grad_bits"]) == (2, 16, 512)
    assert a["affected_gates"] == pytest.approx(2 * 16 * per_mac("gemm/bf16"))
    assert a["token_gates"] == pytest.approx(160 * per_mac("gemm/bf16")) and a["token_gates"] < a["gates"]
    assert (b["steps"], b["affected_gates"]) == (1, 0)
    assert f["a_bits"] == 99 and f["labels"]["delta_grad_bits"].startswith("measured")
    m = F.complete(f, {"output": {"e_bytes": 10}}, SHAPE)
    assert m.a_bits == 99 and m.labels["a_bits"] == "measured: the traced cut" and len(m.parts) == 2
    assert F.of_record(record, through=0)["parts"][0]["steps"] == 1


def test_setup_and_tail_count_in_c_as_parts_with_no_token_blocks():
    """A part with no token blocks has only Fit's compute kind, and no token gates (σ < 1)."""
    step = {"step": 0, "tokens": 8, "struct": "a", "by": {"fwd/theta": 80}}
    record = {"setup": {"by": {"other/theta": 30}, "struct": None}, "tail": {"by": {"other/theta": 5}, "struct": None},
              "steps": [step, step | {"step": 1}],
              "structures": {"a": {"updated": [], "log": [["fwd", "theta", "mm", "[[8]]", 1, 80]]}}}
    whole, prefix = F.of_record(record), F.of_record(record, through=0)
    assert [p["tokens"] for p in whole["parts"]] == [0, 0, 16] and [p["tokens"] for p in prefix["parts"]] == [0, 8]
    m = F.complete(whole, {"output": {"e_bytes": 10}}, SHAPE)
    assert m.gates == pytest.approx(160 * per_mac("gemm/bf16") + 35 * per_mac("elementwise/bf16"))
    assert m.sigma == pytest.approx(160 * per_mac("gemm/bf16") / m.gates)
    assert [set(p["fit"]) for p in F.at(m, 1e13, 1e4)["parts"]] == [{"compute"}, {"compute"},
                                                                    {"compute", "reduction", "update"}]


def test_the_grid_spans_the_reference_shapes_in_decades():
    s = policy.shapes(SHAPE)
    w_fwd = SHAPE["forward_macs_per_token"] * per_mac("gemm/bf16")
    assert s["inference"]["c"] == w_fwd and s["inference"]["k"] == 16 + 1023 * 2 * 6 * 512 * 16
    ref = s["reference"]
    assert (ref["c"], ref["k"]) == (4096 * w_fwd, 80_000)
    serving = F.Measured((F.Part(1024 * w_fwd, 1024, 1024 * w_fwd, 0.0),), 16, 512 * 16, F.even(0.0), F.even(0.0), 0.0,
                         0.0)
    assert F.fit_k(serving, ref["c"], w_fwd)[0] <= ref["k"]  # honest serving passes Fit at the reference policy
    assert s["server"]["c"] == pytest.approx(3 * SHAPE["forward_macs_per_token"] * SHAPE["tokens"] * per_mac("gemm/bf16"))
    assert s["server"]["k"] == SHAPE["parameters"] * 80
    g = policy.grid(SHAPE)
    cs, ks = sorted({c for c, _ in g}), sorted({k for _, k in g})
    assert cs[0] <= w_fwd and cs[-1] >= s["server"]["c"] and ks[0] <= 16384 and ks[-1] >= s["server"]["k"]
    assert all(math.log10(c).is_integer() for c in cs) and len(g) == len(cs) * len(ks)
    assert policy.of_key(policy.key(1e14, 1e4)) == (1e14, 1e4)
    assert policy.c0(SHAPE | {"c0_gates": 5.0}) == (5.0, "counted")


def test_the_cells_are_the_grid_and_the_reference_shapes_at_their_exact_c_and_k():
    cells, s = policy.cells(SHAPE), policy.shapes(SHAPE)
    assert {m: cells[m] for m in policy.MARKS} == {m: (v["c"], v["k"]) for m, v in s.items()}
    assert {v for k, v in cells.items() if k not in policy.MARKS} == set(policy.grid(SHAPE))


def test_the_consoles_c_is_one_full_session_of_the_reference_model():
    """The dashboard's ruling: the console's block is one session of the 45M reference, 4,096 tokens in and 1,024 out,
    attention and the LM head included, every MAC at the bf16 GEMM price: 1.5739 times the old 5,120 x 2N x 740."""
    s = policy.session(45_029_888, 4096, 1024)
    assert s["shape"] == {"d_model": 512, "n_layer": 6, "n_head": 8, "d_ff": 1408, "vocab": 50_304}
    assert s["macs"] == {"matrices": 5120 * 6 * (4 * 512 ** 2 + 3 * 512 * 1408), "lm_head": 5120 * 512 * 50_304,
                         "attention": 2 * 6 * 512 * (4096 ** 2 + 1024 * 4096 + 1024 * 1025 // 2)}
    assert s["gates"] == 362_592_337_920 * 1480.125 == policy.CONSOLE[0] == policy.session_gates(45_029_888, 4096, 1024)
    assert s["k_bits"] == policy.CONSOLE[1] == 65_536
    assert round(policy.CONSOLE[0] / (5_120 * 2 * 45e6 * 740), 6) == 1.573884
    # attention over the whole prompt at each prompt token, as training_slowdown.flops counts the causal SDPA
    assert policy.session(45_029_888, 8, 0)["attended"] == 64 and policy.session(45_029_888, 8, 2)["attended"] == 64 + 19
    with pytest.raises(ValueError, match="no stated shape"):
        policy.session(7e9, 4096, 1024)
    with pytest.raises(ValueError, match="parameters, not"):
        policy.session(45_029_888, 4096, 1024, policy.SHAPES[35_658_240])


def test_each_starts_shape_is_stated_and_its_counted_forward_is_the_sessions_products():
    """Every start's size has a stated shape with its parameters, d_model, layers and vocabulary; a token's products at
    the start's context are its counted forward MACs but its elementwise ops, under 0.1% of them."""
    for name, shape in policy.starts().items():
        sh = policy.SHAPES[shape["parameters"]]
        assert policy.parameters(sh) == shape["parameters"], name
        assert (sh["d_model"], sh["n_layer"], sh["vocab"]) == (shape["d_model"], shape["n_layer"], shape["vocab"]), name
        assert sh["d_model"] // sh["n_head"] == 64, name
        products = sum(policy.session(shape["parameters"], shape["ctx"], 0)["macs"].values()) / shape["ctx"]
        assert 0 < shape["forward_macs_per_token"] - products < 1e-3 * shape["forward_macs_per_token"], name


def test_honest_is_the_least_gates_of_any_checkpoint_and_s_is_at_least_one():
    """The coordinator's 19:50Z rule: H and W* read every checkpoint of every valid record, a mid-schedule one
    included; each cell carries W*'s Slowdown terms."""
    row = {"start": "s", "output": {"e_bytes": 10 ** 6}, "start_bpb": 2.0, "direction": "d", "annotation": {"steps": 1}}

    def r(i, gates_macs, bpb, step, run):
        log = {"by": {"fwd/theta": gates_macs, "fwd/updated": 1},
               "ops": {"fwd theta mm [[1]]": [1, gates_macs], "fwd updated mm [[1]]": [1, 1]}}
        return row | {"id": f"r{i}", "macs_log": log, "val_bpb": bpb, "checkpoint_step": step, "run_id": run}

    rows = [r(1, 10 ** 9, 1.9, 9, "a"), r(2, 10 ** 8, 1.9, 4, "b"), r(3, 10 ** 10, 1.9, 9, "b")]
    shapes = {"s": SHAPE}
    scored, skipped = frontier.formalism_rows(rows, shapes)
    assert not skipped and all(set(x["rho"]) == set(policy.cells(SHAPE)) for x in scored)
    out = frontier.formalism_targets(scored, shapes)["s"]
    last = [t for t in out["targets"] if t["drop"] == frontier.TARGETS[-1]]
    assert len(last) == 1
    t = last[0]
    assert t["honest"] == pytest.approx(10 ** 8 * per_mac("gemm/bf16")) and t["honest_of"]["id"] == "r2"
    assert all(v["S"] >= 1 for v in t["cells"].values() if v["S"] is not None)
    assert set(policy.MARKS) <= set(t["cells"]) and {v["basis"] for v in t["cells"].values()} <= {"fit", "target"}
    assert all(set(frontier.TERMS) <= set(v) for v in t["cells"].values())
    assert set(out["marks"]) == set(policy.MARKS)
    assert len(out["targets"]) == len(frontier.TARGETS)


def test_a_span_block_holds_the_most_tokens_its_reads_and_gates_allow():
    """No context: (k − U)/A tokens, at most C0, none where U > k. With context (kf a position before p, read once a
    block), a block serving p either reads the earlier positions' context or computes them at A each: at k = 14 with
    A = 4, kf = 10, L = 4, positions 0-2 hold 3.5 tokens and position 3 none (30 bits of context, or 16 of states)."""
    assert F._span_tokens(16, 1000, 0, 0, None, 500, 17000).tolist() == [500]
    assert F._span_tokens(16, 1000, 0, 0, None, 2000, 17000).tolist() == [1000]
    assert F._span_tokens(16, 20000, 0, 0, None, 2000, 17000).tolist() == [0]
    assert F._span_tokens(4, 0, 10, 0, 4, math.inf, 14).tolist() == [3.5, 3.5, 3.5, 0]
    assert F._span_tokens(4, 0, 10, 0, 4, math.inf, 40).tolist() == [10, 10, 10, 10]
    assert F._span_tokens(4, 0, 0, 10, 4, math.inf, 14).tolist() == [0, 3.5, 3.5, 3.5]


def tile(gates: float, m: int, K: int, n: int = 1000) -> dict:
    """A forward op of one (1, m, K, n) product at one gate a MAC, bf16: at k = 3600 its tile holds 225·max(m, n)
    gates, the operand of the shorter side free (training_slowdown.tiles)."""
    return {"phase": "fwd", "class": "updated", "op": "mm", "gates": gates, "macs": gates, "products": [[1, m, K, n]],
            "widths": [16, 16]}


def one_segment(U=2000.0, delta=None, decoder=0.0, ops=None) -> F.Measured:
    """One part of 1e12 gates, 1e3 tokens of 1e9 gates each, half of them affected, reading A's entries of one forward
    segment that reads a token's id (16 bits) and U update bits, and an op log of one product (`tile`) whose tile holds
    2.25e8 gates at k = 3600, 6.25e7 at k = 1000."""
    part = F.Part(gates=1e12, tokens=1e3, token_gates=1e12, affected_gates=5e11)
    entries = {"segments": 1, "gates": [1e9], "phase": [0], "frontier_bits": [[0.0]], "ids": [[1.0]], "id_bits": 16,
               "u_bits": [[U]], "positions": None, "context_fwd_bits": None, "context_bwd_bits": None}
    return F.Measured((part,), tau_bits=16, a_bits=16, u=F.even(U), u_in=F.even(U), delta_bits=delta,
                      decoder_gates=decoder, entries=entries, ops=[tile(1e12, 10 ** 6, 10 ** 3)] if ops is None else ops)


def test_the_default_packer_is_the_counting_lemma_at_its_best_union_of_runs_and_tiles():
    """c = 1e11 gates holds 100 tokens: a run block holds min((k − U)/16, 100) tokens of 1e9 gates, and beside it the
    tile (1.75e8 gates at k = 2800), so the term is c·share over min(c, the two), never ∞: where no run block fits
    (U > k) the tile alone. A block reading the update code (Δ = 500 < U) reads less but has c − d gates under
    decoding, c under token; the best of the kinds binds. Without an op log the family has no tiles: unmeasured
    (None)."""
    m, c = one_segment(), 1e11
    assert F.packer(m, c, 3600) == pytest.approx(0.5) and F.packer(m, c, 10 ** 4) == pytest.approx(0.5)
    assert F.packer(m, c, 2800) == pytest.approx(c * 0.5 / (5e10 + 1.75e8))
    assert F.packer(m, c, 1000) == pytest.approx(c * 0.5 / 6.25e7)
    assert F.packer(replace(m, ops=None), c, 1000) is None and F.packer(replace(m, entries=None), c, 3600) is None
    coded = one_segment(delta=500.0, decoder=5e10)
    assert F.packer(coded, c, 2800, decoder=False) == pytest.approx(0.5)
    assert F.packer(coded, c, 2800, decoder=True) == pytest.approx(c * 0.5 / (5e10 + 1.75e8))
    assert F.packer(coded, c, 1000, decoder=False) == pytest.approx(c * 0.5 / (1e9 * 500 / 16 + 6.25e7))


def test_a_union_of_witnesses_bounds_every_gate_so_per_gate_caps_are_a_sensitivity():
    """Two forward segments, one a run block holds at c and one no run block holds (its update cut exceeds k), and two
    products, tiles of 2.25e9 and 2.25e7 gates. A block holding a gate of the segment no run holds, for that gate's
    few operand bits, also holds the run at c: every block is at most c, so the term is the affected share. Each gate
    at its own best witness (`per_gate`) puts the better tile on the segment no run holds (the antitone pairing), c·(1/c
    + 1/2.25e9)/2·(A/|C|)·2, far above it; the bound publishes that as a sensitivity. Without any run, both tiles
    sit in one block beside each other: 2.25e9 + 2.25e7 gates."""
    entries = {"segments": 2, "gates": [1e9, 1e9], "phase": [0, 0], "frontier_bits": [[0.0, 0.0], [0.0]],
               "ids": [[1.0, 1.0], [1.0]], "id_bits": 16, "u_bits": [[0.0, 1e9], [1e9]], "positions": None,
               "context_fwd_bits": None, "context_bwd_bits": None}
    ops = [tile(5e11, 10 ** 7, 50), tile(5e11, 10 ** 5, 5000)]
    part = F.Part(gates=1e12, tokens=500, token_gates=1e12, affected_gates=1e12)
    m, c, k = F.Measured((part,), 16, 16, F.even(0.0), F.even(0.0), None, entries=entries, ops=ops), 1e11, 3600
    assert F.packer(m, c, k) == pytest.approx(1.0)
    assert F.per_gate(m, c, k) == pytest.approx(c * (0.5 / c + 0.5 / 2.25e9))
    half = replace(m, parts=(replace(part, affected_gates=5e11),))
    assert F.packer(half, c, k) == pytest.approx(0.5)
    assert F.per_gate(half, c, k) == pytest.approx(c * 0.5 * (0.5 / c + 0.5 / 2.25e9))
    b = F.bound(m, c, k)
    assert b["terms"]["token"] == pytest.approx(1.0) and b["sensitivities"]["per_gate"] == F.per_gate(m, c, k)
    assert "not closed under unions" in b["labels"]["per_gate"]
    runless = replace(m, entries=entries | {"u_bits": [[1e9, 1e9], [1e9]]})
    assert F.packer(runless, c, k) == pytest.approx(c / (2.25e9 + 2.25e7))
    a, b = np.array([1.0, math.inf]), np.array([3.0, 2.0])
    assert F._antitone(a, np.ones(2), b, np.ones(2)) == pytest.approx((1.0 + 2.0) / 2)


def test_the_one_bound_is_the_max_over_its_kinds_each_resting_on_what_it_names():
    """The max is over the token, decoding and fresh-products kinds; the token kind binds where Δ isn't counted
    (decoding then equals it, and ties bind the kind resting on less); fresh products are unmeasured on a record with
    no product rows, with what its op log leaves unknown where it has one; reading B isn't a reading of the bound."""
    c = 1e11
    coded = one_segment(delta=500.0, decoder=5e10)
    b = F.bound(coded, c, 2800)
    assert (b["terms"]["token"], b["terms"]["decoding"], b["binding"]) == (
        pytest.approx(0.5), pytest.approx(c * 0.5 / (5e10 + 1.75e8)), "decoding")
    assert b["max"] == b["terms"]["decoding"] and b["label"] == f"{F.THE_CODE} {F_VERSION}, reading A"
    assert b["terms"]["fresh_products"] is None and b["unmeasured"]["fresh_products"].startswith(
        f"unmeasured: {F.LOG_UNKNOWN}; 100% of its products' MACs")
    assert b["unmeasured"]["fresh_products"].endswith(f"a trace from {F.THE_CODE} {F.PRODUCTS_FROM} on keeps them")
    assert b["unmeasured"]["fresh_old"] == b["unmeasured"]["fresh_products"]
    assert b["inputs"]["fresh_products"] == {"source": "op log", "unknown_share": 1.0}
    assert b["inputs"]["fresh_old"] == {"source": "op log", "order": "chain", "rows": 1, "unknown_share": 1.0,
                                        "macs": {"theta": 0.0, "attention": 0.0, "unknown": 1e12}}
    assert F.bound(replace(coded, ops=None), c, 2800)["unmeasured"]["fresh_products"] == F.FRESH_UNMEASURED
    assert set(b["rests_on"]) == set(F.KINDS) and "assumption (D)" in b["rests_on"]["decoding"]
    assert "tiles" in b["rests_on"]["token"] and b["rests_on"]["fresh_products"].startswith("theorem")
    plain = F.bound(one_segment(), c, 2800)
    assert plain["binding"] == "token" and plain["terms"]["decoding"] == plain["terms"]["token"]
    assert F.at(coded, c, 2800, "A")["rho"] == pytest.approx(1.0) and F.at(coded, c, 2800, "A")["bound"] == b
    with pytest.raises(ValueError, match="sensitivity of bound"):
        F.at(coded, c, 2800, "B")
    with pytest.raises(ValueError, match="some of"):
        F.bound(coded, c, 2800, kinds=("tokens",))


#: one bf16 product of two fresh operands over K = 512, half of one_segment's 1e12 gates
SQUARE = {"macs": 0.5e12 / (23682 / 16), "outputs": 0.5e12 / (23682 / 16) / 512, "k_dim": 512, "k_whole": 512, "b_a": 16,
          "b_b": 16, "b_c": 32, "gates_per_mac": 23682 / 16, "cell_gates": 23682.0}
#: SQUARE's whole-making system: one class at K = 512 feeding itself and the counted operands, io_bound's one-class rule
SYSTEM = {"classes": [{"k": 512, "c": 1 / 3, "P": 1.0, "feeds": [[512, 1.0]]}], "counted": [[512, 1.0]], "b": 16}


#: SQUARE as a step's products with its system and the products term's M (`he_of_products`)
CHAIN = {"k": policy.CONSOLE[1], "order": "chain", "products": [SQUARE], "system": SYSTEM, "elementwise_products": [],
         "he": F.he_of_products([SQUARE], policy.CONSOLE[1])}


def test_the_old_fresh_products_figure_is_io_bounds_bound_beside_the_term():
    """The sensitivity fresh_old is training_slowdown.io_bound's bound with Λ proved on the record's products and |C|, with
    P-cell, and fresh_no_cell its figure without it, in every bound, and fresh_old on every console row; both rest on
    P-width extended, false on honest circuits. The products are summed in the reference circuit's chains (`ORDER`),
    so they give it whether or not they give an order; a circuit that declares a tree leaves it unmeasured, with
    io_bound's chain figure in the inputs, as a k other than the one the products were classified at does. It is the
    same over S steps of the same products, and the products survive the record's JSON."""
    from training_slowdown import io_bound

    c, k = 1e11, policy.CONSOLE[1]
    chain = replace(one_segment(), products=CHAIN)
    b = F.bound(chain, c, k)
    wc = F.whole_chain(SYSTEM, k)["whole_chain"]
    assert wc == pytest.approx(io_bound.whole_chain(k, 512, 16), rel=1e-9)
    proved = io_bound.bound([SQUARE], c, k, 1e12, wc)["proved"]
    rho, no_cell = proved["rho"], proved["rho_no_cell"]
    assert F.P_CELL and F.ORDER == "chain" and 10 < no_cell < 100 < rho < 1e3
    assert b["sensitivities"]["fresh_old"] == rho and "P-cell" in b["labels"]["fresh_old"]
    assert b["sensitivities"]["fresh_no_cell"] == no_cell and "without assuming" in b["labels"]["fresh_no_cell"]
    assert all(b["labels"][x].startswith("rests on P-width extended, false on honest circuits")
               for x in ("fresh_old", "fresh_no_cell"))
    assert all(x in F.SENSITIVITIES["fresh_old"] for x in (F.P_MATCH, F.P_WIDTH, F.P_CHAIN, "P-share", F.P_CELL_WORDS))
    assert not {"fresh_old", "fresh_no_cell"} & set(b["unmeasured"]) and b["void"] == {}
    assert b["inputs"]["fresh_old"]["io_bound"]["share"] == pytest.approx(0.5, rel=1e-12)
    one = F.published(chain, reading="A")
    assert set(one["sensitivities"]) == {"reduction", "update", "fresh_old"} and one["void"] is None
    assert one["sensitivities"]["fresh_old"] == pytest.approx(rho * policy.CONSOLE[0] / c, rel=1e-12)
    assert F.bound(replace(chain, products=CHAIN | {"order": None}), c, k)["sensitivities"]["fresh_old"] == rho
    tree = F.bound(replace(chain, products=CHAIN | {"order": "tree"}), c, k)
    assert tree["sensitivities"]["fresh_old"] is None and "declares its products summed in a tree" in tree[
        "unmeasured"]["fresh_old"]
    assert "chain constant only" in tree["unmeasured"]["fresh_old"] == tree["unmeasured"]["fresh_no_cell"]
    assert tree["inputs"]["fresh_old"]["io_bound"]["proved"]["rho"] == rho
    elsewhere = F.bound(chain, c, 80_000)
    assert elsewhere["sensitivities"]["fresh_old"] is None and "classified at k = 65536" in elsewhere["unmeasured"][
        "fresh_old"]
    S = 7
    steps = replace(chain, parts=(replace(chain.parts[0], gates=S * 1e12, tokens=S * 1e3, token_gates=S * 1e12,
                                          affected_gates=S * 5e11),),
                    products=CHAIN | {"products": [SQUARE | {"macs": S * SQUARE["macs"],
                                                             "outputs": S * SQUARE["outputs"]}]})
    assert F.bound(steps, c, k)["sensitivities"]["fresh_old"] == pytest.approx(rho, rel=1e-10)
    assert F.from_json(chain.to_json()).products == chain.products
    assert F.bound(replace(chain, products=CHAIN | {"system": None}), c, k)["unmeasured"]["fresh_old"] == \
        F.FRESH_RETRACE
    hand = F.bound(replace(chain, products={"k": k, "order": "chain", "products": [SQUARE]}), c, k)
    assert hand["sensitivities"]["fresh_old"] == io_bound.bound([SQUARE], c, k, 1e12)["proved"]["rho"]
    assert hand["inputs"]["fresh_old"]["whole_making"] == F.ONE_CLASS


def test_the_products_term_is_the_theorem_of_records_on_m():
    """The products kind's term is the theorem of record's ρ on M (`he_figure` on the products' `he`) and the record's
    |C|, linear in c at a fixed k, and the published row rests on (H) at the β it took and (E) at its b_y, which
    `products_rests_on` carries wherever the term is measured. It needs no summation order and no whole-making system,
    so a tree, a step whose rows have no system and products written elementwise leave it measured. It is unmeasured
    at a k the products weren't classified at, on products that come without M (a hand-written set), on rows that
    don't flag a gather's and where an elementwise op fans out into an operand. The published row names the code that
    wrote the products' trace where the trace records it, and None where it doesn't."""
    c, k = 1e11, policy.CONSOLE[1]
    chain = replace(one_segment(), products=CHAIN)
    b = F.bound(chain, c, k)
    fig = F.he_figure(CHAIN["he"], c, 1e12)
    assert (fig["subset"], fig["beta_w"], fig["beta_x"], fig["b_y"]) == ("M", 16, 15.9, 16)
    assert not fig["below_width"] and fig["macs_share"] == 1
    assert fig["F"] == pytest.approx(2 / math.sqrt(3), rel=1e-15) and fig["N"] == SQUARE["macs"]
    assert fig["O"] == SQUARE["outputs"] * 16
    assert b["terms"]["fresh_products"] == fig["rho"] > 1 and "fresh_products" not in b["unmeasured"]
    assert F.bound(chain, 2 * c, k)["terms"]["fresh_products"] == pytest.approx(2 * fig["rho"], rel=1e-12)
    assert F.published(chain)["reading"] == ("A" if F.READING_A else "B")
    one = F.published(chain, reading="A")
    assert one["products_rests_on"] == F.he_rests_on(fig)
    assert all(x in one["products_rests_on"] for x in ("(H)", "β_w = 16 and β_x = 15.9 bits a word", F.BETA_K_LABEL,
                                                       "the casts of one value count as one word", "(E)",
                                                       "b_y = 16 bits an output", "a single counterexample refutes"))
    assert F.CODE_LABEL not in one["products_rests_on"]
    assert one["rests_on"] == (one["products_rests_on"] if one["binding_kind"] == "fresh_products" else
                               F.RESTS_ON_CELL[one["binding_kind"]])
    assert one["capture_code"] is None and F.published(replace(chain, products=CHAIN | {
        "capture_code": F.CAPTURE_CODE_SINCE}), reading="A")["capture_code"] == F.CAPTURE_CODE_SINCE
    for x in ({"order": "tree"}, {"system": None}, {"elementwise_products": [{"op": "mul", "count": 2, "macs": 4.0}]}):
        assert F.bound(replace(chain, products=CHAIN | x), c, k)["terms"]["fresh_products"] == fig["rho"], x
    elsewhere = F.bound(chain, c, 80_000)
    assert elsewhere["terms"]["fresh_products"] is None and "classified at k = 65536" in elsewhere["unmeasured"][
        "fresh_products"]
    no_m = {x: v for x, v in CHAIN.items() if x != "he"}
    assert F.bound(replace(chain, products=no_m), c, k)["unmeasured"]["fresh_products"] == F.HE_NO_ROWS
    assert F.bound(replace(chain, products=CHAIN | {"he": None}), c, k)["unmeasured"]["fresh_products"] == F.HE_RETRACE
    fans = F.bound(replace(chain, products=CHAIN | {"p_match": [{"op": "expand", "fan_out": 16.0, "count": 1}]}), c, k)
    assert fans["terms"]["fresh_products"] is None and "breaks the premise" in fans["unmeasured"]["fresh_products"]
    assert all(x in F.RESTS_ON["fresh_products"] for x in ("(H)", "(E)", "F = 2/√3", "β_eff", "K* = ½√(k/β)",
                                                           F.BETA_K_LABEL, F.CODE_LABEL, "casts of one value"))
    assert all(x in F.RESTS_ON_CELL["fresh_products"] for x in ("(H)", "(E)", "a single counterexample refutes"))
    assert F.CELL_SENSITIVITIES == ("reduction", "update", "fresh_old")


def he_row(phase: str, td: list, m: int, k: int, n: int, count: int = 1, batch: int = 1) -> dict:
    """A product row as training_slowdown.capture.products writes one: both operands affected at bf16, paid, none gathered."""
    gates = next(g for g, b in F.HE_BITS.items() if b == (16, 16))
    return {"phase": phase, "token_dims": td, "dims": {"batch": batch, "m": m, "k": k, "n": n}, "count": count,
            "macs": batch * m * k * n, "outputs": batch * m * n, "affected": [True, True], "self": False,
            "gathered": [False, False], "widths": [16, 16], "rates": [16.0, 16.0], "cell_gates": gates}


#: a step of 1,024 tokens: two layers' 1,024-wide matrix, a 4,096-wide head and an 8-long product with wide outputs,
#: forward, input and weight gradients, beside the optimizer's products and attention's
HE_STEP = [he_row("forward", ["m"], 1024, 1024, 1024, 2), he_row("forward", ["m"], 1024, 1024, 4096),
           he_row("forward", ["m"], 1024, 8, 4096), he_row("backward", ["m"], 1024, 4096, 1024),
           he_row("backward", ["k"], 4096, 1024, 1024), he_row("backward", ["m"], 1024, 1024, 1024, 2),
           he_row("backward", ["k"], 1024, 1024, 1024, 2), he_row("optimizer", [], 1024, 1024, 1024, 4),
           he_row("forward", ["batch", "m", "n"], 256, 64, 256, 1, 4)]


def test_the_products_term_takes_the_best_of_three_ms_at_beta_k_with_the_code_size_beta_w_and_the_first_steps():
    """4.13.1's rules on M (`he_of_rows`, `he_figure`): the optimizer's products and attention's leave M; a weight is an
    operand no token indexes, a backward product contracting over tokens a weight gradient, and the head's are the
    backward products over its vocabulary width (the widest forward output); the trained weights are the weights a
    step's forward reads, each matrix once a token. Activations are charged β(k), 15.9 up to 145,568 and linear in log
    k between io-bound's points, none past 8e8, weights their width or less the record's code bits a trained weight,
    a weight's β the β_w a product reports beside its activation's β_x. The term is the best of all of M, M less its contractions below K_thr (here the 8-long product's wide outputs,
    whose leaving lifts the term) and the weight gradients alone at F = 1; over a run's steps it loses the ⌊k/(τT)⌋
    after the first update, and from 8e7 the head's products leave M. Model products that sum the head's gradients
    with the layers' are unmeasured from 8e7, and every M past 8e8."""
    c, k, C, T = 1e11, 65_536.0, 1e13, 1024
    assert (F.beta_k(1.0), F.beta_k(145_568), F.beta_k(8e6), F.beta_k(8e8)) == (15.9, 15.9, 10.8, 0.37)
    assert F.beta_k(math.sqrt(8e6 * 8e7)) == pytest.approx((10.8 + 2.7) / 2, rel=1e-12) and F.beta_k(8.1e8) is None
    he = F.he_of_rows(HE_STEP, k, 10.0, T)
    opt, att = HE_STEP[7], HE_STEP[8]
    assert he["left_out_macs"] == 10 * (opt["count"] * opt["macs"] + att["count"] * att["macs"])
    assert he["trained_weights"] == 2 * 1024 * 1024 + 1024 * 4096 + 8 * 4096
    assert sorted((g["k_dim"], g["dw"], g["head"], tuple(g["weight"])) for g in he["groups"]) == [
        (8, False, False, (False, True)), (1024, False, False, (False, True)), (1024, False, False, (False, True)),
        (1024, True, False, (False, False)), (1024, True, True, (False, False)), (4096, False, True, (False, True))]
    fig = F.he_figure(he, c, C)
    assert fig["subset"] == F.HE_SUBSETS[1] and fig["rho"] == max(fig["subsets"].values()) > fig["subsets"]["M"]
    assert 8 < fig["K_thr"] < 1024 and fig["beta_k"] == 15.9 and fig["F"] == pytest.approx(2 / math.sqrt(3))
    dw = [g for g in he["groups"] if g["dw"]]
    N, O = sum(g["macs"] for g in dw), sum(g["wires"] for g in dw)
    assert fig["subsets"][F.HE_SUBSETS[2]] == pytest.approx(max(0.0, F.he_rho(N, O, 15.9, 15.9, 16, 1.0, c, k, C)),
                                                            rel=1e-12)
    first = F.he_figure(he, c, C, tau=16)
    assert first["first_steps"] == 4 and first["N"] == pytest.approx(0.6 * fig["N"], rel=1e-12)
    assert fig["first_steps"] is None and F.he_figure(he, c, C, tau=1e3)["first_steps"] == 0
    coded = F.he_figure(he, c, C, beta_code=2.0)
    assert coded["rho"] < fig["rho"] and coded["beta_code"] == 2.0
    one = he | {"groups": [g for g in he["groups"] if g["k_dim"] == 1024 and not g["dw"] and not g["head"]][:1]}
    weighed = F.he_figure(one, c, C, beta_code=8.0)
    assert (weighed["beta_w"], weighed["beta_x"], weighed["code_bound"]) == (8.0, 15.9, True)
    m = replace(one_segment(delta=2.0 * he["trained_weights"]),
                products={"k": k, "order": "chain", "products": [], "he": he})
    term, why, got = F.he_products(m, c, k)
    assert why is None and got["beta_code"] == 2.0 and term == F.he_figure(he, c, m.gates, m.tau_bits, 2.0)["rho"]
    assert (F.CODE_LABEL in F.he_rests_on(got)) == got["code_bound"]
    far = F.he_of_rows(HE_STEP, 8e7)
    assert F.he_figure(far, c, C) == F.he_figure(far | {"groups": [g for g in far["groups"] if not g["head"]]}, c, C)
    assert F.he_figure(he, c, C) != F.he_figure(he | {"groups": [g for g in he["groups"] if not g["head"]]}, c, C)
    for at, why in ((8e7, F.HE_HEAD), (9e8, F.HE_PAST)):
        model = replace(one_segment(), products={"k": at, "order": "chain", "products": [SQUARE],
                                                 "he": F.he_of_products([SQUARE], at)})
        assert F.he_products(model, c, at)[:2] == (None, why), at
    with pytest.raises(ValueError, match="past k"):
        F.he_figure(F.he_of_rows(HE_STEP, 9e8), c, C)


def test_the_adapter_rule_needs_a_frozen_weight_the_update_reaches_and_a_frozen_head_keeps_its_width():
    """standard-workloads' 12:37Z LOOPHOLE: a distilling teacher's weights are frozen but read only tokens, so the
    student's head (its d_model widened to the vocabulary) is no adapter: r_step is None and every activation keeps its
    width, as without the teacher. A LoRA's frozen layers read what its adapters wrote, so B·(Ax) sets r_step to its
    rank and an activation wider than it is charged β_eff. With its head frozen, the vocabulary is still the head's
    width, so the adapters' backward products over d_model aren't the head's; a trained head's are."""
    k, d, V, r = 65_536.0, 512, 32_000, 8

    def frozen(row, *i):
        return row | {"affected": [j not in i for j in (0, 1)]}

    student = [he_row("forward", ["m"], 1024, 256, 704), he_row("forward", ["m"], 1024, 704, 256),
               he_row("forward", ["m"], 1024, 256, V)]
    alone, distilled = (F.he_of_rows(rows, k) for rows in (student, [frozen(he_row("forward", ["m"], 1024, d, V), 0, 1),
                                                                     *student]))
    assert distilled["r_step"] is alone["r_step"] is None
    assert distilled["groups"] == alone["groups"] and all(b == 16 for g in alone["groups"] for b in g["beta"])
    lora = [frozen(he_row("forward", ["m"], 1024, d, d), 1), he_row("forward", ["m"], 1024, d, r),
            he_row("forward", ["m"], 1024, r, d), frozen(he_row("forward", ["m"], 1024, d, V), 1),
            he_row("backward", ["m"], 1024, d, r), he_row("backward", ["k"], d, 1024, r)]
    he = F.he_of_rows(lora, k)
    assert he["r_step"] == r and not any(g["head"] for g in he["groups"])
    dx = [g["beta"] for g in he["groups"] if not g["forward"] and not g["dw"]]
    assert dx == [[round(F.beta_eff(16, r, k), 9), 16]]
    trained = F.he_of_rows([*lora[:3], he_row("forward", ["m"], 1024, d, V), he_row("backward", ["m"], 1024, V, d)], k)
    assert [g["k_dim"] for g in trained["groups"] if g["head"]] == [V]


def test_the_products_term_charges_m_its_least_pair_where_a_layers_arrays_differ_in_beta():
    """F = 2/√3 is proved for one β on a layer's three shared arrays (weights, activations, output gradients); with a β
    each, Motzkin-Straus sums u_r·u_s/(β_r·β_s) over the pairs M's products multiply, so both terms take the least
    pair product among M's products. Weights at 2 (the record's code size), activations at β(k) = 15.9 and output
    gradients at 8: all of M binds at the input gradient's 2·8 = 16, below the forward product's 2·15.9, which would
    read higher, and each threshold's M_P keeps only the products whose pair is at least its own."""
    c, k, C, m, K = 1e11, 65_536.0, 1e13, 256, 65_536

    def group(beta: list, weight: list, forward: bool, dw: bool, scale: float) -> dict:
        return {"beta": beta, "width": [16, 16], "weight": weight, "b_y": 16, "forward": forward, "k_dim": K,
                "dw": dw, "head": False, "macs": scale * m * m * K, "wires": scale * m * m * 16}

    fwd, dx = group([16.0, 16.0], [False, True], True, False, 1), group([8.0, 16.0], [False, True], False, False, 4)
    dw = group([8.0, 16.0], [False, False], False, True, 0.25)
    he = {"k": k, "groups": [fwd, dx, dw], "r_step": None, "candidates_macs": 5.25 * m * m * K, "self_macs": 0.0,
          "gathered_macs": 0.0, "left_out_macs": 0.0, "steps": None, "step_tokens": None, "trained_weights": None}
    fig = F.he_figure(he, c, C, beta_code=2.0)
    share = 2 / math.sqrt(3)

    def rho(M: list, bw: float, bx: float, f: float = share) -> float:
        return F.he_rho(sum(g["macs"] for g in M), sum(g["wires"] for g in M), bw, bx, 16, f, c, k, C)

    assert (fig["subset"], fig["beta_w"], fig["beta_x"], fig["code_bound"]) == ("M", 2.0, 8.0, True)
    assert fig["F"] == pytest.approx(share, rel=1e-15) and fig["macs_share"] == 1
    assert fig["rho"] == pytest.approx(rho([fwd, dx, dw], 2.0, 8.0), rel=1e-12)
    assert rho([fwd, dx, dw], 2.0, 8.0) < rho([fwd, dx, dw], 2.0, 15.9)
    assert fig["rho"] > max(rho([fwd, dw], 2.0, 15.9), rho([dw], 15.9, 8.0), rho([dw], 15.9, 8.0, 1.0)) > 0


def graph(omega: int = 3, forward: int = 2, dw: int = 2, repeated: list | None = None) -> dict:
    """A step's array graph as training_slowdown.arrays' `check` writes one."""
    return {"omega": omega, "clique": [], "omega_forward": forward, "omega_dw": dw, "arrays": 6, "products": 6,
            "repeated": repeated or [], "loops": []}


def test_the_products_term_takes_f_at_the_steps_array_graph_and_refuses_two_products_on_one_pair():
    """HEShared proves F = √(2(1 − 1/ω)) at the array graph's clique number ω, where M's products are on distinct pairs
    of arrays. A step's graph at a layer's triangle (ω = 3), its forward products and weight gradients triangle-free,
    leaves the term as it is without one; a clique of five charges M F = √1.6, a triangle among the weight gradients
    charges them 2/√3, and one among the forward products charges a forward-only M_P, where it binds, 2/√3; two of M's
    products on one pair leave the term unmeasured, with the graph it read."""
    from training_slowdown import arrays, io_bound

    c, k, C = 1e11, 65_536.0, 1e13
    he = F.he_of_rows(HE_STEP, k, 10.0, 1024)
    fig = F.he_figure(he, c, C)
    assert F.he_figure(he, c, C, graph=graph()) == F.he_figure(he, c, C, graph=graph(2)) == fig
    wide = F.he_figure(he, c, C, graph=graph(5))
    assert wide["F"] == pytest.approx(math.sqrt(1.6), rel=1e-15) and wide["subset"] == fig["subset"]
    assert all(wide["subsets"][x] < fig["subsets"][x] for x in F.HE_SUBSETS[:2])
    assert wide["subsets"][F.HE_SUBSETS[2]] == fig["subsets"][F.HE_SUBSETS[2]] and wide["rho"] < fig["rho"]
    dws = F.he_figure(he, c, C, graph=graph(dw=3))["subsets"]
    N, O = (sum(g[x] for g in he["groups"] if g["dw"]) for x in ("macs", "wires"))
    assert dws[F.HE_SUBSETS[2]] == pytest.approx(F.he_rho(N, O, 15.9, 15.9, 16, io_bound.SHARE, c, k, C), rel=1e-12)
    assert dws[F.HE_SUBSETS[2]] < fig["subsets"][F.HE_SUBSETS[2]]
    m, K = 256, 65_536

    def group(beta: list, weight: list, forward: bool, dw: bool, scale: float) -> dict:
        return {"beta": beta, "width": [16, 16], "weight": weight, "b_y": 16, "forward": forward, "k_dim": K,
                "dw": dw, "head": False, "macs": scale * m * m * K, "wires": scale * m * m * 16}

    wide_fwd = {"k": k, "groups": [group([16.0, 16.0], [False, True], True, False, 16),
                                   group([8.0, 16.0], [False, True], False, False, 1),
                                   group([8.0, 16.0], [False, False], False, True, 1)],
                "r_step": None, "candidates_macs": 18 * m * m * K, "self_macs": 0.0, "gathered_macs": 0.0,
                "left_out_macs": 0.0, "steps": None, "step_tokens": None, "trained_weights": None}
    alone = F.he_figure(wide_fwd, c, C)
    assert (alone["subset"], alone["F"], alone["beta_w"], alone["beta_x"]) == ("M", 1.0, 16.0, 15.9)
    assert alone["macs_share"] == pytest.approx(16 / 18, rel=1e-15)
    shared = F.he_figure(wide_fwd, c, C, graph=graph(forward=3))
    assert shared["F"] == pytest.approx(io_bound.SHARE, rel=1e-15) and shared["rho"] < alone["rho"]
    one = replace(one_segment(), products={"k": k, "order": "chain", "products": [], "he": he})
    term, why, got = F.he_products(replace(one, products=one.products | {"arrays": graph()}), c, k)
    assert why is None and term == F.he_products(one, c, k)[0] and got["arrays"]["omega"] == 3
    twice = graph(repeated=[{"pair": ["W0", "X0"], "products": ["f0", "f0r"], "count": 2}])
    term, why, got = F.he_products(replace(one, products=one.products | {"arrays": twice}), c, k)
    assert term is None and why == arrays.refusal(twice) and got["arrays"]["repeated"] == twice["repeated"]
    b = F.bound(replace(one, products=one.products | {"arrays": twice}), c, k)
    assert b["terms"]["fresh_products"] is None and b["unmeasured"]["fresh_products"] == why


#: a code name in a published string: a dotted path (a module's, or its function's), an identifier with an underscore
#: (training_slowdown, io_bound, rows_of, ONE_CLASS, FRESH_VOID, k_whole) other than a one-letter math subscript (b_y), or a
#: command-line flag
CODE_NAME = __import__("re").compile(
    r"\b[A-Za-z]\w*\.[A-Za-z_]\w*|\b(?![A-Za-z]_[A-Za-z0-9]\b)[A-Za-z][A-Za-z0-9]*_\w+\b|--[a-z]")


def test_no_published_string_names_the_code():
    """Every string a panel publishes is plain words, since the dashboard prints them on a reader's page: what each kind
    and each published ρ rests on, every reason a term is unmeasured (an op log, no product rows, another k, a tree, a
    retrace, an unchecked trace, products written elementwise, a broken P-match, a void, a stand-in), the
    sensitivities' labels, the record's labels and the label. The code is "training's slowdown code", and a premise is
    named in words beside its name."""
    c, k = 1e11, policy.CONSOLE[1]
    static = [F.RESTS_ON, F.RESTS_ON_CELL, F.SENSITIVITIES, F.STUBS, F.TRACED, F.DECODER, F.V23,
              F.FRESH_UNMEASURED, F.FRESH_VOID, F.FRESH_RETRACE, F.FRESH_UNCHECKED, F.ONE_CLASS, F.LOG_UNKNOWN,
              F.HE_RETRACE, F.HE_NO_ROWS, F.HE_PAST, F.HE_HEAD, F.STAND_IN, F.ENTRIES, F.OPS, F.PRODUCTS,
              F.TRACED_PRODUCTS, F.CONTEXT, F.TAIL]
    strings = [s for x in static for s in (x.values() if isinstance(x, dict) else [x])]
    chain = CHAIN
    coded = one_segment(delta=500.0, decoder=5e10)
    records = [coded, replace(coded, ops=None), replace(coded, ops=[]), replace(coded, entries=None)] + [
        replace(one_segment(), products=chain | x) for x in (
            {}, {"system": None}, {"elementwise_products": None}, {"order": "tree"},
            {"elementwise_products": [{"op": "mul", "count": 2, "macs": 4.0}]},
            {"p_match": [{"op": "expand", "fan_out": 16.0, "count": 1}]},
            {"arrays": graph(repeated=[{"pair": ["primals_1", "view_2"], "products": ["mm_1", "mm_4"],
                                        "count": 2}])})] + [
        replace(one_segment(), products={x: v for x, v in chain.items() if x not in ("system", "he")})]
    reasons = set()
    for m in records:
        for at in (k, 2800.0, 80_000.0):
            b = F.bound(m, c, at)
            strings += [b["label"], *b["rests_on"].values(), *b["unmeasured"].values(), *b["labels"].values(),
                        *b["void"].values()]
            reasons |= set(b["unmeasured"].values())
            if isinstance(b["inputs"]["fresh_products"].get("whole_making"), str):
                strings.append(b["inputs"]["fresh_products"]["whole_making"])
            for reading in ("A", "B"):
                one = F.published(m, c, at, reading)
                strings += [one["label"], one["rests_on"], one["why"] or "", one["void"] or "",
                            one["products_rests_on"] or ""]
    strings += [F.void_reason(0.17, 0.5), F.void_reason(0.023, 7e-5), F.void_reason(None, None),
                F.he_rests_on({"subset": F.HE_SUBSETS[2], "beta_w": 1.6, "beta_x": 3.2, "b_y": 16.0,
                               "below_width": True, "code_bound": True}), *F.HE_SUBSETS]
    assert {F.FRESH_UNMEASURED, F.FRESH_RETRACE, F.FRESH_UNCHECKED, F.HE_NO_ROWS} <= reasons
    assert all(any(x in r for r in reasons) for x in ("op log", "classified at k", "summed in a tree", "elementwise "
                                                      "steps (", "breaks the premise", "no entries", "no op log",
                                                      "pair(s) of arrays"))
    named = {s: CODE_NAME.findall(s) for s in strings if CODE_NAME.search(s)}
    assert not named, named
    assert all(f"{words} ({name})" in F.SENSITIVITIES["fresh_old"] for name, words in F.PREMISES.items()
               if name != "P-cell") and F.P_CELL_WORDS in F.SENSITIVITIES["fresh_old"]


def test_published_reading_a_where_fit_holds_is_fits_c_over_c_minus_d():
    c, k = 1e13, 1e6
    one = F.published(measured(u1=1e5, dgrad=1e5, decoder=1e12), c, k, "A")
    assert (one["rho"], one["binding_kind"], one["partial"], one["why"]) == (
        pytest.approx(c / (c - 1e12)), "fit", False, None)
    assert one["rests_on"] == F.RESTS_ON_CELL["fit"]


def test_published_reading_a_above_1_is_the_measured_max_partial_where_a_kind_is_unmeasured():
    """The token kind binds at 800 with fresh products unmeasured (an op log, no product rows): partial, only a lower
    bound. With the chain's products every kind is measured and fresh products bind: not partial."""
    c = 1e11
    one = F.published(one_segment(), c, 1000, "A")
    assert one["rho"] == pytest.approx(c * 0.5 / 6.25e7) and one["binding_kind"] == "token"
    assert one["partial"] and one["why"] is None and one["terms"]["fresh_products"] is None
    assert one["rests_on"] == F.RESTS_ON_CELL["token"]
    chain = replace(one_segment(), products=CHAIN)
    full = F.published(chain, 1e11, policy.CONSOLE[1], "A")
    assert full["binding_kind"] == "fresh_products" and full["rho"] == full["terms"]["fresh_products"] > 1
    assert not full["partial"] and full["rests_on"] == full["products_rests_on"]
    assert full["rests_on"].startswith("theorem of record on M, given (H): operands are hard at β_w = 16 and β_x = "
                                       "15.9 bits a word")


def test_published_reading_a_at_most_1_with_a_kind_unmeasured_is_null_never_1():
    """The token kind's term is 0.5 and fresh products are unmeasured: ρ is null, never the floor's 1 (coordinator
    12:20Z), with `why` naming the unmeasured kind; with no kind measured, null too."""
    one = F.published(one_segment(), 1e11, 3600, "A")
    assert (one["rho"], one["binding_kind"], one["rests_on"], one["partial"]) == (None, "unmeasured", "unmeasured", False)
    assert one["terms"]["token"] == pytest.approx(0.5) and one["terms"]["fresh_products"] is None
    assert one["why"] == "the measured kinds give at most 1; fresh products unmeasured"
    none = F.published(replace(one_segment(), ops=None), 1e11, 3600, "A")
    assert (none["rho"], none["binding_kind"], none["why"]) == (None, "unmeasured", "no kind is measured")


def test_published_reading_a_with_every_kind_measured_and_none_above_1_is_1_binding_none():
    """Every kind measured (the chain's products, an op log and entries), each at most 1 at c = 1e8: ρ is 1, binding
    none, resting on no kind exceeding 1; the floor sets it, not the token term."""
    k = policy.CONSOLE[1]
    chain = replace(one_segment(), products=CHAIN)
    one = F.published(chain, 1e8, k, "A")
    assert all(t is not None and t <= 1 for t in one["terms"].values())
    assert (one["rho"], one["binding_kind"], one["partial"], one["why"]) == (1.0, "none", False, None)
    assert one["rests_on"] == F.RESTS_ON_CELL["none"] == "no kind exceeds 1"


def test_products_of_takes_io_bounds_fresh_rows_from_captured_steps():
    """`products_of` is io_bound.rows_of's fresh rows of each captured step at k, their MACs and output entries times
    the segment's steps: a low-rank chain through trained weights, whose a and b products and, at rank 64, last's are
    fresh (test_io_bound's)."""
    import torch
    import torch.nn.functional as tF
    from torch import nn

    from training_slowdown import io_bound
    from training_slowdown.capture import capture_step

    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(256, 256, bias=False), nn.Linear(256, 64, bias=False), nn.Linear(64, 256, bias=False),
                      nn.Linear(256, 256, bias=False))
    g = capture_step(m, torch.optim.SGD(m.parameters(), lr=0.1), lambda mm, b: tF.mse_loss(mm(b[0]), b[1]),
                     (torch.empty(6, 256), torch.empty(6, 256)))
    fresh = [r.product() for r in io_bound.rows_of(g, "bfloat16", policy.CONSOLE[1]) if r.kind == "fresh"]
    got = F.products_of([(3, g), (2, g)])
    assert fresh and got["k"] == policy.CONSOLE[1] and got["order"] == F.ORDER
    assert got["products"] == [p | {"macs": n * p["macs"], "outputs": n * p["outputs"]} for n in (3, 2) for p in fresh]
    assert F.products_of([(1, g)], k=1024, order="chain")["k"] == 1024.0
    with pytest.raises(ValueError, match="one of"):
        F.products_of([(1, g)], order="split")


@pytest.mark.parametrize("name", ("dense_blocks", "lora_blocks", "wildcard_adapter"))
def test_products_of_ops_takes_a_logs_products_and_counts_none(name):
    """`products_of_ops` on a traced step's op log (the fixtures' `.ops.json`, training_slowdown.tiles.of_graph): a row for each
    product the log places, its op's MACs split by the products' sizes, its output entries its MACs over K, its
    operands' widths the log's and its partial sums' io_bound's; θ where its class is registered, else unknown (the
    tiny start's attention is unfused bmm, whose token dims a log doesn't keep). The order is the reference circuit's,
    and nothing is counted, so the fresh-products term is unmeasured with the unknown share in its reason."""
    from training_slowdown import io_bound

    ops = json.loads((FIXTURES / f"{name}.ops.json").read_text())
    placed = [op for op in ops if op["products"]]
    got = F.products_of_ops(ops)
    assert (got["k"], got["order"], got["source"], got["products"]) == (None, F.ORDER, "op log", [])
    assert len(got["rows"]) == sum(len(op["products"]) for op in placed)
    assert sum(r["macs"] for r in got["rows"]) == pytest.approx(sum(op["macs"] for op in placed), rel=1e-12)
    assert sum(got["macs"].values()) == pytest.approx(sum(op["macs"] for op in placed), rel=1e-12)
    for r in got["rows"]:
        assert r["kind"] == ("theta" if r["class"] == "registered" else "unknown") and r["b_c"] == io_bound.ACC_BITS
        assert r["outputs"] * r["dims"][2] == pytest.approx(r["macs"]) and len(r["widths"]) == 2
    assert got["macs"]["attention"] == 0 and 0 < got["unknown_share"] < 1
    b = F.bound(one_segment(ops=ops), 1e11, policy.CONSOLE[1])
    assert b["terms"]["fresh_products"] is None and b["inputs"]["fresh_old"]["rows"] == len(got["rows"])
    assert f"; {got['unknown_share']:.0%} of its products' MACs" in b["unmeasured"]["fresh_products"]


@pytest.mark.slow
def test_a_dense_steps_product_rows_give_io_bounds_bound_on_it(monkeypatch):
    """The 45M start's dense step (training_slowdown.io_bound.capture, Muon on the block matrices, batch 2 at ctx 256) through
    training_slowdown.capture.products and `products_of_trace`, before G leaves out the sequences' first positions, is the
    products io_bound.rows_of counts on it, its identical rows merged, and their whole-making system is
    io_bound.chain_classes', so io_bound's bound is the same: at the console's policy about 69,676 fresh MACs a block,
    0.552 of |C| and ρ 2.87e6 with P-cell. The trace's order is unknown and the reference circuit's chain stands, so
    the term is measured, on G without each sequence's first of 256 positions and each subset with its own
    whole-making factor: the figure with P-cell, 2.869e6, with 1.633e5 beside it as fresh_no_cell. The rows, with
    those at the console dial's other k, are the fixture the
    vectors read (`dense_45m.products.json`). The one check of the strong bound's figure on a model's own shape (about
    6 s on a CPU): the fixtures' tiny start has almost no fresh products."""
    import collections

    from training_slowdown import io_bound
    from training_slowdown.capture import products

    g = io_bound.capture("dense", size="s", vocab=32000, batch=2, ctx=256, micro_steps=1)
    c, k = policy.CONSOLE
    block = products.rows(g, k)
    assert json.loads(json.dumps(products.rows_on_dial(g))) == json.loads(
        (FIXTURES / "dense_45m.products.json").read_text())
    with monkeypatch.context() as mp:
        mp.setattr(F, "SHARED_PREFIX", 0)
        whole = F.products_of_trace(block)
    got = F.products_of_trace(block)
    rows = io_bound.rows_of(g, "bfloat16", k)
    kinds = collections.Counter()
    for r in rows:
        kinds[r.kind] += r.macs
    assert got["kinds_macs"] == pytest.approx(dict.fromkeys(io_bound.KINDS, 0.0) | kinds, rel=1e-12)
    assert len(got["products"]) < sum(r.kind == "fresh" for r in rows)
    C = block["gates"]
    assert C == io_bound.step_gates(g, "bfloat16")
    wc = F.whole_chain(got["system"], k)["whole_chain"]
    assert wc == pytest.approx(io_bound.whole_chain_of(io_bound.chain_classes(g, k), k)["whole_chain"], rel=1e-12)
    want = io_bound.bound([r.product() for r in rows if r.kind == "fresh"], c, k, C, wc)
    b = io_bound.bound(whole["products"], c, k, C, wc)
    assert b["proved"]["rho"] == pytest.approx(want["proved"]["rho"], rel=1e-12) and b["G"] == pytest.approx(want["G"])
    macs = sum(p["macs"] for p in whole["products"])
    assert macs / b["proved"]["blocks"] == pytest.approx(69_676, rel=1e-4)
    assert b["share"] == pytest.approx(0.552, abs=5e-4) and b["proved"]["rho"] == pytest.approx(2.870e6, rel=1e-3)
    assert b["proved"]["rho_no_cell"] == pytest.approx(1.637e5, rel=1e-3)
    assert (got["order"], got["trace_order"]) == (F.ORDER, "unknown")
    assert sum(p["macs"] for p in got["products"]) == pytest.approx(macs * (1 - got["shared_prefix"]["trimmed_macs_share"]))
    t = io_bound.bound(got["products"], c, k, C, made_whole=F.whole_chain(got["system"], k)["mu"])["proved"]
    assert t["rho_no_cell"] == pytest.approx(1.633e5, rel=1e-3) and t["rho"] == pytest.approx(2.869e6, rel=1e-3)
    m = replace(one_segment(), parts=(F.Part(gates=C, tokens=512, token_gates=C, affected_gates=C / 2),), products=got)
    one = F.bound(m, c, k)
    assert one["sensitivities"]["fresh_old"] == t["rho"] and one["sensitivities"]["fresh_no_cell"] == t["rho_no_cell"]
    assert one["inputs"]["fresh_old"]["capture"] == one["inputs"]["fresh_products"]["capture"] == products.CAPTURE
    assert one["inputs"]["fresh_old"]["subset"]["dropped"] == 4
    assert one["terms"]["fresh_products"] == F.he_figure(got["he"], c, C)["rho"]
    he = one["inputs"]["fresh_products"]
    assert round(one["terms"]["fresh_products"]) == 99_481 and (he["subset"], he["beta_w"], he["beta_x"]) == (
        "M", 15.9, 15.9)


def test_the_45m_dense_steps_old_figure_reads_its_whole_making_system_and_a_rank_one_class_voids_it():
    """The 45M dense step's products (its fixture rows, batch 2 at ctx 256) at the console: attention's 64-long chains
    sit below every counted operand, so the step's whole-making system gives whole_chain 1.0887 (the one-class rule at
    its least k_whole gave 1.033), and the old figure (fresh_old), with G without each sequence's first position, is
    2,869,161 with P-cell, on the best subset, which drops the 4 of 19 products the 256-long chains feed (5.8% of the
    fresh MACs), and 163,341 without it on all 19 (2,870,251 and 163,708 before the trim). A rank-1 adapter's class on
    the residual stream added to the system (r = k/(3·1²·16), fed by the 512-long chains and feeding them and the
    counted operands) closes a cycle with no fixed point through every class, and every counted product reads one of
    them, so the old figure is void with no subset's figure (`FRESH_VOID`), never a measured 0, while the products
    term, which reads no whole-making system, is measured and the same. A product no class feeds makes the old figure
    that product's, 24,154 with P-cell on 0.78% of the step's fresh multiplications, with the void's reason beside it.
    A broken P-match leaves both unmeasured with the op that breaks it. No counted product
    reads the embedding's rows before RoPE or attention's output makes them token-specific (layer 0's projections of
    them and their weight gradients are expansion products, PV and dP attention's), so G drops none of the 333; a
    counted row that read them would be dropped, and a 4.12 trace, whose rows don't say, is unmeasured until it is
    traced again (`FRESH_RETRACE`)."""
    rows = json.loads((FIXTURES / "dense_45m.products.json").read_text())
    got = F.products_of_trace(rows)
    c, k = policy.CONSOLE
    C = rows["gates"]

    def measured(products):
        return replace(one_segment(), parts=(F.Part(gates=C, tokens=512, token_gates=C, affected_gates=C / 2),),
                       products=products)

    def term(products):
        return F.fresh_products(measured(products), c, k)

    rho, why, inputs = term(got)
    assert why is None and round(rho) == round(inputs["fresh_cell"]) == 2_869_161 and "void" not in inputs
    assert round(inputs["fresh_no_cell"]) == 163_341
    assert inputs["gathered"] == {"dropped": 0, "of": 333, "dropped_macs_share": 0.0}
    assert inputs["shared_prefix"] == {"length": 1, "sequence": 256, "kept": 255 / 256,
                                       "trimmed_macs_share": pytest.approx(0.0020407, abs=1e-7)}
    kinds = {(F.product_kind(r), tuple(r["gathered"]), r["phase"], r["dims"]["k"]) for r in rows["rows"]
             if any(r["gathered"])}
    assert kinds == {("expansion", (True, False), "forward", 512), ("expansion", (False, True), "backward", 512),
                     ("attention", (False, True), "forward", 256), ("attention", (False, True), "backward", 64)}
    fresh = next(i for i, r in enumerate(rows["rows"]) if F.product_kind(r) == "fresh")
    one = rows | {"rows": [r | {"gathered": [True, False]} if i == fresh else r for i, r in enumerate(rows["rows"])]}
    dropped = F.products_of_trace(one)
    share = rows["rows"][fresh]["count"] * rows["rows"][fresh]["macs"] / sum(
        r["count"] * r["macs"] for r in rows["rows"] if F.product_kind(r) == "fresh")
    assert dropped["gathered"] == {"dropped": rows["rows"][fresh]["count"], "of": 333,
                                   "dropped_macs_share": pytest.approx(share, rel=1e-12)}
    assert len(dropped["products"]) == len(got["products"]) - 1
    old = F.products_of_trace(rows | {"capture": "4.12", "rows": [
        {x: y for x, y in r.items() if x != "gathered"} for r in rows["rows"]]})
    assert old["system"] is None and old["gathered"] is None and term(old)[1] == F.FRESH_RETRACE
    assert inputs["whole_making"]["e"] == pytest.approx(0.058322155, rel=1e-6)
    assert inputs["io_bound"]["whole_chain"] == pytest.approx(1.08874665, rel=1e-6)
    assert inputs["subset"] == {"dropped": 4, "of": 19, "dropped_macs_share": pytest.approx(0.0583702, rel=1e-6),
                                "k_whole_min": 512.0, "rule": "k_whole", "threshold": 512.0,
                                "e": pytest.approx(0.01964116, rel=1e-6), "dropped_fed": [256.0]}
    w = F.whole_chain(got["system"], k)
    assert w["mu"][64] == pytest.approx(0.3451, abs=1e-4) and w["r"][64] == pytest.approx(1 / 3)
    classes = [e | {"feeds": [*e["feeds"], [1, 1.0]]} if e["k"] == 512 else e for e in got["system"]["classes"]]
    system = got["system"] | {"classes": [*classes, {"k": 1, "c": 1 / 3, "P": 1.0, "feeds": [[512, 1.0]]}],
                              "counted": [*got["system"]["counted"], [1, 1.0]]}
    void, why, inputs = term(got | {"system": system})
    assert void is None and why == F.FRESH_VOID and inputs["fresh_cell"] is None and inputs["fresh_no_cell"] is None
    assert inputs["whole_making"] == {"e": None, "whole_chain": None, "diverged": [1, 64, 256, 512, 1408, 32000]}
    n = len(got["products"])
    assert inputs["void"] == {"reason": "the products count diverges; no subset has a proved bound",
                              "kept_macs_share": None, "diverged_fed": n}
    alone = got | {"system": system, "products": [got["products"][0] | {"fed": []}, *got["products"][1:]]}
    kept, why, inputs = term(alone)
    assert why is None and round(kept) == round(inputs["fresh_cell"]) == 24_154
    assert 1 < inputs["fresh_no_cell"] < kept and inputs["void"]["diverged_fed"] == n - 1
    sub = inputs["subset"]
    assert sub["rule"] in ("e", "mu") and sub["e"] == 0 and sub["dropped"] == n - 1
    assert inputs["void"]["kept_macs_share"] == pytest.approx(1 - sub["dropped_macs_share"], rel=1e-12)
    assert inputs["void"]["reason"] == ("the products count diverges; best subset 2.42e+04, keeping 0.778% of the "
                                        "step's fresh multiplications")
    voids = replace(one_segment(), products=got | {"system": system})
    b = F.bound(voids, c, k)
    assert b["terms"]["fresh_products"] == F.he_figure(got["he"], c, voids.gates)["rho"] > 1
    assert "fresh_products" not in b["unmeasured"] and b["void"] == {} and F.uncomputed(b) == []
    assert b["sensitivities"]["fresh_old"] is None and b["unmeasured"]["fresh_old"] == F.FRESH_VOID
    assert b["sensitivities"]["fresh_no_cell"] is None and b["unmeasured"]["fresh_no_cell"] == F.FRESH_VOID
    pub = F.published(voids, c, k, "A")
    assert pub["void"] is None and pub["sensitivities"]["fresh_old"] is None and pub["products_rests_on"]
    one = F.bound(measured(alone), c, k)
    assert one["sensitivities"]["fresh_old"] == kept and one["sensitivities"]["fresh_no_cell"] == inputs["fresh_no_cell"]
    assert not {"fresh_old", "fresh_no_cell"} & set(one["unmeasured"])
    assert one["terms"]["fresh_products"] == F.bound(measured(got), c, k)["terms"]["fresh_products"]
    fans = {"p_match": [{"op": "expand", "fan_out": 2.0, "count": 6}]}
    broken, why, inputs = term(got | fans)
    assert broken is None and f"breaks the premise that {F.P_MATCH}" in why and "expand (fan-out 2, 6×)" in why
    assert inputs["p_match"] == fans["p_match"]
    b = F.bound(measured(got | fans), c, k)
    assert b["terms"]["fresh_products"] is None and b["unmeasured"]["fresh_products"] == why


def test_a_traces_products_at_each_k_of_the_dial_give_the_term_there_which_the_dial_reads_as_c_times_a_number():
    """A trace from training_slowdown 4.12.1 on keeps its product rows at each k of the console's dial (training_slowdown.capture.products'
    `at`), so the products term is measured at each of them on that k's M, and off the dial it is unmeasured with the
    reason. At a fixed k the theorem of record's ρ is c/|C| times a function of k, and io_bound's blocks don't move
    with c, so the dial reads the old figure and the term as c times a number for each k (`fresh_per_c`, [fresh_old/c,
    term/c]). On the 45M dense step the term falls with k (99,481 at the console's 65,536 bits and 56,743 at 80,000 on
    M, 23,374 at the served block's 145,568 and 1,655 at 800,000 on M without its short contractions), and from 8e6
    bits up β(k) is low enough that the outputs' term exceeds the MACs', so it is 0, which ρ's floor reads as 1; the
    old figure is 2,869,161, 2,089,282 and 778,360 at the first three, and void from 800,000 bits up, where its
    whole-making system's one cycle, which feeds every counted product, has no fixed point, so the dial reads it null
    beside the term."""
    from training_slowdown.capture import products

    rows = json.loads((FIXTURES / "dense_45m.products.json").read_text())
    got = F.products_of_trace(rows)
    C = rows["gates"]
    m = replace(one_segment(), parts=(F.Part(gates=C, tokens=512, token_gates=C, affected_gates=C / 2),), products=got)
    assert sorted(got["at"], key=int) == [str(k) for k in products.DIAL_K if k != policy.CONSOLE[1]]
    assert all(b["k"] == float(x) and "at" not in b for x, b in got["at"].items())
    per = F.fresh_per_c(m)
    assert list(per) == [str(k) for k in products.DIAL_K]
    for k in products.DIAL_K:
        for c in (1e12, policy.CONSOLE[0]):
            term, why, inputs = F.he_products(m, c, k)
            assert why is None and inputs["k_classified"] == k
            assert term == pytest.approx(c * per[str(k)][1], rel=1e-12, abs=1e-12)
            old, why, inputs = F.fresh_products(m, c, k)
            if per[str(k)][0] is None:
                assert old is None and why == F.FRESH_VOID and inputs["void"]["kept_macs_share"] is None
            else:
                assert why is None and old == pytest.approx(c * per[str(k)][0], rel=1e-12)
    c = policy.CONSOLE[0]
    assert [round(c * per[str(k)][1]) for k in products.DIAL_K] == [99_481, 56_743, 23_374, 1_655] + [0] * 3
    assert [round(c * per[str(k)][0]) for k in (65_536, 80_000, 145_568)] == [2_869_161, 2_089_282, 778_360]
    assert [per[str(k)][0] is None for k in products.DIAL_K] == [False] * 3 + [True] * 4
    term, why, _ = F.he_products(m, c, 100_000)
    assert term is None and "keep none at it" in why and "each k of the console's dial" in why
    assert F.fresh_per_c(replace(m, products={x: v for x, v in got.items() if x != "at"})) == {"65536": per["65536"]}
    assert F.fresh_per_c(replace(m, products=None)) is None
    assert F.from_json(m.to_json()).products == m.products


def test_a_block_5_slots_term_grows_with_its_rank_and_never_falls_below_what_no_class_feeds():
    """The lowrank worker's single-slot LoRA steps on block 5 of the s-t20 start (one traced step of 64 sequences of
    1,024 tokens each, the console's product rows, `block5_slots.products.json`): each slot's term at the console grows
    with its rank (with P-cell w_down 0.155 at rank 1 to 6,177 at 256, w_gate and w_up 13.1 at 4 to 2,599 at 64, wo
    and wv about 33 at 16 and about 6,000 at 256; without it w_down 0.131 to 328, w_gate and w_up 8.03 to 370, wo and
    wv about 11 and over 300), never below the products no class feeds alone, and two micro-batches read as one.
    From rank 32 on the rank class makes e large (722 on w_down r64), which one factor for the whole step brought to
    1.4e-5, so the best subset drops the products it feeds; at ranks 1 and 4 the system has no cycle and is solved
    exactly; on all three MLP matrices the residual stream closes a cycle with no fixed point, so the term is void
    and is the figure of the product no diverged cycle feeds (187.7 with P-cell, 63.41 without, on 0.595 of the
    step's counted fresh MACs), a valid lower bound (coordinator 21:54Z)."""
    from training_slowdown import io_bound

    fx = json.loads((FIXTURES / "block5_slots.products.json").read_text())
    c, k = policy.CONSOLE
    terms, no_cell, inputs = {}, {}, {}
    for name, p in fx["programs"].items():
        block = {x: v for x, v in p.items() if x != "args"} | {"rows": [fx["rows"][i] for i in p["rows"]]}
        got = F.products_of_trace(block)
        C = block["gates"]
        m = replace(one_segment(), parts=(F.Part(gates=C, tokens=block["tokens"], token_gates=C, affected_gates=C / 2),),
                    products=got)
        terms[name], why, inputs[name] = F.fresh_products(m, c, k)
        no_cell[name] = inputs[name]["fresh_no_cell"]
        assert why is None and terms[name] == inputs[name]["fresh_cell"], name
        assert ("void" in inputs[name]) == bool(inputs[name]["whole_making"]["diverged"]), name
        assert no_cell[name] > 0 and got["shared_prefix"]["sequence"] == 1024, name
        unfed = io_bound.bound([q for q in got["products"] if not any(a > 0 for _, a in q["fed"])], c, k, C, 1.0)
        assert terms[name] >= unfed["proved"]["rho"] * (1 - 1e-12), name
        assert no_cell[name] >= unfed["proved"]["rho_no_cell"] * (1 - 1e-12), name
    slots = {"w_down": (1, 4, 16, 32, 64, 256), "w_gate,w_up": (4, 16, 32, 64), "wo": (16, 256), "wv": (16, 256)}
    for figure, want in ((terms, ([0.155, 2.399, 34.427, 121.743, 477.569, 6176.697],
                                  [13.103, 187.865, 663.727, 2598.916], [34, 6078, 32, 5825])),
                         (no_cell, ([0.131, 1.471, 11.631, 28.98, 67.933, 328.164],
                                    [8.03, 63.467, 157.993, 369.688], [11, 323, 11, 309]))):
        by_rank = {slot: [figure[f"{slot}-r{r}-m64"] for r in ranks] for slot, ranks in slots.items()}
        assert all(ts == sorted(ts) for ts in by_rank.values())
        assert [round(t, 3) for t in by_rank["w_down"]] == want[0]
        assert [round(t, 3) for t in by_rank["w_gate,w_up"]] == want[1]
        assert [round(t) for t in by_rank["wo"] + by_rank["wv"]] == want[2]
    assert terms["w_down-r64-m32"] == pytest.approx(terms["w_down-r64-m64"], rel=1e-9)
    sub = inputs["w_down-r64-m64"]["subset"]
    assert (sub["rule"], sub["e"], sub["dropped"], sub["of"], sub["dropped_fed"]) == ("e", 0, 2, 3, [64.0])
    assert inputs["w_down-r64-m64"]["whole_making"]["e"] == pytest.approx(722.16, rel=1e-5)
    assert inputs["w_down-r1-m64"]["whole_making"] == {"e": 0, "whole_chain": 1.0, "diverged": []}
    mlp = inputs["mlp-r16-m64"]
    assert mlp["whole_making"] == {"e": None, "whole_chain": None, "diverged": [16, 512, 1408, 65536]}
    assert mlp["void"]["diverged_fed"] == 2
    assert (mlp["subset"]["e"], mlp["subset"]["dropped"], mlp["subset"]["dropped_fed"]) == (0, 2, [512.0, 1408.0])
    assert terms["mlp-r16-m64"] == pytest.approx(187.699, rel=1e-5)
    assert no_cell["mlp-r16-m64"] == pytest.approx(63.41, rel=1e-4)
    assert mlp["void"]["kept_macs_share"] == pytest.approx(0.594595, rel=1e-5)
    assert mlp["void"]["reason"] == ("the products count diverges; best subset 188, keeping 59.5% of the step's fresh "
                                     "multiplications")
    assert [name for name, x in inputs.items() if "void" in x] == ["mlp-r16-m64"]


@pytest.mark.parametrize("name", ("dense_blocks", "lora_blocks", "wildcard_adapter"))
def test_a_fixtures_trace_keeps_its_product_rows_and_their_fresh_share(name):
    """Each fixture's product rows (the `.products.json` beside its circuit, training_slowdown.capture.products of the step the
    circuit traces, at training_slowdown 4.12.1): the capture's version, the console's k, the order unknown, the circuit's
    tokens and |C|, and by kind every product the op log places. On the tiny start no product is fresh: every counted
    candidate has an operand read through a repeating view or a broadcast below its width (each view read at its own
    value's rate), so the old figure is a measured 0 with nothing counted, and so is the term: M's operands are read
    far below their widths, and at every β M can take its outputs' part is at least its MACs'. Its whole-making
    system's cycle through attention's 32-long chains (r = 4/3) has no fixed point but feeds no counted product, so
    whole_chain is 1; P-match holds, and G drops nothing for an embedding's rows (the products that read them are expansion, θ or attention
    products) and trims nothing at the sequences' first positions (1,024 long, attention's, affected or not)."""
    from training_slowdown.capture import products
    from training_slowdown.capture.circuit import StepCircuit, measure

    block = json.loads((FIXTURES / f"{name}.products.json").read_text())
    f = measure(StepCircuit.load(FIXTURES / f"{name}.npz"))
    assert (block["capture"], block["k"], block["order"]) == (products.CAPTURE, policy.CONSOLE[1], "unknown")
    assert (block["tokens"], block["gates"]) == (f["tokens"], f["gates"]) and block["p_match"] == []
    got = F.products_of_trace(block)
    ops = json.loads((FIXTURES / f"{name}.ops.json").read_text())
    total = sum(got["kinds_macs"].values())
    assert total == sum(op["macs"] for op in ops if op["products"])
    assert got["kinds_macs"]["fresh"] == 0 and not got["products"]
    assert got["gathered"] == {"dropped": 0, "of": 0, "dropped_macs_share": 0.0}
    assert any(any(r["gathered"]) for r in block["rows"])
    b = F.bound(replace(one_segment(), products=got), 1e11, policy.CONSOLE[1])
    fp, he = b["inputs"]["fresh_old"], b["inputs"]["fresh_products"]
    assert b["sensitivities"]["fresh_no_cell"] == fp["io_bound"]["proved"]["rho_no_cell"] == 0 and "subset" not in fp
    assert fp["kinds_macs"] == got["kinds_macs"]
    assert b["terms"]["fresh_products"] == 0 and "fresh_products" not in b["unmeasured"] and he["below_width"]
    assert he["beta"] < 4 and he["macs_share"] < 1
    w = F.whole_chain(got["system"], policy.CONSOLE[1])
    assert (w["e"], w["whole_chain"]) == (0, 1.0) and bool(w["diverged"]) == (name != "wildcard_adapter")
    assert got["shared_prefix"] == {"length": 1, "sequence": 1024, "kept": 1023 / 1024, "trimmed_macs_share": 0.0}


def test_a_records_trace_rows_are_its_traced_steps_products():
    """`of_record` takes the record's trace's product rows (`formalism.trace.products`) as `Measured.products` over its
    traced steps, the tokens of the steps whose structure has an affected share over the traced step's (before the
    token factor scales a part's tokens), each fresh row times its count, summed in the reference circuit's chains; a
    step before the first update adds none, a record that has its own products keeps them, and one whose trace has no
    rows has none. Where the traced steps differ in shape (another structure, or other tokens), the dominant step's
    products stand in for them, scaled by tokens, and the products and their label say so (`STAND_IN`)."""
    row = {"count": 2, "phase": "forward", "op": "mm", "dims": {"batch": 1, "m": 4, "k": 8, "n": 8},
           "token_dims": ["m"], "macs": 256, "outputs": 32, "affected": [True, True], "widths": [16, 16],
           "rates": [16.0, 16.0], "paid": [True, True], "self": False, "b_c": 32, "k_whole": 8,
           "gates_per_mac": 1480.0, "cell_gates": 23682.0}
    theta = row | {"count": 1, "affected": [True, False], "rates": [16.0, 0.0], "paid": [True, False]}
    block = {"capture": F.PRODUCTS_FROM, "k": 65536.0, "order": "unknown", "tokens": 8, "gates": None,
             "rows": [row, theta]}
    log = [["fwd", "theta", "mm", "[[8]]", 1, 80]]
    record = {"steps": [{"step": 0, "tokens": 8, "struct": "b", "by": {"fwd/theta": 80}},
                        {"step": 1, "tokens": 8, "struct": "a", "by": {"fwd/theta": 80}},
                        {"step": 2, "tokens": 8, "struct": "a", "by": {"fwd/theta": 80}}],
              "structures": {"a": {"updated": [], "log": log, "affected_share": 0.5, "sigma": 1.0,
                                   "token_factor": 2.0, "delta_grad_bits": 10},
                             "b": {"updated": [], "log": log}},
              "formalism": {"trace": {"products": block}}}
    f = F.of_record(record)
    assert [p["tokens"] for p in f["parts"]] == [8, 32]
    p = f["products"]
    assert (p["k"], p["order"], p["trace_order"], p["capture"]) == (65536.0, F.ORDER, "unknown", F.PRODUCTS_FROM)
    assert p["products"] == [{"macs": 2 * 2 * 256, "outputs": 2 * 2 * 32, "k_dim": 8, "k_whole": 8, "b_a": 16,
                              "b_b": 16, "b_c": 32, "gates_per_mac": 1480.0, "cell_gates": 23682.0}]
    assert p["kinds_macs"]["fresh"] == 2 * 2 * 256 and p["kinds_macs"]["theta"] == 2 * 256
    assert F.of_record(record, through=1)["products"]["products"][0]["macs"] == 2 * 256
    m = F.complete(f, {"output": {"e_bytes": 10}}, SHAPE)
    assert m.products == p and m.labels["products"] == F.TRACED_PRODUCTS and "stand_in" not in p
    own = {"k": 1.0, "order": "chain", "products": []}
    assert F.of_record(record | {"formalism": {"trace": {"products": block}, "products": own}})["products"] == own
    assert "products" not in F.of_record(record | {"formalism": {"trace": {}}})
    grown = record | {"steps": [*record["steps"], {"step": 3, "tokens": 4, "struct": "c+traced4",
                                                   "by": {"fwd/theta": 40}}],
                      "structures": record["structures"] | {"c+traced4": record["structures"]["a"]}}
    g = F.of_record(grown)
    assert g["products"]["products"][0]["macs"] == 2.5 * 2 * 256
    assert g["products"]["stand_in"] == {"structures": ["a", "c"], "step_tokens": [4, 8], "traced_tokens": 8,
                                         "why": F.STAND_IN}
    assert g["labels"]["products"] == f"{F.TRACED_PRODUCTS}; {F.STAND_IN}"
    b = F.bound(F.complete(g, {"output": {"e_bytes": 10}}, SHAPE), 1e11, 65536.0)
    assert b["inputs"]["fresh_products"]["stand_in"] == g["products"]["stand_in"]


def test_a_row_an_older_te_core_ingested_reads_its_traces_rows_as_this_one_does():
    """A ledger row keeps the products its ingest took from its trace's rows, and the scorer's traces were ingested by
    the training_slowdown that captured them: at 4.8 with the trace's order unknown, which `fresh_products` never measures, and
    an upcast operand's product an expansion; at 4.10 and 4.11 with no `system`, which it reads as products of no
    trace, by the one-class rule (`ONE_CLASS`) that misses attention's chains; at 4.12 with a system that counts the
    products reading gathered rows, which its rows can't name; at 4.12.1 at the console's k only, though the trace
    keeps rows at each k of the dial (`at`). `complete` reads them as this training_slowdown's ingest of the same rows over the
    same traced steps (`traced_products`: their kinds' MACs over the rows' own, 2.5 here with a stand-in), so for the
    old figure (fresh_old) rows captured before `SYSTEM_FROM` say they need a retrace (`FRESH_RETRACE`) and rows with
    `fed` and `gathered`, from a capture that looked for products written elementwise and found none, are measured on
    their system, at each k their trace keeps rows at (the served block's here), with no re-ingest; products a 4.12
    ingest stored are never read as they are, since no capture before `ELEMENTWISE_FROM` looked for those products
    (`FRESH_UNCHECKED`); products a record has of its own stand. The products term reads M from the same rows: rows
    before `SYSTEM_FROM` need a retrace (`HE_RETRACE`), products stored without rows have no M (`HE_NO_ROWS`), and
    rows with `gathered` are measured at each k their trace keeps rows at (0 at the console's k: the run's 2.5 steps
    after the first are among the ⌊k/(τT)⌋ = 512 it leaves out)."""
    row = {"count": 2, "phase": "forward", "op": "mm", "dims": {"batch": 1, "m": 4, "k": 512, "n": 8},
           "token_dims": ["m"], "macs": 16384, "outputs": 32, "affected": [True, True], "widths": [16, 16],
           "rates": [16.0, 16.0], "paid": [True, False], "self": False, "b_c": 32, "k_whole": 512,
           "gates_per_mac": 1480.0, "cell_gates": 23682.0}
    theta = row | {"count": 1, "affected": [True, False], "rates": [16.0, 0.0], "paid": [True, False]}
    block = {"capture": F.PRODUCTS_FROM, "k": 65536.0, "order": "unknown", "tokens": 8, "gates": None,
             "rows": [row, theta]}
    log = [["fwd", "theta", "mm", "[[8]]", 1, 80]]

    def record(block):
        return {"steps": [{"step": 0, "tokens": 8, "struct": "a", "by": {"fwd/theta": 80}},
                          {"step": 1, "tokens": 8, "struct": "a", "by": {"fwd/theta": 80}},
                          {"step": 2, "tokens": 4, "struct": "a+traced4", "by": {"fwd/theta": 40}}],
                "structures": dict.fromkeys(("a", "a+traced4"), {"updated": [], "log": log, "affected_share": 0.5,
                                                                 "sigma": 1.0, "token_factor": 2.0,
                                                                 "delta_grad_bits": 10}),
                "formalism": {"trace": {"products": block}}}

    c, k, out = 1e11, 65536.0, {"output": {"e_bytes": 10}}

    def fresh(f, k=k):
        b = F.bound(F.complete(f, out, SHAPE), c, k)
        return b["sensitivities"]["fresh_old"], b["unmeasured"].get("fresh_old"), b["inputs"]["fresh_old"]

    def term(f, k=k):
        b = F.bound(F.complete(f, out, SHAPE), c, k)
        return b["terms"]["fresh_products"], b["unmeasured"].get("fresh_products")

    def before_412(f):
        return f | {"products": {x: v for x, v in f["products"].items() if x not in ("system", "p_match", "gathered")}}

    now = F.of_record(record(block))
    assert now["products"]["products"][0]["macs"] == 2.5 * 2 * 16384 and "stand_in" in now["products"]
    assert now["products"]["system"] is None and fresh(now)[:2] == (None, F.FRESH_RETRACE)
    assert term(now) == (None, F.HE_RETRACE)
    at_48 = now | {"products": {"k": 65536.0, "order": "unknown", "capture": F.PRODUCTS_FROM, "products": [],
                                "kinds_macs": {"expansion": 2.5 * 2 * 16384, "theta": 2.5 * 16384},
                                "stand_in": now["products"]["stand_in"]}}
    at_411 = before_412(now)
    stored = F.complete(at_48 | {"trace": {}}, out, SHAPE)
    assert stored.products == at_48["products"] and fresh(at_48 | {"trace": {}})[0] is None
    assert term(at_48 | {"trace": {}}) == (None, F.HE_NO_ROWS)
    one_class = fresh(at_411 | {"trace": {}})
    assert one_class[0] > 0 and one_class[2]["whole_making"] == F.ONE_CLASS
    fed = [r | {"fed": [], "P": 1.0} for r in block["rows"]]
    now_412 = F.of_record(record(block | {"capture": "4.12", "p_match": [], "rows": fed}))
    assert now_412["products"]["system"] is None
    at_412 = before_412(now_412)
    at_412["products"] |= {"system": F.system_of([r | {"gathered": [False, False]} for r in fed]), "p_match": []}
    assert fresh(at_412 | {"trace": {}})[:2] == (None, F.FRESH_UNCHECKED)
    for old, rows in ((at_48, now), (at_411, now), (at_412, now_412)):
        assert F.complete(old, out, SHAPE).products == rows["products"]
        assert fresh(old)[:2] == (None, F.FRESH_RETRACE) and term(old) == (None, F.HE_RETRACE)
    captured = block | {"capture": F.ELEMENTWISE_FROM, "p_match": [], "elementwise_products": [],
                        "rows": [r | {"gathered": [False, False]} for r in fed]}
    now = F.of_record(record(captured))
    assert now["products"]["system"] is not None
    old, why, got = fresh(before_412(now))
    assert (old, why) == fresh(now)[:2] and old > 0 and why is None and got["whole_making"] != F.ONE_CLASS
    assert term(before_412(now)) == term(now) == (0.0, None)
    served = 145_568.0
    dial = F.of_record(record(captured | {"at": {f"{served:.0f}": {"k": served, "rows": captured["rows"],
                                                                   "p_match": []}}}))
    at_4121 = dial | {"products": {x: v for x, v in dial["products"].items() if x != "at"}}
    assert "keep none at it" in fresh(at_4121 | {"trace": {}}, served)[1]
    assert "keep none at it" in term(at_4121 | {"trace": {}}, served)[1]
    old, why, got = fresh(at_4121, served)
    assert why is None and old > 0 and (old, why) == fresh(dial, served)[:2] and got["k_classified"] == served
    assert term(at_4121, served) == term(dial, served) and term(dial, served)[1] is None
    own = {"k": 65536.0, "order": "chain", "products": []}
    assert F.complete(now | {"products": own}, out, SHAPE).products == own


def test_a_trace_of_another_program_than_the_runs_is_never_priced():
    """Coordinator 20:42Z: a products term is priced only on a trace of the run's own program. A trace keeps no list
    of the tensors it trains, so `complete` reads the row's op log beside the trace's rows when the row is read
    (`other_program`, whatever training_slowdown ingested it): where every matrix-product shape the rows run is a product the
    log ran, in either orientation, the term is priced as before; where one isn't (a rank, a width or a batch the
    capture took and the run didn't), the term is unmeasured at every k the trace keeps rows at, with the counts as
    its reason (`FRESH_OTHER`), and so are the old figure's fresh_old and fresh_no_cell. A batched product is no match
    for a plain one, and a row with no op log isn't checked."""
    row = {"count": 2, "phase": "forward", "op": "mm", "dims": {"batch": 1, "m": 4, "k": 512, "n": 8},
           "token_dims": ["m"], "macs": 16384, "outputs": 32, "affected": [True, True], "widths": [16, 16],
           "rates": [16.0, 16.0], "paid": [True, True], "self": False, "b_c": 32, "k_whole": 512,
           "gates_per_mac": 1480.0, "cell_gates": 23682.0, "fed": [], "P": 1.0, "gathered": [False, False]}
    served = 145_568.0
    block = {"capture": F.ELEMENTWISE_FROM, "k": 65536.0, "order": "unknown", "tokens": 8, "gates": None,
             "p_match": [], "elementwise_products": [], "rows": [row],
             "at": {f"{served:.0f}": {"k": served, "rows": [row], "p_match": []}}}
    log = [["fwd", "updated", "mm", "[[4, 512], [512, 8]]", 2, 2 * 16384]]
    f = F.of_record({"steps": [{"step": 0, "tokens": 8, "struct": "a", "by": {"fwd/updated": 2 * 16384}}],
                     "structures": {"a": {"updated": [], "log": log, "affected_share": 0.5, "sigma": 1.0,
                                          "token_factor": 1.0, "delta_grad_bits": 10}},
                     "formalism": {"trace": {"products": block}}})
    assert "other_program" not in f["products"]

    def fresh(ops, k=65536.0):
        r = {"output": {"e_bytes": 10}} | ({} if ops is None else {"macs_log": {"by": {}, "ops": ops}})
        return F.bound(F.complete(f, r, SHAPE), 1e11, k)

    for ops in (None, {}, {"fwd updated mm [[4, 512], [512, 8]]": [2, 32768]},
                {"bwd grad mm [[8, 512], [512, 4]]": [2, 32768], "fwd theta mm [[4, 8], [8, 3]]": [1, 96]}):
        for k in (65536.0, served):
            b = fresh(ops, k)
            assert b["terms"]["fresh_products"] >= 0 and "fresh_products" not in b["unmeasured"], (ops, k)
            assert b["sensitivities"]["fresh_old"] > 0 and "other_program" not in b["inputs"]["fresh_products"]
    for ops, ran in (({"fwd updated mm [[4, 512], [512, 3]]": [2, 12288]}, 1),
                     ({"fwd updated bmm [[1, 4, 512], [1, 512, 3]]": [2, 12288],
                       "fwd updated bmm [[2, 4, 512], [2, 512, 8]]": [1, 32768]}, 1)):
        for k in (65536.0, served):
            b = fresh(ops, k)
            why = F.FRESH_OTHER.format(unmatched=1, shapes=1, ran=ran, largest="4×512 by 512×8")
            assert b["terms"]["fresh_products"] is None and b["unmeasured"]["fresh_products"] == why, (ops, k)
            assert b["inputs"]["fresh_products"]["other_program"] == {"shapes": 1, "unmatched": 1, "ran": ran,
                                                                      "largest": [[4, 512, 8]]}
            assert b["sensitivities"]["fresh_old"] is None and b["sensitivities"]["fresh_no_cell"] is None
            assert b["unmeasured"]["fresh_old"] == b["unmeasured"]["fresh_no_cell"] == why and b["void"] == {}
            assert "io_bound" not in b["inputs"]["fresh_old"] and "beta" not in b["inputs"]["fresh_products"]
    assert F.fresh_per_c(F.complete(f, {"output": {"e_bytes": 10}, "macs_log": {"by": {}, "ops": ops}}, SHAPE)) == {
        "65536": None, f"{served:.0f}": None}


def test_a_ledger_row_publishes_what_its_trace_rows_give_at_every_dial_k_whatever_te_core_ingested_it(monkeypatch):
    """The 45M dense step's capture (`dense_45m.products.json`) as a ledger row's trace. Ingested at 4.11 (rows
    with no `fed`, `P` or `gathered`, products with no system) the row published the one-class rule's number as the old
    figure; read now it publishes the old figure (fresh_old) and the products term unmeasured with their retrace
    reasons until it is traced again. Ingested at
    4.12.1 (products at the console's k only, none at the dial's other k, no `fed` and no shared-prefix trim) it
    publishes at the served block's 145,568 bits, and at every k of the dial, what a fresh ingest of the same record
    does, where its stored products kept none there."""
    from training_slowdown.capture.products import DIAL_K

    fixture = json.loads((FIXTURES / "dense_45m.products.json").read_text())
    log = [["fwd", "theta", "mm", "[[8]]", 1, 80]]

    def row(block):
        record = {"steps": [{"step": 0, "tokens": 512, "struct": "a", "by": {"fwd/theta": 80}}],
                  "structures": {"a": {"updated": [], "log": log, "affected_share": 0.5, "sigma": 1.0,
                                       "token_factor": 1.0, "delta_grad_bits": 1e9}},
                  "formalism": {"trace": {"products": block, "data_dependent": [], "carried_scalars": [],
                                          "affected_share": 0.5, "u1_bits": 1e6}}}
        return {"formalism": F.of_record(record), "output": {"e_bytes": 1e8}}

    def fresh(r, c, k):
        b = F.bound(F.measure(r, SHAPE), c, k)
        return b["sensitivities"]["fresh_old"], b["unmeasured"].get("fresh_old")

    def term(r, c, k):
        b = F.bound(F.measure(r, SHAPE), c, k)
        return b["terms"]["fresh_products"], b["unmeasured"].get("fresh_products")

    def stored(r):
        trace = {x: v for x, v in r["formalism"]["trace"].items() if x != "products"}
        return r | {"formalism": r["formalism"] | {"trace": trace}}

    c = policy.CONSOLE[0]
    rows = [{x: v for x, v in r.items() if x not in ("fed", "P", "gathered")} for r in fixture["rows"]]
    at_411 = row({x: v for x, v in fixture.items() if x not in ("at", "p_match", "elementwise_products")}
                 | {"capture": "4.11", "rows": rows})
    at_411["formalism"]["products"] = {x: v for x, v in at_411["formalism"]["products"].items()
                                       if x not in ("system", "p_match", "gathered", "shared_prefix")}
    one_class, why = fresh(stored(at_411), c, policy.CONSOLE[1])
    assert one_class > 0 and why is None
    assert fresh(at_411, c, policy.CONSOLE[1]) == (None, F.FRESH_RETRACE)
    assert term(stored(at_411), c, policy.CONSOLE[1]) == term(at_411, c, policy.CONSOLE[1]) == (None, F.HE_RETRACE)
    was = F.published(F.measure(stored(at_411), SHAPE), c, policy.CONSOLE[1], "A")
    p = F.published(F.measure(at_411, SHAPE), c, policy.CONSOLE[1], "A")
    assert was["sensitivities"]["fresh_old"] == one_class and p["sensitivities"]["fresh_old"] is None
    assert was["terms"]["fresh_products"] is p["terms"]["fresh_products"] is None

    now = row(fixture)
    with monkeypatch.context() as mp:
        mp.setattr(F, "SHARED_PREFIX", 0)
        at_4121 = row(fixture)
    at_4121["formalism"]["products"] = {x: v for x, v in at_4121["formalism"]["products"].items()
                                        if x not in ("at", "shared_prefix")}
    at_4121["formalism"]["products"]["products"] = [{x: v for x, v in q.items() if x != "fed"}
                                                    for q in at_4121["formalism"]["products"]["products"]]
    served = 145_568.0
    assert "keep none at it" in fresh(stored(at_4121), c, served)[1]
    assert "keep none at it" in term(stored(at_4121), c, served)[1]
    assert F.measure(at_4121, SHAPE).products == F.measure(now, SHAPE).products
    old, why = fresh(at_4121, c, served)
    assert why is None and old == fresh(now, c, served)[0] > 0
    for k in DIAL_K:
        assert fresh(at_4121, c, k) == fresh(now, c, k) and term(at_4121, c, k) == term(now, c, k)
        assert F.published(F.measure(at_4121, SHAPE), c, k, "A") == F.published(F.measure(now, SHAPE), c, k, "A")


def test_a_ledger_rows_log_puts_theta_and_attention_products_aside_and_the_rest_unknown():
    """On a macs_log (training_slowdown.tiles.of_log): a product whose non-token operand is θ (its role theta) is θ's, a fused
    attention op's products are attention's, and an updated one's, whose operands the log says only depend on the
    run's state, is unknown; an op run twice is one row of twice its MACs."""
    sdpa = 2 * 16 * 256 * 64 * 256
    log = {"by": {}, "ops": {
        "fwd theta mm [[2048, 64], [64, 192]]": [1, 2048 * 64 * 192],
        "fwd updated mm [[2048, 64], [64, 192]] bfloat16": [2, 2 * 2048 * 64 * 192],
        "fwd act _scaled_dot_product_flash_attention [[2, 8, 256, 64], [2, 8, 256, 64], [2, 8, 256, 64]] bfloat16":
            [1, sdpa]}}
    m = F.complete({}, {"macs_log": log, "output": {"e_bytes": 10}}, SHAPE)
    got = F.products_of_ops(m.ops)
    assert [(r["kind"], r["dims"]) for r in got["rows"]] == [
        ("theta", [1, 2048, 64, 192]), ("unknown", [1, 2048, 64, 192]), ("attention", [16, 256, 64, 256]),
        ("attention", [16, 256, 256, 64])]
    assert got["macs"] == {"theta": 2048 * 64 * 192, "attention": sdpa, "unknown": 2 * 2048 * 64 * 192}
    assert got["rows"][1]["outputs"] == 2 * 2048 * 192 and got["unknown_share"] == pytest.approx(0.24)


def test_reduction_update_and_the_token_form_are_sensitivities_never_the_bound():
    """They come out labelled beside the kinds: reduction and update where the step's token work fills two or more
    blocks (b = 8 at c = 1e11), unmeasured without the cuts, and reading B's token form, Slowdown's; asking for them in
    the max is refused. A record with no entries has no measured kind, so where Fit fails ρ is unmeasured."""
    m, c, k = measured(u1=1e9), 1e11, 1e5
    b = F.bound(m, c, k)
    assert b["sensitivities"]["token_form"] == F.at(m, c, k)["slowdown"]
    assert b["sensitivities"]["reduction"] is None and set(b["unmeasured"]) == {"fresh_products", "token", "decoding",
                                                                               "per_gate", "reduction", "update",
                                                                               "fresh_old", "fresh_no_cell"}
    assert "no entries" in b["unmeasured"]["token"]
    cut = replace(m, parts=(replace(m.parts[0], partial_bits=1e6, grad_bits=2e6),))
    b = F.bound(cut, c, k)
    assert b["inputs"]["reduction"]["blocks"] == [8] and set(b["unmeasured"]) == {"fresh_products", "token", "decoding",
                                                                                 "per_gate", "fresh_old",
                                                                                 "fresh_no_cell"}
    assert b["sensitivities"]["reduction"] == pytest.approx(c * 7 * 1e6 / (k * 1e12))
    assert b["sensitivities"]["update"] == pytest.approx(c * 2e6 / (k * 1e12))
    assert b["max"] is None and b["binding"] is None and set(b["labels"]) == set(F.SENSITIVITIES)
    assert "split by tokens" in b["labels"]["reduction"] and "reading B" in b["labels"]["token_form"]
    r = F.at(cut, c, k, "A")
    assert not r["fit"] and r["rho"] is None and F.at(cut, c, k)["rho"] == max(1.0, b["sensitivities"]["token_form"])
    for kinds in (("token", "reduction"), ("update",), ("token_form",), ("per_gate",)):
        with pytest.raises(ValueError, match="sensitivities, never in the bound's max"):
            F.bound(cut, c, k, kinds=kinds)


def test_a_rows_op_log_is_its_tiles_and_its_block_keeps_its_own():
    """`complete` reads a block's `ops`, else the row's macs_log (training_slowdown.tiles.of_log), labelled; from_json keeps
    them."""
    log = {"by": {"fwd/updated": 2048 * 64 * 192}, "ops": {"fwd updated mm [[2048, 64], [64, 192]]": [1, 2048 * 64 * 192]}}
    m = F.complete({}, {"macs_log": log, "output": {"e_bytes": 10}}, SHAPE)
    assert m.ops[0]["products"] == [[1, 2048, 64, 192]] and m.labels["ops"] == F.OPS
    assert F.complete({"ops": []}, {"macs_log": log, "output": {"e_bytes": 10}}, SHAPE).ops == []
    assert F.from_json(m.to_json()) == m


