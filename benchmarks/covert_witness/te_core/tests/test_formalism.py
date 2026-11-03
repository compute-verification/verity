"""te_core.formalism and te_core.policy: Fit and Slowdown as the formalism v2 states them (a frozen tail always
charged), the attack side beside them, the stubs, assumptions and upper bounds labelled, the grid."""
import dataclasses
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from te_core import VERSION as F_VERSION
from te_core import formalism as F
from te_core import frontier, policy
from te_core.gates import per_mac

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
    ({"data_dependent": None}, 10, "read-backs unchecked: traced before te_core 4.2 (no data_dependent)"),
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
    # attention over the whole prompt at each prompt token, as te_core.flops counts the causal SDPA
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
    gates, the operand of the shorter side free (te_core.tiles)."""
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
    assert b["max"] == b["terms"]["decoding"] and b["label"] == f"te_core {F_VERSION}, reading A"
    assert b["terms"]["fresh_products"] is None and b["unmeasured"]["fresh_products"].startswith(
        f"unmeasured: {F.LOG_UNKNOWN}; 100% of its products' MACs")
    assert b["unmeasured"]["fresh_products"].endswith(f"a trace with te_core {F.PRODUCTS_FROM} or later keeps them "
                                                      f"(te_core.capture.products)")
    assert b["inputs"]["fresh_products"] == {"source": "op log", "order": "chain", "rows": 1, "unknown_share": 1.0,
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


def test_the_fresh_products_kind_is_io_bounds_bound_on_the_runs_products():
    """The term is te_core.io_bound's bound with Λ proved on the record's products and |C|, without P-cell (a premise
    on the partition), with the whole-cell figure beside it as the sensitivity fresh_cell, on every console row. The
    products are summed in the reference circuit's chains (`ORDER`), so a run's products are measured whether or not
    they give an order; only a circuit that declares a tree is unmeasured, with io_bound's chain figure in the inputs,
    and so is a term at a k other than the one the products were classified at. The term is the same over S steps of
    the same products, and the products survive the record's JSON."""
    from te_core import io_bound

    c, k = 1e11, policy.CONSOLE[1]
    chain = replace(one_segment(), products={"k": k, "order": "chain", "products": [SQUARE], "system": SYSTEM})
    b = F.bound(chain, c, k)
    wc = F.whole_chain(SYSTEM, k)["whole_chain"]
    assert wc == pytest.approx(io_bound.whole_chain(k, 512, 16), rel=1e-9)
    proved = io_bound.bound([SQUARE], c, k, 1e12, wc)["proved"]
    rho = proved["rho_no_cell"]
    assert not F.P_CELL and F.ORDER == "chain" and 10 < rho < 100 < proved["rho"] < 1e3
    assert b["terms"]["fresh_products"] == rho and b["binding"] == "fresh_products"
    assert b["sensitivities"]["fresh_cell"] == proved["rho"] and "P-cell" in b["labels"]["fresh_cell"]
    assert "fresh_products" not in b["unmeasured"] and "fresh_cell" not in b["unmeasured"]
    assert b["inputs"]["fresh_products"]["io_bound"]["share"] == pytest.approx(0.5, rel=1e-12)
    assert F.published(chain)["reading"] == ("A" if F.READING_A else "B")
    one = F.published(chain, reading="A")
    assert one["binding_kind"] == "fresh_products" and one["rho"] == pytest.approx(rho * policy.CONSOLE[0] / c,
                                                                                   rel=1e-12)
    assert one["sensitivities"]["fresh_cell"] == pytest.approx(proved["rho"] * policy.CONSOLE[0] / c, rel=1e-12)
    assert "P-chain" in one["rests_on"] and "without P-cell" in one["rests_on"]
    assert F.bound(replace(chain, products=chain.products | {"order": None}), c, k)["terms"]["fresh_products"] == rho
    tree = F.bound(replace(chain, products=chain.products | {"order": "tree"}), c, k)
    assert tree["terms"]["fresh_products"] is None and "declares its products summed in a tree" in tree["unmeasured"][
        "fresh_products"]
    assert "chain constant only" in tree["unmeasured"]["fresh_products"]
    assert tree["inputs"]["fresh_products"]["io_bound"]["proved"]["rho_no_cell"] == rho
    assert tree["sensitivities"]["fresh_cell"] is None and tree["unmeasured"]["fresh_cell"] == tree["unmeasured"][
        "fresh_products"]
    elsewhere = F.bound(chain, c, 80_000)
    assert elsewhere["terms"]["fresh_products"] is None and "classified at k = 65536" in elsewhere["unmeasured"][
        "fresh_products"]
    S = 7
    steps = replace(chain, parts=(replace(chain.parts[0], gates=S * 1e12, tokens=S * 1e3, token_gates=S * 1e12,
                                          affected_gates=S * 5e11),),
                    products=chain.products | {"products": [SQUARE | {"macs": S * SQUARE["macs"],
                                                                      "outputs": S * SQUARE["outputs"]}]})
    assert F.bound(steps, c, k)["terms"]["fresh_products"] == pytest.approx(rho, rel=1e-10)
    assert F.from_json(chain.to_json()).products == chain.products
    assert "fresh operands" in F.RESTS_ON["fresh_products"] and "P-chain" in F.RESTS_ON["fresh_products"]
    assert all(x in F.RESTS_ON["fresh_products"] for x in ("P-match", "P-width extended"))
    assert all(x in F.RESTS_ON_CELL["fresh_products"] for x in ("P-match", "P-width (extended)", "P-chain"))
    assert F.bound(replace(chain, products=chain.products | {"system": None}), c, k)["unmeasured"][
        "fresh_products"] == F.FRESH_RETRACE
    hand = F.bound(replace(chain, products={"k": k, "order": "chain", "products": [SQUARE]}), c, k)
    assert hand["terms"]["fresh_products"] == io_bound.bound([SQUARE], c, k, 1e12)["proved"]["rho_no_cell"]
    assert hand["inputs"]["fresh_products"]["whole_making"] == F.ONE_CLASS


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
    chain = replace(one_segment(), products={"k": policy.CONSOLE[1], "order": "chain", "products": [SQUARE],
                                             "system": SYSTEM})
    full = F.published(chain, 1e10, policy.CONSOLE[1], "A")
    assert full["binding_kind"] == "fresh_products" and full["rho"] == full["terms"]["fresh_products"] > 1
    assert not full["partial"] and full["rests_on"] == F.RESTS_ON_CELL["fresh_products"]


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
    chain = replace(one_segment(), products={"k": k, "order": "chain", "products": [SQUARE], "system": SYSTEM})
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

    from te_core import io_bound
    from te_core.capture import capture_step

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
    """`products_of_ops` on a traced step's op log (the fixtures' `.ops.json`, te_core.tiles.of_graph): a row for each
    product the log places, its op's MACs split by the products' sizes, its output entries its MACs over K, its
    operands' widths the log's and its partial sums' io_bound's; θ where its class is registered, else unknown (the
    tiny start's attention is unfused bmm, whose token dims a log doesn't keep). The order is the reference circuit's,
    and nothing is counted, so the fresh-products term is unmeasured with the unknown share in its reason."""
    from te_core import io_bound

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
    assert b["terms"]["fresh_products"] is None and b["inputs"]["fresh_products"]["rows"] == len(got["rows"])
    assert f"; {got['unknown_share']:.0%} of its products' MACs" in b["unmeasured"]["fresh_products"]


@pytest.mark.slow
def test_a_dense_steps_product_rows_give_io_bounds_bound_on_it():
    """The 45M start's dense step (te_core.io_bound.capture, Muon on the block matrices, batch 2 at ctx 256) through
    te_core.capture.products and `products_of_trace` is the products io_bound.rows_of counts on it, its identical rows
    merged, and their whole-making system is io_bound.chain_classes', so io_bound's bound is the same: at the console's
    policy about 69,676 fresh MACs a block, 0.552 of |C| and ρ 2.87e6 with P-cell. The trace's order is unknown
    and the reference circuit's chain stands, so the term is measured: the figure without P-cell, 1.637e5, with 2.87e6
    beside it as fresh_cell. The rows, with those at the console dial's other k, are the fixture the
    vectors read (`dense_45m.products.json`). The one check of the strong bound's figure on a model's own shape (about
    6 s on a CPU): the fixtures' tiny start has almost no fresh products."""
    import collections

    from te_core import io_bound
    from te_core.capture import products

    g = io_bound.capture("dense", size="s", vocab=32000, batch=2, ctx=256, micro_steps=1)
    c, k = policy.CONSOLE
    block = products.rows(g, k)
    assert json.loads(json.dumps(products.rows_on_dial(g))) == json.loads(
        (FIXTURES / "dense_45m.products.json").read_text())
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
    b = io_bound.bound(got["products"], c, k, C, wc)
    assert b["proved"]["rho"] == pytest.approx(want["proved"]["rho"], rel=1e-12) and b["G"] == pytest.approx(want["G"])
    macs = sum(p["macs"] for p in got["products"])
    assert macs / b["proved"]["blocks"] == pytest.approx(69_676, rel=1e-4)
    assert b["share"] == pytest.approx(0.552, abs=5e-4) and b["proved"]["rho"] == pytest.approx(2.870e6, rel=1e-3)
    assert b["proved"]["rho_no_cell"] == pytest.approx(1.637e5, rel=1e-3)
    assert (got["order"], got["trace_order"]) == (F.ORDER, "unknown")
    m = replace(one_segment(), parts=(F.Part(gates=C, tokens=512, token_gates=C, affected_gates=C / 2),), products=got)
    one = F.bound(m, c, k)
    assert one["terms"]["fresh_products"] == b["proved"]["rho_no_cell"] and one["binding"] == "fresh_products"
    assert one["sensitivities"]["fresh_cell"] == b["proved"]["rho"]
    assert one["inputs"]["fresh_products"]["capture"] == products.CAPTURE and "subset" not in one["inputs"]["fresh_products"]


def test_the_45m_dense_steps_term_reads_its_whole_making_system_and_a_rank_one_class_voids_it():
    """The 45M dense step's products (its fixture rows, batch 2 at ctx 256) at the console: attention's 64-long chains
    sit below every counted operand, so the step's whole-making system gives whole_chain 1.0887 (the one-class rule at
    its least k_whole gave 1.033), and the term is 163,708 without P-cell, 2,870,251 with it (the io-bound worker's
    table). A rank-1 adapter's class on the residual stream added to the system (r = k/(3·1²·16), fed by the 512-long
    chains and feeding them and the counted operands) has no fixed point, so the term is void: unmeasured
    (`FRESH_VOID`), never a measured 0. A broken P-match is unmeasured with the op that breaks it. No counted product
    reads the embedding's rows before RoPE or attention's output makes them token-specific (layer 0's projections of
    them and their weight gradients are expansion products, PV and dP attention's), so G drops none of the 333; a
    counted row that read them would be dropped, and a 4.12 trace, whose rows don't say, is unmeasured until it is
    traced again (`FRESH_RETRACE`)."""
    rows = json.loads((FIXTURES / "dense_45m.products.json").read_text())
    got = F.products_of_trace(rows)
    c, k = policy.CONSOLE
    C = rows["gates"]

    def term(products):
        m = replace(one_segment(), parts=(F.Part(gates=C, tokens=512, token_gates=C, affected_gates=C / 2),),
                    products=products)
        return F.fresh_products(m, c, k)

    rho, why, inputs = term(got)
    assert why is None and round(rho) == 163_708 and round(inputs["fresh_cell"]) == 2_870_251
    assert inputs["gathered"] == {"dropped": 0, "of": 333, "dropped_macs_share": 0.0}
    kinds = {(F.product_kind(r), tuple(r["gathered"]), r["phase"], r["dims"]["k"]) for r in rows["rows"]
             if any(r["gathered"])}
    assert kinds == {("expansion", (True, False), "forward", 512), ("expansion", (False, True), "backward", 512),
                     ("attention", (False, True), "forward", 256), ("attention", (False, True), "backward", 64)}
    fresh = next(i for i, r in enumerate(rows["rows"]) if F.product_kind(r) == "fresh")
    one = rows | {"rows": [r | {"gathered": [True, False]} if i == fresh else r for i, r in enumerate(rows["rows"])]}
    dropped = F.products_of_trace(one)
    share = rows["rows"][fresh]["count"] * rows["rows"][fresh]["macs"] / sum(p["macs"] for p in got["products"])
    assert dropped["gathered"] == {"dropped": rows["rows"][fresh]["count"], "of": 333,
                                   "dropped_macs_share": pytest.approx(share, rel=1e-12)}
    assert len(dropped["products"]) == len(got["products"]) - 1
    old = F.products_of_trace(rows | {"capture": "4.12", "rows": [
        {x: y for x, y in r.items() if x != "gathered"} for r in rows["rows"]]})
    assert old["system"] is None and old["gathered"] is None and term(old)[1] == F.FRESH_RETRACE
    assert inputs["whole_making"]["e"] == pytest.approx(0.058322155, rel=1e-6)
    assert inputs["io_bound"]["whole_chain"] == pytest.approx(1.08874665, rel=1e-6) and "subset" not in inputs
    w = F.whole_chain(got["system"], k)
    assert w["mu"][64] == pytest.approx(0.3451, abs=1e-4) and w["r"][64] == pytest.approx(1 / 3)
    classes = [e | {"feeds": [*e["feeds"], [1, 1.0]]} if e["k"] == 512 else e for e in got["system"]["classes"]]
    system = got["system"] | {"classes": [*classes, {"k": 1, "c": 1 / 3, "P": 1.0, "feeds": [[512, 1.0]]}],
                              "counted": [*got["system"]["counted"], [1, 1.0]]}
    void, why, inputs = term(got | {"system": system})
    assert void is None and why == F.FRESH_VOID and inputs["fresh_cell"] is None
    b = F.bound(replace(one_segment(), products=got | {"system": system}), c, k)
    assert b["terms"]["fresh_products"] is None and b["unmeasured"]["fresh_products"] == F.FRESH_VOID
    assert b["sensitivities"]["fresh_cell"] is None and b["unmeasured"]["fresh_cell"] == F.FRESH_VOID
    broken, why, inputs = term(got | {"p_match": [{"op": "expand", "fan_out": 2.0, "count": 6}]})
    assert broken is None and "breaks P-match" in why and "expand (fan-out 2, 6×)" in why
    assert inputs["p_match"] == [{"op": "expand", "fan_out": 2.0, "count": 6}]


def test_a_traces_products_at_each_k_of_the_dial_give_the_term_there_which_the_dial_reads_as_c_times_a_number():
    """A trace from te_core 4.12.1 on keeps its product rows at each k of the console's dial (te_core.capture.products'
    `at`), so the fresh-products term is measured at each of them on that k's products and whole-making system, and
    off the dial it is unmeasured with the reason. At a fixed k io_bound's blocks don't move with c, so the dial reads
    the term as c times a number for each k (`fresh_per_c`, [without P-cell, with it]). On the 45M dense step the term
    falls with k (163,708 at the console's 65,536 bits, 131,294 at 80,000 and 64,497 at the served block's 145,568)
    and is void from 800,000 bits up, where its whole-making system has no fixed point."""
    from te_core.capture import products

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
            term, why, inputs = F.fresh_products(m, c, k)
            assert inputs["k_classified"] == k
            if per[str(k)] is None:
                assert term is None and why == F.FRESH_VOID
            else:
                assert why is None and term == pytest.approx(c * per[str(k)][0], rel=1e-12)
                assert inputs["fresh_cell"] == pytest.approx(c * per[str(k)][1], rel=1e-12)
    c = policy.CONSOLE[0]
    assert [round(c * per[str(k)][0]) for k in (65_536, 80_000, 145_568)] == [163_708, 131_294, 64_497]
    assert [per[str(k)] is None for k in products.DIAL_K] == [False] * 3 + [True] * 4
    term, why, _ = F.fresh_products(m, c, 100_000)
    assert term is None and "keep none at it" in why and "DIAL_K" in why
    assert F.fresh_per_c(replace(m, products={x: v for x, v in got.items() if x != "at"})) == {"65536": per["65536"]}
    assert F.fresh_per_c(replace(m, products=None)) is None
    assert F.from_json(m.to_json()).products == m.products


@pytest.mark.parametrize("name", ("dense_blocks", "lora_blocks", "wildcard_adapter"))
def test_a_fixtures_trace_keeps_its_product_rows_and_their_fresh_share(name):
    """Each fixture's product rows (the `.products.json` beside its circuit, te_core.capture.products of the step the
    circuit traces, at te_core 4.12.1): the capture's version, the console's k, the order unknown, the circuit's
    tokens and |C|, and by kind every product the op log places. On the tiny start no product is fresh: every counted
    candidate has an operand read through a repeating view or a broadcast below its width (each view read at its own
    value's rate), so the term is a measured 0 with nothing counted, though its whole-making system (attention's
    32-long chains, r = 4/3) has no fixed point; P-match holds, and G drops nothing for an embedding's rows (the
    products that read them are expansion, θ or attention products)."""
    from te_core.capture import products
    from te_core.capture.circuit import StepCircuit, measure

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
    fp = b["inputs"]["fresh_products"]
    assert b["terms"]["fresh_products"] == fp["io_bound"]["proved"]["rho_no_cell"] == 0 and "subset" not in fp
    assert fp["kinds_macs"] == got["kinds_macs"]
    assert math.isinf(F.whole_chain(got["system"], policy.CONSOLE[1])["whole_chain"]) == (name != "wildcard_adapter")


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


def test_a_ledger_rows_log_puts_theta_and_attention_products_aside_and_the_rest_unknown():
    """On a macs_log (te_core.tiles.of_log): a product whose non-token operand is θ (its role theta) is θ's, a fused
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
                                                                               "fresh_cell"}
    assert "no entries" in b["unmeasured"]["token"]
    cut = replace(m, parts=(replace(m.parts[0], partial_bits=1e6, grad_bits=2e6),))
    b = F.bound(cut, c, k)
    assert b["inputs"]["reduction"]["blocks"] == [8] and set(b["unmeasured"]) == {"fresh_products", "token", "decoding",
                                                                                 "per_gate", "fresh_cell"}
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
    """`complete` reads a block's `ops`, else the row's macs_log (te_core.tiles.of_log), labelled; from_json keeps
    them."""
    log = {"by": {"fwd/updated": 2048 * 64 * 192}, "ops": {"fwd updated mm [[2048, 64], [64, 192]]": [1, 2048 * 64 * 192]}}
    m = F.complete({}, {"macs_log": log, "output": {"e_bytes": 10}}, SHAPE)
    assert m.ops[0]["products"] == [[1, 2048, 64, 192]] and m.labels["ops"] == F.OPS
    assert F.complete({"ops": []}, {"macs_log": log, "output": {"e_bytes": 10}}, SHAPE).ops == []
    assert F.from_json(m.to_json()) == m


