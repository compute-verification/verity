"""te_core.formalism and te_core.policy: Fit and Slowdown as the formalism v2 states them (a frozen tail always
charged), the attack side beside them, the stubs, assumptions and upper bounds labelled, the grid."""
import math
from dataclasses import replace

import pytest

from te_core import formalism as F
from te_core import frontier, policy
from te_core.gates import per_mac

SHAPE = {"d_model": 512, "n_layer": 6, "vocab": 50304, "ctx": 1024, "parameters": 45_029_888, "tokens": 154_206_208,
         "forward_macs_per_token": 51_342_080}


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
                             "decoder_gates", "u", "u_in", "u1_bits", "delta_grad_bits"}
    assert all(m.labels[k].startswith("stub") for k in ("tokens", "affected_gates", "tau_bits"))
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
