"""training_slowdown.formalism: Fit and Slowdown as the formalism v2 states them (a frozen tail always charged), the
attack side beside them, the stubs, assumptions and upper bounds labelled, and the one bound's products term with its
sensitivities and what each published string rests on."""
import dataclasses
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import SHAPE, calculators_count, figure

from training_slowdown import VERSION as F_VERSION
from training_slowdown import calculator, policy, work
from training_slowdown import formalism as F
from training_slowdown.gates import per_mac

FIXTURES = Path(__file__).parent / "fixtures"


def measured(u1=1e8, sigma=0.8, delta=None, dgrad=None, decoder=0.0, tail=0.0, affected=0.5) -> F.Measured:
    """One part of 1e12 gates and 1e3 token blocks, σ of the gates in them and `affected` of those affected; no decoder
    (d = 0) unless given."""
    u, u_in = F.tail(u1, tail)
    part = F.Part(gates=1e12, tokens=1e3, token_gates=sigma * 1e12, affected_gates=affected * sigma * 1e12,
                  delta_grad_bits=dgrad)
    return F.Measured((part,), tau_bits=32, a_bits=8192, u=u, u_in=u_in,
                      delta_bits=delta if delta is not None else u1, decoder_gates=decoder, tail_share=tail)


def term_of(m: F.Measured, c: float, k: float) -> tuple[float | None, str | None, dict]:
    """The one bound's products term on `m` at (c, k), its reason where it has none, and what it read."""
    b = F.bound(m, c, k)
    return b["terms"]["products"], b["unmeasured"].get("products"), b["inputs"]["products"]


def token_form(m: F.Measured, c: float, k: float) -> dict:
    """Slowdown's token form on `m` at (c, k), a sensitivity of the one bound and never its slowdown: its terms (σ,
    x0, s, s*, ᾱ/α_max, and v2's s* as `s_star_a1` where the context is charged) and its value (`token_form`)."""
    b = F.bound(m, c, k)
    return b["inputs"]["token_form"] | {"token_form": b["sensitivities"]["token_form"]}


def test_even_update_token_form_is_the_corollary():
    """Evenly spread with u(1) >> k: x0 <= k/u(1), s* <= k^2/(4 a u(1)), so the token form is about 4 σ s a u(1) / k^2;
    it is a sensitivity, and no slowdown of the record's, which has no product rows."""
    m, c, k = measured(u1=1e9), 1e13, 1e5
    r = token_form(m, c, k)
    s = c / 0.8e9
    assert (r["sigma"], r["s"]) == pytest.approx((0.8, s))
    assert r["x0"] == pytest.approx(k / 1e9)
    assert r["s_star"] == pytest.approx(k * k / (4 * 1e9) / 8192)
    assert r["token_form"] == pytest.approx(4 * 0.8 * s * 8192 * 1e9 / k ** 2, rel=1e-3)
    got = F.at(m, c, k)
    assert not got["fit"] and got["rho"] is None and "slowdown" not in got


def test_a_frozen_tail_is_always_charged():
    """Frozen layers above changed ones with u(1) > k: u_in jumps at once, so x0 = 0 and the tail's work stays charged;
    s* <= t k / a, so the token form is about σ s a / (t k), below the evenly spread update's (the tail leaves a
    block more room)."""
    t, k, u1, c = 0.05, 1e5, 1e9, 1e13
    r = token_form(measured(u1=u1, tail=t), c, k)
    assert r["x0"] == 0 and r["s_star"] == pytest.approx(t * k / 8192)
    assert r["token_form"] == pytest.approx(0.8 * (c / 0.8e9) * 8192 / (t * k), rel=1e-6)
    assert token_form(measured(u1=u1), c, k)["token_form"] > r["token_form"]


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
    r = token_form(m, c, k)
    s_star = F.room(u, k) / 8192
    alpha_max = 0.8e12 / 1e3
    assert r["alpha_ratio"] == pytest.approx((1e12 / 2e3) / alpha_max)
    assert r["token_form"] == pytest.approx((1 - k / 1e9) * c * 1e12 / (alpha_max * 2e12 * s_star))
    assert r["token_form"] == pytest.approx(m.sigma * (1 - k / 1e9) * (c / m.w) / s_star * r["alpha_ratio"])
    same = F.Measured((a, a), 32, 8192, u, u, 1e9)
    assert token_form(same, c, k)["alpha_ratio"] == 1


def test_fit_is_three_kinds_of_block_and_gives_c_over_c_minus_d():
    """Training fits when compute (Δ + sτ), reduction (2Δ_grad) and update application (Δ + Δ_grad) each fit in k;
    ρ is then c/(c − d), 1 with no decoder (d = 0); an unmeasured decoder fits no block. There Δ ≤ k bounds u and u_in
    (`capped`), so x0's bound is 1 and the token form 0."""
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
        m = measured(u1=1e8, delta=1e3, dgrad=1e3, tail=tail)
        wide, form = F.at(m, c, k), token_form(m, c, k)
        assert wide["fit"] and wide["rho"] == 1.0 and form["x0"] == 1.0 and form["token_form"] == 0.0


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
    r = token_form(F.Measured((part,), 32, 8192, u, u, 20.0, decoder_gates=0.0), c, k)
    assert r["x0"] == 1.0 and r["s_star"] == pytest.approx((k - 20.0) / 8192)
    wide = token_form(F.Measured((part,), 32, 8192, u, u, 20.0, decoder_gates=c), c, k)
    assert wide["x0"] == pytest.approx(0.5 + 0.25 * 2 / 3)


def test_update_application_reads_delta_t_where_the_run_has_latents():
    """A straight-through run's update application reads its fp32 latents Δ_t, not its code Δ:
    Δ_t + Δ_grad ≤ k, while compute still reads Δ; a run with no Δ_t reads Δ there."""
    c, k = 1e13, 1e6
    code = measured(u1=1e5, dgrad=1e5)
    latent = dataclasses.replace(code, delta_t_bits=k)
    assert F.at(code, c, k)["fit"] and code.delta_t == code.delta_bits
    assert F.at(latent, c, k)["parts"][0]["fit"] == {"compute": True, "reduction": True, "update": False}
    need = k + 1e5
    assert F.at(latent, c, need)["fit"] and not F.at(latent, c, need * (1 - 1e-9))["parts"][0]["fit"]["update"]


def test_the_attack_side_is_reported_beside_the_bounds():
    """Best-known values from exhibited blocks: x0 1 where the compute block fits; s* from the frozen tail's block,
    t·min(⌊k/(a + τ)⌋, ⌊c/(t α_max)⌋), at most A1's bound; none without a tail."""
    c, k = 1e13, 1e5
    tail = F.at(measured(u1=1e9, tail=0.05), c, k)
    assert tail["x0_known"] == 0 and tail["s_star_known"] == pytest.approx(0.05 * math.floor(k / (8192 + 32)))
    assert tail["s_star_known"] <= token_form(measured(u1=1e9, tail=0.05), c, k)["s_star"]
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
    0.2 (5000 − 1600)/1000. u_in jumps, so x0 = 0, and the token form is σ s times the positions' mean of
    h/max(S0, S1(p)) + (1 − h)/S1(p)."""
    part = F.Part(gates=1e12, tokens=1e3, token_gates=0.8e12, affected_gates=0.4e12)
    m = F.Measured((part,), tau_bits=32, a_bits=1000, u=F.even(8000), u_in=((0, 0), (0, 8000), (1, 8000)),
                   delta_bits=8000)
    s1, s0 = [0.78125, 0.78125, 0.75, 0.5], 0.2 * 3400 / 1000
    assert F.context_rooms(m.u, 5000, 1000, 4) == pytest.approx(s1)
    assert F.room(m.u, 5000, 0.2) / 1000 == pytest.approx(s0)
    mean = sum(0.2 / max(s0, s) + 0.8 / s for s in s1) / 4
    r = token_form(charged(m, 0.2, positions=4, kv=2000), 1e12, 5000)
    assert r["x0"] == 0 and r["s_star_a1"] == pytest.approx(0.78125) and r["s_star"] == pytest.approx(1 / mean)
    assert r["token_form"] == pytest.approx(0.8 * 1250 * mean)
    v2 = token_form(m, 1e12, 5000)
    assert v2["token_form"] == pytest.approx(0.8 * 1250 / 0.78125) and v2["token_form"] < r["token_form"]
    assert "s_star_a1" not in v2
    with pytest.raises(ValueError, match="best span"):
        F.at(charged(m, 0.2, positions=4, kv=999), 1e12, 5000)


def test_a_narrow_cut_whose_best_block_holds_the_sequence_keeps_v2s_token_form():
    """a = 32 and a frozen tail: v2's best block holds k/a = 2500 tokens, more than L = 1024, so every position's
    S1(p) is s* and the charge changes nothing."""
    m = replace(measured(u1=1e9, tail=0.3), a_bits=32)
    v2, r = token_form(m, 1e13, 8e4), token_form(charged(m, 0.6), 1e13, 8e4)
    assert r["s_star"] == pytest.approx(v2["s_star"]) == r["s_star_a1"]
    assert r["token_form"] == pytest.approx(v2["token_form"])


def test_a_dense_cut_is_unbounded_where_k_holds_few_tokens():
    """a = d_model·16 at k = 80,000 bits: a block doing a token's charged work must hold every earlier token, so none
    serves a position past k/a − 1 ≈ 8.8, and the charged share 1 − h left there makes the token form infinite (s* 0).
    At k = 10^8 the block holds the sequence and the token form is v2's. With no affected K/V (h = 1) nothing is
    charged."""
    m = measured(u1=1e9, tail=0.3)
    r = token_form(charged(m, 0.6), 1e13, 8e4)
    assert r["s_star"] == 0 and r["token_form"] == math.inf and r["s_star_a1"] == pytest.approx(0.3 * 8e4 / 8192)
    for h, k in ((0.6, 1e8), (1.0, 8e4)):
        assert token_form(charged(m, h), 1e13, k)["token_form"] == pytest.approx(token_form(m, 1e13, k)["token_form"])
    assert token_form(charged(replace(m, u_in=F.even(1e3)), 0.6), 1e13, 8e4)["token_form"] == 0


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
    assert m.context == emb and m.labels["context"] == F.CONTEXT


def test_fit_reads_delta_and_the_token_form_reads_u1_as_separate_fields():
    """A total update (Fit's Δ) larger than the per-token update (the token form's u(1)), as for an update each
    token's path reads only part of: the compute block fails on Δ while the token form reads u(1). With a small Δ the
    compute block fits and Δ bounds u too. The record says which value each came from."""
    row = {"output": {"e_bytes": 10**9, "rebuild_macs": 0}, "macs_log": {"by": {}, "ops": {}}}
    part = {"gates": 1e12, "tokens": 1e3, "token_gates": 1e12, "affected_gates": 5e11, "steps": 1,
            "delta_grad_bits": None}
    f = {"parts": [part], "u1_bits": 1e9, "labels": {"u1_bits": "upper bound (witness): a test"}}
    m = F.complete(f, row, SHAPE)
    assert (m.delta_bits, m.u1_bits, m.u, m.u_in, m.tail_share) == (8e9, 1e9, F.even(1e9), F.even(1e9), 0.0)
    assert m.labels["u1_bits"] == "upper bound (witness): a test" and m.labels["u"] == F.STUBS["u"]
    assert m.labels["u"].startswith("assumption (A1)") and m.labels["u_in"].startswith("assumption (A1)")
    k = 1e6
    r, form = F.at(m, 1e13, k), token_form(m, 1e13, k)
    assert not r["parts"][0]["fit"]["compute"] and form["x0"] == pytest.approx(k / 1e9)
    assert form["token_form"] == pytest.approx(4 * 1e4 * 512 * 16 * 1e9 / k ** 2, rel=2e-3)
    less = F.complete(f, row | {"output": {"e_bytes": 10, "format": "full"}}, SHAPE)
    small, form = F.at(less, 1e13, k), token_form(less, 1e13, k)
    assert small["parts"][0]["fit"]["compute"] and not small["fit"]
    assert form["x0"] == 1.0 and form["token_form"] == 0.0
    with pytest.raises(ValueError):
        F.complete({"parts": [part], "u": [[0, 0], [1, 5.0]], "u1_bits": 6.0}, row, SHAPE)


def test_a_block_s_tail_share_gives_the_tail_profiles_labelled_as_assumptions():
    row = {"output": {"e_bytes": 10}, "macs_log": {"by": {}, "ops": {}}}
    part = {"gates": 1e12, "tokens": 1e3, "token_gates": 1e12, "affected_gates": 5e11}
    m = F.complete({"parts": [part], "u1_bits": 1e9, "tail_share": 0.25}, row, SHAPE)
    assert m.tail_share == 0.25 and (m.u, m.u_in) == F.tail(1e9, 0.25)
    assert m.labels["u"].startswith("assumption (A1)") and m.labels["u_in"].startswith("assumption (A1)")
    with pytest.raises(ValueError):
        F.complete({"parts": [part], "u1_bits": 1e9, "tail_share": 1.0}, row, SHAPE)


def test_a_version_4_block_s_dep_gates_are_its_affected_gates():
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
    """d is the program's notes.decoder gates for the checkpoint's E, else the output's rebuild MACs at
    F32Fma_v4's 12,637 gates a multiply-add, 0 only where E stores θ' directly; with none,
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
    ({"data_dependent": None}, 10, "read-backs unchecked: traced before training's slowdown code 4.2 (no data_dependent)"),
])
def test_a_trace_the_capture_cannot_vouch_for_is_refused_never_scored(trace, e_bytes, reason):
    """A clip on a read-back estimate leaves the trace with no update (affected share 0, u(1) 0), which would score
    ρ = 1 whatever the update. A data-dependent capture, a Python number folded over steps, a
    nonempty E with no update in its trace and a trace without the read-back lists are each refused with the
    reason."""
    row = traced({k: v for k, v in trace.items() if v is not None}, e_bytes)
    if trace.get("data_dependent", []) is None:
        del row["formalism"]["trace"]["data_dependent"]
    assert F.refusal(row).startswith(reason)
    with pytest.raises(F.Refused, match="^" + reason.split(" (")[0]):
        F.measure(row, SHAPE)


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


def test_a_record_with_version_4_cuts_reads_them_as_delta_grad():
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


def one_part(delta=None, decoder=0.0) -> F.Measured:
    """One part of 1e12 gates and 1e3 tokens of 1e9 gates each, half of its gates affected, its update profiles even
    up to u(1) = 2,000 bits; no product rows unless a test gives them."""
    part = F.Part(gates=1e12, tokens=1e3, token_gates=1e12, affected_gates=5e11)
    return F.Measured((part,), tau_bits=16, a_bits=16, u=F.even(2000.0), u_in=F.even(2000.0), delta_bits=delta,
                      decoder_gates=decoder)


def test_the_one_bound_is_its_products_kind_resting_on_what_it_names():
    """The one bound's max is over its one kind, the products term, resting on the theorem of record and its premises:
    on a record with no product rows it is unmeasured and the bound binds nothing; with a model's stated products it
    binds the products kind. A sensitivity, or a kind the bound doesn't have, is refused in its max."""
    c, k = 1e11, policy.CONSOLE[1]
    b = F.bound(one_part(), c, k)
    assert b["kinds"] == list(F.KINDS) == ["products"] and b["label"] == f"{F.THE_CODE} {F_VERSION}"
    assert b["terms"]["products"] is None and b["unmeasured"]["products"] == work.PRODUCTS_UNMEASURED
    assert (b["max"], b["binding"]) == (None, None)
    assert set(b["rests_on"]) == set(F.KINDS) and b["rests_on"]["products"].startswith("theorem of record")
    stated = replace(one_part(), products=STATED)
    one = F.bound(stated, c, k)
    assert one["binding"] == "products" and one["max"] == one["terms"]["products"] > 1
    r = F.at(stated, c, k)
    assert r["bound"] == one and r["rho"] == one["max"] == one["entry"]["slowdown"]
    with pytest.raises(ValueError, match="some of"):
        F.bound(stated, c, k, kinds=("tokens",))
    with pytest.raises(ValueError, match="sensitivities, never in the bound's max"):
        F.bound(stated, c, k, kinds=("decoding_form",))


#: one bf16 product of two fresh operands over K = 512, half of one_part's 1e12 gates: its MACs and output entries
SQUARE = {"macs": 0.5e12 / (23682 / 16), "outputs": 0.5e12 / (23682 / 16) / 512}


def stated(k: float) -> dict:
    """SQUARE as the products term's M at k, in `he_of_rows`' form: one input gradient, an activation times a trained
    weight at 16 bits each with 16-bit outputs, which doesn't say whether it is the head's."""
    group = {"beta": [16, 16], "width": [16, 16], "weight": [False, True], "b_y": 16, "forward": False, "k_dim": 512,
             "dw": False, "head": None, "macs": SQUARE["macs"], "wires": SQUARE["outputs"] * 16}
    return {"k": float(k), "groups": [group], "r_step": None, "candidates_macs": SQUARE["macs"], "self_macs": 0.0,
            "gathered_macs": 0.0, "left_out_macs": 0.0, "steps": None, "step_tokens": None, "trained_weights": None}


#: SQUARE as a run's products at the console's k
STATED = {"k": policy.CONSOLE[1], "he": stated(policy.CONSOLE[1])}


def test_the_products_term_is_the_theorem_of_records_on_m():
    """The products kind's term is the theorem of record's ρ on M (the entry point's, on the products' `he`) and the record's
    |C|, linear in c at a fixed k, and the published row rests on (H) at the β it took and (E) at its b_y, which
    `products_rests_on` carries wherever the term is measured. It is unmeasured at a k the products weren't
    classified at, on products that come without M (no rows), on rows that don't flag a gather's and where an
    elementwise op fans out into an operand. The published row names the code that wrote the products' trace where
    the trace records it, and None where it doesn't."""
    c, k = 1e11, policy.CONSOLE[1]
    stated = replace(one_part(), products=STATED)
    b = F.bound(stated, c, k)
    fig = figure(STATED["he"], c, 1e12)
    assert (fig["subset"], fig["beta_w"], fig["beta_x"], fig["b_y"]) == ("M", 16, 15.9, 16)
    assert not fig["below_width"] and fig["macs_share"] == 1
    assert fig["F"] == pytest.approx(2 / math.sqrt(3), rel=1e-15) and fig["N"] == SQUARE["macs"]
    assert fig["O"] == SQUARE["outputs"] * 16
    assert b["terms"]["products"] == fig["rho"] > 1 and "products" not in b["unmeasured"]
    assert F.bound(stated, 2 * c, k)["terms"]["products"] == pytest.approx(2 * fig["rho"], rel=1e-12)
    one = F.published(stated)
    assert one["products_rests_on"] == F.he_rests_on(fig)
    assert one["products_rests_on"].startswith(f"theorem of record on M at {F.SHARED_LABEL}, given (H)")
    assert all(x in one["products_rests_on"] for x in ("(H)", "β_w = 16 and β_x = 15.9 bits a word", calculator.BETA_K_LABEL,
                                                       "the casts of one value count as one word", "(E)",
                                                       "b_y = 16 bits an output", "a single counterexample refutes"))
    assert F.CODE_LABEL not in one["products_rests_on"]
    assert one["rests_on"] == (one["products_rests_on"] if one["binding_kind"] == "products" else
                               F.RESTS_ON_CELL[one["binding_kind"]])
    assert one["capture_code"] is None
    assert F.published(replace(stated, products=STATED | {"capture_code": F_VERSION}))["capture_code"] == F_VERSION
    elsewhere = F.bound(stated, c, 80_000)
    assert elsewhere["terms"]["products"] is None and "classified at k = 65536" in elsewhere["unmeasured"]["products"]
    assert F.bound(replace(stated, products={"k": k}), c, k)["unmeasured"]["products"] == work.HE_NO_ROWS
    assert F.bound(replace(stated, products=STATED | {"he": None}), c, k)["unmeasured"]["products"] == work.HE_RETRACE
    fans = F.bound(replace(stated, products=STATED | {"p_match": [{"op": "expand", "fan_out": 16.0, "count": 1}]}), c, k)
    assert fans["terms"]["products"] is None and "breaks the premise" in fans["unmeasured"]["products"]
    assert all(x in F.RESTS_ON["products"] for x in ("(H)", "(E)", "F = 2/√3", "β_eff", "K* = ½√(k/β)",
                                                    calculator.BETA_K_LABEL, F.CODE_LABEL, "casts of one value"))
    assert all(x in F.RESTS_ON_CELL["products"] for x in ("(H)", "(E)", "a single counterexample refutes"))
    assert F.CELL_SENSITIVITIES == ("reduction", "update")


def he_row(phase: str, td: list, m: int, k: int, n: int, count: int = 1, batch: int = 1) -> dict:
    """A product row as training_slowdown.capture.products writes one: both operands affected at bf16, paid, none gathered."""
    gates = next(g for g, b in work.HE_BITS.items() if b == (16, 16))
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
    """4.13.1's rules on M (`work.he_of_rows`, the entry point's choice of the work): the optimizer's products and attention's leave M; a weight is an
    operand no token indexes, a backward product contracting over tokens a weight gradient, and the head's are the
    backward products over its vocabulary width (the widest forward output); the trained weights are the weights a
    step's forward reads, each matrix once a token. Activations are charged β(k), 15.9 up to 145,568, between two
    red-team points the next one's (from 4.13.5), none past 8e8, weights their width or less the record's code bits a trained weight,
    a weight's β the β_w a product reports beside its activation's β_x. The term is the best of all of M, M less its contractions below K_thr (here the 8-long product's wide outputs,
    whose leaving lifts the term) and the weight gradients alone at F = 1; over a run's steps it loses the ⌊k/(τT)⌋
    after the first update, and from 8e7 the head's products leave M. Model products that sum the head's gradients
    with the layers' are unmeasured from 8e7, and every M past 8e8."""
    c, k, C, T = 1e11, 65_536.0, 1e13, 1024
    beta = calculator._min_cut().beta
    assert (beta(1.0), beta(145_568), beta(8e6), beta(8e8)) == (15.9, 15.9, 10.8, 0.37)
    assert beta(8e6 + 1) == beta(math.sqrt(8e6 * 8e7)) == beta(8e7) == 2.7 and beta(8.1e8) is None
    he = work.he_of_rows(HE_STEP, k, 10.0, T)
    opt, att = HE_STEP[7], HE_STEP[8]
    assert he["left_out_macs"] == 10 * (opt["count"] * opt["macs"] + att["count"] * att["macs"])
    assert he["trained_weights"] == 2 * 1024 * 1024 + 1024 * 4096 + 8 * 4096
    assert sorted((g["k_dim"], g["dw"], g["head"], tuple(g["weight"])) for g in he["groups"]) == [
        (8, False, False, (False, True)), (1024, False, False, (False, True)), (1024, False, False, (False, True)),
        (1024, True, False, (False, False)), (1024, True, True, (False, False)), (4096, False, True, (False, True))]
    fig = figure(he, c, C)
    assert fig["subset"] == calculator.HE_SUBSETS[1] and fig["rho"] == max(fig["subsets"].values()) > fig["subsets"]["M"]
    assert 8 < fig["K_thr"] < 1024 and fig["beta_k"] == 15.9 and fig["F"] == pytest.approx(2 / math.sqrt(3))
    dw = [g for g in he["groups"] if g["dw"]]
    N, O = sum(g["macs"] for g in dw), sum(g["wires"] for g in dw)
    alone = {"N": N, "O": O, "beta_w": 15.9, "beta_x": 15.9, "b_y": 16, "F": 1.0, "C_gates": C}
    assert fig["subsets"][calculator.HE_SUBSETS[2]] == calculators_count(c, k, alone)
    first = figure(he, c, C, tau=16)
    assert first["first_steps"] == 4 and first["N"] == pytest.approx(0.6 * fig["N"], rel=1e-12)
    assert fig["first_steps"] is None and figure(he, c, C, tau=1e3)["first_steps"] == 0
    coded = figure(he, c, C, beta_code=2.0)
    assert coded["rho"] < fig["rho"] and coded["beta_code"] == 2.0
    one = he | {"groups": [g for g in he["groups"] if g["k_dim"] == 1024 and not g["dw"] and not g["head"]][:1]}
    weighed = figure(one, c, C, beta_code=8.0)
    assert (weighed["beta_w"], weighed["beta_x"], weighed["code_bound"]) == (8.0, 15.9, True)
    m = replace(one_part(delta=2.0 * he["trained_weights"]), products={"k": k, "he": he})
    term, why, got = term_of(m, c, k)
    assert why is None and got["beta_code"] == 2.0 and term == figure(he, c, m.gates, m.tau_bits, 2.0)["rho"]
    assert (F.CODE_LABEL in F.he_rests_on(got)) == got["code_bound"]
    far = work.he_of_rows(HE_STEP, 8e7)
    assert figure(far, c, C) == figure(far | {"groups": [g for g in far["groups"] if not g["head"]]}, c, C)
    assert figure(he, c, C) != figure(he | {"groups": [g for g in he["groups"] if not g["head"]]}, c, C)
    for at, why in ((8e7, work.HE_HEAD), (9e8, calculator.he_past())):
        model = replace(one_part(), products={"k": at, "he": stated(at)})
        assert term_of(model, c, at)[:2] == (None, why), at
    with pytest.raises(ValueError, match="past k"):
        figure(work.he_of_rows(HE_STEP, 9e8), c, C)


def test_all_of_m_binds_where_the_count_puts_it_above_the_weight_gradients_and_the_words_say_so():
    """The subset that binds is the one whose count is largest, and the row's words name it with its F. Here M's
    forward and input-gradient products give many output wires a MAC, where the statement's second conjunct loses
    most of the count: by the count all of M binds, at F = 2/√3 and the code's β_w, and by the second conjunct the
    weight gradients alone would, so a term read from the second conjunct fails here."""
    k, c, C = policy.CONSOLE[1], 1e14, 1e18

    def group(macs, wires, weight, forward, dw, k_dim):
        return {"beta": [16.0, 16.0], "width": [16.0, 16.0], "weight": weight, "b_y": 16.0, "forward": forward,
                "k_dim": k_dim, "dw": dw, "head": False, "macs": macs, "wires": wires}

    groups = [group(1.5e15, 7.5e11, [False, False], False, True, 4096.0),
              group(1.6e15, 3.3e13, [False, True], True, False, 512.0),
              group(1.6e15, 3.3e13, [False, True], False, False, 512.0)]
    he = {"k": k, "groups": groups, "r_step": None, "candidates_macs": sum(g["macs"] for g in groups),
          "self_macs": 0.0, "gathered_macs": 0.0, "left_out_macs": 0.0, "steps": None, "step_tokens": None,
          "trained_weights": None}
    fig = figure(he, c, C, beta_code=8.0)
    whole = {"N": 4.7e15, "O": 7.5e11 + 2 * 3.3e13, "beta_w": 8.0, "beta_x": 15.9, "b_y": 16.0,
             "F": calculator._min_cut().share(3), "C_gates": C}
    alone = {"N": 1.5e15, "O": 7.5e11, "beta_w": 15.9, "beta_x": 15.9, "b_y": 16.0, "F": 1.0, "C_gates": C}
    count = {name: calculators_count(c, k, w) for name, w in (("M", whole), ("dw", alone))}
    assert (fig["subset"], fig["F"], fig["beta_w"], fig["N"], fig["O"]) == (
        "M", calculator._min_cut().share(3), 8.0, whole["N"], whole["O"])
    assert fig["rho"] == fig["subsets"]["M"] == count["M"] > count["dw"] == fig["subsets"][calculator.HE_SUBSETS[2]]
    words = F.he_rests_on(fig)
    assert words.startswith(f"theorem of record on M at {F.SHARED_LABEL}, given (H): operands are hard at β_w = 8 "
                            f"({F.CODE_LABEL}) and β_x = 15.9 bits a word")
    assert calculator.HE_SUBSETS[2] not in words
    second = calculator._min_cut().slowdown_at_least
    assert second(whole["N"], whole["O"], C, c, k, 8.0 * 15.9, 16.0, whole["F"]) < second(
        alone["N"], alone["O"], C, c, k, 15.9 * 15.9, 16.0, 1.0) < count["dw"]


def test_the_adapter_rule_needs_a_frozen_weight_the_update_reaches_and_a_frozen_head_keeps_its_width():
    """A distilling teacher's weights are frozen but read only tokens, so the
    student's head (its d_model widened to the vocabulary) is no adapter: r_step is None and every activation keeps its
    width, as without the teacher. A LoRA's frozen layers read what its adapters wrote, so B·(Ax) sets r_step to its
    rank and an activation wider than it is charged β_eff. With its head frozen, the vocabulary is still the head's
    width, so the adapters' backward products over d_model aren't the head's; a trained head's are."""
    k, d, V, r = 65_536.0, 512, 32_000, 8

    def frozen(row, *i):
        return row | {"affected": [j not in i for j in (0, 1)]}

    student = [he_row("forward", ["m"], 1024, 256, 704), he_row("forward", ["m"], 1024, 704, 256),
               he_row("forward", ["m"], 1024, 256, V)]
    alone, distilled = (work.he_of_rows(rows, k) for rows in (student, [frozen(he_row("forward", ["m"], 1024, d, V), 0, 1),
                                                                     *student]))
    assert distilled["r_step"] is alone["r_step"] is None
    assert distilled["groups"] == alone["groups"] and all(b == 16 for g in alone["groups"] for b in g["beta"])
    lora = [frozen(he_row("forward", ["m"], 1024, d, d), 1), he_row("forward", ["m"], 1024, d, r),
            he_row("forward", ["m"], 1024, r, d), frozen(he_row("forward", ["m"], 1024, d, V), 1),
            he_row("backward", ["m"], 1024, d, r), he_row("backward", ["k"], d, 1024, r)]
    he = work.he_of_rows(lora, k)
    assert he["r_step"] == r and not any(g["head"] for g in he["groups"])
    dx = [g["beta"] for g in he["groups"] if not g["forward"] and not g["dw"]]
    assert dx == [[round(work.beta_eff(16, r, k), 9), 16]]
    trained = work.he_of_rows([*lora[:3], he_row("forward", ["m"], 1024, d, V), he_row("backward", ["m"], 1024, V, d)], k)
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
    fig = figure(he, c, C, beta_code=2.0)
    share = 2 / math.sqrt(3)

    def rho(M: list, bw: float, bx: float, f: float = share) -> float:
        work = {"N": sum(g["macs"] for g in M), "O": sum(g["wires"] for g in M), "beta_w": bw, "beta_x": bx,
                "b_y": 16, "F": f, "C_gates": C}
        return calculators_count(c, k, work)

    assert (fig["subset"], fig["beta_w"], fig["beta_x"], fig["code_bound"]) == ("M", 2.0, 8.0, True)
    assert fig["F"] == pytest.approx(share, rel=1e-15) and fig["macs_share"] == 1
    assert fig["rho"] == pytest.approx(rho([fwd, dx, dw], 2.0, 8.0), rel=1e-12)
    assert rho([fwd, dx, dw], 2.0, 8.0) < rho([fwd, dx, dw], 2.0, 15.9)
    assert fig["rho"] > max(rho([fwd, dw], 2.0, 15.9), rho([dw], 15.9, 8.0), rho([dw], 15.9, 8.0, 1.0)) > 0


def test_f_is_1_where_the_binding_m_holds_forward_products_alone():
    """F = 2/√3 (the calculator's `min_cut.share`) where the M_P a term takes holds a backward product, since a layer's forward,
    input-gradient and weight-gradient products share three arrays, and 1 where it holds forward products alone: a
    forward product sixteen times the gradients' MACs binds at its own pair, alone, at F = 1, and gradients sixteen
    times its bind at the least pair, with all three, at 2/√3."""
    c, k, C, m, K = 1e11, 65_536.0, 1e13, 256, 65_536

    def group(beta: list, weight: list, forward: bool, dw: bool, scale: float) -> dict:
        return {"beta": beta, "width": [16, 16], "weight": weight, "b_y": 16, "forward": forward, "k_dim": K,
                "dw": dw, "head": False, "macs": scale * m * m * K, "wires": scale * m * m * 16}

    def step(fwd: float, back: float) -> dict:
        return {"k": k, "groups": [group([16.0, 16.0], [False, True], True, False, fwd),
                                   group([8.0, 16.0], [False, True], False, False, back),
                                   group([8.0, 16.0], [False, False], False, True, back)],
                "r_step": None, "candidates_macs": (fwd + 2 * back) * m * m * K, "self_macs": 0.0,
                "gathered_macs": 0.0, "left_out_macs": 0.0, "steps": None, "step_tokens": None,
                "trained_weights": None}

    alone = figure(step(16, 1), c, C)
    assert (alone["subset"], alone["F"], alone["beta_w"], alone["beta_x"]) == ("M", 1.0, 16.0, 15.9)
    assert alone["macs_share"] == pytest.approx(16 / 18, rel=1e-15)
    shared = figure(step(1, 16), c, C)
    assert shared["subset"] == "M" and shared["F"] == calculator._min_cut().share(3) and shared["macs_share"] == 1


#: a code name in a published string: a dotted path (a module's, or its function's), an identifier with an underscore
#: (training_slowdown, he_of_rows, HE_NO_ROWS, k_dim) other than a one-letter math subscript (b_y), or a command-line
#: flag
CODE_NAME = __import__("re").compile(
    r"\b[A-Za-z]\w*\.[A-Za-z_]\w*|\b(?![A-Za-z]_[A-Za-z0-9]\b)[A-Za-z][A-Za-z0-9]*_\w+\b|--[a-z]")


def test_no_published_string_names_the_code():
    """Every string a panel publishes is plain words, since a reader's page prints them: what the kind and each
    published ρ rest on, every reason the term is unmeasured (no product rows, no M, a retrace, another k, a broken
    P-match, a trace of another program), the sensitivities' labels, the record's labels and the label. The code is
    "training's slowdown code", and a premise is named in words beside its name."""
    c, k = 1e11, policy.CONSOLE[1]
    static = [F.RESTS_ON, F.RESTS_ON_CELL, F.SENSITIVITIES, F.STUBS, F.TRACED, F.DECODER, F.V23,
              work.PRODUCTS_UNMEASURED, work.HE_RETRACE, work.HE_NO_ROWS, calculator.he_past(), work.HE_HEAD, work.OTHER_PROGRAM, F.STAND_IN,
              F.PRODUCTS, F.TRACED_PRODUCTS, F.CONTEXT, F.TAIL]
    strings = [s for x in static for s in (x.values() if isinstance(x, dict) else [x])]
    other = {"shapes": 2, "unmatched": 1, "ran": 3, "largest": [[4, 512, 8]]}
    records = [one_part(), one_part(delta=500.0, decoder=5e10), replace(one_part(), products={"k": k})] + [
        replace(one_part(), products=STATED | x) for x in (
            {}, {"he": None}, {"p_match": [{"op": "expand", "fan_out": 16.0, "count": 1}]}, {"other_program": other})]
    reasons = set()
    for m in records:
        for at in (k, 2800.0, 80_000.0):
            b = F.bound(m, c, at)
            strings += [b["label"], *b["rests_on"].values(), *b["unmeasured"].values(), *b["labels"].values()]
            reasons |= set(b["unmeasured"].values())
            one = F.published(m, c, at)
            strings += [one["label"], one["rests_on"], one["why"] or "", one["products_rests_on"] or ""]
    alone = F.he_rests_on({"subset": calculator.HE_SUBSETS[2], "F": 1.0, "beta_w": 1.6, "beta_x": 3.2, "b_y": 16.0,
                           "below_width": True, "code_bound": True})
    assert alone.startswith("theorem of record on the weight gradients alone at F = 1, given (H)")
    strings += [alone, F.SHARED_LABEL, *calculator.HE_SUBSETS]
    assert {work.PRODUCTS_UNMEASURED, work.HE_RETRACE, work.HE_NO_ROWS} <= reasons
    assert all(any(x in r for r in reasons) for x in ("classified at k", "breaks the premise", "own program"))
    named = {s: CODE_NAME.findall(s) for s in strings if CODE_NAME.search(s)}
    assert not named, named


def test_published_where_fit_holds_is_fits_c_over_c_minus_d():
    c, k = 1e13, 1e6
    one = F.published(measured(u1=1e5, dgrad=1e5, decoder=1e12), c, k)
    assert (one["rho"], one["binding_kind"], one["partial"], one["why"]) == (
        pytest.approx(c / (c - 1e12)), "fit", False, None)
    assert one["rests_on"] == F.RESTS_ON_CELL["fit"]


def test_published_above_1_is_the_products_term():
    """With a model's stated products the products term binds where it exceeds 1, and the row rests on it at the β it
    took; every kind is measured, so it isn't partial."""
    full = F.published(replace(one_part(), products=STATED), 1e11, policy.CONSOLE[1])
    assert full["binding_kind"] == "products" and full["rho"] == full["terms"]["products"] > 1
    assert not full["partial"] and full["rests_on"] == full["products_rests_on"]
    assert full["rests_on"].startswith(f"theorem of record on M at {F.SHARED_LABEL}, given (H): operands are hard at "
                                       "β_w = 16 and β_x = 15.9 bits a word")


def test_published_with_the_term_unmeasured_is_null_never_1():
    """With no product rows the one kind is unmeasured: ρ is null, never the floor's 1, which would read as no
    slowdown, with `why` saying no kind is measured."""
    one = F.published(one_part(), 1e11, 3600)
    assert (one["rho"], one["binding_kind"], one["rests_on"], one["partial"], one["why"]) == (
        None, "unmeasured", "unmeasured", False, "no kind is measured")
    assert one["terms"]["products"] is None


def test_published_with_every_kind_measured_and_none_above_1_is_1_binding_none():
    """The stated products' term at c = 1e8 is at most 1: ρ is 1, binding none, resting on no kind exceeding 1; the
    floor sets it, not the term."""
    one = F.published(replace(one_part(), products=STATED), 1e8, policy.CONSOLE[1])
    assert all(t is not None and t <= 1 for t in one["terms"].values())
    assert (one["rho"], one["binding_kind"], one["partial"], one["why"]) == (1.0, "none", False, None)
    assert one["rests_on"] == F.RESTS_ON_CELL["none"] == "no kind exceeds 1"


def test_a_traces_products_at_each_k_of_the_dial_give_the_term_there():
    """A trace keeps its product rows at each k of the console's dial (training_slowdown.capture.products' `at`), so
    the products term is measured at each of them on that k's M, linear in c there, and off the dial it is unmeasured
    with the reason, as at every k but their own for products that keep no rows on the dial. On the 45M dense step
    the term falls with k (220,665 at the console's 65,536 bits, 160,416 at 80,000, 60,188 at the served block's
    145,568 and 2,815 at 800,000 on M, 39 at 8e6 on M without its short contractions), and from 8e7 bits up β(k) is so
    low that the count allows fewer blocks than |C|/c, so the term is below 1, which ρ's floor reads as 1."""
    from training_slowdown.capture import products

    rows = json.loads((FIXTURES / "dense_45m.products.json").read_text())
    got = work.products_of_trace(rows)
    C = rows["gates"]
    m = replace(one_part(), parts=(F.Part(gates=C, tokens=512, token_gates=C, affected_gates=C / 2),), products=got)
    assert sorted(got["at"], key=int) == [str(k) for k in products.DIAL_K if k != policy.CONSOLE[1]]
    assert all(b["k"] == float(x) and "at" not in b for x, b in got["at"].items())
    c, terms = policy.CONSOLE[0], []
    for k in products.DIAL_K:
        term, why, inputs = term_of(m, c, k)
        assert why is None and inputs["k_classified"] == k
        assert term_of(m, 1e12, k)[0] == pytest.approx(term * 1e12 / c, rel=1e-12, abs=1e-12)
        terms.append(round(term))
    assert terms == [220_665, 160_416, 60_188, 2_815, 39, 0, 0]
    term, why, _ = term_of(m, c, 100_000)
    assert term is None and "keep none at it" in why and "each k of the console's dial" in why
    own = replace(m, products={x: v for x, v in got.items() if x != "at"})
    assert term_of(own, c, policy.CONSOLE[1])[0] == term_of(m, c, policy.CONSOLE[1])[0]
    assert term_of(own, c, 80_000)[:2] == (None, why)


@pytest.mark.parametrize("name", ("dense_blocks", "lora_blocks", "wildcard_adapter"))
def test_a_fixtures_trace_keeps_its_product_rows_and_their_products_term(name):
    """Each fixture's product rows (the `.products.json` beside its circuit, training_slowdown.capture.products of the
    step the circuit traces): the capture's version, the console's k, the circuit's tokens and |C|, P-match holding,
    and by kind every product the op log places. On the tiny start no product is fresh: every counted candidate has an
    operand read through a repeating view or a broadcast below its width (each view read at its own value's rate), so
    the products term is measured and far below 1, where the bound says nothing: M's operands are read far below
    their widths, and at every β M can take the count allows fewer blocks than |C|/c."""
    from training_slowdown.capture import products
    from training_slowdown.capture.circuit import StepCircuit, measure

    block = json.loads((FIXTURES / f"{name}.products.json").read_text())
    f = measure(StepCircuit.load(FIXTURES / f"{name}.npz"))
    assert (block["capture"], block["k"]) == (products.CAPTURE, policy.CONSOLE[1])
    assert (block["tokens"], block["gates"]) == (f["tokens"], f["gates"]) and block["p_match"] == []
    got = work.products_of_trace(block)
    ops = json.loads((FIXTURES / f"{name}.ops.json").read_text())
    assert sum(got["kinds_macs"].values()) == sum(op["macs"] for op in ops if op["products"])
    assert got["kinds_macs"]["fresh"] == 0 and any(any(r["gathered"]) for r in block["rows"])
    b = F.bound(replace(one_part(), products=got), 1e11, policy.CONSOLE[1])
    he = b["inputs"]["products"]
    assert 0 < b["terms"]["products"] < 1 and "products" not in b["unmeasured"] and he["below_width"]
    assert he["beta"] < 4 and he["macs_share"] < 1


def test_a_records_trace_rows_are_its_traced_steps_products():
    """`of_record` takes the record's trace's product rows (`formalism.trace.products`) as `Measured.products` over its
    traced steps, the tokens of the steps whose structure has an affected share over the traced step's (before the
    token factor scales a part's tokens), each kind's MACs its rows' times their counts; a step before the first
    update adds none, a record that has its own products keeps them, and one whose trace has no rows has none. Where
    the traced steps differ in shape (another structure, or other tokens), the dominant step's products stand in for
    them, scaled by tokens, and the products and their label say so (`STAND_IN`)."""
    row = {"count": 2, "phase": "forward", "op": "mm", "dims": {"batch": 1, "m": 4, "k": 8, "n": 8},
           "token_dims": ["m"], "macs": 256, "outputs": 32, "affected": [True, True], "widths": [16, 16],
           "rates": [16.0, 16.0], "paid": [True, True], "self": False, "cell_gates": 23682.0}
    theta = row | {"count": 1, "affected": [True, False], "rates": [16.0, 0.0], "paid": [True, False]}
    block = {"capture": work.PRODUCTS_FROM, "k": 65536.0, "tokens": 8, "gates": None, "rows": [row, theta]}
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
    assert (p["k"], p["capture"]) == (65536.0, work.PRODUCTS_FROM)
    assert p["kinds_macs"]["fresh"] == 2 * 2 * 256 and p["kinds_macs"]["theta"] == 2 * 256
    assert F.of_record(record, through=1)["products"]["kinds_macs"]["fresh"] == 2 * 256
    m = F.complete(f, {"output": {"e_bytes": 10}}, SHAPE)
    assert m.products == p and m.labels["products"] == F.TRACED_PRODUCTS and "stand_in" not in p
    own = {"k": 1.0, "he": None}
    assert F.of_record(record | {"formalism": {"trace": {"products": block}, "products": own}})["products"] == own
    assert "products" not in F.of_record(record | {"formalism": {"trace": {}}})
    grown = record | {"steps": [*record["steps"], {"step": 3, "tokens": 4, "struct": "c+traced4",
                                                   "by": {"fwd/theta": 40}}],
                      "structures": record["structures"] | {"c+traced4": record["structures"]["a"]}}
    g = F.of_record(grown)
    assert g["products"]["kinds_macs"]["fresh"] == 2.5 * 2 * 256
    assert g["products"]["stand_in"] == {"structures": ["a", "c"], "step_tokens": [4, 8], "traced_tokens": 8,
                                         "why": F.STAND_IN}
    assert g["labels"]["products"] == f"{F.TRACED_PRODUCTS}; {F.STAND_IN}"
    b = F.bound(F.complete(g, {"output": {"e_bytes": 10}}, SHAPE), 1e11, 65536.0)
    assert b["inputs"]["products"]["stand_in"] == g["products"]["stand_in"]


def test_a_record_an_older_version_ingested_reads_its_traces_rows_as_this_one_does():
    """A record keeps the products its ingest took from its trace's rows, and an older ingest took others: at 4.8 each
    kind's MACs alone, at 4.12.1 rows at the console's k only, though the trace keeps rows at each k of the dial
    (`at`). `complete` reads them as this code's ingest of the same rows over the same traced steps
    (`traced_products`: their kinds' MACs over the rows' own, 2.5 here with a stand-in), so rows before `SYSTEM_FROM`
    need a retrace (`HE_RETRACE`), products stored without rows have no M (`HE_NO_ROWS`), rows with `gathered` are
    measured at each k their trace keeps rows at (0 at the console's k: the run's 2.5 steps after the first are among
    the ⌊k/(τT)⌋ = 512 it leaves out), and products a record has of its own stand."""
    row = {"count": 2, "phase": "forward", "op": "mm", "dims": {"batch": 1, "m": 4, "k": 512, "n": 8},
           "token_dims": ["m"], "macs": 16384, "outputs": 32, "affected": [True, True], "widths": [16, 16],
           "rates": [16.0, 16.0], "paid": [True, False], "self": False, "cell_gates": 23682.0}
    theta = row | {"count": 1, "affected": [True, False], "rates": [16.0, 0.0], "paid": [True, False]}
    block = {"capture": work.PRODUCTS_FROM, "k": 65536.0, "tokens": 8, "gates": None, "rows": [row, theta]}
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

    def term(f, k=k):
        b = F.bound(F.complete(f, out, SHAPE), c, k)
        return b["terms"]["products"], b["unmeasured"].get("products")

    now = F.of_record(record(block))
    assert now["products"]["kinds_macs"]["fresh"] == 2.5 * 2 * 16384 and "stand_in" in now["products"]
    assert term(now) == (None, work.HE_RETRACE)
    at_48 = now | {"products": {"k": 65536.0, "capture": work.PRODUCTS_FROM,
                                "kinds_macs": {"expansion": 2.5 * 2 * 16384, "theta": 2.5 * 16384},
                                "stand_in": now["products"]["stand_in"]}}
    assert F.complete(at_48 | {"trace": {}}, out, SHAPE).products == at_48["products"]
    assert term(at_48 | {"trace": {}}) == (None, work.HE_NO_ROWS)
    assert F.complete(at_48, out, SHAPE).products == now["products"] and term(at_48) == (None, work.HE_RETRACE)
    captured = block | {"capture": work.SYSTEM_FROM, "p_match": [],
                        "rows": [r | {"gathered": [False, False]} for r in block["rows"]]}
    now = F.of_record(record(captured))
    assert now["products"]["he"] is not None and term(now) == (0.0, None)
    served = 145_568.0
    dial = F.of_record(record(captured | {"at": {f"{served:.0f}": {"k": served, "rows": captured["rows"],
                                                                   "p_match": []}}}))
    at_4121 = dial | {"products": {x: v for x, v in dial["products"].items() if x != "at"}}
    assert "keep none at it" in term(at_4121 | {"trace": {}}, served)[1]
    assert term(at_4121, served) == term(dial, served) and term(dial, served)[1] is None
    own = {"k": 65536.0, "he": None}
    assert F.complete(now | {"products": own}, out, SHAPE).products == own


def test_a_trace_of_another_program_than_the_runs_is_never_priced():
    """A products term is priced only on a trace of the run's own program. A trace keeps no list of the tensors it
    trains, so `complete` reads the row's op log beside the trace's rows when the row is read (`other_program`,
    whichever version ingested it): where every matrix-product shape the rows run is a product the log ran, in either
    orientation, the term is priced as before; where one isn't (a rank, a width or a batch the capture took and the
    run didn't), the term is unmeasured at every k the trace keeps rows at, with the counts as its reason
    (`OTHER_PROGRAM`). A batched product is no match for a plain one, and a row with no op log isn't checked."""
    row = {"count": 2, "phase": "forward", "op": "mm", "dims": {"batch": 1, "m": 4, "k": 512, "n": 8},
           "token_dims": ["m"], "macs": 16384, "outputs": 32, "affected": [True, True], "widths": [16, 16],
           "rates": [16.0, 16.0], "paid": [True, True], "self": False, "cell_gates": 23682.0,
           "gathered": [False, False]}
    served = 145_568.0
    block = {"capture": work.SYSTEM_FROM, "k": 65536.0, "tokens": 8, "gates": None, "p_match": [], "rows": [row],
             "at": {f"{served:.0f}": {"k": served, "rows": [row], "p_match": []}}}
    log = [["fwd", "updated", "mm", "[[4, 512], [512, 8]]", 2, 2 * 16384]]
    f = F.of_record({"steps": [{"step": 0, "tokens": 8, "struct": "a", "by": {"fwd/updated": 2 * 16384}}],
                     "structures": {"a": {"updated": [], "log": log, "affected_share": 0.5, "sigma": 1.0,
                                          "token_factor": 1.0, "delta_grad_bits": 10}},
                     "formalism": {"trace": {"products": block}}})
    assert "other_program" not in f["products"]

    def bound(ops, k=65536.0):
        r = {"output": {"e_bytes": 10}} | ({} if ops is None else {"macs_log": {"by": {}, "ops": ops}})
        return F.bound(F.complete(f, r, SHAPE), 1e11, k)

    for ops in (None, {}, {"fwd updated mm [[4, 512], [512, 8]]": [2, 32768]},
                {"bwd grad mm [[8, 512], [512, 4]]": [2, 32768], "fwd theta mm [[4, 8], [8, 3]]": [1, 96]}):
        for k in (65536.0, served):
            b = bound(ops, k)
            assert b["terms"]["products"] >= 0 and "products" not in b["unmeasured"], (ops, k)
            assert "other_program" not in b["inputs"]["products"]
    for ops, ran in (({"fwd updated mm [[4, 512], [512, 3]]": [2, 12288]}, 1),
                     ({"fwd updated bmm [[1, 4, 512], [1, 512, 3]]": [2, 12288],
                       "fwd updated bmm [[2, 4, 512], [2, 512, 8]]": [1, 32768]}, 1)):
        for k in (65536.0, served):
            b = bound(ops, k)
            why = work.OTHER_PROGRAM.format(unmatched=1, shapes=1, ran=ran, largest="4×512 by 512×8")
            assert b["terms"]["products"] is None and b["unmeasured"]["products"] == why, (ops, k)
            assert b["inputs"]["products"]["other_program"] == {"shapes": 1, "unmatched": 1, "ran": ran,
                                                                "largest": [[4, 512, 8]]}
            assert "beta" not in b["inputs"]["products"]


def test_a_records_trace_publishes_what_its_rows_give_at_every_dial_k_whichever_version_ingested_it():
    """The 45M dense step's capture (`dense_45m.products.json`) as a record's trace. Ingested at 4.11 (rows that don't
    flag a gather's) it reads the products term unmeasured with its retrace reason until it is traced again. Ingested
    at 4.12.1 (products at the console's k only, none at the dial's other k) it publishes at the served block's
    145,568 bits, and at every k of the dial, what a fresh ingest of the same record does, where its stored products
    kept none there."""
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

    def term(r, c, k):
        b = F.bound(F.measure(r, SHAPE), c, k)
        return b["terms"]["products"], b["unmeasured"].get("products")

    def stored(r):
        trace = {x: v for x, v in r["formalism"]["trace"].items() if x != "products"}
        return r | {"formalism": r["formalism"] | {"trace": trace}}

    c, k = policy.CONSOLE
    rows = [{x: v for x, v in r.items() if x != "gathered"} for r in fixture["rows"]]
    at_411 = row({x: v for x, v in fixture.items() if x not in ("at", "p_match")} | {"capture": "4.11", "rows": rows})
    assert term(stored(at_411), c, k) == term(at_411, c, k) == (None, work.HE_RETRACE)
    now, at_4121 = row(fixture), row(fixture)
    at_4121["formalism"]["products"] = {x: v for x, v in at_4121["formalism"]["products"].items() if x != "at"}
    served = 145_568.0
    assert "keep none at it" in term(stored(at_4121), c, served)[1]
    assert F.measure(at_4121, SHAPE).products == F.measure(now, SHAPE).products
    assert term(at_4121, c, served) == term(now, c, served) and term(now, c, served)[1] is None
    for k in DIAL_K:
        assert term(at_4121, c, k) == term(now, c, k)
        assert F.published(F.measure(at_4121, SHAPE), c, k) == F.published(F.measure(now, SHAPE), c, k)


def test_reduction_update_and_the_two_forms_are_sensitivities_never_the_bound():
    """They come out labelled beside the kind: reduction and update where the step's token work fills two or more
    blocks (b = 8 at c = 1e11), unmeasured without the cuts, and Slowdown's token form and the decoding form,
    targets; asking for any of them in the max is refused. A record with no product rows has no measured kind, so
    where Fit fails ρ is unmeasured, whatever the token form gives."""
    m, c, k = measured(u1=1e9), 1e11, 1e5
    b = F.bound(m, c, k)
    assert b["sensitivities"]["token_form"] > 1
    assert b["sensitivities"]["reduction"] is None and set(b["unmeasured"]) == {"products", "reduction", "update"}
    cut = replace(m, parts=(replace(m.parts[0], partial_bits=1e6, grad_bits=2e6),))
    b = F.bound(cut, c, k)
    assert b["inputs"]["reduction"]["blocks"] == [8] and set(b["unmeasured"]) == {"products"}
    assert b["sensitivities"]["reduction"] == pytest.approx(c * 7 * 1e6 / (k * 1e12))
    assert b["sensitivities"]["update"] == pytest.approx(c * 2e6 / (k * 1e12))
    assert b["max"] is None and b["binding"] is None and set(b["labels"]) == set(F.SENSITIVITIES)
    assert "split by tokens" in b["labels"]["reduction"] and "a target, not a bound" in b["labels"]["token_form"]
    r = F.at(cut, c, k)
    assert not r["fit"] and r["rho"] is None and r["bound"]["sensitivities"] == b["sensitivities"]
    for kinds in (("products", "reduction"), ("update",), ("token_form",), ("decoding_form",)):
        with pytest.raises(ValueError, match="sensitivities, never in the bound's max"):
            F.bound(cut, c, k, kinds=kinds)


