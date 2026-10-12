"""The γ model (`model/gamma.py`): shapes, the primes it recommends, and the bound's arithmetic."""

from __future__ import annotations

import math
import random
import sys
from dataclasses import replace
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from model import frontier as F  # noqa: E402
from model import gamma as G  # noqa: E402


SMALL_PRIMES = [q for q in range(2, 5000) if all(q % f for f in range(2, int(q**0.5) + 1))]
SIEVE = math.prod(SMALL_PRIMES)


def is_prime(n: int, rounds: int = 40) -> bool:
    if n < 5000:
        return n in SMALL_PRIMES
    if math.gcd(n, SIEVE) != 1:
        return False
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    rng = random.Random(n)
    for _ in range(rounds):
        x = pow(rng.randrange(2, n - 1), d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def test_shapes():
    assert G.LLAMA_8B.linears() == 6_979_321_856 and G.LLAMA_8B.lm_head() == 2 * 525_336_576
    assert G.MISTRAL_24B.linears() == 22_229_811_200
    assert G.LLAMA_70B.linears() == 68_451_041_280
    assert G.LLAMA_70B.lm_head() == 2_101_346_304 and replace(G.LLAMA_70B, lm_head_bytes=1).lm_head() == 1_050_673_152
    assert [s.kv_bytes_per_token() for s in G.SHAPES] == [128 << 10, 160 << 10, 320 << 10]


def test_mu_chip_by_model():
    chip = [G.mu_chip(G.e_bytes(s, 60)) for s in G.SHAPES]
    assert chip[0] == pytest.approx(0.0280, abs=5e-4)
    assert chip[1] == pytest.approx(0.0095, abs=5e-4)
    assert chip[2] == pytest.approx(0.0032, abs=5e-4)


@pytest.mark.parametrize("bits,c", [(64, 189), (128, 173), (256, 189), (512, 569), (768, 825), (1024, 105),
                                    (1536, 3453), (2048, 1557)])
def test_full_slot_primes(bits, c):
    """Each is the smallest c with 2^bits − c prime and ≡ 3 mod 4, so Sloth's square root (x^((p+1)/4)) works."""
    p = (1 << bits) - c
    assert is_prime(p) and p % 4 == 3
    assert not any(d % 4 == 1 and is_prime((1 << bits) - d) for d in range(1, c))
    assert G.mu_slot(bits - 2.0**-50, bits) < 1e-15


@pytest.mark.parametrize("bits,c", [(64, 189), (256, 189), (512, 569), (768, 825)])
def test_e4m3_payloads_fit_below_p(bits, c):
    """A payload of bits/8 E4M3 weights is below p: the largest has top byte 0xFE, since 0xFF is NaN."""
    p = (1 << bits) - c
    assert (0xFF >> 3) & 15 == 15 and 0xFF & 7 == 7  # E4M3's NaN code
    assert ((0xFE + 1) << (bits - 8)) - 1 < p
    assert len(G.e4m3_values()) == 253  # 254 non-NaN codes, ±0 equal


def test_mersenne_and_2_255_19_residues():
    assert ((1 << 61) - 1) % 4 == 3 and ((1 << 521) - 1) % 4 == 3
    p = (1 << 255) - 19
    assert p % 4 == 1 and p % 3 == 1 and (p - 1) % 5 != 0
    assert G.mu_slot(61, 64) == pytest.approx(3 / 64)


def test_e4m3_entropy_range():
    """Order-0 entropy of rounded Gaussian/Laplace weights, independent of scale: s_c = 1 − H/8 in 0.15–0.19."""
    h_gauss = G.e4m3_entropy(G.gauss_cdf, 448 / 5)
    h_lap = G.e4m3_entropy(G.laplace_cdf, 448 / 10)
    assert 6.4 < h_gauss < 6.55 and 6.7 < h_lap < 6.85
    assert G.e4m3_entropy(G.gauss_cdf, 448 / 4) == pytest.approx(h_gauss, abs=0.02)


def test_saving_lp():
    """The closed form equals the LP's optimum over its three options."""
    for chip, f, h_over_8 in [(0.03, 0.1, 0.81), (0.003, 0.02, 0.85), (0.01, 0.5, 0.7)]:
        s_c = 1 - h_over_8
        best = 0.0
        steps = 200
        for i in range(steps + 1):
            u1 = chip * i / steps
            u2 = min((chip - u1) / h_over_8, f)
            u3 = min(f - u2, 1 - u1 - u2)
            best = max(best, u1 + u2 + s_c * u3)
        assert G.saving(chip, f, s_c) == pytest.approx(best, rel=2e-2)


def test_f_cap_forms():
    c = G.MEASURED[2]
    keep = G.f_cap_keep_decode(1.56, 0.0, c.eps, c.r)
    free = G.f_cap_decode_free(1.56, 0.0, c.eps, c.r)
    assert free == pytest.approx(0.117, abs=1e-3)  # the red team's F1, batch 1, context 512
    assert 0.19 * keep == pytest.approx(0.016, abs=1e-3)
    assert G.f_cap_keep_decode(c.eps + c.tau, c.tau, c.eps, c.r) == 0.0  # no 1/r floor when decode fills the ALU


def test_mu_sample_and_lambda():
    assert G.mu_sample(2.0**-40, 16, 10**6) == pytest.approx(1.73e-6, rel=0.01)
    assert G.mu_lambda(4096) == pytest.approx(0.0014, rel=0.01)


def test_s_word_per_codeword():
    """The red team's F14: s_c per stored codeword, 0.20 for 60 payload bits packed at 61."""
    assert G.s_word(G.packed(G.MEASURED[3]), 0.19) == pytest.approx(0.203, abs=1e-3)
    assert G.s_word(G.MEASURED[3], 0.19) == pytest.approx(0.194, abs=1e-3)
    full = replace(G.WIDE["p511"], p_bits=512, payload_bits=512)
    assert G.s_word(full, 0.19) == pytest.approx(0.19)


def test_gamma_points():
    packed = G.Point(G.LLAMA_70B, 1, 4096, True, G.packed(G.MEASURED[3]), idle=0.10).terms()
    assert packed["fits"] and packed["gamma_bw"] == pytest.approx(0.0235, abs=3e-4)
    assert packed["gamma_decode_free"] > 2 * packed["gamma_bw"]
    unpacked = G.Point(G.LLAMA_70B, 1, 4096, True, G.MEASURED[3], idle=0.10).terms()
    assert unpacked["mu_slot"] - packed["mu_slot"] == pytest.approx(3 / 64)
    kv_open = G.Point(G.LLAMA_8B, 1, 4096, False, G.packed(G.MEASURED[3]), idle=0.37).terms()
    assert kv_open["uncertified_reads"] == pytest.approx(0.0617, abs=1e-3) and kv_open["gamma_bw"] > 0.09
    assert not G.Point(G.LLAMA_70B, 32, 4096, True, G.MEASURED[3]).terms()["fits"]


def test_free_kv_payloads():
    """The operator picks the prompts, so the KV cache's words are rebuilt first, from free payloads: at 70B batch 1
    the term is min(b, (budget − εa)/(k_KV·rε)), and more rounds on the KV words alone buy most of it back."""
    c = replace(G.WIDE["p511"], tau=0.011, p_bits=512, payload_bits=512)
    th = G.WIDE_THETA["p511"]
    t = G.Point(G.LLAMA_70B, 1, 4096, True, c, idle=0.10, theta=th).terms()
    a = t["E_GB"] / (t["E_GB"] + t["kv_GB"])
    budget = t["d"] * (1 + th) - c.tau
    assert t["mu_rebuild"] == pytest.approx(min(1 - a, (budget - c.eps * a) / (c.r * c.eps)))
    assert t["mu_rebuild"] == pytest.approx(0.0034, abs=1e-4)
    four = G.Point(G.LLAMA_70B, 1, 4096, True, c, idle=0.10, theta=th, kv_rounds=4).terms()
    assert four["mu_rebuild"] < t["mu_rebuild"] / 3 and four["slowdown"] < 1.04 * t["slowdown"]
    packed = G.Point(G.LLAMA_70B, 1, 4096, True, G.packed(G.MEASURED[3]), idle=0.10, kv_rounds=4).terms()
    assert packed["gamma_bw"] == pytest.approx(0.0106, abs=3e-4)


def test_extrapolated_wide_words():
    assert G.extrapolated(768).eps == pytest.approx(G.WIDE["p767"].eps) and G.extrapolated(768).r == 503
    assert G.extrapolated(512).eps == pytest.approx(G.WIDE["p511"].eps)
    k = G.extrapolated(2048)
    assert k.eps == pytest.approx(1.68) and k.r * k.eps == pytest.approx(2253, abs=1) and G.mu_slot(2048, 2048) == 0


def test_kv_codec_and_its_encode():
    """The KV words' codec apart from E's. The honest encodes each KV word once (a new token's in its step, the
    prompt's at prefill) at the price E_c = k_KV·rε the adversary pays to rebuild one in a window; at 70B batch 8
    a 1536-bit KV codec at one round beats E's 512-bit one at four on both γ_bw and the step."""
    c = replace(G.WIDE["p511"], tau=0.011, p_bits=512, payload_bits=512)
    th = G.WIDE_THETA["p511"]
    base = G.Point(G.LLAMA_70B, 8, 4096, True, c, idle=0.10, theta=th, out_tokens=512)
    assert replace(base, kv_codec=c).terms() == base.terms()
    rounds = replace(base, kv_rounds=4).terms()
    wide = replace(base, kv_codec=replace(G.extrapolated(1536), tau=0.011)).terms()
    assert wide["gamma_bw"] < rounds["gamma_bw"] - 4e-4 and wide["slowdown"] < rounds["slowdown"] - 0.2
    e, kv = rounds["E_GB"], rounds["kv_GB"]
    a, b = e / (e + kv), kv / (e + kv)
    e_c = 4 * c.r * c.eps
    assert rounds["prefill_passes"] == pytest.approx(e_c * b * 4095 / 4096)
    budget = rounds["d"] * (1 + th) - c.tau
    assert rounds["mu_rebuild"] == pytest.approx((budget - c.eps * a) / e_c)  # KV words first, and not all of them
    assert rounds["slowdown_served"] > rounds["slowdown"] + 0.2


def test_squaring_floor():
    """The issue bound of a 768-bit squaring is 720 instructions, 0.22 of a pass; a squaring's cost per byte grows
    with the width, so the conjectured encode per byte grows as its square."""
    assert G.sq_floor(768) == pytest.approx(0.2217, abs=1e-4)
    assert G.sq_floor(768, ops=720) == G.sq_floor(768)
    assert G.sq_floor(1536) / G.sq_floor(768) == pytest.approx(1.8, abs=0.01)
    c = replace(G.WIDE["p767"], p_bits=768)
    a = G.at_floor(c, 0.25, rounds=2)
    assert a.eps == 0.5 and a.r * a.eps == pytest.approx(0.9 * 2 * 766 * 0.25)
    wide = G.at_floor(replace(G.extrapolated(2048), p_bits=2048), G.sq_floor(2048))
    narrow = G.at_floor(c, G.sq_floor(768))
    assert (wide.r * wide.eps) / (narrow.r * narrow.eps) > 6 > G.sq_floor(2048) / G.sq_floor(768)


def test_adversary_at_the_floor():
    """The adversary's costs apart from the honest's: unset is the honest's; at the floor the adversary decodes its
    kept words and encodes faster, so both γ's rise while the step stays."""
    c = replace(G.WIDE["p767"], tau=0.011, p_bits=768, payload_bits=768)
    base = G.Point(G.LLAMA_70B, 1, 4096, True, c, idle=0.10, theta=G.WIDE_THETA["p767"], kv_rounds=4, out_tokens=512)
    assert replace(base, adversary=c).terms() == base.terms()
    t0, t1 = base.terms(), replace(base, adversary=G.at_floor(c, G.sq_floor(768))).terms()
    assert t1["slowdown"] == t0["slowdown"] and t1["prefill_passes"] == t0["prefill_passes"]
    assert t1["gamma_bw"] > t0["gamma_bw"] + 1e-3 and t1["gamma_decode_free"] > 2 * t0["gamma_decode_free"]
    kc = replace(G.extrapolated(2048), tau=0.011)
    kv = replace(base, kv_rounds=1, kv_codec=kc)
    assert kv.terms() == replace(kv, kv_adversary=kc).terms()
    assert replace(kv, kv_adversary=G.at_floor(kc, G.sq_floor(2048))).terms()["gamma_bw"] > kv.terms()["gamma_bw"]


def test_at_batch():
    c = G.WIDE["p511"]
    assert G.at_batch(c, G.WIDE_Y32["p511"], 8) is c and G.at_batch(c, G.WIDE_Y32["p511"], 32).y == 1.74
    assert all(G.MEASURED_Y32[k] > G.MEASURED[k].y for k in G.MEASURED)


def test_wide_words_without_the_data_assumption():
    """Decode-free (no bound on s_c), 70B batch 1: 2^512 − 569 at k = 1 stays under 1%, 2^256 − 189 doesn't."""
    def point(name):
        c = replace(G.WIDE[name], tau=0.011, p_bits=G.WIDE[name].slot_bits, payload_bits=G.WIDE[name].slot_bits)
        c = replace(c, y=G.WIDE_HIDDEN_Y[name])
        return G.Point(G.LLAMA_70B, 1, 4096, True, c, idle=0.10, theta=G.WIDE_THETA[name], kv_rounds=4).terms()
    assert point("p511")["gamma_decode_free"] < 0.01 < point("p255")["gamma_decode_free"]
    assert point("p511")["gamma_bw"] == pytest.approx(0.0040, abs=2e-4)


def test_alu_idle_falls_with_k():
    """The joint priced residual's compute side: k sets the ALU the window leaves unforced (KV decode isn't forced,
    since its payloads are free)."""
    hidden = {k: replace(G.MEASURED[k], y=max(1.05, G.MEASURED[k].eps + 0.011 + 0.05), tau=0.011) for k in (1, 4, 5)}
    idle = {k: G.Point(G.LLAMA_70B, 1, 4096, True, c, idle=0.10).terms()["alu_idle"] for k, c in hidden.items()}
    assert idle[1] == pytest.approx(0.75, abs=0.01)
    assert idle[1] > idle[4] > idle[5] and idle[5] == pytest.approx(0.14, abs=0.01)


def test_frontier_at_the_floor():
    """The design note's point and its neighbours at the squaring floor (70B batch 1, the adversary at the measured
    pipes' floor and the max clock): no squaring reaches the issue bound; the decode rule meets the measured fused
    steps at 512, 768 and 1023 bits; the measured 1023-bit kernel buys nothing over 768 for 0.8 more step; E on 1,536
    bits brings the no-assumption γ under 1% with its decode apart at one block an SM (either end; at the issue bound
    only the low end), thread-per-word only for a further step; 2,048 bits buys little more for more step; an honest
    at the max clock is faster at the same γ."""
    assert F.KAPPA == pytest.approx(2430 / 2092)
    assert [F.issue_count(b) for b in (512, 768, 1023, 1536, 2048)] == [352, 696, 1120, 2184, 3488]
    assert F.pipe_clocks(768) == pytest.approx(7.77, abs=0.01)
    assert [F.pipe_ops(b) for b in (512, 768, 1023, 1536, 2048)] == pytest.approx([1.45, 1.428, 1.38, 1.398, 1.356],
                                                                                   abs=0.001)
    assert [F.rho_sq(b) for b in (768, 1536, 2048)] == pytest.approx([5.88e10, 1.915e10, 1.236e10], rel=0.002)
    assert F.encode_bound(1536) == pytest.approx(1.387e7, rel=0.002)
    assert F.encode_bound(1536) * 192 == pytest.approx(2.66e9, rel=0.005)
    assert F.encode_bound(768) * 96 == pytest.approx(8.19e9, rel=0.005)
    assert F.OFF_ISSUE == pytest.approx((4.57, 7.13), abs=0.01)
    assert F.fused_y(768, 1, F.OFF_ISSUE[0]) == pytest.approx(F.FUSED_768_Y, abs=0.005)
    assert F.fused_y(768, 32, F.OFF_ISSUE[0]) == pytest.approx(F.FUSED_768_Y32)
    assert F.decode_eps(512, 5.17) == pytest.approx(G.WIDE["p511"].eps, abs=0.005)
    assert F.fused_y(1023, 1, F.OFF_ISSUE[1]) == pytest.approx(F.FUSED_1023_Y, abs=0.005)
    assert 1.961 <= F.fused_y(1023, 32, F.OFF_ISSUE[1]) <= 1.961 + 0.07
    t = F.today()
    assert t["gamma_bw"] == pytest.approx(0.0054, abs=2e-4)
    assert t["gamma_decode_free"] == pytest.approx(0.0132, abs=2e-4)
    assert t["slowdown"] == pytest.approx(1.39, abs=0.01)
    assert F.today(ops=1.0)["gamma_decode_free"] == pytest.approx(0.0174, abs=2e-4)
    assert F.kv_attn(1) == pytest.approx(0.52, abs=0.005) and F.SHARED == pytest.approx(0.81, abs=0.005)
    fast = F.today(clock_mhz=G.CLOCK_MAX_MHZ)
    assert fast["slowdown"] < t["slowdown"] - 0.08 and abs(fast["gamma_bw"] - t["gamma_bw"]) < 5e-4
    _, t1023 = F.pick(F.options(G.LLAMA_70B, 1, F.e_codec(1023, 1, F.OFF_ISSUE[1]), off=F.OFF_ISSUE[1]))
    assert t["gamma_decode_free"] - t1023["gamma_decode_free"] < 0.001 and t1023["slowdown"] > t["slowdown"] + 0.7
    assert F.SQUARING[768] == pytest.approx(0.552, abs=0.001)
    assert F.apart_off(1536) == pytest.approx((4.68, 6.06), abs=0.01)
    p = F.point()
    assert p["gamma_bw"] == pytest.approx(0.0043, abs=2e-4)
    assert p["gamma_decode_free"] == pytest.approx(0.0078, abs=2e-4)
    assert p["slowdown"] == pytest.approx(1.94, abs=0.01)
    high = F.point(off=F.apart_off(1536)[1])
    assert high["gamma_decode_free"] < 0.0092 and high["slowdown"] < 2.4
    assert F.point(ops=1.0)["gamma_decode_free"] < 0.0098 < 0.01 < F.point(ops=1.0, off=F.apart_off(1536)[1])[
        "gamma_decode_free"]
    tpw = F.point(off=F.OFF_ISSUE[1])
    assert tpw["gamma_decode_free"] < 0.0097 and tpw["slowdown"] > p["slowdown"] + 0.6
    off = F.apart_off(2048)[0]
    _, wide = F.pick(F.options(G.LLAMA_70B, 1, F.e_codec(2048, 1, off), off=off))
    assert p["gamma_decode_free"] - wide["gamma_decode_free"] < 0.0015 and wide["slowdown"] > p["slowdown"] + 0.15


def test_built_point():
    """The point's codec as ra-codec-tensor built it: its untagged step lands inside the predicted decode bracket at the
    locked clock, its tag (0.44) above the modelled 0.227, and γ_bw stays under 1% with no assumption on W at the pipe
    floor at batch 1 and 32 (the red team's 12:22Z reprice: 0.87% / 0.79%)."""
    lo, hi = F.apart_off(1536)
    assert F.fused_y(1536, 1, lo) < F.FUSED_1536_Y[0] < F.fused_y(1536, 1, hi)
    assert F.SKETCH_1536_Y[0] - F.FUSED_1536_Y[0] == pytest.approx(0.443) and F.TAG < 0.443
    b1, b32 = F.built(), F.built(batch=32)
    assert b1["gamma_bw"] == pytest.approx(0.0045, abs=1e-4)
    assert b1["gamma_decode_free"] == pytest.approx(0.0086, abs=1e-4)
    assert b1["slowdown"] == pytest.approx(2.30, abs=0.01)
    assert b32["gamma_bw"] == pytest.approx(0.0073, abs=1e-4)
    assert b32["gamma_decode_free"] == pytest.approx(0.0077, abs=1e-4)
    assert b32["slowdown"] == pytest.approx(2.58, abs=0.01)
    assert F.built(ops=1.0)["gamma_decode_free"] > 0.01 > F.built(batch=32, ops=1.0)["gamma_decode_free"]
