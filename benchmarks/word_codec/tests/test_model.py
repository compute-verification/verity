"""The γ model (`model/gamma.py`): shapes, the primes it recommends, and the bound's arithmetic."""

from __future__ import annotations

import random
import sys
from dataclasses import replace
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from model import gamma as G  # noqa: E402


def is_prime(n: int, rounds: int = 40) -> bool:
    if n < 4:
        return n in (2, 3)
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
    assert G.LLAMA_8B.linears() == 6_979_321_856 and G.LLAMA_8B.lm_head() == 525_336_576
    assert G.MISTRAL_24B.linears() == 22_229_811_200
    assert G.LLAMA_70B.linears() == 68_451_041_280
    assert [s.kv_bytes_per_token() for s in G.SHAPES] == [128 << 10, 160 << 10, 320 << 10]


def test_mu_chip_by_model():
    chip = [G.mu_chip(G.e_bytes(s, 60)) for s in G.SHAPES]
    assert chip[0] == pytest.approx(0.0300, abs=5e-4)
    assert chip[1] == pytest.approx(0.0098, abs=5e-4)
    assert chip[2] == pytest.approx(0.0032, abs=5e-4)


@pytest.mark.parametrize("bits,c", [(64, 189), (128, 173), (256, 189), (512, 569)])
def test_full_slot_primes(bits, c):
    """Each is the smallest c with 2^bits − c prime and ≡ 3 mod 4, so Sloth's square root (x^((p+1)/4)) works."""
    p = (1 << bits) - c
    assert is_prime(p) and p % 4 == 3
    assert not any(is_prime((1 << bits) - d) and ((1 << bits) - d) % 4 == 3 for d in range(1, c))
    assert G.mu_slot(bits - 2.0**-50, bits) < 1e-15


@pytest.mark.parametrize("bits,c", [(64, 189), (256, 189), (512, 569)])
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
    assert packed["fits"] and packed["gamma_bw"] == pytest.approx(0.0090, abs=3e-4)
    assert packed["gamma_decode_free"] > 6 * packed["gamma_bw"]
    unpacked = G.Point(G.LLAMA_70B, 1, 4096, True, G.MEASURED[3], idle=0.10).terms()
    assert unpacked["gamma_bw"] - packed["gamma_bw"] == pytest.approx(3 / 64, abs=1e-3)
    kv_open = G.Point(G.LLAMA_8B, 1, 4096, False, G.packed(G.MEASURED[3]), idle=0.37).terms()
    assert kv_open["uncertified_reads"] == pytest.approx(0.066, abs=1e-3) and kv_open["gamma_bw"] > 0.09
    assert not G.Point(G.LLAMA_70B, 32, 4096, True, G.MEASURED[3]).terms()["fits"]


def test_wide_words_without_the_data_assumption():
    """Decode-free (no bound on s_c), 70B batch 1: 2^512 − 569 at k = 1 stays under 1%, 2^256 − 189 doesn't."""
    def point(name):
        c = replace(G.WIDE[name], tau=0.011, p_bits=G.WIDE[name].slot_bits, payload_bits=G.WIDE[name].slot_bits)
        c = replace(c, y=G.WIDE_HIDDEN_Y[name])
        return G.Point(G.LLAMA_70B, 1, 4096, True, c, idle=0.10, theta=G.WIDE_THETA[name]).terms()
    assert point("p511")["gamma_decode_free"] < 0.01 < point("p255")["gamma_decode_free"]
    assert point("p511")["gamma_bw"] == pytest.approx(0.0038, abs=2e-4)


def test_alu_idle_falls_with_k():
    """The joint view: μ_rebuild drops out, and k sets the compute the window leaves unforced."""
    hidden = {k: replace(G.MEASURED[k], y=max(1.05, G.MEASURED[k].eps + 0.011 + 0.05), tau=0.011) for k in (1, 4, 5)}
    idle = {k: G.Point(G.LLAMA_70B, 1, 4096, True, c, idle=0.10).terms()["alu_idle"] for k, c in hidden.items()}
    assert idle[1] == pytest.approx(0.75, abs=0.01)
    assert idle[1] > idle[4] > idle[5] and idle[5] == pytest.approx(0.12, abs=0.01)
