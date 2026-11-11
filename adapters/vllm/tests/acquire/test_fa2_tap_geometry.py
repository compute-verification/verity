"""FA2 M1 tap portability to sliding-window / softcap models (Gemma-2 hd 256, Phi-3 hd 96): torch-free checks.

1. `fa2_swapped` mirrors flash_api.cpp:646 `seqlenq_ngroups_swapped` exactly: a GQA decode launch is swapped ONLY without a sliding
   window (and dropout 0, D % 8 == 0, no alibi).  A mirror of `max_seqlen_q == 1 and H > Hk` alone declares the swapped geometry for
   Gemma-2's sliding layers at decode, which the kernel refuses (arm check b*h <= vt.H) -> named COMMIT FAIL, never a wrong binding.
2. The oracle's softcap parameters are the kernel's (flash_api.cpp:156-159) and `apply_softcap` follows the measured MUFU.TANH rules
   (identity below 2^-8, saturation at |x| >= 8) -- the GPU bit-exactness itself is fa2_tap_xcheck.py --cases gemma on a pod.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
for p in (REPO, os.path.join(REPO, "vllm-poc")):
    if p not in sys.path:
        sys.path.insert(0, p)

from verity_vllm.acquire.sources.hidden_source import fa2_kblock_n, fa2_swapped  # noqa: E402


def _oracle():
    spec = importlib.util.spec_from_file_location("fa2_attn_oracle", os.path.join(os.path.dirname(__file__), "fa2_attn_oracle.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("max_q,H,Hk,D,wl,wr,dropout,alibi,expect", [
    (1, 8, 4, 256, -1, -1, 0.0, None, True),       # Gemma-2-2B global layer at decode: swapped (rows = 2 groups, heads = 4)
    (1, 8, 4, 256, 4095, 0, 0.0, None, False),     # Gemma-2-2B sliding layer at decode (vLLM window (sw-1, 0)): NOT swapped
    (1, 32, 8, 64, -1, -1, 0.0, None, True),       # Llama-3.2-1B decode: swapped
    (1, 32, 32, 96, 2046, 0, 0.0, None, False),    # Phi-3-mini (MHA) sliding layer: never swapped (H == Hk)
    (1, 32, 32, 96, -1, -1, 0.0, None, False),
    (17, 8, 4, 256, -1, -1, 0.0, None, False),     # prefill: never swapped
    (1, 8, 4, 256, -1, -1, 0.1, None, False),      # dropout > 0: kernel does not swap
    (1, 8, 4, 256, -1, -1, 0.0, object(), False),  # alibi slopes: kernel does not swap
    (1, 8, 4, 100, -1, -1, 0.0, None, False),      # head_dim % 8 != 0: kernel does not swap
    (1, 8, 4, 256, -1, 0, 0.0, None, False),       # window_right == 0 alone (is_causal spelled as a window): not swapped
])
def test_fa2_swapped_mirrors_flash_api(max_q, H, Hk, D, wl, wr, dropout, alibi, expect):
    assert fa2_swapped(max_q, H, Hk, D, wl, wr, dropout, alibi) is expect


def test_swapped_and_unswapped_decode_have_equal_word_counts_but_different_geometry():
    # the failure mode the kernel-side arm check must refuse: same buffer size, different (HB, M) addressing
    b, H, Hk = 7, 8, 4
    hb_sw, m_sw = b * Hk, H // Hk
    hb_un, m_un = b * H, 1
    assert hb_sw * m_sw == hb_un * m_un and (hb_sw, m_sw) != (hb_un, m_un)


def test_kblock_n_hd256_is_64():
    assert fa2_kblock_n(256, 1) == 64 and fa2_kblock_n(64, 1) == 128
    with pytest.raises(RuntimeError):
        fa2_kblock_n(256, 2)


def test_softcap_params_match_flash_api():
    O = _oracle()
    P = O.P
    cap_bits, sl2 = O.softcap_params(256, 50.0)
    # params.softcap = softmax_scale / softcap (f32 / f32)
    assert P.f32_of(cap_bits) == np.float32(np.float32(256 ** -0.5) / np.float32(50.0))
    # params.scale_softmax_log2 = softcap * M_LOG2E (double product narrowed to f32)
    assert P.f32_of(sl2) == np.float32(50.0 * 1.4426950408889634)
    # Gemma-2-2B: query_pre_attn_scalar (256) == head_dim, so the default scale IS the model's
    assert np.float32(256 ** -0.5) == np.float32(1 / 16)


def test_apply_softcap_follows_mufu_tanh_rules():
    O = _oracle()
    P = O.P
    cap_bits, _ = O.softcap_params(256, 50.0)
    # tiny score: s * c below 2^-8 -> identity: word == FTZ f32 product bits
    s = P.bits_of(np.float32(0.5))
    prod = P.F32MulFtz.evaluate(s, cap_bits)
    assert abs(P.f32_of(prod)) < 2 ** -8 and O.apply_softcap(s, cap_bits) == prod
    # zero (padded key column) -> +0 exactly
    assert O.apply_softcap(0, cap_bits) == 0
    # huge score: s * c >= 8 -> saturates at exactly +1.0 / -1.0
    big = P.bits_of(np.float32(50.0 * 16 * 9))     # (s * scale / cap) = 9 >= 8
    assert O.apply_softcap(big, cap_bits) == 0x3F800000
    assert O.apply_softcap(big | 0x80000000, cap_bits) == 0xBF800000
    # a table word (0.6): equals the measured MUFU.TANH, which is NOT RN32(tanh)
    x = P.bits_of(np.float32(0.6))
    y = P.MufuTanh.evaluate(x)
    assert abs(P.f32_of(y) - np.tanh(0.6)) < 1e-4 and y != P.bits_of(np.float32(np.tanh(np.float32(0.6))))
