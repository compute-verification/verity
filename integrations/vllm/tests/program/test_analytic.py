"""analytic.py: config_of reproduces the hand-written static configs; the rotary table is the analytic function of C
(gate I9); untied checkpoints get an lm_head weights field; the structure gate fails unclassified extras (I8)."""
from __future__ import annotations

import numpy as np
import pytest

from verity_vllm.check.fold_compare import DEFAULT_COS_SIN as B0_FIXTURE
from verity_vllm.program import model as analytic
from verity.ml.definitions.registry import b1

# the relevant keys of the checkpoints' config.json (HF hub, the pinned revisions)
SMOLLM2_135M = {"model_type": "llama", "hidden_size": 576, "num_hidden_layers": 30, "num_attention_heads": 9, "num_key_value_heads": 3,
                "intermediate_size": 1536, "vocab_size": 49152, "rms_norm_eps": 1e-5, "rope_theta": 100000.0,
                "max_position_embeddings": 8192, "tie_word_embeddings": True, "attention_bias": False}
QWEN25_15B = {"model_type": "qwen2", "hidden_size": 1536, "num_hidden_layers": 28, "num_attention_heads": 12, "num_key_value_heads": 2,
              "intermediate_size": 8960, "vocab_size": 151936, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
              "max_position_embeddings": 131072, "tie_word_embeddings": True}
TINYLLAMA_11B = {"model_type": "llama", "hidden_size": 2048, "num_hidden_layers": 22, "num_attention_heads": 32, "num_key_value_heads": 4,
                 "intermediate_size": 5632, "vocab_size": 32000, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
                 "max_position_embeddings": 2048, "tie_word_embeddings": False}
LLAMA32_1B = {"model_type": "llama", "hidden_size": 2048, "num_hidden_layers": 16, "num_attention_heads": 32, "num_key_value_heads": 8,
              "head_dim": 64, "intermediate_size": 8192, "vocab_size": 128256, "rms_norm_eps": 1e-5, "rope_theta": 500000.0,
              "max_position_embeddings": 131072, "tie_word_embeddings": True,
              "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0, "original_max_position_embeddings": 8192,
                               "rope_type": "llama3"}}

STATIC_KEYS = ("H", "NL", "NH", "KVH", "D", "I", "V", "QKV", "GU", "EPS", "ROPE_THETA", "MAX_POS", "BIAS", "tie_word_embeddings")


@pytest.mark.parametrize("hf, ref", [(SMOLLM2_135M, b1.B0_CONFIG), (QWEN25_15B, b1.B1_CONFIG)])
def test_config_of_reproduces_hand_written_configs(hf, ref):
    C, prov = analytic.config_of(hf, model=ref["model"], revision=ref["revision"])
    assert {k: C[k] for k in STATIC_KEYS} == {k: ref[k] for k in STATIC_KEYS}
    assert C["BN"] == b1.fa2_kblock_n(ref["D"])
    assert set(prov) >= set(STATIC_KEYS) and prov["QKV"] == "derived" and prov["H"] == "hf-config"
    if hf["model_type"] == "qwen2":
        assert C["BIAS"] is True and prov["BIAS"] == "derived"     # vLLM's Qwen2Attention: bias=True, not in config.json


def test_config_of_untied_and_hand_fields():
    C, prov = analytic.config_of(TINYLLAMA_11B, hand={"revision": "abc"})
    assert C["tie_word_embeddings"] is False and C["D"] == 64 and C["QKV"] == (32 + 8) * 64 and C["BIAS"] is False
    assert prov["revision"] == "hand" and prov["D"] == "derived"
    fields = [f for f, _ in analytic.weights_type(C).fields]
    assert fields == ["embed", "layers", "norm", "cos_sin", "lm_head"]
    assert [f for f, _ in analytic.weights_type(b1.B0_CONFIG).fields] == ["embed", "layers", "norm", "cos_sin"]
    assert analytic.weights_type(b1.B0_CONFIG) == b1.weights_type(b1.B0_CONFIG)


def test_cos_sin_table_shape_and_first_row():
    tab = analytic.cos_sin_table(b1.B0_CONFIG, max_pos=4)
    assert tab.shape == (4, 64) and tab.dtype == np.uint16
    # position 0: cos = 1.0 (0x3F80), sin = 0.0 (0x0000)
    assert set(tab[0, :32].tolist()) == {0x3F80} and set(tab[0, 32:].tolist()) == {0x0000}


def test_llama3_rope_scaling_changes_low_frequencies_only():
    C, _ = analytic.config_of(LLAMA32_1B)
    scaled = analytic._inv_freq(C)
    plain = analytic._inv_freq({**C, "rope_scaling": None})
    assert scaled[0] == plain[0]                        # high frequency (wavelength < original/high_freq_factor) untouched
    assert scaled[-1] == pytest.approx(plain[-1] / 32.0)   # lowest frequency divided by the factor
    assert (scaled <= plain + 1e-12).all()


@pytest.mark.skipif(not B0_FIXTURE.exists(), reason="B0 rotary fixture not present")
def test_check_cos_sin_against_the_captured_b0_table():
    r = analytic.check_cos_sin(b1.B0_CONFIG, B0_FIXTURE)
    assert r["status"] == "pass" and r["shape_ok"] and r["n_outside_bound"] == 0
    assert r["max_ulp"] <= 2 and r["n_diff"] <= 8          # libm differences: a handful of elements, near zeros of cos/sin
    for key, bad in (("ROPE_THETA", 10000.0), ("D", 128), ("MAX_POS", 4096)):
        assert analytic.check_cos_sin({**b1.B0_CONFIG, key: bad}, B0_FIXTURE)["status"] == "fail", key
    w = np.load(B0_FIXTURE).copy()
    w[100, 3] ^= 0x0004
    assert analytic.check_cos_sin(b1.B0_CONFIG, w)["status"] == "fail"


def test_check_cos_sin_self_consistent_at_qwen_size():
    tab = analytic.cos_sin_table(b1.B1_CONFIG)
    r = analytic.check_cos_sin(b1.B1_CONFIG, tab)
    assert r["status"] == "pass" and r["n_elements"] == 131072 * 128 and r["max_ulp"] == 0
