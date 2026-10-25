"""class_size (V3b): a window class's padded size, projected from one Program lowered at a small probe class.

* today's lowering projected from a probe of 3 positions is, template for template, TINY's exact lowering at a class of 7
  positions (`window_circuit.class_circuit`): every template but the heads and their key gathers is a fixed count a
  position, `TokenFeed_v1` one a prompt position;
* v19's heads (#4263), at a block of 3 keys: a head over T keys is its first block, ceil(T / 3) - 1 full ones and its end;
  its keys reach it through one gather a full block, shared, and its first block's own where that block is neither one key
  nor full;
* a probe that doesn't divide evenly is refused, and so is a template `size` has no measurement for; m is the log of the
  cells.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
from verity_vllm.program.predict.phi import Phi
from verity_vllm.program.predict.steps import Registered
from verity_vllm.program.predict.window_circuit import Grain, WindowClass, class_circuit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import class_size as CZ  # noqa: E402

#: integrations/vllm's `tests.program.predict.test_steps.TINY` and its grain in test_window_circuit
TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128,
        "num_attention_heads": 4, "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256, "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True, "max_position_embeddings": 64, "hidden_act": "silu"}
GRAIN = Grain(gemm_words=512, rows=8, select=64)
PROBE = WindowClass(1, 2, 2)


def _census(cls: WindowClass) -> dict[str, int]:
    phi = Phi(model={"repo": "org/m", "revision": "r", "hf_config": TINY}, target={"compute_capability": [12, 0], "num_sms": 188})
    return CZ.census(class_circuit(Registered.of(phi, default_seed=7), cls, 1, GRAIN))


def test_todays_lowering_projected_from_a_probe_is_the_exact_lowering_at_a_larger_class():
    probe, at = _census(PROBE), WindowClass(1, 5, 3)
    assert PROBE.positions == 3 and at.positions == 7
    assert CZ.project(probe, PROBE, at, heads="v18") == _census(at)


def test_v19s_heads_are_key_blocks_with_their_gathers():
    head = "AttentionHead_v18{{T={},D=4,BN=3,MASKED_FROM=1}}"
    probe = {**{head.format(t): 4 for t in (1, 2, 3)}, "AllGather_v1{WORLD=2,N=4}": 2, "AllGather_v1{WORLD=3,N=4}": 2 + 3,
             "RMSNormTriton_v5{N=4}": 6, "TokenFeed_v1{NP=2,FIRST=true}": 1, "TokenFeed_v1{NP=2,FIRST=false}": 1}
    assert CZ.project(probe, PROBE, WindowClass(1, 5, 3)) == {
        "RMSNormTriton_v5{N=4}": 14, "TokenFeed_v1{NP=5,FIRST=true}": 1, "TokenFeed_v1{NP=5,FIRST=false}": 4,
        "AttnBlockFirst_v1{D=4,NVIS=1,BN=3}": 12, "AttnBlockFirst_v1{D=4,NVIS=2,BN=3}": 8, "AttnBlockFirst_v1{D=4,NVIS=3,BN=3}": 8,
        "AttnBlock_v5{D=4,NVIS=3,FIRST=false,BN=3,CHECK=false}": 16, "AttnBlock_v5{D=4,NVIS=3,FIRST=false,BN=3,CHECK=true}": 4,
        "AttnHeadEnd_v1{D=4}": 28, "AllGather_v1{WORLD=2,N=4}": 4, "AllGather_v1{WORLD=3,N=4}": 7 + 4}
    with pytest.raises(ValueError, match="not a fixed count at each of its 3 positions"):
        CZ.project({**probe, "RMSNormTriton_v5{N=4}": 7}, PROBE, WindowClass(1, 5, 3))
    with pytest.raises(ValueError, match="not one count at each T"):
        CZ.project({**probe, head.format(3): 5}, PROBE, WindowClass(1, 5, 3))


def test_size_prices_the_measured_units_and_refuses_the_unmeasured():
    s = CZ.size({"Gemm_v2{K=2048,N=1}": 3, "AllGather_v1{WORLD=4,N=16}": 1}, times=2)
    assert (s["units"], s["ands"], s["m"]) == (8, 6 * 1_053_595, math.ceil(math.log2(6 * 2**24 + 2 * 2**22)))
    ands, k, src = CZ.measured("AttnBlockFirst_v1{D=64,NVIS=72,BN=128}")
    assert (ands, k) == (round(1_231_456 + (9_075_919 - 1_231_456) * 55 / 111), 26) and "interpolated" in src
    assert CZ.measured("AttnBlockFirst_v1{D=64,NVIS=17,BN=128}")[:2] == (1_231_456, 24)
    with pytest.raises(ValueError, match="no measured ANDs or k_log for Gemm_v2"):
        CZ.size({"Gemm_v2{K=4096,N=16}": 1})
