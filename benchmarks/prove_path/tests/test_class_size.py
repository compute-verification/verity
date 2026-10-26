"""class_size (V3b): a window class's padded size, projected from one Program lowered at a small probe class.

* today's lowering projected from a probe of 3 positions is, template for template, TINY's exact lowering at a class of 7
  positions (`window_circuit.class_circuit`): every template but the heads and their key gathers is a fixed count a
  position, `TokenFeed_v1` one a prompt position and `SlotRow_v1` one a Program;
* v19's heads (#4263), at a block of 3 keys: a head over T keys is its first block, ceil(T / 3) - 1 full ones and its end;
  its keys reach it through one gather a full block, shared, and its first block's own where that block is neither one key
  nor full;
* a probe that doesn't divide evenly is refused, and so is a template `size` has no measurement for; m is the log of the
  cells;
* a gather's k_log is its rows' (its parts and its output, an hm96 slot each), circuits' composed values at every W they
  composed, and a gather of 256 parts needs a block of 2^28 bits, which `size` refuses as C-Flock does;
* with no ``--class``, ``prove_path.py`` takes the class of record.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
from verity_vllm.program.predict.phi import Phi
from verity_vllm.program.predict.steps import Registered
from verity_vllm.program.predict.window_circuit import CLASS_OF_RECORD, Grain, WindowClass, class_circuit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import class_size as CZ  # noqa: E402

#: integrations/vllm's `tests.program.predict.test_steps.TINY` and its grain in test_window_circuit
TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128,
        "num_attention_heads": 4, "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256, "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True, "max_position_embeddings": 64, "hidden_act": "silu"}
GRAIN = Grain(gemm_words=512, rows=9, select=64)
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
    assert (ands, k) == (round(4_526_991 + (5_664_223 - 4_526_991) * 8 / 16), 25) and "interpolated" in src
    assert CZ.measured("AttnBlockFirst_v1{D=64,NVIS=17,BN=128}")[:2] == (1_231_456, 24)
    assert [CZ.measured(f"AttnBlockFirst_v1{{D=64,NVIS={v},BN=128}}")[1] for v in (4, 5, 15, 16, 33, 34, 90, 91, 128)] == [
        22, 23, 23, 24, 24, 25, 25, 26, 26]
    assert [CZ.measured(f"AllGather_v1{{WORLD={w},N=64}}")[1] for w in (2, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128)] == [
        22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27]
    wn = ((64, 16), (32, 16), (128, 4), (167, 1), (255, 16))
    assert [CZ.measured(f"AllGather_v1{{WORLD={w},N={n}}}")[1] for w, n in wn] == [26, 25, 27, 27, 27]
    with pytest.raises(ValueError, match=r"no measured ANDs or k_log at Llama-3.2-1B for Gemm_v2\{K=4096,N=16\}, Foo_v1\{\}"):
        CZ.size({"Gemm_v2{K=4096,N=16}": 1, "AllGather_v1{WORLD=256,N=16}": 1, "Foo_v1{}": 1})
    with pytest.raises(ValueError, match=r"AllGather_v1\{WORLD=256,N=4\} \(2\^28\): an instance needs a block past 2\^27 bits"):
        CZ.size({"AllGather_v1{WORLD=128,N=4}": 4, "AllGather_v1{WORLD=256,N=4}": 2})
    assert CZ.measured("TokenFeed_v1{NP=1024,FIRST=false}")[:2] == (62, 23)
    assert CZ.measured("TokenFeed_v1{NP=2048,FIRST=true}")[:2] == (30, 24)
    assert (CZ.measured("SlotRow_v1{NP=1024}")[:2], CZ.measured("SlotMask_v1{}")[:2]) == ((0, 23), (32, 22))


def test_window_circuit_and_class_size_default_to_the_class_of_record():
    """Without ``--class``, ``window-circuit`` and ``class-size`` take the class of record, the certifier run of record's
    envelope at ASK-122's fixed slots (8 slots, 4 at each of buckets 0 and 16, EVERY 16; 1,024 prompt ids, 64 outputs); a
    given class is read as SLOTS,PROMPT,OUTPUTS[,LANES,EVERY], every slot at one boundary without LANES."""
    import prove_path as PP

    assert PP.window_class_of(None) == CLASS_OF_RECORD == WindowClass(8, 1024, 64, lanes=4, every=16)
    assert PP.window_class_of("8,1024,64,4,16") == CLASS_OF_RECORD
    assert PP.window_class_of("1,2,2") == PROBE and PROBE.boundaries == (0,)
