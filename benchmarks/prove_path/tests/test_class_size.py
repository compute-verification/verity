"""class_size (V3b): a window class's padded size, projected from one Program lowered at a small probe class.

* the lowering projected from a probe of 3 positions is, template for template, TINY's exact lowering at a class of 7
  positions (`window_circuit.class_circuit`), at a class of two lanes (its scan state row another) and at 129 positions,
  a head's second key block: every template but the heads is a fixed count a position, `TokenFeed_v1` one a prompt
  position past the first, `ScanSlots_v2` and `SelectEntry_v1` one a Program and `SelectCollect_v1` one at every
  other position;
* so it is for int8 Llama-3.2-1B on the H200, whose P·V groups read a key's value as two slices (VS 32), at key blocks
  of 192 and groups of 96 (`key_blocks`, FA3's `AttnBlockSoftmax_v2` below a head's first block);
* the cut's heads, at blocks of 4 keys and groups of 2: a head over T keys is its first visited block's softmax step and
  groups, ceil(T / 4) - 1 blocks below it (none guarded: a one-row decode's kernel guards its first visited block only),
  each its softmax step and 2 groups, block 0's softmax LAST, and its end;
* a probe that doesn't divide evenly, or whose heads aren't the cut's, is refused, and so is a template `size` has no
  measurement for (the cut's steps among them); m is the log of the cells; a select with its collector (and the entry)
  is its Calls' ANDs at the combine's k_log;
* a unit of few ANDs takes its rows' k_log (an hm96 slot each), as composed at every W + 1 rows measured, and a unit of
  257 rows needs a block of 2^28 bits, which `size` refuses as C-Flock does;
* with no ``--class``, ``prove_path.py`` takes the class of record.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
from verity_vllm.program.model import INT8_EXACT_QUANT
from verity_vllm.program.predict.phi import Phi
from verity_vllm.program.predict.steps import Registered
from verity_vllm.program.predict.window_circuit import CLASS_OF_RECORD, Grain, WindowClass, class_circuit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import class_size as CZ  # noqa: E402

#: integrations/vllm's `tests.program.predict.test_steps.TINY` and its grain in test_window_circuit
TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128,
        "num_attention_heads": 4, "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256, "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True, "max_position_embeddings": 64, "hidden_act": "silu"}
GRAIN = Grain(gemm_words=1024, rows=256, select=64)
PROBE = WindowClass(1, 2, 2)
LLAMA_1B = Path(__file__).resolve().parents[3] / "integrations/vllm/tests/program/predict/fixtures/llama32-1b.config.json"


def _registered(config: dict) -> Registered:
    phi = Phi(model={"repo": "org/m", "revision": "r", "hf_config": config}, target={"compute_capability": [12, 0], "num_sms": 188})
    return Registered.of(phi, default_seed=7)


def _census(cls: WindowClass, config: dict = TINY, grain: Grain = GRAIN) -> dict[str, int]:
    return CZ.census(class_circuit(_registered(config), cls, 1, grain, provable=False))


def test_the_lowering_projected_from_a_probe_is_the_exact_lowering_at_a_larger_class():
    probe, at, two = _census(PROBE), WindowClass(1, 5, 3), WindowClass(2, 5, 3)
    blocks = CZ.key_blocks(_registered(TINY), PROBE)
    assert PROBE.positions == 3 and at.positions == 7 and blocks == {"BN": 128, "later": "AttnBlockSoftmax_v1"}
    assert CZ.project(probe, PROBE, at, **blocks) == _census(at)
    assert CZ.project(probe, PROBE, two, **blocks) == _census(two)
    assert "ScanSlots_v2{NS=4550,NP=5,IDB=17,K0=12,K=45}" in _census(two)
    long, past = TINY | {"max_position_embeddings": 256}, WindowClass(1, 2, 128)
    exact = _census(past, long)
    assert past.positions == 129 and exact["AttnBlockSoftmax_v1{D=16,NVIS=128,FIRST=false,CHECK=false,PK=128,LAST=true}"] == 8
    assert exact["AttnBlockSoftmax_v1{D=16,NVIS=1,FIRST=true,CHECK=true,PK=128,LAST=false}"] == 8
    assert CZ.project(_census(PROBE, long), PROBE, past, **CZ.key_blocks(_registered(long), PROBE)) == exact


def test_value_slices_project_as_the_exact_lowering():
    """int8 Llama-3.2-1B on the H200, whose keys' values are each two rows of the value GEMM: its P·V groups are
    `AttnBlockPv_v1{..., VS=32}` at groups of 96 keys, and the probe's projection is still the exact lowering."""
    cfg = json.loads(LLAMA_1B.read_text()) | {"quantization_config": dict(INT8_EXACT_QUANT)}
    phi = Phi(model={"repo": "org/m", "revision": "r", "hf_config": cfg},
              target={"compute_capability": [9, 0], "num_sms": 132, "int8_exact": "linears"})
    reg, at = Registered.of(phi, default_seed=7), WindowClass(1, 5, 3)
    probe, exact = (CZ.census(class_circuit(reg, c, 1, provable=False)) for c in (PROBE, at))
    ts = range(1, at.positions + 1)
    assert {lab for lab in exact if lab.startswith("AttnBlock")} == {
        *(f"AttnBlockSoftmax_v1{{D=64,NVIS={t},FIRST=true,CHECK=true,PK=96,LAST=true}}" for t in ts),
        *(f"AttnBlockPv_v1{{D=64,N={t},VS=32,FIRST=true}}" for t in ts)}
    blocks = CZ.key_blocks(reg, PROBE)
    assert blocks == {"BN": 192, "later": "AttnBlockSoftmax_v2"}
    assert CZ.project(probe, PROBE, at, **blocks) == exact


def test_the_cuts_heads_are_key_blocks_of_softmax_steps_and_pv_groups():
    soft, pv = "AttnBlockSoftmax_v{}{{D=4,NVIS={},FIRST={},CHECK={},PK=2,LAST={}}}", "AttnBlockPv_v1{{D=4,N={},VS=2,FIRST={}}}"
    probe = {**{soft.format(1, t, "true", "true", "true"): 4 for t in (1, 2, 3)}, pv.format(1, "true"): 4,
             pv.format(2, "true"): 8, pv.format(1, "false"): 4, "AttnHeadEnd_v1{D=4}": 12, "RMSNormTriton_v5{N=4}": 6,
             "TokenFeed_v1{NP=2,FIRST=false}": 1}
    at = WindowClass(1, 5, 3)
    for v, later in ((1, "AttnBlockSoftmax_v1"), (2, "AttnBlockSoftmax_v2")):
        assert CZ.project(probe, PROBE, at, 4, later) == {
            "RMSNormTriton_v5{N=4}": 14, "TokenFeed_v1{NP=5,FIRST=false}": 4,
            **{soft.format(1, t, "true", "true", "true"): 4 for t in (1, 2, 3, 4)},
            **{soft.format(1, t, "true", "true", "false"): 4 for t in (1, 2, 3)},
            soft.format(v, 4, "false", "false", "true"): 12,
            pv.format(1, "true"): 8, pv.format(2, "true"): 20, pv.format(1, "false"): 8, pv.format(2, "false"): 28,
            "AttnHeadEnd_v1{D=4}": 28}
    with pytest.raises(ValueError, match="not a fixed count at each of its 3 positions"):
        CZ.project({**probe, "RMSNormTriton_v5{N=4}": 7}, PROBE, at, 4)
    with pytest.raises(ValueError, match="the probe's heads are not 4 a position of the cut at D 4, BN 4, PK 2 and VS 2"):
        CZ.project({**probe, soft.format(1, 3, "true", "true", "true"): 5}, PROBE, at, 4)
    with pytest.raises(ValueError, match=r"not 3 a position of the cut .* \(11 AttnHeadEnd_v1\{D=4\} units\)"):
        CZ.project({**probe, "AttnHeadEnd_v1{D=4}": 11}, PROBE, at, 4)
    with pytest.raises(ValueError, match="the probe's heads are not 4 a position of the cut at D 4, BN 2"):
        CZ.project(probe, PROBE, at, 2)
    with pytest.raises(ValueError, match=r"2 TokenFeed_v1\{NP=2,FIRST=false\} units, not the 1 its slot places"):
        CZ.project({**probe, "TokenFeed_v1{NP=2,FIRST=false}": 2}, PROBE, at, 4)


def test_size_prices_the_measured_units_and_refuses_the_unmeasured(monkeypatch):
    s = CZ.size({"Gemm_v2{K=2048,N=1}": 3, "OutCollect_v1{KO=64,FIRST=true}": 1}, times=2)
    assert (s["units"], s["ands"], s["m"]) == (8, 6 * 1_053_595 + 2 * 155, math.ceil(math.log2(6 * 2**24 + 2 * 2**22)))
    step = "AttnBlockSoftmax_v1{D=64,NVIS=17,FIRST=true,CHECK=true,PK=64,LAST=true}"
    with pytest.raises(ValueError, match=r"no measured ANDs or k_log at Llama-3.2-1B for AttnBlockSoftmax_v1\{D=64,NVIS=17,"):
        CZ.size({step: 1, "Gemm_v2{K=2048,N=1}": 1})
    assert [CZ.rows_k(r) for r in (3, 8, 9, 16, 17, 32, 33, 64, 65, 128, 129, 256, 257)] == [
        22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28]
    with pytest.raises(ValueError, match=r"no measured ANDs or k_log at Llama-3.2-1B for Gemm_v2\{K=4096,N=16\}, Foo_v1\{\}"):
        CZ.size({"Gemm_v2{K=4096,N=16}": 1, "SlotEntry_v1{KO=64,IDB=17,NP=1024}": 1, "Foo_v1{}": 1})
    monkeypatch.setitem(CZ.MEASURED, "Foo_v1{}", (1, CZ.rows_k(257), "a unit of 257 rows"))
    with pytest.raises(ValueError, match=r"Foo_v1\{\} \(2\^28\): an instance needs a block past 2\^27 bits"):
        CZ.size({"Gemm_v2{K=2048,N=1}": 4, "Foo_v1{}": 2})
    assert [CZ.measured(f"TokenFeed_v1{{NP={n},FIRST=false}}")[:2] for n in (1024, 2048)] == [(62, 23), (62, 24)]
    assert [CZ.measured(lab)[:2] for lab in ("ScanSlots_v2{NS=9353,NP=1024,IDB=17,K0=12,K=45}",
                                             "OutCollect_v1{KO=64,FIRST=true}", "OutCollect_v1{KO=64,FIRST=false}",
                                             "SlotEntry_v1{KO=64,IDB=17,NP=1024}")] == [
        (120, 24), (155, 22), (2_141, 22), (5_676, 22)]
    assert [CZ.measured(lab)[:2] for lab in ("SelectCollect_v1{V=128256,C=16,KO=64,FIRST=true}",
                                             "SelectCollect_v1{V=128256,C=16,KO=64,FIRST=false}",
                                             "SelectEntry_v1{V=128256,C=16,KO=64,FIRST=false,IDB=17,NP=1024}")] == [
        (1_346_657 + 155, 26), (1_346_657 + 2_141, 26), (1_346_657 + 2_141 + 5_676, 26)]


def test_window_circuit_and_class_size_default_to_the_class_of_record():
    """Without ``--class``, ``window-circuit`` and ``class-size`` take the class of record, the certifier run of record's
    envelope at the fixed slots (8 slots, 4 at each of buckets 0 and 16, EVERY 16; 1,024 prompt ids, 64 outputs); a
    given class is read as SLOTS,PROMPT,OUTPUTS[,LANES,EVERY], every slot at one boundary without LANES."""
    import prove_path as PP

    assert PP.window_class_of(None) == CLASS_OF_RECORD == WindowClass(8, 1024, 64, lanes=4, every=16)
    assert PP.window_class_of("8,1024,64,4,16") == CLASS_OF_RECORD
    assert PP.window_class_of("1,2,2") == PROBE and PROBE.boundaries == (0,)
