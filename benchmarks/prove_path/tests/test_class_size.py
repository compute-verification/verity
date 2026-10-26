"""class_size (V3b): a window class's padded size, projected from one Program lowered at a small probe class.

* the lowering projected from a probe of 3 positions is, template for template, TINY's exact lowering at a class of 7
  positions (`window_circuit.class_circuit`), at a class of two lanes (its scan state row another), and at TINY of 2,048
  positions at a class of 261, past two key blocks of 128: every template but the heads and their key gathers is a fixed
  count a position, `TokenFeed_v1` one a prompt position, `ScanSlots_v1`, `SlotRow_v1` and `SlotEntry_v1` one a Program and
  `OutCollect_v1` one a position;
* v19's heads (#4263), at a block of 3 keys: a head over T keys is its first block, ceil(T / 3) - 1 full ones (none
  guarded: a one-row decode's FA2 guards its first visited block only) and its end; its keys reach it through one gather a
  full block, shared, and its first block's own where that block is neither one key nor full;
* with ``VERITY_PROVE_PATH_CLASS_OF_RECORD=1``, the class of record at Llama-3.2-1B lowers exactly, within C-Flock's block,
  as the projection says;
* a probe that doesn't divide evenly is refused, and so is a template `size` has no measurement for; m is the log of the
  cells;
* a gather's k_log is its rows' (its parts and its output, an hm96 slot each), circuits' composed values at every W they
  composed, and a gather of 256 parts needs a block of 2^28 bits, which `size` refuses as C-Flock does;
* with no ``--class``, ``prove_path.py`` takes the class of record.
"""
from __future__ import annotations

import json
import math
import os
import resource
import sys
from pathlib import Path

import pytest
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict.phi import Phi
from verity_vllm.program.predict.steps import Registered
from verity_vllm.program.predict.window_circuit import CLASS_OF_RECORD, Grain, WindowClass, class_circuit

from verity.ml.windows import interface as IF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import class_size as CZ  # noqa: E402

#: integrations/vllm's `tests.program.predict.test_steps.TINY` and its grain in test_window_circuit
TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128,
        "num_attention_heads": 4, "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256, "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True, "max_position_embeddings": 64, "hidden_act": "silu"}
GRAIN = Grain(gemm_words=512, rows=9, select=64)
PROBE = WindowClass(1, 2, 2)
LLAMA_1B = Path(__file__).resolve().parents[3] / "integrations/vllm/tests/program/predict/fixtures/llama32-1b.config.json"


def _registered(config: dict) -> Registered:
    phi = Phi(model={"repo": "org/m", "revision": "r", "hf_config": config}, target={"compute_capability": [12, 0], "num_sms": 188})
    return Registered.of(phi, default_seed=7)


def _census(cls: WindowClass, config: dict = TINY, grain: Grain = GRAIN) -> dict[str, int]:
    return CZ.census(class_circuit(_registered(config), cls, 1, grain))


def test_the_lowering_projected_from_a_probe_is_the_exact_lowering_at_a_larger_class():
    probe, at, two = _census(PROBE), WindowClass(1, 5, 3), WindowClass(2, 5, 3)
    assert PROBE.positions == 3 and at.positions == 7
    assert CZ.project(probe, PROBE, at) == _census(at)
    assert CZ.project(probe, PROBE, two) == _census(two) and "ScanSlots_v1{NS=4550,NP=5,IDB=17,K0=12,K=45}" in _census(two)


@pytest.mark.slow
def test_the_projection_is_the_exact_lowering_past_two_key_blocks():
    """At 261 positions (TINY of 2,048 positions, a grain of 256 rows) the heads past 128 and 256 keys are a full block's
    Calls and their chain, and the full blocks' gathers are made once each; the projection from 3 positions is the exact
    lowering's census, template for template."""
    config, grain, at = {**TINY, "max_position_embeddings": 2048}, Grain(gemm_words=512, rows=256, select=64), WindowClass(1, 260, 2)
    exact = _census(at, config, grain)
    assert exact["AttnBlock_v5{D=16,NVIS=128,FIRST=false,BN=128,CHECK=false}"] == 8 * (128 + 2 * 5)
    assert exact["AllGather_v1{WORLD=128,N=16}"] == 2 * 8
    assert CZ.project(_census(PROBE, config, grain), PROBE, at) == exact


def test_v19s_heads_are_key_blocks_with_their_gathers():
    first = "AttnBlockFirst_v1{{D=4,NVIS={},BN=3}}"
    probe = {**{first.format(t): 4 for t in (1, 2, 3)}, "AttnHeadEnd_v1{D=4}": 12, "AllGather_v1{WORLD=2,N=4}": 2,
             "AllGather_v1{WORLD=3,N=4}": 2 + 3, "RMSNormTriton_v5{N=4}": 6, "TokenFeed_v1{NP=2,FIRST=true}": 1,
             "TokenFeed_v1{NP=2,FIRST=false}": 1}
    assert CZ.project(probe, PROBE, WindowClass(1, 5, 3)) == {
        "RMSNormTriton_v5{N=4}": 14, "TokenFeed_v1{NP=5,FIRST=true}": 1, "TokenFeed_v1{NP=5,FIRST=false}": 4,
        "AttnBlockFirst_v1{D=4,NVIS=1,BN=3}": 12, "AttnBlockFirst_v1{D=4,NVIS=2,BN=3}": 8, "AttnBlockFirst_v1{D=4,NVIS=3,BN=3}": 8,
        "AttnBlock_v5{D=4,NVIS=3,FIRST=false,BN=3,CHECK=false}": 20, "AttnHeadEnd_v1{D=4}": 28, "AllGather_v1{WORLD=2,N=4}": 4,
        "AllGather_v1{WORLD=3,N=4}": 7 + 4}
    with pytest.raises(ValueError, match="not a fixed count at each of its 3 positions"):
        CZ.project({**probe, "RMSNormTriton_v5{N=4}": 7}, PROBE, WindowClass(1, 5, 3))
    with pytest.raises(ValueError, match="not one count of first blocks at each T"):
        CZ.project({**probe, first.format(3): 5}, PROBE, WindowClass(1, 5, 3))
    with pytest.raises(ValueError, match=r"11 AttnHeadEnd_v1\{D=4\} units, not one a head"):
        CZ.project({**probe, "AttnHeadEnd_v1{D=4}": 11}, PROBE, WindowClass(1, 5, 3))


@pytest.mark.skipif(not os.environ.get("VERITY_PROVE_PATH_CLASS_OF_RECORD"),
                    reason="the class of record at Llama-3.2-1B: 368M units, some 22 minutes and 42 GB on a pod; set "
                           "VERITY_PROVE_PATH_CLASS_OF_RECORD=1")
def test_the_class_of_record_lowers_at_llama_3_2_1b(monkeypatch):
    """The class of record (8 slots of 1,087 positions) at Llama-3.2-1B's geometry and CIR-95's grain lowers whole: no unit
    past 256 rows in and out (`Grain.rows`), so none past C-Flock's 2^27-bit block (`size` refuses one), and its census is 8
    times the projection of one Program from a probe of 3 positions. Its numbers (nodes, units, wires, peak RSS, then ANDs
    and m) are printed as JSON lines."""
    reg = _registered(json.loads(LLAMA_1B.read_text()))
    built, program = WC.class_program(reg, CLASS_OF_RECORD), WC.class_program
    monkeypatch.setattr(WC, "class_program", lambda r, cls: built if cls == CLASS_OF_RECORD else program(r, cls))
    circuit = class_circuit(reg, CLASS_OF_RECORD)
    census = CZ.census(circuit)
    rows = max(sum(map(len, IF.unit_rows(fn, circuit.cuts.get(IF.descriptor_id(fn))))) for fn, _ in circuit.templates)
    print(json.dumps({"positions": CLASS_OF_RECORD.positions, "nodes": len(built.program.fn.body.nodes),
                      "units": sum(census.values()), "templates": len(circuit.templates), "wires": len(circuit.wires),
                      "reads": len(circuit.registered.reads), "entries": len(circuit.blocks.inference), "most_rows": rows,
                      "max_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1), "census": census},
                     sort_keys=True), flush=True)
    assert rows <= WC._GRAIN.rows
    s = CZ.size(census)
    print(json.dumps({"units": s["units"], "ands": s["ands"], "m": s["m"], "templates": s["templates"]}), flush=True)
    one = CZ.project(CZ.census(class_circuit(reg, PROBE, 1)), PROBE, CLASS_OF_RECORD)
    assert census == {lab: CLASS_OF_RECORD.slots * n for lab, n in one.items()}


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
    assert [CZ.measured(lab)[:2] for lab in ("ScanSlots_v1{NS=9353,NP=1024,IDB=17,K0=12,K=45}", "OutCollect_v1{KO=64,FIRST=true}",
                                             "OutCollect_v1{KO=64,FIRST=false}")] == [(0, 23), (155, 22), (2_141, 22)]


def test_window_circuit_and_class_size_default_to_the_class_of_record():
    """Without ``--class``, ``window-circuit`` and ``class-size`` take the class of record, the certifier run of record's
    envelope at ASK-122's fixed slots (8 slots, 4 at each of buckets 0 and 16, EVERY 16; 1,024 prompt ids, 64 outputs); a
    given class is read as SLOTS,PROMPT,OUTPUTS[,LANES,EVERY], every slot at one boundary without LANES."""
    import prove_path as PP

    assert PP.window_class_of(None) == CLASS_OF_RECORD == WindowClass(8, 1024, 64, lanes=4, every=16)
    assert PP.window_class_of("8,1024,64,4,16") == CLASS_OF_RECORD
    assert PP.window_class_of("1,2,2") == PROBE and PROBE.boundaries == (0,)
