"""window_cost (VLLM-128): the class-of-record window's commit priced from a class_rows census.

* TINY's census rebuilt through `split` and `window_units` is class_rows' total in the census model, and its private bodies
  are the census's bytes and a salt a row;
* a probe of one Program projected to a class is that class's own census, unit for unit and byte for byte;
* the attention at key blocks of 4 over 6 positions is the hand count, and in 2-key groups (`L3a`) too;
* at the class of record, every lever leaves the commit's compressions no larger and every step of the plan lowers them,
  and the plan from ``L1 + L2`` on leaves no pad pass at any traffic;
* the H200's AND tables are `BS.compile`'s counts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from verity_vllm.program.model import INT8_EXACT_QUANT
from verity_vllm.program.predict.phi import Phi
from verity_vllm.program.predict.steps import Registered
from verity_vllm.program.predict.window_circuit import Grain, WindowClass, class_circuit

from verity.ml.windows import interface as IF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import class_rows as CR
import window_cost as W

#: integrations/vllm's `tests.program.predict.test_steps.TINY` and its grain in test_window_circuit (test_class_rows')
TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128,
        "num_attention_heads": 4, "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256, "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True, "max_position_embeddings": 64, "hidden_act": "silu"}
GRAIN = Grain(gemm_words=512, rows=9, select=64)
LLAMA_1B = Path(__file__).resolve().parents[3] / "integrations/vllm/tests/program/predict/fixtures/llama32-1b.config.json"


def _census(cls: WindowClass, probe: int | None = None) -> dict:
    phi = Phi(model={"repo": "org/m", "revision": "r", "hf_config": TINY}, target={"compute_capability": [12, 0], "num_sms": 188})
    doc = CR.census(class_circuit(Registered.of(phi, default_seed=7), cls, probe, GRAIN), IF)
    doc["class"] = {"slots": cls.slots, "prompt": cls.prompt, "outputs": cls.outputs, "positions": cls.positions}
    return doc


def _shapes(units):
    return sorted((u.label, u.n, u.wired, u.registered, u.other, u.out, u.out_bytes, u.x, u.wired_bytes, u.registered_bytes,
                   u.other_bytes) for u in units)


def test_tinys_census_rebuilt_is_class_rows_total_and_its_private_bodies_its_bytes_and_salts():
    doc = _census(WindowClass(2, 4, 2))
    rest, heads = W.split(W.units_of(doc), doc["class"], "sm120")
    us = W.window_units(rest, heads, frozenset())
    assert sum(u.compressions(W.CENSUS) for u in us) == doc["total"]["compressions"]
    t = doc["total"]
    rows = sum(v for k, v in t.items() if k.startswith("rows."))
    assert W.Registration.of(us, False).private_bytes == sum(v for k, v in t.items() if k.startswith("bytes.")) + \
        rows * W.SALT_BYTES
    assert sum(u.n * (u.ins + u.out) for u in us) == rows


def test_a_probe_of_one_program_projected_to_a_class_is_that_class_census():
    probe, exact = _census(WindowClass(1, 2, 2), 1), _census(WindowClass(2, 4, 2))
    rest, heads, cls = W.project(probe, 2, 4, 2, "sm120")
    rest2, heads2 = W.split(W.units_of(exact), exact["class"], "sm120")
    assert cls == exact["class"] and heads == heads2 and _shapes(rest) == _shapes(rest2)


def _heads() -> W.Heads:
    first = W.Unit("", 1, wired=3, out=3, out_bytes=276, x=8, wired_bytes=128)
    full = W.Unit("", 1, wired=6, out=3, out_bytes=276, x=8, wired_bytes=404)
    end = W.Unit("AttnHeadEnd_v1{D=8}", 1, wired=2, out=1, out_bytes=128, x=3, wired_bytes=272)
    return W.Heads(S=1, P=6, H=1, G=2, D=8, BN=4, block="AttnBlock_v5", first=first, full=full, end=end)


def test_the_attention_at_blocks_of_4_over_6_positions_is_the_hand_count():
    got = {u.label: (u.n, u.wired, u.wired_bytes) for u in _heads().units()}
    first = "AttnBlockFirst_v1{D=8,NVIS=%d,BN=4}"
    assert got == {"AttnHeadEnd_v1{D=8}": (6, 2, 272),
                   first % 1: (2, 3, 160), first % 2: (2, 3, 192), first % 3: (1, 3, 224), first % 4: (1, 3, 256),
                   "AttnBlock_v5{D=8,NVIS=4,FIRST=false,BN=4,CHECK=false}": (2, 6, 532),
                   "AllGather_v1{WORLD=2,N=8}": (4, 2, 32), "AllGather_v1{WORLD=3,N=8}": (2, 3, 48),
                   "AllGather_v1{WORLD=4,N=8}": (2, 4, 64)}
    grouped = {u.label: (u.n, u.wired) for u in _heads().units(group=2)}
    assert grouped[first % 3] == (1, 5) and grouped[first % 2] == (2, 3)
    assert grouped["AllGather_v1{WORLD=2,N=8}"] == (6, 2) and grouped["AllGather_v1{WORLD=4,N=8}"] == (2, 2)
    assert "AllGather_v1{WORLD=3,N=8}" not in grouped


@pytest.mark.slow
def test_at_the_class_of_record_every_lever_lowers_the_commit_and_the_plan_leaves_no_pad_pass():
    """Lowers int8 Llama-3.2-1B's probe for the H200 (about 5 s), as `class_rows.py --llama1b --int8 --probe 1,2,2`."""
    cfg = json.loads(LLAMA_1B.read_text()) | {"quantization_config": dict(INT8_EXACT_QUANT)}
    phi = Phi(model={"repo": "org/m", "revision": "r", "hf_config": cfg},
              target={"compute_capability": [9, 0], "num_sms": 132, "int8_exact": "linears"})
    probe = CR.census(class_circuit(Registered.of(phi, default_seed=7), WindowClass(1, 2, 2), 1), IF)
    probe["class"] = {"slots": 1, "prompt": 2, "outputs": 2, "positions": 3}
    doc = W.price(probe, "h200", to=(8, 1024, 64))
    by = {r["levers"]: r for r in doc["rows"]}
    base = by["baseline (engine)"]["commit_compressions"]
    assert all(r["commit_compressions"] <= base for r in doc["rows"] if r["costs"] == "engine")
    plan = [r for r in doc["rows"] if r["levers"].startswith("plan: ")]
    assert len(plan) == len(W.PLAN) - 1
    assert all(b["commit_compressions"] < a["commit_compressions"] for a, b in zip(plan, plan[1:]))
    assert all(p["steps"] == 0 for r in plan for p in r["pad_pass"].values())
    assert by["baseline (engine)"]["pad_pass"][W.TRAFFIC[0].name]["steps"] == 1087 - W.EXACT_KEYS


def _ands(t) -> int:
    from verity.ml.circuits.evaluation import bits as BS
    return BS.compile(t).counts["and"]


def test_the_h200s_small_and_counts_are_bs_compiles():
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity.ml.definitions.registry import boolean_attention as BA
    assert _ands(bind(BG.ScaledMmS8Coordinate, K=2048, **BG.GEMM_SM90_S8)) == W.S8_COORD[2048]
    assert _ands(bind(BG.GemmCoordinate, K=2048, DOT=BG.HopperBF16WgmmaDot16)) == W.BF16_COORD[2048]
    for nvis in (1, 17):
        assert _ands(bind(BA.AttnBlockV9, D=64, NVIS=nvis, FIRST=True, BN=192, DOT=BG.HopperBF16WgmmaDot16,
                          CHECK=True)) == W.FA3_FIRST[nvis - 1]


@pytest.mark.slow
def test_the_h200s_large_and_counts_are_bs_compiles():
    """The quantizer at K 2,048, FA3's full blocks at 128 and 192 keys and its last first block: each 8 to 12 million ANDs,
    tens of seconds and a few GB a compile, and their counts move apart from the small blocks' (the online softmax over a
    block, the quantizer's row max), which the quick test can't see."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity.ml.definitions.registry import boolean_attention as BA
    from verity.ml.definitions.registry import boolean_int8 as X
    assert _ands(bind(X.Int8RowQuantV2, K=2048)) == W.ROW_QUANT[2048]
    for bn, check in ((128, False), (192, True)):
        assert _ands(bind(BA.AttnBlockV9, D=64, NVIS=bn, FIRST=False, BN=bn, DOT=BG.HopperBF16WgmmaDot16,
                          CHECK=check)) == W.FA3_FULL[(bn, check)]
    assert _ands(bind(BA.AttnBlockV9, D=64, NVIS=192, FIRST=True, BN=192, DOT=BG.HopperBF16WgmmaDot16,
                      CHECK=True)) == W.FA3_FIRST[191]
