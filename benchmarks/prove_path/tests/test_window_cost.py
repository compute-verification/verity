"""window_cost: the class-of-record window's commit priced from a class_rows census.

* TINY's census rebuilt through `split` and `window_units` is class_rows' total in the census model, and its private bodies
  are the census's bytes and a salt a row;
* a probe of one Program projected to a class is that class's own census, unit for unit and byte for byte, at more outputs
  and fewer lanes too (a slot's units at their Definitions' rows there), and a probe's slot unit its Definition doesn't
  give is refused;
* the attention at key blocks of 4 over 6 positions is the hand count, each block reading its keys' and values' rows, and
  with a block's state in one row (`L3b`) too;
* at the class of record, every lever leaves the commit's compressions no larger and every step of the plan lowers them,
  and the plan from ``L1 + L2`` on leaves no pad pass at any traffic, once every template of Llama-3.2-1B's probe has a
  measured price: its probe lowers, first blocks `AttnBlockFirst_v2` priced as v1 and quantizers `Int8RowQuant_v3` as
  v1, but the scan off the class of record's lanes, `SiluMul_v2` at I 2,048 and the norm's add parts and tail have none
  yet;
* serving's stream commit is its stream's 4 KiB rows under both schedules, its FA3 interiors the hand count of each launch,
  and ``F`` leaves out exactly those;
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
GRAIN = Grain(gemm_words=1024, rows=256, select=64)
LLAMA_1B = Path(__file__).resolve().parents[3] / "integrations/vllm/tests/program/predict/fixtures/llama32-1b.config.json"
#: Llama-3.2-1B's arena sizing record (`step_bound_bytes`'s widths)
SIZING = Path(__file__).resolve().parents[3] / "integrations/vllm/tests/pipeline/fixtures/admission/b16.json"


def _census(cls: WindowClass, probe: int | None = None) -> dict:
    phi = Phi(model={"repo": "org/m", "revision": "r", "hf_config": TINY}, target={"compute_capability": [12, 0], "num_sms": 188})
    doc = CR.census(class_circuit(Registered.of(phi, default_seed=7), cls, probe, GRAIN, provable=False), IF)
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
    assert W.RegistrationSize.of(us, False).private_bytes == sum(v for k, v in t.items() if k.startswith("bytes.")) + \
        rows * W.SALT_BYTES
    assert sum(u.n * (u.ins + u.out) for u in us) == rows


@pytest.mark.parametrize("to", [WindowClass(2, 4, 2), WindowClass(2, 4, 3), WindowClass(2, 4, 2, 1, 4)])
def test_a_probe_of_one_program_projected_to_a_class_is_that_class_census(to):
    probe, exact = _census(WindowClass(1, 2, 2), 1), _census(to)
    rest, heads, cls = W.project(probe, to.slots, to.prompt, to.outputs, "sm120", to.lanes, to.every)
    rest2, heads2 = W.split(W.units_of(exact), exact["class"], "sm120")
    assert cls == exact["class"] and heads == heads2 and _shapes(rest) == _shapes(rest2)


def test_a_probe_unit_its_definition_doesnt_give_is_refused():
    probe = _census(WindowClass(1, 2, 2), 1)
    scan = next(lab for lab in probe["templates"] if lab.startswith("ScanSlots_v2{"))
    probe["templates"][scan]["bytes"]["value"] += 2
    with pytest.raises(ValueError, match=r"ScanSlots_v2\{.*\} is not its Definition's rows at the probe's class"):
        W.project(probe, 2, 4, 2, "sm120")


def _heads() -> W.Heads:
    first = W.Unit("", 1, wired=3, out=3, out_bytes=276, x=8, wired_bytes=128)
    full = W.Unit("", 1, wired=6, out=3, out_bytes=276, x=8, wired_bytes=404)
    end = W.Unit("AttnHeadEnd_v1{D=8}", 1, wired=2, out=1, out_bytes=128, x=3, wired_bytes=272)
    return W.Heads(S=1, P=6, H=1, R=2, D=8, BN=4, block="AttnBlock_v5", first=first, full=full, end=end)


def test_the_attention_at_blocks_of_4_over_6_positions_is_the_hand_count():
    got = {u.label: (u.n, u.wired, u.wired_bytes) for u in _heads().units()}
    first = "AttnBlockFirst_v1{D=8,NVIS=%d,BN=4}"
    full = "AttnBlock_v5{D=8,NVIS=4,FIRST=false,BN=4,CHECK=false}"
    assert got == {"AttnHeadEnd_v1{D=8}": (6, 2, 272),
                   first % 1: (2, 5, 160), first % 2: (2, 7, 192), first % 3: (1, 9, 224), first % 4: (1, 11, 256),
                   full: (2, 14, 532)}
    packed = {u.label: (u.out, u.wired) for u in _heads().units(packed=True)}
    assert (packed[first % 4], packed[full], packed["AttnHeadEnd_v1{D=8}"]) == ((1, 11), (1, 12), (1, 1))


@pytest.mark.parametrize("levers", [(), ("L3b",), ("W", "R"), ("L3b", "W", "R")])
def test_a_windows_position_layouts_are_its_output_rows_and_bytes(levers):
    doc = _census(WindowClass(2, 4, 2))
    rest, heads = W.split(W.units_of(doc), doc["class"], "sm120")
    us = W.window_units(rest, heads, frozenset(levers))
    rows = nbytes = 0
    for T in range(1, heads.P + 1):
        for g in W.position_layout(rest, heads, frozenset(levers), T)["groups"]:
            rows += heads.S * sum(len(u) for u in g["units"])
            nbytes += heads.S * g["sources"][0]
    assert (rows, nbytes) == (sum(u.n * u.out for u in us), sum(u.n * u.out_bytes for u in us))


@pytest.mark.slow
@pytest.mark.xfail(raises=ValueError, strict=True, reason="Llama-3.2-1B's probe lowers, and some of its templates have no "
                   "measured price on the H200")
def test_at_the_class_of_record_every_lever_lowers_the_commit_and_the_plan_leaves_no_pad_pass():
    """Lowers int8 Llama-3.2-1B's probe for the H200 (about 5 s), as `class_rows.py --llama1b --int8 --probe 1,2,2`."""
    cfg = json.loads(LLAMA_1B.read_text()) | {"quantization_config": dict(INT8_EXACT_QUANT)}
    phi = Phi(model={"repo": "org/m", "revision": "r", "hf_config": cfg},
              target={"compute_capability": [9, 0], "num_sms": 132, "int8_exact": "linears"})
    probe = CR.census(class_circuit(Registered.of(phi, default_seed=7), WindowClass(1, 2, 2), 1, provable=False), IF)
    probe["class"] = {"slots": 1, "prompt": 2, "outputs": 2, "positions": 3}
    doc = W.price(probe, "h200", to=(8, 1024, 64), sizing=json.loads(SIZING.read_text())["sizing"])
    assert all(0 < r["F"]["ands"] < r["all"]["ands"] for r in doc["stream"].values())
    assert "the plan + F" in W.report(doc)
    by = {r["levers"]: r for r in doc["rows"]}
    base = by["baseline (engine)"]["commit_compressions"]
    assert all(r["commit_compressions"] <= base for r in doc["rows"] if r["costs"] == "engine")
    plan = [r for r in doc["rows"] if r["levers"].startswith("plan: ")]
    assert len(plan) == len(W.PLAN) - 1
    assert all(b["commit_compressions"] < a["commit_compressions"] for a, b in zip(plan, plan[1:]))
    assert all(p["steps"] == 0 for r in plan for p in r["pad_pass"].values())
    assert by["baseline (engine)"]["pad_pass"][W.TRAFFIC[0].name]["steps"] == 1087 - W.EXACT_KEYS


def test_serving_streams_commit_is_its_4_kib_rows_and_F_leaves_out_its_fa3_interiors():
    sizing = json.loads(SIZING.read_text())["sizing"]
    one, anchors = W.stream(sizing, {"slots": 4, "prompt": 40, "outputs": 3}, "h200", 1e12, 1.0, lanes=2, every=2).values()

    def fa3(seqs: int, rows: int, BN: int) -> int:
        """One launch's FA3 M1 bytes over the layers, in whole 64-word chunks: a slab a (sequence, KV head) of rows · 4 query
        heads, one key block of BN keys, each row 2.5 BN words of scores and probabilities, 8 + D of the block's state and 4."""
        words = 8 * seqs * 4 * rows * (2 * BN + BN // 2 + 8 + 64 + 4)
        return sizing["model"]["layers"] * 4 * -(-words // 64) * 64

    # 40 ids · 4 heads a KV head > 64 rows: blocks of 128; a decode launch's 4 rows: blocks of 192
    assert one["hidden_bytes"] == fa3(4, 40, 128) + 2 * fa3(4, 1, 192)
    # 2 prompts at step 0 and 2 at step 2, whose launch pads the 2 decoding slabs to the 40-id prompts
    assert anchors["hidden_bytes"] == fa3(2, 40, 128) + fa3(2, 1, 192) + fa3(4, 40, 128) + 2 * fa3(2, 1, 192)
    for r in (one, anchors):
        assert r["all"]["compressions"] == -(-r["bytes"] // W.STREAM_ROW_BYTES) * W.STREAM_ROW
        assert r["F"]["compressions"] == -(-(r["bytes"] - r["hidden_bytes"]) // W.STREAM_ROW_BYTES) * W.STREAM_ROW
        assert r["all"]["seal_s"] - r["F"]["seal_s"] == pytest.approx(r["hidden_bytes"] / W.SEAL_BYTES_S)


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
