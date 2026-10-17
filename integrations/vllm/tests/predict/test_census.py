"""The census (`verity.ml.circuits.census`) and the stacking property (`verity.ml.circuits.stacking`) on the Programs
`predict.build` emits for an RTX PRO 6000.

The pinned classes are the stacking campaign's counter's (count_budgets.py, branch cursor/stack-property-6cd3 at 4268176afa,
`classify(count(fn, taints))` on the same Programs; the comparison is art:8f73f34585b4f5b15ebd37bdbcaedc5251a4eca3675886da132b5a97e3a8bc75). The census charges two primitives the
counter left free: `F32Fabs_v1` as a comparison and `F32BitsShl23_v1` as an integer operation, so those two classes are
compared net of them. Today's bf16 Programs fail the property: every matmul is an FP tensor-core step."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity.ml.circuits import census as C
from verity.ml.circuits import stacking as P
from verity.ml.circuits.program import Program, is_input
from verity_vllm.predict.build import build_program, model_shape
from verity_vllm.predict.phi import Phi
from verity_vllm.program.frontend.target_profile import TargetProfile

CFG = Path(__file__).resolve().parents[2] / "data" / "hf_configs"
TP = TargetProfile.default().with_overrides(compute_capability=(12, 0), num_sms=188)

#: the counter's classes: {(config, wrapper, M, LP, T): classes}
COUNTER = {
    ("LLAMA32_1B", "request", 1, 3, 2): {
        "act": 665770, "act_ex2": 7680, "cast": 4621568, "cmp": 1161980, "f32_arith": 1573842, "gather": 10240,
        "gather_fanin": 1313341440, "int": 384765, "prod_rr": 4608005, "prod_rs": 5654155429, "prod_rsel": 0, "tc_acc_rr": 194560,
        "tc_acc_rs": 353337344, "tc_acc_rsel": 0},
    ("QWEN3_30B_A3B", "step", 1, None, None): {
        "act": 307778, "act_ex2": 7680, "cast": 3602304, "cmp": 914302, "f32_arith": 3701802, "gather": 1811941376,
        "gather_fanin": 232239398912, "int": 329727, "prod_rr": 5474369, "prod_rs": 1230609281, "prod_rsel": 1811939328,
        "tc_acc_rr": 208896, "tc_acc_rs": 76857344, "tc_acc_rsel": 113246208},
}


def counter_classes(t: dict) -> dict:
    """A census's counts in the counter's classes."""
    p = t["primitives"]
    return {"act": t["roundings"]["special"], "act_ex2": p.get("MufuEx2Ftz_v1", 0), "cast": t["roundings"]["cast"],
            "cmp": t["compare_select"] - p.get("F32Fabs_v1", 0), "f32_arith": t["roundings"]["arith"], "gather": t["gathers"],
            "gather_fanin": t["gather_fanin"], "int": t["roundings"]["integer"] - p.get("F32BitsShl23_v1", 0),
            **{"prod_" + k: v for k, v in t["products"].items()}, **{"tc_acc_" + k: v for k, v in t["mma_acc"].items()}}


def forward_starts(program: Program) -> list[int]:
    """Each forward's first Call: the first of a run of Embedding Calls."""
    calls = [nd for nd in program.root.body.nodes if not is_input(nd.fn)]
    starts, prev = [], False
    for i, nd in enumerate(calls):
        emb = nd.fn.id.startswith("Embedding")
        if emb and not prev:
            starts.append(i)
        prev = emb
    return starts


def _census(config: str, wrapper: str, M: int, LP, T):
    hf = json.loads((CFG / f"{config}.config.json").read_text())
    phi = Phi(model={"repo": "-", "revision": "-", "hf_config": hf}, target=TP.to_json())
    program = Program(build_program(phi, wrapper, M=M, LP=LP, T=T, with_desc=False).program.fn, registered=("weights",))
    return hf, C.census(program, forward_starts(program) if wrapper == "request" else None)


@pytest.mark.parametrize("key", list(COUNTER))
def test_the_census_equals_the_counter(key):
    _hf, cen = _census(*key)
    assert counter_classes(cen.total) == COUNTER[key]
    assert not cen.total["unclassified"]


def test_a_requests_steps_and_memory():
    """Llama-3.2-1B, 3 prompt tokens then 2 decode steps: the prompt's KV lives 2 steps, the first decode token's 1, each sampled
    token (32 bits) 1."""
    hf, cen = _census("LLAMA32_1B", "request", 1, 3, 2)
    s = model_shape(hf)
    kv = s.NL * 2 * s.KVH * s.D * 16
    assert kv == 262_144
    mem = [st["memory"] for st in cen.steps]
    assert [st["calls"] for st in cen.steps] == [488, 164, 164]
    assert [m["lifetimes"] for m in mem] == [{"1": 32, "2": 3 * kv}, {"1": kv + 32}, {}]
    assert [m["inputs"]["run_bits"] for m in mem] == [96, 0, 0]
    assert cen.memory["inputs"]["registered_bits"] == 19_773_035_520
    assert cen.memory["max_lifetime"] == 2 and cen.memory["held_input_bits"] == 0
    assert [st["macs"]["other"] for st in cen.steps] == [3_183_673_344, 1_236_402_176, 1_236_434_944]


def test_bf16_programs_fail_the_property():
    _hf, cen = _census("LLAMA32_1B", "request", 1, 3, 2)
    windows = [P.Window(range(3), 10 ** 12)]
    v = P.check(cen, c=1, R_in=12, L=2, windows=windows)
    assert [r["code"] for r in v.reasons] == ["fp-mma"]
    assert v.reasons[0]["steps"] == {"HopperBF16WgmmaDot16_v1": 353_531_904}
    assert v.numbers["macs"] == {"exact_s8": 0, "other": 5_656_510_464}
    strict = P.check(cen, c=1, R_in=11, L=1, windows=[P.Window(range(3), 1_000_000)])
    assert [r["code"] for r in strict.reasons] == ["fp-mma", "roundings", "inputs", "lifetime"]
    assert strict.reasons[1]["roundings"] == P.roundings(cen.total) == 8_418_165
