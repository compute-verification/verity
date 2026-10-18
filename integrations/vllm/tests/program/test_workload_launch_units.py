"""The workload Program's launch units (`launch_units`, option 1): a MoE align launch shared by several requests runs once, at the
workload root, and each member request's `schedule` argument takes its pairs' blocks and the launch's block count from it.  Toy
request Programs stand in for the TP rank Programs: `schedule` is a request-local parameter whose linked words they read like any
other, so the workload's outputs show which words each request got."""

import gzip
import json
from dataclasses import replace

import numpy as np
import pytest

from verity.ml.circuits.codec import canonical_json, encode_program, program_digest
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, REGISTRY
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import coll_concat
from verity.ml.circuits.types import Array, Value
from verity_vllm.pipeline import components as C
from verity_vllm.program import workload as W
from verity_vllm.program.kernels.moe_align_rows import moe_align_blocks, moe_align_ranks
from verity_vllm.program.kernels.serve3_reference import materialize

_ADD = PrimitiveDefinition("GpTestAdd32", 1, (("a", Value(32)), ("b", Value(32))), Value(32), lambda a, b: (a + b) & 0xFFFFFFFF)
WEIGHTS, KEYS, IDS, KEY = [5, 6, 7, 8], [3, 1, 2, 0], [2, 0, 2, 1], [1, 0, 0, 2]


def _toy(LP: int, Wn: int):
    """`ToyScheduled{LP,W}(prompt: Array<LP,32>, schedule: Array<W,32>, weights: Array<4,32>) -> Array<W,32>`: each schedule word
    plus weights[0]."""
    def sig(S):
        return ((("prompt", Array(S.LP, Value(32))), ("schedule", Array(S.W, Value(32))), ("weights", Array(4, Value(32)))),
                Array(S.W, Value(32)))

    def body(B, S, prompt, schedule, weights):
        return coll_concat(Array(S.W, Value(32)), [B.call(_ADD, schedule[w], weights[0]) for w in range(S.W)])

    d = REGISTRY["ToyScheduled_v1"] if "ToyScheduled_v1" in REGISTRY else CompositeDefinition("ToyScheduled", 1, ("LP", "W"), sig, body)
    return d.bind(LP=LP, W=Wn)


def _component(LP, Wn, schedule):
    wp = [("prompt", f"Array<{LP},Value<32>>"), ("schedule", f"Array<{Wn},Value<32>>"), ("weights", "Array<4,Value<32>>")]
    return replace(W.request_component(encode_program(Program(_toy(LP, Wn))), REGISTRY, wp), schedule=tuple(schedule))


def _linked(k, rows, offset, step, L=2, BM=8):
    return {"k": k, "shape": [rows, 2], "pairs": rows * 2, "offset": offset, "form": "linked", "words": rows * 2 + 1, "step": step,
            "launch_tokens": L, "E": 4, "BM": BM}


def _wl(r1_prompt=1, **launch):
    """r0 (prompt 2, cap 2) prefills alone at engine step 0 (a keyed align over its 4 pairs) and decodes at step 1 beside r1's prefill
    (prompt 1, cap 1, arriving at step 1): one launch of 2 tokens x top-2."""
    r0 = _component(2, 7, [{"k": 0, "shape": [2, 2], "pairs": 4, "offset": 0}, _linked(1, 1, 4, 1, **launch)])
    r1 = _component(r1_prompt, 3, [_linked(0, 1, 0, 0, **launch)])
    return W.WorkloadProgram([W.WorkloadRequest("r0", 0, 2, 2, 0, r0), W.WorkloadRequest("r1", 1, r1_prompt, 1, 1, r1)],
                             manifest_digest="m" * 8, lag=0)


def _launch(ids, key):
    statics = {"E": 4, "P": len(ids), "BM": 8}
    rank = moe_align_ranks(statics, [np.array(ids), np.array(key), np.arange(len(ids))])
    out = moe_align_blocks(statics, [np.array(ids), rank["0"], rank["1"]])
    return [int(x) for x in out["0"]], int(out["1"][0])


def test_a_shared_launch_runs_once_and_feeds_each_member_its_slots_and_the_block_count():
    wl = _wl()
    assert wl.fn.defn is W.WORKLOAD_V2
    assert wl.launches == ({"e": 1, "j": 0, "E": 4, "BM": 8, "P": 4, "members": ((0, 1, 4, 0, 2), (1, 0, 0, 2, 2))},)
    params, _ = wl.fn.signature()
    # r0 keeps its keyed call's 4 words as its schedule input; r1's schedule is all linked words, so it has none
    assert [(n, t.n) for n, t in params] == [("shared.weights", 4), ("request[r0].prompt", 2), ("request[r0].schedule", 4),
                                             ("request[r1].prompt", 1), ("launch[e1.0].ids", 4), ("launch[e1.0].key", 4)]
    body = wl.program.root.body
    assert [n.fn.id for n in body.nodes[6:12]] == [f"Const32[0x0000000{p}]_v1" for p in range(4)] + [
        "MoeAlignRanks_v1{E=4,P=4,BM=8}", "MoeAlignBlocks_v1{P=4,BM=8}"]
    t = materialize(wl.program, [WEIGHTS, [1, 2], KEYS, [9], IDS, KEY])
    out = [t[g] for g in wl.program.circuit.root_outputs()]
    blk, par = _launch(IDS, KEY)
    assert out == [(v + WEIGHTS[0]) & 0xFFFFFFFF for v in KEYS + blk[:2] + [par] + blk[2:] + [par]]
    assert program_digest(encode_program(wl.program)) == wl.digest


def test_the_address_map_names_the_unit_and_the_population_counts_its_gates():
    wl = _wl()
    am = wl.address_map()
    assert [i["owner"] for i in am["inputs"]] == ["shared", "r0", "r0", "r1", "launch", "launch"]
    (unit,) = am["launches"]
    assert unit["root_nodes"] == [6, 12] and unit["members"] == [
        {"request_id": "r0", "align_call": 1, "word": 4, "slot": 0, "pairs": 2},
        {"request_id": "r1", "align_call": 0, "word": 0, "slot": 2, "pairs": 2}]
    r0, r1 = am["requests"]
    assert r0["root_node"] == 12 and r1["root_node"] == 13
    assert r0["inputs"][1]["global"] == "request[r0].schedule" and r0["inputs"][1]["linked"] == [{"word": 4, "unit": 0, "slot": 0, "pairs": 2}]
    assert r1["inputs"][1] == {"local_param": 1, "name": "schedule", "global": None, "linked": [{"word": 0, "unit": 0, "slot": 2, "pairs": 2}]}
    pop = wl.population()
    assert pop["sum_equals_total"] and pop["components_partition_non_input_gates"]
    assert pop["launch_gates"] == unit["gates"][1] - unit["gates"][0] > 0 and pop["other_root_nodes"] == 6


def test_a_launch_its_members_do_not_fill_or_disagree_on_is_refused():
    with pytest.raises(ValueError, match="do not fill the launch's 3 tokens"):
        _wl(L=3)
    with pytest.raises(ValueError, match="routes 2 pairs, not its declared 2 rows x 2"):
        _wl(r1_prompt=2)
    r0 = _component(2, 7, [{"k": 0, "shape": [2, 2], "pairs": 4, "offset": 0}, _linked(1, 1, 4, 1)])
    r1 = _component(1, 3, [_linked(0, 1, 0, 0, BM=16)])
    with pytest.raises(ValueError, match="disagree on the launch"):
        W.WorkloadProgram([W.WorkloadRequest("r0", 0, 2, 2, 0, r0), W.WorkloadRequest("r1", 1, 1, 1, 1, r1)], manifest_digest="m" * 8, lag=0)


def test_a_component_loads_with_the_align_calls_its_build_recorded(tmp_path):
    desc = encode_program(Program(_toy(1, 3)))
    with gzip.open(tmp_path / "descriptor.json.gz", "wb") as f:
        f.write(canonical_json(desc))
    with gzip.open(tmp_path / "instances.json.gz", "wt") as f:
        json.dump({"params": [["prompt", "Array<1,Value<32>>"], ["schedule", "Array<3,Value<32>>"], ["weights", "Array<4,Value<32>>"]],
                   "rows": []}, f)
    rec = [_linked(0, 1, 0, 0)]
    (tmp_path / "result.json").write_text(json.dumps({"ok": True, "program": {"digest": program_digest(desc)},
                                                      "tp": {"schedule": {"calls": 1, "pairs": 2, "record": rec}}}))
    assert C.load_component(str(tmp_path), {})["comp"].schedule == tuple(rec)


def test_a_workload_without_linked_calls_stays_workload_v1():
    c = _component(2, 4, [{"k": 0, "shape": [2, 2], "pairs": 4, "offset": 0}])
    wl = W.WorkloadProgram([W.WorkloadRequest("r0", 0, 2, 1, 0, c)], manifest_digest="m" * 8, lag=0)
    assert wl.fn.defn is W.WORKLOAD and wl.launches == () and "launches" not in wl.address_map()
    assert wl.digest == W.WorkloadProgram([W.WorkloadRequest("r0", 0, 2, 1, 0, replace(c, schedule=()))], manifest_digest="m" * 8,
                                          lag=0).digest
