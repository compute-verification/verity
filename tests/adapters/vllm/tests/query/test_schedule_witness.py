"""A Program's `schedule` input (`format.SCHEDULE_PARAM`) is neither token ids nor an identity of the manifest, and nothing commits
or binds it: every Call that reads it is refused by name."""
from __future__ import annotations

import pytest

from tests.query.test_module_body import W, _call
from tests.query.test_tp_moe_members import tp_moe_program
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.query.program_view import InputView, Port, ProgramView
from verity_vllm.query.required import QUERIES, request_manifest
from verity_vllm.query.units import ValueRef

EMBED_W, GATE_W, W13, LM_W = f"{W}.embed.weight", f"{W}.layers.0.ffn.gate.weight", f"{W}.layers.0.ffn.experts.w13_weight", f"{W}.lm_head.weight"
EXPERTS = f"{W}.layers.0.ffn.experts"


def aligned_program(reader: str = "MoeAlignRanks_v1") -> ProgramView:
    """A world-2 rank Program whose experts body ranks its pairs by the `schedule` witness (MoeAlignRanks -> MoeAlignBlocks), runs the
    expert Gemm on the schedule, and all-reduces it with the received peer partial."""
    weights = InputView(-1, "weights", "{...}", tuple(Port(n, 8 * i, 8, 8, 16) for i, n in enumerate((EMBED_W, GATE_W, W13, LM_W))))
    ids = InputView(-2, "prompt", "Value<32>", (Port("prompt", 0, 1, 1, 32),))
    sched = InputView(-3, "schedule", "Array<4,Value<32>>", (Port("schedule", 0, 4, 4, 32),))
    peers = InputView(-4, "peers_0", "Array<1,Array<4,Value<16>>>", (Port("peers_0", 0, 4, 4, 16),))
    R = ValueRef
    calls = (
        _call(0, "Gather_v1", f"{W}.embed/embedding/out[0]", [R(-2, "prompt"), R(-1, EMBED_W)]),
        _call(1, "Gemm_v1", f"{W}.layers.0.ffn.gate/linear[0]", [R(0, "out"), R(-1, GATE_W)]),
        _call(2, reader, f"{EXPERTS}/marlin_moe_align/rank[0]", [R(1, "out"), R(-3, "schedule")], ports=("0", "1")),
        _call(3, "MoeAlignBlocks_v1", f"{EXPERTS}/marlin_moe_align[0]", [R(1, "out"), R(2, "0"), R(2, "1")], ports=("0", "1")),
        _call(4, "MarlinMoeExpertGemm_v1", f"{EXPERTS}/marlin_moe_mxfp4_gemm[0]", [R(0, "out"), R(1, "out"), R(3, "0"), R(3, "1"), R(-1, W13)]),
        _call(5, "AllReduce2_v1", f"{EXPERTS}/all_reduce[0]/all_reduce", [R(4, "out"), R(-4, "peers_0")]),
        _call(6, "Add_v1", f"{W}/add[0]", [R(0, "out"), R(5, "out")]),
        _call(7, "Gemm_v1", f"{W}.lm_head/linear[0]", [R(6, "out"), R(-1, LM_W)]),
        _call(8, "ArgmaxTokenSelect_v1", "argmax[0]", [R(7, "out")], n=1),
    )
    return ProgramView(calls, (weights, ids, sched, peers), (ValueRef(8, "out"),), "synthetic", "synthetic")


def _manifest(P: ProgramView, rank: int, query: str) -> dict:
    res = {"program": {"digest": f"d{rank}", "function": "f"}, "target": {"declared": {"compute_capability": [9, 0]}}, "tp": {"world": 2, "rank": rank}}
    return request_manifest(P, Correspondence(P), res, request_id="r0", rank=rank, ranks=2, query=query).manifest


@pytest.mark.parametrize("reader", ["MoeAlignRanks_v1", "MoeAlignRanksByIndex_v1"])
def test_a_reader_of_the_schedule_is_refused_by_name(reader):
    m = _manifest(aligned_program(reader=reader), 0, QUERIES[0])
    assert not m["complete"]
    assert [k for k in m["unmodelled"] if "reads the `schedule` witness" in k and reader in k], m["unmodelled"]


def test_a_program_without_the_witness_states_none():
    P = tp_moe_program()
    res = {"program": {"digest": "d0", "function": "f"}, "target": {"declared": {"compute_capability": [8, 9]}}, "tp": {"world": 2, "rank": 0}}
    m = request_manifest(P, Correspondence(P), res, request_id="r0", rank=0, ranks=2).manifest
    assert "witness_inputs" not in m["query"]["tp"]
