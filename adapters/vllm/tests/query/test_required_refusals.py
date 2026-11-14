"""What `query.required` refuses or reports of a hand-built request Program, each beside the Program it takes.

A `Population` refuses a query it does not know and a Program with no sampling event.  `request_manifest` lists as
`unmodelled`, and so calls incomplete, a manifest whose Program has bodies sharing a parameter that not exactly one of
them owns, a required member of no width, a member two Definitions produce, or a rank Program's collective with a
rank-local operand it cannot name.  A Value of no single width is spelt `mixed`, and a warm Program's cached rows are no
token ids.
"""
from __future__ import annotations

from typing import Any

import pytest

from tests.query.test_module_body import W, _call
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.query.program_view import CallView, InputView, Port, ProgramView
from verity_vllm.query.required import (
    Q_MODULE_BODY_ID,
    Q_WORD_ID,
    Population,
    cached_prefix_inputs,
    request_manifest,
    shared_instances,
)
from verity_vllm.query.units import ValueRef

R = ValueRef
N = 4
#: the implementation's name of the Program module `W`: the wrapper holds the served model under one more `model`
IMPL = "model"
EMBED_W, A_W, B_W, TIED_W = f"{W}.embed.weight", f"{W}.a.weight", f"{W}.b.weight", f"{W}.tied.weight"
IDS = InputView(-2, "ids", "Value<32>", (Port("ids", 0, 1, 1, 32),))
RES: dict[str, Any] = {"program": {"digest": "d0", "function": "f"}, "target": {"declared": {"compute_capability": [9, 0]}}}


def _weights(*names: str, width: int | None = 16) -> InputView:
    return InputView(-1, "weights", "{...}", tuple(Port(n, N * i, N * (i + 1), N, width) for i, n in enumerate(names)))


def _program(weights: InputView, middle: tuple[CallView, ...], *more: InputView, select: str = "ArgmaxTokenSelect_v1") -> ProgramView:
    """The embedding, then `middle` (Calls 1 on), then a Call of `select` over the last of them, which the Program returns."""
    last = len(middle) + 1
    calls = (_call(0, "Gather_v1", f"{W}.embed/embedding/out[0]", [R(-2, "ids"), R(-1, EMBED_W)]), *middle,
             _call(last, select, "argmax[0]", [R(last - 1, middle[-1].port_table[0].name)], n=1))
    return ProgramView(calls, (weights, IDS, *more), (R(last, "out"),), "synthetic", "synthetic")


def _two_bodies(*read: str) -> ProgramView:
    """Two module bodies, `a` and `b`, of one Call each of one Definition, both reading the parameters `read`."""
    return _program(_weights(EMBED_W, *read),
                    (_call(1, "Gemm_v1", f"{W}.a/linear[0]", [R(0, "out"), *(R(-1, f) for f in read)]),
                     _call(2, "Gemm_v1", f"{W}.b/linear[0]", [R(1, "out"), *(R(-1, f) for f in read)])))


def _corr(P: ProgramView) -> Correspondence:
    """`P`'s correspondence where the implementation's paths are the Program's less the wrapper's prefix."""
    corr = Correspondence(P, implementation_paths=(f"{IMPL}.embed",))
    assert corr.prefix == "model." and corr.implementation_path(f"{W}.embed") == f"{IMPL}.embed"
    return corr


def _manifest(P: ProgramView, res: dict[str, Any] = RES, **kw: Any) -> dict[str, Any]:
    return request_manifest(P, _corr(P), res, request_id="r0", **kw).manifest


def _rows(m: dict[str, Any], family: str) -> list[dict[str, Any]]:
    return [x for x in m["identities"] if x["family"] == family]


def test_a_population_refuses_a_query_it_does_not_know_and_a_program_with_no_sampling_event():
    P = _two_bodies(A_W)
    assert {Population(P, Correspondence(P), query=q).query for q in (Q_WORD_ID, Q_MODULE_BODY_ID)} == {Q_WORD_ID, Q_MODULE_BODY_ID}
    with pytest.raises(ValueError, match=r"^unknown query 'Q_other': one of \("):
        Population(P, Correspondence(P), query="Q_other")
    unsampled = _program(_weights(EMBED_W, A_W), (_call(1, "Gemm_v1", f"{W}.a/linear[0]", [R(0, "out"), R(-1, A_W)]),), select="Gemm_v1")
    with pytest.raises(ValueError, match=r"^Program has no sampling event: no step boundary$"):
        Population(unsampled, Correspondence(unsampled))


def test_a_population_prescribes_its_programs_record_fields_and_supplied_inputs_and_no_other_value():
    supplied = InputView(-3, "values", "Array<4,Value<16>>", (Port("values", 0, N, N, 16),))
    prefix = InputView(-4, "prefix", "Array<1,Array<4,Value<16>>>", (Port("prefix", 0, N, N, 16),))
    P = _program(_weights(EMBED_W, A_W),
                 (_call(1, "Gemm_v1", f"{W}.a/linear[0]", [R(0, "out"), R(-1, A_W), R(-3, "values"), R(-4, "prefix")]),), supplied, prefix)
    for q in (Q_WORD_ID, Q_MODULE_BODY_ID):
        policy = Population(P, Correspondence(P), query=q).policy
        assert policy.prescribed_inputs == {R(-1, EMBED_W), R(-1, A_W), R(-3, "values")}, "the prompt and the cached rows are not"
        assert policy.prescribed_inputs < set(P.inputs)


@pytest.mark.parametrize(("read", "owners"), [((TIED_W,), 0), ((A_W, B_W), 2)])
def test_bodies_sharing_parameters_that_not_one_of_them_owns_are_not_aliased_and_the_manifest_is_incomplete(read, owners):
    owned = _two_bodies(A_W)
    assert shared_instances(owned, Correspondence(owned), {R(-1, A_W): A_W}) == ({f"{W}.a": (f"{W}.a", 0), f"{W}.b": (f"{W}.a", 1)}, [])
    m = _manifest(owned)
    assert (m["complete"], m["unmodelled"], m["op_path_aliases"]) == (True, {}, {f"{IMPL}.b": [f"{IMPL}.a", 1]})

    P = _two_bodies(*read)
    fault = f"shared parameters {sorted(read)} read by 2 identical bodies with {owners} registered owner(s): not aliased"
    assert shared_instances(P, Correspondence(P), {R(-1, f): f for f in read}) == ({}, [fault])
    m = _manifest(P)
    assert (m["complete"], m["unmodelled"], m["op_path_aliases"]) == (False, {fault: 1}, {})


def test_a_value_and_a_weight_of_no_single_width_are_spelt_mixed():
    out = CallView(1, "Gemm_v1{N=4}", "Gemm_v1", "call", f"{W}.a/linear[0]", (Port("out", 0, N, N, None),), (R(0, "out"), R(-1, A_W)))
    m = _manifest(_program(_weights(EMBED_W, A_W, width=None), (out,)))
    assert m["complete"]
    assert [(x["op_path"], x["dtype"]) for x in _rows(m, "weights")] == [(f"{IMPL}.a.weight", "mixed"), (f"{IMPL}.embed.weight", "mixed")]
    assert [(x["op_path"], x["dtype"]) for x in _rows(m, "instance_outputs")] == [(f"{IMPL}.a", "mixed"), (f"{IMPL}.embed", "bf16")]


def test_a_required_member_of_no_width_is_unmodelled():
    P = _program(_weights(EMBED_W, A_W), (_call(1, "Gemm_v1", f"{W}.a/linear[0]", [R(0, "out"), R(-1, A_W)], n=0),))
    m = _manifest(P)
    assert (m["complete"], m["unmodelled"]) == (False, {f"Gemm_v1 under {IMPL}.a (out: width 0)": 1})
    assert [x["op_path"] for x in _rows(m, "instance_outputs")] == [f"{IMPL}.embed"]


def test_one_member_that_calls_of_two_definitions_produce_is_unmodelled_and_keeps_the_first():
    def middle(second: str) -> tuple[CallView, ...]:
        return (_call(1, "Gemm_v1", f"{W}.a/node[0]", [R(0, "out"), R(-1, A_W)]), _call(2, second, f"{W}.a/node[1]", [R(1, "out")]))

    one = _manifest(_program(_weights(EMBED_W, A_W), middle("Gemm_v1")))
    assert one["complete"] and [(x["spec"], x["rows"]) for x in _rows(one, "instance_outputs") if x["op_path"] == f"{IMPL}.a"] == [("Gemm_v1", 2)]
    m = _manifest(_program(_weights(EMBED_W, A_W), middle("Mul_v1")))
    assert (m["complete"], m["unmodelled"]) == (False, {f"Mul_v1 under {IMPL}.a (out: two producers Gemm_v1 4 / Mul_v1 4)": 1})
    assert [(x["spec"], x["rows"], x["position"]) for x in _rows(m, "instance_outputs") if x["op_path"] == f"{IMPL}.a"] == [("Gemm_v1", 1, [1, 0])]


def test_a_warm_programs_cached_rows_are_no_token_ids():
    prefix = InputView(-3, "prefix", "Array<1,Array<4,Value<16>>>", (Port("prefix", 0, N, N, 16),))
    P = _program(_weights(EMBED_W, A_W), (_call(1, "Gemm_v1", f"{W}.a/linear[0]", [R(0, "out"), R(-1, A_W), R(-3, "prefix")]),), prefix)

    def token_ids(res: dict[str, Any]) -> list[str]:
        return [x["op_path"] for x in _rows(_manifest(P, res), "input_token_ids")]

    assert cached_prefix_inputs(P, RES) == cached_prefix_inputs(P, {**RES, "prefix": None}) == frozenset()
    assert token_ids(RES) == [f"{IMPL}.a", f"{IMPL}.embed"], "a Program whose derive names no prefix reads the input as the prompt's"
    assert cached_prefix_inputs(P, {**RES, "prefix": {"input": 2}}) == {R(-3, "prefix")}
    assert token_ids({**RES, "prefix": {"input": 2}}) == [f"{IMPL}.embed"]
    assert cached_prefix_inputs(P, {**RES, "prefix": {"input": 1}}) == {R(-2, "ids")}, "the wrapper's input index, not a name"


WORLD = 2
TP: dict[str, Any] = {"world": WORLD, "rank": 0}


def _rank_program(collectives: tuple[CallView, ...], peers: int) -> ProgramView:
    """Rank 0 of two: an embedding shard, then `collectives` (Calls 1 on), which read it and the `peers` received parts."""
    received = tuple(InputView(-3 - k, f"peers_{k}", f"Array<1,Array<{N},Value<16>>>", (Port(f"peers_{k}", 0, N, N, 16),)) for k in range(peers))
    last = len(collectives) + 1
    calls = (_call(0, "EmbeddingShard_v1", f"{W}.embed/embedding_shard[0]/shard", [R(-2, "ids"), R(-1, EMBED_W)]), *collectives,
             _call(last, "ArgmaxTokenSelect_v1", "argmax[0]", [R(last - 1, collectives[-1].port_table[0].name)], n=1))
    return ProgramView(calls, (_weights(EMBED_W), IDS, *received), (R(last, "out"),), "synthetic", "synthetic")


def _rank_manifest(P: ProgramView) -> dict[str, Any]:
    return _manifest(P, {**RES, "tp": TP}, rank=0, ranks=WORLD)


def test_one_value_is_the_rank_local_partial_of_one_collective_only():
    def reduce(cid: int, node: str) -> CallView:
        return _call(cid, "AllReduceMultimem_v1", f"{W}.embed/{node}/all_reduce", [R(0, "out"), R(-2 - cid, f"peers_{cid - 1}")])

    P = _rank_program((reduce(1, "all_reduce"), reduce(2, "all_reduce_1")), 2)
    pop = Population(P, _corr(P), tp=TP)
    fault = f"EmbeddingShard_v1 under {IMPL}.embed: one Value is the rank-local partial of two collectives"
    assert pop.tp_faults == [fault]
    assert (pop.collectives, pop.partial_of, pop.carries) == ({1: "all_reduce", 2: "all_reduce"}, {R(0, "out"): 1}, set())
    assert pop.operand == {1: R(0, "out"), 2: R(0, "out")}
    m = _rank_manifest(P)
    assert (m["complete"], m["unmodelled"]) == (False, {fault: 1})
    assert [x["output_member"] for x in _rows(m, "tp_rank_partials")] == ["part_rank0"]

    alone = _rank_program((reduce(1, "all_reduce"),), 1)
    assert Population(alone, _corr(alone), tp=TP).tp_faults == [] and _rank_manifest(alone)["complete"]


@pytest.mark.parametrize(("family", "good", "bad", "shown"), [
    ("AllReduceMultimem_v1", (Port("0", 0, N, N, 16), Port("1", 0, N, N, 16)), (Port("0", 0, N, N, 16), Port("1", 0, 2 * N, 2 * N, 16)),
     "[('0', 4), ('1', 8)]"),
    ("AllGather2_v1", (Port("out", 0, 2 * N, 2 * N, 16),), (Port("out", 0, 3, 3, 16),), "[('out', 3)]"),
    ("AllGather2_v1", (Port("out", 0, 2 * N, 2 * N, 16),), (Port("0", 0, N, N, 16), Port("1", 0, N, N, 16)), "[('0', 4), ('1', 4)]"),
])
def test_a_collective_that_reads_another_bodys_value_has_outputs_its_partial_can_be_cut_from(family, good, bad, shown):
    def program(ports: tuple[Port, ...]) -> ProgramView:
        return _rank_program((CallView(1, f"{family}{{N={N}}}", family, "call", f"{W}.site/collective/collective", ports,
                                       (R(0, "out"), R(-3, "peers_0"))),), 1)

    P = program(good)
    carried = Population(P, _corr(P), tp=TP)
    assert (carried.tp_faults, carried.carries, carried.operand) == ([], {1}, {1: R(0, "out")})

    P = program(bad)
    pop = Population(P, _corr(P), tp=TP)
    fault = (f"{family} under {IMPL}.site: an operand from another body and output ports {shown} (all_reduce: outputs of one width; "
             f"all_gather: one output, a multiple of world {WORLD}, expected)")
    assert (pop.tp_faults, pop.carries, pop.operand, pop.site) == ([fault], set(), {1: R(0, "out")}, {})
    m = _rank_manifest(P)
    assert (m["complete"], m["unmodelled"]) == (False, {fault: 1})
    assert _rows(m, "tp_rank_partials") == []
