"""What `query.required` refuses or reports of a hand-built request Program, each beside the Program it takes.

A `Population` refuses a query it does not know and a Program with no sampling event.  `request_manifest` lists as
`unmodelled`, and so calls incomplete, a manifest whose Program has bodies sharing a parameter that not exactly one of
them owns, a required member of no width, a member two Definitions produce, or a rank Program's collective with a
rank-local operand it cannot name.  A Value of no single width is spelt `mixed`, and a warm Program's cached rows are no
token ids.  Of what a `Population` reports: a literal is no boundary Value, a returned Value is one unless it is a check, and
inside a MoE module with router rows only the Values of a plane's kind feed the planes.  Of a rank Program: which inputs are
the peers' parts, which operands of a collective are its own, where a carried collective is, and the step of a Call after
the last sampling event.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from verity.circuits.types import Array, RecordType, Type, Value
from verity.circuits.types import Tuple as TupleT

from tests.query.test_module_body import W, _call
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.query.program_view import (
    CallView,
    InputView,
    Port,
    ProgramView,
    ports_of_type,
)
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


def test_a_call_that_may_not_read_the_schedule_witness_is_refused_under_the_implementations_path():
    """The `schedule` parameter is the witness under either query, and a Call of another family that reads it is a fault that
    names the family and the module by the implementation's path.  The manifest is then incomplete, and the read is no token id."""
    sched = InputView(-3, "schedule", "Array<4,Value<32>>", (Port("schedule", 0, 4, 4, 32),))
    P = _program(_weights(EMBED_W, A_W), (_call(1, "Gemm_v1", f"{W}.a/linear[0]", [R(0, "out"), R(-1, A_W), R(-3, "schedule")]),), sched)
    fault = (f"Gemm_v1 under {IMPL}.a reads the `schedule` witness (only ['MoeAlignRanks_v1'] may: the replay binds a reader's keys to "
             "its own committed rank)")
    for q in (Q_WORD_ID, Q_MODULE_BODY_ID):
        pop = Population(P, _corr(P), query=q)
        assert (pop.witness, pop.witness_readers, pop.tp_faults) == ({R(-3, "schedule")}, [1], [fault])
    m = _manifest(P)
    assert not m["complete"] and m["unmodelled"] == {fault: 1}
    assert [x["op_path"] for x in _rows(m, "input_token_ids")] == [f"{IMPL}.embed"]


def _typed(pid: int, name: str, ty: Type) -> InputView:
    """The parameter a view makes of a type: the type's own text and its ports."""
    return InputView(pid, name, repr(ty), ports_of_type(ty))


def test_a_parameter_is_a_record_by_its_type_or_by_its_ports_and_a_supplied_input_is_never_one():
    """Today's answer, held as that.  A record of one field is a record parameter by its type, and a tuple of two fields by its
    ports: its fields are prescribed under their positions, and the manifest lists a weight at "0" and at "1".  A tuple of one
    field is neither, so the Call that reads it is listed as reading token ids.  A supplied input of two fields is supplied and
    no record."""
    row, only_w = Array(N, Value(16)), f"{W}.only.weight"
    one, pair = _typed(-3, "one", RecordType((only_w, row))), _typed(-4, "pair", TupleT(row, row))
    single, run = _typed(-5, "single", TupleT(row)), _typed(-6, "run_0", RecordType(("a", row), ("b", row)))
    assert [(p.type_repr[:1], p.ports) for p in (one, pair, single, run)] == [("{", (only_w,)), ("(", ("0", "1")), ("(", ("0",)), ("{", ("a", "b"))]

    def program(*reads: ValueRef) -> ProgramView:
        return _program(_weights(EMBED_W, A_W), (_call(1, "Gemm_v1", f"{W}.a/linear[0]", [R(0, "out"), R(-1, A_W), *reads]),), one, pair, single, run)

    others = (R(-3, only_w), R(-4, "0"), R(-4, "1"), R(-6, "a"), R(-6, "b"))
    P = program(*others, R(-5, "0"))
    fields = {R(-1, EMBED_W): EMBED_W, R(-1, A_W): A_W, R(-3, only_w): only_w, R(-4, "0"): "0", R(-4, "1"): "1"}
    for q in (Q_WORD_ID, Q_MODULE_BODY_ID):
        pop = Population(P, Correspondence(P), query=q)
        assert [p.name for p in pop.record_params] == ["weights", "one", "pair"]
        assert (pop.record_fields, pop.supplied) == (fields, {R(-6, "a"): "a", R(-6, "b"): "b"})
        assert pop.policy.prescribed_inputs == {*fields, R(-6, "a"), R(-6, "b")}
    m = _manifest(P)
    assert m["complete"]
    assert [x["op_path"] for x in _rows(m, "weights")] == ["0", "1", f"{IMPL}.a.weight", f"{IMPL}.embed.weight", f"{IMPL}.only.weight"]
    assert [x["op_path"] for x in _rows(m, "input_token_ids")] == [f"{IMPL}.a", f"{IMPL}.embed"]
    assert [x["op_path"] for x in _rows(_manifest(program(*others)), "input_token_ids")] == [f"{IMPL}.embed"], "without the tuple of one field"


def _from_calls(pop: Population) -> set[ValueRef]:
    return {v for v in pop.B.values if v.call >= 0}


def test_a_literal_is_no_boundary_value_whichever_body_reads_it():
    """A Call that reads no Value is a literal, written in the Program.  Under either query its Value is no boundary Value and
    is not required, though another module's body reads it."""
    P = _program(_weights(EMBED_W, A_W), (_call(1, "Const_v1", f"{W}.c/const[0]", []),
                                          _call(2, "Gemm_v1", f"{W}.a/linear[0]", [R(0, "out"), R(1, "out"), R(-1, A_W)])))
    for q in (Q_WORD_ID, Q_MODULE_BODY_ID):
        pop = Population(P, _corr(P), query=q)
        assert pop.literals == {1}
        assert _from_calls(pop) == {R(0, "out"), R(2, "out"), R(3, "out")} and R(1, "out") not in pop.required.values


def test_a_returned_value_is_a_boundary_value_unless_it_is_a_check():
    """Of the Values a Program returns, one that is not a check is a boundary Value though no Call reads it, and a check is
    none.  With the same Values returned and none marked a check, each is one."""
    P = _program(_weights(EMBED_W, A_W), (_call(1, "Gemm_v1", f"{W}.a/linear[0]", [R(0, "out"), R(-1, A_W)]),
                                          _call(2, "Gemm_v1", f"{W}.b/linear[0]", [R(1, "out"), R(-1, A_W)]),
                                          _call(3, "I32Eq_v1", f"{W}.b/eq[0]", [R(1, "out")], n=1),
                                          _call(4, "Gemm_v1", f"{W}.a/linear_1[0]", [R(1, "out"), R(-1, A_W)])))
    unmarked = replace(P, outputs=(*P.outputs, R(2, "out"), R(3, "out")))
    checked = replace(unmarked, checks=(R(3, "out"),))
    others = {R(cid, "out") for cid in (0, 1, 2, 4, 5)}
    for q in (Q_WORD_ID, Q_MODULE_BODY_ID):
        assert _from_calls(Population(checked, _corr(checked), query=q)) == others
        assert _from_calls(Population(unmarked, _corr(unmarked), query=q)) == others | {R(3, "out")}


def _moe(module: str, first: str = "MoeRouterTopK_v1") -> ProgramView:
    """Under `module`: a row of `first` with two ports, an `Add_v1`, the up launch, the SiluMul, the down launch and the sum; then
    the logits.  Each row but the sum is read under `module` alone."""
    return _program(_weights(EMBED_W, A_W), (_call(1, first, f"{module}/router[0]", [R(0, "out")], ports=("w", "ids")),
                                             _call(2, "Add_v1", f"{module}/add[0]", [R(0, "out")]),
                                             _call(3, "MoeExpertGemm_v1", f"{module}/up[0]", [R(2, "out"), R(1, "ids")]),
                                             _call(4, "SiluMul_v1", f"{module}/act[0]", [R(3, "out")]),
                                             _call(5, "MoeExpertGemmW_v1", f"{module}/down[0]", [R(4, "out"), R(1, "w")]),
                                             _call(6, "MoeSum_v1", f"{module}/moe_sum[0]", [R(5, "out")]),
                                             _call(7, "Gemm_v1", f"{W}.lm_head/linear[0]", [R(6, "out"), R(-1, A_W)])))


def test_inside_a_moe_module_with_router_rows_only_a_planes_kind_feeds_the_planes():
    """Under the word query the Values a module's body keeps to itself are its interior, but for those that feed the MoE planes:
    the rows of a plane's kind under a module that has router rows.  A SiluMul is of a plane's kind under the runner module
    (`...experts`) alone, and a module with no router row has no plane.  Under the module-body query no Value is inside a body."""
    router, kept = {R(1, "w"), R(1, "ids")}, {R(2, "out")}
    up, act, down = R(3, "out"), R(4, "out"), R(5, "out")
    runner, parent = f"{W}.layers.3.mlp.experts", f"{W}.layers.3.mlp"
    for P, fed, interior in ((_moe(runner), router | {up, act, down}, kept),
                             (_moe(parent), router | {up, down}, kept | {act}),
                             (_moe(runner, first="Add_v1"), set(), router | kept | {up, act, down})):
        word, body = Population(P, _corr(P), query=Q_WORD_ID), Population(P, _corr(P), query=Q_MODULE_BODY_ID)
        assert (word.plane_fed, word.interior) == (fed, interior)
        assert (body.plane_fed, body.interior) == (frozenset(), frozenset())


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


AR = "AllReduceMultimem_v1"
NORM, NORM_2 = f"{W}.norm", f"{W}.norm_2"


def test_a_peer_input_is_an_input_that_collectives_alone_read_and_no_record_field_or_supplied_input():
    """Of the inputs a collective reads, the peers' parts are those no other Call reads: the prompt, which the embedding read
    before it, is none, nor a part a later Call reads too, a weight or a supplied input."""
    def program(*after: CallView) -> ProgramView:
        reduce = _call(1, AR, f"{W}.embed/all_reduce/all_reduce",
                       [R(0, "out"), R(-3, "peers_0"), R(-4, "peers_1"), R(-2, "ids"), R(-5, "values"), R(-1, EMBED_W)])
        P = _rank_program((reduce, *after), 2)
        return replace(P, params=(*P.params, InputView(-5, "values", "Array<4,Value<16>>", (Port("values", 0, N, N, 16),))))

    P = program()
    pop = Population(P, _corr(P), tp=TP)
    assert (pop.tp_faults, pop.operand, pop.supplied) == ([], {1: R(0, "out")}, {R(-5, "values"): "values"})
    assert pop.peer_inputs == {R(-3, "peers_0"), R(-4, "peers_1")}
    P = program(_call(2, "Gemm_v1", f"{W}.a/linear[0]", [R(1, "out"), R(-4, "peers_1")]))
    assert Population(P, _corr(P), tp=TP).peer_inputs == {R(-3, "peers_0")}


def test_a_collective_with_no_rank_local_operand_is_named_and_has_no_partial():
    P = _rank_program((_call(1, AR, f"{W}.embed/all_reduce/all_reduce", [R(-3, "peers_0"), R(-4, "peers_1")]),), 2)
    pop = Population(P, _corr(P), tp=TP)
    fault = f"{AR} under {IMPL}.embed: 0 rank-local operands (one part expected)"
    assert (pop.tp_faults, pop.collectives, pop.operand, pop.partial_of, pop.carries) == ([fault], {1: "all_reduce"}, {}, {}, set())
    m = _rank_manifest(P)
    assert (m["complete"], m["unmodelled"]) == (False, {fault: 1})
    assert _rows(m, "tp_rank_partials") == []


@pytest.mark.parametrize("parts", [(R(0, "out"), R(1, "out")), (R(1, "out"), R(0, "out"))])
def test_a_collective_one_of_whose_two_parts_is_another_collectives_partial_is_refused(parts):
    """A collective may have two rank-local parts of its own body, and both are its partials.  Where one of them is already
    another collective's, the collective is refused and takes neither; the fault names the family of that part's producer,
    whichever of the two it is, and of the first where both are; the collective's operand is its first part."""
    def program(*before: CallView) -> ProgramView:
        k = len(before) + 2
        return _rank_program((_call(1, "Gemm_v1", f"{W}.embed/linear[0]", [R(0, "out")]), *before,
                              _call(k, AR, f"{W}.embed/all_reduce_1/all_reduce", [*parts, R(-4, "peers_1")])), 2)

    alone = program()
    pop = Population(alone, _corr(alone), tp=TP)
    assert (pop.tp_faults, pop.operand, pop.partial_of) == ([], {2: parts[0]}, {R(0, "out"): 2, R(1, "out"): 2})
    P = program(_call(2, AR, f"{W}.embed/all_reduce/all_reduce", [R(0, "out"), R(-3, "peers_0")]))
    pop = Population(P, _corr(P), tp=TP)
    fault = f"EmbeddingShard_v1 under {IMPL}.embed: one Value is the rank-local partial of two collectives"
    assert (pop.tp_faults, pop.operand, pop.partial_of) == ([fault], {2: R(0, "out"), 3: parts[0]}, {R(0, "out"): 2})
    m = _rank_manifest(P)
    assert (m["complete"], m["unmodelled"]) == (False, {fault: 1})
    P = program(_call(2, AR, f"{W}.embed/all_reduce/all_reduce", [R(0, "out"), R(-3, "peers_0")]),
                _call(3, AR, f"{W}.embed/all_reduce_2/all_reduce", [R(1, "out"), R(-3, "peers_0")]))
    pop = Population(P, _corr(P), tp=TP)
    fault = f"{P.call(parts[0].call).family} under {IMPL}.embed: one Value is the rank-local partial of two collectives"
    assert (pop.tp_faults, pop.partial_of) == ([fault], {R(0, "out"): 2, R(1, "out"): 3})


def _carried(*after: CallView, weights: tuple[str, ...] = ()) -> ProgramView:
    """A one-output all-reduce under `site` whose part the embedding's body produced, then the Calls `after` (Calls 2 on)."""
    P = _rank_program((_call(1, AR, f"{W}.site/collective/collective", [R(0, "out"), R(-3, "peers_0")]), *after), 1)
    return replace(P, params=(_weights(EMBED_W, *weights), *P.params[1:]))


def _norm(cid: int, module: str, *reads: ValueRef) -> CallView:
    return _call(cid, "RmsNorm_v1", f"{module}/norm[0]", [*reads, R(-1, f"{module}.weight")])


def test_a_carried_collective_is_at_the_site_its_correspondence_names_unless_one_norm_reads_it(monkeypatch):
    """A carried collective that no norm reads is at its site path, which the correspondence composes: stood in here by the
    spelling a record gives, the module and the fx node.  One that the Calls of one norm read is that norm's and has no site.
    One that two norms read is refused under the implementation's path of its site, and keeps the site."""
    def population(P: ProgramView) -> Population:
        corr = _corr(P)
        monkeypatch.setattr(corr, "site_path", lambda cid: f"{corr.name(cid).module}.{corr.name(cid).node}")
        return Population(P, corr, tp=TP)

    site = f"{W}.site.collective"
    pop = population(_carried())
    assert (pop.tp_faults, pop.carries, pop.site, pop.norm_site) == ([], {1}, {1: site}, {})
    pop = population(_carried(_norm(2, NORM, R(1, "out")), weights=(f"{NORM}.weight",)))
    assert (pop.tp_faults, pop.carries, pop.site, pop.norm_site) == ([], {1}, {}, {1: NORM})
    pop = population(_carried(_norm(2, NORM, R(1, "out")), _norm(3, NORM_2, R(1, "out"), R(2, "out")),
                              weights=(f"{NORM}.weight", f"{NORM_2}.weight")))
    fault = (f"{AR} at {IMPL}.site.collective: not one norm's collective (one `<norm>.weight` read, and no other collective of the "
             "step for that norm unless all are fused, expected)")
    assert (pop.tp_faults, pop.carries, pop.site, pop.norm_site) == ([fault], {1}, {1: site}, {})


def test_the_fused_routes_residual_and_norm_weight_are_none_of_its_parts():
    """The fused all-reduce and norm reads the residual and the norm's weight after its parts.  A residual its own module
    produced is no rank-local operand of it: the part another body produced is the one operand, carried, and the norm whose
    weight it reads is its norm."""
    family = "AllReduceRmsNormMnnvl_v1"
    fused = CallView(2, f"{family}{{N={N}}}", family, "call", f"{W}.site/fused/fused", (Port("0", 0, N, N, 16), Port("1", 0, N, N, 16)),
                     (R(0, "out"), R(-3, "peers_0"), R(1, "out"), R(-1, f"{NORM}.weight")))
    P = _rank_program((_call(1, "Gemm_v1", f"{W}.site/linear[0]", [R(0, "out")]), fused), 1)
    P = replace(P, params=(_weights(EMBED_W, f"{NORM}.weight"), *P.params[1:]))
    pop = Population(P, _corr(P), tp=TP)
    assert (pop.tp_faults, pop.operand, pop.partial_of, pop.carries) == ([], {2: R(0, "out")}, {}, {2})
    assert (pop.site, pop.norm_site, pop.peer_inputs) == ({}, {2: NORM}, {R(-3, "peers_0")})


def test_a_check_after_the_last_sampling_event_is_of_the_last_step_and_any_other_call_after_it_opens_a_step():
    """A Call is of the step its next sampling event ends.  After the last event a check is of the last step, and any other
    Call is of a step after it; a check before an event is of that event's step."""
    P = _program(_weights(EMBED_W, A_W), (_call(1, "Gemm_v1", f"{W}.a/linear[0]", [R(0, "out"), R(-1, A_W)]),
                                          _call(2, "I32Eq_v1", f"{W}.a/eq[0]", [R(1, "out")], n=1)))
    after = (_call(4, "I32Eq_v1", f"{W}.a/eq_1[0]", [R(3, "out")], n=1), _call(5, "Gemm_v1", f"{W}.a/linear_1[0]", [R(3, "out"), R(-1, A_W)]))
    checks = (R(2, "out"), R(4, "out"))
    P = replace(P, calls=(*P.calls, *after), outputs=(*P.outputs, *checks, R(5, "out")), checks=checks, check_cids=frozenset({2, 4}))
    for q in (Q_WORD_ID, Q_MODULE_BODY_ID):
        pop = Population(P, _corr(P), query=q)
        assert pop.events == [3]
        assert [pop.step_of(c.id) for c in P.calls] == [0, 0, 0, 0, 0, 1]
