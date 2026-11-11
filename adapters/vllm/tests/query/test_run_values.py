"""The `run_values` family: a Program's `run` Input field `<op_path>/<member>` (a DSA row's Waksman settings past INDEX_TOPK) is one
required identity of the step whose forward reads it, element range [0, its leaves), anchored on its reader's module; a Program
with no such field has a manifest byte-for-byte the one it had before the family existed.  The checks a Program returns of those
fields are its outputs, never required Values."""

from __future__ import annotations

import json
from dataclasses import replace

from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.query import program_view as pv
from verity_vllm.query.manifest import format as fmt
from verity_vllm.query.program_view import CallView, InputView, Port, ProgramView
from verity_vllm.query.required import request_manifest
from verity_vllm.query.units import ValueRef as R

W = "model.model"
ATTN = "model.layers.0.self_attn.attn"
EMBED_W, HEAD_W = f"{W}.embed.weight", f"{W}.lm_head.weight"
RES = {"program": {"digest": "d0", "function": "f"}, "target": {"declared": {"compute_capability": [9, 0]}}}
LEAVES = {5: 10, 6: 12}


def _field(pos: int) -> str:
    return f"{ATTN}/topk_switches[t={pos}]"


def _call(cid, spec, name, reads, *, n=4, width=16):
    return CallView(cid, spec, pv.spec_base(spec), "call", name, (Port("out", 0, n, n, width),), tuple(reads))


def _program(fields=(), reads=None, checked=False) -> ProgramView:
    """Two greedy steps; `fields`: the `run` Input's (name, leaves); `reads[step]`: the positions whose field that step's order Call
    (`DsaWaksmanOrder_v1{N=pos+1,TOPK=4}` under ATTN) reads; `checked`: each order checked (`DsaTopKLongRow_v1`), the bits returned
    after the token as the Program's checks."""
    params = [InputView(-1, "weights", "{...}", (Port(EMBED_W, 0, 8, 8, 16), Port(HEAD_W, 8, 16, 8, 16))),
              InputView(-2, "ids", "Value<32>", (Port("out", 0, 1, 1, 32),))]
    ports, lo = [], 0
    for name, n in fields:
        ports.append(Port(name, lo, lo + n, n, 1))
        lo += n
    if fields:
        params.append(InputView(-3, fmt.RUN_PARAM, "{...}", tuple(ports)))
    calls, tok, checks = [], R(-2, "out"), []
    for step in (0, 1):
        k0 = len(calls)
        calls += [_call(k0, "Embedding_v1{V=8,H=4}", f"{W}.embed/embedding[0]", [tok, R(-1, EMBED_W)]),
                  _call(k0 + 1, "Gemm_v1{K=4,N=4}", f"{W}.lm_head/linear[0]", [R(k0, "out"), R(-1, HEAD_W)])]
        for p in (reads or {}).get(step, ()):
            calls.append(_call(len(calls), f"DsaWaksmanOrder_v1{{N={p + 1},TOPK=4}}", f"{ATTN}/waksman_order_{p}[0]", [R(-3, _field(p))],
                               n=4, width=32))
            if checked:
                calls.append(_call(len(calls), f"DsaTopKLongRow_v1{{N={p + 1},TOPK=4}}", f"{ATTN}/topk_long_row_{p}[0]",
                                   [R(len(calls) - 1, "out")], n=1, width=1))
                checks.append(R(len(calls) - 1, "out"))
        calls.append(_call(len(calls), "ArgmaxTokenSelect_v1", "argmax[0]", [R(k0 + 1, "out")], n=1, width=32))
        tok = R(len(calls) - 1, "out")
    return ProgramView(tuple(calls), tuple(params), (tok, *checks), "synthetic", "synthetic", tuple(checks))


def _manifest(P: ProgramView) -> dict:
    return request_manifest(P, Correspondence(P), RES, request_id="r0", rank=0, ranks=1).manifest


def _served() -> dict:
    return _manifest(_program([(_field(p), n) for p, n in LEAVES.items()], reads={0: [5], 1: [6]}))


def test_each_run_field_is_one_identity_of_the_step_that_reads_it():
    m = _served()
    assert m["complete"], m["unmodelled"]
    rows = [r for r in m["identities"] if r["family"] == fmt.RUN_FAMILY]
    assert [(r["step"], r["engine_step"], r["invocation"], r["op_path"], r["output_member"], r["element_range"], r["rank"], r["request_id"],
             r["leaves"], r["dtype"], r["attention_module"], r["statics"], r["spec"]) for r in rows] == \
        [(0, 0, 0, ATTN, "topk_switches[t=5]", [0, 10], 0, "r0", [0, 10], "u8", ATTN, {"N": 6, "TOPK": 4}, "DsaWaksmanOrder_v1"),
         (1, 1, 0, ATTN, "topk_switches[t=6]", [0, 12], 0, "r0", [10, 22], "u8", ATTN, {"N": 7, "TOPK": 4}, "DsaWaksmanOrder_v1")]
    assert m["populations"]["by_family"][fmt.RUN_FAMILY] == 2
    assert m["query"]["required_families"][-1] == fmt.RUN_FAMILY and fmt.RUN_FAMILY in fmt.FAMILIES


def test_a_program_without_run_fields_adds_nothing_to_its_manifest():
    """No `run` Input, or one whose fields name no `<op_path>/<member>` (a fold's sampling values, registration's, prescribed like the
    weights): no `run_values` row, family or policy line, the same query header and nothing unmodelled."""
    base = _manifest(_program())
    fold = _manifest(_program([("top_p", 1)]))
    assert base["complete"] and fold["unmodelled"] == base["unmodelled"] and fold["query"] == base["query"]
    for m in (base, fold):
        assert fmt.RUN_FAMILY not in json.dumps(m)
        assert not any(f"`{fmt.RUN_PARAM}` field" in p for p in m["query"]["policy"]["protocol_required"])


def test_the_checks_a_program_returns_are_never_required():
    """A Program returning a check of each order beside its token: no identity of its manifest is a check (nothing commits one; the
    replay evaluates it), and the same outputs not marked as its checks would each be required, the manifest otherwise the same."""
    def rows(m: dict) -> list:
        return sorted((r["family"], r["step"], r["op_path"], r["output_member"], r.get("spec")) for r in m["identities"])

    checked = _program([(_field(p), n) for p, n in LEAVES.items()], {0: [5], 1: [6]}, checked=True)
    m, unmarked = _manifest(checked), rows(_manifest(replace(checked, checks=())))
    want = [("instance_outputs", s, ATTN, "out", "DsaTopKLongRow_v1") for s in (0, 1)]
    assert m["complete"] and [r for r in unmarked if r[-1] == "DsaTopKLongRow_v1"] == want
    assert rows(m) == [r for r in unmarked if r not in want]


def test_a_descriptors_last_returned_leaves_are_its_checks():
    """`from_descriptor(checks=n)`: the Values of the return's last n leaves are the view's checks, still among its outputs."""
    from verity.circuits.codec import encode_program
    from verity.circuits.defs import REGISTRY, CompositeDefinition, bind
    from verity.circuits.program import Program
    from verity.circuits.refs import tuple_of
    from verity.circuits.types import Tuple, Value
    from verity.definitions.registry import prims

    i32 = Value(32)
    checked = CompositeDefinition("QChecked", 1, (), lambda S: ((("a", i32), ("b", i32)), Tuple(i32, Value(1))),
                                  lambda B, S, a, b: tuple_of(B.call(prims.I32Add, a, b), B.call(prims.I32Eq, a, b)), register=False)
    desc = encode_program(Program(bind(checked)))
    plain, view = pv.from_descriptor(desc, registry=REGISTRY), pv.from_descriptor(desc, registry=REGISTRY, checks=1)
    assert plain.checks == () and view.outputs == plain.outputs and len(view.outputs) == 2 and view.checks == view.outputs[1:]
    assert view.call(view.checks[0].call).family == "I32Eq_v1"


def test_a_run_field_no_step_or_two_steps_read_is_unmodelled_by_name():
    m = _manifest(_program([(_field(5), 10), (_field(6), 12)], reads={0: [5], 1: [5]}))
    assert not m["complete"]
    assert sorted(m["unmodelled"]) == [f"run value {_field(5)}: read at steps [0, 1] (one step's forward reads it)",
                                       f"run value {_field(6)}: read at steps [] (one step's forward reads it)"]


def test_the_plan_anchors_one_value_class_per_layer_on_its_readers_module():
    from verity_vllm.acquire import plan as PL
    from verity_vllm.correspondence.reader_for_acquire import AnnotationCorrespondence, runtime_tree_from_paths

    m = _served()
    assert PL.ROUTE_OF_FAMILY[fmt.RUN_FAMILY] == "run_tap"
    (v,) = [v for v in PL.value_classes(m) if v.family == fmt.RUN_FAMILY]
    assert (v.op_path, v.output_member, v.identities, v.anchor) == (ATTN, "*", 2, ATTN)
    rt = runtime_tree_from_paths({"": ["M"], ATTN: ["DeepseekV32Attention"]}, buffers=[], runner={"sample": True}, source="synthetic")
    legal, refused = PL.sites_of(v, AnnotationCorrespondence(rt, wrapper_prefix="", aliases={}))
    assert [(s.kind, s.module, s.slot) for s in legal] == [("native_tap", ATTN, "*")] and not refused
    gone = runtime_tree_from_paths({"": ["M"]}, buffers=[], runner={"sample": True}, source="synthetic")
    legal, refused = PL.sites_of(v, AnnotationCorrespondence(gone, wrapper_prefix="", aliases={}))
    assert not legal and "is not a module of the runtime tree" in refused[0]["why"]
