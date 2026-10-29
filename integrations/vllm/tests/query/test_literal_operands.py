"""The literal-operand rule of `Q_module_body_v1`: a Call that reads no Value is a literal of the Program (its output is written in the
descriptor), never a boundary Value and never required -- wherever the correspondence places it and whatever it is named.

`sampler_program()` is a seeded-Gumbel shape under the v1-annotation fallback (a v1-frontend Build carries no correspondence
record), two steps::

    ids --Embedding[model.model.embed]--> h --Gemm[model.model.lm_head]--> logits --GumbelTopPTokenSelect[root]--> tok --Embedding--> ...
                          Const32[top_p], Const64[pos], Const32[temperature] --------^   (+ the seed / splits Inputs)

The frontend names the three literals `<select node>/top_p` ...; the fallback parses `<a>/<b>` as `<module>/<fx node>`, so they form a
body `gumbel_topp_token_select` of their own and their Values cross into the root body: the 32 extra `gumbel_topp_token_select_<s>/out`
identities (and the `two producers` incompleteness) under ENGINE=v2.
"""

from __future__ import annotations

from verity_vllm.query.units import ValueRef, boundary
from verity_vllm.query import program_view as pv
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.query.module_body import Q_module_body_v1, literal_calls, module_body_boundary
from verity_vllm.query.program_view import CallView, InputView, Port, ProgramView
from verity_vllm.query.required import request_manifest

R = ValueRef
W = "model.model"
EMBED_W, HEAD_W = f"{W}.embed.weight", f"{W}.lm_head.weight"
RES = {"program": {"digest": "d0", "function": "f"}, "target": {"declared": {"compute_capability": [8, 9]}}}
SELECT = "GumbelTopPTokenSelect_v1"
TOP_P, TEMP = "Const32[0x3f733333]_v1", "Const32[0x3f4ccccd]_v1"


def _call(cid, spec, name, reads, *, n=4, width=16, ports=("out",)):
    table, lo = [], 0
    for p in ports:
        table.append(Port(p, lo, lo + n, n, width))
        lo += n
    return CallView(cid, spec, pv.spec_base(spec), "call", name, tuple(table), tuple(reads))


def _params():
    return (InputView(-1, "weights", "{...}", (Port(EMBED_W, 0, 8, 8, 16), Port(HEAD_W, 8, 16, 8, 16))),
            InputView(-2, "ids", "Value<32>", (Port("out", 0, 1, 1, 32),)),
            InputView(-3, "seed", "Value<64>", (Port("out", 0, 1, 1, 64),)),
            InputView(-4, "splits", "Array<2,Value<32>>", (Port("out", 0, 2, 2, 32),)))


def _step(k0: int, node: str, token: ValueRef, pos: int) -> tuple[CallView, ...]:
    return (
        _call(k0, "Embedding_v1{V=8,H=4}", f"{W}.embed/embedding[0]", [token, R(-1, EMBED_W)]),
        _call(k0 + 1, "Gemm_v1{K=4,N=4}", f"{W}.lm_head/linear[0]", [R(k0, "out"), R(-1, HEAD_W)]),
        _call(k0 + 2, TOP_P, f"{node}/top_p", [], n=1, width=32),
        _call(k0 + 3, f"Const64[0x{pos:016x}]_v1", f"{node}/pos", [], n=1, width=64),
        _call(k0 + 4, TEMP, f"{node}/temperature", [], n=1, width=32),
        _call(k0 + 5, f"{SELECT}{{V=4}}", f"{node}[0]",
              [R(k0 + 1, "out"), R(k0 + 2, "out"), R(-3, "out"), R(k0 + 3, "out"), R(k0 + 4, "out"), R(-4, "out")], n=1, width=32),
    )


def sampler_program() -> ProgramView:
    calls = _step(0, "gumbel_topp_token_select", R(-2, "out"), 256) + _step(6, "gumbel_topp_token_select_1", R(5, "out"), 257)
    return ProgramView(calls, _params(), (R(11, "out"),), "synthetic", "synthetic")


LITERALS = frozenset({2, 3, 4, 8, 9, 10})


def constant_s_program(S: int = 32) -> ProgramView:
    """A single-request workload's Program: the same two steps with S its own `Const32[S]` literal (`<select node>/splits`, emitted
    after temperature) in place of the `splits` Input."""
    calls = []
    for k0, node, tok, pos in ((0, "gumbel_topp_token_select", R(-2, "out"), 256), (7, "gumbel_topp_token_select_1", R(6, "out"), 257)):
        *head, sel = _step(k0, node, tok, pos)
        calls += [*head, _call(k0 + 5, f"Const32[0x{S:08x}]_v1", f"{node}/splits", [], n=1, width=32),
                  _call(k0 + 6, sel.definition, sel.name, [*sel.reads[:5], R(k0 + 5, "out")], n=1, width=32)]
    return ProgramView(tuple(calls), _params()[:3], (R(13, "out"),), "synthetic", "synthetic")


def columnar(P: ProgramView) -> ProgramView:
    """The same body through the columnar builder (`from_instances`' representation): reads as operand runs."""
    k_of = {p.id: k for k, p in enumerate(P.params)}
    b = pv._Builder(P.params)
    for c in P.calls:
        runs = []
        for v in c.reads:
            if v.call >= 0:
                port = P.call(v.call).port(v.port)
                runs.append([("n", v.call, port.lo, port.hi)])
            else:
                port = next(q for q in P.input(v.call).port_table if q.name == v.port)
                runs.append([("p", k_of[v.call], port.lo, port.hi)])
        b.add(c.id, c.definition, c.form, c.name, c.port_table, runs)
    return b.view(P.outputs, P.digest, "instances")


def test_literal_calls_are_the_calls_that_read_nothing():
    P = sampler_program()
    assert literal_calls(P) == LITERALS
    C = columnar(P)
    assert C.columns is not None and literal_calls(C) == LITERALS, "the columnar fast path (read offsets) agrees"
    # decided by the reads alone: a zero-operand Call of any Definition is a literal; a Call reading only a literal is not
    extra = (_call(12, "Iota_v1{N=4}", f"{W}/iota[0]", []), _call(13, "Convert_v1{N=1}", f"{W}/convert[0]", [R(2, "out")], n=1))
    P2 = ProgramView(tuple(P.calls) + extra, P.params, P.outputs, None, "synthetic")
    assert literal_calls(P2) == LITERALS | {12}


def test_literals_cross_a_body_boundary_but_are_not_boundary_values():
    P = sampler_program()
    corr = Correspondence(P)
    assert corr.source == "v1_annotations" and corr.module_of(2) == "gumbel_topp_token_select" and corr.module_of(5) == ""
    raw = boundary(P, Q_module_body_v1(P, corr))
    lit = {R(k, "out") for k in LITERALS}
    assert lit <= raw.values, "the mechanism: the literals' body is not the select's, so the plain boundary carries them"
    B = module_body_boundary(P, corr)
    assert B.values == raw.values - lit and not (lit & B.produced_in.keys()) and not (lit & B.read_by.keys())
    assert B.units == raw.units and B.owner == raw.owner, "the partition is untouched: only the literals' Values leave the boundary"
    assert {R(-2, "out"), R(-3, "out"), R(-4, "out"), R(-1, EMBED_W), R(-1, HEAD_W), R(1, "out"), R(5, "out"), R(11, "out")} <= B.values
    C = columnar(P)
    assert module_body_boundary(C, Correspondence(C)).values == B.values


def test_request_manifest_requires_no_literal():
    P = sampler_program()
    for return_slots in (True, False):
        r = request_manifest(P, Correspondence(P), RES, request_id="r0", return_slots=return_slots)
        man = r.manifest
        assert not any(v.call in LITERALS for v in r.required.values)
        assert man["complete"] and man["unmodelled"] == {}
        rows = sorted((x["family"], x["step"], x["op_path"], x["output_member"], tuple(x["element_range"])) for x in man["identities"])
        per_step = lambda s: [("input_token_ids", s, f"{W}.embed", "in", (0, 1)), ("instance_outputs", s, f"{W}.embed", "out", (0, 4)),  # noqa: E731
                              ("instance_outputs", s, f"{W}.lm_head", "out", (0, 4)), ("sampled_token_ids", s, "runner.sampler", "sampled_token_ids", (0, 1)),
                              ("sampler_seed", s, "runner.sampler", "seed", (0, 1)), ("sampler_splits", s, "runner.sampler", "splits", (0, 1))]
        assert rows == sorted([("weights", "*", EMBED_W, "weight", (0, None)), ("weights", "*", HEAD_W, "weight", (0, None))] + per_step(0) + per_step(1))
        assert man["populations"]["literal_calls"] == 6 and man["populations"]["steps"] == 2
        assert "minus the Program's literals" in man["query"]["query_text"]


def test_a_constant_s_program_requires_the_seed_word_and_no_splits_word():
    """S is a literal of a single-request Program: the manifest keeps `sampler_seed` per step and has no `sampler_splits` family, so
    Commit neither collects nor links a `runner.sampler/splits` word for it."""
    P = constant_s_program()
    lits = literal_calls(P)
    assert lits == {2, 3, 4, 5, 9, 10, 11, 12} and [p.name for p in P.params] == ["weights", "ids", "seed"]
    r = request_manifest(P, Correspondence(P), RES, request_id="r0")
    fams = sorted({(x["family"], x["step"]) for x in r.manifest["identities"] if x["op_path"] == "runner.sampler"})
    assert r.manifest["complete"] and not any(v.call in lits for v in r.required.values)
    assert fams == [("sampled_token_ids", 0), ("sampled_token_ids", 1), ("sampler_seed", 0), ("sampler_seed", 1)]


def test_a_literal_named_as_a_return_slot_of_a_crossing_node_is_not_required():
    """A pad guard's `stop = I32Eq(y, Const32[EOS])`: the literal is named `<module>/eq_3/eos` (a return-slot hint to the RETURN-SLOT
    policy) and shares module + fx node with the I32Eq row whose output crosses into the root body (TENSOR-RETURN).  Neither policy
    makes the literal required; the crossing I32Eq row stays."""
    params = (InputView(-1, "weights", "{...}", (Port(EMBED_W, 0, 8, 8, 16), Port(HEAD_W, 8, 16, 8, 16))),
              InputView(-2, "h", "Array<4,Value<16>>", (Port("out", 0, 4, 4, 16),)))
    calls = (
        _call(0, "Gemm_v1{K=4,N=4}", f"{W}.lm_head/linear[0]", [R(-2, "out"), R(-1, HEAD_W)]),
        _call(1, "ArgmaxTokenSelect_v1{V=4}", "argmax[0]", [R(0, "out")], n=1, width=32),
        _call(2, "Const32[0x00000002]_v1", f"{W}/eq_3/eos", [], n=1, width=32),
        _call(3, "I32Eq_v1", f"{W}/eq_3[0]", [R(1, "out"), R(2, "out")], n=1, width=1),
        _call(4, "SelectI32_v1", "where[0]", [R(3, "out"), R(1, "out")], n=1, width=32),
        _call(5, "Embedding_v1{V=8,H=4}", f"{W}.embed/embedding[0]", [R(4, "out"), R(-1, EMBED_W)]),
        _call(6, "Gemm_v1{K=4,N=4}", f"{W}.lm_head/linear[0]", [R(5, "out"), R(-1, HEAD_W)]),
        _call(7, "ArgmaxTokenSelect_v1{V=4}", "argmax_1[0]", [R(6, "out")], n=1, width=32),
    )
    P = ProgramView(calls, params, (R(7, "out"),), None, "synthetic")
    corr = Correspondence(P)
    assert corr.name(2).hint == "eos" and (corr.module_of(2), corr.name(2).node) == (corr.module_of(3), corr.name(3).node) == (W, "eq_3")
    r = request_manifest(P, corr, RES, request_id="r0")
    assert R(3, "out") in r.required.boundary and not any(v.call == 2 for v in r.required.values)
    body = {(x["step"], x["output_member"]) for x in r.manifest["identities"] if x["op_path"] == W and x["family"] == "instance_outputs"}
    assert body == {(1, "out")}, "the I32Eq row is required; no `eos` member for the literal"
    assert r.manifest["complete"]
