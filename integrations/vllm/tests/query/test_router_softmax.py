"""The router-softmax policy (`verity_vllm.query.router_softmax`): the committed interior values of the selection-order router are, in
order, exactly the words the tap writes (with no recompute); off, the manifest is byte-identical; on, one `router_softmax` identity per
(step, MoE layer) over its router Calls, the header says so, and the plan, the replay coverage, form (B) and `Q_word_v1` see the family."""
from __future__ import annotations

import json
import types

import pytest
from verity_vllm.query.manifest.bind import Binder
from verity_vllm.query.units import ValueRef
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.query import router_softmax as RS
from verity_vllm.query.manifest import format as fmt
from verity_vllm.query.program_view import CallView, InputView, Port, ProgramView
from verity_vllm.query.required import request_manifest

W = "model.model"
H, E, K = 8, 4, 2
RES = {"program": {"digest": "d0", "function": "f"}, "attention_impls_observed": {}, "target": {"declared": {"compute_capability": [8, 9]}}}
R = ValueRef


@pytest.mark.parametrize("E_,renorm", [(64, False), (64, True), (128, False), (128, True)])
def test_the_committed_softmax_is_the_taps_layout_with_no_recompute(E_, renorm):
    from verity.ml.definitions import word as W_

    fn = RS.router_fn(E_, 8, 8, renorm)
    G = W_.Graph(fn)
    cut = W_.units(G, 16, 32)["cut"]
    assert cut.ok and G.recomputed == [], cut.codes
    s = RS.slots(E_, 8, 8, renorm)
    assert len(s) == RS.words(E_, 8, renorm) == (2 * E_ + 2 + (9 if renorm else 0)) and list(s) == sorted(s)
    rule = W_.unit_rule(fn)
    assert rule["committed_interior_words"] == len(s) and not rule["violations"] and rule["max_out_bits"] == 32


def program(rows: int = 3, family: str = "MoeRouterTopKOrdered_v1") -> ProgramView:
    """Per token row: embed -> the router gate Gemm (E logits) -> the router of MoE layer 1 -> lm_head over the embedding; one argmax."""
    weights = InputView(-1, "weights", "{...}", tuple(Port(f"{W}.{n}.weight", 8 * i, 8 * i + 8, 8, 16) for i, n in
                                                      enumerate(("embed", "layers.1.mlp.gate", "lm_head"))))
    ids = InputView(-2, "ids", f"Array<{rows},Value<32>>", (Port("ids", 0, rows, rows, 32),))
    calls = []

    def add(spec, name, reads, ports=(("out", H),), width=16):
        calls.append(CallView(len(calls), spec, spec.split("{", 1)[0], "call", name, tuple(Port(p, 0, n, n, width) for p, n in ports), tuple(reads)))
        return len(calls) - 1

    last = None
    for r in range(rows):
        e = add(f"Embedding_v1{{V=8,H={H}}}", f"{W}.embed/embedding/out[{r}]", [R(-2, "ids"), R(-1, f"{W}.embed.weight")])
        lg = add(f"Gemm_v1{{K={H},N={E}}}", f"{W}.layers.1.mlp.gate/linear[{r}]", [R(e, "out"), R(-1, f"{W}.layers.1.mlp.gate.weight")],
                 ports=(("out", E),))
        add(f"{family}{{E={E},TOPK={K},VPT={E}}}", f"{W}.layers.1.mlp.experts/topk_softmax[{r}]", [R(lg, "out")],
            ports=(("0", K), ("1", K)), width=32)
        last = add(f"Gemm_v1{{K={H},N={H}}}", f"{W}.lm_head/linear[{r}]", [R(e, "out"), R(-1, f"{W}.lm_head.weight")])
    add("ArgmaxTokenSelect_v1{N=1}", "argmax[0]", [R(last, "out")], ports=(("out", 1),), width=32)
    return ProgramView(tuple(calls), (weights, ids), (R(len(calls) - 1, "out"),), "synthetic", "synthetic")


def _manifest(P, on=False):
    corr = Correspondence(P, implementation_paths=["model.lm_head"])
    return request_manifest(P, corr, RES, request_id="r0", taps=(RS.POLICY,) if on else ()).manifest


def test_off_is_byte_identical():
    P = program()
    a = _manifest(P)
    b = request_manifest(P, Correspondence(P, implementation_paths=["model.lm_head"]), RES, request_id="r0").manifest
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert RS.FAMILY not in a["populations"]["by_family"] and not RS.policy_of(a) and "router_softmax" not in a["query"]


@pytest.mark.parametrize("family,renorm", [("MoeRouterTopKOrdered_v1", False), ("MoeRouterTopKOrderedNorm_v1", True)])
def test_one_identity_per_step_and_layer_over_its_router_rows(family, renorm):
    P = program(rows=3, family=family)
    off, on = _manifest(P), _manifest(P, on=True)
    rows = [x for x in on["identities"] if x["family"] == RS.FAMILY]
    n = RS.words(E, K, renorm)
    assert [(x["op_path"], x["output_member"], x["element_range"], x["rows"], x["width"], x["dtype"], x["spec"], x["renormalize"]) for x in rows] == \
        [("moe/L1", RS.MEMBER, [0, 3 * n], 3, n, "f32", family, renorm)]
    assert rows[0]["moe_module"].endswith("layers.1.mlp.experts") and (rows[0]["experts"], rows[0]["topk"]) == (E, K)
    strip = lambda m: sorted(json.dumps(x, sort_keys=True) for x in m["identities"] if x["family"] != RS.FAMILY)  # noqa: E731
    assert strip(on) == strip(off), "the policy only adds identities"
    q = on["query"]
    assert RS.policy_of(on) and q["required_families"][-1] == RS.FAMILY and q["query_id_digest"] != off["query"]["query_id_digest"]
    assert q["router_softmax"]["words"] == 3 * n and q["router_softmax"]["rows_by_family"] == {family: 3}
    assert RS.manifest_layers(on) == {1: {"module": rows[0]["moe_module"], "experts": E, "topk": K, "renormalize": renorm, "width": n}}
    assert RS.words_per_token(on) == n and RS.words_per_token(off) == 0


def test_the_kernel_order_router_has_no_tapped_softmax():
    on = _manifest(program(family="MoeRouterTopK_v1"), on=True)
    assert not [x for x in on["identities"] if x["family"] == RS.FAMILY] and on["query"]["router_softmax"]["words"] == 0


def test_manifest_layers_refuses_a_layer_under_two_geometries():
    man = _manifest(program(), on=True)
    row = next(x for x in man["identities"] if x["family"] == RS.FAMILY)
    man["identities"].append(dict(row, step=1, width=row["width"] + 9, renormalize=True))
    with pytest.raises(ValueError, match="two geometries"):
        RS.manifest_layers(man)


def test_the_family_is_in_every_vocabulary():
    from verity_vllm.acquire import plan
    from verity_vllm.check import oracle_compare as OC
    from verity_vllm.check.replay import coverage

    assert RS.FAMILY in fmt.FAMILIES and plan.ROUTE_OF_FAMILY[RS.FAMILY] == "router_tap" and RS.FAMILY in coverage.QUERY_FAMILY_MECHANISM
    entry = {"identity": {"op_path": "moe/L1", "output_member": RS.MEMBER}, "name": f"moe/L1/{RS.MEMBER}"}
    assert OC._family_of(entry) == "hidden_stream", "the Match record holds no value for the router's softmax: form (B) never compares it"
    row = {"family": RS.FAMILY, "op_path": "moe/L1", "output_member": RS.MEMBER, "step": 0}
    assert Binder([row]).bind(0, [f"moe/L1/{RS.MEMBER}"]) == [("moe/L1", RS.MEMBER, 0, "label")]


def test_the_plan_anchors_the_tap_on_the_moe_module():
    from verity_vllm.acquire import plan as PL
    from verity_vllm.correspondence.reader_for_acquire import AnnotationCorrespondence, runtime_tree_from_paths

    man = _manifest(program(), on=True)
    mod = next(x["moe_module"] for x in man["identities"] if x["family"] == RS.FAMILY)
    rt = runtime_tree_from_paths({"": ["M"], mod: ["FusedMoE"]}, buffers=[], runner={"sample": True}, source="synthetic")
    (v,) = [v for v in PL.value_classes(man) if v.family == RS.FAMILY]
    legal, refused = PL.sites_of(v, AnnotationCorrespondence(rt, wrapper_prefix="", aliases={}))
    assert [(s.kind, s.module, s.slot) for s in legal] == [("native_tap", "moe/L1", RS.MEMBER)] and not refused
    gone = runtime_tree_from_paths({"": ["M"]}, buffers=[], runner={"sample": True}, source="synthetic")
    legal, refused = PL.sites_of(v, AnnotationCorrespondence(gone, wrapper_prefix="", aliases={}))
    assert not legal and "is not a module of the runtime tree" in refused[0]["why"]


def test_word_check_sees_the_tapped_softmax_as_committed():
    from verity.ml.definitions import word as W_

    groups = [{"family": fam, "statics": {"E": 64, "TOPK": 8, "VPT": 8}, "definition": f"{fam}{{E=64,TOPK=8,VPT=8}}", "calls": 3,
               "where": "model.layers.0.mlp.experts"} for fam in RS.ROUTERS]
    tapped = W_.check_calls(groups, acquired=RS.TAP_WORDS)
    assert tapped["ok"] and [g["acquired_interior_words"] for g in tapped["groups"]] == [3 * 130, 3 * 139]
    with pytest.raises(W_.QueryRuleViolation, match="interior-not-acquired"):
        W_.check_calls(groups[:1], acquired={"MoeRouterTopKOrdered_v1": 129})


def test_the_manifest_cli_and_verify_carry_the_policy(monkeypatch):
    from verity_vllm.pipeline import manifest as M
    from verity_vllm.pipeline import manifest_verify as MV

    assert M.taps_of() == () and M.taps_of(router_softmax=True) == (RS.FAMILY,) and M.TAPS[RS.FAMILY] is RS.POLICY
    seen = {}
    monkeypatch.setattr(M, "build", lambda program, **kw: seen.setdefault("kw", kw) and {})
    MV.rebuild(program="p", taps=M.taps_of(router_softmax=True))
    assert seen["kw"]["taps"] == (RS.FAMILY,)


def test_the_source_attaches_only_with_a_build_and_a_manifest_that_names_the_family():
    from verity_vllm.acquire.sources import router_softmax_source as S

    com = types.SimpleNamespace(name="rec", sources=[], attach_source=lambda src: com.sources.append(src))
    assert S.attach(com, None, {}, ("a",)) == (("a",), None) and com.sources == []
    with pytest.raises(SystemExit):
        S.attach(com, "/nonexistent/verity_router_tap.so", _manifest(program()), ("a",))
