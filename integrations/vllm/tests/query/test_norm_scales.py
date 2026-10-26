"""The norm-scale policy (`verity_vllm.query.norm_scales`) on hand-built request Programs: off, the manifest is byte-identical; on, one
`norm_scales` identity per (step, kernel norm module) over its Call rows, an ATen-chain norm's rsqrt output protocol-required as
`norm_scale`, the header says so, and the plan, the replay coverage, form (B) and `Q_word_v1` see the family."""
from __future__ import annotations

import copy
import json

import pytest
from verity_vllm.query.manifest.bind import Binder
from verity_vllm.query.units import ValueRef
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.query import norm_scales as NS
from verity_vllm.query.manifest import format as fmt
from verity_vllm.query.program_view import CallView, InputView, Port, ProgramView
from verity_vllm.query.required import Q_MODULE_BODY_ID, request_manifest

W = "model.model"
H = 8
RES = {"program": {"digest": "d0", "function": "f"}, "attention_impls_observed": {}, "target": {"declared": {"compute_capability": [8, 9]}}}
R = ValueRef


def _call(cid, spec, name, reads, ports=(("out", H),), width=16):
    return CallView(cid, spec, spec.split("{", 1)[0], "call", name, tuple(Port(p, 0, n, n, width) for p, n in ports), tuple(reads))


def kernel_program(rows: int = 2) -> ProgramView:
    """Per token row: embed -> the first norm (Triton, no residual) -> Gemm(o_proj) -> the final fused add-norm (residual = the embedding
    row; ports 0 normed, 1 residual) -> lm_head Gemm; one argmax over the last row."""
    weights = InputView(-1, "weights", "{...}", tuple(Port(f"{W}.{n}.weight", 8 * i, 8 * i + 8, 8, 16) for i, n in
                                                      enumerate(("embed", "layers.0.input_layernorm", "layers.0.o_proj", "norm", "lm_head"))))
    ids = InputView(-2, "ids", f"Array<{rows},Value<32>>", (Port("ids", 0, rows, rows, 32),))
    calls, k = [], 0

    def add(spec, name, reads, ports=(("out", H),), width=16):
        nonlocal k
        calls.append(_call(k, spec, name, reads, ports, width))
        k += 1
        return k - 1

    last = None
    for r in range(rows):
        e = add(f"Embedding_v1{{V=8,H={H}}}", f"{W}.embed/embedding/out[{r}]", [R(-2, "ids"), R(-1, f"{W}.embed.weight")])
        n0 = add(f"RMSNormTriton_v1{{N={H},EPS=1e-05}}", f"{W}.layers.0.input_layernorm/triton_launch/out[{r}]",
                 [R(e, "out"), R(-1, f"{W}.layers.0.input_layernorm.weight")])
        o = add(f"Gemm_v1{{K={H},N={H}}}", f"{W}.layers.0.o_proj/linear[{r}]", [R(n0, "out"), R(-1, f"{W}.layers.0.o_proj.weight")])
        n1 = add(f"RMSNormFusedCuda_v2{{N={H},EPS=1e-05}}", f"{W}.norm/fused_add_rms_norm/out[{r}]", [R(o, "out"), R(e, "out"), R(-1, f"{W}.norm.weight")],
                 ports=(("0", H), ("1", H)))
        last = add(f"Gemm_v1{{K={H},N={H}}}", f"{W}.lm_head/linear[{r}]", [R(n1, "0"), R(-1, f"{W}.lm_head.weight")])
    add("ArgmaxTokenSelect_v1{N=1}", "argmax[0]", [R(last, "out")], ports=(("out", 1),), width=32)
    return ProgramView(tuple(calls), (weights, ids), (R(k - 1, "out"),), "synthetic", "synthetic")


def chain_program(rows: int = 3, rsqrt_n: int = 1) -> ProgramView:
    """Per token row, an ATen-chain norm in one module body (square, mean, + eps, rsqrt, scale row, weight, narrow; only the narrowed
    output is a return slot), read by a Gemm."""
    weights = InputView(-1, "weights", "{...}", (Port(f"{W}.layers.0.input_layernorm.weight", 0, H, H, 16), Port(f"{W}.lm_head.weight", H, 2 * H, H, 16)))
    xs = InputView(-2, "x", f"Array<{rows * H},Value<16>>", (Port("x", 0, rows * H, rows * H, 16),))
    calls, k = [], 0
    N = f"{W}.layers.0.input_layernorm"

    def add(spec, name, reads, n=H, width=32):
        nonlocal k
        calls.append(_call(k, spec, name, reads, (("out", n),), width))
        k += 1
        return k - 1

    last = None
    for r in range(rows):
        sq = add(f"SquareBf16_v1{{N={H}}}", f"{N}/pow[{r}]", [R(-2, "x")])
        mu = add(f"MeanTriton_v1{{N={H}}}", f"{N}/mean[{r}]", [R(sq, "out")], n=1)
        ve = add("AddScalarF32_v1{N=1,C=1e-06}", f"{N}/add[{r}]", [R(mu, "out")], n=1)
        rs = add(f"RsqrtF32_v1{{N={rsqrt_n}}}", f"{N}/rsqrt[{r}]", [R(ve, "out")], n=1)
        sc = add(f"ScaleRowBf16_v1{{N={H}}}", f"{N}/mul[{r}]", [R(-2, "x"), R(rs, "out")])
        mv = add(f"MulVecF32_v1{{N={H}}}", f"{N}/mul_1[{r}]", [R(sc, "out"), R(-1, f"{N}.weight")])
        nb = add(f"NarrowF32ToBf16_v1{{N={H}}}", f"{N}/to/out[{r}]", [R(mv, "out")], width=16)
        last = add(f"Gemm_v1{{K={H},N={H}}}", f"{W}.lm_head/linear[{r}]", [R(nb, "out"), R(-1, f"{W}.lm_head.weight")], width=16)
    add("ArgmaxTokenSelect_v1{N=1}", "argmax[0]", [R(last, "out")], n=1)
    return ProgramView(tuple(calls), (weights, xs), (R(k - 1, "out"),), "synthetic", "synthetic")


def _manifest(P, norm_scales=None, **kw):
    corr = Correspondence(P, implementation_paths=["model.lm_head"])
    return request_manifest(P, corr, RES, request_id="r0", norm_scales=NS.POLICY if norm_scales else norm_scales, **kw)


def _rows(man, family):
    return sorted((x["op_path"], x["output_member"], tuple(x["element_range"]), x.get("spec"), x.get("dtype")) for x in man["identities"]
                  if x["family"] == family)


@pytest.mark.parametrize("program", [kernel_program, chain_program])
def test_off_is_byte_identical(program):
    P = program()
    a, b = _manifest(P).manifest, _manifest(P, norm_scales=False).manifest
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert NS.FAMILY not in a["populations"]["by_family"] and not NS.policy_of(a) and "norm_scales" not in a["query"]


def test_kernel_norms_commit_one_scale_per_call_row():
    P = kernel_program(rows=2)
    off, on = _manifest(P).manifest, _manifest(P, norm_scales=True).manifest
    assert _rows(on, NS.FAMILY) == [("model.layers.0.input_layernorm", NS.MEMBER, (0, 2), "RMSNormTriton_v1", "f32"),
                                    ("model.norm", NS.MEMBER, (0, 2), "RMSNormFusedCuda_v2", "f32")]
    strip = lambda m: sorted(json.dumps(x, sort_keys=True) for x in m["identities"] if x["family"] != NS.FAMILY)  # noqa: E731
    assert strip(on) == strip(off), "the policy only adds identities"
    assert on["manifest_digest"] != off["manifest_digest"]
    q = on["query"]
    assert NS.policy_of(on) and q["required_families"][-1] == NS.FAMILY and q["query_id_digest"] != off["query"]["query_id_digest"]
    assert q["norm_scales"]["words"] == 4 and q["norm_scales"]["words_by_family"] == {"RMSNormFusedCuda_v2": 2, "RMSNormTriton_v1": 2}
    assert NS.manifest_modules(on) == {"model.layers.0.input_layernorm": "RMSNormTriton_v1", "model.norm": "RMSNormFusedCuda_v2"}
    assert NS.words_per_token(on) == 2 and NS.words_per_token(off) == 0


def test_an_aten_chain_norm_commits_its_rsqrt_output():
    """Under the module-body query the chain is interior to the norm's body and the policy promotes the rsqrt output alone.  Under Q_word
    (the record) every Call boundary of the chain is required whatever the policy says; the policy only spells the rsqrt output as the
    norm scale instead of a `call_boundaries` row."""
    P = chain_program(rows=3)
    off, on = _manifest(P, query=Q_MODULE_BODY_ID), _manifest(P, norm_scales=True, query=Q_MODULE_BODY_ID)
    rsqrt = {R(c.id, "out") for c in P.calls if c.family == "RsqrtF32_v1"}
    assert not rsqrt & off.required.values and rsqrt <= on.required.values
    assert _rows(on.manifest, NS.FAMILY) == [("model.layers.0.input_layernorm", NS.MEMBER, (0, 3), "RsqrtF32_v1", "f32")]
    interior = {R(c.id, "out") for c in P.calls if c.family in ("SquareBf16_v1", "MeanTriton_v1", "AddScalarF32_v1")}
    assert not interior & on.required.values, "the rest of the chain stays interior to the norm's body"
    w_off, w_on = _manifest(P), _manifest(P, norm_scales=True)
    assert rsqrt | interior <= w_off.required.values and rsqrt | interior <= w_on.required.values
    assert _rows(w_on.manifest, NS.FAMILY) == _rows(on.manifest, NS.FAMILY)
    assert not any(x["family"] == "call_boundaries" and x.get("spec") == "RsqrtF32_v1" for x in w_on.manifest["identities"])
    assert any(x["family"] == "call_boundaries" and x.get("spec") == "RsqrtF32_v1" for x in w_off.manifest["identities"])
    q = on.manifest["query"]
    assert q["norm_scales"]["words_by_family"] == {"RsqrtF32_v1": 3}
    assert NS.manifest_modules(on.manifest) == {"model.layers.0.input_layernorm": "RsqrtF32_v1"}


def test_a_vector_rsqrt_is_not_a_norm_scale():
    assert [c.family for c in NS.scale_calls(chain_program(rows=1))] == ["RsqrtF32_v1"]
    assert NS.scale_calls(chain_program(rows=1, rsqrt_n=8)) == []


def test_manifest_modules_refuses_a_module_under_two_families():
    man = _manifest(kernel_program(), norm_scales=True).manifest
    bad = copy.deepcopy(man)
    bad["identities"].append(dict(next(x for x in bad["identities"] if x["family"] == NS.FAMILY), spec="RMSNormFusedCuda_v2", step=5))
    with pytest.raises(ValueError, match="two norm families"):
        NS.manifest_modules(bad)
    bad["identities"][-1]["spec"] = "Gemm_v1"
    with pytest.raises(ValueError, match="carries no norm scale"):
        NS.manifest_modules(bad)


def test_the_family_is_in_every_vocabulary():
    from verity_vllm.acquire import plan
    from verity_vllm.check import oracle_compare as OC
    from verity_vllm.check.replay import coverage

    assert NS.FAMILY in fmt.FAMILIES and plan.ROUTE_OF_FAMILY[NS.FAMILY] == "norm_tap" and NS.FAMILY in coverage.QUERY_FAMILY_MECHANISM
    entry = {"identity": {"op_path": "model.norm", "output_member": NS.MEMBER}, "name": f"model.norm/{NS.MEMBER}"}
    assert OC._family_of(entry) == "hidden_stream", "the Match record holds no value for a norm's scale: form (B) never compares it"
    row = {"family": NS.FAMILY, "op_path": "model.norm", "output_member": NS.MEMBER, "step": 0}
    assert Binder([row]).bind(0, [f"model.norm/{NS.MEMBER}"]) == [("model.norm", NS.MEMBER, 0, "label")]


def test_the_plan_takes_the_scale_at_the_norm_module():
    from verity_vllm.acquire import plan as PL
    from verity_vllm.correspondence.reader_for_acquire import AnnotationCorrespondence, runtime_tree_from_paths

    man = _manifest(kernel_program(), norm_scales=True).manifest
    rt = runtime_tree_from_paths({"": ["M"], "model.layers.0.input_layernorm": ["N"], "model.norm": ["N"]}, buffers=[],
                                 runner={"sample": True}, source="synthetic")
    corr = AnnotationCorrespondence(rt, wrapper_prefix="", aliases={})
    vals = {v.value_id: v for v in PL.value_classes(man) if v.family == NS.FAMILY}
    assert sorted(vals) == [f"model.layers.0.input_layernorm/{NS.MEMBER}", f"model.norm/{NS.MEMBER}"]
    for v in vals.values():
        legal, refused = PL.sites_of(v, corr)
        assert [(s.kind, s.module, s.slot) for s in legal] == [("native_tap", v.op_path, NS.MEMBER)] and not refused
    gone = runtime_tree_from_paths({"": ["M"]}, buffers=[], runner={"sample": True}, source="synthetic")
    legal, refused = PL.sites_of(vals["model.norm/norm_scale"], AnnotationCorrespondence(gone, wrapper_prefix="", aliases={}))
    assert not legal and refused[0]["why"] == "the norm module is not a module of the runtime tree"


def test_word_check_sees_the_tapped_scale_as_committed():
    from verity.ml.definitions import word as W_

    g = {"family": "RMSNormTriton_v1", "statics": {"N": 64, "EPS": 1e-05}, "definition": "RMSNormTriton_v1{N=64,EPS=1e-05}", "calls": 3,
         "where": "model.layers.0.input_layernorm"}
    plain = W_.check_calls([g])
    assert "acquired_interior_words" not in plain and plain["committed_interior_words"] == 3, "without a tap the output is unchanged"
    tapped = W_.check_calls([g], acquired=NS.TAP_WORDS)
    assert tapped["ok"] and tapped["acquired_interior_words"] == 3 and tapped["groups"][0]["acquired_interior_words"] == 3
    with pytest.raises(W_.QueryRuleViolation, match="interior-not-acquired"):
        W_.check_calls([g], acquired={"RMSNormTriton_v1": 2})


def test_manifest_verify_rebuilds_under_the_callers_policy(tmp_path, monkeypatch):
    from verity_vllm.pipeline import manifest as M
    from verity_vllm.pipeline import manifest_verify as MV
    from verity_vllm.query.manifest import verify as V

    P = kernel_program()
    on = _manifest(P, norm_scales=True).manifest
    off = _manifest(P).manifest
    assert V.verify(on, on)["ok"] and not V.verify(on, off)["ok"] and not V.verify(off, on)["ok"]
    seen = {}
    monkeypatch.setattr(M, "build", lambda program, **kw: seen.setdefault("kw", kw) and on)
    MV.rebuild(program="p", norm_scales=True)
    assert seen["kw"]["norm_scales"] is True
