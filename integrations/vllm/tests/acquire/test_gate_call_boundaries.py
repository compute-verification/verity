"""`gate call-boundaries` on a row dir after its Build, torch-free: a row passes when every `call_boundaries` identity of its manifest has
a source its Commit attaches -- covered by the call-boundary source's host plan over the Build dirs, or claimed by an attached source --
and stops naming the first one without (an identity the plan cannot cover, a claim naming a source the Commit does not attach, any call
boundary of a TP manifest); a manifest naming none passes."""
from __future__ import annotations

import json

from tests.acquire.call_boundary_programs import LP, PF, build_dir, dense_program
from verity_vllm.acquire import gate as G
from verity_vllm.acquire.sources import call_boundary_source as CB
from verity_vllm.acquire.sources import taps
from verity_vllm.pipeline import cli


def _row_dir(tmp_path, **kw):
    """A row dir after its Build: the dense stand-in's Build dir and its manifest.json."""
    b, _at = dense_program(**kw)
    _d, man = build_dir(tmp_path, b.doc(LP), [PF[len("model."):]])
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    return man


def _gate(tmp_path, capsys):
    rc = cli.main(["gate", "call-boundaries", str(tmp_path), "--out", str(tmp_path / "call_boundaries.json")])
    return rc, capsys.readouterr().out.strip().splitlines()[-1], json.loads((tmp_path / "call_boundaries.json").read_text())


def _cb(man):
    return sum(1 for r in man["identities"] if r["family"] == CB.FAMILY)


def test_a_row_whose_every_call_boundary_the_plan_covers_passes(tmp_path, capsys):
    man = _row_dir(tmp_path)
    n = _cb(man)
    rc, line, rec = _gate(tmp_path, capsys)
    assert rc == 0 and rec["pass"] is True and n > 0
    assert (rec["identities"], rec["claimed"], rec["covered"], rec["uncovered"], rec["unattached"]) == (n, 0, n, 0, 0)
    assert rec["sources"] == ["call_boundary"] and rec["program_dirs"] == {"r0": str(tmp_path / "build_request")}
    assert line.startswith(f"identities={n} claimed=0 covered={n} uncovered=0 unattached=0 sources=call_boundary wall=") and "first:" not in line


def test_an_identity_the_plan_cannot_cover_stops_the_row_by_name(tmp_path, capsys):
    man = _row_dir(tmp_path, embed=PF)
    plan = CB.plan_of(man, str(tmp_path))
    assert plan.uncovered
    rc, line, rec = _gate(tmp_path, capsys)
    assert rc == 1 and rec["pass"] is False
    assert (rec["identities"], rec["covered"], rec["uncovered"]) == (_cb(man), plan.covered, len(plan.uncovered))
    assert rec["named"]["uncovered"][0] == plan.uncovered[0]
    assert f"uncovered={len(plan.uncovered)} " in line and line.endswith(f"first: {plan.uncovered[0]}")
    assert line.split("first: ")[1].startswith(f"{PF[len('model.'):]}/") and "reads Program input 'prompt'" in line


def test_a_claim_naming_a_source_the_commit_does_not_attach_stops_the_row_by_name(tmp_path, capsys, monkeypatch):
    man = _row_dir(tmp_path)
    logits = [r for r in man["identities"] if r["family"] == CB.FAMILY and r["op_path"].startswith("logits_processor")]
    assert logits
    monkeypatch.setattr(CB, "CLAIMS", [lambda r: "logits_tap" if r["op_path"].startswith("logits_processor") else None])
    rc, line, rec = _gate(tmp_path, capsys)
    n = _cb(man)
    assert rc == 1 and rec["pass"] is False
    assert (rec["claimed"], rec["covered"], rec["uncovered"], rec["unattached"]) == (len(logits), n - len(logits), 0, len(logits))
    r = logits[0]
    assert line.endswith(f"first: {r['op_path']}/{r['output_member']} of {r['request_id']} @ engine step {r['engine_step']}: claimed by "
                         f"logits_tap, a source the Commit does not attach")


def test_a_claim_naming_an_attached_source_passes(tmp_path, monkeypatch):
    man = _row_dir(tmp_path)
    man = {**man, "identities": [*man["identities"], {"family": "norm_scales"}]}
    assert "norm_scale" in taps.attached(man)
    monkeypatch.setattr(CB, "CLAIMS", [lambda r: "norm_scale" if r["op_path"].startswith("logits_processor") else None])
    rec = G.call_boundaries(man, str(tmp_path))
    assert rec["pass"] is True and rec["claimed"] > 0 and rec["covered"] == rec["identities"] - rec["claimed"]
    monkeypatch.setattr(CB, "CLAIMS", [lambda r: "norm_scale"])
    rec = G.call_boundaries(man, str(tmp_path))
    assert rec["pass"] is True and rec["claimed"] == rec["identities"] and rec["sources"] == ["norm_scale"]


def test_a_manifest_naming_no_call_boundaries_passes(tmp_path, capsys):
    (tmp_path / "manifest.json").write_text(json.dumps({"identities": [{"family": "instance_outputs"}, {"family": "weights"}]}))
    rc, line, rec = _gate(tmp_path, capsys)
    assert rc == 0 and rec["pass"] is True and rec["identities"] == 0
    assert line.startswith("identities=0 claimed=0 covered=0 uncovered=0 unattached=0 sources=none wall=")


def test_every_call_boundary_of_a_tp_manifest_stops_the_row(tmp_path):
    man = {**_row_dir(tmp_path), "ranks": 2}
    rec = G.call_boundaries(man, str(tmp_path))
    assert rec["pass"] is False and rec["uncovered"] == rec["identities"] == _cb(man) and "call_boundary" not in rec["sources"]
    assert rec["named"]["uncovered"][0].endswith(": a manifest of 2 ranks (the source serves one rank)")


def _sparse_mla_manifest(model="zai-org/GLM-5.3", ranks=8):
    """GLM-5.3's TP manifest shape: one SparseMla call boundary per (rank, layer, engine step), its rows 256 at prefill and 1 after."""
    ids = [{"family": CB.FAMILY, "request_id": "r0", "step": s, "engine_step": s, "invocation": 0, "rank": k, "output_member": "out",
            "op_path": f"inner.model.model.layers.{lay}.self_attn/dsa_indexer_attn" + (f"_{s * 2 + lay}" if s * 2 + lay else ""),
            "dtype": "bf16", "spec": "SparseMlaSplit_v1", "element_range": [0, (256 if s == 0 else 1) * 4096]}
           for k in range(ranks) for s in range(2) for lay in range(2)]
    return {"ranks": ranks, "fixture": {"model": model}, "identities": ids}


def test_the_attention_source_claims_every_sparse_mla_call_boundary_of_glm53_tp8(tmp_path):
    man = _sparse_mla_manifest()
    assert "attention" in taps.attached(man)
    rec = G.call_boundaries(man, str(tmp_path))
    n = len(man["identities"])
    assert rec["pass"] is True and (rec["identities"], rec["claimed"], rec["uncovered"], rec["unattached"]) == (n, n, 0, 0)
    assert rec["sources"] == ["attention"]


def test_a_sparse_mla_call_boundary_without_an_attached_attention_source_stops_the_row(tmp_path):
    for man in (_sparse_mla_manifest(model="other/model"), _sparse_mla_manifest(ranks=4)):
        assert "attention" not in taps.attached(man)
        rec = G.call_boundaries(man, str(tmp_path))
        assert rec["pass"] is False and rec["unattached"] == rec["identities"] == len(man["identities"])
        assert rec["named"]["unattached"][0].endswith("claimed by attention, a source the Commit does not attach")


#: DeepSeek-V4-Pro's Marlin block as its TP manifest of record spells it: (fx node, Definition, port) of each call boundary of a layer
MARLIN_BLOCK = (("marlin_moe_mxfp4_gemm", "MarlinMoeExpertGemm_v1", "out"), ("silu_and_mul_with_clamp_1", "SiluMulClamp_v1", "out"),
                ("marlin_moe_mxfp4_gemm_1", "MarlinMoeExpertGemmW_v1", "out"), ("moe_sum", "MoeSum_v1", "out"),
                ("triton_launch", "MoeRouterSqrtSoftplus_v1", "0"), ("triton_launch", "MoeRouterSqrtSoftplus_v1", "1"))


def _marlin_moe_manifest(ranks=8, planes=False):
    """A hash layer (its router `MoeRouterHash_v1`) and a scored one (`dsv4_topk`, two ports) of the Marlin block per (rank, engine
    step), and no `moe_block_stream` plane unless `planes`."""
    layers = {0: (("topk_softplus_sqrt", "MoeRouterHash_v1", "out"), *MARLIN_BLOCK[:4]), 3: MARLIN_BLOCK}
    ids = [{"family": CB.FAMILY, "request_id": "r0", "step": s, "engine_step": s, "invocation": 0, "rank": k, "dtype": "bf16",
            "op_path": f"model.layers.{lay}.ffn.experts/{node}", "output_member": port, "port": port, "spec": spec, "element_range": [0, 8]}
           for k in range(ranks) for s in range(2) for lay, calls in layers.items() for node, spec, port in calls]
    return {"ranks": ranks, "identities": ids + ([{"family": "moe_block_stream"}] if planes else [])}


def test_the_moe_source_claims_every_marlin_block_call_boundary_of_dsv4_tp8(tmp_path):
    man = _marlin_moe_manifest()
    assert "moe" in taps.attached(man)
    rec = G.call_boundaries(man, str(tmp_path))
    n = len(man["identities"])
    assert rec["pass"] is True and (rec["identities"], rec["claimed"], rec["uncovered"], rec["unattached"]) == (n, n, 0, 0)
    assert rec["sources"] == ["moe"]


def test_a_marlin_block_call_boundary_the_moe_source_binds_no_label_of_is_not_its(tmp_path):
    man = _marlin_moe_manifest()
    man["identities"][0] = {**man["identities"][0], "port": "1", "output_member": "1"}
    assert CB.claimed(man["identities"][0]) is None
    rec = G.call_boundaries(man, str(tmp_path))
    assert rec["pass"] is False and rec["claimed"] == rec["identities"] - 1 and rec["uncovered"] == 1


def test_a_marlin_block_call_boundary_without_the_moe_source_at_calls_stops_the_row(tmp_path):
    for man in (_marlin_moe_manifest(planes=True), _marlin_moe_manifest(ranks=1)):
        assert "moe" not in taps.attached(man)
        rec = G.call_boundaries(man, str(tmp_path))
        n = sum(1 for r in man["identities"] if r["family"] == CB.FAMILY)
        assert rec["pass"] is False and rec["unattached"] == rec["identities"] == n
        assert rec["named"]["unattached"][0].endswith("claimed by moe, a source the Commit does not attach")
