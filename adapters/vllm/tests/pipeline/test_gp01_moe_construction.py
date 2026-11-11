"""GP-01's `moe_construction` header block (`global_program.workload_moe_construction`): the
fused-MoE construction is a declared-and-bound Program static of the workload -- read from the manifest's target, coupled to every component
(its derive-time `target.declared` AND its `spec_histogram` Definitions), never attested.  Fail closed, by name:

  - undeclared -> indexed-read-ordered (MOE-01, the router in selection order); a declared `indexed-read` composes the kernel-order rows;
  - `padded` with components that bind `MoeFfnFixedL_v1` -> the block, `moe_block_present`, the padded population label;
  - a component derived under the OTHER construction (target.declared) -> refused, both directions;
  - a component whose Definitions are the other family's (spec_histogram) -> refused, both directions (the digest would be another row's);
  - `padded` on a workload with no MoE block anywhere -> refused (idle declaration);
  - a misspelt / foreign target field -> refused (parse), not absorbed.
"""
from __future__ import annotations

import json
import os

import pytest

from verity_vllm.pipeline import global_program as GP

MOE01 = {"Gemm_v1{K=2048,N=64}": 16, "MoeExpertGemm_v1{E=64,K=2048,N=2048}": 128, "MoeExpertGemmW_v1{E=64,K=1024,N=2048}": 128,
         "MoeRouterTopK_v1{E=64,TOPK=8}": 16, "MoeSum_v1{TOPK=8,H=2048}": 16, "RmsNorm_v1{H=2048}": 33}
PADDED = {"Gemm_v1{K=2048,N=64}": 16, "MoeFfnFixedL_v1{E=64,TOPK=8,VPT=8,H=2048,I=1024}": 16, "RmsNorm_v1{H=2048}": 33}
ORDERED = {**{k: v for k, v in MOE01.items() if not k.startswith("MoeRouterTopK_v1")}, "MoeRouterTopKOrdered_v1{E=64,TOPK=8}": 16}
DENSE = {"Gemm_v1{K=2048,N=8192}": 16, "RmsNorm_v1{H=2048}": 33}


def component(tmp_path, name: str, hist: dict | None, declared: dict | None):
    d = tmp_path / name
    d.mkdir()
    res = {"ok": True}
    if hist is not None:
        res["spec_histogram"] = hist
    if declared is not None:
        res["target"] = {"declared": declared}
    (d / "result.json").write_text(json.dumps(res))
    return str(d)


def manifest(**target):
    return {"engine": {"target": target}} if target else {}


def test_undeclared_is_indexed_read_ordered_and_declared_indexed_read_composes_as_before(tmp_path):
    shapes = {(256, 31): component(tmp_path, "c0", ORDERED, {"compute_capability": [8, 9], "num_sms": 142})}
    out = GP.workload_moe_construction(manifest(), shapes)
    assert out["refusal"] is None and out["value"] == "indexed-read-ordered" and out["declared_explicitly"] is False and out["attested"] is False
    assert out["moe_block_present"] is True and out["components"]["256,31"]["binds_declared_construction"] is True
    assert out["components"]["256,31"]["moe_construction"] == "indexed-read-ordered" and out["components"]["256,31"]["declared_explicitly"] is False
    assert "MOE-01" in out["population_label"] and "indexed-read-ordered" in out["population_label"]
    # the kernel-order router is the other construction's under the default (refused by name) and composes when declared
    out = GP.workload_moe_construction(manifest(), {(256, 31): component(tmp_path, "k0", MOE01, None)})
    assert out["refusal"] and "MoeRouterTopK_v1" in out["refusal"] and "the indexed-read construction" in out["refusal"]
    decl = {"compute_capability": [8, 9], "num_sms": 142, "moe_construction": "indexed-read"}
    out = GP.workload_moe_construction(manifest(moe_construction="indexed-read"), {(256, 31): component(tmp_path, "k1", MOE01, decl)})
    assert out["refusal"] is None and out["value"] == "indexed-read" and out["declared_explicitly"] is True and out["moe_block_present"] is True
    # a dense workload under the default is fine: no MoE block, nothing declared, nothing idle
    dense = {(256, 31): component(tmp_path, "d0", DENSE, None)}
    out = GP.workload_moe_construction(manifest(), dense)
    assert out["refusal"] is None and out["moe_block_present"] is False and "note" in out["components"]["256,31"]


def test_padded_declared_and_bound_is_the_block_with_the_actual_population_label(tmp_path):
    decl = {"compute_capability": [8, 9], "num_sms": 142, "moe_construction": "padded"}
    shapes = {(256, 31): component(tmp_path, "c0", PADDED, decl), (1024, 127): component(tmp_path, "c1", PADDED, decl)}
    out = GP.workload_moe_construction(manifest(moe_construction="padded"), shapes)
    assert out["refusal"] is None and out["value"] == "padded" and out["declared_explicitly"] is True and out["attested"] is False
    assert out["moe_block_present"] is True and "ACTUAL population" in out["population_label"]
    assert all(c["moe_construction"] == "padded" and c["binds_declared_construction"] and not c["binds_other_construction"]
               for c in out["components"].values())
    assert "MoeFfnFixedL_v1{E,TOPK,VPT,H,I}" in out["definitions"] and "LMoeFfnL_v1" in out["lift_of_record"]
    assert "R17-42" in out["rule"] and "never attested" in out["rule"]


@pytest.mark.parametrize("workload, comp", [("padded", "indexed-read"), ("indexed-read", "padded")])
def test_a_component_derived_under_the_other_construction_is_refused_by_name(tmp_path, workload, comp):
    hist = PADDED if workload == "padded" else MOE01           # the Definitions agree with the workload; the derive-time declaration does not
    shapes = {(256, 31): component(tmp_path, "c0", hist, {"compute_capability": [8, 9], "num_sms": 142, "moe_construction": comp})}
    out = GP.workload_moe_construction(manifest(moe_construction=workload), shapes)
    assert out["refusal"] and f"derived under moe_construction={comp!r}" in out["refusal"] and f"says {workload!r}" in out["refusal"]
    assert out["components"]["256,31"]["moe_construction"] == comp


@pytest.mark.parametrize("workload, hist, other", [("padded", MOE01, "MoeExpertGemm_v1"), ("indexed-read", PADDED, "MoeFfnFixedL_v1")])
def test_a_component_whose_definitions_are_the_other_familys_is_refused_by_name(tmp_path, workload, hist, other):
    decl = {"compute_capability": [8, 9], "num_sms": 142, "moe_construction": workload}
    shapes = {(256, 31): component(tmp_path, "c0", hist, decl)}
    out = GP.workload_moe_construction(manifest(moe_construction=workload), shapes)
    assert out["refusal"] and other in out["refusal"] and "refused by name" in out["refusal"]
    assert other in out["components"]["256,31"]["binds_other_construction"]
    # and with no target.declared at all (pre-v3 component) the Definitions alone still refuse it
    shapes = {(256, 31): component(tmp_path, "c1", hist, None)}
    out = GP.workload_moe_construction(manifest(moe_construction=workload), shapes)
    assert out["refusal"] and other in out["refusal"]


def test_an_idle_padded_declaration_on_a_dense_workload_is_refused(tmp_path):
    decl = {"compute_capability": [8, 9], "num_sms": 142, "moe_construction": "padded"}
    shapes = {(256, 31): component(tmp_path, "c0", DENSE, decl)}
    out = GP.workload_moe_construction(manifest(moe_construction="padded"), shapes)
    assert out["refusal"] and "idle declaration" in out["refusal"] and out["moe_block_present"] is False
    # the same dense components under the default compose (nothing declared, nothing idle)
    assert GP.workload_moe_construction(manifest(), {(256, 31): component(tmp_path, "c1", DENSE, {"compute_capability": [8, 9], "num_sms": 142})})["refusal"] is None


def test_a_foreign_or_misspelt_field_is_refused_not_absorbed(tmp_path):
    shapes = {(256, 31): component(tmp_path, "c0", PADDED, {"compute_capability": [8, 9], "num_sms": 142, "moe_construction": "padded"})}
    out = GP.workload_moe_construction(manifest(moe_constructon="padded"), shapes)
    assert out["value"] is None and "does not parse" in out["refusal"]
    out = GP.workload_moe_construction(manifest(moe_construction="MOE-01"), shapes)
    assert out["value"] is None and "does not parse" in out["refusal"]
    shapes = {(256, 31): component(tmp_path, "c1", PADDED, {"compute_capability": [8, 9], "num_sms": 142, "moe_construction": "fixed-slot"})}
    out = GP.workload_moe_construction(manifest(moe_construction="padded"), shapes)
    assert out["refusal"] and "target.declared does not parse" in out["refusal"]


def test_gm01_carries_the_block_from_the_artifact_and_labels_a_pre_r17_42_artifact_by_its_absence(tmp_path):
    """`global_match.from_gprog` carries GP-01's `moe_construction` block into the checker's Program dict (the block GM-01 REPORTS beside
    the verdict; never compared with the record), and None for an artifact without it."""
    from verity_vllm.check.match import global_match as GM
    block = {"schema": "verity/workload-program/v1/moe_construction", "value": "padded", "declared_explicitly": True, "attested": False,
             "moe_block_present": True, "population_label": "ACTUAL population |Q(Lift(P_fixed-slot))|"}
    art = {"schema": "verity/workload-program/v1", "ok": True, "workload_digest": "00" * 32, "workload_fn": "Workload_v1",
           "requests": [{"index": 0, "request_id": "r0", "prompt_len": 1024, "cap": 128, "arrive_step": 0}],
           "inputs": {"derived": {}}, "manifest_digest": "11" * 32}
    p = tmp_path / "workload_program.json"
    p.write_text(json.dumps({**art, "moe_construction": block}))
    gp = GM.from_gprog(str(p))
    assert gp["moe_construction"] == block and gp["gprog_artifact"]["digest_verification"]["verified"] is False   # carried verbatim; unverified (no descriptor)
    p.write_text(json.dumps(art))                                                                                   # an artifact without the block
    assert GM.from_gprog(str(p))["moe_construction"] is None
