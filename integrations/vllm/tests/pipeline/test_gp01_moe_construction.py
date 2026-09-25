"""[R17-42 / R17-45 (a) route A (3), lane moe] GP-01's `moe_construction` header block (`global_program.workload_moe_construction`): the
fused-MoE construction is a declared-and-bound Program static of the workload -- read from the manifest's target, coupled to every component
(its derive-time `target.declared` AND its `spec_histogram` Definitions), never attested.  Fail closed, by name:

  - undeclared -> indexed-read (MOE-01), the pre-R17 rows unchanged (no key in the target, no digest change);
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
    return {"target": target} if target else {}


def test_undeclared_is_indexed_read_and_moe01_components_compose_as_before(tmp_path):
    shapes = {(256, 31): component(tmp_path, "c0", MOE01, {"compute_capability": [8, 9], "num_sms": 142})}
    out = GP.workload_moe_construction(manifest(), shapes)
    assert out["refusal"] is None and out["value"] == "indexed-read" and out["declared_explicitly"] is False and out["attested"] is False
    assert out["moe_block_present"] is True and out["components"]["256,31"]["binds_declared_construction"] is True
    assert out["components"]["256,31"]["moe_construction"] == "indexed-read" and out["components"]["256,31"]["declared_explicitly"] is False
    assert "MOE-01" in out["population_label"] or "indexed-read" in out["population_label"]
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
    decl = {"compute_capability": [8, 9], "num_sms": 142}
    if workload == "padded":
        decl["moe_construction"] = "padded"
    shapes = {(256, 31): component(tmp_path, "c0", hist, decl)}
    out = GP.workload_moe_construction(manifest(**({"moe_construction": "padded"} if workload == "padded" else {})), shapes)
    assert out["refusal"] and other in out["refusal"] and "refused by name" in out["refusal"]
    assert other in out["components"]["256,31"]["binds_other_construction"]
    # and with no target.declared at all (pre-v3 component) the Definitions alone still refuse it
    shapes = {(256, 31): component(tmp_path, "c1", hist, None)}
    out = GP.workload_moe_construction(manifest(**({"moe_construction": "padded"} if workload == "padded" else {})), shapes)
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
    out = GP.workload_moe_construction({"target": {"moe_constructon": "padded"}}, shapes)
    assert out["value"] is None and "does not parse" in out["refusal"]
    out = GP.workload_moe_construction({"target": {"moe_construction": "MOE-01"}}, shapes)
    assert out["value"] is None and "does not parse" in out["refusal"]
    shapes = {(256, 31): component(tmp_path, "c1", PADDED, {"compute_capability": [8, 9], "num_sms": 142, "moe_construction": "fixed-slot"})}
    out = GP.workload_moe_construction(manifest(moe_construction="padded"), shapes)
    assert out["refusal"] and "target.declared does not parse" in out["refusal"]


def test_gm01_carries_the_block_from_the_artifact_and_labels_a_pre_r17_42_artifact_by_its_absence(tmp_path):
    """[route A (4) fix, found on the first padded GM-01 run] `global_match.from_gprog` must carry GP-01's `moe_construction` block into the
    checker's Program dict (the block GM-01 REPORTS beside the verdict; never compared with the record).  Before this fix the block was dropped
    on the way and a padded Program was labelled "pre-R17-42 Build" by GM-01 -- the row's population label would have been the wrong one."""
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
    p.write_text(json.dumps(art))                                                                                   # a pre-R17-42 artifact: no block
    assert GM.from_gprog(str(p))["moe_construction"] is None
