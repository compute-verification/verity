"""What the check's small readers make of a field their document leaves out.

Each read here is of a document a file brings: a checkpoint's config.json, the Build's roles and its result.json, a
replay's record, a verdict document.  The field may be absent, null or empty, and each means the same thing."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from verity_vllm.check import cublas_route_pin, fp8_moe_scales
from verity_vllm.check import fp8_ue8m0 as U
from verity_vllm.check import verdict as V
from verity_vllm.check.component_programs import post_load_tensors

NO_BLOCK_SIZE = [{}, {"quantization_config": None}, {"quantization_config": {}}]
PINNED = "GemmF32Out_v1"


@pytest.mark.parametrize("config", NO_BLOCK_SIZE)
def test_block_scales_of_a_checkpoint_that_states_no_block_size_are_refused_by_name(config):
    scales = "m.experts.0.gate_proj.weight_scale_inv"
    with pytest.raises(KeyError, match=r"no square weight_block_size \(got None\)"):
        fp8_moe_scales.on_rank_grid([np.zeros((2, 2), np.float32)], [scales], "w13", SimpleNamespace(config=config), 8, {})


@pytest.mark.parametrize("config", [*NO_BLOCK_SIZE, {"quantization_config": {"weight_block_size": None}}])
def test_a_checkpoint_that_states_no_block_size_has_no_field_under_the_rewrite(config):
    field = "m.layers.0.self_attn.q_b_proj.weight"
    rw = U.Rewrites([field], {field: {"role": U.REQUANT_ROLES[0]}}, lambda f: None, config)
    assert rw.G == 0 and rw.requant == {field} and not rw.covers(field)


def test_a_field_the_build_gave_no_role_takes_no_rewrite():
    requant, absorb, unnamed, null, roleless = "m.a.weight", "m.b.W_UV", "m.c.weight", "m.d.weight", "m.e.weight"
    roles = {requant: {"role": U.REQUANT_ROLES[0]}, absorb: {"role": U.ABSORB_ROLE}, null: None, roleless: {"rule": "1:1"}}
    rw = U.Rewrites([requant, absorb, unnamed, null, roleless], roles, lambda f: None, {"quantization_config": {"weight_block_size": [128, 128]}})
    assert rw.requant == {requant} and rw.absorb == {absorb: "m.b.kv_b_proj.weight"}
    assert [rw.covers(f) for f in (requant, absorb, unnamed, null, roleless)] == [True, True, False, False, False]


def test_a_build_whose_result_states_no_post_load_tensors_registers_none(tmp_path):
    results = [{}, {"post_load_tensors": None}, {"post_load_tensors": {"applies": False}},
               {"post_load_tensors": {"applies": True, "parameters": None}}, {"post_load_tensors": {"applies": True, "parameters": {}}},
               {"post_load_tensors": {"applies": True, "parameters": {"m.x": {"shape": [2, 2], "dtype": "float32"}}}}]
    for r, doc in enumerate(results):
        d = tmp_path / f"rank{r}" / "build_request_LP4_T2"
        d.mkdir(parents=True)
        (d / "instances.json.gz").write_bytes(b"")
        (d / "result.json").write_text(json.dumps(doc))
    names = post_load_tensors({r: str(tmp_path / f"rank{r}") for r in range(len(results))})
    assert names == {"0": [], "1": [], "2": [], "3": [], "4": [], "5": ["m.x"]}


@pytest.mark.parametrize("sr", [None, {}, {"population": None, "unreplayable": None}, {"population": {}, "unreplayable": []},
                                {"population": {"by_family": None}}, {"population": {"by_family": {}}}])
def test_a_replay_record_that_names_no_family_binds_no_pinned_route(sr):
    assert cublas_route_pin.route_pin_gate(sr, rank=0) == (None, None)


def test_the_pinned_families_are_counted_from_the_population_and_the_unreplayable_alike():
    def families(sr: dict) -> dict[str, int]:
        block, _why = cublas_route_pin.route_pin_gate(sr, rank=0)
        assert block is not None
        return block["families"]
    assert families({"population": {"by_family": {PINNED: 2, "Gemm_v2": 9}}}) == {PINNED: 2}
    assert families({"unreplayable": [{"family": f"{PINNED} (no evaluator)", "n": 3}, {"family": PINNED, "n": None}]}) == {PINNED: 3}
    assert families({"population": {"by_family": {PINNED: 2}}, "unreplayable": [{"family": PINNED, "n": 3}]}) == {PINNED: 5}


@pytest.mark.parametrize("doc", [{}, {"checks": None}, {"checks": []}])
def test_a_verdict_document_without_checks_reads_as_no_checks(doc):
    v = V.CommitChecks.from_dict({"outcome": V.PASS, **doc})
    assert v.checks == [] and v.outcome == V.PASS


def test_a_wrapper_result_without_a_program_block_states_no_correspondence_digest():
    digest = "c" * 64
    silent: dict[str, Any] = {"a": None, "b": {}, "c": {"program": None}, "d": {"program": {}}, "e": {"program": {"correspondence_digest": None}}}
    assert V.correspondence_of_build(None, result_jsons=silent) == (None, "absent", {})
    assert V.correspondence_of_build(None, result_jsons=None) == (None, "absent", {})
    stated = {**silent, "request": {"program": {"correspondence_digest": digest}}}
    assert V.correspondence_of_build(None, result_jsons=stated) == (digest, "result.json:request", {"request": digest})
