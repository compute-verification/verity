"""What the check's small readers make of a field their document leaves out.

Each read here is of a document a file brings: a checkpoint's config.json, the Build's roles and its result.json, a
replay's record, a verdict document, the checkpoints manifest's entry, a shard pin another process took, the required
manifest's rows, the facts of the Programs of record, the binding map, a run directory's manifest and its artifacts.
The field may be absent, null or empty, and each means the same thing."""
from __future__ import annotations

import json
import os
import re
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.check.test_oracle_compare import make_bmap, make_match, reader
from tests.check.test_oracle_compare_v2_producers import gemma_like_program
from tests.check.test_weights_of_record import write_safetensors
from verity_vllm.check import checkpoint_index as CI
from verity_vllm.check import cublas_route_pin, fp8_moe_scales
from verity_vllm.check import fp8_ue8m0 as U
from verity_vllm.check import gates as G
from verity_vllm.check import oracle_compare as OC
from verity_vllm.check import producers as PR
from verity_vllm.check import verdict as V
from verity_vllm.check.component_programs import post_load_tensors
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.query import program_view

NO_BLOCK_SIZE = [{}, {"quantization_config": None}, {"quantization_config": {}}]
PINNED = "GemmF32Out_v1"
SHARD = "model-00001-of-00001.safetensors"
ENTRY = {"repo": "test/absent", "revision": "abc"}
ROW = {"family": "instance_outputs", "op_path": "model", "output_member": "out", "spec": "X_v1{N=4}"}
NO_ENTRIES: list[dict[str, Any]] = [{}, {"entries": None}, {"entries": []}]
NO_IDENTITY: list[dict[str, Any]] = [{"name": "x"}, {"identity": None, "name": "x"}, {"identity": {}, "name": "x"}, {"identity": {"request_id": "r0"}, "name": "x"}]
HOST_UNSTATED = {"cpu": None, "nproc": None, "cpu_quota": None, "clocksource": None, "clock_read_us": None}
FOREIGN = {"requests": [{"request_id": "cccccccc-2222-2222-2222-222222222222", "index": 0, "prompt_len": 5, "max_tokens": 1}]}


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


@pytest.fixture
def one_shard(tmp_path):
    """A checkpoint of one shard: its directory and the shard's sha256."""
    d = str(tmp_path / "snap")
    os.makedirs(d)
    write_safetensors(os.path.join(d, SHARD), {"w": ("F32", np.zeros((2, 2), np.float32))})
    return d, CI.sha256_file(os.path.join(d, SHARD))


def pin_under(tmp_path, d: str, entry: dict[str, Any] | None) -> dict[str, Any]:
    """`shard_pin` of `d` against a manifest whose entry for it states `entry` beside its repo, revision and path (None: no entry)."""
    manifest = tmp_path / "checkpoints.json"
    manifest.write_text(json.dumps({"checkpoints": [] if entry is None else [{**ENTRY, "local_path": d, **entry}]}))
    return CI.shard_pin(d, checkpoints_manifest=str(manifest), model=None, revision=None)


@pytest.mark.parametrize("entry", [{}, {"files": None}, {"files": {}}, {"files": {SHARD: None}}, {"files": {SHARD: {}}}, {"files": {SHARD: {"sha256": None}}},
                                   {"files": [SHARD]}, {"safetensors_sha256": None}, {"safetensors_sha256": {}}, {"files": {}, "safetensors_sha256": {}}])
def test_a_manifest_entry_that_states_no_sha256_for_a_shard_pins_none(tmp_path, one_shard, entry):
    d, sha = one_shard
    assert pin_under(tmp_path, d, entry) == {"manifest": str(tmp_path / "checkpoints.json"), "entry": ENTRY,
                                             "shards": {SHARD: {"sha256": sha, "pinned_sha256": None, "pinned": None}}}


def test_a_shard_is_pinned_by_the_entrys_files_and_else_by_its_sha256_by_name(tmp_path, one_shard):
    d, sha = one_shard
    other = "0" * 64
    assert pin_under(tmp_path, d, None) == {"manifest": str(tmp_path / "checkpoints.json"), "entry": None,
                                            "shards": {SHARD: {"sha256": sha, "pinned_sha256": None, "pinned": None}}}
    stated: list[tuple[dict[str, Any], str]] = [
        ({"files": {SHARD: {"sha256": sha}}}, sha), ({"safetensors_sha256": {SHARD: sha}}, sha), ({"files": {}, "safetensors_sha256": {SHARD: sha}}, sha),
        ({"files": [SHARD], "safetensors_sha256": {SHARD: sha}}, sha), ({"files": {SHARD: {}}, "safetensors_sha256": {SHARD: sha}}, sha),
        ({"files": {SHARD: {"sha256": sha}}, "safetensors_sha256": {SHARD: other}}, sha), ({"files": {SHARD: {"sha256": other}}}, other)]
    for entry, pinned in stated:
        assert pin_under(tmp_path, d, entry)["shards"][SHARD] == {"sha256": sha, "pinned_sha256": pinned, "pinned": pinned == sha}, entry


@pytest.mark.parametrize("pin", [{}, {"shards": None}, {"shards": {}}, {"shards": []}])
def test_a_pin_that_states_no_shard_covers_none_and_is_refused_by_name(one_shard, pin):
    d, _sha = one_shard
    with pytest.raises(ValueError, match=re.escape(f"the pin covers shards [], the checkpoint has ['{SHARD}'] (0 vs 1)")):
        CI.CheckpointIndex(d, pin=pin)


def test_a_pin_of_more_shards_is_refused_with_both_counts(one_shard):
    d, sha = one_shard
    shards = {name: {"sha256": sha, "pinned_sha256": None, "pinned": None} for name in ("z.safetensors", SHARD)}
    with pytest.raises(ValueError, match=re.escape(f"the pin covers shards ['{SHARD}', 'z.safetensors'], the checkpoint has ['{SHARD}'] (2 vs 1)")):
        CI.CheckpointIndex(d, pin={"shards": shards})
    assert CI.CheckpointIndex(d, pin={"shards": {SHARD: shards[SHARD]}}).pin == {"manifest": None, "entry": None, "shards": {SHARD: shards[SHARD]}}


@pytest.mark.parametrize("manifest", [None, {}, {"identities": None}, {"identities": []}])
def test_a_manifest_without_rows_names_no_producer_and_promotes_nothing(manifest):
    facts = {("model", "out"): {"spec": "X_v1", "consumers": ["model.layers.0"]}}
    assert PR.producers_of(manifest) == {} and PR.runtime_rows_of(manifest) == {}
    assert PR.promote_handed_down(manifest, facts) == {"members": {}, "rows": 0}


@pytest.mark.parametrize("named", [{}, {"consumers": None}, {"consumers": []}])
def test_a_promoted_row_that_names_no_consumer_adds_none(named):
    row = {**ROW, "promoted": True, **named}
    producer = {"spec": "X_v1", "ordinal": None, "operand": None, "promoted": True}
    assert PR.producers_of({"identities": [row]}) == {("model", "out"): {**producer, "consumers": []}}
    rows = [{**row, "consumers": ["a", "b"]}, row, {**row, "consumers": ["b", "c"]}]
    assert PR.producers_of({"identities": rows}) == {("model", "out"): {**producer, "consumers": ["a", "b", "c"]}}


@pytest.mark.parametrize("fact", [{}, {"consumers": None}, {"consumers": []}])
def test_a_fact_that_names_no_consumer_hands_nothing_down(fact):
    manifest = {"identities": [dict(ROW)]}
    assert PR.promote_handed_down(manifest, {("model", "out"): {"spec": "X_v1", **fact}}) == {"members": {}, "rows": 0}
    assert manifest == {"identities": [ROW]}
    readers = ["model.layers.0"]
    assert PR.promote_handed_down(manifest, {("model", "out"): {"spec": "X_v1", "consumers": readers}}) == {"members": {"model/out": readers}, "rows": 1}
    assert manifest == {"identities": [{**ROW, "promoted": True, "consumers": readers}]}


@pytest.mark.parametrize("built", [{}, {"result": None}, {"result": {}}, {"result": {"tp": None}}, {"result": {"tp": {}}}])
def test_a_program_handed_over_without_a_result_gives_the_facts_of_its_rows(monkeypatch, built):
    P = gemma_like_program()
    monkeypatch.setattr(program_view, "from_instances", lambda instances: P)
    assert PR.producers_of_programs({"r0": {"instances": object(), **built}}) == PR.program_producer_facts(P, Correspondence(P))


@pytest.fixture
def matched(tmp_path):
    """A Match record, a binding map of its values and a reader of the committed bytes equal to them."""
    d, vals = make_match(str(tmp_path))
    bmap, committed = make_bmap(vals)
    return d, bmap, reader(committed)


@pytest.mark.parametrize("bmap", NO_ENTRIES)
def test_a_binding_map_without_entries_compares_nothing(matched, bmap):
    d, full, read = matched
    res = OC.oracle_compare(OC.MatchOracle(d), bmap, read)
    assert (res["result"], res["grade"], res["compared"], res["not_compared"], res["steps_compared"]) == (None, "partial", 0, {}, [])
    assert OC.oracle_compare(OC.MatchOracle(d), bmap, read, workload=FOREIGN)["not_compared"] == {"foreign oracle": 0}
    assert OC.oracle_compare(OC.MatchOracle(d), full, read, workload=FOREIGN)["not_compared"] == {"foreign oracle": len(full["entries"])}
    assert OC.same_name_invocations(bmap) == {}


def test_an_entry_that_binds_no_identity_or_no_step_is_skipped(matched):
    d, bmap, read = matched
    with_them = {"entries": bmap["entries"] + NO_IDENTITY}
    assert OC.oracle_compare(OC.MatchOracle(d), with_them, read) == OC.oracle_compare(OC.MatchOracle(d), bmap, read)
    assert OC.same_name_invocations(with_them) == OC.same_name_invocations(bmap) and OC.same_name_invocations({"entries": NO_IDENTITY}) == {}


@pytest.mark.parametrize("stated", [{}, {"encoding": None}, {"encoding": {}}, {"encoding": {"dtype": None}}, {"encoding": {"shape": [2]}}])
def test_an_entry_that_states_no_dtype_is_counted_as_without_a_byte_range(matched, stated):
    d, bmap, read = matched
    good = next(e for e in bmap["entries"] if e["name"] == "model.embed_tokens" and e["identity"]["request_id"] == "r0")
    entry = {**{k: v for k, v in good.items() if k != "encoding"}, "identity": {**good["identity"], "invocation": 7}, **stated}
    base = OC.oracle_compare(OC.MatchOracle(d), bmap, read)
    res = OC.oracle_compare(OC.MatchOracle(d), {"entries": bmap["entries"] + [entry]}, read)
    assert res["not_compared"] == {**base["not_compared"], "identity without a byte range (dtype/range)": 1}
    assert (res["compared"], res["equal"], res["mismatch_n"]) == (base["compared"], base["equal"], base["mismatch_n"])


def test_the_invocations_bound_under_a_name_are_counted_per_step():
    entries = [{"identity": {"step": 0}, "name": "a"}, {"identity": {"step": 0, "invocation": 1}, "name": "a"}, {"identity": {"step": 0, "invocation": None}, "name": "a"},
               {"identity": {"step": 1}, "name": None}, {"identity": {"step": 1}}, {"identity": {"step": 2, "invocation": 3}, "name": "b"}]
    assert OC.same_name_invocations({"entries": entries}) == {(0, "a"): 2, (2, "b"): 1}


def run_of(tmp_path, manifest: dict[str, Any] | None = None, **artifacts: Any) -> G.RunArtifacts:
    """A run directory that holds `artifacts` (each by its key of `gates.ARTIFACTS`) under the run manifest `manifest`."""
    for key, doc in artifacts.items():
        path = tmp_path / G.ARTIFACTS[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc))
    return G.RunArtifacts(tmp_path, manifest={} if manifest is None else manifest)


@pytest.mark.parametrize("stated", [{}, {"evidence": None, "detail": None}, {"evidence": [], "detail": {}}])
def test_a_gate_result_read_without_evidence_or_detail_has_none(stated):
    result = G.GateResult.from_json({"id": G.OBSERVATION, "status": "pass", **stated})
    assert (result.evidence, result.detail, result.reason, result.name) == ([], {}, None, G.GATE_NAMES[G.OBSERVATION])
    evidence, detail = ["capture/tokens.json"], {"steps": 2}
    read = G.GateResult.from_json({"id": G.OBSERVATION, "status": "pass", "evidence": evidence, "detail": detail})
    assert read.evidence == evidence and read.evidence is not evidence and read.detail == detail and read.detail is not detail


@pytest.mark.parametrize("manifest", [{}, {"artifacts": None}, {"artifacts": {}}, {"artifacts": {"tokens": None}}, {"artifacts": {"tokens": ""}}])
def test_an_artifact_the_manifest_does_not_place_is_at_its_default_path(tmp_path, manifest):
    assert G.RunArtifacts(tmp_path, manifest=manifest).path("tokens") == tmp_path / G.ARTIFACTS["tokens"]
    placed = G.RunArtifacts(tmp_path, manifest={"artifacts": {"tokens": "x/t.json", "stats": "/elsewhere/s.json"}})
    assert (placed.rel("tokens"), placed.rel("stats"), placed.rel("log")) == ("x/t.json", "/elsewhere/s.json", G.ARTIFACTS["log"])


@pytest.mark.parametrize("manifest", [{}, {"workload": None}, {"workload": {}}, {"workload": {"requests": None}}, {"workload": {"requests": []}}])
def test_a_manifest_that_states_no_request_takes_them_from_the_tokens(tmp_path, manifest):
    requests = [{"request_id": "a", "prompt_token_ids": [1, 2, 3], "max_tokens": 4}, {"request_id": "b"},
                {"request_id": "c", "prompt_token_ids": None}, {"request_id": "d", "prompt_token_ids": [], "max_tokens": 5}]
    assert G.RunArtifacts(tmp_path, manifest=manifest).workload_requests() == []
    assert run_of(tmp_path, manifest, control_tokens={"requests": requests}).workload_requests() == [
        {"request_id": "a", "prompt_len": 3, "max_tokens": 4}, {"request_id": "b", "prompt_len": 0, "max_tokens": None},
        {"request_id": "c", "prompt_len": 0, "max_tokens": None}, {"request_id": "d", "prompt_len": 0, "max_tokens": 5}]
    stated = {"workload": {"requests": [{"request_id": "z", "max_tokens": 2}]}}
    assert G.RunArtifacts(tmp_path, manifest=stated).workload_requests() == stated["workload"]["requests"]


@pytest.mark.parametrize("timing", [{}, {"timing": None}, {"timing": {}}, {"timing": {"engine_steps": None}}])
def test_a_control_run_that_states_no_engine_steps_expects_the_largest_max_tokens(tmp_path, timing):
    requests = [{"request_id": "a", "max_tokens": 4}, {"request_id": "b"}, {"request_id": "c", "max_tokens": 5}]
    assert run_of(tmp_path, control_tokens={"requests": requests, **timing}).expected_steps() == 5
    assert G.RunArtifacts(tmp_path / "empty", manifest={}).expected_steps() is None


@pytest.mark.parametrize("steps", [0, 3])
def test_a_control_run_that_states_its_engine_steps_expects_them(tmp_path, steps):
    control = {"requests": [{"request_id": "a", "max_tokens": 4}], "timing": {"engine_steps": steps}}
    assert run_of(tmp_path, control_tokens=control).expected_steps() == steps


def test_a_control_request_that_states_no_generated_id_generated_none(tmp_path):
    requests = [{"request_id": "a", "generated_token_ids": [7, 8]}, {"request_id": "b"}, {"request_id": "c", "generated_token_ids": None},
                {"request_id": "d", "generated_token_ids": []}]
    assert run_of(tmp_path, control_tokens={"requests": requests}).control_tokens() == ({"a": [7, 8], "b": [], "c": [], "d": []}, "control/tokens.json")


@pytest.mark.parametrize("report", [{}, {"tokens": None}, {"tokens": {}}, {"tokens": {"c_bare": None}}, {"tokens": {"c_bare": {}}}, []])
def test_a_run_without_control_tokens_or_a_bare_variant_has_no_control(tmp_path, report):
    assert run_of(tmp_path, noninterference=report).control_tokens() == (None, "none")
    assert G.RunArtifacts(tmp_path / "empty", manifest={}).control_tokens() == (None, "none")


def test_the_bare_variant_stands_in_for_the_control_tokens(tmp_path):
    assert run_of(tmp_path, noninterference={"tokens": {"c_bare": {"a": [1, 2]}}}).control_tokens() == ({"a": [1, 2]}, "noninterference c_bare")


@pytest.mark.parametrize("histogram", [{}, {"spec_histogram": None}, {"spec_histogram": {}}])
def test_a_decode_step_that_states_no_histogram_has_an_empty_one(histogram):
    decode = {"phase": "decode", "n_instances": 3, **histogram}
    read = G.diag_structure({"per_step": [{"phase": "prefill", "step": 0, "n_instances": 2}, {**decode, "step": 1}, {**decode, "step": 2}]})
    assert (read["status"], read["decode_multiset_mod_T"], read["histograms_equal_mod_T"], read["histogram_diffs"]) == ("ok", {}, True, [])
    differs = G.diag_structure({"per_step": [{**decode, "step": 1}, {"phase": "decode", "step": 2, "n_instances": 3, "spec_histogram": {"A{T=2}": 1}}]})
    assert differs["histograms_equal_mod_T"] is False and differs["histogram_diffs"] == [{"step": 2, "differs": {"A{}": [0, 1]}}]


@pytest.mark.parametrize("classes", [{}, {"classes": None}, {"classes": {}}])
def test_a_structure_report_that_states_no_class_has_none(classes):
    read = G.diag_family_match({"structure": {"status": "ok", **classes}})
    assert (read["classes"], read["unclassified"], read["unclassified_classes"]) == ({}, 0, [])
    stated = G.diag_family_match({"structure": {"status": "ok", "classes": {"unclassified:x": {"count": 2}, "a": 1}}})
    assert (stated["classes"], stated["unclassified"], stated["unclassified_classes"]) == ({"unclassified:x": 2, "a": 1}, 1, ["unclassified:x"])


@pytest.mark.parametrize("versions, recorded", [(None, {}), (None, {"versions": None}), (None, {"versions": {}}), (None, {"versions": {"host": None}}),
                                                ({}, {"versions": {"host": {}}}), ({"host": None}, {})])
@pytest.mark.parametrize("reps", [{}, {"reps": None}, {"reps": []}])
def test_an_overhead_record_that_states_no_host_and_no_repetition_has_no_rung(versions, recorded, reps):
    read = G.diag_overhead({"results": {"r0": {"label": "x", **reps}}, "baseline_step_wall_s": 1.0, **recorded}, versions)
    assert (read["status"], read["host"], read["rungs"]) == ("ok", HOST_UNSTATED, {})


def test_the_overhead_ladder_reads_the_host_from_the_capture_and_else_from_its_own_record():
    m0 = {"results": {"r0": {"label": "x", "reps": [{"step_wall_mean_s": 2.0, "recorded_calls": 4, "observed_steps": 2}, "junk", {"step_wall_mean_s": None}]}},
          "baseline_step_wall_s": 1.0, "versions": {"host": {"cpu": "recorded", "clock_read_us": 0.5}}}
    assert G.diag_overhead(m0, {"host": {"cpu": "captured"}})["host"] == {**HOST_UNSTATED, "cpu": "captured"}
    own = G.diag_overhead(m0, None)
    assert own["host"] == {**HOST_UNSTATED, "cpu": "recorded", "clock_read_us": 0.5}
    assert own["rungs"] == {"r0": {"label": "x", "step_wall_ms": 2000.0, "x_baseline": 2.0, "events_per_step": 2.0, "overhead_per_event_us": 500000.0,
                                   "overhead_per_event_us_minus_clock": 499998.0}}
