"""What the check's small readers make of a field their document leaves out.

Each read here is of a document a file brings: a checkpoint's config.json, the Build's roles and its result.json, a
replay's record, a verdict document, the checkpoints manifest's entry, a shard pin another process took, the required
manifest's rows, the facts of the Programs of record, the binding map, a run directory's manifest and its artifacts,
and the reports each gate reads.  The field may be absent, null or empty, and each means the same thing."""
from __future__ import annotations

import json
import os
import re
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.check.test_gates_fixtures import gate, passing_artifacts, query_use_doc
from tests.check.test_oracle_compare import make_bmap, make_match, reader
from tests.check.test_oracle_compare_v2_producers import gemma_like_program
from tests.check.test_verdict import CLASSES, _base
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
ABSENT = object()   # a field its document leaves out, beside the null and the empty one
NONINT, CENSUS, RESOLUTION, FOLD, ROOTS = "nonint/noninterference.json", "census_reconcile.json", "resolution.json", "fold_summary.json", "root_policy.json"
REPLAY, WIRING, CHAIN, GOLDEN, HOLDOUTS, QUERY_USE = "replay_a.json", "wiring.json", "replay_chain.json", "golden.json", "holdouts.json", "query_use.json"
HOLDOUT = "qwen15_2req"
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


def instrumented(pair: int, commit_pass: Any) -> dict[str, Any]:
    """An instrumented run's line of runs.jsonl whose Commit record states `commit_pass`, or leaves it out."""
    commit = _base()
    if commit_pass is not ABSENT:
        commit["commit_pass"] = commit_pass
    return {"arm": "instrumented", "pair": pair, "tokens_equal_ref": True, "commit": commit}


@pytest.mark.parametrize("stated, outcome", [((True, True), V.PASS), ((True, False), V.FAIL), ((True, ABSENT), V.PASS), ((True, None), V.PASS)])
def test_a_verdict_document_that_states_no_decision_takes_the_runs_own_where_every_run_states_one(stated, outcome):
    v = V.build([instrumented(i, p) for i, p in enumerate(stated)], {"required_classes": list(CLASSES)})
    mismatch = ["RECONSTRUCTION MISMATCH: recorded commit_pass=False but the rule-by-rule outcome is PASS; the recorded decision stands"]
    assert (v.outcome, v.notes) == (outcome, mismatch if outcome == V.FAIL else [])


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


def unstated(*path: Any, empty: Any) -> list[tuple[tuple, Any]]:
    """The three ways a document states nothing at `path`: the field absent, null, and `empty`."""
    return [(path, ABSENT), (path, None), (path, empty)]


def gate_over(tmp_path, gid: str, rel: str, path: tuple, value: Any, arts: dict[str, Any] | None = None) -> G.GateResult:
    """Gate `gid` over the fixtures' passing run (or `arts`) whose artifact `rel` leaves `path` out or holds `value` there."""
    arts = passing_artifacts() if arts is None else arts
    at = arts[rel]
    for key in path[:-1]:
        at = at[key]
    if value is ABSENT:
        del at[path[-1]]
    else:
        at[path[-1]] = value
    return gate(tmp_path / "run", arts, gid)


@pytest.mark.parametrize("path, value", unstated("tokens", empty={}))
def test_a_noninterference_report_that_states_no_tokens_misses_every_variant(tmp_path, path, value):
    r = gate_over(tmp_path, "G1", NONINT, path, value)
    assert (r.status, r.reason) == ("fail", "tokens missing for variants ['a_observer', 'b_hooks', 'c_bare']")
    assert r.detail["noninterference"]["tokens_equal_control"] == {"a_observer": None, "b_hooks": None, "c_bare": None}


@pytest.mark.parametrize("path, value", unstated("tokens", "b_hooks", empty={}))
def test_a_variant_whose_tokens_the_report_leaves_out_is_missing_by_name(tmp_path, path, value):
    r = gate_over(tmp_path, "G1", NONINT, path, value)
    assert (r.status, r.reason) == ("fail", "tokens missing for variants ['b_hooks']")
    assert r.detail["noninterference"]["tokens_equal_control"] == {"a_observer": True, "b_hooks": None, "c_bare": True}


@pytest.mark.parametrize("path, value", unstated("steps", empty=[]))
def test_a_noninterference_report_that_states_no_step_compared_none(tmp_path, path, value):
    r = gate_over(tmp_path, "G1", NONINT, path, value)
    assert (r.status, r.reason) == ("fail", "0 steps, expected 4 (control run engine steps, else workload max_tokens)")
    assert r.detail["noninterference"]["steps"] == 0


@pytest.mark.parametrize("rel, path, value", [(NONINT, *case) for field in ("mismatches", "problems") for case in unstated(field, empty=[])]
                         + [(CENSUS, *case) for case in unstated("steps", empty=[])])
def test_a_report_that_lists_no_mismatch_problem_or_census_step_counts_none(tmp_path, rel, path, value):
    r = gate_over(tmp_path, "G1", rel, path, value)
    assert (r.status, r.reason) == ("pass", None)
    assert (r.detail["noninterference"]["n_mismatches"], r.detail["noninterference"]["n_problems"]) == (0, 0)


def test_the_mismatches_and_problems_a_report_lists_are_counted_and_fail(tmp_path):
    arts = passing_artifacts()
    arts[NONINT]["mismatches"] = [{"step": 1}, {"step": 2}]
    r = gate_over(tmp_path, "G1", NONINT, ("problems",), ["p1", "p2", "p3"], arts)
    assert (r.status, r.reason) == ("fail", "2 hash mismatches; 3 problems recorded: ['p1', 'p2']")
    assert (r.detail["noninterference"]["n_mismatches"], r.detail["noninterference"]["n_problems"]) == (2, 3)
    r = gate_over(tmp_path, "G1", CENSUS, ("steps",), [{"step": 0}, {"error": "no log step"}, "junk"])
    assert (r.status, r.reason) == ("fail", "1 census steps without a resolved log step")


@pytest.mark.parametrize("path, value", unstated("primary_report", empty={}) + unstated("primary_report", "unsupported_ops", empty={}))
def test_unsupported_resolutions_are_counted_without_the_ops_a_report_leaves_out(tmp_path, path, value):
    arts = passing_artifacts()
    arts[RESOLUTION]["totals"]["unsupported"] = 2
    r = gate_over(tmp_path, "G2", RESOLUTION, path, value, arts)
    assert (r.status, r.reason) == ("fail", "2 Unsupported resolutions ([])")


def test_unsupported_resolutions_name_the_first_ops_the_report_lists(tmp_path):
    arts = passing_artifacts()
    arts[RESOLUTION]["totals"]["unsupported"] = 6
    r = gate_over(tmp_path, "G2", RESOLUTION, ("primary_report", "unsupported_ops"), {f"aten::op{i}": 1 for i in range(6)}, arts)
    assert r.reason == "6 Unsupported resolutions (['aten::op0', 'aten::op1', 'aten::op2', 'aten::op3'])"


@pytest.mark.parametrize("path, value", unstated("totals", empty={}))
def test_a_resolution_report_without_totals_is_refused_by_the_first_count_it_lacks(tmp_path, path, value):
    r = gate_over(tmp_path, "G2", RESOLUTION, path, value)
    assert (r.status, r.reason) == ("error", "resolution: key 'totals.unsupported' missing")


def test_the_resolutions_compute_counts_are_carried_as_the_totals_state_them(tmp_path):
    r = gate_over(tmp_path, "G2", RESOLUTION, ("totals", "computes"), ABSENT)
    assert r.status == "pass" and (r.detail["computes"], r.detail["compute_rows"]) == (None, 1200)
    r = gate_over(tmp_path, "G2", RESOLUTION, ("totals", "compute_rows"), None)
    assert r.status == "pass" and (r.detail["computes"], r.detail["compute_rows"]) == (1000, None)


@pytest.mark.parametrize("rel, path, value, read", [(FOLD, *case, ("unknown_roots", [])) for case in unstated("unknown_roots", empty=[])]
                         + [(RESOLUTION, *case, ("unknown_roots", [])) for case in unstated("roots", empty={}) + unstated("roots", "unknown", empty=[])]
                         + [(ROOTS, *case, ("new_root_classes", [])) for case in unstated("new_root_classes", empty=[])]
                         + [(ROOTS, *case, ("classes", {})) for case in unstated("classes", empty={})]
                         + [(ROOTS, *case, ("roots", 0)) for case in unstated("roots", empty=[])]
                         + [(ROOTS, *case, ("alias_claims", 0)) for case in unstated("alias_claims", empty=[])])
def test_a_root_report_that_lists_nothing_under_a_field_has_none_of_it(tmp_path, rel, path, value, read):
    r = gate_over(tmp_path, "G3", rel, path, value)
    assert (r.status, r.reason) == ("pass", None)
    assert r.detail[read[0]] == read[1] and (r.detail["roots_failed"], r.detail["alias_unproven"]) == (0, 0)


def test_the_roots_a_report_lists_are_counted_and_fail(tmp_path):
    arts = passing_artifacts()
    arts[RESOLUTION]["roots"]["unknown"] = ["r1"]
    arts[ROOTS].update(new_root_classes=["c1"], alias_claims=[{"name": "al", "status": "open"}, {"name": "ok", "status": "pass"}, "junk"])
    r = gate_over(tmp_path, "G3", FOLD, ("unknown_roots",), ["u1", "u2"], arts)
    assert r.reason == ("fold: 2 unknown roots: ['u1', 'u2']; resolution: 1 unknown roots: ['r1']; 1 new root classes: ['c1']; "
                        "1 alias claims unproven: ['al']")
    assert (r.detail["unknown_roots"], r.detail["new_root_classes"], r.detail["alias_claims"], r.detail["alias_unproven"]) == (["u1", "u2"], ["c1"], 3, 1)


@pytest.mark.parametrize("res", [None, [], {}, {"primary_report": None}, {"primary_report": {}}, {"by_spec": None}, {"by_spec": {}},
                                 {"spec_modules": None, "primary_report": {"spec_modules": None}}])
def test_a_resolution_report_that_names_no_module_or_specialization_has_no_quarantine_spec(res):
    assert G.quarantine_specs(res) == []


def test_quarantine_specs_are_read_at_the_top_of_the_report_and_else_under_the_resolvers_own():
    own = {"spec_modules": {"A_v1": "registry.quarantine.a", "B_v1": "registry.core"}, "registries": {"C_v1": "quarantine"}}
    assert G.quarantine_specs({"primary_report": own}) == ["A_v1", "C_v1"]
    assert G.quarantine_specs({"spec_modules": {"D_v1": "quarantine"}, "primary_report": own}) == ["C_v1", "D_v1"]
    assert G.quarantine_specs({"quarantine_specs": ["Q_v1"], "by_spec": {"Quarantine_v1{N=2}": 3, "Gemm_v1": 1}}) == ["Q_v1", "Quarantine_v1{N=2}"]


@pytest.mark.parametrize("path, value, read", [(*case, {"specs": 2, "covered": 2, "uncovered_specs": []}) for case in unstated("uncovered_specs", empty=[])]
                         + [(*case, {"specs": 0, "covered": 0, "pass": 0, "fail": 0, "skipped": 0}) for case in unstated("coverage", empty={})])
def test_a_sample_report_that_lists_no_uncovered_spec_or_no_coverage_passes_on_its_status(tmp_path, path, value, read):
    r = gate_over(tmp_path, "G4", REPLAY, path, value)
    assert (r.status, r.reason) == ("pass", None)
    assert {k: r.detail[k] for k in read} == read


@pytest.mark.parametrize("path, value", unstated("differential", empty={}))
def test_a_quarantine_report_that_states_no_differential_count_fails(tmp_path, path, value):
    arts = passing_artifacts()
    arts["quarantine.json"] = {"status": "pass", "differential": {"passed": 12, "total": 12}, "negative_matcher_tests": "pass"}
    r = gate_over(tmp_path, "G4", "quarantine.json", path, value, arts)
    assert (r.status, r.reason) == ("fail", "quarantine differential tests: None/None (need total > 0)")
    assert r.detail["quarantine"]["differential"] == {"passed": None, "total": None}


@pytest.mark.parametrize("path, value", unstated("streams", empty={}))
def test_a_wiring_report_without_a_stream_census_fails_by_name(tmp_path, path, value):
    r = gate_over(tmp_path, "G5", WIRING, path, value)
    assert (r.status, r.reason) == ("fail", "wiring: stream census absent (run provenance with the raw log)")
    assert (r.detail["provenance"]["single_stream"], r.detail["provenance"]["streams"]) == (None, None)


@pytest.mark.parametrize("path, value", unstated("schema", empty={}))
def test_a_wiring_report_without_a_schema_census_counts_no_write_outside_it(tmp_path, path, value):
    r = gate_over(tmp_path, "G5", WIRING, path, value)
    assert (r.status, r.reason) == ("pass", None)
    assert [r.detail["provenance"][k] for k in ("writes_outside_schema", "schema_roots", "triton_roots", "declared_unwritten")] == [None] * 4


@pytest.mark.parametrize("path, value", unstated("failures", empty=[]))
def test_a_failed_wiring_report_that_names_no_failure_fails_on_its_status_alone(tmp_path, path, value):
    arts = passing_artifacts()
    arts[WIRING]["status"] = "fail"
    assert gate_over(tmp_path, "G5", WIRING, path, value, arts).reason == "wiring status 'fail': "
    arts[WIRING]["failures"] = ["f1", "f2"]
    assert gate(tmp_path / "run", arts, "G5").reason == "wiring status 'fail': f1; f2"


@pytest.mark.parametrize("path, value", unstated("tokens", empty={}))
def test_a_sample_report_without_a_tokens_section_fails_by_name(tmp_path, path, value):
    r = gate_over(tmp_path, "G5", REPLAY, path, value)
    assert (r.status, r.reason) == ("fail", "tier a report has no `tokens` section (served-token reproduction)")
    assert "tokens" not in r.detail


@pytest.mark.parametrize("path, value", unstated("tokens", "steps", empty=[]))
def test_served_tokens_checked_at_no_step_miss_the_prefill_and_the_decode_steps(tmp_path, path, value):
    r = gate_over(tmp_path, "G5", REPLAY, path, value)
    decode = f"served tokens checked at 0 decode step(s) < MIN_DECODE_TOKEN_STEPS={G.MIN_DECODE_TOKEN_STEPS}"
    assert (r.status, r.reason) == ("fail", f"served tokens: step 0 (prefill) not checked; {decode}")
    assert r.detail["tokens"]["steps"] == []


@pytest.mark.parametrize("path, value", unstated("tokens", "mismatch", empty=[]))
def test_served_tokens_without_a_mismatch_list_have_none(tmp_path, path, value):
    r = gate_over(tmp_path, "G5", REPLAY, path, value)
    assert (r.status, r.reason, r.detail["tokens"]["mismatch"]) == ("pass", None, [])


def test_the_served_tokens_a_report_lists_as_unequal_fail_and_the_first_five_are_kept(tmp_path):
    unequal = [{"step": s} for s in range(7)]
    r = gate_over(tmp_path, "G5", REPLAY, ("tokens", "mismatch"), unequal)
    assert r.reason == f"7 served token(s) not reproduced from the logits boundary: {unequal[:3]}"
    assert r.detail["tokens"]["mismatch"] == unequal[:5]


@pytest.mark.parametrize("path, value, read", [(*case, []) for case in unstated("slices", empty=[])]
                         + [(*case, [{"step": 0, "status": "pass", "edges": 5, "specs": []}]) for case in unstated("slices", 0, "specs", empty=[])])
def test_a_chain_report_that_lists_no_slice_or_a_slice_no_spec_passes_on_its_counts(tmp_path, path, value, read):
    r = gate_over(tmp_path, "G5", CHAIN, path, value)
    assert (r.status, r.reason) == ("pass", None)
    assert r.detail["chain"]["slices"][:1] == read and r.detail["chain"]["n_slices"] == 3


def test_a_chained_slice_without_an_edge_fails_and_its_specs_lose_their_parameters(tmp_path):
    r = gate_over(tmp_path, "G5", CHAIN, ("slices", 1, "edges_chained"), None)
    assert (r.status, r.reason) == ("fail", "a chained slice chained no producer->consumer edge")
    assert r.detail["chain"]["slices"][1] == {"step": 1, "status": "pass", "edges": None, "specs": ["Gemm_v1", "RMSNormFusedCuda_v2"]}


@pytest.mark.parametrize("stated, equal", [({}, None), ({"tokens": None}, None), ({"tokens": {}}, None), ({"tokens": {"equal": 4}}, 4)])
def test_a_full_replay_that_states_no_tokens_carries_no_equal_count(tmp_path, stated, equal):
    arts = passing_artifacts()
    arts[G.replay_b_rel(1)] = {"status": "pass", "n_instances": 57, **stated}
    r = gate(tmp_path / "run", arts, "G5")
    assert r.status == "pass" and r.detail["full_replay"]["1"]["tokens_equal"] == equal


@pytest.mark.parametrize("path, value", unstated("entries", empty=[]))
def test_a_golden_report_that_lists_no_entry_passes_on_its_status(tmp_path, path, value):
    r = gate_over(tmp_path, "G6", GOLDEN, path, value)
    assert (r.status, r.reason, r.detail["golden"]) == ("pass", None, {"status": "pass", "entries": 0, "changed": []})


@pytest.mark.parametrize("path, value", unstated("holdouts", empty=[]))
def test_a_holdouts_report_that_lists_none_fails(tmp_path, path, value):
    r = gate_over(tmp_path, "G7", HOLDOUTS, path, value)
    assert (r.status, r.reason) == ("fail", "no holdouts")
    assert (r.detail["n_holdouts"], r.detail["holdouts"]) == (0, {})


@pytest.mark.parametrize("path, value", unstated("holdouts", 0, "gates", empty={}))
def test_a_holdout_that_states_no_gate_passes_none(tmp_path, path, value):
    r = gate_over(tmp_path, "G7", HOLDOUTS, path, value)
    assert (r.status, r.reason) == ("fail", f"holdout {HOLDOUT}: {list(G.HOLDOUT_GATES)} not passing")
    assert r.detail["holdouts"][HOLDOUT]["gates"] == dict.fromkeys(G.HOLDOUT_GATES) and r.detail["holdouts"][HOLDOUT]["pass"] is False


@pytest.mark.parametrize("path, value", unstated("verdict", "partition", "codes", empty=[]) + unstated("verdict", "width", "violators", empty=[])
                         + unstated("verdict", "width", "unsupported", empty=[]) + unstated("verdict", "readiness_32", "violators", empty=[]))
def test_a_query_use_verdict_that_lists_no_code_violator_or_undecided_class_has_none(tmp_path, path, value):
    arts = passing_artifacts()
    arts[QUERY_USE] = query_use_doc()
    r = gate_over(tmp_path, "G8", QUERY_USE, path, value, arts)
    assert (r.status, r.reason) == ("pass", None)
    assert r.detail["partition"]["codes"] == [] and r.detail["readiness_32"] == {"ok": True, "max_w": 32, "n_violators": 0}
    assert [r.detail["width"][k] for k in ("n_violators", "violators", "n_unsupported", "unsupported")] == [0, [], 0, []]


def test_the_codes_violators_and_undecided_classes_a_query_use_verdict_lists_are_carried(tmp_path):
    arts = passing_artifacts()
    arts[QUERY_USE] = query_use_doc(partition="overlap", codes=["P1", "P2"], width="over", violators=["V_v1"], unsupported=["W_v1"])
    r = gate_over(tmp_path, "G8", QUERY_USE, ("verdict", "readiness_32", "violators"), ["V_v1", "X_v1"], arts)
    assert r.status == "fail" and r.reason.startswith("partition 'overlap' ['P1', 'P2']; width 'over'; 1 member class(es) with w_out > w_max=32: ['V_v1']")
    assert r.detail["partition"]["codes"] == ["P1", "P2"] and r.detail["readiness_32"]["n_violators"] == 2
    assert [r.detail["width"][k] for k in ("n_violators", "violators", "n_unsupported", "unsupported")] == [1, ["V_v1"], 1, ["W_v1"]]
    assert r.detail["undecided"] == ["1 member class(es) with undecidable w_out: ['W_v1']"]


@pytest.mark.parametrize("path, value", [*unstated("use", empty={}), (("use",), [G.QUERY_USE_KIND])])
def test_a_query_use_document_that_states_no_use_fails_on_its_kind(tmp_path, path, value):
    arts = passing_artifacts()
    arts[QUERY_USE] = query_use_doc()
    r = gate_over(tmp_path, "G8", QUERY_USE, path, value, arts)
    assert (r.status, r.reason) == ("fail", f"use.kind None != {G.QUERY_USE_KIND!r}")
    assert r.detail["use"] == {"kind": None, "w_max": None} and r.detail["width"]["w_max"] == query_use_doc()["verdict"]["width"]["w_max"]


@pytest.mark.parametrize("path, value", [*unstated("verdict", "readiness_32", empty={}), (("verdict", "readiness_32"), [True])])
def test_a_query_use_verdict_that_states_no_readiness_carries_none(tmp_path, path, value):
    arts = passing_artifacts()
    arts[QUERY_USE] = query_use_doc()
    r = gate_over(tmp_path, "G8", QUERY_USE, path, value, arts)
    assert (r.status, r.reason, r.detail["readiness_32"]) == ("pass", None, None)
