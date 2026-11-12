"""What the check's small readers make of a field their document leaves out.

Each read here is of a document a file brings: a checkpoint's config.json, the Build's roles and its result.json, a
replay's record, a verdict document, the checkpoints manifest's entry, a shard pin another process took, the required
manifest's rows, the facts of the Programs of record, the binding map.  The field may be absent, null or empty, and
each means the same thing."""
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
