"""`verity_vllm.records.schema_ids`: each constant holds the id its records on disk carry, so an id can be read from here
and never moves."""
from __future__ import annotations

from verity_vllm.observe import events, runtime_tree
from verity_vllm.records import schema_ids

#: the ids as the records on disk have them
ON_DISK = {
    "AOT_COMPARISON_SCHEMA": "verity-vllm/aot-comparison/v1",
    "BATCH_DECOMP_SCHEMA": "cb-a/batch-decomp/v1",
    "CAPTURE_LOG_SCHEMA": "verity-capture/log/v1",
    "CAPTURE_SNAPSHOTS_SCHEMA": "verity-capture/snapshots/v1",
    "GLOBAL_PROGRAM_SCHEMA": "cb-a/global-program/v1",
    "MANIFEST_VERIFY_SCHEMA": "verity-sweep/manifest-verify/v1",
    "MOE_PLANE_ALIAS_CHECK_SCHEMA": "verity-sweep/moe-plane-alias-check/v1",
    "NODE_REGISTRATION_SCHEMA": "verity-vllm/node-registration/v1",
    "PHI_SCHEMA": "verity-predict/phi/v1",
    "POUW_H1_CALLS_SCHEMA": "verity-vllm/pouw-h1-calls/v2",
    "POUW_H1_HANDOFF_SCHEMA": "verity-vllm/pouw-h1-handoff/v1",
    "POUW_NCP_V2_REPLAY_SCHEMA": "verity-vllm/pouw-ncp-v2-replay/v3",
    "PREDICT_NODE_SCHEMA": "verity-predict/node/v1",
    "PROGRAM_COMPARE_SCHEMA": "cb-a/compare/v1",
    "RANK_MATCH_SCHEMA": "tp2/rank_match/v1",
    "RUNTIME_TREE_SCHEMA": "verity-vllm/runtime-tree/v1",
    "SAMPLED_REPLAY_SCHEMA": "verity-sweep/sampled-replay/v1",
    "TP2_COMMIT_BINDING_SCHEMA": "verity/tp2/commit-binding/v1",
    "X09_PROFILE_SCHEMA": "cb-a/x09-profile/v1",
}


def test_every_constant_is_the_id_on_disk_and_the_module_holds_no_other():
    held = {name: value for name, value in vars(schema_ids).items() if name.isupper()}
    assert held == ON_DISK
    assert len(set(held.values())) == len(held)


def test_the_observation_reads_its_ids_from_the_module():
    assert events.SCHEMA is schema_ids.CAPTURE_LOG_SCHEMA
    assert runtime_tree.RUNTIME_TREE_SCHEMA is schema_ids.RUNTIME_TREE_SCHEMA
    assert runtime_tree.RuntimeTree().to_json()["schema"] == "verity-vllm/runtime-tree/v1"
