"""`verity_vllm.records.schema_ids`: each constant holds the id its records on disk carry, so an id can be read from here
and never moves."""
from __future__ import annotations

from verity_vllm.observe import events
from verity_vllm.records import schema_ids

#: the ids as the records on disk have them
ON_DISK = {
    "AOT_COMPARISON_SCHEMA": "verity-vllm/aot-comparison/v1",
    "BATCH_DECOMP_SCHEMA": "cb-a/batch-decomp/v1",
    "BOUNDARY_LINKAGE_SCHEMA": "verity-sweep/boundary-linkage/v1",
    "BUILD_RELABEL_GUARD_SCHEMA": "verity-vllm/build-relabel-guard/v1",
    "BUILD_RELABEL_SCHEMA": "verity-vllm/build-relabel/v1",
    "CALL_BOUNDARIES_GATE_SCHEMA": "verity-vllm/call-boundaries-gate/v1",
    "CAPTURE_LOG_SCHEMA": "verity-capture/log/v1",
    "CAPTURE_SNAPSHOTS_SCHEMA": "verity-capture/snapshots/v1",
    "CENSUS_RANK_SCHEMA": "verity-capture/census-rank/v0",
    "COVERAGE_DECLARATION_SCHEMA": "verity/commit-integ/coverage-declaration/v1",
    "ENGINE_PROFILE_SCHEMA": "verity-vllm/profile/v1",
    "ENGINE_SPEC_SCHEMA": "verity-vllm/engine-spec/v0",
    "GLOBAL_PROGRAM_SCHEMA": "cb-a/global-program/v1",
    "IMPORT_PATH_RUNTIME_SCHEMA": "verity/tp2/import-path-runtime/v1",
    "INT8_ERROR_REPORT_SCHEMA": "verity-vllm/int8-error-report/v1",
    "INT8_K_SCALES_SCHEMA": "verity-vllm/int8-k-scales/v1",
    "LATE_READ_CHECK_SCHEMA": "verity/commit-integ/t6_4-check/v1",
    "MANIFEST_VERIFY_SCHEMA": "verity-sweep/manifest-verify/v1",
    "MOE_PLANE_ALIAS_CHECK_SCHEMA": "verity-sweep/moe-plane-alias-check/v1",
    "NODE_REGISTRATION_SCHEMA": "verity-vllm/node-registration/v1",
    "PHI_SCHEMA": "verity-predict/phi/v1",
    "POUW_CIRCUIT_REPLAY_SCHEMA": "verity-vllm/pouw-ncp-v2-replay/v3",
    "POUW_H1_CALLS_SCHEMA": "verity-vllm/pouw-h1-calls/v2",
    "POUW_H1_HANDOFF_SCHEMA": "verity-vllm/pouw-h1-handoff/v1",
    "PREDICT_NODE_SCHEMA": "verity-predict/node/v1",
    "PROFILE_FALLBACK_SCHEMA": "verity-capture/profile-fallback/v1",
    "PROGRAM_COMPARE_SCHEMA": "cb-a/compare/v1",
    "RANK_COMMIT_BINDING_SCHEMA": "verity/tp2/commit-binding/v1",
    "RANK_MATCH_SCHEMA": "tp2/rank_match/v1",
    "SAMPLED_REPLAY_SCHEMA": "verity-sweep/sampled-replay/v1",
    "SERVED_SWITCHES_SCHEMA": "verity-vllm/served-switches/v1",
    "STATIC_SCHEDULE_SCHEMA": "verity-vllm/static-schedule/v1",
    "TP_FOLD_MATCH_SCHEMA": "verity/tp2/fold-match/v1",
    "TP_LINKS_SCHEMA": "cb-a/tp_links/v2",
    "TP_PARAM_HASHES_SCHEMA": "verity-capture/tp2-param-hashes/v0",
    "TP_RANK_REPLAY_SCHEMA": "verity-vllm/tp-rank-replay/v1",
    "TP_REPLAY_DEFERRED_SCHEMA": "verity-vllm/tp-replay-deferred/v1",
    "TP_SAMPLED_REPLAY_SCHEMA": "verity-sweep/tp-sampled-replay/v1",
    "X09_PROFILE_SCHEMA": "cb-a/x09-profile/v1",
    "XRANK_PICKS_SCHEMA": "verity/tp2/xrank-picks/v1",
}


def test_every_constant_is_the_id_on_disk_and_the_module_holds_no_other():
    held = {name: value for name, value in vars(schema_ids).items() if name.isupper()}
    assert held == ON_DISK
    assert len(set(held.values())) == len(held)


def test_the_observation_reads_its_ids_from_the_module():
    assert events.SCHEMA is schema_ids.CAPTURE_LOG_SCHEMA
