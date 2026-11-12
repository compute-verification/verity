"""The schema ids of the records that have no type in this package yet: the string a record carries as ``schema``.

Each id is written once, here; its writers and readers read the constant.  Beside each: the record, and what writes it.
An id leaves this module when its record gets a dataclass in this package, whose ``SCHEMA`` then owns it.  One constant
is no ``schema``: `SERVED_REQUIRED_MANIFEST_KIND`, the ``kind`` a served Build carries beside its ``schema``.
"""
from __future__ import annotations

# --- the observation of one eager run (`verity_vllm.observe`)
#: the raw observation log (its `RunHeader.schema`), which `observe.capture.make_header` writes
CAPTURE_LOG_SCHEMA = "verity-capture/log/v1"
#: a capture's snapshot index, `snapshots.json`, which `observe.capture.Capture._write_snapshot_index` writes
CAPTURE_SNAPSHOTS_SCHEMA = "verity-capture/snapshots/v1"

# --- the query (`verity_vllm.query`)
#: whether a Commit used the manifest of record, which `query.manifest.verify.verify` and `acquire.compiled`'s `main`
#: write
MANIFEST_VERIFY_SCHEMA = "verity-sweep/manifest-verify/v1"

# --- the acquisition (`verity_vllm.acquire`)
#: a committer's coverage declaration (`acquire.committer_api.DECLARATION_SCHEMA`), which `declaration` writes
COVERAGE_DECLARATION_SCHEMA = "verity/commit-integ/coverage-declaration/v1"
#: the record of `verity-vllm gate call-boundaries` (`acquire.gate.CALL_BOUNDARIES_SCHEMA`), which `call_boundaries_of`
#: writes
CALL_BOUNDARIES_GATE_SCHEMA = "verity-vllm/call-boundaries-gate/v1"
#: what the acquisition plan's hooks saw in one run, occurrence counts and shapes only, which
#: `acquire.install.OccurrenceSink.to_json` builds and the capture writes as `acquisition_observed.json`
ACQUISITION_OBSERVED_SCHEMA = "verity-vllm/acquisition-observed/v1"
#: the `kind` of a served Build (`acquire.served.KIND`), a manifest of `query.manifest.format.SCHEMA`, which
#: `acquire.served.assemble` writes
SERVED_REQUIRED_MANIFEST_KIND = "verity/served-required-manifest/v1"

# --- the Commit (`verity_vllm.commit`)
#: a dumped store's `store.json` (`commit.store_dump.SCHEMA`), which `dump_files` writes and `load` refuses any other
#: than
COMMITTED_STORE_SCHEMA = "verity-vllm/committed-store/v1"
#: a window's `index.json` (`commit.index.INDEX_SCHEMA`), which `index_doc` writes
SERVING_ROWS_INDEX_SCHEMA = "verity-vllm/serving-rows/index/v0"
#: the reconciliation of what a Commit did not execute, which `commit.executed_prefix.reconcile_not_executed` writes
NOT_EXECUTED_RECONCILIATION_SCHEMA = "verity-sweep/not-executed-reconciliation/v3"
#: a capture map (`commit.capture_map.SCHEMA`), a JSON under `manifests/capture-maps/`, which `load` refuses any other
#: than
CAPTURE_MAP_SCHEMA = "verity-vllm/capture-map/v0"

# --- the Commit's driver (`verity_vllm.pipeline`)
#: the Commit plan's document (`pipeline.commit_plan.SCHEMA`), which `derive` writes; `use` takes no plan of another
COMMIT_PLAN_SCHEMA = "verity-vllm/commit-plan/v1"
#: a slim keep's `slim.json` (`pipeline.cpu_replay.SLIM_SCHEMA`), which `keep` writes and `open_keep` refuses any other
#: than
REPLAY_SLIM_SCHEMA = "verity-vllm/replay-slim/v1"
#: a replay bundle's `context.json` and its manifest (`pipeline.replay_bundle.SCHEMA`), which `seal` writes;
#: `open_bundle` refuses a manifest with any other
REPLAY_BUNDLE_SCHEMA = "verity-vllm/replay-bundle/v1"
#: a capture's `spec_tokens.json`, the workload as known before any engine step, which `pipeline.m1_capture`'s `main`
#: writes
SPEC_TOKENS_SCHEMA = "verity-gen/spec-tokens/v1"
#: a row's admission plan, the host memory reserved for each part of the row (`pipeline.telemetry.admission.SCHEMA`),
#: which `predict` writes
ADMISSION_PLAN_SCHEMA = "verity/admission-plan/v1"
#: the Commit's admission bound (`pipeline.admission_bound.SCHEMA`): a launch's predicted host bytes against the
#: container's limit, which the Commit's driver writes as `admission_<kind>.json`, and the record of the scheduled steps
#: inside it
ADMISSION_BOUND_SCHEMA = "verity-sweep/admission/f-da-15/v1"
#: a leased Commit's lease record (`pipeline.gpu_lease.run`): when the GPU was requested, held and released, and why it
#: was kept where it was; the row's driver writes it as `gpu_lease.json` beside the row
GPU_LEASE_SCHEMA = "verity-vllm/gpu-lease/v1"

# --- the Build (`verity_vllm.pipeline`)
#: the Build's result, a derive dir's `result.json`, which `pipeline.build.run` makes
DERIVE_STEP_SCHEMA = "cb-a/derive_step/v3"
#: `construction_manifest.json`, what the construction read and which rules fired, which `pipeline.build._write_manifest`
#: writes
CONSTRUCTION_MANIFEST_SCHEMA = "cb-a/construction_manifest/v3"
#: the construction's version (its rules and the hash of its sources), which `pipeline.build.construction_version`
#: returns and the Build's result holds
CONSTRUCTION_VERSION_SCHEMA = "cb-a/aot-artifact/v1"
#: the head of a derive dir's instance sequence, `instances.json.gz`, which `pipeline.build_instances.write_instances`
#: writes
INSTANCE_SEQUENCE_SCHEMA = "cb-a/instance-sequence/v2"
#: an entry of the Program cache (`pipeline.program_cache.SCHEMA`)
PROGRAM_CACHE_ENTRY_SCHEMA = "verity-vllm/program-cache-entry/v1"
#: a generated workload's plan (`pipeline.workload.SWEEP_SCHEMA`), which `plan` returns; not a row's workload file,
#: whose id `records.workload.SCHEMA` owns
GENERATED_WORKLOAD_SCHEMA = "verity-sweep/workload/v1"
#: neutrality's report, `neutrality.json`, which `pipeline.neutrality`'s `main` writes
NEUTRALITY_REPORT_SCHEMA = "verity-vllm/neutrality/v1"

# --- the Match's driver (`verity_vllm.pipeline`)
#: a Match run's manifest, `run.json`, which `pipeline.match.write_manifest` writes
MATCH_RUN_SCHEMA = "verity-gen/run/v1"
#: a Match run's `stages.json`, which `pipeline.match.save_stages` writes
MATCH_STAGES_SCHEMA = "verity-gen/stages/v1"
#: the realized Program's `program.json` (`pipeline.match.PROGRAM_SCHEMA`), which `persist_program` writes
REALIZED_PROGRAM_SCHEMA = "verity-gen/program/v1"
#: the head of the realized Program's `instances.jsonl` in its record form (`pipeline.match.INSTANCES_SCHEMA`), which
#: `persist_program` writes
REALIZED_INSTANCES_SCHEMA = "verity-gen/instances/v1"
#: the same head in the compact form, operand references as progressions (`pipeline.match.INSTANCES_SCHEMA_V2`)
REALIZED_INSTANCES_PROGRESSIONS_SCHEMA = "verity-gen/instances/v2"

# --- the row driver (`verity_vllm.pipeline`)
#: a row's `config_record.json` (`pipeline.config_record.SCHEMA`), which `record` writes
CONFIG_RECORD_SCHEMA = "verity-vllm/config-record/v1"
#: the circuit database's index, `manifests/records.json` (`pipeline.config_record.INDEX_SCHEMA`): `load_index` gives
#: an empty index with it and refuses any other
RECORDS_INDEX_SCHEMA = "verity-vllm/records-index/v1"
#: a run's `spec.json`, what was to run as known before execution (`pipeline.experiment.SPEC_SCHEMA`), which
#: `build_spec` writes
EXPERIMENT_SPEC_SCHEMA = "verity-gen/spec/v1"
#: a run's `experiment.json` (`pipeline.experiment.EXPERIMENT_SCHEMA`), which `build_experiment` writes
EXPERIMENT_SCHEMA = "verity-gen/experiment/v1"
#: the configuration card (`pipeline.report.CARD_SCHEMA`), which `build_card` writes
CONFIGURATION_CARD_SCHEMA = "verity-gen/card/v1"
#: one event of a row's `timeline.jsonl` (`pipeline.timeline.SCHEMA`), which `Timeline.event` writes; `report` writes
#: it with `/report` appended
TIMELINE_SCHEMA = "verity-sweep/timeline/v1"
#: the `row.json` beside a row, its stages' Attempts, which `pipeline.research._row_json` writes
ROW_ATTEMPTS_SCHEMA = "verity-sweep/row-v2/v1"
#: the configuration a row command ran with, `<row dir>/effective_config.json` (`pipeline.row.EFFECTIVE_SCHEMA`), which
#: `effective_config` writes
EFFECTIVE_CONFIG_SCHEMA = "verity-vllm/effective-config/v1"
#: a run's `tree.json`, the tree a row runs from with its export and its release line, which
#: `pipeline.row_stages.SingleRow.prepare` writes
EXECUTING_TREE_SCHEMA = "verity-sweep/executing-tree/v1"
#: the document of `epoch digests`, which `pipeline.epoch`'s `main` writes
REBASELINE_DIGESTS_SCHEMA = "verity-vllm/rebaseline-digests/v1"

# --- the Match (`verity_vllm.check.match`)
#: the global workload Program's interface as the Match reads it (`check.match.declaration.GP_SCHEMA`)
GLOBAL_PROGRAM_SCHEMA = "cb-a/global-program/v1"
#: the batch decomposition's document, which `check.match.batch_decomp.decompose` writes
BATCH_DECOMP_SCHEMA = "cb-a/batch-decomp/v1"
#: the batch decomposition's phase profile, which `check.match.batch_decomp.decompose` writes
X09_PROFILE_SCHEMA = "cb-a/x09-profile/v1"
#: the cross-rank check's result, which `check.match.rank_match.cross_rank_check` writes
RANK_MATCH_SCHEMA = "tp2/rank_match/v1"
#: the ranks' commit binding, which `check.match.rank_match.commit_binding` writes (not `records.commit_binding`, a
#: Commit record's `binding` block)
RANK_COMMIT_BINDING_SCHEMA = "verity/tp2/commit-binding/v1"
#: the comparison of two Programs, which `check.match.program_compare`'s `main` writes
PROGRAM_COMPARE_SCHEMA = "cb-a/compare/v1"

# --- the sampled replay (`verity_vllm.check.replay`)
#: the sampled replay's record (`check.replay.driver.SCHEMA`), which `check.replay.driver` writes; `pipeline.c2_replay`,
#: `pipeline.replay_bundle` and `pipeline.tp.rank_replay` write it for a replay that did not run
SAMPLED_REPLAY_SCHEMA = "verity-sweep/sampled-replay/v1"
#: the MoE planes' alias check, which `check.replay.population.moe_plane_alias_check` writes
MOE_PLANE_ALIAS_CHECK_SCHEMA = "verity-sweep/moe-plane-alias-check/v1"
#: the PoUW circuit's replay, which `check.replay.pouw_circuit.replay` writes
POUW_CIRCUIT_REPLAY_SCHEMA = "verity-vllm/pouw-ncp-v2-replay/v3"
#: the boundary linkage's record (`check.replay.linkage.LINKAGE_SCHEMA`); `pipeline.c2_replay`, `pipeline.replay_bundle`
#: and `pipeline.tp.rank_replay` write it for a linkage that did not run
BOUNDARY_LINKAGE_SCHEMA = "verity-sweep/boundary-linkage/v1"
#: the stochastic recompute's report, which `check.replay.stoch_recompute.recompute_run` writes
STOCH_RECOMPUTE_SCHEMA = "verity-r15/stoch-recompute/v2"

# --- the checks (`verity_vllm.check`)
#: the Commit's token record, `commit/tokens_p<pair>.json` (`check.commit_tokens.CommitTokens.SCHEMA`), which
#: `commit_tokens_record` writes and `link_to_commit_account` refuses any other than
COMMIT_TOKENS_SCHEMA = "verity-vllm/commit-tokens/v1"
#: the gates document, `gates.json` (`check.gates.GATES_SCHEMA`), which `gates_json` writes
GATES_SCHEMA = "verity-gen/gates/v1"
#: the fold summary, `fold_summary.json` (`check.gates.FOLD_SUMMARY_SCHEMA`), which `pipeline.match`'s fold writes
FOLD_SUMMARY_SCHEMA = "verity-gen/fold-summary/v1"
#: the query-use document, `query_use.json` (`check.gates.QUERY_USE_SCHEMA`); the partition gate refuses any other
QUERY_USE_SCHEMA = "verity-ir/query-use/v1"
#: the Commit's comparison with the Match record (`check.oracle_compare.SCHEMA`), which `oracle_compare` writes
ORACLE_COMPARE_SCHEMA = "verity-sweep/oracle-compare/v1"
#: the weights of record (`check.weights_of_record.SCHEMA`), which `weights_of_record` writes and `root_of_fields`
#: hashes into the root
WEIGHTS_OF_RECORD_SCHEMA = "verity/commit/weights-of-record/v2"
#: the weights of record as derived before the root was host-independent (`check.weights_of_record.SCHEMA_V1`); no code
#: writes it, and a row calls a record that carries it stale
WEIGHTS_OF_RECORD_PRIOR_SCHEMA = "verity/commit/weights-of-record/v1"

# --- the TP row (`verity_vllm.pipeline.tp`)
#: the TP Commit's record of the replays it left for `tp-commit-replay` (`pipeline.tp.c2.SCHEMA`), which `seal` writes
TP_REPLAY_DEFERRED_SCHEMA = "verity-vllm/tp-replay-deferred/v1"
#: one rank's replay result (`pipeline.tp.commit_replay.RESULT_SCHEMA`), which `replay_rank` writes
TP_RANK_REPLAY_SCHEMA = "verity-vllm/tp-rank-replay/v1"
#: the ranks' fold match (`pipeline.tp.fold_match.SCHEMA`), which its `main` writes
TP_FOLD_MATCH_SCHEMA = "verity/tp2/fold-match/v1"
#: the TP Commit's sampled replay (`pipeline.tp.rank_replay.SCHEMA`), which `in_process` and `defer` write, and
#: `pipeline.tp.commit`'s `main`
TP_SAMPLED_REPLAY_SCHEMA = "verity-sweep/tp-sampled-replay/v1"
#: the Build's relabel report (`pipeline.tp.emit_ranks.SCHEMA`), which `finish` and `read_report` write
BUILD_RELABEL_SCHEMA = "verity-vllm/build-relabel/v1"
#: the relabel's guard record (`pipeline.tp.relabel_build.SCHEMA`), which `derive` writes
BUILD_RELABEL_GUARD_SCHEMA = "verity-vllm/build-relabel-guard/v1"
#: a rank's links, which `pipeline.tp.links.write_tp_links` writes
TP_LINKS_SCHEMA = "cb-a/tp_links/v2"
#: a rank's cross-rank picks, which `pipeline.tp.xrank_dump.dump` writes
XRANK_PICKS_SCHEMA = "verity/tp2/xrank-picks/v1"
#: which `pipeline.tp.import_path.write_import_path_runtime` writes
IMPORT_PATH_RUNTIME_SCHEMA = "verity/tp2/import-path-runtime/v1"
#: which `pipeline.tp.rank_worker`'s `tp2_param_hashes` writes
TP_PARAM_HASHES_SCHEMA = "verity-capture/tp2-param-hashes/v0"
#: which `pipeline.tp.rank_worker`'s `tp2_profiler_stop` writes
CENSUS_RANK_SCHEMA = "verity-capture/census-rank/v0"
#: the late-read check on a fused-MoE block's received output, which `acquire.sources.partial_source.late_read_check`
#: writes for one rank, and `pipeline.tp.rank_worker` and `pipeline.tp.commit` for the ranks
LATE_READ_CHECK_SCHEMA = "verity/commit-integ/t6_4-check/v1"
#: the TP Commit's value check, `value_check.json`, which `pipeline.tp.commit`'s `main` writes
TP_VALUE_CHECK_SCHEMA = "verity/commit-integ/value-check-tp/v1"
#: the pairs' cross-rank checks folded, `xrank_collectives.json`, which `pipeline.tp.commit`'s `main` writes
XRANK_COLLECTIVES_FOLD_SCHEMA = "verity/tp2/xrank-collectives-fold/v1"
#: the ranks' weights pins folded, `weights_pin.json`, which `pipeline.tp.commit`'s `main` writes
TP_WEIGHTS_PIN_SCHEMA = "verity-sweep/tp-weights-pin/v1"

# --- the engine (`verity_vllm.engine`)
#: the engine's profile manifest, which `engine.engine_profile.build_manifest` writes and `pipeline.tp.rank_worker` reads
ENGINE_PROFILE_SCHEMA = "verity-vllm/profile/v1"
#: the manifest `engine.run_facts.profile_manifest` writes in place of the engine's profile when building that one raises
PROFILE_FALLBACK_SCHEMA = "verity-capture/profile-fallback/v1"
#: the record of the served switches (`engine.served.SCHEMA`), which `record` writes, and `require` on a failing summary
SERVED_SWITCHES_SCHEMA = "verity-vllm/served-switches/v1"
#: an engine spec (`engine.spec.SCHEMA`); `from_mapping` and `load` refuse any other
ENGINE_SPEC_SCHEMA = "verity-vllm/engine-spec/v0"
#: the static schedule's check (`engine.static_schedule.SCHEMA`), which `check` writes
STATIC_SCHEDULE_SCHEMA = "verity-vllm/static-schedule/v1"
#: the registered int8 K-scales manifest (`engine.int8_k_scales.SCHEMA`), which `with_entry` writes into a new one
INT8_K_SCALES_SCHEMA = "verity-vllm/int8-k-scales/v1"
#: an int8 checkpoint's error report, which `engine.int8_checkpoint.report` writes
INT8_ERROR_REPORT_SCHEMA = "verity-vllm/int8-error-report/v1"
#: a served run's step trace (`engine.step_trace.SCHEMA`): `StepTrace` writes it in the header, the trace's first record,
#: and `load` refuses a trace that opens with any other
STEP_TRACE_SCHEMA = "verity-vllm/step-trace/v1"
#: the inventory of the modules vLLM builds for a checkpoint, by rank (`engine.meta_inventory.SCHEMA`), which
#: `live_document` and the `meta-inventory` command write
META_INVENTORY_SCHEMA = "verity-vllm/meta-inventory/v1"
#: the generated kernels of one process by content hash, which `engine.kernel_identity.collect` writes
KERNEL_IDENTITY_SCHEMA = "verity-vllm/kernel-identity/v1"

# --- the target-family preflight (`verity_vllm.target_family`)
#: the preflight's record, which `target_family.check` writes
TARGET_FAMILY_SCHEMA = "verity-sweep/target-family/v1"

# --- the ahead-of-time compiler (`verity_vllm.program.predict`)
#: a phi (`program.predict.phi.PHI_SCHEMA`), which `Phi.from_json` refuses any other than
PHI_SCHEMA = "verity-predict/phi/v1"
#: the comparison of a predicted Build with the Build of record (`program.predict.aot.SCHEMA`)
AOT_COMPARISON_SCHEMA = "verity-vllm/aot-comparison/v1"
#: `node.json`, which `program.predict.cli.node_main` writes
PREDICT_NODE_SCHEMA = "verity-predict/node/v1"
#: `registration.json`, which `program.predict.node_registration.register` writes
NODE_REGISTRATION_SCHEMA = "verity-vllm/node-registration/v1"

# --- PoUW's H1 (`verity_vllm.protocols.pouw.h1`)
#: a served window's calls record, `pouw-h1-calls.json` (`protocols.pouw.h1.RECORD`), which `H1Run.finish` writes
POUW_H1_CALLS_SCHEMA = "verity-vllm/pouw-h1-calls/v2"
#: a suspended window's hand-off state (`protocols.pouw.h1.handoff.SCHEMA`), which `suspend` writes and `resume` refuses
#: any other than
POUW_H1_HANDOFF_SCHEMA = "verity-vllm/pouw-h1-handoff/v1"

# --- the protocol options (`verity_vllm.protocols`)
#: the record of the options one stage installed (`protocols.RECORD_SCHEMA`), which `Composition.record` builds and
#: `into_verdict` writes as a Commit's `protocols.json`
PROTOCOL_COMPOSITION_SCHEMA = "verity-vllm/protocol-composition/v0"
