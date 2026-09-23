"""Generate tools/move_map.txt from placement rules.  Usage: python tools/gen_move_map.py [integration_root] [out].  The committed
artefact is the map; re-run this only when new files appear (e.g. on the staging lane) and review the diff."""
import ast, collections, os, re, subprocess, sys

ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'tools', 'move_map.txt')
files = subprocess.run(['git', 'ls-files'], cwd=ROOT, capture_output=True, text=True).stdout.split()

DEAD = 'vllm-dead-code'; PROF = 'vllm-profiles-generic'; TPN = 'vllm-tp-n'; V2D = 'vllm-v2-default'

# ---- explicit non-test module placement (old -> new[, lane]) ------------------------------------------------------------
E = {}
def put(old, new, lane=None): E[old] = (new, lane)

# --- verity_vllm top level (v2 + the W2 POC) ---
put('verity_vllm/__init__.py', 'verity_vllm/__init__.py')
put('verity_vllm/capture.py', 'verity_vllm/observe/capture_v1.py')
put('verity_vllm/commit.py', 'verity_vllm/commit/merkle.py')
put('verity_vllm/hashing.py', 'verity_vllm/commit/hashing.py')
put('verity_vllm/identity.py', 'verity_vllm/commit/identity.py')
put('verity_vllm/description.py', 'verity_vllm/check/poc_description.py', DEAD)
put('verity_vllm/dtypes.py', 'verity_vllm/program/dtypes.py')
put('verity_vllm/relations.py', 'verity_vllm/check/relations.py')
put('verity_vllm/required.py', 'verity_vllm/check/poc_required_interface.py', DEAD)
put('verity_vllm/rows.py', 'verity_vllm/check/poc_rows.py', DEAD)
put('verity_vllm/synthetic.py', 'verity_vllm/harness/synthetic.py', DEAD)
put('verity_vllm/observe.py', 'verity_vllm/observe/contract.py')
put('verity_vllm/resolve.py', 'verity_vllm/correspondence/resolve.py')
put('verity_vllm/resolve_decomp.py', 'verity_vllm/correspondence/resolve_decomp.py')
put('verity_vllm/runtime_correspondence.py', 'verity_vllm/correspondence/runtime.py')
put('verity_vllm/research_tools.py', 'verity_vllm/harness/research_tools.py')
# commit_bench (CPERF prototypes; fasttree/semantic/stream are the live engine pieces)
put('verity_vllm/commit_bench/__init__.py', 'DROP')
put('verity_vllm/commit_bench/fasttree.py', 'verity_vllm/commit/fasttree.py')
put('verity_vllm/commit_bench/semantic.py', 'verity_vllm/commit/semantic_layout.py')
put('verity_vllm/commit_bench/stream.py', 'verity_vllm/commit/stream_merkle.py')
put('verity_vllm/commit_bench/test_semantic.py', 'tests/commit/test_semantic_layout.py')
put('verity_vllm/commit_bench/test_stream.py', 'tests/commit/test_stream_merkle.py')
# correspondence (existing package)
put('verity_vllm/correspondence/__init__.py', 'verity_vllm/correspondence/__init__.py')
put('verity_vllm/correspondence/core.py', 'verity_vllm/correspondence/capture_identities.py')
put('verity_vllm/correspondence/program.py', 'verity_vllm/correspondence/capture_identities_program.py', DEAD)
# ir
put('verity_vllm/ir/__init__.py', 'DROP')
put('verity_vllm/ir/boundary.py', 'verity_vllm/query/boundary.py')
put('verity_vllm/ir/partition.py', 'verity_vllm/query/partition.py')
put('verity_vllm/ir/query_artifact.py', 'verity_vllm/query/query_artifact.py')
put('verity_vllm/ir/descriptor_equivalence.py', 'verity_vllm/program/descriptor_equivalence.py')
put('verity_vllm/ir/profile_descriptor.py', 'verity_vllm/program/profile_descriptor.py')
put('verity_vllm/ir/sampling_event.py', 'verity_vllm/program/sampling_event.py')
put('verity_vllm/ir/workload.py', 'verity_vllm/program/workload.py')
# frontend / registry / lifting -> program/
put('verity_vllm/frontend/emit_correspondence.py', 'verity_vllm/correspondence/emit.py')
# query
put('verity_vllm/query/correspondence.py', 'verity_vllm/correspondence/reader_for_query.py')
# acquire
put('verity_vllm/acquire/correspondence.py', 'verity_vllm/correspondence/reader_for_acquire.py')
put('verity_vllm/acquire/runtime_tree.py', 'verity_vllm/correspondence/runtime_tree.py')
put('verity_vllm/acquire/switch.py', 'verity_vllm/acquire/switch.py', V2D)
put('verity_vllm/acquire/v1_decision.py', 'verity_vllm/acquire/v1_decision.py', V2D)
put('verity_vllm/acquire/gate.py', 'verity_vllm/acquire/gate.py', V2D)
put('verity_vllm/acquire/match_compare.py', 'verity_vllm/acquire/match_compare.py', V2D)
put('verity_vllm/acquire/pod_gate.sh', 'verity_vllm/ops/pod_gate.sh', V2D)
put('verity_vllm/acquire/pod_match_v2.sh', 'verity_vllm/ops/pod_match_v2.sh', V2D)
# numerics / adapter
put('verity_vllm_numerics/__init__.py', 'verity_vllm/program/numerics/__init__.py')
# built binaries that were tracked by mistake; lane/vllm-cleanup-2 untracks them
put('verity_vllm_numerics/cpp/libfa2_model.so', 'verity_vllm/program/numerics/cpp/libfa2_model.so', 'vllm-cleanup-2')
put('verity_vllm_numerics/cpp/librms_triton_model.so', 'verity_vllm/program/numerics/cpp/librms_triton_model.so', 'vllm-cleanup-2')
put('verity_vllm_adapter/__init__.py', 'DROP')
put('verity_vllm_adapter/driver.py', 'verity_vllm/observe/engine_driver.py')
put('verity_vllm_adapter/profile.py', 'verity_vllm/observe/engine_profile.py')
put('verity_vllm_adapter/kernel_identity.py', 'verity_vllm/check/kernel_identity.py')
put('verity_vllm_adapter/verify_bindings.py', 'verity_vllm/check/poc_verify_bindings.py', DEAD)
put('verity_vllm_adapter/compiled_fx/__init__.py', 'DROP')
put('verity_vllm_adapter/compiled_fx/kernels.py', 'verity_vllm/check/compiled_fx_kernels.py')

# --- verity_capture top level ---
put('verity_capture/__init__.py', 'verity_vllm/observe/__init__.py')
put('verity_capture/allowlist.py', 'verity_vllm/check/kernel_allowlist.py')
put('verity_capture/analytic.py', 'verity_vllm/input_provenance/analytic.py')
put('verity_capture/arrivals.py', 'verity_vllm/observe/arrivals.py')
put('verity_capture/card.py', 'verity_vllm/harness/card.py')
put('verity_capture/census.py', 'verity_vllm/check/census.py')
put('verity_capture/chunk_attribution.py', 'verity_vllm/correspondence/chunk_attribution.py')
put('verity_capture/compare.py', 'verity_vllm/check/fold_compare.py')
put('verity_capture/difftest.py', 'verity_vllm/check/difftest.py')
put('verity_capture/events.py', 'verity_vllm/observe/events.py')
put('verity_capture/experiment.py', 'verity_vllm/harness/experiment.py')
put('verity_capture/fold.py', 'verity_vllm/observe/fold.py')
put('verity_capture/gates.py', 'verity_vllm/check/gates.py')
put('verity_capture/gc_tuning.py', 'verity_vllm/harness/gc_tuning.py')
put('verity_capture/golden.py', 'verity_vllm/check/golden.py')
put('verity_capture/golden/corpus.json', 'verity_vllm/check/golden/corpus.json')
put('verity_capture/holdout.py', 'verity_vllm/check/holdout.py', DEAD)
put('verity_capture/log.py', 'verity_vllm/observe/log.py')
put('verity_capture/memory.py', 'verity_vllm/observe/memory.py')
put('verity_capture/noninterference.py', 'verity_vllm/check/noninterference.py')
put('verity_capture/observer.py', 'verity_vllm/observe/observer.py')
put('verity_capture/patterns.py', 'verity_vllm/observe/patterns.py')
put('verity_capture/patterns_fp8.py', 'verity_vllm/observe/patterns_fp8.py')
put('verity_capture/patterns_prefix.py', 'verity_vllm/observe/patterns_prefix.py', DEAD)
put('verity_capture/prefix_cache.py', 'verity_vllm/observe/prefix_cache.py', DEAD)
put('verity_capture/protected.py', 'verity_vllm/check/protected.py', DEAD)
put('verity_capture/provenance.py', 'verity_vllm/check/operand_provenance.py')
put('verity_capture/quarantine_lint.py', 'verity_vllm/check/quarantine_lint.py', DEAD)
put('verity_capture/replay.py', 'verity_vllm/check/replay.py')
put('verity_capture/replay_budgeted.py', 'verity_vllm/check/replay_budgeted.py', DEAD)
put('verity_capture/resolver.py', 'verity_vllm/observe/resolver.py')
put('verity_capture/root_policy.py', 'verity_vllm/input_provenance/root_policy.py')
put('verity_capture/storage.py', 'verity_vllm/observe/storage.py')
put('verity_capture/tree.py', 'verity_vllm/observe/tree.py')
put('verity_capture/triton_adapter.py', 'verity_vllm/observe/triton_adapter.py')
put('verity_capture/twins.py', 'verity_vllm/check/twins.py')
put('verity_capture/views.py', 'verity_vllm/observe/views.py')
put('verity_capture/vllm_adapter.py', 'verity_vllm/observe/vllm_adapter.py')
put('verity_capture/vu_canonical.py', 'verity_vllm/query/vu_canonical.py')
put('verity_capture/vu_query.py', 'verity_vllm/query/vu_query.py')
# bench
put('verity_capture/bench/__init__.py', 'DROP')
put('verity_capture/bench/commit_delta.py', 'verity_vllm/harness/commit_delta.py')
put('verity_capture/bench/commit_verdict.py', 'verity_vllm/check/commit_verdict.py')
put('verity_capture/bench/compiled_merge.py', 'verity_vllm/harness/compiled_merge.py')
put('verity_capture/bench/gen_adversarial.py', 'verity_vllm/check/adversarial.py', DEAD)
put('verity_capture/bench/gen_batch_attribution.py', 'verity_vllm/correspondence/batch_attribution.py', DEAD)
put('verity_capture/bench/gen_batch_candidate.py', 'verity_vllm/correspondence/batch_candidate.py', DEAD)
put('verity_capture/bench/gen_batch_provenance.py', 'verity_vllm/correspondence/batch_provenance.py', DEAD)
put('verity_capture/bench/gen_coverage_workloads.py', 'verity_vllm/harness/coverage_workloads.py', DEAD)
put('verity_capture/bench/gen_ov_sampling_libdevice.py', 'verity_vllm/program/numerics/libdevice_sampling.py', DEAD)
put('verity_capture/bench/gen_sampling_rng.py', 'verity_vllm/program/numerics/sampling_rng.py')
put('verity_capture/bench/m1_capture.py', 'verity_vllm/observe/m1_capture.py')
put('verity_capture/bench/resolve_log.py', 'verity_vllm/observe/resolve_log.py')
put('verity_capture/bench/run_config.py', 'verity_vllm/harness/run_config.py')
put('verity_capture/bench/stoch_recompute.py', 'verity_vllm/check/stoch_recompute.py')
put('verity_capture/bench/topp_split_probe.py', 'verity_vllm/harness/topp_split_probe.py', DEAD)
put('verity_capture/bench/tp2_analyze.py', 'verity_vllm/tp/analyze.py', DEAD)
put('verity_capture/bench/tp2_capture.py', 'verity_vllm/tp/capture.py')
put('verity_capture/bench/tp2_commit.py', 'verity_vllm/tp/commit_delta.py')
put('verity_capture/bench/tp2_fold_match.py', 'verity_vllm/tp/fold_match.py')
put('verity_capture/bench/tp2_match.py', 'verity_vllm/tp/match.py')
put('verity_capture/bench/tp2_worker.py', 'verity_vllm/tp/worker.py')
put('verity_capture/bench/fixtures/llama32-1b__bf16__l40s__tp2__b2__i64__o8__equal__greedy__bi-eager.json',
    'tests/tp/fixtures/llama32-1b__bf16__l40s__tp2__b2__i64__o8__equal__greedy__bi-eager.json')
# commit/
put('verity_capture/commit/__init__.py', 'verity_vllm/commit/__init__.py')
put('verity_capture/commit/admission.py', 'verity_vllm/harness/admission_bound.py')
put('verity_capture/commit/admission_planner.py', 'verity_vllm/harness/admission_planner.py')
put('verity_capture/commit/binding.py', 'verity_vllm/commit/binding.py')
put('verity_capture/commit/compiled_manifest.py', 'verity_vllm/query/compiled_manifest.py')
put('verity_capture/commit/executed_prefix.py', 'verity_vllm/check/executed_prefix.py')
put('verity_capture/commit/leafhash.py', 'verity_vllm/acquire/leafhash.py')
put('verity_capture/commit/native_collect.py', 'verity_vllm/acquire/native_collect.py')
put('verity_capture/commit/native_collect.cpp', 'verity_vllm/acquire/native_collect.cpp')
put('verity_capture/commit/native_gather.cu', 'verity_vllm/acquire/native_gather.cu')
put('verity_capture/commit/native_leafhash.cu', 'verity_vllm/acquire/native_leafhash.cu')
put('verity_capture/commit/native_tree.cu', 'verity_vllm/acquire/native_tree.cu')
put('verity_capture/commit/native_jit.py', 'verity_vllm/acquire/native_jit.py')
put('verity_capture/commit/oracle_compare.py', 'verity_vllm/check/oracle_compare.py')
put('verity_capture/commit/padding_steps.py', 'verity_vllm/commit/padding_steps.py')
put('verity_capture/commit/required_manifest.py', 'verity_vllm/query/required_manifest.py')
put('verity_capture/commit/required_values.py', 'verity_vllm/query/required_values.py')
put('verity_capture/commit/sampled_replay.py', 'verity_vllm/check/sampled_replay.py')
put('verity_capture/commit/verdict.py', 'verity_vllm/check/verdict.py')
put('verity_capture/commit/weights_of_record.py', 'verity_vllm/input_provenance/weights_of_record.py')
put('verity_capture/commit/xrank_collectives.py', 'verity_vllm/tp/xrank_collectives.py')
# experimental/
put('verity_capture/experimental/__init__.py', 'DROP')
put('verity_capture/experimental/fp16_rules.py', 'verity_vllm/check/fp16_rules.py', DEAD)
put('verity_capture/experimental/prefix_g3_rules.py', 'verity_vllm/check/prefix_g3_rules.py', DEAD)
put('verity_capture/experimental/cb_a/__init__.py', 'DROP')
put('verity_capture/experimental/cb_a/batch_decomp.py', 'verity_vllm/correspondence/batch_decomp.py')
put('verity_capture/experimental/cb_a/compact.py', 'verity_vllm/program/compact.py')
put('verity_capture/experimental/cb_a/compare.py', 'verity_vllm/check/program_compare.py')
put('verity_capture/experimental/cb_a/derive_step.py', 'verity_vllm/harness/derive_step.py')
put('verity_capture/experimental/cb_a/export_compat.py', 'verity_vllm/program/frontend/export_compat.py')
put('verity_capture/experimental/cb_a/global_match.py', 'verity_vllm/check/global_match.py')
put('verity_capture/experimental/cb_a/global_match_fast.py', 'verity_vllm/check/global_match_fast.py')
put('verity_capture/experimental/cb_a/global_program.py', 'verity_vllm/program/global_program.py')
put('verity_capture/experimental/cb_a/inputs_trace.py', 'verity_vllm/program/frontend/inputs_trace.py')
put('verity_capture/experimental/cb_a/instances_form.py', 'verity_vllm/program/instances_form.py', DEAD)
put('verity_capture/experimental/cb_a/tp_ops.py', 'verity_vllm/tp/export_ops.py')
put('verity_capture/experimental/cb_a/triton_capture.py', 'verity_vllm/program/frontend/triton_capture.py')
put('verity_capture/experimental/cb_a/vllm_meta.py', 'verity_vllm/program/frontend/vllm_meta.py')
put('verity_capture/experimental/commit/__init__.py', 'verity_vllm/commit/reference_engine/__init__.py', DEAD)
put('verity_capture/experimental/commit/engine.py', 'verity_vllm/commit/reference_engine/engine.py', DEAD)
put('verity_capture/experimental/commit/positions.py', 'verity_vllm/commit/reference_engine/positions.py', DEAD)
put('verity_capture/experimental/commit/torch_sha256.py', 'verity_vllm/commit/reference_engine/torch_sha256.py', DEAD)
put('verity_capture/experimental/commit/triton_sha256.py', 'verity_vllm/commit/reference_engine/triton_sha256.py', DEAD)
put('verity_capture/experimental/commit_integ/__init__.py', 'DROP')
put('verity_capture/experimental/commit_integ/api.py', 'verity_vllm/acquire/committer_api.py')
put('verity_capture/experimental/commit_integ/cmt_ref_adapter.py', 'verity_vllm/commit/reference_engine_adapter.py', DEAD)
put('verity_capture/experimental/commit_integ/compiled_autotune.py', 'verity_vllm/check/compiled_autotune.py')
put('verity_capture/experimental/commit_integ/compiled_kernel_check.py', 'verity_vllm/check/compiled_kernel_check.py')
put('verity_capture/experimental/commit_integ/compiled_kernel_source.py', 'verity_vllm/acquire/compiled_kernel_source.py')
put('verity_capture/experimental/commit_integ/compiled_norm_literal.py', 'verity_vllm/program/numerics/compiled_norm_literal.py')
put('verity_capture/experimental/commit_integ/compiled_relations.py', 'verity_vllm/program/numerics/compiled_relations.py')
put('verity_capture/experimental/commit_integ/compiled_source.py', 'verity_vllm/acquire/compiled_source.py')
put('verity_capture/experimental/commit_integ/compiled_value_check.py', 'verity_vllm/check/compiled_value_check.py')
put('verity_capture/experimental/commit_integ/fa2_plane_classes.py', 'verity_vllm/acquire/fa2_plane_classes.py')
put('verity_capture/experimental/commit_integ/hidden_source.py', 'verity_vllm/acquire/hidden_source.py')
put('verity_capture/experimental/commit_integ/moe_source.py', 'verity_vllm/acquire/moe_source.py')
put('verity_capture/experimental/commit_integ/native_host.py', 'verity_vllm/acquire/native_host.py')
put('verity_capture/experimental/commit_integ/row4_derived.py', 'verity_vllm/check/row4_derived.py', DEAD)
put('verity_capture/experimental/commit_integ/spans.py', 'verity_vllm/harness/spans.py')
put('verity_capture/experimental/commit_integ/tp_partial_source.py', 'verity_vllm/tp/partial_source.py')
put('verity_capture/experimental/commit_integ/value_check.py', 'verity_vllm/check/value_check.py')
put('verity_capture/experimental/fa2_commit/__init__.py', 'DROP')
put('verity_capture/experimental/fa2_commit/engine.py', 'verity_vllm/commit/hidden_engine.py')
put('verity_capture/experimental/fa2_commit/hidden_stream.py', 'verity_vllm/commit/hidden_stream.py')
for f in ('derived', 'encoding', 'fixture', 'kernel_dump', 'layouts', 'oracle', 'reference'):
    put(f'verity_capture/experimental/fa2_commit/{f}.py', f'verity_vllm/commit/fa2_prototype/{f}.py', DEAD)
# profiles -> observe/profiles (lane prof: one generic constructor + facts table; per-model modules go)
put('verity_capture/profiles/__init__.py', 'verity_vllm/observe/profiles/__init__.py')
put('verity_capture/profiles/dense_generic.py', 'verity_vllm/observe/profiles/dense_generic.py')
put('verity_capture/profiles/vllm_d9105ea80_sm89_eager.py', 'verity_vllm/observe/profiles/vllm_d9105ea80_sm89_eager.py')
put('verity_capture/profiles/corpus_coverage.json', 'verity_vllm/observe/profiles/corpus_coverage.json')
# sweep
put('verity_capture/sweep/__init__.py', 'DROP')
put('verity_capture/sweep/commit_summary.py', 'verity_vllm/harness/commit_summary.py')
put('verity_capture/sweep/fa2_attn_oracle.py', 'verity_vllm/check/fa2_attn_oracle.py', DEAD)
put('verity_capture/sweep/finalize.py', 'verity_vllm/harness/finalize.py')
put('verity_capture/sweep/hot_commit.py', 'verity_vllm/harness/hot_commit.py')
put('verity_capture/sweep/launch_context.py', 'verity_vllm/harness/launch_context.py')
put('verity_capture/sweep/rebuild_digest_gate.py', 'verity_vllm/harness/rebuild_digest_gate.py')
put('verity_capture/sweep/release_json.py', 'verity_vllm/harness/release_json.py')
put('verity_capture/sweep/research_outputs.py', 'verity_vllm/harness/research_outputs.py')
put('verity_capture/sweep/research_result.py', 'verity_vllm/harness/research_result.py')
put('verity_capture/sweep/source_identity.py', 'verity_vllm/harness/source_identity.py')
put('verity_capture/sweep/target_family.py', 'verity_vllm/harness/target_family.py')
put('verity_capture/sweep/timeline.py', 'verity_vllm/harness/timeline.py')
put('verity_capture/sweep/workload.py', 'verity_vllm/harness/workload.py')
put('verity_capture/sweep/known_roots.json', 'verity_vllm/ops/known_roots.json')
put('verity_capture/sweep/row_pod_tp2.sh', 'verity_vllm/ops/row_pod_tp2.sh', TPN)
# telemetry
put('verity_capture/telemetry/__init__.py', 'verity_vllm/harness/telemetry/__init__.py')
put('verity_capture/telemetry/admission.py', 'verity_vllm/harness/telemetry/admission.py')
# tp
put('verity_capture/tp/__init__.py', 'verity_vllm/tp/__init__.py')
put('verity_capture/tp/collective_link.py', 'verity_vllm/tp/collective_link.py')
put('verity_capture/tp/collective_pattern.py', 'verity_vllm/tp/collective_pattern.py')
put('verity_capture/tp/collective_record.py', 'verity_vllm/tp/collective_record.py')
put('verity_capture/tp/collective_sites.py', 'verity_vllm/tp/collective_sites.py')
put('verity_capture/tp/rank_match.py', 'verity_vllm/tp/rank_match.py')
# root
put('pod_bootstrap.sh', 'verity_vllm/ops/pod_bootstrap.sh')

# ---- prefix rules (dir -> dir[, lane]); exact entries win --------------------------------------------------------------
P = [
    ('verity_vllm/registry/', 'verity_vllm/program/registry/', None),
    ('verity_vllm/frontend/', 'verity_vllm/program/frontend/', None),
    ('verity_vllm/lifting/', 'verity_vllm/program/lifting/', None),
    ('verity_vllm/query/', 'verity_vllm/query/', None),
    ('verity_vllm/acquire/', 'verity_vllm/acquire/', None),
    ('verity_vllm/commit_engine/', 'verity_vllm/commit/engine_rs/', None),
    ('verity_vllm_numerics/', 'verity_vllm/program/numerics/', None),
    ('verity_capture/profiles/quarantine/', 'verity_vllm/observe/profiles/quarantine/', PROF),
    ('verity_capture/profiles/hf_configs/', 'verity_vllm/observe/profiles/hf_configs/', None),
    ('verity_capture/profiles/', 'verity_vllm/observe/profiles/', PROF),        # per-model gen_*.py / workload_*.json
    ('verity_capture/sweep/fa2_tap_src/', 'verity_vllm/acquire/fa2_tap_src/', None),
    ('verity_capture/sweep/fa3_tap_src/', 'verity_vllm/acquire/fa3_tap_src/', None),
    ('verity_capture/sweep/hidden_gpu_src/', 'verity_vllm/acquire/hidden_gpu_src/', None),
    ('verity_capture/sweep/', 'verity_vllm/ops/', None),                          # the remaining *.sh
    ('verity_capture/bench/', 'verity_vllm/ops/', None),                          # the remaining *.sh
    ('verity_capture/experimental/fa2_commit/fixtures/', 'verity_vllm/commit/fa2_prototype/fixtures/', DEAD),
    ('verity_capture/commit/tests/fixtures/', 'tests/harness/fixtures/', None),
    ('verity_capture/tests/fixtures/tp2/', 'tests/tp/fixtures/tp2/', None),
    ('tests/ir/data/', 'tests/program/data/', None),
    ('tests/query/gate/', 'tests/query/gate/', None),
    ('tests/regression/', 'tests/regression/', None),
    ('tests/acquire/', 'tests/acquire/', None),
    ('workloads/', 'workloads/', None), ('manifests/', 'manifests/', None), ('fixtures/', 'fixtures/', None),
    ('data/', 'data/', None), ('docs/', 'docs/', None),
]
STAY = {'README.md', 'pyproject.toml', '.gitignore', 'tests/__init__.py', 'tests/by_name_allowlist.json',
        'tests/test_no_by_name_rules.py', 'tests/query/__init__.py', 'tests/acquire/__init__.py'}

def place(old):
    if old in E: return E[old]
    if old in STAY: return (old, None)
    for pre, new, lane in P:
        if old.startswith(pre): return (new + old[len(pre):], lane)
    return None

# ---- tests: mirror the package (majority vote over the imported integration modules) ------------------------------------
TEST_RE = re.compile(r'(^|/)(tests?/|test_[^/]+\.py$|conftest\.py$)')
def is_test(p): return bool(TEST_RE.search(p)) and p.endswith('.py') and not p.startswith('tests/regression/')

mod_of = {}   # dotted old module -> new subpackage
for f in files:
    if not f.endswith('.py') or is_test(f): continue
    r = place(f)
    if not r or r[0] == 'DROP': continue
    new = r[0]
    m = f[:-3].replace('/', '.'); m = m[:-9] if m.endswith('.__init__') else m
    parts = new.split('/')
    sub = parts[1] if parts[0] == 'verity_vllm' and len(parts) > 2 else None
    if sub: mod_of[m] = sub

KEY = [('tp2', 'tp'), ('tp_', 'tp'), ('row_pod', 'ops'), ('commit_delta', 'harness'), ('run_config', 'harness'),
       ('admission', 'harness'), ('global_match', 'check'), ('sampled_replay', 'check'), ('required', 'query'),
       ('weights', 'input_provenance'), ('native', 'acquire'), ('lifted', 'program'), ('derive', 'program'),
       ('fold', 'observe'), ('gen_', 'observe'), ('compare', 'check'), ('oracle', 'check'), ('verdict', 'check'),
       ('x03', 'acquire'), ('regression', 'regression')]
LANE_OF_STEM = {}
for f0 in files:
    r0 = place(f0)
    if f0.endswith('.py') and not is_test(f0) and r0 and r0[0] != 'DROP':
        LANE_OF_STEM.setdefault(os.path.basename(f0)[:-3], r0[1])
def test_lane(path):
    if path in ('verity_capture/bench/tests/test_engine_schedule.py',): return None
    base = os.path.basename(path)
    x = base[len('test_'):-3] if base.startswith('test_') else None
    if not x: return None
    best = None
    for stem, lane in LANE_OF_STEM.items():
        if x == stem or x.startswith(stem + '_'):
            if best is None or len(stem) > len(best[0]): best = (stem, lane)
    if best: return best[1]
    d = os.path.dirname(path)
    if d in ('verity_capture/experimental/commit', 'verity_capture/experimental/fa2_commit'): return DEAD
    return None
STEMS = {}   # module stem -> new subpackage (namesake rule: test_<stem>[_...].py follows the module)
for m, sub in mod_of.items():
    STEMS.setdefault(m.rsplit('.', 1)[-1], sub)
def namesake(base):
    x = base[len('test_'):-3] if base.startswith('test_') else None
    if not x: return None
    best = None
    for stem, sub in STEMS.items():
        if stem == '__init__': continue
        if x == stem or x.startswith(stem + '_'):
            if best is None or len(stem) > len(best[0]): best = (stem, sub)
    return best[1] if best else None
OVERRIDE = {
    'tests/ir/test_tp_collective_site.py': 'correspondence',   # `import test_tp_collective_site` from test_runtime_correspondence (sibling import)
    'tests/query/test_compare.py': 'query', 'verity_capture/tests/test_gp01_moe_construction.py': 'program',
    'tests/ir/test_workload_program.py': 'program', 'tests/ir/test_workload_compose_padrev.py': 'program',
    'tests/ir/test_artifact_applicability_independent.py': 'program', 'verity_capture/tests/test_resolve_m1.py': 'observe',
    'verity_capture/bench/tests/test_engine_schedule.py': 'harness', 'verity_capture/commit/tests/test_admit_r19_host_working_set.py': 'harness',
    'verity_capture/tests/test_coverage_workloads.py': 'harness', 'verity_capture/tests/test_vu_conformance.py': 'query',
    'tests/ir/test_topp_split_geometry.py': 'program', 'tests/ir/test_conformance_record.py': 'program',
    'tests/acquire/test_plan.py': 'acquire', 'tests/ir/test_observe_resolve.py': 'observe',
    'verity_capture/commit/tests/test_prescribed_input_linkage.py': 'harness',
    'verity_capture/experimental/commit_integ/tests/schemes.py': 'acquire', 'tests/ir/padded_commit_tiny.py': 'program',
}
def vote(path):
    if path in OVERRIDE: return OVERRIDE[path]
    base = os.path.basename(path)
    ns = namesake(base)
    if ns: return ns
    for k, sub in KEY[:3]:
        if k in base: return sub
    src = open(os.path.join(ROOT, path)).read()
    try: t = ast.parse(src)
    except Exception: t = None
    c = collections.Counter()
    names = set()
    if t is not None:
        for n in ast.walk(t):
            if isinstance(n, ast.Import):
                for a in n.names: names.add(a.name)
            elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
                names.add(n.module)
                for a in n.names: names.add(n.module + '.' + a.name)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith(('verity_vllm.', 'verity_capture.')):
                names.add(n.value)
    for d in sorted(names):
        while d and d not in mod_of: d = d.rsplit('.', 1)[0] if '.' in d else ''
        if d: c[mod_of[d]] += 1
    if c:
        return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]   # deterministic tie-break
    base = os.path.basename(path)
    for k, sub in KEY:
        if k in base: return sub
    return None

tests = collections.defaultdict(list)
placed = {}
for f in files:
    if not is_test(f): continue
    if f in E or f in STAY: continue
    base = os.path.basename(f)
    if base in ('__init__.py', 'conftest.py'):
        continue
    sub = vote(f)
    if sub is None:
        # test-support modules: go where their importers go (second pass) -- default program/
        sub = 'program'
    placed[f] = sub
# support modules (non test_*.py in test dirs): follow the majority of tests in their old dir
for f in list(placed):
    pass
by_dir = collections.defaultdict(collections.Counter)
for f, sub in placed.items():
    if os.path.basename(f).startswith('test_'): by_dir[os.path.dirname(f)][sub] += 1
for f in files:
    if is_test(f) and f not in placed and f not in E and f not in STAY and os.path.basename(f) not in ('__init__.py',):
        d = os.path.dirname(f)
        placed[f] = sorted(by_dir[d].items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if by_dir[d] else 'program'
# collisions -> suffix by old dir tag
targets = collections.defaultdict(list)
for f, sub in placed.items(): targets[(sub, os.path.basename(f))].append(f)
TAG = {'verity_capture/tests': 'capture', 'verity_capture/commit/tests': 'commit', 'verity_capture/bench/tests': 'bench',
       'verity_capture/experimental/cb_a/tests': 'cb_a', 'verity_capture/experimental/commit_integ/tests': 'commit_integ',
       'verity_capture/telemetry/tests': 'telemetry', 'tests/ir': 'ir', 'tests/query': 'query', 'tests/acquire': 'acquire'}
test_map = {}
for (sub, base), olds in targets.items():
    if len(olds) == 1:
        test_map[olds[0]] = f'tests/{sub}/{base}'
    else:
        for o in olds:
            tag = TAG.get(os.path.dirname(o), 'x')
            stem, ext = os.path.splitext(base)
            test_map[o] = f'tests/{sub}/{stem}_{tag}{ext}'
# conftest / __init__ of dissolved test dirs
for f in files:
    if is_test(f) and f not in test_map and f not in E and f not in STAY:
        base = os.path.basename(f)
        if base == '__init__.py':
            test_map[f] = 'DROP'
        elif base == 'conftest.py':
            test_map[f] = 'tests/conftest.py'

# ---- emit -----------------------------------------------------------------------------------------------------------
lines = []
unmapped = []
for f in files:
    if f in test_map:
        lines.append((f, test_map[f], test_lane(f) if test_map[f] != 'DROP' else None)); continue
    r = place(f)
    if r is None: unmapped.append(f); continue
    lines.append((f, r[0], r[1]))
hdr = """# verity-vllm relayout move map (integrations/vllm/tools/relayout.py reads this; paths are relative to integrations/vllm).
#
#   <old path> -> <new path>                    move (git mv)
#   <old path> -> <new path>  DELETE-BY:<lane>  the named cleanup lane deletes this file; if it still exists when the tool runs
#                                               it is moved to <new path> (so the map is valid whichever lanes land)
#   <old path> -> <new path>  ADDED-BY:<lane>   a file a cleanup lane adds (absent on main; moved when present)
#   <old path> -> DROP                          a docstring-only __init__.py of a package that dissolves (the tool refuses if it
#                                               holds code); its prose is superseded by the README layout section
#   <old dir>/ -> <new dir>/                    prefix rule for files the lanes add inside an existing directory; exact lines win
#
# Layers (Draft 3.1 / README "Vocabulary"): program query correspondence observe acquire input_provenance commit check tp harness ops.
"""
with open(OUT, 'w') as fh:
    fh.write(hdr)
    for old, new, lane in lines:
        fh.write(f"{old} -> {new}" + (f"  DELETE-BY:{lane}" if lane else '') + "\n")
    fh.write("\n# prefix rules (for files added later inside these directories)\n")
    for pre, new, lane in P:
        fh.write(f"{pre} -> {new}" + (f"  DELETE-BY:{lane}" if lane else '') + "\n")
    fh.write("tests/ir/ -> tests/program/\nverity_capture/tests/ -> tests/observe/\nverity_capture/commit/tests/ -> tests/commit/\n"
             "verity_capture/bench/tests/ -> tests/harness/\nverity_capture/experimental/cb_a/tests/ -> tests/check/\n"
             "verity_capture/experimental/commit_integ/tests/ -> tests/acquire/\nverity_capture/telemetry/tests/ -> tests/harness/\n"
             "verity_capture/experimental/ -> verity_vllm/check/\nverity_capture/commit/ -> verity_vllm/commit/\n"
             "verity_capture/tp/ -> verity_vllm/tp/\nverity_capture/telemetry/ -> verity_vllm/harness/telemetry/\n"
             "verity_capture/ -> verity_vllm/observe/\nverity_vllm_adapter/ -> verity_vllm/observe/\nverity_vllm/ -> verity_vllm/\ntests/ -> tests/\ntools/ -> tools/\n")
    fh.write("""
# ghosts (DELETE-BY:ghost): modules that no longer exist but are still named in prose / test tables; named as their layer would
verity_capture/experimental/cb_a/artifact.py -> verity_vllm/check/artifact.py  DELETE-BY:ghost
verity_capture/experimental/cb_a/artifact_identity.py -> verity_vllm/check/artifact_identity.py  DELETE-BY:ghost
verity_capture/experimental/cb_a/applicability.py -> verity_vllm/check/applicability.py  DELETE-BY:ghost
verity_capture/sweep/match_diff.py -> verity_vllm/harness/match_diff.py  DELETE-BY:ghost
verity_capture/bench/add_checkpoint.py -> verity_vllm/input_provenance/add_checkpoint.py  DELETE-BY:ghost
verity_vllm_adapter/derived_b0_check.py -> verity_vllm/check/poc_derived_b0_check.py  DELETE-BY:ghost
verity_vllm_adapter/derive_logprobs.py -> verity_vllm/check/poc_derive_logprobs.py  DELETE-BY:ghost
verity_vllm_adapter/verify_cpu.py -> verity_vllm/check/poc_verify_cpu.py  DELETE-BY:ghost
verity_vllm_adapter/verify.py -> verity_vllm/check/poc_verify.py  DELETE-BY:ghost
verity_vllm_adapter/conformance.py -> verity_vllm/check/poc_conformance.py  DELETE-BY:ghost
verity_vllm_adapter/binding.py -> verity_vllm/commit/poc_binding.py  DELETE-BY:ghost
verity_vllm_adapter/production.py -> verity_vllm/observe/poc_production.py  DELETE-BY:ghost
verity_vllm_adapter/tp_worker.py -> verity_vllm/tp/poc_tp_worker.py  DELETE-BY:ghost
verity_vllm/ir/store_nan_scan.py -> verity_vllm/check/store_nan_scan.py  DELETE-BY:ghost

# files other lanes add / remove (seen on lane/vllm-cleanup-2 @ 6811222)
tests/test_pod_bootstrap.py -> tests/ops/test_pod_bootstrap.py  ADDED-BY:vllm-cleanup-2
verity_vllm/ir/store_nan_scan.py -> verity_vllm/check/store_nan_scan.py  DELETE-BY:ghost

# files other lanes add / remove (seen on lane/vllm-cleanup-2 @ 6811222)
tests/test_pod_bootstrap.py -> tests/ops/test_pod_bootstrap.py  ADDED-BY:vllm-cleanup-2

# explicit literal rewrites (REWRITE <old file> :: <old literal> => <new literal>; `<DEL>` removes the lines holding the literal).
# These are the places where `verity_capture` / `verity_vllm_adapter` name a PACKAGE (their members now all live in verity_vllm):
# the generic passes cannot know whether a bare package name in a tuple should become verity_vllm or disappear.
REWRITE verity_capture/sweep/source_identity.py :: "verity_capture": ("verity_capture", "integrations/vllm"), => <DEL>
REWRITE verity_capture/sweep/source_identity.py :: "verity_vllm_adapter": ("verity_vllm_adapter", "integrations/vllm"), => <DEL>
REWRITE verity_capture/sweep/source_identity.py :: "verity", "verity_vllm", "verity_capture", "verity_vllm_adapter" => "verity", "verity_vllm"
REWRITE verity_capture/sweep/source_identity.py :: (verity, verity_vllm, verity_capture, verity_vllm_adapter, hidden_gpu, research) => (verity, verity_vllm, hidden_gpu, research)
REWRITE verity_capture/tests/test_source_identity.py :: "verity", "verity_vllm", "verity_capture", "verity_vllm_adapter" => "verity", "verity_vllm"
REWRITE verity_capture/tests/test_source_identity.py :: doc["packages"]["verity_capture"] => doc["packages"]["verity_vllm"]
REWRITE verity_capture/tests/test_source_identity.py :: assert doc["packages"]["verity_vllm_adapter"]["sys_path_entry"] == os.path.realpath(INTEGRATION) => <DEL>
REWRITE verity_capture/tests/test_source_identity.py :: # verity_capture (the package that ran the precheck) => # verity_vllm (the package that ran the precheck)
REWRITE verity_capture/tests/test_source_identity.py :: `verity_vllm` / `verity_capture` / => `verity_vllm` from integrations/vllm,
REWRITE verity_capture/tests/test_source_identity.py :: `verity_vllm_adapter` from integrations/vllm, `research` from tools/research/src. => `research` from tools/research/src.
REWRITE verity_capture/tests/test_source_identity.py :: "integrations/vllm/verity_vllm_adapter", "tools/research/src/research/telemetry" => "tools/research/src/research/telemetry"
REWRITE verity_capture/sweep/hot_commit.py :: CODE_ROOTS = ("verity_capture", "verity_vllm", "verity_vllm_adapter", "verity_vllm_sampler", "verity_vllm_numerics", => CODE_ROOTS = ("verity_vllm", "verity_vllm_sampler",
REWRITE verity_capture/bench/tp2_commit.py :: roots = ["verity_capture", "record_v5", "e2e", "verity_vllm", "verity_vllm_adapter"] => roots = ["record_v5", "e2e", "verity_vllm"]
REWRITE verity_capture/tests/test_ship_roots.py :: for r in ("verity_capture", "record_v5", "verity.ir", "workloads"): => for r in ("verity_vllm", "record_v5", "verity.ir", "workloads"):
REWRITE verity_capture/experiment.py :: "--", "verity_capture", "verity_vllm", => "--", "verity_vllm",
REWRITE verity_capture/sweep/finalize.py :: "verity_capture.sweep.finalize/v1" => "verity_vllm.harness.finalize/v1"
REWRITE verity_capture/sweep/research_result.py :: "verity_capture.sweep.row_pod stages.txt" => "verity_vllm.ops.row_pod stages.txt"
REWRITE verity_capture/sweep/row_pod.sh :: (verity verity_vllm verity_capture verity_vllm_adapter hidden_gpu research) => (verity verity_vllm hidden_gpu research)
REWRITE verity_capture/sweep/row_pod_tp2.sh :: (verity verity_vllm verity_capture verity_vllm_adapter hidden_gpu research) => (verity verity_vllm hidden_gpu research)
REWRITE tests/test_no_by_name_rules.py :: SCAN = ("verity_vllm", "verity_capture") => SCAN = ("verity_vllm",)
REWRITE tests/test_no_by_name_rules.py :: `verity_vllm` / `verity_capture` => `verity_vllm`
REWRITE tests/test_no_by_name_rules.py :: integrations/vllm/{verity_vllm,verity_capture} => integrations/vllm/verity_vllm
REWRITE tests/by_name_allowlist.json :: integrations/vllm/{verity_vllm,verity_capture} => integrations/vllm/verity_vllm
REWRITE tests/ir/test_artifact_applicability_independent.py :: SOURCE_ROOTS = ("verity.ir", "verity_capture", => SOURCE_ROOTS = ("verity.ir", "verity_vllm",
REWRITE tests/ir/padded_commit_tiny.py :: "verity_vllm.commit_bench (production host core => "verity_vllm.commit.{semantic_layout,stream_merkle} (production host core
REWRITE tests/ir/padded_commit_tiny.py :: "verity_vllm.commit_bench (production host core => "verity_vllm.commit.{semantic_layout,stream_merkle} (production host core
REWRITE pod_bootstrap.sh :: check("verity_capture_from_tree", lambda: under("verity_capture", root)) => <DEL>
REWRITE verity_capture/quarantine_lint.py :: "numpy.random", "verity_capture", "importlib", => "numpy.random", "verity_vllm.observe", "verity_vllm.acquire", "verity_vllm.commit", "verity_vllm.check", "importlib",
REWRITE verity_capture/quarantine_lint.py :: numpy.random, verity_capture (any), importlib, => numpy.random, verity_vllm.{observe,acquire,commit,check} (any), importlib,
""")
print(f"mapped {len(lines)} files; unmapped {len(unmapped)}")
for u in unmapped: print("  UNMAPPED", u)
# collisions among non-test targets
tg = collections.Counter(new for _, new, _ in lines if new != 'DROP')
for k, v in tg.items():
    if v > 1: print("  COLLISION", k, [o for o, n, _ in lines if n == k])
