"""The admission planner's facts (`pipeline.telemetry.admission`): the per-model table, the dense default for a model it
has no coefficients for, and the measured constants, each with the evidence that pins it."""
from __future__ import annotations

from typing import Any

MiB = 1 << 20
GiB = 1 << 30
#: The model an attempt is planned as when its directory names no row.
DEFAULT_MODEL = "llama32-1b"

# ---------------------------------------------------------------------------------------------------------------------
# Model table: structural constants + measured per-model coefficients.  `instances_per_layer_token` and
# `per_token_bytes` are structural (they follow from the declaration / the op graph), not tunables.
# ---------------------------------------------------------------------------------------------------------------------
MODELS: dict[str, dict[str, Any]] = {
    "llama32-1b": {"family": "dense", "layers": 16, "hidden": 2048, "heads": 32, "dtype_bytes": 2, "weights_bytes": 2_471_645_608,
                       "per_token_bytes": 1_437_700, "instances_per_layer_token": 10.1, "hidden_per_token_K": 4.4e-3,
                       "c2_oracle_per_instance": 13_000, "c2_programs_per_instance": 14_500, "build_graph_per_row_ctx_layer": 3_300, "largest_tensor_per_prompt": 392_500_000},
    "phi3-mini": {"family": "dense", "layers": 32, "hidden": 3072, "heads": 32, "dtype_bytes": 2, "weights_bytes": 7_642_159_168,
                      "per_token_bytes": 3_950_596, "instances_per_layer_token": 10.1, "hidden_per_token_K": 4.0e-3,
                      "c2_oracle_per_instance": 17_500, "c2_programs_per_instance": 21_500, "build_graph_per_row_ctx_layer": 6_800, "largest_tensor_per_prompt": 554_000_000},
    "mistral-7b": {"family": "dense", "layers": 32, "hidden": 4096, "heads": 32, "dtype_bytes": 2, "weights_bytes": 14_483_464_192,
                       "per_token_bytes": 5_500_000, "instances_per_layer_token": 10.1, "hidden_per_token_K": 4.4e-3,
                       "c2_oracle_per_instance": 17_500, "c2_programs_per_instance": 21_500, "build_graph_per_row_ctx_layer": 6_800,
                       "capture_lower_bound": "R17 #60: capture 133,461 MiB observed vs 102,249 predicted; the wide-op operand snapshot (intermediate 14,336) is unmodelled"},
    "gemma2-2b": {"family": "dense", "layers": 26, "hidden": 2304, "heads": 8, "dtype_bytes": 2, "weights_bytes": 5_228_683_776,
                      "per_token_bytes": 2_500_000, "instances_per_layer_token": 40.5, "hidden_per_token_K": 1.1e-3,
                      "c2_oracle_per_instance": 17_500, "c2_programs_per_instance": 21_500, "build_graph_per_row_ctx_layer": 6_800,
                      "capture_lower_bound": "R17 #57: capture >=198,347 MiB observed vs 122,673 predicted; GeGLU operand snapshot at 40.5 instances/layer-token unmodelled"},
    "olmoe-1b-7b": {"family": "moe", "layers": 16, "hidden": 2048, "heads": 16, "dtype_bytes": 2, "weights_bytes": 13_887_000_000,
                        "per_token_bytes": 2_374_660, "instances_per_layer_token": 36.0, "hidden_per_token_K": 1.8e-3,
                        "c2_oracle_per_instance": 4_000, "c2_programs_per_instance": 5_000, "build_graph_per_row_ctx_layer": 3_300, "largest_tensor_per_prompt": 310_600_000},
    "qwen3-4b-fp8": {"family": "dense", "layers": 36, "hidden": 2560, "heads": 32, "dtype_bytes": 1, "weights_bytes": 4_400_000_000,
                         "per_token_bytes": 2_800_000, "instances_per_layer_token": 10.1, "hidden_per_token_K": 4.4e-3,
                         "c2_oracle_per_instance": 17_500, "c2_programs_per_instance": 21_500, "build_graph_per_row_ctx_layer": 6_800},
}
# Unknown dense model: the largest measured dense coefficients (phi3), named as such in the plan.
DENSE_DEFAULT = {"family": "dense", "layers": 32, "hidden": 4096, "heads": 32, "dtype_bytes": 2, "weights_bytes": 16 * GiB, "per_token_bytes": 4_000_000,
                     "instances_per_layer_token": 10.1, "hidden_per_token_K": 4.4e-3, "c2_oracle_per_instance": 17_500,
                     "c2_programs_per_instance": 21_500, "build_graph_per_row_ctx_layer": 6_800, "largest_tensor_per_prompt": 554_000_000}


def unknown_model_note(model: str) -> str:
    """The plan's note for an attempt whose model `MODELS` doesn't have: it is planned under `DENSE_DEFAULT`."""
    return f"model {model!r} not in MODELS: dense defaults (phi3 coefficients) used"


def uncalibrated_model_note(key: str) -> str:
    """The plan's note for a workload row whose model `MODELS` doesn't have: every term comes from `DENSE_DEFAULT`."""
    return f"no calibration for model '{key}': dense default coefficients (phi3's, the largest measured dense set) -- every term is an estimate"

# ---------------------------------------------------------------------------------------------------------------------
# Measured constants.  value + unit + the evidence that pins it.  These are the only numbers in the model.
# ---------------------------------------------------------------------------------------------------------------------
CALIBRATION: dict[str, dict[str, Any]] = {
    "serving.engine_host": {"value": 3_200 * MiB, "unit": "bytes", "what": "committer process before the first instrumented run: python + torch + vLLM client + CUDA host context",
                                "source": "prep.committer_setup max_python_rss_mib: b1_llama 2314, llama_b8 2410, r11 2385, r22 3182, r23 2955 (max 3182)"},
    "serving.weights_mmap": {"value": 1.0, "unit": "x safetensors bytes", "what": "model weights file-mapped by the engine during load; file-backed, resident (unreclaimable while mapped)",
                                 "source": "#67 commit_oom_cgroup: 18 GiB 'file' at the kill vs OLMoE-1B-7B safetensors 13.9 GB; rss_file of the engine"},
    "process.python_base": {"value": 3_000 * MiB, "unit": "bytes", "what": "a CPU-side Verity process (build derive / match fold / capture observer) before its own objects: interpreter + torch + program imports",
                                "source": "build.derive_step 1059-3000 MiB (r60, r57); match.resolve 1963-6898 MiB"},
    "build.program_graph.per_row_ctx_layer": {"unit": "bytes per (decode row x context token x layer)", "what": "Program object graph during derive_request: per-step decode graphs reference the reachable context",
                                                  "source": "r11 llama LP4096/T512: 126057 MiB -> 3.3 KB; phi3 LP1024_T54: 15638 MiB -> 6.8 KB; llama_b8 LP1024_T64: 8603 MiB -> 5.0 KB"},
    "match.capture.instance_records": {"value": 12_300, "unit": "bytes per instance", "what": "capture observer: one record per matched instance (ids, operand digests, positions)",
                                           "source": "fit over llama_b8 (38141 MiB, 698k inst), r11 (106351, 747k), r23 (141211, 2.86M), r67 (126273, 5.7M) with the two value terms below"},
    "match.capture.token_values": {"value": 1.03, "unit": "x scheduled token_bytes", "what": "capture observer: snapshot of the committed per-token values (declaration widths x tokens)",
                                       "source": "same fit; token_bytes from the F-dA-15 composition (llama_b8 5913 MiB, r23 24214, r67 22571)"},
    "match.capture.hidden_values": {"value": 0.67, "unit": "x scheduled hidden_bytes", "what": "capture observer: the M1 hidden-stream snapshot (heads x prompt x NB per arriving prompt)",
                                        "source": "r11 vs llama_b8: +101 GB hidden_bytes -> +68 GB RSS (0.67); r23 check 152 vs 141 GB observed"},
    "match.fold.instance_state": {"value": 26_000, "unit": "bytes per instance (dense); 12000 (moe)", "what": "fold: folded instance dicts + per-instance program rows",
                                      "source": "llama_b8 27551 MiB -> 28 KB; r23 86639 -> 25.5 KB; r60 49426 -> 21 KB; r67 82881 -> 12 KB (moe)"},
    "match.fold.attention_operands": {"value": 3_400, "unit": "bytes per (step x context token x layer)", "what": "fold: per-step attention operand rows over the reachable context",
                                          "source": "r11 149634 MiB fold with steps 512 x K 4608 x 16 layers -> 3.4 KB (the only long-context fold); llama_b8 check 25 vs 27.5 GB"},
    "commit.pinned_pool.levels_factor": {"value": 1.25, "unit": "x scheduled raw bytes", "what": "pinned host staging pool: raw committed bytes + Merkle levels, allocated during warm-up, resident until commit_delta exits under --retain host (torch host allocator caches it)",
                                             "source": "F-dA-15 ledger vs 'staging (pinned, in use)': phi3 108803 vs 107614 MiB (0.99); llama_b8 44715 vs 42017 (0.94); r22 103352 vs 98195 (0.95)"},
    "commit.run_record.bytes_per_leaf": {"value": 268, "unit": "raw bytes per leaf", "what": "per-run leaf digest table: 32 B per leaf, leaves = raw / 268",
                                             "source": "phi3 commit.log: 87,073,246,248 B in 340,129,939 leaves -> 256 B/leaf raw; x32 B digest = 10.9 GB (observed warm-up excess 15.6 GB over pool+engine; r22 16.6 GB)"},
    "commit.binding.population_index": {"value": 2_000, "unit": "bytes per instance", "what": "binding map + population/coverage structures (map entries ~0.43 x instances)",
                                            "source": "binding_coverage RSS deltas: llama_b8 5.6 GB/698k = 8 KB; r22 3.2 GB/1.62M = 2 KB; phi3 1.9 GB/1.3M = 1.4 KB; r67 6.5 GB/5.7M = 1.1 KB (median 2 KB; spread noted)"},
    "commit.c2.evaluator_base": {"value": 6_700 * MiB, "unit": "bytes", "what": "C2 evaluator fixed cost: program registry, family tables, sampled-replay bookkeeping (independent of instance count)",
                                     "source": "b1_llama: oracle+replay delta 11.8 GB - 187k x 27.5 KB = 6.7 GB; llama_b8 check 25.9 vs 24.4 GB"},
    "commit.c2.match_oracle.per_instance": {"unit": "bytes per instance (per model)", "what": "MatchOracle: match/instances parsed into python objects for oracle_compare; freed at pair end (heap residual retained)",
                                                "source": "oracle_compare RSS deltas: llama_b8 8.8 GB/698k; r22 21.5 GB/1.62M -> 13 KB; phi3 25.7 GB/1.3M -> 17.5 KB net of base; #67 (moe) anon at kill bounds oracle+programs <= 8.6 KB"},
    "commit.c2.programs.per_instance": {"unit": "bytes per instance (per model)", "what": "sampled-replay Program object graph (pair 0 only under the replay cache); freed after pair 0 (heap residual retained)",
                                            "source": "sampled_replay RSS deltas: llama_b8 15.6 GB/698k; r22 23 GB/1.62M -> 14.5 KB; phi3 32.3 GB/1.3M -> 21.5 KB net of base"},
    "commit.c2.heap_residual.retention": {"value": 0.91, "unit": "fraction of pair-0 C2 heap", "what": "pymalloc arenas / glibc heap not returned after pair 0's oracle+programs are freed; resident for the process lifetime",
                                              "source": "RSS after pair-0 release minus pre-C2 RSS over the pair-0 C2 delta: r22 40.4/44.5 = 0.91; llama_b8 21/24.4 = 0.86; phi3 47.2/58 = 0.81; b1 9.2/11.8 = 0.78 (max used)"},
    "commit.c2.replay_workers.small_vu": {"value": 250 * MiB, "unit": "bytes per worker", "what": "a forked replay worker evaluating ordinary VUs: its private (COW-dirtied + evaluator scratch) pages",
                                              "source": "r22 laneC, 32 forked workers over 800 s: cgroup anon+shmem - parent RSS plateau 6.7-7.6 GB -> <= 240 MB each, no spike"},
    "commit.c2.replay_workers.largest_vu": {"value": 2.9, "unit": "x largest committed tensor bytes", "what": "a worker evaluating the largest VU (the step-0 hidden-M1 stream: heads x prompt x NB for the step's prompts) materialises its decoded operand + the recompute + evaluator temporaries",
                                                "source": "#23 B64 commit2: exec2 ps at the kill: two workers at ~72 GB each vs 'largest tensor bound 25120 MiB' -> 2.9x; cgroup 359518 - parent 256722 = >= 102.8 GB in children"},
    "commit.c2.replay_workers.largest_vu_picks": {"value": 2, "unit": "workers", "what": "sampled_replay picks per stratum (per-stratum 2): the largest stratum's two VUs can be in two workers at once",
                                                      "source": "commit.log 'C2 SAMPLED REPLAY ... (per-stratum 2, ...)' on every R17 row"},
    "commit.largest_tensor.per_prompt": {"unit": "bytes per prompt at 1024/128 (per model), x (LP x K)/(1024 x 1152)", "what": "the largest committed tensor (windowed staging ledger): scales with the prompts in the step and LP x K",
                                             "source": "'largest tensor bound': llama B1 392, B8 3140, B32 12560, B64 25120 MiB (392.5 x B); r11 LP4096 6274 = 392 x 16; phi3 B8 4228 (528.5 x B); olmoe B32 9480 (296 x B)"},
    "margin.pod_cotenants": {"value": 2_600 * MiB, "unit": "bytes", "what": "row_pod.sh, hot_commit client, the resource sampler, sshd: pod-resident processes outside the row's stage process",
                                 "source": "r22 laneC anon+shmem before commit_delta: 2512-2521 MiB; r23 commit.prepare 987-2955 MiB"},
    "margin.kernel_unreclaimable": {"value": 2 * GiB, "unit": "bytes", "what": "kernel memory charged to the cgroup that the OOM path could not reclaim (slab, page tables of a 250 GB heap, dirty pages)",
                                        "source": "#67: 175 GiB limit = 154 GiB anon+shmem + 18 GiB file at the kill; #11 memory.peak = memory.max"},
    "instances_per_layer_token": {"unit": "instances per (layer x token)", "what": "fold instance count law: n_instances = c x layers x (prompt tokens + decode rows)",
                                      "source": "llama_b8 698398/(16 x 4313) = 10.1; phi3 1315324/(32 x 4090) = 10.05; r22 1622848/(16 x 10034) = 10.1; r11 747358/(16 x 4608) = 10.1; r60 1211108/(32 x 3758) = 10.07; r57 gemma2-2b 4368950/(26 x 4150) = 40.5; r67 olmoe 5745046/(16 x 9967) = 36.0"},
    "fa2.tap_window.device": {"value": int(24.53 * GiB), "unit": "bytes (device)", "what": "FA2 hidden-M1 tap window on device: heads x prompt x NB for the largest step, ring of 8 x 256 MiB + scratch",
                                  "source": "F-r17-23: 24.53 GiB device; r23 commit.log 'windowed staging (P0): 8 x 256 MiB ring + scratch (largest tensor bound 25120 MiB)'"},
}
