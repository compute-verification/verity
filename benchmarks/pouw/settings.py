"""Every environment variable PoUW's benchmarks read, declared once (`verity_config.settings`): the Pearl-C kernels'
(``pearl_c_sm120``, ``pearl_c_sm90``, ``pearl_c4``), their vLLM arm (``pearl_c_vllm``), the GPU harness (``harness``) and
served H1 (``served_h1``).  Code reads them through ``of(__file__)``::

    from verity_config.settings import of
    S = of(__file__)
    S.PEARLC_SCHEDULE

The registry owns ``PEARLC_``, ``PEARLC4_`` and ``PEARL_C4_``: a variable with one of them that nothing declares fails the
first `of` of a process.  It declares the ``POUW_`` variables these directories read without owning the prefix, which the
distribution's other directories read too.

The Pearl-C runners and the SASS gate (`STANDALONE`) run outside the tree with stdlib alone: a runner beside its cubin on the
card, or loaded by path from a kernel directory (the vLLM integration's ``pearl_c``), the gate and ieee_pin.py copied beside a
build.  So they read the environment themselves; each name they read is declared here, and ``tests/test_settings.py`` holds
their reads to these declarations.
"""
from __future__ import annotations

from verity_config.settings import DEBUG, VALUE, WHERE, Registry, Setting as S, ints, not_zero, one, present, words

#: the files that run outside the tree, relative to this one
STANDALONE = ("pearl_c_sm120/run.py", "pearl_c_sm90/run.py", "pearl_c_sm90/grouped.py", "pearl_c4/dev.py",
              "harness/sass_gate.py", "harness/ieee_pin.py")


def places(raw: str) -> frozenset[str]:
    """``+``-joined names; ``base`` (and an empty name) is none."""
    return frozenset(raw.split("+")) - {"", "base"}


LIB = "a library the runner loads by name, which a test points at its stand-in; the driver it opens computes nothing itself"

REGISTRY = Registry(
    __file__, ("PEARLC_", "PEARLC4_", "PEARL_C4_"),
    # ---------------------------------------------------------------- the Pearl-C runners (pearl_c_sm120, pearl_c_sm90)
    S("PEARLC_SPLIT", not_zero, True, VALUE, "decode's split kernels (words_k, stats_r); 0 runs the kernels they replace "
                                             "(words, stats_s5), the same words"),
    S("PEARLC_LIVE", not_zero, True, VALUE, "the 64 x 64 GEMMs on the real rows alone; 0 runs (sm90: stores) them over all m "
                                            "rows, the same real rows"),
    S("PEARLC_DEEP", not_zero, True, VALUE, "decode's chain on 128-deep stages; 0 runs the 64-deep one it replaces, the same "
                                            "words"),
    S("PEARLC_PREFETCH_FORM", int, 2, VALUE, "prefetch_l2's form for B~ ahead of a decode call's chain: 0 a line a thread, 1 the "
                                             "same evict_last, 2 a 16 KiB bulk prefetch a thread"),
    S("PEARLC_PREFETCH_CTAS", int, 64, VALUE, "prefetch_l2's blocks"),
    S("PEARLC_PREFETCH_MB", int, 0, VALUE, "B~'s leading MiB the prefetch takes; 0 all of them"),
    S("PEARLC_WORDS_WIN", not_zero, True, VALUE, "sm90: A'F_B a CTA a window (gemm_words_win_d); 0 runs gemm_words_d and pa"),
    S("PEARLC_FORM_EP", not_zero, True, VALUE, "sm90: A's E' in its forming's launch (form_s5_ep); 0 runs eprime and form_s5"),
    S("PEARLC_WORDS_FORM", not_zero, True, VALUE, "sm90: A'F_B's windows in the forming's launch too (form_s5_epw); 0 runs "
                                                  "gemm_words_win_d"),
    S("PEARLC_GRID_BALANCE", not_zero, True, VALUE, "sm90: a persistent GEMM takes the fewest CTAs that keep its rounds of "
                                                    "tiles; 0 takes every SM it may"),
    S("PEARLC_ONE_COMMIT", not_zero, True, VALUE, "sm90: every call's A commitment launches one pd_commit variant; 0 launches "
                                                  "the shape's"),
    S("PEARLC_PDL", not_zero, True, VALUE, "sm90: the PDL kernels launch with programmatic stream serialization; 0 plainly"),
    S("PEARLC_BF16_ROWS", not_zero, True, VALUE, "sm90: a split pipeline's pd_commit reads a call's BF16 rows in its FP32 "
                                                 "words' place; 0 says no"),
    S("PEARLC_RS_FAST_NT", int, 128, VALUE, "hash_fast.cuh's segment kernels' block, as the cubin was built"),
    S("PEARLC_SHA", str, "fast", VALUE, "the SHA-512 kernels: fast (hash_fast.cuh's segments and folds) or ref"),
    S("PEARLC_CHAIN", str, "pl", VALUE, "sm90: the GEMM chain (run.py's CHAINS)"),
    S("PEARLC_LIBCUDA", str, "libcuda.so.1", WHERE, f"the CUDA driver: {LIB}"),
    S("PEARLC_LIBNVML", str, "libnvidia-ml.so.1", WHERE, f"NVML, for clocks and power: {LIB}"),
    S("PEARLC_EXPECT_UUID", str, None, WHERE, "the UUID the one visible GPU must have (the lease's); a mismatch exits before "
                                              "any launch"),
    S("PEARLC_ANY_GPU", present, False, DEBUG, "skip the GPU identity check: a dry run on a stand-in driver"),
    S("PEARLC_UNGATED", present, False, DEBUG, "time regardless of the check, every row marked ungated (with PEARLC_ANY_GPU)"),
    S("PEARLC_GROUPED_PLACE", places, frozenset(), DEBUG, "sm90 grouped: the older forms, +-joined, for an A/B in one "
                                                          "session (grouped.py's PLACE)"),
    S("VARIANTS", words, None, VALUE, "the variants a runner checks and times; unset, its own (sm120 and sm120-unpromoted, "
                                      "h100); empty, none", empty=True),
    S("SHAPES", words, ("8192",), VALUE, "the timed shapes, N (m = n = k) or MxNxK; empty, the check alone", empty=True),
    S("REPS", int, 5, VALUE, "each kernel's timed repetitions after a warm-up"),
    # ---------------------------------------------------------------- pearl_c4
    S("PEARLC4_LIBCUDA", str, "libcuda.so.1", WHERE, f"the CUDA driver: {LIB}"),
    S("PEARLC4_CUBIN", str, None, VALUE, "the cubin the Pearl-C4 arm loads; unset, pearl_c4.cubin beside it"),
    S("PEARLC4_VERIFY_WORKERS", int, 16, WHERE, "the verifier's worker processes: how fast it runs, not what it checks"),
    S("PEARLC4_GATE_TILES", int, 2, VALUE, "the tiles the arm's gate checks a call"),
    S("PEARL_C4_F1", str, None, VALUE, "real.py's F1 check: fast checks a drawn tile through the fast path, else exact "
                                       "(real.py --f1 sets it for the processes it starts)"),
    S("VERITY_COMMIT", str, None, WHERE, "the commit a Pearl-C4 replay names for its provenance; unset, unknown"),
    # ---------------------------------------------------------------- the vLLM arm (pearl_c_vllm)
    S("PEARLC_KERNEL", str, None, VALUE, "the kernel directory the arm serves with; unset, the tree's pearl_c_sm120"),
    S("PEARLC_CUBIN", str, None, VALUE, "the cubin the arm loads; unset, the kernel directory's build.sh cubin"),
    S("PEARLC_SCHEME", str, None, VALUE, "the arm's scheme record; unset, the kernel directory's (bench_arm.KERNEL_SCHEMES)"),
    S("PEARLC_SCHEDULE", str, "deferred", VALUE, "when the arm's hashes run: deferred (the served schedule) or serial"),
    S("PEARLC_BENCH_EPOCH", int, 20261014, VALUE, "the arm's PoUW epoch"),
    S("PEARLC_CHALLENGER_GATE", str, None, WHERE, "infra's gate as an ssh command: with PEARLC_CHALLENGER_PEER, the salt coin "
                                                  "is a live session's on the auditor host"),
    S("PEARLC_CHALLENGER_PEER", str, None, WHERE, "this node's name in the challenger host's peers"),
    S("PEARLC_CHALLENGER_LANE", str, "vllm", WHERE, "the lane the gate key is registered for"),
    S("PEARLC_GPU_UTIL", float, 0.55, VALUE, "vLLM's gpu_memory_utilization under the arm"),
    S("PEARLC_STEP_TOKENS", int, 4096, VALUE, "vLLM's max_num_batched_tokens under the arm"),
    S("PEARLC_WHOLE_DEFER_A", not_zero, True, VALUE, "the whole-step arm's whole_defer_a option; 0 leaves it out"),
    S("PEARLC_WHOLE_FORK_STATS", not_zero, True, VALUE, "the whole-step arm's whole_fork_stats option; 0 leaves it out"),
    S("PEARLC_LAG_ENDS", not_zero, True, VALUE, "the whole-step arm's lag_ends option; 0 leaves it out"),
    S("PEARLC_WHOLE_TREE_SM", not_zero, True, VALUE, "the whole-step arm's whole_tree_sm option; 0 leaves it out"),
    S("PEARLC_ASIDE_ENDS", not_zero, True, VALUE, "the whole-step arm's aside_ends option; 0 leaves it out"),
    S("PEARLC_WHOLE_PREFETCH_B", not_zero, True, VALUE, "the whole-step arm's whole_prefetch_b option (pearl_c_sm120's); 0 "
                                                        "leaves it out"),
    S("PEARLC_WHOLE_SHAPE_JOINS", one, False, VALUE, "1 adds the whole-step arm's whole_shape_joins option"),
    S("PEARLC_WHOLE_EAGER", str, None, VALUE, "the whole-step arm's whole_eager form (pearl_c_sm120's); 0 leaves it out"),
    S("PEARLC_WHOLE_EXECUTION", str, "compiled", VALUE, "the whole-step arm's vLLM execution"),
    S("PEARLC_CAPTURE_SIZES", ints, None, VALUE, "the whole-step arm's CUDA graph capture sizes; unset, 1, 8, 32 and 128"),
    S("VLLM_ENABLE_V1_MULTIPROCESSING", str, None, WHERE, "vLLM's engine core in a process of its own; the arm's scripts set 0, "
                                                          "so its hooks and profiles see the engine in theirs"),
    S("VERITY_INDUCTOR_DIR_PER_PROCESS", one, False, WHERE, "1: TORCHINDUCTOR_CACHE_DIR is this process's own (the vLLM "
                                                            "integration's engine.env.INDUCTOR_DIR_MARK), else served_step.py "
                                                            "gives a serving-defaults arm a fresh one"),
    S("E2E_GPU_UUID", str, "0", WHERE, "the GPU nvidia-smi reads (window.sh sets the lease's), since NVML ignores "
                                       "CUDA_VISIBLE_DEVICES"),
    S("POUW_COIN_STATE", str, None, WHERE, "a file holding the challenger's entropy (e2e.py's epoch coins), whose commitment "
                                           "the run publishes"),
    S("PANEL_BY", str, "bc-dd22acf8", WHERE, "who panel_rows.py labels its rows by"),
    S("SERVED_TREE", str, None, WHERE, "the source tree served_profile.py profiles; unset, the working directory"),
    # ---------------------------------------------------------------- the harness
    S("POUW_CUDA_BIN", str, None, WHERE, "the CUDA toolkit's bin directory the SASS gate takes cuobjdump and nvdisasm from"),
    S("POUW_SASS_GATE_CACHE", str, None, WHERE, "the SASS gate's cache; unset, ~/.cache/pouw-harness/sass-gate"),
    S("POUW_HARNESS_LIB", str, None, WHERE, "the harness's native library; unset, native/build/libpouw_harness.so"),
    S("POUW_HARNESS_LT2", str, None, VALUE, "the second cuBLASLt's library; unset, libpouw_lt2.so beside the harness's if "
                                            "built; none leaves it out"),
    S("POUW_LIBNVML", str, "libnvidia-ml.so.1", WHERE, f"NVML, for the harness's clocks and power: {LIB}"),
    S("POUW_TIER", str, "2a", VALUE, "the evidence tier verify.py records a variant at (ledger.TIERS)"),
    S("POUW_LEDGER_PUBLISH", one, False, WHERE, "1: verify.py publishes the variant's record to the ledger"),
    # ---------------------------------------------------------------- served H1
    S("POUW_CHALLENGER", str, None, WHERE, "the window's challenger session's address (infra's gate)"),
    S("POUW_CHALLENGER_SESSION", str, None, WHERE, "the window's challenger session's id"),
    S("POUW_CHALLENGER_SECRET_FILE", str, None, WHERE, "the file holding the session's secret"),
    S("POUW_CHALLENGER_COMMITMENT", str, None, WHERE, "the session's commitment"),
    S("POUW_CHALLENGER_GATE", str, None, WHERE, "infra's gate, USER@HOST"),
    S("H1_SKIP_DEAD", one, False, VALUE, "1: served H1's producer skips the units of row blocks wholly past the served rows"),
    standalone=STANDALONE,
)
