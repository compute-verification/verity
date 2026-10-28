"""Every environment variable the integration reads, declared once: vLLM's ``vllm/envs.py``, for ``verity_vllm``.  `REGISTRY` is
a `Registry` of ``verity/settings.py``, the machinery every registry in the tree is built on, which says what a `Setting` is,
what its roles (`VALUE`, `WHERE`, `DEBUG`) mean and how a read parses; this module holds the integration's declarations and
forwards its names to `REGISTRY`.

A setting is ``value`` unless it is ``where`` (only where and how long things run: paths, devices, hosts, interpreters, run ids
and liveness bounds) or ``debug`` (diagnostics that change nothing a run computes: dumps, traces, profiles, injected delays).
`record` carries every value setting in force, and the value families (`FAMILIES`), so a value setting can't stay out of a run's
record.  A credential's value never enters a record: `record` names it with `REDACTED` in place of its value, and so does
`redact`, through which every other record that copies the environment by family passes it.

Read a setting as an attribute, ``settings.VERITY_WINDOW_MB`` (or `get` by name), and write one with `put`, `setdefault` and
`unset`; `environ` is a copy of the whole environment, for code that passes it on.  `is_set` here counts an empty string as
set, which `Registry.is_set` does only for a setting that keeps it (`Setting.empty`).

`validate_environ` refuses a ``VERITY_*`` variable that neither this registry nor the rest of the tree reads (`ELSEWHERE`), naming
it: ``verity-vllm`` runs it at startup, so a misspelt or retired setting fails the run instead of being ignored.

The library reads and writes ``os.environ`` here, through `REGISTRY`, and nowhere else (P7's ``environ`` kind,
``tests/lint/test_p07_declared_inputs.py``, whose allowlist holds the sites not yet moved here).  Nothing from ``verity_vllm``:
every layer may import it.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from verity.settings import (DEBUG, VALUE, WHERE, Registry, Setting, csv, flag, int_or_zero, lowered, not_off, not_zero, number,
                             one, paths, present, stripped, yes)
from verity.settings import REDACTED as REDACTED, ROLES as ROLES, SECRET_WORDS as SECRET_WORDS, UnknownSetting as UnknownSetting

RECORD_SCHEMA = "verity-vllm/settings/v1"

S = Setting
_DECLARED = (
    # ---- the Commit's committers (commit/committer/)
    S("VERITY_ABL", csv, (), VALUE, "parts of the host committer turned off, comma-separated (an A/B ablation)"),
    S("VERITY_ARENA_GROW", yes, False, VALUE, "the host arena may grow when no block can ever fit a step (else it waits)"),
    S("VERITY_ARENA_WAIT_S", float, 600.0, WHERE, "seconds the serving thread waits for a free host-arena block before failing"),
    S("VERITY_COLLECT_DEAD_GRACE_S", float, 120.0, WHERE, "seconds a dead collector waits for progress before its watchdog exits"),
    S("VERITY_COLLECT_WATCHDOG_MARKER", str, "", WHERE, "the file the collector's watchdog writes when it exits (the Commit sets it)"),
    S("VERITY_CONSUMER_DELAY_MS", float, 0.0, DEBUG, "milliseconds the collector's consumer sleeps per step (a stall injected for tests)"),
    S("VERITY_DRAIN_SIDE", flag, True, VALUE, "the collector drains a gather replay on its side stream (native_collect.cpp)"),
    S("VERITY_FA2_COMPACT", not_off, True, VALUE, "the attention tap writes its classes compacted"),
    S("VERITY_FA2_TAP_CAP_MB", int_or_zero, None, VALUE, "the attention tap's residency cap in MiB, 0 (or empty) unbounded; unset, the "
      "source's own", empty=True),
    S("VERITY_FA2_ZERO", not_off, True, VALUE, "the FA2 tap zeroes its buffers after each read"),
    S("VERITY_FA3_ZERO", not_off, None, VALUE, "the FA3 tap zeroes its buffers after each read (empty: on); unset, VERITY_FA2_ZERO's",
      empty=True),
    S("VERITY_FAULT", stripped, "", VALUE, "a deliberate fault the committer injects (negative tests): the record names it"),
    S("VERITY_FOOTPRINT", str, None, DEBUG, "the file the host committer traces its memory footprint to"),
    S("VERITY_GATHER_DESC_DEV", flag, True, VALUE, "the gather kernel reads its descriptors from device memory (native_collect.cpp)"),
    S("VERITY_HI_PRIO", one, True, VALUE, "1: native hooks stage on a high-priority stream (empty: off)", empty=True),
    S("VERITY_INLINE_D2H", one, False, VALUE, "1: the host committer copies device to host inline, not on its worker"),
    S("VERITY_LEAFHASH_BUILD", str, None, WHERE, "the leaf-hash extension's build directory (default: torch's)"),
    S("VERITY_LEARN_HOST_BUDGET_MB", float, 4096.0, VALUE, "the host budget in MiB the windowed collector learns its ring in"),
    S("VERITY_MEMTRACE", present, False, DEBUG, "the host committer prints its memory trace (any value, 0 included)"),
    S("VERITY_PIN", one, False, VALUE, "1: the host committer's worker pins its CPU affinity"),
    S("VERITY_PINNED_EXACT", flag, True, VALUE, "the pinned pool allocates exactly what is asked, not rounded up"),
    S("VERITY_PINNED_POPULATE_THREADS", int, None, VALUE, "helper threads that fault a fresh pinned pool in (`commit --pinned-populate-threads`)"),
    S("VERITY_RETAIN", lowered, None, VALUE, "where raw copies are retained: host, device, gpu, none or sample; unset, the committer's; "
      "empty is refused", empty=True),
    S("VERITY_RETAIN_EXCLUDE", str, "", VALUE, "families whose raw copies aren't retained (`native_collect.retain_exclude_families`)"),
    S("VERITY_RETAIN_PINNED", not_off, True, VALUE, "host retention goes to pinned memory (else pageable)"),
    S("VERITY_ROWS_FORM", str, None, VALUE, "the rows-sha512 leaf form, fused or split; unset, the device's (native_rows.cu)"),
    S("VERITY_ROWS_PHASES", int, None, VALUE, "the launches a row's blocks take in the split leaf form; unset, 4 (native_rows.cu)"),
    S("VERITY_SKIP", csv, (), VALUE, "module classes whose outputs a marginal-cost ablation doesn't acquire, comma-separated"),
    S("VERITY_STAGING_BOUNDED", yes, None, VALUE, "bounded staging; unset or blank, on exactly when the collector is windowed"),
    S("VERITY_WEIGHTS_HASH", str, "host", VALUE, "where the weights' leaves are hashed: host, device or device-checked; empty is "
      "refused", empty=True),
    S("VERITY_WINDOW_MB", number, None, VALUE, "the device staging window in MiB, 0 per-step blocks; unset or blank, the committer's "
      "default"),
    S("VERITY_WINDOW_SLOTS", int, None, VALUE, "device windows in the staging ring; unset, `native_collect.WINDOW_SLOTS`"),
    S("VERITY_WIN_WAIT_TIMEOUT_S", float, None, WHERE, "seconds the serving thread waits for a free window slot; unset, the collector's"),
    S("NATIVE_GATHER_BLOCKS", int, None, VALUE, "thread blocks per staged descriptor in the gather kernel; unset, 1024 size-adaptive, "
      "32 with native hooks, else 4"),
    S("NATIVE_GATHER_AUTO_BYTES", int, 32768, VALUE, "staged bytes per thread block of the size-adaptive gather"),
    # ---- the Commit (pipeline/commit.py and its helpers)
    S("VERITY_ADMIT_HIDDEN_FACTOR", float, 1.05, VALUE, "admission: the planned hidden-class bytes' safety factor"),
    S("VERITY_ADMIT_OTHER_FACTOR", float, 2.0, VALUE, "admission: the other planned bytes' safety factor"),
    S("VERITY_ADMIT_OVER_BOUND", one, False, VALUE, "admission: 1 runs a Commit whose plan is over its bound (recorded as an override)"),
    S("VERITY_ADMIT_SLACK_GIB", float, 6.0, VALUE, "admission: GiB of host memory kept free"),
    S("VERITY_CENSUS_GPU_UTIL", float, None, VALUE, "the kernel census engine's gpu_memory_utilization"),
    S("VERITY_DUMP_STEP", str, None, DEBUG, "the Commit dumps this step's values"),
    S("VERITY_HOT_RESTART_REASON", str, None, WHERE, "why the hot worker restarted (it sets it for its child)"),
    S("VERITY_HOT_SUBMIT", one, False, WHERE, "1: the Commit submits itself to the pod's hot worker"),
    S("VERITY_SOURCE_IDENTITY_DIR", str, None, WHERE, "where the source identity is written (the Commit sets it)"),
    S("VERITY_SWITCH", float, None, DEBUG, "the interpreter's switch interval in seconds (a GIL-interference experiment)"),
    S("VERITY_TRACEDUMP", float, None, DEBUG, "every so many seconds, dump every thread's traceback"),
    S("VERITY_VU_EXPORT_DIR", str, None, WHERE, "where pair 0's replay exports its VU sample (`commit --vu-export-dir`)"),
    # ---- the engine (engine/)
    S("VERITY_INDUCTOR_DIR_PER_PROCESS", flag, False, WHERE, "TORCHINDUCTOR_CACHE_DIR is this process's own (`engine.env.INDUCTOR_DIR_MARK`)"),
    S("VERITY_INT8_CREDITED_READ", str, None, VALUE, "MODE:PAGED_SO, the int8 attention's credited read every engine installs"),
    S("VERITY_INT8_FUSED_PRODUCERS", flag, True, VALUE, "the int8 linears' norms and silu-and-mul quantize their outputs in the "
      "same launch (`engine.int8_producers`)"),
    S("VERITY_INT8_PARTS", csv, (), VALUE, "the int8 parts a rank registers when it starts (`engine.int8_ranks`)"),
    S("VERITY_ROPE_CACHE_FUSED", flag, True, VALUE, "the rope and the KV-cache write run as one launch a layer "
      "(`engine.rope_cache`)"),
    S("VERITY_SPARSE_MLA_NUM_SPLITS", stripped, "0", VALUE, "FA3's sparse MLA splits: 1 pins one split, 0 leaves vLLM's heuristic; "
      "empty is refused", empty=True),
    # ---- Build, Match and the row (pipeline/, check/, query/)
    S("VERITY_CAMPAIGN", str, "r15", VALUE, "the campaign label the TP Match record carries"),
    S("VERITY_CORE", str, None, WHERE, "the core tree a row's stages import (the row sets it)"),
    S("VERITY_GC_THRESHOLDS", lowered, "", VALUE, "the cyclic GC's thresholds, a,b,c; empty, tuned; default, off, none or 0, the interpreter's"),
    S("VERITY_INSTANCES_FORM", lowered, None, VALUE, "the stored form of the fold's instances.jsonl: runs or progressions"),
    S("VERITY_LEASE_MAX_MIN", int, None, WHERE, "minutes a leased Commit holds its GPUs (the row sets it)"),
    S("VERITY_LEASE_STARTED", str, None, WHERE, "the lease marker a leased Commit takes its GPUs by (the row sets it)"),
    S("VERITY_MANIFEST_JOBS", int, None, VALUE, "concurrent component builds of the required-value manifest"),
    S("VERITY_QUERY_ID", str, None, VALUE, "the verification query id a row runs"),
    S("VERITY_QWORD_MAX_GATES", str, None, VALUE, "the strict word check's gate bound for the row"),
    S("VERITY_QWORD_MAX_GATES_ALLOWED", str, None, VALUE, "the one raised VERITY_QWORD_MAX_GATES a GO allows"),
    S("VERITY_REPLAY_DUMP_DIR", str, None, DEBUG, "the replay dumps the families VERITY_REPLAY_DUMP_FAMILIES names here"),
    S("VERITY_REPLAY_DUMP_FAMILIES", csv, (), DEBUG, "the families the replay dumps, comma-separated; none, every family"),
    S("VERITY_REPLAY_DUMP_MAX", int, 1024, DEBUG, "the most values the replay dumps"),
    S("VERITY_REPLAY_FORK_GC_FREEZE", not_off, True, VALUE, "the replay freezes the GC before it forks its workers"),
    S("VERITY_RESEARCH", str, None, WHERE, "the research tree a row's stages import (the row sets it)"),
    S("VERITY_TIMELINE", str, None, WHERE, "the stage telemetry file"),
    S("VERITY_TREE", str, None, WHERE, "the source tree that runs (the mixed-tree refusal)"),
    S("VERITY_UNIT_RULE_CACHE", str, None, WHERE, "the host's unit-rule cache directory (`sweep --unit-rule-cache`)"),
    S("VERITY_WORD_CHECK", str, None, VALUE, "the strict word check's widths, X/W"),
    S("VERITY_WORKLOADS", paths, (), WHERE, "directories of row files beside workloads/, os.pathsep-separated: a row id resolves to "
      "the one file of its name among them (`config.row_file`), whose path and sha256 the row's effective config records"),
    S("MATCH_IMPL", str, None, VALUE, "the global match's checker, fast or record (the audit); unset, `global_match`'s default, fast, "
      "though `global_match_fast.enabled` reads unset as off (`--match-impl`, which the row exports)"),
    S("MATCH_WORKERS", stripped, "", VALUE, "forked workers for the match's per-request parts; empty, min(cpu_count, 8)"),
    S("MATCH_COMPACT", not_zero, True, VALUE, "the match holds operands as run progressions; 0, as runs (the representation's A/B)"),
    S("STOCH_DECLARED_CHRONOLOGY_S", one, False, VALUE, "1: a workload Program composes components derived under stoch's "
      "experimental per-event S schedule, which it otherwise refuses"),
    S("X09_PROFILE_OUT", str, None, DEBUG, "the file the batch decomposition writes its phase profile to; unset or blank, no profile"),
    S("X09_FORCE_DAG", str, None, VALUE, "1 runs the batch decomposition's DAG diagnostic even when the sequence criterion held, adding "
      "its fields to the record (diagnostic only, never on a workload of record)"),
    # ---- the hot Commit worker (pipeline/hot.py) and the native builds it keys on
    S("HOT_ROOT", str, "/workspace/cp/hot", WHERE, "the hot workers' directory (`hot --root`)"),
    S("HOT_IDLE", float, 900.0, WHERE, "seconds a hot worker waits idle before it exits (`hot --idle`)"),
    S("HOT_TIMEOUT", float, None, WHERE, "seconds a hot submit waits for its job; unset, no bound"),
    S("HOT_RELEASING_MARKER", str, "/workspace/cp/RELEASING", WHERE, "the marker pod_release.sh holds while it ships a release"),
    S("HOT_RELEASE_JSON", str, "/workspace/cp/RELEASE.json", WHERE, "the release (tag, commit) of the tree on the pod"),
    S("RELEASING_STALE_S", float, 900.0, WHERE, "seconds after which a release marker nobody refreshed is stale"),
    S("NATIVE_COLLECT_BUILD", str, None, WHERE, "the parent directory of the native collector's builds"),
    S("HIDDEN_GPU_BUILD", str, None, WHERE, "the directory of the hidden-GPU committer's builds"),
    # ---- the program (program/)
    S("VERITY_DENSE_THREADS", str, None, VALUE, "the dense chain's thread cap, read at import; not a positive integer, no cap"),
    S("VERITY_NUMERICS_BUILD_DIR", str, None, WHERE, "where the numerics' JIT-built libraries live"),
    S("VERITY_REAL_LOGITS", str, None, WHERE, "the top-p difftest's real logits rows, an .npy"),
    # ---- operations (ops/*.sh)
    S("VERITY_RELEASE_TAG", str, None, WHERE, "the release a compiled Commit logs"),
    S("VERITY_TREE_COMMIT", str, None, WHERE, "the commit of VERITY_TREE a compiled Commit logs"),
    # ---- other programs' variables the library reads
    S("CUBLASLT_WORKSPACE_SIZE", str, None, VALUE, "cuBLASLt's workspace in KiB; the replay's dependencies record it"),
    S("CUBLAS_WORKSPACE_CONFIG", str, None, VALUE, "cuBLAS's workspace configuration, :SIZE:COUNT; the replay's dependencies record it"),
    S("CUDA_HOME", str, None, WHERE, "the CUDA toolkit torch's extension builds use; a tap build sets its own default"),
    S("CUDA_VISIBLE_DEVICES", str, None, WHERE, "the devices this process may use; empty, none", empty=True),
    S("HF_HOME", str, None, WHERE, "Hugging Face's cache root; unset, ~/.cache/huggingface"),
    S("HF_HUB_OFFLINE", str, None, WHERE, "1: Hugging Face reads only its cache (`engine.env.EXPORT` and an engine's pins set it); a "
      "Build records it"),
    S("MAX_JOBS", str, None, VALUE, "parallel compile jobs of torch's extension builds; a tap build sets its own default"),
    S("RESEARCH_RUN_ID", str, None, WHERE, "the research run this process is part of"),
    S("TOKENIZERS_PARALLELISM", str, None, WHERE, "the tokenizers library's threads, true or false (`engine.env.EXPORT` sets false); "
      "a Build records it"),
    S("TORCHINDUCTOR_CACHE_DIR", str, None, WHERE, "Inductor's output-code cache; unset, torch's default"),
    S("TORCH_CUDA_ARCH_LIST", str, None, VALUE, "the architectures torch's extension builds compile for (a tap build sets it)"),
    S("TRITON_CACHE_DIR", str, None, WHERE, "Triton's compiled-kernel cache; unset, ~/.triton/cache"),
    S("VLLM_API_KEY", str, None, VALUE, "the key vLLM's OpenAI-compatible server asks of its clients (its family's role): a "
      "credential, which a record names and never holds", secret=True),
    S("VLLM_ATTENTION_BACKEND", str, None, VALUE, "vLLM's attention backend, which the pinned vLLM no longer defines: a Build "
      "records it as ignored"),
    S("VLLM_BATCH_INVARIANT", str, None, VALUE, "vLLM's batch-invariant kernels, 1 as an engine profile's env sets it; the replay's "
      "dependencies record it"),
    S("VLLM_DISABLE_COMPILE_CACHE", str, None, VALUE, "1: vLLM compiles afresh instead of loading its compile cache; the compiled "
      "records say which"),
    S("VLLM_LOGGING_LEVEL", str, None, DEBUG, "vLLM's log level (`engine.env.EXPORT` sets WARNING); a Build records it"),
    S("VLLM_USE_DEEP_GEMM", str, None, VALUE, "vLLM's DeepGEMM kernels for block-scaled FP8 (the target profile's `fp8_block_gemm` "
      "sets it); a Build records it"),
    S("VLLM_USE_LAYERNAME", str, None, VALUE, "vLLM's layer-name custom ops, read when vLLM is imported (`engine.env.EXPORT` sets 0); "
      "a Build records it"),
)
del S

#: Namespaces of other libraries' variables, read or written as a whole (vLLM's ``vllm/envs.py`` declares its own): the role of every
#: variable of the family that isn't declared in `SETTINGS`.
FAMILIES: dict[str, str] = {"VLLM_": VALUE, "NCCL_": VALUE, "CUBLAS": VALUE}

#: The ``VERITY_*`` variables the rest of the tree reads (``verity/``, ``benchmarks/``, ``tools/``, the integration's tests and
#: tools), which a verity-vllm process may inherit.  A name ending ``_`` names a family.
ELSEWHERE = frozenset("""
VERITY_BENCH_ VERITY_SUITE_ VERITY_VLLM_BENCH_
VERITY_ADV_CACHE VERITY_ADV_SEED VERITY_AMBIENT VERITY_AMBIENT_TEST VERITY_CAPTURED_101_GUMBEL VERITY_CENSUS
VERITY_CHECKER_VARIANT VERITY_CIRCUIT_CHECK_SLOW VERITY_COMMIT VERITY_DENSE_PROGRAM_DIRS VERITY_E2E_B VERITY_E2E_BOUNDED
VERITY_E2E_CASE VERITY_E2E_EXCLUDE VERITY_E2E_GPU_UTIL VERITY_E2E_HIDDEN_SO VERITY_E2E_LEARN_MB VERITY_E2E_UTIL
VERITY_E2E_WINDOW_MB VERITY_E2E_WORKLOAD VERITY_ENCODINGS_ADVICE_V0 VERITY_EVALUATE_FULL_LOGF VERITY_EXHAUSTIVE VERITY_FA2_CSRC
VERITY_FLOCK_CACHE VERITY_FLOCK_INNER_FOLD VERITY_FLOCK_SAME_SHAPE_FULL VERITY_GIT_GUARD VERITY_GPU_TORCH_ONLY
VERITY_H1_REGISTER VERITY_HP_ARCHIVE VERITY_KV_SO VERITY_LEAF_LAYOUT VERITY_LEAN_MEM_FLOOR_GIB VERITY_ML_BOOLEAN_EXHAUSTIVE
VERITY_MOE_PROGRAM_DIRS VERITY_MUFU_TABLES VERITY_MUFU_TANH_TABLES VERITY_NETWORK_RECORDS_EXE VERITY_NUMERICAL_STORE
VERITY_PAGED_SO VERITY_POUS_SLOW VERITY_POUW_CUDA_ARCH VERITY_POUW_CUDA_BUILD VERITY_POUW_H1_PIN
VERITY_POUW_H1U_SO VERITY_POUW_NOISE_CUBIN VERITY_POUW_NVCC VERITY_POUW_WINDOW_LOWERED_LANES VERITY_REAL_GIT VERITY_REGRESSION
VERITY_REGRESSION_CANDIDATE VERITY_REGRESSION_CANDIDATE_CACHE VERITY_REGRESSION_ORACLE VERITY_REGRESSION_REBUILD
VERITY_REGRESSION_RECORD_ACTUAL VERITY_REGRESSION_ROWS_ROOT VERITY_REGRESSION_SCRATCH VERITY_REGRESSION_STOCH_JOBS
VERITY_REGRESSION_TIERS VERITY_REGRESSION_VERIFY_ALL VERITY_REPIN VERITY_RESEARCH_POSTGRES VERITY_RMS_TABLES VERITY_SESSION_SO
VERITY_SKIP_STORE VERITY_SLOW VERITY_STORE_REQUIRED VERITY_TC_PROBE_FP4_NVCC VERITY_TEST_CACHE VERITY_TEST_CACHE_MIN_S
VERITY_TRACE_DIR VERITY_VEC_IN VERITY_VEC_OUT VERITY_VLLM_BOOLEAN_NORMS_FULL VERITY_VLLM_BOOLEAN_SWEEPS
VERITY_VLLM_INT8_K_SCALES VERITY_VLLM_TANH_EXHAUSTIVE VERITY_VSTAR_STAGE VERITY_VSTAR_STAGE_OUT
""".split())

#: ``VERITY_*`` names the library spells out that are not environment variables: the C preprocessor macros of the tap builds.
NOT_SETTINGS = frozenset("""
VERITY_FA3_MS VERITY_HAVE_FA3_HDIM VERITY_HAVE_FWD_HDIM64_BF16 VERITY_HAVE_SPLIT_HDIM VERITY_MAT_MS VERITY_MAT_SRC
VERITY_ROW_GUARD VERITY_TAP
""".split())


REGISTRY = Registry(__file__, ("VERITY_",), *_DECLARED, schema=RECORD_SCHEMA, families=FAMILIES, elsewhere=ELSEWHERE)
SETTINGS = REGISTRY.settings

setting, raw, get, put, unset, setdefault, environ = (REGISTRY.setting, REGISTRY.raw, REGISTRY.get, REGISTRY.put, REGISTRY.unset,
                                                      REGISTRY.setdefault, REGISTRY.environ)
role_of, is_secret, redact, record = REGISTRY.role_of, REGISTRY.is_secret, REGISTRY.redact, REGISTRY.record
known, unknown, validate_environ = REGISTRY.known, REGISTRY.unknown, REGISTRY.validate_environ


def is_set(name: str, environ: Mapping[str, str] | None = None) -> bool:
    """The variable is set (to anything, the empty string included)."""
    return raw(name, environ) is not None


def __getattr__(name: str) -> Any:
    if name in SETTINGS:
        return get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *SETTINGS])
