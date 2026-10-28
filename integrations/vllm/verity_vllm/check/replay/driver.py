"""Sampled exact VU replay -- the local replay check of record (serialized as `C2`).

For a sampled VU `V` (one Call of the Program) the check is

    F_V( tau(in(V)) ) == tau(out(V))      bit-for-bit,

where `tau` reads the COMMITTED bytes of the Values the Program's dataflow names (the Call's `args`), and `F_V` is the
registered relation of the Call's Definition evaluated by the registry's own reference (`verity.ml.definitions.numerics.derived_rows`,
the same numpy models the Match stage's fold checks use).  Nothing here reads the collector's emissions to decide WHAT is
required or WHERE an input lives: the population is the Program (its Calls, the instances.json.gz rows), the addresses are the required-value
manifest's Value rule -- the v2 query's population the manifest of record was built from (`required.Population.members_of`) -- and the
binding map fixed at finalize
(occurrence resolution: a Call's Value -> its occurrence key -> committed bytes).

    row.args = [[["n", j, lo, hi]], [["p", 1, lo, hi]]]
        "n" j lo hi -> producer Call j's OUTPUT element slice [lo, hi) -> its occurrence key (op_path, member, request-local row) -> committed bytes
        "p" 1 lo hi -> a slice of the registered `weights` parameter (bound by reference: read from the model's parameters)
        "p" 0 k k+1 -> the declared prompt id k -> committed `input_token_ids` row k of the request's arrival step
        a TokenSelect producer -> committed `sampled_token_ids` of THAT Call's step (the previous step's designated output)

Strata (recorded): (spec incl. shape, request, phase in {prefill, first-decode, mid, last}; `last` = the last EXECUTED step) x `per_stratum`, seeded -- distinct
numerical mechanisms x requests x lifetime boundaries.  Not-evaluable Calls are counted by reason (never silently skipped); see NOT_YET.
Attention_v2/v3 Calls ARE evaluated: the query at position T-1 over the T committed K/V Values the Program names by ref -- the
resolver reads every producer at its own step, so the request's history is committed words, never a cache.

Ordered boundary linkage (the `linkage` check; exhaustive, NOT sampled): every request's committed `input_token_ids` at its arrival step == the DECLARED
prompt (workload manifest), and at every later step == that request's committed `sampled_token_ids` of the previous step.  This is
the prescribed-input / designated-output linkage of §4 and replaces the Match-oracle occurrence-resolution gate at B>=2.

PROTOCOL SUMMARY (Commit stage: local replay (`C2`) of record = sampled exact VU replay + exhaustive boundary linkage (`linkage`))
  Purpose: from the COMMITTED bytes alone (binding map + committed reader), re-evaluate a stratified sample of Calls with the
    registry's own relations and require bit equality; link every request's committed input ids to its declared prompt and to its
    own previous-step sampled ids over the executed prefix.  Needs no Match record.
  Inputs: the request Programs' instances docs (ProgramIndex), the binding map, `committed(step, op_path, member, rank)`, the weights
    reader, arrivals, the required-value manifest (Q(P) of record), `executed_prefix` facts per request, the seed (+ an optional verifier challenge).
  Outputs: the replay record {result True|False|None, grade, sample{seed, seed_source, challenge}, strata, population{query_total,
    not_executed_by_rule, ...}, uncovered_by_name, query_checks, catch_probability}; the linkage record {result, why, pairs, linked, unlinked}.
  Invariants: result True only if EVERY sampled VU replays bit-exactly AND every non-empty stratum was evaluated (None = partial,
    never PASS); the seed is never prover-choosable in advance (run root after commitment, or sha256(challenge|root)[:8]); Calls at
    request-relative steps >= n_exec leave the pool into a NAMED class before any draw; a request linked over fewer steps than its
    cap must end on a declared EOS id at steps[-1 - lag] (undeclared EOS / unattested lag = unlinked by name); weights_check.equal
    False fails the replay by name.

The check runs here; its steps live beside it: `index` (the Program rows), `population` and `coverage` (what may be sampled
and what the query population names), `sample` (the stratified draw and its seed), `opening` (committed words by
identity), `evaluate` and `compare` (one VU), and `linkage` (the exhaustive boundary linkage).  A caller wires the check
through this module: the names in `__all__` are its interface.
"""
from __future__ import annotations

import ctypes
import multiprocessing
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from verity.ml.randomness import Key

from verity_vllm import settings
from verity_vllm.check.replay import cached_prefix as CP
from verity_vllm.check.replay import fork as FK
from verity_vllm.check.replay import replay_codes as RC
from verity_vllm.check.replay.fork import WorkerDied
from verity_vllm.check.replay.replay_codes import NOT_YET, WHY_CLASSES_BY_NAME, WHY_CLASSES_FAULT, why_class
from verity_vllm.commit.opened import opened_in_children
from verity.ml.definitions.numerics.rows import EVALUATOR_PROVENANCE, EVALUATORS
from verity_vllm.check.replay.index import V2_QUERY, ProgramIndex, node_runs
from verity_vllm.program.compact import p_item_weight_span
from verity_vllm.query import norm_scales
from verity_vllm.check.replay.opening import CommittedStore, model_weights_provider
from verity_vllm.check.replay.evaluate import INTERIOR_DEPTH, ROW_EVALUATORS, plan_vu, replay_vu
from verity_vllm.check.replay.population import (executed_prefix_of_record, moe_plane_alias_check, phase_anchor, population,
                                                split_not_executed)
from verity_vllm.check.replay.coverage import _uncovered_mechanisms, controller_rows_of, query_population
from verity_vllm.check.replay.sample import _catch_probability, challenge_seed, sample
from verity_vllm.check.replay.linkage import arrivals_of_record, boundary_linkage, linkage_horizon, splits_of_record


__all__ = ["arrivals_of_record", "boundary_linkage", "challenge_seed", "engine_facts_of_record", "executed_prefix_of_record",
           "linkage_horizon", "model_weights_provider", "ProgramIndex", "sampled_replay", "SCHEMA", "splits_of_record", "V2_QUERY"]


SCHEMA = "verity-sweep/sampled-replay/v1"

_FORK_FN: Callable | None = None      # the per-VU closure the forked children run (set in the parent right before the fork)
_FORK_VUS: list[dict] = []


class _DeviceInChild(RuntimeError):
    """A forked child touched the accelerator (a DEVICE-retained committed store, CUDA after fork): its 'evaluator error' is
    an environment fault of the pool, not a verdict about the VU -- raised so the parent falls back to the serial path (its own CUDA context)."""


_DEVICE_ERROR_RE = re.compile(r"AcceleratorError|CUDA|cuda|hip|HIP", re.ASCII)


def _fork_worker(i: int) -> tuple[dict, dict, float]:
    assert _FORK_FN is not None
    vu, rec, dt = _FORK_FN(_FORK_VUS[i])
    why = str(rec.get("why") or "")
    if rec.get("result") is None and why.startswith("evaluator error:") and _DEVICE_ERROR_RE.search(why):
        raise _DeviceInChild(f"child pid {os.getpid()} hit the device: {why[:200]}")
    return vu, _plain(rec), dt


def _plain(x: Any) -> Any:
    """JSON-plain copy of a replay record (no numpy scalars / arrays cross the pipe)."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _prewarm_weights(indices: dict, picked: list[dict], weights: Callable[[str, int, int], np.ndarray | None]) -> None:
    """Read every weight field the sample consumes into the provider's cache in the PARENT (the children must not touch the model) --
    INCLUDING the weights of the interior producers a VU composes (the pre-bias Gemm under a BiasAdd): [measured on
    a Qwen2.5 L40S B1 record] the BiasAdd VUs' own args name only the bias slice, the composed Gemm's qkv weight was first read
    in a forked child -> AcceleratorError 'CUDA initialization error' -> every BiasAdd stratum 'not evaluated' -> partial -> FAIL closed."""
    seen: set[str] = set()

    def walk(pi: ProgramIndex, r: dict, depth: int) -> None:
        wp = pi.pnames.index("weights") if "weights" in pi.pnames else None
        for group in r.get("args") or []:
            for it in group if wp is not None else ():
                span = p_item_weight_span(it, wp)                                  # a "P" progression's span, never its runs
                ws = pi.weight_slice(*span) if span is not None else None
                if ws is not None and ws[0] not in seen:
                    seen.add(ws[0])
                    weights(ws[0], ws[1], ws[2])
            for sl in node_runs(group) if depth < INTERIOR_DEPTH else ():
                a, lo, hi = int(sl[1]), int(sl[2]), int(sl[3])
                if pi.member_at(a, lo, hi) is None:
                    pr = pi.by_i.get(a)
                    pd = pi.describe(pr) if pr is not None else None
                    if pd is not None and pd["members"] == [] and pd["family"] in EVALUATORS:
                        walk(pi, pr, depth + 1)                                    # the interior producer's weights are the VU's too

    for vu in picked:
        pi = indices[vu["request_id"]]
        walk(pi, pi.by_i[vu["i"]], 0)


#: spec family -> the C++ twin libraries its evaluator dlopens (built JIT on first use)
_FAMILY_LIBS = {"RMSNormTriton_v1": ("rms",), "RMSNormTriton_v5": ("rms",), "RMSNormFusedCuda_v2": ("rms",), "Gemm_v1": (), "Gemm_v2": (), "RoPE_v1": (), "SiluMul_v1": (), "SiluMul_v2": (), "BiasAdd_v1": (),
                "Embedding_v1": (), "TokenSelect_v1": (), "TokenSelect_v3": (), "TokenSelect_v4": (), "GumbelTopPTokenSelect_v1": (),
                # the attention rows' softmax reads the measured MUFU.EX2 / MUFU.RCP tables: loaded in the parent
                "Attention_v3": ("fa2",), "Attention_v2": ("fa2",), "Attention_v4": ("fa2",), "Attention_v5": ("fa2",), "Attention_v18": ("fa2",),
                "AttentionSoftcap_v2": ("fa2",),
                # the router's NvExpf reads the measured MUFU.EX2 table (fa2 tables, ~1.5 s to load): loaded in the parent
                "MoeRouterTopK_v1": ("fa2",), "MoeRouterTopKNorm_v1": ("fa2",),
                "MoeRouterTopKOrdered_v1": ("fa2",), "MoeRouterTopKOrderedNorm_v1": ("fa2",), "MoeExpertGemm_v1": (), "MoeExpertGemmW_v1": (), "MoeExpertGemm_v2": (),
                "MoeExpertGemmW_v2": (), "MoeSum_v1": ()}


def _prewarm_evaluators(picked: list[dict]) -> dict[str, str]:
    """Build + dlopen every twin library the sample's families need in the PARENT before the fork:
    the children inherit the mapping and never race a first-use JIT build (a sibling used to dlopen the half-written .so -> 'file too
    short' -> a stratum 'not evaluated').  The build itself is atomic + locked (`verity.ml.definitions.numerics._jit`); this makes the fast path
    the only path.  Returns {library: path} for the record; a failure here is reported, not fatal (the children fall back to their own
    locked build)."""
    fams = {str(vu.get("spec") or "").split("{", 1)[0] for vu in picked}
    need = {lib for f in fams for lib in _FAMILY_LIBS.get(f, ())}
    out: dict[str, str] = {}
    for ev in ROW_EVALUATORS[-1:]:                                  # an evaluator on another IR compiles what the sample reaches
        out.update(getattr(ev, "prewarm", lambda _p: {})(picked))
    if "rms" in need:
        try:
            from verity.ml.definitions.numerics import prims_twins
            rm, _t = prims_twins.rms_twin()
            rm.lib()
            out["rms_triton_model"] = str(rm.LIB)
        except Exception as e:  # noqa: BLE001
            out["rms_triton_model"] = f"prewarm failed: {type(e).__name__}: {str(e)[:120]}"
    if "fa2" in need:
        try:
            from verity.ml.definitions.numerics import prims_twins
            _fa2_model, _tables = prims_twins.fa2_twin()
            out["fa2_mufu_tables"] = f"loaded ({type(_tables).__name__})"
        except Exception as e:  # noqa: BLE001
            out["fa2_mufu_tables"] = f"prewarm failed: {type(e).__name__}: {str(e)[:120]}"
    return out


#: what the last forked pool did to the parent's heap before the fork: `gc_frozen_objects` = the
#: objects moved to the permanent generation (never traversed by a collection in the children), `gc_collected` = what the pre-fork
#: collection freed, `gc_freeze` = the policy (env VERITY_REPLAY_FORK_GC_FREEZE=0 opts out for a measurement, never for a record).
_FORK_STATS: dict[str, Any] = {}
FORK_GC_FREEZE = settings.VERITY_REPLAY_FORK_GC_FREEZE
CGROUP_MEMORY = ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current")


class PoolDoesNotFit(RuntimeError):
    """The largest pick does not fit in the cgroup's memory.max less what the parent holds after the prewarm: no worker, and no serial
    replay in the parent, could evaluate it.  Raised before the fork, by name, never left to the OOM killer."""


def _cgroup_memory() -> tuple[int, int] | None:
    read = []
    try:
        for p in CGROUP_MEMORY:
            with open(p) as f:
                read.append(f.read().strip())
    except OSError:
        return None
    lim, cur = read
    return (int(lim), int(cur)) if lim.isdigit() and cur.isdigit() else None


def _malloc_trim() -> bool:
    try:
        return bool(ctypes.CDLL("libc.so.6").malloc_trim(0))
    except (OSError, AttributeError):
        return False


def default_workers(processes: int = 1) -> int:
    """The forked pool's size when the caller names none: half the cores, shared by ``processes`` replays running at once, at most 32."""
    return max(1, min(32, (os.cpu_count() or 2) // 2 // processes))


def fit_pool(want: int, picked: list[dict], worker_peak: int) -> tuple[int, dict[str, Any] | None]:
    """How many forked workers fit, measured after the prewarm, and the fit's record (None without a cgroup limit).  The room is the
    cgroup's memory.max less its use once the parent's free heap is returned (`malloc_trim`); a pick's peak is ``worker_peak`` plus what
    the row evaluator states its evaluation allocates (``peak_bytes(vu)``, `evaluate.ROW_EVALUATORS`), and ``want`` workers are cut to
    the most whose largest picks fit side by side.  PoolDoesNotFit when the largest pick alone does not."""
    before = _cgroup_memory()
    trimmed = _malloc_trim()
    mem = _cgroup_memory()
    if mem is None:
        return want, None
    lim, cur = mem
    room = lim - cur
    stated = getattr(ROW_EVALUATORS[-1], "peak_bytes", None) if ROW_EVALUATORS else None
    peaks = sorted(((worker_peak + (int(stated(vu)) if stated else 0), str(vu.get("spec"))) for vu in picked), reverse=True)
    peaks += [(worker_peak, "an idle worker")] * (want - len(peaks))
    n, total = 0, 0
    for p, _ in peaks[:want]:
        if total + p > room:
            break
        n, total = n + 1, total + p
    big, spec = peaks[0]
    if n == 0:
        raise PoolDoesNotFit(f"the largest pick {spec} needs {_gib(big)} GiB, more than the cgroup's memory.max {_gib(lim)} GiB less the "
                             f"{_gib(cur)} GiB in use after the prewarm ({_gib(room)} GiB): the job needs at least {_gib(cur + big)} GiB")
    note = (f"{want} requested, {f'fitted to {n}' if n < want else 'all fit'} after the prewarm: {n} workers at the largest picks' peaks "
            f"({_gib(total)} GiB, the largest {_gib(big)} GiB for {spec}) in memory.max {_gib(lim)} GiB less the {_gib(cur)} GiB in use")
    return n, {"memory_max": lim, "in_use": cur, "in_use_before_trim": before[1] if before else None, "malloc_trim": trimmed, "room": room,
               "worker_peak": worker_peak, "peaks_stated_by": type(ROW_EVALUATORS[-1]).__name__ if stated else None,
               "largest": {"spec": spec, "bytes": big}, "workers_requested": want, "workers_fitted": n, "fitted_bytes": total, "note": note}


def _gib(b: int) -> str:
    return f"{b / 2**30:.1f}"


def _parallel_replay(fn: Callable[[dict], tuple[dict, dict, float]], picked: list[dict], workers: int) -> list[tuple[dict, dict, float]]:
    """Forked pool over the picked VUs; bounded by a timeout (a child deadlocked on a lock inherited at fork must not hang the Commit).

    [the fork-COW children term] the children inherit the parent's whole C2 heap (the Programs of
    every instance + the MatchOracle + the binding index: 43.5 GiB at #60 B8 Mistral, ~75 GB at #23 B64 Llama) copy-on-write.  A
    child evaluates a few VUs, but the FIRST cyclic garbage collection its own allocations trigger traverses EVERY container object
    of the inherited heap and writes each one's gc header -> every page of the parent heap is copied into that child.  That is the
    #23 datum ("two ForkPoolWorkers at ~72 GB each"): 72 GB is the parent's C2 heap (programs 39.5 + oracle
    35.4 GB modelled), not 2.9 x the largest committed tensor -- a VU reads its own rows only (`CommittedStore.words` slices), and the
    hidden-M1 stream the "largest tensor" is has no VU row at all.  `gc.freeze()` before the fork moves every live object into the
    permanent generation, which the children's collections never traverse (they collect only what they allocate after the fork), so a
    child's private pages are its own scratch + the refcount-dirtied pages of the objects it touches (the 250 MiB ordinary-worker term).
    Semantics-free: gc.freeze changes WHEN unreachable cycles are reclaimed, never a value; the parent unfreezes right after the pool
    joins.  The record carries what was frozen (`evaluator.fork`)."""
    global _FORK_FN, _FORK_VUS
    import gc
    _FORK_FN, _FORK_VUS = fn, picked
    _FORK_STATS.clear()
    _FORK_STATS.update({"gc_freeze": bool(FORK_GC_FREEZE), "workers": int(workers), "picked": len(picked)})
    frozen = False
    try:
        if FORK_GC_FREEZE:
            _FORK_STATS["gc_collected"] = int(gc.collect())          # garbage is not frozen into the permanent generation
            gc.freeze()
            frozen = True
            _FORK_STATS["gc_frozen_objects"] = int(gc.get_freeze_count())
        # the canary line (commit.log): the planner's `compare` reads it to model the record with 250 MiB workers
        print(f"[replay] fork gc.freeze={'on' if frozen else 'OFF'}: {workers} workers over {len(picked)} VUs"
              + (f", {_FORK_STATS['gc_frozen_objects']} objects frozen ({_FORK_STATS['gc_collected']} collected) before the fork" if frozen else
                 " (VERITY_REPLAY_FORK_GC_FREEZE=0: the children may copy the parent's C2 heap on write)"), flush=True)
        oom0 = FK.oom_kills()
        with FK.fork_pool(workers) as pool:
            pids = {p.pid for p in pool._pool}
            res = pool.map_async(_fork_worker, range(len(picked)), chunksize=max(1, len(picked) // (workers * 8)))
            per_vu = max([3.0] + [float(getattr(ev, "seconds_per_vu", 3.0)) for ev in ROW_EVALUATORS[-1:]])
            deadline = time.monotonic() + max(300.0, per_vu * len(picked))
            while not res.ready():
                if (why := FK.dead_worker(pool, pids, oom0, workers, "--replay-workers")) is not None:
                    print(f"[replay] {why}", file=sys.stderr, flush=True)
                    raise WorkerDied(why)
                if time.monotonic() > deadline:
                    raise multiprocessing.TimeoutError
                res.wait(FK.POOL_POLL_S)
            out = res.get()
        return out
    finally:
        if frozen:
            gc.unfreeze()
        _FORK_FN, _FORK_VUS = None, []


ENGINE_FACTS_RULE = ("the evaluators' semantics are the engine's configuration of record (on cc 9.0 the Gemm_v2 Hopper chain "
                     "equals cuBLASLt only under VLLM_BATCH_INVARIANT=1, which disables split-K by CUBLAS_WORKSPACE_CONFIG=:16:8 / "
                     "CUBLASLT_WORKSPACE_SIZE=1 / allow_bf16_reduced_precision_reduction False; on cc 8.x it installs the Triton persistent "
                     "matmul; without batch invariance on cc 9.0 the cuBLAS GEMMs are the route pin's, which holds only where `cublas` is "
                     "bf16_gemm_check.ROUTE_CONDITIONS) -- MEASURED in the process the replay ran in (the Commit stage runs in the engine's "
                     "process), never declared; a fact this process cannot see is 'unavailable: <why>' by name (a laptop re-read of a stored "
                     "record has no engine)")
#: the workspace variables cuBLAS and cuBLASLt size their workspaces by, which move their routes
CUBLAS_ENV = ("CUBLAS_WORKSPACE_CONFIG", "CUBLASLT_WORKSPACE_SIZE")


def _unavailable(e: BaseException) -> str:
    return f"unavailable: {type(e).__name__}: {e}"[:200]


def _mapped_library(prefix: str, maps: str) -> str:
    """The one file named `prefix`* that `maps` (a /proc/<pid>/maps) lists: the library torch loaded, never the linker's search order,
    which can name another CUDA install's library than the one the GEMMs run."""
    files = {f[-1] for f in (ln.split() for ln in Path(maps).read_text().splitlines()) if len(f) >= 6}
    got = sorted(f for f in files if f.rsplit("/", 1)[-1].startswith(prefix))
    if len(got) != 1:
        raise LookupError(f"{len(got)} {prefix}* mapped {got}")
    return got[0]


def cublas_versions(maps: str = "/proc/self/maps") -> dict[str, Any]:
    """The cuBLAS and cuBLASLt this process maps, by their public API (`cublasGetProperty`'s major, minor and patch level, the
    number `cublasGetVersion` returns; `cublasLtGetVersion`), without a handle or a device."""
    out: dict[str, Any] = {}
    try:
        path = _mapped_library("libcublas.so", maps)
        lib = ctypes.CDLL(path)
        lib.cublasGetProperty.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        parts = [ctypes.c_int() for _ in range(3)]                                  # MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL
        if any(lib.cublasGetProperty(i, ctypes.byref(p)) for i, p in enumerate(parts)):
            raise RuntimeError("cublasGetProperty returned an error status")
        out["cublas_library"], out["cublas_version"] = path, parts[0].value * 10000 + parts[1].value * 100 + parts[2].value
    except Exception as e:                                                          # noqa: BLE001 -- named, never guessed
        out["cublas_library"] = out["cublas_version"] = _unavailable(e)
    try:
        lt = ctypes.CDLL(_mapped_library("libcublasLt.so", maps))
        lt.cublasLtGetVersion.restype = ctypes.c_size_t
        out["cublaslt_version"] = int(lt.cublasLtGetVersion())
    except Exception as e:                                                          # noqa: BLE001
        out["cublaslt_version"] = _unavailable(e)
    return out


def _cublas_facts(env: dict[str, Any]) -> dict[str, Any]:
    """What decides this process's cuBLAS routes beside the launch's shape, each read as the route pin's scan read it (the keys of
    `bf16_gemm_check.ROUTE_CONDITIONS`), with the cuBLASLt version and the library's path beside them."""
    out: dict[str, Any] = {}
    try:
        import torch                                                                 # type: ignore
    except Exception as e:                                                          # noqa: BLE001
        return {"torch": _unavailable(e)} | cublas_versions() | {"env": env}
    m = torch.backends.cuda.matmul
    reads = {"device": torch.cuda.get_device_name, "compute_capability": lambda: list(torch.cuda.get_device_capability()),
             "multiprocessors": lambda: int(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count),
             "torch": lambda: str(torch.__version__),
             "allow_fp16_reduced_precision_reduction": lambda: repr(m.allow_fp16_reduced_precision_reduction),
             "allow_bf16_reduced_precision_reduction": lambda: repr(m.allow_bf16_reduced_precision_reduction),
             "preferred_blas_library": lambda: str(torch.backends.cuda.preferred_blas_library())}
    for k, read in reads.items():
        try:
            out[k] = read()
        except Exception as e:                                                      # noqa: BLE001
            out[k] = _unavailable(e)
    return out | cublas_versions() | {"env": env}


def engine_facts_of_record() -> dict[str, Any]:
    """The batch-invariant configuration beside the evaluator strings: what the replay's pins were conditioned on;
    `cublas`, what the route pin's routes were conditioned on (`_cublas_facts`)."""
    facts: dict[str, Any] = {k: settings.raw(k) for k in ("VLLM_BATCH_INVARIANT", *CUBLAS_ENV)}
    try:
        from vllm.model_executor.determinism import batch_invariant as bi          # type: ignore
        facts["vllm_batch_invariant_mode"] = bool(bi.vllm_is_batch_invariant()) if hasattr(bi, "vllm_is_batch_invariant") else "unavailable: no vllm_is_batch_invariant()"
    except Exception as e:                                                          # noqa: BLE001 -- named, never guessed
        facts["vllm_batch_invariant_mode"] = f"unavailable: {type(e).__name__}"
    try:
        import torch                                                                 # type: ignore
        bf16 = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction   # (allow, allow for split-K) once set as a pair
        facts["torch_allow_bf16_reduced_precision_reduction"] = bool(bf16[0] if isinstance(bf16, tuple) else bf16)
        try:
            facts["torch_preferred_blas_library"] = str(torch.backends.cuda.preferred_blas_library()).split(".")[-1]
        except Exception as e:                                                      # noqa: BLE001
            facts["torch_preferred_blas_library"] = f"unavailable: {type(e).__name__}"
        facts["compute_capability"] = list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else "unavailable: no CUDA device"
    except Exception as e:                                                          # noqa: BLE001
        facts["torch_allow_bf16_reduced_precision_reduction"] = facts["torch_preferred_blas_library"] = facts["compute_capability"] = f"unavailable: {type(e).__name__}"
    facts["cublas"] = _cublas_facts({k: facts[k] for k in CUBLAS_ENV})
    facts["rule"] = ENGINE_FACTS_RULE
    return facts


def _picks_by_request_step(results: list) -> dict[str, dict[str, list[int]]]:
    # the picks BY (request, step): the strata carry phases, not steps, so "the other request's mid VUs evaluated
    # beside a stopped request's padding steps" was inferred from the phase arithmetic; this states it -- per request, per step: picked /
    # evaluated / mismatch (a padded record's reader splits `mid` at the other request's transition from these)
    picks_by_step: dict[str, dict[int, list[int]]] = {}
    for vu, rec, _dt in results:
        row = picks_by_step.setdefault(str(vu["request_id"]), {}).setdefault(int(vu["step"]), [0, 0, 0])
        row[0] += 1
        if rec.get("result") is not None:
            row[1] += 1
            if not rec["result"]:
                row[2] += 1
    return {r: {str(s): v for s, v in sorted(d.items())} for r, d in sorted(picks_by_step.items())}


#: the replay's own population, kept for one reader after it (the VU exporter, `pipeline.vu_store`, which arms `keep_population`
#: and takes it right after the replay record); never read by the replay or a verdict
_POPULATION: dict[str, Any] = {}
_KEEP = [False]


def keep_population(on: bool) -> None:
    _KEEP[0] = bool(on)


def take_population() -> dict[str, Any]:
    """The last replay's {indices (request -> ProgramIndex), vus, pool, not_executed}, emptied here ({} when none was kept)."""
    out = dict(_POPULATION)
    _POPULATION.clear()
    return out


def _kept(split: tuple[list[dict], list[dict]], indices: dict, all_vus: list[dict]) -> tuple[list[dict], list[dict]]:
    if _KEEP[0]:
        _POPULATION.update(indices={k: v for k, v in indices.items() if isinstance(k, str)}, vus=all_vus, pool=split[0], not_executed=split[1])
    return split


def manifest_query(manifest: dict[str, Any] | None) -> str | None:
    """The query the manifest was built under, when it is one the population knows (`required.QUERIES`); else None (the query of record)."""
    from verity_vllm.query.required import QUERIES
    q = ((manifest or {}).get("query") or {}).get("query_id")
    return q if q in QUERIES else None


DRAWS = ("family", "uniform")


def family_strata(vus: list[dict], k: int | None, per_stratum: int = 2, draw: str = "family") -> tuple[list[dict], int]:
    """With `k` (the config run), every VU re-keyed to one stratum per `draw` (the key's fields kept) and `k` picks per stratum.
    `uniform`: one stratum (`all|all|all|all|all|all`), so the k units are a uniform draw without replacement from every unit.
    `family`: one stratum per family (`family|all|all|all|all|all`); with the cap at `k`, the round-robin order of `sample` then takes
    k units in all, one per family first, a small family used up before the others, the rest split evenly.  Without `k`, the VUs and
    `per_stratum` as given.  Any other `draw` is refused by name."""
    if draw not in DRAWS:
        raise ValueError(f"replay draw {draw!r} is not one of {DRAWS}")
    if not k:
        return vus, per_stratum
    return [dict(v, stratum=f"{'all' if draw == 'uniform' else v['family']}|all|all|all|all|all") for v in vus], int(k)


def populate(programs: dict[str, dict[str, Any]], ranks: list[int], arrivals: dict[str, int], executed_prefix: dict[str, dict] | None,
             ring_slots: int | None, *, attention_paths: list[str] | None, tp: dict[str, Any] | None, program_dirs: dict[str, str] | None,
             manifest: dict[str, Any] | None, memo: dict | None = None, cached_prefixes: dict[str, dict] | None = None) -> tuple:
    """`sampled_replay`'s enumeration: every Program's index and its request's population, with the not-evaluable counts, their codes,
    the manifest addresses, the `last` anchors and the address correspondence.  ``memo`` keeps the indices and the populations across calls
    on the same Programs objects (the slim planner fills it before the run's seed exists).  ``cached_prefixes``: per warm request
    {"cached": C, "producers": [request label per cached block]} (`program.prefix_plan`), what its `prefix` reads resolve to."""
    memo = {} if memo is None else memo
    all_vus, not_ev, not_ev_fam, indices, codes = [], Counter(), Counter(), {}, {}   # codes: not-evaluable reason -> its structured code
    addresses: dict[tuple, dict] = {}                                             # manifest address -> class of the rows producing it
    anchors: dict[str, dict] = {}                                                 # per request: the step `last` is anchored to, named
    addr_rec: dict[str, Any] = {"rule": V2_QUERY, "correspondence": {}}
    rank = ranks[0] if len(ranks) == 1 else 0
    for rid, inst in programs.items():
        pi = indices.get(id(inst)) or memo.get(("index", id(inst)))
        if pi is None:
            pi = memo[("index", id(inst))] = ProgramIndex(inst, attention_paths=attention_paths, tp=tp, program_dir=(program_dirs or {}).get(rid),
                                                         query=manifest_query(manifest), norm_scales=norm_scales.policy_of(manifest))
        indices[id(inst)] = pi
        addr_rec["correspondence"][str(rid)] = pi.correspondence
        # a per-rank map (tp2_worker: every entry carries identity.rank == r) labels its strata with THAT rank; a
        # multi-rank map is replayed as rank 0's module outputs at v1 (named below)
        # the `last` phase is anchored to the request's last EXECUTED step (n_exec - 1 under the executed prefix = the
        # lag step of an EOS-stopped request), not to the cap: the EOS-stopped request has a `last` stratum too, sampled deterministically
        anchors[rid] = phase_anchor(pi.n_steps, (executed_prefix or {}).get(str(rid)))
        key = ("population", id(pi), rid, rank, int(arrivals.get(rid, 0)), ring_slots, int(anchors[rid]["last_step"]))
        if key not in memo:                                                       # a request's addresses are its own (keyed by it)
            got_a: dict[tuple, dict] = {}
            got_c: dict[str, dict] = {}
            memo[key] = (*population(pi, rid, rank=rank, arrive=int(arrivals.get(rid, 0)), ring_slots=ring_slots, addresses=got_a,
                                     last_step=int(anchors[rid]["last_step"]), codes=got_c), got_a, got_c)
        vus, ne, nef, got_a, got_c = memo[key]
        all_vus.extend(vus)
        not_ev.update(ne)
        not_ev_fam.update(nef)
        addresses.update(got_a)
        codes.update(got_c)
        indices[rid] = pi
    for rid, pc in (cached_prefixes or {}).items():
        pi = indices.get(rid)
        if pi is None or not int(pc.get("cached") or 0):
            continue
        if pi.kv_prefix is None or int(pi.kv_prefix["C"]) != int(pc["cached"]):
            raise ValueError(f"{rid} is served {pc['cached']} cached prompt rows but its Program of record takes "
                             f"{pi.kv_prefix['C'] if pi.kv_prefix else 'no'} cached prefix")
        pi.cached_prefix[rid] = CP.CachedPrefix(pi.kv_prefix, [str(q) for q in pc["producers"]], indices, arrivals)
    return all_vus, not_ev, not_ev_fam, indices, codes, addresses, anchors, addr_rec


def planned(picked: list[dict], indices: dict, store: CommittedStore, arrivals: dict[str, int], t_start: float) -> dict[str, Any]:
    """A plan of `sampled_replay`: every picked VU's committed reads made without evaluating it (`evaluate.plan_vu`); the draw's size, its
    picks ([request, request-relative step, Program row, op_path] in draw order) and the VUs whose inputs did not resolve."""
    unresolved = [w for vu in picked if (w := plan_vu(indices[vu["request_id"]], store, vu, arrive=int(arrivals.get(vu["request_id"], 0))))]
    return {"plan": True, "picked": len(picked), "picks": [[vu["request_id"], int(vu["step"]), int(vu["i"]), vu["op_path"]] for vu in picked],
            "unresolved_n": len(unresolved), "unresolved": unresolved[:8], "wall_s": round(time.perf_counter() - t_start, 3)}


def _evaluated(one: Callable[[dict], tuple[dict, dict, float]], picked: list[dict], indices: dict, weights: Callable, workers: int,
               worker_peak: int | None) -> tuple[list[tuple[dict, dict, float]], int, str | None, dict[str, str], dict[str, Any] | None]:
    """`one` over the picked VUs, in picked order: a forked pool after the prewarm, fitted to the cgroup's memory when ``worker_peak``
    is given (`fit_pool`), else serial in the parent; a prewarm or pool failure falls back to serial.  Returns the results, the
    workers that ran them, the pool's note, the prewarmed libraries and the fit's record."""
    note, prewarmed, memory = None, {}, None
    if workers > 1 and len(picked) > 1:
        try:
            _prewarm_weights(indices, picked, weights)
            prewarmed = _prewarm_evaluators(picked)
        except Exception as e:  # noqa: BLE001
            note, workers = f"parallel evaluator unavailable ({type(e).__name__}: {str(e)[:120]}); serial fallback", 1
    if workers > 1 and len(picked) > 1 and worker_peak is not None:
        fitted, memory = fit_pool(workers, picked, int(worker_peak))
        if memory is not None:
            print(f"[replay] WORKERS: {memory['note']}", flush=True)
        if fitted < workers:
            note, workers = memory["note"], fitted
    if workers > 1 and len(picked) > 1:
        try:
            return _parallel_replay(one, picked, workers), workers, note, prewarmed, memory
        except Exception as e:  # noqa: BLE001
            note = f"parallel evaluator unavailable ({type(e).__name__}: {str(e)[:120]}); serial fallback"
    return [one(vu) for vu in picked], 1, note, prewarmed, memory


def sampled_replay(programs: dict[str, dict[str, Any]], bmap: dict[str, Any], committed: Callable[[int, str, int, int], bytes | None],
                   weights: Callable[[str, int, int], np.ndarray | None], *, arrivals: dict[str, int] | None = None, per_stratum: int = 2,
                   seed: int, seed_source: str = "caller", max_evaluate: int = 2000, attention_paths: list[str] | None = None, tp: dict[str, Any] | None = None,
                   weights_check: dict[str, Any] | None = None, targets: list[str] | None = None, workers: int | None = None, worker_peak: int | None = None,
                   ring_slots: int | None = None, retention: str | None = None, seed_challenge: str | None = None,
                   manifest: dict[str, Any] | None = None, executed_prefix: dict[str, dict] | None = None,
                   program_dirs: dict[str, str] | None = None, challenge_key: Key | None = None, replay_k: int | None = None, replay_draw: str = "family", plan: bool = False, memo: dict | None = None,
                   cached_prefixes: dict[str, dict] | None = None) -> dict[str, Any]:
    """C2 of record.  `challenge_key` (off legacy, `challenge.replay_key`) draws the sample in place of `seed`.  `cached_prefixes`: each warm
    request's cached rows and their producers (`populate`): its attention reads its cached prefix from what they committed.  `executed_prefix`: request -> `executed_prefix_of_record(...)` from the Commit's own committed tokens;
    rows at request-relative steps >= n_exec leave the sampling pool into the named class `population.not_executed_by_rule` (never a
    stratum that grades 'not evaluated'); a request with prefix faults is not classified and fails `query_checks` by name.  `programs`: request label -> Program instances doc ({"rows", "params", ...}; requests sharing a shape share it).
    `manifest`: the row's required-value manifest (Q(P)) -- the population of record the replay's reach is reconciled against,
    identity by identity (`population.query_total` / `sampled_strata` / `uncovered_by_name` / `query_checks`); None is recorded as such.
    `weights_check`: the caller's proof that the weight slices the relations consume are the COMMITTED weights ({"equal": bool, ...})
    -- `equal` False fails the replay by name; None is recorded as 'by reference, unverified at replay'.
    `max_evaluate` caps the stratified sample but never below one VU per stratum (round-robin order; the effective cap is recorded);
    `targets` (`<op_path>[@<step>]`) add every VU of a module to the sample (targeted negatives; not subject to the cap); `replay_k` (the config run): that many units, drawn by `replay_draw` (`family_strata`).
    Addresses: the v2 query's population; a request's Build dir (`program_dirs`: request label -> dir) supplies the correspondence the
    manifest was built with (recorded as `addresses`).
    Result: True = every sampled VU replays bit-exactly and every non-empty stratum was evaluated; False = a mismatch (named);
    None = nothing evaluable / a stratum could not be evaluated (partial, never PASS).  `plan`: the same draw, read only (`planned`).
    `worker_peak` (bytes): the forked pool is fitted after the prewarm to the cgroup's memory (`fit_pool`, recorded as
    `evaluator.memory`), and a pick too large for it is refused (PoolDoesNotFit); None forks the workers asked for."""
    t_start = time.perf_counter()
    store = CommittedStore.of_record(bmap, committed, manifest)                 # reads through the manifest's alias table
    arrivals = arrivals or {}
    ranks = sorted({int((e.get("identity") or {}).get("rank", 0) or 0) for e in (bmap.get("entries") or [])}) or [0]
    all_vus, not_ev, not_ev_fam, indices, codes, addresses, anchors, addr_rec = populate(
        programs, ranks, arrivals, executed_prefix, ring_slots, attention_paths=attention_paths, tp=tp, program_dirs=program_dirs, manifest=manifest, memo=memo,
        cached_prefixes=cached_prefixes)
    pool_vus, not_exec_vus = _kept(split_not_executed(all_vus, executed_prefix), indices, all_vus)   # classified out of the pool
    picked_all, strata, tgt = sample(*family_strata(pool_vus, replay_k, per_stratum, replay_draw), seed, targets=targets, key=challenge_key)
    targeted, n_targeted, n_strat = tgt["hits"], int(tgt["added"]), int(tgt["stratified"])
    cap = max(int(replay_k or max_evaluate), len(strata))                         # never below one VU per stratum
    picked = picked_all[:min(cap, n_strat)] + picked_all[n_strat:]                # targeted VUs are outside the cap
    if plan:
        return planned(picked, indices, store, arrivals, t_start)
    if len(ranks) > 1:                                                            # TP: the map carries rank-local partials this version does not replay
        not_ev[f"multi-rank map (ranks {ranks}): rank-local partials / collectives not replayed at v1 (rank 0 module outputs only)"] += 0
    evaluated, equal, mism, not_evald = 0, 0, [], []
    strata_eval: Counter = Counter()
    strata_mism: Counter = Counter()
    fam_time, fam_n = Counter(), Counter()

    def _one(vu: dict) -> tuple[dict, dict, float]:
        t0 = time.perf_counter()
        rec = replay_vu(indices[vu["request_id"]], store, weights, vu, arrive=int(arrivals.get(vu["request_id"], 0)))
        return vu, rec, time.perf_counter() - t0

    # PARALLEL evaluation: the numpy k16-chain evaluators are GIL-bound (many small array ops), so threads gave ~2x on 16 cores; a FORKED
    # process pool shares the committed store + the weights cache copy-on-write and returns small records.  The children never touch CUDA
    # (weights are pre-read into the cache in the parent, the committed reader slices host tensors); a pool failure / timeout falls back
    # to the serial path.  Results are folded in the picked order below, so the record is independent of the schedule.
    workers = default_workers() if workers is None else workers
    workers_requested = int(workers)
    pool_note = None
    # a DEVICE-retained committed store (`--retain device`) cannot be read by forked children (CUDA after fork ->
    # AcceleratorError; every VU would come back 'not evaluated' and the honest row would FAIL closed): the caller names the retention and
    # the replay runs SERIAL in the parent's own CUDA context.  A child that hits the device anyway is a pool failure -> serial fallback.
    if retention == "device" and int(workers) > 1:
        pool_note = "device-retained committed store: forked evaluator disabled, serial replay in the parent's CUDA context"
        workers = 1
    results, workers, note, prewarmed, memory = _evaluated(_one, picked, indices, weights, int(workers), worker_peak)
    pool_note = note or pool_note
    unev_by_stratum: dict[str, dict] = {}
    child_reads = opened_in_children(results, forked=int(workers) > 1)
    for vu, rec, dt in results:
        fam_time[vu["family"]] += dt
        fam_n[vu["family"]] += 1
        if rec.get("result") is None:
            ne = {k: rec.get(k) for k in ("spec", "request_id", "step", "op_path", "row", "why")}
            ne["why_class"] = why_class(ne.get("why"))
            not_evald.append(ne)
            u = unev_by_stratum.setdefault(vu["stratum"], {"stratum": vu["stratum"], "spec": vu["spec"], "request_id": vu["request_id"], "phase": vu["phase"],
                                                            "op_path": vu["op_path"], "n": 0, "why": ne.get("why"), "why_class": ne["why_class"], "classes": []})
            u["n"] += 1
            if ne["why_class"] not in u["classes"]:
                u["classes"].append(ne["why_class"])
            continue
        evaluated += 1
        strata_eval[vu["stratum"]] += 1
        if rec["result"]:
            equal += 1
        else:
            mism.append(rec)
            strata_mism[vu["stratum"]] += 1
    for k in strata:
        strata[k]["evaluated"] = int(strata_eval.get(k, 0))
        if strata_mism.get(k):
            strata[k]["mismatch"] = int(strata_mism[k])                       # the stratum's own count, uncapped
    picks_by_request_step = _picks_by_request_step(results)
    # EVERY mismatch counted by name, uncapped (the first-12 list below carries the inputs/outputs detail only): a
    # reader of the record learns which mechanisms / requests / phases / modules disagree without re-running the replay
    mism_by = {"by_family": dict(Counter(m["spec"].split("{")[0] for m in mism)),
               "by_request": dict(Counter(str(m["request_id"]) for m in mism)),
               "by_phase": dict(Counter(str(m.get("phase")) for m in mism)),
               "by_family_phase": dict(sorted(Counter(f"{m['spec'].split('{')[0]}|{m.get('phase')}" for m in mism).items())),
               "by_family_request_phase": dict(sorted(Counter(f"{m['spec'].split('{')[0]}|{m['request_id']}|{m.get('phase')}" for m in mism).items())),
               "by_module": dict(sorted(Counter(f"{m['spec'].split('{')[0]}|{m.get('op_path')}" for m in mism).items())),
               "strata": {k: int(v) for k, v in sorted(strata_mism.items())}}
    strata_unevaluated = sorted(k for k, v in strata.items() if v["population"] > 0 and v["evaluated"] == 0)
    fams = sorted({v["family"] for v in all_vus})
    wc = dict(weights_check) if isinstance(weights_check, dict) else None
    if wc is not None and wc.get("equal") is False:
        result, grade = False, "mismatch"
        why = f"weights at replay != committed weights root ({wc.get('why') or 'the registered inputs the relations consume are not the committed ones'})"
    elif mism:
        result, grade = False, "mismatch"
        why = mism[0]["why"] + (f" (+{len(mism) - 1} more)" if len(mism) > 1 else "")
    elif evaluated == 0:
        result, grade = None, "not-evaluated"
        why = "no VU could be evaluated: " + (not_evald[0]["why"] if not_evald else "empty population")
    elif strata_unevaluated:
        result, grade = None, "partial"
        why = f"{len(strata_unevaluated)} strata not evaluated: {strata_unevaluated[:3]}"
    elif not_evald:
        # picks not evaluated INSIDE strata that still had an evaluated pick used to leave the grade 'complete'
        # with sample.not_evaluated_n > 0 -- a run-time fault (missing committed input, bytes not retained, evaluator error, device) on
        # some picks then rode a PASS unnamed.  'complete' means every picked VU was evaluated: any not-evaluated pick -> partial, the first
        # fault named (fault classes first; a by-name pick inside a mixed stratum is named too and, having no fully unevaluated stratum,
        # cannot be lifted by the named-gap rule -- fail closed).  Real records: 0 complete-with-not-evaluated picks (rev, 38 pairs).
        faults = [ne for ne in not_evald if ne.get("why_class") not in WHY_CLASSES_BY_NAME]
        first = (faults or not_evald)[0]
        by_cls = Counter(ne.get("why_class") or "other" for ne in not_evald)
        result, grade = None, "partial"
        why = (f"{len(not_evald)} pick(s) not evaluated inside evaluated strata ({', '.join(f'{c} x {n}' for c, n in sorted(by_cls.items()))})"
               f"{' -- run-time fault, not a gap on the record' if faults else ''}: {first.get('spec')} {first.get('op_path')} "
               f"(req {first.get('request_id')}, step {first.get('step')}): {str(first.get('why'))[:160]}")
    else:
        result, grade, why = True, "complete", None
    steps_seen = sorted({v["step"] for v in picked})
    catch = _catch_probability(strata)
    # the population of record is Q(P) (the required manifest), never the replay's own enumeration: reconcile the reach
    # identity by identity and carry query_total / sampled_strata / uncovered_by_name on the record (a required identity the exact replay
    # has no VU for is NAMED with its mechanism of record -- never inside the grade, never silently "qualified")
    evaluated_keys = {(vu["request_id"], int(vu["step"]), vu["op_path"]) for vu, rec, _dt in results if rec.get("result") is not None}
    qpop = query_population(manifest, addresses, evaluated_keys, ranks=ranks, strata=strata, executed_prefix=executed_prefix,
                            controller_rows=controller_rows_of(programs, indices))   # [R4 c2] the replay's own guard-row count per request
    return {"schema": SCHEMA, "method": "sampled-exact-replay", "result": result, "grade": grade, "why": why, "catch_probability": catch,
            "population": {"vus": len(all_vus), "source": "Program instances.json.gz rows (every row; evaluable = families with a registered evaluator)",
                           "requests": sorted(programs), "by_family": dict(Counter(v["family"] for v in all_vus)), "not_evaluable_by_family": dict(not_ev_fam),
                           "not_evaluable": dict(not_ev), "not_evaluable_codes": {k: codes.get(k) or RC.reason_code("other") for k in not_ev}, "strata_n": len(strata),
                           "authority": "Q(P) -- the required manifest; `vus` above is the replay's enumeration of the Program rows, reconciled below",
                           "query_total": qpop["query_total"], "sampled_strata": qpop["sampled_strata"], "uncovered_by_name": qpop["uncovered_by_name"],
                           "not_executed_by_family": qpop["not_executed_by_family"], "not_executed_rule": qpop["not_executed_rule"],
                           "not_executed_by_rule": dict(qpop["not_executed_by_rule"], vus=len(not_exec_vus),
                                                        vus_by_request=dict(Counter(v["request_id"] for v in not_exec_vus)),
                                                        strata_excluded={k: {"population": int(n), "label": "not evaluable: unexecuted"}
                                                                         for k, n in sorted(Counter(v["stratum"] for v in not_exec_vus).items())}),
                           "uncovered_identities": qpop["uncovered_identities"], "controller": qpop.get("controller"),   # [R4] the lifted controller, by name
                           "query_checks": qpop["query_checks"], "scope": qpop["scope"]},
            "addresses": addr_rec,
            "sample": {"per_stratum": int(replay_k or per_stratum), "seed": int(seed), "seed_source": seed_source, "strata_by": replay_draw if replay_k else "unit", **({"k": int(replay_k)} if replay_k else {}),
                       # under the verifier-challenge form the FULL challenge is on the record: a reader recomputes
                       # sha256(challenge|run_root)[:8] (challenge_seed) as a fact -- the source text alone (challenge[:16], repr) cannot be
                       **({"challenge": str(seed_challenge)} if seed_challenge is not None else {}),
                       "picked": len(picked), "evaluated": evaluated, "equal": equal,
                       "mismatch_n": len(mism), "not_evaluated_n": len(not_evald), "max_evaluate": int(replay_k or max_evaluate), "cap_effective": int(cap),
                       "stratified_picked": min(cap, n_strat), "targeted": targeted, "targeted_n": n_targeted,
                       "order": "round-robin across strata (every stratum's first pick before any second); targeted VUs appended, outside the cap",
                       "picks_by_request_step": picks_by_request_step,      # {request: {step: [picked, evaluated, mismatch]}}
                       # the step is the REQUEST-RELATIVE step (the executed prefix's n_exec counts the request's own steps); the engine
                       # step of a pick is arrival + step -- the arrival per request beside it, so a mixed-arrivals reader adds it, not guesses
                       "picks_by_request_step_key": "[picked, evaluated, mismatch] per (request, request-relative step); engine step = picks_by_request_arrival[request] + step; targeted picks included",
                       "picks_by_request_arrival": {r: int(arrivals.get(r, 0)) for r in picks_by_request_step}},
            "timing": {"wall_s": round(time.perf_counter() - t_start, 3), "workers": int(workers), "cpu_s_sum": round(sum(fam_time.values()), 3),
                       "per_family_s": {f: round(fam_time[f], 3) for f in sorted(fam_time)}, "per_family_n": {f: int(fam_n[f]) for f in sorted(fam_n)}},
            # how the committed store was read: the collector's retention and the evaluator that ran (forked pool / serial)
            "evaluator": {"retention": retention, "workers_requested": workers_requested, "workers": int(workers),
                          "mode": "forked pool" if int(workers) > 1 else "serial (parent process)", "note": pool_note,
                          "prewarmed_libraries": prewarmed,               # twin .so's dlopen'ed in the parent before the fork
                          # gc.freeze before the fork: the children's collections never traverse the
                          # inherited C2 heap (otherwise a 72 GB-per-child COW); {} when no pool was forked
                          "fork": dict(_FORK_STATS) if int(workers) > 1 else {}, **({"memory": memory} if memory else {})},
            "strata": strata, "strata_unevaluated": strata_unevaluated, **({"value_reads_in_children": child_reads} if child_reads else {}),
            # EVERY unevaluated stratum with its reason and class, uncapped (the verdict's named-gap rule decides per stratum)
            "unevaluated_strata": [unev_by_stratum[k] for k in strata_unevaluated if k in unev_by_stratum],
            "why_classes": {"by_name": list(WHY_CLASSES_BY_NAME), "fault": list(WHY_CLASSES_FAULT),
                            "rule": "a by-name class is a gap of this replay version (may be carried on a PASS when form (B) covers the values); "
                                    "a fault class is a run that did not evaluate (never carried)"},
            "strata_keys": {"mechanisms": sorted({k.split("|")[0] for k in strata}), "requests": sorted({k.split("|")[1] for k in strata}),
                            "ranks": (["rank 0 (tp1, or the rank-0 shard of a per-rank map)"] if ranks == [0] else [f"rank {r}" for r in ranks]),
                            "steps": {"early": sorted(s for s in steps_seen if s <= 1), "late": sorted(s for s in steps_seen if s > 1)},
                            "phases": sorted({k.split("|")[3] for k in strata}),
                            "modules": len({k.split("|")[4] for k in strata if k.count("|") >= 4}),
                            "key": "spec | request | rank | phase | module op_path | lifetime (window); attention rows: spec with T -> NB = ceil(T/BN) "
                                   "(the block-count mechanism; the row's own spec keeps T)",
                            "lifetime": {"values": dict(Counter(v["lifetime"] for v in all_vus)), "ring_slots": ring_slots,
                                         "rule": "engine step >= ring_slots -> recycled-window (storage the ring already committed under an earlier step's "
                                                 "identities); else fresh-window; unwindowed collectors: one value, stated",
                                         "sampled": dict(Counter(v["lifetime"] for v in picked))},
                            "phase": ("prefill / first-decode / mid / last EXECUTED step of each request (`last` is anchored to n_exec - 1 "
                                      "under the executed prefix -- the lag step of an EOS-stopped request -- else to the cap; steps past the "
                                      "prefix are 'not-executed', split out of the pool by name)"),
                            # per request: the step `last` is anchored to, the anchor's rule, and the lag step when there is one
                            "phase_anchor": {r: anchors[r] for r in sorted(anchors)}},
            "mismatch": [{k: m.get(k) for k in ("spec", "request_id", "step", "engine_step", "op_path", "row", "phase", "inputs", "outputs", "why")} for m in mism[:12]],
            "mismatch_by": mism_by,                                       # every mismatch by name, uncapped
            "not_evaluated": not_evald[:12],
            # mechanisms of the Program's rows this replay does NOT read even when the row IS evaluated (named, counted)
            "uncovered_mechanisms": _uncovered_mechanisms(all_vus, not_ev_fam),
            # Not-evaluable families BY NAME WITH COUNTS + reason (footnoted on the row; bounds the claim)
            "unreplayable": [{"family": f, "n": int(not_ev_fam[f]),
                              "why": NOT_YET.get(f.split(" (")[0], "structure / interior: no identity" if "(structure" in f
                                                 else ("interior producer: composed into its consumer's VU" if "(interior" in f
                                                       else ("MoE plane row -> plane slice mapping failed: see population.not_evaluable" if "(plane mapping failed)" in f
                                                             else "no registered evaluator")))}
                             for f in sorted(not_ev_fam)],
            "weights": (wc if wc is not None else {"equal": None, "note": "by reference, unverified at replay (no weights check supplied)"}),
            # the committed up plane is not the down plane's bytes, per (request, step, layer) -- beside the replay
            "moe_planes": {"alias_check": moe_plane_alias_check(store)},
            # the engine configuration the evaluators were conditioned on, measured in this process (VLLM_BATCH_INVARIANT & co.)
            "engine_of_record": engine_facts_of_record(),
            # the composed INTERIOR producers' evaluators are named too (the quant twin is load-bearing for every FP8 GEMM VU)
            "evaluators": {f: EVALUATORS[f] for f in sorted(set(fams) | {k.split(" (")[0] for k in not_ev_fam if "(interior" in k}) if f in EVALUATORS},
            # the fitted twins name what pinned them (fixture sha256, kernels, device, date, tests) -- only for families present
            "evaluator_provenance": {f: EVALUATOR_PROVENANCE[f] for f in sorted(set(fams) | {k.split(" (")[0] for k in not_ev_fam if "(interior" in k})
                                     if f in EVALUATOR_PROVENANCE},
            "applicability": {"families_evaluated": fams, "families_not_yet": {k: v for k, v in NOT_YET.items() if k in not_ev_fam},
                              "ranks": ranks,
                              "note": "honest-implementation qualification of the collector's mapping on a seeded stratified sample; not a proof over the population"}}
