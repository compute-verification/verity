"""Δ_commit — serving overhead of one-stage commitment on the SAME batch-invariant vLLM workload (CMT-3).

    Δ_commit = T(BI serving + commitment) / T(BI serving) − 1

Two arms in ONE process on ONE engine (vLLM d9105ea80, eager, bf16, VLLM_BATCH_INVARIANT=1, the adapter's declared profile):
  control       Verity disabled entirely — no observer, no hooks, no wrapped runner methods; the loop below is the only code around
                `engine.step()` (the same loop runs the instrumented arm, so it is in both numerator and denominator)
  instrumented  a committer installed for the run (`--committer native_collect_v2b`, the windowed worker, which commits rows-sha512;
                other committers plug in through the same `Committer` interface), uninstalled afterwards; the harness asserts the engine
                is pristine again before the next control run.
Pairs are interleaved (control, instrumented) with the order alternating per pair; ≥ 5 pairs.  Warm-up: two control passes and one
instrumented pass on throwaway prompts of the same shape (Triton JIT, allocator, pinned pool), reported separately.  Weight registration
(the registered-input class) is done once and timed apart from steady state.

Per run: wall (first add_request → last step() return), per-step wall (host clock between step() returns), tokens, tokens/s,
peak GPU memory (torch max_memory_allocated / reserved), host RSS/HWM, generated tokens (must be identical across arms: BI=1).
Instrumented runs add the component spans (acquisition / movement / hashing / root_finalisation / opening_generation / wait), the
per-step hashing time and backlog (steady state), root-ready delay, retained bytes, openings generated + verified, coverage declaration.

Outputs (DIR): runs.jsonl (one row per run), steps.jsonl (one row per step), summary.json, summary.md, declaration_<committer>.json,
versions.json.  Every number is MEASURED unless the key says otherwise.

    PYTHONPATH=. verity-vllm commit \
        --workload workloads/workload_cov_b0_b1_c256.json --pairs 5 --out /workspace/cp/cmt3/b0_b1

PROTOCOL SUMMARY (Commit phase entry: the instrumented runs + the fail-closed X-03 verdict)
  Purpose: run the row's workload on ONE engine in interleaved (control, instrumented) pairs; the instrumented arm commits every
    required value through the committer (row_pod: native_collect_v2b + the FA2/FA3 M1 tap); the verdict is then computed from the
    committed bytes -- the checks: required-value coverage (`coverage`), openings, local replay (`C2`: oracle compare vs the Match
    record + sampled replay), boundary linkage (`linkage`), checkpoint binding (`weights_pin`), execution extent (`executed_prefix`)
    -- and Δ_commit (the overhead) is measured beside it.
  Inputs: --workload wl.json --case ROLE; --required-manifest manifest.json (the required-value manifest) + --program-digest (the Build's); --program-dir |
    --programs-root (the Build Programs); --match-dir (this attempt's Match record); --weights-of-record | --checkpoint-dir;
    --retain none|host|device (default none: openings by replay, --open-after-release on), --openings, --pairs,
    --window-mb/--window-slots and --bounded-staging (the collector's defaults: `resolve_staging`).
  Outputs (--out DIR): runs.jsonl, steps.jsonl, summary.json/.md, verdict.json, versions.json, source_identity.json,
    binding_map_p<k>.json, sampled_replay_p<k>.json, host_admission_<kind>.json, weights_of_record.json, declaration_<committer>.json, protocols.json.
  Invariants: the Program/source identity precheck (`identity`) runs before any engine (exit 3 PRECHECK_FAIL_SOURCE_IDENTITY); a
    required-value manifest whose program_digest is not the Build's is refused before the engine starts; the population is Q(P) at the CAP shape -- EOS-stopped
    positions are classified not-executed by rule, never cut out; control and instrumented tokens must be equal; PASS only when
    every commit_verdict check holds (the first failing reason is the verdict; never "PASS with errors recorded").
"""

from __future__ import annotations

import time

from verity_vllm.records.commit_run import SCHEMA

_T_PROCESS0 = time.time()      # process start (before the torch / vLLM imports): startup.imports = engine.build start - this

import contextlib
import dataclasses
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from verity_vllm import protocols as PO
from verity_vllm.acquire.sources import hidden_source as HS, taps as TAPS
from verity_vllm import settings  # noqa: E402 (after the process clock, like the imports around it)
from verity_vllm.config import group, option, workload_arg
from verity_vllm.engine import engine_profile as prof  # noqa: F401

prof.apply_env()

import torch  # noqa: E402

from verity_vllm.engine import arrivals as AR, served as SV, vllm_adapter as va  # noqa: E402
from verity_vllm.pipeline.timeline import Timeline  # noqa: E402  (torch-free; VERITY_TIMELINE -> append-only JSONL, else a no-op sink)
from verity_vllm.pipeline.commit_attach import attach_of_record, classes_of_record  # noqa: E402
from verity_vllm.commit import challenge as CH, opened as OV, step_rows as SR  # noqa: E402
from verity_vllm.pipeline import boundary_plan as BP, c2_replay as C2, commit_plan as CP, gpu_lease as GL, group_reuse as GR, replay_bundle as RB  # noqa: E402
from verity_vllm.pipeline.c2_replay import rss_kb  # noqa: E402
from verity_vllm.pipeline.report import bootstrap_ci  # noqa: E402
from verity_vllm.pipeline import commit_coins as CC, remote_session as RS  # noqa: E402

# automatic stage telemetry of THIS process: hot/cold, engine built or reused (+ reason), warm-up, arms, validation
TL = Timeline.from_env("commit_delta", hot=False)


def _rid_of(order: dict, rid) -> str:
    """The engine's per-step row id -> the workload's `r<i>`: vLLM may suffix the id we passed to add_request with `-<8 hex>`
    (verity_vllm.engine.engine_driver.external_id strips it); an id the workload does not name is returned raw (named downstream)."""
    r = str(rid)
    if r in order:
        return order[r]
    try:
        from verity_vllm.engine.engine_driver import external_id
        e = external_id(r)
        if e in order:
            return order[e]
    except Exception:  # noqa: BLE001
        pass
    return r


def git_head() -> str | None:
    for p in (Path.cwd() / "EXPORT.json",):
        if p.exists():
            try:
                return json.loads(p.read_text()).get("commit")
            except Exception:  # noqa: BLE001
                pass
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


C2_PROGRAMS_OBJECT_FACTOR = 15.0        # Python-object bytes per json byte of an instance file: B8 FP8 row 4,924 MB json -> 70,300 MiB (FP8-STAGING §6)
# the C2 sampled replay's forked evaluator children: their private (copy-on-write) pages are charged to the cgroup
# while the Programs + frozen pool are live in the parent -- B8 attempt 1 measured cgroup +7,000 MiB over 33 children (pool blocks shared);
# 8 GiB on the record as an EMPIRICAL term of the c2 phase (the run's 1 Hz cgroup timeline measures it).  A private-anon pinned pool block
# is COPIED per child by the kernel at fork (attempt 2) -- the pool's blocks are shared mappings now; this term is the children's own pages.
FORK_CHILDREN_BYTES = 8 << 30
# two EMPIRICAL terms of the later phases, measured by attempt 3's
# footprint (rss - pinned pool, per phase): the freed Programs leave 31,460 MiB resident (44 % of the 71,445 loaded: pymalloc arenas / glibc
# heap keep the pages) for the rest of the process; and 20,047 MiB of the previous pair's pool pages stay resident in the next pair, once
# (measured; mechanism hypothesised: blocks whose munmap release() deferred under live tensor views).  The residual is taken as a FRACTION of the
# Programs forecast (the term scales with the Programs), the carry as a constant; the run's footprint measures both (HOST PEAK gap accounting).
C2_RESIDUAL_FRACTION = 0.45
POOL_CARRY_BYTES = 20 << 30


def c2_programs_forecast(programs_root: str | None, program_dir: str | None, factor: float = C2_PROGRAMS_OBJECT_FACTOR) -> dict[str, Any]:
    """EMPIRICAL FORECAST of the host bytes the C2 sampled replay's loaded Programs take in this process: the
    instance files it will load (`build_request_*/instances.json.gz` under --programs-root, or --program-dir's), streamed and counted
    (rows = `{"i": ` occurrences, json bytes = decompressed size; nothing is parsed or kept), x `factor` json byte -> object bytes.  One
    calibration point (B8, 7,853,122 rows, 628 B/row json, 9,387 B/row objects); the run measures the real number (C2 PROGRAMS FREED)."""
    import glob
    import gzip
    files: list[str] = []
    if programs_root:
        files = sorted(glob.glob(os.path.join(str(programs_root), "build_request_*", "instances.json.gz")))
    if not files and program_dir:
        p = os.path.join(str(program_dir), "instances.json.gz")
        files = [p] if os.path.exists(p) else []
    rows = 0
    jb = 0
    per_file = []
    needle = b'{"i": '
    t0 = time.perf_counter()
    for f in files:
        n = 0
        b = 0
        tail = b""
        try:
            with gzip.open(f, "rb") as fh:
                while True:
                    buf = fh.read(64 << 20)
                    if not buf:
                        break
                    b += len(buf)
                    data = tail + buf
                    n += data.count(needle)
                    tail = data[-(len(needle) - 1):]
        except Exception as e:  # noqa: BLE001  -- an unreadable file is named, never silently 0
            per_file.append({"file": f, "error": repr(e)[:200]})
            continue
        per_file.append({"file": f, "rows": n, "json_bytes": b, "gz_bytes": os.path.getsize(f)})
        rows += n
        jb += b
    return {"bytes": int(jb * factor), "rows": rows, "json_bytes": jb, "factor": factor, "files": per_file, "wall_s": round(time.perf_counter() - t0, 2),
            "kind": "empirical_forecast", "calibration": "B8 qwen3-4b-fp8 1024/128: 4,924 MB json -> 70,300 MiB objects (9,387 B/row), FP8-STAGING §6",
            "why": None if files else "no instance files found (no --programs-root / --program-dir instances.json.gz): the C2 term is 0 by absence, not by measurement"}


def cgroup_mem_current() -> dict[str, int | None]:
    """The container's memory accounting (what the OOM-killer reads): current / peak / max of the cgroup, bytes (None outside one)."""
    out: dict[str, int | None] = {"current": None, "peak": None, "max": None}
    for k, paths in (("current", ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes")),
                     ("peak", ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes")),
                     ("max", ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"))):
        for p in paths:
            try:
                v = Path(p).read_text().strip()
                if v.isdigit() and int(v) < (1 << 60):
                    out[k] = int(v)
                    break
            except OSError:
                pass
    return out


def nvidia_smi_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=20).stdout.strip().splitlines()
        return int(out[0]) if out else None
    except Exception:  # noqa: BLE001
        return None


def run_workload(engine: va.Engine, requests: list[dict], sampling: dict, step: Any = None) -> dict:
    """The serving loop (identical for both arms): admit requests at their arrive_step, step until done, time every step() return.
    ``step(eng.step)``, if given, runs each engine step (the remote session's `Lockstep.step`)."""
    va.fresh_prefix_cache(engine)
    eng = engine.llm.llm_engine
    pending = sorted(range(len(requests)), key=lambda i: (int(requests[i].get("arrive_step") or 0), i))
    tokens: dict[str, list[int]] = {}
    finish: dict[str, str | None] = {}
    finished: set[str] = set()
    step_t: list[float] = []
    step_tokens: list[int] = []
    it, forwards = 0, AR.ForwardCounter(eng, AR.scheduler_of(engine.llm))
    t0 = time.perf_counter()
    while pending or eng.has_unfinished_requests():
        while pending and int(requests[pending[0]].get("arrive_step") or 0) <= forwards.n:
            r = requests[pending.pop(0)]
            eng.add_request(request_id=r["request_id"], prompt={"prompt_token_ids": list(r["prompt_token_ids"])},
                            params=va.sampling_params(r, sampling))
        if not eng.has_unfinished_requests():
            r = requests[pending.pop(0)]
            eng.add_request(request_id=r["request_id"], prompt={"prompt_token_ids": list(r["prompt_token_ids"])},
                            params=va.sampling_params(r, sampling))
            continue
        outs = eng.step() if step is None else step(eng.step)
        forwards.stepped(outs)
        now = time.perf_counter()
        step_t.append(now)
        n = 0
        for o in outs:
            for c in o.outputs:
                tokens[o.request_id], finish[o.request_id] = [int(t) for t in c.token_ids], c.finish_reason
                n += 1
            if o.finished:
                finished.add(o.request_id)
        step_tokens.append(n)
        it += 1
    t1 = time.perf_counter()
    missing = [r["request_id"] for r in requests if r["request_id"] not in finished]
    if missing:
        raise RuntimeError(f"{len(missing)} requests did not finish")
    walls = [step_t[0] - t0] + [b - a for a, b in zip(step_t, step_t[1:])]
    gen = sum(len(v) for v in tokens.values())
    return {"wall_s": t1 - t0, "iterations": it, "step_wall_s": walls, "step_tokens_out": step_tokens, "generated_tokens": gen,
            "prompt_tokens": sum(len(r["prompt_token_ids"]) for r in requests), "tokens": tokens, "finish_reasons": finish, "t_last_step": step_t[-1]}


# the compiled check's sampled steps are drawn from the run root AFTER commitment: seed = int(run_root.hex()[:8], 16)
# (first 4 bytes).  Recorded on value_correspondence.compiled_check.seed_source so the finalizer recomputes it from the same runs.jsonl as a fact.
COMPILED_REPLAY_SEED_SOURCE = "run root (first 4 bytes, after commitment): int(run_root.hex()[:8], 16)"


def admission_lag(wl: dict) -> int:
    """The lag rows the host admission counts: the declared LAG of the loaded workload's target (`global_program.workload_target`)."""
    from verity_vllm.pipeline import global_program as _GP
    return int(_GP.workload_target(wl.get("target")).lag())


def stop_rule_of_record(eng, wl: dict, versions: dict) -> dict[str, Any]:
    """the Program's DECLARED stop rule, read ONCE for every Commit consumer (required
    population, composition, boundary linkage, sampled-replay population): `eos` = global_program.stop_rule's eos_token_ids over the pinned
    checkpoint's generation_config / config + the workload record's sampling block, read from the file `wl` was loaded from, as GP-01
    reads it (None = undeclared); `lag` = the workload Program's declared
    LAG (TargetProfile.lag() of the manifest's declared target -- the SAME fact GP-01 composes into Workload_v1{..., lag}), kept only when
    the engine's RESOLVED read-back (versions.engine_kwargs.async_scheduling == the RunHeader's attestation) AGREES; a disagreement or an
    unattested engine leaves it None (every EOS stop then fails by name) -- never the engine value as the rule, never inferred from the
    trace.  `notes` names what was refused."""
    out: dict[str, Any] = {"eos": None, "lag": None, "declared_lag": None, "engine_lag": None, "async_scheduling": None, "notes": [], "srule": None}
    try:
        from verity_vllm.pipeline import global_program as _GP
        from verity_vllm.program.workload import eos_lag_rows_declared
        from verity_vllm.records.workload import Workload
        _srule = _GP.stop_rule(eng.cp.get("local_path"), None, Workload.load(wl["source"]).sampling)
        out["srule"] = {k: _srule.get(k) for k in ("eos_token_ids", "source", "refusal") if k in _srule}
        out["eos"] = None if _srule.get("refusal") else _srule.get("eos_token_ids")
        _decl_lag = _GP.workload_target(wl.get("target")).lag()
        _eng_async = (versions.get("engine_kwargs") or {}).get("async_scheduling")
        # one field, one reader: GP-01's eos_lag_rows_declared over the engine's resolved read-back
        _eng_lag = eos_lag_rows_declared({"async_scheduling": (versions.get("engine_kwargs") or {}).get("async_scheduling")})
        out["declared_lag"], out["engine_lag"], out["async_scheduling"] = _decl_lag, _eng_lag, _eng_async
        if _eng_lag is None:
            out["notes"].append(f"the Program declares LAG {_decl_lag} but this engine does not attest async_scheduling: EOS stops fail by name")
        elif _eng_lag != _decl_lag:
            out["notes"].append(f"the Program declares LAG {_decl_lag} but this engine resolved async_scheduling={_eng_async!r} (LAG {_eng_lag}): "
                                f"not an execution of the declared scheduler -- EOS stops fail by name")
        else:
            out["lag"] = int(_decl_lag)
    except Exception as _ee:  # noqa: BLE001  -- undeclared, never assumed
        out["notes"].append(f"stop rule of record unavailable ({_ee!r}): early-stopped requests fail by name")
    for n in out["notes"]:
        print(f"[delta] {n}", flush=True)
    return out


def engine_schedule(request_rows: dict[int, list[tuple[str, int, int]]] | None, declared: dict[str, int], caps: dict[str, int]) -> dict:
    """The engine's schedule of record vs the declaration.  `request_rows` is the scheduler's per-committed-step row
    order as the collector recorded it (which requests ran in which step); a request's OBSERVED arrival is the first committed step it ran
    in.  The serving loop admits a request once the engine's forwards reach its declared arrive_step (`arrivals.ForwardCounter`: the
    bookkeeping iteration after the last request finishes runs no forward pass) OR immediately when the engine is idle (nothing running),
    so on a workload with gaps (r0's cap shorter than the next arrival) the committed-step axis and the declared arrive_steps drift apart --
    the required population must be placed on the axis the record is written on.  It is USED only when every declared request was
    observed, each ran a contiguous run of steps (no preemption) of exactly its expected count; otherwise the declaration stands and coverage
    fails by name.  `caps` = the EXPECTED steps per request = its executed prefix of record (the cap on a length stop,
    served + LAG on an EOS stop -- executed_prefix.executed_prefix_of_record)."""
    observed: dict[str, int] = {}
    steps_of: dict[str, list[int]] = {}
    for st in sorted(request_rows or {}):
        for rid, _lo, _hi in request_rows[st]:
            observed.setdefault(str(rid), int(st))
            steps_of.setdefault(str(rid), []).append(int(st))
    contiguous = {r: v == list(range(v[0], v[0] + len(v))) for r, v in steps_of.items()}
    count_eq = {r: len(steps_of.get(r, [])) == int(caps[r]) for r in caps}
    complete = bool(declared) and set(observed) >= set(declared)
    use = "observed" if (request_rows and complete and all(contiguous.values()) and all(count_eq.values())) else "declared"
    why = (None if use == "observed" else
           ("no per-step request rows (single request or none recorded): declared arrive_steps" if not request_rows else
            (f"requests never scheduled: {sorted(set(declared) - set(observed))[:4]}" if not complete else
             (f"non-contiguous scheduling (preemption?): {sorted(r for r, ok in contiguous.items() if not ok)[:4]}" if not all(contiguous.values()) else
              f"scheduled steps != expected (executed prefix of record): {sorted(r for r, ok in count_eq.items() if not ok)[:4]}"))))
    return {"basis": "engine schedule of record: per-committed-step request rows (the scheduler's row order as the collector recorded it); "
                     "arrival = the first committed step the request ran in",
            "declared_arrive_steps": dict(declared), "observed_arrive_steps": dict(sorted(observed.items(), key=lambda kv: int(kv[0][1:]) if kv[0][1:].isdigit() else 0)),
            "equal_declared": {r: observed.get(r) == v for r, v in declared.items()} if observed else None,
            "all_equal_declared": bool(observed) and all(observed.get(r) == v for r, v in declared.items()),
            "scheduled_steps": {r: len(v) for r, v in steps_of.items()}, "contiguous": all(contiguous.values()) if contiguous else None,
            "steps_equal_caps": all(count_eq.values()) if count_eq else None, "complete": complete, "use": use, "why": why}


def assert_pristine(engine: va.Engine, baseline: dict) -> None:
    runner, model = engine.runner, engine.model
    for nm in ("execute_model", "sample"):
        if nm in runner.__dict__ and not baseline["runner_had"].get(nm):
            raise AssertionError(f"runner.{nm} still wrapped")
    n_hooks = sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for _, m in model.named_modules())
    if n_hooks != baseline["hooks"]:
        raise AssertionError(f"forward hooks present: {n_hooks} != baseline {baseline['hooks']}")
    n_wrapped = sum(1 for _, m in model.named_modules() if "forward" in m.__dict__)
    if n_wrapped != baseline.get("wrapped_forwards", 0):
        raise AssertionError(f"wrapped module.forward present: {n_wrapped} != baseline {baseline.get('wrapped_forwards', 0)}")
    smp = getattr(runner, "sampler", None)
    if isinstance(smp, torch.nn.Module) and (len(smp._forward_hooks) + len(smp._forward_pre_hooks)) != baseline["sampler_hooks"]:
        raise AssertionError("sampler hooks present")


def baseline_state(engine: va.Engine) -> dict:
    runner, model = engine.runner, engine.model
    smp = getattr(runner, "sampler", None)
    return {"runner_had": {nm: nm in runner.__dict__ for nm in ("execute_model", "sample")},
            "hooks": sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for _, m in model.named_modules()),
            "wrapped_forwards": sum(1 for _, m in model.named_modules() if "forward" in m.__dict__),
            "sampler_hooks": (len(smp._forward_hooks) + len(smp._forward_pre_hooks)) if isinstance(smp, torch.nn.Module) else 0,
            "cuda_alloc_mib": int(torch.cuda.memory_allocated() >> 20) if torch.cuda.is_available() else 0}


def _process_state() -> dict[str, Any]:
    import threading
    st: dict[str, Any] = {"threads": threading.active_count(), "gc_enabled": gc.isenabled(), "gc_threshold": list(gc.get_threshold()),
                          "gc_count": list(gc.get_count()), "gc_tracked_objects": len(gc.get_objects()), "gc_frozen": gc.get_freeze_count(),
                          "switch_interval_s": sys.getswitchinterval(), "torch_threads": torch.get_num_threads(),
                          "torch_interop_threads": torch.get_num_interop_threads()}
    with contextlib.suppress(Exception):
        st["affinity_cores"] = len(os.sched_getaffinity(0))
    with contextlib.suppress(Exception):
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(("VmRSS", "Threads")):
                    k, v = line.split(":", 1)
                    st[k.strip().lower()] = v.strip()
    return st


def cuda_census() -> dict[int, tuple[int, str]]:
    """{storage data_ptr: (nbytes, tensor type)} of every CUDA tensor the Python gc can see (the hot worker's post-job residue audit
    diffs this against the census taken right after the engine build; storage held only from C++ is invisible here, so a residue that shows
    in torch.cuda.memory_allocated but not in the census points at a native holder)."""
    out: dict[int, tuple[int, str]] = {}
    for o in gc.get_objects():
        try:
            if isinstance(o, torch.Tensor) and o.is_cuda and o.numel() > 0:
                st = o.untyped_storage()
                p = int(st.data_ptr())
                if p not in out:
                    out[p] = (int(st.nbytes()), type(o).__name__)
        except Exception:  # noqa: BLE001
            pass
    return out


def cuda_residue(base: dict[int, tuple[int, str]], now: dict[int, tuple[int, str]], top: int = 6) -> dict:
    """Python-visible CUDA storages present now but absent from the baseline census: count, MiB, the largest few with their holders."""
    new = {p: v for p, v in now.items() if p not in base}
    largest = sorted(new.items(), key=lambda kv: -kv[1][0])[:top]
    holders = []
    if largest:
        want = {p for p, _ in largest}
        for o in gc.get_objects():
            try:
                if isinstance(o, torch.Tensor) and o.is_cuda and int(o.untyped_storage().data_ptr()) in want:
                    refs = []
                    for r in gc.get_referrers(o)[:6]:
                        if isinstance(r, dict):          # an instance __dict__: name the owner
                            owners = [type(x).__name__ for x in gc.get_referrers(r) if getattr(x, "__dict__", None) is r]
                            refs.append("dict<-" + (owners[0] if owners else "?"))
                        else:
                            refs.append(type(r).__name__)
                    holders.append({"mib": round(o.untyped_storage().nbytes() / 2**20, 1), "shape": list(o.shape), "dtype": str(o.dtype), "referrers": refs})
                    want.discard(int(o.untyped_storage().data_ptr()))
                    if not want:
                        break
            except Exception:  # noqa: BLE001
                pass
    return {"count": len(new), "mib": round(sum(v[0] for v in new.values()) / 2**20, 1), "largest": holders}


def make_committer(kind: str, engine: va.Engine, args: Options, run_id: str):
    if kind == "native_collect_v2b":
        # the windowed worker (candidate B: native hook bodies and FA2 tap glue, size-adaptive gather windows, the native worker step),
        # which commits rows-sha512 (`step_rows`) over a ring of device windows; VERITY_HI_PRIO=1 adds the high-priority serving stream
        # variant (labelled `_hp`)
        from verity_vllm.commit.committer.native_collect import NativeCollectCommitter

        return NativeCollectCommitter(engine, run_id=run_id, arena_bytes=((args.arena_mb << 20) if args.arena_mb else None), gather=True,
                                      native_hooks=True, auto_blocks=True, native_worker=True,
                                      populate_threads=getattr(args, "pinned_populate_threads", None),
                                      commit_scheme=getattr(args, "commit_scheme", None))
    if kind == "hooks_only":
        from verity_vllm.commit.committer.native_host import NativeHostCommitter

        class HooksOnly(NativeHostCommitter):        # ablation: acquisition hooks fire, nothing is copied or hashed (NOT a commitment)
            name = "hooks_only"

            def _acquire(self, name, t, job):
                t0 = time.perf_counter()
                if t.device.type == "cuda" and t.numel():
                    job.metas.append(None)
                self.spans.add("acquisition", time.perf_counter() - t0, step=job.step)

            def _commit_step(self, job):
                job.event.synchronize()
                from verity_vllm.acquire.committer_api import StepCommitment
                zero = b"\0" * SR.ROOT_BYTES
                self._layouts[job.step] = []
                self._done[job.step] = StepCommitment(job.step, zero, zero, 0, 0, len(job.metas), b"\0" * 32, job.t_step_end,
                                                      time.perf_counter(), job.backlog)

        com = HooksOnly(engine, run_id=run_id, hash_threads=args.hash_threads, staging="direct")   # no arena: nothing is copied
        com.chunk = args.chunk
        return com
    if kind.startswith("plugin:"):
        # `plugin:<module>:<factory>` — an outside committer exposing the api.Committer interface: factory(engine, run_id=..., **kw)
        _, mod, fn = kind.split(":", 2)
        import importlib

        factory = getattr(importlib.import_module(mod), fn)
        return factory(engine, run_id=run_id)
    raise ValueError(f"--committer {kind}: the Commit's committers are native_collect_v2b (the windowed worker, rows-sha512), hooks_only "
                     "and plugin:<module>:<factory>")


from verity_vllm.check.commit_rules import REQUIRED_CLASSES_DEFAULT, class_coverage, required_classes_default  # noqa: E402
from verity_vllm.check.cublas_route_pin import stamp as route_pin_stamp  # noqa: E402
from verity_vllm.check.verdict import commit_verdict  # noqa: E402  (X-03 fail-closed verdict)


def tensor_digests(com, out, pair: int, prev: dict) -> int:
    import hashlib as _hl
    from collections import Counter as _C
    cur: dict[int, dict[str, str]] = {}
    zero_groups: dict[str, bool] = {}
    t0 = time.perf_counter()
    for step, (blk, used) in sorted(com._gpu_blocks.items()):
        if blk is None:
            continue
        # `blk` is the contiguous keep tensor (device | host) or the windowed `_HostWindows` (host windows): both support `blk[lo:hi]`
        def mv(lo: int, hi: int) -> memoryview:
            seg = blk[lo:hi]
            return memoryview((seg.cpu() if seg.is_cuda else seg).numpy())

        d = {}
        for m in com._layouts[step]:
            d[m.name] = _hl.blake2b(mv(m.dev_off, m.dev_off + m.nbytes), digest_size=8).hexdigest()
            if m.name.startswith(("fa2.m1/", "fa3.m1/")):
                # an FA2/FA3 row bound as all zeros = staged after it was zeroed (unsound).  Compacted rows are six planes per (layer, sequence);
                # the matReq build never writes the RP plane, so judge the (layer, sequence) group: zero only if EVERY plane is zero.
                key = f"{step}:{m.name.rsplit('/', 1)[0]}" if m.name.count("/") >= 3 else f"{step}:{m.name}"
                z = not any(mv(m.dev_off, m.dev_off + m.nbytes))
                zero_groups[key] = zero_groups.get(key, True) and z
            if m.name.startswith("fa2.m1/") and "/s" not in m.name:
                # whole-buffer FA2 row: digest the six planes separately (S, P, RP, ROW, O, FIN -- hidden_stream.StreamLayout order)
                HB, M, NB = (int(x) for x in m.name.rsplit("/g", 1)[1].split("x"))
                N = NB * 128
                off = 0
                for pl, w in zip(("S", "P", "RP", "ROW", "O", "FIN"), (N, N, N // 2, NB * 8, NB * 64, 4)):
                    nb = HB * M * w * 4
                    d[f"{m.name}/{pl}"] = _hl.blake2b(mv(m.dev_off + off, m.dev_off + off + nb), digest_size=8).hexdigest()
                    off += nb
        cur[step] = d
        dump = settings.VERITY_DUMP_STEP
        if dump and int(dump) == step:
            # raw bytes of this step's FA2 L0 row (and the whole layout) for offline byte-level diffs across pairs
            for m in com._layouts[step]:
                if m.name.startswith("fa2.m1/L0/"):
                    (out / f"dump_step{step}_pair{pair}_{m.name.replace('/', '_')}.bin").write_bytes(bytes(mv(m.dev_off, m.dev_off + m.nbytes)))
            (out / f"dump_step{step}_pair{pair}_layout.json").write_text(json.dumps([[m.name, m.dtype, list(m.shape), m.nbytes, m.dev_off] for m in com._layouts[step]]))
    (out / f"tensor_digests_pair{pair}.json").write_text(json.dumps(cur) + "\n")
    if prev:
        p0, pk = prev["pair"], prev["digests"]
        diff_steps, fam, perm_steps = [], _C(), 0
        for step, d in cur.items():
            e = pk.get(step) or {}
            names = [n for n in d if e.get(n) != d[n]]
            if names:
                diff_steps.append(step)
                for n in names:
                    fam[n.split("/", 1)[0].rsplit(".", 1)[-1] if not n.startswith("fa2.m1") else ("fa2.m1/" + n.rsplit("/", 1)[1] if n.count("/") >= 3 else "fa2.m1")] += 1
                if _C(d.values()) == _C(e.values()):
                    perm_steps += 1
        ex = ""
        if diff_steps:
            s0 = diff_steps[0]
            ex = " e.g. " + ", ".join(n for n in cur[s0] if (pk.get(s0) or {}).get(n) != cur[s0][n])[:600]
        print(f"[delta] TENSOR DIGESTS pair {pair} vs pair {p0}: {len(diff_steps)} steps differ {diff_steps[:24]}; differing tensors by family {dict(fam)}; "
              f"steps whose digest MULTISET is equal (pure permutation) {perm_steps}; {time.perf_counter() - t0:.1f}s{ex}", flush=True)
    prev["pair"], prev["digests"] = pair, cur
    zero_rows = sorted(k for k, z in zero_groups.items() if z)
    if zero_rows:
        print(f"[delta] FA2 ROWS BOUND AS ZEROS (pair {pair}): {len(zero_rows)} e.g. {zero_rows[:6]}", flush=True)
    return len(zero_rows)


def binding_record(com, rc, reqs, sampling, args, pair: int, *, served_tokens: dict | None = None, stop: dict[str, Any] | None = None,
                   finish_reasons: dict | None = None) -> dict[str, Any]:
    """`served_tokens` / `finish_reasons`: {workload request_id: ...} of THIS run (run_workload's); `stop`: stop_rule_of_record's {eos, lag}
    -- together the executed prefix of record every population consumer is placed over."""
    """The binding map (fixed at finalize, digest recorded), the collector-INDEPENDENT expected population (model structure x
    workload requests/steps/ranks), the identity-coverage verdict, and openings drawn from the EXPECTED population (incl. the last step
    and every family) verified against the original roots.  Requests: `r<i>` in workload order; B>1 needs the per-step request->row map
    from the global Program (gprog/manifest) -- without it the map binds everything to r0 and the coverage of r1.. FAILS (fail-closed)."""
    from verity_vllm.commit import binding as B, executed_prefix as EP, partition as PT
    from verity_vllm.check import commit_tokens as CT

    # the map's key `program_digest` is the Program of record = the required manifest's (the Build's digest; at B>=2
    # gprog's workload digest or the labelled components fold).  The committer's own digest seeds the step roots and is recorded beside
    # as `root_program_digest` -- it is NOT changed (known-good canary roots pin it).  `--program-digest <Build digest>` makes a manifest
    # of another Program fail closed as wrong-Program.
    root_pd = com.program_digest.hex()
    man_doc_pd, wrong_program = None, None
    if getattr(args, "required_manifest", None):
        _hdr = json.loads(open(args.required_manifest).read())
        man_doc_pd = _hdr.get("program_digest")
        if not isinstance(man_doc_pd, str) or not man_doc_pd:
            wrong_program = f"required manifest carries no string program_digest ({type(man_doc_pd).__name__}): not a Program of record"
        elif getattr(args, "program_digest", None) and str(args.program_digest) != man_doc_pd:
            wrong_program = f"wrong Program: required manifest program_digest {man_doc_pd[:16]} != Build digest {str(args.program_digest)[:16]}"
    pd = man_doc_pd if isinstance(man_doc_pd, str) and man_doc_pd else root_pd
    n_req = len(reqs)
    max_new = max(int(r.get("max_tokens") or sampling.get("max_tokens") or 1) for r in reqs)
    # a stochastic workload (declared temperature > 0, every request seeded) requires the sampler's consumed seed per step
    stochastic = (sampling.get("greedy") is not True) and float(sampling.get("temperature", 0.0) or 0.0) > 0.0 and bool(reqs) and all(r.get("seed") is not None for r in reqs)
    # a derived-S Program (its required manifest names `sampler_splits`: the Program declares a `splits` input its
    # selector reads; equivalently the workload declares `sampling.splits == "derived"`) also requires the sampler's split count per step
    _man_fams = ((_hdr.get("query") or {}).get("required_families") or []) if getattr(args, "required_manifest", None) else []
    derived_splits = stochastic and ("sampler_splits" in _man_fams or sampling.get("splits") == "derived")
    structure = B.structure_of(com, stochastic=stochastic, derived_splits=derived_splits)
    collector = {"variant": com.name, "retention": getattr(com, "retain", None), "window_mb": args.window_mb, "window_slots": getattr(args, "window_slots", None),
                 "arena_mb": args.arena_mb, "layout": "chunk-leaf-v1", **PT.partition_entry(_hdr if getattr(args, "required_manifest", None) else None)}
    # the FA2 tap's bounded-residency record (cap, the one allocation, peaks, split counts) beside the collector variant
    for _s in getattr(com, "sources", []) or []:
        if hasattr(_s, "tap_cap_bytes"):
            with contextlib.suppress(Exception):
                _s.sync_stats()
            collector["fa2_tap_bounded"] = dict(getattr(_s, "stats", {}).get("bounded_tap") or {"cap_bytes": int(_s.tap_cap_bytes)})
    # request ids are r<index>: the workload's declared `index` when present, else declaration position
    order = {str(r.get("request_id")): f"r{int(r.get('index', i))}" for i, r in enumerate(reqs)}
    arrivals = {f"r{int(r.get('index', i))}": int(r.get("arrive_step") or 0) for i, r in enumerate(reqs)}
    step_rows = getattr(com, "step_rows", None) or {}

    def _logical(rid: str) -> str:
        return B.logical_request(rid, order)

    request_rows = {int(s): [(_logical(rid), lo, hi) for rid, lo, hi in rows] for s, rows in step_rows.items()} if n_req > 1 else None
    caps = {f"r{int(r.get('index', i))}": int(r.get("max_tokens") or sampling.get("max_tokens") or 1) for i, r in enumerate(reqs)}
    # the EXECUTED PREFIX OF RECORD of every request: cap on a length stop; served + LAG on an EOS stop
    # (the driver's served tokens of THIS run; the Program's declared EOS set and LAG -- stop_rule_of_record; nothing from the trace).
    # Every consumer below counts on it: the required population, the composition, engine_schedule's expected steps.  The first B>=2
    # record with an EOS stop (OLMoE B2 i32/o32 mixedstop) had r0 committed over 12 = 11 + 1 steps and r1 over 32, and the
    # cap-shaped population required r0's 20 unexecuted steps (bound 13808/19988).  A problem (early stop without a declared EOS / LAG, a
    # non-EOS last token, served > cap) leaves the population the cap's and fails coverage by name.
    served_by_rid = None if served_tokens is None else {_logical(str(k)): [int(t) for t in v] for k, v in served_tokens.items()}
    epr = EP.executed_prefix_of_record(sorted(caps, key=lambda k: int(k[1:])), served_by_rid, caps=caps,
                                       eos_of_record=(stop or {}).get("eos"), lag=(stop or {}).get("lag"))
    n_exec = {k: int(epr["requests"][k]["executed_prefix_len"]) for k in caps}
    if epr["any_eos_stop"] or not epr["ok"]:
        print(f"[delta] execution_extent: executed prefix of record (pair {pair}): " + ", ".join(
            f"{k} served {v['served_output_len']} / executed {v['executed_prefix_len']} / cap {v['cap']}" + (" (EOS stop)" if v.get("eos_stop") else "")
            for k, v in epr["requests"].items()) + f"; LAG {epr['lag']}; EOS {epr['eos_of_record']}"
              + (f"; PROBLEMS {epr['problems'][:3]}" if epr["problems"] else ""), flush=True)
    # the population is placed on the committed-step axis by the engine's schedule of record (engine_schedule);
    # the declared arrive_steps are recorded beside it and the difference is named.  'exactly its cap' = exactly its executed prefix.
    schedule = engine_schedule(request_rows, arrivals, n_exec)
    if schedule["use"] == "observed":
        arrivals = {**arrivals, **{k: int(v) for k, v in schedule["observed_arrive_steps"].items() if k in arrivals}}
    if not schedule["all_equal_declared"]:
        print(f"[delta] engine schedule of record: arrivals observed {schedule['observed_arrive_steps']} vs declared {schedule['declared_arrive_steps']} "
              f"-> population placed by the {schedule['use'].upper()} schedule{'' if not schedule['why'] else ' (' + schedule['why'] + ')'}", flush=True)
    # the manifest's identities, placed by the same schedule, bind the committed tensors (`verity_vllm.query.manifest.bind`)
    binder = None
    if getattr(args, "required_manifest", None):
        from verity_vllm.query.manifest.bind import Binder, source_ports
        binder = Binder.of_manifest(_hdr, rank=0, ports=source_ports(getattr(com, "sources", []) or []), arrivals=arrivals)
        if binder.rebind:
            collector = dict(collector, promoted_inputs={"names": {k: list(v) for k, v in binder.rebind.items()}, "rule": "binding.PROMOTED_INPUT_RULE"})
    bmap = B.build_binding_map(com._layouts, rc.steps, rc.run_root, pd, request_id="r0", rank=0, layout_version=collector["layout"], chunk=com.chunk,
                               weight_registration=rc.weight_registration, collector=collector, request_rows=request_rows, binder=binder)
    horizon = max(arrivals[k] + n_exec[k] for k in caps)
    request_steps = EP.request_steps_of_record(epr, arrivals)
    workload = {"program_digest": pd, "requests": sorted(caps, key=lambda k: int(k[1:])), "steps": horizon, "ranks": [0], "request_steps": request_steps}
    man_hdr = None
    comp = None
    if getattr(args, "required_manifest", None):
        # the Program x query manifest is the population of record
        pop, man_hdr = B.load_required_manifest(args.required_manifest, pd, requests=workload["requests"], steps=horizon, ranks=[0], arrivals=arrivals,
                                                request_steps=request_steps)
        man_hdr["schedule"] = schedule
        man_hdr["root_program_digest"] = root_pd
        man_hdr["build_program_digest"] = getattr(args, "program_digest", None)
        # declared per-step composition vs the map's partition, over the executed steps of record
        comp = B.composition_check(man_hdr, bmap, request_steps=request_steps)
    else:
        pop = list(B.expected_population(structure, workload))
    pop_digest, pop_n = B.population_digest(pop)
    cov = B.coverage_check(pop, bmap)
    cov["schedule"] = schedule
    # the two lengths of every request beside the population they placed, and the rows-past-the-stop check
    cov["executed_prefix_of_record"] = {k: v for k, v in epr.items() if k != "rule"} | {"rule": EP.RULE, "request_steps": {k: [v[0], v[-1]] if v else [] for k, v in request_steps.items()}}
    past = EP.rows_past_the_stop(bmap, request_steps)
    cov["rows_past_the_stop"] = past
    # the prefix is accepted only LINKED to an account of the same attempt: the Match record (--match-dir: capture/tokens.json
    # + ../global_match.json) and, where a prefix is reduced and the sampled ids are committed, the Commit's token record (commit/tokens_p<pair>.json,
    # every served token opened from the run root); on a config run (no Match) the token record alone; never the committer's own count alone.
    links = CT.account_links(com, rc, bmap, epr, out_dir=args.out, pair=pair, order=order, arrivals=arrivals, match_dir=getattr(args, "match_dir", None), stop=stop or {},
                             served=served_by_rid, finish=None if finish_reasons is None else {_logical(str(k)): v for k, v in finish_reasons.items()}, past=past, program_digest=pd)
    cov["match_account_link"], cov["commit_account_link"] = links["match_account_link"], links["commit_account_link"]
    if comp is not None:
        cov["composition"] = {k: v for k, v in comp.items() if k != "problems"} | {"problems": (comp.get("problems") or [])[:3]}
    if wrong_program:                                                              # a manifest of another Program is never a population
        cov = dict(cov, ok=False, missing=list(cov.get("missing") or []), error=wrong_program)
    elif man_hdr is not None and man_hdr.get("complete") is False:                 # a manifest with unmodelled Program families is not a population
        cov = dict(cov, ok=False, missing=list(cov.get("missing") or []),
                   error="manifest incomplete: unmodelled Program families " + "; ".join(f"{k} x{v}" for k, v in sorted(man_hdr["unmodelled"].items())))
    elif not epr["ok"]:                                                            # an early stop the declared rule cannot place, by name
        cov = dict(cov, ok=False, missing=list(cov.get("missing") or []), error="executed prefix of record: " + "; ".join(epr["problems"][:3]))
    elif not past["ok"]:                                                           # committed rows of a stopped request past its prefix
        cov = dict(cov, ok=False, missing=list(cov.get("missing") or []), error=past["error"])
    elif links["error"]:                                                           # a reduced prefix linked to no account of record
        cov = dict(cov, ok=False, missing=list(cov.get("missing") or []), error=links["error"])
    elif comp is not None and not comp.get("ok"):                                  # an omitted / relabelled / duplicated request, named
        cov = dict(cov, ok=False, missing=list(cov.get("missing") or []), error=comp.get("error") or "composition check failed")
    ctx_ok, ctx_why = B.map_context_check(bmap, rc)
    picks = B.challenge_identities(pop, bmap, rc.run_root, args.openings, salt=b"population/" + str(pair).encode(), **CC.identities(rc))
    bound = [(i, e, leaf) for i, e, leaf in picks if e is not None]
    unbound = [B.identity_dict(i) for i, e, _ in picks if e is None]
    ops, ok = [], []
    po_skipped: str | None = None
    po_deferred: list[tuple[int, int]] = []
    if getattr(com, "retain", "device") == "none":
        po_skipped = "retain=none: values by replay" if bound else None
    elif bound and getattr(com, "retain_exclude", None):
        # retain=host + exclude: a population pick whose leaf lies in an excluded family's interior has no host bytes here;
        # it is handed to openings_after_release (com._oar_deferred) and opened there by REPLAY against this run's root.  The host share
        # opens now and must verify (binding_check).  Nothing is dropped: every drawn identity is opened by one of the two paths.
        want = [(e["identity"]["step"], leaf) for _, e, leaf in bound]
        held = [p for p in want if com.leaf_retained(*p)]
        po_deferred = [p for p in want if not com.leaf_retained(*p)]
        ops = com.open(held) if held else []
        ok = [com.verify(rc, o) for o in ops]
        if po_deferred:
            po_skipped = (f"retain=host excluding {sorted(com.retain_exclude)}: {len(po_deferred)} of {len(want)} population-drawn leaves lie in "
                          "excluded families -> opened by replay (openings_after_release)")
            com._oar_deferred = list(getattr(com, "_oar_deferred", []) or []) + po_deferred
    elif bound:
        ops = com.open([(e["identity"]["step"], leaf) for _, e, leaf in bound])
        ok = [com.verify(rc, o) for o in ops]
    rec = {"rules_version": B.RULES_VERSION, "map_schema": bmap["schema"], "map_digest": bmap["digest"], "map_entries_n": bmap["entries_n"], "layout": bmap["layout"],
           "program_digest": pd, "program_digest_source": ("required-manifest" if man_hdr is not None else "committer"), "root_program_digest": root_pd,
           "collector": collector, "population": {"source": (f"required-manifest ({((man_hdr or {}).get('query') or {}).get('query_id') or 'Q(P)'} over the Program, families selected by the canonical VU query)" if man_hdr else
                                                             "structure(model modules x MEMBER_RULES) x workload(requests, steps=max_new, ranks)"), "manifest": man_hdr,
                                                  "digest": pop_digest, "n": pop_n, "requests": n_req, "steps": horizon, "ranks": 1, "step_rows_recorded": len(step_rows), "step_rows_why": getattr(com, "step_rows_why", None), "step_rows_src": getattr(com, "step_rows_src", None),
                                                  "modules": len(structure["modules"]), "attention_layers": len(structure["attention_layers"]), "hidden": structure["hidden"]},
           "coverage": {k: v for k, v in cov.items() if k != "map_digest"}, "context": {"ok": ctx_ok, "why": ctx_why},
           "openings_population": {"n": len(picks), "bound": len(bound), "unbound_n": len(unbound), "unbound": unbound[:8], "opened": len(ops), "verified": sum(ok),
                                   "late_step_in_sample": max((e["identity"]["step"] for _, e, _ in bound), default=-1) == horizon - 1,
                                   "families": sorted({e["identity"]["op_path"].split(".")[-1].split("/")[0] for _, e, _ in bound}),
                                   "skipped": po_skipped, "deferred_replay_n": len(po_deferred),
                                   "deferred_positions": [list(p) for p in po_deferred], "deferred_digest": _positions_digest(po_deferred)}}
    if args.out:
        (Path(args.out) / f"binding_map_p{pair}.json").write_text(json.dumps({k: v for k, v in bmap.items() if k != "entries"} | {"entries": bmap["entries"]}, separators=(",", ":"), default=str))
        (Path(args.out) / f"structure_p{pair}.json").write_text(json.dumps(structure, indent=1, default=str))
    return rec


def _positions_digest(positions) -> str | None:
    """sha256 of the sorted, de-duplicated (step, leaf) list -- the same set recorded by two stages has the same digest."""
    pos = sorted({(int(s), int(i)) for s, i in (positions or [])})
    if not pos:
        return None
    return hashlib.sha256(json.dumps(pos, separators=(",", ":")).encode()).hexdigest()


def openings_after_release(com, rc, eng, reqs, sampling, base, args, pair: int, tokens_ref: dict | None = None) -> dict[str, Any]:
    """The P0 acceptance step.  Freeze the finished run's authentication state (roots, layouts, levels), RELEASE every raw device copy, then
    open the challenge positions (+ forced picks: a leaf of the prefill step, a leaf inside an FA2 hidden `fa2.m1/` row, a leaf of the last
    step) and verify each against the ORIGINAL roots.  Value bytes come from the host copy when the committer retained one (`--retain host`),
    otherwise from a REPLAY: the same workload is executed again under the collector (values kept on host for this run only); BI determinism
    makes the replay byte-identical, which the ORIGINAL roots check (a differing byte fails to open).  Nothing here changes what was bound."""
    import copy

    from verity_vllm.commit.challenge import challenge_positions
    from verity_vllm.commit.committer.native_host import value_family

    frozen = copy.copy(com)                     # shallow: keeps THIS run's per-step dicts (reset() rebinds the live committer's, never mutates them)
    frozen.installed = True                     # guard: never install()/uninstall() the frozen view
    freed = frozen.release_staging(levels_to_host=True)
    retained_after = frozen._retained_device_bytes()
    has_host = all(b is not None and not b.is_cuda for b, _u in frozen._gpu_blocks.values()) and bool(frozen._gpu_blocks)
    sample = list(challenge_positions(rc.run_root, [s.n_leaves for s in rc.steps], args.openings, salt=b"after-release/" + str(pair).encode(), **CC.positions(rc)))
    forced: list[tuple[int, int]] = []
    steps_sorted = sorted(frozen._layouts)
    forced_why: dict[tuple[int, int], str] = {}
    if steps_sorted:
        big = max(steps_sorted, key=lambda st: rc.steps[st].n_bytes)          # the prefill step (largest)
        metas = frozen._layouts[big]
        fa2 = [m for m in metas if "fa2.m1/" in m.name or "fa3.m1/" in m.name]     # forced pick inside a hidden row of either tap
        if fa2:
            m = fa2[len(fa2) // 2]
            forced.append((big, (m.stream_off + m.nbytes // 2) // frozen.chunk)); forced_why[forced[-1]] = "largest step: inside a hidden-m1 row"
        gem = [m for m in metas if "down_proj" in m.name]
        if gem:
            forced.append((big, gem[-1].stream_off // frozen.chunk)); forced_why[forced[-1]] = "largest step: last down_proj output"
        forced.append((big, 0)); forced_why[forced[-1]] = "largest step: first leaf (first window)"
        last = steps_sorted[-1]
        forced.append((last, rc.steps[last].n_leaves - 1)); forced_why[forced[-1]] = "last step: last leaf (the FINAL window, opened after staging release)"
        first = steps_sorted[0]
        if first != big:
            forced.append((first, 0)); forced_why[forced[-1]] = "first step: first leaf"
    forced = [f for f in forced if f not in sample]
    # the in-run challenge / population picks this run could NOT open from host (excluded families' interior; handed over
    # by the openings and binding stages via com._oar_deferred) are opened here by replay -- every drawn leaf is opened by one of the paths
    handed = list(dict.fromkeys(tuple(p) for p in (getattr(com, "_oar_deferred", None) or [])))
    deferred = [p for p in handed if p not in sample and p not in forced]    # the rest are drawn here anyway (and open by replay when not held)
    com._oar_deferred = []
    picks = sample + forced + deferred
    # which picks the frozen run can open from its OWN retained bytes: all of them (retain=host), none (retain=none), or
    # all but the excluded families' interior chunks (retain=host + VERITY_RETAIN_EXCLUDE -- the FA3 hidden plane opens by replay).
    # `--replay-openings always` replays even when every pick is held (root-parity demonstration on the small fixtures).
    held = [p for p in picks if has_host and frozen.leaf_retained(*p)]
    not_held = [p for p in picks if p not in set(held)]
    replay_all = str(getattr(args, "replay_openings", "auto")) == "always"
    need_replay = bool(not_held) or replay_all
    source = "host" if not need_replay else ("replay" if (replay_all or not held) else "host+replay")
    replay_root, replay_tokens_equal = None, None
    values_from, replay_retain = None, None
    replay_wall_s, replay_errors_n, replay_bounded = None, None, None
    if need_replay:
        saved = com.retain
        saved_levels = getattr(com, "retain_levels", True)
        com.retain_levels = True                # the replay's levels are the authentication paths of a run that kept none
        if hasattr(com, "learn_only"):
            com.learn_only = False              # a replay is a record run: a learning step over budget fails by name, never a placeholder
        if getattr(com, "windowed", False) and getattr(com, "win_bytes", 0):
            # windowed replay: keep ONLY the windows that hold a challenged/forced leaf (a chunk never straddles a window: W is a multiple
            # of the chunk) -- the whole raw stream on host is what the pod's 188 GB cgroup cannot take at B>=64
            com.retain = "sample"
            Wb = int(com.win_bytes)
            com.retain_windows = {(st, (idx * frozen.chunk) // Wb) for st, idx in picks}
            if getattr(com, "bounded", False):
                # bounded: the replay keeps only the challenged CHUNKS (256 B each) of those windows, not the windows -- its
                # authentication paths come from its own levels, so the replay's host footprint is its levels + picks x chunk
                com.retain_chunks = set(picks)
                replay_retain = f"sample ({len(picks)} chunks x {frozen.chunk} B in {len(com.retain_windows)} windows)"
            else:
                replay_retain = f"sample ({len(com.retain_windows)} windows x {Wb >> 20} MiB)"
        else:
            com.retain = "host"                 # the replay keeps its values on host so they can be read; the device still holds only arena blocks
            replay_retain = "host"
        com.pool = com.pool.fresh()             # the frozen view owns the original pinned blocks (its levels live there); the replay gets a fresh pool
        t_r = time.perf_counter()
        try:
            com.replay_row_keys = dict(frozen.row_keys) if getattr(com, "rows", False) else None   # rows-sha512: the run's salts, or no root to match
            com.reset()
            com.install()
            r2 = run_workload(eng, reqs, sampling)
            rc2 = com.finalize()
            com.uninstall()
            assert_pristine(eng, base)
        finally:
            com.retain = saved
            com.retain_levels = saved_levels
            com.retain_windows = set()
            if hasattr(com, "retain_chunks"):
                com.retain_chunks = set()
        replay_wall_s = time.perf_counter() - t_r
        replay_root = rc2.run_root.hex()
        replay_errors_n = int((rc2.extra or {}).get("errors_n") or 0)
        replay_bounded = (rc2.extra or {}).get("bounded_staging")
        replay_tokens_equal = (r2.get("tokens") == tokens_ref) if tokens_ref is not None else None    # computed, on record
        values_from = com
    t0 = time.perf_counter()
    if not need_replay:
        ops = frozen.open(picks)
        n_host, n_replay = len(ops), 0
    elif replay_all or not held:
        # every pick by replay: regenerated VALUES and regenerated AUTHENTICATION PATHS (the replay's levels) against the ORIGINAL roots
        ops = frozen.open(picks, values_from=values_from, paths_from_replay=True)
        n_host, n_replay = 0, len(ops)
    else:
        ops_h = frozen.open(held)
        ops_r = frozen.open(not_held, values_from=values_from, paths_from_replay=True)
        by_pick = {(o.step, o.index): o for o in ops_h + ops_r}
        ops = [by_pick[p] for p in picks]                                   # original pick order (sample first, forced after)
        n_host, n_replay = len(ops_h), len(ops_r)
    t_open = time.perf_counter() - t0
    t0 = time.perf_counter()
    ok = [frozen.verify(rc, o) for o in ops]
    t_ver = time.perf_counter() - t0
    fams: dict[str, int] = {}
    for o in ops:
        fam = value_family(o.identity.get("name", ""))
        fams[fam] = fams.get(fam, 0) + 1
    replay_set = set(picks) if (need_replay and (replay_all or not held)) else set(not_held)
    replay_fams: dict[str, int] = {}
    for o in ops:
        if (o.step, o.index) in replay_set:
            fam = value_family(o.identity.get("name", ""))
            replay_fams[fam] = replay_fams.get(fam, 0) + 1
    steps_opened = sorted({o.step for o in ops})
    n_sf = len(sample) + len(forced)
    # the windows the openings touched, on the record: every pick's (step, window index) -- the challenge is salted
    # across the whole run, the forced picks pin the first window of the largest step and the LAST window of the LAST step
    _Wb = int(getattr(frozen, "win_bytes", 0) or 0)
    windows_opened = sorted({(int(o.step), int((o.index * frozen.chunk) // _Wb) if _Wb else 0) for o in ops})
    res = {"n": len(ops), "verified": sum(ok), "challenge_n": len(sample), "forced_n": len(forced), "forced": forced,
           "forced_why": [{"step": int(st), "leaf": int(ix), "why": forced_why.get((st, ix), "")} for st, ix in forced],
           "windows_opened_n": len(windows_opened), "windows_opened_steps": len({w[0] for w in windows_opened}),
           "last_window_opened": bool(steps_sorted) and any((st == steps_sorted[-1] and ix == rc.steps[steps_sorted[-1]].n_leaves - 1) for st, ix in picks),
           "forced_families": [value_family(o.identity.get("name", "")) for o in ops[len(sample):n_sf]], "families": fams,
           # the in-run challenge / population picks deferred to this stage (excluded families): all opened by replay, all must verify
           "deferred_n": len(deferred), "deferred_received_n": len(handed), "deferred_verified": sum(ok[n_sf:]),
           "deferred_families": sorted({value_family(o.identity.get("name", "")) for o in ops[n_sf:]}),
           # the received POSITIONS (every one is in `picks`, opened + verified above); the verdict compares the union of the
           # in-run phases' deferred positions with these, not the counts
           "deferred_received_positions": [list(p) for p in handed], "deferred_received_digest": _positions_digest(handed),
           "steps_opened": steps_opened, "source": source, "freed": freed, "retained_device_after": retained_after,
           "replay_run_root": replay_root, "replay_root_equal": (replay_root == rc.run_root.hex()) if replay_root else None,
           "replay_tokens_equal": replay_tokens_equal,
           # what the replay depends on, named on the record: the same engine process (weights of record
           # in-process, pinned by weights_pin), the same fixture requests + sampling, BI=1; the release before it; its equality with the
           # original is the whole-run root (every regenerated byte) + the tokens
           "replay_dependencies": ({"engine": "same vLLM engine process as the original run (no reload; weights of record in-process)",
                                    "requests": len(reqs), "sampling": dict(sampling) if isinstance(sampling, dict) else str(sampling),
                                    "batch_invariant": settings.raw("VLLM_BATCH_INVARIANT"), "release_before_replay": freed,
                                    "committed_weights_root": (rc.weight_registration or {}).get("root") if isinstance(rc.weight_registration, dict) else None,
                                    "original_run_root": rc.run_root.hex(), "checked_by": ["replay_run_root == original", "replay_tokens_equal",
                                                                                            "every opening verified against the ORIGINAL root"]}
                                   if need_replay else None),
           "generate_s": t_open, "verify_s": t_ver, "retain_mode": com.retain, "replay_retain": replay_retain,
           # the replay-opened share of the picks: regenerated values + regenerated authentication paths, verified against the
           # original prechallenge root under the original layout; the replay's own wall (not C2) and any collector error it raised
           "host_n": n_host, "replay_n": n_replay, "replay_families": replay_fams, "replay_wall_s": replay_wall_s,
           "replay_paths": bool(need_replay), "replay_errors_n": replay_errors_n, "replay_bounded_staging": replay_bounded,
           "failed": [{"step": o.step, "index": o.index, "source": "replay" if (o.step, o.index) in replay_set else "host",
                       "family": value_family(o.identity.get("name", ""))} for o, k_ok in zip(ops, ok) if not k_ok][:16]}
    if values_from is not None:
        com.reset()                             # drop the replay's host copies
        com.pool = com.pool.fresh()             # ... and FREE their pinned blocks now: `reset()` only returns them to the pool's free list, and
    #                                             the frozen view + its pool sit in a reference cycle (committer <-> C++ collector) that the
    #                                             cyclic GC frees on ITS schedule -- at B=32 the replay's 99 GB survived into the next pair's
    #                                             replay -> 198 GB > the pod's 188 GB cgroup -> SIGKILL.  Collect explicitly.
    del frozen, values_from
    import gc

    gc.collect()
    try:
        with open("/proc/self/status") as f:
            _st = {l.split(":")[0]: int(l.split()[1]) for l in f if l.startswith(("VmRSS:", "VmHWM:"))}
        res["host_rss_gb_after"] = round(_st["VmRSS"] / 1e6, 2)
        # the process's kernel high-water mark AFTER the replay -- the in-process peak the bounded HOST BOUND must
        # cover (the replay phase is where a bounded arm peaks); sampler-independent (try 4's external sampler followed another process)
        res["host_rss_hwm_gb"] = round(_st["VmHWM"] / 1e6, 2)
    except (OSError, StopIteration, KeyError):
        pass
    return res


# hot-engine reuse: `verity_vllm.pipeline.hot serve` sets _HOT = {"jobs": n, "restart_reason": why} and calls
# main(argv) once per Commit job.  The first job builds the engine and parks it here (with the exact kwargs set and the pristine baseline
# taken BEFORE any committer); later jobs reuse it when `engine_kwargs_for` yields the same kwargs / env flags / checkpoint (else the job is
# refused with HOT-KEY-MISMATCH and the worker restarts, reason recorded).  Every committer built in a job is registered in _HOT["coms"] so
# the worker can close it (uninstall + worker stop + pool shutdown) and re-check the engine is pristine after the job.  _HOT is None in a
# plain (cold) process: nothing below changes.  A grouped Commit (`commit_group`) sets it too, with ``group`` (`group_reuse`) and ``last``.
_HOT: dict[str, Any] | None = None

HOT_UNSAFE = ("compiled_taps", "arms", "only_arm", "late_read", "value_check", "arena_unchecked", "profile_hooks", "moe_source")


def hot_unsafe_reasons(args: Options) -> list[str]:
    """Why a job must run in a fresh (cold) process: flags that mutate module/process state beyond the committer's lifetime."""
    why = []
    if args.compiled_taps != "off":
        why.append(f"--compiled-taps {args.compiled_taps} (Inductor hooks / runner patches are process-wide)")
    if args.arms:
        why.append("--arms (multi-committer arms)")
    if args.only_arm == "control" or (args.only_arm and args.compiled_taps != "off"):
        why.append(f"--only-arm {args.only_arm} (paired-process protocol)")
    if args.late_read:
        why.append("--late-read (mutates native_collect.GATHER_FLUSH_LEAVES)")
    if args.value_check:
        why.append("--value-check (reference taps)")
    if args.arena_mb and args.arena_unchecked:
        why.append("--arena-unchecked (negative test)")
    if args.moe_source == "on":
        why.append("--moe-source on")
    if str(args.committer).startswith("plugin:"):
        why.append("plugin committer")
    if settings.VERITY_FAULT:
        why.append(f"VERITY_FAULT={settings.raw('VERITY_FAULT')!r} (deliberate fault: negative test)")
    if settings.raw("VERITY_SWITCH"):
        why.append("VERITY_SWITCH (interpreter switch interval is process-wide)")
    if args.remote_verifier:
        why.append("--remote-verifier (the store binder goes on the engine at its build)")
    return why


def resolve_staging(args: Options) -> Options:
    """The collector's staging defaults made explicit before anything reads them (the replay planner, the collector record, the
    replay's ring): an omitted --window-mb is `native_collect.window_mb_default` when every committer kind is the windowed worker
    (per-step blocks otherwise), an omitted --window-slots the collector's ring, an omitted --bounded-staging is on whenever staging is
    windowed."""
    from verity_vllm.commit.committer.native_collect import WINDOW_SLOTS, window_mb_default

    window, slots = args.window_mb, args.window_slots
    if window is None:
        kinds = args.arms.split(",") if args.arms else [args.committer]
        window = window_mb_default(all(k == "native_collect_v2b" for k in kinds), args.retain)
    if slots is None and window > 0:
        slots = WINDOW_SLOTS
    bounded = args.bounded_staging if args.bounded_staging is not None else window > 0
    return dataclasses.replace(args, window_mb=window, window_slots=slots, bounded_staging=bounded)


def staging_environ(args: Options) -> None:
    """The resolved staging options into the environment the committers read (`resolve_staging`), and the late-read
    negative's flush points."""
    settings.put("VERITY_RETAIN", args.retain)
    settings.put("VERITY_WINDOW_MB", str(args.window_mb))
    if args.window_slots is not None:
        settings.put("VERITY_WINDOW_SLOTS", str(args.window_slots))
    settings.put("VERITY_STAGING_BOUNDED", "1" if args.bounded_staging else "0")
    if args.bounded_staging:
        settings.put("VERITY_LEARN_HOST_BUDGET_MB", str(int(args.learn_budget_mb)))
    if args.retain_exclude:
        if not args.bounded_staging or args.retain != "host" or not args.open_after_release:
            raise SystemExit("--retain-exclude needs --bounded-staging --retain host --open-after-release (the excluded families open by replay)")
        settings.put("VERITY_RETAIN_EXCLUDE", args.retain_exclude)
    if args.footprint:
        settings.put("VERITY_FOOTPRINT", args.footprint)
        Path(args.footprint).parent.mkdir(parents=True, exist_ok=True)
    if args.late_read:
        # deliberate late read: the P0 flush after qkv_proj is deferred to the next legal point (past RoPE's in-place q/k rewrite)
        import verity_vllm.commit.committer.native_collect as _nc

        _nc.GATHER_FLUSH_LEAVES = frozenset(_nc.GATHER_FLUSH_LEAVES - {args.late_read})
        print(f"[delta] LATE-READ NEGATIVE TEST: gather flush points {sorted(_nc.GATHER_FLUSH_LEAVES)} (dropped {args.late_read!r}); roots must NOT match arm 2", flush=True)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Options:
    workload: str = option("--workload", type=workload_arg, default="workloads/workload_cov_b0_b1_c256.json")
    checkpoints: str = option("--checkpoints", default="manifests/checkpoints.json")
    case: str | None = option("--case", default=None)
    out: str = option("--out")
    pairs: int = option("--pairs", type=int, default=5)
    committer: str = option("--committer", default="native_collect_v2b", help="native_collect_v2b (default: the windowed worker, which commits rows-sha512) | hooks_only (an ablation, not a commitment) | plugin:<module>:<factory>")
    commit_scheme: str = option("--commit-scheme", default=SR.NAME, choices=(SR.NAME,), help="the commitment scheme, rows-sha512 (hm96-sha512 rows in frame-v3-sha512 trees, 64-byte roots), the windowed worker's (native_collect_v2b with --window-mb > 0)")
    chunk: int = option("--chunk", type=int, default=65536, help="hooks_only: the leaf, in bytes of the packed step stream, it reports (it hashes nothing); a rows-sha512 row is step_rows.ROW_BYTES")
    arena_unchecked: bool = option("--arena-unchecked", default=False, action="store_true", help="NEGATIVE TEST ONLY: skip the X-02 arm-time bound check for an explicit --arena-mb so a " "too-small block reaches the collector (its refusal must then yield Commit FAIL under X-03)")
    arena_mb: int = option("--arena-mb", type=int, default=0, help="device staging block (MB); default 0 = X-02 auto: sized at arm time from the workload envelope " "(B, max prompt, max_num_batched_tokens, max_model_len; + FA2 hidden stream); 2 blocks preallocated, grown on demand (counted). " "An explicit value smaller than the step bound is refused before the run. Memory headroom for the arena comes from " "--engine-arg gpu_memory_utilization=<x> (0.5-0.6 on 48 GB for <=2B models at 1024-token prompts; 0.75 on 80 GB)")
    weights_hash: str = option("--weights-hash", choices=("host", "device", "device-checked"), env="VERITY_WEIGHTS_HASH", default="host", help="where the weights registration hashes: host (default; Python on the CPU), device (the same leaves and per-tensor trees " "on the GPU: rows-sha512's hiding rows by native_rows.cu, verity_vllm.commit.committer.weights_rows), device-checked (both, each tensor's device root compared with its host " "root and a mismatch refused; validation runs); a leased Commit's row sets device unless the item sets it " "(pipeline.gpu_lease).  Recorded in versions.json weights.hash")
    hash_threads: int = option("--hash-threads", env="HASH_THREADS", type=int, default=2, help="hashlib worker threads (pod CPU tune: 2 threads x 4 KB = 2.2 GB/s; 8 threads x 4 KB = 0.4 GB/s, GIL thrash)")
    pinned_populate_threads: int | None = option("--pinned-populate-threads", env="VERITY_PINNED_POPULATE_THREADS", type=int, default=None, help="helper threads that fault a fresh exact pinned block's pages in (MADV_POPULATE_WRITE) before the caller registers it (default: commit.committer.pinned_pool.POPULATE_THREADS; recorded as pinned_populate_threads); 0 = the registration faults them in")
    openings: int = option("--openings", type=int, default=64)
    max_num_seqs: int = option("--max-num-seqs", type=int, default=8)
    engine_arg: list[str] | None = option("--engine-arg", action="append", default=None)
    target: str | None = option("--target", default=None, help="TargetProfile JSON overrides / path (default: the workload's `target` block over this device)")
    protocols_config: str | None = option("--protocols-config", default=None, help="the protocols' settings, a JSON file (default: the workload's `protocols_config`); never hashed")
    warmup: int = option("--warmup", type=int, default=2)
    label: str = option("--label", default="")
    hidden_so: str | None = option("--hidden-so", default=None, help="CMT-2 M1 tap FA2 extension (.so): bind the FA2 hidden stream (rows R5-R8) into the same run root")
    hidden_ns: str | None = option("--hidden-ns", default=None, help="torch.ops namespace of the tapped .so (default verity_fa2_matAll; verity_fa2_matReq for the src-59 build)")
    tap_cap_mb: int | None = option("--tap-cap-mb", type=int, default=None, help="cap (MiB) on the FA2 M1 tap's device buffer: a multi-sequence launch whose padded tap buffer exceeds it is " "split into sequence groups tapped one after another into ONE bounded buffer (names/bytes/roots unchanged); default 2048 " "(VERITY_FA2_TAP_CAP_MB); 0 = the unbounded per-geometry buffers (B64 llama 1024: 24.53 GiB, which OOMs)")
    tap_sources: TAPS.TapSources = group(TAPS.TapSources)
    value_check: bool = option("--value-check", default=False, action="store_true", help="one extra instrumented run with reference taps + re-execution (value_check.json)")
    arms: str | None = option("--arms", default=None, help="comma-separated committer kinds run as separate treatment arms in one process, interleaved with the control (order rotates per round)")
    profile_hooks: bool = option("--profile-hooks", default=False, action="store_true", help="native_collect: time the hook bodies (perf_counter per call) -- profiling run, not a headline timing")
    expect_root: str | None = option("--expect-root", default=None, help="fail if the instrumented run root (hex) differs from this reference root (the arm2/arm3 equality gate)")
    late_read: str | None = option("--late-read", default=None, help="NEGATIVE TEST: drop this leaf (e.g. qkv_proj) from the gather arm's flush points so the " "staged values are read after the in-place overwriter -> the run root MUST differ from arm 2 (roots_equal_across_arms=false)")
    flip_weight: str | None = option("--flip-weight", default=None, help="NEGATIVE TEST ONLY (a substituted weight input): " "<served param name>:<flat index>[:<value>] -- ONE element of that live parameter is changed in the instrumented process BEFORE " "register_weights, so the committer serves a model that is not the checkpoint the Program names; the weights pin MUST FAIL naming " "that field (served weight(s) differ from the checkpoint of record) and the Commit is FAIL rc 3.  Needs the pin armed " "(--weights-of-record / --checkpoint-dir + the Program).  Recorded in verdict.json negatives.flip_weight; never a row-procedure flag")
    window_mb: int | None = option("--window-mb", type=int, default=None, help="windowed staging -- lay the packed step stream over a ring of device windows of this size " "(MiB, multiple of 1) and absorb/release each window as soon as it is filled (sets VERITY_WINDOW_MB); omitted = the collector's " "(native_collect.window_mb_default: 256 on native_collect_v2b unless --retain device, else per-step blocks); 0 = per-step blocks")
    window_slots: int | None = option("--window-slots", type=int, default=None, help="P0: number of ring slots (default 8; raised automatically so the largest tensor fits)")
    retain: str = option("--retain", default="none", choices=["device", "host", "none"], help="retention policy for absorbed raw staging (sets VERITY_RETAIN): device = a compact device copy per step " "(the collector footprint that OOMs Llama 1024/128 at B=2); host = copy to host memory, arena block recycled; none (default) = " "keep only the authentication state, openings by deterministic replay (--open-after-release)")
    footprint: str | None = option("--footprint", default=None, help="write the per-step device-memory timeline (JSONL) to this path (sets VERITY_FOOTPRINT)")
    bounded_staging: bool | None = option("--bounded-staging", default=None, action="boolean_optional", help="bounded host staging (sets VERITY_STAGING_BOUNDED; windowed path only; omitted = on whenever " "staging is windowed): a learning step's transient " "host copies are capped by --learn-budget-mb and returned right after the re-hash (over budget: the warm-up pass learns the " "plan and DISCARDS the windows, a record run FAILS by name); hit-plan steps take no hedge copy (a plan mismatch is a named " "FAIL); roots identical to the unbounded path (same headers, leaf order, padding). --no-bounded-staging: the hedge copies")
    learn_budget_mb: int = option("--learn-budget-mb", type=int, default=4096, help="bounded staging: transient pinned-host budget for a learning step (MiB; default 4096)")
    retain_exclude: str | None = option("--retain-exclude", default=None, help="with --bounded-staging --retain host: comma-separated value families (e.g. fa3_hidden_m1) whose chunk-aligned " "interior is NOT copied to host (sets VERITY_RETAIN_EXCLUDE): those leaves open by REPLAY against the original root in " "openings_after_release (needs --open-after-release); everything else opens from host as before")
    host_admit: float | None = option("--host-admit", type=float, default=None, metavar="MARGIN", help="with --bounded-staging: REFUSE (rc 3, named) before any run when the host peak bound exceeds (1 - MARGIN) x the " "cgroup memory limit (e.g. 0.15); without it the bound is only printed / recorded (host_admission_<kind>.json)")
    replay_openings: str = option("--replay-openings", default="auto", choices=["auto", "always"], help="openings-after-release: 'auto' replays only when a pick is not retained on host; 'always' replays every time and opens " "EVERY pick from the replay (regenerated values + regenerated authentication paths vs the ORIGINAL roots)")
    open_after_release: bool = option("--open-after-release", default=True, action="boolean_optional", help="(default on) after each instrumented run, release every raw device copy, then open the 64 challenge positions " "(+ forced picks from the prefill step and the FA2 hidden rows) from the host copy or from a REPLAY run, and verify them " "against the ORIGINAL run's roots (recorded as commit.openings_after_release; a failure is a Commit FAIL); " "--no-open-after-release opens in the run only (not under --retain none)")
    required_classes: str | None = option("--required-classes", default=None, help="comma-separated class ids that MUST be declared with tensors > 0 by every " f"instrumented run (default {','.join(REQUIRED_CLASSES_DEFAULT)}; the FA2 hidden class is required because the harness has no other " "evidence the attention producer fired). Override only when the model's producers genuinely differ; the set is recorded in verdict.json")
    match_dir: str | None = option("--match-dir", "--oracle", default=None, help="the Match stage's run dir of THIS attempt (<row>/match: capture/snapshots.json + values/, instances.jsonl, " "capture/log.jsonl.gz): live C2 -- every bound identity's committed bytes are compared with the Match record's value for that " "identity (oracle-compare); any mismatch = COMMIT FAIL naming the identity; steps the Match did not snapshot are reported partial")
    program_dir: str | None = option("--program-dir", default=None, help="B=1: the Build's request Program dir (instances.json.gz + result.json) -- enables the C2 OF RECORD: " "sampled exact VU replay of the committed bytes through the registered relation evaluators (commit.sampled_replay) and the " "exhaustive ordered boundary linkage (commit.boundary_linkage); a mismatch / unlinked boundary = COMMIT FAIL naming the VU")
    programs_root: str | None = option("--programs-root", default=None, help="B>=2: the directory holding the per-request Programs (build_request*/ or *LP<n>_T<t>; build-global's resolution) for the sampled replay")
    replay_per_stratum: int = option("--replay-per-stratum", type=int, default=2, help="sampled replay: VUs per stratum (spec x request x phase); default 2")
    replay_cache: bool = option("--replay-cache", default=False, action="store_true", help="reuse an earlier pair's C2 sampled-replay + boundary-linkage record within this Commit stage when the FULL key " "(run root, manifest digest, seed, binding map, weights root, sample parameters, code) is identical; the copy carries reused_from and " "evaluated_here=0 (default off)")
    replay_bundle_cap_gb: float = option("--replay-bundle-cap-gb", type=float, default=300.0, help="--replay-deferred: the largest replay bundle this Commit writes (GB); a larger one is not written and the C2 replay runs in this process")
    replay_bundle_overlap: int = option("--replay-bundle-overlap", type=int, default=1, choices=(0, 1), help="--replay-deferred full: 1 writes the bundle's store from the moment the run root is fixed, beside the binding and manifest coverage (replay_bundle.start; the same files and digests); 0 writes it at the C2 site, after them")
    replay_deferred: str | None = option("--replay-deferred", nargs="?", const="full", default=None, choices=("full", "slim"), help="the C2 sampled replay + boundary linkage do not run in this process: each instrumented run writes a replay bundle (verity_vllm.pipeline.replay_bundle, <out>/replay_bundle_p<pair>/) and its verdict stays PENDING until the replay runs on CPU from the bundle; exit 0 when every check this process runs passes.  `slim`: the replay's reads are planned here and only the members they open are sealed, with their Merkle paths (the planning adds to the GPU hold)")
    replay_seed: int | None = option("--replay-seed", type=int, default=None, help="sampled replay: a TARGETED sampler seed (negatives / reproduction); default = derived from the run root after commitment (recorded with its source)")
    replay_challenge: str | None = option("--replay-challenge", default=None, help="sampled replay: a VERIFIER-ISSUED post-commitment challenge (hex/string) -- the seed becomes sha256(challenge || run_root) so an " "adaptive prover cannot grind the sample by recommitting; default = the run root alone (honest-prover measurement)")
    challenger_gate: str | None = option("--challenger-gate", default=None, metavar="CMD", help="the auditor host's challenger gate, its command as one string (e.g. 'ssh -i KEY -o IdentitiesOnly=yes research@HOST'): the Commit reserves one session before its engine and opens it on its first run's root of record, and each run's openings and served C2 replay challenge are its coins, opened after that run's root (verity_vllm.pipeline.commit_coins); without it each run's coins are this process's own, a measurement that credits nothing")
    challenger_peer: str | None = option("--challenger-peer", default=None, metavar="NAME", help="with --challenger-gate: this machine's name in the auditor host's peers, where its challenger reaches the Commit (its certifier coin link)")
    weights_of_record: str | None = option("--weights-of-record", default=None, metavar="JSON", help="the weights of record: per-field digests of the served weights COMPOSED FROM THE PINNED CHECKPOINT SHARDS " "(verity_vllm.check.weights_of_record, written beside the Build's Program); every live parameter the committer registers is compared " "with it by name -- a served model that differs from the checkpoint in one element is a named COMMIT FAIL")
    checkpoint_dir: str | None = option("--checkpoint-dir", default=None, help="derive the weights of record in-process from this pinned checkpoint snapshot + the Program of record (--program-dir / " "--programs-root) when --weights-of-record is not given; the record is written to <out>/weights_of_record.json")
    replay_k: int | None = option("--replay-k", type=int, default=None, help="the config run's sampled replay: k units in all (drawn by --replay-draw) in place of the row's per-stratum sample")
    replay_draw: str = option("--replay-draw", default="uniform", choices=("uniform", "family"), help="with --replay-k: uniform over every unit, or stratified by family (one stratum per family, round-robin, at least one each)")
    replay_max: int = option("--replay-max", type=int, default=2000, help="sampled replay: cap on the stratified sample per pair (never below one VU per stratum; recorded)")
    replay_workers: int | None = option("--replay-workers", type=int, default=None, help="sampled replay: evaluator threads (default min(16, cpus/2); recorded)")
    vu_export_dir: str | None = option("--vu-export-dir", env="VERITY_VU_EXPORT_DIR", default=None, help="opt-in, not on the record: after pair 0's replay PASSes, export a stratified sample of its VUs as benchmark instance sets into this directory (verity_vllm.pipeline.vu_export; limits.json / provenance.json there are read)")
    serving_rows: str | None = option("--serving-rows", default=None, metavar="REQUEST_JSON", help="opt-in, not on the record: after pair 0's roots are final and before its seed, commit every unit of the verifier's partition in M0's serving row format (verity_vllm.pipeline.serving_rows; the row stage writes the request: dir, partition, run, committer); with --replay-deferred (not slim) at the C2 site, before the bundle is sealed (c2_replay.serving_rows)")
    remote_verifier: str | None = option("--remote-verifier", default=None, metavar="HOST:PORT", help="the remote bandwidth session's verifier: the store binder goes on the engine at its build, and the Commit proves its store to it in lockstep with its first instrumented run, a pass an engine step of that run (verity_vllm.protocols.remote_bandwidth.remote.CommitSession); a rejected session fails the Commit")
    remote_plan: str = option("--remote-plan", default="serve", help="the session's mode: serve, or servedrop:K (negative: the first K blocks appended withheld)")
    remote_kv_pages: int = option("--remote-kv-pages", type=int, default=32, help="the KV pool's pages under the session (num_gpu_blocks_override; outside the registered store); 0: the engine's own pool, at the row's gpu_memory_utilization")
    remote_filler_gib: float = option("--remote-filler-gib", type=float, default=0.0, help="GiB of zeros on the engine's GPU registered after the build as the session's filler, beside the weights (the KV pool is never registered: verity_vllm.protocols.remote_bandwidth.binder.Binder.register_filler)")
    remote_weight_blocks: int | None = option("--remote-weight-blocks", type=int, default=None, help="register weight tensors in order up to this many 4 KiB blocks (default: all)")
    remote_kv_lib: str | None = option("--remote-kv-lib", default=None, help="kv.cu built at the codec's field and mask: the store's encode on the GPU (default: the CPU reference)")
    remote_session_lib: str | None = option("--remote-session-lib", default=None, help="session.cu's build (benchmarks/remote_session): the session reads the store where the engine holds it, its weights and filler encoded in place by --remote-kv-lib, with no copy (verity_vllm.protocols.remote_bandwidth.in_place)")
    remote_step_ms: float = option("--remote-step-ms", type=float, default=0.0, help="the engine step a paced session's pass waits on, in ms: added to the calibration's pass times so the verifier's allowance covers it (a verifier at --paced 1:T_MS:DRAWS:WINDOWS runs each pass's step inside its coin's window); 0 for a session in stated windows")
    memory_map: str | None = option("--memory-map", default=None, metavar="REQUEST_JSON", help="opt-in, not on the record: commit each KV layer's `<layer>/slot_mapping` (acquire.sources.kv_slots_source) and, after pair 0's roots are final, write memory-map/v1 of the run's KV to the request's path, its blocks from each request's Program of record at the pages those slots name, its digests from the committed K and V words (verity_vllm.pipeline.memory_map; the request: path, registration, store layout); the prover's witness, never published")
    plan: str | None = option("--plan", default=None, metavar="JSON", help="the Commit's plan, derived before the GPU (verity_vllm.pipeline.commit_plan; the row's `plan` stage): the producer facts are taken from it when its key (Program digest, manifest, workload, Programs, code identity, vLLM version) equals this Commit's; else derived here and recorded as plan_recomputed with the reason (verdict.json plan)")
    call_boundary_plan: str | None = option("--call-boundary-plan", default=None, metavar="JSON", help="the call-boundary source's host plan, derived at the Build's call-boundary check (verity_vllm.pipeline.boundary_plan): taken when its key (as --plan's) equals this Commit's; else derived here and recorded as plan_recomputed with the reason (verdict.json call_boundary_plan)")
    replay_target: list[str] | None = option("--replay-target", action="append", default=None, metavar="OP_PATH[@STEP]", help="sampled replay: add EVERY VU of a module (at a request-local step) to the sample, outside the cap -- the targeted negatives of record " "(a single-identity fault at a mid-phase step); repeatable; recorded under sample.targeted")
    program_digest: str | None = option("--program-digest", default=None, help="the Build's Program digest of record (B=1: build_request's program_digest; B>=2: gprog's workload digest): the " "required manifest's program_digest must equal it or the Commit fails closed as wrong-Program; also the binding map's key digest")
    required_manifest: str | None = option("--required-manifest", default=None, help="the Program/query-derived required-value manifest (verity_vllm.pipeline.cli build ...): every " "identity it lists (step, invocation, op_path, output_member, element range, request, rank) must be covered by a committed " "tensor of every instrumented run (coverage_check); replaces the counts+classes source, which is the labelled fallback")
    tensor_digests: bool = option("--tensor-digests", default=False, action="store_true", help="diagnostic: per-tensor digests of the retained host bytes per instrumented run, diffed across pairs")
    moe_source: str = option("--moe-source", choices=["auto", "on", "off"], default="auto", help="[MOE-03] attach the fused-MoE block source (router_w/ids, up, act, down, sum per MoE layer per step; class moe_block_stream, " "added to the required classes): auto = when the model has MoE-runner modules (the default), on = refuse if it has none, off = never")
    compiled_taps: str = option("--compiled-taps", choices=["off", "on", "kernels", "served"], default="off", help="served = the "
                                "served-compiled row: every value of every engine step at vLLM's defaults (ServedCompiledSource), the served Build "
                                "(--required-manifest, verity-vllm served-match) checked word for word on the committed values (served_check) with a "
                                "seed drawn after commitment, the Build rederived in this process (build_rebuilt.json); control: --compiled-taps off "
                                "--only-arm control (verity_vllm/ops/served_commit.sh).  kernels = P_compiled: acquire every non-weight buffer of every Inductor kernel launch through Inductor's " "wrapper-level hook seam (kernels/fusion untouched; captured pieces copy into preallocated slots at graph capture, drained per step; " "verity_vllm/acquire/sources/compiled_kernel_source.py); requires --required-manifest from compiled_manifest.py. " "on = [withdrawn as the workload's path] record ops compiled INTO the graphs (graph-replay-safe " "staging ring; verity_vllm/acquire/sources/compiled_source.py) for Execution=compiled rows (enforce_eager=false). " "The op nodes live in the graphs for the whole process, so the paired control must be a separate process with " "`--compiled-taps off --only-arm control` and the same engine args (verity_vllm/ops/compiled_commit.sh)")
    compiled_ring_rows: int = option("--compiled-ring-rows", type=int, default=0, help="rows per staging slot; 0 = auto: the engine's largest CUDA graph capture size (larger, uncaptured steps use the direct path)")
    only_arm: str | None = option("--only-arm", choices=["control", "instrumented"], default=None, help="run only this arm (`--pairs` runs of it); the paired summary is produced by compiled_commit.sh from both processes")


def main(args: Options, argv: list[str] | None = None) -> None:
    global TL
    _rs = RS.session_of(args)
    if settings.VERITY_HOT_SUBMIT and _HOT is None:
        # route this invocation through the pod's hot Commit worker (verity_vllm/pipeline/hot.py) -- lets an unchanged harness
        # (the e2e negatives in tests/test_commit_fail_closed.py, ad-hoc timing pairs) exercise engine reuse; the worker refuses HOT-UNSAFE
        # jobs and the client falls back to a cold process with the reason on record, so nothing runs that would not run cold.
        from verity_vllm.pipeline import hot as _HC
        sys.exit(_HC.submit(sys.argv[2:] if argv is None else list(argv), root=Path(settings.HOT_ROOT),
                            idle_s=settings.HOT_IDLE, timeout_s=settings.HOT_TIMEOUT))
    TL = Timeline.from_env("commit_delta", hot=_HOT is not None)      # hot worker: the job's VERITY_TIMELINE (per row attempt), not the worker's
    if (tracedump := settings.VERITY_TRACEDUMP) is not None:
        # diagnostics: dump every thread's Python stack to stderr every N seconds (hang isolation on pods without ptrace)
        import faulthandler
        faulthandler.dump_traceback_later(tracedump, repeat=True)
    # source identity precheck BEFORE anything else runs: verity.ml.circuits, verity_vllm
    # (and the telemetry package) must all resolve from THIS tree (the one commit_delta.py lives in; $VERITY_TREE must agree).
    # A mismatch exits 3 with PRECHECK_FAIL_SOURCE_IDENTITY on stderr -- no engine, no inference; the record is written beside the out dir.
    from verity_vllm.pipeline import source_identity as _sid

    _sid_out = (Path(args.out) / "source_identity.json") if getattr(args, "out", None) else None
    try:
        source_identity_doc = _sid.precheck("commit", out=_sid_out)
    except _sid.SourceIdentityError as e:
        print(str(e), file=sys.stderr, flush=True)
        raise SystemExit(3) from None
    print(f"[delta] source identity OK: " + " ".join(f"{k}={(v.get('git_sha') or '-')[:9]}" for k, v in source_identity_doc["packages"].items())
          + f" tree={source_identity_doc['requested_root']}", flush=True)
    settings.put("VERITY_WEIGHTS_HASH", args.weights_hash)       # read by native_host.register_weights (this process and the forked replay)
    if _HOT is not None:
        _unsafe = hot_unsafe_reasons(args)
        if _unsafe:
            # the worker records this as the restart reason and the client re-runs the job in a fresh process; the engine stays untouched
            raise SystemExit("HOT-UNSAFE: " + "; ".join(_unsafe))
    args = resolve_staging(args)
    _planner = RB.prepare(args)
    compiled_src = None
    fault = settings.VERITY_FAULT
    compiled_kernels = None
    if args.compiled_taps == "kernels":
        if args.only_arm == "control":
            raise SystemExit("--compiled-taps kernels is the treatment process; the control runs with --compiled-taps off --only-arm control")
        if args.arms:
            raise SystemExit("--compiled-taps kernels: single committer only (no --arms)")
        if not getattr(args, "required_manifest", None):
            raise SystemExit("--compiled-taps kernels needs --required-manifest (verity_vllm.acquire.compiled build ... from the control process's Inductor dir)")
        from verity_vllm.engine import engine_profile as _prof
        from verity_vllm.acquire.sources.compiled_kernel_source import CompiledKernelSource

        _prof.apply_env()
        compiled_kernels = CompiledKernelSource(fault=fault if fault.startswith("compiled_") else None)
        compiled_kernels.install_hooks()                       # before vLLM compiles: the generated wrappers carry the hook lines
        print(f"[delta] COMPILED KERNEL TAPS (P_compiled): Inductor wrapper hooks {compiled_kernels.hooks_info}, fault {compiled_kernels.fault!r}", flush=True)
    if args.compiled_taps == "on":
        if args.only_arm == "control":
            raise SystemExit("--compiled-taps on is the treatment process; the control process runs with --compiled-taps off --only-arm control")
        if args.arms:
            raise SystemExit("--compiled-taps on: single committer only (no --arms)")
        from verity_vllm.engine import engine_profile as _prof
        from verity_vllm.acquire.sources.compiled_source import CompiledGraphSource

        _prof.apply_env()                                    # before any vllm import (build_engine does the same, idempotent)
        compiled_src = CompiledGraphSource(rows=args.compiled_ring_rows)
        compiled_src.prepare()
        print(f"[delta] COMPILED TAPS: record ops will be compiled into the graphs (ring rows {args.compiled_ring_rows}, fault {compiled_src.fault!r}, "
              f"runner patches {compiled_src.stats.get('patched_runners')}); vLLM compile cache disabled", flush=True)
    served = None
    if args.compiled_taps == "served":
        if args.only_arm == "control" or args.arms or not args.required_manifest:
            raise SystemExit("--compiled-taps served is the treatment process (one committer) and needs --required-manifest, the served Build "
                             "of record (verity-vllm served-match); the control runs with --compiled-taps off --only-arm control")
        from verity_vllm.pipeline.served_match import ServedCommit

        served = ServedCommit(json.load(open(args.required_manifest)), fault if fault.startswith("compiled_") else None)   # hooks before vLLM compiles
        print(f"[delta] SERVED TAPS: every value of every engine step at vLLM's defaults; fault {served.src.fault!r}", flush=True)
    required_classes = tuple(c for c in (args.required_classes.split(",") if args.required_classes is not None else REQUIRED_CLASSES_DEFAULT) if c)
    required_classes_cli = args.required_classes is not None
    moe_attached = False
    required_manifest = None
    if args.required_manifest:
        from verity_vllm.commit.binding import require_build_program_digest
        require_build_program_digest(args.required_manifest, args.program_digest)          # named refusal, before any engine
        from verity_vllm.query.manifest.bind import coverage_check as manifest_coverage_check, source_ports
        from verity_vllm.commit.partition import require_partition
        required_manifest = require_partition(json.load(open(args.required_manifest)), args.programs_root, args.program_dir)
        print(f"[delta] required-value manifest {args.required_manifest}: {required_manifest['populations']['identities']} identities {required_manifest['populations']['by_family']} digest {required_manifest['manifest_digest'][:16]} (Program {str(required_manifest['program_digest'])[:16]})", flush=True)
        if args.compiled_taps == "kernels":
            compiled_kernels.set_role_maps(required_manifest)          # BEFORE the engine compiles: capture-time copies + eager acquisitions bind the manifest's roles
            print(f"[delta] compiled kernel taps: argument roles from the manifest's role_maps {compiled_kernels.hooks_info.get('role_maps')} "
                  f"({'signature-resolved per launch site' if compiled_kernels.role_maps else 'ABSENT -> name-keyed table'})", flush=True)
    staging_environ(args)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if compiled_src is not None:
        compiled_src.use_plan(required_manifest, out / "compiled_source_plan")     # [v2 acquire] the engine build's wrap_model wraps the plan's boundaries
    # the collector's fail-fast watchdog writes this marker (then _exit(3)) when a "collector dead" run does not fail
    # closed within VERITY_COLLECT_DEAD_GRACE_S; row_pod reads rc 3 as COMMIT FAIL, and a present marker names the hang in the row evidence.
    settings.setdefault("VERITY_COLLECT_WATCHDOG_MARKER", str(out / "COLLECTOR_DEAD_HUNG.json"))
    wl = va.load_workload(args.workload)
    engine_args = {**(wl.get("engine_args_required") or {}), **va.parse_engine_args(args.engine_arg)}
    TL.event("startup.imports", "end", s=round(time.time() - _T_PROCESS0, 3), label=args.label, workload=str(args.workload),
             hot_job=(_HOT or {}).get("jobs"))
    # `--arms k1,k2,...` runs several committers as separate treatment arms in ONE process against the same control, interleaved
    # (control, k1, k2, ...) with the order rotated every round so every arm takes every position; single-committer mode is unchanged
    # (a control-only process has no committer at all: nothing installed, nothing compiled in)
    kinds = [] if args.only_arm == "control" else [k for k in (args.arms.split(",") if args.arms else [args.committer]) if k]
    multi = bool(args.arms)
    coins = CC.of_options(args, args.pairs * len(kinds))       # the auditor host's session, reserved before any root (or None: a measurement)
    coms: dict[str, Any] = {}
    # the Programs' producer facts for a v2-query manifest (which carries none), derived on the CPU before the engine (and a leased Commit's
    # GPU) and the committer: the members the producer module's body hands down are marked promoted here, so the acquisition observes them
    # at their first consumer's input, not the module's forward output (oracle_compare.promote_handed_down); form (B) reads them every pair
    oracle_producers: tuple[dict | None, dict] | None = None
    plan_rec, (cb_plan, cb_rec) = None, BP.use(args, required_manifest if kinds else None, tl=TL)
    if required_manifest is not None and kinds:
        from verity_vllm.check import oracle_compare as _OCP
        oracle_producers, plan_rec = CP.producer_facts(args, required_manifest, wl["requests"], wl["sampling"], tl=TL)
        print(f"[delta] form (B) producer facts from the Programs of record: {oracle_producers[1]}", flush=True)
        if oracle_producers[0]:
            _hd = _OCP.promote_handed_down(required_manifest, oracle_producers[0])
            oracle_producers[1]["handed_down"] = _hd
            if _hd["rows"]:
                print(f"[delta] v2 manifest: {_hd['rows']} row(s) of {len(_hd['members'])} member(s) handed down by their module's body (every "
                      f"reader inside its subtree) marked promoted -> observed at the first consumer's input: "
                      f"{ {k: v[:2] for k, v in sorted(_hd['members'].items())[:6]} }", flush=True)
    GL.acquire(TL)            # a leased Commit's GPU from here to its last run's GL.release (pipeline.gpu_lease); the target below reads the device
    _settings = PO.own_auditor(args.protocols_config or wl.get("protocols_config"), va.target_of_workload(wl, args.target))   # PoUW's coins: this process's challenger (the Commit is its run's own auditor)
    _ekw = dict(checkpoints=args.checkpoints, max_num_seqs=args.max_num_seqs, case=va.case_of(wl, args.case), engine_args=engine_args, target=PO.commit_guard(va.target_of_workload(wl, args.target), hot=_HOT is not None, settings=_settings, commit_scheme=args.commit_scheme), execution=va.execution_of_workload(wl))
    _rs is not None and _rs.before_build(_ekw)      # the hot worker refuses --remote-verifier: the session's engine is built here
    _live = (_HOT or {}).get("eng")
    if _live is not None:
        # hot worker with a live engine: reuse iff THIS job would build exactly the same engine (kwargs set + env flags +
        # checkpoint record, `engine_kwargs_for` = the same code path build_engine takes).  Anything else is a genuine restart trigger.
        _kw, _envf, _cp = va.engine_kwargs_for(_ekw["checkpoints"], max_num_seqs=_ekw["max_num_seqs"], case=_ekw["case"], engine_args=_ekw["engine_args"],
                                               target=_ekw["target"], execution=_ekw["execution"])
        _want = {"kwargs": _kw, "env_flags": _envf, "checkpoint": _cp}
        _have = {"kwargs": dict(_live.kwargs), "env_flags": dict(_live.env_flags), "checkpoint": dict(_live.cp)}
        if json.dumps(_want, sort_keys=True, default=str) != json.dumps(_have, sort_keys=True, default=str):
            _diff = sorted(k for k in set(_want["kwargs"]) | set(_have["kwargs"]) if _want["kwargs"].get(k) != _have["kwargs"].get(k))
            raise SystemExit(f"HOT-KEY-MISMATCH: live engine differs from the job's engine (kwargs keys {_diff or 'same'}; env_flags "
                             f"{_want['env_flags'] == _have['env_flags']}; checkpoint {_want['checkpoint'] == _have['checkpoint']})")
        eng = _live
        TL.event("engine.reuse", "mark", model=eng.cp["repo"], enforce_eager=eng.kwargs.get("enforce_eager"), engine_reused=True,
                 hot_job=_HOT["jobs"], build_s_saved=round(eng.build_s, 3), engine_age_s=round(time.time() - _HOT.get("t_built", time.time()), 1))
        print(f"[delta] HOT: engine reused (job {_HOT['jobs']} of this worker; saves the {eng.build_s:.1f}s build; engine age "
              f"{time.time() - _HOT.get('t_built', time.time()):.0f}s; {eng.cp['repo']} max_num_seqs {eng.kwargs.get('max_num_seqs')})", flush=True)
    else:
        TL.start("engine.build", case=_ekw["case"], max_num_seqs=args.max_num_seqs, execution=_ekw["execution"])
        eng = va.build_engine(_ekw["checkpoints"], max_num_seqs=_ekw["max_num_seqs"], case=_ekw["case"], engine_args=_ekw["engine_args"],
                              target=_ekw["target"], execution=_ekw["execution"])     # the row's execution decides enforce_eager (header: versions.json)
        TL.end("engine.build", build_s=round(eng.build_s, 3), model=eng.cp["repo"], enforce_eager=eng.kwargs.get("enforce_eager"),
               engine_reused=False, restart_reason=((_HOT or {}).get("restart_reason") or settings.raw("VERITY_HOT_RESTART_REASON")
                                                    or "fresh process (one commit_delta per Commit stage)"))
        if _HOT is not None:
            gc.collect()
            torch.cuda.synchronize()
            _HOT.update(eng=eng, t_built=time.time(), base0=baseline_state(eng), cuda0=cuda_census())   # pristine baseline BEFORE any committer of any job
    # the attention implementation the engine executes (FA2 / FA3) -- selects the hidden class the verdict requires and the tap class
    fa_version = va.observed_flash_attn_version(eng.model)
    if not required_classes_cli and served is not None:
        # the served values: the step's ids, the sampled ids, the logits and every other value of the pieces (compiled_kernel_buffers);
        # FA3's outputs are values of the pieces, not a tap's hidden stream
        required_classes, hidden_family = HS.with_hidden_class(required_classes_default(fa_version), None) + ("compiled_kernel_buffers",), None
        print(f"[delta] required classes (served): {required_classes}", flush=True)
    elif not required_classes_cli:
        required_classes, hidden_family = classes_of_record(required_manifest, eng.model, fa_version)
        if compiled_kernels is not None:
            # the execution's leaf class is P_compiled's kernel buffers, not the eager module outputs (never fired: not wrapped)
            required_classes = tuple("compiled_kernel_buffers" if c == "instance_outputs_native" else c for c in required_classes)
            print(f"[delta] required classes (P_compiled): {required_classes}", flush=True)
    print(f"[delta] engine attention: vllm_flash_attn_version={fa_version}; required classes {','.join(required_classes)}", flush=True)
    print(f"[delta] engine {'REUSED (hot; originally built' if _live is not None else 'built'} in {eng.build_s:.1f}s{')' if _live is not None else ''} "
          f"(case {eng.case}, {eng.cp['repo']}), env {eng.env_flags}", flush=True)
    _rs is not None and _rs.check(eng.llm, eng.runner)
    versions = va.versions_doc(eng)
    versions["served"] = SV.require(eng.llm, wl, _ekw["target"], out)
    flip_weight_rec: dict | None = None                    # --flip-weight NEGATIVE TEST record (set once, before register_weights)
    versions["git_head"] = git_head()
    versions["source_identity"] = source_identity_doc      # resolved_path / repo_root / git_sha per package + sys.path[0]
    versions["argv"] = list(argv) if argv is not None else sys.argv
    # hot/cold process + engine reused or restarted (+ reason) in the record itself, not only on the timeline
    versions["hot"] = {"hot_process": _HOT is not None, "engine_reused": _live is not None,
                       "restart_reason": None if _live is not None else ((_HOT or {}).get("restart_reason") or settings.raw("VERITY_HOT_RESTART_REASON")
                                                                          or "fresh process (one commit_delta per Commit stage)"),
                       "worker_job": (_HOT or {}).get("jobs"), "worker_pid": os.getpid() if _HOT is not None else None,
                       "engine_built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime((_HOT or {}).get("t_built") or time.time())),
                       # process state that decides the speed of the CPU-bound phases (register_weights, forked replay): a hot worker measured
                       # ~+5 s on each vs a cold process -- these numbers are for locating the cause, not a verdict field
                       "process_state": _process_state()}
    versions["engine_kwargs"] = {k: v for k, v in eng.kwargs.items() if k not in ("model", "tokenizer")}
    # the RESOLVED async-scheduling mode (scheduler_config after the build), same fact the RunHeader attests
    versions["engine_kwargs"]["async_scheduling"] = va.resolved_async_scheduling(eng.llm)
    versions["enforce_eager"] = eng.kwargs.get("enforce_eager")
    # what is actually compiled / graphed in THIS process (CompilationMode, CUDAGraphMode, capture sizes, splitting ops):
    # recorded for every attempt, control and treatment alike -- the paired processes must agree (compiled_commit.sh checks)
    from verity_vllm.acquire.sources.compiled_source import compile_settings as _compile_settings

    versions["compilation"] = _compile_settings(eng.llm)
    versions["compilation"]["compiled_taps"] = args.compiled_taps
    versions["only_arm"] = args.only_arm
    versions["weights_hash"] = args.weights_hash        # which hashing produced the weights root (runs.jsonl weight_registration.hash = the effective one)
    print(f"[delta] compilation: {versions['compilation']}", flush=True)
    if versions["compilation"].get("enforce_eager") is False:
        # IDENTIFY the fused-interior implementation even though it is not compared launch-by-launch: the
        # Inductor output code of every compiled piece (wrapper module = one piece: its launch program `call()` and the Triton kernel sources
        # it embeds) is digested from THIS process's Inductor cache dir (kernel_identity.collect, W1c's rule: kernel_set_sha256 = sha256 of the
        # sorted {(kernel name, source sha256)}); control and treatment record it, compiled_merge compares the two kernel sets.  Requires a
        # per-process TORCHINDUCTOR_CACHE_DIR and VLLM_DISABLE_COMPILE_CACHE=1 (compiled_commit.sh sets both), else the dir is shared/empty.
        gk = va.generated_kernels_of_this_process()                    # same helper as the record path (RunHeader.compilation.generated_kernels)
        versions["compilation"]["generated_kernels"] = gk
        if gk.get("error"):
            print(f"[delta] generated kernels: {gk['error']}", flush=True)
        else:
            (out / "inductor_kernels.json").write_text(json.dumps(gk, indent=1, default=str) + "\n")
            print(f"[delta] generated kernels (Inductor output code, per-process dir {gk['per_process_cache_dir']}): pieces {gk['wrappers']} "
                  f"distinct kernel sources {gk['distinct_kernel_sources']} kernel_set {gk['kernel_set_sha256'][:16]} call_set {gk['call_set_sha256'][:16]}", flush=True)
    if compiled_kernels is not None:
        compiled_kernels.note_engine(eng.llm)
        versions["compiled_source"] = compiled_kernels.summary()
        print(f"[delta] compiled kernel source: capture sizes {sorted(compiled_kernels.capture_sizes)}, slots {len(compiled_kernels.slots)} "
              f"({compiled_kernels.stats['slot_bytes'] >> 20} MiB) written at graph capture: copies {compiled_kernels.stats['capture_copies']}, "
              f"unknown-kernel args {compiled_kernels.stats['unknown_kernel_args']}, errors {compiled_kernels.errors}", flush=True)
        if compiled_kernels.errors or not compiled_kernels.slots:
            raise SystemExit(f"--compiled-taps kernels: no captured slots / errors {compiled_kernels.errors}")
    if served is not None:
        versions.update(served.engine(eng))
        print(f"[delta] served source: slots {len(served.src.slots)} plans {len(served.src.plans)} written at graph capture, "
              f"{len(served.src.modules)} identities every step", flush=True)
    if compiled_src is not None:
        if compiled_src.model is None:
            raise SystemExit(f"--compiled-taps on: the model was not wrapped before compilation (runner patch missed; errors {compiled_src.errors})")
        if compiled_src.errors:
            raise SystemExit(f"--compiled-taps on: errors while wrapping/renumbering: {compiled_src.errors}")
        versions["compiled_source"] = compiled_src.summary()
        print(f"[delta] compiled source: {len(compiled_src.wrapped)} modules wrapped before compile; renumber {compiled_src.renumber}; slots {len(compiled_src.slots)} "
              f"({compiled_src.stats['slot_bytes'] >> 20} MiB) written during warm-up/capture: capture calls {compiled_src.stats['capture_calls']}, eager calls "
              f"{compiled_src.stats['eager_calls']}, captured keys {len(compiled_src.captured_keys)}", flush=True)
    (out / "versions.json").write_text(json.dumps(versions, indent=1, default=str) + "\n")

    base = baseline_state(eng)
    reqs, sampling = wl["requests"], wl["sampling"]

    # warm-up (control)
    for i in range(args.warmup):
        with TL.span(f"prep.warmup_control{i}"):
            r = run_workload(eng, va.throwaway_requests(eng, reqs, seed=100 + i), sampling)
        print(f"[delta] warm-up control {i}: {r['wall_s']:.3f}s {r['iterations']} it", flush=True)

    run_id = f"{Path(args.workload).stem}"
    if switch := settings.raw("VERITY_SWITCH"):          # GIL-interference experiment: interpreter switch interval (default 5 ms)
        import sys as _sys

        _sys.setswitchinterval(float(switch))
        print(f"[delta] sys.setswitchinterval({switch})", flush=True)
    for kind in kinds:
        TL.start("prep.committer_setup", kind=kind)
        com = make_committer(kind, eng, args, run_id)
        if _HOT is not None:
            _HOT.setdefault("coms", []).append(GR.adopt(_HOT, com))          # closed after the job (hot worker, grouped Commit: on the group's pinned pool)
        if compiled_kernels is not None:
            if not hasattr(com, "_nc") or not hasattr(com, "attach_source"):
                raise SystemExit(f"--compiled-taps kernels needs the native_collect committer (got {kind})")
            compiled_kernels.attach(com, required_manifest)
            com.program_digest = SR.program_field(required_manifest["program_digest"])   # P_compiled's digest binds the roots (manifest = P_compiled x query)
            print(f"[delta] compiled kernel source attached -> committer {com.name}; program digest (P_compiled) {required_manifest['program_digest'][:16]}", flush=True)
        if served is not None:
            served.attach(com)                               # the committer wraps nothing: the source acquires every value of the Build
            com.program_digest = SR.program_field(required_manifest["program_digest"])   # the served Build's digest binds the roots
        if compiled_src is not None:
            if not hasattr(com, "_nc") or not hasattr(com, "attach_source"):
                raise SystemExit(f"--compiled-taps on needs the native_collect committer (got {kind})")
            compiled_src.attach(com)                         # sets com.external_acquisition: the committer does not wrap the compiled forwards
            print(f"[delta] compiled source attached -> committer {com.name}", flush=True)
        _att = attach_of_record(com, eng, args, required_manifest, out, kind=kind, required_classes=required_classes,
                                required_classes_cli=required_classes_cli, hidden_family=hidden_family if not required_classes_cli else None,
                                fa_version=fa_version, cb_plan=cb_plan, served=served)
        required_classes, moe_attached, _hidden_required = _att.required_classes, _att.moe_attached or moe_attached, _att.hidden_required
        if hasattr(com, "size_arena") and getattr(com, "arena", None) is not None and not (args.arena_unchecked and args.arena_mb):
            # X-02: size the device staging block from the workload envelope (refuses here, with needed vs available bytes, if it cannot fit)
            envelope = {"batch": args.max_num_seqs, "max_prompt": max(len(r["prompt_token_ids"]) for r in reqs),
                        "max_new": max(int(r.get("max_tokens") or sampling.get("max_tokens") or 1) for r in reqs),
                        "max_num_batched_tokens": eng.kwargs.get("max_num_batched_tokens"), "max_model_len": eng.kwargs.get("max_model_len")}
            asz = com.size_arena(envelope, hidden_m1=bool(args.hidden_so) and _hidden_required is not False)
            print(f"[delta] arena sizing (X-02, {kind}): decode block {asz['decode_block_bytes'] >> 20} MiB x{asz['blocks_preallocated']} + prefill block "
                  f"{asz['prefill_block_bytes'] >> 20} MiB "
                  f"(step bound {asz['step_bound_bytes'] >> 20} MiB: prefill {asz['prefill_tokens']} tok -> {asz['prefill_step_bytes'] >> 20} MiB "
                  f"[hidden {asz['prefill_hidden_bytes'] >> 20} MiB], decode -> {asz['decode_step_bytes'] >> 20} MiB; per-token {asz['per_token_bytes']} B), "
                  f"retained bound {asz['retained_bytes_bound'] >> 20} MiB; device free {asz['device_free_bytes'] >> 20} MiB; "
                  f"gpu_memory_utilization <= {asz['gpu_memory_utilization_max']}", flush=True)
            (out / f"arena_sizing_{kind}.json").write_text(json.dumps(asz, indent=1, default=str) + "\n")
            # `--retain host` pins the whole run's raw bytes (+ ~1/4 for levels) in HOST memory; inside a container the
            # bound is the cgroup limit, not `free`. Over it the kernel OOM-killer SIGKILLs the process mid-warm-up with no Python error
            # (B=32 Llama 1024/128 on a 125 GB cp-dA container). Say so up front; the sound alternative is `--retain none` + replay openings.
            host_cap = None; host_cap_src = None
            for cg in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
                try:
                    v = Path(cg).read_text().strip()
                    if v.isdigit() and int(v) < (1 << 60):
                        host_cap = int(v); host_cap_src = cg
                        asz["host_memory_cap_source"] = cg
                        break
                except OSError:
                    pass
            if host_cap is None:
                # no cgroup limit visible (bare host / 'max'): the machine's MemTotal keeps the admission a bound instead of
                # admitting unconditionally; the source is named on the record either way
                try:
                    for line in Path("/proc/meminfo").read_text().splitlines():
                        if line.startswith("MemTotal:"):
                            host_cap = int(line.split()[1]) * 1024; host_cap_src = "/proc/meminfo MemTotal (no cgroup limit visible)"
                            break
                except OSError:
                    pass
            asz["host_memory_cap_bytes"] = host_cap
            asz["host_memory_cap_source"] = host_cap_src
            # the record NAMES the admission that governs this run (a field, not an
            # inference from which file is present): a run must satisfy the admission whose domain is its retention mode -- the bounded
            # HOST BOUND (below) for --bounded-staging, the host-retention envelope for the unbounded --retain host path; the other
            # stays on the record beside it and does not refuse alone.  Rides on arena_sizing_<kind>.json and host_admission_<kind>.json.
            asz["admission_of_record"] = {
                "governing": ("bounded_host_admission" if getattr(com, "bounded", False)
                              else ("f_da_15_host_retention_envelope" if getattr(com, "retain", None) == "host" else None)),
                "retain": getattr(com, "retain", None), "bounded_staging": bool(getattr(com, "bounded", False)),
                "rule": "the admission whose domain is the run's retention mode governs; the other-mode admission is on the record and does not refuse alone"}
            if host_cap is not None and getattr(com, "retain", None) == "host" and not getattr(com, "bounded", False):
                need = int(asz["retained_bytes_bound"] * 1.25)
                if need > host_cap:
                    print(f"[delta] WARNING host retention bound {need >> 20} MiB (raw x1.25 for levels) exceeds the container memory limit "
                          f"{host_cap >> 20} MiB: expect a silent OOM-kill (rc 137) during the instrumented run; use --retain none (openings by "
                          f"replay) or a larger container", flush=True)
            if getattr(com, "bounded", False):
                # the HOST peak bound of a bounded run, every term named (bytes):
                #   engine       the process RSS now (weights are on device; this is vLLM + torch + the tokenizer + the pinned pool so far)
                #   levels       the run's tree levels on pinned host: 32 B per 256 B chunk per level -> <= raw / 4 (kept for openings)
                #   raw          retain=host: the raw bytes kept on host = the bound minus the excluded families' prefill hidden bytes
                #                (decode-step hidden rows are counted as retained: conservative); retain=none|sample: 0
                #   learn        the learning-step transient budget (given back after each re-hash; at most one step at a time)
                #   replay       openings-after-release: a second run's levels (raw / 4) + the challenged windows on host (<= picks x W)
                #   ring         the windowed device ring is DEVICE memory (not counted); the pinned pool rounds slices up to 64 MiB blocks
                # The bound is compared with the cgroup limit; `--host-admit <margin>` REFUSES the run (rc 3, named) when it does not fit.
                _rss_now = int((rss_kb()[0] or 0) * 1024)
                # TERM BY TERM against the B1 1024/128 measurements on an H100 (runs.jsonl of the record and
                # bounded-x arms): the stream has two components the sizing models with very different accuracy --
                #   hidden (fa3_hidden_m1, the FA3 slab padded to the step's longest query): sized 19,636,365,312 B, measured (excluded_bytes)
                #            19,635,194,880 B -> exact to 1.2 MB; factor 1.05
                #   other  (every other family, per-token widths): sized ~3.43 GB, measured (host kept under exclude) 5,440,526,844 B -> x1.59;
                #            factor 2.0 (the per-token widths of the declaration under-size these planes; bounded, not explained)
                # A blanket x1.15 on the total held on B1 only because hidden is 78 % of the stream; on B8 (hidden 92 %) it under-sizes the
                # kept raw by ~7 GiB while over-sizing the levels. Env overrides for a re-derivation, never for a launch.
                _f_hidden = settings.VERITY_ADMIT_HIDDEN_FACTOR
                _f_other = settings.VERITY_ADMIT_OTHER_FACTOR
                # the run's stream under the workload's DECLARED schedule (per step; the hidden family's share), not the envelope's
                # any-packing worst case (`retained_bytes_bound` = ceil(B x max_prompt / T_prefill) prefills of the worst step: 3x for the
                # B8 1024/128 row whose 8 prompts prefill in ONE step) -- both are on the record
                from verity_vllm.commit.committer.native_host import schedule_stream_bound
                _ssb = schedule_stream_bound(asz, reqs, sampling)
                asz["schedule_stream_bound"] = _ssb
                _excl = set(getattr(com, "retain_exclude", ()) or ())
                from verity_vllm.commit.committer.native_host import bounded_host_bound
                # the C2 sampled replay's Programs of record become Python objects in THIS process (B8: 70,300 MiB,
                # the peak of attempt 1); forecast them BEFORE the run from the instance files the replay will load (streamed count, ~17 s
                # for the B8 row's 4.9 GB of json; nothing is built), so the bound printed here names them as a term of their own
                _c2f = c2_programs_forecast(getattr(args, "programs_root", None), getattr(args, "program_dir", None)) if args.replay_per_stratum else \
                    {"bytes": 0, "rows": 0, "json_bytes": 0, "files": [], "why": "no sampled replay (--replay-per-stratum 0)"}
                asz["c2_programs_forecast"] = _c2f
                print(f"[delta] C2 PROGRAMS FORECAST: {_c2f['rows']:,} instance rows in {len(_c2f['files'])} files, json {_c2f['json_bytes'] >> 20} MiB -> "
                      f"{_c2f['bytes'] >> 20} MiB of Python objects (x{_c2f.get('factor')}, {_c2f.get('calibration')})"
                      f"{(' -- ' + str(_c2f.get('why'))) if _c2f.get('why') else ''}", flush=True)
                _admit = bounded_host_bound(_ssb, engine_rss_bytes=_rss_now, retain=com.retain, retain_exclude=_excl,
                                            c2_programs_bytes=int(_c2f["bytes"]), c2_freed_before_replay=True,       # freed after the replay + linkage records (this file)
                                            fork_children_bytes=(FORK_CHILDREN_BYTES if args.replay_per_stratum else 0),
                                            c2_residual_bytes=(int(_c2f["bytes"] * C2_RESIDUAL_FRACTION) if args.replay_per_stratum else 0),
                                            pool_carry_bytes=(POOL_CARRY_BYTES if int(getattr(args, "pairs", 1) or 1) >= 2 else 0),
                                            pool_exact=(getattr(getattr(com, "pool", None), "exact", False) is True), replay_cache=bool(getattr(args, "replay_cache", False)),
                                            learn_budget_bytes=int(getattr(com, "learn_budget_bytes", 0) or 0), openings=int(args.openings),
                                            chunk_bytes=int(com.chunk), window_bytes=int(getattr(com, "win_bytes", 0) or 0),
                                            bounded=bool(getattr(com, "bounded", False)), open_after_release=bool(args.open_after_release),
                                            host_cap_bytes=host_cap, hidden_factor=_f_hidden, other_factor=_f_other,
                                            # engine growth after admission (torch host allocator, vLLM bookkeeping, CUDA context, the warm-ups'
                                            # buffers) + pool rounding: B1 timelines RSS 2,218 -> 5,757 MiB before the first pool block (+3,539
                                            # MiB), rounding <= 190 MiB -> 6 GiB (the earlier 4 GiB left 516 MiB on bounded-none try 3)
                                            slack_bytes=int(settings.VERITY_ADMIT_SLACK_GIB * (1 << 30)),
                                            # a 2nd+ pair's frozen pool beside the pinned cache of the first (B1 PAIRS=3 curve, FP8-STAGING §3.4)
                                            pairs=int(getattr(args, "pairs", 1) or 1))
                _admit.update({"envelope_retained_bytes_bound": int(asz["retained_bytes_bound"]), "schedule_run_bytes": int(_ssb["run_bytes"]),
                               "schedule_hidden_bytes": int(_ssb["hidden_bytes"]), "schedule_largest_step": _ssb.get("largest_step"),
                               "prefill_tokens": asz.get("prefill_tokens"), "per_token_bytes": asz.get("per_token_bytes"),
                               # the DECLARED schedule the bound came from: a hit-plan deviation from it fails by name on the run
                               "declared_schedule": {k: _ssb.get(k) for k in ("model", "requests", "steps", "horizon", "prefill_steps",
                                                                               "hidden_prefill_bytes", "decode_step_bytes_max")},
                               "cgroup_path": asz.get("host_memory_cap_source"),
                               # this admission GOVERNS the run it is printed before (the bound printed
                               # BEFORE the run is the bound; commit_delta compares its own VmHWM with it at the end -> verdict.host_peak)
                               "governs_this_run": bool(getattr(com, "bounded", False)), "admission_of_record": asz.get("admission_of_record"),
                               # not a literal: the admission's own print time; verdict.host_peak compares it with the first
                               # pair's start (first_run_started_utc) and records printed_before_run from the two timestamps
                               "admission_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "admission_monotonic_s": round(time.monotonic(), 3)})
                _bound, _raw_bound, _levels, _raw_kept, _hidden_prefill, _learn, _replay, _slack, _pairs_x = (
                    _admit["host_peak_bound_bytes"], _admit["raw_bound_bytes"], _admit["levels_bytes"], _admit["raw_kept_bytes"],
                    _admit["excluded_prefill_hidden_bytes"], _admit["learn_budget_bytes"], _admit["replay_bytes"], _admit["slack_bytes"],
                    _admit["pairs_extra_bytes"])
                _hidden_sized, _other_sized = _admit["hidden_sized_bytes"], _admit["other_sized_bytes"]
                asz["bounded_host_admission"] = _admit
                com.host_admission = _admit          # rides on the record: extra.bounded_staging.host_admission
                print(f"[delta] stream of record (declared schedule, {_ssb['steps']} steps, {_ssb['requests']} requests): run {int(_ssb['run_bytes']) >> 20} MiB "
                      f"(hidden {int(_ssb['hidden_bytes']) >> 20} MiB), largest step {(_ssb.get('largest_step') or {}).get('bytes', 0) >> 20} MiB "
                      f"({(_ssb.get('largest_step') or {}).get('tokens')} tok); envelope any-packing bound {int(asz['retained_bytes_bound']) >> 20} MiB; "
                      f"raw bound {_raw_bound >> 20} MiB = hidden {_hidden_sized >> 20} x {_f_hidden} + other {_other_sized >> 20} x {_f_other} "
                      f"(factors measured on B1)", flush=True)
                _ph = _admit.get("phases_bytes") or {}
                print(f"[delta] bounded staging HOST BOUND ({kind}): {_bound >> 20} MiB = max(phases) + pairs {_pairs_x >> 20} ({_admit['pairs']} pairs) + slack {_slack >> 20}; "
                      f"phases: run {_ph.get('run', 0) >> 20} = engine {_rss_now >> 20} + levels {_levels >> 20} + raw kept {_raw_kept >> 20} (bound {_raw_bound >> 20}, "
                      f"excluded hidden {_hidden_prefill >> 20}) + learn {_learn >> 20} + pool_rounding {_admit.get('pool_rounding_bytes', 0) >> 20} "
                      f"(pool {'exact' if _admit.get('pool_exact') else 'torch-pow2'}) | c2 {_ph.get('c2_sampled_replay', 0) >> 20} = run + c2_programs "
                      f"{_admit.get('c2_programs_bytes', 0) >> 20} + fork_children {_admit.get('fork_children_bytes', 0) >> 20} (cgroup scope) | "
                      f"replay {_ph.get('openings_after_release', 0) >> 20} = run + replay {_replay >> 20}"
                      f"{' + c2_residual ' + str(_admit.get('c2_residual_bytes', 0) >> 20) if _admit.get('c2_freed_before_replay', True) else ' + c2_programs'}"
                      f"{(' | later_pair_replay ' + str(_ph.get('later_pair_replay', 0) >> 20) + ' = replay + pool_carry ' + str(_admit.get('pool_carry_bytes', 0) >> 20)) if 'later_pair_replay' in _ph else ''}"
                      f" -> peak phase {_admit.get('peak_phase')} MiB; "
                      f"cgroup {((host_cap or 0) >> 20) or None} MiB; margin "
                      f"{(_admit['margin_bytes'] or 0) >> 20 if host_cap else None} MiB ({round(100 * (_admit['margin_frac'] or 0), 1) if host_cap else None} %)", flush=True)
                # the dominant contributor by name, on the print and on any refusal -- never merely "predicted OOM"
                print(f"[delta] bounded staging DOMINANT CONTRIBUTOR ({kind}): {(_admit.get('dominant_contributor') or {}).get('text')}", flush=True)
                (out / f"host_admission_{kind}.json").write_text(json.dumps(_admit, indent=1, default=str) + "\n")
                (out / f"arena_sizing_{kind}.json").write_text(json.dumps(asz, indent=1, default=str) + "\n")   # + cap, admission_of_record, bound
                if getattr(args, "host_admit", None) is not None:
                    if host_cap is None:
                        print("[delta] REFUSE (--host-admit): no cgroup memory limit readable to admit against", flush=True)
                        sys.exit(3)
                    if _bound > host_cap * (1.0 - float(args.host_admit)):
                        print(f"[delta] REFUSE (--host-admit {args.host_admit}): host peak bound {_bound >> 20} MiB > {(1.0 - float(args.host_admit)):.2f} x cgroup "
                              f"{host_cap >> 20} MiB -- predicted OOM, not launched; dominant contributor: "
                              f"{(_admit.get('dominant_contributor') or {}).get('text')}", flush=True)
                        sys.exit(3)
            # step: arriving prompts, live decode rows, the M1 stream geometry the record shows; + the Program's LAG rows), x1.25 for levels,
            # against the cgroup.  A Commit predicted to exceed it is NOT launched: refused here by name before the engine runs a step
            # (rc 3 = COMMIT FAIL in row_pod; the record carries the numbers).  VERITY_ADMIT_OVER_BOUND=1 overrides -- named on the record.
            try:
                from verity_vllm.pipeline import admission_bound as ADM
                adm = ADM.admission(asz, reqs, host_cap, retain=str(getattr(com, "retain", "host")),
                                    max_num_batched_tokens=eng.kwargs.get("max_num_batched_tokens"), lag=admission_lag(wl),
                                    envelope_bound_bytes=int(asz["retained_bytes_bound"]), resident=ADM.committer_resident(wl, host_cap, pairs=args.pairs, replay_workers=args.replay_workers))
                adm["override"] = settings.VERITY_ADMIT_OVER_BOUND
                adm["host_cap_source"] = host_cap_src
                adm["domain"] = "--retain host (the whole run's raw bytes pinned on host); a non-host retention is recorded, never refused"
                # a --bounded-staging run pins the NON-excluded families sparsely and opens the excluded
                # one by replay: its governing admission is the bounded HOST BOUND printed above (asz["admission_of_record"]); this envelope
                # admission counts the whole run's raw bytes (the excluded family too) and is RECORDED beside it, never refuses alone
                adm["governs_this_run"] = not bool(getattr(com, "bounded", False))
                asz["admission"] = adm
                (out / f"admission_{kind}.json").write_text(json.dumps(adm, indent=1, default=str) + "\n")
                _s = adm["scheduled"]
                print(f"[delta] HOST ADMISSION ({kind}): scheduled raw {adm['scheduled_raw_bytes'] >> 20} MiB over {_s['steps']} steps "
                      f"({_s['prompt_tokens']} prompt tokens + {_s['decode_rows']} decode rows, lag {_s['lag']}; peak step {_s['peak_step']} = "
                      f"{_s['peak_step_bytes'] >> 20} MiB) -> predicted host {adm['predicted_host_bytes'] >> 20} MiB vs limit "
                      f"{(host_cap or 0) >> 20 if host_cap else 'none'} MiB: {'ADMITTED' if adm['admitted'] else 'REFUSED'} -- {adm['why']}", flush=True)
                if not adm["admitted"] and not adm["override"] and adm["governs_this_run"]:
                    print(f"[delta] REFUSE: host admission -- {adm['why']} (VERITY_ADMIT_OVER_BOUND=1 launches anyway, named on the record)", flush=True)
                    sys.exit(3)
                if not adm["admitted"] and not adm["governs_this_run"]:
                    print(f"[delta] NOTE host-retention envelope admission not met ({adm['why'][:160]}) -- recorded, not governing: this is a --bounded-staging run "
                          f"(the bounded HOST BOUND above governs)", flush=True)
                elif not adm["admitted"]:
                    print("[delta] WARNING host admission OVERRIDDEN by VERITY_ADMIT_OVER_BOUND=1 (named on the record): the run may be OOM-killed", flush=True)
            except SystemExit:
                raise
            except Exception as _ae:  # noqa: BLE001  -- an unreadable schedule is named, never a silent skip of the admission
                # an uncomputed admission is not an admission: a --retain host run is REFUSED by name (rc 3) unless
                # VERITY_ADMIT_OVER_BOUND=1; a non-host retention (nothing pinned for the run) records the error and proceeds
                _host = str(getattr(com, "retain", "host")) == "host" and not bool(getattr(com, "bounded", False))   # a bounded run: the HOST BOUND above governs
                _ovr = settings.VERITY_ADMIT_OVER_BOUND
                asz["admission"] = {"admitted": (None if not _host else False), "override": _ovr, "host_cap_source": host_cap_src,
                                    "why": f"admission not computed: {_ae!r}"[:300] + (" -- refused: an uncomputed admission is not an admission (--retain host)" if _host and not _ovr else "")}
                (out / f"admission_{kind}.json").write_text(json.dumps(asz["admission"], indent=1, default=str) + "\n")
                if _host and not _ovr:
                    print(f"[delta] REFUSE: host admission not computed on a --retain host run: {_ae!r} (VERITY_ADMIT_OVER_BOUND=1 launches anyway, named on the record)", flush=True)
                    sys.exit(3)
                print(f"[delta] WARNING host admission not computed: {_ae!r}" + (" (OVERRIDDEN by VERITY_ADMIT_OVER_BOUND=1, named on the record)" if _host else " (retain != host: nothing pinned for the run)"), flush=True)
            if getattr(com, "windowed", False):
                # windowed staging ring instead of per-step blocks; device staging = slots x window regardless of the step size
                wi = com.enable_windows(asz)
                print(f"[delta] windowed staging (P0): {wi['slots']} x {wi['window_bytes'] >> 20} MiB ring + scratch (largest tensor bound "
                      f"{wi['largest_tensor_bound'] >> 20} MiB); layout {wi['layout']}; in-place hashing: {wi['in_place_hashing']}", flush=True)
                (out / f"window_ring_{kind}.json").write_text(json.dumps(wi, indent=1, default=str) + "\n")
        if args.flip_weight:
            # NEGATIVE TEST ONLY: one element of one served parameter, changed before registration -- the pin (below, per pair) must name it
            _fw = str(args.flip_weight).split(":")
            if len(_fw) < 2:
                raise SystemExit("--flip-weight <served param name>:<flat index>[:<value>]")
            import torch as _torch
            _pmap = dict(eng.model.named_parameters())
            if _fw[0] not in _pmap:
                raise SystemExit(f"--flip-weight names no served parameter {_fw[0]!r} (first served names: {sorted(_pmap)[:5]})")
            with _torch.no_grad():
                _flat = _pmap[_fw[0]].data.view(-1)
                _idx = int(_fw[1]) % _flat.numel()
                _before = float(_flat[_idx].item())
                _flat[_idx] = float(_fw[2]) if len(_fw) > 2 else (1.0 if _before != 1.0 else -1.0)
            flip_weight_rec = {"param": _fw[0], "index": _idx, "before": _before, "after": float(_flat[_idx].item()), "numel": int(_flat.numel()),
                               "expect": "weights pin FAIL naming the field; Commit FAIL rc 3"}
            if flip_weight_rec["after"] == _before:
                raise SystemExit(f"--flip-weight did not change the element ({_before} -> {flip_weight_rec['after']}): pick another value")
            print(f"[delta] FLIP-WEIGHT NEGATIVE TEST ({kind}): {flip_weight_rec} -- checkpoint_binding (the weights pin) MUST FAIL naming {_fw[0]}", flush=True)
        with TL.span("prep.register_weights", kind=kind):
            w = GR.register_weights(_HOT, com, args.weights_hash)
        TL.end("prep.committer_setup", kind=kind)
        print(f"[delta] weight registration ({kind}): {w}", flush=True)
        # the weights of record are the CHECKPOINT's, not the committer's: sha256 every live parameter
        # the registration just hashed (same tensors, same moment) for the field-by-field comparison with the checkpoint-composed digests of
        # record (per pair, below).  Once per process, excluded from steady state like the registration itself.
        _live_wd: dict[str, Any] | None = None
        _wor_rec: dict[str, Any] | None = None
        if getattr(args, "weights_of_record", None) or getattr(args, "checkpoint_dir", None):
            from verity_vllm.check import weights_of_record as WR
            TL.start("validate.weights_of_record_digests", kind=kind)
            try:
                if getattr(args, "weights_of_record", None):
                    _wor_rec = json.loads(Path(args.weights_of_record).read_text())
                    _wor_rec["source"] = str(args.weights_of_record)
                elif served is not None:
                    _wor_rec = WR.served_record(required_manifest, args.checkpoint_dir, checkpoints_manifest=args.checkpoints,
                                                kv_cache_dtype=str(engine_args.get("kv_cache_dtype") or "auto"))
                    _wor_rec["source"] = f"derived in-process from the served Build's weight reads x {args.checkpoint_dir}"
                    (Path(args.out) / "weights_of_record.json").write_text(json.dumps(_wor_rec, indent=1) + "\n")
                else:
                    _pd = args.program_dir
                    if not _pd and getattr(args, "programs_root", None):
                        # prefer a COMPONENT Program (build_request_LP*_T*) over the default wrapper shape build_request: at
                        # B>=2 only the components are of record (the record binds to the manifest's component digests)
                        _cands = sorted(p for p in Path(args.programs_root).glob("build_request_LP*_T*") if (p / "instances.json.gz").exists()) or \
                                 sorted(p for p in Path(args.programs_root).glob("build_request*") if (p / "instances.json.gz").exists())
                        _pd = str(_cands[0]) if _cands else None
                    if not _pd:
                        raise RuntimeError("--checkpoint-dir needs the Program of record (--program-dir / --programs-root)")
                    _wor_rec = WR.build_record(_pd, args.checkpoint_dir, checkpoints_manifest=args.checkpoints,   # the engine's own checkpoints.json (shard pins)
                                               kv_cache_dtype=str(engine_args.get("kv_cache_dtype") or "auto"))   # the kv scales' engine assumption
                    _wor_rec["source"] = f"derived in-process from {_pd} x {args.checkpoint_dir}"
                    if getattr(args, "programs_root", None):
                        # stamp the WHOLE of-record set (every component's digest, struct-checked, + the workload digest +
                        # the row manifest's): check() then requires the run's set to EQUAL it -- no dependence on which component was picked
                        _man = args.required_manifest if getattr(args, "required_manifest", None) and Path(str(args.required_manifest)).exists() else None
                        WR.stamp_of_record_set(_wor_rec, args.programs_root, manifest=_man)
                    if hasattr(WR, "derivation_provenance"):      # the in-process derivation is a logged derivation too
                        _wor_rec["derived"] = WR.derivation_provenance(_pd, args.checkpoint_dir, args.checkpoints, None)
                        _wor_rec["derived"]["tool"] = "verity_vllm.pipeline.commit (in-process --checkpoint-dir fallback)"
                    (Path(args.out) / "weights_of_record.json").write_text(json.dumps(_wor_rec, indent=1) + "\n")
                _live_wd = GR.live_digests(_HOT, eng.model, WR.live_param_digests)
                print(f"[delta] weights of record ({kind}): root {str(_wor_rec.get('root_of_record'))[:16]} fields {_wor_rec.get('n_pinned')}/{_wor_rec.get('n_fields')} pinned, "
                      f"unpinned {len(_wor_rec.get('unpinned') or [])}, unresolved {len(_wor_rec.get('unresolved') or [])}, shards pinned {(_wor_rec.get('checkpoint') or {}).get('all_shards_pinned')}; "
                      f"live digests {_live_wd.get('n')} tensors in {_live_wd.get('wall_s')} s", flush=True)
            except Exception as _we:  # noqa: BLE001  -- fail closed: recorded as not evaluated, the verdict refuses it by name
                _wor_rec = {"error": repr(_we)[:300], "complete": False, "why": f"weights of record not derived: {_we!r}"[:300], "fields": []}
                print(f"[delta] WARNING weights of record not derived: {_we!r}", flush=True)
            TL.end("validate.weights_of_record_digests", kind=kind, ok=not (_wor_rec or {}).get("error"))
        # warm-up (instrumented): sizes the pinned pool, JITs nothing else
        # bounded staging: the warm-up is also the LEARN-ONLY pass -- every shape class of the workload (same lengths, same schedule
        # as the record runs: throwaway_requests) gets its plan here; a learning step over the transient budget records its plan and
        # discards its windows (placeholder root: this run is never a record).  The record runs then hit every plan and stage bounded.
        # Its call boundaries are placeholder words of their shapes (TAPS.warmup): the record pass evaluates its own.
        TL.start("prep.warmup_instrumented", kind=kind)
        TAPS.warmup(com, True)
        com.install()
        r = run_workload(eng, va.throwaway_requests(eng, reqs, seed=200), sampling)
        rc = com.finalize()
        com.uninstall()
        TAPS.warmup(com, False)
        if getattr(com, "bounded", False):
            _bs = rc.extra.get("bounded_staging") or {}
            print(f"[delta] bounded staging warm-up ({kind}): learned-only steps {_bs.get('learned_only_steps')} ({(_bs.get('learned_only_bytes') or 0) >> 20} MiB "
                  f"discarded), transient peak {(_bs.get('transient_peak_bytes') or 0) >> 20} MiB (budget {(_bs.get('learn_budget_bytes') or 0) >> 20} MiB), "
                  f"plan mismatches {_bs.get('plan_mismatches')}, plans cached {(rc.extra.get('native_collect') or {}).get('plans_cached')}, "
                  f"pinned pool {(rc.extra.get('pinned_pool_bytes') or 0) >> 20} MiB", flush=True)
            (out / f"bounded_warmup_{kind}.json").write_text(json.dumps({**_bs, "pinned_pool_bytes": rc.extra.get("pinned_pool_bytes"),
                                                                          "plans_cached": (rc.extra.get("native_collect") or {}).get("plans_cached"),
                                                                          "host_mem_kb": dict(zip(("rss", "hwm"), rss_kb())), "cgroup_memory_current": cgroup_mem_current(),
                                                                          "errors": rc.extra.get("errors")}, indent=1, default=str) + "\n")
        com.reset()
        assert_pristine(eng, base)
        TL.end("prep.warmup_instrumented", kind=kind, wall_s=round(r["wall_s"], 3))
        print(f"[delta] warm-up instrumented ({kind}): {r['wall_s']:.3f}s root-ready {rc.root_ready_delay_s * 1e3:.1f} ms spans {rc.spans['total_s']} "
              f"errors {rc.extra.get('errors')}", flush=True)
        coms[kind] = com
    com = coms[kinds[0]] if kinds else None
    if served is not None:
        _rb = served.rebuild(json.loads(Path(args.workload).read_text()), str(out))   # every kernel has launched in the warm-ups
        print(f"[delta] served Build of this process: program {str(_rb['program_digest'])[:16]} complete {_rb['complete']}, equal to the Build "
              f"of record {_rb['program_digest'] == required_manifest['program_digest']}", flush=True)
    protocols = PO.at_commit(_ekw["target"], eng.model, versions, _settings)   # after every committer registered the weights
    if args.value_check and com is not None and hasattr(com, "_layouts"):
        # VALUE-CORRECTNESS (VALUE-FIRST §5): one extra instrumented run with reference taps + re-execution of the registered relation
        from verity_vllm.check.value_check import run_value_check
        vres = run_value_check(com, lambda: run_workload(eng, reqs, sampling), lambda: assert_pristine(eng, base))
        vres.update({"committer": com.name, "case": eng.case, "model": eng.cp["repo"], "gpu": torch.cuda.get_device_name(0), "label": "measured"})
        (out / "value_check.json").write_text(json.dumps(vres, indent=1, default=str) + "\n")
        print(f"[delta] VALUE CHECK {vres['verdict']}: {vres['tensors']} tensors, tap-equal {vres['tap_equal']}, tap-mismatch {vres['tap_mismatch_n']}; "
              f"re-exec {vres['reexec_equal']}/{vres['reexec_tensors']} equal, mismatch {vres['reexec_mismatch_n']}; repeated-call modules {vres['repeated_calls']}; "
              f"missing {len(vres['missing_in_commit'])}; not verified {vres['not_verified_n']}; skipped {vres['reexec_skipped']}; by class {vres['by_class']}; "
              f"opened {vres['value_source']['reads']} reads / {vres['value_source']['bytes']} B in {vres['value_source']['seconds']} s", flush=True)
        del vres

    rows: list[dict] = []
    step_rows: list[dict] = []
    tokens_ref: dict | None = None
    decl = None
    decls: dict[str, dict] = {}
    prev_digests: dict = {}
    replay_cache: dict[str, dict[str, Any]] = {}      # --replay-cache: full-key -> pair 0's C2 record (this Commit phase only)
    deferred: dict[int, dict[str, Any]] = {}          # --replay-deferred: pair -> its replay bundle's context
    # the Program's declared stop rule + LAG, read once for every consumer of every pair (population, composition,
    # linkage, replay); recorded in versions so the finalizer reads the same fact the Commit counted on
    stop_of_record = stop_rule_of_record(eng, wl, versions)
    versions["stop_rule_of_record"] = {k: v for k, v in stop_of_record.items() if k != "srule"} | {"eos_source": (stop_of_record.get("srule") or {}).get("source")}
    (out / "versions.json").write_text(json.dumps(versions, indent=1, default=str) + "\n")
    arms_all = ["control"] + (kinds if multi else ["instrumented"])
    # the measured runs start here; verdict.host_peak.printed_before_run = admission_monotonic_s < this (timestamps, not a literal)
    _RUN0 = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "monotonic_s": round(time.monotonic(), 3)}
    for pair in range(args.pairs):
        if multi:
            shift = pair % len(arms_all)
            order = arms_all[shift:] + arms_all[:shift]
        else:
            order = ["control", "instrumented"] if pair % 2 == 0 else ["instrumented", "control"]
        if args.only_arm:
            order = [a for a in order if a == args.only_arm]
        for k, arm in enumerate(order):
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            rss0, _ = rss_kb()
            rc = None
            com = coms[arm] if multi and arm != "control" else (coms[kinds[0]] if kinds else None)
            _wstage = f"work.pair{pair}.{arm}"
            TL.start(_wstage, pair=pair, arm=arm)
            # the remote session audits the first instrumented run, a pass an engine step of it; the control arm is never audited
            _lock = _rs.begin() if _rs is not None and arm != "control" and _rs.lockstep is None else None
            if arm != "control":
                com.reset()
                if compiled_src is not None:
                    compiled_src.stats_reset_step_counters()
                if served is not None:
                    served.src.begin_run()
                com.install()
            r = run_workload(eng, reqs, sampling, step=None if _lock is None else _lock.step)
            if _lock is not None:
                r["remote_session"] = RS.end_run(_rs, pair, arm)
            if arm != "control":
                rc = com.finalize()
                t_ready = time.perf_counter()
                com.uninstall()
                assert_pristine(eng, base)
            else:
                assert_pristine(eng, base)
            torch.cuda.synchronize()
            TL.end(_wstage, pair=pair, arm=arm, wall_s=round(r["wall_s"], 4), steps=len(r["step_wall_s"]))
            rss1, hwm = rss_kb()
            if tokens_ref is None:
                tokens_ref = r["tokens"]
            row: dict[str, Any] = {
                "schema": SCHEMA, "label": args.label, "workload": args.workload, "workload_sha256": wl["sha256"], "case": eng.case,
                "model": eng.cp["repo"], "pair": pair, "order_in_pair": k, "arm": arm,
                "committer": None if arm == "control" else (arm if multi else args.committer), "committer_name": com.name if arm != "control" else None,
                "wall_s": r["wall_s"], "iterations": r["iterations"], "engine_steps": len(r["step_wall_s"]),
                "generated_tokens": r["generated_tokens"], "prompt_tokens": r["prompt_tokens"],
                "generated_tokens_per_s": r["generated_tokens"] / r["wall_s"], "total_tokens_per_s": (r["generated_tokens"] + r["prompt_tokens"]) / r["wall_s"],
                "step_wall_s": r["step_wall_s"], "step_wall_prefill_s": r["step_wall_s"][0],
                "step_wall_decode_mean_s": statistics.fmean(r["step_wall_s"][1:]) if len(r["step_wall_s"]) > 1 else None,
                "tokens_equal_ref": r["tokens"] == tokens_ref,
                "tokens_digest": hashlib.sha256(json.dumps(r["tokens"], default=str, separators=(",", ":")).encode()).hexdigest(),   # cross-process pairing (compiled rows)
                "gpu_mem": {"torch_max_allocated_gb": torch.cuda.max_memory_allocated() / 2**30, "torch_reserved_gb": torch.cuda.memory_reserved() / 2**30, "nvidia_smi_used_mib": nvidia_smi_used_mib()},
                "host_mem_kb": {"rss_before": rss0, "rss_after": rss1, "hwm": hwm},
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            if "remote_session" in r:
                row["remote_session"] = r["remote_session"]
            if rc is not None:
                if fault.startswith("retained_flip"):
                    row["fault_retained_flip"] = com.fault_retained_flip(fault)
                    print(f"[delta] VERITY_FAULT retained_flip (negative test): {row['fault_retained_flip']}", flush=True)
                # the run's coins, drawn once its root of record is fixed (the auditor host's session; without one, this process's own)
                from verity_vllm.commit.challenge import challenge_positions, run_root_of
                sample = challenge_positions(run_root_of(rc), [s.n_leaves for s in rc.steps], args.openings, salt=str(pair).encode(), **CC.positions(rc, coins))
                TL.start("validate.openings", pair=pair)
                t_o = time.perf_counter()
                skipped = None
                deferred_n = 0
                deferred_pos: list[tuple[int, int]] = []
                if getattr(com, "retain", "device") == "none":
                    # retain=none: no raw copy anywhere -> the openings come from the REPLAY in openings_after_release (required by the verdict)
                    ops, skipped = [], "retain=none: values by replay (openings_after_release)"
                    if not args.open_after_release:
                        raise SystemExit("--retain none needs --open-after-release (the only opening path is replay)")
                elif getattr(com, "retain_exclude", None):
                    # retain=host + exclude: a challenged leaf inside an excluded family's interior has no host bytes -> it
                    # opens by REPLAY in openings_after_release (required by the verdict when `skipped` is set); every other leaf opens
                    # from host here.  The in-run record keeps n / verified of the host openings (a failure among them is still a FAIL).
                    held = [p for p in sample if com.leaf_retained(*p)]
                    deferred_pos = [(int(s), int(i)) for s, i in sample if not com.leaf_retained(s, i)]
                    deferred_n = len(deferred_pos)
                    ops = com.open(held)
                    if deferred_n:
                        com._oar_deferred = list(getattr(com, "_oar_deferred", []) or []) + deferred_pos
                        skipped = (f"retain=host excluding {sorted(com.retain_exclude)}: {deferred_n} of {len(sample)} challenged leaves lie in "
                                   "excluded families -> opened by replay (openings_after_release)")
                        if not args.open_after_release:
                            raise SystemExit("--retain-exclude needs --open-after-release (the excluded families open by replay)")
                else:
                    ops = com.open(sample)
                t_open = time.perf_counter() - t_o
                t_v = time.perf_counter()
                ok = [com.verify(rc, o) for o in ops]
                t_ver = time.perf_counter() - t_v
                TL.end("validate.openings", pair=pair, n=len(ops), verified=sum(ok))
                row["commit"] = {
                    "run_root": rc.run_root.hex(), "steps": len(rc.steps), "leaves": sum(s.n_leaves for s in rc.steps),
                    "bytes_bound": sum(s.n_bytes for s in rc.steps), "tensors": sum(s.n_tensors for s in rc.steps),
                    "spans_total_s": rc.spans["total_s"], "spans_calls": rc.spans["calls"], "spans_bytes": rc.spans["bytes"], "spans_throughput": rc.spans.get("throughput"),
                    "root_ready_delay_s": rc.root_ready_delay_s, "finalize_s": rc.t_finalize_s, "t_ready_minus_last_step_s": t_ready - r["t_last_step"],
                    "persistent_bytes": rc.persistent_bytes, "temporary_bytes_peak": rc.temporary_bytes_peak, "extra": rc.extra,
                    "openings": {"n": len(ops), "verified": sum(ok), "generate_s": t_open, "verify_s": t_ver, "skipped": skipped,
                                 "deferred_replay_n": deferred_n, "challenge_n": len(sample),
                                 # the deferred POSITIONS (step, leaf) on record: openings_after_release must receive them all
                                 "deferred_positions": [list(p) for p in deferred_pos], "deferred_digest": _positions_digest(deferred_pos),
                                 "bytes_mean": statistics.fmean(len(o.value) + 32 * sum(1 for p in o.path if p is not None) for o in ops) if ops else None,
                                 # fail-closed diagnostics: WHICH positions failed (step, leaf index, identity) -- a failed opening is a Commit FAIL
                                 "failed": [{"step": o.step, "index": o.index, "identity": {k: (v if isinstance(v, (int, str, bool)) else str(v)) for k, v in (o.identity or {}).items()}}
                                            for o, k_ok in zip(ops, ok) if not k_ok][:16]},
                    "weight_registration": rc.weight_registration}
                for s in rc.steps:
                    step_rows.append({"pair": pair, "arm": arm, "step": s.step, "wall_s": r["step_wall_s"][s.step] if s.step < len(r["step_wall_s"]) else None,
                                      "n_leaves": s.n_leaves, "n_bytes": s.n_bytes, "n_tensors": s.n_tensors, "hashing_s": s.hashing_s,
                                      "root_finalisation_s": s.root_finalisation_s, "root_ready_after_step_end_s": s.t_root_ready - s.t_step_end,
                                      "backlog_at_step_end": s.backlog_at_step_end,
                                      "spans": rc.spans["per_step"].get(s.step, {})})
                # the coverage declaration of THIS run feeds the verdict -- every required class declared, every
                # declared leaf-binding class with tensors > 0 -- so a producer that never fired cannot pass on equal attempt counts
                d = com.coverage_declaration(rc)
                d["run"] = {"workload": args.workload, "case": eng.case, "model": eng.cp["repo"]}
                row["commit"]["classes"] = class_coverage(d)
                # X-03: fail-closed verdict per instrumented run (recorded; the process exits non-zero if any run fails)
                row["commit"]["expected_tensors"] = rc.extra.get("expected_tensors")
                row["commit"]["committed_tensors"] = rc.extra.get("committed_tensors")
                row["commit"]["errors_n"] = rc.extra.get("errors_n", len(rc.extra.get("errors") or []))
                # binding map + identity coverage against the collector-independent expected population, and openings drawn
                # from that population (the map and its digest are fixed here, before either challenge is derived from them)
                _rb_started = RB.start(out, pair, com, rc, args, protocols)
                GL.release(TL, eng, list(coms.values()), args, last=pair == args.pairs - 1 and k == len(order) - 1 and (_HOT or {}).get("last", _HOT is None), after=_wstage, bundle=_rb_started is not None)   # nothing below reads the device
                TL.start("validate.binding_coverage", pair=pair)
                try:
                    row["commit"]["binding"] = binding_record(com, rc, reqs, sampling, args, pair, served_tokens=r["tokens"], stop=stop_of_record, finish_reasons=r.get("finish_reasons"))
                    row["commit"]["rules_version"] = row["commit"]["binding"]["rules_version"]
                    b = row["commit"]["binding"]
                    print(f"[delta] BINDING (pair {pair}): map {b['map_entries_n']} entries digest {b['map_digest'][:16]}; population {b['population']['n']} "
                          f"(digest {b['population']['digest'][:16]}); coverage {'OK' if b['coverage']['ok'] else 'FAIL'} bound {b['coverage']['bound_n']}/{b['coverage']['expected_n']} "
                          f"missing {b['coverage']['missing_n']} extra {b['coverage']['extra_n']}; population openings {b['openings_population']['verified']}/{b['openings_population']['opened']} "
                          f"(unbound {b['openings_population']['unbound_n']}, late step {b['openings_population']['late_step_in_sample']})", flush=True)
                    cmp_ = (b["coverage"] or {}).get("composition")
                    if cmp_ is not None:
                        print(f"[delta] COMPOSITION (pair {pair}): {'OK' if cmp_.get('ok') else 'FAIL'} engine steps {cmp_.get('checked_steps')} problem steps {cmp_.get('problem_steps_n', 0)}"
                              f"{(' ' + str(cmp_.get('skipped'))) if cmp_.get('skipped') else ''}{(' -- ' + str(cmp_.get('error'))) if cmp_.get('error') else ''}", flush=True)
                    if b["coverage"].get("error"):
                        print(f"[delta] required_value_coverage: identity coverage ERROR (pair {pair}): {b['coverage']['error']}", flush=True)
                    if b["coverage"]["missing"]:
                        print(f"[delta]   first missing: {b['coverage']['missing'][:3]}", flush=True)
                except Exception as e:  # noqa: BLE001
                    # name the SITE too (last frames): a bare "TypeError('int' ...)" does not say which reader raised
                    import traceback
                    site = " | ".join(ln.strip() for ln in traceback.format_exc().splitlines() if ln.strip().startswith("File "))[-400:]
                    row["commit"]["binding"] = {"error": repr(e)[:300], "site": site, "rules_version": "r14-v1"}
                    print(f"[delta] BINDING ERROR (pair {pair}): {e!r} at {site}", flush=True)
                TL.end("validate.binding_coverage", pair=pair, ok=bool(((row["commit"].get("binding") or {}).get("coverage") or {}).get("ok")))
                _ids_full = None                                                          # the engine-step-annotated identities the reconciler re-derives the cell counts from
                if required_manifest is not None and hasattr(com, "layouts_dump"):
                    TL.start("validate.manifest_coverage", pair=pair)
                    # identity coverage against the Program/query manifest (the collector's own layouts are the CLAIM, the manifest the requirement)
                    lay = com.layouts_dump()
                    # this layouts-only check keys on the manifest's `engine_step` annotation, which the compiled
                    # manifest derived under the DECLARED arrive_steps; on a workload with gaps the engine admits a pending request as soon
                    # as it is idle, so the population is placed on the committed-step axis by the engine schedule of record (the same
                    # engine_schedule the binding-map coverage and the kernel check use): engine_step = request step + observed arrival
                    _man_cov, _cov_sched, _epr_cov, _ne_class, _ids_all_n = required_manifest, None, None, None, None
                    # the placement below ran at B >= 2 ONLY (`len(reqs) > 1`):
                    # at B=1 the cap-length manifest (row_pod's Program of record at Commit is the CAP-shape build_request on this tree, the
                    # served-shape rule retired there) was required IN FULL and the never-executed steps of an EOS-stopped request were
                    # 'uncovered' instead of the named class -- measured: 8,295 identities of r0's steps 11..31
                    # {fa2_hidden_m1_stream 630, instance_outputs 7,623, ids 21+21}, COMMIT FAIL at MANIFEST COVERAGE.  The single request
                    # is r0 (index order, the manifest's request_id), the same executed_prefix_of_record / request_steps_of_record /
                    # restrict_* run -- as at B >= 2, nothing new; a request that ran to its cap drops nothing.
                    if reqs and isinstance(required_manifest, dict) and required_manifest.get("identities"):
                        _o = {str(q.get("request_id")): f"r{i}" for i, q in enumerate(reqs)}
                        _a = {f"r{i}": int(q.get("arrive_step") or 0) for i, q in enumerate(reqs)}
                        _c = {f"r{i}": int(q.get("max_tokens") or sampling.get("max_tokens") or 1) for i, q in enumerate(reqs)}
                        # the SAME executed prefix of record binding_record placed the population over (served tokens of
                        # this run + the declared stop rule / LAG): the layouts-only requirement and its per-step composition sizes follow it
                        from verity_vllm.commit import executed_prefix as _EP
                        _epr_cov = _EP.executed_prefix_of_record(sorted(_c, key=lambda k: int(k[1:])), {_rid_of(_o, str(k)): [int(t) for t in v] for k, v in r["tokens"].items()},
                                                                 caps=_c, eos_of_record=stop_of_record.get("eos"), lag=stop_of_record.get("lag"))
                        _n_exec = {k: int(_epr_cov["requests"][k]["executed_prefix_len"]) for k in _c}
                        _r = {int(st): [(_rid_of(_o, rid), lo, hi) for rid, lo, hi in rows] for st, rows in (getattr(com, "step_rows", None) or {}).items()} or None
                        _cov_sched = engine_schedule(_r, _a, _n_exec)
                        _ids = list(required_manifest["identities"])
                        if _cov_sched["use"] == "observed" and not _cov_sched["all_equal_declared"]:
                            _obs = {k: int(v) for k, v in _cov_sched["observed_arrive_steps"].items()}
                            _a = {**_a, **_obs}
                            _ids = [dict(x, engine_step=int(x["step"]) + _obs[str(x["request_id"])], engine_step_declared=x.get("engine_step"))
                                    if x.get("step") is not None and str(x.get("request_id")) in _obs else x for x in _ids]
                        _rsteps = _EP.request_steps_of_record(_epr_cov, _a)
                        _ne_class = _EP.new_classification()
                        _ids_all_n = len(_ids)
                        _ids_full = list(_ids)
                        _ids, _dropped = _EP.restrict_manifest_identities(_ids, _rsteps, stats=_ne_class)
                        _es_raw = list(required_manifest.get("engine_steps") or [])
                        if _es_raw and not isinstance(_es_raw[0], dict):
                            # a B=1 (non-global) manifest lists its engine steps as plain ints -- no per-step composition
                            # members (COMPOSITION: "manifest declares no engine-step composition") -- so the restriction is by step alone:
                            # a step no request executes at is not executed (restrict_engine_steps over the int list raises
                            # TypeError: 'int' object is not subscriptable).
                            _alw = {int(s) for v in _rsteps.values() for s in v}
                            _es = [e for e in _es_raw if int(e) in _alw]
                            _es_r = {"members_dropped": 0, "steps_changed": 0, "steps_dropped": len(_es_raw) - len(_es)}
                        else:
                            _es, _es_r = _EP.restrict_engine_steps(_es_raw, _rsteps)
                        _man_cov = dict(required_manifest, identities=_ids, engine_steps=_es)
                        if _dropped or _es_r["members_dropped"]:
                            print(f"[delta] manifest requirement placed over the executed prefix of record (pair {pair}): identities not-executed {_dropped} "
                                  f"(by family {_ne_class['by_family']}; manifest of record untouched: {_ids_all_n} identities, digest {str(required_manifest.get('manifest_digest'))[:16]}), "
                                  f"composition members dropped {_es_r['members_dropped']} over {_es_r['steps_changed']} steps, steps not executed {_es_r['steps_dropped']}", flush=True)
                    try:
                        cov = manifest_coverage_check(_man_cov, lay, request_id=None, rank=0, ports=source_ports(getattr(com, "sources", []) or []))
                    except Exception as e:  # noqa: BLE001  -- fail closed, keep the bind record (e.g. B>1 global manifests with `step: "*"` rows)
                        cov = {"ok": False, "error": repr(e)[:300], "checked": 0, "missing_n": -1, "missing_by_family": {}, "families_missing_entirely": [],
                               "committed_unmatched_n": -1, "manifest_source": str(args.required_manifest)}
                        print(f"[delta] MANIFEST COVERAGE ERROR (pair {pair}, {arm}): {e!r}", flush=True)
                    if _cov_sched is not None:
                        cov["schedule"] = {k: _cov_sched.get(k) for k in ("use", "basis", "why", "all_equal_declared", "declared_arrive_steps", "observed_arrive_steps")}
                        cov["engine_step_basis"] = ("engine schedule of record (request step + observed arrival)" if _man_cov is not required_manifest
                                                    else "manifest engine_step annotation (declared arrive_steps)")
                    row["commit"]["coverage"] = cov
                    rc.extra["coverage"] = cov
                    import gzip as _gz
                    if _epr_cov is not None:
                        cov["executed_prefix_of_record"] = {k: {kk: vv for kk, vv in v.items() if kk in ("served_output_len", "executed_prefix_len", "cap", "cap_exhausted", "eos_stop")}
                                                            for k, v in _epr_cov["requests"].items()} | {"lag": _epr_cov["lag"], "ok": _epr_cov["ok"], "problems": _epr_cov["problems"][:3]}
                        # the classification beside the requirement it placed; the manifest of record's own count + digest beside both
                        cov["not_executed_by_rule"] = _ne_class
                        cov["manifest_of_record"] = {"digest": required_manifest.get("manifest_digest"), "identities_n": _ids_all_n, "required_n": len(_man_cov.get("identities") or []),
                                                     "not_executed_n": (_ne_class or {}).get("n"), "edited": False}
                        if not _epr_cov["ok"] and cov.get("ok"):
                            cov["ok"], cov["error"] = False, "executed prefix of record: " + "; ".join(_epr_cov["problems"][:3])   # in place: row + rc.extra alias it
                    with _gz.open(out / f"layouts_pair{pair}_{arm}.json.gz", "wt") as f:
                        json.dump(lay, f)
                    print(f"[delta] required_value_coverage: MANIFEST COVERAGE (pair {pair}, {arm}): {'OK' if cov['ok'] else 'FAIL'} checked {cov['checked']} missing {cov['missing_n']} "
                          f"{cov['missing_by_family']} families-missing-entirely {cov['families_missing_entirely']} unmatched-committed {cov['committed_unmatched_n']}", flush=True)
                    TL.end("validate.manifest_coverage", pair=pair, ok=bool(cov.get("ok")), n=cov.get("checked"))
                # value correspondence is a separate, labelled check -- never folded into the coverage verdict
                # value_correspondence is a COMPOSITE of independently graded components, each recorded with its own
                # result + reference so the finalizer grades each by its own rule (oracle-compare: live Match oracle of THIS attempt;
                # root-compare: an INDEPENDENT pinned root -- never a same-tree reproduction; replay: openings_after_release).
                # `result` = AND over the components that ran (None if none ran); `why` names the failing component's actual cause.
                vc_methods = ["offline-suite (tests/commit/test_binding_negatives.py: late-read, relabel)"]
                components: dict[str, Any] = {}
                vc_result = None
                if args.expect_root:
                    vc_methods.append("root-compare-vs-independent-run (--expect-root)")
                    vc_result = rc.run_root.hex() == args.expect_root
                row["commit"]["value_correspondence"] = {"method": " + ".join(vc_methods), "result": vc_result,
                                                         "note": "coverage + openings cannot see a late read or a relabel (brief §4); live evidence = root equality vs an independent run / replay"}
                if compiled_kernels is not None:
                    # C2 of record: stratified sampled exact replay of P_compiled's kernel relations from the
                    # committed bytes + ordered boundary linkage (ids(s) == token(s-1); argmax lm_head(hidden_s) == token(s)); seed from the run root
                    from verity_vllm.check.replay.compiled_kernel_check import CompiledKernelCheck

                    row["commit"]["compiled_source"] = compiled_kernels.summary()
                    row["commit"]["compilation"] = versions["compilation"]
                    # tokens / arrivals / declared prompts per request `r<i>` (workload order) + the engine's per-step row order (B>=2)
                    _order = {str(q.get("request_id")): f"r{i}" for i, q in enumerate(reqs)}
                    _toks = {_order.get(str(k), str(k)): v for k, v in r["tokens"].items()}
                    _arr = {f"r{i}": int(q.get("arrive_step") or 0) for i, q in enumerate(reqs)}
                    _prompts = {f"r{i}": list(q.get("prompt_token_ids") or [])[int(q.get("cached") or 0):] for i, q in enumerate(reqs)}   # the prefill's rows
                    _rows = {int(st): [(_rid_of(_order, rid), lo, hi) for rid, lo, hi in rows] for st, rows in (getattr(com, "step_rows", None) or {}).items()} or None
                    _caps = {f"r{i}": int(q.get("max_tokens") or sampling.get("max_tokens") or 1) for i, q in enumerate(reqs)}
                    _sched = engine_schedule(_rows, _arr, _caps)                  # the linkage's k = s - arrival counts on the committed-step axis too
                    if _sched["use"] == "observed":
                        _arr = {**_arr, **{k: int(v) for k, v in _sched["observed_arrive_steps"].items() if k in _arr}}
                    try:
                        ckc = CompiledKernelCheck(com, eng.model, _toks, seed=CH.kernel_check_seed(rc.run_root), request_rows=_rows, arrivals=_arr,
                                                  prompts=_prompts, seed_source=COMPILED_REPLAY_SEED_SOURCE).run(required_manifest)
                    except Exception as e:  # noqa: BLE001
                        import traceback as _tb

                        ckc = {"ok": False, "error": repr(e)[:300], "traceback": _tb.format_exc()[-1200:], "launches_checked": 0}
                        print(f"[delta] compiled kernel check raised:\n{_tb.format_exc()[-1200:]}", flush=True)
                    # record shape of the C2 form (A) graded by the finalizer: components.sampled_replay + components.boundary_linkage
                    lk0 = ckc.get("linkage") or {}
                    _kinds = sorted((ckc.get("by_kind") or {}).keys())
                    sr_rec = {"result": bool(ckc.get("ok")) and not ckc.get("error"),
                              "population": {"n": required_manifest["populations"]["identities"], "digest": required_manifest.get("manifest_digest") or required_manifest.get("digest"),
                                             "source": "P_compiled x query required manifest (compiled_manifest.py from the control process's Inductor output code)",
                                             "per_step": required_manifest["populations"].get("per_step")},
                              "n": ckc.get("launches_checked"), "outputs_checked": ckc.get("outputs_checked"), "outputs_exact": ckc.get("outputs_exact"),
                              "launches_exact": ckc.get("launches_exact"), "launches_missing": ckc.get("launches_missing"),
                              "strata": {"mechanisms": _kinds, "requests": required_manifest.get("requests"), "ranks": ["n/a (tp1: rank 0)"],
                                         "steps": ckc.get("sampled_steps"), "phases": ["prefill (uncaptured: hooks acquire directly)", "decode (CUDA-graph replay: slot drain per step)"],
                                         "pieces": f"every piece ({required_manifest['program']['pieces']}) x every kernel kind of a sampled step",
                                         "lifetime": "n/a (unwindowed: per-launch static slots written at graph capture, drained before the step's root)"},
                              "seed": ckc.get("seed"), "unreplayable": [], "by_kind": ckc.get("by_kind"), "mismatches": ckc.get("mismatch"), "missing": ckc.get("missing"),
                              "errors": ckc.get("errors"), "why": ckc.get("error") or ("all sampled launches replay bit-for-bit from the committed bytes" if ckc.get("ok") else "see mismatches/missing/errors"),
                              "method": "sampled-exact-replay: P_compiled kernel relations (compiled_relations.py) evaluated on the committed input buffers + registered weights, "
                                        "compared bit-for-bit with the committed output buffers (compiled_kernel_check.py)"}
                    link_rec = {"result": bool(lk0.get("ok")), "method": "per-(request, engine step) token gate under graph replay: embedding ids(s) == sampled token(s-1); "
                                                                   "argmax lm_head(final_norm_out(s)) == sampled token(s) (eager lm_head on the committed final hidden)",
                                "checked": int(lk0.get("ids_checked", 0)) + int(lk0.get("argmax_checked", 0)), "ids": f"{lk0.get('ids_equal')}/{lk0.get('ids_checked')}",
                                "argmax": f"{lk0.get('argmax_equal')}/{lk0.get('argmax_checked')}", "mismatch": lk0.get("mismatch"),
                                # per-(step, request) attribution: the gate ran over EVERY committed step x every
                                # request row of the step (complete) -- the compiled row's stand-in for form (B)'s attribution at B>=2
                                "attribution": lk0.get("attribution"),
                                "why": None if lk0.get("ok") else "a step's ids / argmax does not tie to the request's own tokens (engine outputs == control arm)"}
                    neg_rec = {"suite": "VERITY_FAULT compiled_omit:<kernel> (one launch's buffers never acquired under replay -> C1 coverage FAIL) ; "
                                        "compiled_late_drain (slots read one step late under replay = post-overwrite read -> C2 linkage FAIL)",
                               "artifact": "negatives/ beside this attempt's commit/ (treatment processes with the fault; verdict.json each)", "results": None}
                    # the components are ADDED to the composite the O1 gate reads (commit_verdict: components.
                    # sampled_replay / boundary_linkage stand in for the top level only when the method says sampled-exact-replay); result = AND.
                    # `components` / `vc_methods` / `vc_result` are the shared locals the composite below is built from (the dict is
                    # (re)assigned to row.commit.value_correspondence after the compiled/root/oracle blocks); the compiled-only top-level
                    # fields are merged in right after that assignment (`_compiled_vc_extra`).
                    components.update({"sampled_replay": sr_rec, "boundary_linkage": link_rec, "negatives": neg_rec})
                    vc_methods.insert(0, "sampled-exact-replay (P_compiled kernel relations from committed bytes; compiled_kernel_check.py) + boundary linkage")
                    vc_result = bool(ckc.get("ok")) if vc_result is None else (vc_result and bool(ckc.get("ok")))
                    _compiled_vc_extra = {"population": required_manifest["populations"], "n": ckc.get("launches_checked"), "seed": ckc.get("seed"),
                                          "sampled_steps": ckc.get("sampled_steps"), "compiled_check": ckc}
                    cs = row["commit"]["compiled_source"]["stats"]
                    lk = ckc.get("linkage") or {}
                    _ae = ckc.get("autotune_equivalence") or {}
                    if _ae:
                        print(f"[delta] AUTOTUNE EQUIVALENCE (pair {pair}): " + "; ".join(
                            f"{k}: {v.get('n_candidates')} candidates, {v.get('shown')}/{v.get('launches')} launches shown{'' if v.get('all_shown') else ' (NOT shown: ' + str(v.get('why'))[:90] + ')'}"
                            if (v.get("n_candidates") or 0) > 1 else f"{k}: 1 candidate" for k, v in sorted(_ae.items())), flush=True)
                    print(f"[delta] COMPILED KERNELS (pair {pair}): steps {cs['steps']} replayed {cs['replayed_steps']} eager {cs['eager_steps']} replay-drains {cs['replay_drains']} "
                          f"eager-acq {cs['eager_acquisitions']} omitted {cs['omitted']} outside-step {cs['hook_calls_outside_step']}; C2 {'OK' if ckc.get('ok') else 'FAIL'}: launches "
                          f"{ckc.get('launches_exact')}/{ckc.get('launches_checked')} exact, missing {ckc.get('launches_missing')}, outputs {ckc.get('outputs_exact')}/{ckc.get('outputs_checked')}, "
                          f"by kind {ckc.get('by_kind')}, linkage ids {lk.get('ids_equal')}/{lk.get('ids_checked')} argmax {lk.get('argmax_equal')}/{lk.get('argmax_checked')} "
                          f"positions {lk.get('pos_equal', 0)}/{lk.get('pos_checked', 0)} "
                          f"attribution {(lk.get('attribution') or {}).get('pairs_checked')}/{(lk.get('attribution') or {}).get('pairs_total')} pairs over "
                          f"{(lk.get('attribution') or {}).get('steps_checked')}/{(lk.get('attribution') or {}).get('population_steps')} steps complete={(lk.get('attribution') or {}).get('complete')} "
                          f"coincident {(lk.get('attribution') or {}).get('coincident_pairs')} (tokens-only {(lk.get('attribution') or {}).get('coincident_pairs_tokens')}) "
                          f"steps {ckc.get('sampled_steps')} {ckc.get('mismatch', [])[:2]} {ckc.get('missing', [])[:2]} {ckc.get('errors', [])[:1]} {ckc.get('error', '')}", flush=True)
                if served is not None:
                    row["commit"]["compiled_source"], row["commit"]["compilation"] = served.src.summary(), versions["compilation"]
                    _chal, _chal_rec = CC.replay_challenge(rc, args.replay_challenge)
                    _sc2 = served.c2(com, rc, reqs, _chal, args.replay_seed)   # the challenge drawn after the run root
                    components.update(_sc2["components"])
                    components["replay_challenge"] = _chal_rec
                    vc_methods.insert(0, "sampled-exact-replay (the served Build's Program on the committed values; served_check.py) + boundary linkage")
                    vc_result = _sc2["ok"] if vc_result is None else (vc_result and _sc2["ok"])
                    print(f"[delta] SERVED C2 (pair {pair}): {_sc2['line']}", flush=True)
                if compiled_src is not None:
                    # C2 under graph replay: no Python tap can observe a value inside a replayed graph, so correspondence is
                    # checked from the committed bytes alone (Linear re-execution from committed inputs, lm_head(norm) == logits, greedy
                    # argmax == sampled ids, embed-in(s+1) == sampled(s)); a late drain (VERITY_FAULT=compiled_late_drain) fails here
                    from verity_vllm.check.compiled_value_check import CompiledValueCheck

                    row["commit"]["compiled_source"] = compiled_src.summary()
                    row["commit"]["compilation"] = versions["compilation"]
                    try:
                        cvc = CompiledValueCheck(com, eng.model, eng.runner, sampling).run()
                    except Exception as e:  # noqa: BLE001
                        cvc = {"ok": False, "error": repr(e)[:300], "checked": 0}
                    row["commit"]["value_correspondence"]["compiled_check"] = cvc
                    row["commit"]["value_correspondence"]["method"] += " + compiled_value_check (committed-bytes relations; compiled_value_check.py)"
                    vc_result = bool(cvc.get("ok")) if vc_result is None else (vc_result and bool(cvc.get("ok")))
                    row["commit"]["value_correspondence"]["result"] = vc_result
                    cs = row["commit"]["compiled_source"]["stats"]
                    print(f"[delta] COMPILED (pair {pair}): steps {cs['steps']} replay-drains {cs['replay_drains']} eager-drains {cs['eager_drains']} direct-steps {cs['direct_steps']} "
                          f"direct-calls {cs['direct_calls']} unwritten-skipped {cs['unwritten_skipped']} refused-capture-alloc {cs['refused_capture_alloc']} shape-mismatch {cs['shape_mismatch']}; "
                          f"VALUE CHECK {'OK' if cvc.get('ok') else 'FAIL'}: reexec {cvc.get('reexec_equal')}/{cvc.get('reexec_checked')} logits {cvc.get('logits_equal')}/{cvc.get('logits_checked')} "
                          f"argmax {cvc.get('argmax_equal')}/{cvc.get('argmax_checked')} chain {cvc.get('chain_equal')}/{cvc.get('chain_checked')} "
                          f"mismatch {cvc.get('reexec_mismatch', [])[:2]} {cvc.get('chain_mismatch', [])[:2]} {cvc.get('error', '')}", flush=True)
                    components["root_compare"] = {"result": vc_result, "run_root": rc.run_root.hex(), "expect_root": args.expect_root,
                                                  "reference": "--expect-root (caller-pinned root; independent only if produced by another tree/pod)",
                                                  "why": None if vc_result else f"run root {rc.run_root.hex()[:16]} != expected root {str(args.expect_root)[:16]} (--expect-root)"}
                row["commit"]["value_correspondence"] = {"method": " + ".join(vc_methods), "result": vc_result, "components": components, "why": None,
                                                         "note": "coverage + openings cannot see a late read or a relabel (brief §4); live evidence = the Match oracle of this attempt / an independent pinned root / replay"}
                if compiled_kernels is not None:
                    row["commit"]["value_correspondence"].update(_compiled_vc_extra)          # the C2-of-record fields (compiled_check = the full report)
                if served is not None:
                    row["commit"]["value_correspondence"].update(_sc2["value_correspondence"])
                if getattr(args, "match_dir", None):
                    # LIVE C2 = oracle-compare: every bound identity's committed bytes against the Match phase's
                    # value record of the SAME attempt (separate process + acquisition path); any mismatch fails the verdict naming the identity.
                    # the oracle must be THIS workload's Match record: request ids / prompt lengths / caps are checked by name
                    # before any value compare (a foreign oracle = FAIL 'foreign oracle', never a value mismatch that happens to point the right way).
                    TL.start("validate.oracle_compare", pair=pair)
                    try:
                        from verity_vllm.check import oracle_compare as OC
                        _bm = json.loads((Path(args.out) / f"binding_map_p{pair}.json").read_text())
                        _oracle = OC.MatchOracle(args.match_dir)
                        _wl = {"requests": [{"request_id": r.get("request_id"), "index": r.get("index", i), "prompt_len": r.get("prompt_len"),
                                             "max_tokens": r.get("max_tokens") or sampling.get("max_tokens")} for i, r in enumerate(reqs)]}
                        _prod = OC.producers_of(required_manifest)
                        if oracle_producers is None:
                            oracle_producers = CP.derived(args, required_manifest, reqs, sampling)
                            print(f"[delta] form (B) producer facts from the Programs of record: {oracle_producers[1]}", flush=True)
                        if oracle_producers[0]:
                            _prod = OC.merge_producers(_prod, oracle_producers[0])
                        # the manifest's alias table: a shared module's invocation k is compared at ITS Program op
                        _ocr = OC.OpenedReader(com, rc)
                        oc = OC.oracle_compare(_oracle, _bm, _ocr, producers=_prod, workload=_wl,
                                               aliases=(required_manifest.get("op_path_aliases") if isinstance(required_manifest, dict) else None),
                                               query_families=OC.query_families_of(required_manifest),   # identities per Q(P) family
                                               runtime_rows=OC.runtime_rows_of(required_manifest if isinstance(required_manifest, dict) else None),
                                               own=frozenset().union(*(getattr(s, "own_names", ()) for s in getattr(com, "sources", ()))))
                        oc["producer_facts"] = oracle_producers[1]
                        oc["value_source"] = _ocr.record()
                        print(f"[delta] C2 ORACLE COMPARE opened values (pair {pair}): {OV.filed(row['commit'], 'oracle_compare', oc['value_source'])}", flush=True)
                    except Exception as e:  # noqa: BLE001  -- fail closed: an unreadable oracle is 'not checked', never PASS
                        oc = {"schema": OC.SCHEMA if "OC" in dir() else "verity-sweep/oracle-compare/v1", "method": "oracle-compare", "result": None,
                              "why": f"oracle compare not run: {e!r}"[:300], "compared": 0, "mismatch_n": 0, "error": repr(e)[:300]}
                    row["commit"]["oracle_compare"] = oc
                    vc = row["commit"]["value_correspondence"]
                    vc["method"] = ("oracle-compare" if oc.get("complete") else "oracle-compare(partial)") + " + " + vc["method"]
                    vc["oracle_compare"] = {k: oc.get(k) for k in ("result", "why", "grade", "compared", "equal", "mismatch_n", "steps_compared", "steps_not_snapshotted", "complete", "attribution", "workload_check", "value_source")}
                    vc["oracle_provenance"] = oc.get("oracle")
                    components["oracle_compare"] = dict(vc["oracle_compare"], reference="live Match oracle of this attempt (--match-dir)", provenance=oc.get("oracle"))
                    if oc.get("result") is False:
                        vc["result"] = False
                    elif oc.get("result") is True and vc_result in (None, True):
                        vc["result"] = True
                    _at = oc.get("attribution") or {}
                    TL.end("validate.oracle_compare", pair=pair, ok=oc.get("result"), n=oc.get("compared"))
                    print(f"[delta] C2 ORACLE COMPARE (pair {pair}): {'MISMATCH' if oc.get('result') is False else ('OK complete' if oc.get('complete') else str(oc.get('grade') or 'partial'))} "
                          f"attribution {_at.get('complete')} ({_at.get('pairs')} step/request pairs, unverified {_at.get('unverified_n')}, ambiguous {_at.get('ambiguous_steps')}) compared {oc.get('compared')} "
                          f"equal {oc.get('equal')} mismatch {oc.get('mismatch_n')} steps {oc.get('steps_compared')} not-snapshotted {oc.get('steps_not_snapshotted')} "
                          f"by-family {oc.get('compared_by_family')} not-compared {oc.get('not_compared')}{(' -- ' + str(oc.get('why'))) if oc.get('why') else ''}", flush=True)
                _c2 = C2.Site(args=args, out=Path(args.out), pair=pair, com=com, rc=rc, row=row, reqs=reqs, sampling=sampling, tokens=r["tokens"], manifest=required_manifest, stop_of_record=stop_of_record, ids_full=_ids_full, versions=versions, components=components, vc_result=vc_result, tl=TL, protocols=protocols, weights_of=lambda p: C2.live_weights(PO.weights_view(protocols, eng.model), p), weights_check=lambda: C2.live_weights_check(lambda: PO.register_weights(protocols, com), com, rc), after_replay=C2.vu_export(args, pair, rc, required_manifest, versions, eng))
                _c2_deferred = bool(args.replay_deferred and not C2.live_only(_c2) and RB.defer(out, pair, com, rc, args, tl=TL, arm=arm, versions=versions, reqs=reqs, sampling=sampling, tokens=r["tokens"], stop_of_record=stop_of_record, ids_full=_ids_full, declaration=d, required_classes=required_classes, vc_result=vc_result, plan=(lambda: C2.plan(_c2, _planner)) if args.replay_deferred == "slim" else None, row=row, deferred=deferred, started=_rb_started))
                if _c2_deferred and args.serving_rows:
                    C2.serving_rows(_c2)
                elif not _c2_deferred and (getattr(args, "program_dir", None) or getattr(args, "programs_root", None)):
                    C2.run(_c2, replay_cache)
                if _wor_rec is not None or _live_wd is not None:
                    # WEIGHTS PIN: the served parameters the committer registered == the checkpoint of record, field by field
                    from verity_vllm.check import weights_of_record as WR
                    TL.start("validate.weights_pin", pair=pair)
                    if _wor_rec is None or _live_wd is None or _wor_rec.get("error"):
                        _wp = {"result": None, "why": (_wor_rec or {}).get("why") or "live parameter digests not taken", "method": "MAN-09 weights pin (not evaluated)"}
                    else:
                        # the record must be the one of THIS Program and THIS checkpoint: its program_digest against the
                        # Build digest of record (--program-digest) / the manifest's component digests, its checkpoints.json entry against
                        # the workload's (model, engine.revision) pin -- a record built for another Program/checkpoint is refused by name
                        _pdr: set[str] = set()
                        if getattr(args, "program_digest", None):
                            _pdr.add(str(args.program_digest))
                        if isinstance(required_manifest, dict):
                            _pdr.update(str(x) for x in (required_manifest.get("component_digests") or []))
                            for _c in (required_manifest.get("components") or {}).values():
                                if isinstance(_c, dict) and _c.get("program_digest"):
                                    _pdr.add(str(_c["program_digest"]))
                        _wp = WR.check(_wor_rec, _live_wd, program_digests_of_record=(_pdr or None),
                                       checkpoint_of_record={"repo": wl.get("model"), "revision": wl.get("revision")},
                                       kv_cache_dtype=str(engine_args.get("kv_cache_dtype") or "auto"))
                    _wp["source"] = (_wor_rec or {}).get("source")
                    _wp["committed_weights_root"] = (rc.weight_registration or {}).get("root") if isinstance(rc.weight_registration, dict) else None
                    row["commit"]["weights_pin"] = _wp
                    components = row["commit"]["value_correspondence"].setdefault("components", {})
                    components["weights_pin"] = {"result": _wp.get("result"), "why": _wp.get("why"), "root_of_record": _wp.get("root_of_record"), "compared": _wp.get("compared"),
                                                 "mismatched": _wp.get("mismatched"), "missing": _wp.get("missing"), "unpinned_n": len(_wp.get("unpinned") or []),
                                                 "unresolved": _wp.get("unresolved"), "all_shards_pinned": _wp.get("all_shards_pinned"),
                                                 "analytic": [{k: a.get(k) for k in ("served_name", "table", "status", "n_outside_bound", "max_ulp", "why")} for a in (_wp.get("analytic") or [])],
                                                 "reference": "the pinned checkpoint shards (manifests/checkpoints.json sha256), composed by the implementation's stacking rule (weights_of_record)"}
                    if _wp.get("result") is False:
                        row["commit"]["value_correspondence"]["result"] = False
                    TL.end("validate.weights_pin", pair=pair, ok=_wp.get("result"), n=_wp.get("compared"))
                    # the line names the verdict class (ok / mismatch / missing / not evaluated), the DERIVED root and the Program
                    # the record was derived for, so a row.log alone tells which pin was compared
                    _wp_cls = ("ok" if _wp.get("result") else ("missing" if _wp.get("result") is False and (_wp.get("missing") or []) and not (_wp.get("mismatched") or [])
                                                            else ("mismatch" if _wp.get("result") is False else "not evaluated")))
                    print(f"[delta] checkpoint_binding (WEIGHTS PIN, pair {pair}): {'OK' if _wp.get('result') else ('FAIL' if _wp.get('result') is False else 'NOT EVALUATED')} [{_wp_cls}] "
                          f"root {str(_wp.get('root_of_record') or '?')[:16]} (record for Program {str((_wor_rec or {}).get('program_digest') or '?')[:16]}) "
                          f"compared {_wp.get('compared')} mismatched {len(_wp.get('mismatched') or [])} missing {len(_wp.get('missing') or [])} unpinned {len(_wp.get('unpinned') or [])} "
                          f"shards pinned {_wp.get('all_shards_pinned')}{(' -- ' + str(_wp.get('why'))) if _wp.get('why') else ''}", flush=True)
                # the composite's `why` = the FAILING component's own cause (the oracle's partial-coverage note is not a failure cause)
                _vc = row["commit"]["value_correspondence"]
                _fails = [(k, c) for k, c in components.items() if c.get("result") is False]
                _vc["why"] = "; ".join(f"{k}: {c.get('why')}" for k, c in _fails) if _fails else None
                if args.open_after_release and hasattr(com, "release_staging"):
                    with TL.span("validate.openings_after_release", pair=pair) as _tx:
                        row["commit"]["openings_after_release"] = openings_after_release(com, rc, eng, reqs, sampling, base, args, pair, tokens_ref=r["tokens"])
                        _tx["source"] = row["commit"]["openings_after_release"].get("source")
                        _tx["n"] = row["commit"]["openings_after_release"].get("n")
                    oar = row["commit"]["openings_after_release"]
                    if "replay" in str(oar.get("source") or ""):          # replay | host+replay (the excluded families by replay)
                        _vc = row["commit"]["value_correspondence"]
                        _vc["method"] += " + replay (openings_after_release.replay_root_equal)"
                        _rep = bool(oar.get("replay_root_equal"))
                        _vc["components"]["replay"] = {"result": _rep, "reference": "replay of the committed values (openings_after_release)",
                                                       "why": None if _rep else "replay root != committed root"}
                        _cur = _vc["result"]          # never overwrite a live C2 FAIL (oracle mismatch) with a replay result
                        _vc["result"] = _rep if _cur is None else (_cur and _rep)
                        if not _rep:
                            _vc["why"] = ((_vc.get("why") + "; ") if _vc.get("why") else "") + "replay: replay root != committed root"
                    print(f"[delta] OPENINGS AFTER RELEASE (pair {pair}): freed {oar['freed']['raw_device_bytes'] >> 20} MiB raw + "
                          f"{oar['freed']['levels_device_bytes'] >> 20} MiB levels from the device; source {oar['source']}; "
                          f"{oar['verified']}/{oar['n']} verified ({oar['forced_n']} forced: {oar['forced_families']}; host {oar.get('host_n')} / replay {oar.get('replay_n')} "
                          f"{oar.get('replay_families')} in {oar.get('replay_wall_s')} s); replay root equal {oar['replay_root_equal']}; "
                          f"device retained after {oar['retained_device_after'] >> 20} MiB; replay retain {oar.get('replay_retain')}; host rss after {oar.get('host_rss_gb_after')} GB "
                          f"(process VmHWM {oar.get('host_rss_hwm_gb')} GB)", flush=True)
                if args.tensor_digests and getattr(com, "_gpu_blocks", None):
                    # diagnostic: per-tensor digests of the retained (host) bytes, compared across paired instrumented runs to say
                    # WHICH identities differ when run roots disagree (values vs permutation vs schedule)
                    row["commit"]["fa2_zero_rows"] = tensor_digests(com, out, pair, prev_digests)
                route_pin_stamp(row["commit"])
                row["commit"]["commit_pass"], row["commit"]["commit_fail_reason"] = (RB.pending_verdict if pair in deferred else commit_verdict)(
                    row["commit"], row["tokens_equal_ref"], min_openings=min(64, args.openings), declaration=d, required_classes=required_classes)
                if compiled_kernels is not None and row["commit"]["commit_pass"]:
                    ckc = row["commit"]["value_correspondence"]["compiled_check"]
                    if not ckc.get("ok"):
                        row["commit"]["commit_pass"], row["commit"]["commit_fail_reason"] = False, (f"compiled C2 FAIL: {ckc.get('launches_exact')}/{ckc.get('launches_checked')} launches exact, "
                                                                                                    f"missing {ckc.get('launches_missing')}, linkage {ckc.get('linkage')} {ckc.get('error', '')}")
                if served is not None and row["commit"]["commit_pass"]:
                    scc = row["commit"]["value_correspondence"]["compiled_check"]
                    if not scc.get("ok"):
                        row["commit"]["commit_pass"], row["commit"]["commit_fail_reason"] = False, (f"served C2 FAIL: {scc.get('equal')}/{scc.get('checked')} values exact, "
                                                                                                    f"steps unrun {scc.get('steps_unrun')} {scc.get('error') or ''}")
                if compiled_src is not None and row["commit"]["commit_pass"]:
                    # the compiled boundary's own fail-closed conditions (C2 value relations; no refused/mis-shaped in-graph acquisition)
                    cvc = row["commit"]["value_correspondence"]["compiled_check"]
                    cs = row["commit"]["compiled_source"]["stats"]
                    if not cvc.get("ok"):
                        row["commit"]["commit_pass"], row["commit"]["commit_fail_reason"] = False, f"compiled value check FAIL: {cvc.get('equal')}/{cvc.get('checked')} relations hold ({cvc.get('error', '')})"
                    elif cs["refused_capture_alloc"] or cs["shape_mismatch"] or cs["oversize_in_capture"]:
                        row["commit"]["commit_pass"], row["commit"]["commit_fail_reason"] = False, (f"compiled ring refused acquisitions: refused_capture_alloc {cs['refused_capture_alloc']} "
                                                                                                    f"shape_mismatch {cs['shape_mismatch']} oversize_in_capture {cs['oversize_in_capture']}")
                if not row["commit"]["commit_pass"]:
                    print(f"[delta] COMMIT {'FAIL' if row['commit']['commit_pass'] is False else 'PENDING'} (pair {pair}, {arm}): {row['commit']['commit_fail_reason']}", flush=True)
                row["root_equal_expected"] = (rc.run_root.hex() == args.expect_root) if args.expect_root else None
                if args.expect_root and rc.run_root.hex() != args.expect_root:
                    print(f"[delta] ROOT MISMATCH: run root {rc.run_root.hex()} != expected {args.expect_root}", flush=True)
                if arm not in decls:
                    (out / f"declaration_{com.name}.json").write_text(json.dumps(d, indent=1) + "\n")
                    decls[arm] = d
                    if decl is None:
                        decl = d
            else:
                for i, wsec in enumerate(r["step_wall_s"]):
                    step_rows.append({"pair": pair, "arm": arm, "step": i, "wall_s": wsec})
            rows.append(row)
            extra = ""
            if rc is not None:
                sp = rc.spans["total_s"]
                extra = (f" | spans acq {sp['acquisition'] * 1e3:.1f} mov {sp['movement'] * 1e3:.1f} hash {sp['hashing'] * 1e3:.1f} "
                         f"root {sp['root_finalisation'] * 1e3:.1f} wait {sp['wait'] * 1e3:.1f} ms; root-ready {rc.root_ready_delay_s * 1e3:.1f} ms; "
                         f"open {row['commit']['openings']['verified']}/{row['commit']['openings']['n']}; bound {row['commit'].get('committed_tensors')}/"
                         f"{row['commit'].get('expected_tensors')} errors {row['commit'].get('errors_n')} -> {'PASS' if row['commit'].get('commit_pass') else 'FAIL'}")
            print(f"[delta] pair {pair} {arm:12s}: wall {r['wall_s'] * 1e3:.1f} ms steps {len(r['step_wall_s'])} prefill {r['step_wall_s'][0] * 1e3:.2f} ms "
                  f"decode-mean {(row['step_wall_decode_mean_s'] or 0) * 1e3:.2f} ms tok/s {row['generated_tokens_per_s']:.0f} tokens_eq {row['tokens_equal_ref']}{extra}",
                  flush=True)
        with (out / "runs.jsonl").open("w") as f:
            for rw in rows:
                f.write(json.dumps(rw, default=str) + "\n")
        with (out / "steps.jsonl").open("w") as f:
            for rw in step_rows:
                f.write(json.dumps(rw, default=str) + "\n")

    # repeated graph replay must be covered, not only compilation/warm-up: re-read the code
    # objects LOADED in this process after the LAST measured run and compare with the build-time snapshot (compile + autotune + capture):
    # the CUDA graphs replay the launchers that existed at capture, so an unchanged executed set (no new module / kernel / binary / config)
    # is the evidence that the artifact identity recorded at build is the identity of every replayed and eager step of every run
    _gk0 = (versions.get("compilation") or {}).get("generated_kernels") if isinstance(versions.get("compilation"), dict) else None
    if isinstance(_gk0, dict) and isinstance(_gk0.get("loaded"), dict) and not _gk0["loaded"].get("error"):
        _end = va.loaded_code_objects()
        _delta = va.loaded_code_objects_delta(_gk0["loaded"], _end)
        _gk0["loaded_end"] = {k: v for k, v in _end.items() if k not in ("per_kernel", "modules_list")}
        _gk0["replay_covered"] = _delta
        (out / "inductor_kernels_end.json").write_text(json.dumps({"loaded": _end, "delta": _delta, "runs": len(rows)}, indent=1, default=str) + "\n")
        (out / "inductor_kernels.json").write_text(json.dumps(_gk0, indent=1, default=str) + "\n")
        (out / "versions.json").write_text(json.dumps(versions, indent=1, default=str) + "\n")
        print(f"[delta] loaded code objects after the last run: modules {_end.get('modules')} kernels {_end.get('kernels')} executed_set "
              f"{str(_end.get('executed_set_sha256'))[:16]} vs build {str(_gk0['loaded'].get('executed_set_sha256'))[:16]} -> replay covered: {_delta.get('ok')}"
              + (f" ({_delta.get('why')})" if not _delta.get("ok") else ""), flush=True)
    for c in coms.values():
        c.close()
    if coins is not None and (_why := coins.close()):
        print(f"[delta] the challenger session did not close (no ledger end: its coins credit nothing): {_why}", flush=True)
    TL.start("closure.summary")
    if args.only_arm:
        # one arm per process: the paired summary (delta, CI) is produced by compiled_commit.sh from both runs.jsonl files
        summary = {"schema": SCHEMA + "/only-arm", "only_arm": args.only_arm, "runs": len(rows), "versions": versions, "declaration": decl,
                   "wall_s": [r["wall_s"] for r in rows], "commit_pass": (RB.pass_of(rows) if args.only_arm == "instrumented" else None),
                   "table_md": f"only-arm {args.only_arm}: {len(rows)} runs, wall {[round(r['wall_s'], 4) for r in rows]}", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
        (out / "summary.md").write_text(summary["table_md"] + "\n")
        print(summary["table_md"], flush=True)
        if args.only_arm == "control":
            (out / "verdict.json").write_text(json.dumps({"only_arm": "control", "runs": len(rows), "compilation": versions["compilation"]}, indent=1) + "\n")
            print(f"[delta] wrote {out} (control arm only)", flush=True)
            TL.end("closure.summary", only_arm="control")
            TL.event("process.end", "mark", rc=0, s=round(time.time() - _T_PROCESS0, 3))
            return
    elif not multi:
        summary = summarize(rows, step_rows, args, versions, decl)
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
        (out / "summary.md").write_text(summary["table_md"] + "\n")
        print(summary["table_md"], flush=True)
    else:
        # one paired summary per treatment arm (vs the control run of the same round), plus cross-arm root equality per round
        summaries: dict[str, dict] = {}
        md = []
        for kind in kinds:
            sub = [dict(r, arm=("instrumented" if r["arm"] == kind else r["arm"])) for r in rows if r["arm"] in ("control", kind)]
            sub_steps = [dict(s, arm=("instrumented" if s["arm"] == kind else s["arm"])) for s in step_rows if s["arm"] in ("control", kind)]
            a2 = dataclasses.replace(args, committer=kind)
            summaries[kind] = summarize(sub, sub_steps, a2, versions, decls.get(kind))
            md.append(summaries[kind]["table_md"])
        roots_by_round: dict[int, dict[str, str]] = {}
        for r in rows:
            if r["arm"] != "control":
                roots_by_round.setdefault(r["pair"], {})[r["arm"]] = r["commit"]["run_root"]
        roots_equal_across_arms = all(len(set(v.values())) == 1 for v in roots_by_round.values()) and bool(roots_by_round)
        md.append(f"\nrun roots identical across treatment arms in every round: **{roots_equal_across_arms}** "
                  f"({len(roots_by_round)} rounds; arms {kinds}); tokens identical across all arms: **{all(r['tokens_equal_ref'] for r in rows)}**")
        summary = {"schema": SCHEMA + "/multi-arm", "arms": kinds, "rounds": args.pairs, "per_arm": summaries, "roots_by_round": roots_by_round,
                   "roots_equal_across_arms": roots_equal_across_arms, "versions": versions, "table_md": "\n".join(md),
                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
        (out / "summary.md").write_text(summary["table_md"] + "\n")
        print(summary["table_md"], flush=True)
    print(f"[delta] wrote {out}", flush=True)
    # X-03 fail-closed: Commit PASS only if every instrumented run passed `commit_verdict` (manifest count == bound count, no collector
    # error, openings all verified, tokens equal); otherwise the first reason is printed and recorded and the process exits 3
    fails = [(r["pair"], r["arm"], r["commit"].get("commit_fail_reason")) for r in rows if r["arm"] != "control" and not (r.get("commit") or {}).get("commit_pass") and not RB.awaits_replay(r["commit"])]
    pending = [r["pair"] for r in rows if r["arm"] != "control" and RB.awaits_replay(r.get("commit") or {})]
    verdict = {"commit_pass": RB.pass_of(rows), "runs_instrumented": sum(1 for r in rows if r["arm"] != "control"),
               "runs_failed": len(fails), "first_fail_reason": (fails[0][2] if fails else None),
               "required_classes": list(required_classes), "required_classes_source": (
                   f"program/query manifest {required_manifest['manifest_digest']} (R14 MAN-01; classes kept as a secondary check)" if required_manifest is not None else
                   "cli" if args.required_classes is not None else
                   (f"default for FA{fa_version} + moe_block_stream (MOE-03: MoE source attached; REL-02 interim fallback)" if moe_attached else
                    f"default for FA{fa_version} (REL-02 interim fallback: counts+classes, not the Program/query manifest)")),
               "required_manifest": ({"path": args.required_manifest, "manifest_digest": required_manifest["manifest_digest"], "program_digest": required_manifest["program_digest"],
                                      "populations": required_manifest["populations"], "query": required_manifest.get("query"),
                                      "coverage": [r["commit"].get("coverage") for r in rows if r["arm"] != "control" and r.get("commit")]} if required_manifest is not None else None),
               "moe_source": {"mode": args.moe_source, "attached": moe_attached},
               "negatives": ({"flip_weight": flip_weight_rec} if flip_weight_rec else None),        # NEGATIVE TEST flags armed on this run, by name
               "run_roots": [r["commit"]["run_root"] for r in rows if r["arm"] != "control" and r.get("commit")]}
    verdict.update({**CP.verdict_entry(args, plan_rec), **BP.verdict_entry(args, cb_rec)})
    # replay cache of record: the ORIGINAL pair's artifact lists the pairs that share its replay (root equality
    # explicit); the verdict carries the evaluated-vs-reused count so a reader sees at once that C2 ran once per distinct root
    if getattr(args, "replay_cache", False):
        _rc_rec = []
        for _v in replay_cache.values():
            _shared = _v.get("shared_by") or []
            if _shared:
                with contextlib.suppress(Exception):
                    _p = Path(_v["artifact"])
                    _doc = json.loads(_p.read_text())
                    _doc["sampled_replay"]["shared_by"] = _shared
                    _p.write_text(json.dumps(_doc, indent=1, default=str) + "\n")
            _rc_rec.append({"evaluated_pair": _v["pair"], "run_root": _v["run_root"], "shared_by_pairs": [s["pair"] for s in _shared], "roots_equal": all(s["root_equal"] for s in _shared)})
        verdict["replay_cache"] = {"enabled": True, "distinct_replays": len(replay_cache), "reused_pairs": sum(len(r["shared_by_pairs"]) for r in _rc_rec),
                                   "rule": "C2 sampled replay once per distinct (run root, manifest digest, seed) per Commit stage; equal-root pairs reference that record",
                                   "records": _rc_rec}
    # the process's kernel high-water mark over the WHOLE Commit (engine build, warm-ups, every
    # pair's run + openings-by-replay) against the bounded HOST BOUND printed at admission -- in-process, so no sampler can follow the
    # wrong pid (try 4's did); recorded, and a miss is printed by name (a finding against the derivation, not a verdict rule)
    _hwm_kb = rss_kb()[1]
    _adm = sorted(out.glob("host_admission_*.json"))
    if _hwm_kb is not None and _adm:
        with contextlib.suppress(Exception):
            _a = json.loads(_adm[-1].read_text())
            _b = int(_a["host_peak_bound_bytes"])
            _peak = int(_hwm_kb) * 1024
            # admission_of_record / governs_this_run / printed_before_run ON the verdict (the record, not only the code):
            # printed_before_run from the admission's own monotonic print time vs the first pair's start
            _adm_mono, _run0_mono = _a.get("admission_monotonic_s"), (_RUN0 or {}).get("monotonic_s")
            _pbr = (float(_adm_mono) < float(_run0_mono)) if (_adm_mono is not None and _run0_mono is not None) else None
            # per-term predicted vs measured where a term is measurable in-process: engine (RSS at admission vs first
            # pool block), the whole (bound vs VmHWM); the other terms are pool capacities, read from footprint.jsonl / the record
            _terms = {k: int(_a[k]) for k in ("engine_rss_bytes", "levels_bytes", "raw_kept_bytes", "learn_budget_bytes", "replay_bytes",
                                                "slack_bytes", "pairs_extra_bytes") if _a.get(k) is not None}
            verdict["host_peak"] = {"process_vmhwm_bytes": _peak, "host_peak_bound_bytes": _b, "under_bound": _peak <= _b,
                                    "margin_bytes": _b - _peak, "admission": _adm[-1].name,
                                    "admission_of_record": _a.get("admission_of_record"), "governs_this_run": _a.get("governs_this_run"),
                                    "admission_utc": _a.get("admission_utc"), "first_run_started_utc": (_RUN0 or {}).get("utc"),
                                    "printed_before_run": _pbr, "dominant_contributor": _a.get("dominant_contributor"),
                                    "terms_predicted_bytes": _terms, "terms_sum_bytes": sum(_terms.values()),
                                    "measured_over_predicted": round(_peak / _b, 4) if _b else None}
            print(f"[delta] host peak (process VmHWM) {_peak >> 20} MiB vs bounded HOST BOUND {_b >> 20} MiB: "
                  f"{'under' if _peak <= _b else 'OVER'} by {abs(_b - _peak) >> 20} MiB (measured/predicted {round(_peak / _b, 3) if _b else None}; "
                  f"admission {_a.get('admission_utc')} printed_before_run={_pbr}, first run {(_RUN0 or {}).get('utc')}; "
                  f"admission_of_record {(_a.get('admission_of_record') or {}).get('governing') if isinstance(_a.get('admission_of_record'), dict) else _a.get('admission_of_record')})", flush=True)
    PO.into_verdict(protocols, verdict, out, _ekw["target"], program_digest=args.program_digest, manifest=required_manifest)
    RB.seal_all(out, deferred, rows, verdict, pending, tl=TL)
    RS.into_verdict(_rs, verdict)
    (out / "verdict.json").write_text(json.dumps(verdict, indent=1) + "\n")
    TL.end("closure.summary", commit_pass=verdict["commit_pass"], runs=verdict["runs_instrumented"])
    TL.event("process.end", "mark", rc=(3 if fails else 0), s=round(time.time() - _T_PROCESS0, 3), commit_pass=verdict["commit_pass"])
    if fails:
        print(f"[delta] COMMIT FAIL: {len(fails)}/{verdict['runs_instrumented']} instrumented runs failed the fail-closed check; first: pair {fails[0][0]} {fails[0][1]}: {fails[0][2]}", flush=True)
        sys.exit(3)
    print(f"[delta] COMMIT PENDING: {len(pending)}/{verdict['runs_instrumented']} instrumented runs pass every check of this process; their C2 replay runs from {[b['path'] for b in verdict['replay_deferred']['bundles']]}" if pending else
          f"[delta] COMMIT PASS (fail-closed X-03): {verdict['runs_instrumented']} instrumented runs, manifest == bound, 0 errors, openings verified, tokens equal", flush=True)
    if not RS.accepted(_rs):
        sys.exit(3)


def summarize(rows: list[dict], step_rows: list[dict], args: Options, versions: dict, decl: dict | None) -> dict:
    ctrl = {r["pair"]: r for r in rows if r["arm"] == "control"}
    inst = {r["pair"]: r for r in rows if r["arm"] == "instrumented"}
    pairs = sorted(set(ctrl) & set(inst))
    d_wall = [inst[p]["wall_s"] / ctrl[p]["wall_s"] - 1 for p in pairs]
    d_dec = [inst[p]["step_wall_decode_mean_s"] / ctrl[p]["step_wall_decode_mean_s"] - 1 for p in pairs if ctrl[p]["step_wall_decode_mean_s"]]
    d_pre = [inst[p]["step_wall_prefill_s"] / ctrl[p]["step_wall_prefill_s"] - 1 for p in pairs]
    abs_wall_ms = [(inst[p]["wall_s"] - ctrl[p]["wall_s"]) * 1e3 for p in pairs]
    ci = bootstrap_ci(d_wall)
    ci_dec = bootstrap_ci(d_dec)
    ci_pre = bootstrap_ci(d_pre)
    sp_keys = ("acquisition", "movement", "hashing", "root_finalisation", "opening_generation", "wait")
    spans = {k: statistics.fmean(inst[p]["commit"]["spans_total_s"][k] for p in pairs) for k in sp_keys} if pairs else {}
    rr = [inst[p]["commit"]["root_ready_delay_s"] for p in pairs]
    # steady state: per-step hashing time vs step index (instrumented), decode steps only
    steps_inst = [s for s in step_rows if s["arm"] == "instrumented" and s["step"] > 0]
    by_step: dict[int, list[float]] = {}
    for s in steps_inst:
        by_step.setdefault(s["step"], []).append(s["hashing_s"])
    steady = {k: statistics.fmean(v) for k, v in sorted(by_step.items())}
    backlog_max = max((s["backlog_at_step_end"] for s in step_rows if s["arm"] == "instrumented"), default=0)
    first_half = [v for k, v in steady.items() if k <= max(steady) // 2] if steady else []
    second_half = [v for k, v in steady.items() if k > max(steady) // 2] if steady else []
    growth = (statistics.fmean(second_half) / statistics.fmean(first_half) - 1) if first_half and second_half and statistics.fmean(first_half) > 0 else None
    tokens_eq = all(r["tokens_equal_ref"] for r in rows)
    c0 = ctrl[pairs[0]] if pairs else rows[0]
    lines = [f"### Δ_commit — {c0['model']} ({c0['case']}), workload `{Path(args.workload).name}`, committer `{args.committer}`, {len(pairs)} interleaved pairs [measured]",
             "",
             "| metric | control (mean) | instrumented (mean) | Δ (mean over pairs) | 95% CI (bootstrap over pairs) |",
             "|---|---:|---:|---:|---:|"]
    if pairs:
        cw = statistics.fmean(ctrl[p]["wall_s"] for p in pairs) * 1e3
        iw = statistics.fmean(inst[p]["wall_s"] for p in pairs) * 1e3
        cd = statistics.fmean(ctrl[p]["step_wall_decode_mean_s"] for p in pairs) * 1e3
        idd = statistics.fmean(inst[p]["step_wall_decode_mean_s"] for p in pairs) * 1e3
        cp_ = statistics.fmean(ctrl[p]["step_wall_prefill_s"] for p in pairs) * 1e3
        ip = statistics.fmean(inst[p]["step_wall_prefill_s"] for p in pairs) * 1e3
        ct = statistics.fmean(ctrl[p]["generated_tokens_per_s"] for p in pairs)
        itk = statistics.fmean(inst[p]["generated_tokens_per_s"] for p in pairs)
        lines += [f"| end-to-end wall (ms) | {cw:.2f} | {iw:.2f} | **{statistics.fmean(d_wall) * 100:+.2f} %** ({statistics.fmean(abs_wall_ms):+.2f} ms) | [{ci[0] * 100:+.2f} %, {ci[1] * 100:+.2f} %] |",
                  f"| decode step wall (ms, mean) | {cd:.3f} | {idd:.3f} | {statistics.fmean(d_dec) * 100:+.2f} % | [{ci_dec[0] * 100:+.2f} %, {ci_dec[1] * 100:+.2f} %] |",
                  f"| prefill step wall (ms) | {cp_:.3f} | {ip:.3f} | {statistics.fmean(d_pre) * 100:+.2f} % | [{ci_pre[0] * 100:+.2f} %, {ci_pre[1] * 100:+.2f} %] |",
                  f"| generated tokens/s | {ct:.1f} | {itk:.1f} | {(itk / ct - 1) * 100:+.2f} % | |",
                  f"| peak GPU alloc (GB) | {statistics.fmean(ctrl[p]['gpu_mem']['torch_max_allocated_gb'] for p in pairs):.3f} | {statistics.fmean(inst[p]['gpu_mem']['torch_max_allocated_gb'] for p in pairs):.3f} | | |",
                  f"| host RSS after run (MB) | {statistics.fmean(ctrl[p]['host_mem_kb']['rss_after'] or 0 for p in pairs) / 1024:.0f} | {statistics.fmean(inst[p]['host_mem_kb']['rss_after'] or 0 for p in pairs) / 1024:.0f} | | |",
                  "", f"tokens identical across all runs/arms: **{tokens_eq}** (BI=1)", "", "| component (instrumented, per run, mean) | ms | bytes | note |", "|---|---:|---:|---|"]
        sb = inst[pairs[0]]["commit"]["spans_bytes"]
        for k in sp_keys:
            lines.append(f"| {k} | {spans[k] * 1e3:.2f} | {sb.get(k, '')} | host wall of the component (hashing/root on the worker thread, overlapped) |")
        c = inst[pairs[0]]["commit"]
        lines += ["", f"root-ready delay (last step end → run root): mean {statistics.fmean(rr) * 1e3:.2f} ms, max {max(rr) * 1e3:.2f} ms; "
                  f"backlog max {backlog_max} steps; bytes bound per run {c['bytes_bound']:,} in {c['leaves']:,} leaves ({c['tensors']} tensors, {c['steps']} steps); "
                  f"persistent {c['persistent_bytes']:,} B; staging (pinned, in use) {c['temporary_bytes_peak']:,} B; values retained {c['extra'].get('values_retained_bytes'):,} B; "
                  f"openings {c['openings']['verified']}/{c['openings']['n']} verified ({(c['openings']['bytes_mean'] or 0):.0f} B mean); "
                  f"weight registration {c['weight_registration']['wall_s']:.2f} s for {c['weight_registration']['bytes']:,} B (once, excluded)",
                  "", "steady state — mean hashing time per decode step (ms) by step index: " + ", ".join(f"{k}:{v * 1e3:.2f}" for k, v in steady.items()),
                  f"growth second half / first half: {growth:+.1%}" if growth is not None else "growth: n/a"]
    return {"schema": SCHEMA, "workload": args.workload, "committer": args.committer, "pairs": len(pairs), "delta_wall": d_wall, "delta_wall_mean": statistics.fmean(d_wall) if d_wall else None,
            "delta_wall_ci95": ci, "delta_decode": d_dec, "delta_decode_ci95": ci_dec, "delta_prefill": d_pre, "delta_prefill_ci95": ci_pre,
            "abs_wall_ms": abs_wall_ms, "spans_mean_s": spans, "root_ready_delay_s": rr, "steady_hashing_by_step_s": steady, "steady_growth": growth,
            "backlog_max": backlog_max, "tokens_equal": tokens_eq, "versions": versions, "declaration": decl, "table_md": "\n".join(lines),
            # X-03 fail-closed verdict over the instrumented runs (see commit_verdict)
            "commit_pass": RB.pass_of(rows),
            "commit_fail_reasons": [(r.get("commit") or {}).get("commit_fail_reason") for r in rows if r["arm"] != "control" and not (r.get("commit") or {}).get("commit_pass")],
            "expected_tensors": [(r.get("commit") or {}).get("expected_tensors") for r in rows if r["arm"] != "control"],
            "committed_tensors": [(r.get("commit") or {}).get("committed_tensors") for r in rows if r["arm"] != "control"],
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
