"""Hot Commit worker: one commit_delta process serves consecutive Commit jobs of the SAME engine key
instead of building a fresh vLLM engine per invocation.  Not a distributed service: one worker per pod, files under one directory,
the client is a drop-in for `verity-vllm commit ARGS` (row_pod.sh `HOT_ENGINE=1`).

    verity-vllm hot submit [--root /workspace/cp/hot] [--idle 900] -- <commit_delta args...>
    verity-vllm hot serve  <worker dir> --idle <s> --reason <why this worker started>
    verity-vllm hot key    -- <commit_delta args...>        # print the key (diagnostics)
    verity-vllm hot stop   [--root ...]                      # stop every worker under root

Key (what may share an engine) = sha256 of
  * CODE identity: `source_identity.code_identity` of the executing repository -- content hashes of its import closure (the integration
    plus Verity core, `research_tools.CLOSURE`), so an edited pod tree is a different key ("hot" never executes stale code),
  * the native extensions the job would load (`--hidden-so`, $NATIVE_COLLECT_BUILD/*.so, the hidden_gpu_tree torch extension) by sha256,
  * the interpreter (`sys.executable`) and the process environment that shapes the engine / collector (PYTHONPATH, VLLM_*, VERITY_* except
    the per-job VERITY_TIMELINE / VERITY_FOOTPRINT, CUDA_*, HF_*),
  * the engine-shaping arguments: the workload file's engine-relevant blocks (case / model, its `engine` block, execution), --checkpoints
    (content), --case, --max-num-seqs, --engine-arg, --target.
The worker additionally compares the EXACT `LLM(**kw)` keyword set the job would build (`vllm_adapter.engine_kwargs_for`) against its live
engine (commit_delta: HOT-KEY-MISMATCH -> the client re-runs cold).  Jobs whose flags mutate process state (compiled taps, arms, faults, ...)
are refused by commit_delta (HOT-UNSAFE) and re-run cold by the client; the reason is recorded in the row log and on the timeline.

State between jobs (the worker, after every job): every committer built in the job is `close()`d (uninstall -> hooks/wrappers removed, worker
thread stopped, pool shut down) and dropped; gc + torch.cuda.synchronize + empty_cache; no unfinished engine requests; forward hooks / wrapped
forwards / sampler hooks == the baseline taken right after the engine build (commit_delta.assert_pristine).  A failed check STOPS the worker
(recorded as the next worker's restart reason) -- it never serves another job on a dirty engine.  KV/prefix cache: the engine is built with
enable_prefix_caching=False (engine.build.engine_kwargs_for) unless the row's engine args turn it on, and then every workload run starts from
an empty cache (vllm_adapter.fresh_prefix_cache), so a job's hits are its own requests'; RNG: greedy rows are RNG-free, seeded rows carry
per-request seeds (vllm_adapter.sampling_params), so the process RNG state does not enter the result -- the fresh-process comparison
(cold vs hot roots) is the check of record, not this paragraph.

Files (worker dir = <root>/<key16>): KEY, worker.json (pid, started, reason, key, tree), queue/<job>.json, running/, done/<job>.json
(rc, engine_reused, timings, pristine), <job>.log (the job's stdout+stderr, streamed by the client), stop (request), worker.log.
"""

from __future__ import annotations

import contextlib
import glob
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm import settings
from verity_vllm.config import option, workload_file

SCHEMA = "verity-sweep/hot-commit/v1"
# PATH / LD_LIBRARY_PATH are part of the identity: they decide which nvcc / ninja / CUDA libs a JIT build in the worker would use (a worker
# spawned from a shell without the venv on PATH cannot build the native collector), so a client with a different toolchain view
# must not inherit that engine
ENV_PREFIXES = ("PYTHONPATH", "PATH", "LD_LIBRARY_PATH", "VLLM_", "VERITY_", "CUDA_", "HF_", "TORCH", "NATIVE_COLLECT_BUILD", "HIDDEN_GPU_BUILD",
                "TOKENIZERS_PARALLELISM", "OMP_")
ENV_PER_JOB = {"VERITY_TIMELINE", "VERITY_FOOTPRINT", "VERITY_TRACEDUMP", "VERITY_HOT_RESTART_REASON", "VERITY_HOT_SUBMIT", "VERITY_WEIGHTS_HASH", "HASH_THREADS"}
WORKLOAD_ENGINE_KEYS = ("case", "model", "engine", "execution")
RC_HOT_INFRA = 70          # the worker died / refused before running the job (client falls back to cold)
RELEASING_MARKER = Path(settings.HOT_RELEASING_MARKER)     # pod_release.sh holds it while it ships
RELEASING_STALE_S = settings.RELEASING_STALE_S    # pod_release.sh refreshes the marker per step; older = that ship died


def releasing_live() -> bool:
    """A release is in progress iff the marker exists AND was refreshed within RELEASING_STALE_S (same rule as pod_release.sh busy()): a
    marker left by a killed ship must not turn every hot job into a worker start."""
    try:
        return time.time() - RELEASING_MARKER.stat().st_mtime < RELEASING_STALE_S
    except OSError:
        return False
RELEASE_JSON = Path(settings.HOT_RELEASE_JSON)


def release_json_sig() -> str | None:
    """(tag, commit) of the tree on the pod, or None when there is no RELEASE.json (a tree run by hand)."""
    try:
        d = json.loads(RELEASE_JSON.read_text())
        return f"{d.get('tag')}@{str(d.get('commit'))[:9]}"
    except Exception:  # noqa: BLE001
        return None


def sha256_file(p: str | os.PathLike, h: "hashlib._Hash | None" = None) -> str:
    hh = h or hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hh.update(chunk)
    return hh.hexdigest()


def code_identity(tree: str | os.PathLike = ".") -> dict[str, Any]:
    """`source_identity.code_identity` of the repository `tree` belongs to (`tree` = its root or its integrations/vllm), plus the
    EXPORT.json commit and RELEASE.json tag of `tree`."""
    from verity_vllm.pipeline import source_identity

    tree = Path(tree)
    ident = source_identity.code_identity(source_identity.normalise_tree(tree))
    export = {}
    for name, cands in (("EXPORT.json", (tree / "EXPORT.json",)), ("RELEASE.json", (tree / "RELEASE.json", RELEASE_JSON))):
        for f in cands:                                        # RELEASE.json lives beside the tree (/workspace/cp/RELEASE.json), not in it
            if f.exists():
                with contextlib.suppress(Exception):
                    export[name] = json.loads(f.read_text())
                break
    return {"sha256": ident["sha256"], "files": ident["files"], "tree": str(tree.resolve()),
            "commit": (export.get("EXPORT.json") or {}).get("commit"), "release": (export.get("RELEASE.json") or {}).get("tag")}


def _argval(argv: list[str], flag: str, default: Any = None) -> Any:
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default


def _argvals(argv: list[str], flag: str) -> list[str]:
    out = []
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            out.append(argv[i + 1])
        elif a.startswith(flag + "="):
            out.append(a.split("=", 1)[1])
    return out


def so_identity(argv: list[str]) -> dict[str, str | None]:
    """sha256 of the native extensions a job would load (missing file -> 'missing:<path>' so the key still differs)."""
    paths: list[str] = []
    hs = _argval(argv, "--hidden-so")
    if hs:
        paths.append(hs)
    ncb = settings.NATIVE_COLLECT_BUILD
    if ncb:
        # NATIVE_COLLECT_BUILD is the PARENT; the .so this tree loads is in <parent>/<source digest12>/ (native_jit) --
        # name that one when the digest is computable here, else every .so one level down (plus the flat legacy location)
        keyed: list[str] = []
        try:
            from verity_vllm.commit.committer import native_ext as _ne, native_jit as _nj

            keyed = sorted(glob.glob(os.path.join(_nj.keyed_build_dir(ncb, _ne.build_source_digest()), "*.so")))
        except Exception:  # noqa: BLE001 -- no torch / no sources here: fall back to the directory listing
            keyed = []
        paths += keyed or sorted(glob.glob(os.path.join(ncb, "*.so")) + glob.glob(os.path.join(ncb, "*", "*.so")))
    hgb = settings.HIDDEN_GPU_BUILD
    # the .so lives one level down, in the source-keyed child (native_jit.jit_load); the flat legacy location is listed too
    cands = sorted(glob.glob(os.path.join(hgb, "*.so")) + glob.glob(os.path.join(hgb, "*", "*.so"))) if hgb else sorted(
        glob.glob(os.path.expanduser("~/.cache/torch_extensions/*/hidden_gpu_tree/*.so"))
        + glob.glob(os.path.expanduser("~/.cache/torch_extensions/*/hidden_gpu_tree/*/*.so")))
    paths += cands
    out: dict[str, str | None] = {}
    for p in paths:
        out[p] = sha256_file(p) if os.path.exists(p) else f"missing:{p}"
    return out


def env_identity() -> dict[str, str]:
    return settings.redact({k: v for k, v in sorted(settings.environ().items()) if k.startswith(ENV_PREFIXES) and k not in ENV_PER_JOB})


def engine_identity(argv: list[str]) -> dict[str, Any]:
    wl_path = _argval(argv, "--workload", "workloads/workload_cov_b0_b1_c256.json")
    wl: dict[str, Any] = {}
    with contextlib.suppress(Exception):
        wl = json.loads(workload_file(wl_path).read_text())
    ck = _argval(argv, "--checkpoints", "manifests/checkpoints.json")
    return {"workload": {k: wl.get(k) for k in WORKLOAD_ENGINE_KEYS if k in wl}, "workload_path": wl_path,
            "checkpoints_sha256": sha256_file(ck) if os.path.exists(ck) else f"missing:{ck}",
            "case": _argval(argv, "--case"), "max_num_seqs": _argval(argv, "--max-num-seqs", "8"), "engine_args": sorted(_argvals(argv, "--engine-arg")),
            "target": _argval(argv, "--target")}


def unsafe_hints(argv: list[str], env: dict[str, str] | None = None) -> list[str]:
    """argv/env-level mirror of commit_delta.hot_unsafe_reasons (which needs the parsed args and torch): the flags that make a job cold.
    The worker's check stays authoritative (a miss here only costs a refused job + cold fallback); a hit here saves starting a worker."""
    env = settings.environ() if env is None else env
    why = []
    ct = _argval(argv, "--compiled-taps", "off")
    if ct != "off":
        why.append(f"--compiled-taps {ct}")
    for flag in ("--arms", "--late-read", "--value-check"):
        if flag in argv:
            why.append(flag)
    oa = _argval(argv, "--only-arm")
    if oa == "control" or (oa and ct != "off"):
        why.append(f"--only-arm {oa}")
    if "--arena-unchecked" in argv and _argval(argv, "--arena-mb") not in (None, "0"):
        why.append("--arena-unchecked (negative test)")
    if _argval(argv, "--moe-source") == "on":
        why.append("--moe-source on")
    if str(_argval(argv, "--committer", "")).startswith("plugin:"):
        why.append("plugin committer")
    if env.get("VERITY_FAULT", "").strip():
        why.append(f"VERITY_FAULT={env['VERITY_FAULT']!r} (deliberate fault: negative test)")
    if env.get("VERITY_SWITCH"):
        why.append("VERITY_SWITCH")
    return why


def job_key(argv: list[str], tree: str | os.PathLike = ".") -> dict[str, Any]:
    ident = {"schema": SCHEMA, "python": sys.executable, "code": code_identity(tree), "so": so_identity(argv), "env": env_identity(),
             "engine": engine_identity(argv)}
    ident["key"] = hashlib.sha256(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()
    return ident


# ---------------------------------------------------------------------------------------------------------------------------- worker state
def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _read_json(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def worker_info(wdir: Path) -> dict[str, Any] | None:
    w = _read_json(wdir / "worker.json")
    if not w:
        return None
    w["alive"] = _alive(w.get("pid")) and not (wdir / "exited.json").exists()
    return w


def stop_workers(root: Path, *, except_dir: Path | None = None, wait_s: float = 120.0, busy_wait_s: float = 3600.0, idle_only: bool = False,
                 log=print) -> list[str]:
    """Ask every live worker under root (but `except_dir`) to stop; a worker BUSY with a job is waited for (up to busy_wait_s) -- another
    row's Commit is never killed to make room -- then asked to stop; SIGTERM only if it ignores the request for wait_s.  One engine per pod.
    idle_only: leave busy workers alone (pod_release.sh before its busy check: an IDLE engine must not make a release wait, a running row must)."""
    stopped = []
    for wj in sorted(root.glob("*/worker.json")):
        wdir = wj.parent
        if except_dir is not None and wdir.resolve() == except_dir.resolve():
            continue
        w = worker_info(wdir)
        if not w or not w["alive"]:
            continue
        if idle_only and w.get("current"):
            log(f"[hot] worker {w['pid']} ({wdir.name}) is busy with job {w.get('current')}: left alone (idle-only stop)")
            continue
        t0 = time.time()
        while w and w["alive"] and w.get("current") and time.time() - t0 < busy_wait_s:
            if int(time.time() - t0) % 60 == 0:
                log(f"[hot] worker {w['pid']} ({wdir.name}) is busy with job {w.get('current')}: waiting ({time.time() - t0:.0f}s)")
            time.sleep(2.0)
            w = worker_info(wdir)
        if not w or not w["alive"]:
            continue
        (wdir / "stop").write_text(f"{time.time()} stop requested by pid {os.getpid()} ({os.getcwd()})\n")
        log(f"[hot] stopping worker {w['pid']} ({wdir.name}, key {str(w.get('key'))[:16]}) -- one engine per pod")
        t0 = time.time()
        while _alive(w["pid"]) and time.time() - t0 < wait_s:
            time.sleep(1.0)
        if _alive(w["pid"]):
            log(f"[hot] worker {w['pid']} did not exit in {wait_s:.0f}s: SIGTERM")
            with contextlib.suppress(ProcessLookupError):
                os.kill(w["pid"], signal.SIGTERM)
            t0 = time.time()
            while _alive(w["pid"]) and time.time() - t0 < 30:
                time.sleep(1.0)
            if _alive(w["pid"]):
                with contextlib.suppress(ProcessLookupError):
                    os.kill(w["pid"], signal.SIGKILL)
        stopped.append(wdir.name)
    return stopped


# ---------------------------------------------------------------------------------------------------------------------------- client
def submit(argv: list[str], *, root: Path, idle_s: float, timeout_s: float | None, spawn: bool = True, log=print) -> int:
    """Run commit_delta ARGS on the pod's hot worker for this key (start one if none is alive); stream its output; return the job's rc.
    Falls back to a fresh (cold) process when the worker refuses the job (HOT-UNSAFE / HOT-KEY-MISMATCH) or dies before finishing."""
    from verity_vllm.pipeline.timeline import Timeline

    tl = Timeline.from_env("hot_commit", hot=True)
    hints = unsafe_hints(argv)
    if hints:
        # torch-free pre-check of commit_delta.hot_unsafe_reasons: such a job would only be refused by the worker, so it must neither start
        # one nor stop a live engine of another key on its way
        log(f"[hot] HOT-UNSAFE by argv/env, no worker touched: {'; '.join(hints)}")
        return cold_run(argv, f"HOT-UNSAFE (client pre-check): {'; '.join(hints)}", log=log)
    t_key0 = time.time()
    ident = job_key(argv)
    key = ident["key"]
    wdir = root / key[:16]
    wdir.mkdir(parents=True, exist_ok=True)
    for sub in ("queue", "running", "done"):
        (wdir / sub).mkdir(exist_ok=True)
    (wdir / "KEY.json").write_text(json.dumps(ident, indent=1, default=str) + "\n")
    tl.event("hot.key", "mark", s=round(time.time() - t_key0, 3), key=key[:16], code_sha256=ident["code"]["sha256"][:16], code_files=ident["code"]["files"],
             so=len(ident["so"]), release=ident["code"].get("release"))
    log(f"[hot] key {key[:16]} (code {ident['code']['sha256'][:16]} over {ident['code']['files']} files, release {ident['code'].get('release')}, "
        f"{len(ident['so'])} native .so, {len(ident['env'])} env keys, model {ident['engine']['workload'].get('model') or ident['engine']['workload'].get('case')} "
        f"max_num_seqs {ident['engine']['max_num_seqs']}); worker dir {wdir}")
    w = worker_info(wdir)
    reason = None
    if w and w["alive"] and w.get("key") == key:
        log(f"[hot] worker {w['pid']} alive since {w.get('started_utc')} (jobs served {w.get('jobs', '?')}): reusing")
    else:
        if w and not w["alive"]:
            ex = _read_json(wdir / "exited.json") or {}
            reason = f"previous worker exited: {ex.get('reason') or 'unknown'}"
        elif w and w.get("key") != key:
            reason = "worker dir key differs (tree / extension / env changed)"
        else:
            reason = "no live hot worker for this key"
        others = stop_workers(root, except_dir=wdir, log=log)
        if others:
            reason += f"; stopped other-key workers {others}"
        if not spawn:
            log(f"[hot] no worker and spawning disabled: cold run ({reason})")
            return cold_run(argv, reason, log=log)
        cmd = [sys.executable, "-m", "verity_vllm.pipeline.cli", "hot", "serve", str(wdir), "--idle", str(idle_s), "--reason", reason]
        with open(wdir / "worker.log", "a") as wl:
            wl.write(f"$ {shlex.join(cmd)}  (cwd {os.getcwd()}, by pid {os.getpid()})\n")
            wl.flush()
            p = subprocess.Popen(cmd, stdout=wl, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=os.getcwd(), env={k: v for k, v in settings.environ().items() if k not in ENV_PER_JOB},
                                 start_new_session=True)
        log(f"[hot] worker started pid {p.pid}: {reason}")
        tl.event("hot.worker_start", "mark", pid=p.pid, restart_reason=reason)
    job_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}"
    env = settings.environ()
    job = {"schema": SCHEMA, "id": job_id, "argv": list(argv), "cwd": os.getcwd(), "key": key, "submitted": time.time(),
           "env": {k: env[k] for k in ENV_PER_JOB if k in env}, "log": str(wdir / f"{job_id}.log")}
    tmp = wdir / "queue" / f"{job_id}.json.tmp"
    tmp.write_text(json.dumps(job, indent=1) + "\n")
    tmp.rename(wdir / "queue" / f"{job_id}.json")
    tl.start("hot.job_wait", job=job_id)
    log(f"[hot] job {job_id} queued: {shlex.join(argv)[:300]}", flush=True)
    # stream the job log; finish on done/<id>.json; give up if the worker dies
    log_path = Path(job["log"])
    done_path = wdir / "done" / f"{job_id}.json"
    pos = 0
    t0 = time.time()
    last_alive = time.time()
    while True:
        if log_path.exists():
            with open(log_path, "rb") as f:
                f.seek(pos)
                chunk = f.read()
                if chunk:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.flush()
                    pos += len(chunk)
        if done_path.exists():
            res = _read_json(done_path)
            if res is not None:
                break
        w = worker_info(wdir)
        if w and w["alive"]:
            last_alive = time.time()
        elif time.time() - last_alive > 15:
            ex = _read_json(wdir / "exited.json") or {}
            log(f"\n[hot] worker died before finishing job {job_id}: {ex.get('reason') or 'no exit record'}")
            tl.end("hot.job_wait", job=job_id, ok=False, rc=RC_HOT_INFRA)
            with contextlib.suppress(OSError):
                (wdir / "queue" / f"{job_id}.json").unlink()
            return cold_run(argv, f"hot worker died: {ex.get('reason') or 'no exit record'}", log=log)
        if timeout_s is not None and time.time() - t0 > timeout_s:
            log(f"\n[hot] job {job_id} exceeded {timeout_s:.0f}s: stopping the worker (SIGTERM)")
            if w:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(w["pid"], signal.SIGTERM)
            tl.end("hot.job_wait", job=job_id, ok=False, rc=124)
            return 124
        time.sleep(0.5)
    rc = int(res.get("rc", RC_HOT_INFRA))
    tl.end("hot.job_wait", job=job_id, rc=rc, engine_reused=res.get("engine_reused"), job_s=res.get("job_s"), pristine=res.get("pristine_ok"))
    log(f"[hot] job {job_id} done rc={rc} engine_reused={res.get('engine_reused')} job {res.get('job_s')}s (worker job #{res.get('worker_job')}, "
        f"post-job pristine {res.get('pristine_ok')}, refused={res.get('refused')})")
    if res.get("refused"):
        # HOT-UNSAFE / HOT-KEY-MISMATCH: the engine was never touched; the job runs in a fresh process with the reason on record
        return cold_run(argv, f"hot worker refused: {res.get('refused')}", log=log)
    return rc


def cold_run(argv: list[str], reason: str, log=print) -> int:
    from verity_vllm.pipeline.timeline import Timeline

    tl = Timeline.from_env("hot_commit", hot=False)
    tl.event("hot.fallback_cold", "mark", restart_reason=reason)
    log(f"[hot] COLD fallback: fresh commit_delta process ({reason})", flush=True)
    cmd = [sys.executable, "-m", "verity_vllm.pipeline.cli", "commit", *argv]
    env = dict(settings.environ(), VERITY_HOT_RESTART_REASON=reason)
    env.pop("VERITY_HOT_SUBMIT", None)          # the cold process must not route back through the worker (commit_delta.__main__ hook)
    sys.stdout.flush()
    return subprocess.call(cmd, env=env)


# ---------------------------------------------------------------------------------------------------------------------------- server
# committer attributes that own device / pinned memory (native_collect window ring + scratch, the FA2/FA3 tap buffers of the hidden
# sources, the staging arena, the pinned pool, the retained device store).  A cold process frees them at exit; the hot worker must drop them
# explicitly after `close()` so the caching allocator can hand the memory to the next job (else +2.3 GiB stays allocated per hot job).
DEVICE_ATTRS = ("win_tensors", "win_scratch", "arena", "pool", "_gpu_blocks", "_gpu_levels", "_levels", "_done", "_layouts", "_streams",
                "_metas_cache", "_wrapped", "_handles", "_cur", "sources", "_nc", "_hi")


def release_committer_device(com: Any) -> list[str]:
    """Best-effort: after `close()`, drop every attribute of the committer (and of its hidden sources: tap buffers) that pins device or
    pinned-host memory.  Returns the attribute names it cleared.  The committer object is dead afterwards (the hot worker never reuses it)."""
    cleared: list[str] = []
    w = getattr(com, "_worker", None)
    if w is not None and getattr(w, "is_alive", lambda: False)():
        return ["SKIPPED:cmt-worker-alive"]          # close() could not join the commit worker thread: leave its state alone
    for src in list(getattr(com, "sources", None) or []):
        for nm in ("bufs", "_w", "buf"):
            if hasattr(src, nm):
                with contextlib.suppress(Exception):
                    v = getattr(src, nm)
                    v.clear() if isinstance(v, dict) else setattr(src, nm, None)
                    cleared.append(f"src.{nm}")
    for nm in DEVICE_ATTRS:
        if hasattr(com, nm):
            with contextlib.suppress(Exception):
                v = getattr(com, nm)
                if isinstance(v, dict):
                    v.clear()
                elif isinstance(v, list):
                    v.clear()
                else:
                    setattr(com, nm, None)
                cleared.append(nm)
    return cleared


def serve(wdir: Path, *, idle_s: float, reason: str) -> int:
    import gc

    t_start = time.time()
    (wdir / "queue").mkdir(parents=True, exist_ok=True)
    (wdir / "running").mkdir(exist_ok=True)
    (wdir / "done").mkdir(exist_ok=True)
    with contextlib.suppress(OSError):
        (wdir / "stop").unlink()
    with contextlib.suppress(OSError):
        (wdir / "exited.json").unlink()
    key_doc = _read_json(wdir / "KEY.json") or {}
    env0 = {k: v for k, v in settings.environ().items() if k not in ENV_PER_JOB}     # per-job keys come from each job, never from the spawning client
    cwd0 = os.getcwd()
    wlog = open(wdir / "worker.log", "a", buffering=1)

    def log(msg: str) -> None:
        wlog.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [hot-worker {os.getpid()}] {msg}\n")

    def write_state(**extra: Any) -> None:
        doc = {"schema": SCHEMA, "pid": os.getpid(), "started": t_start, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)),
               "reason": reason, "key": key_doc.get("key"), "code_sha256": (key_doc.get("code") or {}).get("sha256"), "release": (key_doc.get("code") or {}).get("release"),
               "cwd": cwd0, "python": sys.executable, "idle_s": idle_s, **extra}
        tmp = wdir / "worker.json.tmp"
        tmp.write_text(json.dumps(doc, indent=1, default=str) + "\n")
        tmp.rename(wdir / "worker.json")

    def exit_with(why: str, rc: int = 0) -> int:
        log(f"exit: {why}")
        (wdir / "exited.json").write_text(json.dumps({"pid": os.getpid(), "reason": why, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                      "jobs": jobs, "reuses": reuses}, indent=1) + "\n")
        write_state(jobs=jobs, reuses=reuses, exited=True, exit_reason=why)
        # tear the engine down explicitly (vLLM worker processes / CUDA context) so the next worker finds the GPU free
        with contextlib.suppress(Exception):
            eng = (cd._HOT or {}).get("eng")
            if eng is not None:
                del eng
                cd._HOT["eng"] = None
                gc.collect()
        wlog.close()
        return rc

    jobs = 0
    reuses = 0
    write_state(jobs=0, reuses=0)
    log(f"serving {wdir} (key {str(key_doc.get('key'))[:16]}, reason: {reason}); importing commit_delta")
    t_imp = time.time()
    import torch
    from verity_vllm.pipeline import cli, commit as cd

    cd._HOT = {"jobs": 0, "restart_reason": reason}
    rel_sig0 = release_json_sig()
    log(f"imports {time.time() - t_imp:.1f}s; waiting for jobs (idle timeout {idle_s:.0f}s; RELEASE.json {rel_sig0})")

    def _sigterm(*_: Any) -> None:
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, _sigterm)
    last_job = time.time()
    try:
        while True:
            if (wdir / "stop").exists():
                return exit_with("stop requested")
            queue = sorted(p for p in (wdir / "queue").glob("*.json") if not p.name.endswith(".tmp"))
            if not queue:
                if time.time() - last_job > idle_s:
                    return exit_with(f"idle {idle_s:.0f}s")
                # a release ship must never wait on an idle engine: the RELEASING marker (pod_release.sh) or a changed
                # RELEASE.json (a new tree is on the pod: this worker's code identity is stale by definition) ends the worker at once
                if releasing_live():
                    return exit_with("release in progress (RELEASING marker) -- idle engine released")
                if release_json_sig() != rel_sig0:
                    return exit_with("RELEASE.json changed under the worker -- code identity stale, engine released")
                time.sleep(0.5)
                continue
            jp = queue[0]
            job = _read_json(jp)
            if job is None:
                time.sleep(0.2)
                continue
            rp = wdir / "running" / jp.name
            jp.rename(rp)
            jobs += 1
            cd._HOT["jobs"] = jobs
            cd._HOT["coms"] = []
            write_state(jobs=jobs, reuses=reuses, current=job["id"])
            log(f"job {job['id']} #{jobs}: {shlex.join(job['argv'])[:400]}")
            # per-job process state: the worker's own env restored, then the job's per-job keys, then what the engine's build wrote
            # into the process (Engine.environ: vLLM's batch invariance sets cuBLAS' switches as it builds); the job's cwd
            cli.set_environment({**env0, **(job.get("env") or {}), **(getattr(cd._HOT.get("eng"), "environ", None) or {})})
            os.chdir(job.get("cwd") or cwd0)
            res: dict[str, Any] = {"schema": SCHEMA, "id": job["id"], "worker_pid": os.getpid(), "worker_job": jobs, "started": time.time()}
            had_engine = (cd._HOT.get("eng") is not None)
            # the job's stdout/stderr (Python prints AND vLLM's C-level logs) -> its own log file, streamed by the client into commit.log
            fd_out, fd_err = os.dup(1), os.dup(2)
            sys.stdout.flush()
            sys.stderr.flush()
            with open(job["log"], "ab", buffering=0) as jl:
                os.dup2(jl.fileno(), 1)
                os.dup2(jl.fileno(), 2)
                rc: int | None = None
                refused = None
                try:
                    cd.main(cli.parse("commit", list(job["argv"])), argv=list(job["argv"]))
                    rc = 0
                except SystemExit as e:
                    code = e.code
                    if isinstance(code, str):
                        print(code, flush=True)
                        if code.startswith("HOT-UNSAFE") or code.startswith("HOT-KEY-MISMATCH"):
                            refused = code
                            rc = RC_HOT_INFRA
                        else:
                            rc = 1
                    else:
                        rc = int(code or 0)
                except KeyboardInterrupt:
                    print("[hot-worker] SIGTERM during job", flush=True)
                    rc = 143
                except BaseException as e:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    print(f"[hot-worker] job crashed: {e!r}", flush=True)
                    rc = 1
                finally:
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os.dup2(fd_out, 1)
                    os.dup2(fd_err, 2)
                    os.close(fd_out)
                    os.close(fd_err)
            engine_reused = had_engine and refused is None
            if engine_reused:
                reuses += 1
            # ---------------------------------------------------------------- post-job: close committers, free, check pristine
            t_pj = time.time()
            pristine_ok, pristine_why = True, None
            coms = cd._HOT.pop("coms", []) or []
            released: list[str] = []
            for com in coms:
                for meth in ("uninstall", "close"):
                    with contextlib.suppress(Exception):
                        getattr(com, meth)()
                released += release_committer_device(com)
            n_coms = len(coms)
            del coms
            gc.collect()
            eng = cd._HOT.get("eng")
            unfinished = None
            residue = None
            if eng is not None:
                with contextlib.suppress(Exception):
                    torch.cuda.synchronize()
                with contextlib.suppress(Exception):
                    unfinished = bool(eng.llm.llm_engine.has_unfinished_requests())
                with contextlib.suppress(Exception):
                    torch.cuda.empty_cache()
                try:
                    cd.assert_pristine(eng, cd._HOT["base0"])
                except AssertionError as e:
                    pristine_ok, pristine_why = False, f"engine not pristine after job: {e}"
                if unfinished:
                    pristine_ok, pristine_why = False, "engine has unfinished requests after job"
                # residue audit: CUDA storages the job left behind (Python-visible) vs the census taken right after the engine build; the
                # allocator counter tells the rest (a growth the census cannot see is held from C++)
                with contextlib.suppress(Exception):
                    if cd._HOT.get("cuda0") is not None:
                        residue = cd.cuda_residue(cd._HOT["cuda0"], cd.cuda_census())
                        residue["alloc_growth_mib"] = int((torch.cuda.memory_allocated() >> 20) - int(cd._HOT["base0"].get("cuda_alloc_mib", 0)))
            mem = None
            with contextlib.suppress(Exception):
                mem = {"allocated_mib": torch.cuda.memory_allocated() >> 20, "reserved_mib": torch.cuda.memory_reserved() >> 20}
            res.update(rc=rc, refused=refused, engine_reused=engine_reused, engine_built_this_job=(not had_engine and cd._HOT.get("eng") is not None and refused is None),
                       job_s=round(time.time() - res["started"], 3), post_job_s=round(time.time() - t_pj, 3), committers_closed=n_coms,
                       pristine_ok=pristine_ok, pristine_why=pristine_why, unfinished_requests=unfinished, cuda=mem, cuda_residue=residue,
                       device_released=sorted(set(released)), restart_reason=reason if not had_engine else None)
            tmp = wdir / "done" / (jp.name + ".tmp")
            tmp.write_text(json.dumps(res, indent=1, default=str) + "\n")
            tmp.rename(wdir / "done" / jp.name)
            with contextlib.suppress(OSError):
                rp.unlink()
            log(f"job {job['id']} rc={rc} reused={engine_reused} refused={refused} {res['job_s']}s; post-job {res['post_job_s']}s committers {n_coms} "
                f"pristine={pristine_ok} {pristine_why or ''} cuda {mem} residue {json.dumps(residue, default=str) if residue else None} released {sorted(set(released))}")
            write_state(jobs=jobs, reuses=reuses, current=None, last_job=job["id"], last_rc=rc)
            last_job = time.time()
            if not pristine_ok:
                return exit_with(pristine_why or "not pristine", rc=3)
            if refused and refused.startswith("HOT-KEY-MISMATCH"):
                return exit_with(refused)
            if rc == 143:
                return exit_with("SIGTERM")
    except KeyboardInterrupt:
        return exit_with("SIGTERM while idle", rc=143)
    except Exception as e:  # noqa: BLE001
        import traceback
        log(traceback.format_exc())
        return exit_with(f"worker loop crashed: {e!r}", rc=1)


# ---------------------------------------------------------------------------------------------------------------------------- CLI
@dataclass(frozen=True, kw_only=True)
class SubmitOptions:
    root: str = option("--root", env="HOT_ROOT", default="/workspace/cp/hot")
    idle: float = option("--idle", type=float, env="HOT_IDLE", default=900.0)
    timeout: float | None = option("--timeout", type=float, default=None)
    no_spawn: bool = option("--no-spawn", default=False, action="store_true",
                            help="use a live worker if one exists, else run cold")
    commit_args: list[str] = option("commit_args", nargs="*", default=[], metavar="-- COMMIT_ARGS")
    cmd: str = field(default="submit", init=False)


@dataclass(frozen=True, kw_only=True)
class ServeOptions:
    wdir: str = option("wdir")
    idle: float = option("--idle", type=float, default=900.0)
    reason: str = option("--reason", default="started by hand")
    cmd: str = field(default="serve", init=False)


@dataclass(frozen=True, kw_only=True)
class KeyOptions:
    tree: str = option("--tree", default=".")
    commit_args: list[str] = option("commit_args", nargs="*", default=[], metavar="-- COMMIT_ARGS")
    cmd: str = field(default="key", init=False)


@dataclass(frozen=True, kw_only=True)
class StopOptions:
    root: str = option("--root", env="HOT_ROOT", default="/workspace/cp/hot")
    idle_only: bool = option("--idle-only", default=False, action="store_true",
                             help="stop idle workers only; a worker running a job is left alone (release ships)")
    cmd: str = field(default="stop", init=False)


COMMANDS = {"submit": SubmitOptions, "serve": ServeOptions, "key": KeyOptions, "stop": StopOptions}


def main(a: SubmitOptions | ServeOptions | KeyOptions | StopOptions) -> int:
    if a.cmd == "submit":
        def _log(msg: str, flush: bool = True) -> None:
            print(msg, flush=flush)
        return submit(list(a.commit_args), root=Path(a.root), idle_s=a.idle, timeout_s=a.timeout, spawn=not a.no_spawn, log=_log)
    if a.cmd == "serve":
        return serve(Path(a.wdir), idle_s=a.idle, reason=a.reason)
    if a.cmd == "key":
        print(json.dumps(job_key(list(a.commit_args), a.tree), indent=1, default=str))
        return 0
    if a.cmd == "stop":
        print(json.dumps({"stopped": stop_workers(Path(a.root), idle_only=bool(a.idle_only))}))
        return 0
    return 2
